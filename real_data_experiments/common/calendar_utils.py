"""Calendar utility functions for calendar-based baselines across experiments.

Provides:
- load_calendar_features: load and validate 15min calendar CSV
- get_sample_target_times: extract target_time from various dataset types
- collect_targets_and_calendar: collect targets and calendar rows for each sample
- evaluate_calendar_profile_naive_for_clients: evaluate CalendarProfileNaive baseline
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from real_data_experiments.common.calendar_baselines import (
    build_client_seasonal_profile,
    calendar_profile_naive_predict,
)
from real_data_experiments.common.metrics import compute_regression_metrics


def load_calendar_features(
    calendar_features_path: str | Path,
    expected_time_count: int | None = None,
) -> pd.DataFrame:
    """Load and validate 15min calendar features CSV.

    Args:
        calendar_features_path: Path to the 15min calendar CSV.
        expected_time_count: If provided, verify that the calendar covers
            at least this many rows (time_index from 0 to expected_time_count-1).

    Returns:
        DataFrame sorted by time_index with required columns.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required columns are missing or time coverage is insufficient.
    """
    path = Path(calendar_features_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Calendar features file not found: {path}. "
            f"Run real_data_experiments/tools/build_calendar_features_2017.py first."
        )

    cal = pd.read_csv(path)
    required = ["time_index", "slot_of_day", "is_effective_workday"]
    missing = [col for col in required if col not in cal.columns]
    if missing:
        raise ValueError(
            f"Calendar features CSV missing required columns: {missing}. "
            f"Available columns: {list(cal.columns)}"
        )

    cal = cal.sort_values("time_index").reset_index(drop=True)

    if expected_time_count is not None:
        if len(cal) < expected_time_count:
            raise ValueError(
                f"Calendar features ({len(cal)} rows) does not cover "
                f"the expected time range ({expected_time_count} time steps)."
            )

    return cal


def get_sample_target_times(dataset) -> np.ndarray:
    """Extract target_time (calendar time_index) for every sample in a dataset.

    Supports:
    - RegionClientWindowDataset (has first_target_time, per_region_window_count)
    - torch.utils.data.Subset (recursively resolves underlying dataset)
    - Any wrapper with a `.dataset` attribute

    Args:
        dataset: A PyTorch Dataset instance.

    Returns:
        np.ndarray of shape (len(dataset),) with integer time_index values.

    Raises:
        TypeError: if the dataset type cannot be resolved.
    """
    # Unwrap Subset
    if isinstance(dataset, Subset):
        base_times = get_sample_target_times(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=int)
        return base_times[indices]

    # Unwrap other wrappers via .dataset attribute
    if hasattr(dataset, "dataset") and not hasattr(dataset, "first_target_time"):
        return get_sample_target_times(dataset.dataset)

    # RegionClientWindowDataset
    if hasattr(dataset, "first_target_time") and hasattr(dataset, "per_region_window_count"):
        first = int(dataset.first_target_time)
        window_count = int(dataset.per_region_window_count)
        n_regions = len(dataset.region_ids) if hasattr(dataset, "region_ids") else 1
        expected_len = n_regions * window_count
        if len(dataset) != expected_len:
            raise ValueError(
                f"Dataset length {len(dataset)} != n_regions * per_region_window_count "
                f"({n_regions} * {window_count} = {expected_len})"
            )
        # target_time = first_target_time + (sample_idx % per_region_window_count)
        single_region_times = np.arange(first, first + window_count, dtype=int)
        return np.tile(single_region_times, n_regions)

    raise TypeError(
        f"Cannot extract target times from dataset type {type(dataset).__name__}. "
        f"Expected RegionClientWindowDataset or Subset wrapping one."
    )


def collect_targets_and_calendar(
    dataset,
    calendar_features: pd.DataFrame,
    target_channel: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Collect target values and calendar rows for each sample in a dataset.

    Uses get_sample_target_times() to map each sample to a calendar time_index,
    then looks up the corresponding row in the calendar features DataFrame.
    Uses vectorized tensor access when possible (RegionClientWindowDataset),
    falling back to per-sample iteration for wrappers like Subset.

    Args:
        dataset: A RegionClientWindowDataset or Subset thereof.
        calendar_features: DataFrame loaded by load_calendar_features().
        target_channel: Which tensor channel to extract as the target value.

    Returns:
        (y, calendar_rows) where:
        - y: np.ndarray of shape (len(dataset),) with target values
        - calendar_rows: pd.DataFrame with len(dataset) rows from calendar_features

    Raises:
        ValueError: if sample count mismatches or time_index is out of range.
    """
    target_times = get_sample_target_times(dataset)
    n_samples = len(target_times)
    if n_samples != len(dataset):
        raise ValueError(
            f"Target times count ({n_samples}) != dataset length ({len(dataset)})"
        )

    # Vectorized path for RegionClientWindowDataset (raw, unwrapped)
    base_dataset = dataset
    if hasattr(base_dataset, "dataset") and hasattr(base_dataset.dataset, "tensor"):
        base_dataset = base_dataset.dataset

    if hasattr(base_dataset, "tensor") and hasattr(base_dataset, "first_target_time"):
        # Direct tensor access: targets are at tensor[target_channel, region_ids, target_time]
        tensor = base_dataset.tensor
        region_ids = list(base_dataset.region_ids) if hasattr(base_dataset, "region_ids") else [base_dataset.region_id]
        # Build target_times for the base dataset
        n_regions = len(region_ids)
        per_region = len(target_times) // n_regions
        # For each sample, map sample_index -> region_pos, target_pos
        sample_to_region = np.repeat(np.arange(n_regions), per_region)
        sample_to_region_idx = np.array([region_ids[r] for r in sample_to_region])

        # Extract targets: tensor[target_channel, region_id, time_idx]
        # We need to handle that some datasets may be Subset
        if isinstance(dataset, Subset):
            # For Subset: target_times is already subsetted, need per-sample iteration
            # But we can batch the tensor extraction
            y = np.empty(n_samples, dtype=np.float64)
            for i in range(n_samples):
                _, target = dataset[i]
                y[i] = float(target.view(-1)[0].item())
        else:
            # Direct tensor extraction
            y = tensor[target_channel, sample_to_region_idx, target_times].numpy().astype(np.float64)
    else:
        # Fallback: per-sample iteration
        y = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            _, target = dataset[i]
            y[i] = float(target.view(-1)[0].item())

    # Calendar lookup: vectorized via merge
    cal_lookup = calendar_features.set_index("time_index")
    cal_rows = cal_lookup.reindex(target_times).reset_index(drop=True)
    cal_rows = cal_rows.ffill().bfill()  # safety for any missing indices

    return y, cal_rows


def evaluate_calendar_profile_naive_for_clients(
    clients,
    calendar_features: pd.DataFrame,
    target_channel: int = 0,
    method_name: str = "CalendarProfileNaive",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate CalendarProfileNaive baseline for a list of clients.

    For each client:
    1. Collect y_train + calendar rows from raw_train_dataset
    2. Collect y_test + calendar rows from raw_test_dataset
    3. Build client-specific weekday/weekend slot profile
    4. Predict using profile on test calendar rows
    5. Compute regression metrics

    Args:
        clients: List of client data objects, each must have:
            - client_id: int
            - raw_train_dataset: RegionClientWindowDataset
            - raw_test_dataset: RegionClientWindowDataset
        calendar_features: DataFrame from load_calendar_features().
        target_channel: Tensor channel index for target values.
        method_name: Label for the method column in outputs.

    Returns:
        (client_metrics_df, prediction_df) where:
        - client_metrics_df: one row per client with method, metrics, client metadata
        - prediction_df: per-sample predictions with method, client_id, sample_index,
          y_true, y_pred, target_time, date, slot_of_day, is_effective_workday
    """
    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for client in clients:
        client_id = int(client.client_id)

        if client.raw_train_dataset is None:
            raise ValueError(
                f"Client {client_id} has no raw_train_dataset; "
                f"CalendarProfileNaive requires raw (un-normalized) datasets."
            )
        if client.raw_test_dataset is None:
            raise ValueError(
                f"Client {client_id} has no raw_test_dataset; "
                f"CalendarProfileNaive requires raw (un-normalized) datasets."
            )

        # Collect train data
        y_train, cal_train = collect_targets_and_calendar(
            client.raw_train_dataset, calendar_features, target_channel=target_channel
        )
        # Collect test data
        y_test, cal_test = collect_targets_and_calendar(
            client.raw_test_dataset, calendar_features, target_channel=target_channel
        )

        # Build profile and predict
        profile = build_client_seasonal_profile(y_train, cal_train, "weekday_weekend")
        preds = calendar_profile_naive_predict(
            y_train=y_train,
            y_test_raw=y_test,
            test_time_indices=get_sample_target_times(client.raw_test_dataset),
            calendar_features_test=cal_test,
            profile=profile,
        )

        # Compute metrics
        metrics: dict[str, Any] = compute_regression_metrics(y_test, preds)
        metrics["method"] = method_name
        metrics["client_id"] = client_id

        # Add client metadata if available
        for attr in ("entity_kind", "entity_id", "cell_count", "region_count"):
            if hasattr(client, attr):
                val = getattr(client, attr)
                if isinstance(val, (list, tuple)):
                    val = ",".join(str(v) for v in val)
                metrics[attr] = val

        client_rows.append(metrics)

        # Build prediction DataFrame
        pred_df = pd.DataFrame({
            "method": method_name,
            "client_id": client_id,
            "sample_index": np.arange(len(y_test), dtype=int),
            "y_true": y_test,
            "y_pred": preds,
        })

        # Add calendar context columns if available
        cal_context_cols = ["date", "slot_of_day", "is_effective_workday"]
        for col in cal_context_cols:
            if col in cal_test.columns:
                pred_df[col] = cal_test[col].values

        prediction_frames.append(pred_df)

    client_df = pd.DataFrame(client_rows)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    return client_df, prediction_df
