"""Partition active pooled-grid regions into multi-region clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .tensor_dataset import GridTensorBundle


@dataclass
class RegionPartitionResult:
    """Structured region-partition outputs for downstream experiments."""

    assignment_df: pd.DataFrame
    client_distribution_summary_df: pd.DataFrame
    non_iid_summary_df: pd.DataFrame
    feature_df: pd.DataFrame
    client_region_ids: dict[int, list[int]]
    requested_partition_method: str
    resolved_partition_method: str


def _require_partition_inputs(feature_df: pd.DataFrame, num_clients: int) -> None:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if feature_df.empty:
        raise ValueError("No active regions are available for partitioning.")
    if len(feature_df) < num_clients:
        raise ValueError(
            f"Requested {num_clients} region clients, but only {len(feature_df)} active regions are available."
        )


def _build_client_region_ids(assignment_df: pd.DataFrame) -> dict[int, list[int]]:
    return {
        int(client_id): group.sort_values(["pooled_row", "pooled_col", "region_id"])["region_id"].astype(int).tolist()
        for client_id, group in assignment_df.groupby("client_id", sort=True)
    }


def _build_non_iid_summary(client_summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "region_count": client_summary_df["region_count"].to_numpy(dtype=float),
        "source_node_count_sum": client_summary_df["source_node_count_sum"].to_numpy(dtype=float),
        "mean_total_flow_mean": client_summary_df["mean_total_flow_mean"].to_numpy(dtype=float),
        "mean_total_flow_sum": client_summary_df["mean_total_flow_sum"].to_numpy(dtype=float),
        "sample_count_estimate": client_summary_df["sample_count_estimate"].to_numpy(dtype=float),
    }
    rows: list[dict[str, Any]] = []
    for metric_name, values in metrics.items():
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))
        gap_value = float(np.max(values) - np.min(values))
        cv_value = float(std_value / mean_value) if abs(mean_value) > 1e-12 else 0.0
        rows.extend(
            [
                {"metric": metric_name, "statistic": "mean", "value": mean_value},
                {"metric": metric_name, "statistic": "std", "value": std_value},
                {"metric": metric_name, "statistic": "gap", "value": gap_value},
                {"metric": metric_name, "statistic": "cv", "value": cv_value},
            ]
        )
    return pd.DataFrame(rows)


def build_region_feature_frame(
    bundle: GridTensorBundle,
    target_channel: int = 0,
    use_active_regions_only: bool = True,
    input_length: int = 12,
    horizon: int = 1,
) -> pd.DataFrame:
    """Build region-level statistics used by partition methods."""

    regions_df = bundle.regions_df.copy()
    if use_active_regions_only:
        regions_df = regions_df[regions_df["is_active_region"]].copy()
    if regions_df.empty:
        raise ValueError("No regions remain after active-region filtering.")

    channel_tensor = bundle.tensor[int(target_channel)].detach().cpu().numpy()
    mean_total_flow = channel_tensor.mean(axis=1)
    std_total_flow = channel_tensor.std(axis=1)
    peak_total_flow = channel_tensor.max(axis=1)

    regions_df["mean_total_flow"] = regions_df["region_id"].map(lambda rid: float(mean_total_flow[int(rid)]))
    regions_df["std_total_flow"] = regions_df["region_id"].map(lambda rid: float(std_total_flow[int(rid)]))
    regions_df["peak_total_flow"] = regions_df["region_id"].map(lambda rid: float(peak_total_flow[int(rid)]))
    regions_df["valid_window_count"] = max(0, int(bundle.tensor.shape[2]) - int(input_length) - int(horizon) + 1)
    regions_df["sample_count_estimate"] = regions_df["valid_window_count"].astype(int)
    if "source_node_count" not in regions_df.columns:
        regions_df["source_node_count"] = 0
    regions_df["source_node_count"] = pd.to_numeric(regions_df["source_node_count"], errors="coerce").fillna(0).astype(int)
    return regions_df.sort_values(["region_id"]).reset_index(drop=True)


def spatial_block_partition(feature_df: pd.DataFrame, num_clients: int) -> pd.DataFrame:
    """Partition active regions by a row-major snake spatial ordering and balanced chunking."""

    _require_partition_inputs(feature_df, num_clients)
    ordered_df = feature_df.copy()
    row_order = {int(row): idx for idx, row in enumerate(sorted(ordered_df["pooled_row"].astype(int).unique().tolist()))}
    ordered_df["row_order"] = ordered_df["pooled_row"].astype(int).map(row_order)
    ordered_df["snake_col"] = ordered_df.apply(
        lambda row: int(row["pooled_col"]) if int(row["row_order"]) % 2 == 0 else -int(row["pooled_col"]),
        axis=1,
    )
    ordered_df = ordered_df.sort_values(["row_order", "snake_col", "region_id"]).reset_index(drop=True)

    client_ids = np.empty(len(ordered_df), dtype=int)
    for client_id, chunk in enumerate(np.array_split(np.arange(len(ordered_df)), num_clients)):
        if len(chunk) == 0:
            raise ValueError("spatial_block partition produced an empty client chunk.")
        client_ids[chunk] = int(client_id)

    assignment_df = ordered_df.drop(columns=["row_order", "snake_col"]).copy()
    assignment_df.insert(0, "client_id", client_ids)
    return assignment_df


def flow_kmeans_partition(feature_df: pd.DataFrame, num_clients: int, seed: int = 42) -> pd.DataFrame:
    """Partition active regions with KMeans over spatial-flow features."""

    _require_partition_inputs(feature_df, num_clients)
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "partition_method=flow_kmeans requires scikit-learn. Please install sklearn or use spatial_block."
        ) from exc

    feature_columns = ["centroid_lon", "centroid_lat", "mean_total_flow", "std_total_flow", "peak_total_flow"]
    model_df = feature_df.copy()
    matrix = model_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).to_numpy(dtype=float)
    matrix_mean = matrix.mean(axis=0, keepdims=True)
    matrix_std = matrix.std(axis=0, keepdims=True) + 1e-12
    matrix_z = (matrix - matrix_mean) / matrix_std

    labels = KMeans(n_clusters=num_clients, random_state=seed, n_init=10).fit_predict(matrix_z)
    label_counts = np.bincount(labels, minlength=num_clients)
    while np.any(label_counts == 0):
        empty_label = int(np.where(label_counts == 0)[0][0])
        donor_label = int(np.argmax(label_counts))
        donor_indices = np.where(labels == donor_label)[0]
        if donor_indices.size <= 1:
            raise ValueError("flow_kmeans could not rebalance an empty cluster.")
        labels[int(donor_indices[-1])] = empty_label
        label_counts = np.bincount(labels, minlength=num_clients)

    assignment_df = model_df.copy()
    assignment_df.insert(0, "client_id", labels.astype(int))
    return assignment_df.sort_values(["client_id", "pooled_row", "pooled_col", "region_id"]).reset_index(drop=True)


def assign_region_clients(
    bundle: GridTensorBundle,
    num_clients: int,
    partition_method: str = "spatial_block",
    use_active_regions_only: bool = True,
    target_channel: int = 0,
    input_length: int = 12,
    horizon: int = 1,
    seed: int = 42,
) -> RegionPartitionResult:
    """Assign active pooled-grid regions to multi-region clients."""

    feature_df = build_region_feature_frame(
        bundle=bundle,
        target_channel=target_channel,
        use_active_regions_only=use_active_regions_only,
        input_length=input_length,
        horizon=horizon,
    )

    requested_partition_method = str(partition_method)
    if partition_method == "spatial_block":
        assignment_df = spatial_block_partition(feature_df, num_clients=num_clients)
        resolved_partition_method = "spatial_block"
    elif partition_method == "flow_kmeans":
        assignment_df = flow_kmeans_partition(feature_df, num_clients=num_clients, seed=seed)
        resolved_partition_method = "flow_kmeans"
    else:
        raise ValueError(f"Unsupported partition_method: {partition_method}")

    client_region_ids = _build_client_region_ids(assignment_df)
    client_summary_df = (
        assignment_df.groupby("client_id", as_index=False)
        .agg(
            region_count=("region_id", "count"),
            source_node_count_sum=("source_node_count", "sum"),
            mean_total_flow_mean=("mean_total_flow", "mean"),
            mean_total_flow_sum=("mean_total_flow", "sum"),
            std_total_flow_mean=("std_total_flow", "mean"),
            peak_total_flow_max=("peak_total_flow", "max"),
            pooled_row_min=("pooled_row", "min"),
            pooled_row_max=("pooled_row", "max"),
            pooled_col_min=("pooled_col", "min"),
            pooled_col_max=("pooled_col", "max"),
            sample_count_estimate=("sample_count_estimate", "sum"),
        )
        .sort_values("client_id")
        .reset_index(drop=True)
    )
    client_summary_df["region_ids"] = client_summary_df["client_id"].map(
        lambda client_id: ",".join(str(region_id) for region_id in client_region_ids[int(client_id)])
    )
    non_iid_summary_df = _build_non_iid_summary(client_summary_df)

    assignment_columns = [
        "client_id",
        "region_id",
        "pooled_row",
        "pooled_col",
        "centroid_lon",
        "centroid_lat",
        "source_node_count",
        "mean_total_flow",
        "std_total_flow",
        "peak_total_flow",
        "is_active_region",
    ]
    available_assignment_columns = [column for column in assignment_columns if column in assignment_df.columns]

    return RegionPartitionResult(
        assignment_df=assignment_df[available_assignment_columns].copy(),
        client_distribution_summary_df=client_summary_df,
        non_iid_summary_df=non_iid_summary_df,
        feature_df=feature_df,
        client_region_ids=client_region_ids,
        requested_partition_method=requested_partition_method,
        resolved_partition_method=resolved_partition_method,
    )
