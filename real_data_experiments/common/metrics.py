"""Unified regression metrics for real-data experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EPS = 1.0
METRIC_COLUMNS = ["mse", "rmse", "mae", "mape", "smape", "r2"]


def _to_numpy(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def compute_regression_metrics(
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
    mape_eps: float = DEFAULT_EPS,
) -> dict[str, float]:
    """Compute unified regression metrics with safe handling near zero."""
    true_arr = _to_numpy(y_true)
    pred_arr = _to_numpy(y_pred)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={true_arr.shape}, y_pred={pred_arr.shape}")
    if true_arr.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    error = pred_arr - true_arr
    abs_error = np.abs(error)
    mse = float(np.mean(np.square(error)))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_error))

    denom_mape = np.maximum(np.abs(true_arr), float(mape_eps))
    mape = float(np.mean(abs_error / denom_mape) * 100.0)

    denom_smape = np.maximum(np.abs(true_arr) + np.abs(pred_arr), float(mape_eps))
    smape = float(np.mean((2.0 * abs_error) / denom_smape) * 100.0)

    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(true_arr - np.mean(true_arr))))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "r2": float(r2),
    }


def coefficient_of_variation(values: Iterable[float] | np.ndarray, eps: float = 1e-8) -> float:
    """Compute coefficient of variation."""
    arr = _to_numpy(values)
    mean_value = float(np.mean(arr))
    std_value = float(np.std(arr, ddof=0))
    return std_value / max(abs(mean_value), eps)


def summarize_metric_frame(metric_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize metrics with mean, std, and CV for each metric column."""
    rows: list[dict[str, float | str | int]] = []
    grouped = metric_df.groupby(group_cols, dropna=False) if group_cols else [((), metric_df)]
    for group_key, group_df in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, key_values)) if group_cols else {}
        for metric in METRIC_COLUMNS:
            if metric not in group_df.columns:
                continue
            values = group_df[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "cv": float(coefficient_of_variation(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def build_client_variability_frame(metric_df: pd.DataFrame, metric_name: str = "rmse") -> pd.DataFrame:
    """Build a lightweight client variability table for reviewer-facing outputs."""
    if metric_name not in metric_df.columns:
        raise ValueError(f"Metric column not found: {metric_name}")
    values = metric_df[metric_name].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return pd.DataFrame(columns=["metric", "mean", "std", "cv", "min", "max", "count"])
    return pd.DataFrame(
        [
            {
                "metric": metric_name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "cv": float(coefficient_of_variation(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": int(values.size),
            }
        ]
    )
