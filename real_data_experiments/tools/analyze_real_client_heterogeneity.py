"""Analyze client heterogeneity across real experiment partitions.

Inputs:
    - partition JSON files (similarity_k5, spatial_block, etc.)
    - grid tensor
    - calendar CSV
    - existing run output CSVs

Outputs:
    - results/real_data_experiments/analysis/client_heterogeneity_summary.csv
    - real_data_experiments/real_client_heterogeneity_analysis_zh.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from real_data_experiments.common.tensor_dataset import load_grid_tensor_bundle


def compute_heterogeneity_metrics(
    partition_path: str,
    tensor_path: str,
    target_channel: int = 0,
) -> pd.DataFrame:
    """Compute per-client heterogeneity metrics from partition and tensor."""
    payload = json.loads(Path(partition_path).read_text(encoding="utf-8"))
    bundle = load_grid_tensor_bundle(tensor_path)
    target_tensor = bundle.tensor[target_channel].detach().cpu().numpy().astype(np.float64)
    global_profile = np.mean(target_tensor, axis=0)

    rows: list[dict[str, Any]] = []
    for client in payload["clients"]:
        cid = client["client_id"]
        cell_ids = client["cell_ids"]
        client_series = target_tensor[cell_ids]
        client_mean_profile = np.mean(client_series, axis=0)

        rows.append({
            "client_id": int(cid),
            "num_grid_cells": int(len(cell_ids)),
            "num_samples_train_est": int(client.get("train_samples_estimate", 0)),
            "num_samples_val_est": int(client.get("val_samples_estimate", 0)),
            "num_samples_test_est": int(client.get("test_samples_estimate", 0)),
            "mean_flow": float(np.mean(client_series)),
            "std_flow": float(np.std(client_series)),
            "peak_flow": float(np.max(client_series)),
            "flow_cv": float(np.std(client_series) / (np.mean(client_series) + 1e-12)),
            "daily_profile_corr_to_global": float(np.corrcoef(client_mean_profile, global_profile)[0, 1]),
            "internal_mean_pairwise_corr": client.get("internal_mean_pairwise_corr", float("nan")),
            "source_node_count_sum": int(client.get("source_node_count_sum", 0)),
            "mean_total_flow_mean": float(client.get("mean_total_flow_mean", 0)),
            "mean_total_flow_sum": float(client.get("mean_total_flow_sum", 0)),
            "flow_cv_mean": float(client.get("flow_cv_mean", 0)),
            "lag1_autocorr_mean": float(client.get("lag1_autocorr_mean", 0)),
            "pooled_row_min": client.get("pooled_row_min"),
            "pooled_row_max": client.get("pooled_row_max"),
            "pooled_col_min": client.get("pooled_col_min"),
            "pooled_col_max": client.get("pooled_col_max"),
        })

    df = pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)
    return df


def build_summary(df: pd.DataFrame, label: str) -> dict[str, Any]:
    """Compute summary statistics (CV, gap) for heterogeneity reporting."""
    metrics = ["num_grid_cells", "mean_flow", "std_flow", "peak_flow", "flow_cv",
               "daily_profile_corr_to_global", "internal_mean_pairwise_corr"]
    summary = {"label": label, "num_clients": len(df)}
    for m in metrics:
        if m in df.columns and df[m].notna().any():
            vals = df[m].dropna().values
            summary[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{m}_std"] = round(float(np.std(vals)), 4)
            summary[f"{m}_cv"] = round(float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)), 4)
            summary[f"{m}_gap"] = round(float(np.max(vals) - np.min(vals)), 4)
    return summary


def main():
    partitions = [
        ("similarity_k5", "real_data_experiments/region_client_full_cells/partitions/similarity_k5.json"),
        ("similarity_k8", "real_data_experiments/region_client_full_cells/partitions/similarity_k8.json"),
        ("similarity_k10", "real_data_experiments/region_client_full_cells/partitions/similarity_k10.json"),
    ]
    tensor_path = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"

    output_dir = Path("results/real_data_experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    for label, path in partitions:
        try:
            df = compute_heterogeneity_metrics(path, tensor_path)
            df["partition"] = label
            all_rows.append(df)
            summaries.append(build_summary(df, label))
            print(f"[{label}] {len(df)} clients, cells: {df['num_grid_cells'].tolist()}")
        except Exception as e:
            print(f"[{label}] ERROR: {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(output_dir / "client_heterogeneity_summary.csv", index=False)
        print(f"\nOutput: {output_dir / 'client_heterogeneity_summary.csv'}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(output_dir / "client_heterogeneity_comparison.csv", index=False)
        print(f"Output: {output_dir / 'client_heterogeneity_comparison.csv'}")
        print(summary_df.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
