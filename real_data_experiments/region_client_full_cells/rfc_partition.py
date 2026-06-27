"""Inventory and partition helpers for the full-cells region-client experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.tensor_dataset import GridTensorBundle, load_grid_tensor_bundle


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation or NaN if it cannot be computed stably."""

    if left.size != right.size or left.size < 2:
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def downsample_profile(series: np.ndarray, profile_length: int = 24) -> np.ndarray:
    """Downsample one time series to a fixed-length z-score profile."""

    if series.size == 0:
        return np.zeros(profile_length, dtype=np.float64)
    z = series.astype(np.float64)
    z = z - float(np.mean(z))
    std = float(np.std(z))
    if std > 1e-12:
        z = z / std
    chunks = np.array_split(z, profile_length)
    return np.array([float(np.mean(chunk)) if chunk.size else 0.0 for chunk in chunks], dtype=np.float64)


def compute_valid_window_count(time_count: int, sequence_length: int = 12, prediction_horizon: int = 1) -> int:
    """Return the number of contiguous windows available before split slicing."""

    return max(0, int(time_count) - int(sequence_length) - int(prediction_horizon) + 1)


def build_full_cell_inventory(
    bundle: GridTensorBundle,
    sequence_length: int = 12,
    prediction_horizon: int = 1,
    target_channel: int = 0,
) -> pd.DataFrame:
    """Build an inventory over all pooled grid cells and mark valid cells."""

    valid_window_count = compute_valid_window_count(bundle.tensor.shape[2], sequence_length, prediction_horizon)
    regions_df = bundle.regions_df.copy()
    target_tensor = bundle.tensor[int(target_channel)].detach().cpu().numpy().astype(np.float64)

    rows: list[dict[str, Any]] = []
    for row in regions_df.to_dict(orient="records"):
        region_id = int(row["region_id"])
        series = target_tensor[region_id]
        flow_mean = float(np.mean(series))
        flow_std = float(np.std(series, ddof=0))
        flow_min = float(np.min(series))
        flow_max = float(np.max(series))
        flow_cv = float(flow_std / flow_mean) if abs(flow_mean) > 1e-12 else 0.0
        lag1 = safe_corr(series[1:], series[:-1]) if series.size > 1 else float("nan")
        source_node_count = int(row.get("source_node_count", 0) or 0)
        is_active_region = bool(row.get("is_active_region", False))

        invalid_reasons: list[str] = []
        if not is_active_region:
            invalid_reasons.append("inactive_region")
        if source_node_count <= 0:
            invalid_reasons.append("no_source_nodes")
        if valid_window_count <= 0:
            invalid_reasons.append("insufficient_windows")
        if not np.isfinite(series).all():
            invalid_reasons.append("non_finite_series")
        if np.allclose(series, 0.0, atol=1e-12):
            invalid_reasons.append("all_zero_series")

        rows.append(
            {
                "cell_id": region_id,
                "region_id": region_id,
                "pooled_row": row.get("pooled_row"),
                "pooled_col": row.get("pooled_col"),
                "centroid_lon": row.get("centroid_lon"),
                "centroid_lat": row.get("centroid_lat"),
                "source_node_count": source_node_count,
                "mean_total_flow": flow_mean,
                "flow_mean": flow_mean,
                "flow_std": flow_std,
                "flow_min": flow_min,
                "flow_max": flow_max,
                "flow_cv": flow_cv,
                "lag1_autocorr": lag1,
                "valid_sample_count": valid_window_count if not invalid_reasons else 0,
                "is_valid_cell": len(invalid_reasons) == 0,
                "invalid_reason": ";".join(invalid_reasons),
            }
        )

    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def build_inventory_markdown(inventory_df: pd.DataFrame, output_csv_path: str) -> str:
    """Render a concise inventory markdown report."""

    total_cells = int(len(inventory_df))
    valid_df = inventory_df[inventory_df["is_valid_cell"]].copy()
    invalid_df = inventory_df[~inventory_df["is_valid_cell"]].copy()
    lines = [
        "# 全量有效 grid cells 清单报告",
        "",
        "## 1. 目的",
        "",
        "本报告用于只读盘点 pooled-grid tensor 中的全部 grid cells，并识别可用于 full-cells region-client 实验的有效 cells。",
        "",
        "## 2. 统计摘要",
        "",
        f"- total grid cells: `{total_cells}`",
        f"- valid grid cells: `{int(len(valid_df))}`",
        f"- invalid / empty cells: `{int(len(invalid_df))}`",
        f"- inventory csv: `{output_csv_path}`",
    ]
    if not valid_df.empty:
        lines.extend(
            [
                f"- mean source_node_count: `{valid_df['source_node_count'].mean():.3f}`",
                f"- mean_total_flow mean: `{valid_df['mean_total_flow'].mean():.3f}`",
                f"- flow_cv mean: `{valid_df['flow_cv'].mean():.6f}`",
                f"- lag1_autocorr mean: `{valid_df['lag1_autocorr'].mean():.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 3. 说明",
            "",
            "- `valid cell` 定义为：active pooled region、`source_node_count > 0`、存在可用时间窗、序列有限且非全零。",
            "- 若坐标列在 sidecar 中缺失，则 CSV 会保留列名但不伪造坐标值。",
        ]
    )
    if not invalid_df.empty:
        top_reasons = invalid_df["invalid_reason"].value_counts().head(8)
        lines.extend(["", "## 4. 主要无效原因", ""])
        for reason, count in top_reasons.items():
            lines.append(f"- `{reason}`: `{int(count)}`")
    return "\n".join(lines) + "\n"


def _chunk_sizes(total_items: int, num_clients: int) -> list[int]:
    base = total_items // num_clients
    remainder = total_items % num_clients
    return [base + (1 if idx < remainder else 0) for idx in range(num_clients)]


def spatial_partition(valid_df: pd.DataFrame, num_clients: int) -> dict[int, list[int]]:
    """Assign valid cells into contiguous snake-ordered spatial blocks."""

    if len(valid_df) < num_clients * 2:
        raise ValueError(f"Need at least {num_clients * 2} valid cells so each client has >=2 cells.")
    order_df = valid_df.copy()
    row_order = {int(value): idx for idx, value in enumerate(sorted(order_df["pooled_row"].dropna().astype(int).unique().tolist()))}
    order_df["row_order"] = order_df["pooled_row"].astype(int).map(row_order)
    order_df["snake_col"] = order_df.apply(
        lambda row: int(row["pooled_col"]) if int(row["row_order"]) % 2 == 0 else -int(row["pooled_col"]),
        axis=1,
    )
    order_df = order_df.sort_values(["row_order", "snake_col", "cell_id"]).reset_index(drop=True)
    memberships: dict[int, list[int]] = {}
    cursor = 0
    for client_id, size in enumerate(_chunk_sizes(len(order_df), num_clients)):
        members = order_df.iloc[cursor : cursor + size]["cell_id"].astype(int).tolist()
        memberships[int(client_id)] = members
        cursor += size
    return memberships


def _fallback_kmeans(matrix: np.ndarray, num_clients: int, seed: int, max_iter: int = 100) -> np.ndarray:
    """A deterministic NumPy KMeans fallback used when sklearn is unavailable."""

    rng = np.random.default_rng(seed)
    init_indices = rng.choice(matrix.shape[0], size=num_clients, replace=False)
    centers = matrix[init_indices].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(num_clients):
            cluster_points = matrix[labels == cluster_id]
            if cluster_points.size > 0:
                centers[cluster_id] = cluster_points.mean(axis=0)
    return labels


def _rebalance_min_cluster_size(labels: np.ndarray, matrix: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Move samples from large clusters so every cluster has at least min_size members."""

    labels = labels.copy()
    num_clients = int(labels.max()) + 1
    while True:
        counts = np.bincount(labels, minlength=num_clients)
        small_clusters = [idx for idx, count in enumerate(counts) if count < min_size]
        if not small_clusters:
            return labels
        donor = int(np.argmax(counts))
        if counts[donor] <= min_size:
            raise ValueError("Could not rebalance clusters to satisfy min_size.")
        donor_indices = np.where(labels == donor)[0]
        donor_center = matrix[donor_indices].mean(axis=0)
        ranked = sorted(
            donor_indices.tolist(),
            key=lambda idx: float(np.linalg.norm(matrix[idx] - donor_center)),
            reverse=True,
        )
        labels[ranked[0]] = int(small_clusters[0])


def _similarity_feature_matrix(valid_df: pd.DataFrame, bundle: GridTensorBundle, target_channel: int = 0) -> tuple[np.ndarray, list[int]]:
    """Build similarity features from stats plus downsampled normalized time profiles."""

    cell_ids = valid_df["cell_id"].astype(int).tolist()
    target_tensor = bundle.tensor[int(target_channel)].detach().cpu().numpy().astype(np.float64)
    stat_matrix = valid_df[["flow_mean", "flow_std", "flow_cv", "lag1_autocorr"]].to_numpy(dtype=np.float64)
    stat_mean = stat_matrix.mean(axis=0, keepdims=True)
    stat_std = stat_matrix.std(axis=0, keepdims=True) + 1e-12
    stat_z = (stat_matrix - stat_mean) / stat_std
    profile_matrix = np.stack([downsample_profile(target_tensor[cell_id]) for cell_id in cell_ids], axis=0)
    return np.concatenate([stat_z, profile_matrix], axis=1), cell_ids


def similarity_partition(
    valid_df: pd.DataFrame,
    bundle: GridTensorBundle,
    num_clients: int,
    seed: int = 2026,
    target_channel: int = 0,
) -> dict[int, list[int]]:
    """Assign valid cells into similarity-based clients using pre-training features only."""

    if len(valid_df) < num_clients * 2:
        raise ValueError(f"Need at least {num_clients * 2} valid cells so each client has >=2 cells.")
    matrix, cell_ids = _similarity_feature_matrix(valid_df, bundle=bundle, target_channel=target_channel)
    try:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=num_clients, random_state=seed, n_init=10).fit_predict(matrix)
    except Exception:
        labels = _fallback_kmeans(matrix, num_clients=num_clients, seed=seed)
    labels = _rebalance_min_cluster_size(np.asarray(labels, dtype=int), matrix, min_size=2)
    memberships: dict[int, list[int]] = {}
    for client_id in range(num_clients):
        members = [cell_ids[idx] for idx in range(len(cell_ids)) if int(labels[idx]) == client_id]
        memberships[int(client_id)] = sorted(int(cell_id) for cell_id in members)
    return memberships


def _client_internal_mean_corr(cell_ids: list[int], bundle: GridTensorBundle, target_channel: int = 0) -> float:
    if len(cell_ids) < 2:
        return float("nan")
    target_tensor = bundle.tensor[int(target_channel)].detach().cpu().numpy().astype(np.float64)
    correlations: list[float] = []
    for idx, left_id in enumerate(cell_ids):
        for right_id in cell_ids[idx + 1 :]:
            corr = safe_corr(target_tensor[left_id], target_tensor[right_id])
            if np.isfinite(corr):
                correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else float("nan")


def build_partition_payload(
    inventory_df: pd.DataFrame,
    memberships: dict[int, list[int]],
    partition_mode: str,
    num_clients: int,
    bundle: GridTensorBundle,
    seed: int = 2026,
    method: str = "spatial_block",
    sequence_length: int = 12,
    prediction_horizon: int = 1,
) -> dict[str, Any]:
    """Build a JSON-serializable partition payload with client summaries."""

    valid_ids = sorted(inventory_df[inventory_df["is_valid_cell"]]["cell_id"].astype(int).tolist())
    assigned_ids = sorted(int(cell_id) for cells in memberships.values() for cell_id in cells)
    if valid_ids != assigned_ids:
        raise ValueError("Partition does not cover exactly the valid cells.")

    client_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    inventory_index = inventory_df.set_index("cell_id")
    train_ratio = 0.7
    val_ratio = 0.15
    for client_id, cell_ids in sorted(memberships.items()):
        client_df = inventory_index.loc[cell_ids].reset_index()
        valid_sample_count = int(client_df["valid_sample_count"].iloc[0]) if not client_df.empty else 0
        train_samples = int(valid_sample_count * train_ratio * len(cell_ids))
        val_samples = int(valid_sample_count * val_ratio * len(cell_ids))
        test_samples = int(valid_sample_count * (1.0 - train_ratio - val_ratio) * len(cell_ids))
        internal_corr = _client_internal_mean_corr(cell_ids, bundle=bundle)
        client_rows.append(
            {
                "client_id": int(client_id),
                "cell_count": int(len(cell_ids)),
                "cell_ids": [int(cell_id) for cell_id in cell_ids],
                "source_node_count_sum": int(client_df["source_node_count"].sum()),
                "mean_total_flow_mean": float(client_df["mean_total_flow"].mean()),
                "mean_total_flow_sum": float(client_df["mean_total_flow"].sum()),
                "flow_cv_mean": float(client_df["flow_cv"].mean()),
                "lag1_autocorr_mean": float(client_df["lag1_autocorr"].mean()),
                "train_samples_estimate": train_samples,
                "val_samples_estimate": val_samples,
                "test_samples_estimate": test_samples,
                "pooled_row_min": int(client_df["pooled_row"].min()) if "pooled_row" in client_df.columns else None,
                "pooled_row_max": int(client_df["pooled_row"].max()) if "pooled_row" in client_df.columns else None,
                "pooled_col_min": int(client_df["pooled_col"].min()) if "pooled_col" in client_df.columns else None,
                "pooled_col_max": int(client_df["pooled_col"].max()) if "pooled_col" in client_df.columns else None,
                "internal_mean_pairwise_corr": internal_corr,
            }
        )
        for _, row in client_df.iterrows():
            assignment_rows.append(
                {
                    "client_id": int(client_id),
                    "cell_id": int(row["cell_id"]),
                    "pooled_row": None if pd.isna(row["pooled_row"]) else int(row["pooled_row"]),
                    "pooled_col": None if pd.isna(row["pooled_col"]) else int(row["pooled_col"]),
                    "centroid_lon": None if pd.isna(row["centroid_lon"]) else float(row["centroid_lon"]),
                    "centroid_lat": None if pd.isna(row["centroid_lat"]) else float(row["centroid_lat"]),
                    "source_node_count": int(row["source_node_count"]),
                    "mean_total_flow": float(row["mean_total_flow"]),
                }
            )

    procedure = (
        "spatial snake-order contiguous block partition over pooled rows/cols"
        if partition_mode == "spatial"
        else "kmeans over pre-training flow statistics and downsampled normalized time-series profiles"
    )
    return {
        "partition_mode": partition_mode,
        "num_clients": int(num_clients),
        "method": method,
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "prediction_horizon": int(prediction_horizon),
        "valid_cell_count": int(len(valid_ids)),
        "cell_membership": {str(client_id): [int(cell_id) for cell_id in cell_ids] for client_id, cell_ids in sorted(memberships.items())},
        "clients": client_rows,
        "assignments": assignment_rows,
        "cluster_procedure": procedure,
    }


def write_partition_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Persist one partition payload as JSON."""

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI for partition generation."""

    parser = argparse.ArgumentParser(description="Generate full-cells client partitions.")
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument(
        "--regions-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv",
    )
    parser.add_argument("--partition-mode", choices=["spatial", "similarity"], required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--method", type=str, default="kmeans")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-file", type=str, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = load_grid_tensor_bundle(args.tensor_path, args.regions_path)
    inventory_df = build_full_cell_inventory(bundle)
    valid_df = inventory_df[inventory_df["is_valid_cell"]].copy()
    if args.partition_mode == "spatial":
        memberships = spatial_partition(valid_df, num_clients=args.num_clients)
        partition_method = "spatial_block"
    else:
        memberships = similarity_partition(valid_df, bundle=bundle, num_clients=args.num_clients, seed=args.seed)
        partition_method = args.method
    payload = build_partition_payload(
        inventory_df=inventory_df,
        memberships=memberships,
        partition_mode=args.partition_mode,
        num_clients=args.num_clients,
        bundle=bundle,
        seed=args.seed,
        method=partition_method,
    )
    write_partition_payload(payload, args.output_file)
    print(f"[partition_mode] {args.partition_mode}")
    print(f"[num_clients] {args.num_clients}")
    print(f"[output_file] {resolve_path(args.output_file)}")


if __name__ == "__main__":
    main()
