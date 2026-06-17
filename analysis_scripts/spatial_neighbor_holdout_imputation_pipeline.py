from __future__ import annotations

import argparse
import gc
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-6
SCENARIO_ID = "snh_mix"
MECHANISM = "spatial_neighbor_holdout"
METHOD_ORDER = [
    "mean_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "function_curve_fit",
    "road_topology_neighbor_fill",
    "correlation_topology_neighbor_fill",
    "adaptive_spatio_temporal_fill",
    "adaptive_topology_function_hybrid",
]
METHOD_DIR_ABBR = {
    "mean_fill": "mf",
    "forward_fill": "ff",
    "historical_linear_extrapolation": "hle",
    "function_curve_fit": "fcf",
    "road_topology_neighbor_fill": "rtn",
    "correlation_topology_neighbor_fill": "ctn",
    "adaptive_spatio_temporal_fill": "ast",
    "adaptive_topology_function_hybrid": "atfh",
}
FLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]
LENGTH_GROUP_LABELS = ["short", "mid", "long"]
NEIGHBOR_COVERAGE_GROUPS = ["low", "mid", "high"]
CONSTRAINT_LEVELS = ["strict_anchor", "relaxed_anchor", "weak_neighbor_available", "none"]


@dataclass(frozen=True)
class StagePaths:
    scenario_root: Path
    miss_root: Path
    imp_root: Path
    masks_root: Path
    miss_data_root: Path
    imp_data_root: Path
    manifests_root: Path
    summaries_root: Path
    figures_root: Path
    audits_root: Path
    run_config_path: Path
    run_commands_path: Path
    input_check_csv_path: Path
    input_check_json_path: Path
    chunk_status_path: Path
    detail_path: Path
    summary_exclude_warmup_path: Path
    summary_by_length_path: Path
    summary_by_flow_path: Path
    summary_by_neighbor_coverage_path: Path
    summary_by_constraint_level_path: Path
    audit_json_path: Path
    audit_md_path: Path
    flow_threshold_path: Path


@dataclass
class ChunkPrepared:
    file_name: str
    chunk_index: int
    day_index: int
    clean_df: pd.DataFrame
    missing_df: pd.DataFrame
    sorted_original_rows: np.ndarray
    clean_matrix: np.ndarray
    missing_matrix: np.ndarray
    mask_matrix: np.ndarray
    node_indices: np.ndarray
    slot_indices: np.ndarray
    true_values: np.ndarray
    flow_group_ids: np.ndarray
    length_groups: np.ndarray
    available_neighbor_counts: np.ndarray
    neighbor_scopes: np.ndarray
    spatial_constraint_levels: np.ndarray
    is_warmup: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spatial neighbor holdout imputation and summary generation.")
    parser.add_argument("--stage", required=True, choices=["prepare", "impute", "summarize", "audit", "all"])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--scenario_dir", required=True, type=Path)
    parser.add_argument("--topology_file", required=True, type=Path)
    parser.add_argument("--missing_rates", required=True, type=str)
    parser.add_argument("--methods", required=True, type=str)
    parser.add_argument("--history_days", required=True, type=int)
    parser.add_argument("--correlation_history_days", required=True, type=int)
    parser.add_argument("--neighbor_scope", required=True, type=int)
    parser.add_argument("--allow_current_time_neighbors", required=True, type=str)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--node_col", required=True, type=str)
    parser.add_argument("--time_col", required=True, type=str)
    parser.add_argument("--period", required=True, type=int)
    return parser.parse_args()


def parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {raw}")


def parse_rates(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("missing_rates is empty")
    return values


def parse_methods(raw: str) -> list[str]:
    methods = [token.strip() for token in raw.split(",") if token.strip()]
    unique_methods: list[str] = []
    for method in methods:
        if method == "zero_fill":
            raise ValueError("zero_fill is not allowed in snh_mix")
        if method not in METHOD_ORDER:
            raise ValueError(f"unsupported method: {method}")
        if method not in unique_methods:
            unique_methods.append(method)
    if unique_methods != METHOD_ORDER:
        raise ValueError(f"--methods must exactly match formal snh methods: {METHOD_ORDER}")
    return unique_methods


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def extract_day_index(file_name: str) -> int:
    return int(Path(file_name).stem.rsplit("_", 1)[-1])


def scenario_rate_tag(rate: float) -> str:
    return f"snh_r{int(round(rate * 100)):02d}_mix_s42"


def imputed_dir_name(rate: float, method: str) -> str:
    return f"{scenario_rate_tag(rate)}_m_{METHOD_DIR_ABBR[method]}"


def build_paths(scenario_dir: Path) -> StagePaths:
    imp_root = scenario_dir / "imp"
    return StagePaths(
        scenario_root=scenario_dir,
        miss_root=scenario_dir / "miss_set",
        imp_root=imp_root,
        masks_root=scenario_dir / "miss_set" / "masks",
        miss_data_root=scenario_dir / "miss_set" / "miss_data",
        imp_data_root=imp_root / "imp_data",
        manifests_root=imp_root / "manifests",
        summaries_root=imp_root / "summaries",
        figures_root=imp_root / "figures",
        audits_root=imp_root / "audits",
        run_config_path=imp_root / "run_config_imputation.json",
        run_commands_path=imp_root / "run_commands_imputation.txt",
        input_check_csv_path=imp_root / "manifests" / "snh_imputation_input_check.csv",
        input_check_json_path=imp_root / "manifests" / "snh_imputation_input_check.json",
        chunk_status_path=imp_root / "manifests" / "snh_imputed_chunk_status.csv",
        detail_path=imp_root / "summaries" / "snh_imputation_quality_detail.csv",
        summary_exclude_warmup_path=imp_root / "summaries" / "snh_imputation_quality_summary_exclude_warmup.csv",
        summary_by_length_path=imp_root / "summaries" / "snh_imputation_quality_by_length_group.csv",
        summary_by_flow_path=imp_root / "summaries" / "snh_imputation_quality_by_flow_group.csv",
        summary_by_neighbor_coverage_path=imp_root / "summaries" / "snh_imputation_quality_by_neighbor_coverage.csv",
        summary_by_constraint_level_path=imp_root / "summaries" / "snh_imputation_quality_by_constraint_level.csv",
        audit_json_path=imp_root / "audits" / "snh_spatial_imputation_audit.json",
        audit_md_path=imp_root / "audits" / "snh_spatial_imputation_audit_zh.md",
        flow_threshold_path=imp_root / "manifests" / "snh_flow_group_thresholds.json",
    )


def load_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no input parquet files found under {input_dir}")
    return files


def build_row_layout(df: pd.DataFrame, node_col: str, time_col: str, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = df[[node_col, time_col]].copy()
    local["row_index"] = np.arange(len(local), dtype=np.int64)
    local = local.sort_values([node_col, time_col], kind="mergesort").reset_index(drop=True)
    node_ids = local[node_col].astype(np.int64).to_numpy(copy=False).reshape(-1, period)[:, 0]
    sorted_original_rows = local["row_index"].to_numpy(dtype=np.int64, copy=False)
    inverse_sort_idx = np.empty(len(local), dtype=np.int64)
    inverse_sort_idx[sorted_original_rows] = np.arange(len(local), dtype=np.int64)
    return node_ids, sorted_original_rows, inverse_sort_idx


def build_topology_candidates(topology_file: Path, canonical_node_ids: np.ndarray, max_scope: int) -> dict[int, dict[str, np.ndarray]]:
    topo_df = pd.read_csv(topology_file, usecols=["起始节点ID", "结束节点ID", "长度"])
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(canonical_node_ids.tolist())}
    first_hop_sets: list[set[int]] = [set() for _ in range(len(canonical_node_ids))]
    first_hop_lengths: list[dict[int, float]] = [dict() for _ in range(len(canonical_node_ids))]
    for start, end, length in topo_df.itertuples(index=False, name=None):
        start_idx = node_to_idx.get(int(start))
        end_idx = node_to_idx.get(int(end))
        if start_idx is None or end_idx is None or start_idx == end_idx:
            continue
        safe_length = float(length) if pd.notna(length) and float(length) > 0 else 1.0
        first_hop_sets[start_idx].add(end_idx)
        first_hop_sets[end_idx].add(start_idx)
        prev = first_hop_lengths[start_idx].get(end_idx)
        if prev is None or safe_length < prev:
            first_hop_lengths[start_idx][end_idx] = safe_length
        prev = first_hop_lengths[end_idx].get(start_idx)
        if prev is None or safe_length < prev:
            first_hop_lengths[end_idx][start_idx] = safe_length
    second_hop_lengths: list[dict[int, float]] = [dict() for _ in range(len(canonical_node_ids))]
    if max_scope >= 2:
        for node_idx, neighbors in enumerate(first_hop_sets):
            for mid_idx in neighbors:
                length_to_mid = first_hop_lengths[node_idx][mid_idx]
                for second_idx in first_hop_sets[mid_idx]:
                    if second_idx == node_idx or second_idx in neighbors:
                        continue
                    total_length = length_to_mid + first_hop_lengths[mid_idx][second_idx]
                    prev = second_hop_lengths[node_idx].get(second_idx)
                    if prev is None or total_length < prev:
                        second_hop_lengths[node_idx][second_idx] = total_length
    output: dict[int, dict[str, np.ndarray]] = {}
    for node_idx in range(len(canonical_node_ids)):
        first_neighbors = np.asarray(sorted(first_hop_sets[node_idx]), dtype=np.int64)
        first_lengths = np.asarray(
            [float(first_hop_lengths[node_idx][neighbor]) for neighbor in first_neighbors.tolist()],
            dtype=np.float32,
        )
        if len(first_neighbors) > 0:
            neighbors = first_neighbors
            lengths = first_lengths
            scope = 1
        else:
            second_neighbors = np.asarray(sorted(second_hop_lengths[node_idx].keys()), dtype=np.int64)
            second_lengths = np.asarray(
                [float(second_hop_lengths[node_idx][neighbor]) for neighbor in second_neighbors.tolist()],
                dtype=np.float32,
            )
            neighbors = second_neighbors
            lengths = second_lengths
            scope = 2 if len(second_neighbors) > 0 else 0
        output[node_idx] = {
            "neighbors": neighbors,
            "lengths": lengths,
            "scope": np.asarray([scope], dtype=np.int64),
        }
    return output


def compute_flow_group_thresholds(input_files: list[Path], target_col: str, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    sample_values: list[np.ndarray] = []
    for file_path in input_files[:8]:
        series = pd.read_parquet(file_path, columns=[target_col])[target_col].dropna().astype(np.float32).to_numpy(copy=False)
        if len(series) > 200_000:
            step = max(len(series) // 200_000, 1)
            series = series[::step]
        sample_values.append(series)
    merged = np.concatenate(sample_values).astype(np.float32, copy=False)
    q33, q66 = np.quantile(merged, [1 / 3, 2 / 3])
    payload = {
        "estimation_method": "sampled_global_clean_quantiles",
        "quantiles": {
            "q33": float(q33),
            "q66": float(q66),
        },
        "groups": {
            "low_flow": {"min_inclusive": 0.0, "max_exclusive": float(q33)},
            "mid_flow": {"min_inclusive": float(q33), "max_exclusive": float(q66)},
            "high_flow": {"min_inclusive": float(q66), "max_inclusive": float(np.max(merged))},
        },
    }
    write_json(output_path, payload)
    return payload


def build_flow_group_ids(true_values: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    q33 = float(thresholds["quantiles"]["q33"])
    q66 = float(thresholds["quantiles"]["q66"])
    group_ids = np.full(len(true_values), 2, dtype=np.int8)
    group_ids[true_values < q66] = 1
    group_ids[true_values < q33] = 0
    return group_ids


def prepare_chunk(
    *,
    clean_file: Path,
    missing_file: Path,
    mask_file: Path,
    node_col: str,
    time_col: str,
    target_col: str,
    period: int,
    thresholds: dict[str, Any],
    warmup_days: int,
) -> ChunkPrepared:
    clean_df = pd.read_parquet(clean_file)
    missing_df = pd.read_parquet(missing_file)
    mask_df = pd.read_parquet(mask_file)
    node_ids, sorted_original_rows, inverse_sort_idx = build_row_layout(clean_df, node_col, time_col, period)
    clean_sorted = clean_df.iloc[sorted_original_rows][target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1, period)
    missing_sorted = missing_df.iloc[sorted_original_rows][target_col].to_numpy(dtype=np.float32, copy=False).reshape(-1, period)
    mask_positions = inverse_sort_idx[mask_df["row_index"].to_numpy(dtype=np.int64, copy=False)]
    mask_flat = np.zeros(clean_sorted.size, dtype=bool)
    mask_flat[mask_positions] = True
    mask_matrix = mask_flat.reshape(clean_sorted.shape)
    length_group_flat = np.full(clean_sorted.size, "", dtype="<U8")
    length_group_flat[mask_positions] = mask_df["length_group"].astype(str).to_numpy(copy=False)
    available_neighbor_flat = np.zeros(clean_sorted.size, dtype=np.int32)
    available_neighbor_flat[mask_positions] = mask_df["available_neighbor_count"].astype(np.int32).to_numpy(copy=False)
    neighbor_scope_flat = np.zeros(clean_sorted.size, dtype=np.int8)
    neighbor_scope_flat[mask_positions] = mask_df["neighbor_scope"].astype(np.int8).to_numpy(copy=False)
    constraint_level_flat = np.full(clean_sorted.size, "strict_anchor", dtype="<U32")
    if "spatial_constraint_level" in mask_df.columns:
        constraint_level_flat[mask_positions] = mask_df["spatial_constraint_level"].astype(str).to_numpy(copy=False)
    node_indices, slot_indices = np.where(mask_matrix)
    true_values = clean_sorted[mask_matrix]
    flow_group_ids = build_flow_group_ids(true_values, thresholds)
    chunk_index = extract_day_index(clean_file.name)
    return ChunkPrepared(
        file_name=clean_file.name,
        chunk_index=chunk_index,
        day_index=chunk_index,
        clean_df=clean_df,
        missing_df=missing_df,
        sorted_original_rows=sorted_original_rows,
        clean_matrix=clean_sorted,
        missing_matrix=missing_sorted,
        mask_matrix=mask_matrix,
        node_indices=node_indices.astype(np.int32, copy=False),
        slot_indices=slot_indices.astype(np.int16, copy=False),
        true_values=true_values.astype(np.float32, copy=False),
        flow_group_ids=flow_group_ids,
        length_groups=length_group_flat.reshape(clean_sorted.shape)[mask_matrix],
        available_neighbor_counts=available_neighbor_flat.reshape(clean_sorted.shape)[mask_matrix],
        neighbor_scopes=neighbor_scope_flat.reshape(clean_sorted.shape)[mask_matrix],
        spatial_constraint_levels=constraint_level_flat.reshape(clean_sorted.shape)[mask_matrix],
        is_warmup=bool(chunk_index < warmup_days),
    )


def build_current_day_forward_fill(missing_matrix: np.ndarray, previous_last_state: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(np.isnan(imputed), dtype=bool)
    if previous_last_state is not None:
        first_missing = np.isnan(imputed[:, 0])
        imputed[first_missing, 0] = previous_last_state[first_missing]
    for slot in range(1, imputed.shape[1]):
        missing_here = np.isnan(imputed[:, slot])
        if np.any(missing_here):
            imputed[missing_here, slot] = imputed[missing_here, slot - 1]
    unresolved = np.isnan(imputed)
    if np.any(unresolved):
        imputed[unresolved] = 0.0
        fallback_mask[unresolved] = True
    return imputed.astype(np.float32, copy=False), fallback_mask


def historical_stack(history: deque[np.ndarray], last_n: int | None = None) -> np.ndarray | None:
    if not history:
        return None
    values = list(history)[-last_n:] if last_n is not None else list(history)
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def impute_mean_fill(missing_matrix: np.ndarray, mask_matrix: np.ndarray, history: deque[np.ndarray], forward_fill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback = np.zeros_like(mask_matrix, dtype=bool)
    stack = historical_stack(history)
    if stack is None or stack.shape[0] == 0:
        imputed[mask_matrix] = forward_fill[mask_matrix]
        fallback[mask_matrix] = True
        return imputed, fallback
    available = np.isfinite(stack)
    values = np.where(available, stack, 0.0)
    same_slot_count = available.sum(axis=0)
    same_slot_sum = values.sum(axis=0, dtype=np.float32)
    same_slot_mean = np.full_like(imputed, np.nan, dtype=np.float32)
    np.divide(same_slot_sum, same_slot_count, out=same_slot_mean, where=same_slot_count > 0)
    use_primary = mask_matrix & (same_slot_count > 0)
    imputed[use_primary] = same_slot_mean[use_primary]
    remaining = mask_matrix & ~use_primary
    if np.any(remaining):
        node_count = available.sum(axis=(0, 2))
        node_sum = values.sum(axis=(0, 2), dtype=np.float32)
        node_mean = np.full(imputed.shape[0], np.nan, dtype=np.float32)
        np.divide(node_sum, node_count, out=node_mean, where=node_count > 0)
        node_matrix = np.broadcast_to(node_mean[:, None], imputed.shape)
        use_node = remaining & np.broadcast_to((node_count > 0)[:, None], imputed.shape)
        imputed[use_node] = node_matrix[use_node]
        remaining = remaining & ~use_node
        if np.any(remaining):
            slot_count = available.sum(axis=(0, 1))
            slot_sum = values.sum(axis=(0, 1), dtype=np.float32)
            slot_mean = np.full(imputed.shape[1], np.nan, dtype=np.float32)
            np.divide(slot_sum, slot_count, out=slot_mean, where=slot_count > 0)
            slot_matrix = np.broadcast_to(slot_mean[None, :], imputed.shape)
            use_slot = remaining & np.broadcast_to((slot_count > 0)[None, :], imputed.shape)
            imputed[use_slot] = slot_matrix[use_slot]
            remaining = remaining & ~use_slot
        if np.any(remaining):
            global_valid = available.sum()
            if global_valid > 0:
                global_mean = float(values.sum(dtype=np.float32) / float(global_valid))
                imputed[remaining] = global_mean
            else:
                imputed[remaining] = forward_fill[remaining]
            fallback[mask_matrix & ~use_primary] = True
    return imputed.astype(np.float32, copy=False), fallback


def impute_historical_linear(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    day_index: int,
    history: deque[np.ndarray],
    mean_fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback = np.zeros_like(mask_matrix, dtype=bool)
    stack = historical_stack(history)
    if stack is None or stack.shape[0] < 2:
        imputed[mask_matrix] = mean_fill[mask_matrix]
        fallback[mask_matrix] = True
        return imputed, fallback
    history_count = stack.shape[0]
    x = np.arange(day_index - history_count, day_index, dtype=np.float32)
    n = float(history_count)
    sx = float(np.sum(x))
    sxx = float(np.sum(x * x))
    sy = np.sum(stack, axis=0, dtype=np.float32)
    sxy = np.tensordot(x, stack, axes=(0, 0)).astype(np.float32)
    denominator = (n * sxx) - (sx * sx)
    if abs(denominator) <= EPSILON:
        imputed[mask_matrix] = mean_fill[mask_matrix]
        fallback[mask_matrix] = True
        return imputed, fallback
    slope = ((n * sxy) - (sx * sy)) / denominator
    intercept = (sy - (slope * sx)) / n
    predicted = (slope * float(day_index)) + intercept
    imputed[mask_matrix] = predicted[mask_matrix]
    nan_mask = np.isnan(imputed) & mask_matrix
    if np.any(nan_mask):
        imputed[nan_mask] = mean_fill[nan_mask]
        fallback[nan_mask] = True
    return imputed.astype(np.float32, copy=False), fallback


def build_fourier_basis(period: int, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    time_index = np.arange(period, dtype=np.float32)
    columns = [np.ones(period, dtype=np.float32)]
    for harmonic in range(1, order + 1):
        angle = (2.0 * math.pi * harmonic * time_index) / float(period)
        columns.append(np.cos(angle).astype(np.float32))
        columns.append(np.sin(angle).astype(np.float32))
    basis = np.column_stack(columns).astype(np.float32)
    pinv = np.linalg.pinv(basis).astype(np.float32)
    return basis, pinv


def impute_function_curve_fit(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    history: deque[np.ndarray],
    basis: np.ndarray,
    pinv: np.ndarray,
    mean_fill: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback = np.zeros_like(mask_matrix, dtype=bool)
    stack = historical_stack(history)
    if stack is None or stack.shape[0] == 0:
        imputed[mask_matrix] = mean_fill[mask_matrix]
        fallback[mask_matrix] = True
        return imputed, fallback, mean_fill.copy()
    mean_profile = np.mean(stack, axis=0, dtype=np.float32)
    coeff = mean_profile @ pinv.T
    predicted = coeff @ basis.T
    imputed[mask_matrix] = predicted[mask_matrix]
    nan_mask = np.isnan(imputed) & mask_matrix
    if np.any(nan_mask):
        imputed[nan_mask] = mean_fill[nan_mask]
        fallback[nan_mask] = True
    return imputed.astype(np.float32, copy=False), fallback, predicted.astype(np.float32, copy=False)


def correlation_for_pair(target_series: np.ndarray, neighbor_series: np.ndarray) -> float:
    valid = np.isfinite(target_series) & np.isfinite(neighbor_series)
    if int(valid.sum()) < 8:
        return 0.0
    x = target_series[valid].astype(np.float64, copy=False)
    y = neighbor_series[valid].astype(np.float64, copy=False)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= EPSILON or y_std <= EPSILON:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def build_target_neighbor_cache(
    *,
    target_idx: int,
    history_stack_corr: np.ndarray | None,
    topology_candidates: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    candidate_info = topology_candidates[target_idx]
    neighbors = candidate_info["neighbors"]
    lengths = candidate_info["lengths"]
    scope = int(candidate_info["scope"][0]) if len(candidate_info["scope"]) else 0
    if history_stack_corr is None or history_stack_corr.shape[0] == 0 or len(neighbors) == 0:
        return {
            "neighbors": neighbors,
            "lengths": lengths,
            "scope": scope,
            "corr": np.zeros(len(neighbors), dtype=np.float32),
            "mu_i": 0.0,
            "sigma_i": 1.0,
            "mu_j": np.zeros(len(neighbors), dtype=np.float32),
            "sigma_j": np.ones(len(neighbors), dtype=np.float32),
        }
    target_hist = history_stack_corr[:, target_idx, :].reshape(-1).astype(np.float32, copy=False)
    mu_i = float(np.nanmean(target_hist)) if np.isfinite(target_hist).any() else 0.0
    sigma_i = float(np.nanstd(target_hist)) if np.isfinite(target_hist).any() else 1.0
    corr_values = np.zeros(len(neighbors), dtype=np.float32)
    mu_j = np.zeros(len(neighbors), dtype=np.float32)
    sigma_j = np.ones(len(neighbors), dtype=np.float32)
    for idx, neighbor_idx in enumerate(neighbors.tolist()):
        neighbor_hist = history_stack_corr[:, int(neighbor_idx), :].reshape(-1).astype(np.float32, copy=False)
        corr_values[idx] = correlation_for_pair(target_hist, neighbor_hist)
        mu_j[idx] = float(np.nanmean(neighbor_hist)) if np.isfinite(neighbor_hist).any() else 0.0
        sigma_j[idx] = float(np.nanstd(neighbor_hist)) if np.isfinite(neighbor_hist).any() else 1.0
    return {
        "neighbors": neighbors,
        "lengths": lengths,
        "scope": scope,
        "corr": corr_values,
        "mu_i": mu_i,
        "sigma_i": sigma_i,
        "mu_j": mu_j,
        "sigma_j": sigma_j,
    }


def spatial_current_prediction(
    *,
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    mean_fill: np.ndarray,
    node_indices: np.ndarray,
    slot_indices: np.ndarray,
    topology_candidates: dict[int, dict[str, np.ndarray]],
    history_stack_corr: np.ndarray | None,
    correlation_mode: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imputed = mean_fill.copy()
    fallback = mask_matrix.copy()
    neighbor_count_matrix = np.zeros_like(mean_fill, dtype=np.float32)
    coverage_matrix = np.zeros_like(mean_fill, dtype=np.float32)
    positive_corr_matrix = np.zeros_like(mean_fill, dtype=np.float32)
    success_matrix = np.zeros_like(mask_matrix, dtype=bool)
    unique_nodes = np.unique(node_indices)
    target_cache: dict[int, dict[str, Any]] = {}
    for target_idx in unique_nodes.tolist():
        selector = node_indices == target_idx
        slots = slot_indices[selector]
        cache = target_cache.get(int(target_idx))
        if cache is None:
            cache = build_target_neighbor_cache(
                target_idx=int(target_idx),
                history_stack_corr=history_stack_corr,
                topology_candidates=topology_candidates,
            )
            target_cache[int(target_idx)] = cache
        neighbors = cache["neighbors"]
        if len(neighbors) == 0:
            continue
        current_values = missing_matrix[neighbors][:, slots]
        available = np.isfinite(current_values)
        available_count = available.sum(axis=0).astype(np.float32)
        total_neighbor_count = float(len(neighbors))
        neighbor_count_matrix[int(target_idx), slots] = available_count
        coverage_matrix[int(target_idx), slots] = available_count / max(total_neighbor_count, 1.0)
        if correlation_mode:
            positive = available & (cache["corr"][:, None] > 0.05)
            positive_count = positive.sum(axis=0)
            if np.any(positive_count > 0):
                positive_corr_values = np.where(positive, cache["corr"][:, None], 0.0)
                positive_corr_matrix[int(target_idx), slots] = np.divide(
                    positive_corr_values.sum(axis=0),
                    np.maximum(positive_count, 1),
                    dtype=np.float32,
                )
            base_weights = np.where(
                cache["corr"] > 0.05,
                np.power(np.maximum(cache["corr"], 0.0), 1.0) / np.power(cache["lengths"] + EPSILON, 1.0),
                0.0,
            ).astype(np.float32)
            weight_matrix = base_weights[:, None] * positive.astype(np.float32)
            z_values = (current_values - cache["mu_j"][:, None]) / (cache["sigma_j"][:, None] + EPSILON)
            numerator = np.sum(weight_matrix * z_values, axis=0, dtype=np.float32)
            denominator = np.sum(weight_matrix, axis=0, dtype=np.float32)
            success = denominator > EPSILON
            predictions = np.full(len(slots), np.nan, dtype=np.float32)
            predictions[success] = cache["mu_i"] + (max(cache["sigma_i"], EPSILON) * (numerator[success] / denominator[success]))
        else:
            base_weights = (1.0 / (cache["lengths"] + EPSILON)).astype(np.float32)
            weight_matrix = base_weights[:, None] * available.astype(np.float32)
            numerator = np.sum(weight_matrix * np.where(available, current_values, 0.0), axis=0, dtype=np.float32)
            denominator = np.sum(weight_matrix, axis=0, dtype=np.float32)
            success = denominator > EPSILON
            predictions = np.full(len(slots), np.nan, dtype=np.float32)
            predictions[success] = numerator[success] / denominator[success]
        if np.any(success):
            imputed[int(target_idx), slots[success]] = predictions[success]
            fallback[int(target_idx), slots[success]] = False
            success_matrix[int(target_idx), slots[success]] = True
    return (
        imputed.astype(np.float32, copy=False),
        fallback,
        neighbor_count_matrix,
        coverage_matrix,
        positive_corr_matrix,
        success_matrix,
    )


def estimate_historical_errors(
    *,
    target_idx: int,
    history_stack_corr: np.ndarray | None,
    topology_candidates: dict[int, dict[str, np.ndarray]],
    basis: np.ndarray,
    pinv: np.ndarray,
) -> tuple[float, float]:
    if history_stack_corr is None or history_stack_corr.shape[0] < 2:
        return 1.0, 1.0
    holdout = history_stack_corr[-1]
    train = history_stack_corr[:-1]
    target_actual = holdout[target_idx]
    target_train = train[:, target_idx, :]
    mean_profile = np.mean(target_train, axis=0, dtype=np.float32)
    coeff = mean_profile @ pinv.T
    func_pred = coeff @ basis.T
    func_error = float(np.sqrt(np.mean((func_pred - target_actual) ** 2)))
    cache = build_target_neighbor_cache(target_idx=target_idx, history_stack_corr=train, topology_candidates=topology_candidates)
    neighbors = cache["neighbors"]
    if len(neighbors) == 0:
        return 1.0, max(func_error, EPSILON)
    current_values = holdout[neighbors]
    available = np.isfinite(current_values)
    positive = available & (cache["corr"] > 0.05)
    if not np.any(positive):
        return 1.0, max(func_error, EPSILON)
    weights = np.where(
        positive,
        np.power(np.maximum(cache["corr"], 0.0), 1.0) / np.power(cache["lengths"] + EPSILON, 1.0),
        0.0,
    ).astype(np.float32)
    z_values = (current_values - cache["mu_j"]) / (cache["sigma_j"] + EPSILON)
    numerator = float(np.sum(weights * z_values))
    denominator = float(np.sum(weights))
    if denominator <= EPSILON:
        return 1.0, max(func_error, EPSILON)
    space_pred = cache["mu_i"] + (max(cache["sigma_i"], EPSILON) * (numerator / denominator))
    space_error = float(np.sqrt(np.mean((space_pred - float(np.mean(target_actual))) ** 2)))
    return max(space_error, EPSILON), max(func_error, EPSILON)


def infer_neighbor_coverage_group(values: np.ndarray) -> np.ndarray:
    labels = np.full(len(values), "high", dtype="<U8")
    labels[values < 0.67] = "mid"
    labels[values < 0.34] = "low"
    return labels


def metric_row(
    *,
    missing_rate: float,
    method: str,
    chunk_index: int,
    day_index: int,
    group_dimension: str,
    flow_group: str,
    length_group: str,
    neighbor_coverage_group: str,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    fallback_flags: np.ndarray,
    neighbor_available_values: np.ndarray,
    neighbor_coverage_values: np.ndarray,
    mean_positive_corr_values: np.ndarray,
    is_warmup_day: bool,
) -> dict[str, Any]:
    finite_mask = np.isfinite(true_values) & np.isfinite(pred_values)
    valid_true = true_values[finite_mask]
    valid_pred = pred_values[finite_mask]
    valid_neighbor_count = neighbor_available_values[finite_mask]
    valid_neighbor_coverage = neighbor_coverage_values[finite_mask]
    valid_positive_corr = mean_positive_corr_values[finite_mask]
    errors = valid_pred - valid_true
    abs_errors = np.abs(errors)
    sq_errors = errors * errors
    nonzero_mask = np.abs(valid_true) > EPSILON
    ape_values = np.abs(errors[nonzero_mask] / valid_true[nonzero_mask]) if np.any(nonzero_mask) else np.array([], dtype=np.float32)
    smape_denom = np.abs(valid_true) + np.abs(valid_pred)
    smape_mask = smape_denom > EPSILON
    smape_values = (
        2.0 * np.abs(errors[smape_mask]) / smape_denom[smape_mask] if np.any(smape_mask) else np.array([], dtype=np.float32)
    )
    valid_eval_count = int(len(valid_true))
    missing_count = int(len(true_values))
    fallback_count = int(np.count_nonzero(fallback_flags))
    mae = float(abs_errors.mean()) if valid_eval_count else math.nan
    rmse = float(np.sqrt(sq_errors.mean())) if valid_eval_count else math.nan
    mape = float(ape_values.mean()) if len(ape_values) > 0 else math.nan
    smape = float(smape_values.mean()) if len(smape_values) > 0 else math.nan
    if valid_eval_count:
        true_min = float(valid_true.min())
        true_max = float(valid_true.max())
        data_range = true_max - true_min
        nrmse = float(rmse / data_range) if data_range > EPSILON else math.nan
    else:
        true_min = math.nan
        true_max = math.nan
        nrmse = math.nan
    return {
        "missing_rate": float(missing_rate),
        "method": method,
        "chunk_index": int(chunk_index),
        "day_index": int(day_index),
        "group_dimension": group_dimension,
        "flow_group": flow_group,
        "length_group": length_group,
        "neighbor_coverage_group": neighbor_coverage_group,
        "is_warmup_day": bool(is_warmup_day),
        "missing_count": missing_count,
        "valid_eval_count": valid_eval_count,
        "fallback_count": fallback_count,
        "neighbor_available_count_sum": float(valid_neighbor_count.sum()) if valid_eval_count else 0.0,
        "neighbor_coverage_sum": float(valid_neighbor_coverage.sum()) if valid_eval_count else 0.0,
        "mean_positive_corr_sum": float(valid_positive_corr.sum()) if valid_eval_count else 0.0,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "nrmse": nrmse,
        "abs_error_sum": float(abs_errors.sum()) if valid_eval_count else 0.0,
        "sq_error_sum": float(sq_errors.sum()) if valid_eval_count else 0.0,
        "ape_sum": float(ape_values.sum()) if len(ape_values) > 0 else 0.0,
        "ape_count": int(len(ape_values)),
        "smape_sum": float(smape_values.sum()) if len(smape_values) > 0 else 0.0,
        "smape_count": int(len(smape_values)),
        "true_min": true_min,
        "true_max": true_max,
    }


def build_detail_rows_for_method(
    *,
    missing_rate: float,
    method: str,
    prepared: ChunkPrepared,
    pred_values: np.ndarray,
    fallback_flags: np.ndarray,
    neighbor_available_values: np.ndarray,
    neighbor_coverage_values: np.ndarray,
    mean_positive_corr_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        metric_row(
            missing_rate=missing_rate,
            method=method,
            chunk_index=prepared.chunk_index,
            day_index=prepared.day_index,
            group_dimension="overall",
            flow_group="overall",
            length_group="overall",
            neighbor_coverage_group="overall",
            true_values=prepared.true_values,
            pred_values=pred_values,
            fallback_flags=fallback_flags,
            neighbor_available_values=neighbor_available_values,
            neighbor_coverage_values=neighbor_coverage_values,
            mean_positive_corr_values=mean_positive_corr_values,
            is_warmup_day=prepared.is_warmup,
        )
    ]
    for group_id, group_label in enumerate(FLOW_GROUP_LABELS):
        selector = prepared.flow_group_ids == group_id
        if np.any(selector):
            rows.append(
                metric_row(
                    missing_rate=missing_rate,
                    method=method,
                    chunk_index=prepared.chunk_index,
                    day_index=prepared.day_index,
                    group_dimension="flow_group",
                    flow_group=group_label,
                    length_group="overall",
                    neighbor_coverage_group="overall",
                    true_values=prepared.true_values[selector],
                    pred_values=pred_values[selector],
                    fallback_flags=fallback_flags[selector],
                    neighbor_available_values=neighbor_available_values[selector],
                    neighbor_coverage_values=neighbor_coverage_values[selector],
                    mean_positive_corr_values=mean_positive_corr_values[selector],
                    is_warmup_day=prepared.is_warmup,
                )
            )
    for group_label in LENGTH_GROUP_LABELS:
        selector = prepared.length_groups == group_label
        if np.any(selector):
            rows.append(
                metric_row(
                    missing_rate=missing_rate,
                    method=method,
                    chunk_index=prepared.chunk_index,
                    day_index=prepared.day_index,
                    group_dimension="length_group",
                    flow_group="overall",
                    length_group=group_label,
                    neighbor_coverage_group="overall",
                    true_values=prepared.true_values[selector],
                    pred_values=pred_values[selector],
                    fallback_flags=fallback_flags[selector],
                    neighbor_available_values=neighbor_available_values[selector],
                    neighbor_coverage_values=neighbor_coverage_values[selector],
                    mean_positive_corr_values=mean_positive_corr_values[selector],
                    is_warmup_day=prepared.is_warmup,
                )
            )
    for group_label in CONSTRAINT_LEVELS:
        selector = prepared.spatial_constraint_levels == group_label
        if np.any(selector):
            rows.append(
                metric_row(
                    missing_rate=missing_rate,
                    method=method,
                    chunk_index=prepared.chunk_index,
                    day_index=prepared.day_index,
                    group_dimension="spatial_constraint_level",
                    flow_group="overall",
                    length_group=group_label,
                    neighbor_coverage_group="overall",
                    true_values=prepared.true_values[selector],
                    pred_values=pred_values[selector],
                    fallback_flags=fallback_flags[selector],
                    neighbor_available_values=neighbor_available_values[selector],
                    neighbor_coverage_values=neighbor_coverage_values[selector],
                    mean_positive_corr_values=mean_positive_corr_values[selector],
                    is_warmup_day=prepared.is_warmup,
                )
            )
    coverage_groups = infer_neighbor_coverage_group(neighbor_coverage_values)
    for group_label in NEIGHBOR_COVERAGE_GROUPS:
        selector = coverage_groups == group_label
        if np.any(selector):
            rows.append(
                metric_row(
                    missing_rate=missing_rate,
                    method=method,
                    chunk_index=prepared.chunk_index,
                    day_index=prepared.day_index,
                    group_dimension="neighbor_coverage_group",
                    flow_group="overall",
                    length_group="overall",
                    neighbor_coverage_group=group_label,
                    true_values=prepared.true_values[selector],
                    pred_values=pred_values[selector],
                    fallback_flags=fallback_flags[selector],
                    neighbor_available_values=neighbor_available_values[selector],
                    neighbor_coverage_values=neighbor_coverage_values[selector],
                    mean_positive_corr_values=mean_positive_corr_values[selector],
                    is_warmup_day=prepared.is_warmup,
                )
            )
    return rows


def summarize_from_detail(detail_df: pd.DataFrame, exclude_warmup: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = detail_df.loc[~detail_df["is_warmup_day"].astype(bool)].copy() if exclude_warmup else detail_df.copy()

    def aggregate(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
        grouped = (
            df.groupby(group_keys, dropna=False)
            .agg(
                missing_count=("missing_count", "sum"),
                valid_eval_count=("valid_eval_count", "sum"),
                fallback_count=("fallback_count", "sum"),
                neighbor_available_count_sum=("neighbor_available_count_sum", "sum"),
                neighbor_coverage_sum=("neighbor_coverage_sum", "sum"),
                mean_positive_corr_sum=("mean_positive_corr_sum", "sum"),
                abs_error_sum=("abs_error_sum", "sum"),
                sq_error_sum=("sq_error_sum", "sum"),
                ape_sum=("ape_sum", "sum"),
                ape_count=("ape_count", "sum"),
                smape_sum=("smape_sum", "sum"),
                smape_count=("smape_count", "sum"),
                true_min=("true_min", "min"),
                true_max=("true_max", "max"),
            )
            .reset_index()
        )
        grouped["mae"] = grouped["abs_error_sum"] / grouped["valid_eval_count"].replace(0, np.nan)
        grouped["rmse"] = np.sqrt(grouped["sq_error_sum"] / grouped["valid_eval_count"].replace(0, np.nan))
        grouped["mape"] = grouped["ape_sum"] / grouped["ape_count"].replace(0, np.nan)
        grouped["smape"] = grouped["smape_sum"] / grouped["smape_count"].replace(0, np.nan)
        grouped["nrmse"] = grouped["rmse"] / ((grouped["true_max"] - grouped["true_min"]).replace(0, np.nan))
        grouped["neighbor_available_count"] = grouped["neighbor_available_count_sum"] / grouped["valid_eval_count"].replace(0, np.nan)
        grouped["neighbor_coverage"] = grouped["neighbor_coverage_sum"] / grouped["valid_eval_count"].replace(0, np.nan)
        grouped["mean_positive_corr"] = grouped["mean_positive_corr_sum"] / grouped["valid_eval_count"].replace(0, np.nan)
        return grouped

    overall = aggregate(filtered.loc[filtered["group_dimension"] == "overall"].copy(), ["missing_rate", "method"])
    by_length = aggregate(
        filtered.loc[filtered["group_dimension"] == "length_group"].copy(),
        ["missing_rate", "method", "length_group"],
    )
    by_flow = aggregate(
        filtered.loc[filtered["group_dimension"] == "flow_group"].copy(),
        ["missing_rate", "method", "flow_group"],
    )
    by_neighbor = aggregate(
        filtered.loc[filtered["group_dimension"] == "neighbor_coverage_group"].copy(),
        ["missing_rate", "method", "neighbor_coverage_group"],
    )
    by_constraint = aggregate(
        filtered.loc[filtered["group_dimension"] == "spatial_constraint_level"].copy(),
        ["missing_rate", "method", "length_group"],
    ).rename(columns={"length_group": "spatial_constraint_level"})
    return overall, by_length, by_flow, by_neighbor, by_constraint


def run_prepare(args: argparse.Namespace, paths: StagePaths) -> pd.DataFrame:
    for directory in [
        paths.imp_root,
        paths.imp_data_root,
        paths.manifests_root,
        paths.summaries_root,
        paths.figures_root,
        paths.audits_root,
    ]:
        ensure_dir(directory)
    input_files = load_input_files(args.input_dir)
    rates = parse_rates(args.missing_rates)
    rows: list[dict[str, Any]] = []
    for rate in rates:
        scenario_tag = scenario_rate_tag(rate)
        mask_dir = paths.masks_root / scenario_tag
        miss_dir = paths.miss_data_root / scenario_tag
        mask_files = sorted(mask_dir.glob("*_mask.parquet"))
        miss_files = sorted(miss_dir.glob("*.parquet"))
        rows.append(
            {
                "missing_rate": float(rate),
                "scenario_tag": scenario_tag,
                "mask_file_count": int(len(mask_files)),
                "miss_data_file_count": int(len(miss_files)),
                "expected_chunk_count": int(len(input_files)),
                "masks_complete": bool(len(mask_files) == len(input_files)),
                "miss_data_complete": bool(len(miss_files) == len(input_files)),
            }
        )
    input_check_df = pd.DataFrame(rows)
    input_check_df.to_csv(paths.input_check_csv_path, index=False, encoding="utf-8-sig")
    write_json(paths.input_check_json_path, {"rows": input_check_df.to_dict(orient="records")})
    write_json(
        paths.run_config_path,
        {
            "scenario_id": SCENARIO_ID,
            "mechanism": MECHANISM,
            "evaluation_protocol": "online_spatial_interpolation",
            "input_dir": str(args.input_dir),
            "scenario_dir": str(paths.scenario_root),
            "topology_file": str(args.topology_file),
            "missing_rates": args.missing_rates,
            "methods": args.methods,
            "history_days": int(args.history_days),
            "correlation_history_days": int(args.correlation_history_days),
            "neighbor_scope": int(args.neighbor_scope),
            "allow_current_time_neighbors": parse_bool(args.allow_current_time_neighbors),
            "target_col": args.target_col,
            "node_col": args.node_col,
            "time_col": args.time_col,
            "period": int(args.period),
        },
    )
    command = (
        "E:\\anaconda3\\envs\\analysis\\python.exe analysis_scripts\\spatial_neighbor_holdout_imputation_pipeline.py "
        f"--stage all --input_dir {args.input_dir} --scenario_dir {paths.scenario_root} --topology_file {args.topology_file} "
        f"--missing_rates {args.missing_rates} --methods {args.methods} --history_days {args.history_days} "
        f"--correlation_history_days {args.correlation_history_days} --neighbor_scope {args.neighbor_scope} "
        f"--allow_current_time_neighbors {args.allow_current_time_neighbors} --target_col {args.target_col} "
        f"--node_col {args.node_col} --time_col {args.time_col} --period {args.period}"
    )
    paths.run_commands_path.write_text(command + "\n", encoding="utf-8")
    return input_check_df


def run_impute(args: argparse.Namespace, paths: StagePaths) -> pd.DataFrame:
    input_files = load_input_files(args.input_dir)
    rates = parse_rates(args.missing_rates)
    methods = parse_methods(args.methods)
    first_clean = pd.read_parquet(input_files[0], columns=[args.node_col, args.time_col])
    canonical_node_ids, _, _ = build_row_layout(first_clean, args.node_col, args.time_col, args.period)
    thresholds = compute_flow_group_thresholds(input_files, args.target_col, paths.flow_threshold_path)
    topology_candidates = build_topology_candidates(args.topology_file, canonical_node_ids, args.neighbor_scope)
    basis, pinv = build_fourier_basis(args.period)
    chunk_status_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for rate in rates:
        scenario_tag = scenario_rate_tag(rate)
        history_clean: deque[np.ndarray] = deque(maxlen=max(args.history_days, args.correlation_history_days))
        previous_last_state: np.ndarray | None = None
        for clean_file in input_files:
            missing_file = paths.miss_data_root / scenario_tag / clean_file.name
            mask_file = paths.masks_root / scenario_tag / clean_file.name.replace(".parquet", "_mask.parquet")
            prepared = prepare_chunk(
                clean_file=clean_file,
                missing_file=missing_file,
                mask_file=mask_file,
                node_col=args.node_col,
                time_col=args.time_col,
                target_col=args.target_col,
                period=args.period,
                thresholds=thresholds,
                warmup_days=args.history_days,
            )
            forward_matrix, forward_internal_fallback = build_current_day_forward_fill(prepared.missing_matrix, previous_last_state)
            mean_matrix, mean_fallback = impute_mean_fill(
                prepared.missing_matrix,
                prepared.mask_matrix,
                history_clean,
                forward_matrix,
            )
            hle_matrix, hle_fallback = impute_historical_linear(
                prepared.missing_matrix,
                prepared.mask_matrix,
                prepared.day_index,
                history_clean,
                mean_matrix,
            )
            func_matrix, func_fallback, func_primary = impute_function_curve_fit(
                prepared.missing_matrix,
                prepared.mask_matrix,
                history_clean,
                basis,
                pinv,
                mean_matrix,
            )
            corr_stack = historical_stack(history_clean, last_n=args.correlation_history_days)
            road_matrix, road_fallback, road_neighbor_count_matrix, road_coverage_matrix, _, road_success = spatial_current_prediction(
                missing_matrix=prepared.missing_matrix,
                mask_matrix=prepared.mask_matrix,
                mean_fill=mean_matrix,
                node_indices=prepared.node_indices,
                slot_indices=prepared.slot_indices,
                topology_candidates=topology_candidates,
                history_stack_corr=corr_stack,
                correlation_mode=False,
            )
            ctn_matrix, ctn_fallback, ctn_neighbor_count_matrix, ctn_coverage_matrix, ctn_positive_corr_matrix, ctn_success = spatial_current_prediction(
                missing_matrix=prepared.missing_matrix,
                mask_matrix=prepared.mask_matrix,
                mean_fill=mean_matrix,
                node_indices=prepared.node_indices,
                slot_indices=prepared.slot_indices,
                topology_candidates=topology_candidates,
                history_stack_corr=corr_stack,
                correlation_mode=True,
            )
            ast_matrix = func_matrix.copy()
            ast_fallback = func_fallback.copy()
            ctn_neighbor_values = ctn_neighbor_count_matrix[prepared.mask_matrix]
            ctn_coverage_values = ctn_coverage_matrix[prepared.mask_matrix]
            ctn_positive_values = ctn_positive_corr_matrix[prepared.mask_matrix]
            ctn_success_values = ctn_success[prepared.mask_matrix]
            length_factor_lookup = {"short": 0.4, "mid": 0.6, "long": 0.8}
            length_factors = np.asarray([length_factor_lookup.get(label, 0.6) for label in prepared.length_groups.tolist()], dtype=np.float32)
            alpha_values = np.clip(ctn_coverage_values * np.maximum(ctn_positive_values, 0.0) * length_factors, 0.1, 0.8)
            func_values = func_matrix[prepared.mask_matrix]
            ctn_values = ctn_matrix[prepared.mask_matrix]
            ast_values = func_values.copy()
            ast_values[ctn_success_values] = (
                alpha_values[ctn_success_values] * ctn_values[ctn_success_values]
                + (1.0 - alpha_values[ctn_success_values]) * func_values[ctn_success_values]
            )
            ast_matrix[prepared.mask_matrix] = ast_values
            ast_fallback[prepared.mask_matrix] = (~ctn_success_values) & func_fallback[prepared.mask_matrix]
            atfh_matrix = func_matrix.copy()
            atfh_fallback = func_fallback.copy()
            unique_targets = np.unique(prepared.node_indices)
            error_cache: dict[int, tuple[float, float]] = {}
            atfh_values = func_values.copy()
            for target_idx in unique_targets.tolist():
                if target_idx not in error_cache:
                    error_cache[target_idx] = estimate_historical_errors(
                        target_idx=int(target_idx),
                        history_stack_corr=corr_stack,
                        topology_candidates=topology_candidates,
                        basis=basis,
                        pinv=pinv,
                    )
                space_error, func_error = error_cache[target_idx]
                selector = prepared.node_indices == target_idx
                if not np.any(selector):
                    continue
                alpha = (1.0 / (space_error + EPSILON)) / (
                    (1.0 / (space_error + EPSILON)) + (1.0 / (func_error + EPSILON))
                )
                local_success = ctn_success_values[selector]
                local_ctn = ctn_values[selector]
                local_func = func_values[selector]
                atfh_values[selector] = local_func
                atfh_values[np.flatnonzero(selector)[local_success]] = (
                    alpha * local_ctn[local_success] + (1.0 - alpha) * local_func[local_success]
                )
            atfh_matrix[prepared.mask_matrix] = atfh_values
            atfh_fallback[prepared.mask_matrix] = (~ctn_success_values) & func_fallback[prepared.mask_matrix]
            method_outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
                "mean_fill": (
                    mean_matrix,
                    mean_fallback[prepared.mask_matrix],
                    prepared.available_neighbor_counts.astype(np.float32),
                    prepared.available_neighbor_counts.astype(np.float32)
                    / np.maximum(prepared.available_neighbor_counts.astype(np.float32), 1.0),
                    np.zeros_like(prepared.available_neighbor_counts, dtype=np.float32),
                ),
                "forward_fill": (
                    forward_matrix,
                    forward_internal_fallback[prepared.mask_matrix],
                    prepared.available_neighbor_counts.astype(np.float32),
                    prepared.available_neighbor_counts.astype(np.float32)
                    / np.maximum(prepared.available_neighbor_counts.astype(np.float32), 1.0),
                    np.zeros_like(prepared.available_neighbor_counts, dtype=np.float32),
                ),
                "historical_linear_extrapolation": (
                    hle_matrix,
                    hle_fallback[prepared.mask_matrix],
                    prepared.available_neighbor_counts.astype(np.float32),
                    prepared.available_neighbor_counts.astype(np.float32)
                    / np.maximum(prepared.available_neighbor_counts.astype(np.float32), 1.0),
                    np.zeros_like(prepared.available_neighbor_counts, dtype=np.float32),
                ),
                "function_curve_fit": (
                    func_matrix,
                    func_fallback[prepared.mask_matrix],
                    prepared.available_neighbor_counts.astype(np.float32),
                    prepared.available_neighbor_counts.astype(np.float32)
                    / np.maximum(prepared.available_neighbor_counts.astype(np.float32), 1.0),
                    np.zeros_like(prepared.available_neighbor_counts, dtype=np.float32),
                ),
                "road_topology_neighbor_fill": (
                    road_matrix,
                    road_fallback[prepared.mask_matrix],
                    road_neighbor_count_matrix[prepared.mask_matrix],
                    road_coverage_matrix[prepared.mask_matrix],
                    np.zeros_like(road_neighbor_count_matrix[prepared.mask_matrix], dtype=np.float32),
                ),
                "correlation_topology_neighbor_fill": (
                    ctn_matrix,
                    ctn_fallback[prepared.mask_matrix],
                    ctn_neighbor_count_matrix[prepared.mask_matrix],
                    ctn_coverage_matrix[prepared.mask_matrix],
                    ctn_positive_corr_matrix[prepared.mask_matrix],
                ),
                "adaptive_spatio_temporal_fill": (
                    ast_matrix,
                    ast_fallback[prepared.mask_matrix],
                    ctn_neighbor_count_matrix[prepared.mask_matrix],
                    ctn_coverage_matrix[prepared.mask_matrix],
                    ctn_positive_corr_matrix[prepared.mask_matrix],
                ),
                "adaptive_topology_function_hybrid": (
                    atfh_matrix,
                    atfh_fallback[prepared.mask_matrix],
                    ctn_neighbor_count_matrix[prepared.mask_matrix],
                    ctn_coverage_matrix[prepared.mask_matrix],
                    ctn_positive_corr_matrix[prepared.mask_matrix],
                ),
            }
            for method in methods:
                imputed_matrix, fallback_flags, neighbor_count_values, neighbor_coverage_values, positive_corr_values = method_outputs[method]
                out_values = imputed_matrix.reshape(-1)
                output_series = np.empty(len(prepared.clean_df), dtype=np.float32)
                output_series[prepared.sorted_original_rows] = out_values
                out_df = prepared.missing_df.copy()
                out_df[args.target_col] = output_series
                method_dir = paths.imp_data_root / imputed_dir_name(rate, method)
                ensure_dir(method_dir)
                out_path = method_dir / prepared.file_name
                out_df.to_parquet(out_path, index=False)
                pred_values = imputed_matrix[prepared.mask_matrix]
                detail_rows.extend(
                    build_detail_rows_for_method(
                        missing_rate=rate,
                        method=method,
                        prepared=prepared,
                        pred_values=pred_values,
                        fallback_flags=fallback_flags.astype(bool, copy=False),
                        neighbor_available_values=neighbor_count_values.astype(np.float32, copy=False),
                        neighbor_coverage_values=neighbor_coverage_values.astype(np.float32, copy=False),
                        mean_positive_corr_values=positive_corr_values.astype(np.float32, copy=False),
                    )
                )
                chunk_status_rows.append(
                    {
                        "missing_rate": float(rate),
                        "method": method,
                        "chunk_index": int(prepared.chunk_index),
                        "day_index": int(prepared.day_index),
                        "file_name": prepared.file_name,
                        "output_path": str(out_path),
                        "row_count": int(len(out_df)),
                        "missing_count": int(prepared.mask_matrix.sum()),
                        "fallback_count": int(np.count_nonzero(fallback_flags)),
                    }
                )
            previous_last_state = prepared.clean_matrix[:, -1].copy()
            history_clean.append(prepared.clean_matrix.copy())
            del prepared, forward_matrix, mean_matrix, hle_matrix, func_matrix, func_primary
            del road_matrix, ctn_matrix, ast_matrix, atfh_matrix
            gc.collect()
    chunk_status_df = pd.DataFrame(chunk_status_rows).sort_values(["missing_rate", "method", "chunk_index"]).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows).sort_values(
        ["missing_rate", "method", "chunk_index", "group_dimension", "flow_group", "length_group", "neighbor_coverage_group"]
    ).reset_index(drop=True)
    chunk_status_df.to_csv(paths.chunk_status_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(paths.detail_path, index=False, encoding="utf-8-sig")
    return detail_df


def run_summarize(paths: StagePaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_df = pd.read_csv(paths.detail_path)
    overall_df, by_length_df, by_flow_df, by_neighbor_df, by_constraint_df = summarize_from_detail(detail_df, exclude_warmup=True)
    overall_df.to_csv(paths.summary_exclude_warmup_path, index=False, encoding="utf-8-sig")
    by_length_df.to_csv(paths.summary_by_length_path, index=False, encoding="utf-8-sig")
    by_flow_df.to_csv(paths.summary_by_flow_path, index=False, encoding="utf-8-sig")
    by_neighbor_df.to_csv(paths.summary_by_neighbor_coverage_path, index=False, encoding="utf-8-sig")
    by_constraint_df.to_csv(paths.summary_by_constraint_level_path, index=False, encoding="utf-8-sig")
    return overall_df, by_length_df, by_flow_df, by_neighbor_df, by_constraint_df


def run_audit(args: argparse.Namespace, paths: StagePaths) -> dict[str, Any]:
    summary_df = pd.read_csv(paths.summary_exclude_warmup_path)
    by_length_df = pd.read_csv(paths.summary_by_length_path)
    by_flow_df = pd.read_csv(paths.summary_by_flow_path)
    by_neighbor_df = pd.read_csv(paths.summary_by_neighbor_coverage_path)
    by_constraint_df = pd.read_csv(paths.summary_by_constraint_level_path)
    input_check_df = pd.read_csv(paths.input_check_csv_path)
    payload = {
        "scenario_id": SCENARIO_ID,
        "mechanism": MECHANISM,
        "evaluation_protocol": "online_spatial_interpolation",
        "uses_current_time_neighbors": True,
        "uses_target_current_true_value": False,
        "uses_future_time_steps": False,
        "uses_future_days": False,
        "allowed_current_information": "observed_neighbors_only",
        "masked_position_error_only": True,
        "not_traffic_prediction_error": True,
        "neighbor_observed_enforced": True,
        "not_strict_history_only": True,
        "methods": METHOD_ORDER,
        "input_checks": input_check_df.to_dict(orient="records"),
        "summary_outputs": {
            "detail": str(paths.detail_path),
            "summary_exclude_warmup": str(paths.summary_exclude_warmup_path),
            "by_length_group": str(paths.summary_by_length_path),
            "by_flow_group": str(paths.summary_by_flow_path),
            "by_neighbor_coverage": str(paths.summary_by_neighbor_coverage_path),
            "by_constraint_level": str(paths.summary_by_constraint_level_path),
        },
        "contains_8_methods": bool(set(summary_df["method"].astype(str).unique().tolist()) == set(METHOD_ORDER)),
        "contains_length_groups": sorted(by_length_df["length_group"].astype(str).unique().tolist()),
        "contains_flow_groups": sorted(by_flow_df["flow_group"].astype(str).unique().tolist()),
        "contains_neighbor_coverage_groups": sorted(by_neighbor_df["neighbor_coverage_group"].astype(str).unique().tolist()),
        "contains_constraint_levels": sorted(by_constraint_df["spatial_constraint_level"].astype(str).unique().tolist()),
    }
    write_json(paths.audit_json_path, payload)
    write_markdown(
        paths.audit_md_path,
        [
            "# snh_mix 空间插补审计",
            "",
            f"- scenario_id: `{SCENARIO_ID}`",
            f"- mechanism: `{MECHANISM}`",
            "- evaluation_protocol: `online_spatial_interpolation`",
            "- 允许使用目标节点缺失时刻的邻居观测。",
            "- 不允许使用目标节点当前真实值。",
            "- 不允许使用未来时间片或未来天。",
            "- 当前指标仅在人工 mask 位置计算，不是交通流预测误差。",
        ],
    )
    return payload


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.scenario_dir = ensure_absolute(project_root, args.scenario_dir)
    args.topology_file = ensure_absolute(project_root, args.topology_file)
    if not parse_bool(args.allow_current_time_neighbors):
        raise ValueError("snh_mix requires --allow_current_time_neighbors true")
    paths = build_paths(args.scenario_dir)
    run_prepare(args, paths)
    if args.stage in {"impute", "all"}:
        run_impute(args, paths)
    if args.stage in {"summarize", "all"}:
        if not paths.detail_path.exists():
            raise FileNotFoundError("detail summary is missing; run impute first")
        run_summarize(paths)
    if args.stage in {"audit", "all"}:
        if not paths.summary_exclude_warmup_path.exists():
            raise FileNotFoundError("summary outputs are missing; run summarize first")
        run_audit(args, paths)


if __name__ == "__main__":
    main()
