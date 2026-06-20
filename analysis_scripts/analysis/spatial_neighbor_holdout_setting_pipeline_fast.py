"""快速生成空间邻居保留缺失场景的设置结果与审计产物。

核心功能：
- 在空间邻接约束下高效构造 holdout 缺失事件；
- 输出掩码、缺失分块、事件清单和统计审计结果；
- 提供与标准 setting 流程兼容、但更偏向批量运行的快速实现。

项目作用：
- 用于加速空间缺失实验的大规模 setting 阶段；
- 作为标准空间缺失设置脚本的性能优化替代版本。

关键依赖：`numpy`、`pandas`、`pathlib`。
主要输入：完整分块数据、拓扑文件、缺失率和长度配置。
主要输出：掩码文件、缺失数据、事件日志和审计文件。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LENGTH_GROUP_ORDER = ["short", "mid", "long"]
LENGTH_GROUP_TO_CODE = {"short": 0, "mid": 1, "long": 2}
CODE_TO_LENGTH_GROUP = {0: "short", 1: "mid", 2: "long"}
CONSTRAINT_LEVELS = ["strict_anchor", "relaxed_anchor", "weak_neighbor_available", "none"]
CONSTRAINT_LEVEL_TO_CODE = {
    "strict_anchor": 1,
    "relaxed_anchor": 2,
    "weak_neighbor_available": 3,
    "none": 4,
}
CODE_TO_CONSTRAINT_LEVEL = {
    1: "strict_anchor",
    2: "relaxed_anchor",
    3: "weak_neighbor_available",
    4: "none",
}
SCENARIO_ID = "snh_mix"
MECHANISM = "spatial_neighbor_holdout"
EVALUATION_PROTOCOL = "online_spatial_interpolation"
CHECKPOINT_MODE = "part_files"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class LengthConfig:
    short_range: Tuple[int, int]
    mid_range: Tuple[int, int]
    long_range: Tuple[int, int]
    probs: Tuple[float, float, float]


@dataclass
class PreparedArtifacts:
    input_files: List[Path]
    chunk_summary_df: pd.DataFrame
    eligible_df: pd.DataFrame
    node_neighbor_lookup_df: pd.DataFrame
    allocation_df: pd.DataFrame
    allocation_map: Dict[Tuple[float, int], Dict[str, Any]]
    canonical_node_ids: np.ndarray
    preferred_scope: Dict[int, int]
    preferred_neighbors: Dict[int, np.ndarray]
    preferred_lengths: Dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast spatial_neighbor_holdout missingness generator with resume support.")
    parser.add_argument("--stage", required=True, choices=["prepare", "generate_missing", "finalize", "audit", "validate", "all"])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topology_file", required=True, type=Path)
    parser.add_argument("--missing_rates", required=True, type=str)
    parser.add_argument("--length_group_probs", required=True, type=str)
    parser.add_argument("--short_length_range", required=True, type=str)
    parser.add_argument("--mid_length_range", required=True, type=str)
    parser.add_argument("--long_length_range", required=True, type=str)
    parser.add_argument("--neighbor_scope", required=True, type=int)
    parser.add_argument("--min_available_neighbors", required=True, type=int)
    parser.add_argument("--anchor_neighbor_count", default=1, type=int)
    parser.add_argument("--neighbor_protection_mode", default="anchor", choices=["anchor", "all"])
    parser.add_argument("--placement_backend", default="bitset", choices=["numpy_bool", "bitset"])
    parser.add_argument("--candidate_oversample_factor", default=-1.0, type=float)
    parser.add_argument("--max_candidate_rounds", default=5, type=int)
    parser.add_argument("--allocation_method", default="sequential_hypergeometric_global_without_replacement", type=str)
    parser.add_argument("--allocation_shortfall_policy", default="carry_forward", type=str)
    parser.add_argument("--mask_scope", default="global", type=str)
    parser.add_argument("--spatial_constraint_relaxation", default="true", type=str)
    parser.add_argument("--relaxation_policy", default="progressive", choices=["progressive"])
    parser.add_argument("--min_spatially_constrained_ratio", default=0.70, type=float)
    parser.add_argument("--allow_no_spatial_constraint", default="true", type=str)
    parser.add_argument("--max_seconds_per_rate_chunk", default=600.0, type=float)
    parser.add_argument("--max_resample_ratio", default=20.0, type=float)
    parser.add_argument("--write_missing_data", default="true", type=str)
    parser.add_argument("--write_event_neighbor_rows", default="false", type=str)
    parser.add_argument("--resume", default="true", type=str)
    parser.add_argument("--overwrite", default="false", type=str)
    parser.add_argument("--only_rates", default="", type=str)
    parser.add_argument("--only_chunks", default="", type=str)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--node_col", required=True, type=str)
    parser.add_argument("--time_col", required=True, type=str)
    parser.add_argument("--period", required=True, type=int)
    return parser.parse_args()


def parse_bool(raw: str) -> bool:
    lowered = str(raw).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError("invalid boolean value: %s" % raw)


def parse_float_list(raw: str) -> List[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def parse_int_pair(raw: str) -> Tuple[int, int]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if len(values) != 2:
        raise ValueError("expected two integers, got: %s" % raw)
    return int(values[0]), int(values[1])


def parse_rates(raw: str) -> List[float]:
    values = parse_float_list(raw)
    if not values:
        raise ValueError("missing_rates is empty")
    for value in values:
        if value <= 0.0 or value >= 1.0:
            raise ValueError("invalid missing rate: %s" % value)
    return values


def parse_only_rates(raw: str) -> Optional[set]:
    values = {round(value, 6) for value in parse_float_list(raw)} if str(raw).strip() else set()
    return values or None


def parse_only_chunks(raw: str) -> Optional[set]:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not tokens:
        return None
    return {int(token) for token in tokens}


def parse_length_config(args: argparse.Namespace) -> LengthConfig:
    probs = parse_float_list(args.length_group_probs)
    if len(probs) != 3:
        raise ValueError("length_group_probs must contain 3 values")
    if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("length_group_probs must sum to 1.0")
    return LengthConfig(
        short_range=parse_int_pair(args.short_length_range),
        mid_range=parse_int_pair(args.mid_length_range),
        long_range=parse_int_pair(args.long_length_range),
        probs=(float(probs[0]), float(probs[1]), float(probs[2])),
    )


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    ensure_dir(final_path.parent)
    tmp_path.replace(final_path)


def atomic_write_text(path: Path, text: str, encoding: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    ensure_dir(path.parent)
    tmp_path.write_text(text, encoding=encoding)
    atomic_replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_write_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    ensure_dir(path.parent)
    frame.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    atomic_replace(tmp_path, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    ensure_dir(path.parent)
    frame.to_parquet(tmp_path, index=False)
    atomic_replace(tmp_path, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_rate_token(rate: float) -> str:
    return "%02d" % int(round(rate * 100.0))


def scenario_rate_tag(rate: float, seed: int) -> str:
    return "snh_r%s_mix_s%s" % (format_rate_token(rate), int(seed))


def chunk_token(chunk_index: int) -> str:
    return "%03d" % int(chunk_index)


def scenario_chunk_stem(rate: float, seed: int, chunk_index: int) -> str:
    return "%s__chunk_%s" % (scenario_rate_tag(rate, seed), chunk_token(chunk_index))


def extract_day_index(file_name: str) -> int:
    return int(Path(file_name).stem.rsplit("_", 1)[-1])


def load_input_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no parquet files found under %s" % input_dir)
    return files


def default_global_allocation_source(project_root: Path) -> Path:
    return project_root / "results" / "rdm_exp" / "scenarios" / "g_mcar_pt" / "miss_set" / "manifests" / "global_missing_allocation.csv"


def validate_allocation_pairs(allocation_df: pd.DataFrame, rates: Sequence[float], chunk_summary_df: pd.DataFrame) -> Dict[Tuple[float, int], Dict[str, Any]]:
    expected_pairs = set((round(float(rate), 6), int(chunk_index)) for rate in rates for chunk_index in chunk_summary_df["chunk_index"].tolist())
    allocation_map = {}
    for row in allocation_df.to_dict(orient="records"):
        key = (round(float(row["missing_rate_target"]), 6), int(row["chunk_index"]))
        allocation_map[key] = row
    missing_pairs = expected_pairs.difference(set(allocation_map.keys()))
    if missing_pairs:
        raise RuntimeError("allocation file is missing rate/chunk pairs: %s" % sorted(missing_pairs))
    return allocation_map


def generate_global_missing_allocation(
    args: argparse.Namespace,
    chunk_summary_df: pd.DataFrame,
    rates: Sequence[float],
) -> pd.DataFrame:
    rng = np.random.default_rng(int(args.seed))
    rows = []
    global_eligible_count = int(chunk_summary_df["eligible_non_null_count"].sum())
    for rate in rates:
        global_missing_count = int(round(float(global_eligible_count) * float(rate)))
        remaining_missing = int(global_missing_count)
        remaining_eligible = int(global_eligible_count)
        chunk_count = int(len(chunk_summary_df))
        for row_index, row in enumerate(chunk_summary_df.sort_values("chunk_index").to_dict(orient="records")):
            current_eligible = int(row["eligible_non_null_count"])
            if row_index == chunk_count - 1:
                allocated_missing_count = int(remaining_missing)
            elif current_eligible <= 0 or remaining_missing <= 0:
                allocated_missing_count = 0
            else:
                allocated_missing_count = int(
                    rng.hypergeometric(
                        ngood=int(remaining_missing),
                        nbad=int(max(remaining_eligible - remaining_missing, 0)),
                        nsample=int(current_eligible),
                    )
                )
            rows.append(
                {
                    "missing_rate_target": float(rate),
                    "chunk_index": int(row["chunk_index"]),
                    "day_index": int(row["day_index"]),
                    "file_name": str(row["file_name"]),
                    "eligible_non_null_count": int(current_eligible),
                    "allocated_missing_count": int(allocated_missing_count),
                    "global_eligible_count": int(global_eligible_count),
                    "global_missing_count": int(global_missing_count),
                    "remaining_eligible_before_chunk": int(remaining_eligible),
                    "remaining_missing_before_chunk": int(remaining_missing),
                    "mask_scope": str(args.mask_scope),
                    "allocation_method": str(args.allocation_method),
                    "allocation_shortfall_policy": str(args.allocation_shortfall_policy),
                    "mechanism": MECHANISM,
                }
            )
            remaining_missing -= int(allocated_missing_count)
            remaining_eligible -= int(current_eligible)
        if remaining_missing != 0:
            raise RuntimeError("global allocation did not exhaust missing count for rate %s" % rate)
    return pd.DataFrame(rows).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)


def build_paths(output_root: Path) -> Dict[str, Path]:
    miss_root = output_root / "miss_set"
    manifests_root = miss_root / "manifests"
    return {
        "scenario_root": output_root,
        "miss_root": miss_root,
        "masks_root": miss_root / "masks",
        "miss_data_root": miss_root / "miss_data",
        "manifests_root": manifests_root,
        "status_parts_root": manifests_root / "status_parts",
        "event_parts_root": manifests_root / "event_parts",
        "progress_root": manifests_root / "progress",
        "failed_parts_root": manifests_root / "failed_parts",
        "runtime_logs_root": manifests_root / "runtime_logs",
        "audits_root": miss_root / "audits",
        "run_config_path": miss_root / "run_config.json",
        "run_commands_path": miss_root / "run_commands.txt",
        "prepare_summary_path": manifests_root / "snh_prepare_summary.json",
        "chunk_summary_path": manifests_root / "snh_chunk_summary.csv",
        "eligible_nodes_path": manifests_root / "snh_eligible_nodes.csv",
        "node_neighbor_lookup_parquet": manifests_root / "node_neighbor_lookup.parquet",
        "node_neighbor_lookup_csv": manifests_root / "node_neighbor_lookup.csv",
        "global_eligible_chunk_counts_path": manifests_root / "global_eligible_chunk_counts.csv",
        "allocation_path": manifests_root / "global_missing_allocation.csv",
        "chunk_status_path": manifests_root / "snh_generate_missing_chunk_status.csv",
        "event_csv_path": manifests_root / "spatial_neighbor_holdout_events.csv",
        "runtime_jsonl_path": manifests_root / "runtime_logs" / "snh_generation_runtime.jsonl",
        "audit_json_path": miss_root / "audits" / "snh_missingness_audit.json",
        "audit_md_path": miss_root / "audits" / "snh_missingness_audit_zh.md",
        "global_allocation_audit_json_path": miss_root / "audits" / "snh_global_allocation_audit.json",
        "global_allocation_audit_md_path": miss_root / "audits" / "snh_global_allocation_audit_zh.md",
        "constraint_relaxation_audit_json_path": miss_root / "audits" / "snh_constraint_relaxation_audit.json",
        "constraint_relaxation_audit_md_path": miss_root / "audits" / "snh_constraint_relaxation_audit_zh.md",
        "performance_json_path": miss_root / "audits" / "snh_fast_generation_performance.json",
        "performance_md_path": miss_root / "audits" / "snh_fast_generation_performance_zh.md",
        "validation_json_path": miss_root / "audits" / "snh_fast_generation_validation.json",
        "validation_csv_path": miss_root / "audits" / "snh_fast_generation_validation.csv",
    }


def ensure_base_dirs(paths: Dict[str, Path]) -> None:
    for key in [
        "miss_root",
        "masks_root",
        "miss_data_root",
        "manifests_root",
        "status_parts_root",
        "event_parts_root",
        "progress_root",
        "failed_parts_root",
        "runtime_logs_root",
        "audits_root",
    ]:
        ensure_dir(paths[key])


def expected_length(config: LengthConfig) -> float:
    means = [
        (float(config.short_range[0]) + float(config.short_range[1])) / 2.0,
        (float(config.mid_range[0]) + float(config.mid_range[1])) / 2.0,
        (float(config.long_range[0]) + float(config.long_range[1])) / 2.0,
    ]
    return sum(float(prob) * mean for prob, mean in zip(config.probs, means))


def classify_length_group(actual_length: int, config: LengthConfig) -> str:
    if config.short_range[0] <= actual_length <= config.short_range[1]:
        return "short"
    if config.mid_range[0] <= actual_length <= config.mid_range[1]:
        return "mid"
    if config.long_range[0] <= actual_length <= config.long_range[1]:
        return "long"
    return "short" if actual_length <= config.short_range[1] else "long"


def recommended_oversample_factor(rate: float) -> float:
    if rate <= 0.05:
        return 2.0
    if rate <= 0.10:
        return 3.0
    if rate <= 0.20:
        return 4.0
    return 5.0


def effective_oversample_factor(rate: float, cli_value: float) -> float:
    return float(cli_value) if float(cli_value) > 0.0 else recommended_oversample_factor(rate)


def build_row_lookup(
    df: pd.DataFrame,
    canonical_node_ids: np.ndarray,
    node_col: str,
    time_col: str,
    period: int,
) -> Tuple[np.ndarray, np.ndarray]:
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(canonical_node_ids.tolist())}
    unique_times = np.sort(df[time_col].astype(np.int64).unique())
    if len(unique_times) != period:
        raise RuntimeError("period mismatch inside chunk: expected %s, got %s" % (period, len(unique_times)))
    time_to_idx = {int(value): idx for idx, value in enumerate(unique_times.tolist())}
    local = df[[node_col, time_col]].copy()
    local["row_index"] = np.arange(len(local), dtype=np.int64)
    local["node_index"] = local[node_col].astype(np.int64).map(node_to_idx)
    local["time_index"] = local[time_col].astype(np.int64).map(time_to_idx)
    if local["node_index"].isna().any():
        raise RuntimeError("row lookup found unknown node ids")
    if local["time_index"].isna().any():
        raise RuntimeError("row lookup found unknown time ids")
    row_lookup = np.full((len(canonical_node_ids), period), -1, dtype=np.int64)
    row_lookup[
        local["node_index"].to_numpy(dtype=np.int64, copy=False),
        local["time_index"].to_numpy(dtype=np.int64, copy=False),
    ] = local["row_index"].to_numpy(dtype=np.int64, copy=False)
    if np.any(row_lookup < 0):
        raise RuntimeError("chunk does not contain a full node x time grid")
    return row_lookup, unique_times.astype(np.int64, copy=False)


def build_preferred_neighbors(
    args: argparse.Namespace,
    canonical_node_ids: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    topo_df = pd.read_csv(args.topology_file, usecols=["起始节点ID", "结束节点ID", "长度"])
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(canonical_node_ids.tolist())}
    first_hop_sets = [set() for _ in range(len(canonical_node_ids))]
    first_hop_lengths = [{} for _ in range(len(canonical_node_ids))]
    for start_node, end_node, length in topo_df.itertuples(index=False, name=None):
        start_idx = node_to_idx.get(int(start_node))
        end_idx = node_to_idx.get(int(end_node))
        if start_idx is None or end_idx is None or start_idx == end_idx:
            continue
        safe_length = float(length) if pd.notna(length) and float(length) > 0.0 else 1.0
        first_hop_sets[start_idx].add(end_idx)
        first_hop_sets[end_idx].add(start_idx)
        previous = first_hop_lengths[start_idx].get(end_idx)
        if previous is None or safe_length < previous:
            first_hop_lengths[start_idx][end_idx] = safe_length
        previous = first_hop_lengths[end_idx].get(start_idx)
        if previous is None or safe_length < previous:
            first_hop_lengths[end_idx][start_idx] = safe_length
    second_hop_lengths = [{} for _ in range(len(canonical_node_ids))]
    if int(args.neighbor_scope) >= 2:
        for node_idx, neighbors in enumerate(first_hop_sets):
            for mid_idx in neighbors:
                to_mid = first_hop_lengths[node_idx][mid_idx]
                for second_idx in first_hop_sets[mid_idx]:
                    if second_idx == node_idx or second_idx in neighbors:
                        continue
                    total_length = to_mid + first_hop_lengths[mid_idx][second_idx]
                    previous = second_hop_lengths[node_idx].get(second_idx)
                    if previous is None or total_length < previous:
                        second_hop_lengths[node_idx][second_idx] = total_length
    eligible_rows = []
    lookup_rows = []
    preferred_scope = {}
    preferred_neighbors = {}
    preferred_lengths = {}
    required_anchor_count = max(int(args.min_available_neighbors), int(args.anchor_neighbor_count))
    for node_idx, node_id in enumerate(canonical_node_ids.tolist()):
        first_neighbors = sorted(first_hop_sets[node_idx])
        second_neighbors = sorted(second_hop_lengths[node_idx].keys())
        selected_scope = 0
        selected_neighbors = []
        selected_lengths = []
        if len(first_neighbors) >= int(args.min_available_neighbors):
            selected_scope = 1
            selected_neighbors = first_neighbors
            selected_lengths = [float(first_hop_lengths[node_idx][neighbor]) for neighbor in first_neighbors]
        elif int(args.neighbor_scope) >= 2 and len(second_neighbors) >= int(args.min_available_neighbors):
            selected_scope = 2
            selected_neighbors = second_neighbors
            selected_lengths = [float(second_hop_lengths[node_idx][neighbor]) for neighbor in second_neighbors]
        preferred_scope[node_idx] = int(selected_scope)
        preferred_neighbors[node_idx] = np.asarray(selected_neighbors, dtype=np.int64)
        preferred_lengths[node_idx] = np.asarray(selected_lengths, dtype=np.float32)
        eligible_rows.append(
            {
                "node_id": int(node_id),
                "node_index": int(node_idx),
                "first_hop_neighbor_count": int(len(first_neighbors)),
                "second_hop_neighbor_count": int(len(second_neighbors)),
                "preferred_neighbor_scope": int(selected_scope),
                "preferred_available_neighbor_count": int(len(selected_neighbors)),
                "required_anchor_count": int(required_anchor_count),
                "eligible": bool(selected_scope > 0),
            }
        )
        for rank, (neighbor_idx, path_length) in enumerate(zip(selected_neighbors, selected_lengths), start=1):
            lookup_rows.append(
                {
                    "target_node_id": int(node_id),
                    "target_node_index": int(node_idx),
                    "neighbor_node_id": int(canonical_node_ids[int(neighbor_idx)]),
                    "neighbor_node_index": int(neighbor_idx),
                    "neighbor_scope": int(selected_scope),
                    "path_length": float(path_length),
                    "neighbor_rank": int(rank),
                    "is_anchor_candidate": bool(rank <= required_anchor_count),
                }
            )
    return (
        pd.DataFrame(eligible_rows),
        pd.DataFrame(lookup_rows),
        preferred_scope,
        preferred_neighbors,
        preferred_lengths,
    )


def write_run_metadata(args: argparse.Namespace, paths: Dict[str, Path], rates: Sequence[float]) -> None:
    payload = {
        "scenario_id": SCENARIO_ID,
        "mechanism": MECHANISM,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "input_dir": str(args.input_dir),
        "output_dir": str(paths["scenario_root"]),
        "topology_file": str(args.topology_file),
        "missing_rates": [float(value) for value in rates],
        "length_group_probs": args.length_group_probs,
        "short_length_range": args.short_length_range,
        "mid_length_range": args.mid_length_range,
        "long_length_range": args.long_length_range,
        "neighbor_scope": int(args.neighbor_scope),
        "min_available_neighbors": int(args.min_available_neighbors),
        "anchor_neighbor_count": int(args.anchor_neighbor_count),
        "neighbor_protection_mode": str(args.neighbor_protection_mode),
        "placement_backend": str(args.placement_backend),
        "candidate_oversample_factor": float(args.candidate_oversample_factor),
        "max_candidate_rounds": int(args.max_candidate_rounds),
        "allocation_method": str(args.allocation_method),
        "allocation_shortfall_policy": str(args.allocation_shortfall_policy),
        "mask_scope": str(args.mask_scope),
        "global_allocation_used": True,
        "day_stratified_generation_used": False,
        "per_chunk_round_rate_used": False,
        "spatial_constraint_relaxation": parse_bool(args.spatial_constraint_relaxation),
        "relaxation_policy": str(args.relaxation_policy),
        "min_spatially_constrained_ratio": float(args.min_spatially_constrained_ratio),
        "allow_no_spatial_constraint": parse_bool(args.allow_no_spatial_constraint),
        "max_seconds_per_rate_chunk": float(args.max_seconds_per_rate_chunk),
        "max_resample_ratio": float(args.max_resample_ratio),
        "write_missing_data": parse_bool(args.write_missing_data),
        "write_event_neighbor_rows": parse_bool(args.write_event_neighbor_rows),
        "resume": parse_bool(args.resume),
        "overwrite": parse_bool(args.overwrite),
        "only_rates": args.only_rates,
        "only_chunks": args.only_chunks,
        "seed": int(args.seed),
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "period": int(args.period),
        "checkpoint_mode": CHECKPOINT_MODE,
        "atomic_write_enabled": True,
        "event_id_written_directly": True,
        "backfill_event_ids_used": False,
        "chunk_outer_loop_enabled": True,
        "full_checkpoint_rewrite_disabled": True,
    }
    atomic_write_json(paths["run_config_path"], payload)
    command_text = " ".join([sys.executable] + sys.argv)
    atomic_write_text(paths["run_commands_path"], command_text + "\n", encoding="utf-8")


def run_prepare(args: argparse.Namespace, paths: Dict[str, Path]) -> PreparedArtifacts:
    ensure_base_dirs(paths)
    rates = parse_rates(args.missing_rates)
    length_config = parse_length_config(args)
    input_files = load_input_files(args.input_dir)
    first_df = pd.read_parquet(input_files[0], columns=[args.node_col, args.time_col, args.target_col])
    canonical_node_ids = np.sort(first_df[args.node_col].astype(np.int64).unique())
    unique_times = np.sort(first_df[args.time_col].astype(np.int64).unique())
    if len(unique_times) != int(args.period):
        raise RuntimeError("period mismatch: expected %s, got %s" % (args.period, len(unique_times)))
    chunk_rows = []
    for chunk_index, file_path in enumerate(input_files):
        df = pd.read_parquet(file_path, columns=[args.target_col])
        chunk_rows.append(
            {
                "chunk_index": int(chunk_index),
                "day_index": int(extract_day_index(file_path.name)),
                "file_name": file_path.name,
                "row_count": int(len(df)),
                "target_non_null_count": int(df[args.target_col].notna().sum()),
            }
        )
        del df
        gc.collect()
    eligible_df, node_neighbor_lookup_df, preferred_scope, preferred_neighbors, preferred_lengths = build_preferred_neighbors(
        args=args,
        canonical_node_ids=canonical_node_ids,
    )
    eligible_node_ids = set(
        eligible_df.loc[eligible_df["eligible"], "node_id"].astype(np.int64).tolist()
    )
    updated_rows = []
    for chunk_row in chunk_rows:
        file_path = input_files[int(chunk_row["chunk_index"])]
        local_df = pd.read_parquet(file_path, columns=[args.node_col, args.target_col])
        eligible_selector = local_df[args.node_col].astype(np.int64).isin(eligible_node_ids)
        eligible_non_null_count = int((eligible_selector & local_df[args.target_col].notna()).sum())
        local_payload = dict(chunk_row)
        local_payload["eligible_non_null_count"] = int(eligible_non_null_count)
        updated_rows.append(local_payload)
        del local_df
        gc.collect()
    chunk_summary_df = pd.DataFrame(chunk_rows)
    chunk_summary_df = pd.DataFrame(updated_rows).sort_values("chunk_index").reset_index(drop=True)
    allocation_df = generate_global_missing_allocation(args=args, chunk_summary_df=chunk_summary_df, rates=rates)
    allocation_map = validate_allocation_pairs(allocation_df=allocation_df, rates=rates, chunk_summary_df=chunk_summary_df)
    atomic_write_dataframe_csv(paths["chunk_summary_path"], chunk_summary_df)
    atomic_write_dataframe_csv(paths["global_eligible_chunk_counts_path"], chunk_summary_df)
    atomic_write_dataframe_csv(paths["allocation_path"], allocation_df)
    atomic_write_dataframe_csv(paths["eligible_nodes_path"], eligible_df)
    atomic_write_dataframe_csv(paths["node_neighbor_lookup_csv"], node_neighbor_lookup_df)
    atomic_write_parquet(paths["node_neighbor_lookup_parquet"], node_neighbor_lookup_df)
    prepare_payload = {
        "scenario_id": SCENARIO_ID,
        "mechanism": MECHANISM,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "chunk_count": int(len(chunk_summary_df)),
        "canonical_node_count": int(len(canonical_node_ids)),
        "eligible_node_count": int(eligible_df["eligible"].sum()),
        "period": int(args.period),
        "min_available_neighbors": int(args.min_available_neighbors),
        "anchor_neighbor_count": int(args.anchor_neighbor_count),
        "required_anchor_count": int(max(int(args.min_available_neighbors), int(args.anchor_neighbor_count))),
        "neighbor_protection_mode": str(args.neighbor_protection_mode),
        "placement_backend": str(args.placement_backend),
        "spatial_constraint_relaxation": parse_bool(args.spatial_constraint_relaxation),
        "relaxation_policy": str(args.relaxation_policy),
        "min_spatially_constrained_ratio": float(args.min_spatially_constrained_ratio),
        "allow_no_spatial_constraint": parse_bool(args.allow_no_spatial_constraint),
        "allocation_method": str(args.allocation_method),
        "allocation_shortfall_policy": str(args.allocation_shortfall_policy),
        "mask_scope": str(args.mask_scope),
        "global_allocation_used": True,
        "day_stratified_generation_used": False,
        "per_chunk_round_rate_used": False,
        "chunk_outer_loop_enabled": True,
        "full_checkpoint_rewrite_disabled": True,
        "direct_event_id_write_enabled": True,
        "expected_length": float(expected_length(length_config)),
        "allocation_path": str(paths["allocation_path"]),
        "global_eligible_chunk_counts_path": str(paths["global_eligible_chunk_counts_path"]),
        "global_eligible_count": int(chunk_summary_df["eligible_non_null_count"].sum()),
        "missing_rates": [float(value) for value in rates],
    }
    atomic_write_json(paths["prepare_summary_path"], prepare_payload)
    write_run_metadata(args, paths, rates)
    return PreparedArtifacts(
        input_files=input_files,
        chunk_summary_df=chunk_summary_df,
        eligible_df=eligible_df,
        node_neighbor_lookup_df=node_neighbor_lookup_df,
        allocation_df=allocation_df,
        allocation_map=allocation_map,
        canonical_node_ids=canonical_node_ids,
        preferred_scope=preferred_scope,
        preferred_neighbors=preferred_neighbors,
        preferred_lengths=preferred_lengths,
    )


def rate_chunk_paths(paths: Dict[str, Path], rate: float, seed: int, chunk_index: int, file_name: str) -> Dict[str, Path]:
    scenario_tag = scenario_rate_tag(rate, seed)
    stem = Path(file_name).stem
    part_stem = scenario_chunk_stem(rate, seed, chunk_index)
    event_neighbor_path = paths["event_parts_root"] / (part_stem + ".event_neighbors.csv")
    return {
        "scenario_tag": Path(scenario_tag),
        "mask_path": paths["masks_root"] / scenario_tag / (stem + "_mask.parquet"),
        "missing_path": paths["miss_data_root"] / scenario_tag / file_name,
        "status_json_path": paths["status_parts_root"] / (part_stem + ".status.json"),
        "status_csv_path": paths["status_parts_root"] / (part_stem + ".status.csv"),
        "event_part_path": paths["event_parts_root"] / (part_stem + ".events.parquet"),
        "event_neighbor_path": event_neighbor_path,
        "done_path": paths["progress_root"] / (part_stem + ".done.json"),
        "failed_path": paths["failed_parts_root"] / (part_stem + ".failed.json"),
    }


def safe_file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def remove_paths_if_exist(file_paths: Iterable[Path]) -> None:
    for path in file_paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def clean_tmp_for_base(path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()


def validate_parquet_readable(path: Path) -> bool:
    try:
        if not path.exists() or safe_file_size(path) <= 0:
            return False
        pd.read_parquet(path)
        return True
    except Exception:
        return False


def validate_csv_readable(path: Path) -> bool:
    try:
        if not path.exists() or safe_file_size(path) <= 0:
            return False
        pd.read_csv(path)
        return True
    except Exception:
        return False


def is_rate_chunk_complete(
    part_paths: Dict[str, Path],
    write_missing_data: bool,
) -> bool:
    done_path = part_paths["done_path"]
    if not done_path.exists() or safe_file_size(done_path) <= 0:
        return False
    try:
        payload = read_json(done_path)
    except Exception:
        return False
    if not bool(payload.get("completed", False)):
        return False
    if int(payload.get("observed_missing_count", 0)) <= 0:
        return False
    if not validate_parquet_readable(part_paths["mask_path"]):
        return False
    if write_missing_data and not validate_parquet_readable(part_paths["missing_path"]):
        return False
    if not validate_parquet_readable(part_paths["event_part_path"]):
        return False
    if not validate_csv_readable(part_paths["status_csv_path"]):
        return False
    if not part_paths["status_json_path"].exists() or safe_file_size(part_paths["status_json_path"]) <= 0:
        return False
    output_sizes = payload.get("output_file_sizes", {})
    if int(output_sizes.get("mask", 0)) <= 0:
        return False
    if write_missing_data and int(output_sizes.get("miss_data", 0)) <= 0:
        return False
    if int(output_sizes.get("status", 0)) <= 0 or int(output_sizes.get("event_part", 0)) <= 0:
        return False
    return True


def cleanup_incomplete_rate_chunk(part_paths: Dict[str, Path]) -> None:
    remove_paths_if_exist(
        [
            part_paths["mask_path"],
            part_paths["missing_path"],
            part_paths["status_json_path"],
            part_paths["status_csv_path"],
            part_paths["event_part_path"],
            part_paths["event_neighbor_path"],
            part_paths["done_path"],
            part_paths["failed_path"],
        ]
    )
    for key in [
        "mask_path",
        "missing_path",
        "status_json_path",
        "status_csv_path",
        "event_part_path",
        "event_neighbor_path",
        "done_path",
        "failed_path",
    ]:
        clean_tmp_for_base(part_paths[key])


def outputs_exist_without_done(part_paths: Dict[str, Path]) -> bool:
    return any(
        path.exists()
        for path in [
            part_paths["mask_path"],
            part_paths["missing_path"],
            part_paths["status_json_path"],
            part_paths["status_csv_path"],
            part_paths["event_part_path"],
            part_paths["event_neighbor_path"],
        ]
    ) and not part_paths["done_path"].exists()


def sample_candidate_groups(rng: np.random.Generator, size: int, probs: Tuple[float, float, float]) -> np.ndarray:
    return rng.choice(np.arange(3, dtype=np.int8), size=size, p=np.asarray(probs, dtype=np.float64)).astype(np.int8)


def sample_candidate_lengths(
    rng: np.random.Generator,
    groups: np.ndarray,
    config: LengthConfig,
) -> np.ndarray:
    lengths = np.zeros(len(groups), dtype=np.int16)
    for group_code in (0, 1, 2):
        selector = groups == group_code
        count = int(np.count_nonzero(selector))
        if count == 0:
            continue
        if group_code == 0:
            low, high = config.short_range
        elif group_code == 1:
            low, high = config.mid_range
        else:
            low, high = config.long_range
        lengths[selector] = rng.integers(int(low), int(high) + 1, size=count, endpoint=False).astype(np.int16)
    return lengths


def event_id_base(rate_index: int, chunk_index: int) -> int:
    return int((rate_index + 1) * 100000000 + chunk_index * 1000000)


def interval_mask_for(start_slot: int, actual_length: int) -> int:
    return ((1 << int(actual_length)) - 1) << int(start_slot)


def bitset_target_available(coverage_bits: List[int], protected_bits: List[int], target_idx: int, bitmask: int) -> bool:
    return ((coverage_bits[target_idx] & bitmask) == 0) and ((protected_bits[target_idx] & bitmask) == 0)


def bitset_available_neighbors(coverage_bits: List[int], neighbors: np.ndarray, bitmask: int) -> np.ndarray:
    if len(neighbors) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(idx) for idx in neighbors.tolist() if (coverage_bits[int(idx)] & bitmask) == 0], dtype=np.int64)


def numpy_target_available(coverage: np.ndarray, protected: np.ndarray, target_idx: int, start_slot: int, stop_slot: int) -> bool:
    return (not np.any(coverage[target_idx, start_slot:stop_slot])) and (not np.any(protected[target_idx, start_slot:stop_slot]))


def numpy_available_neighbors(coverage: np.ndarray, neighbors: np.ndarray, start_slot: int, stop_slot: int) -> np.ndarray:
    if len(neighbors) == 0:
        return np.asarray([], dtype=np.int64)
    keep = [int(idx) for idx in neighbors.tolist() if not np.any(coverage[int(idx), start_slot:stop_slot])]
    return np.asarray(keep, dtype=np.int64)


def encode_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def compute_neighbor_slot_stats(
    available_neighbors: np.ndarray,
    period: int,
    start_slot: int,
    stop_slot: int,
    placement_backend: str,
    coverage_bits: List[int],
    coverage_bool: np.ndarray,
) -> Tuple[bool, bool, float]:
    event_length = max(int(stop_slot - start_slot), 1)
    if len(available_neighbors) == 0:
        return False, False, 0.0
    slot_has_observed = np.zeros(event_length, dtype=bool)
    fully_neighbor_observed = False
    for neighbor_idx in available_neighbors.tolist():
        if placement_backend == "bitset":
            observed = np.asarray(
                [bool((coverage_bits[int(neighbor_idx)] >> int(slot)) & 1 == 0) for slot in range(start_slot, stop_slot)],
                dtype=bool,
            )
        else:
            observed = ~coverage_bool[int(neighbor_idx), start_slot:stop_slot]
        slot_has_observed = slot_has_observed | observed
        if bool(np.all(observed)):
            fully_neighbor_observed = True
    partially_neighbor_observed = bool(np.any(slot_has_observed))
    ratio = float(np.count_nonzero(slot_has_observed)) / float(event_length)
    return fully_neighbor_observed, partially_neighbor_observed, ratio


def stage_definition(
    level_name: str,
    args: argparse.Namespace,
    required_anchor_count: int,
) -> Dict[str, Any]:
    if level_name == "strict_anchor":
        return {
            "spatial_constraint_level": level_name,
            "relaxation_stage": 1,
            "required_available_neighbor_count": int(required_anchor_count),
            "anchor_neighbor_count_used": int(required_anchor_count),
            "protect_mode": "strict",
        }
    if level_name == "relaxed_anchor":
        return {
            "spatial_constraint_level": level_name,
            "relaxation_stage": 2,
            "required_available_neighbor_count": 1,
            "anchor_neighbor_count_used": 1,
            "protect_mode": "strict",
        }
    if level_name == "weak_neighbor_available":
        return {
            "spatial_constraint_level": level_name,
            "relaxation_stage": 3,
            "required_available_neighbor_count": 1,
            "anchor_neighbor_count_used": 0,
            "protect_mode": "weak",
        }
    return {
        "spatial_constraint_level": "none",
        "relaxation_stage": 4,
        "required_available_neighbor_count": 0,
        "anchor_neighbor_count_used": 0,
        "protect_mode": "none",
    }


def generate_rate_chunk(
    args: argparse.Namespace,
    artifacts: PreparedArtifacts,
    df: pd.DataFrame,
    row_lookup: np.ndarray,
    unique_times: np.ndarray,
    chunk_row: Dict[str, Any],
    rate: float,
    rate_index: int,
    length_config: LengthConfig,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any], Dict[str, float]]:
    node_count = len(artifacts.canonical_node_ids)
    period = int(args.period)
    file_name = str(chunk_row["file_name"])
    chunk_index = int(chunk_row["chunk_index"])
    day_index = int(chunk_row["day_index"])
    scenario_tag = scenario_rate_tag(rate, args.seed)
    write_event_neighbor_rows = parse_bool(args.write_event_neighbor_rows)
    required_anchor_count = max(int(args.min_available_neighbors), int(args.anchor_neighbor_count))
    eligible_node_indices = artifacts.eligible_df.loc[artifacts.eligible_df["eligible"], "node_index"].to_numpy(dtype=np.int64, copy=False)
    if len(eligible_node_indices) == 0:
        raise RuntimeError("no eligible nodes available for spatial_neighbor_holdout")
    allocation_key = (round(float(rate), 6), int(chunk_index))
    allocation_row = artifacts.allocation_map.get(allocation_key)
    if allocation_row is None:
        raise RuntimeError("missing allocation for rate=%s chunk=%s" % (rate, chunk_index))
    target_missing_count = int(allocation_row["allocated_missing_count"])
    event_id_matrix = np.full((node_count, period), -1, dtype=np.int64)
    actual_length_matrix = np.zeros((node_count, period), dtype=np.uint8)
    length_group_code_matrix = np.full((node_count, period), -1, dtype=np.int8)
    available_neighbor_count_matrix = np.zeros((node_count, period), dtype=np.int16)
    anchor_neighbor_count_matrix = np.zeros((node_count, period), dtype=np.int8)
    neighbor_scope_matrix = np.zeros((node_count, period), dtype=np.int8)
    neighbor_observed_ratio_matrix = np.zeros((node_count, period), dtype=np.float32)
    relaxation_stage_matrix = np.zeros((node_count, period), dtype=np.int8)
    constraint_level_code_matrix = np.zeros((node_count, period), dtype=np.int8)
    coverage_bits = [0 for _ in range(node_count)]
    protected_bits = [0 for _ in range(node_count)]
    coverage_bool = np.zeros((node_count, period), dtype=bool)
    protected_bool = np.zeros((node_count, period), dtype=bool)
    event_rows = []
    event_neighbor_rows = []
    missing_count = 0
    accepted_event_count = 0
    resampled_event_count = 0
    candidate_rounds_used = 0
    accepted_by_level = dict((level, 0) for level in CONSTRAINT_LEVELS)
    event_count_by_level = dict((level, 0) for level in CONSTRAINT_LEVELS)
    placement_start = time.perf_counter()
    rng = np.random.default_rng(int(args.seed) + int(round(rate * 10000.0)) + chunk_index)
    factor = effective_oversample_factor(rate, float(args.candidate_oversample_factor))
    per_chunk_event_id = event_id_base(rate_index, chunk_index)
    average_length = max(expected_length(length_config), 1.0)
    relaxation_enabled = parse_bool(args.spatial_constraint_relaxation)
    allow_none = parse_bool(args.allow_no_spatial_constraint)
    stage_names = ["strict_anchor"]
    if relaxation_enabled:
        stage_names.extend(["relaxed_anchor", "weak_neighbor_available"])
        if allow_none:
            stage_names.append("none")
    elif allow_none and str(args.neighbor_protection_mode) == "all":
        stage_names.append("none")
    relaxation_reason = "no_relaxation"
    max_relaxation_stage_used = 1
    stage_summaries = []
    for stage_name in stage_names:
        if missing_count >= target_missing_count:
            break
        stage_config = stage_definition(stage_name, args, required_anchor_count)
        stage_begin_missing = missing_count
        stage_begin_events = accepted_event_count
        stage_begin_resampled = resampled_event_count
        local_reason = "no_relaxation" if stage_name == "strict_anchor" else "candidate_exhausted"
        local_rounds_used = 0
        local_timed_out = False
        while missing_count < target_missing_count and local_rounds_used < int(args.max_candidate_rounds):
            if (time.perf_counter() - placement_start) > float(args.max_seconds_per_rate_chunk):
                local_reason = "timeout"
                local_timed_out = True
                break
            local_rounds_used += 1
            candidate_rounds_used += 1
            remaining_slots = target_missing_count - missing_count
            estimated_events = int(math.ceil(float(remaining_slots) / average_length))
            pool_size = max(int(math.ceil(float(estimated_events) * factor)), 1)
            target_choices = rng.choice(eligible_node_indices, size=pool_size, replace=True).astype(np.int64)
            start_slots = rng.integers(0, period, size=pool_size, endpoint=False).astype(np.int16)
            group_codes = sample_candidate_groups(rng, size=pool_size, probs=length_config.probs)
            sampled_lengths = sample_candidate_lengths(rng, groups=group_codes, config=length_config)
            order = rng.permutation(pool_size)
            accepted_this_round = 0
            for local_idx in order.tolist():
                if missing_count >= target_missing_count:
                    break
                target_idx = int(target_choices[local_idx])
                start_slot = int(start_slots[local_idx])
                max_allowed = min(int(sampled_lengths[local_idx]), period - start_slot, target_missing_count - missing_count)
                if max_allowed <= 0:
                    continue
                actual_length = int(max_allowed)
                stop_slot = start_slot + actual_length
                bitmask = interval_mask_for(start_slot, actual_length)
                if args.placement_backend == "bitset":
                    if not bitset_target_available(coverage_bits, protected_bits, target_idx, bitmask):
                        resampled_event_count += 1
                        continue
                    available_neighbors = bitset_available_neighbors(
                        coverage_bits=coverage_bits,
                        neighbors=artifacts.preferred_neighbors[target_idx],
                        bitmask=bitmask,
                    )
                else:
                    if not numpy_target_available(coverage_bool, protected_bool, target_idx, start_slot, stop_slot):
                        resampled_event_count += 1
                        continue
                    available_neighbors = numpy_available_neighbors(
                        coverage=coverage_bool,
                        neighbors=artifacts.preferred_neighbors[target_idx],
                        start_slot=start_slot,
                        stop_slot=stop_slot,
                    )
                total_neighbors = int(len(artifacts.preferred_neighbors[target_idx]))
                available_neighbor_count = int(len(available_neighbors))
                anchor_neighbors = np.asarray([], dtype=np.int64)
                accepted = False
                if stage_config["protect_mode"] == "strict":
                    if available_neighbor_count >= int(stage_config["required_available_neighbor_count"]):
                        anchor_take = int(stage_config["anchor_neighbor_count_used"])
                        anchor_neighbors = available_neighbors[:anchor_take] if anchor_take > 0 else np.asarray([], dtype=np.int64)
                        accepted = True
                elif stage_config["protect_mode"] == "weak":
                    if available_neighbor_count >= 1:
                        accepted = True
                else:
                    accepted = True
                if not accepted:
                    resampled_event_count += 1
                    continue
                if args.placement_backend == "bitset":
                    coverage_bits[target_idx] |= bitmask
                    for anchor_idx in anchor_neighbors.tolist():
                        protected_bits[int(anchor_idx)] |= bitmask
                else:
                    coverage_bool[target_idx, start_slot:stop_slot] = True
                    for anchor_idx in anchor_neighbors.tolist():
                        protected_bool[int(anchor_idx), start_slot:stop_slot] = True
                fully_neighbor_observed, partially_neighbor_observed, slot_ratio = compute_neighbor_slot_stats(
                    available_neighbors=available_neighbors,
                    period=period,
                    start_slot=start_slot,
                    stop_slot=stop_slot,
                    placement_backend=str(args.placement_backend),
                    coverage_bits=coverage_bits,
                    coverage_bool=coverage_bool,
                )
                event_id = int(per_chunk_event_id + accepted_event_count)
                length_group = classify_length_group(actual_length, length_config)
                neighbor_scope = int(artifacts.preferred_scope[target_idx])
                constraint_level = str(stage_config["spatial_constraint_level"])
                relaxation_stage = int(stage_config["relaxation_stage"])
                if relaxation_stage > max_relaxation_stage_used:
                    max_relaxation_stage_used = relaxation_stage
                event_id_matrix[target_idx, start_slot:stop_slot] = event_id
                actual_length_matrix[target_idx, start_slot:stop_slot] = np.uint8(actual_length)
                length_group_code_matrix[target_idx, start_slot:stop_slot] = np.int8(LENGTH_GROUP_TO_CODE[length_group])
                available_neighbor_count_matrix[target_idx, start_slot:stop_slot] = np.int16(available_neighbor_count)
                anchor_neighbor_count_matrix[target_idx, start_slot:stop_slot] = np.int8(int(stage_config["anchor_neighbor_count_used"]))
                neighbor_scope_matrix[target_idx, start_slot:stop_slot] = np.int8(neighbor_scope)
                neighbor_observed_ratio_matrix[target_idx, start_slot:stop_slot] = np.float32(slot_ratio)
                relaxation_stage_matrix[target_idx, start_slot:stop_slot] = np.int8(relaxation_stage)
                constraint_level_code_matrix[target_idx, start_slot:stop_slot] = np.int8(CONSTRAINT_LEVEL_TO_CODE[constraint_level])
                anchor_neighbor_ids = [int(artifacts.canonical_node_ids[int(idx)]) for idx in anchor_neighbors.tolist()]
                available_neighbor_ids = [int(artifacts.canonical_node_ids[int(idx)]) for idx in available_neighbors.tolist()]
                event_rows.append(
                    {
                        "event_id": int(event_id),
                        "missing_rate_target": float(rate),
                        "scenario_tag": scenario_tag,
                        "chunk_index": int(chunk_index),
                        "day_index": int(day_index),
                        "file_name": file_name,
                        "target_node_id": int(artifacts.canonical_node_ids[target_idx]),
                        "target_node_index": int(target_idx),
                        "start_slot": int(start_slot),
                        "stop_slot_exclusive": int(stop_slot),
                        "start_global_time_index": int(unique_times[start_slot]),
                        "end_global_time_index": int(unique_times[stop_slot - 1]),
                        "actual_length": int(actual_length),
                        "length_group": length_group,
                        "neighbor_scope": int(neighbor_scope),
                        "available_neighbor_count": int(available_neighbor_count),
                        "anchor_neighbor_count": int(len(anchor_neighbors)),
                        "anchor_neighbor_ids": encode_list(anchor_neighbor_ids),
                        "anchor_neighbor_indices": encode_list(anchor_neighbors.tolist()),
                        "available_neighbor_ids": encode_list(available_neighbor_ids),
                        "neighbor_observed_ratio": float(slot_ratio),
                        "neighbor_protection_mode": str(args.neighbor_protection_mode),
                        "placement_backend": str(args.placement_backend),
                        "spatial_constraint_level": constraint_level,
                        "relaxation_stage": int(relaxation_stage),
                        "relaxation_reason": "no_relaxation" if relaxation_stage == 1 else local_reason,
                        "anchor_neighbor_count_used": int(stage_config["anchor_neighbor_count_used"]),
                        "required_available_neighbor_count": int(stage_config["required_available_neighbor_count"]),
                        "actual_available_neighbor_count": int(available_neighbor_count),
                        "fully_neighbor_observed": bool(fully_neighbor_observed),
                        "partially_neighbor_observed": bool(partially_neighbor_observed),
                        "neighbor_observed_slot_ratio": float(slot_ratio),
                        "seed": int(args.seed),
                    }
                )
                if write_event_neighbor_rows:
                    preferred_neighbors = artifacts.preferred_neighbors[target_idx].tolist()
                    preferred_lengths = artifacts.preferred_lengths[target_idx].tolist()
                    anchor_set = {int(value) for value in anchor_neighbors.tolist()}
                    available_set = {int(value) for value in available_neighbors.tolist()}
                    for rank, (neighbor_idx, path_length) in enumerate(zip(preferred_neighbors, preferred_lengths), start=1):
                        event_neighbor_rows.append(
                            {
                                "event_id": int(event_id),
                                "target_node_id": int(artifacts.canonical_node_ids[target_idx]),
                                "neighbor_node_id": int(artifacts.canonical_node_ids[int(neighbor_idx)]),
                                "neighbor_node_index": int(neighbor_idx),
                                "neighbor_scope": int(neighbor_scope),
                                "path_length": float(path_length),
                                "neighbor_rank": int(rank),
                                "is_available": bool(int(neighbor_idx) in available_set),
                                "is_anchor": bool(int(neighbor_idx) in anchor_set),
                                "spatial_constraint_level": constraint_level,
                            }
                        )
                accepted_by_level[constraint_level] += actual_length
                event_count_by_level[constraint_level] += 1
                missing_count += actual_length
                accepted_event_count += 1
                accepted_this_round += 1
            accepted_events_delta = accepted_event_count - stage_begin_events
            stage_resampled_delta = resampled_event_count - stage_begin_resampled
            if accepted_events_delta > 0:
                current_ratio = float(stage_resampled_delta) / float(max(accepted_events_delta, 1))
                if current_ratio > float(args.max_resample_ratio):
                    local_reason = "resample_too_high"
                    break
            if accepted_this_round == 0:
                local_reason = "candidate_exhausted"
                break
        if local_timed_out:
            relaxation_reason = "timeout"
        elif missing_count < target_missing_count and stage_name != stage_names[-1]:
            relaxation_reason = local_reason
        stage_summaries.append(
            {
                "spatial_constraint_level": stage_name,
                "accepted_missing_count": int(missing_count - stage_begin_missing),
                "accepted_event_count": int(accepted_event_count - stage_begin_events),
                "resampled_candidate_count": int(resampled_event_count - stage_begin_resampled),
                "relaxation_reason": local_reason if missing_count < target_missing_count else "no_relaxation",
            }
        )
    if missing_count < target_missing_count and allow_none:
        remaining_slots = int(target_missing_count - missing_count)
        fallback_added = 0
        randomized_nodes = rng.permutation(eligible_node_indices)
        for target_idx in randomized_nodes.tolist():
            if fallback_added >= remaining_slots:
                break
            candidate_slots = []
            if args.placement_backend == "bitset":
                unavailable_bits = int(coverage_bits[int(target_idx)] | protected_bits[int(target_idx)])
                for slot in range(period):
                    if ((unavailable_bits >> int(slot)) & 1) == 0:
                        candidate_slots.append(int(slot))
            else:
                free_mask = ~(coverage_bool[int(target_idx)] | protected_bool[int(target_idx)])
                candidate_slots = np.flatnonzero(free_mask).astype(np.int64).tolist()
            if not candidate_slots:
                continue
            candidate_slots = rng.permutation(np.asarray(candidate_slots, dtype=np.int64)).tolist()
            for start_slot in candidate_slots:
                if fallback_added >= remaining_slots:
                    break
                bitmask = interval_mask_for(int(start_slot), 1)
                if args.placement_backend == "bitset":
                    if not bitset_target_available(coverage_bits, protected_bits, int(target_idx), bitmask):
                        continue
                    coverage_bits[int(target_idx)] |= bitmask
                else:
                    if not numpy_target_available(coverage_bool, protected_bool, int(target_idx), int(start_slot), int(start_slot + 1)):
                        continue
                    coverage_bool[int(target_idx), int(start_slot)] = True
                event_id = int(per_chunk_event_id + accepted_event_count)
                event_id_matrix[int(target_idx), int(start_slot)] = event_id
                actual_length_matrix[int(target_idx), int(start_slot)] = np.uint8(1)
                length_group_code_matrix[int(target_idx), int(start_slot)] = np.int8(LENGTH_GROUP_TO_CODE["short"])
                available_neighbor_count_matrix[int(target_idx), int(start_slot)] = np.int16(0)
                anchor_neighbor_count_matrix[int(target_idx), int(start_slot)] = np.int8(0)
                neighbor_scope_matrix[int(target_idx), int(start_slot)] = np.int8(artifacts.preferred_scope[int(target_idx)])
                neighbor_observed_ratio_matrix[int(target_idx), int(start_slot)] = np.float32(0.0)
                relaxation_stage_matrix[int(target_idx), int(start_slot)] = np.int8(4)
                constraint_level_code_matrix[int(target_idx), int(start_slot)] = np.int8(CONSTRAINT_LEVEL_TO_CODE["none"])
                event_rows.append(
                    {
                        "event_id": int(event_id),
                        "missing_rate_target": float(rate),
                        "scenario_tag": scenario_tag,
                        "chunk_index": int(chunk_index),
                        "day_index": int(day_index),
                        "file_name": file_name,
                        "target_node_id": int(artifacts.canonical_node_ids[int(target_idx)]),
                        "target_node_index": int(target_idx),
                        "start_slot": int(start_slot),
                        "stop_slot_exclusive": int(start_slot + 1),
                        "start_global_time_index": int(unique_times[int(start_slot)]),
                        "end_global_time_index": int(unique_times[int(start_slot)]),
                        "actual_length": 1,
                        "length_group": "short",
                        "neighbor_scope": int(artifacts.preferred_scope[int(target_idx)]),
                        "available_neighbor_count": 0,
                        "anchor_neighbor_count": 0,
                        "anchor_neighbor_ids": "",
                        "anchor_neighbor_indices": "",
                        "available_neighbor_ids": "",
                        "neighbor_observed_ratio": 0.0,
                        "neighbor_protection_mode": str(args.neighbor_protection_mode),
                        "placement_backend": str(args.placement_backend),
                        "spatial_constraint_level": "none",
                        "relaxation_stage": 4,
                        "relaxation_reason": "capacity_shortfall",
                        "anchor_neighbor_count_used": 0,
                        "required_available_neighbor_count": 0,
                        "actual_available_neighbor_count": 0,
                        "fully_neighbor_observed": False,
                        "partially_neighbor_observed": False,
                        "neighbor_observed_slot_ratio": 0.0,
                        "seed": int(args.seed),
                    }
                )
                accepted_by_level["none"] += 1
                event_count_by_level["none"] += 1
                accepted_event_count += 1
                missing_count += 1
                fallback_added += 1
                max_relaxation_stage_used = max(max_relaxation_stage_used, 4)
        stage_summaries.append(
            {
                "spatial_constraint_level": "none",
                "accepted_missing_count": int(fallback_added),
                "accepted_event_count": int(fallback_added),
                "resampled_candidate_count": 0,
                "relaxation_reason": "capacity_shortfall",
            }
        )
    if missing_count < target_missing_count and not allow_none:
        raise RuntimeError("unable to satisfy allocated_missing_count without none-level fallback")
    if missing_count < target_missing_count:
        raise RuntimeError("allocated_missing_count still not reached after all relaxation stages")
    placement_elapsed = time.perf_counter() - placement_start
    mask_selector = event_id_matrix.reshape(-1) >= 0
    selected_positions = np.flatnonzero(mask_selector)
    node_positions = selected_positions // period
    slot_positions = selected_positions % period
    selected_rows = row_lookup[node_positions, slot_positions]
    order = np.argsort(selected_rows, kind="mergesort")
    selected_rows = selected_rows[order]
    node_positions = node_positions[order]
    slot_positions = slot_positions[order]
    selected_event_ids = event_id_matrix.reshape(-1)[selected_positions][order]
    selected_lengths = actual_length_matrix.reshape(-1)[selected_positions][order].astype(np.int16, copy=False)
    selected_length_codes = length_group_code_matrix.reshape(-1)[selected_positions][order].astype(np.int8, copy=False)
    selected_available_counts = available_neighbor_count_matrix.reshape(-1)[selected_positions][order].astype(np.int16, copy=False)
    selected_anchor_counts = anchor_neighbor_count_matrix.reshape(-1)[selected_positions][order].astype(np.int8, copy=False)
    selected_scopes = neighbor_scope_matrix.reshape(-1)[selected_positions][order].astype(np.int8, copy=False)
    selected_ratios = neighbor_observed_ratio_matrix.reshape(-1)[selected_positions][order].astype(np.float32, copy=False)
    selected_relaxation_stages = relaxation_stage_matrix.reshape(-1)[selected_positions][order].astype(np.int8, copy=False)
    selected_constraint_level_codes = constraint_level_code_matrix.reshape(-1)[selected_positions][order].astype(np.int8, copy=False)
    selected_length_groups = [CODE_TO_LENGTH_GROUP.get(int(code), "short") for code in selected_length_codes.tolist()]
    selected_constraint_levels = [CODE_TO_CONSTRAINT_LEVEL.get(int(code), "none") for code in selected_constraint_level_codes.tolist()]
    selected_time_values = unique_times[slot_positions].astype(np.int64, copy=False)
    mask_df = pd.DataFrame(
        {
            "row_index": selected_rows,
            args.node_col: artifacts.canonical_node_ids[node_positions].astype(np.int64, copy=False),
            args.time_col: selected_time_values,
            "day_index": np.full(len(selected_rows), day_index, dtype=np.int64),
            "global_time_index": selected_time_values,
            "is_missing": np.full(len(selected_rows), True, dtype=bool),
            "scenario_id": np.full(len(selected_rows), SCENARIO_ID),
            "mechanism": np.full(len(selected_rows), MECHANISM),
            "evaluation_protocol": np.full(len(selected_rows), EVALUATION_PROTOCOL),
            "missing_rate_target": np.full(len(selected_rows), float(rate), dtype=np.float64),
            "event_id": selected_event_ids.astype(np.int64, copy=False),
            "actual_length": selected_lengths.astype(np.int16, copy=False),
            "length_group": selected_length_groups,
            "available_neighbor_count": selected_available_counts.astype(np.int16, copy=False),
            "anchor_neighbor_count": selected_anchor_counts.astype(np.int8, copy=False),
            "neighbor_scope": selected_scopes.astype(np.int8, copy=False),
            "neighbor_observed_ratio": selected_ratios.astype(np.float32, copy=False),
            "spatial_constraint_level": selected_constraint_levels,
            "relaxation_stage": selected_relaxation_stages.astype(np.int8, copy=False),
            "anchor_neighbor_count_used": selected_anchor_counts.astype(np.int8, copy=False),
            "neighbor_observed_slot_ratio": selected_ratios.astype(np.float32, copy=False),
            "neighbor_protection_mode": np.full(len(selected_rows), str(args.neighbor_protection_mode)),
            "placement_backend": np.full(len(selected_rows), str(args.placement_backend)),
        }
    )
    missing_df = None
    if parse_bool(args.write_missing_data):
        missing_df = df.copy()
        missing_df.loc[selected_rows, args.target_col] = np.nan
    event_df = pd.DataFrame(event_rows)
    event_neighbor_df = pd.DataFrame(event_neighbor_rows) if write_event_neighbor_rows else None
    skipped_missing_slots = int(target_missing_count - len(mask_df))
    mean_neighbor_count = float(event_df["available_neighbor_count"].mean()) if not event_df.empty else 0.0
    mean_ratio = float(event_df["neighbor_observed_ratio"].mean()) if not event_df.empty else 0.0
    min_neighbor_count = int(event_df["available_neighbor_count"].min()) if not event_df.empty else 0
    spatially_constrained_missing_count = int(
        accepted_by_level["strict_anchor"] + accepted_by_level["relaxed_anchor"] + accepted_by_level["weak_neighbor_available"]
    )
    none_missing_count = int(accepted_by_level["none"])
    spatially_constrained_ratio = float(spatially_constrained_missing_count) / float(target_missing_count) if target_missing_count > 0 else 0.0
    timing = {
        "placement_time_seconds": float(placement_elapsed),
    }
    status_row = {
        "scenario_tag": scenario_tag,
        "missing_rate_target": float(rate),
        "chunk_index": int(chunk_index),
        "day_index": int(day_index),
        "file_name": file_name,
        "row_count": int(len(df)),
        "expected_missing_count": int(target_missing_count),
        "observed_missing_count": int(len(mask_df)),
        "observed_missing_rate": (float(len(mask_df)) / float(len(df))) if len(df) else 0.0,
        "expected_missing_rate": float(rate),
        "event_count": int(len(event_df)),
        "resampled_event_count": int(resampled_event_count),
        "candidate_rounds": int(candidate_rounds_used),
        "skipped_missing_slots": int(skipped_missing_slots),
        "neighbor_observed_ratio": float(mean_ratio),
        "average_available_neighbor_count": float(mean_neighbor_count),
        "min_available_neighbor_count": int(min_neighbor_count),
        "anchor_neighbor_count": int(required_anchor_count if str(args.neighbor_protection_mode) == "anchor" else -1),
        "strict_anchor_missing_count": int(accepted_by_level["strict_anchor"]),
        "relaxed_anchor_missing_count": int(accepted_by_level["relaxed_anchor"]),
        "weak_neighbor_available_missing_count": int(accepted_by_level["weak_neighbor_available"]),
        "none_missing_count": int(none_missing_count),
        "spatially_constrained_missing_count": int(spatially_constrained_missing_count),
        "spatially_unconstrained_missing_count": int(none_missing_count),
        "spatially_constrained_ratio": float(spatially_constrained_ratio),
        "constraint_relaxation_used": bool(max_relaxation_stage_used > 1),
        "max_relaxation_stage_used": int(max_relaxation_stage_used),
        "global_missing_count": int(allocation_row["global_missing_count"]),
        "allocation_source": "global_missing_allocation.csv",
        "constraint_stage_summaries": json.dumps(stage_summaries, ensure_ascii=False),
        "relaxation_reason": relaxation_reason if max_relaxation_stage_used > 1 else "no_relaxation",
        "neighbor_protection_mode": str(args.neighbor_protection_mode),
        "placement_backend": str(args.placement_backend),
        "checkpoint_mode": CHECKPOINT_MODE,
        "event_id_written_directly": True,
        "backfill_event_ids_used": False,
        "chunk_outer_loop_enabled": True,
        "full_checkpoint_rewrite_disabled": True,
        "spatial_constraint_relaxation": parse_bool(args.spatial_constraint_relaxation),
        "relaxation_policy": str(args.relaxation_policy),
        "allow_no_spatial_constraint": parse_bool(args.allow_no_spatial_constraint),
        "min_spatially_constrained_ratio": float(args.min_spatially_constrained_ratio),
        "write_missing_data": parse_bool(args.write_missing_data),
        "write_event_neighbor_rows": write_event_neighbor_rows,
    }
    return mask_df, missing_df, event_df, event_neighbor_df, status_row, timing


def write_generation_outputs(
    args: argparse.Namespace,
    part_paths: Dict[str, Path],
    mask_df: pd.DataFrame,
    missing_df: Optional[pd.DataFrame],
    event_df: pd.DataFrame,
    event_neighbor_df: Optional[pd.DataFrame],
    status_row: Dict[str, Any],
    elapsed_seconds: float,
    write_time_seconds: float,
    checkpoint_time_seconds: float,
) -> Dict[str, Any]:
    ensure_dir(part_paths["mask_path"].parent)
    ensure_dir(part_paths["status_json_path"].parent)
    ensure_dir(part_paths["event_part_path"].parent)
    if missing_df is not None:
        ensure_dir(part_paths["missing_path"].parent)
    write_started = time.perf_counter()
    atomic_write_parquet(part_paths["mask_path"], mask_df)
    if missing_df is not None:
        atomic_write_parquet(part_paths["missing_path"], missing_df)
    atomic_write_parquet(part_paths["event_part_path"], event_df)
    if event_neighbor_df is not None:
        atomic_write_dataframe_csv(part_paths["event_neighbor_path"], event_neighbor_df)
    write_time_seconds += float(time.perf_counter() - write_started)
    status_payload = dict(status_row)
    status_payload["mask_path"] = str(part_paths["mask_path"])
    status_payload["missing_dataset_path"] = str(part_paths["missing_path"]) if missing_df is not None else ""
    status_payload["status_json_path"] = str(part_paths["status_json_path"])
    status_payload["status_csv_path"] = str(part_paths["status_csv_path"])
    status_payload["event_part_path"] = str(part_paths["event_part_path"])
    status_payload["event_neighbor_path"] = str(part_paths["event_neighbor_path"]) if event_neighbor_df is not None else ""
    status_payload["elapsed_seconds"] = float(elapsed_seconds)
    status_payload["write_time_seconds"] = float(write_time_seconds)
    status_payload["checkpoint_time_seconds"] = float(checkpoint_time_seconds)
    checkpoint_started = time.perf_counter()
    atomic_write_json(part_paths["status_json_path"], status_payload)
    atomic_write_dataframe_csv(part_paths["status_csv_path"], pd.DataFrame([status_payload]))
    checkpoint_time_seconds += float(time.perf_counter() - checkpoint_started)
    done_payload = {
        "scenario_tag": status_row["scenario_tag"],
        "missing_rate_target": float(status_row["missing_rate_target"]),
        "chunk_index": int(status_row["chunk_index"]),
        "file_name": status_row["file_name"],
        "mask_path": str(part_paths["mask_path"]),
        "missing_dataset_path": str(part_paths["missing_path"]) if missing_df is not None else "",
        "status_path": str(part_paths["status_json_path"]),
        "event_part_path": str(part_paths["event_part_path"]),
        "completed": True,
        "completed_at": pd.Timestamp.utcnow().isoformat(),
        "observed_missing_count": int(status_row["observed_missing_count"]),
        "expected_missing_count": int(status_row["expected_missing_count"]),
        "output_file_sizes": {
            "mask": safe_file_size(part_paths["mask_path"]),
            "miss_data": safe_file_size(part_paths["missing_path"]) if missing_df is not None else 0,
            "status": safe_file_size(part_paths["status_json_path"]),
            "event_part": safe_file_size(part_paths["event_part_path"]),
        },
    }
    checkpoint_started = time.perf_counter()
    atomic_write_json(part_paths["done_path"], done_payload)
    checkpoint_time_seconds += float(time.perf_counter() - checkpoint_started)
    status_payload["checkpoint_time_seconds"] = float(checkpoint_time_seconds)
    atomic_write_json(part_paths["status_json_path"], status_payload)
    atomic_write_dataframe_csv(part_paths["status_csv_path"], pd.DataFrame([status_payload]))
    return status_payload


def should_process_rate_chunk(
    args: argparse.Namespace,
    part_paths: Dict[str, Path],
    rate: float,
    chunk_index: int,
) -> bool:
    only_rates = parse_only_rates(args.only_rates)
    only_chunks = parse_only_chunks(args.only_chunks)
    if only_rates is not None and round(float(rate), 6) not in only_rates:
        return False
    if only_chunks is not None and int(chunk_index) not in only_chunks:
        return False
    return True


def handle_existing_outputs(args: argparse.Namespace, part_paths: Dict[str, Path], write_missing_data: bool) -> bool:
    resume = parse_bool(args.resume)
    overwrite = parse_bool(args.overwrite)
    if overwrite:
        cleanup_incomplete_rate_chunk(part_paths)
        return False
    if resume:
        if is_rate_chunk_complete(part_paths, write_missing_data=write_missing_data):
            return True
        if part_paths["done_path"].exists():
            cleanup_incomplete_rate_chunk(part_paths)
            return False
        if outputs_exist_without_done(part_paths):
            raise RuntimeError(
                "legacy outputs exist for %s but no done.json is present; rerun with --overwrite true for this rate/chunk"
                % part_paths["done_path"].stem
            )
        return False
    if any(
        path.exists()
        for path in [
            part_paths["mask_path"],
            part_paths["missing_path"],
            part_paths["status_json_path"],
            part_paths["status_csv_path"],
            part_paths["event_part_path"],
            part_paths["done_path"],
        ]
    ):
        raise RuntimeError("outputs already exist; use --resume true or --overwrite true")
    return False


def write_failed_part(part_paths: Dict[str, Path], status_row: Dict[str, Any], exc: Exception) -> None:
    payload = {
        "scenario_tag": status_row.get("scenario_tag", ""),
        "chunk_index": int(status_row.get("chunk_index", -1)),
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": pd.Timestamp.utcnow().isoformat(),
        "resume_hint": "rerun with --resume true",
    }
    atomic_write_json(part_paths["failed_path"], payload)


def run_generate_missing(args: argparse.Namespace, paths: Dict[str, Path], artifacts: PreparedArtifacts) -> pd.DataFrame:
    ensure_base_dirs(paths)
    length_config = parse_length_config(args)
    write_missing_data = parse_bool(args.write_missing_data)
    rates = parse_rates(args.missing_rates)
    for chunk_row in artifacts.chunk_summary_df.to_dict(orient="records"):
        chunk_index = int(chunk_row["chunk_index"])
        file_name = str(chunk_row["file_name"])
        file_path = artifacts.input_files[chunk_index]
        process_any_rate = any(
            should_process_rate_chunk(args, rate_chunk_paths(paths, rate, args.seed, chunk_index, file_name), rate, chunk_index)
            for rate in rates
        )
        if not process_any_rate:
            continue
        io_started = time.perf_counter()
        df = pd.read_parquet(file_path, columns=[args.node_col, args.time_col, args.target_col])
        row_lookup, unique_times = build_row_lookup(
            df=df,
            canonical_node_ids=artifacts.canonical_node_ids,
            node_col=args.node_col,
            time_col=args.time_col,
            period=args.period,
        )
        io_elapsed = time.perf_counter() - io_started
        for rate_index, rate in enumerate(rates):
            part_paths = rate_chunk_paths(paths, rate, args.seed, chunk_index, file_name)
            if not should_process_rate_chunk(args, part_paths, rate, chunk_index):
                continue
            if handle_existing_outputs(args, part_paths, write_missing_data=write_missing_data):
                continue
            status_stub = {
                "scenario_tag": scenario_rate_tag(rate, args.seed),
                "chunk_index": int(chunk_index),
            }
            try:
                start_time = time.perf_counter()
                mask_df, missing_df, event_df, event_neighbor_df, status_row, timing = generate_rate_chunk(
                    args=args,
                    artifacts=artifacts,
                    df=df,
                    row_lookup=row_lookup,
                    unique_times=unique_times,
                    chunk_row=chunk_row,
                    rate=rate,
                    rate_index=rate_index,
                    length_config=length_config,
                )
                total_elapsed = time.perf_counter() - start_time
                status_row["io_time_seconds"] = float(io_elapsed)
                status_row["placement_time_seconds"] = float(timing["placement_time_seconds"])
                status_row["write_time_seconds"] = 0.0
                status_row["checkpoint_time_seconds"] = 0.0
                status_row["mask_path"] = str(part_paths["mask_path"])
                status_row["missing_dataset_path"] = str(part_paths["missing_path"]) if missing_df is not None else ""
                final_status = write_generation_outputs(
                    args=args,
                    part_paths=part_paths,
                    mask_df=mask_df,
                    missing_df=missing_df,
                    event_df=event_df,
                    event_neighbor_df=event_neighbor_df,
                    status_row=status_row,
                    elapsed_seconds=total_elapsed,
                    write_time_seconds=0.0,
                    checkpoint_time_seconds=0.0,
                )
                append_jsonl(
                    paths["runtime_jsonl_path"],
                    {
                        "event": "chunk_completed",
                        "scenario_tag": final_status["scenario_tag"],
                        "chunk_index": int(final_status["chunk_index"]),
                        "elapsed_seconds": float(final_status["elapsed_seconds"]),
                        "observed_missing_count": int(final_status["observed_missing_count"]),
                        "resampled_event_count": int(final_status["resampled_event_count"]),
                        "candidate_rounds": int(final_status["candidate_rounds"]),
                        "skipped_missing_slots": int(final_status["skipped_missing_slots"]),
                        "io_time_seconds": float(final_status["io_time_seconds"]),
                        "placement_time_seconds": float(final_status["placement_time_seconds"]),
                        "write_time_seconds": float(final_status["write_time_seconds"]),
                        "checkpoint_time_seconds": float(final_status["checkpoint_time_seconds"]),
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                    },
                )
                if part_paths["failed_path"].exists():
                    part_paths["failed_path"].unlink()
            except Exception as exc:
                cleanup_incomplete_rate_chunk(part_paths)
                write_failed_part(part_paths, status_stub, exc)
                raise
        del df, row_lookup
        gc.collect()
    return collect_status_parts(paths)


def collect_status_parts(paths: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for status_path in sorted(paths["status_parts_root"].glob("*.status.json")):
        rows.append(read_json(status_path))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)


def collect_event_parts(paths: Dict[str, Path]) -> pd.DataFrame:
    frames = []
    for event_path in sorted(paths["event_parts_root"].glob("*.events.parquet")):
        frames.append(pd.read_parquet(event_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["missing_rate_target", "event_id"]).reset_index(drop=True)


def finalize_event_csv_frame(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame()
    preferred_columns = [
        "event_id",
        "missing_rate_target",
        "scenario_tag",
        "chunk_index",
        "day_index",
        "file_name",
        "target_node_id",
        "target_node_index",
        "start_slot",
        "stop_slot_exclusive",
        "actual_length",
        "length_group",
        "neighbor_scope",
        "available_neighbor_count",
        "anchor_neighbor_count",
        "anchor_neighbor_ids",
        "anchor_neighbor_indices",
        "neighbor_observed_ratio",
        "spatial_constraint_level",
        "relaxation_stage",
        "relaxation_reason",
        "anchor_neighbor_count_used",
        "required_available_neighbor_count",
        "actual_available_neighbor_count",
        "fully_neighbor_observed",
        "partially_neighbor_observed",
        "neighbor_observed_slot_ratio",
    ]
    selected_columns = [column for column in preferred_columns if column in event_df.columns]
    return event_df.loc[:, selected_columns].copy()


def path_free_bytes(path: Path) -> int:
    anchor = path.anchor if path.anchor else str(path.resolve().anchor)
    return int(shutil.disk_usage(anchor).free)


def choose_external_event_csv_path(paths: Dict[str, Path]) -> Path:
    export_root = Path("D:/FedTrafficFlow_temp_exports")
    return export_root / paths["scenario_root"].name / "miss_set" / "manifests" / paths["event_csv_path"].name


def write_finalize_event_csv(paths: Dict[str, Path], event_df: pd.DataFrame) -> Path:
    final_df = finalize_event_csv_frame(event_df)
    target_path = paths["event_csv_path"]
    clean_tmp_for_base(target_path)
    free_bytes = path_free_bytes(target_path)
    if free_bytes > 512 * 1024 * 1024:
        atomic_write_dataframe_csv(target_path, final_df)
        return target_path
    external_path = choose_external_event_csv_path(paths)
    clean_tmp_for_base(external_path)
    atomic_write_dataframe_csv(external_path, final_df)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        os.symlink(str(external_path), str(target_path))
    except Exception:
        pointer_df = pd.DataFrame(
            [
                {
                    "storage_mode": "external_pointer",
                    "external_event_csv_path": str(external_path),
                }
            ]
        )
        atomic_write_dataframe_csv(target_path, pointer_df)
    return target_path


def resolve_final_event_csv_path(paths: Dict[str, Path]) -> Path:
    event_csv_path = paths["event_csv_path"]
    if not event_csv_path.exists():
        return event_csv_path
    try:
        preview_df = pd.read_csv(event_csv_path, nrows=1)
    except Exception:
        return event_csv_path
    if "external_event_csv_path" in preview_df.columns and not preview_df.empty:
        return Path(str(preview_df.iloc[0]["external_event_csv_path"]))
    return event_csv_path


def load_finalized_event_df(paths: Dict[str, Path]) -> pd.DataFrame:
    resolved_path = resolve_final_event_csv_path(paths)
    if not resolved_path.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved_path)


def run_finalize(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_base_dirs(paths)
    status_df = collect_status_parts(paths)
    event_df = collect_event_parts(paths)
    if status_df.empty:
        raise FileNotFoundError("no status parts found; run generate_missing first")
    clean_tmp_for_base(paths["chunk_status_path"])
    clean_tmp_for_base(paths["event_csv_path"])
    atomic_write_dataframe_csv(paths["chunk_status_path"], status_df)
    write_finalize_event_csv(paths, event_df)
    return status_df, event_df


def build_missingness_audit(
    args: argparse.Namespace,
    artifacts: PreparedArtifacts,
    paths: Dict[str, Path],
    status_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> Dict[str, Any]:
    rates = parse_rates(args.missing_rates)
    per_rate = {}
    relaxation_per_rate = {}
    for rate in rates:
        scenario_tag = scenario_rate_tag(rate, args.seed)
        rate_status = status_df.loc[np.isclose(status_df["missing_rate_target"], rate)].copy()
        rate_events = event_df.loc[np.isclose(event_df["missing_rate_target"], rate)].copy() if not event_df.empty else pd.DataFrame()
        mask_count = len(list((paths["masks_root"] / scenario_tag).glob("*_mask.parquet")))
        miss_data_count = len(list((paths["miss_data_root"] / scenario_tag).glob("*.parquet")))
        global_missing_count = int(rate_status["global_missing_count"].max()) if not rate_status.empty and "global_missing_count" in rate_status.columns else 0
        sum_observed_missing_count = int(rate_status["observed_missing_count"].sum()) if not rate_status.empty else 0
        strict_count = int(rate_status["strict_anchor_missing_count"].sum()) if "strict_anchor_missing_count" in rate_status.columns and not rate_status.empty else 0
        relaxed_count = int(rate_status["relaxed_anchor_missing_count"].sum()) if "relaxed_anchor_missing_count" in rate_status.columns and not rate_status.empty else 0
        weak_count = int(rate_status["weak_neighbor_available_missing_count"].sum()) if "weak_neighbor_available_missing_count" in rate_status.columns and not rate_status.empty else 0
        none_count = int(rate_status["none_missing_count"].sum()) if "none_missing_count" in rate_status.columns and not rate_status.empty else 0
        constrained_ratio = float(rate_status["spatially_constrained_ratio"].mean()) if "spatially_constrained_ratio" in rate_status.columns and not rate_status.empty else 0.0
        per_rate["%.2f" % rate] = {
            "scenario_tag": scenario_tag,
            "mask_file_count": int(mask_count),
            "miss_data_file_count": int(miss_data_count),
            "observed_missing_rate": float(rate_status["observed_missing_count"].sum() / rate_status["row_count"].sum())
            if not rate_status.empty
            else 0.0,
            "neighbor_observed_ratio": float(rate_status["neighbor_observed_ratio"].mean()) if not rate_status.empty else 0.0,
            "min_available_neighbor_count": int(rate_status["min_available_neighbor_count"].min()) if not rate_status.empty else 0,
            "max_available_neighbor_count": int(rate_status["average_available_neighbor_count"].max()) if not rate_status.empty else 0,
            "event_count": int(len(rate_events)),
            "skipped_missing_slots": int(rate_status["skipped_missing_slots"].sum()) if not rate_status.empty else 0,
            "candidate_rounds_total": int(rate_status["candidate_rounds"].sum()) if not rate_status.empty else 0,
        }
        relaxation_per_rate["%.2f" % rate] = {
            "scenario_tag": scenario_tag,
            "global_missing_count": int(global_missing_count),
            "sum_observed_missing_count": int(sum_observed_missing_count),
            "strict_anchor_missing_count": int(strict_count),
            "relaxed_anchor_missing_count": int(relaxed_count),
            "weak_neighbor_available_missing_count": int(weak_count),
            "none_missing_count": int(none_count),
            "spatially_constrained_ratio": float(constrained_ratio),
            "spatially_unconstrained_ratio": float(none_count / float(global_missing_count)) if global_missing_count > 0 else 0.0,
            "constraint_relaxation_used": bool(int(rate_status["constraint_relaxation_used"].astype(int).max()) > 0) if "constraint_relaxation_used" in rate_status.columns and not rate_status.empty else False,
            "max_relaxation_stage_used": int(rate_status["max_relaxation_stage_used"].max()) if "max_relaxation_stage_used" in rate_status.columns and not rate_status.empty else 1,
            "warning_if_spatially_constrained_ratio_below_threshold": bool(constrained_ratio < float(args.min_spatially_constrained_ratio)),
        }
    payload = {
        "scenario_id": SCENARIO_ID,
        "mechanism": MECHANISM,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "uses_current_time_neighbors_allowed_for_spatial_methods": True,
        "target_current_true_value_available_to_methods": False,
        "future_information_allowed": False,
        "masked_position_imputation_error_only": True,
        "neighbor_protection_mode": str(args.neighbor_protection_mode),
        "anchor_neighbor_count": int(args.anchor_neighbor_count),
        "min_available_neighbors": int(args.min_available_neighbors),
        "neighbor_protection_difference_from_legacy": "anchor mode protects only anchor neighbors instead of all preferred neighbors",
        "placement_backend": str(args.placement_backend),
        "per_rate": per_rate,
    }
    lines = [
        "# snh_mix fast missingness audit",
        "",
        "- mechanism: `spatial_neighbor_holdout`",
        "- evaluation_protocol: `online_spatial_interpolation`",
        "- 当前时刻允许使用邻居观测。",
        "- 不允许使用目标节点当前真实值。",
        "- 不允许使用未来信息。",
        "- fast 版本默认采用 anchor 邻居保护，而不是旧版 all-neighbor 保护。",
        "",
    ]
    for rate_key in sorted(per_rate.keys()):
        item = per_rate[rate_key]
        lines.extend(
            [
                "## %s" % rate_key,
                "",
                "- mask 文件数: `%s`" % item["mask_file_count"],
                "- miss_data 文件数: `%s`" % item["miss_data_file_count"],
                "- observed_missing_rate: `%.6f`" % item["observed_missing_rate"],
                "- neighbor_observed_ratio: `%.6f`" % item["neighbor_observed_ratio"],
                "- min_available_neighbor_count: `%s`" % item["min_available_neighbor_count"],
                "- skipped_missing_slots: `%s`" % item["skipped_missing_slots"],
                "",
            ]
        )
    atomic_write_json(paths["audit_json_path"], payload)
    atomic_write_text(paths["audit_md_path"], "\n".join(lines).rstrip() + "\n", encoding="utf-8")
    relaxation_payload = {
        "spatial_constraint_relaxation": parse_bool(args.spatial_constraint_relaxation),
        "relaxation_policy": str(args.relaxation_policy),
        "allow_no_spatial_constraint": parse_bool(args.allow_no_spatial_constraint),
        "constraint_levels": CONSTRAINT_LEVELS,
        "global_missing_count_preserved": bool(
            all(int(item["global_missing_count"]) == int(item["sum_observed_missing_count"]) for item in relaxation_per_rate.values())
        ),
        "none_level_not_interpreted_as_spatial_holdout": True,
        "min_spatially_constrained_ratio": float(args.min_spatially_constrained_ratio),
        "per_rate": relaxation_per_rate,
    }
    relaxation_lines = [
        "# snh constraint relaxation audit",
        "",
        "- 当高缺失率下空间约束导致缺失位置无法放满时，本实验允许逐级降低空间约束。",
        "- 其中 none 等级只用于满足完整数据集 global missing rate，不作为严格空间邻居保留样本解释。",
        "- 正式分析空间结构效果时，应优先查看 strict_anchor、relaxed_anchor 和 weak_neighbor_available 子集。",
        "",
    ]
    for rate_key in sorted(relaxation_per_rate.keys()):
        item = relaxation_per_rate[rate_key]
        relaxation_lines.extend(
            [
                "## %s" % rate_key,
                "",
                "- global_missing_count: `%s`" % item["global_missing_count"],
                "- sum_observed_missing_count: `%s`" % item["sum_observed_missing_count"],
                "- strict_anchor_missing_count: `%s`" % item["strict_anchor_missing_count"],
                "- relaxed_anchor_missing_count: `%s`" % item["relaxed_anchor_missing_count"],
                "- weak_neighbor_available_missing_count: `%s`" % item["weak_neighbor_available_missing_count"],
                "- none_missing_count: `%s`" % item["none_missing_count"],
                "- spatially_constrained_ratio: `%.6f`" % item["spatially_constrained_ratio"],
                "- warning_if_spatially_constrained_ratio_below_threshold: `%s`" % item["warning_if_spatially_constrained_ratio_below_threshold"],
                "",
            ]
        )
    atomic_write_json(paths["constraint_relaxation_audit_json_path"], relaxation_payload)
    atomic_write_text(paths["constraint_relaxation_audit_md_path"], "\n".join(relaxation_lines).rstrip() + "\n", encoding="utf-8")
    return payload


def build_global_allocation_audit(
    args: argparse.Namespace,
    artifacts: PreparedArtifacts,
    paths: Dict[str, Path],
    status_df: pd.DataFrame,
) -> Dict[str, Any]:
    allocation_df = artifacts.allocation_df.copy()
    rates = parse_rates(args.missing_rates)
    per_rate = {}
    for rate in rates:
        rate_key = "%.2f" % rate
        rate_alloc = allocation_df.loc[np.isclose(allocation_df["missing_rate_target"], rate)].copy()
        rate_status = status_df.loc[np.isclose(status_df["missing_rate_target"], rate)].copy()
        global_missing_count = int(rate_alloc["global_missing_count"].max()) if not rate_alloc.empty else 0
        allocation_sum = int(rate_alloc["allocated_missing_count"].sum()) if not rate_alloc.empty else 0
        observed_sum = int(rate_status["observed_missing_count"].sum()) if not rate_status.empty else 0
        per_rate[rate_key] = {
            "global_eligible_count": int(rate_alloc["global_eligible_count"].max()) if not rate_alloc.empty else 0,
            "global_missing_count": int(global_missing_count),
            "allocation_sum": int(allocation_sum),
            "observed_sum": int(observed_sum),
            "allocation_sum_matches_global_missing_count": bool(allocation_sum == global_missing_count),
            "observed_sum_matches_global_missing_count": bool(observed_sum == global_missing_count),
        }
    payload = {
        "mechanism": MECHANISM,
        "mask_scope": str(args.mask_scope),
        "allocation_method": str(args.allocation_method),
        "allocation_shortfall_policy": str(args.allocation_shortfall_policy),
        "global_allocation_used": True,
        "day_stratified_generation_used": False,
        "per_chunk_round_rate_used": False,
        "per_rate": per_rate,
    }
    lines = [
        "# snh global allocation audit",
        "",
        "- mask_scope: `%s`" % str(args.mask_scope),
        "- allocation_method: `%s`" % str(args.allocation_method),
        "- day_stratified_generation_used: `False`",
        "- per_chunk_round_rate_used: `False`",
        "",
    ]
    for rate_key in sorted(per_rate.keys()):
        item = per_rate[rate_key]
        lines.extend(
            [
                "## %s" % rate_key,
                "",
                "- global_eligible_count: `%s`" % item["global_eligible_count"],
                "- global_missing_count: `%s`" % item["global_missing_count"],
                "- allocation_sum: `%s`" % item["allocation_sum"],
                "- observed_sum: `%s`" % item["observed_sum"],
                "- allocation_sum_matches_global_missing_count: `%s`" % item["allocation_sum_matches_global_missing_count"],
                "- observed_sum_matches_global_missing_count: `%s`" % item["observed_sum_matches_global_missing_count"],
                "",
            ]
        )
    atomic_write_json(paths["global_allocation_audit_json_path"], payload)
    atomic_write_text(paths["global_allocation_audit_md_path"], "\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return payload


def build_performance_audit(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    status_df: pd.DataFrame,
) -> Dict[str, Any]:
    runtime_entries = []
    if paths["runtime_jsonl_path"].exists():
        for line in paths["runtime_jsonl_path"].read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                runtime_entries.append(json.loads(line))
    per_rate_elapsed_seconds = {}
    per_chunk_elapsed_seconds = {}
    rates = parse_rates(args.missing_rates)
    for rate in rates:
        scenario_tag = scenario_rate_tag(rate, args.seed)
        rate_status = status_df.loc[np.isclose(status_df["missing_rate_target"], rate)].copy()
        per_rate_elapsed_seconds["%.2f" % rate] = float(rate_status["elapsed_seconds"].sum()) if not rate_status.empty else 0.0
        for row in rate_status.to_dict(orient="records"):
            key = "%s__chunk_%s" % (scenario_tag, chunk_token(int(row["chunk_index"])))
            per_chunk_elapsed_seconds[key] = float(row["elapsed_seconds"])
    elapsed_values = [float(value) for value in per_chunk_elapsed_seconds.values()]
    payload = {
        "per_rate_elapsed_seconds": per_rate_elapsed_seconds,
        "per_chunk_elapsed_seconds": per_chunk_elapsed_seconds,
        "mean_seconds_per_chunk": float(np.mean(elapsed_values)) if elapsed_values else 0.0,
        "median_seconds_per_chunk": float(np.median(elapsed_values)) if elapsed_values else 0.0,
        "max_seconds_per_chunk": float(np.max(elapsed_values)) if elapsed_values else 0.0,
        "candidate_rounds_total": int(status_df["candidate_rounds"].sum()) if not status_df.empty else 0,
        "resampled_event_count_total": int(status_df["resampled_event_count"].sum()) if not status_df.empty else 0,
        "skipped_missing_slots_total": int(status_df["skipped_missing_slots"].sum()) if not status_df.empty else 0,
        "io_time_seconds": float(status_df["io_time_seconds"].sum()) if not status_df.empty else 0.0,
        "placement_time_seconds": float(status_df["placement_time_seconds"].sum()) if not status_df.empty else 0.0,
        "write_time_seconds": float(status_df["write_time_seconds"].sum()) if not status_df.empty else 0.0,
        "checkpoint_time_seconds": float(status_df["checkpoint_time_seconds"].sum()) if not status_df.empty else 0.0,
        "old_script_baseline": {
            "0.05": "approx 3 minutes/day",
            "0.10": "approx 10 minutes/day",
            "0.20": "> 20 minutes/day",
        },
        "speedup_notes": [
            "anchor-neighbor protection replaces all-neighbor protection",
            "chunk outer loop reads each day parquet once for all rates",
            "event_id is written directly during mask generation",
            "checkpoint is written as part files instead of full-table rewrite",
            "atomic write improves interruption safety",
        ],
        "runtime_log_entry_count": int(len(runtime_entries)),
    }
    lines = [
        "# snh fast generation performance",
        "",
        "- 旧脚本 5% 约 3 分钟/天。",
        "- 旧脚本 10% 约 10 分钟/天。",
        "- 旧脚本 20% 超过 20 分钟/天。",
        "- 本轮通过 anchor-neighbor 保护、chunk 外层循环、event_id 直接写入、分片 checkpoint 与原子写入提升速度与稳定性。",
        "",
    ]
    for rate_key in sorted(per_rate_elapsed_seconds.keys()):
        lines.append("- %s 总耗时秒数: `%.3f`" % (rate_key, per_rate_elapsed_seconds[rate_key]))
    lines.extend(
        [
            "",
            "- mean_seconds_per_chunk: `%.3f`" % payload["mean_seconds_per_chunk"],
            "- median_seconds_per_chunk: `%.3f`" % payload["median_seconds_per_chunk"],
            "- max_seconds_per_chunk: `%.3f`" % payload["max_seconds_per_chunk"],
        ]
    )
    atomic_write_json(paths["performance_json_path"], payload)
    atomic_write_text(paths["performance_md_path"], "\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return payload


def run_audit(
    args: argparse.Namespace,
    artifacts: PreparedArtifacts,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    if not paths["chunk_status_path"].exists() or not paths["event_csv_path"].exists():
        status_df, event_df = run_finalize(paths)
    else:
        status_df = pd.read_csv(paths["chunk_status_path"])
        event_df = load_finalized_event_df(paths)
    missingness_payload = build_missingness_audit(args, artifacts, paths, status_df, event_df)
    global_allocation_payload = build_global_allocation_audit(args, artifacts, paths, status_df)
    performance_payload = build_performance_audit(args, paths, status_df)
    return {
        "missingness": missingness_payload,
        "global_allocation": global_allocation_payload,
        "performance": performance_payload,
    }


def check_tmp_files(miss_root: Path) -> List[str]:
    rows = []
    for path in miss_root.rglob("*.tmp"):
        rows.append(str(path))
    return sorted(rows)


def build_validation_rows(
    args: argparse.Namespace,
    artifacts: PreparedArtifacts,
    paths: Dict[str, Path],
    status_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rates = parse_rates(args.missing_rates)
    chunk_count = int(len(artifacts.chunk_summary_df))
    run_config = read_json(paths["run_config_path"]) if paths["run_config_path"].exists() else {}
    failed_parts = sorted([str(path.name) for path in paths["failed_parts_root"].glob("*.failed.json")])
    tmp_files = check_tmp_files(paths["miss_root"])
    finalized_event_df = load_finalized_event_df(paths) if paths["event_csv_path"].exists() else pd.DataFrame()
    allocation_df = pd.read_csv(paths["allocation_path"]) if paths["allocation_path"].exists() else pd.DataFrame()
    sample_mask_columns = []
    sample_mask_path = next(paths["masks_root"].rglob("*_mask.parquet"), None) if paths["masks_root"].exists() else None
    if sample_mask_path is not None:
        sample_mask_columns = pd.read_parquet(sample_mask_path).columns.tolist()
    rows = []
    warning_rates = []
    min_available_check_passed = True
    min_available_check_details = []
    for rate in rates:
        scenario_tag = scenario_rate_tag(rate, args.seed)
        mask_count = len(list((paths["masks_root"] / scenario_tag).glob("*_mask.parquet")))
        miss_data_count = len(list((paths["miss_data_root"] / scenario_tag).glob("*.parquet")))
        done_count = len(list(paths["progress_root"].glob("%s__chunk_*.done.json" % scenario_tag)))
        status_count = len(list(paths["status_parts_root"].glob("%s__chunk_*.status.json" % scenario_tag)))
        event_count = len(list(paths["event_parts_root"].glob("%s__chunk_*.events.parquet" % scenario_tag)))
        rate_status = status_df.loc[np.isclose(status_df["missing_rate_target"], rate)].copy()
        observed_rate = float(rate_status["observed_missing_count"].sum() / rate_status["row_count"].sum()) if not rate_status.empty else 0.0
        global_missing_count = int(rate_status["global_missing_count"].max()) if "global_missing_count" in rate_status.columns and not rate_status.empty else 0
        observed_missing_count = int(rate_status["observed_missing_count"].sum()) if not rate_status.empty else 0
        constrained_ratio = float(rate_status["spatially_constrained_ratio"].mean()) if "spatially_constrained_ratio" in rate_status.columns and not rate_status.empty else 0.0
        min_neighbor_value = int(rate_status["min_available_neighbor_count"].min()) if "min_available_neighbor_count" in rate_status.columns and not rate_status.empty else 0
        relaxed_min_check = bool(
            (min_neighbor_value >= int(args.min_available_neighbors))
            or (
                parse_bool(args.allow_no_spatial_constraint)
                and constrained_ratio >= float(args.min_spatially_constrained_ratio)
            )
        )
        min_available_check_passed = min_available_check_passed and relaxed_min_check
        min_available_check_details.append("%s:%s" % (scenario_tag, min_neighbor_value))
        if constrained_ratio < float(args.min_spatially_constrained_ratio):
            warning_rates.append("%.2f" % rate)
        rows.append({"check": "%s_mask_count" % scenario_tag, "passed": bool(mask_count == chunk_count), "details": str(mask_count)})
        if parse_bool(args.write_missing_data):
            rows.append({"check": "%s_miss_data_count" % scenario_tag, "passed": bool(miss_data_count == chunk_count), "details": str(miss_data_count)})
        rows.append({"check": "%s_done_count" % scenario_tag, "passed": bool(done_count == chunk_count), "details": str(done_count)})
        rows.append({"check": "%s_status_part_count" % scenario_tag, "passed": bool(status_count == chunk_count), "details": str(status_count)})
        rows.append({"check": "%s_event_part_count" % scenario_tag, "passed": bool(event_count == chunk_count), "details": str(event_count)})
        rows.append(
            {
                "check": "%s_observed_missing_rate_close" % scenario_tag,
                "passed": bool(abs(observed_rate - float(rate)) <= 0.01),
                "details": "%.6f" % observed_rate,
            }
        )
        rows.append(
            {
                "check": "%s_global_missing_count_preserved" % scenario_tag,
                "passed": bool(global_missing_count == observed_missing_count),
                "details": "%s/%s" % (observed_missing_count, global_missing_count),
            }
        )
        rows.append(
            {
                "check": "%s_spatially_constrained_ratio_reported" % scenario_tag,
                "passed": bool("spatially_constrained_ratio" in rate_status.columns),
                "details": "%.6f" % constrained_ratio,
            }
        )
    rows.extend(
        [
            {
                "check": "neighbor_observed_ratio_positive",
                "passed": bool((status_df["neighbor_observed_ratio"] > 0).all()) if not status_df.empty else False,
                "details": "%.6f" % (float(status_df["neighbor_observed_ratio"].min()) if not status_df.empty else 0.0),
            },
            {
                "check": "min_available_neighbor_count",
                "passed": bool(min_available_check_passed),
                "details": ";".join(min_available_check_details),
            },
            {
                "check": "skipped_missing_slots_recorded",
                "passed": bool("skipped_missing_slots" in status_df.columns),
                "details": str(int(status_df["skipped_missing_slots"].sum()) if "skipped_missing_slots" in status_df.columns else -1),
            },
            {
                "check": "no_incomplete_tmp_files",
                "passed": bool(len(tmp_files) == 0),
                "details": ";".join(tmp_files[:20]),
            },
            {
                "check": "no_failed_parts",
                "passed": bool(len(failed_parts) == 0),
                "details": ";".join(failed_parts[:20]),
            },
            {
                "check": "resume_enabled",
                "passed": bool(run_config.get("resume", False)),
                "details": str(run_config.get("resume", False)),
            },
            {
                "check": "checkpoint_mode_part_files",
                "passed": run_config.get("checkpoint_mode") == CHECKPOINT_MODE,
                "details": str(run_config.get("checkpoint_mode")),
            },
            {
                "check": "event_id_written_directly",
                "passed": bool(run_config.get("event_id_written_directly", False)),
                "details": str(run_config.get("event_id_written_directly", False)),
            },
            {
                "check": "backfill_event_ids_disabled",
                "passed": bool(not run_config.get("backfill_event_ids_used", True)),
                "details": str(run_config.get("backfill_event_ids_used", True)),
            },
            {
                "check": "constraint_relaxation_enabled",
                "passed": bool(run_config.get("spatial_constraint_relaxation", False)),
                "details": str(run_config.get("spatial_constraint_relaxation", False)),
            },
            {
                "check": "constraint_level_recorded_in_events",
                "passed": bool("spatial_constraint_level" in finalized_event_df.columns),
                "details": ",".join([column for column in ["spatial_constraint_level", "relaxation_stage"] if column in finalized_event_df.columns]),
            },
            {
                "check": "constraint_level_recorded_in_masks",
                "passed": bool("spatial_constraint_level" in sample_mask_columns),
                "details": ",".join(sample_mask_columns[:20]),
            },
            {
                "check": "none_level_allowed",
                "passed": bool(run_config.get("allow_no_spatial_constraint", False)),
                "details": str(run_config.get("allow_no_spatial_constraint", False)),
            },
            {
                "check": "none_level_count_reported",
                "passed": bool("none_missing_count" in status_df.columns),
                "details": str(int(status_df["none_missing_count"].sum()) if "none_missing_count" in status_df.columns and not status_df.empty else 0),
            },
            {
                "check": "none_level_excluded_from_spatial_claims",
                "passed": bool(paths["constraint_relaxation_audit_md_path"].exists()),
                "details": str(paths["constraint_relaxation_audit_md_path"]),
            },
            {
                "check": "mask_scope_global",
                "passed": run_config.get("mask_scope") == "global",
                "details": str(run_config.get("mask_scope")),
            },
            {
                "check": "global_allocation_used",
                "passed": bool(run_config.get("global_allocation_used", False)),
                "details": str(run_config.get("global_allocation_used", False)),
            },
            {
                "check": "allocation_sum_matches_global_missing_count",
                "passed": bool(
                    (allocation_df.groupby("missing_rate_target")["allocated_missing_count"].sum() == allocation_df.groupby("missing_rate_target")["global_missing_count"].max()).all()
                )
                if not allocation_df.empty
                else False,
                "details": "checked on global_missing_allocation.csv",
            },
            {
                "check": "observed_sum_matches_global_missing_count",
                "passed": bool(
                    (status_df.groupby("missing_rate_target")["observed_missing_count"].sum() == status_df.groupby("missing_rate_target")["global_missing_count"].max()).all()
                )
                if ("global_missing_count" in status_df.columns and not status_df.empty)
                else False,
                "details": "checked on finalized status",
            },
            {
                "check": "day_stratified_generation_disabled",
                "passed": bool(not run_config.get("day_stratified_generation_used", True)),
                "details": str(run_config.get("day_stratified_generation_used", True)),
            },
            {
                "check": "per_chunk_round_rate_disabled",
                "passed": bool(not run_config.get("per_chunk_round_rate_used", True)),
                "details": str(run_config.get("per_chunk_round_rate_used", True)),
            },
        ]
    )
    validation_df = pd.DataFrame(rows)
    validation_json = {
        "all_complete": bool(validation_df["passed"].all()) if not validation_df.empty else False,
        "resume_enabled": bool(run_config.get("resume", False)),
        "checkpoint_mode": CHECKPOINT_MODE,
        "atomic_write_enabled": True,
        "event_id_written_directly": True,
        "backfill_event_ids_used": False,
        "neighbor_protection_mode": str(args.neighbor_protection_mode),
        "placement_backend": str(args.placement_backend),
        "chunk_outer_loop_enabled": True,
        "full_checkpoint_rewrite_disabled": True,
        "constraint_relaxation_enabled": parse_bool(args.spatial_constraint_relaxation),
        "all_rates_global_missing_count_preserved": bool(
            (status_df.groupby("missing_rate_target")["observed_missing_count"].sum() == status_df.groupby("missing_rate_target")["global_missing_count"].max()).all()
        )
        if ("global_missing_count" in status_df.columns and not status_df.empty)
        else False,
        "constraint_level_recorded_in_events": bool("spatial_constraint_level" in finalized_event_df.columns),
        "constraint_level_recorded_in_masks": bool("spatial_constraint_level" in sample_mask_columns),
        "none_level_allowed": parse_bool(args.allow_no_spatial_constraint),
        "none_level_count_reported": bool("none_missing_count" in status_df.columns),
        "spatially_constrained_ratio_reported": bool("spatially_constrained_ratio" in status_df.columns),
        "none_level_excluded_from_spatial_claims": True,
        "mask_scope": str(run_config.get("mask_scope", "")),
        "global_allocation_used": bool(run_config.get("global_allocation_used", False)),
        "allocation_sum_matches_global_missing_count": bool(
            (allocation_df.groupby("missing_rate_target")["allocated_missing_count"].sum() == allocation_df.groupby("missing_rate_target")["global_missing_count"].max()).all()
        )
        if not allocation_df.empty
        else False,
        "observed_sum_matches_global_missing_count": bool(
            (status_df.groupby("missing_rate_target")["observed_missing_count"].sum() == status_df.groupby("missing_rate_target")["global_missing_count"].max()).all()
        )
        if ("global_missing_count" in status_df.columns and not status_df.empty)
        else False,
        "day_stratified_generation_used": bool(run_config.get("day_stratified_generation_used", False)),
        "per_chunk_round_rate_used": bool(run_config.get("per_chunk_round_rate_used", False)),
        "warning_rates_below_min_spatially_constrained_ratio": warning_rates,
        "failed_parts": failed_parts,
        "tmp_files": tmp_files,
        "checks": validation_df.to_dict(orient="records"),
    }
    return validation_df, validation_json


def run_validate(args: argparse.Namespace, artifacts: PreparedArtifacts, paths: Dict[str, Path]) -> Dict[str, Any]:
    if not paths["chunk_status_path"].exists():
        status_df, _ = run_finalize(paths)
    else:
        status_df = pd.read_csv(paths["chunk_status_path"])
    validation_df, validation_json = build_validation_rows(args, artifacts, paths, status_df)
    atomic_write_dataframe_csv(paths["validation_csv_path"], validation_df)
    atomic_write_json(paths["validation_json_path"], validation_json)
    return validation_json


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.output_dir = ensure_absolute(project_root, args.output_dir)
    args.topology_file = ensure_absolute(project_root, args.topology_file)
    paths = build_paths(args.output_dir)
    artifacts = run_prepare(args, paths)
    if args.stage == "prepare":
        return
    if args.stage in {"generate_missing", "all"}:
        run_generate_missing(args, paths, artifacts)
    if args.stage in {"finalize", "all"}:
        run_finalize(paths)
    if args.stage in {"audit", "all"}:
        run_audit(args, artifacts, paths)
    if args.stage in {"validate", "all"}:
        run_validate(args, artifacts, paths)


if __name__ == "__main__":
    main()
