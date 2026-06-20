"""生成空间邻居保留缺失场景的掩码、缺失数据与审计结果。

核心功能：
- 在空间邻居约束下构造节点 holdout 缺失事件；
- 输出掩码文件、缺失分块、事件清单和审计结果；
- 为后续空间缺失补全实验提供可复现输入。

项目作用：
- 作为空间缺失实验的 setting 阶段脚本；
- 统一空间缺失事件生成、状态记录和结果落盘格式。

关键依赖：`numpy`、`pandas`、`pathlib`。
主要输入：完整分块、拓扑文件、缺失率和长度配置。
主要输出：掩码文件、缺失数据、事件日志和审计报告。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LENGTH_GROUP_ORDER = ["short", "mid", "long"]
EPSILON = 1e-6
SCENARIO_ID = "snh_mix"
MECHANISM = "spatial_neighbor_holdout"


@dataclass(frozen=True)
class LengthConfig:
    short_range: tuple[int, int]
    mid_range: tuple[int, int]
    long_range: tuple[int, int]
    probs: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spatial neighbor holdout missingness setting only.")
    parser.add_argument("--stage", required=True, choices=["prepare", "generate_missing", "audit", "all"])
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
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--node_col", required=True, type=str)
    parser.add_argument("--time_col", required=True, type=str)
    parser.add_argument("--period", required=True, type=int)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_int_pair(raw: str) -> tuple[int, int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) != 2:
        raise ValueError(f"expected two integers, got: {raw}")
    return int(values[0]), int(values[1])


def parse_rates(raw: str) -> list[float]:
    values = parse_float_list(raw)
    if not values:
        raise ValueError("missing_rates is empty")
    for value in values:
        if not (0.0 < value < 1.0):
            raise ValueError(f"invalid missing rate: {value}")
    return values


def parse_length_config(args: argparse.Namespace) -> LengthConfig:
    probs = parse_float_list(args.length_group_probs)
    if len(probs) != 3:
        raise ValueError("length_group_probs must contain 3 values")
    if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"length_group_probs must sum to 1.0, got {sum(probs)}")
    return LengthConfig(
        short_range=parse_int_pair(args.short_length_range),
        mid_range=parse_int_pair(args.mid_length_range),
        long_range=parse_int_pair(args.long_length_range),
        probs=(float(probs[0]), float(probs[1]), float(probs[2])),
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def extract_day_index(file_name: str) -> int:
    stem = Path(file_name).stem
    return int(stem.rsplit("_", 1)[-1])


def format_rate_token(rate: float) -> str:
    return f"{int(round(rate * 100)):02d}"


def scenario_rate_tag(rate: float, seed: int) -> str:
    return f"snh_r{format_rate_token(rate)}_mix_s{seed}"


def classify_length_group(actual_length: int, config: LengthConfig) -> str:
    if config.short_range[0] <= actual_length <= config.short_range[1]:
        return "short"
    if config.mid_range[0] <= actual_length <= config.mid_range[1]:
        return "mid"
    if config.long_range[0] <= actual_length <= config.long_range[1]:
        return "long"
    raise ValueError(f"length {actual_length} is outside configured ranges")


def sample_length(rng: np.random.Generator, config: LengthConfig, max_allowed: int) -> tuple[int, str]:
    group_index = int(rng.choice(np.arange(3), p=np.asarray(config.probs, dtype=np.float64)))
    group_name = LENGTH_GROUP_ORDER[group_index]
    if group_name == "short":
        low, high = config.short_range
    elif group_name == "mid":
        low, high = config.mid_range
    else:
        low, high = config.long_range
    high = min(int(high), int(max_allowed))
    if high < low:
        actual_length = int(max_allowed)
        return actual_length, classify_length_group(actual_length, config)
    actual_length = int(rng.integers(low, high + 1))
    return actual_length, group_name


def build_paths(output_root: Path) -> dict[str, Path]:
    scenario_root = output_root
    miss_root = scenario_root / "miss_set"
    return {
        "scenario_root": scenario_root,
        "miss_root": miss_root,
        "masks_root": miss_root / "masks",
        "miss_data_root": miss_root / "miss_data",
        "manifests_root": miss_root / "manifests",
        "audits_root": miss_root / "audits",
        "run_config_path": miss_root / "run_config.json",
        "run_commands_path": miss_root / "run_commands.txt",
        "prepare_summary_path": miss_root / "manifests" / "snh_prepare_summary.json",
        "chunk_summary_path": miss_root / "manifests" / "snh_chunk_summary.csv",
        "eligible_nodes_path": miss_root / "manifests" / "snh_eligible_nodes.csv",
        "event_path": miss_root / "manifests" / "spatial_neighbor_holdout_events.csv",
        "neighbor_lists_root": miss_root / "manifests" / "neighbor_lists",
        "chunk_status_path": miss_root / "manifests" / "snh_generate_missing_chunk_status.csv",
        "audit_json_path": miss_root / "audits" / "snh_missingness_audit.json",
        "audit_md_path": miss_root / "audits" / "snh_missingness_audit_zh.md",
    }


def load_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {input_dir}")
    return files


def chunk_output_paths(paths: dict[str, Path], scenario_tag: str, file_name: str) -> dict[str, Path]:
    stem = Path(file_name).stem
    return {
        "mask_path": paths["masks_root"] / scenario_tag / f"{stem}_mask.parquet",
        "missing_path": paths["miss_data_root"] / scenario_tag / file_name,
        "neighbor_path": paths["neighbor_lists_root"] / scenario_tag / f"{stem}_neighbors.csv",
    }


def chunk_output_complete(paths: dict[str, Path], scenario_tag: str, file_name: str) -> bool:
    outputs = chunk_output_paths(paths, scenario_tag, file_name)
    return outputs["mask_path"].exists() and outputs["missing_path"].exists() and outputs["neighbor_path"].exists()


def persist_checkpoint(
    paths: dict[str, Path],
    status_by_key: dict[tuple[str, int], dict[str, Any]],
    events_by_key: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    status_rows = list(status_by_key.values())
    event_rows = [row for rows in events_by_key.values() for row in rows]
    if status_rows:
        status_df = pd.DataFrame(status_rows).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)
        status_df.to_csv(paths["chunk_status_path"], index=False, encoding="utf-8-sig")
    if event_rows:
        event_df = pd.DataFrame(event_rows).sort_values(["missing_rate_target", "event_id"]).reset_index(drop=True)
        event_df.to_csv(paths["event_path"], index=False, encoding="utf-8-sig")


def build_prepare_artifacts(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    input_files = load_input_files(args.input_dir)
    first_df = pd.read_parquet(input_files[0], columns=[args.node_col, args.time_col, args.target_col])
    canonical_node_ids = np.sort(first_df[args.node_col].astype(np.int64).unique())
    unique_times = np.sort(first_df[args.time_col].astype(np.int64).unique())
    if len(unique_times) != args.period:
        raise RuntimeError(f"period mismatch: expected {args.period}, got {len(unique_times)}")
    chunk_rows: list[dict[str, Any]] = []
    for chunk_index, file_path in enumerate(input_files):
        df = pd.read_parquet(file_path, columns=[args.target_col])
        day_index = extract_day_index(file_path.name)
        chunk_rows.append(
            {
                "chunk_index": int(chunk_index),
                "day_index": int(day_index),
                "file_name": file_path.name,
                "row_count": int(len(df)),
                "target_non_null_count": int(df[args.target_col].notna().sum()),
            }
        )
        del df
        gc.collect()
    topo_df = pd.read_csv(args.topology_file, usecols=["起始节点ID", "结束节点ID", "长度"])
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
        previous = first_hop_lengths[start_idx].get(end_idx)
        if previous is None or safe_length < previous:
            first_hop_lengths[start_idx][end_idx] = safe_length
        previous = first_hop_lengths[end_idx].get(start_idx)
        if previous is None or safe_length < previous:
            first_hop_lengths[end_idx][start_idx] = safe_length
    second_hop_lengths: list[dict[int, float]] = [dict() for _ in range(len(canonical_node_ids))]
    for node_idx, neighbors in enumerate(first_hop_sets):
        for mid_idx in neighbors:
            length_to_mid = first_hop_lengths[node_idx][mid_idx]
            for second_idx in first_hop_sets[mid_idx]:
                if second_idx == node_idx or second_idx in neighbors:
                    continue
                total_length = length_to_mid + first_hop_lengths[mid_idx][second_idx]
                previous = second_hop_lengths[node_idx].get(second_idx)
                if previous is None or total_length < previous:
                    second_hop_lengths[node_idx][second_idx] = total_length
    eligible_rows: list[dict[str, Any]] = []
    preferred_scope: dict[int, int] = {}
    preferred_neighbors: dict[int, np.ndarray] = {}
    preferred_lengths: dict[int, np.ndarray] = {}
    for node_idx, node_id in enumerate(canonical_node_ids.tolist()):
        first_neighbors = np.asarray(sorted(first_hop_sets[node_idx]), dtype=np.int64)
        second_neighbors = np.asarray(sorted(second_hop_lengths[node_idx].keys()), dtype=np.int64)
        selected_scope = 0
        selected_neighbors = np.asarray([], dtype=np.int64)
        selected_lengths = np.asarray([], dtype=np.float32)
        if len(first_neighbors) >= args.min_available_neighbors:
            selected_scope = 1
            selected_neighbors = first_neighbors
            selected_lengths = np.asarray(
                [float(first_hop_lengths[node_idx][neighbor]) for neighbor in first_neighbors.tolist()],
                dtype=np.float32,
            )
        elif args.neighbor_scope >= 2 and len(second_neighbors) >= args.min_available_neighbors:
            selected_scope = 2
            selected_neighbors = second_neighbors
            selected_lengths = np.asarray(
                [float(second_hop_lengths[node_idx][neighbor]) for neighbor in second_neighbors.tolist()],
                dtype=np.float32,
            )
        preferred_scope[node_idx] = int(selected_scope)
        preferred_neighbors[node_idx] = selected_neighbors
        preferred_lengths[node_idx] = selected_lengths
        eligible_rows.append(
            {
                "node_id": int(node_id),
                "node_index": int(node_idx),
                "first_hop_neighbor_count": int(len(first_neighbors)),
                "second_hop_neighbor_count": int(len(second_neighbors)),
                "preferred_neighbor_scope": int(selected_scope),
                "preferred_available_neighbor_count": int(len(selected_neighbors)),
                "eligible": bool(selected_scope > 0),
            }
        )
    chunk_summary_df = pd.DataFrame(chunk_rows)
    eligible_df = pd.DataFrame(eligible_rows)
    chunk_summary_df.to_csv(paths["chunk_summary_path"], index=False, encoding="utf-8-sig")
    eligible_df.to_csv(paths["eligible_nodes_path"], index=False, encoding="utf-8-sig")
    prepare_payload = {
        "scenario_id": SCENARIO_ID,
        "mechanism": MECHANISM,
        "input_dir": str(args.input_dir),
        "topology_file": str(args.topology_file),
        "chunk_count": int(len(chunk_summary_df)),
        "canonical_node_count": int(len(canonical_node_ids)),
        "period": int(args.period),
        "eligible_node_count": int(eligible_df["eligible"].sum()),
        "preferred_first_hop_count": int((eligible_df["preferred_neighbor_scope"] == 1).sum()),
        "preferred_second_hop_count": int((eligible_df["preferred_neighbor_scope"] == 2).sum()),
        "min_available_neighbors": int(args.min_available_neighbors),
    }
    write_json(paths["prepare_summary_path"], prepare_payload)
    return {
        "input_files": input_files,
        "chunk_summary_df": chunk_summary_df,
        "canonical_node_ids": canonical_node_ids,
        "preferred_scope": preferred_scope,
        "preferred_neighbors": preferred_neighbors,
        "preferred_lengths": preferred_lengths,
        "eligible_df": eligible_df,
        "prepare_payload": prepare_payload,
    }


def build_row_lookup(
    df: pd.DataFrame,
    canonical_node_ids: np.ndarray,
    node_col: str,
    time_col: str,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(canonical_node_ids.tolist())}
    unique_times = np.sort(df[time_col].astype(np.int64).unique())
    if len(unique_times) != period:
        raise RuntimeError(f"period mismatch inside chunk: expected {period}, got {len(unique_times)}")
    time_to_idx = {int(time_value): idx for idx, time_value in enumerate(unique_times.tolist())}
    local = df[[node_col, time_col]].copy()
    local["row_index"] = np.arange(len(local), dtype=np.int64)
    local["node_index"] = local[node_col].astype(np.int64).map(node_to_idx)
    local["time_index"] = local[time_col].astype(np.int64).map(time_to_idx)
    if local["node_index"].isna().any():
        raise RuntimeError("row lookup found node ids not in canonical node list")
    if local["time_index"].isna().any():
        raise RuntimeError("row lookup found time values not in local period mapping")
    if local["time_index"].min() < 0 or local["time_index"].max() >= period:
        raise RuntimeError("time index is outside configured period")
    row_lookup = np.full((len(canonical_node_ids), period), -1, dtype=np.int64)
    row_lookup[
        local["node_index"].to_numpy(dtype=np.int64, copy=False),
        local["time_index"].to_numpy(dtype=np.int64, copy=False),
    ] = local["row_index"].to_numpy(dtype=np.int64, copy=False)
    if np.any(row_lookup < 0):
        raise RuntimeError("chunk does not contain complete node x time grid")
    return row_lookup, unique_times.astype(np.int64, copy=False)


def choose_target_nodes(
    rng: np.random.Generator,
    eligible_node_indices: np.ndarray,
    target_missing_count: int,
    period: int,
) -> np.ndarray:
    probabilities = np.full(len(eligible_node_indices), 1.0 / float(len(eligible_node_indices)), dtype=np.float64)
    missing_slots_per_node = rng.multinomial(target_missing_count, probabilities).astype(np.int64)
    max_slots = np.full(len(missing_slots_per_node), period, dtype=np.int64)
    overflow = missing_slots_per_node > max_slots
    while np.any(overflow):
        excess = int(np.sum(missing_slots_per_node[overflow] - max_slots[overflow]))
        missing_slots_per_node[overflow] = max_slots[overflow]
        recipients = np.flatnonzero(missing_slots_per_node < max_slots)
        if excess <= 0 or len(recipients) == 0:
            break
        topup = rng.multinomial(excess, np.full(len(recipients), 1.0 / float(len(recipients)), dtype=np.float64))
        missing_slots_per_node[recipients] += topup.astype(np.int64)
        overflow = missing_slots_per_node > max_slots
    return missing_slots_per_node


def try_place_event(
    coverage: np.ndarray,
    *,
    target_idx: int,
    start_slot: int,
    actual_length: int,
    neighbor_indices: np.ndarray,
) -> bool:
    stop_slot = start_slot + actual_length
    if np.any(coverage[target_idx, start_slot:stop_slot]):
        return False
    if len(neighbor_indices) > 0 and np.any(coverage[neighbor_indices, start_slot:stop_slot]):
        return False
    coverage[target_idx, start_slot:stop_slot] = True
    return True


def generate_chunk_payload(
    *,
    rng: np.random.Generator,
    target_missing_count: int,
    eligible_node_indices: np.ndarray,
    preferred_scope: dict[int, int],
    preferred_neighbors: dict[int, np.ndarray],
    preferred_lengths: dict[int, np.ndarray],
    canonical_node_ids: np.ndarray,
    period: int,
    unique_times: np.ndarray,
    length_config: LengthConfig,
    missing_rate: float,
    day_index: int,
    chunk_index: int,
    file_name: str,
    scenario_tag: str,
    event_id_start: int,
    neighbor_lists_dir: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    coverage = np.zeros((len(canonical_node_ids), period), dtype=bool)
    actual_length_matrix = np.zeros((len(canonical_node_ids), period), dtype=np.uint8)
    event_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    resampled_event_count = 0
    skipped_event_count = 0
    event_id = int(event_id_start)
    missing_slots_per_node = choose_target_nodes(rng, eligible_node_indices, target_missing_count, period)
    shuffled_positions = rng.permutation(len(eligible_node_indices))
    for sampled_idx in shuffled_positions.tolist():
        node_idx = int(eligible_node_indices[sampled_idx])
        remaining = int(missing_slots_per_node[sampled_idx])
        if remaining <= 0:
            continue
        neighbors = preferred_neighbors[node_idx]
        neighbor_lengths = preferred_lengths[node_idx]
        neighbor_scope = int(preferred_scope[node_idx])
        attempts = 0
        while remaining > 0 and attempts < max(20, remaining * 4):
            actual_length, length_group = sample_length(rng, length_config, max_allowed=min(remaining, period))
            if actual_length <= 0:
                break
            start_slot = int(rng.integers(0, period - actual_length + 1))
            accepted = try_place_event(
                coverage,
                target_idx=node_idx,
                start_slot=start_slot,
                actual_length=actual_length,
                neighbor_indices=neighbors,
            )
            if not accepted:
                attempts += 1
                resampled_event_count += 1
                continue
            stop_slot = start_slot + actual_length
            actual_length_matrix[node_idx, start_slot:stop_slot] = np.uint8(actual_length)
            neighbor_file = neighbor_lists_dir / f"{Path(file_name).stem}_neighbors.csv"
            event_rows.append(
                {
                    "event_id": int(event_id),
                    "missing_rate_target": float(missing_rate),
                    "target_node_id": int(canonical_node_ids[node_idx]),
                    "start_global_time_index": int(unique_times[start_slot]),
                    "end_global_time_index": int(unique_times[stop_slot - 1]),
                    "actual_length": int(actual_length),
                    "length_group": length_group,
                    "neighbor_scope": int(neighbor_scope),
                    "available_neighbor_count": int(len(neighbors)),
                    "neighbor_node_ids_file": str(neighbor_file.relative_to(neighbor_lists_dir.parent)),
                    "resample_count": int(attempts),
                    "seed": int(seed),
                    "chunk_index": int(chunk_index),
                    "day_index": int(day_index),
                    "file_name": file_name,
                    "scenario_tag": scenario_tag,
                }
            )
            for neighbor_idx, neighbor_length in zip(neighbors.tolist(), neighbor_lengths.tolist()):
                neighbor_rows.append(
                    {
                        "event_id": int(event_id),
                        "target_node_id": int(canonical_node_ids[node_idx]),
                        "neighbor_node_id": int(canonical_node_ids[int(neighbor_idx)]),
                        "neighbor_scope": int(neighbor_scope),
                        "path_length": float(neighbor_length),
                    }
                )
            remaining -= int(actual_length)
            attempts = 0
            event_id += 1
        if remaining > 0:
            skipped_event_count += int(remaining)
    unresolved = int(target_missing_count - int(coverage.sum()))
    topup_attempts = 0
    while unresolved > 0 and topup_attempts < max(1000, unresolved * 3):
        sampled_idx = int(rng.integers(0, len(eligible_node_indices)))
        node_idx = int(eligible_node_indices[sampled_idx])
        neighbors = preferred_neighbors[node_idx]
        neighbor_lengths = preferred_lengths[node_idx]
        neighbor_scope = int(preferred_scope[node_idx])
        start_slot = int(rng.integers(0, period))
        accepted = try_place_event(
            coverage,
            target_idx=node_idx,
            start_slot=start_slot,
            actual_length=1,
            neighbor_indices=neighbors,
        )
        if not accepted:
            topup_attempts += 1
            resampled_event_count += 1
            continue
        actual_length_matrix[node_idx, start_slot] = np.uint8(1)
        neighbor_file = neighbor_lists_dir / f"{Path(file_name).stem}_neighbors.csv"
        event_rows.append(
            {
                "event_id": int(event_id),
                "missing_rate_target": float(missing_rate),
                "target_node_id": int(canonical_node_ids[node_idx]),
                "start_global_time_index": int(unique_times[start_slot]),
                "end_global_time_index": int(unique_times[start_slot]),
                "actual_length": 1,
                "length_group": "short",
                "neighbor_scope": int(neighbor_scope),
                "available_neighbor_count": int(len(neighbors)),
                "neighbor_node_ids_file": str(neighbor_file.relative_to(neighbor_lists_dir.parent)),
                "resample_count": 0,
                "seed": int(seed),
                "chunk_index": int(chunk_index),
                "day_index": int(day_index),
                "file_name": file_name,
                "scenario_tag": scenario_tag,
            }
        )
        for neighbor_idx, neighbor_length in zip(neighbors.tolist(), neighbor_lengths.tolist()):
            neighbor_rows.append(
                {
                    "event_id": int(event_id),
                    "target_node_id": int(canonical_node_ids[node_idx]),
                    "neighbor_node_id": int(canonical_node_ids[int(neighbor_idx)]),
                    "neighbor_scope": int(neighbor_scope),
                    "path_length": float(neighbor_length),
                }
            )
        unresolved -= 1
        event_id += 1
    stats = {
        "resampled_event_count": int(resampled_event_count),
        "skipped_event_count": int(skipped_event_count + max(unresolved, 0)),
        "events_with_available_neighbors": int(len(event_rows)),
        "events_without_available_neighbors": 0,
        "average_available_neighbor_count": float(np.mean([row["available_neighbor_count"] for row in event_rows]))
        if event_rows
        else 0.0,
        "neighbor_observed_ratio": 1.0 if event_rows else 0.0,
    }
    return coverage, actual_length_matrix, event_rows, neighbor_rows, stats, int(event_id)


def write_chunk_outputs(
    *,
    args: argparse.Namespace,
    df: pd.DataFrame,
    row_lookup: np.ndarray,
    coverage: np.ndarray,
    actual_length_matrix: np.ndarray,
    unique_times: np.ndarray,
    missing_rate: float,
    scenario_tag: str,
    day_index: int,
    file_name: str,
    paths: dict[str, Path],
    chunk_index: int,
    stats: dict[str, Any],
    preferred_scope: dict[int, int],
    preferred_neighbors: dict[int, np.ndarray],
    canonical_node_ids: np.ndarray,
) -> dict[str, Any]:
    selected_positions = np.flatnonzero(coverage.reshape(-1))
    selected_lengths = actual_length_matrix.reshape(-1)[selected_positions].astype(np.int64, copy=False)
    node_positions = selected_positions // args.period
    time_positions = selected_positions % args.period
    selected_rows = row_lookup[node_positions, time_positions]
    selected_order = np.argsort(selected_rows, kind="mergesort")
    selected_rows = selected_rows[selected_order]
    node_positions = node_positions[selected_order]
    time_positions = time_positions[selected_order]
    selected_lengths = selected_lengths[selected_order]
    selected_groups = [classify_length_group(int(value), parse_length_config(args)) for value in selected_lengths.tolist()]
    selected_scope = np.asarray([preferred_scope[int(node_idx)] for node_idx in node_positions.tolist()], dtype=np.int64)
    selected_available_neighbors = np.asarray(
        [len(preferred_neighbors[int(node_idx)]) for node_idx in node_positions.tolist()],
        dtype=np.int64,
    )
    selected_time_values = unique_times[time_positions].astype(np.int64, copy=False)
    mask_df = pd.DataFrame(
        {
            "row_index": selected_rows,
            args.node_col: canonical_node_ids[node_positions].astype(np.int64, copy=False),
            args.time_col: selected_time_values,
            "day_index": np.full(len(selected_rows), day_index, dtype=np.int64),
            "global_time_index": selected_time_values,
            "is_missing": np.full(len(selected_rows), True, dtype=bool),
            "mechanism": np.full(len(selected_rows), MECHANISM),
            "missing_rate_target": np.full(len(selected_rows), missing_rate, dtype=np.float64),
            "event_id": -np.ones(len(selected_rows), dtype=np.int64),
            "actual_length": selected_lengths,
            "length_group": selected_groups,
            "available_neighbor_count": selected_available_neighbors,
            "neighbor_scope": selected_scope,
        }
    )
    mask_dir = paths["masks_root"] / scenario_tag
    miss_data_dir = paths["miss_data_root"] / scenario_tag
    ensure_dir(mask_dir)
    ensure_dir(miss_data_dir)
    mask_path = mask_dir / file_name.replace(".parquet", "_mask.parquet")
    missing_path = miss_data_dir / file_name
    out_df = df.copy()
    out_df.loc[selected_rows, args.target_col] = np.nan
    mask_df.to_parquet(mask_path, index=False)
    out_df.to_parquet(missing_path, index=False)
    return {
        "scenario_tag": scenario_tag,
        "missing_rate_target": float(missing_rate),
        "chunk_index": int(chunk_index),
        "day_index": int(day_index),
        "file_name": file_name,
        "row_count": int(len(df)),
        "observed_missing_count": int(len(selected_rows)),
        "observed_missing_rate": float(len(selected_rows) / float(len(df))) if len(df) else 0.0,
        "mask_path": str(mask_path),
        "missing_dataset_path": str(missing_path),
        "neighbor_observed_ratio": float(stats["neighbor_observed_ratio"]),
        "events_with_available_neighbors": int(stats["events_with_available_neighbors"]),
        "events_without_available_neighbors": int(stats["events_without_available_neighbors"]),
        "resampled_event_count": int(stats["resampled_event_count"]),
        "skipped_event_count": int(stats["skipped_event_count"]),
        "average_available_neighbor_count": float(stats["average_available_neighbor_count"]),
    }


def backfill_event_ids(mask_dir: Path, file_name: str, event_rows: list[dict[str, Any]]) -> None:
    mask_path = mask_dir / file_name.replace(".parquet", "_mask.parquet")
    mask_df = pd.read_parquet(mask_path)
    event_id_by_key: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in event_rows:
        start_time = int(row["start_global_time_index"])
        end_time = int(row["end_global_time_index"])
        chunk_times = sorted(mask_df["global_time_index"].astype(np.int64).unique().tolist())
        chunk_time_to_idx = {value: idx for idx, value in enumerate(chunk_times)}
        start_local = chunk_time_to_idx.get(start_time)
        end_local = chunk_time_to_idx.get(end_time)
        if start_local is None or end_local is None:
            continue
        for local_idx in range(start_local, end_local + 1):
            event_id_by_key[(int(row["target_node_id"]), int(chunk_times[local_idx]), int(row["actual_length"]))].append(
                int(row["event_id"])
            )
    assigned: list[int] = []
    for node_id, time_idx, actual_length in zip(
        mask_df["节点ID"].astype(np.int64).tolist() if "节点ID" in mask_df.columns else mask_df.iloc[:, 1].astype(np.int64).tolist(),
        mask_df["global_time_index"].astype(np.int64).tolist(),
        mask_df["actual_length"].astype(np.int64).tolist(),
    ):
        key = (int(node_id), int(time_idx), int(actual_length))
        bucket = event_id_by_key.get(key)
        assigned.append(int(bucket[0]) if bucket else -1)
    mask_df["event_id"] = np.asarray(assigned, dtype=np.int64)
    mask_df.to_parquet(mask_path, index=False)


def build_event_row_from_group(
    group_df: pd.DataFrame,
    *,
    event_id: int,
    missing_rate: float,
    seed: int,
    chunk_index: int,
    day_index: int,
    file_name: str,
    scenario_tag: str,
    neighbor_file_relative: str,
    node_col: str,
) -> dict[str, Any]:
    first_row = group_df.iloc[0]
    return {
        "event_id": int(event_id),
        "missing_rate_target": float(missing_rate),
        "target_node_id": int(first_row[node_col]),
        "start_global_time_index": int(group_df["global_time_index"].min()),
        "end_global_time_index": int(group_df["global_time_index"].max()),
        "actual_length": int(first_row["actual_length"]),
        "length_group": str(first_row["length_group"]),
        "neighbor_scope": int(first_row["neighbor_scope"]),
        "available_neighbor_count": int(first_row["available_neighbor_count"]),
        "neighbor_node_ids_file": neighbor_file_relative,
        "resample_count": 0,
        "seed": int(seed),
        "chunk_index": int(chunk_index),
        "day_index": int(day_index),
        "file_name": file_name,
        "scenario_tag": scenario_tag,
    }


def reconstruct_event_rows_from_mask(
    mask_df: pd.DataFrame,
    *,
    next_event_id: int,
    missing_rate: float,
    seed: int,
    chunk_index: int,
    day_index: int,
    file_name: str,
    scenario_tag: str,
    neighbor_file_relative: str,
    node_col: str,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    if mask_df.empty:
        return [], mask_df["event_id"].to_numpy(dtype=np.int64, copy=True), int(next_event_id)
    assigned_event_ids = mask_df["event_id"].fillna(-1).to_numpy(dtype=np.int64, copy=True)
    event_rows: list[dict[str, Any]] = []
    positive_df = mask_df.loc[mask_df["event_id"].fillna(-1).astype(np.int64) >= 0].copy()
    for event_id, group_df in positive_df.groupby("event_id", sort=True):
        event_rows.append(
            build_event_row_from_group(
                group_df.sort_values("global_time_index"),
                event_id=int(event_id),
                missing_rate=missing_rate,
                seed=seed,
                chunk_index=chunk_index,
                day_index=day_index,
                file_name=file_name,
                scenario_tag=scenario_tag,
                neighbor_file_relative=neighbor_file_relative,
                node_col=node_col,
            )
        )
        next_event_id = max(int(next_event_id), int(event_id) + 1)
    pending_df = (
        mask_df.loc[mask_df["event_id"].fillna(-1).astype(np.int64) < 0]
        .copy()
        .reset_index()
        .rename(columns={"index": "_mask_row_position"})
        .sort_values([node_col, "global_time_index", "row_index"])
        .reset_index(drop=True)
    )
    for _, node_group in pending_df.groupby(node_col, sort=False):
        group_rows = node_group.to_dict(orient="records")
        used = [False] * len(group_rows)
        for start_idx, row in enumerate(group_rows):
            if used[start_idx]:
                continue
            actual_length = int(row["actual_length"])
            start_time = int(row["global_time_index"])
            matched_positions: list[int] = []
            for step in range(max(actual_length, 1)):
                target_time = start_time + step
                candidate_idx = None
                for local_idx in range(start_idx, len(group_rows)):
                    candidate = group_rows[local_idx]
                    if used[local_idx]:
                        continue
                    if (
                        int(candidate["global_time_index"]) == target_time
                        and int(candidate["actual_length"]) == actual_length
                    ):
                        candidate_idx = local_idx
                        break
                if candidate_idx is None:
                    matched_positions = [start_idx]
                    actual_length = int(row["actual_length"])
                    break
                matched_positions.append(candidate_idx)
            event_id = int(next_event_id)
            next_event_id += 1
            absolute_positions: list[int] = []
            for local_idx in matched_positions:
                used[local_idx] = True
                absolute_positions.append(int(group_rows[local_idx]["_mask_row_position"]))
            assigned_event_ids[np.asarray(absolute_positions, dtype=np.int64)] = event_id
            inferred_group = mask_df.iloc[absolute_positions].sort_values("global_time_index")
            event_rows.append(
                build_event_row_from_group(
                    inferred_group,
                    event_id=event_id,
                    missing_rate=missing_rate,
                    seed=seed,
                    chunk_index=chunk_index,
                    day_index=day_index,
                    file_name=file_name,
                    scenario_tag=scenario_tag,
                    neighbor_file_relative=neighbor_file_relative,
                    node_col=node_col,
                )
            )
    event_rows.sort(key=lambda row: int(row["event_id"]))
    return event_rows, assigned_event_ids, int(next_event_id)


def reconstruct_checkpoint_for_chunk(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    chunk_row: dict[str, Any],
    missing_rate: float,
    scenario_tag: str,
    next_event_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    chunk_index = int(chunk_row["chunk_index"])
    day_index = int(chunk_row["day_index"])
    file_name = str(chunk_row["file_name"])
    outputs = chunk_output_paths(paths, scenario_tag, file_name)
    mask_df = pd.read_parquet(outputs["mask_path"])
    neighbor_file_relative = str(outputs["neighbor_path"].relative_to(paths["manifests_root"]))
    event_rows, assigned_event_ids, next_event_id = reconstruct_event_rows_from_mask(
        mask_df,
        next_event_id=next_event_id,
        missing_rate=float(missing_rate),
        seed=int(args.seed),
        chunk_index=chunk_index,
        day_index=day_index,
        file_name=file_name,
        scenario_tag=scenario_tag,
        neighbor_file_relative=neighbor_file_relative,
        node_col=args.node_col,
    )
    if not np.array_equal(mask_df["event_id"].to_numpy(dtype=np.int64, copy=False), assigned_event_ids):
        mask_df["event_id"] = assigned_event_ids
        mask_df.to_parquet(outputs["mask_path"], index=False)
    event_count = len(event_rows)
    available_neighbor_values = [int(row["available_neighbor_count"]) for row in event_rows]
    status_row = {
        "scenario_tag": scenario_tag,
        "missing_rate_target": float(missing_rate),
        "chunk_index": chunk_index,
        "day_index": day_index,
        "file_name": file_name,
        "row_count": int(chunk_row["row_count"]),
        "observed_missing_count": int(len(mask_df)),
        "observed_missing_rate": float(len(mask_df) / float(chunk_row["row_count"])) if int(chunk_row["row_count"]) else 0.0,
        "mask_path": str(outputs["mask_path"]),
        "missing_dataset_path": str(outputs["missing_path"]),
        "neighbor_observed_ratio": 1.0 if event_count > 0 else 0.0,
        "events_with_available_neighbors": int(sum(value > 0 for value in available_neighbor_values)),
        "events_without_available_neighbors": int(sum(value <= 0 for value in available_neighbor_values)),
        "resampled_event_count": 0,
        "skipped_event_count": 0,
        "average_available_neighbor_count": float(np.mean(available_neighbor_values)) if available_neighbor_values else 0.0,
    }
    return status_row, event_rows, int(next_event_id)


def load_existing_checkpoint(
    args: argparse.Namespace,
    paths: dict[str, Path],
    chunk_summary_df: pd.DataFrame,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]], set[tuple[str, int]], int]:
    status_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    events_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if paths["chunk_status_path"].exists():
        status_df = pd.read_csv(paths["chunk_status_path"])
        for row in status_df.to_dict(orient="records"):
            key = (str(row["scenario_tag"]), int(row["chunk_index"]))
            status_by_key[key] = row
    if paths["event_path"].exists():
        event_df = pd.read_csv(paths["event_path"])
        for (scenario_tag, chunk_index), group_df in event_df.groupby(["scenario_tag", "chunk_index"], sort=False):
            events_by_key[(str(scenario_tag), int(chunk_index))] = group_df.to_dict(orient="records")
    next_event_id = 0
    for event_rows in events_by_key.values():
        for row in event_rows:
            next_event_id = max(next_event_id, int(row["event_id"]) + 1)
    rebuilt = False
    for missing_rate in parse_rates(args.missing_rates):
        scenario_tag = scenario_rate_tag(missing_rate, args.seed)
        for chunk_row in chunk_summary_df.to_dict(orient="records"):
            key = (scenario_tag, int(chunk_row["chunk_index"]))
            file_name = str(chunk_row["file_name"])
            if not chunk_output_complete(paths, scenario_tag, file_name):
                continue
            needs_status = key not in status_by_key
            needs_events = key not in events_by_key or not events_by_key[key]
            if not needs_status and not needs_events:
                continue
            status_row, event_rows, next_event_id = reconstruct_checkpoint_for_chunk(
                args=args,
                paths=paths,
                chunk_row=chunk_row,
                missing_rate=missing_rate,
                scenario_tag=scenario_tag,
                next_event_id=next_event_id,
            )
            status_by_key[key] = status_row
            events_by_key[key] = event_rows
            persist_checkpoint(paths, status_by_key, events_by_key)
            rebuilt = True
    if rebuilt or status_by_key or events_by_key:
        persist_checkpoint(paths, status_by_key, events_by_key)
    completed_keys = {
        key
        for key in status_by_key
        if key in events_by_key and chunk_output_complete(paths, key[0], str(status_by_key[key]["file_name"]))
    }
    return status_by_key, events_by_key, completed_keys, int(next_event_id)


def run_prepare(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    for key in ["miss_root", "masks_root", "miss_data_root", "manifests_root", "audits_root", "neighbor_lists_root"]:
        ensure_dir(paths[key])
    artifacts = build_prepare_artifacts(args, paths)
    write_json(
        paths["run_config_path"],
        {
            "scenario_id": SCENARIO_ID,
            "mechanism": MECHANISM,
            "evaluation_protocol": "online_spatial_interpolation",
            "input_dir": str(args.input_dir),
            "output_dir": str(paths["scenario_root"]),
            "topology_file": str(args.topology_file),
            "missing_rates": args.missing_rates,
            "length_group_probs": args.length_group_probs,
            "short_length_range": args.short_length_range,
            "mid_length_range": args.mid_length_range,
            "long_length_range": args.long_length_range,
            "neighbor_scope": int(args.neighbor_scope),
            "min_available_neighbors": int(args.min_available_neighbors),
            "seed": int(args.seed),
            "target_col": args.target_col,
            "node_col": args.node_col,
            "time_col": args.time_col,
            "period": int(args.period),
        },
    )
    command = (
        "E:\\anaconda3\\envs\\analysis\\python.exe analysis_scripts\\spatial_neighbor_holdout_setting_pipeline.py "
        f"--stage all --input_dir {args.input_dir} --output_dir {paths['scenario_root']} "
        f"--topology_file {args.topology_file} --missing_rates {args.missing_rates} "
        f"--length_group_probs {args.length_group_probs} --short_length_range {args.short_length_range} "
        f"--mid_length_range {args.mid_length_range} --long_length_range {args.long_length_range} "
        f"--neighbor_scope {args.neighbor_scope} --min_available_neighbors {args.min_available_neighbors} "
        f"--seed {args.seed} --target_col {args.target_col} --node_col {args.node_col} "
        f"--time_col {args.time_col} --period {args.period}"
    )
    paths["run_commands_path"].write_text(command + "\n", encoding="utf-8")
    return artifacts


def run_generate_missing(args: argparse.Namespace, paths: dict[str, Path], artifacts: dict[str, Any]) -> pd.DataFrame:
    input_files: list[Path] = artifacts["input_files"]
    chunk_summary_df: pd.DataFrame = artifacts["chunk_summary_df"]
    canonical_node_ids: np.ndarray = artifacts["canonical_node_ids"]
    eligible_df: pd.DataFrame = artifacts["eligible_df"]
    preferred_scope: dict[int, int] = artifacts["preferred_scope"]
    preferred_neighbors: dict[int, np.ndarray] = artifacts["preferred_neighbors"]
    preferred_lengths: dict[int, np.ndarray] = artifacts["preferred_lengths"]
    eligible_node_indices = eligible_df.loc[eligible_df["eligible"], "node_index"].to_numpy(dtype=np.int64, copy=False)
    if len(eligible_node_indices) == 0:
        raise RuntimeError("no eligible nodes available for spatial_neighbor_holdout")
    length_config = parse_length_config(args)
    rates = parse_rates(args.missing_rates)
    status_by_key, events_by_key, completed_keys, next_event_id = load_existing_checkpoint(args, paths, chunk_summary_df)
    for missing_rate in rates:
        scenario_tag = scenario_rate_tag(missing_rate, args.seed)
        ensure_dir(paths["neighbor_lists_root"] / scenario_tag)
        for chunk_row in chunk_summary_df.to_dict(orient="records"):
            chunk_index = int(chunk_row["chunk_index"])
            file_name = str(chunk_row["file_name"])
            key = (scenario_tag, chunk_index)
            if key in completed_keys:
                continue
            file_path = input_files[chunk_index]
            day_index = int(chunk_row["day_index"])
            target_missing_count = int(round(int(chunk_row["target_non_null_count"]) * float(missing_rate)))
            df = pd.read_parquet(file_path, columns=[args.node_col, args.time_col, args.target_col])
            row_lookup, unique_times = build_row_lookup(df, canonical_node_ids, args.node_col, args.time_col, args.period)
            rng = np.random.default_rng(args.seed + int(round(missing_rate * 1000)) + chunk_index)
            coverage, actual_length_matrix, event_rows, neighbor_rows, stats, next_event_id = generate_chunk_payload(
                rng=rng,
                target_missing_count=target_missing_count,
                eligible_node_indices=eligible_node_indices,
                preferred_scope=preferred_scope,
                preferred_neighbors=preferred_neighbors,
                preferred_lengths=preferred_lengths,
                canonical_node_ids=canonical_node_ids,
                period=args.period,
                unique_times=unique_times,
                length_config=length_config,
                missing_rate=missing_rate,
                day_index=day_index,
                chunk_index=chunk_index,
                file_name=file_name,
                scenario_tag=scenario_tag,
                event_id_start=next_event_id,
                neighbor_lists_dir=paths["neighbor_lists_root"] / scenario_tag,
                seed=args.seed,
            )
            neighbor_file = paths["neighbor_lists_root"] / scenario_tag / f"{Path(file_name).stem}_neighbors.csv"
            pd.DataFrame(neighbor_rows).to_csv(neighbor_file, index=False, encoding="utf-8-sig")
            status_row = write_chunk_outputs(
                args=args,
                df=df,
                row_lookup=row_lookup,
                coverage=coverage,
                actual_length_matrix=actual_length_matrix,
                unique_times=unique_times,
                missing_rate=missing_rate,
                scenario_tag=scenario_tag,
                day_index=day_index,
                file_name=file_name,
                paths=paths,
                chunk_index=chunk_index,
                stats=stats,
                preferred_scope=preferred_scope,
                preferred_neighbors=preferred_neighbors,
                canonical_node_ids=canonical_node_ids,
            )
            backfill_event_ids(paths["masks_root"] / scenario_tag, file_name, event_rows)
            status_by_key[key] = status_row
            events_by_key[key] = event_rows
            completed_keys.add(key)
            persist_checkpoint(paths, status_by_key, events_by_key)
            del df, coverage, actual_length_matrix, row_lookup
            gc.collect()
    persist_checkpoint(paths, status_by_key, events_by_key)
    status_df = pd.DataFrame(list(status_by_key.values())).sort_values(["missing_rate_target", "chunk_index"]).reset_index(
        drop=True
    )
    return status_df


def infer_mask_stats(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    rates = parse_rates(args.missing_rates)
    status_df = pd.read_csv(paths["chunk_status_path"])
    event_df = pd.read_csv(paths["event_path"])
    per_rate: dict[str, Any] = {}
    for missing_rate in rates:
        rate_key = f"{missing_rate:.2f}"
        scenario_tag = scenario_rate_tag(missing_rate, args.seed)
        mask_dir = paths["masks_root"] / scenario_tag
        mask_files = sorted(mask_dir.glob("*_mask.parquet"))
        length_group_event_count = {label: 0 for label in LENGTH_GROUP_ORDER}
        length_group_mask_count = {label: 0 for label in LENGTH_GROUP_ORDER}
        min_available = math.inf
        max_available = 0
        for mask_path in mask_files:
            mask_df = pd.read_parquet(mask_path, columns=["length_group", "available_neighbor_count"])
            for label, count in mask_df["length_group"].value_counts(dropna=False).to_dict().items():
                if str(label) in length_group_mask_count:
                    length_group_mask_count[str(label)] += int(count)
            if len(mask_df) > 0:
                min_available = min(min_available, int(mask_df["available_neighbor_count"].min()))
                max_available = max(max_available, int(mask_df["available_neighbor_count"].max()))
        rate_events = event_df.loc[np.isclose(event_df["missing_rate_target"], missing_rate)].copy()
        for label, count in rate_events["length_group"].value_counts(dropna=False).to_dict().items():
            if str(label) in length_group_event_count:
                length_group_event_count[str(label)] += int(count)
        rate_status = status_df.loc[np.isclose(status_df["missing_rate_target"], missing_rate)].copy()
        observed_missing_rate = float(rate_status["observed_missing_count"].sum() / rate_status["row_count"].sum())
        events_with_neighbors = int(rate_status["events_with_available_neighbors"].sum())
        events_without_neighbors = int(rate_status["events_without_available_neighbors"].sum())
        per_rate[rate_key] = {
            "scenario_tag": scenario_tag,
            "mask_file_count": int(len(mask_files)),
            "miss_data_file_count": int(len(list((paths["miss_data_root"] / scenario_tag).glob("*.parquet")))),
            "observed_missing_rate": observed_missing_rate,
            "length_group_event_count": length_group_event_count,
            "length_group_mask_count": length_group_mask_count,
            "neighbor_observed_ratio": float(rate_status["neighbor_observed_ratio"].mean()) if len(rate_status) else 0.0,
            "average_available_neighbor_count": float(rate_status["average_available_neighbor_count"].mean())
            if len(rate_status)
            else 0.0,
            "min_available_neighbor_count": 0 if min_available is math.inf else int(min_available),
            "max_available_neighbor_count": int(max_available),
            "resampled_event_count": int(rate_status["resampled_event_count"].sum()),
            "skipped_event_count": int(rate_status["skipped_event_count"].sum()),
            "events_with_available_neighbors": events_with_neighbors,
            "events_without_available_neighbors": events_without_neighbors,
        }
    return per_rate


def run_audit(args: argparse.Namespace, paths: dict[str, Path], artifacts: dict[str, Any]) -> dict[str, Any]:
    per_rate_stats = infer_mask_stats(paths, args)
    payload = {
        "scenario_id": SCENARIO_ID,
        "mechanism": MECHANISM,
        "evaluation_protocol": "online_spatial_interpolation",
        "missing_rates": parse_rates(args.missing_rates),
        "length_groups": LENGTH_GROUP_ORDER,
        "neighbor_observed_enforced": True,
        "uses_current_time_neighbors_allowed_for_spatial_methods": True,
        "target_current_true_value_available_to_methods": False,
        "future_information_allowed": False,
        "per_rate": per_rate_stats,
    }
    write_json(paths["audit_json_path"], payload)
    lines = [
        "# spatial_neighbor_holdout 缺失机制审计",
        "",
        f"- scenario_id: `{SCENARIO_ID}`",
        f"- mechanism: `{MECHANISM}`",
        "- evaluation_protocol: `online_spatial_interpolation`",
        "- 该机制允许使用目标节点缺失时刻的邻居观测，但不允许使用目标节点当前真实值和未来信息。",
        "- 邻居可观测约束已在缺失生成阶段强制执行。",
        "",
        "## 每个缺失率",
        "",
    ]
    for rate_key, stats in per_rate_stats.items():
        lines.extend(
            [
                f"### {rate_key}",
                "",
                f"- mask 文件数: `{stats['mask_file_count']}`",
                f"- miss_data 文件数: `{stats['miss_data_file_count']}`",
                f"- observed_missing_rate: `{stats['observed_missing_rate']:.6f}`",
                f"- neighbor_observed_ratio: `{stats['neighbor_observed_ratio']:.6f}`",
                f"- average_available_neighbor_count: `{stats['average_available_neighbor_count']:.3f}`",
                f"- min_available_neighbor_count: `{stats['min_available_neighbor_count']}`",
                f"- max_available_neighbor_count: `{stats['max_available_neighbor_count']}`",
                f"- resampled_event_count: `{stats['resampled_event_count']}`",
                f"- skipped_event_count: `{stats['skipped_event_count']}`",
                "",
            ]
        )
    write_markdown(paths["audit_md_path"], lines)
    return payload


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.output_dir = ensure_absolute(project_root, args.output_dir)
    args.topology_file = ensure_absolute(project_root, args.topology_file)
    paths = build_paths(args.output_dir)
    artifacts = run_prepare(args, paths)
    if args.stage in {"generate_missing", "all"}:
        run_generate_missing(args, paths, artifacts)
    if args.stage in {"audit", "all"}:
        if not paths["chunk_status_path"].exists() or not paths["event_path"].exists():
            raise FileNotFoundError("generate_missing outputs are missing; run prepare/generate_missing first")
        run_audit(args, paths, artifacts)


if __name__ == "__main__":
    main()
