from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NODE_TEMPORAL_BLOCK = "node_temporal_block"
NODE_SUBSET_TEMPORAL_OUTAGE = "node_subset_temporal_outage"
DEFAULT_MECHANISMS = [NODE_TEMPORAL_BLOCK, NODE_SUBSET_TEMPORAL_OUTAGE]
OUTAGE_NODE_SUBSET_RATIOS = [0.005, 0.01, 0.02]
LENGTH_MODE_MIXED = "mixed_short_mid_long"
LENGTH_GROUP_ORDER = ("short", "mid", "long")
EVENT_FLUSH_BATCH_SIZE = 5000
CHUNK_STATUS_FILE_NAME = "structured_missing_chunk_status.csv"
CHUNK_RUNTIME_STATE_FILE_NAME = "structured_missing_chunk_runtime_state.jsonl"


@dataclass(frozen=True)
class StagePaths:
    root: Path
    masks_dir: Path
    missing_datasets_dir: Path
    manifests_dir: Path
    audits_dir: Path
    outage_node_lists_dir: Path
    run_config_path: Path
    run_commands_path: Path
    design_doc_path: Path


@dataclass(frozen=True)
class PrepareArtifacts:
    chunk_summary_df: pd.DataFrame
    node_ids: np.ndarray
    global_time_summary: Dict[str, Any]


@dataclass(frozen=True)
class LengthSamplingConfig:
    length_mode: str
    length_group_probs: Tuple[float, float, float]
    short_length_range: Tuple[int, int]
    mid_length_range: Tuple[int, int]
    long_length_range: Tuple[int, int]


@dataclass(frozen=True)
class ScenarioDefinition:
    mechanism: str
    missing_rate: float
    scenario_tag: str
    parameter_setting: str
    length_mode: str
    length_group_probs: Tuple[float, float, float]
    short_length_range: Tuple[int, int]
    mid_length_range: Tuple[int, int]
    long_length_range: Tuple[int, int]


class LengthStatsAccumulator:
    def __init__(self) -> None:
        self.event_count = 0
        self.sum_length = 0.0
        self.sum_sq_length = 0.0
        self.short_event_count = 0
        self.mid_event_count = 0
        self.long_event_count = 0
        self.actual_length_value_counts: Dict[int, int] = {}

    def add(self, actual_length: int, length_group: str) -> None:
        self.event_count += 1
        self.sum_length += float(actual_length)
        self.sum_sq_length += float(actual_length * actual_length)
        self.actual_length_value_counts[int(actual_length)] = self.actual_length_value_counts.get(int(actual_length), 0) + 1
        if length_group == "short":
            self.short_event_count += 1
        elif length_group == "mid":
            self.mid_event_count += 1
        elif length_group == "long":
            self.long_event_count += 1
        else:
            raise ValueError(f"unsupported length group: {length_group}")

    def to_summary(self, config: LengthSamplingConfig) -> Dict[str, Any]:
        if self.event_count <= 0:
            return {
                "fixed_lengths_only": False,
                "length_is_event_level_random_variable": True,
                "length_mode": config.length_mode,
                "short_length_range": list(config.short_length_range),
                "mid_length_range": list(config.mid_length_range),
                "long_length_range": list(config.long_length_range),
                "length_group_probs": list(config.length_group_probs),
                "length_min": 0,
                "length_max": 0,
                "length_mean": 0.0,
                "length_std": 0.0,
                "length_median": 0.0,
                "length_p25": 0.0,
                "length_p75": 0.0,
                "short_event_count": 0,
                "mid_event_count": 0,
                "long_event_count": 0,
                "short_event_ratio": 0.0,
                "mid_event_ratio": 0.0,
                "long_event_ratio": 0.0,
                "actual_length_value_counts": {},
            }
        mean = self.sum_length / float(self.event_count)
        variance = max((self.sum_sq_length / float(self.event_count)) - (mean * mean), 0.0)
        sorted_lengths = sorted(self.actual_length_value_counts.items())
        return {
            "fixed_lengths_only": False,
            "length_is_event_level_random_variable": True,
            "length_mode": config.length_mode,
            "short_length_range": list(config.short_length_range),
            "mid_length_range": list(config.mid_length_range),
            "long_length_range": list(config.long_length_range),
            "length_group_probs": list(config.length_group_probs),
            "length_min": int(sorted_lengths[0][0]),
            "length_max": int(sorted_lengths[-1][0]),
            "length_mean": float(mean),
            "length_std": float(math.sqrt(variance)),
            "length_median": float(quantile_from_counts(self.actual_length_value_counts, self.event_count, 0.50)),
            "length_p25": float(quantile_from_counts(self.actual_length_value_counts, self.event_count, 0.25)),
            "length_p75": float(quantile_from_counts(self.actual_length_value_counts, self.event_count, 0.75)),
            "short_event_count": int(self.short_event_count),
            "mid_event_count": int(self.mid_event_count),
            "long_event_count": int(self.long_event_count),
            "short_event_ratio": float(self.short_event_count / float(self.event_count)),
            "mid_event_ratio": float(self.mid_event_count / float(self.event_count)),
            "long_event_ratio": float(self.long_event_count / float(self.event_count)),
            "actual_length_value_counts": {
                str(length): int(count) for length, count in sorted(self.actual_length_value_counts.items())
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新增结构化缺失设置，不执行任何补全。")
    parser.add_argument("--stage", required=True, choices=["prepare", "generate_missing", "audit", "all"])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--mechanisms", required=True, type=str)
    parser.add_argument("--missing_rates", required=True, type=str)
    parser.add_argument("--length_group_probs", required=True, type=str)
    parser.add_argument("--short_length_range", required=True, type=str)
    parser.add_argument("--mid_length_range", required=True, type=str)
    parser.add_argument("--long_length_range", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--node_col", required=True, type=str)
    parser.add_argument("--time_col", required=True, type=str)
    parser.add_argument("--period", required=True, type=int)
    parser.add_argument("--tolerance", required=True, type=float)
    return parser.parse_args()


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def parse_float_list(raw: str) -> List[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def parse_int_pair(raw: str) -> Tuple[int, int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) != 2:
        raise ValueError(f"expected 2 integers, got: {raw}")
    return values[0], values[1]


def parse_mechanisms(raw: str) -> List[str]:
    mechanisms = [token.strip() for token in raw.split(",") if token.strip()]
    if not mechanisms:
        raise ValueError("mechanisms is empty")
    invalid = sorted(set(mechanisms) - set(DEFAULT_MECHANISMS))
    if invalid:
        raise ValueError(f"unsupported mechanisms: {invalid}")
    return mechanisms


def format_rate_tag(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), ensure_ascii=False) + "\n")


def append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    header = not path.exists()
    df.to_csv(path, mode=mode, header=header, index=False, encoding="utf-8-sig")


def build_paths(output_dir: Path) -> StagePaths:
    return StagePaths(
        root=output_dir,
        masks_dir=output_dir / "masks",
        missing_datasets_dir=output_dir / "missing_datasets",
        manifests_dir=output_dir / "manifests",
        audits_dir=output_dir / "audits",
        outage_node_lists_dir=output_dir / "manifests" / "outage_node_lists",
        run_config_path=output_dir / "run_config.json",
        run_commands_path=output_dir / "run_commands.txt",
        design_doc_path=output_dir / "structured_missingness_design_zh.md",
    )


def chunk_status_path(paths: StagePaths) -> Path:
    return paths.manifests_dir / CHUNK_STATUS_FILE_NAME


def chunk_runtime_state_path(paths: StagePaths) -> Path:
    return paths.manifests_dir / CHUNK_RUNTIME_STATE_FILE_NAME


def mkdirs(paths: StagePaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.masks_dir.mkdir(parents=True, exist_ok=True)
    paths.missing_datasets_dir.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    paths.audits_dir.mkdir(parents=True, exist_ok=True)
    paths.outage_node_lists_dir.mkdir(parents=True, exist_ok=True)


def validate_range_pair(name: str, range_pair: Tuple[int, int]) -> None:
    start, end = range_pair
    if start <= 0 or end <= 0 or start > end:
        raise ValueError(f"{name} must satisfy 0 < start <= end, got {range_pair}")


def validate_args(args: argparse.Namespace) -> Tuple[List[str], List[float], LengthSamplingConfig]:
    mechanisms = parse_mechanisms(args.mechanisms)
    missing_rates = parse_float_list(args.missing_rates)
    probs = tuple(parse_float_list(args.length_group_probs))
    if len(probs) != 3:
        raise ValueError("length_group_probs must contain exactly 3 values")
    prob_sum = sum(probs)
    if not math.isclose(prob_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"length_group_probs must sum to 1.0, got {prob_sum}")
    for probability in probs:
        if probability <= 0:
            raise ValueError("length_group_probs must all be positive")
    short_range = parse_int_pair(args.short_length_range)
    mid_range = parse_int_pair(args.mid_length_range)
    long_range = parse_int_pair(args.long_length_range)
    validate_range_pair("short_length_range", short_range)
    validate_range_pair("mid_length_range", mid_range)
    validate_range_pair("long_length_range", long_range)
    if not (short_range[1] < mid_range[0] and mid_range[1] < long_range[0]):
        raise ValueError("length ranges must be strictly ordered and non-overlapping")
    if args.period <= 0:
        raise ValueError("period must be positive")
    if args.tolerance <= 0:
        raise ValueError("tolerance must be positive")
    for rate in missing_rates:
        if rate <= 0 or rate >= 1:
            raise ValueError(f"missing rate must be within (0, 1): {rate}")
    return mechanisms, missing_rates, LengthSamplingConfig(
        length_mode=LENGTH_MODE_MIXED,
        length_group_probs=(float(probs[0]), float(probs[1]), float(probs[2])),
        short_length_range=short_range,
        mid_length_range=mid_range,
        long_length_range=long_range,
    )


def list_chunk_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("node_flow_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no chunk parquet files found in {input_dir}")
    return files


def extract_day_index(file_name: str) -> int:
    match = re.search(r"node_flow_chunk_(\d+)\.parquet$", file_name)
    if not match:
        raise ValueError(f"unexpected chunk file name: {file_name}")
    return int(match.group(1))


def build_parameter_setting(length_config: LengthSamplingConfig) -> str:
    short_prob, mid_prob, long_prob = length_config.length_group_probs
    short_start, short_end = length_config.short_length_range
    mid_start, mid_end = length_config.mid_length_range
    long_start, long_end = length_config.long_length_range
    return (
        f"{length_config.length_mode}: "
        f"short={short_start}-{short_end}@{short_prob:.1f}, "
        f"mid={mid_start}-{mid_end}@{mid_prob:.1f}, "
        f"long={long_start}-{long_end}@{long_prob:.1f}"
    )


def build_scenarios(
    mechanisms: List[str],
    missing_rates: List[float],
    length_config: LengthSamplingConfig,
    seed: int,
) -> List[ScenarioDefinition]:
    scenarios: List[ScenarioDefinition] = []
    parameter_setting = build_parameter_setting(length_config)
    for mechanism in mechanisms:
        for rate in missing_rates:
            scenarios.append(
                ScenarioDefinition(
                    mechanism=mechanism,
                    missing_rate=rate,
                    scenario_tag=(
                        f"mechanism_{mechanism}__rate_{format_rate_tag(rate)}__"
                        f"{length_config.length_mode}__seed_{seed}"
                    ),
                    parameter_setting=parameter_setting,
                    length_mode=length_config.length_mode,
                    length_group_probs=length_config.length_group_probs,
                    short_length_range=length_config.short_length_range,
                    mid_length_range=length_config.mid_length_range,
                    long_length_range=length_config.long_length_range,
                )
            )
    return scenarios


def write_run_artifacts(
    args: argparse.Namespace,
    paths: StagePaths,
    mechanisms: List[str],
    missing_rates: List[float],
    length_config: LengthSamplingConfig,
) -> None:
    config = {
        "experiment_name": "real_data_structured_missingness_setting",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "stage": args.stage,
        "mechanisms": mechanisms,
        "missing_rates": missing_rates,
        "length_mode": length_config.length_mode,
        "length_group_probs": list(length_config.length_group_probs),
        "short_length_range": list(length_config.short_length_range),
        "mid_length_range": list(length_config.mid_length_range),
        "long_length_range": list(length_config.long_length_range),
        "fixed_lengths_only": False,
        "length_is_event_level_random_variable": True,
        "node_subset_ratio_candidates": OUTAGE_NODE_SUBSET_RATIOS,
        "seed": args.seed,
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "period": args.period,
        "tolerance": args.tolerance,
        "imputation_enabled": False,
    }
    write_json(paths.run_config_path, config)
    lines = [
        "Structured missingness setting pipeline commands",
        f"python {' '.join(sys.argv)}",
    ]
    paths.run_commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_prepare_artifacts(paths: StagePaths) -> PrepareArtifacts:
    chunk_summary_path = paths.manifests_dir / "structured_prepare_chunk_summary.csv"
    node_index_path = paths.manifests_dir / "node_index.csv"
    global_time_path = paths.manifests_dir / "global_time_index_summary.json"
    if not chunk_summary_path.exists() or not node_index_path.exists() or not global_time_path.exists():
        raise FileNotFoundError("prepare outputs not found; run --stage prepare first")
    chunk_summary_df = pd.read_csv(chunk_summary_path)
    node_ids = pd.read_csv(node_index_path)["node_id"].to_numpy(dtype=np.int64, copy=False)
    global_time_summary = json.loads(global_time_path.read_text(encoding="utf-8"))
    return PrepareArtifacts(
        chunk_summary_df=chunk_summary_df,
        node_ids=node_ids,
        global_time_summary=global_time_summary,
    )


def prepare_chunk_layout(
    df: pd.DataFrame,
    canonical_node_ids: np.ndarray,
    node_col: str,
    time_col: str,
    period: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_values = df[node_col].to_numpy(dtype=np.int64, copy=False)
    time_values = df[time_col].to_numpy(dtype=np.int64, copy=False)
    sort_idx = np.lexsort((time_values, node_values))
    sorted_nodes = node_values[sort_idx]
    sorted_times = time_values[sort_idx]
    unique_nodes, counts = np.unique(sorted_nodes, return_counts=True)
    if not np.array_equal(unique_nodes, canonical_node_ids):
        raise RuntimeError("chunk node ids do not match canonical node index")
    if not np.all(counts == period):
        raise RuntimeError("each node must have exactly one observation for every time slot within the chunk")
    unique_times = np.unique(sorted_times)
    if len(unique_times) != period:
        raise RuntimeError(f"expected {period} time slots per chunk, got {len(unique_times)}")
    time_matrix = sorted_times.reshape(len(unique_nodes), period)
    if not np.array_equal(time_matrix, np.broadcast_to(unique_times, time_matrix.shape)):
        raise RuntimeError("time slots are not aligned across nodes after sorting")
    sorted_original_rows = np.arange(len(df), dtype=np.int64)[sort_idx]
    return unique_nodes, unique_times.astype(np.int64, copy=False), sorted_original_rows


def scenario_mask_dir(paths: StagePaths, scenario: ScenarioDefinition) -> Path:
    return paths.masks_dir / scenario.scenario_tag


def scenario_missing_dir(paths: StagePaths, scenario: ScenarioDefinition) -> Path:
    return paths.missing_datasets_dir / scenario.scenario_tag


def scenario_event_path(paths: StagePaths, scenario: ScenarioDefinition) -> Path:
    event_stem = "node_temporal_block_events" if scenario.mechanism == NODE_TEMPORAL_BLOCK else "node_subset_temporal_outage_events"
    return paths.manifests_dir / f"{event_stem}__{scenario.scenario_tag}.csv"


def legacy_event_path(paths: StagePaths, scenario: ScenarioDefinition) -> Path:
    if scenario.mechanism == NODE_TEMPORAL_BLOCK:
        return paths.manifests_dir / "node_temporal_block_events.csv"
    return paths.manifests_dir / "node_subset_temporal_outage_events.csv"


def write_design_doc(paths: StagePaths, length_config: LengthSamplingConfig) -> None:
    short_start, short_end = length_config.short_length_range
    mid_start, mid_end = length_config.mid_length_range
    long_start, long_end = length_config.long_length_range
    short_prob, mid_prob, long_prob = length_config.length_group_probs
    lines = [
        "# 结构化缺失设计说明",
        "",
        "1. 已有 `results\\real_data_global_missingness_setting` 继续保留为 global MCAR point 随机点缺失基准。",
        "2. 本轮新增 `node_temporal_block` 和 `node_subset_temporal_outage` 两类结构化缺失机制。",
        "3. 每个机制、每个缺失率只生成一套结构化缺失数据集，不再按固定长度拆分为多个 block 目录。",
        "4. 连续缺失长度采用事件级随机变量 `mixed_short_mid_long`。",
        f"5. short_block 范围为 `{short_start}-{short_end}` 个时间片，采样概率 `{short_prob}`。",
        f"6. mid_block 范围为 `{mid_start}-{mid_end}` 个时间片，采样概率 `{mid_prob}`。",
        f"7. long_block 范围为 `{long_start}-{long_end}` 个时间片，采样概率 `{long_prob}`。",
        "8. `mask` 与 `missing_datasets` 仍精确到 `row_index`，且只修改目标列。",
        "9. 两类机制都不会把同一时间片下全部路口整体置缺失，也不会覆盖现有 global MCAR point 结果。",
        "",
    ]
    paths.design_doc_path.write_text("\n".join(lines), encoding="utf-8")


def run_prepare(args: argparse.Namespace, paths: StagePaths) -> PrepareArtifacts:
    files = list_chunk_files(args.input_dir)
    records: List[Dict[str, Any]] = []
    canonical_node_ids: Optional[np.ndarray] = None
    expected_row_count: Optional[int] = None

    for chunk_index, file_path in enumerate(files):
        df = pd.read_parquet(file_path, columns=[args.node_col, args.time_col, args.target_col])
        if {args.node_col, args.time_col, args.target_col} - set(df.columns):
            raise KeyError(f"{file_path.name} missing required columns")
        day_index = extract_day_index(file_path.name)
        unique_nodes = np.unique(df[args.node_col].to_numpy(dtype=np.int64, copy=False))
        unique_times = np.unique(df[args.time_col].to_numpy(dtype=np.int64, copy=False))

        if canonical_node_ids is None:
            canonical_node_ids = unique_nodes.astype(np.int64, copy=False)
        elif not np.array_equal(canonical_node_ids, unique_nodes):
            raise RuntimeError(f"{file_path.name} node set differs from previous chunks")

        if expected_row_count is None:
            expected_row_count = int(len(df))
        elif expected_row_count != int(len(df)):
            raise RuntimeError(f"{file_path.name} row_count differs from previous chunks")

        if len(unique_times) != args.period:
            raise RuntimeError(f"{file_path.name} time_slot_count != period")

        records.append(
            {
                "chunk_index": chunk_index,
                "day_index": day_index,
                "file_name": file_path.name,
                "row_count": int(len(df)),
                "node_count": int(len(unique_nodes)),
                "time_slot_count": int(len(unique_times)),
                "time_slot_min": int(unique_times.min()),
                "time_slot_max": int(unique_times.max()),
                "target_non_null_count": int(df[args.target_col].notna().sum()),
                "target_col": args.target_col,
                "node_col": args.node_col,
                "time_col": args.time_col,
            }
        )
        del df, unique_nodes, unique_times
        gc.collect()

    if canonical_node_ids is None:
        raise RuntimeError("no chunk files found")

    chunk_summary_df = pd.DataFrame(records).sort_values(["chunk_index"]).reset_index(drop=True)
    chunk_summary_df.to_csv(
        paths.manifests_dir / "structured_prepare_chunk_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"node_id": canonical_node_ids}).to_csv(
        paths.manifests_dir / "node_index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_time_summary = {
        "chunk_count": int(len(chunk_summary_df)),
        "period": int(args.period),
        "global_time_index_min": int(chunk_summary_df["time_slot_min"].min()),
        "global_time_index_max": int(chunk_summary_df["time_slot_max"].max()),
        "time_slots_per_day": int(args.period),
    }
    write_json(paths.manifests_dir / "global_time_index_summary.json", global_time_summary)
    return PrepareArtifacts(
        chunk_summary_df=chunk_summary_df,
        node_ids=canonical_node_ids,
        global_time_summary=global_time_summary,
    )


def quantile_from_counts(value_counts: Dict[int, int], total_count: int, quantile: float) -> int:
    if total_count <= 0:
        return 0
    threshold = max(1, int(math.ceil(total_count * quantile)))
    cumulative = 0
    for value in sorted(value_counts):
        cumulative += int(value_counts[value])
        if cumulative >= threshold:
            return int(value)
    return int(max(value_counts))


def scenario_length_config(scenario: ScenarioDefinition) -> LengthSamplingConfig:
    return LengthSamplingConfig(
        length_mode=scenario.length_mode,
        length_group_probs=scenario.length_group_probs,
        short_length_range=scenario.short_length_range,
        mid_length_range=scenario.mid_length_range,
        long_length_range=scenario.long_length_range,
    )


def classify_length_group(actual_length: int, config: LengthSamplingConfig) -> str:
    if config.short_length_range[0] <= actual_length <= config.short_length_range[1]:
        return "short"
    if config.mid_length_range[0] <= actual_length <= config.mid_length_range[1]:
        return "mid"
    if config.long_length_range[0] <= actual_length <= config.long_length_range[1]:
        return "long"
    raise ValueError(f"actual length {actual_length} is outside configured ranges")


def sample_length_group(rng: np.random.Generator, config: LengthSamplingConfig) -> str:
    index = int(rng.choice(np.arange(len(LENGTH_GROUP_ORDER)), p=np.array(config.length_group_probs, dtype=np.float64)))
    return LENGTH_GROUP_ORDER[index]


def sample_length_from_group(rng: np.random.Generator, length_group: str, config: LengthSamplingConfig) -> int:
    if length_group == "short":
        low, high = config.short_length_range
    elif length_group == "mid":
        low, high = config.mid_length_range
    elif length_group == "long":
        low, high = config.long_length_range
    else:
        raise ValueError(f"unsupported length group: {length_group}")
    return int(rng.integers(low, high + 1))


def sample_event_length(
    rng: np.random.Generator,
    config: LengthSamplingConfig,
    max_allowed: Optional[int] = None,
) -> Tuple[int, str]:
    length_group = sample_length_group(rng, config)
    actual_length = sample_length_from_group(rng, length_group, config)
    if max_allowed is not None:
        actual_length = min(actual_length, int(max_allowed))
        if actual_length <= 0:
            raise ValueError("max_allowed must produce positive actual_length")
        length_group = classify_length_group(actual_length, config)
    return int(actual_length), length_group


def flush_event_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    append_csv(pd.DataFrame(rows), path)
    rows.clear()


def split_available_intervals(
    intervals: List[Tuple[int, int]],
    interval_index: int,
    start_time: int,
    end_time: int,
) -> None:
    interval_start, interval_end = intervals.pop(interval_index)
    left_end = start_time - 2
    right_start = end_time + 2
    new_intervals: List[Tuple[int, int]] = []
    if interval_start <= left_end:
        new_intervals.append((interval_start, left_end))
    if right_start <= interval_end:
        new_intervals.append((right_start, interval_end))
    for offset, item in enumerate(new_intervals):
        intervals.insert(interval_index + offset, item)


def choose_interval_index(
    eligible_intervals: Sequence[Tuple[int, int]],
    actual_length: int,
    rng: np.random.Generator,
) -> int:
    weights = np.array([(end - start - actual_length + 2) for start, end in eligible_intervals], dtype=np.float64)
    weights = weights / weights.sum()
    return int(rng.choice(np.arange(len(eligible_intervals)), p=weights))


def generate_block_scenario(
    scenario: ScenarioDefinition,
    node_ids: np.ndarray,
    total_time_slots: int,
    total_observation_count: int,
    seed: int,
    tolerance: float,
    event_path: Path,
    event_id_start: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    node_count = len(node_ids)
    target_missing_count = int(round(total_observation_count * scenario.missing_rate))
    rng = np.random.default_rng(stable_seed(seed, scenario.mechanism, scenario.scenario_tag))
    coverage = np.zeros((node_count, total_time_slots), dtype=bool)
    actual_length_matrix = np.zeros((node_count, total_time_slots), dtype=np.uint8)
    missing_slots_per_node = rng.multinomial(target_missing_count, np.full(node_count, 1.0 / float(node_count))).astype(np.int64)
    stats = LengthStatsAccumulator()
    config = scenario_length_config(scenario)
    event_rows: List[Dict[str, Any]] = []
    observed_missing_count = 0
    event_id = int(event_id_start)

    for node_position, node_target in enumerate(missing_slots_per_node.tolist()):
        if node_target <= 0:
            continue
        available_intervals: List[Tuple[int, int]] = [(0, total_time_slots - 1)]
        remaining_for_node = int(node_target)
        while remaining_for_node > 0:
            max_free_length = max((end - start + 1 for start, end in available_intervals), default=0)
            if max_free_length <= 0:
                raise RuntimeError("node_temporal_block ran out of available intervals before meeting target")
            actual_length, length_group = sample_event_length(
                rng=rng,
                config=config,
                max_allowed=min(remaining_for_node, max_free_length),
            )
            eligible_intervals = [(start, end) for start, end in available_intervals if (end - start + 1) >= actual_length]
            picked_within_eligible = choose_interval_index(eligible_intervals, actual_length, rng)
            picked_interval = eligible_intervals[picked_within_eligible]
            picked_interval_index = available_intervals.index(picked_interval)
            interval_start, interval_end = picked_interval
            start_time = int(rng.integers(interval_start, interval_end - actual_length + 2))
            end_time = start_time + actual_length - 1
            if coverage[node_position, max(0, start_time - 1): min(total_time_slots, end_time + 2)].any():
                raise RuntimeError("node_temporal_block attempted to overwrite or merge an existing interval")
            coverage[node_position, start_time:end_time + 1] = True
            actual_length_matrix[node_position, start_time:end_time + 1] = np.uint8(actual_length)
            split_available_intervals(available_intervals, picked_interval_index, start_time, end_time)
            observed_missing_count += int(actual_length)
            remaining_for_node -= int(actual_length)
            stats.add(actual_length, length_group)
            event_rows.append(
                {
                    "mechanism": scenario.mechanism,
                    "missing_rate_target": scenario.missing_rate,
                    "event_id": event_id,
                    "node_id": int(node_ids[node_position]),
                    "start_global_time_index": int(start_time),
                    "end_global_time_index": int(end_time),
                    "actual_length": int(actual_length),
                    "length_group": length_group,
                    "length_mode": scenario.length_mode,
                    "seed": seed,
                }
            )
            event_id += 1
            if len(event_rows) >= EVENT_FLUSH_BATCH_SIZE:
                flush_event_rows(event_rows, event_path)
        gc.collect()

    flush_event_rows(event_rows, event_path)
    observed_count_from_mask = int(coverage.sum())
    if observed_count_from_mask != observed_missing_count:
        raise RuntimeError("node_temporal_block observed count mismatch after generation")
    absolute_error = abs((observed_missing_count / float(total_observation_count)) - scenario.missing_rate)
    scenario_summary = {
        "mechanism": scenario.mechanism,
        "missing_rate_target": scenario.missing_rate,
        "parameter_setting": scenario.parameter_setting,
        "target_missing_count": int(target_missing_count),
        "observed_missing_count": int(observed_missing_count),
        "observed_missing_rate": float(observed_missing_count / float(total_observation_count)),
        "absolute_error": float(absolute_error),
        "is_within_tolerance": bool(absolute_error <= tolerance),
        "point_topup_count": 0,
        "event_count": int(stats.event_count),
    }
    scenario_summary.update(stats.to_summary(config))
    return coverage, actual_length_matrix, scenario_summary


def outage_node_count_for_ratio(node_count: int, ratio: float) -> int:
    return max(1, min(node_count - 1, int(round(node_count * ratio))))


def try_place_outage_event(
    coverage: np.ndarray,
    node_positions: np.ndarray,
    start_time: int,
    end_time: int,
) -> bool:
    guard_start = max(0, start_time - 1)
    guard_end = min(coverage.shape[1] - 1, end_time + 1)
    if coverage[node_positions, guard_start:guard_end + 1].any():
        return False
    prospective = coverage[:, start_time:end_time + 1].copy()
    prospective[node_positions, :] = True
    if prospective.all(axis=0).any():
        return False
    return True


def generate_outage_scenario(
    scenario: ScenarioDefinition,
    node_ids: np.ndarray,
    total_time_slots: int,
    total_observation_count: int,
    seed: int,
    tolerance: float,
    outage_node_lists_dir: Path,
    event_path: Path,
    event_id_start: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    node_count = len(node_ids)
    target_missing_count = int(round(total_observation_count * scenario.missing_rate))
    tolerance_count = max(1, int(round(total_observation_count * tolerance)))
    rng = np.random.default_rng(stable_seed(seed, scenario.mechanism, scenario.scenario_tag))
    coverage = np.zeros((node_count, total_time_slots), dtype=bool)
    actual_length_matrix = np.zeros((node_count, total_time_slots), dtype=np.uint8)
    stats = LengthStatsAccumulator()
    config = scenario_length_config(scenario)
    event_rows: List[Dict[str, Any]] = []
    observed_missing_count = 0
    event_id = int(event_id_start)
    ratio_candidates = [(float(ratio), outage_node_count_for_ratio(node_count, float(ratio))) for ratio in OUTAGE_NODE_SUBSET_RATIOS]

    while target_missing_count - observed_missing_count > tolerance_count:
        remaining = target_missing_count - observed_missing_count
        candidate_order = list(range(len(ratio_candidates)))
        rng.shuffle(candidate_order)
        placed = False
        for ratio_index in candidate_order:
            ratio, selected_node_count = ratio_candidates[ratio_index]
            max_length_allowed = (remaining + tolerance_count) // selected_node_count
            if max_length_allowed <= 0:
                continue
            actual_length, length_group = sample_event_length(
                rng=rng,
                config=config,
                max_allowed=min(max_length_allowed, config.long_length_range[1]),
            )
            for _attempt in range(200):
                start_time = int(rng.integers(0, total_time_slots - actual_length + 1))
                end_time = start_time + actual_length - 1
                node_positions = np.sort(rng.choice(node_count, size=selected_node_count, replace=False).astype(np.int64, copy=False))
                if not try_place_outage_event(coverage, node_positions, start_time, end_time):
                    continue
                coverage[node_positions, start_time:end_time + 1] = True
                actual_length_matrix[node_positions, start_time:end_time + 1] = np.uint8(actual_length)
                observed_missing_count += int(selected_node_count * actual_length)
                node_list = node_ids[node_positions]
                node_list_path = outage_node_lists_dir / f"event_{event_id:06d}_nodes.csv"
                pd.DataFrame({"event_id": event_id, "node_id": node_list.astype(np.int64, copy=False)}).to_csv(
                    node_list_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                event_rows.append(
                    {
                        "mechanism": scenario.mechanism,
                        "missing_rate_target": scenario.missing_rate,
                        "event_id": event_id,
                        "node_subset_ratio": float(ratio),
                        "selected_node_count": int(selected_node_count),
                        "start_global_time_index": int(start_time),
                        "end_global_time_index": int(end_time),
                        "actual_length": int(actual_length),
                        "length_group": length_group,
                        "length_mode": scenario.length_mode,
                        "seed": seed,
                        "node_list_file": str(node_list_path.relative_to(outage_node_lists_dir.parent)),
                    }
                )
                event_id += 1
                stats.add(actual_length, length_group)
                placed = True
                if len(event_rows) >= EVENT_FLUSH_BATCH_SIZE:
                    flush_event_rows(event_rows, event_path)
                break
            if placed:
                break
        if not placed:
            raise RuntimeError("node_subset_temporal_outage failed to place a valid event within search budget")

    flush_event_rows(event_rows, event_path)
    observed_missing_count = int(coverage.sum())
    absolute_error = abs((observed_missing_count / float(total_observation_count)) - scenario.missing_rate)
    scenario_summary = {
        "mechanism": scenario.mechanism,
        "missing_rate_target": scenario.missing_rate,
        "parameter_setting": scenario.parameter_setting,
        "target_missing_count": int(target_missing_count),
        "observed_missing_count": int(observed_missing_count),
        "observed_missing_rate": float(observed_missing_count / float(total_observation_count)),
        "absolute_error": float(absolute_error),
        "is_within_tolerance": bool(absolute_error <= tolerance),
        "point_topup_count": 0,
        "event_count": int(stats.event_count),
        "node_subset_ratio_candidates": OUTAGE_NODE_SUBSET_RATIOS,
    }
    scenario_summary.update(stats.to_summary(config))
    return coverage, actual_length_matrix, scenario_summary


def write_scenario_outputs(
    *,
    args: argparse.Namespace,
    paths: StagePaths,
    scenario: ScenarioDefinition,
    prepare_artifacts: PrepareArtifacts,
    coverage: np.ndarray,
    actual_length_matrix: np.ndarray,
    current_status_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    chunk_summary_df = prepare_artifacts.chunk_summary_df
    node_ids = prepare_artifacts.node_ids
    mask_dir = scenario_mask_dir(paths, scenario)
    missing_dir = scenario_missing_dir(paths, scenario)
    mask_dir.mkdir(parents=True, exist_ok=True)
    missing_dir.mkdir(parents=True, exist_ok=True)
    config = scenario_length_config(scenario)

    status_rows: List[Dict[str, Any]] = []
    for row in chunk_summary_df.to_dict(orient="records"):
        chunk_index = int(row["chunk_index"])
        day_index = int(row["day_index"])
        file_name = str(row["file_name"])
        file_path = args.input_dir / file_name
        df = pd.read_parquet(file_path)
        required_columns = {args.node_col, args.time_col, args.target_col}
        if required_columns - set(df.columns):
            raise KeyError(f"{file_name} missing required columns in generate_missing stage")

        _, unique_times, sorted_original_rows = prepare_chunk_layout(
            df=df,
            canonical_node_ids=node_ids,
            node_col=args.node_col,
            time_col=args.time_col,
            period=args.period,
        )
        day_start = int(unique_times.min())
        day_end = int(unique_times.max())
        day_mask = coverage[:, day_start:day_end + 1]
        day_lengths = actual_length_matrix[:, day_start:day_end + 1]
        if day_mask.shape[1] != args.period:
            raise RuntimeError(f"{file_name} day mask width mismatch")

        selected_sorted_positions = np.flatnonzero(day_mask.reshape(-1))
        selected_rows = sorted_original_rows[selected_sorted_positions].astype(np.int64, copy=False)
        selected_lengths = day_lengths.reshape(-1)[selected_sorted_positions].astype(np.int64, copy=False)
        row_sort_order = np.argsort(selected_rows, kind="mergesort")
        selected_rows = selected_rows[row_sort_order]
        selected_lengths = selected_lengths[row_sort_order]
        selected_groups = [classify_length_group(int(length), config) for length in selected_lengths.tolist()]
        existing_status_row = get_existing_status_row(current_status_df, scenario, chunk_index)
        if can_reuse_existing_chunk(
            paths=paths,
            scenario=scenario,
            chunk_index=chunk_index,
            expected_missing_count=int(len(selected_rows)),
            file_name=file_name,
            existing_status_row=existing_status_row,
        ):
            status_row = dict(existing_status_row)
            status_rows.append(status_row)
            append_jsonl(
                chunk_runtime_state_path(paths),
                {
                    "action": "skip_existing_chunk",
                    "scenario_tag": scenario.scenario_tag,
                    "mechanism": scenario.mechanism,
                    "missing_rate_target": scenario.missing_rate,
                    "chunk_index": chunk_index,
                    "day_index": day_index,
                    "file_name": file_name,
                    "observed_missing_count": int(status_row["observed_missing_count"]),
                },
            )
            del df
            gc.collect()
            continue

        if len(selected_rows) > 0:
            selected_nodes = df.iloc[selected_rows][args.node_col].to_numpy(dtype=np.int64, copy=False)
            selected_times = df.iloc[selected_rows][args.time_col].to_numpy(dtype=np.int64, copy=False)
            mask_df = pd.DataFrame(
                {
                    "row_index": selected_rows,
                    args.node_col: selected_nodes,
                    args.time_col: selected_times,
                    "day_index": np.full(len(selected_rows), day_index, dtype=np.int64),
                    "global_time_index": selected_times,
                    "is_missing": np.full(len(selected_rows), True, dtype=bool),
                    "mechanism": np.full(len(selected_rows), scenario.mechanism),
                    "missing_rate_target": np.full(len(selected_rows), scenario.missing_rate, dtype=np.float64),
                    "actual_length": selected_lengths,
                    "length_group": selected_groups,
                    "length_mode": np.full(len(selected_rows), scenario.length_mode),
                }
            )
        else:
            mask_df = pd.DataFrame(
                columns=[
                    "row_index",
                    args.node_col,
                    args.time_col,
                    "day_index",
                    "global_time_index",
                    "is_missing",
                    "mechanism",
                    "missing_rate_target",
                    "actual_length",
                    "length_group",
                    "length_mode",
                ]
            )

        mask_path = mask_dir / file_name.replace(".parquet", "_mask.parquet")
        mask_df.to_parquet(mask_path, index=False)

        out_df = df.copy()
        original_target = out_df[args.target_col].copy()
        out_df.loc[selected_rows, args.target_col] = np.nan
        for column in [column for column in out_df.columns if column != args.target_col]:
            if not out_df[column].equals(df[column]):
                raise RuntimeError(f"{file_name} modified non-target columns")

        output_path = missing_dir / file_name
        out_df.to_parquet(output_path, index=False)
        all_nodes_missing_same_time = bool(day_mask.all(axis=0).any())
        entire_day_missing = bool(day_mask.all())
        status_row = {
            "mechanism": scenario.mechanism,
            "missing_rate_target": scenario.missing_rate,
            "scenario_tag": scenario.scenario_tag,
            "parameter_setting": scenario.parameter_setting,
            "length_mode": scenario.length_mode,
            "chunk_index": chunk_index,
            "day_index": day_index,
            "file_name": file_name,
            "mask_path": str(mask_path),
            "missing_dataset_path": str(output_path),
            "mask_file_count": 1,
            "missing_dataset_file_count": 1,
            "row_count": int(len(df)),
            "target_non_null_count": int(original_target.notna().sum()),
            "observed_missing_count": int(len(selected_rows)),
            "observed_missing_rate": float(len(selected_rows) / float(len(df))),
            "drops_entire_day": entire_day_missing,
            "drops_all_nodes_at_same_time": all_nodes_missing_same_time,
            "uses_row_index_mask": True,
            "modifies_only_target_col": True,
        }
        status_rows.append(status_row)
        current_status_df = upsert_chunk_status_row(current_status_df, status_row)
        persist_chunk_status_df(paths, current_status_df)
        append_jsonl(
            chunk_runtime_state_path(paths),
            {
                "action": "write_chunk",
                "scenario_tag": scenario.scenario_tag,
                "mechanism": scenario.mechanism,
                "missing_rate_target": scenario.missing_rate,
                "chunk_index": chunk_index,
                "day_index": day_index,
                "file_name": file_name,
                "observed_missing_count": int(len(selected_rows)),
                "mask_path": str(mask_path),
                "missing_dataset_path": str(output_path),
            },
        )

        del df, out_df, original_target, mask_df
        gc.collect()

    return pd.DataFrame(status_rows).sort_values(["chunk_index"]).reset_index(drop=True), current_status_df


def scenario_key(scenario: ScenarioDefinition) -> Tuple[str, float]:
    return scenario.mechanism, round(float(scenario.missing_rate), 6)


def next_event_id(event_path: Path) -> int:
    if not event_path.exists():
        return 0
    max_event_id = -1
    for chunk_df in pd.read_csv(event_path, usecols=["event_id"], chunksize=200000):
        if not chunk_df.empty:
            max_event_id = max(max_event_id, int(chunk_df["event_id"].max()))
    return max_event_id + 1


def scenario_output_counts(paths: StagePaths, scenario: ScenarioDefinition) -> Tuple[int, int]:
    mask_dir = scenario_mask_dir(paths, scenario)
    missing_dir = scenario_missing_dir(paths, scenario)
    mask_count = len(list(mask_dir.glob("*.parquet"))) if mask_dir.exists() else 0
    missing_count = len(list(missing_dir.glob("*.parquet"))) if missing_dir.exists() else 0
    return mask_count, missing_count


def scenario_is_complete(paths: StagePaths, scenario: ScenarioDefinition, expected_chunk_count: int) -> bool:
    mask_count, missing_count = scenario_output_counts(paths, scenario)
    return mask_count == expected_chunk_count and missing_count == expected_chunk_count


def load_existing_chunk_status(paths: StagePaths) -> pd.DataFrame:
    status_path = chunk_status_path(paths)
    if not status_path.exists():
        return pd.DataFrame()
    return pd.read_csv(status_path)


def persist_chunk_status_df(paths: StagePaths, status_df: pd.DataFrame) -> None:
    if status_df.empty:
        return
    status_df.sort_values(["mechanism", "missing_rate_target", "chunk_index"]).to_csv(
        chunk_status_path(paths),
        index=False,
        encoding="utf-8-sig",
    )


def upsert_chunk_status_row(
    current_status_df: pd.DataFrame,
    row: Dict[str, Any],
) -> pd.DataFrame:
    row_df = pd.DataFrame([row])
    if current_status_df.empty:
        return row_df
    keep_mask = ~(
        (current_status_df["scenario_tag"].astype(str) == str(row["scenario_tag"]))
        & (current_status_df["chunk_index"].astype(int) == int(row["chunk_index"]))
    )
    return pd.concat([current_status_df.loc[keep_mask].copy(), row_df], ignore_index=True)


def get_existing_status_row(
    current_status_df: pd.DataFrame,
    scenario: ScenarioDefinition,
    chunk_index: int,
) -> Optional[Dict[str, Any]]:
    if current_status_df.empty:
        return None
    filtered_df = current_status_df.loc[
        (current_status_df["scenario_tag"].astype(str) == scenario.scenario_tag)
        & (current_status_df["chunk_index"].astype(int) == int(chunk_index))
    ]
    if filtered_df.empty:
        return None
    return filtered_df.iloc[0].to_dict()


def can_reuse_existing_chunk(
    *,
    paths: StagePaths,
    scenario: ScenarioDefinition,
    chunk_index: int,
    expected_missing_count: int,
    file_name: str,
    existing_status_row: Optional[Dict[str, Any]],
) -> bool:
    if existing_status_row is None:
        return False
    mask_path = scenario_mask_dir(paths, scenario) / file_name.replace(".parquet", "_mask.parquet")
    missing_path = scenario_missing_dir(paths, scenario) / file_name
    if not mask_path.exists() or not missing_path.exists():
        return False
    try:
        mask_meta = pq.ParquetFile(mask_path).metadata
        missing_meta = pq.ParquetFile(missing_path).metadata
    except Exception:
        return False
    if int(mask_meta.num_rows) != int(expected_missing_count):
        return False
    if int(existing_status_row.get("observed_missing_count", -1)) != int(expected_missing_count):
        return False
    if int(existing_status_row.get("chunk_index", -1)) != int(chunk_index):
        return False
    return int(missing_meta.num_rows) > 0


def cleanup_resume_artifacts(
    paths: StagePaths,
    scenarios: List[ScenarioDefinition],
    expected_chunk_count: int,
) -> None:
    for temp_path in paths.manifests_dir.glob("*.tmp"):
        temp_path.unlink()
    for scenario in scenarios:
        event_path = scenario_event_path(paths, scenario)
        temp_path = event_path.with_suffix(".tmp")
        if temp_path.exists():
            temp_path.unlink()


def reconstruct_existing_chunk_status(
    args: argparse.Namespace,
    paths: StagePaths,
    scenario: ScenarioDefinition,
    prepare_artifacts: PrepareArtifacts,
) -> pd.DataFrame:
    records_by_file = {
        str(row["file_name"]): row for row in prepare_artifacts.chunk_summary_df.to_dict(orient="records")
    }
    node_count = len(prepare_artifacts.node_ids)
    status_rows: List[Dict[str, Any]] = []
    mask_dir = scenario_mask_dir(paths, scenario)
    missing_dir = scenario_missing_dir(paths, scenario)
    for file_name, row in sorted(records_by_file.items(), key=lambda item: int(item[1]["chunk_index"])):
        mask_path = mask_dir / file_name.replace(".parquet", "_mask.parquet")
        missing_path = missing_dir / file_name
        mask_df = pd.read_parquet(mask_path)
        observed_missing_count = int(len(mask_df))
        if observed_missing_count > 0:
            slot_counts = mask_df.groupby("global_time_index").size()
            all_nodes_missing_same_time = bool((slot_counts >= node_count).any())
        else:
            all_nodes_missing_same_time = False
        status_rows.append(
            {
                "mechanism": scenario.mechanism,
                "missing_rate_target": scenario.missing_rate,
                "scenario_tag": scenario.scenario_tag,
                "parameter_setting": scenario.parameter_setting,
                "length_mode": scenario.length_mode,
                "chunk_index": int(row["chunk_index"]),
                "day_index": int(row["day_index"]),
                "file_name": file_name,
                "mask_path": str(mask_path),
                "missing_dataset_path": str(missing_path),
                "mask_file_count": 1,
                "missing_dataset_file_count": 1,
                "row_count": int(row["row_count"]),
                "target_non_null_count": int(row["target_non_null_count"]),
                "observed_missing_count": observed_missing_count,
                "observed_missing_rate": float(observed_missing_count / float(row["row_count"])),
                "drops_entire_day": bool(observed_missing_count == int(row["row_count"])),
                "drops_all_nodes_at_same_time": all_nodes_missing_same_time,
                "uses_row_index_mask": True,
                "modifies_only_target_col": True,
            }
        )
    return pd.DataFrame(status_rows).sort_values(["chunk_index"]).reset_index(drop=True)


def load_event_stats_for_scenario(
    event_paths: Sequence[Path],
    scenario: ScenarioDefinition,
) -> LengthStatsAccumulator:
    stats = LengthStatsAccumulator()
    visited: set[str] = set()
    for event_path in event_paths:
        resolved = str(event_path)
        if resolved in visited or not event_path.exists():
            continue
        visited.add(resolved)
        for chunk_df in pd.read_csv(
            event_path,
            usecols=["mechanism", "missing_rate_target", "actual_length", "length_group"],
            chunksize=200000,
        ):
            filtered_df = chunk_df.loc[
                (chunk_df["mechanism"].astype(str) == scenario.mechanism)
                & np.isclose(chunk_df["missing_rate_target"].astype(float), float(scenario.missing_rate))
            ]
            for row in filtered_df.itertuples(index=False):
                stats.add(int(row.actual_length), str(row.length_group))
    return stats


def load_length_stats_from_masks(
    paths: StagePaths,
    scenario: ScenarioDefinition,
    prepare_artifacts: PrepareArtifacts,
) -> LengthStatsAccumulator:
    stats = LengthStatsAccumulator()
    mask_dir = scenario_mask_dir(paths, scenario)
    if not mask_dir.exists():
        return stats
    for row in prepare_artifacts.chunk_summary_df.to_dict(orient="records"):
        file_name = str(row["file_name"])
        mask_path = mask_dir / file_name.replace(".parquet", "_mask.parquet")
        if not mask_path.exists():
            continue
        mask_df = pd.read_parquet(mask_path, columns=["row_index", "actual_length", "length_group"])
        if mask_df.empty:
            continue
        event_df = (
            mask_df.loc[:, ["row_index", "actual_length", "length_group"]]
            .drop_duplicates()
            .sort_values(["row_index", "actual_length", "length_group"], kind="mergesort")
        )
        for item in event_df.itertuples(index=False):
            stats.add(int(item.actual_length), str(item.length_group))
    return stats


def build_existing_scenario_summary(
    scenario: ScenarioDefinition,
    chunk_status_df: pd.DataFrame,
    total_observation_count: int,
    tolerance: float,
    event_path: Path,
    paths: StagePaths,
    prepare_artifacts: PrepareArtifacts,
) -> Dict[str, Any]:
    config = scenario_length_config(scenario)
    stats = load_event_stats_for_scenario([event_path, legacy_event_path(paths, scenario)], scenario)
    if stats.event_count <= 0:
        stats = load_length_stats_from_masks(paths, scenario, prepare_artifacts)
    observed_missing_count = int(chunk_status_df["observed_missing_count"].sum())
    absolute_error = abs((observed_missing_count / float(total_observation_count)) - scenario.missing_rate)
    summary = {
        "mechanism": scenario.mechanism,
        "missing_rate_target": scenario.missing_rate,
        "parameter_setting": scenario.parameter_setting,
        "target_missing_count": int(round(total_observation_count * scenario.missing_rate)),
        "observed_missing_count": observed_missing_count,
        "observed_missing_rate": float(observed_missing_count / float(total_observation_count)),
        "absolute_error": float(absolute_error),
        "is_within_tolerance": bool(absolute_error <= tolerance),
        "point_topup_count": 0,
        "event_count": int(stats.event_count),
        "drops_all_nodes_at_same_time": bool(chunk_status_df["drops_all_nodes_at_same_time"].astype(bool).any()),
        "drops_entire_day": bool(chunk_status_df["drops_entire_day"].astype(bool).any()),
        "chunk_count": int(len(chunk_status_df)),
        "mask_file_count": int(chunk_status_df["mask_file_count"].sum()),
        "missing_dataset_file_count": int(chunk_status_df["missing_dataset_file_count"].sum()),
        "uses_row_index_mask": True,
        "modifies_only_target_col": True,
        "scenario_tag": scenario.scenario_tag,
    }
    summary.update(stats.to_summary(config))
    if scenario.mechanism == NODE_SUBSET_TEMPORAL_OUTAGE:
        summary["node_subset_ratio_candidates"] = OUTAGE_NODE_SUBSET_RATIOS
    return summary


def allocate_chunk_missing_counts(
    chunk_summary_df: pd.DataFrame,
    scenario: ScenarioDefinition,
    seed: int,
) -> pd.DataFrame:
    records = chunk_summary_df.sort_values(["chunk_index"]).to_dict(orient="records")
    global_eligible_count = int(chunk_summary_df["target_non_null_count"].sum())
    global_missing_count = int(round(global_eligible_count * scenario.missing_rate))
    remaining_total = global_eligible_count
    remaining_missing = global_missing_count
    rng = np.random.default_rng(stable_seed(seed, scenario.mechanism, scenario.scenario_tag, "chunk_allocation"))
    rows: List[Dict[str, Any]] = []
    for pos, record in enumerate(records):
        eligible_count = int(record["target_non_null_count"])
        if pos == len(records) - 1:
            allocated_missing_count = remaining_missing
        elif eligible_count == 0 or remaining_missing == 0:
            allocated_missing_count = 0
        else:
            allocated_missing_count = int(
                rng.hypergeometric(
                    ngood=eligible_count,
                    nbad=remaining_total - eligible_count,
                    nsample=remaining_missing,
                )
            )
        allocated_missing_count = max(0, min(allocated_missing_count, eligible_count, remaining_missing))
        rows.append(
            {
                "chunk_index": int(record["chunk_index"]),
                "day_index": int(record["day_index"]),
                "file_name": str(record["file_name"]),
                "target_non_null_count": eligible_count,
                "allocated_missing_count": int(allocated_missing_count),
            }
        )
        remaining_total -= eligible_count
        remaining_missing -= allocated_missing_count
    if remaining_total != 0 or remaining_missing != 0:
        raise RuntimeError(f"chunk allocation did not terminate cleanly for {scenario.scenario_tag}")
    return pd.DataFrame(rows).sort_values(["chunk_index"]).reset_index(drop=True)


def build_chunk_target_lookup(allocation_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    return {int(row["chunk_index"]): row for row in allocation_df.to_dict(orient="records")}


def ensure_event_file_schema(path: Path, required_columns: Sequence[str]) -> None:
    if not path.exists():
        return
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        path.unlink()
        return
    if list(columns) != list(required_columns):
        path.unlink()


def generate_block_chunk_payload(
    *,
    scenario: ScenarioDefinition,
    node_ids: np.ndarray,
    unique_times: np.ndarray,
    target_missing_count: int,
    seed: int,
    chunk_index: int,
    day_index: int,
    file_name: str,
    event_id_start: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], int]:
    node_count = len(node_ids)
    period = len(unique_times)
    config = scenario_length_config(scenario)
    for attempt in range(20):
        rng = np.random.default_rng(
            stable_seed(seed, scenario.mechanism, scenario.scenario_tag, chunk_index, day_index, attempt)
        )
        coverage = np.zeros((node_count, period), dtype=bool)
        actual_length_matrix = np.zeros((node_count, period), dtype=np.uint8)
        missing_slots_per_node = rng.multinomial(target_missing_count, np.full(node_count, 1.0 / float(node_count))).astype(
            np.int64
        )
        event_rows: List[Dict[str, Any]] = []
        event_id = int(event_id_start)
        for node_position, node_target in enumerate(missing_slots_per_node.tolist()):
            if node_target <= 0:
                continue
            available_intervals: List[Tuple[int, int]] = [(0, period - 1)]
            remaining_for_node = int(node_target)
            while remaining_for_node > 0:
                max_free_length = max((end - start + 1 for start, end in available_intervals), default=0)
                if max_free_length <= 0:
                    raise RuntimeError(f"{scenario.scenario_tag} chunk {chunk_index} ran out of available intervals")
                actual_length, length_group = sample_event_length(
                    rng=rng,
                    config=config,
                    max_allowed=min(remaining_for_node, max_free_length),
                )
                eligible_intervals = [
                    (start, end) for start, end in available_intervals if (end - start + 1) >= actual_length
                ]
                picked_within_eligible = choose_interval_index(eligible_intervals, actual_length, rng)
                picked_interval = eligible_intervals[picked_within_eligible]
                picked_interval_index = available_intervals.index(picked_interval)
                interval_start, interval_end = picked_interval
                start_local = int(rng.integers(interval_start, interval_end - actual_length + 2))
                end_local = start_local + actual_length - 1
                coverage[node_position, start_local : end_local + 1] = True
                actual_length_matrix[node_position, start_local : end_local + 1] = np.uint8(actual_length)
                split_available_intervals(available_intervals, picked_interval_index, start_local, end_local)
                event_rows.append(
                    {
                        "mechanism": scenario.mechanism,
                        "missing_rate_target": scenario.missing_rate,
                        "event_id": event_id,
                        "node_id": int(node_ids[node_position]),
                        "start_global_time_index": int(unique_times[start_local]),
                        "end_global_time_index": int(unique_times[end_local]),
                        "actual_length": int(actual_length),
                        "length_group": length_group,
                        "length_mode": scenario.length_mode,
                        "seed": seed,
                        "chunk_index": chunk_index,
                        "day_index": day_index,
                        "file_name": file_name,
                    }
                )
                event_id += 1
                remaining_for_node -= int(actual_length)
        if not coverage.all(axis=0).any():
            return coverage, actual_length_matrix, event_rows, 0
    raise RuntimeError(f"{scenario.scenario_tag} chunk {chunk_index} failed to satisfy non-full-slot constraint")


def generate_outage_chunk_payload(
    *,
    scenario: ScenarioDefinition,
    node_ids: np.ndarray,
    unique_times: np.ndarray,
    target_missing_count: int,
    seed: int,
    chunk_index: int,
    day_index: int,
    file_name: str,
    event_id_start: int,
    outage_node_lists_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], int]:
    node_count = len(node_ids)
    period = len(unique_times)
    config = scenario_length_config(scenario)
    ratio_candidates = [(float(ratio), outage_node_count_for_ratio(node_count, float(ratio))) for ratio in OUTAGE_NODE_SUBSET_RATIOS]
    min_selected_node_count = min(selected for _, selected in ratio_candidates)
    scenario_outage_dir = outage_node_lists_dir / scenario.scenario_tag / Path(file_name).stem
    scenario_outage_dir.mkdir(parents=True, exist_ok=True)
    for existing_csv in scenario_outage_dir.glob("*.csv"):
        existing_csv.unlink()
    for attempt in range(20):
        rng = np.random.default_rng(
            stable_seed(seed, scenario.mechanism, scenario.scenario_tag, chunk_index, day_index, attempt)
        )
        coverage = np.zeros((node_count, period), dtype=bool)
        actual_length_matrix = np.zeros((node_count, period), dtype=np.uint8)
        event_rows: List[Dict[str, Any]] = []
        event_id = int(event_id_start)
        observed_missing_count = 0
        while True:
            remaining = target_missing_count - observed_missing_count
            if remaining < min_selected_node_count:
                break
            candidate_order = list(range(len(ratio_candidates)))
            rng.shuffle(candidate_order)
            placed = False
            for ratio_index in candidate_order:
                ratio, selected_node_count = ratio_candidates[ratio_index]
                max_length_allowed = remaining // selected_node_count
                if max_length_allowed <= 0:
                    continue
                actual_length, length_group = sample_event_length(
                    rng=rng,
                    config=config,
                    max_allowed=min(max_length_allowed, config.long_length_range[1]),
                )
                if actual_length <= 0:
                    continue
                for _attempt in range(200):
                    start_local = int(rng.integers(0, period - actual_length + 1))
                    end_local = start_local + actual_length - 1
                    node_positions = np.sort(
                        rng.choice(node_count, size=selected_node_count, replace=False).astype(np.int64, copy=False)
                    )
                    if not try_place_outage_event(coverage, node_positions, start_local, end_local):
                        continue
                    coverage[node_positions, start_local : end_local + 1] = True
                    actual_length_matrix[node_positions, start_local : end_local + 1] = np.uint8(actual_length)
                    observed_missing_count += int(selected_node_count * actual_length)
                    node_list = node_ids[node_positions]
                    node_list_path = scenario_outage_dir / f"event_{event_id:06d}_nodes.csv"
                    pd.DataFrame({"event_id": event_id, "node_id": node_list.astype(np.int64, copy=False)}).to_csv(
                        node_list_path,
                        index=False,
                        encoding="utf-8-sig",
                    )
                    event_rows.append(
                        {
                            "mechanism": scenario.mechanism,
                            "missing_rate_target": scenario.missing_rate,
                            "event_id": event_id,
                            "node_subset_ratio": float(ratio),
                            "selected_node_count": int(selected_node_count),
                            "start_global_time_index": int(unique_times[start_local]),
                            "end_global_time_index": int(unique_times[end_local]),
                            "actual_length": int(actual_length),
                            "length_group": length_group,
                            "length_mode": scenario.length_mode,
                            "seed": seed,
                            "node_list_file": str(node_list_path.relative_to(outage_node_lists_dir.parent)),
                            "chunk_index": chunk_index,
                            "day_index": day_index,
                            "file_name": file_name,
                        }
                    )
                    event_id += 1
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                break
        point_topup_count = target_missing_count - int(coverage.sum())
        if point_topup_count < 0:
            raise RuntimeError(f"{scenario.scenario_tag} chunk {chunk_index} overshot target count")
        if point_topup_count > 0:
            uncovered_positions = np.flatnonzero(~coverage.reshape(-1))
            if point_topup_count > len(uncovered_positions):
                raise RuntimeError(f"{scenario.scenario_tag} chunk {chunk_index} lacks uncovered positions for point top-up")
            topup_positions = np.sort(rng.choice(uncovered_positions, size=point_topup_count, replace=False).astype(np.int64))
            topup_nodes = topup_positions // period
            topup_times = topup_positions % period
            coverage[topup_nodes, topup_times] = True
            actual_length_matrix[topup_nodes, topup_times] = np.uint8(1)
        if not coverage.all(axis=0).any():
            return coverage, actual_length_matrix, event_rows, int(point_topup_count)
    raise RuntimeError(f"{scenario.scenario_tag} chunk {chunk_index} failed to satisfy non-full-slot constraint")


def write_single_chunk_outputs(
    *,
    args: argparse.Namespace,
    paths: StagePaths,
    scenario: ScenarioDefinition,
    row: Dict[str, Any],
    node_ids: np.ndarray,
    coverage: np.ndarray,
    actual_length_matrix: np.ndarray,
    current_status_df: pd.DataFrame,
    point_topup_count: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    chunk_index = int(row["chunk_index"])
    day_index = int(row["day_index"])
    file_name = str(row["file_name"])
    file_path = args.input_dir / file_name
    df = pd.read_parquet(file_path)
    required_columns = {args.node_col, args.time_col, args.target_col}
    if required_columns - set(df.columns):
        raise KeyError(f"{file_name} missing required columns in generate_missing stage")
    _, unique_times, sorted_original_rows = prepare_chunk_layout(
        df=df,
        canonical_node_ids=node_ids,
        node_col=args.node_col,
        time_col=args.time_col,
        period=args.period,
    )
    if coverage.shape[1] != len(unique_times):
        raise RuntimeError(f"{file_name} local coverage width mismatch")
    config = scenario_length_config(scenario)
    selected_sorted_positions = np.flatnonzero(coverage.reshape(-1))
    selected_rows = sorted_original_rows[selected_sorted_positions].astype(np.int64, copy=False)
    selected_lengths = actual_length_matrix.reshape(-1)[selected_sorted_positions].astype(np.int64, copy=False)
    row_sort_order = np.argsort(selected_rows, kind="mergesort")
    selected_rows = selected_rows[row_sort_order]
    selected_lengths = selected_lengths[row_sort_order]
    selected_groups = [classify_length_group(int(length), config) for length in selected_lengths.tolist()]
    if len(selected_rows) > 0:
        selected_nodes = df.iloc[selected_rows][args.node_col].to_numpy(dtype=np.int64, copy=False)
        selected_times = df.iloc[selected_rows][args.time_col].to_numpy(dtype=np.int64, copy=False)
        mask_df = pd.DataFrame(
            {
                "row_index": selected_rows,
                args.node_col: selected_nodes,
                args.time_col: selected_times,
                "day_index": np.full(len(selected_rows), day_index, dtype=np.int64),
                "global_time_index": selected_times,
                "is_missing": np.full(len(selected_rows), True, dtype=bool),
                "mechanism": np.full(len(selected_rows), scenario.mechanism),
                "missing_rate_target": np.full(len(selected_rows), scenario.missing_rate, dtype=np.float64),
                "actual_length": selected_lengths,
                "length_group": selected_groups,
                "length_mode": np.full(len(selected_rows), scenario.length_mode),
            }
        )
    else:
        mask_df = pd.DataFrame(
            columns=[
                "row_index",
                args.node_col,
                args.time_col,
                "day_index",
                "global_time_index",
                "is_missing",
                "mechanism",
                "missing_rate_target",
                "actual_length",
                "length_group",
                "length_mode",
            ]
        )
    mask_dir = scenario_mask_dir(paths, scenario)
    missing_dir = scenario_missing_dir(paths, scenario)
    mask_dir.mkdir(parents=True, exist_ok=True)
    missing_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / file_name.replace(".parquet", "_mask.parquet")
    output_path = missing_dir / file_name
    mask_df.to_parquet(mask_path, index=False)
    out_df = df.copy()
    original_target = out_df[args.target_col].copy()
    out_df.loc[selected_rows, args.target_col] = np.nan
    for column in [column for column in out_df.columns if column != args.target_col]:
        if not out_df[column].equals(df[column]):
            raise RuntimeError(f"{file_name} modified non-target columns")
    out_df.to_parquet(output_path, index=False)
    status_row = {
        "mechanism": scenario.mechanism,
        "missing_rate_target": scenario.missing_rate,
        "scenario_tag": scenario.scenario_tag,
        "parameter_setting": scenario.parameter_setting,
        "length_mode": scenario.length_mode,
        "chunk_index": chunk_index,
        "day_index": day_index,
        "file_name": file_name,
        "mask_path": str(mask_path),
        "missing_dataset_path": str(output_path),
        "mask_file_count": 1,
        "missing_dataset_file_count": 1,
        "row_count": int(len(df)),
        "target_non_null_count": int(original_target.notna().sum()),
        "observed_missing_count": int(len(selected_rows)),
        "observed_missing_rate": float(len(selected_rows) / float(len(df))),
        "drops_entire_day": bool(coverage.all()),
        "drops_all_nodes_at_same_time": bool(coverage.all(axis=0).any()),
        "uses_row_index_mask": True,
        "modifies_only_target_col": True,
        "point_topup_count": int(point_topup_count),
    }
    current_status_df = upsert_chunk_status_row(current_status_df, status_row)
    persist_chunk_status_df(paths, current_status_df)
    append_jsonl(
        chunk_runtime_state_path(paths),
        {
            "action": "write_chunk",
            "scenario_tag": scenario.scenario_tag,
            "mechanism": scenario.mechanism,
            "missing_rate_target": scenario.missing_rate,
            "chunk_index": chunk_index,
            "day_index": day_index,
            "file_name": file_name,
            "observed_missing_count": int(len(selected_rows)),
            "point_topup_count": int(point_topup_count),
            "mask_path": str(mask_path),
            "missing_dataset_path": str(output_path),
        },
    )
    del df, out_df, original_target, mask_df
    gc.collect()
    return status_row, current_status_df


def run_generate_missing(
    args: argparse.Namespace,
    paths: StagePaths,
    prepare_artifacts: PrepareArtifacts,
    mechanisms: List[str],
    missing_rates: List[float],
    length_config: LengthSamplingConfig,
) -> pd.DataFrame:
    write_design_doc(paths, length_config)
    scenarios = build_scenarios(mechanisms, missing_rates, length_config, args.seed)
    expected_chunk_count = int(len(prepare_artifacts.chunk_summary_df))
    total_observation_count = int(prepare_artifacts.chunk_summary_df["target_non_null_count"].sum())
    cleanup_resume_artifacts(paths, scenarios, expected_chunk_count)
    current_status_df = load_existing_chunk_status(paths)

    chunk_status_rows: List[pd.DataFrame] = []
    scenario_summary_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        event_path = scenario_event_path(paths, scenario)
        if scenario.mechanism == NODE_TEMPORAL_BLOCK:
            ensure_event_file_schema(
                event_path,
                [
                    "mechanism",
                    "missing_rate_target",
                    "event_id",
                    "node_id",
                    "start_global_time_index",
                    "end_global_time_index",
                    "actual_length",
                    "length_group",
                    "length_mode",
                    "seed",
                    "chunk_index",
                    "day_index",
                    "file_name",
                ],
            )
        else:
            ensure_event_file_schema(
                event_path,
                [
                    "mechanism",
                    "missing_rate_target",
                    "event_id",
                    "node_subset_ratio",
                    "selected_node_count",
                    "start_global_time_index",
                    "end_global_time_index",
                    "actual_length",
                    "length_group",
                    "length_mode",
                    "seed",
                    "node_list_file",
                    "chunk_index",
                    "day_index",
                    "file_name",
                ],
            )
        if scenario_is_complete(paths, scenario, expected_chunk_count):
            existing_chunk_status_df = reconstruct_existing_chunk_status(
                args=args,
                paths=paths,
                scenario=scenario,
                prepare_artifacts=prepare_artifacts,
            )
            existing_summary = build_existing_scenario_summary(
                scenario=scenario,
                chunk_status_df=existing_chunk_status_df,
                total_observation_count=total_observation_count,
                tolerance=args.tolerance,
                event_path=event_path,
                paths=paths,
                prepare_artifacts=prepare_artifacts,
            )
            for status_row in existing_chunk_status_df.to_dict(orient="records"):
                current_status_df = upsert_chunk_status_row(current_status_df, status_row)
            persist_chunk_status_df(paths, current_status_df)
            chunk_status_rows.append(existing_chunk_status_df)
            scenario_summary_rows.append(existing_summary)
            pd.DataFrame(scenario_summary_rows).sort_values(["mechanism", "missing_rate_target", "scenario_tag"]).to_csv(
                paths.manifests_dir / "structured_missing_scenario_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            continue

        allocation_df = allocate_chunk_missing_counts(prepare_artifacts.chunk_summary_df, scenario, args.seed)
        allocation_lookup = build_chunk_target_lookup(allocation_df)
        next_id = next_event_id(event_path)
        scenario_chunk_status_rows: List[Dict[str, Any]] = []
        for row in prepare_artifacts.chunk_summary_df.to_dict(orient="records"):
            chunk_index = int(row["chunk_index"])
            day_index = int(row["day_index"])
            file_name = str(row["file_name"])
            allocation_row = allocation_lookup[chunk_index]
            target_missing_count = int(allocation_row["allocated_missing_count"])
            existing_status_row = get_existing_status_row(current_status_df, scenario, chunk_index)
            if can_reuse_existing_chunk(
                paths=paths,
                scenario=scenario,
                chunk_index=chunk_index,
                expected_missing_count=target_missing_count,
                file_name=file_name,
                existing_status_row=existing_status_row,
            ):
                reused_row = dict(existing_status_row)
                if "point_topup_count" not in reused_row:
                    reused_row["point_topup_count"] = 0
                scenario_chunk_status_rows.append(reused_row)
                append_jsonl(
                    chunk_runtime_state_path(paths),
                    {
                        "action": "skip_existing_chunk",
                        "scenario_tag": scenario.scenario_tag,
                        "mechanism": scenario.mechanism,
                        "missing_rate_target": scenario.missing_rate,
                        "chunk_index": chunk_index,
                        "day_index": day_index,
                        "file_name": file_name,
                        "observed_missing_count": int(reused_row["observed_missing_count"]),
                    },
                )
                continue

            df = pd.read_parquet(args.input_dir / file_name, columns=[args.node_col, args.time_col])
            _, unique_times, _ = prepare_chunk_layout(
                df=df,
                canonical_node_ids=prepare_artifacts.node_ids,
                node_col=args.node_col,
                time_col=args.time_col,
                period=args.period,
            )
            del df
            gc.collect()
            if scenario.mechanism == NODE_TEMPORAL_BLOCK:
                coverage, actual_length_matrix, event_rows, point_topup_count = generate_block_chunk_payload(
                    scenario=scenario,
                    node_ids=prepare_artifacts.node_ids,
                    unique_times=unique_times,
                    target_missing_count=target_missing_count,
                    seed=args.seed,
                    chunk_index=chunk_index,
                    day_index=day_index,
                    file_name=file_name,
                    event_id_start=next_id,
                )
            else:
                coverage, actual_length_matrix, event_rows, point_topup_count = generate_outage_chunk_payload(
                    scenario=scenario,
                    node_ids=prepare_artifacts.node_ids,
                    unique_times=unique_times,
                    target_missing_count=target_missing_count,
                    seed=args.seed,
                    chunk_index=chunk_index,
                    day_index=day_index,
                    file_name=file_name,
                    event_id_start=next_id,
                    outage_node_lists_dir=paths.outage_node_lists_dir,
                )
            status_row, current_status_df = write_single_chunk_outputs(
                args=args,
                paths=paths,
                scenario=scenario,
                row=row,
                node_ids=prepare_artifacts.node_ids,
                coverage=coverage,
                actual_length_matrix=actual_length_matrix,
                current_status_df=current_status_df,
                point_topup_count=point_topup_count,
            )
            scenario_chunk_status_rows.append(status_row)
            if event_rows:
                append_csv(pd.DataFrame(event_rows), event_path)
            next_id += len(event_rows)
            del coverage, actual_length_matrix
            gc.collect()

        chunk_status_df = pd.DataFrame(scenario_chunk_status_rows).sort_values(["chunk_index"]).reset_index(drop=True)
        chunk_status_rows.append(chunk_status_df)
        scenario_summary = build_existing_scenario_summary(
            scenario=scenario,
            chunk_status_df=chunk_status_df,
            total_observation_count=total_observation_count,
            tolerance=args.tolerance,
            event_path=event_path,
            paths=paths,
            prepare_artifacts=prepare_artifacts,
        )
        scenario_summary["point_topup_count"] = int(chunk_status_df.get("point_topup_count", pd.Series(dtype=int)).sum())
        scenario_summary_rows.append(scenario_summary)
        pd.DataFrame(scenario_summary_rows).sort_values(["mechanism", "missing_rate_target", "scenario_tag"]).to_csv(
            paths.manifests_dir / "structured_missing_scenario_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        del chunk_status_df
        gc.collect()

    all_chunk_status_df = pd.concat(chunk_status_rows, ignore_index=True)
    persist_chunk_status_df(paths, all_chunk_status_df)
    pd.DataFrame(scenario_summary_rows).sort_values(["mechanism", "missing_rate_target", "scenario_tag"]).to_csv(
        paths.manifests_dir / "structured_missing_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return all_chunk_status_df


def build_audit_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# 结构化缺失设置审计报告",
        "",
        "## 1. 基本信息",
        "",
        f"- input_dir: `{payload['input_dir']}`",
        f"- output_dir: `{payload['output_dir']}`",
        f"- mechanisms: `{', '.join(payload['mechanisms'])}`",
        f"- missing_rates: `{payload['missing_rates']}`",
        f"- length_mode: `{payload['length_mode']}`",
        f"- short_length_range: `{payload['short_length_range']}`",
        f"- mid_length_range: `{payload['mid_length_range']}`",
        f"- long_length_range: `{payload['long_length_range']}`",
        f"- length_group_probs: `{payload['length_group_probs']}`",
        f"- seed: `{payload['seed']}`",
        f"- period: `{payload['period']}`",
        f"- chunk_count: `{payload['chunk_count']}`",
        "",
        "## 2. 机制说明",
        "",
        "- 现有 global MCAR point 目录保留不变，本轮输出全部写入全新目录。",
        "- node_temporal_block 用于模拟单节点连续离线。",
        "- node_subset_temporal_outage 用于模拟部分节点在连续窗口内统一离线。",
        "- 连续缺失长度为事件级随机变量，而不是实验等级。",
        "",
        "## 3. 场景审计",
        "",
    ]
    for summary in payload["scenario_summaries"]:
        lines.extend(
            [
                f"### {summary['scenario_tag']}",
                "",
                f"- mechanism: `{summary['mechanism']}`",
                f"- missing_rate_target: `{summary['missing_rate_target']}`",
                f"- parameter_setting: `{summary['parameter_setting']}`",
                f"- observed_missing_rate: `{summary['observed_missing_rate']}`",
                f"- absolute_error: `{summary['absolute_error']}`",
                f"- is_within_tolerance: `{summary['is_within_tolerance']}`",
                f"- fixed_lengths_only: `{summary['fixed_lengths_only']}`",
                f"- length_is_event_level_random_variable: `{summary['length_is_event_level_random_variable']}`",
                f"- length_min: `{summary['length_min']}`",
                f"- length_max: `{summary['length_max']}`",
                f"- length_mean: `{summary['length_mean']}`",
                f"- length_std: `{summary['length_std']}`",
                f"- short_event_ratio: `{summary['short_event_ratio']}`",
                f"- mid_event_ratio: `{summary['mid_event_ratio']}`",
                f"- long_event_ratio: `{summary['long_event_ratio']}`",
                f"- mask_file_count: `{summary['mask_file_count']}`",
                f"- missing_dataset_file_count: `{summary['missing_dataset_file_count']}`",
                f"- drops_entire_day: `{summary['drops_entire_day']}`",
                f"- drops_all_nodes_at_same_time: `{summary['drops_all_nodes_at_same_time']}`",
                f"- uses_row_index_mask: `{summary['uses_row_index_mask']}`",
                f"- modifies_only_target_col: `{summary['modifies_only_target_col']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    args: argparse.Namespace,
    paths: StagePaths,
    prepare_artifacts: PrepareArtifacts,
    mechanisms: List[str],
    missing_rates: List[float],
    length_config: LengthSamplingConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    chunk_status_path = paths.manifests_dir / "structured_missing_chunk_status.csv"
    scenario_summary_path = paths.manifests_dir / "structured_missing_scenario_summary.csv"
    if not chunk_status_path.exists() or not scenario_summary_path.exists():
        raise FileNotFoundError("generate_missing outputs not found; run --stage generate_missing first")

    chunk_status_df = pd.read_csv(chunk_status_path)
    scenario_summary_df = pd.read_csv(scenario_summary_path)
    chunk_rate_df = chunk_status_df[
        [
            "mechanism",
            "missing_rate_target",
            "scenario_tag",
            "parameter_setting",
            "length_mode",
            "chunk_index",
            "day_index",
            "file_name",
            "row_count",
            "target_non_null_count",
            "observed_missing_count",
            "observed_missing_rate",
            "drops_entire_day",
            "drops_all_nodes_at_same_time",
            "uses_row_index_mask",
            "modifies_only_target_col",
        ]
    ].copy()
    chunk_rate_df.to_csv(
        paths.audits_dir / "structured_missingness_rate_by_chunk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "mechanisms": mechanisms,
        "missing_rates": missing_rates,
        "length_mode": length_config.length_mode,
        "short_length_range": list(length_config.short_length_range),
        "mid_length_range": list(length_config.mid_length_range),
        "long_length_range": list(length_config.long_length_range),
        "length_group_probs": list(length_config.length_group_probs),
        "fixed_lengths_only": False,
        "length_is_event_level_random_variable": True,
        "seed": args.seed,
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "period": args.period,
        "chunk_count": int(len(prepare_artifacts.chunk_summary_df)),
        "uses_row_index_mask": True,
        "drops_entire_day": False,
        "drops_all_nodes_at_same_time": False,
        "modifies_only_target_col": True,
        "scenario_summaries": scenario_summary_df.to_dict(orient="records"),
    }
    if scenario_summary_df["drops_entire_day"].astype(bool).any():
        audit_payload["drops_entire_day"] = True
    if scenario_summary_df["drops_all_nodes_at_same_time"].astype(bool).any():
        audit_payload["drops_all_nodes_at_same_time"] = True
    if not scenario_summary_df["uses_row_index_mask"].astype(bool).all():
        audit_payload["uses_row_index_mask"] = False
    if not scenario_summary_df["modifies_only_target_col"].astype(bool).all():
        audit_payload["modifies_only_target_col"] = False

    write_json(paths.audits_dir / "structured_missingness_audit.json", audit_payload)
    build_audit_markdown(paths.audits_dir / "structured_missingness_audit_zh.md", audit_payload)
    return chunk_rate_df, audit_payload


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.output_dir = ensure_absolute(project_root, args.output_dir)
    paths = build_paths(args.output_dir)
    mkdirs(paths)
    mechanisms, missing_rates, length_config = validate_args(args)
    write_run_artifacts(args, paths, mechanisms, missing_rates, length_config)

    if args.stage in {"prepare", "all"}:
        prepare_artifacts = run_prepare(args, paths)
    else:
        prepare_artifacts = load_prepare_artifacts(paths)

    if args.stage in {"generate_missing", "all"}:
        run_generate_missing(
            args=args,
            paths=paths,
            prepare_artifacts=prepare_artifacts,
            mechanisms=mechanisms,
            missing_rates=missing_rates,
            length_config=length_config,
        )

    if args.stage in {"audit", "all"}:
        run_audit(
            args=args,
            paths=paths,
            prepare_artifacts=prepare_artifacts,
            mechanisms=mechanisms,
            missing_rates=missing_rates,
            length_config=length_config,
        )


if __name__ == "__main__":
    main()
