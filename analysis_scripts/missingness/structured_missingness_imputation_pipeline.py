"""执行结构化缺失场景的补全与效果评估。

核心功能：
- 读取时间块缺失或节点离线缺失场景生成的 miss_set 结果；
- 执行多方法补全，并记录分块运行状态与误差明细；
- 输出按长度组、流量组聚合的质量统计和审计文件。

项目作用：
- 作为结构化缺失实验的核心补全脚本；
- 为方法比较、误差复核和结果可视化提供标准输入。

关键依赖：`numpy`、`pandas`、`scipy`、`matplotlib`。
主要输入：结构化缺失数据、拓扑信息和运行配置。
主要输出：补全结果、质量汇总、审计日志和图形输入。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


EXPECTED_EXPERIMENT_NAME = "real_data_structured_missingness_setting"
SUPPORTED_MECHANISMS = {
    "node_temporal_block": {
        "artifact_prefix": "structured",
        "display_title": "Structured Temporal Block",
        "run_config_name": "run_config_imputation.json",
        "run_commands_name": "run_commands_imputation.txt",
        "input_check_csv_name": "structured_imputation_input_check.csv",
        "input_check_json_name": "structured_imputation_input_check.json",
        "chunk_status_name": "structured_imputed_chunk_status.csv",
        "chunk_state_log_name": "structured_imputed_chunk_runtime_state.jsonl",
        "resume_scan_name": "structured_imputed_resume_scan.csv",
        "detail_summary_name": "structured_imputation_quality_detail.csv",
        "summary_all_days_name": "structured_imputation_quality_summary_all_days.csv",
        "summary_exclude_warmup_name": "structured_imputation_quality_summary_exclude_warmup.csv",
        "summary_by_flow_group_name": "structured_imputation_quality_by_flow_group.csv",
        "summary_by_length_group_name": "structured_imputation_quality_by_length_group.csv",
        "audit_json_name": "structured_causal_imputation_audit.json",
        "audit_md_name": "structured_causal_imputation_audit_zh.md",
    },
    "node_subset_temporal_outage": {
        "artifact_prefix": "outage",
        "display_title": "Structured Node Subset Outage",
        "run_config_name": "run_config_imputation_outage.json",
        "run_commands_name": "run_commands_imputation_outage.txt",
        "input_check_csv_name": "outage_imputation_input_check.csv",
        "input_check_json_name": "outage_imputation_input_check.json",
        "chunk_status_name": "outage_imputed_chunk_status.csv",
        "chunk_state_log_name": "outage_imputed_chunk_runtime_state.jsonl",
        "resume_scan_name": "outage_imputed_resume_scan.csv",
        "detail_summary_name": "outage_imputation_quality_detail.csv",
        "summary_all_days_name": "outage_imputation_quality_summary_all_days.csv",
        "summary_exclude_warmup_name": "outage_imputation_quality_summary_exclude_warmup.csv",
        "summary_by_flow_group_name": "outage_imputation_quality_by_flow_group.csv",
        "summary_by_length_group_name": "outage_imputation_quality_by_length_group.csv",
        "audit_json_name": "outage_causal_imputation_audit.json",
        "audit_md_name": "outage_causal_imputation_audit_zh.md",
    },
}
EXPECTED_SCENARIO_TAG = "mixed_short_mid_long"
METHOD_ALIASES = {
    "geo_neighbor_fill": "road_topology_neighbor_fill",
}
METHOD_ORDER = [
    "mean_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "road_topology_neighbor_fill",
    "function_curve_fit",
    "correlation_topology_neighbor_fill",
]
REMOVED_METHODS = {"zero_fill"}
METHOD_DIR_ABBREVIATIONS = {
    "mean_fill": "mf",
    "forward_fill": "ff",
    "historical_linear_extrapolation": "hle",
    "road_topology_neighbor_fill": "rtn",
    "function_curve_fit": "fcf",
    "correlation_topology_neighbor_fill": "ctn",
}
METHOD_FALLBACK_POLICY = {
    "mean_fill": "same_slot_7day_mean -> node_7day_mean -> slot_7day_mean -> global_7day_mean -> current_day_forward_fill",
    "forward_fill": "use_previous_slot_or_previous_day_last_slot_then_global_safe_fallback_zero",
    "historical_linear_extrapolation": "fallback_to_current_day_forward_fill_when_history_is_insufficient",
    "road_topology_neighbor_fill": "fallback_to_current_day_forward_fill_when_no_topology_history_is_available",
    "function_curve_fit": "fallback_to_current_day_forward_fill_when_no_history_profile_is_available",
    "correlation_topology_neighbor_fill": "same-time positive-correlation topology neighbors -> mean_fill",
}
FLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]
LENGTH_GROUP_LABELS = ["short", "mid", "long"]
EPSILON = 1e-6


@dataclass(frozen=True)
class StagePaths:
    root: Path
    manifests_dir: Path
    audits_dir: Path
    summaries_dir: Path
    figures_dir: Path
    imputed_datasets_dir: Path
    artifact_prefix: str
    mechanism_title: str
    run_config_path: Path
    run_commands_path: Path
    input_check_csv_path: Path
    input_check_json_path: Path
    chunk_status_path: Path
    chunk_state_log_path: Path
    resume_scan_path: Path
    detail_summary_path: Path
    summary_all_days_path: Path
    summary_exclude_warmup_path: Path
    summary_by_flow_group_path: Path
    summary_by_length_group_path: Path
    audit_json_path: Path
    audit_md_path: Path


@dataclass
class DayLayout:
    sort_idx: np.ndarray
    inverse_sort_idx: np.ndarray
    node_ids: np.ndarray


@dataclass
class PreparedChunk:
    chunk_index: int
    day_index: int
    clean_file: Path
    clean_df: pd.DataFrame
    missing_df: pd.DataFrame
    layout: DayLayout
    clean_sorted: np.ndarray
    missing_sorted: np.ndarray
    mask_matrix: np.ndarray
    true_masked_values: np.ndarray
    flow_group_ids: np.ndarray
    length_group_values: np.ndarray
    is_warmup: bool


@dataclass
class RateScanResult:
    missing_rate: float
    resume_chunk_index: int
    valid_prefix_chunk_count: int
    valid_status_rows: list[dict[str, Any]]
    valid_detail_rows: list[dict[str, Any]]
    forward_last_state: np.ndarray | None
    history_observed: deque[np.ndarray]
    history_linear: deque[np.ndarray]
    history_road: deque[np.ndarray]
    history_function: deque[np.ndarray]
    history_clean: deque[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于已有 structured missingness 缺失数据执行严格历史因果补全。")
    parser.add_argument("--stage", required=True, choices=["prepare", "impute", "summarize", "validate", "plot", "all"])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--missingness_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--missing_rates", required=True, type=str)
    parser.add_argument("--mechanism", required=True, type=str)
    parser.add_argument("--scenario_tag", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--impute_methods", required=True, type=str)
    parser.add_argument("--causal_history_only", action="store_true")
    parser.add_argument("--history_days", required=True, type=int)
    parser.add_argument("--context_days_before", required=True, type=int)
    parser.add_argument("--context_days_after", required=True, type=int)
    parser.add_argument("--warmup_days", required=True, type=int)
    parser.add_argument("--exclude_warmup_from_main_metrics", required=True, type=str)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--node_col", required=True, type=str)
    parser.add_argument("--time_col", required=True, type=str)
    parser.add_argument("--period", required=True, type=int)
    parser.add_argument("--topology_file", required=True, type=Path)
    parser.add_argument("--correlation_history_days", default=14, type=int)
    parser.add_argument("--neighbor_scope", default=2, type=int)
    parser.add_argument("--allow_current_time_neighbors", default="true", type=str)
    parser.add_argument("--manifests_subdir", default="manifests", type=str)
    parser.add_argument("--summaries_subdir", default="summaries", type=str)
    parser.add_argument("--audits_subdir", default="audits", type=str)
    parser.add_argument("--figures_subdir", default="figures", type=str)
    parser.add_argument("--run_config_name", default="", type=str)
    parser.add_argument("--run_commands_name", default="", type=str)
    return parser.parse_args()


def parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {raw}")


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def parse_missing_rates(raw: str) -> list[float]:
    rates = []
    for token in raw.split(","):
        value = float(token.strip())
        if not (0 <= value <= 1):
            raise ValueError(f"missing rate out of range: {value}")
        rates.append(value)
    if not rates:
        raise ValueError("missing_rates is empty")
    return rates


def parse_methods(raw: str) -> list[str]:
    parsed: list[str] = []
    for token in raw.split(","):
        normalized = METHOD_ALIASES.get(token.strip(), token.strip())
        if normalized in REMOVED_METHODS:
            raise ValueError(f"unsupported imputation method (removed from formal set): {token}")
        if normalized not in METHOD_ORDER:
            raise ValueError(f"unsupported imputation method: {token}")
        if normalized not in parsed:
            parsed.append(normalized)
    if not parsed:
        raise ValueError("impute_methods is empty")
    return parsed


def format_rate_tag(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def mechanism_settings(mechanism: str) -> dict[str, str]:
    if mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(f"unsupported mechanism: {mechanism}")
    return SUPPORTED_MECHANISMS[mechanism]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), ensure_ascii=False) + "\n")


def build_paths(args: argparse.Namespace) -> StagePaths:
    output_dir = args.output_dir
    mechanism = args.mechanism
    settings = mechanism_settings(mechanism)
    run_config_name = args.run_config_name.strip() or settings["run_config_name"]
    run_commands_name = args.run_commands_name.strip() or settings["run_commands_name"]
    return StagePaths(
        root=output_dir,
        manifests_dir=output_dir / args.manifests_subdir,
        audits_dir=output_dir / args.audits_subdir,
        summaries_dir=output_dir / args.summaries_subdir,
        figures_dir=output_dir / args.figures_subdir,
        imputed_datasets_dir=output_dir / "imp_data",
        artifact_prefix=settings["artifact_prefix"],
        mechanism_title=settings["display_title"],
        run_config_path=output_dir / run_config_name,
        run_commands_path=output_dir / run_commands_name,
        input_check_csv_path=(output_dir / args.manifests_subdir) / settings["input_check_csv_name"],
        input_check_json_path=(output_dir / args.manifests_subdir) / settings["input_check_json_name"],
        chunk_status_path=(output_dir / args.manifests_subdir) / settings["chunk_status_name"],
        chunk_state_log_path=(output_dir / args.manifests_subdir) / settings["chunk_state_log_name"],
        resume_scan_path=(output_dir / args.manifests_subdir) / settings["resume_scan_name"],
        detail_summary_path=(output_dir / args.summaries_subdir) / settings["detail_summary_name"],
        summary_all_days_path=(output_dir / args.summaries_subdir) / settings["summary_all_days_name"],
        summary_exclude_warmup_path=(output_dir / args.summaries_subdir) / settings["summary_exclude_warmup_name"],
        summary_by_flow_group_path=(output_dir / args.summaries_subdir) / settings["summary_by_flow_group_name"],
        summary_by_length_group_path=(output_dir / args.summaries_subdir) / settings["summary_by_length_group_name"],
        audit_json_path=(output_dir / args.audits_subdir) / settings["audit_json_name"],
        audit_md_path=(output_dir / args.audits_subdir) / settings["audit_md_name"],
    )


def mkdirs(paths: StagePaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    paths.audits_dir.mkdir(parents=True, exist_ok=True)
    paths.summaries_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.imputed_datasets_dir.mkdir(parents=True, exist_ok=True)


def write_run_artifacts(args: argparse.Namespace, paths: StagePaths) -> None:
    config = {
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "stage": args.stage,
        "input_dir": str(args.input_dir),
        "missingness_dir": str(args.missingness_dir),
        "output_dir": str(args.output_dir),
        "missing_rates": args.missing_rates_parsed,
        "mechanism": args.mechanism,
        "scenario_tag": args.scenario_tag,
        "seed": args.seed,
        "impute_methods": args.impute_methods_parsed,
        "causal_history_only": args.causal_history_only,
        "history_days": args.history_days,
        "context_days_before": args.context_days_before,
        "context_days_after": args.context_days_after,
        "warmup_days": args.warmup_days,
        "exclude_warmup_from_main_metrics": args.exclude_warmup_from_main_metrics,
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "period": args.period,
        "topology_file": str(args.topology_file),
        "correlation_history_days": args.correlation_history_days,
        "neighbor_scope": args.neighbor_scope,
        "allow_current_time_neighbors": parse_bool(args.allow_current_time_neighbors),
        "manifests_subdir": args.manifests_subdir,
        "summaries_subdir": args.summaries_subdir,
        "audits_subdir": args.audits_subdir,
        "figures_subdir": args.figures_subdir,
        "run_config_name": paths.run_config_path.name,
        "run_commands_name": paths.run_commands_path.name,
    }
    write_json(paths.run_config_path, config)
    command_lines = [
        "Structured missingness imputation pipeline commands",
        f"python {' '.join(sys.argv)}",
    ]
    paths.run_commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(f"this pipeline only supports mechanisms={sorted(SUPPORTED_MECHANISMS)}")
    if args.scenario_tag != EXPECTED_SCENARIO_TAG:
        raise ValueError(f"this pipeline only supports scenario_tag={EXPECTED_SCENARIO_TAG}")
    if not args.causal_history_only:
        raise ValueError("causal_history_only must be true for this pipeline")
    if args.history_days != 7 or args.context_days_before != 7 or args.context_days_after != 0:
        raise ValueError("this pipeline requires history_days=7, context_days_before=7, context_days_after=0")
    if args.warmup_days != 7:
        raise ValueError("this pipeline requires warmup_days=7")
    if not args.exclude_warmup_from_main_metrics:
        raise ValueError("exclude_warmup_from_main_metrics must be true")
    if args.period <= 0:
        raise ValueError("period must be positive")
    if args.neighbor_scope <= 0:
        raise ValueError("neighbor_scope must be positive")
    if args.correlation_history_days <= 0:
        raise ValueError("correlation_history_days must be positive")
    if "correlation_topology_neighbor_fill" in args.impute_methods_parsed and not parse_bool(args.allow_current_time_neighbors):
        raise ValueError("correlation_topology_neighbor_fill requires --allow_current_time_neighbors true")
def list_clean_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("node_flow_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no clean chunk files found in {input_dir}")
    return files


def extract_day_index(file_name: str) -> int:
    stem = file_name.replace(".parquet", "").replace("_mask", "")
    return int(stem.split("_")[-1])


def scenario_dir_name(rate: float, mechanism: str, scenario_tag: str, seed: int) -> str:
    prefix = "ntb" if mechanism == "node_temporal_block" else "nso"
    return f"{prefix}_r{format_rate_tag(rate).replace('0p', '')}_mix_s{seed}"


def missing_subdir(base_dir: Path, rate: float, mechanism: str, scenario_tag: str, seed: int) -> Path:
    return base_dir / "miss_data" / scenario_dir_name(rate, mechanism, scenario_tag, seed)


def mask_subdir(base_dir: Path, rate: float, mechanism: str, scenario_tag: str, seed: int) -> Path:
    return base_dir / "masks" / scenario_dir_name(rate, mechanism, scenario_tag, seed)


def imputed_subdir(base_dir: Path, rate: float, mechanism: str, scenario_tag: str, seed: int, method: str) -> Path:
    return base_dir / "imp_data" / (
        f"{scenario_dir_name(rate, mechanism, scenario_tag, seed)}_m_{METHOD_DIR_ABBREVIATIONS[method]}"
    )


def infer_length_group_from_length(actual_length: int) -> str:
    if 1 <= actual_length <= 4:
        return "short"
    if 5 <= actual_length <= 12:
        return "mid"
    if 13 <= actual_length <= 24:
        return "long"
    raise ValueError(f"cannot infer length_group from actual_length={actual_length}")


def normalize_length_group_series(mask_df: pd.DataFrame) -> pd.Series:
    if "length_group" in mask_df.columns:
        series = mask_df["length_group"].astype(str).str.strip().str.lower()
        invalid = ~series.isin(LENGTH_GROUP_LABELS)
        if invalid.any():
            raise RuntimeError(f"mask file has invalid length_group values: {sorted(series.loc[invalid].unique().tolist())}")
        return series
    if "actual_length" not in mask_df.columns:
        raise RuntimeError("mask file does not include length_group or actual_length")
    return mask_df["actual_length"].astype(int).map(infer_length_group_from_length)


def prepare_reference_layout(args: argparse.Namespace, clean_files: list[Path]) -> tuple[np.ndarray, int]:
    ref_df = pd.read_parquet(clean_files[0], columns=[args.node_col, args.time_col])
    layout = build_day_layout(
        node_values=ref_df[args.node_col].to_numpy(np.int64, copy=False),
        time_values=ref_df[args.time_col].to_numpy(np.int64, copy=False),
        period=args.period,
        ref_node_ids=None,
    )
    del ref_df
    gc.collect()
    return layout.node_ids, len(layout.node_ids)


def build_day_layout(
    node_values: np.ndarray,
    time_values: np.ndarray,
    period: int,
    ref_node_ids: np.ndarray | None,
) -> DayLayout:
    sort_idx = np.lexsort((time_values, node_values))
    sorted_nodes = node_values[sort_idx]
    sorted_times = time_values[sort_idx]
    unique_nodes, counts = np.unique(sorted_nodes, return_counts=True)
    if not np.all(counts == period):
        raise RuntimeError("each node must have exactly one observation for every time slot in each chunk")
    unique_times = np.unique(time_values)
    if len(unique_times) != period:
        raise RuntimeError(f"expected {period} unique time slots per chunk, got {len(unique_times)}")
    time_matrix = sorted_times.reshape(len(unique_nodes), period)
    if not np.array_equal(time_matrix, np.broadcast_to(unique_times, time_matrix.shape)):
        raise RuntimeError("time slots per node are not consistent within the chunk after lexsort")
    if ref_node_ids is not None and not np.array_equal(unique_nodes, ref_node_ids):
        raise RuntimeError("node set/order mismatch across chunks after sorting by node/time")
    inverse_sort_idx = np.empty_like(sort_idx)
    inverse_sort_idx[sort_idx] = np.arange(len(sort_idx), dtype=np.int64)
    return DayLayout(sort_idx=sort_idx, inverse_sort_idx=inverse_sort_idx, node_ids=unique_nodes)


def compute_flow_group_thresholds(
    clean_files: list[Path],
    target_col: str,
    seed: int,
    output_path: Path,
    sample_per_chunk: int = 50000,
) -> dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for file_path in clean_files:
        series = pd.read_parquet(file_path, columns=[target_col])[target_col].to_numpy(dtype=np.float32, copy=False)
        sample_size = min(sample_per_chunk, len(series))
        sample_idx = rng.choice(len(series), size=sample_size, replace=False)
        samples.append(np.asarray(series[sample_idx], dtype=np.float32))
    sample_values = np.concatenate(samples)
    q33, q66 = np.quantile(sample_values, [1 / 3, 2 / 3])
    payload = {
        "estimation_method": "sampled_global_clean_quantiles",
        "sample_per_chunk": sample_per_chunk,
        "sample_count_total": int(len(sample_values)),
        "quantiles": {
            "q33": float(q33),
            "q66": float(q66),
        },
        "groups": {
            "low_flow": {"min_inclusive": 0.0, "max_exclusive": float(q33)},
            "mid_flow": {"min_inclusive": float(q33), "max_exclusive": float(q66)},
            "high_flow": {"min_inclusive": float(q66), "max_inclusive": float(np.max(sample_values))},
        },
    }
    write_json(output_path, payload)
    return payload


def run_prepare(args: argparse.Namespace, paths: StagePaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean_files = list_clean_files(args.input_dir)
    chunk_count = len(clean_files)
    rows: list[dict[str, Any]] = []
    event_manifest_path = paths.manifests_dir / f"{args.mechanism}_events.csv"

    for rate in args.missing_rates_parsed:
        mask_dir = mask_subdir(args.missingness_dir, rate, args.mechanism, args.scenario_tag, args.seed)
        missing_dir = missing_subdir(args.missingness_dir, rate, args.mechanism, args.scenario_tag, args.seed)
        mask_files = sorted(mask_dir.glob("node_flow_chunk_*_mask.parquet"))
        missing_files = sorted(missing_dir.glob("node_flow_chunk_*.parquet"))
        sample_mask_columns: list[str] = []
        uses_row_index_mask = False
        has_length_group = False
        has_actual_length = False
        has_node_subset_event = True
        if mask_files:
            sample_mask_df = pd.read_parquet(mask_files[0])
            sample_mask_columns = list(sample_mask_df.columns)
            uses_row_index_mask = "row_index" in sample_mask_columns
            has_length_group = "length_group" in sample_mask_columns
            has_actual_length = "actual_length" in sample_mask_columns
            normalize_length_group_series(sample_mask_df)
            if args.mechanism == "node_subset_temporal_outage":
                has_node_subset_event = bool(event_manifest_path.exists()) or has_length_group or has_actual_length
        elif args.mechanism == "node_subset_temporal_outage":
            has_node_subset_event = bool(event_manifest_path.exists())

        rows.append(
            {
                "mechanism": args.mechanism,
                "missing_rate": rate,
                "mask_dir": str(mask_dir),
                "mask_chunk_count": len(mask_files),
                "missing_dataset_dir": str(missing_dir),
                "missing_dataset_chunk_count": len(missing_files),
                "is_complete": bool(len(mask_files) == chunk_count and len(missing_files) == chunk_count),
                "uses_row_index_mask": bool(uses_row_index_mask),
                "has_length_group": bool(has_length_group),
                "has_actual_length": bool(has_actual_length),
                "has_node_subset_event": bool(has_node_subset_event),
                "chunk_count_expected": int(chunk_count),
                "scenario_tag": args.scenario_tag,
                "sample_mask_columns": json.dumps(sample_mask_columns, ensure_ascii=False),
            }
        )

    check_df = pd.DataFrame(rows).sort_values("missing_rate").reset_index(drop=True)
    check_df.to_csv(paths.input_check_csv_path, index=False, encoding="utf-8-sig")
    check_json = {
        "input_dir": str(args.input_dir),
        "missingness_dir": str(args.missingness_dir),
        "mechanism": args.mechanism,
        "scenario_tag": args.scenario_tag,
        "missing_rates": args.missing_rates_parsed,
        "chunk_count_expected": chunk_count,
        "node_subset_event_manifest": str(event_manifest_path),
        "checks": check_df.to_dict(orient="records"),
        "notes_zh": [
            f"本轮只检查并补全 {args.mechanism} / {args.scenario_tag}。",
            "如果任何缺失率不是 61/61，则不应进入 impute 阶段。",
        ],
    }
    write_json(paths.input_check_json_path, check_json)
    incomplete_df = check_df.loc[
        (~check_df["is_complete"])
        | (~check_df["uses_row_index_mask"])
        | (~(check_df["has_length_group"] | check_df["has_actual_length"]))
        | (
            (args.mechanism == "node_subset_temporal_outage")
            & (~check_df["has_node_subset_event"])
        )
    ]
    if not incomplete_df.empty:
        raise RuntimeError(
            f"structured {args.mechanism} input check failed for one or more missing rates; "
            f"see {paths.input_check_csv_path.name}/{paths.input_check_json_path.name}"
        )
    return check_df, check_json


def copy_history(history: deque[np.ndarray], maxlen: int) -> deque[np.ndarray]:
    return deque((value.copy() for value in history), maxlen=maxlen)


def now_utc_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def prepare_chunk(
    *,
    args: argparse.Namespace,
    clean_file: Path,
    missing_dir: Path,
    mask_dir: Path,
    ref_node_ids: np.ndarray,
    node_count: int,
    thresholds: dict[str, Any],
) -> PreparedChunk:
    chunk_index = extract_day_index(clean_file.name)
    clean_df = pd.read_parquet(clean_file)
    missing_df = pd.read_parquet(missing_dir / clean_file.name)
    mask_df = pd.read_parquet(mask_dir / clean_file.name.replace(".parquet", "_mask.parquet"))

    clean_node = clean_df[args.node_col].to_numpy(np.int64, copy=False)
    clean_time = clean_df[args.time_col].to_numpy(np.int64, copy=False)
    layout = build_day_layout(clean_node, clean_time, args.period, ref_node_ids)

    clean_target = clean_df[args.target_col].to_numpy(dtype=np.float32, copy=False)
    missing_target = missing_df[args.target_col].to_numpy(dtype=np.float32, copy=False)

    clean_sorted = clean_target[layout.sort_idx].reshape(node_count, args.period)
    missing_sorted = missing_target[layout.sort_idx].reshape(node_count, args.period)

    mask_positions = layout.inverse_sort_idx[mask_df["row_index"].to_numpy(np.int64, copy=False)]
    mask_matrix = np.zeros(node_count * args.period, dtype=bool)
    mask_matrix[mask_positions] = True
    mask_matrix = mask_matrix.reshape(node_count, args.period)

    if not np.all(np.isnan(missing_sorted[mask_matrix])):
        raise RuntimeError(f"{clean_file.name} missing dataset does not have NaN on all mask positions")
    if not np.allclose(clean_sorted[~mask_matrix], missing_sorted[~mask_matrix], equal_nan=True):
        raise RuntimeError(f"{clean_file.name} non-mask positions changed in missing dataset")

    mask_length_groups = normalize_length_group_series(mask_df)
    length_group_flat = np.full(node_count * args.period, "", dtype="<U8")
    length_group_flat[mask_positions] = mask_length_groups.to_numpy(dtype=str, copy=False)
    length_group_values = length_group_flat.reshape(node_count, args.period)[mask_matrix]
    true_masked_values = clean_sorted[mask_matrix]
    flow_group_ids = build_group_ids(true_masked_values, thresholds)
    return PreparedChunk(
        chunk_index=chunk_index,
        day_index=chunk_index,
        clean_file=clean_file,
        clean_df=clean_df,
        missing_df=missing_df,
        layout=layout,
        clean_sorted=clean_sorted,
        missing_sorted=missing_sorted,
        mask_matrix=mask_matrix,
        true_masked_values=true_masked_values,
        flow_group_ids=flow_group_ids,
        length_group_values=length_group_values,
        is_warmup=chunk_index < args.warmup_days,
    )


def compute_method_outputs(
    *,
    prepared: PreparedChunk,
    forward_last_state: np.ndarray | None,
    history_observed: deque[np.ndarray],
    history_linear: deque[np.ndarray],
    history_road: deque[np.ndarray],
    history_function: deque[np.ndarray],
    history_clean: deque[np.ndarray],
    weight_matrix: sparse.csr_matrix,
    row_sums: np.ndarray,
    topology_candidates: dict[int, dict[str, np.ndarray]],
    correlation_history_days: int,
    basis: np.ndarray,
    pseudo_inverse: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    forward_imputed, forward_fallback = impute_forward_fill(
        prepared.missing_sorted,
        prepared.mask_matrix,
        forward_last_state,
    )
    mean_imputed, mean_fallback = impute_mean_fill(
        prepared.missing_sorted,
        prepared.mask_matrix,
        history_observed,
        forward_imputed,
    )
    linear_imputed, linear_fallback = impute_historical_linear(
        prepared.missing_sorted,
        prepared.mask_matrix,
        prepared.day_index,
        history_linear,
        forward_imputed,
    )
    road_imputed, road_fallback, road_primary_pred, road_primary_available = impute_road_topology_neighbor_fill(
        prepared.missing_sorted,
        prepared.mask_matrix,
        history_road,
        weight_matrix,
        row_sums,
        forward_imputed,
    )
    function_imputed, function_fallback, function_primary_pred, function_primary_available = impute_function_curve_fit(
        prepared.missing_sorted,
        prepared.mask_matrix,
        history_function,
        basis,
        pseudo_inverse,
        forward_imputed,
    )
    masked_node_indices, masked_slot_indices = np.where(prepared.mask_matrix)
    corr_stack = historical_stack(history_clean, last_n=correlation_history_days)
    ctn_imputed, ctn_fallback = spatial_current_prediction(
        missing_matrix=prepared.missing_sorted,
        mask_matrix=prepared.mask_matrix,
        mean_fill=mean_imputed,
        node_indices=masked_node_indices.astype(np.int64, copy=False),
        slot_indices=masked_slot_indices.astype(np.int64, copy=False),
        topology_candidates=topology_candidates,
        history_stack_corr=corr_stack,
        correlation_mode=True,
    )
    return {
        "mean_fill": (mean_imputed, mean_fallback),
        "forward_fill": (forward_imputed, forward_fallback),
        "historical_linear_extrapolation": (linear_imputed, linear_fallback),
        "road_topology_neighbor_fill": (road_imputed, road_fallback),
        "function_curve_fit": (function_imputed, function_fallback),
        "correlation_topology_neighbor_fill": (ctn_imputed, ctn_fallback),
    }


def update_histories_from_outputs(
    *,
    observed_matrix: np.ndarray,
    method_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    history_observed: deque[np.ndarray],
    history_linear: deque[np.ndarray],
    history_road: deque[np.ndarray],
    history_function: deque[np.ndarray],
) -> np.ndarray:
    forward_imputed = method_outputs.get("forward_fill", (observed_matrix, None))[0]
    linear_imputed = method_outputs.get("historical_linear_extrapolation", (observed_matrix, None))[0]
    road_imputed = method_outputs.get("road_topology_neighbor_fill", (observed_matrix, None))[0]
    function_imputed = method_outputs.get("function_curve_fit", (observed_matrix, None))[0]
    history_observed.append(observed_matrix.copy())
    history_linear.append(linear_imputed.copy())
    history_road.append(road_imputed.copy())
    history_function.append(function_imputed.copy())
    return forward_imputed[:, -1].copy()


def build_output_flat(prepared: PreparedChunk, imputed_sorted: np.ndarray) -> np.ndarray:
    out_flat = np.empty(len(prepared.missing_df), dtype=np.float32)
    out_flat[prepared.layout.sort_idx] = imputed_sorted.reshape(-1)
    return out_flat


def build_status_row(
    *,
    missing_rate: float,
    method: str,
    prepared: PreparedChunk,
    output_path: Path,
    fallback_matrix: np.ndarray,
    file_size_bytes: int | None,
    output_mtime_utc: str | None,
    write_duration_seconds: float | None,
    source: str,
) -> dict[str, Any]:
    return {
        "missing_rate": missing_rate,
        "method": method,
        "fallback_policy": METHOD_FALLBACK_POLICY[method],
        "chunk_index": prepared.chunk_index,
        "day_index": prepared.day_index,
        "file_name": prepared.clean_file.name,
        "output_path": str(output_path),
        "missing_count": int(prepared.mask_matrix.sum()),
        "fallback_count": int(np.count_nonzero(fallback_matrix[prepared.mask_matrix])),
        "non_mask_positions_preserved": True,
        "validation_passed": True,
        "status": "completed",
        "source": source,
        "write_duration_seconds": write_duration_seconds,
        "file_size_bytes": file_size_bytes,
        "output_mtime_utc": output_mtime_utc,
    }


def persist_status_snapshot(status_rows: list[dict[str, Any]], paths: StagePaths) -> None:
    if not status_rows:
        return
    status_df = (
        pd.DataFrame(status_rows)
        .sort_values(["missing_rate", "method", "chunk_index"])
        .reset_index(drop=True)
    )
    status_df.to_csv(paths.chunk_status_path, index=False, encoding="utf-8-sig")


def persist_detail_snapshot(detail_rows: list[dict[str, Any]], paths: StagePaths) -> None:
    if not detail_rows:
        return
    detail_df = (
        pd.DataFrame(detail_rows)
        .sort_values(["missing_rate", "method", "chunk_index", "group_dimension", "flow_group", "length_group"])
        .reset_index(drop=True)
    )
    detail_df.to_csv(paths.detail_summary_path, index=False, encoding="utf-8-sig")


def load_output_matrix_from_file(args: argparse.Namespace, prepared: PreparedChunk, output_path: Path) -> np.ndarray:
    output_df = pd.read_parquet(output_path, columns=[args.target_col])
    output_target = output_df[args.target_col].to_numpy(dtype=np.float32, copy=False)
    return output_target[prepared.layout.sort_idx].reshape(prepared.clean_sorted.shape)


def load_existing_progress(paths: StagePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paths.chunk_status_path.exists():
        status_df = pd.read_csv(paths.chunk_status_path)
    else:
        status_df = pd.DataFrame()
    if paths.detail_summary_path.exists():
        detail_df = pd.read_csv(paths.detail_summary_path)
    else:
        detail_df = pd.DataFrame()
    return status_df, detail_df


def expected_detail_rows_per_chunk(method_count: int) -> int:
    return method_count * (1 + len(FLOW_GROUP_LABELS) + len(LENGTH_GROUP_LABELS))


def has_complete_chunk_records(
    *,
    status_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    rate: float,
    chunk_index: int,
    methods: list[str],
) -> bool:
    required_status_cols = {"missing_rate", "chunk_index", "method"}
    required_detail_cols = {"missing_rate", "chunk_index", "method", "group_dimension", "flow_group", "length_group"}
    if status_df.empty or not required_status_cols.issubset(status_df.columns):
        return False
    if detail_df.empty or not required_detail_cols.issubset(detail_df.columns):
        return False

    status_subset = status_df.loc[
        np.isclose(status_df["missing_rate"], rate) & (status_df["chunk_index"] == chunk_index)
    ].copy()
    if len(status_subset) != len(methods):
        return False
    if set(status_subset["method"].tolist()) != set(methods):
        return False

    detail_subset = detail_df.loc[
        np.isclose(detail_df["missing_rate"], rate) & (detail_df["chunk_index"] == chunk_index)
    ].copy()
    if len(detail_subset) != expected_detail_rows_per_chunk(len(methods)):
        return False
    expected_pairs = {
        (method, "overall", "overall", "overall") for method in methods
    } | {
        (method, "flow_group", group, "overall") for method in methods for group in FLOW_GROUP_LABELS
    } | {
        (method, "length_group", "overall", group) for method in methods for group in LENGTH_GROUP_LABELS
    }
    actual_pairs = set(
        zip(
            detail_subset["method"].tolist(),
            detail_subset["group_dimension"].tolist(),
            detail_subset["flow_group"].tolist(),
            detail_subset["length_group"].tolist(),
        )
    )
    return actual_pairs == expected_pairs


def build_topology_matrix(topology_file: Path, ref_node_ids: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    topo_df = pd.read_csv(
        topology_file,
        usecols=["起始节点ID", "结束节点ID", "长度"],
    )
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(ref_node_ids.tolist())}
    accumulator: dict[tuple[int, int], float] = {}

    for start, end, length in topo_df.itertuples(index=False, name=None):
        start_idx = node_to_idx.get(int(start))
        end_idx = node_to_idx.get(int(end))
        if start_idx is None or end_idx is None or start_idx == end_idx:
            continue
        safe_length = float(length) if pd.notna(length) and float(length) > 0 else EPSILON
        weight = 1.0 / (safe_length + EPSILON)
        accumulator[(start_idx, end_idx)] = accumulator.get((start_idx, end_idx), 0.0) + weight
        accumulator[(end_idx, start_idx)] = accumulator.get((end_idx, start_idx), 0.0) + weight

    if not accumulator:
        raise RuntimeError("no valid topology edges overlap with node_intersection_flow nodes")

    rows = np.fromiter((key[0] for key in accumulator.keys()), dtype=np.int32)
    cols = np.fromiter((key[1] for key in accumulator.keys()), dtype=np.int32)
    data = np.fromiter(accumulator.values(), dtype=np.float32)
    weight_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(ref_node_ids), len(ref_node_ids)), dtype=np.float32)
    row_sums = np.asarray(weight_matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    return weight_matrix, row_sums


def build_topology_candidates(topology_file: Path, ref_node_ids: np.ndarray, max_scope: int) -> dict[int, dict[str, np.ndarray]]:
    topo_df = pd.read_csv(topology_file, usecols=["起始节点ID", "结束节点ID", "长度"])
    node_to_idx = {int(node_id): idx for idx, node_id in enumerate(ref_node_ids.tolist())}
    first_hop_sets: list[set[int]] = [set() for _ in range(len(ref_node_ids))]
    first_hop_lengths: list[dict[int, float]] = [dict() for _ in range(len(ref_node_ids))]
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
    second_hop_lengths: list[dict[int, float]] = [dict() for _ in range(len(ref_node_ids))]
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
    for node_idx in range(len(ref_node_ids)):
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


def historical_stack(history: deque[np.ndarray], last_n: int | None = None) -> np.ndarray | None:
    if not history:
        return None
    values = list(history)[-last_n:] if last_n is not None else list(history)
    return np.stack(values, axis=0).astype(np.float32, copy=False)


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
) -> tuple[np.ndarray, np.ndarray]:
    imputed = mean_fill.copy()
    fallback = mask_matrix.copy()
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
        if correlation_mode:
            positive = available & (cache["corr"][:, None] > 0.05)
            base_weights = np.where(
                cache["corr"] > 0.05,
                np.maximum(cache["corr"], 0.0) / np.power(cache["lengths"] + EPSILON, 1.0),
                0.0,
            ).astype(np.float32)
            weight_matrix = base_weights[:, None] * positive.astype(np.float32)
            z_values = np.where(
                positive,
                (current_values - cache["mu_j"][:, None]) / (cache["sigma_j"][:, None] + EPSILON),
                0.0,
            )
            numerator = np.sum(weight_matrix * z_values, axis=0, dtype=np.float32)
            denominator = np.sum(weight_matrix, axis=0, dtype=np.float32)
            success = denominator > EPSILON
            predictions = np.full(len(slots), np.nan, dtype=np.float32)
            predictions[success] = cache["mu_i"] + (
                max(cache["sigma_i"], EPSILON) * (numerator[success] / denominator[success])
            )
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
    return imputed.astype(np.float32, copy=False), fallback


def compute_global_median(history: deque[np.ndarray]) -> float:
    if not history:
        return 0.0
    medians = [float(np.median(day_matrix)) for day_matrix in history]
    return float(np.median(np.asarray(medians, dtype=np.float32)))


def impute_mean_fill(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    history_observed: deque[np.ndarray],
    forward_fill_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(mask_matrix, dtype=bool)
    stack = historical_stack(history_observed)
    if stack is None or stack.shape[0] == 0:
        imputed[mask_matrix] = forward_fill_matrix[mask_matrix]
        fallback_mask[mask_matrix] = True
        return imputed, fallback_mask

    available = np.isfinite(stack)
    values = np.where(available, stack, 0.0).astype(np.float32, copy=False)

    same_slot_count = available.sum(axis=0)
    same_slot_sum = values.sum(axis=0, dtype=np.float32)
    same_slot_mean = np.full_like(missing_matrix, np.nan, dtype=np.float32)
    np.divide(same_slot_sum, same_slot_count, out=same_slot_mean, where=same_slot_count > 0)
    use_primary = mask_matrix & (same_slot_count > 0)
    imputed[use_primary] = same_slot_mean[use_primary]

    remaining = mask_matrix & ~use_primary
    if not np.any(remaining):
        return imputed, fallback_mask

    node_count = available.sum(axis=(0, 2))
    node_sum = values.sum(axis=(0, 2), dtype=np.float32)
    node_mean = np.full(missing_matrix.shape[0], np.nan, dtype=np.float32)
    np.divide(node_sum, node_count, out=node_mean, where=node_count > 0)
    node_available = np.broadcast_to((node_count > 0)[:, None], missing_matrix.shape)
    node_mean_matrix = np.broadcast_to(node_mean[:, None], missing_matrix.shape)
    use_node = remaining & node_available
    imputed[use_node] = node_mean_matrix[use_node]

    remaining = remaining & ~use_node
    if not np.any(remaining):
        fallback_mask[mask_matrix & ~use_primary] = True
        return imputed, fallback_mask

    slot_count = available.sum(axis=(0, 1))
    slot_sum = values.sum(axis=(0, 1), dtype=np.float32)
    slot_mean = np.full(missing_matrix.shape[1], np.nan, dtype=np.float32)
    np.divide(slot_sum, slot_count, out=slot_mean, where=slot_count > 0)
    slot_available = np.broadcast_to((slot_count > 0)[None, :], missing_matrix.shape)
    slot_mean_matrix = np.broadcast_to(slot_mean[None, :], missing_matrix.shape)
    use_slot = remaining & slot_available
    imputed[use_slot] = slot_mean_matrix[use_slot]

    remaining = remaining & ~use_slot
    if not np.any(remaining):
        fallback_mask[mask_matrix & ~use_primary] = True
        return imputed, fallback_mask

    global_count = int(available.sum())
    if global_count > 0:
        global_mean = float(values.sum(dtype=np.float32) / float(global_count))
        imputed[remaining] = global_mean
        fallback_mask[mask_matrix & ~use_primary] = True
        return imputed, fallback_mask

    imputed[remaining] = forward_fill_matrix[remaining]
    fallback_mask[mask_matrix & ~use_primary] = True
    return imputed, fallback_mask


def impute_forward_fill(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    previous_last_state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(mask_matrix, dtype=bool)

    if previous_last_state is not None:
        first_missing = np.isnan(imputed[:, 0])
        imputed[first_missing, 0] = previous_last_state[first_missing]

    for slot in range(1, imputed.shape[1]):
        current_missing = np.isnan(imputed[:, slot])
        if np.any(current_missing):
            imputed[current_missing, slot] = imputed[current_missing, slot - 1]

    unresolved = np.isnan(imputed)
    if np.any(unresolved):
        imputed[unresolved] = 0.0
        fallback_mask[unresolved & mask_matrix] = True

    return imputed, fallback_mask


def impute_historical_linear(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    current_day_index: int,
    history: deque[np.ndarray],
    forward_fill_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(mask_matrix, dtype=bool)
    stack = historical_stack(history)
    if stack is None or stack.shape[0] < 2:
        imputed[mask_matrix] = forward_fill_matrix[mask_matrix]
        fallback_mask[mask_matrix] = True
        return imputed, fallback_mask

    history_count = stack.shape[0]
    x = np.arange(current_day_index - history_count, current_day_index, dtype=np.float32)
    n = float(history_count)
    sx = float(np.sum(x))
    sxx = float(np.sum(x * x))
    sy = np.sum(stack, axis=0, dtype=np.float32)
    sxy = np.tensordot(x, stack, axes=(0, 0)).astype(np.float32)
    denominator = (n * sxx) - (sx * sx)
    if abs(denominator) <= EPSILON:
        imputed[mask_matrix] = forward_fill_matrix[mask_matrix]
        fallback_mask[mask_matrix] = True
        return imputed, fallback_mask

    slope = ((n * sxy) - (sx * sy)) / denominator
    intercept = (sy - (slope * sx)) / n
    predicted = (slope * float(current_day_index)) + intercept
    imputed[mask_matrix] = predicted[mask_matrix]
    remaining_nan = np.isnan(imputed) & mask_matrix
    if np.any(remaining_nan):
        imputed[remaining_nan] = forward_fill_matrix[remaining_nan]
        fallback_mask[remaining_nan] = True
    return imputed, fallback_mask


def topology_primary_prediction(
    history: deque[np.ndarray],
    weight_matrix: sparse.csr_matrix,
    row_sums: np.ndarray,
    history_days: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    stack = historical_stack(history)
    if stack is None or stack.shape[0] == 0:
        return None, None
    betas = 1.0 / np.arange(1, stack.shape[0] + 1, dtype=np.float32)
    numerator = np.zeros_like(stack[0], dtype=np.float32)
    beta_sum = 0.0
    for lag_idx, day_matrix in enumerate(stack[::-1], start=1):
        beta = 1.0 / float(lag_idx)
        numerator += beta * weight_matrix.dot(day_matrix)
        beta_sum += beta
    denominator = (row_sums[:, None] * beta_sum).astype(np.float32, copy=False)
    available = np.broadcast_to(denominator > 0, numerator.shape)
    predicted = np.full_like(stack[0], np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=predicted, where=denominator > 0)
    return predicted, available


def impute_road_topology_neighbor_fill(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    history: deque[np.ndarray],
    weight_matrix: sparse.csr_matrix,
    row_sums: np.ndarray,
    forward_fill_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(mask_matrix, dtype=bool)
    primary_pred, primary_available = topology_primary_prediction(history, weight_matrix, row_sums, history_days=7)
    if primary_pred is None or primary_available is None:
        imputed[mask_matrix] = forward_fill_matrix[mask_matrix]
        fallback_mask[mask_matrix] = True
        return imputed, fallback_mask, None, None

    use_primary = mask_matrix & primary_available
    use_fallback = mask_matrix & ~primary_available
    imputed[use_primary] = primary_pred[use_primary]
    imputed[use_fallback] = forward_fill_matrix[use_fallback]
    fallback_mask[use_fallback] = True
    return imputed, fallback_mask, primary_pred, primary_available


def build_fourier_basis(period: int, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    time_index = np.arange(period, dtype=np.float32)
    basis_columns = [np.ones(period, dtype=np.float32)]
    for harmonic in range(1, order + 1):
        angle = (2.0 * math.pi * harmonic * time_index) / float(period)
        basis_columns.append(np.cos(angle).astype(np.float32))
        basis_columns.append(np.sin(angle).astype(np.float32))
    basis = np.column_stack(basis_columns).astype(np.float32)
    pseudo_inverse = np.linalg.pinv(basis).astype(np.float32)
    return basis, pseudo_inverse


def function_primary_prediction(
    history: deque[np.ndarray],
    basis: np.ndarray,
    pseudo_inverse: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    stack = historical_stack(history)
    if stack is None or stack.shape[0] == 0:
        return None, None
    mean_profile = np.mean(stack, axis=0, dtype=np.float32)
    coefficients = mean_profile @ pseudo_inverse.T
    predicted = coefficients @ basis.T
    available = np.ones_like(predicted, dtype=bool)
    return predicted.astype(np.float32), available


def impute_function_curve_fit(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    history: deque[np.ndarray],
    basis: np.ndarray,
    pseudo_inverse: np.ndarray,
    forward_fill_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(mask_matrix, dtype=bool)
    primary_pred, primary_available = function_primary_prediction(history, basis, pseudo_inverse)
    if primary_pred is None or primary_available is None:
        imputed[mask_matrix] = forward_fill_matrix[mask_matrix]
        fallback_mask[mask_matrix] = True
        return imputed, fallback_mask, None, None
    imputed[mask_matrix] = primary_pred[mask_matrix]
    remaining_nan = np.isnan(imputed) & mask_matrix
    if np.any(remaining_nan):
        imputed[remaining_nan] = forward_fill_matrix[remaining_nan]
        fallback_mask[remaining_nan] = True
    return imputed, fallback_mask, primary_pred, primary_available


def build_group_ids(true_values: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    q33 = float(thresholds["quantiles"]["q33"])
    q66 = float(thresholds["quantiles"]["q66"])
    group_ids = np.full(len(true_values), 2, dtype=np.int8)
    group_ids[true_values < q66] = 1
    group_ids[true_values < q33] = 0
    return group_ids


def metric_row(
    *,
    missing_rate: float,
    method: str,
    chunk_index: int,
    day_index: int,
    group_dimension: str,
    flow_group: str,
    length_group: str,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    fallback_flags: np.ndarray,
    is_warmup: bool,
) -> dict[str, Any]:
    finite_mask = np.isfinite(true_values) & np.isfinite(pred_values)
    valid_true = true_values[finite_mask]
    valid_pred = pred_values[finite_mask]
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
    missing_count = int(len(true_values))
    valid_eval_count = int(len(valid_true))
    fallback_count = int(np.count_nonzero(fallback_flags))

    if valid_eval_count == 0:
        mae = rmse = mape = smape = nrmse = math.nan
        true_min = true_max = math.nan
    else:
        mae = float(abs_errors.mean())
        rmse = float(np.sqrt(sq_errors.mean()))
        mape = float(ape_values.mean()) if len(ape_values) > 0 else math.nan
        smape = float(smape_values.mean()) if len(smape_values) > 0 else math.nan
        true_min = float(valid_true.min())
        true_max = float(valid_true.max())
        data_range = true_max - true_min
        nrmse = float(rmse / data_range) if data_range > EPSILON else math.nan

    return {
        "missing_rate": missing_rate,
        "method": method,
        "chunk_index": chunk_index,
        "day_index": day_index,
        "group_dimension": group_dimension,
        "flow_group": flow_group,
        "length_group": length_group,
        "is_warmup_day": bool(is_warmup),
        "missing_count": missing_count,
        "valid_eval_count": valid_eval_count,
        "fallback_count": fallback_count,
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
    chunk_index: int,
    day_index: int,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    fallback_flags: np.ndarray,
    flow_group_ids: np.ndarray,
    length_group_values: np.ndarray,
    is_warmup: bool,
) -> list[dict[str, Any]]:
    rows = [
        metric_row(
            missing_rate=missing_rate,
            method=method,
            chunk_index=chunk_index,
            day_index=day_index,
            group_dimension="overall",
            flow_group="overall",
            length_group="overall",
            true_values=true_values,
            pred_values=pred_values,
            fallback_flags=fallback_flags,
            is_warmup=is_warmup,
        )
    ]
    for group_id, group_label in enumerate(FLOW_GROUP_LABELS):
        selector = flow_group_ids == group_id
        rows.append(
            metric_row(
                missing_rate=missing_rate,
                method=method,
                chunk_index=chunk_index,
                day_index=day_index,
                group_dimension="flow_group",
                flow_group=group_label,
                length_group="overall",
                true_values=true_values[selector],
                pred_values=pred_values[selector],
                fallback_flags=fallback_flags[selector],
                is_warmup=is_warmup,
            )
        )
    for group_label in LENGTH_GROUP_LABELS:
        selector = length_group_values == group_label
        rows.append(
            metric_row(
                missing_rate=missing_rate,
                method=method,
                chunk_index=chunk_index,
                day_index=day_index,
                group_dimension="length_group",
                flow_group="overall",
                length_group=group_label,
                true_values=true_values[selector],
                pred_values=pred_values[selector],
                fallback_flags=fallback_flags[selector],
                is_warmup=is_warmup,
            )
        )
    return rows


def summarize_aggregates(filtered: pd.DataFrame, group_cols: list[str], exclude_warmup: bool) -> pd.DataFrame:
    agg_df = (
        filtered.groupby(group_cols, dropna=False)
        .agg(
            chunk_count=("chunk_index", "count"),
            missing_count=("missing_count", "sum"),
            valid_eval_count=("valid_eval_count", "sum"),
            fallback_count=("fallback_count", "sum"),
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

    if len(agg_df) == 0:
        return agg_df

    agg_df["mae"] = agg_df["abs_error_sum"] / agg_df["valid_eval_count"].replace(0, np.nan)
    agg_df["rmse"] = np.sqrt(agg_df["sq_error_sum"] / agg_df["valid_eval_count"].replace(0, np.nan))
    agg_df["mape"] = agg_df["ape_sum"] / agg_df["ape_count"].replace(0, np.nan)
    agg_df["smape"] = agg_df["smape_sum"] / agg_df["smape_count"].replace(0, np.nan)
    value_range = agg_df["true_max"] - agg_df["true_min"]
    agg_df["nrmse"] = agg_df["rmse"] / value_range.replace(0, np.nan)
    agg_df["exclude_warmup"] = exclude_warmup
    return agg_df.sort_values(group_cols).reset_index(drop=True)


def summary_from_detail(detail_df: pd.DataFrame, exclude_warmup: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = detail_df.loc[~detail_df["is_warmup_day"]] if exclude_warmup else detail_df
    overall_df = summarize_aggregates(
        filtered.loc[filtered["group_dimension"] == "overall"].copy(),
        ["missing_rate", "method"],
        exclude_warmup,
    )
    flow_df = summarize_aggregates(
        filtered.loc[filtered["group_dimension"] == "flow_group"].copy(),
        ["missing_rate", "method", "flow_group"],
        exclude_warmup,
    )
    length_df = summarize_aggregates(
        filtered.loc[filtered["group_dimension"] == "length_group"].copy(),
        ["missing_rate", "method", "length_group"],
        exclude_warmup,
    )
    return overall_df, flow_df, length_df


def plot_metric_by_rate(summary_df: pd.DataFrame, metric: str, output_png: Path, output_pdf: Path, title: str) -> None:
    overall_df = summary_df.copy()
    if overall_df.empty:
        raise RuntimeError(f"cannot plot {metric}: overall summary is empty")
    methods = [method for method in METHOD_ORDER if method in set(overall_df["method"].astype(str))]
    plt.figure(figsize=(10, 6))
    for method in methods:
        method_df = overall_df.loc[overall_df["method"] == method].sort_values("missing_rate")
        plt.plot(method_df["missing_rate"], method_df[metric], marker="o", linewidth=2, label=method)
    plt.xlabel("Missing Rate")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.xticks(argsort_unique(overall_df["missing_rate"].to_numpy(dtype=float)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.savefig(output_pdf)
    plt.close()


def plot_metric_by_length_group(length_df: pd.DataFrame, metric: str, output_png: Path, output_pdf: Path, title: str) -> None:
    if length_df.empty:
        raise RuntimeError(f"cannot plot {metric}: length-group summary is empty")
    methods = [method for method in METHOD_ORDER if method in set(length_df["method"].astype(str))]
    fig, axes = plt.subplots(1, len(LENGTH_GROUP_LABELS), figsize=(18, 5), sharey=True)
    for axis, group_label in zip(axes, LENGTH_GROUP_LABELS):
        group_df = length_df.loc[length_df["length_group"] == group_label]
        for method in methods:
            method_df = group_df.loc[group_df["method"] == method].sort_values("missing_rate")
            axis.plot(method_df["missing_rate"], method_df[metric], marker="o", linewidth=1.8, label=method)
        axis.set_title(group_label)
        axis.set_xlabel("Missing Rate")
        axis.set_xticks(argsort_unique(group_df["missing_rate"].to_numpy(dtype=float)))
        axis.grid(alpha=0.3)
    axes[0].set_ylabel(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_rmse_by_length_group_and_method(length_df: pd.DataFrame, output_png: Path, output_pdf: Path, title: str) -> None:
    if length_df.empty:
        raise RuntimeError("cannot plot RMSE by length_group and method because grouped summary is empty")
    methods = [method for method in METHOD_ORDER if method in set(length_df["method"].astype(str))]
    x_positions = np.arange(len(LENGTH_GROUP_LABELS), dtype=np.float32)
    width = 0.12
    plt.figure(figsize=(11, 6))
    for method_idx, method in enumerate(methods):
        subset = (
            length_df.loc[length_df["method"] == method]
            .groupby("length_group", dropna=False)["rmse"]
            .mean()
            .reindex(LENGTH_GROUP_LABELS)
        )
        offsets = x_positions + (method_idx - (len(methods) - 1) / 2.0) * width
        plt.bar(offsets, subset.to_numpy(dtype=float), width=width, label=method)
    plt.xticks(x_positions, LENGTH_GROUP_LABELS)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.savefig(output_pdf)
    plt.close()


def argsort_unique(values: np.ndarray) -> list[float]:
    return sorted(set(float(value) for value in values.tolist()))


def scan_existing_outputs_for_rate(
    *,
    args: argparse.Namespace,
    paths: StagePaths,
    rate: float,
    clean_files: list[Path],
    ref_node_ids: np.ndarray,
    node_count: int,
    thresholds: dict[str, Any],
    status_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> RateScanResult:
    missing_dir = missing_subdir(args.missingness_dir, rate, args.mechanism, args.scenario_tag, args.seed)
    mask_dir = mask_subdir(args.missingness_dir, rate, args.mechanism, args.scenario_tag, args.seed)

    valid_status_rows: list[dict[str, Any]] = []
    valid_detail_rows: list[dict[str, Any]] = []
    history_observed: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_linear: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_road: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_function: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_clean: deque[np.ndarray] = deque(maxlen=max(args.history_days, args.correlation_history_days))
    forward_last_state: np.ndarray | None = None

    resume_chunk_index = len(clean_files)
    valid_prefix_chunk_count = len(clean_files)

    for clean_file in clean_files:
        chunk_index = extract_day_index(clean_file.name)
        if not has_complete_chunk_records(
            status_df=status_df,
            detail_df=detail_df,
            rate=rate,
            chunk_index=chunk_index,
            methods=args.impute_methods_parsed,
        ):
            resume_chunk_index = chunk_index
            valid_prefix_chunk_count = chunk_index
            break

        prepared = prepare_chunk(
            args=args,
            clean_file=clean_file,
            missing_dir=missing_dir,
            mask_dir=mask_dir,
            ref_node_ids=ref_node_ids,
            node_count=node_count,
            thresholds=thresholds,
        )
        per_method_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        missing_existing_file = False
        for method in args.impute_methods_parsed:
            output_path = imputed_subdir(paths.root, rate, args.mechanism, args.scenario_tag, args.seed, method) / clean_file.name
            if not output_path.exists():
                missing_existing_file = True
                break
            imputed_sorted = load_output_matrix_from_file(args, prepared, output_path)
            fallback_matrix = np.zeros_like(prepared.mask_matrix, dtype=bool)
            per_method_outputs[method] = (imputed_sorted, fallback_matrix)

        if missing_existing_file:
            resume_chunk_index = chunk_index
            valid_prefix_chunk_count = chunk_index
            break

        valid_status_rows.extend(
            status_df.loc[
                np.isclose(status_df["missing_rate"], rate) & (status_df["chunk_index"] == chunk_index)
            ].to_dict(orient="records")
        )
        valid_detail_rows.extend(
            detail_df.loc[
                np.isclose(detail_df["missing_rate"], rate) & (detail_df["chunk_index"] == chunk_index)
            ].to_dict(orient="records")
        )
        forward_last_state = update_histories_from_outputs(
            observed_matrix=prepared.missing_sorted,
            method_outputs=per_method_outputs,
            history_observed=history_observed,
            history_linear=history_linear,
            history_road=history_road,
            history_function=history_function,
        )
        history_clean.append(prepared.clean_sorted.copy())

    return RateScanResult(
        missing_rate=rate,
        resume_chunk_index=resume_chunk_index,
        valid_prefix_chunk_count=valid_prefix_chunk_count,
        valid_status_rows=valid_status_rows,
        valid_detail_rows=valid_detail_rows,
        forward_last_state=None if forward_last_state is None else forward_last_state.copy(),
        history_observed=copy_history(history_observed, args.history_days),
        history_linear=copy_history(history_linear, args.history_days),
        history_road=copy_history(history_road, args.history_days),
        history_function=copy_history(history_function, args.history_days),
        history_clean=copy_history(history_clean, max(args.history_days, args.correlation_history_days)),
    )


def run_impute(
    args: argparse.Namespace,
    paths: StagePaths,
    input_check_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if input_check_df is None:
        input_check_df = pd.read_csv(paths.input_check_csv_path)

    clean_files = list_clean_files(args.input_dir)
    ref_node_ids, node_count = prepare_reference_layout(args, clean_files)
    thresholds = compute_flow_group_thresholds(
        clean_files=clean_files,
        target_col=args.target_col,
        seed=args.seed,
        output_path=paths.manifests_dir / "structured_flow_group_thresholds.json",
    )
    weight_matrix, row_sums = build_topology_matrix(args.topology_file, ref_node_ids)
    topology_candidates = build_topology_candidates(args.topology_file, ref_node_ids, args.neighbor_scope)
    basis, pseudo_inverse = build_fourier_basis(args.period, order=3)
    existing_status_df, existing_detail_df = load_existing_progress(paths)

    detail_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    scan_summaries: list[dict[str, Any]] = []

    if paths.chunk_state_log_path.exists():
        append_jsonl(
            paths.chunk_state_log_path,
            {
                "event_type": "session_restart",
                "event_time_utc": now_utc_iso(),
                "message": "new imputation session started; resume point will be derived from saved status snapshot",
            },
        )
    else:
        append_jsonl(
            paths.chunk_state_log_path,
            {
                "event_type": "session_start",
                "event_time_utc": now_utc_iso(),
                "message": "imputation session started; resume point will be derived from saved status snapshot",
            },
        )

    for rate in args.missing_rates_parsed:
        scan_result = scan_existing_outputs_for_rate(
            args=args,
            paths=paths,
            rate=rate,
            clean_files=clean_files,
            ref_node_ids=ref_node_ids,
            node_count=node_count,
            thresholds=thresholds,
            status_df=existing_status_df,
            detail_df=existing_detail_df,
        )
        detail_rows.extend(scan_result.valid_detail_rows)
        status_rows.extend(scan_result.valid_status_rows)
        scan_summaries.append(
            {
                "missing_rate": rate,
                "valid_prefix_chunk_count": scan_result.valid_prefix_chunk_count,
                "resume_chunk_index": scan_result.resume_chunk_index,
            }
        )

        history_observed = copy_history(scan_result.history_observed, args.history_days)
        history_linear = copy_history(scan_result.history_linear, args.history_days)
        history_road = copy_history(scan_result.history_road, args.history_days)
        history_function = copy_history(scan_result.history_function, args.history_days)
        history_clean = copy_history(scan_result.history_clean, max(args.history_days, args.correlation_history_days))
        forward_last_state = None if scan_result.forward_last_state is None else scan_result.forward_last_state.copy()

        missing_dir = missing_subdir(args.missingness_dir, rate, args.mechanism, args.scenario_tag, args.seed)
        mask_dir = mask_subdir(args.missingness_dir, rate, args.mechanism, args.scenario_tag, args.seed)
        resume_start = scan_result.resume_chunk_index
        if resume_start >= len(clean_files):
            print(f"[resume-scan] rate={rate:.2f} already complete; skipped all {len(clean_files)} chunks")
            continue
        print(
            f"[resume-scan] rate={rate:.2f} valid_prefix={scan_result.valid_prefix_chunk_count}/{len(clean_files)}; "
            f"resume from chunk={resume_start}"
        )

        for clean_file in clean_files[resume_start:]:
            prepared = prepare_chunk(
                args=args,
                clean_file=clean_file,
                missing_dir=missing_dir,
                mask_dir=mask_dir,
                ref_node_ids=ref_node_ids,
                node_count=node_count,
                thresholds=thresholds,
            )
            method_outputs = compute_method_outputs(
                prepared=prepared,
                forward_last_state=forward_last_state,
                history_observed=history_observed,
                history_linear=history_linear,
                history_road=history_road,
                history_function=history_function,
                history_clean=history_clean,
                weight_matrix=weight_matrix,
                row_sums=row_sums,
                topology_candidates=topology_candidates,
                correlation_history_days=args.correlation_history_days,
                basis=basis,
                pseudo_inverse=pseudo_inverse,
            )

            chunk_started_at = time.perf_counter()
            chunk_event_methods: list[dict[str, Any]] = []
            for method in args.impute_methods_parsed:
                imputed_sorted, fallback_matrix = method_outputs[method]
                if not np.allclose(
                    imputed_sorted[~prepared.mask_matrix],
                    prepared.missing_sorted[~prepared.mask_matrix],
                    equal_nan=True,
                ):
                    raise RuntimeError(
                        f"{clean_file.name} rate={rate:.2f} method={method} modified non-mask positions"
                    )
                if np.any(np.isnan(imputed_sorted[prepared.mask_matrix])):
                    raise RuntimeError(f"{clean_file.name} rate={rate:.2f} method={method} still has NaN on mask positions")

                output_dir = imputed_subdir(paths.root, rate, args.mechanism, args.scenario_tag, args.seed, method)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / clean_file.name
                out_flat = build_output_flat(prepared, imputed_sorted)
                out_df = prepared.missing_df.copy()
                out_df[args.target_col] = out_flat

                started_at = time.perf_counter()
                out_df.to_parquet(output_path, index=False)
                write_duration_seconds = float(time.perf_counter() - started_at)
                stat = output_path.stat()
                output_mtime_utc = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()

                predicted_masked_values = imputed_sorted[prepared.mask_matrix]
                fallback_flags = fallback_matrix[prepared.mask_matrix]
                detail_rows.extend(
                    build_detail_rows_for_method(
                        missing_rate=rate,
                        method=method,
                        chunk_index=prepared.chunk_index,
                        day_index=prepared.day_index,
                        true_values=prepared.true_masked_values,
                        pred_values=predicted_masked_values,
                        fallback_flags=fallback_flags,
                        flow_group_ids=prepared.flow_group_ids,
                        length_group_values=prepared.length_group_values,
                        is_warmup=prepared.is_warmup,
                    )
                )
                status_row = build_status_row(
                    missing_rate=rate,
                    method=method,
                    prepared=prepared,
                    output_path=output_path,
                    fallback_matrix=fallback_matrix,
                    file_size_bytes=int(stat.st_size),
                    output_mtime_utc=output_mtime_utc,
                    write_duration_seconds=write_duration_seconds,
                    source="recomputed",
                )
                status_rows.append(status_row)
                chunk_event_methods.append(
                    {
                        "method": method,
                        "output_path": str(output_path),
                        "fallback_count": status_row["fallback_count"],
                        "file_size_bytes": status_row["file_size_bytes"],
                        "write_duration_seconds": write_duration_seconds,
                    }
                )

                del out_df, out_flat
                gc.collect()

            forward_last_state = update_histories_from_outputs(
                observed_matrix=prepared.missing_sorted,
                method_outputs=method_outputs,
                history_observed=history_observed,
                history_linear=history_linear,
                history_road=history_road,
                history_function=history_function,
            )
            history_clean.append(prepared.clean_sorted.copy())
            persist_status_snapshot(status_rows, paths)
            persist_detail_snapshot(detail_rows, paths)
            append_jsonl(
                paths.chunk_state_log_path,
                {
                    "event_type": "chunk_completed",
                    "event_time_utc": now_utc_iso(),
                    "missing_rate": rate,
                    "chunk_index": prepared.chunk_index,
                    "day_index": prepared.day_index,
                    "file_name": prepared.clean_file.name,
                    "duration_seconds": float(time.perf_counter() - chunk_started_at),
                    "missing_count": int(prepared.mask_matrix.sum()),
                    "methods": chunk_event_methods,
                    "validation": {
                        "mask_positions_filled": True,
                        "non_mask_positions_preserved": True,
                    },
                },
            )
            gc.collect()

    detail_df = (
        pd.DataFrame(detail_rows)
        .sort_values(["missing_rate", "method", "chunk_index", "group_dimension", "flow_group", "length_group"])
        .reset_index(drop=True)
    )
    status_df = (
        pd.DataFrame(status_rows)
        .sort_values(["missing_rate", "method", "chunk_index"])
        .reset_index(drop=True)
    )
    detail_df.to_csv(paths.detail_summary_path, index=False, encoding="utf-8-sig")
    status_df.to_csv(paths.chunk_status_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(scan_summaries).sort_values(["missing_rate"]).to_csv(
        paths.resume_scan_path,
        index=False,
        encoding="utf-8-sig",
    )
    return detail_df, status_df


def run_summarize(
    paths: StagePaths,
    detail_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if detail_df is None:
        if not paths.detail_summary_path.exists():
            raise FileNotFoundError(f"{paths.detail_summary_path.name} not found; run --stage impute first")
        detail_df = pd.read_csv(paths.detail_summary_path)

    summary_all_days, _, _ = summary_from_detail(detail_df, exclude_warmup=False)
    summary_exclude_warmup, summary_by_flow_group, summary_by_length_group = summary_from_detail(
        detail_df, exclude_warmup=True
    )

    summary_all_days.to_csv(paths.summary_all_days_path, index=False, encoding="utf-8-sig")
    summary_exclude_warmup.to_csv(paths.summary_exclude_warmup_path, index=False, encoding="utf-8-sig")
    summary_by_flow_group.to_csv(paths.summary_by_flow_group_path, index=False, encoding="utf-8-sig")
    summary_by_length_group.to_csv(paths.summary_by_length_group_path, index=False, encoding="utf-8-sig")
    return summary_all_days, summary_exclude_warmup, summary_by_flow_group, summary_by_length_group


def run_validate(args: argparse.Namespace, paths: StagePaths) -> dict[str, Any]:
    status_path = paths.chunk_status_path
    summary_path = paths.summary_exclude_warmup_path
    length_summary_path = paths.summary_by_length_group_path
    if not status_path.exists():
        raise FileNotFoundError(f"{status_path.name} not found; run --stage impute first")
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path.name} not found; run --stage summarize first")
    if not length_summary_path.exists():
        raise FileNotFoundError(f"{length_summary_path.name} not found; run --stage summarize first")

    status_df = pd.read_csv(status_path)
    summary_df = pd.read_csv(summary_path)
    length_summary_df = pd.read_csv(length_summary_path)
    clean_files = list_clean_files(args.input_dir)
    chunk_count = len(clean_files)

    completeness_rows = []
    for rate in args.missing_rates_parsed:
        for method in args.impute_methods_parsed:
            subset = status_df.loc[
                np.isclose(status_df["missing_rate"], rate) & (status_df["method"] == method)
            ].copy()
            completeness_rows.append(
                {
                    "missing_rate": rate,
                    "method": method,
                    "imputed_chunk_count": int(len(subset)),
                    "expected_chunk_count": chunk_count,
                    "is_complete": bool(len(subset) == chunk_count),
                    "non_mask_positions_preserved": bool(subset["non_mask_positions_preserved"].all()) if len(subset) else False,
                }
            )
            if len(subset) != chunk_count:
                raise RuntimeError(f"rate={rate:.2f} method={method} imputed chunk count mismatch")
            if not bool(subset["non_mask_positions_preserved"].all()):
                raise RuntimeError(f"rate={rate:.2f} method={method} changed non-mask positions")

    required_metrics = {"mae", "rmse", "smape", "nrmse"}
    if not required_metrics.issubset(summary_df.columns):
        raise RuntimeError("summary file is missing required metrics")
    if not required_metrics.issubset(length_summary_df.columns):
        raise RuntimeError("length-group summary file is missing required metrics")
    if set(length_summary_df["length_group"].astype(str).unique().tolist()) != set(LENGTH_GROUP_LABELS):
        raise RuntimeError("length-group summary does not cover short/mid/long completely")

    audit_payload = {
        "input_missingness_dir": str(args.missingness_dir),
        "output_imputation_dir": str(paths.imputed_datasets_dir),
        "mechanism": args.mechanism,
        "scenario_tag": args.scenario_tag,
        "missing_rates": args.missing_rates_parsed,
        "methods": args.impute_methods_parsed,
        "added_methods": ["mean_fill"],
        "removed_methods": ["zero_fill"],
        "causal_history_only": True,
        "context_days_before": args.context_days_before,
        "history_days": args.history_days,
        "context_days_after": args.context_days_after,
        "uses_future_days": False,
        "uses_same_day_future_slots": False,
        "uses_bfill": False,
        "uses_bidirectional_interpolation": False,
        "warmup_days": args.warmup_days,
        "main_metrics_exclude_warmup": args.exclude_warmup_from_main_metrics,
        "fallback_policy": {
            method: METHOD_FALLBACK_POLICY[method]
            for method in args.impute_methods_parsed
            if method in METHOD_FALLBACK_POLICY
        },
        "non_mask_positions_preserved": True,
        "evaluation_only_on_mask_positions": True,
        "length_group_metrics_enabled": True,
        "completeness": completeness_rows,
        "output_files": {
            "chunk_status": str(status_path),
            "chunk_runtime_state_log": str(paths.chunk_state_log_path),
            "resume_scan": str(paths.resume_scan_path),
            "input_check_csv": str(paths.input_check_csv_path),
            "input_check_json": str(paths.input_check_json_path),
            "detail_summary": str(paths.detail_summary_path),
            "summary_all_days": str(paths.summary_all_days_path),
            "summary_exclude_warmup": str(paths.summary_exclude_warmup_path),
            "summary_by_flow_group": str(paths.summary_by_flow_group_path),
            "summary_by_length_group": str(paths.summary_by_length_group_path),
        },
        "resume_mode": "status_snapshot_only",
    }
    write_json(paths.audit_json_path, audit_payload)
    write_audit_markdown(paths.audit_md_path, audit_payload)
    return audit_payload


def write_audit_markdown(path: Path, audit_payload: dict[str, Any]) -> None:
    mechanism = str(audit_payload["mechanism"])
    handled_label = mechanism
    ignored_label = "node_temporal_block" if mechanism == "node_subset_temporal_outage" else "node_subset_temporal_outage"
    lines = [
        "# 结构化连续缺失补全因果审计报告",
        "",
        "## 1. 范围",
        "",
        f"- 本轮只基于已有 `{handled_label}` 的 `masks` 与 `missing_datasets` 执行补全。",
        f"- 本轮未处理 `{ignored_label}`。",
        "- 未重新生成缺失设置。",
        "- 未修改原始 `input_dir`。",
        "- 本轮结果只代表缺失值补全误差，不代表交通预测误差。",
        "",
        "## 2. 严格历史因果约束",
        "",
        f"- mechanism: `{audit_payload['mechanism']}`",
        f"- scenario_tag: `{audit_payload['scenario_tag']}`",
        f"- causal_history_only: `{audit_payload['causal_history_only']}`",
        f"- context_days_before: `{audit_payload['context_days_before']}`",
        f"- history_days: `{audit_payload['history_days']}`",
        f"- context_days_after: `{audit_payload['context_days_after']}`",
        f"- uses_future_days: `{audit_payload['uses_future_days']}`",
        f"- uses_same_day_future_slots: `{audit_payload['uses_same_day_future_slots']}`",
        f"- uses_bfill: `{audit_payload['uses_bfill']}`",
        f"- uses_bidirectional_interpolation: `{audit_payload['uses_bidirectional_interpolation']}`",
        f"- warmup_days: `{audit_payload['warmup_days']}`",
        f"- main_metrics_exclude_warmup: `{audit_payload['main_metrics_exclude_warmup']}`",
        "",
        "## 3. 评价口径",
        "",
        f"- evaluation_only_on_mask_positions: `{audit_payload['evaluation_only_on_mask_positions']}`",
        f"- non_mask_positions_preserved: `{audit_payload['non_mask_positions_preserved']}`",
        f"- length_group_metrics_enabled: `{audit_payload['length_group_metrics_enabled']}`",
        "- road_topology_neighbor_fill 使用的是 `rnsd_processed.csv` 中的路网拓扑邻接与道路长度权重，不是经纬度距离近邻。",
        "",
        "## 4. 方法变更",
        "",
        f"- added_methods: `{audit_payload['added_methods']}`",
        f"- removed_methods: `{audit_payload['removed_methods']}`",
        "",
        "## 5. Fallback 策略",
        "",
    ]
    for method, description in audit_payload["fallback_policy"].items():
        lines.append(f"- {method}: `{description}`")
    lines.extend(["", "## 6. 输出完整性", ""])
    for row in audit_payload["completeness"]:
        lines.append(
            f"- rate={row['missing_rate']:.2f}, method={row['method']}, imputed_chunk_count={row['imputed_chunk_count']}, expected={row['expected_chunk_count']}, is_complete={row['is_complete']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_plot(paths: StagePaths) -> dict[str, Any]:
    summary_path = paths.summary_exclude_warmup_path
    length_summary_path = paths.summary_by_length_group_path
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path.name} not found; run --stage summarize first")
    if not length_summary_path.exists():
        raise FileNotFoundError(f"{length_summary_path.name} not found; run --stage summarize first")
    summary_df = pd.read_csv(summary_path)
    length_df = pd.read_csv(length_summary_path)

    plot_metric_by_rate(
        summary_df,
        metric="rmse",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_multirate_rmse_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_multirate_rmse_by_method.pdf",
        title=f"{paths.mechanism_title} RMSE by Method",
    )
    plot_metric_by_rate(
        summary_df,
        metric="mae",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_multirate_mae_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_multirate_mae_by_method.pdf",
        title=f"{paths.mechanism_title} MAE by Method",
    )
    plot_metric_by_rate(
        summary_df,
        metric="smape",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_multirate_smape_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_multirate_smape_by_method.pdf",
        title=f"{paths.mechanism_title} sMAPE by Method",
    )
    plot_metric_by_rate(
        summary_df,
        metric="nrmse",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_multirate_nrmse_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_multirate_nrmse_by_method.pdf",
        title=f"{paths.mechanism_title} NRMSE by Method",
    )
    for obsolete_path in [
        paths.figures_dir / f"{paths.artifact_prefix}_rmse_by_method_nonzero_zoom.png",
        paths.figures_dir / f"{paths.artifact_prefix}_rmse_by_method_nonzero_zoom.pdf",
    ]:
        if obsolete_path.exists():
            obsolete_path.unlink()
    plot_metric_by_length_group(
        length_df,
        metric="rmse",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_length_group_rmse_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_length_group_rmse_by_method.pdf",
        title=f"{paths.mechanism_title} RMSE by Length Group and Method",
    )
    plot_metric_by_length_group(
        length_df,
        metric="mae",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_length_group_mae_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_length_group_mae_by_method.pdf",
        title=f"{paths.mechanism_title} MAE by Length Group and Method",
    )
    plot_metric_by_length_group(
        length_df,
        metric="smape",
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_length_group_smape_by_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_length_group_smape_by_method.pdf",
        title=f"{paths.mechanism_title} sMAPE by Length Group and Method",
    )
    plot_rmse_by_length_group_and_method(
        length_df,
        output_png=paths.figures_dir / f"{paths.artifact_prefix}_rmse_by_length_group_and_method.png",
        output_pdf=paths.figures_dir / f"{paths.artifact_prefix}_rmse_by_length_group_and_method.pdf",
        title=f"{paths.mechanism_title} RMSE by Length Group and Method",
    )
    for obsolete_path in [
        paths.figures_dir / f"{paths.artifact_prefix}_length_group_rmse_nonzero_zoom.png",
        paths.figures_dir / f"{paths.artifact_prefix}_length_group_rmse_nonzero_zoom.pdf",
    ]:
        if obsolete_path.exists():
            obsolete_path.unlink()
    return {"nonzero_zoom_created": False, "length_group_nonzero_zoom_created": False}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.missingness_dir = ensure_absolute(project_root, args.missingness_dir)
    args.output_dir = ensure_absolute(project_root, args.output_dir)
    args.topology_file = ensure_absolute(project_root, args.topology_file)
    args.missing_rates_parsed = [rate for rate in parse_missing_rates(args.missing_rates) if rate > 0]
    args.impute_methods_parsed = parse_methods(args.impute_methods)
    args.exclude_warmup_from_main_metrics = parse_bool(args.exclude_warmup_from_main_metrics)
    validate_args(args)

    paths = build_paths(args)
    mkdirs(paths)
    if args.stage in {"prepare", "impute", "all"}:
        write_run_artifacts(args, paths)

    input_check_df: pd.DataFrame | None = None
    detail_df: pd.DataFrame | None = None

    if args.stage in {"prepare", "all"}:
        input_check_df, _ = run_prepare(args, paths)
    if args.stage in {"impute", "all"}:
        if input_check_df is None:
            input_check_df = pd.read_csv(paths.input_check_csv_path)
        detail_df, _ = run_impute(args, paths, input_check_df)
    if args.stage in {"summarize", "all"}:
        run_summarize(paths, detail_df)
    if args.stage in {"validate", "all"}:
        run_validate(args, paths)
    if args.stage in {"plot", "all"}:
        run_plot(paths)


if __name__ == "__main__":
    main()
