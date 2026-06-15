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


EXPECTED_EXPERIMENT_NAME = "real_data_global_missingness_setting"
EXPECTED_MISSINGNESS_TYPE = "global_mcar_point"
EXPECTED_MASK_SCOPE = "global"
EXPECTED_MISSING_UNIT = "node_time_observation"
METHOD_ALIASES = {
    "geo_neighbor_fill": "road_topology_neighbor_fill",
    "geo_func_hybrid": "topology_function_hybrid",
}
METHOD_ORDER = [
    "zero_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "road_topology_neighbor_fill",
    "function_curve_fit",
    "topology_function_hybrid",
]
FLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]
EPSILON = 1e-6


@dataclass(frozen=True)
class StagePaths:
    root: Path
    manifests_dir: Path
    audits_dir: Path
    summaries_dir: Path
    figures_dir: Path
    imputed_datasets_dir: Path
    run_config_path: Path
    run_commands_path: Path
    chunk_status_path: Path
    chunk_state_log_path: Path
    resume_scan_path: Path


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
    group_ids: np.ndarray
    is_warmup: bool


@dataclass
class RateScanResult:
    missing_rate: float
    resume_chunk_index: int
    valid_prefix_chunk_count: int
    valid_status_rows: list[dict[str, Any]]
    valid_detail_rows: list[dict[str, Any]]
    forward_last_state: np.ndarray | None
    history_linear: deque[np.ndarray]
    history_road: deque[np.ndarray]
    history_function: deque[np.ndarray]
    history_hybrid: deque[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于已有 global MCAR 缺失数据执行严格历史因果补全。")
    parser.add_argument("--stage", required=True, choices=["prepare", "impute", "summarize", "validate", "plot", "all"])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--missingness_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--missing_rates", required=True, type=str)
    parser.add_argument("--mechanism", required=True, type=str)
    parser.add_argument("--mask_scope", required=True, type=str)
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
        if normalized not in METHOD_ORDER:
            raise ValueError(f"unsupported imputation method: {token}")
        if normalized not in parsed:
            parsed.append(normalized)
    if parsed != METHOD_ORDER:
        missing_methods = [method for method in METHOD_ORDER if method not in parsed]
        if missing_methods:
            raise ValueError(f"required methods missing from --impute_methods: {missing_methods}")
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), ensure_ascii=False) + "\n")


def build_paths(output_dir: Path) -> StagePaths:
    return StagePaths(
        root=output_dir,
        manifests_dir=output_dir / "manifests",
        audits_dir=output_dir / "audits",
        summaries_dir=output_dir / "summaries",
        figures_dir=output_dir / "figures",
        imputed_datasets_dir=output_dir / "imputed_datasets",
        run_config_path=output_dir / "run_config_imputation.json",
        run_commands_path=output_dir / "run_commands_imputation.txt",
        chunk_status_path=output_dir / "manifests" / "imputed_chunk_status.csv",
        chunk_state_log_path=output_dir / "manifests" / "imputed_chunk_runtime_state.jsonl",
        resume_scan_path=output_dir / "manifests" / "imputed_resume_scan.csv",
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
        "stage": "imputation_only",
        "input_dir": str(args.input_dir),
        "missingness_dir": str(args.missingness_dir),
        "output_dir": str(args.output_dir),
        "missing_rates": args.missing_rates_parsed,
        "mechanism": args.mechanism,
        "mask_scope": args.mask_scope,
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
    }
    write_json(paths.run_config_path, config)
    command_lines = [
        "Global missingness imputation pipeline commands",
        f"python {' '.join(sys.argv)}",
    ]
    paths.run_commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.mechanism != "mcar_point":
        raise ValueError("this pipeline only supports mechanism=mcar_point")
    if args.mask_scope != EXPECTED_MASK_SCOPE:
        raise ValueError("this pipeline only supports mask_scope=global")
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


def list_clean_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("node_flow_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no clean chunk files found in {input_dir}")
    return files


def extract_day_index(file_name: str) -> int:
    stem = file_name.replace(".parquet", "").replace("_mask", "")
    return int(stem.split("_")[-1])


def missing_subdir(base_dir: Path, rate: float, mechanism: str, seed: int) -> Path:
    return base_dir / "missing_datasets" / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"


def mask_subdir(base_dir: Path, rate: float, mechanism: str, seed: int) -> Path:
    return base_dir / "masks" / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"


def imputed_subdir(base_dir: Path, rate: float, mechanism: str, seed: int, method: str) -> Path:
    return base_dir / "imputed_datasets" / (
        f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}__method_{method}"
    )


def load_missingness_audit(missingness_dir: Path) -> dict[str, Any]:
    audit_path = missingness_dir / "audits" / "global_missingness_setting_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"missing required audit: {audit_path}")
    return json.loads(audit_path.read_text(encoding="utf-8"))


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
    audit_payload = load_missingness_audit(args.missingness_dir)
    if audit_payload["mask_scope"] != EXPECTED_MASK_SCOPE:
        raise RuntimeError("missingness input must use mask_scope=global")
    if audit_payload["missing_unit"] != EXPECTED_MISSING_UNIT:
        raise RuntimeError("missingness input must use node_time_observation")
    if not bool(audit_payload["mask_uses_row_index"]):
        raise RuntimeError("missingness masks must store row_index")
    if audit_payload["missingness_type"] != EXPECTED_MISSINGNESS_TYPE:
        raise RuntimeError("missingness input must be global_mcar_point")

    clean_files = list_clean_files(args.input_dir)
    chunk_count = len(clean_files)
    rows: list[dict[str, Any]] = []
    sample_mask_file: Path | None = None
    sample_mask_columns: list[str] | None = None

    for rate in args.missing_rates_parsed:
        mask_dir = mask_subdir(args.missingness_dir, rate, args.mechanism, args.seed)
        missing_dir = missing_subdir(args.missingness_dir, rate, args.mechanism, args.seed)
        mask_files = sorted(mask_dir.glob("node_flow_chunk_*_mask.parquet"))
        missing_files = sorted(missing_dir.glob("node_flow_chunk_*.parquet"))
        if len(mask_files) != chunk_count:
            raise RuntimeError(f"rate={rate:.2f} mask chunk count mismatch: expected {chunk_count}, got {len(mask_files)}")
        if len(missing_files) != chunk_count:
            raise RuntimeError(
                f"rate={rate:.2f} missing_dataset chunk count mismatch: expected {chunk_count}, got {len(missing_files)}"
            )
        if sample_mask_file is None:
            sample_mask_file = mask_files[0]
            sample_mask_columns = list(pd.read_parquet(sample_mask_file).columns)
            if "row_index" not in sample_mask_columns:
                raise RuntimeError("mask file does not include row_index")

        rows.append(
            {
                "missing_rate": rate,
                "mask_dir": str(mask_dir),
                "mask_chunk_count": len(mask_files),
                "missing_dataset_dir": str(missing_dir),
                "missing_dataset_chunk_count": len(missing_files),
                "mask_scope": audit_payload["mask_scope"],
                "missing_unit": audit_payload["missing_unit"],
                "mask_uses_row_index": bool(audit_payload["mask_uses_row_index"]),
                "chunk_count_expected": chunk_count,
            }
        )

    check_df = pd.DataFrame(rows).sort_values("missing_rate").reset_index(drop=True)
    check_df.to_csv(paths.manifests_dir / "imputation_input_check.csv", index=False, encoding="utf-8-sig")
    check_json = {
        "input_dir": str(args.input_dir),
        "missingness_dir": str(args.missingness_dir),
        "missing_rates": args.missing_rates_parsed,
        "chunk_count_expected": chunk_count,
        "mask_scope": audit_payload["mask_scope"],
        "missing_unit": audit_payload["missing_unit"],
        "mask_uses_row_index": bool(audit_payload["mask_uses_row_index"]),
        "sample_mask_file": str(sample_mask_file) if sample_mask_file is not None else None,
        "sample_mask_columns": sample_mask_columns or [],
        "checks": check_df.to_dict(orient="records"),
        "control_group_note_zh": "0% control 不要求存在 missing_datasets 副本，也不参与 masked-position 补全误差计算。",
    }
    write_json(paths.manifests_dir / "imputation_input_check.json", check_json)
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

    true_masked_values = clean_sorted[mask_matrix]
    group_ids = build_group_ids(true_masked_values, thresholds)
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
        group_ids=group_ids,
        is_warmup=chunk_index < args.warmup_days,
    )


def compute_method_outputs(
    *,
    prepared: PreparedChunk,
    forward_last_state: np.ndarray | None,
    history_linear: deque[np.ndarray],
    history_road: deque[np.ndarray],
    history_function: deque[np.ndarray],
    history_hybrid: deque[np.ndarray],
    weight_matrix: sparse.csr_matrix,
    row_sums: np.ndarray,
    basis: np.ndarray,
    pseudo_inverse: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    zero_imputed, zero_fallback = impute_zero_fill(prepared.missing_sorted, prepared.mask_matrix)
    forward_imputed, forward_fallback = impute_forward_fill(
        prepared.missing_sorted,
        prepared.mask_matrix,
        forward_last_state,
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
    hybrid_imputed, hybrid_fallback = impute_topology_function_hybrid(
        prepared.missing_sorted,
        prepared.mask_matrix,
        road_primary_pred,
        road_primary_available,
        function_primary_pred,
        function_primary_available,
        forward_imputed,
        lam=0.5,
    )
    return {
        "zero_fill": (zero_imputed, zero_fallback),
        "forward_fill": (forward_imputed, forward_fallback),
        "historical_linear_extrapolation": (linear_imputed, linear_fallback),
        "road_topology_neighbor_fill": (road_imputed, road_fallback),
        "function_curve_fit": (function_imputed, function_fallback),
        "topology_function_hybrid": (hybrid_imputed, hybrid_fallback),
    }


def update_histories_from_outputs(
    *,
    method_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    history_linear: deque[np.ndarray],
    history_road: deque[np.ndarray],
    history_function: deque[np.ndarray],
    history_hybrid: deque[np.ndarray],
) -> np.ndarray:
    forward_imputed = method_outputs["forward_fill"][0]
    history_linear.append(method_outputs["historical_linear_extrapolation"][0].copy())
    history_road.append(method_outputs["road_topology_neighbor_fill"][0].copy())
    history_function.append(method_outputs["function_curve_fit"][0].copy())
    history_hybrid.append(method_outputs["topology_function_hybrid"][0].copy())
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
        .sort_values(["missing_rate", "method", "chunk_index", "flow_group"])
        .reset_index(drop=True)
    )
    detail_df.to_csv(paths.summaries_dir / "imputation_quality_detail.csv", index=False, encoding="utf-8-sig")


def load_output_matrix_from_file(args: argparse.Namespace, prepared: PreparedChunk, output_path: Path) -> np.ndarray:
    output_df = pd.read_parquet(output_path, columns=[args.target_col])
    output_target = output_df[args.target_col].to_numpy(dtype=np.float32, copy=False)
    return output_target[prepared.layout.sort_idx].reshape(prepared.clean_sorted.shape)


def load_existing_progress(paths: StagePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paths.chunk_status_path.exists():
        status_df = pd.read_csv(paths.chunk_status_path)
    else:
        status_df = pd.DataFrame()
    detail_path = paths.summaries_dir / "imputation_quality_detail.csv"
    if detail_path.exists():
        detail_df = pd.read_csv(detail_path)
    else:
        detail_df = pd.DataFrame()
    return status_df, detail_df


def expected_detail_rows_per_chunk(method_count: int) -> int:
    return method_count * (1 + len(FLOW_GROUP_LABELS))


def has_complete_chunk_records(
    *,
    status_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    rate: float,
    chunk_index: int,
    methods: list[str],
) -> bool:
    required_status_cols = {"missing_rate", "chunk_index", "method"}
    required_detail_cols = {"missing_rate", "chunk_index", "method", "flow_group"}
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
    expected_pairs = {(method, group) for method in methods for group in ["overall", *FLOW_GROUP_LABELS]}
    actual_pairs = set(zip(detail_subset["method"].tolist(), detail_subset["flow_group"].tolist()))
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


def historical_stack(history: deque[np.ndarray]) -> np.ndarray | None:
    if not history:
        return None
    return np.stack(list(history), axis=0).astype(np.float32, copy=False)


def compute_global_median(history: deque[np.ndarray]) -> float:
    if not history:
        return 0.0
    medians = [float(np.median(day_matrix)) for day_matrix in history]
    return float(np.median(np.asarray(medians, dtype=np.float32)))


def impute_zero_fill(missing_matrix: np.ndarray, mask_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    imputed[mask_matrix] = 0.0
    return imputed, np.zeros_like(mask_matrix, dtype=bool)


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


def impute_topology_function_hybrid(
    missing_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    road_primary_pred: np.ndarray | None,
    road_primary_available: np.ndarray | None,
    function_primary_pred: np.ndarray | None,
    function_primary_available: np.ndarray | None,
    forward_fill_matrix: np.ndarray,
    lam: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    imputed = missing_matrix.copy()
    fallback_mask = np.zeros_like(mask_matrix, dtype=bool)
    road_available = road_primary_available if road_primary_available is not None else np.zeros_like(mask_matrix, dtype=bool)
    func_available = (
        function_primary_available if function_primary_available is not None else np.zeros_like(mask_matrix, dtype=bool)
    )

    both_available = mask_matrix & road_available & func_available
    road_only = mask_matrix & road_available & ~func_available
    func_only = mask_matrix & ~road_available & func_available
    neither_available = mask_matrix & ~road_available & ~func_available

    if road_primary_pred is not None and function_primary_pred is not None:
        imputed[both_available] = (lam * road_primary_pred[both_available]) + (
            (1.0 - lam) * function_primary_pred[both_available]
        )
    if road_primary_pred is not None:
        imputed[road_only] = road_primary_pred[road_only]
    if function_primary_pred is not None:
        imputed[func_only] = function_primary_pred[func_only]
    if np.any(neither_available):
        imputed[neither_available] = forward_fill_matrix[neither_available]
        fallback_mask[neither_available] = True
    return imputed, fallback_mask


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
    flow_group: str,
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
        "flow_group": flow_group,
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
    group_ids: np.ndarray,
    is_warmup: bool,
) -> list[dict[str, Any]]:
    rows = [
        metric_row(
            missing_rate=missing_rate,
            method=method,
            chunk_index=chunk_index,
            day_index=day_index,
            flow_group="overall",
            true_values=true_values,
            pred_values=pred_values,
            fallback_flags=fallback_flags,
            is_warmup=is_warmup,
        )
    ]
    for group_id, group_label in enumerate(FLOW_GROUP_LABELS):
        selector = group_ids == group_id
        rows.append(
            metric_row(
                missing_rate=missing_rate,
                method=method,
                chunk_index=chunk_index,
                day_index=day_index,
                flow_group=group_label,
                true_values=true_values[selector],
                pred_values=pred_values[selector],
                fallback_flags=fallback_flags[selector],
                is_warmup=is_warmup,
            )
        )
    return rows


def summary_from_detail(detail_df: pd.DataFrame, exclude_warmup: bool) -> pd.DataFrame:
    filtered = detail_df.loc[~detail_df["is_warmup_day"]] if exclude_warmup else detail_df
    group_cols = ["missing_rate", "method", "flow_group"]
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
    return agg_df.sort_values(["missing_rate", "method", "flow_group"]).reset_index(drop=True)


def plot_metric(summary_df: pd.DataFrame, metric: str, output_png: Path, output_pdf: Path) -> None:
    overall_df = summary_df.loc[summary_df["flow_group"] == "overall"].copy()
    if overall_df.empty:
        raise RuntimeError(f"cannot plot {metric}: overall summary is empty")
    plt.figure(figsize=(10, 6))
    for method in METHOD_ORDER:
        method_df = overall_df.loc[overall_df["method"] == method].sort_values("missing_rate")
        plt.plot(method_df["missing_rate"], method_df[metric], marker="o", linewidth=2, label=method)
    plt.xlabel("Missing Rate")
    plt.ylabel(metric.upper())
    plt.title(f"Global MCAR Imputation {metric.upper()} by Method")
    plt.xticks(argsort_unique(overall_df["missing_rate"].to_numpy(dtype=float)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.savefig(output_pdf)
    plt.close()


def plot_nonzero_zoom(summary_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> bool:
    overall_df = summary_df.loc[summary_df["flow_group"] == "overall"].copy()
    zero_df = overall_df.loc[overall_df["method"] == "zero_fill"]
    other_df = overall_df.loc[overall_df["method"] != "zero_fill"]
    if zero_df.empty or other_df.empty:
        return False
    if float(zero_df["rmse"].max()) <= 1.25 * float(other_df["rmse"].max()):
        return False

    plt.figure(figsize=(10, 6))
    for method in [method for method in METHOD_ORDER if method != "zero_fill"]:
        method_df = overall_df.loc[overall_df["method"] == method].sort_values("missing_rate")
        plt.plot(method_df["missing_rate"], method_df["rmse"], marker="o", linewidth=2, label=method)
    plt.xlabel("Missing Rate")
    plt.ylabel("RMSE")
    plt.title("Global MCAR Imputation RMSE by Method (Non-Zero Methods Only)")
    plt.xticks(argsort_unique(overall_df["missing_rate"].to_numpy(dtype=float)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.savefig(output_pdf)
    plt.close()
    return True


def plot_flow_group_rmse(summary_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    flow_df = summary_df.loc[summary_df["flow_group"].isin(FLOW_GROUP_LABELS)].copy()
    if flow_df.empty:
        raise RuntimeError("cannot plot flow-group RMSE because grouped summary is empty")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, group_label in zip(axes, FLOW_GROUP_LABELS):
        group_df = flow_df.loc[flow_df["flow_group"] == group_label]
        for method in METHOD_ORDER:
            method_df = group_df.loc[group_df["method"] == method].sort_values("missing_rate")
            axis.plot(method_df["missing_rate"], method_df["rmse"], marker="o", linewidth=1.8, label=method)
        axis.set_title(group_label)
        axis.set_xlabel("Missing Rate")
        axis.set_xticks(argsort_unique(group_df["missing_rate"].to_numpy(dtype=float)))
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Global MCAR Imputation RMSE by Flow Group and Method")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


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
    missing_dir = missing_subdir(args.missingness_dir, rate, args.mechanism, args.seed)
    mask_dir = mask_subdir(args.missingness_dir, rate, args.mechanism, args.seed)

    valid_status_rows: list[dict[str, Any]] = []
    valid_detail_rows: list[dict[str, Any]] = []
    history_linear: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_road: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_function: deque[np.ndarray] = deque(maxlen=args.history_days)
    history_hybrid: deque[np.ndarray] = deque(maxlen=args.history_days)
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
            output_path = imputed_subdir(paths.root, rate, args.mechanism, args.seed, method) / clean_file.name
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
            method_outputs=per_method_outputs,
            history_linear=history_linear,
            history_road=history_road,
            history_function=history_function,
            history_hybrid=history_hybrid,
        )

    return RateScanResult(
        missing_rate=rate,
        resume_chunk_index=resume_chunk_index,
        valid_prefix_chunk_count=valid_prefix_chunk_count,
        valid_status_rows=valid_status_rows,
        valid_detail_rows=valid_detail_rows,
        forward_last_state=None if forward_last_state is None else forward_last_state.copy(),
        history_linear=copy_history(history_linear, args.history_days),
        history_road=copy_history(history_road, args.history_days),
        history_function=copy_history(history_function, args.history_days),
        history_hybrid=copy_history(history_hybrid, args.history_days),
    )


def run_impute(
    args: argparse.Namespace,
    paths: StagePaths,
    input_check_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if input_check_df is None:
        input_check_df = pd.read_csv(paths.manifests_dir / "imputation_input_check.csv")

    clean_files = list_clean_files(args.input_dir)
    ref_node_ids, node_count = prepare_reference_layout(args, clean_files)
    thresholds = compute_flow_group_thresholds(
        clean_files=clean_files,
        target_col=args.target_col,
        seed=args.seed,
        output_path=paths.manifests_dir / "flow_group_thresholds.json",
    )
    weight_matrix, row_sums = build_topology_matrix(args.topology_file, ref_node_ids)
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

        history_linear = copy_history(scan_result.history_linear, args.history_days)
        history_road = copy_history(scan_result.history_road, args.history_days)
        history_function = copy_history(scan_result.history_function, args.history_days)
        history_hybrid = copy_history(scan_result.history_hybrid, args.history_days)
        forward_last_state = None if scan_result.forward_last_state is None else scan_result.forward_last_state.copy()

        missing_dir = missing_subdir(args.missingness_dir, rate, args.mechanism, args.seed)
        mask_dir = mask_subdir(args.missingness_dir, rate, args.mechanism, args.seed)
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
                history_linear=history_linear,
                history_road=history_road,
                history_function=history_function,
                history_hybrid=history_hybrid,
                weight_matrix=weight_matrix,
                row_sums=row_sums,
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

                output_dir = imputed_subdir(paths.root, rate, args.mechanism, args.seed, method)
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
                        group_ids=prepared.group_ids,
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
                method_outputs=method_outputs,
                history_linear=history_linear,
                history_road=history_road,
                history_function=history_function,
                history_hybrid=history_hybrid,
            )
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
        .sort_values(["missing_rate", "method", "chunk_index", "flow_group"])
        .reset_index(drop=True)
    )
    status_df = (
        pd.DataFrame(status_rows)
        .sort_values(["missing_rate", "method", "chunk_index"])
        .reset_index(drop=True)
    )
    detail_df.to_csv(paths.summaries_dir / "imputation_quality_detail.csv", index=False, encoding="utf-8-sig")
    status_df.to_csv(paths.chunk_status_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(scan_summaries).sort_values(["missing_rate"]).to_csv(
        paths.resume_scan_path,
        index=False,
        encoding="utf-8-sig",
    )
    return detail_df, status_df


def run_summarize(paths: StagePaths, detail_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if detail_df is None:
        detail_path = paths.summaries_dir / "imputation_quality_detail.csv"
        if not detail_path.exists():
            raise FileNotFoundError("imputation_quality_detail.csv not found; run --stage impute first")
        detail_df = pd.read_csv(detail_path)

    summary_all_days = summary_from_detail(detail_df, exclude_warmup=False)
    summary_exclude_warmup = summary_from_detail(detail_df, exclude_warmup=True)
    summary_by_flow_group = summary_exclude_warmup.loc[summary_exclude_warmup["flow_group"] != "overall"].copy()

    summary_all_days.to_csv(
        paths.summaries_dir / "imputation_quality_summary_all_days.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_exclude_warmup.to_csv(
        paths.summaries_dir / "imputation_quality_summary_exclude_warmup.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_by_flow_group.to_csv(
        paths.summaries_dir / "imputation_quality_by_flow_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary_all_days, summary_exclude_warmup, summary_by_flow_group


def run_validate(args: argparse.Namespace, paths: StagePaths) -> dict[str, Any]:
    status_path = paths.chunk_status_path
    summary_path = paths.summaries_dir / "imputation_quality_summary_exclude_warmup.csv"
    if not status_path.exists():
        raise FileNotFoundError("imputed_chunk_status.csv not found; run --stage impute first")
    if not summary_path.exists():
        raise FileNotFoundError("imputation_quality_summary_exclude_warmup.csv not found; run --stage summarize first")

    status_df = pd.read_csv(status_path)
    summary_df = pd.read_csv(summary_path)
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

    audit_payload = {
        "input_missingness_dir": str(args.missingness_dir),
        "output_imputation_dir": str(paths.imputed_datasets_dir),
        "missing_rates": args.missing_rates_parsed,
        "methods": args.impute_methods_parsed,
        "causal_history_only": True,
        "history_days": args.history_days,
        "context_days_after": args.context_days_after,
        "uses_future_days": False,
        "uses_same_day_future_slots": False,
        "uses_bfill": False,
        "uses_bidirectional_interpolation": False,
        "warmup_days": args.warmup_days,
        "main_metrics_exclude_warmup": args.exclude_warmup_from_main_metrics,
        "fallback_policy": {
            "zero_fill": "direct_zero_fill_no_fallback",
            "forward_fill": "use_global_safe_fallback_zero_when_no_causal_history_exists",
            "historical_linear_extrapolation": "fallback_to_current_day_forward_fill_when_history_is_insufficient",
            "road_topology_neighbor_fill": "fallback_to_current_day_forward_fill_when_no_topology_history_is_available",
            "function_curve_fit": "fallback_to_current_day_forward_fill_when_no_history_profile_is_available",
            "topology_function_hybrid": "blend_topology_and_function_primary_predictions_or_fallback_to_current_day_forward_fill",
        },
        "non_mask_positions_preserved": True,
        "evaluation_only_on_mask_positions": True,
        "completeness": completeness_rows,
        "output_files": {
            "imputed_chunk_status": str(status_path),
            "chunk_runtime_state_log": str(paths.chunk_state_log_path),
            "resume_scan": str(paths.resume_scan_path),
            "detail_summary": str(paths.summaries_dir / "imputation_quality_detail.csv"),
            "summary_all_days": str(paths.summaries_dir / "imputation_quality_summary_all_days.csv"),
            "summary_exclude_warmup": str(paths.summaries_dir / "imputation_quality_summary_exclude_warmup.csv"),
            "summary_by_flow_group": str(paths.summaries_dir / "imputation_quality_by_flow_group.csv"),
        },
        "resume_mode": "status_snapshot_only",
    }
    write_json(paths.audits_dir / "causal_imputation_audit.json", audit_payload)
    write_audit_markdown(paths.audits_dir / "causal_imputation_audit_zh.md", audit_payload)
    return audit_payload


def write_audit_markdown(path: Path, audit_payload: dict[str, Any]) -> None:
    lines = [
        "# 全局 MCAR 缺失补全因果审计报告",
        "",
        "## 1. 范围",
        "",
        "- 本轮只基于已有 `masks` 与 `missing_datasets` 执行补全。",
        "- 未重新生成缺失设置。",
        "- 未修改原始 `input_dir`。",
        "- 本轮结果只代表缺失值补全误差，不代表交通预测误差。",
        "",
        "## 2. 严格历史因果约束",
        "",
        f"- causal_history_only: `{audit_payload['causal_history_only']}`",
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
        "- road_topology_neighbor_fill 使用的是 `rnsd_processed.csv` 中的路网拓扑邻接与道路长度权重，不是经纬度距离近邻。",
        "",
        "## 4. Fallback 策略",
        "",
    ]
    for method, description in audit_payload["fallback_policy"].items():
        lines.append(f"- {method}: `{description}`")
    lines.extend(["", "## 5. 输出完整性", ""])
    for row in audit_payload["completeness"]:
        lines.append(
            f"- rate={row['missing_rate']:.2f}, method={row['method']}, imputed_chunk_count={row['imputed_chunk_count']}, expected={row['expected_chunk_count']}, is_complete={row['is_complete']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_plot(paths: StagePaths) -> dict[str, Any]:
    summary_path = paths.summaries_dir / "imputation_quality_summary_exclude_warmup.csv"
    if not summary_path.exists():
        raise FileNotFoundError("imputation_quality_summary_exclude_warmup.csv not found; run --stage summarize first")
    summary_df = pd.read_csv(summary_path)

    plot_metric(
        summary_df,
        metric="rmse",
        output_png=paths.figures_dir / "multirate_rmse_by_method.png",
        output_pdf=paths.figures_dir / "multirate_rmse_by_method.pdf",
    )
    plot_metric(
        summary_df,
        metric="mae",
        output_png=paths.figures_dir / "multirate_mae_by_method.png",
        output_pdf=paths.figures_dir / "multirate_mae_by_method.pdf",
    )
    plot_metric(
        summary_df,
        metric="smape",
        output_png=paths.figures_dir / "multirate_smape_by_method.png",
        output_pdf=paths.figures_dir / "multirate_smape_by_method.pdf",
    )
    plot_metric(
        summary_df,
        metric="nrmse",
        output_png=paths.figures_dir / "multirate_nrmse_by_method.png",
        output_pdf=paths.figures_dir / "multirate_nrmse_by_method.pdf",
    )
    zoom_created = plot_nonzero_zoom(
        summary_df,
        output_png=paths.figures_dir / "multirate_rmse_by_method_nonzero_zoom.png",
        output_pdf=paths.figures_dir / "multirate_rmse_by_method_nonzero_zoom.pdf",
    )
    plot_flow_group_rmse(
        summary_df,
        output_png=paths.figures_dir / "multirate_flow_group_rmse_by_method.png",
        output_pdf=paths.figures_dir / "multirate_flow_group_rmse_by_method.pdf",
    )
    return {"nonzero_zoom_created": zoom_created}


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.missingness_dir = ensure_absolute(project_root, args.missingness_dir)
    args.output_dir = ensure_absolute(project_root, args.output_dir)
    args.topology_file = ensure_absolute(project_root, args.topology_file)
    args.missing_rates_parsed = [rate for rate in parse_missing_rates(args.missing_rates) if rate > 0]
    args.impute_methods_parsed = parse_methods(args.impute_methods)
    args.exclude_warmup_from_main_metrics = parse_bool(args.exclude_warmup_from_main_metrics)
    validate_args(args)

    paths = build_paths(args.output_dir)
    mkdirs(paths)
    write_run_artifacts(args, paths)

    input_check_df: pd.DataFrame | None = None
    detail_df: pd.DataFrame | None = None

    if args.stage in {"prepare", "all"}:
        input_check_df, _ = run_prepare(args, paths)
    if args.stage in {"impute", "all"}:
        if input_check_df is None:
            input_check_df = pd.read_csv(paths.manifests_dir / "imputation_input_check.csv")
        detail_df, _ = run_impute(args, paths, input_check_df)
    if args.stage in {"summarize", "all"}:
        run_summarize(paths, detail_df)
    if args.stage in {"validate", "all"}:
        run_validate(args, paths)
    if args.stage in {"plot", "all"}:
        run_plot(paths)


if __name__ == "__main__":
    main()
