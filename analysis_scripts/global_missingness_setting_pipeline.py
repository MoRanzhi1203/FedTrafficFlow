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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXPECTED_EXPERIMENT_NAME = "real_data_global_missingness_setting"
ALLOCATION_METHOD = "sequential_hypergeometric_global_without_replacement"
MASK_SCOPE = "global"
MISSINGNESS_TYPE = "global_mcar_point"


@dataclass(frozen=True)
class StagePaths:
    root: Path
    masks_dir: Path
    missing_datasets_dir: Path
    manifests_dir: Path
    audits_dir: Path
    run_config_path: Path
    run_commands_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只生成完整数据全局 MCAR 缺失值设置，不执行任何补全。")
    parser.add_argument("--stage", required=True, choices=["prepare", "generate_missing", "audit", "all"])
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--missing_rates", required=True, type=str)
    parser.add_argument("--mechanism", required=True, type=str)
    parser.add_argument("--mask_scope", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target_col", required=True, type=str)
    parser.add_argument("--node_col", required=True, type=str)
    parser.add_argument("--time_col", required=True, type=str)
    parser.add_argument("--period", required=True, type=int)
    return parser.parse_args()


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def parse_missing_rates(raw: str) -> list[float]:
    rates: list[float] = []
    for token in raw.split(","):
        value = float(token.strip())
        if value < 0 or value > 1:
            raise ValueError(f"missing rate out of range: {value}")
        rates.append(value)
    if not rates:
        raise ValueError("missing_rates is empty")
    return rates


def format_rate_tag(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_paths(output_dir: Path) -> StagePaths:
    return StagePaths(
        root=output_dir,
        masks_dir=output_dir / "masks",
        missing_datasets_dir=output_dir / "missing_datasets",
        manifests_dir=output_dir / "manifests",
        audits_dir=output_dir / "audits",
        run_config_path=output_dir / "run_config.json",
        run_commands_path=output_dir / "run_commands.txt",
    )


def mkdirs(paths: StagePaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.masks_dir.mkdir(parents=True, exist_ok=True)
    paths.missing_datasets_dir.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    paths.audits_dir.mkdir(parents=True, exist_ok=True)


def list_chunk_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("node_flow_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no chunk files found in {input_dir}")
    return files


def extract_day_index(file_name: str) -> int:
    match = re.search(r"node_flow_chunk_(\d+)\.parquet$", file_name)
    if not match:
        raise ValueError(f"unexpected chunk file name: {file_name}")
    return int(match.group(1))


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_serializable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_run_artifacts(args: argparse.Namespace, paths: StagePaths) -> None:
    missing_rates = parse_missing_rates(args.missing_rates)
    effective_rates = [rate for rate in missing_rates if rate > 0]
    config = {
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "stage": "missingness_setting_only",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "missing_rates": missing_rates,
        "effective_missing_rates_for_dataset_generation": effective_rates,
        "mechanism": args.mechanism,
        "mask_scope": args.mask_scope,
        "seed": args.seed,
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "imputation_enabled": False,
    }
    write_json(paths.run_config_path, config)

    command_lines = [
        "Global missingness setting pipeline commands",
        f"python {' '.join(sys.argv)}",
    ]
    paths.run_commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def load_prepare_outputs(paths: StagePaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts_path = paths.manifests_dir / "global_eligible_chunk_counts.csv"
    summary_path = paths.manifests_dir / "global_eligible_summary.json"
    if not counts_path.exists() or not summary_path.exists():
        raise FileNotFoundError("prepare outputs not found; run --stage prepare first")
    counts_df = pd.read_csv(counts_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return counts_df, summary


def validate_common_args(args: argparse.Namespace) -> None:
    if args.mechanism != "mcar_point":
        raise ValueError("this pipeline only supports mechanism=mcar_point")
    if args.mask_scope != MASK_SCOPE:
        raise ValueError("this pipeline only supports mask_scope=global")
    if args.period <= 0:
        raise ValueError("period must be positive")


def run_prepare(args: argparse.Namespace, paths: StagePaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = list_chunk_files(args.input_dir)
    records: list[dict[str, Any]] = []
    required_columns = {args.node_col, args.time_col, args.target_col}

    for chunk_index, file_path in enumerate(files):
        day_index = extract_day_index(file_path.name)
        parquet_file = pq.ParquetFile(file_path)
        schema_names = set(parquet_file.schema_arrow.names)
        missing_columns = required_columns - schema_names
        if missing_columns:
            raise KeyError(f"{file_path.name} missing required columns: {sorted(missing_columns)}")

        row_count = int(parquet_file.metadata.num_rows)
        target_series = pd.read_parquet(file_path, columns=[args.target_col])[args.target_col]
        eligible_count = int(target_series.notna().sum())
        if row_count <= 0:
            raise ValueError(f"{file_path.name} has non-positive row count")

        records.append(
            {
                "chunk_index": chunk_index,
                "day_index": day_index,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "row_count": row_count,
                "eligible_count": eligible_count,
                "target_col": args.target_col,
            }
        )

    counts_df = pd.DataFrame(records).sort_values(["chunk_index", "day_index"]).reset_index(drop=True)
    global_eligible_count = int(counts_df["eligible_count"].sum())
    summary = {
        "experiment_name": EXPECTED_EXPERIMENT_NAME,
        "input_dir": str(args.input_dir),
        "chunk_count": int(len(counts_df)),
        "global_eligible_count": global_eligible_count,
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "period": args.period,
        "mechanism": args.mechanism,
        "mask_scope": args.mask_scope,
        "seed": args.seed,
    }

    counts_df.to_csv(paths.manifests_dir / "global_eligible_chunk_counts.csv", index=False, encoding="utf-8-sig")
    write_json(paths.manifests_dir / "global_eligible_summary.json", summary)
    return counts_df, summary


def allocate_for_rate(counts_df: pd.DataFrame, rate: float, seed: int, mechanism: str) -> list[dict[str, Any]]:
    global_eligible_count = int(counts_df["eligible_count"].sum())
    global_missing_count = int(round(global_eligible_count * rate))
    remaining_total = global_eligible_count
    remaining_missing = global_missing_count
    rng = np.random.default_rng(stable_seed(seed, "allocation", f"{rate:.8f}"))
    rows: list[dict[str, Any]] = []

    for pos, record in enumerate(counts_df.to_dict(orient="records")):
        eligible_count = int(record["eligible_count"])
        if eligible_count < 0:
            raise ValueError("eligible_count must be non-negative")

        if pos == len(counts_df) - 1:
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
        observed_chunk_missing_rate = (
            float(allocated_missing_count / eligible_count) if eligible_count > 0 else math.nan
        )

        rows.append(
            {
                "missing_rate_target": rate,
                "chunk_index": int(record["chunk_index"]),
                "day_index": int(record["day_index"]),
                "file_name": str(record["file_name"]),
                "eligible_count": eligible_count,
                "allocated_missing_count": int(allocated_missing_count),
                "observed_chunk_missing_rate": observed_chunk_missing_rate,
                "global_eligible_count": global_eligible_count,
                "global_missing_count": global_missing_count,
                "mask_scope": MASK_SCOPE,
                "mechanism": mechanism,
                "allocation_method": ALLOCATION_METHOD,
                "seed": seed,
            }
        )

        remaining_total -= eligible_count
        remaining_missing -= allocated_missing_count

    if remaining_total != 0 or remaining_missing != 0:
        raise RuntimeError(f"allocation did not terminate cleanly for rate={rate:.2f}")
    if sum(int(row["allocated_missing_count"]) for row in rows) != global_missing_count:
        raise RuntimeError(f"allocation total mismatch for rate={rate:.2f}")
    return rows


def build_allocation_table(counts_df: pd.DataFrame, rates: Iterable[float], seed: int, mechanism: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rate in rates:
        rows.extend(allocate_for_rate(counts_df=counts_df, rate=rate, seed=seed, mechanism=mechanism))
    allocation_df = pd.DataFrame(rows).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)
    allocation_df.to_csv(
        StagePaths(
            root=Path(),
            masks_dir=Path(),
            missing_datasets_dir=Path(),
            manifests_dir=Path(),
            audits_dir=Path(),
            run_config_path=Path(),
            run_commands_path=Path(),
        ).manifests_dir / "unused.csv",
        index=False,
    )
    return allocation_df


def mask_subdir(paths: StagePaths, rate: float, mechanism: str, seed: int) -> Path:
    return paths.masks_dir / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"


def missing_subdir(paths: StagePaths, rate: float, mechanism: str, seed: int) -> Path:
    return paths.missing_datasets_dir / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"


def write_allocation_csv(paths: StagePaths, allocation_df: pd.DataFrame) -> None:
    allocation_df.to_csv(paths.manifests_dir / "global_missing_allocation.csv", index=False, encoding="utf-8-sig")


def build_rate_lookup(allocation_df: pd.DataFrame) -> dict[float, dict[int, dict[str, Any]]]:
    lookup: dict[float, dict[int, dict[str, Any]]] = {}
    for row in allocation_df.to_dict(orient="records"):
        rate = float(row["missing_rate_target"])
        day_index = int(row["day_index"])
        lookup.setdefault(rate, {})[day_index] = row
    return lookup


def generate_masks_and_datasets(
    args: argparse.Namespace,
    paths: StagePaths,
    counts_df: pd.DataFrame,
    allocation_df: pd.DataFrame,
) -> pd.DataFrame:
    positive_rates = [rate for rate in parse_missing_rates(args.missing_rates) if rate > 0]
    files = list_chunk_files(args.input_dir)
    allocation_lookup = build_rate_lookup(allocation_df)
    status_rows: list[dict[str, Any]] = []

    for file_path in files:
        day_index = extract_day_index(file_path.name)
        df = pd.read_parquet(file_path)
        required_columns = {args.node_col, args.time_col, args.target_col}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise KeyError(f"{file_path.name} missing required columns after read: {sorted(missing_columns)}")

        row_index = np.arange(len(df), dtype=np.int64)
        eligible_mask = df[args.target_col].notna().to_numpy()
        eligible_indices = np.flatnonzero(eligible_mask).astype(np.int64)
        expected_eligible_count = int(
            counts_df.loc[counts_df["day_index"] == day_index, "eligible_count"].iloc[0]
        )
        if len(eligible_indices) != expected_eligible_count:
            raise RuntimeError(f"{file_path.name} eligible_count changed between prepare and generate_missing")

        if not pd.api.types.is_integer_dtype(df[args.time_col]):
            raise TypeError(f"{file_path.name} time column must be integer for global_time_index")
        total_per_time_slot = df.groupby(args.time_col, sort=False).size().to_dict()

        for rate in positive_rates:
            allocation_row = allocation_lookup[rate][day_index]
            missing_count = int(allocation_row["allocated_missing_count"])
            per_rate_mask_dir = mask_subdir(paths, rate, args.mechanism, args.seed)
            per_rate_dataset_dir = missing_subdir(paths, rate, args.mechanism, args.seed)
            per_rate_mask_dir.mkdir(parents=True, exist_ok=True)
            per_rate_dataset_dir.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(stable_seed(args.seed, "mask", f"{rate:.8f}", file_path.name))
            if missing_count > 0:
                selected_rows = np.sort(rng.choice(eligible_indices, size=missing_count, replace=False).astype(np.int64))
            else:
                selected_rows = np.array([], dtype=np.int64)

            selected_time_slots = df.iloc[selected_rows][args.time_col].to_numpy(dtype=np.int64, copy=False)
            unique_time_slots, time_slot_missing_counts = np.unique(selected_time_slots, return_counts=True)
            drops_entire_time_slot = False
            for slot, slot_missing_count in zip(unique_time_slots.tolist(), time_slot_missing_counts.tolist()):
                if int(slot_missing_count) == int(total_per_time_slot.get(slot, -1)):
                    drops_entire_time_slot = True
                    break
            if drops_entire_time_slot:
                raise RuntimeError(
                    f"{file_path.name} rate={rate:.2f} drops an entire time slot; this violates point-level MCAR"
                )

            mask_df = pd.DataFrame(
                {
                    "row_index": selected_rows,
                    args.node_col: df.iloc[selected_rows][args.node_col].to_numpy(copy=False),
                    args.time_col: df.iloc[selected_rows][args.time_col].to_numpy(copy=False),
                    "day_index": np.full(len(selected_rows), day_index, dtype=np.int64),
                    "global_time_index": (day_index * int(args.period) + df.iloc[selected_rows][args.time_col].to_numpy(copy=False)).astype(np.int64, copy=False),
                    "is_missing": np.full(len(selected_rows), True, dtype=bool),
                }
            )
            mask_path = per_rate_mask_dir / file_path.name.replace(".parquet", "_mask.parquet")
            mask_df.to_parquet(mask_path, index=False)

            out_df = df.copy()
            out_df.loc[selected_rows, args.target_col] = np.nan
            output_path = per_rate_dataset_dir / file_path.name
            out_df.to_parquet(output_path, index=False)

            status_rows.append(
                {
                    "missing_rate_target": rate,
                    "chunk_index": int(
                        counts_df.loc[counts_df["day_index"] == day_index, "chunk_index"].iloc[0]
                    ),
                    "day_index": day_index,
                    "file_name": file_path.name,
                    "mask_path": str(mask_path),
                    "missing_dataset_path": str(output_path),
                    "eligible_count": expected_eligible_count,
                    "missing_count": missing_count,
                    "observed_chunk_missing_rate": float(missing_count / expected_eligible_count)
                    if expected_eligible_count > 0
                    else math.nan,
                    "mask_scope": MASK_SCOPE,
                    "mechanism": args.mechanism,
                    "seed": args.seed,
                    "missing_unit": "node_time_observation",
                    "mask_uses_row_index": True,
                    "drops_entire_time_slot": False,
                    "drops_all_nodes_at_same_time": False,
                    "creates_temporal_blocks": False,
                    "forces_non_contiguous_missing": False,
                    "missingness_type": MISSINGNESS_TYPE,
                }
            )

            del out_df, mask_df
            gc.collect()

        del df
        gc.collect()

    status_df = pd.DataFrame(status_rows).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)
    status_df.to_csv(paths.manifests_dir / "generate_missing_chunk_status.csv", index=False, encoding="utf-8-sig")
    return status_df


def is_day_stratified_like(observed_rates: np.ndarray) -> bool:
    if observed_rates.size == 0:
        return False
    return bool(np.allclose(observed_rates, observed_rates[0], rtol=0.0, atol=1e-15))


def build_audit_tables(
    args: argparse.Namespace,
    paths: StagePaths,
    counts_df: pd.DataFrame,
    allocation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    positive_rates = [rate for rate in parse_missing_rates(args.missing_rates) if rate > 0]
    chunk_rate_rows: list[dict[str, Any]] = []
    rate_summaries: list[dict[str, Any]] = []

    for rate in parse_missing_rates(args.missing_rates):
        rate_df = allocation_df.loc[np.isclose(allocation_df["missing_rate_target"], rate)].copy()
        observed_rates = rate_df["observed_chunk_missing_rate"].to_numpy(dtype=float)
        global_eligible_count = int(rate_df["global_eligible_count"].iloc[0])
        global_missing_count = int(rate_df["global_missing_count"].iloc[0])
        sum_allocated_missing_count = int(rate_df["allocated_missing_count"].sum())
        observed_global_missing_rate = (
            float(sum_allocated_missing_count / global_eligible_count) if global_eligible_count > 0 else math.nan
        )
        day_stratified = is_day_stratified_like(observed_rates)

        per_day_missing_rate_std = float(np.nanstd(observed_rates))
        summary_row = {
            "missing_rate_target": rate,
            "global_missing_count": global_missing_count,
            "observed_global_missing_rate": observed_global_missing_rate,
            "per_day_missing_rate_min": float(np.nanmin(observed_rates)) if observed_rates.size else math.nan,
            "per_day_missing_rate_max": float(np.nanmax(observed_rates)) if observed_rates.size else math.nan,
            "per_day_missing_rate_mean": float(np.nanmean(observed_rates)) if observed_rates.size else math.nan,
            "per_day_missing_rate_std": per_day_missing_rate_std,
            "sum_allocated_missing_count": sum_allocated_missing_count,
            "is_global_count_exact": bool(sum_allocated_missing_count == global_missing_count),
            "is_day_stratified_like": bool(day_stratified),
        }
        if rate > 0:
            if not summary_row["is_global_count_exact"]:
                raise RuntimeError(f"rate={rate:.2f} global missing count is not exact")
            if day_stratified:
                raise RuntimeError(f"rate={rate:.2f} still looks day-stratified because all chunk rates are identical")
            if per_day_missing_rate_std <= 0:
                raise RuntimeError(f"rate={rate:.2f} per-day missing rate std must be > 0")
        rate_summaries.append(summary_row)

        for row in rate_df.to_dict(orient="records"):
            chunk_rate_rows.append(
                {
                    "missing_rate_target": rate,
                    "chunk_index": int(row["chunk_index"]),
                    "day_index": int(row["day_index"]),
                    "eligible_count": int(row["eligible_count"]),
                    "missing_count": int(row["allocated_missing_count"]),
                    "observed_chunk_missing_rate": float(row["observed_chunk_missing_rate"]),
                    "global_eligible_count": global_eligible_count,
                    "global_missing_count": global_missing_count,
                    "observed_global_missing_rate": observed_global_missing_rate,
                    "mask_scope": MASK_SCOPE,
                    "mechanism": args.mechanism,
                    "seed": args.seed,
                    "missing_unit": "node_time_observation",
                    "mask_uses_row_index": True,
                    "drops_entire_time_slot": False,
                    "drops_all_nodes_at_same_time": False,
                    "creates_temporal_blocks": False,
                    "forces_non_contiguous_missing": False,
                    "missingness_type": MISSINGNESS_TYPE,
                }
            )

    chunk_rate_df = pd.DataFrame(chunk_rate_rows).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)
    chunk_rate_df.to_csv(
        paths.audits_dir / "global_missingness_rate_by_chunk.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_payload = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target_col": args.target_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "chunk_count": int(len(counts_df)),
        "global_eligible_count": int(counts_df["eligible_count"].sum()),
        "missing_rates": parse_missing_rates(args.missing_rates),
        "mask_scope": MASK_SCOPE,
        "mechanism": args.mechanism,
        "seed": args.seed,
        "allocation_method": ALLOCATION_METHOD,
        "missing_unit": "node_time_observation",
        "mask_uses_row_index": True,
        "drops_entire_time_slot": False,
        "drops_all_nodes_at_same_time": False,
        "creates_temporal_blocks": False,
        "forces_non_contiguous_missing": False,
        "missingness_type": MISSINGNESS_TYPE,
        "rate_summaries": rate_summaries,
        "control_group_note_en": "0% is treated as no-missing control. No mask positions are generated. No missing dataset copy is written. No imputation metrics are applicable.",
        "control_group_note_zh": "0% 作为无缺失对照组，不生成大体积缺失数据副本，不参与缺失位置补全误差计算。",
    }
    write_json(paths.audits_dir / "global_missingness_setting_audit.json", audit_payload)
    return chunk_rate_df, audit_payload


def write_audit_markdown(path: Path, audit_payload: dict[str, Any]) -> None:
    lines = [
        "# 完整数据全局缺失值设置审计报告",
        "",
        "## 1. 任务范围",
        "",
        "- 本次仅生成完整 61 天真实路口流量数据的 global MCAR 缺失值设置。",
        "- 未执行任何补全方法。",
        "- 未生成 imputed_datasets。",
        "- 未计算任何插补误差。",
        "",
        "## 2. 数据范围",
        "",
        f"- input_dir: `{audit_payload['input_dir']}`",
        f"- output_dir: `{audit_payload['output_dir']}`",
        f"- chunk_count: `{audit_payload['chunk_count']}`",
        f"- global_eligible_count: `{audit_payload['global_eligible_count']}`",
        f"- target_col: `{audit_payload['target_col']}`",
        f"- node_col: `{audit_payload['node_col']}`",
        f"- time_col: `{audit_payload['time_col']}`",
        "",
        "## 3. 缺失机制",
        "",
        f"- mechanism: `{audit_payload['mechanism']}`",
        f"- mask_scope: `{audit_payload['mask_scope']}`",
        f"- seed: `{audit_payload['seed']}`",
        f"- allocation_method: `{audit_payload['allocation_method']}`",
        f"- missing_unit: `{audit_payload['missing_unit']}`",
        f"- mask_uses_row_index: `{audit_payload['mask_uses_row_index']}`",
        f"- drops_entire_time_slot: `{audit_payload['drops_entire_time_slot']}`",
        f"- drops_all_nodes_at_same_time: `{audit_payload['drops_all_nodes_at_same_time']}`",
        f"- creates_temporal_blocks: `{audit_payload['creates_temporal_blocks']}`",
        f"- forces_non_contiguous_missing: `{audit_payload['forces_non_contiguous_missing']}`",
        f"- missingness_type: `{audit_payload['missingness_type']}`",
        "",
        "说明：缺失单位是单个 `(day_index, 节点ID, 时间段)` 行级观测，mask 记录具体 `row_index`，不会因为某个时间段被抽中而把该时间段下所有路口整体置缺失，也不会人为构造连续 block 或强制缺失点彼此不相邻。",
        "",
        "## 4. 0% 对照组",
        "",
        f"- 英文说明：{audit_payload['control_group_note_en']}",
        f"- 中文说明：{audit_payload['control_group_note_zh']}",
        "",
        "## 5. 各缺失率审计结果",
        "",
    ]
    for summary in audit_payload["rate_summaries"]:
        lines.extend(
            [
                f"### 缺失率 {summary['missing_rate_target']:.0%}",
                "",
                f"- global_missing_count: `{summary['global_missing_count']}`",
                f"- observed_global_missing_rate: `{summary['observed_global_missing_rate']}`",
                f"- per_day_missing_rate_min: `{summary['per_day_missing_rate_min']}`",
                f"- per_day_missing_rate_max: `{summary['per_day_missing_rate_max']}`",
                f"- per_day_missing_rate_mean: `{summary['per_day_missing_rate_mean']}`",
                f"- per_day_missing_rate_std: `{summary['per_day_missing_rate_std']}`",
                f"- sum_allocated_missing_count: `{summary['sum_allocated_missing_count']}`",
                f"- is_global_count_exact: `{summary['is_global_count_exact']}`",
                f"- is_day_stratified_like: `{summary['is_day_stratified_like']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace, paths: StagePaths, counts_df: pd.DataFrame | None = None) -> dict[str, Any]:
    if counts_df is None:
        counts_df, _ = load_prepare_outputs(paths)
    allocation_path = paths.manifests_dir / "global_missing_allocation.csv"
    if not allocation_path.exists():
        raise FileNotFoundError("global_missing_allocation.csv not found; run generate_missing first")
    allocation_df = pd.read_csv(allocation_path)
    _, audit_payload = build_audit_tables(args=args, paths=paths, counts_df=counts_df, allocation_df=allocation_df)
    write_audit_markdown(paths.audits_dir / "global_missingness_setting_audit_zh.md", audit_payload)
    return audit_payload


def run_generate_missing(args: argparse.Namespace, paths: StagePaths, counts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if counts_df is None:
        counts_df, _ = load_prepare_outputs(paths)
    allocation_df = pd.DataFrame(
        [row for rate in parse_missing_rates(args.missing_rates) for row in allocate_for_rate(counts_df, rate, args.seed, args.mechanism)]
    ).sort_values(["missing_rate_target", "chunk_index"]).reset_index(drop=True)
    write_allocation_csv(paths, allocation_df)
    generate_masks_and_datasets(args=args, paths=paths, counts_df=counts_df, allocation_df=allocation_df)
    return allocation_df


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    args.input_dir = ensure_absolute(project_root, args.input_dir)
    args.output_dir = ensure_absolute(project_root, args.output_dir)
    validate_common_args(args)

    paths = build_paths(args.output_dir)
    mkdirs(paths)
    write_run_artifacts(args, paths)

    counts_df: pd.DataFrame | None = None

    if args.stage in {"prepare", "all"}:
        counts_df, _ = run_prepare(args=args, paths=paths)
    if args.stage in {"generate_missing", "all"}:
        if counts_df is None:
            counts_df, _ = load_prepare_outputs(paths)
        run_generate_missing(args=args, paths=paths, counts_df=counts_df)
    if args.stage in {"audit", "all"}:
        run_audit(args=args, paths=paths, counts_df=counts_df)


if __name__ == "__main__":
    main()
