from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_METHODS = [
    "zero_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "geo_neighbor_fill",
    "function_curve_fit",
    "geo_func_hybrid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读重建完整路口缺失补全 manifests 与 detail 汇总。")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--missing_rate", required=True, type=float)
    parser.add_argument("--mechanism", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT_DIR / args.output_dir
    return args


def build_rate_tag(value: float) -> str:
    text = "{0:.4f}".format(value).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def get_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def extract_day_index(file_name: str) -> int:
    match = re.search(r"node_flow_chunk_(\d+)", file_name)
    if not match:
        raise ValueError(f"无法从文件名解析 chunk 编号: {file_name}")
    return int(match.group(1))


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def load_chunk_manifest(output_dir: Path) -> pd.DataFrame:
    chunk_path = output_dir / "manifests" / "chunk_index_summary.csv"
    if not chunk_path.exists():
        raise FileNotFoundError(f"缺少 chunk 清单: {chunk_path}")
    chunk_df = pd.read_csv(chunk_path)
    required = {"day_index", "file_name", "warmup"}
    missing = required - set(chunk_df.columns)
    if missing:
        raise ValueError(f"chunk_index_summary.csv 缺少字段: {sorted(missing)}")
    chunk_df = chunk_df.copy()
    chunk_df["day_index"] = chunk_df["day_index"].astype(int)
    chunk_df["file_name"] = chunk_df["file_name"].astype(str)
    chunk_df["warmup"] = chunk_df["warmup"].apply(to_bool)
    return chunk_df[["day_index", "file_name", "warmup"]].sort_values("day_index").reset_index(drop=True)


def load_detail_file(detail_path: Path, warmup_by_day: dict[int, bool]) -> pd.DataFrame:
    detail_df = pd.read_csv(detail_path)
    required = {"missing_rate", "mechanism", "seed", "impute_method", "day_index", "file_name", "flow_group", "count"}
    missing = required - set(detail_df.columns)
    if missing:
        raise ValueError(f"detail 文件缺少字段 {sorted(missing)}: {detail_path}")
    detail_df = detail_df.copy()
    detail_df["day_index"] = detail_df["day_index"].astype(int)
    detail_df["file_name"] = detail_df["file_name"].astype(str)
    detail_df["is_warmup"] = detail_df["day_index"].map(warmup_by_day).fillna(False)
    ordered_columns = [
        "missing_rate",
        "mechanism",
        "seed",
        "impute_method",
        "day_index",
        "file_name",
        "is_warmup",
        "flow_group",
        "count",
        "sum_abs_error",
        "sum_squared_error",
        "sum_pct_error",
        "count_pct",
        "sum_smape",
    ]
    missing_optional = [column for column in ordered_columns if column not in detail_df.columns]
    if missing_optional:
        raise ValueError(f"detail 文件缺少汇总字段 {missing_optional}: {detail_path}")
    return detail_df[ordered_columns]


def build_paths(output_dir: Path, rate_tag: str, mechanism: str, seed: int, method: str, day_index: int) -> dict[str, Path]:
    chunk_name = f"node_flow_chunk_{day_index:03d}"
    return {
        "mask": output_dir / "masks" / f"rate_{rate_tag}__mechanism_{mechanism}__seed_{seed}" / f"{chunk_name}_mask.parquet",
        "missing": output_dir / "missing_datasets" / f"rate_{rate_tag}__mechanism_{mechanism}__seed_{seed}" / f"{chunk_name}_missing.parquet",
        "imputed": output_dir
        / "imputed_datasets"
        / f"rate_{rate_tag}__mechanism_{mechanism}__seed_{seed}__method_{method}"
        / f"{chunk_name}_imputed.parquet",
        "detail": output_dir
        / "manifests"
        / "detail_runs"
        / f"rate_{rate_tag}__mechanism_{mechanism}__seed_{seed}__method_{method}"
        / f"{chunk_name}_detail.csv",
    }


def rebuild(output_dir: Path, missing_rate: float, mechanism: str, seed: int) -> dict[str, int]:
    manifests_dir = output_dir / "manifests"
    summaries_dir = output_dir / "summaries"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    rate_tag = build_rate_tag(missing_rate)
    chunk_df = load_chunk_manifest(output_dir)
    warmup_by_day = dict(zip(chunk_df["day_index"], chunk_df["warmup"]))
    file_name_by_day = dict(zip(chunk_df["day_index"], chunk_df["file_name"]))

    detail_frames: list[pd.DataFrame] = []
    imputation_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method in TARGET_METHODS:
        method_detail_dir = manifests_dir / "detail_runs" / f"rate_{rate_tag}__mechanism_{mechanism}__seed_{seed}__method_{method}"
        detail_paths = sorted(method_detail_dir.glob("node_flow_chunk_*_detail.csv"))
        completed_chunk_count = 0

        for detail_path in detail_paths:
            detail_df = load_detail_file(detail_path, warmup_by_day)
            detail_frames.append(detail_df)

            day_index = extract_day_index(detail_path.name)
            file_name = file_name_by_day.get(day_index, detail_df["file_name"].iloc[0])
            all_row = detail_df.loc[detail_df["flow_group"] == "all"]
            actual_missing_count = int(round(float(all_row["count"].iloc[0]))) if not all_row.empty else 0

            paths = build_paths(output_dir, rate_tag, mechanism, seed, method, day_index)
            mask_exists = paths["mask"].exists()
            missing_exists = paths["missing"].exists()
            imputed_exists = paths["imputed"].exists()
            detail_exists = paths["detail"].exists()
            filled_missing_count = actual_missing_count if imputed_exists and detail_exists else 0
            residual_missing_count = 0 if imputed_exists and detail_exists else actual_missing_count
            status = "completed" if imputed_exists and detail_exists else "missing_artifact"
            note = "只读重建：基于已有 detail 与 imputed 文件登记"

            imputation_rows.append(
                {
                    "missing_rate": missing_rate,
                    "mechanism": mechanism,
                    "seed": seed,
                    "impute_method": method,
                    "day_index": day_index,
                    "file_name": file_name,
                    "imputed_dataset_path": get_relative_path(paths["imputed"]),
                    "filled_missing_count": filled_missing_count,
                    "residual_missing_count": residual_missing_count,
                    "detail_path": get_relative_path(paths["detail"]),
                }
            )
            status_rows.append(
                {
                    "stage": "impute",
                    "missing_rate": missing_rate,
                    "mechanism": mechanism,
                    "seed": seed,
                    "impute_method": method,
                    "day_index": day_index,
                    "file_name": file_name,
                    "status": status,
                    "mask_exists": mask_exists,
                    "missing_dataset_exists": missing_exists,
                    "imputed_dataset_exists": imputed_exists,
                    "detail_exists": detail_exists,
                    "actual_missing_count": actual_missing_count,
                    "filled_missing_count": filled_missing_count,
                    "residual_missing_count": residual_missing_count,
                    "note": note,
                }
            )
            if status == "completed":
                completed_chunk_count += 1

        summary_rows.append(
            {
                "stage": "impute",
                "missing_rate": missing_rate,
                "mechanism": mechanism,
                "seed": seed,
                "impute_method": method,
                "status": "completed" if completed_chunk_count == len(chunk_df) else "partial",
                "chunk_count": completed_chunk_count,
            }
        )

    if not detail_frames:
        raise FileNotFoundError("未找到任何 detail_runs CSV，无法重建 summaries/imputation_quality_detail.csv。")

    detail_all = pd.concat(detail_frames, ignore_index=True)
    detail_all = detail_all.sort_values(["missing_rate", "mechanism", "seed", "impute_method", "day_index", "flow_group"]).reset_index(drop=True)
    imputation_runs = pd.DataFrame(imputation_rows).sort_values(["missing_rate", "mechanism", "seed", "impute_method", "day_index"]).reset_index(drop=True)
    impute_status = pd.DataFrame(status_rows).sort_values(["missing_rate", "mechanism", "seed", "impute_method", "day_index"]).reset_index(drop=True)
    impute_stage = pd.DataFrame(summary_rows).sort_values(["missing_rate", "mechanism", "seed", "impute_method"]).reset_index(drop=True)

    detail_all.to_csv(summaries_dir / "imputation_quality_detail.csv", index=False, encoding="utf-8-sig")
    imputation_runs.to_csv(manifests_dir / "imputation_runs.csv", index=False, encoding="utf-8-sig")
    impute_status.to_csv(manifests_dir / "impute_chunk_status.csv", index=False, encoding="utf-8-sig")
    impute_stage.to_csv(manifests_dir / "impute_stage_summary.csv", index=False, encoding="utf-8-sig")

    return {
        "detail_rows": int(len(detail_all)),
        "imputation_runs_rows": int(len(imputation_runs)),
        "impute_chunk_status_rows": int(len(impute_status)),
        "impute_stage_summary_rows": int(len(impute_stage)),
    }


def main() -> None:
    args = normalize_args(parse_args())
    counts = rebuild(args.output_dir, args.missing_rate, args.mechanism, args.seed)
    print("rebuild completed")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
