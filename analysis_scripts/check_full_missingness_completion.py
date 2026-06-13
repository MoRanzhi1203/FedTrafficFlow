from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
METHODS = [
    "zero_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "geo_neighbor_fill",
    "function_curve_fit",
    "geo_func_hybrid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复核 61 chunk 历史因果缺失补全主实验完成度。")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_chunks", required=True, type=int)
    parser.add_argument("--missing_rate", required=True, type=float)
    parser.add_argument("--mechanism", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT_DIR / args.output_dir
    return args


def get_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_rate_tag(value: float) -> str:
    text = "{0:.4f}".format(value).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def list_chunk_indices(paths: List[Path], suffix_token: str) -> List[int]:
    chunk_ids: List[int] = []
    for path in paths:
        name = path.name
        if not name.startswith("node_flow_chunk_") or suffix_token not in name:
            continue
        try:
            number = name.split("node_flow_chunk_")[1].split(suffix_token)[0]
            chunk_ids.append(int(number))
        except Exception:
            continue
    return sorted(set(chunk_ids))


def expected_chunk_set(expected_chunks: int) -> Set[int]:
    return set(range(expected_chunks))


def summarize_method(
    output_dir: Path,
    rate_tag: str,
    mechanism: str,
    seed: int,
    method: str,
    expected: Set[int],
) -> Dict[str, Any]:
    imputed_dir = output_dir / "imputed_datasets" / (
        "rate_{0}__mechanism_{1}__seed_{2}__method_{3}".format(rate_tag, mechanism, seed, method)
    )
    detail_dir = output_dir / "manifests" / "detail_runs" / (
        "rate_{0}__mechanism_{1}__seed_{2}__method_{3}".format(rate_tag, mechanism, seed, method)
    )
    imputed_files = sorted(imputed_dir.glob("node_flow_chunk_*_imputed.parquet")) if imputed_dir.exists() else []
    detail_files = sorted(detail_dir.glob("node_flow_chunk_*_detail.csv")) if detail_dir.exists() else []
    imputed_chunks = set(list_chunk_indices(imputed_files, "_imputed.parquet"))
    detail_chunks = set(list_chunk_indices(detail_files, "_detail.csv"))
    missing_imputed = sorted(expected - imputed_chunks)
    missing_detail = sorted(expected - detail_chunks)
    return {
        "method": method,
        "imputed_dir_exists": imputed_dir.exists(),
        "detail_dir_exists": detail_dir.exists(),
        "imputed_count": len(imputed_chunks),
        "detail_count": len(detail_chunks),
        "missing_imputed_chunks": json.dumps(missing_imputed, ensure_ascii=False),
        "missing_detail_chunks": json.dumps(missing_detail, ensure_ascii=False),
        "complete_61_of_61": len(imputed_chunks) == len(expected) and len(detail_chunks) == len(expected),
        "imputed_dir": get_relative_path(imputed_dir),
        "detail_dir": get_relative_path(detail_dir),
    }


def file_check_row(label: str, path: Path) -> Dict[str, Any]:
    return {
        "category": "artifact_check",
        "name": label,
        "exists": path.exists(),
        "count": "",
        "expected": "",
        "missing_chunks": "",
        "details": get_relative_path(path),
    }


def count_chunk_dir(label: str, path: Path, pattern: str, suffix_token: str, expected: Set[int]) -> Dict[str, Any]:
    files = sorted(path.glob(pattern)) if path.exists() else []
    chunks = set(list_chunk_indices(files, suffix_token))
    missing = sorted(expected - chunks)
    return {
        "category": "chunk_count",
        "name": label,
        "exists": path.exists(),
        "count": len(chunks),
        "expected": len(expected),
        "missing_chunks": json.dumps(missing, ensure_ascii=False),
        "details": get_relative_path(path),
    }


def render_markdown(
    output_dir: Path,
    expected_count: int,
    chunk_rows: List[Dict[str, Any]],
    method_rows: List[Dict[str, Any]],
    file_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Completion Check Before Repair")
    lines.append("")
    lines.append("- 输出目录：`{0}`".format(get_relative_path(output_dir)))
    lines.append("- 预期 chunk 数：`{0}`".format(expected_count))
    lines.append("")
    lines.append("## Chunk 级资产")
    lines.append("")
    for row in chunk_rows:
        lines.append(
            "- {0}：`{1}/{2}`，缺失 chunk：`{3}`".format(
                row["name"], row["count"], row["expected"], row["missing_chunks"]
            )
        )
    lines.append("")
    lines.append("## 方法覆盖")
    lines.append("")
    for row in method_rows:
        lines.append(
            "- `{0}`：imputed=`{1}`，detail=`{2}`，缺失 imputed chunk=`{3}`".format(
                row["method"], row["imputed_count"], row["detail_count"], row["missing_imputed_chunks"]
            )
        )
    lines.append("")
    lines.append("## 关键文件")
    lines.append("")
    for row in file_rows:
        lines.append("- {0}：`{1}`".format(row["name"], str(row["exists"]).lower()))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = normalize_args(parse_args())
    output_dir = args.output_dir
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rate_tag = build_rate_tag(args.missing_rate)
    expected = expected_chunk_set(args.expected_chunks)

    input_df = load_csv_if_exists(manifests_dir / "input_files.csv")
    chunk_df = load_csv_if_exists(manifests_dir / "chunk_index_summary.csv")
    generate_status_df = load_csv_if_exists(manifests_dir / "generate_missing_chunk_status.csv")
    impute_status_df = load_csv_if_exists(manifests_dir / "impute_chunk_status.csv")
    imputation_runs_df = load_csv_if_exists(manifests_dir / "imputation_runs.csv")
    detail_df = load_csv_if_exists(output_dir / "summaries" / "imputation_quality_detail.csv")

    mask_dir = output_dir / "masks" / "rate_{0}__mechanism_{1}__seed_{2}".format(rate_tag, args.mechanism, args.seed)
    missing_dir = output_dir / "missing_datasets" / "rate_{0}__mechanism_{1}__seed_{2}".format(rate_tag, args.mechanism, args.seed)

    chunk_rows = [
        {
            "category": "chunk_count",
            "name": "input_chunks",
            "exists": not input_df.empty,
            "count": int(len(input_df)),
            "expected": args.expected_chunks,
            "missing_chunks": json.dumps(sorted(expected - set(range(len(input_df)))), ensure_ascii=False),
            "details": get_relative_path(manifests_dir / "input_files.csv"),
        },
        count_chunk_dir("masks", mask_dir, "node_flow_chunk_*_mask.parquet", "_mask.parquet", expected),
        count_chunk_dir("missing_datasets", missing_dir, "node_flow_chunk_*_missing.parquet", "_missing.parquet", expected),
    ]

    method_rows = [
        summarize_method(output_dir, rate_tag, args.mechanism, args.seed, method, expected)
        for method in METHODS
    ]

    file_rows = [
        file_check_row("generate_missing_chunk_status.csv", manifests_dir / "generate_missing_chunk_status.csv"),
        file_check_row("impute_chunk_status.csv", manifests_dir / "impute_chunk_status.csv"),
        file_check_row("imputation_runs.csv", manifests_dir / "imputation_runs.csv"),
        file_check_row("imputation_quality_detail.csv", output_dir / "summaries" / "imputation_quality_detail.csv"),
        file_check_row("imputation_quality_summary_all_days.csv", output_dir / "summaries" / "imputation_quality_summary_all_days.csv"),
        file_check_row("imputation_quality_summary_exclude_warmup.csv", output_dir / "summaries" / "imputation_quality_summary_exclude_warmup.csv"),
        file_check_row("full_intersection_missingness_validation.json", output_dir / "full_intersection_missingness_validation.json"),
        file_check_row("full_intersection_missingness_audit.json", output_dir / "full_intersection_missingness_audit.json"),
        file_check_row("figures", output_dir / "figures"),
    ]

    extra_rows: List[Dict[str, Any]] = [
        {
            "category": "manifest_rows",
            "name": "chunk_index_summary_rows",
            "exists": not chunk_df.empty,
            "count": int(len(chunk_df)),
            "expected": args.expected_chunks,
            "missing_chunks": "",
            "details": get_relative_path(manifests_dir / "chunk_index_summary.csv"),
        },
        {
            "category": "manifest_rows",
            "name": "generate_missing_chunk_status_rows",
            "exists": not generate_status_df.empty,
            "count": int(len(generate_status_df)),
            "expected": "",
            "missing_chunks": "",
            "details": get_relative_path(manifests_dir / "generate_missing_chunk_status.csv"),
        },
        {
            "category": "manifest_rows",
            "name": "impute_chunk_status_rows",
            "exists": not impute_status_df.empty,
            "count": int(len(impute_status_df)),
            "expected": "",
            "missing_chunks": "",
            "details": get_relative_path(manifests_dir / "impute_chunk_status.csv"),
        },
        {
            "category": "manifest_rows",
            "name": "imputation_runs_rows",
            "exists": not imputation_runs_df.empty,
            "count": int(len(imputation_runs_df)),
            "expected": "",
            "missing_chunks": "",
            "details": get_relative_path(manifests_dir / "imputation_runs.csv"),
        },
        {
            "category": "manifest_rows",
            "name": "imputation_quality_detail_rows",
            "exists": not detail_df.empty,
            "count": int(len(detail_df)),
            "expected": "",
            "missing_chunks": "",
            "details": get_relative_path(output_dir / "summaries" / "imputation_quality_detail.csv"),
        },
    ]

    rows: List[Dict[str, Any]] = []
    rows.extend(chunk_rows)
    rows.extend(
        [
            {
                "category": "method_completion",
                "name": row["method"],
                "exists": row["imputed_dir_exists"] or row["detail_dir_exists"],
                "count": row["imputed_count"],
                "expected": args.expected_chunks,
                "missing_chunks": row["missing_imputed_chunks"],
                "details": "detail_count={0}; detail_missing={1}".format(
                    row["detail_count"], row["missing_detail_chunks"]
                ),
            }
            for row in method_rows
        ]
    )
    rows.extend(file_rows)
    rows.extend(extra_rows)

    csv_path = manifests_dir / "completion_check_before_repair.csv"
    md_path = manifests_dir / "completion_check_before_repair.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(
        render_markdown(output_dir, args.expected_chunks, chunk_rows, method_rows, file_rows),
        encoding="utf-8",
    )

    print("completion_check_before_repair.csv: {0}".format(csv_path))
    print("completion_check_before_repair.md: {0}".format(md_path))
    for row in method_rows:
        print(
            "{0}: imputed={1}, detail={2}, missing_imputed={3}".format(
                row["method"], row["imputed_count"], row["detail_count"], row["missing_imputed_chunks"]
            )
        )


if __name__ == "__main__":
    main()
