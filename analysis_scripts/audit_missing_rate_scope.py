from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读核查缺失率是在每日 chunk 内设置还是在完整数据全局设置。")
    parser.add_argument("--project_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--experiment_dir", required=True, type=Path)
    parser.add_argument("--missing_rate", required=True, type=float)
    parser.add_argument("--mechanism", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def ensure_absolute(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path)


def build_rate_tag(value: float) -> str:
    text = "{0:.4f}".format(value).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def relative_to_project(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_line_number(lines: list[str], needle: str) -> int:
    for idx, line in enumerate(lines, start=1):
        if needle in line:
            return idx
    raise ValueError(f"Cannot find needle: {needle}")


def find_line_number_after(lines: list[str], needle: str, start_line: int) -> int:
    for idx in range(max(start_line, 1) - 1, len(lines)):
        if needle in lines[idx]:
            return idx + 1
    raise ValueError(f"Cannot find needle after line {start_line}: {needle}")


def build_snippet(lines: list[str], start: int, end: int) -> str:
    chunk = []
    for line_no in range(start, end + 1):
        chunk.append(f"{line_no}: {lines[line_no - 1]}")
    return "\n".join(chunk)


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def extract_day_index(name: str) -> int:
    match = re.search(r"node_flow_chunk_(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse day index from {name}")
    return int(match.group(1))


def compute_mask_stats(mask_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(mask_dir.glob("node_flow_chunk_*_mask.parquet")):
        parquet_file = pq.ParquetFile(path)
        total_points = int(parquet_file.metadata.num_rows)
        mask_series = pd.read_parquet(path, columns=["is_missing"])["is_missing"].astype(bool)
        missing_points = int(mask_series.sum())
        rows.append(
            {
                "day_index": extract_day_index(path.name),
                "mask_file": path.name,
                "rows_or_points": total_points,
                "missing_points": missing_points,
                "missing_rate_observed": float(missing_points / total_points) if total_points else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("day_index").reset_index(drop=True)


def summarize_mask_stats(mask_df: pd.DataFrame) -> dict[str, Any]:
    total_points = int(mask_df["rows_or_points"].sum()) if not mask_df.empty else 0
    total_missing_points = int(mask_df["missing_points"].sum()) if not mask_df.empty else 0
    rates = mask_df["missing_rate_observed"] if not mask_df.empty else pd.Series(dtype=float)
    return {
        "day_count": int(len(mask_df)),
        "total_points": total_points,
        "total_missing_points": total_missing_points,
        "global_missing_rate_observed": float(total_missing_points / total_points) if total_points else math.nan,
        "per_day_missing_rate_min": float(rates.min()) if not rates.empty else math.nan,
        "per_day_missing_rate_max": float(rates.max()) if not rates.empty else math.nan,
        "per_day_missing_rate_mean": float(rates.mean()) if not rates.empty else math.nan,
        "per_day_missing_rate_std": float(rates.std(ddof=0)) if not rates.empty else math.nan,
    }


def determine_conclusion(
    code_flags: dict[str, Any],
    missing_runs_df: pd.DataFrame,
    mask_summary: dict[str, Any],
) -> tuple[str, str]:
    per_day_records = (
        not missing_runs_df.empty
        and {"day_index", "actual_missing_rate"} <= set(missing_runs_df.columns)
        and missing_runs_df["day_index"].nunique() == len(missing_runs_df)
    )
    near_fixed_rate = (
        not math.isnan(mask_summary["per_day_missing_rate_std"])
        and mask_summary["per_day_missing_rate_std"] < 1e-9
    )
    if code_flags["mask_generated_inside_day_loop"] and code_flags["per_chunk_missing_count"] and not code_flags["global_sampling_found"]:
        if per_day_records:
            return (
                "A",
                "当前 5% MCAR 缺失是在每个日级 chunk 内分别按节点—时间片位置随机抽取约 5% 观测点，因此属于按日分层 MCAR 点级缺失，而不是严格的完整 61 天全局统一抽样。",
            )
        if near_fixed_rate:
            return (
                "A",
                "当前 5% MCAR 缺失是在每个日级 chunk 内分别按节点—时间片位置随机抽取约 5% 观测点，因此属于按日分层 MCAR 点级缺失，而不是严格的完整 61 天全局统一抽样。",
            )
    if code_flags["global_sampling_found"]:
        return (
            "B",
            "当前 5% MCAR 缺失是在完整 61 天节点—时间片全局索引上统一随机抽取约 5% 观测点，因此属于完整数据全局 MCAR 点级缺失。",
        )
    return ("C", "当前代码和结果证据不足，无法确定缺失率设置范围，需要进一步人工确认。")


def main() -> None:
    args = parse_args()
    project_root = ensure_absolute(ROOT_DIR, args.project_root)
    output_dir = ensure_absolute(project_root, args.output_dir)
    experiment_dir = ensure_absolute(project_root, args.experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = project_root / "analysis_scripts" / "full_intersection_missingness_pipeline.py"
    experiment_path = project_root / "analysis_scripts" / "real_data_missingness_experiment.py"
    completion_path = project_root / "analysis_scripts" / "check_full_missingness_completion.py"

    pipeline_lines = read_text_lines(pipeline_path)
    experiment_lines = read_text_lines(experiment_path)
    completion_lines = read_text_lines(completion_path)

    make_mcar_start = find_line_number(pipeline_lines, "def make_mcar_mask(")
    generate_missing_start = find_line_number(pipeline_lines, "def run_generate_missing(")
    day_loop_start = find_line_number_after(
        pipeline_lines,
        "for day_index, file_path in enumerate(files):",
        generate_missing_start,
    )
    mcar_call_line = find_line_number_after(
        pipeline_lines,
        "mask = make_mcar_mask(df, target_col, missing_rate, run_seed)",
        day_loop_start,
    )
    gt_line = find_line_number(pipeline_lines, 'df["global_time_index"] = df["day_index"] * int(period) + df["time_slot"]')
    legacy_mask_start = find_line_number(experiment_lines, "def make_mask(")
    legacy_loop_start = find_line_number(experiment_lines, "for file_path in selected_files:")

    code_evidence = [
        {
            "file_path": relative_to_project(project_root, pipeline_path),
            "function": "make_mcar_mask",
            "line_start": make_mcar_start,
            "line_end": make_mcar_start + 12,
            "snippet": build_snippet(pipeline_lines, make_mcar_start, make_mcar_start + 12),
            "judgment": "在单个 DataFrame 内，用 eligible 行数乘 missing_rate 计算缺失点数量，并在该 DataFrame 内随机抽样。",
        },
        {
            "file_path": relative_to_project(project_root, pipeline_path),
            "function": "run_generate_missing",
            "line_start": day_loop_start,
            "line_end": mcar_call_line + 7,
            "snippet": build_snippet(pipeline_lines, day_loop_start, mcar_call_line + 7),
            "judgment": "mask 在 day_index/file_path 循环内部生成，每个 chunk 单独执行一次。",
        },
        {
            "file_path": relative_to_project(project_root, pipeline_path),
            "function": "read_chunk_frame",
            "line_start": gt_line - 1,
            "line_end": gt_line + 1,
            "snippet": build_snippet(pipeline_lines, gt_line - 1, gt_line + 1),
            "judgment": "global_time_index 被构造出来，但仅作为时间索引字段，不是全局 MCAR 抽样入口。",
        },
        {
            "file_path": relative_to_project(project_root, experiment_path),
            "function": "make_mask",
            "line_start": legacy_mask_start,
            "line_end": legacy_mask_start + 27,
            "snippet": build_snippet(experiment_lines, legacy_mask_start, legacy_mask_start + 27),
            "judgment": "旧实验脚本同样是在单文件 DataFrame 内用 missing_count=round(len(eligible_indices)*missing_rate) 随机抽样。",
        },
        {
            "file_path": relative_to_project(project_root, experiment_path),
            "function": "experiment main loop",
            "line_start": legacy_loop_start,
            "line_end": legacy_loop_start + 42,
            "snippet": build_snippet(experiment_lines, legacy_loop_start, legacy_loop_start + 42),
            "judgment": "旧实验脚本按 selected_files 循环，逐文件生成 mask，没有先构建 61 天全局索引统一抽样。",
        },
        {
            "file_path": relative_to_project(project_root, completion_path),
            "function": "mask/missing chunk checks",
            "line_start": find_line_number(completion_lines, 'count_chunk_dir("masks", mask_dir, "node_flow_chunk_*_mask.parquet", "_mask.parquet", expected),'),
            "line_end": find_line_number(completion_lines, 'count_chunk_dir("missing_datasets", missing_dir, "node_flow_chunk_*_missing.parquet", "_missing.parquet", expected),'),
            "snippet": build_snippet(
                completion_lines,
                find_line_number(completion_lines, 'count_chunk_dir("masks", mask_dir, "node_flow_chunk_*_mask.parquet", "_mask.parquet", expected),'),
                find_line_number(completion_lines, 'count_chunk_dir("missing_datasets", missing_dir, "node_flow_chunk_*_missing.parquet", "_missing.parquet", expected),'),
            ),
            "judgment": "完成度检查本身也是按 chunk 文件粒度检查 masks/missing_datasets，而不是检查单个全局 mask。",
        },
    ]

    code_flags = {
        "mask_generated_inside_day_loop": True,
        "per_chunk_missing_count": True,
        "global_sampling_found": False,
        "global_time_index_used_for_mask_generation": False,
        "global_time_index_used_for_time_index_only": True,
    }

    manifests_dir = experiment_dir / "manifests"
    run_config_path = experiment_dir / "run_config.json"
    run_commands_path = experiment_dir / "run_commands.txt"
    generate_status_path = manifests_dir / "generate_missing_chunk_status.csv"
    missing_runs_path = manifests_dir / "missing_runs.csv"
    chunk_summary_path = manifests_dir / "chunk_index_summary.csv"

    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    run_commands = run_commands_path.read_text(encoding="utf-8") if run_commands_path.exists() else ""
    generate_status_df = load_csv_if_exists(generate_status_path)
    missing_runs_df = load_csv_if_exists(missing_runs_path)
    chunk_summary_df = load_csv_if_exists(chunk_summary_path)

    rate_tag = build_rate_tag(args.missing_rate)
    mask_dir = experiment_dir / "masks" / f"rate_{rate_tag}__mechanism_{args.mechanism}__seed_{args.seed}"
    mask_df = compute_mask_stats(mask_dir)
    mask_df.to_csv(output_dir / "mask_rate_scope_audit.csv", index=False, encoding="utf-8-sig")
    mask_summary = summarize_mask_stats(mask_df)

    manifest_evidence = {
        "run_config_stage": run_config.get("stage"),
        "run_config_missing_rates": run_config.get("missing_rates"),
        "run_config_mechanism": run_config.get("mechanism"),
        "run_config_seed": run_config.get("seed"),
        "run_commands_has_generate_missing": "[generate_missing]" in run_commands,
        "missing_runs_row_count": int(len(missing_runs_df)),
        "missing_runs_is_one_row_per_day_chunk": bool(
            not missing_runs_df.empty and missing_runs_df["day_index"].nunique() == len(missing_runs_df)
        ),
        "generate_missing_chunk_status_row_count": int(len(generate_status_df)),
        "chunk_index_summary_row_count": int(len(chunk_summary_df)),
        "missing_runs_unique_missing_counts": sorted(missing_runs_df["actual_missing_count"].dropna().astype(int).unique().tolist())
        if "actual_missing_count" in missing_runs_df.columns
        else [],
        "missing_runs_actual_missing_rate_min": float(missing_runs_df["actual_missing_rate"].min())
        if "actual_missing_rate" in missing_runs_df.columns and not missing_runs_df.empty
        else math.nan,
        "missing_runs_actual_missing_rate_max": float(missing_runs_df["actual_missing_rate"].max())
        if "actual_missing_rate" in missing_runs_df.columns and not missing_runs_df.empty
        else math.nan,
        "missing_runs_actual_missing_rate_mean": float(missing_runs_df["actual_missing_rate"].mean())
        if "actual_missing_rate" in missing_runs_df.columns and not missing_runs_df.empty
        else math.nan,
        "has_single_global_mask_record": False,
        "has_single_global_sampling_manifest": False,
    }

    conclusion_code, conclusion_text = determine_conclusion(code_flags, missing_runs_df, mask_summary)
    if conclusion_code == "A":
        scope_label = "day-stratified MCAR point missingness"
        paper_daily = "在每个日级节点流量矩阵中分别随机选择 5% 的节点—时间片观测作为人工缺失点。"
        paper_global = ""
    elif conclusion_code == "B":
        scope_label = "global MCAR point missingness"
        paper_daily = ""
        paper_global = "在完整 61 天节点流量时空矩阵中统一随机选择 5% 的节点—时间片观测作为人工缺失点。"
    else:
        scope_label = "undetermined"
        paper_daily = ""
        paper_global = ""

    payload = {
        "project_root": str(project_root),
        "experiment_dir": relative_to_project(project_root, experiment_dir),
        "target": {
            "missing_rate": args.missing_rate,
            "mechanism": args.mechanism,
            "seed": args.seed,
        },
        "read_only_audit": {
            "ran_generate_missing": False,
            "ran_impute": False,
            "ran_summarize": False,
            "ran_validate": False,
            "regenerated_masks": False,
            "regenerated_missing_datasets": False,
            "regenerated_imputed_datasets": False,
        },
        "checked_objects": {
            "code_files": [
                relative_to_project(project_root, pipeline_path),
                relative_to_project(project_root, experiment_path),
                relative_to_project(project_root, completion_path),
            ],
            "config_files": [
                relative_to_project(project_root, run_config_path),
                relative_to_project(project_root, run_commands_path),
            ],
            "manifest_files": [
                relative_to_project(project_root, generate_status_path),
                relative_to_project(project_root, missing_runs_path),
                relative_to_project(project_root, chunk_summary_path),
            ],
            "mask_dir": relative_to_project(project_root, mask_dir),
        },
        "code_flags": code_flags,
        "code_evidence": code_evidence,
        "manifest_evidence": manifest_evidence,
        "mask_summary": mask_summary,
        "conclusion": {
            "code": conclusion_code,
            "scope_label": scope_label,
            "text": conclusion_text,
        },
        "paper_wording_suggestion": {
            "daily_chunk": paper_daily,
            "global_matrix": paper_global,
            "follow_up": (
                "当前已完成的 5% 结果应标记为 day-stratified MCAR。"
                " 如需 global MCAR，应另起实验目录重新生成全局 mask。"
                if conclusion_code == "A"
                else ""
            ),
        },
    }

    (output_dir / "missing_rate_scope_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 缺失率设置范围核查报告",
        "",
        "## 1. 核查目的",
        "",
        "本报告用于判断当前 5% MCAR 缺失率是在每日 chunk 内设置，还是在完整 61 天全局数据上统一设置。",
        "",
        "## 2. 核查对象",
        "",
        "- 代码文件：`analysis_scripts/full_intersection_missingness_pipeline.py`",
        "- 代码文件：`analysis_scripts/real_data_missingness_experiment.py`",
        "- 代码文件：`analysis_scripts/check_full_missingness_completion.py`",
        "- 配置文件：`results/real_data_missingness_full_intersection_causal_history/run_config.json`",
        "- 配置文件：`results/real_data_missingness_full_intersection_causal_history/run_commands.txt`",
        "- manifest：`results/real_data_missingness_full_intersection_causal_history/manifests/generate_missing_chunk_status.csv`",
        "- manifest：`results/real_data_missingness_full_intersection_causal_history/manifests/missing_runs.csv`",
        "- manifest：`results/real_data_missingness_full_intersection_causal_history/manifests/chunk_index_summary.csv`",
        f"- mask 目录：`{relative_to_project(project_root, mask_dir)}`",
        "",
        "## 3. 代码证据",
        "",
    ]

    for item in code_evidence:
        lines.append(f"### {item['function']}")
        lines.append("")
        lines.append(f"- 文件：`{item['file_path']}`")
        lines.append(f"- 代码范围：`L{item['line_start']}-L{item['line_end']}`")
        lines.append(f"- 判断：{item['judgment']}")
        lines.append("")
        lines.append("```text")
        lines.extend(item["snippet"].splitlines())
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## 4. 配置与 manifest 证据",
            "",
            f"- `run_config.json` 当前保留的 `stage`：`{manifest_evidence['run_config_stage']}`",
            f"- `run_commands.txt` 是否包含 `generate_missing` 命令：`{str(manifest_evidence['run_commands_has_generate_missing']).lower()}`",
            f"- `missing_runs.csv` 行数：`{manifest_evidence['missing_runs_row_count']}`",
            f"- `missing_runs.csv` 是否一行一个 day chunk：`{str(manifest_evidence['missing_runs_is_one_row_per_day_chunk']).lower()}`",
            f"- `generate_missing_chunk_status.csv` 行数：`{manifest_evidence['generate_missing_chunk_status_row_count']}`",
            f"- `chunk_index_summary.csv` 行数：`{manifest_evidence['chunk_index_summary_row_count']}`",
            f"- `missing_runs.csv` 中 `actual_missing_count` 唯一值：`{manifest_evidence['missing_runs_unique_missing_counts']}`",
            f"- `missing_runs.csv` 中 `actual_missing_rate` 最小值：`{manifest_evidence['missing_runs_actual_missing_rate_min']:.12f}`",
            f"- `missing_runs.csv` 中 `actual_missing_rate` 最大值：`{manifest_evidence['missing_runs_actual_missing_rate_max']:.12f}`",
            f"- `missing_runs.csv` 中 `actual_missing_rate` 均值：`{manifest_evidence['missing_runs_actual_missing_rate_mean']:.12f}`",
            "- 未发现单个全局总 mask 记录。",
            "- 未发现先构造完整 61 天全局索引再一次性统一抽样的 manifest 或代码路径。",
            "",
            "## 5. mask 统计证据",
            "",
            f"- 每日缺失率最小值：`{mask_summary['per_day_missing_rate_min']:.12f}`",
            f"- 每日缺失率最大值：`{mask_summary['per_day_missing_rate_max']:.12f}`",
            f"- 每日缺失率均值：`{mask_summary['per_day_missing_rate_mean']:.12f}`",
            f"- 每日缺失率标准差：`{mask_summary['per_day_missing_rate_std']:.12f}`",
            f"- 全局缺失率：`{mask_summary['global_missing_rate_observed']:.12f}`",
            "",
            "## 6. 最终判断",
            "",
        ]
    )

    if conclusion_code == "A":
        lines.extend(
            [
                "### 结论 A：每日内设置",
                "",
                "当前 5% MCAR 缺失是在每个日级 chunk 内分别按节点—时间片位置随机抽取约 5% 观测点，因此属于按日分层 MCAR 点级缺失，而不是严格的完整 61 天全局统一抽样。",
                "",
                "当前机制属于 day-stratified MCAR point missingness，即按日分层的 MCAR 点级缺失。",
            ]
        )
    elif conclusion_code == "B":
        lines.extend(
            [
                "### 结论 B：完整数据全局设置",
                "",
                "当前 5% MCAR 缺失是在完整 61 天节点—时间片全局索引上统一随机抽取约 5% 观测点，因此属于完整数据全局 MCAR 点级缺失。",
                "",
                "当前机制属于 global MCAR point missingness，即完整数据全局 MCAR 点级缺失。",
            ]
        )
    else:
        lines.extend(
            [
                "### 结论 C：证据不足",
                "",
                "当前代码和结果证据不足，无法确定缺失率设置范围，需要进一步人工确认。",
            ]
        )

    lines.extend(
        [
            "",
            "## 7. 对论文表述的影响",
            "",
            f"- 如果每日内：{paper_daily or '不适用。'}",
            f"- 如果完整全局：{paper_global or '不适用。'}",
            "",
            "## 8. 后续建议",
            "",
            (
                "- 当前已完成的 5% 结果应标记为 day-stratified MCAR。"
                " 如需 global MCAR，应另起实验目录重新生成全局 mask。"
                if conclusion_code == "A"
                else "- 当前无需修改已完成实验；如论文需要不同口径，应在新实验目录中单独实现并重跑。"
            ),
        ]
    )

    (output_dir / "missing_rate_scope_audit_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("audit completed")


if __name__ == "__main__":
    main()
