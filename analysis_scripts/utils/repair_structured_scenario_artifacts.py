"""修复结构化缺失场景中不完整的 manifest、报告和图像产物。

核心功能：
- 为单个结构化场景重建缺失的 manifest 文件和分布报告；
- 复用结构化缺失分析与 setting 模块中的公共能力；
- 恢复后续整理、可视化所需的一致中间产物。

项目作用：
- 用于修复历史实验目录中缺失或损坏的中间结果；
- 保障实验重组与审计流程可以读取到完整产物。

关键依赖：`pandas`、`numpy` 和结构化缺失辅助模块。
主要输入：单个结构化场景目录及其 `run_config`、状态文件。
主要输出：修复后的 manifest、报告和诊断图像。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from analysis_scripts.missingness.analyze_structured_missingness_distribution import (
        analyze_scenario,
        build_markdown_report,
        discover_complete_scenarios,
        load_original_columns,
        load_prepare_summary,
        resolve_missing_root,
        save_length_group_bar,
        save_rate_bar,
        save_time_slot_heatmap,
        write_json,
    )
    from analysis_scripts.missingness.structured_missingness_setting_pipeline import (
        LengthSamplingConfig,
        ScenarioDefinition,
        build_existing_scenario_summary,
        build_parameter_setting,
        build_paths,
        load_prepare_artifacts,
        scenario_output_name,
        scenario_event_path,
    )
except ImportError:
    from analyze_structured_missingness_distribution import (
        analyze_scenario,
        build_markdown_report,
        discover_complete_scenarios,
        load_original_columns,
        load_prepare_summary,
        resolve_missing_root,
        save_length_group_bar,
        save_rate_bar,
        save_time_slot_heatmap,
        write_json,
    )
    from structured_missingness_setting_pipeline import (
        LengthSamplingConfig,
        ScenarioDefinition,
        build_existing_scenario_summary,
        build_parameter_setting,
        build_paths,
        load_prepare_artifacts,
        scenario_output_name,
        scenario_event_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair per-scenario structured missingness manifests and distribution reports."
    )
    parser.add_argument(
        "--scenario_dir",
        type=Path,
        required=True,
        help="Path to a single structured scenario miss_set directory, e.g. results/rdm_exp/scenarios/nso_mix/miss_set",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/analysis/node_intersection_flow_parquet"),
    )
    parser.add_argument("--target_col", type=str, default="路口车流量")
    parser.add_argument("--node_col", type=str, default="节点ID")
    parser.add_argument("--time_col", type=str, default="时间段")
    parser.add_argument("--period", type=int, default=96)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    scenario_dir = args.scenario_dir.resolve()
    input_dir = args.input_dir.resolve() if args.input_dir.is_absolute() else (Path.cwd() / args.input_dir).resolve()
    if not scenario_dir.exists():
        raise FileNotFoundError(f"scenario_dir not found: {scenario_dir}")
    if not scenario_dir.is_dir():
        raise NotADirectoryError(f"scenario_dir is not a directory: {scenario_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")
    if args.period <= 0:
        raise ValueError("period must be positive")
    for field_name in ["target_col", "node_col", "time_col"]:
        if not str(getattr(args, field_name)).strip():
            raise ValueError(f"{field_name} must not be empty")
    return scenario_dir, input_dir


def load_run_config(experiment_dir: Path) -> dict[str, Any]:
    run_config_path = experiment_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {run_config_path}")
    return json.loads(run_config_path.read_text(encoding="utf-8"))


def build_length_config(run_config: dict[str, Any]) -> LengthSamplingConfig:
    return LengthSamplingConfig(
        length_mode=str(run_config["length_mode"]),
        length_group_probs=tuple(float(item) for item in run_config["length_group_probs"]),
        short_length_range=tuple(int(item) for item in run_config["short_length_range"]),
        mid_length_range=tuple(int(item) for item in run_config["mid_length_range"]),
        long_length_range=tuple(int(item) for item in run_config["long_length_range"]),
    )


def build_scenario_definition(group_df: pd.DataFrame, length_config: LengthSamplingConfig) -> ScenarioDefinition:
    mechanisms = sorted(group_df["mechanism"].astype(str).unique().tolist())
    if len(mechanisms) != 1:
        raise RuntimeError(f"expected exactly one mechanism per scenario group, got {mechanisms}")
    scenario_tags = sorted(group_df["scenario_tag"].astype(str).unique().tolist())
    if len(scenario_tags) != 1:
        raise RuntimeError(f"expected exactly one scenario_tag per scenario group, got {scenario_tags}")
    missing_rates = sorted({float(value) for value in group_df["missing_rate_target"].astype(float).tolist()})
    if len(missing_rates) != 1:
        raise RuntimeError(f"expected exactly one missing_rate_target per scenario group, got {missing_rates}")
    parameter_settings = sorted(group_df["parameter_setting"].astype(str).unique().tolist())
    parameter_setting = parameter_settings[0] if parameter_settings else build_parameter_setting(length_config)
    length_modes = sorted(group_df["length_mode"].astype(str).unique().tolist()) if "length_mode" in group_df.columns else []
    return ScenarioDefinition(
        mechanism=mechanisms[0],
        missing_rate=missing_rates[0],
        scenario_tag=scenario_tags[0],
        parameter_setting=parameter_setting,
        length_mode=length_modes[0] if length_modes else length_config.length_mode,
        length_group_probs=length_config.length_group_probs,
        short_length_range=length_config.short_length_range,
        mid_length_range=length_config.mid_length_range,
        long_length_range=length_config.long_length_range,
    )


def rebuild_manifest_summary(experiment_dir: Path) -> pd.DataFrame:
    paths = build_paths(experiment_dir)
    prepare_artifacts = load_prepare_artifacts(paths)
    status_path = paths.manifests_dir / "structured_missing_chunk_status.csv"
    if not status_path.exists():
        raise FileNotFoundError(f"structured_missing_chunk_status.csv not found: {status_path}")
    status_df = pd.read_csv(status_path)
    if status_df.empty:
        raise RuntimeError(f"chunk status is empty: {status_path}")
    run_config = load_run_config(experiment_dir)
    length_config = build_length_config(run_config)
    total_observation_count = int(prepare_artifacts.chunk_summary_df["target_non_null_count"].sum())
    tolerance = float(run_config["tolerance"])
    local_scenarios = {path.name for path in paths.masks_dir.iterdir() if path.is_dir()} if paths.masks_dir.exists() else set()
    if not local_scenarios:
        raise RuntimeError(f"no local scenario directories found under {paths.masks_dir}")

    scenario_rows: list[dict[str, Any]] = []
    for _, group_df in status_df.groupby("scenario_tag", sort=True, dropna=False):
        scenario = build_scenario_definition(group_df, length_config)
        scenario_name = scenario_output_name(scenario)
        if scenario_name not in local_scenarios:
            continue
        summary = build_existing_scenario_summary(
            scenario=scenario,
            chunk_status_df=group_df.reset_index(drop=True),
            total_observation_count=total_observation_count,
            tolerance=tolerance,
            event_path=scenario_event_path(paths, scenario),
            paths=paths,
            prepare_artifacts=prepare_artifacts,
        )
        if "point_topup_count" in group_df.columns:
            summary["point_topup_count"] = int(pd.to_numeric(group_df["point_topup_count"], errors="coerce").fillna(0).sum())
        summary["scenario_output_name"] = scenario_name
        scenario_rows.append(summary)

    if not scenario_rows:
        raise RuntimeError(f"no manifest scenarios matched local directories under {experiment_dir}")
    summary_df = (
        pd.DataFrame(scenario_rows)
        .sort_values(["mechanism", "missing_rate_target", "scenario_tag"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(paths.manifests_dir / "structured_missing_scenario_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def validate_manifest_distribution_consistency(
    manifest_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    *,
    rate_tolerance: float = 1e-12,
) -> pd.DataFrame:
    manifest_lookup = manifest_df.copy()
    if "scenario_output_name" not in manifest_lookup.columns:
        raise RuntimeError("manifest_df missing scenario_output_name")
    distribution_lookup = distribution_df.copy().rename(columns={"scenario": "scenario_output_name"})

    manifest_keys = set(manifest_lookup["scenario_output_name"].astype(str).tolist())
    distribution_keys = set(distribution_lookup["scenario_output_name"].astype(str).tolist())
    missing_in_distribution = sorted(manifest_keys - distribution_keys)
    extra_in_distribution = sorted(distribution_keys - manifest_keys)
    if missing_in_distribution or extra_in_distribution:
        raise RuntimeError(
            "scenario membership mismatch between manifest and distribution: "
            f"missing_in_distribution={missing_in_distribution}, extra_in_distribution={extra_in_distribution}"
        )

    rows: list[dict[str, Any]] = []
    for scenario_name, manifest_row in manifest_lookup.set_index("scenario_output_name").iterrows():
        distribution_row = distribution_lookup.set_index("scenario_output_name").loc[scenario_name]
        mechanism_match = str(manifest_row["mechanism"]) == str(distribution_row["mechanism"])
        rate_match = bool(
            np.isclose(
                float(manifest_row["missing_rate_target"]),
                float(distribution_row["missing_rate_target"]),
                atol=rate_tolerance,
                rtol=0.0,
            )
        )
        count_match = int(manifest_row["observed_missing_count"]) == int(distribution_row["observed_missing_count"])
        observed_rate_match = bool(
            np.isclose(
                float(manifest_row["observed_missing_rate"]),
                float(distribution_row["observed_missing_rate"]),
                atol=rate_tolerance,
                rtol=0.0,
            )
        )
        is_consistent = mechanism_match and rate_match and count_match and observed_rate_match
        rows.append(
            {
                "scenario_output_name": scenario_name,
                "scenario_tag": str(manifest_row["scenario_tag"]),
                "mechanism_manifest": str(manifest_row["mechanism"]),
                "mechanism_distribution": str(distribution_row["mechanism"]),
                "missing_rate_manifest": float(manifest_row["missing_rate_target"]),
                "missing_rate_distribution": float(distribution_row["missing_rate_target"]),
                "observed_missing_count_manifest": int(manifest_row["observed_missing_count"]),
                "observed_missing_count_distribution": int(distribution_row["observed_missing_count"]),
                "observed_missing_rate_manifest": float(manifest_row["observed_missing_rate"]),
                "observed_missing_rate_distribution": float(distribution_row["observed_missing_rate"]),
                "is_consistent": bool(is_consistent),
            }
        )
        if not is_consistent:
            raise RuntimeError(
                f"inconsistent manifest/distribution summary for {scenario_name}: "
                f"mechanism_match={mechanism_match}, rate_match={rate_match}, "
                f"count_match={count_match}, observed_rate_match={observed_rate_match}"
            )
    return pd.DataFrame(rows).sort_values(["scenario_output_name"]).reset_index(drop=True)


def rebuild_distribution_artifacts(
    experiment_dir: Path,
    input_dir: Path,
    *,
    target_col: str,
    node_col: str,
    period: int,
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask_root = experiment_dir / "masks"
    missing_root = resolve_missing_root(experiment_dir)
    manifests_dir = experiment_dir / "manifests"
    audits_dir = experiment_dir / "audits"
    figures_dir = audits_dir
    audits_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prepare_df = load_prepare_summary(manifests_dir / "structured_prepare_chunk_summary.csv")
    expected_count = int(len(prepare_df))
    total_rows = int(prepare_df["row_count"].sum())
    rows_per_file = {
        str(row["file_name"]): int(row["row_count"])
        for row in prepare_df.loc[:, ["file_name", "row_count"]].to_dict(orient="records")
    }
    original_columns = load_original_columns(input_dir)
    complete_scenarios = discover_complete_scenarios(mask_root, missing_root, expected_count)
    if not complete_scenarios:
        raise RuntimeError(f"no complete structured scenarios found in {experiment_dir}")

    summary_rows: list[dict[str, Any]] = []
    top_node_frames: list[pd.DataFrame] = []
    slot_frames: list[pd.DataFrame] = []
    day_frames: list[pd.DataFrame] = []
    for scenario in complete_scenarios:
        summary, top_nodes, slot_profile, day_summary = analyze_scenario(
            scenario=scenario,
            mask_dir=mask_root / scenario,
            total_rows=total_rows,
            rows_per_file=rows_per_file,
            node_col=node_col,
            period=period,
        )
        summary_rows.append(summary)
        top_nodes.insert(0, "scenario", scenario)
        top_node_frames.append(top_nodes)
        slot_frames.append(slot_profile)
        day_frames.append(day_summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(["mechanism", "missing_rate_target", "scenario"]).reset_index(drop=True)
    top_nodes_df = pd.concat(top_node_frames, ignore_index=True)
    slot_df = pd.concat(slot_frames, ignore_index=True)
    day_df = pd.concat(day_frames, ignore_index=True)

    validation_df = validate_manifest_distribution_consistency(manifest_df, summary_df)

    summary_df.to_csv(audits_dir / "structured_missingness_distribution_summary.csv", index=False, encoding="utf-8-sig")
    top_nodes_df.to_csv(audits_dir / "structured_missingness_top_nodes.csv", index=False, encoding="utf-8-sig")
    slot_df.to_csv(audits_dir / "structured_missingness_time_slot_profile.csv", index=False, encoding="utf-8-sig")
    day_df.to_csv(audits_dir / "structured_missingness_day_profile.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(audits_dir / "structured_missingness_consistency_validation.csv", index=False, encoding="utf-8-sig")

    save_rate_bar(summary_df, figures_dir / "structured_missingness_observed_rate.png")
    save_length_group_bar(summary_df, figures_dir / "structured_missingness_length_group_ratio.png")
    save_time_slot_heatmap(slot_df, figures_dir / "structured_missingness_time_slot_heatmap.png")

    payload = {
        "experiment_dir": str(experiment_dir),
        "complete_scenario_count": int(len(complete_scenarios)),
        "complete_scenarios": complete_scenarios,
        "target_col": target_col,
        "original_columns": original_columns,
        "missing_pattern_overall": "non_random_structured_missingness",
        "scenario_summaries": summary_df.to_dict(orient="records"),
        "manifest_consistency": {
            "validated": True,
            "row_count": int(len(validation_df)),
            "all_consistent": bool(validation_df["is_consistent"].all()),
            "validation_csv": str(audits_dir / "structured_missingness_consistency_validation.csv"),
        },
        "figures": {
            "observed_rate": str(figures_dir / "structured_missingness_observed_rate.png"),
            "length_group_ratio": str(figures_dir / "structured_missingness_length_group_ratio.png"),
            "time_slot_heatmap": str(figures_dir / "structured_missingness_time_slot_heatmap.png"),
        },
    }
    write_json(audits_dir / "structured_missingness_distribution_report.json", payload)
    write_json(audits_dir / "structured_missingness_consistency_validation.json", payload["manifest_consistency"])
    build_markdown_report(
        output_path=audits_dir / "structured_missingness_distribution_report_zh.md",
        summary_df=summary_df,
        top_nodes_df=top_nodes_df,
        original_columns=original_columns,
        target_col=target_col,
        node_col=node_col,
    )
    with (audits_dir / "structured_missingness_distribution_report_zh.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## 4. Manifest 一致性校验\n\n"
            f"- validated: `True`\n"
            f"- all_consistent: `{bool(validation_df['is_consistent'].all())}`\n"
            f"- validation_rows: `{int(len(validation_df))}`\n"
            f"- validation_csv: `structured_missingness_consistency_validation.csv`\n"
        )
    return summary_df, validation_df


def repair_structured_scenario_artifacts(
    scenario_dir: Path,
    input_dir: Path,
    *,
    target_col: str,
    node_col: str,
    period: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_df = rebuild_manifest_summary(scenario_dir)
    distribution_df, validation_df = rebuild_distribution_artifacts(
        scenario_dir,
        input_dir,
        target_col=target_col,
        node_col=node_col,
        period=period,
        manifest_df=manifest_df,
    )
    return manifest_df, distribution_df, validation_df


def repair_structured_scenarios_under_target_root(project_root: Path, target_root: Path) -> None:
    input_dir = project_root / "data" / "analysis" / "node_intersection_flow_parquet"
    for scenario_id in ("ntb_mix", "nso_mix"):
        scenario_dir = target_root / "scenarios" / scenario_id / "miss_set"
        if scenario_dir.exists():
            repair_structured_scenario_artifacts(
                scenario_dir=scenario_dir,
                input_dir=input_dir,
                target_col="路口车流量",
                node_col="节点ID",
                period=96,
            )


def main() -> None:
    args = parse_args()
    scenario_dir, input_dir = validate_args(args)
    repair_structured_scenario_artifacts(
        scenario_dir=scenario_dir,
        input_dir=input_dir,
        target_col=args.target_col,
        node_col=args.node_col,
        period=args.period,
    )


if __name__ == "__main__":
    main()
