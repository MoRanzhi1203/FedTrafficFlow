from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze structured missingness distribution from masks and manifests."
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=Path("results/real_data_structured_missingness_setting"),
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_rate(value: float) -> str:
    return f"{value:.4f}"


def discover_complete_scenarios(mask_root: Path, missing_root: Path, expected_count: int) -> list[str]:
    scenarios: list[str] = []
    if not mask_root.exists() or not missing_root.exists():
        return scenarios
    mask_dirs = {path.name: path for path in mask_root.iterdir() if path.is_dir()}
    missing_dirs = {path.name: path for path in missing_root.iterdir() if path.is_dir()}
    shared = sorted(set(mask_dirs) & set(missing_dirs))
    for scenario in shared:
        mask_count = len(list(mask_dirs[scenario].glob("*.parquet")))
        missing_count = len(list(missing_dirs[scenario].glob("*.parquet")))
        if mask_count == expected_count and missing_count == expected_count:
            scenarios.append(scenario)
    return scenarios


def load_prepare_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"prepare summary not found: {path}")
    return pd.read_csv(path)


def load_original_columns(input_dir: Path) -> list[str]:
    sample_files = sorted(input_dir.glob("node_flow_chunk_*.parquet"))
    if not sample_files:
        raise FileNotFoundError(f"no parquet input files found in {input_dir}")
    sample_df = pd.read_parquet(sample_files[0])
    return sample_df.columns.tolist()


def compute_gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.astype(np.float64, copy=False))
    total = float(sorted_values.sum())
    if total <= 0:
        return 0.0
    index = np.arange(1, len(sorted_values) + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (len(sorted_values) * total)) - ((len(sorted_values) + 1) / len(sorted_values)))


def scenario_to_display_name(scenario: str) -> str:
    parts = scenario.split("__")
    mechanism = next((part.replace("mechanism_", "") for part in parts if part.startswith("mechanism_")), scenario)
    rate = next((part.replace("rate_", "") for part in parts if part.startswith("rate_")), "unknown")
    return f"{mechanism}\n{rate}"


def analyze_scenario(
    *,
    scenario: str,
    mask_dir: Path,
    total_rows: int,
    node_col: str,
    period: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_missing = 0
    node_counts: dict[int, int] = {}
    slot_counts = np.zeros(period, dtype=np.int64)
    day_rows: list[dict[str, Any]] = []
    length_counts: dict[int, int] = {}
    length_group_counts = {"short": 0, "mid": 0, "long": 0}

    mask_files = sorted(mask_dir.glob("*.parquet"))
    if not mask_files:
        raise RuntimeError(f"no mask parquet found in {mask_dir}")

    for file_path in mask_files:
        mask_df = pd.read_parquet(file_path)
        file_missing = int(len(mask_df))
        total_missing += file_missing
        if file_missing == 0:
            day_rows.append(
                {
                    "scenario": scenario,
                    "file_name": file_path.name,
                    "observed_missing_count": 0,
                    "observed_missing_rate": 0.0,
                }
            )
            continue

        node_series = mask_df[node_col].value_counts()
        for node_id, count in node_series.items():
            node_counts[int(node_id)] = node_counts.get(int(node_id), 0) + int(count)

        slot_indices = (mask_df["global_time_index"].to_numpy(dtype=np.int64, copy=False) % period).astype(np.int64, copy=False)
        slot_counts += np.bincount(slot_indices, minlength=period)

        if "actual_length" in mask_df.columns:
            length_series = mask_df["actual_length"].value_counts()
            for length_value, count in length_series.items():
                length_counts[int(length_value)] = length_counts.get(int(length_value), 0) + int(count)
        if "length_group" in mask_df.columns:
            group_series = mask_df["length_group"].value_counts()
            for group_name in ["short", "mid", "long"]:
                length_group_counts[group_name] += int(group_series.get(group_name, 0))

        day_rows.append(
            {
                "scenario": scenario,
                "file_name": file_path.name,
                "observed_missing_count": file_missing,
                "observed_missing_rate": float(file_missing / 4034976.0),
            }
        )

    node_count_values = np.array(list(node_counts.values()), dtype=np.int64)
    top_nodes = (
        pd.DataFrame(
            {
                node_col: list(node_counts.keys()),
                "missing_count": list(node_counts.values()),
            }
        )
        .sort_values(["missing_count", node_col], ascending=[False, True])
        .head(20)
        .reset_index(drop=True)
    )
    slot_profile = pd.DataFrame(
        {
            "scenario": scenario,
            "time_slot_within_day": np.arange(period, dtype=np.int64),
            "missing_count": slot_counts,
            "missing_ratio_within_scenario": np.divide(
                slot_counts,
                max(int(total_missing), 1),
                dtype=np.float64,
            ),
        }
    )
    day_summary = pd.DataFrame(day_rows).sort_values("file_name").reset_index(drop=True)

    observed_rate = float(total_missing / float(total_rows))
    summary = {
        "scenario": scenario,
        "mechanism": scenario.split("__")[0].replace("mechanism_", ""),
        "missing_rate_target": scenario.split("__")[1].replace("rate_", "").replace("p", "."),
        "complete_mask_file_count": int(len(mask_files)),
        "observed_missing_count": int(total_missing),
        "observed_missing_rate": observed_rate,
        "field_missing_rate_target_col": observed_rate,
        "field_missing_rate_non_target_cols": 0.0,
        "missing_pattern": "non_random_structured_missingness",
        "is_random_missing": False,
        "is_non_random_missing": True,
        "node_missing_count_mean": float(node_count_values.mean()) if node_count_values.size else 0.0,
        "node_missing_count_std": float(node_count_values.std()) if node_count_values.size else 0.0,
        "node_missing_count_p95": float(np.percentile(node_count_values, 95)) if node_count_values.size else 0.0,
        "node_missing_count_max": int(node_count_values.max()) if node_count_values.size else 0,
        "node_missing_gini": compute_gini(node_count_values),
        "short_missing_ratio": float(length_group_counts["short"] / max(total_missing, 1)),
        "mid_missing_ratio": float(length_group_counts["mid"] / max(total_missing, 1)),
        "long_missing_ratio": float(length_group_counts["long"] / max(total_missing, 1)),
        "length_value_counts": {str(key): int(value) for key, value in sorted(length_counts.items())},
    }
    return summary, top_nodes, slot_profile, day_summary


def save_rate_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df), dtype=np.int64)
    axis.bar(x, summary_df["observed_missing_rate"], color="#4C78A8")
    axis.set_xticks(x, [scenario_to_display_name(value) for value in summary_df["scenario"]], rotation=0)
    axis.set_ylabel("Observed missing rate")
    axis.set_title("Observed Missing Rate by Structured Scenario")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_length_group_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(summary_df), dtype=np.int64)
    axis.bar(x, summary_df["short_missing_ratio"], label="short", color="#59A14F")
    axis.bar(x, summary_df["mid_missing_ratio"], bottom=summary_df["short_missing_ratio"], label="mid", color="#F28E2B")
    axis.bar(
        x,
        summary_df["long_missing_ratio"],
        bottom=summary_df["short_missing_ratio"] + summary_df["mid_missing_ratio"],
        label="long",
        color="#E15759",
    )
    axis.set_xticks(x, [scenario_to_display_name(value) for value in summary_df["scenario"]], rotation=0)
    axis.set_ylabel("Ratio within masked rows")
    axis.set_title("Length Group Composition by Structured Scenario")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_time_slot_heatmap(slot_df: pd.DataFrame, output_path: Path) -> None:
    pivot = slot_df.pivot(index="scenario", columns="time_slot_within_day", values="missing_ratio_within_scenario").fillna(0.0)
    fig, axis = plt.subplots(figsize=(14, max(4, 0.8 * len(pivot))))
    im = axis.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
    axis.set_yticks(np.arange(len(pivot.index), dtype=np.int64), [scenario_to_display_name(value) for value in pivot.index])
    axis.set_xlabel("Time slot within day")
    axis.set_title("Temporal Distribution of Missing Rows Within Day")
    fig.colorbar(im, ax=axis, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_markdown_report(
    *,
    output_path: Path,
    summary_df: pd.DataFrame,
    top_nodes_df: pd.DataFrame,
    original_columns: list[str],
    target_col: str,
    node_col: str,
) -> None:
    lines = [
        "# 结构化缺失分布分析报告",
        "",
        "## 1. 总体结论",
        "",
        "- 本报告仅基于已完整生成的结构化缺失场景。",
        f"- 原始字段列表：`{', '.join(original_columns)}`",
        f"- 目标缺失字段：`{target_col}`",
        "- 缺失模式判断：当前场景均为结构化非随机缺失，不属于 MCAR。",
        "",
        "## 2. 场景摘要",
        "",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.extend(
            [
                f"### {row['scenario']}",
                "",
                f"- observed_missing_rate: `{format_rate(float(row['observed_missing_rate']))}`",
                f"- field_missing_rate_target_col: `{format_rate(float(row['field_missing_rate_target_col']))}`",
                f"- field_missing_rate_non_target_cols: `{format_rate(float(row['field_missing_rate_non_target_cols']))}`",
                f"- missing_pattern: `{row['missing_pattern']}`",
                f"- node_missing_gini: `{format_rate(float(row['node_missing_gini']))}`",
                f"- short_missing_ratio: `{format_rate(float(row['short_missing_ratio']))}`",
                f"- mid_missing_ratio: `{format_rate(float(row['mid_missing_ratio']))}`",
                f"- long_missing_ratio: `{format_rate(float(row['long_missing_ratio']))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 3. 空间分布 Top Nodes",
            "",
        ]
    )
    for scenario, group_df in top_nodes_df.groupby("scenario"):
        lines.append(f"### {scenario}")
        lines.append("")
        for _, row in group_df.head(10).iterrows():
            lines.append(f"- 节点 `{int(row[node_col])}`: `{int(row['missing_count'])}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else (project_root / args.experiment_dir)
    input_dir = args.input_dir if args.input_dir.is_absolute() else (project_root / args.input_dir)
    mask_root = experiment_dir / "masks"
    missing_root = experiment_dir / "missing_datasets"
    manifests_dir = experiment_dir / "manifests"
    audits_dir = experiment_dir / "audits"
    figures_dir = experiment_dir / "figures"
    ensure_dir(audits_dir)
    ensure_dir(figures_dir)

    prepare_df = load_prepare_summary(manifests_dir / "structured_prepare_chunk_summary.csv")
    expected_count = int(len(prepare_df))
    total_rows = int(prepare_df["row_count"].sum())
    original_columns = load_original_columns(input_dir)
    complete_scenarios = discover_complete_scenarios(mask_root, missing_root, expected_count)
    if not complete_scenarios:
        raise RuntimeError("no fully generated structured missingness scenarios found yet")

    summary_rows: list[dict[str, Any]] = []
    top_node_frames: list[pd.DataFrame] = []
    slot_frames: list[pd.DataFrame] = []
    day_frames: list[pd.DataFrame] = []
    for scenario in complete_scenarios:
        summary, top_nodes, slot_profile, day_summary = analyze_scenario(
            scenario=scenario,
            mask_dir=mask_root / scenario,
            total_rows=total_rows,
            node_col=args.node_col,
            period=args.period,
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

    summary_df.to_csv(audits_dir / "structured_missingness_distribution_summary.csv", index=False, encoding="utf-8-sig")
    top_nodes_df.to_csv(audits_dir / "structured_missingness_top_nodes.csv", index=False, encoding="utf-8-sig")
    slot_df.to_csv(audits_dir / "structured_missingness_time_slot_profile.csv", index=False, encoding="utf-8-sig")
    day_df.to_csv(audits_dir / "structured_missingness_day_profile.csv", index=False, encoding="utf-8-sig")

    save_rate_bar(summary_df, figures_dir / "structured_missingness_observed_rate.png")
    save_length_group_bar(summary_df, figures_dir / "structured_missingness_length_group_ratio.png")
    save_time_slot_heatmap(slot_df, figures_dir / "structured_missingness_time_slot_heatmap.png")

    payload = {
        "experiment_dir": str(experiment_dir),
        "complete_scenario_count": int(len(complete_scenarios)),
        "complete_scenarios": complete_scenarios,
        "target_col": args.target_col,
        "original_columns": original_columns,
        "missing_pattern_overall": "non_random_structured_missingness",
        "scenario_summaries": summary_df.to_dict(orient="records"),
        "figures": {
            "observed_rate": str(figures_dir / "structured_missingness_observed_rate.png"),
            "length_group_ratio": str(figures_dir / "structured_missingness_length_group_ratio.png"),
            "time_slot_heatmap": str(figures_dir / "structured_missingness_time_slot_heatmap.png"),
        },
    }
    write_json(audits_dir / "structured_missingness_distribution_report.json", payload)
    build_markdown_report(
        output_path=audits_dir / "structured_missingness_distribution_report_zh.md",
        summary_df=summary_df,
        top_nodes_df=top_nodes_df,
        original_columns=original_columns,
        target_col=args.target_col,
        node_col=args.node_col,
    )


if __name__ == "__main__":
    main()
