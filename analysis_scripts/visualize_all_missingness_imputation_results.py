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

METHOD_ALIASES = {
    "geo_neighbor_fill": "road_topology_neighbor_fill",
    "geo_func_hybrid": "topology_function_hybrid",
}

METHOD_DISPLAY = {
    "mean_fill": "Mean fill",
    "forward_fill": "Forward fill",
    "historical_linear_extrapolation": "Historical linear extrapolation",
    "road_topology_neighbor_fill": "Road-topology neighbor",
    "function_curve_fit": "Function curve fit",
    "topology_function_hybrid": "Topology-function hybrid",
}

SCENARIO_DISPLAY = {
    "global_mcar_point": "Global MCAR point",
    "node_temporal_block": "Node temporal block",
    "node_subset_temporal_outage": "Node subset temporal outage",
}

SCENARIO_PREFIX = {
    "global_mcar_point": "global_mcar",
    "node_temporal_block": "block",
    "node_subset_temporal_outage": "outage",
}

SCENARIO_ORDER = [
    "global_mcar_point",
    "node_temporal_block",
    "node_subset_temporal_outage",
]

DEFAULT_METHODS = [
    "mean_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "road_topology_neighbor_fill",
    "function_curve_fit",
    "topology_function_hybrid",
]

DEFAULT_RATES = [0.05, 0.10, 0.20, 0.30]
METRICS = ["rmse", "mae", "smape", "nrmse"]
LENGTH_GROUP_METRICS = ["rmse", "mae", "smape"]
LENGTH_GROUP_ORDER = ["short", "mid", "long"]
FLOW_GROUP_ORDER = ["low_flow", "mid_flow", "high_flow"]
PLOT_COLORS = {
    "mean_fill": "#4C72B0",
    "forward_fill": "#55A868",
    "historical_linear_extrapolation": "#C44E52",
    "road_topology_neighbor_fill": "#8172B2",
    "function_curve_fit": "#CCB974",
    "topology_function_hybrid": "#64B5CD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualization-only comparisons for completed missingness imputation experiments."
    )
    parser.add_argument(
        "--global_dir",
        type=Path,
        default=Path("results/rdm_exp/scenarios/g_mcar_pt/imp"),
    )
    parser.add_argument(
        "--structured_dir",
        type=Path,
        default=Path("results/rdm_exp/scenarios"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/rdm_exp/comparison"),
    )
    parser.add_argument("--missing_rates", type=str, default="0.05,0.10,0.20,0.30")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_rate_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_method_list(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    normalized = [METHOD_ALIASES.get(method, method) for method in methods]
    if "zero_fill" in normalized:
        raise ValueError("zero_fill has been removed from the formal method set")
    return normalized


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required summary file not found: {path}")
    return pd.read_csv(path)


def resolve_optional_csv(summary_dir: Path, exact_name: str, glob_pattern: str) -> Path:
    exact_path = summary_dir / exact_name
    if exact_path.exists():
        return exact_path
    matches = sorted(summary_dir.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"missing summary file: {exact_name} or {glob_pattern}")
    return matches[0]


def normalize_methods(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["method"] = normalized["method"].replace(METHOD_ALIASES)
    return normalized


def normalize_main_summary(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_methods(df)
    if "flow_group" not in normalized.columns:
        normalized["flow_group"] = "overall"
    numeric_cols = [col for col in ["missing_rate", "mae", "rmse", "smape", "nrmse", "mape"] if col in normalized.columns]
    for column in numeric_cols:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def normalize_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    normalized = normalize_methods(df)
    if group_col not in normalized.columns:
        raise RuntimeError(f"group summary missing required column: {group_col}")
    numeric_cols = [col for col in ["missing_rate", "mae", "rmse", "smape", "nrmse", "mape"] if col in normalized.columns]
    for column in numeric_cols:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def filter_rates(df: pd.DataFrame, rates: list[float]) -> pd.DataFrame:
    rounded_rates = {round(rate, 2) for rate in rates}
    return df.loc[df["missing_rate"].round(2).isin(rounded_rates)].copy()


def validate_metric_columns(df: pd.DataFrame, required_metrics: list[str], scenario: str, file_label: str) -> None:
    missing = [metric for metric in required_metrics if metric not in df.columns]
    if missing:
        raise RuntimeError(f"{scenario} {file_label} missing required metric columns: {missing}")


def validate_expected_scope(
    df: pd.DataFrame,
    scenario: str,
    rates: list[float],
    methods: list[str],
    group_col: str,
    expected_groups: list[str] | None,
) -> None:
    observed_rates = sorted({round(float(value), 2) for value in df["missing_rate"].dropna().tolist()})
    expected_rates = [round(rate, 2) for rate in rates]
    if observed_rates != expected_rates:
        raise RuntimeError(f"{scenario} rates mismatch: expected {expected_rates}, got {observed_rates}")
    observed_methods = sorted(df["method"].dropna().unique().tolist())
    if observed_methods != sorted(methods):
        raise RuntimeError(f"{scenario} methods mismatch: expected {sorted(methods)}, got {observed_methods}")
    for method in methods:
        method_df = df.loc[df["method"] == method]
        method_rates = sorted({round(float(value), 2) for value in method_df["missing_rate"].dropna().tolist()})
        if method_rates != expected_rates:
            raise RuntimeError(f"{scenario} missing rates for method {method}: {method_rates}")
    if expected_groups is not None:
        observed_groups = sorted(df[group_col].dropna().unique().tolist())
        if sorted(expected_groups) != observed_groups:
            raise RuntimeError(f"{scenario} {group_col} mismatch: expected {sorted(expected_groups)}, got {observed_groups}")


def prepare_overall_summary(df: pd.DataFrame, scenario: str, rates: list[float], methods: list[str]) -> pd.DataFrame:
    normalized = normalize_main_summary(df)
    validate_metric_columns(normalized, METRICS, scenario, "main summary")
    overall_df = normalized.loc[normalized["flow_group"] == "overall"].copy()
    if overall_df.empty:
        raise RuntimeError(f"{scenario} main summary does not contain overall rows")
    overall_df = filter_rates(overall_df, rates)
    overall_df = overall_df.loc[overall_df["method"].isin(methods)].copy()
    validate_expected_scope(overall_df, scenario, rates, methods, "flow_group", ["overall"])
    overall_df["scenario"] = scenario
    overall_df["scenario_display"] = SCENARIO_DISPLAY[scenario]
    return overall_df.sort_values(["missing_rate", "method"]).reset_index(drop=True)


def prepare_flow_summary(df: pd.DataFrame, scenario: str, rates: list[float], methods: list[str]) -> pd.DataFrame:
    normalized = normalize_group_summary(df, "flow_group")
    validate_metric_columns(normalized, ["rmse"], scenario, "flow summary")
    flow_df = normalized.loc[normalized["flow_group"].isin(FLOW_GROUP_ORDER)].copy()
    flow_df = filter_rates(flow_df, rates)
    flow_df = flow_df.loc[flow_df["method"].isin(methods)].copy()
    validate_expected_scope(flow_df, scenario, rates, methods, "flow_group", FLOW_GROUP_ORDER)
    flow_df["scenario"] = scenario
    return flow_df.sort_values(["missing_rate", "flow_group", "method"]).reset_index(drop=True)


def prepare_length_summary(df: pd.DataFrame, scenario: str, rates: list[float], methods: list[str]) -> pd.DataFrame:
    normalized = normalize_group_summary(df, "length_group")
    validate_metric_columns(normalized, LENGTH_GROUP_METRICS, scenario, "length summary")
    length_df = normalized.loc[normalized["length_group"].isin(LENGTH_GROUP_ORDER)].copy()
    length_df = filter_rates(length_df, rates)
    length_df = length_df.loc[length_df["method"].isin(methods)].copy()
    validate_expected_scope(length_df, scenario, rates, methods, "length_group", LENGTH_GROUP_ORDER)
    length_df["scenario"] = scenario
    return length_df.sort_values(["missing_rate", "length_group", "method"]).reset_index(drop=True)


def pct_labels(rates: list[float]) -> list[str]:
    return [f"{int(round(rate * 100))}%" for rate in rates]


def save_line_plot_by_rate(
    summary_df: pd.DataFrame,
    methods: list[str],
    metric: str,
    title: str,
    ylabel: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    x_values = sorted(summary_df["missing_rate"].unique().tolist())
    for method in methods:
        method_df = summary_df.loc[summary_df["method"] == method].sort_values("missing_rate")
        axis.plot(
            method_df["missing_rate"],
            method_df[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            color=PLOT_COLORS[method],
            label=METHOD_DISPLAY[method],
        )
    axis.set_title(title)
    axis.set_xlabel("Missing rate")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_values, pct_labels(x_values))
    axis.grid(alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_length_group_plot(
    length_df: pd.DataFrame,
    methods: list[str],
    rates: list[float],
    metric: str,
    title: str,
    ylabel: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    x_positions = np.arange(len(LENGTH_GROUP_ORDER), dtype=float)
    axes_flat = axes.flatten()
    for axis, rate in zip(axes_flat, rates):
        rate_df = length_df.loc[np.isclose(length_df["missing_rate"], rate)].copy()
        for method in methods:
            method_df = rate_df.loc[rate_df["method"] == method].set_index("length_group").reindex(LENGTH_GROUP_ORDER)
            axis.plot(
                x_positions,
                method_df[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                markersize=5,
                color=PLOT_COLORS[method],
                label=METHOD_DISPLAY[method],
            )
        axis.set_title(f"{int(round(rate * 100))}%")
        axis.set_xticks(x_positions, [label.title() for label in LENGTH_GROUP_ORDER])
        axis.set_xlabel("Length group")
        axis.grid(alpha=0.3)
    axes_flat[0].set_ylabel(ylabel)
    axes_flat[2].set_ylabel(ylabel)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scenario_comparison_plot(
    combined_df: pd.DataFrame,
    methods: list[str],
    rates: list[float],
    metric: str,
    title: str,
    ylabel: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    x_positions = np.arange(len(SCENARIO_ORDER), dtype=float)
    axes_flat = axes.flatten()
    for axis, rate in zip(axes_flat, rates):
        rate_df = combined_df.loc[np.isclose(combined_df["missing_rate"], rate)].copy()
        for method in methods:
            method_df = rate_df.loc[rate_df["method"] == method].set_index("scenario").reindex(SCENARIO_ORDER)
            axis.plot(
                x_positions,
                method_df[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                markersize=5,
                color=PLOT_COLORS[method],
                label=METHOD_DISPLAY[method],
            )
        axis.set_title(f"{int(round(rate * 100))}%")
        axis.set_xticks(x_positions, [SCENARIO_DISPLAY[name] for name in SCENARIO_ORDER], rotation=12)
        axis.set_xlabel("Missing scenario")
        axis.grid(alpha=0.3)
    axes_flat[0].set_ylabel(ylabel)
    axes_flat[2].set_ylabel(ylabel)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_ranking_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        scenario_df = overall_df.loc[overall_df["scenario"] == scenario]
        for rate in sorted(scenario_df["missing_rate"].unique().tolist()):
            rate_df = scenario_df.loc[np.isclose(scenario_df["missing_rate"], rate)]
            for metric in METRICS:
                metric_df = rate_df.sort_values([metric, "method"]).reset_index(drop=True)
                for idx, row in metric_df.iterrows():
                    rows.append(
                        {
                            "scenario": scenario,
                            "missing_rate": float(rate),
                            "metric": metric.upper(),
                            "rank": int(idx + 1),
                            "method": row["method"],
                            "value": float(row[metric]),
                        }
                    )
    return pd.DataFrame(rows)


def build_best_method_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, rate, metric), group_df in ranking_df.groupby(["scenario", "missing_rate", "metric"], dropna=False):
        ordered = group_df.sort_values(["rank", "method"]).reset_index(drop=True)
        best_row = ordered.iloc[0]
        worst_row = ordered.iloc[-1]
        second_row = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "missing_rate": float(rate),
                "metric": metric,
                "best_method": best_row["method"],
                "best_value": float(best_row["value"]),
                "worst_method": worst_row["method"],
                "worst_value": float(worst_row["value"]),
                "second_best_method": second_row["method"],
                "notes": "Masked-position imputation error only; ranking by lowest metric value without statistical testing.",
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "missing_rate", "metric"]).reset_index(drop=True)


def build_length_group_best_table(length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tracked_methods = [
        "forward_fill",
        "function_curve_fit",
        "road_topology_neighbor_fill",
        "topology_function_hybrid",
    ]
    for (scenario, rate, length_group), group_df in length_df.groupby(
        ["scenario", "missing_rate", "length_group"], dropna=False
    ):
        for metric in LENGTH_GROUP_METRICS:
            ordered = group_df.sort_values([metric, "method"]).reset_index(drop=True)
            rank_map = {row["method"]: int(idx + 1) for idx, row in ordered.iterrows()}
            best_row = ordered.iloc[0]
            rows.append(
                {
                    "scenario": scenario,
                    "missing_rate": float(rate),
                    "length_group": length_group,
                    "metric": metric.upper(),
                    "best_method": best_row["method"],
                    "best_value": float(best_row[metric]),
                    "forward_fill_rank": int(rank_map["forward_fill"]),
                    "function_curve_fit_rank": int(rank_map["function_curve_fit"]),
                    "road_topology_neighbor_fill_rank": int(rank_map["road_topology_neighbor_fill"]),
                    "topology_function_hybrid_rank": int(rank_map["topology_function_hybrid"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "missing_rate", "length_group", "metric"]).reset_index(drop=True)


def save_rank_heatmap(
    ranking_df: pd.DataFrame,
    title: str,
    output_png: Path,
    output_pdf: Path,
    x_labels: list[str],
    y_labels: list[str],
    matrix: np.ndarray,
) -> None:
    fig, axis = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), max(5, len(y_labels) * 0.7)))
    image = axis.imshow(matrix, cmap="YlGn_r", aspect="auto")
    axis.set_title(title)
    axis.set_xticks(np.arange(len(x_labels)), x_labels, rotation=20)
    axis.set_yticks(np.arange(len(y_labels)), [METHOD_DISPLAY[label] if label in METHOD_DISPLAY else label for label in y_labels])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, shrink=0.85, label="Rank")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_heatmap_matrix(
    ranking_df: pd.DataFrame,
    *,
    scenario: str | None,
    metric: str,
    methods: list[str],
    rates: list[float],
) -> tuple[list[str], list[str], np.ndarray]:
    subset = ranking_df.loc[ranking_df["metric"] == metric.upper()].copy()
    if scenario is not None:
        subset = subset.loc[subset["scenario"] == scenario].copy()
        x_labels = pct_labels(rates)
        matrix = np.zeros((len(methods), len(rates)), dtype=int)
        for row_idx, method in enumerate(methods):
            for col_idx, rate in enumerate(rates):
                value = subset.loc[
                    (subset["method"] == method) & np.isclose(subset["missing_rate"], rate),
                    "rank",
                ]
                matrix[row_idx, col_idx] = int(value.iloc[0])
        return x_labels, methods, matrix
    x_labels = [f"{SCENARIO_DISPLAY[name]}\n{int(round(rate * 100))}%" for name in SCENARIO_ORDER for rate in rates]
    matrix = np.zeros((len(methods), len(SCENARIO_ORDER) * len(rates)), dtype=int)
    for row_idx, method in enumerate(methods):
        col_idx = 0
        for scenario_name in SCENARIO_ORDER:
            for rate in rates:
                value = subset.loc[
                    (subset["scenario"] == scenario_name)
                    & (subset["method"] == method)
                    & np.isclose(subset["missing_rate"], rate),
                    "rank",
                ]
                matrix[row_idx, col_idx] = int(value.iloc[0])
                col_idx += 1
    return x_labels, methods, matrix


def load_all_summaries(args: argparse.Namespace, rates: list[float], methods: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    global_summary_dir = args.global_dir / "summaries"
    block_summary_dir = args.structured_dir / "ntb_mix" / "imp" / "summaries"
    outage_summary_dir = args.structured_dir / "nso_mix" / "imp" / "summaries"

    source_paths = {
        "global_main": global_summary_dir / "imputation_quality_summary_exclude_warmup.csv",
        "global_flow": global_summary_dir / "imputation_quality_by_flow_group.csv",
        "block_main": resolve_optional_csv(
            block_summary_dir,
            "structured_imputation_quality_summary_exclude_warmup.csv",
            "structured_*summary_exclude_warmup*.csv",
        ),
        "block_flow": resolve_optional_csv(
            block_summary_dir,
            "structured_imputation_quality_by_flow_group.csv",
            "structured_*by_flow_group*.csv",
        ),
        "block_length": resolve_optional_csv(
            block_summary_dir,
            "structured_imputation_quality_by_length_group.csv",
            "structured_*by_length_group*.csv",
        ),
        "outage_main": resolve_optional_csv(
            outage_summary_dir,
            "outage_imputation_quality_summary_exclude_warmup.csv",
            "outage_*summary_exclude_warmup*.csv",
        ),
        "outage_flow": resolve_optional_csv(
            outage_summary_dir,
            "outage_imputation_quality_by_flow_group.csv",
            "outage_*by_flow_group*.csv",
        ),
        "outage_length": resolve_optional_csv(
            outage_summary_dir,
            "outage_imputation_quality_by_length_group.csv",
            "outage_*by_length_group*.csv",
        ),
    }

    global_main = prepare_overall_summary(load_csv(source_paths["global_main"]), "global_mcar_point", rates, methods)
    global_flow = prepare_flow_summary(load_csv(source_paths["global_flow"]), "global_mcar_point", rates, methods)

    block_main = prepare_overall_summary(load_csv(source_paths["block_main"]), "node_temporal_block", rates, methods)
    block_flow = prepare_flow_summary(load_csv(source_paths["block_flow"]), "node_temporal_block", rates, methods)
    block_length = prepare_length_summary(load_csv(source_paths["block_length"]), "node_temporal_block", rates, methods)

    outage_main = prepare_overall_summary(load_csv(source_paths["outage_main"]), "node_subset_temporal_outage", rates, methods)
    outage_flow = prepare_flow_summary(load_csv(source_paths["outage_flow"]), "node_subset_temporal_outage", rates, methods)
    outage_length = prepare_length_summary(load_csv(source_paths["outage_length"]), "node_subset_temporal_outage", rates, methods)

    scenario_data = {
        "global_mcar_point": {"main": global_main, "flow": global_flow, "length": None},
        "node_temporal_block": {"main": block_main, "flow": block_flow, "length": block_length},
        "node_subset_temporal_outage": {"main": outage_main, "flow": outage_flow, "length": outage_length},
    }
    return scenario_data, source_paths


def append_figure_index(
    records: list[dict[str, Any]],
    figure_file: Path,
    figure_type: str,
    scenario_scope: str,
    metric: str,
    method_scope: str,
    rate_scope: str,
    is_formal_main_figure: bool,
    notes: str,
) -> None:
    records.append(
        {
            "figure_file": str(figure_file.relative_to(figure_file.parents[1])),
            "figure_type": figure_type,
            "scenario_scope": scenario_scope,
            "metric": metric,
            "method_scope": method_scope,
            "rate_scope": rate_scope,
            "is_formal_main_figure": bool(is_formal_main_figure),
            "notes": notes,
        }
    )


def generate_all_outputs(
    args: argparse.Namespace,
    scenario_data: dict[str, Any],
    source_paths: dict[str, Any],
    rates: list[float],
    methods: list[str],
) -> dict[str, Any]:
    output_root = args.output_dir
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    audits_dir = output_root / "audits"
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)
    ensure_dir(audits_dir)

    figure_records: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        scenario_main = scenario_data[scenario]["main"]
        prefix = SCENARIO_PREFIX[scenario]
        for metric in METRICS:
            png_path = figures_dir / f"{prefix}_{metric}_by_method.png"
            pdf_path = figures_dir / f"{prefix}_{metric}_by_method.pdf"
            save_line_plot_by_rate(
                scenario_main,
                methods,
                metric,
                f"{SCENARIO_DISPLAY[scenario]} {metric.upper()} by Method",
                metric.upper(),
                png_path,
                pdf_path,
            )
            append_figure_index(
                figure_records,
                png_path,
                "line_plot",
                scenario,
                metric.upper(),
                "six_methods",
                "all_rates",
                True,
                "Masked-position imputation error; formal six-method direct comparison.",
            )
        for obsolete_path in [
            figures_dir / f"{prefix}_rmse_nonzero_zoom.png",
            figures_dir / f"{prefix}_rmse_nonzero_zoom.pdf",
        ]:
            if obsolete_path.exists():
                obsolete_path.unlink()

    for scenario in ["node_temporal_block", "node_subset_temporal_outage"]:
        length_df = scenario_data[scenario]["length"]
        assert length_df is not None
        prefix = SCENARIO_PREFIX[scenario]
        for metric in LENGTH_GROUP_METRICS:
            png_path = figures_dir / f"{prefix}_length_group_{metric}_by_method.png"
            pdf_path = figures_dir / f"{prefix}_length_group_{metric}_by_method.pdf"
            save_length_group_plot(
                length_df,
                methods,
                rates,
                metric,
                f"{SCENARIO_DISPLAY[scenario]} {metric.upper()} by Length Group",
                metric.upper(),
                png_path,
                pdf_path,
            )
            append_figure_index(
                figure_records,
                png_path,
                "length_group_plot",
                scenario,
                metric.upper(),
                "six_methods",
                "all_rates",
                False,
                "Four-panel figure faceted by missing rate with x-axis as length group.",
            )

    combined_main = pd.concat([scenario_data[scenario]["main"] for scenario in SCENARIO_ORDER], ignore_index=True)
    for metric in ["rmse", "mae", "smape"]:
        png_path = figures_dir / f"scenario_comparison_{metric}_by_method.png"
        pdf_path = figures_dir / f"scenario_comparison_{metric}_by_method.pdf"
        save_scenario_comparison_plot(
            combined_main,
            methods,
            rates,
            metric,
            f"Scenario Comparison {metric.upper()} by Method",
            metric.upper(),
            png_path,
            pdf_path,
        )
        append_figure_index(
            figure_records,
            png_path,
            "scenario_comparison",
            "all_scenarios",
            metric.upper(),
            "six_methods",
            "all_rates",
            metric == "rmse",
            "Cross-scenario comparison faceted by missing rate.",
        )

    rmse_overall_png = figures_dir / "scenario_comparison_rmse_overall.png"
    rmse_overall_pdf = figures_dir / "scenario_comparison_rmse_overall.pdf"
    save_scenario_comparison_plot(
        combined_main,
        methods,
        rates,
        "rmse",
        "Scenario Comparison RMSE Overall",
        "RMSE",
        rmse_overall_png,
        rmse_overall_pdf,
    )
    append_figure_index(
        figure_records,
        rmse_overall_png,
        "scenario_comparison",
        "all_scenarios",
        "RMSE",
        "six_methods",
        "all_rates",
        True,
        "Primary cross-scenario RMSE figure.",
    )

    for obsolete_path in [
        figures_dir / "scenario_comparison_rmse_nonzero_zoom.png",
        figures_dir / "scenario_comparison_rmse_nonzero_zoom.pdf",
    ]:
        if obsolete_path.exists():
            obsolete_path.unlink()

    ranking_df = build_ranking_table(combined_main)
    ranking_path = tables_dir / "method_ranking_by_scenario_rate_metric.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    best_summary_df = build_best_method_summary(ranking_df)
    best_summary_path = tables_dir / "best_method_summary.csv"
    best_summary_df.to_csv(best_summary_path, index=False, encoding="utf-8-sig")

    combined_length = pd.concat(
        [
            scenario_data["node_temporal_block"]["length"],
            scenario_data["node_subset_temporal_outage"]["length"],
        ],
        ignore_index=True,
    )
    best_length_df = build_length_group_best_table(combined_length)
    best_length_path = tables_dir / "best_method_by_length_group.csv"
    best_length_df.to_csv(best_length_path, index=False, encoding="utf-8-sig")

    for scenario in SCENARIO_ORDER:
        heatmap_png = figures_dir / f"rank_heatmap_{SCENARIO_PREFIX[scenario]}_rmse.png"
        heatmap_pdf = figures_dir / f"rank_heatmap_{SCENARIO_PREFIX[scenario]}_rmse.pdf"
        x_labels, y_labels, matrix = build_heatmap_matrix(
            ranking_df,
            scenario=scenario,
            metric="rmse",
            methods=methods,
            rates=rates,
        )
        save_rank_heatmap(
            ranking_df,
            f"{SCENARIO_DISPLAY[scenario]} RMSE Rank Heatmap",
            heatmap_png,
            heatmap_pdf,
            x_labels,
            y_labels,
            matrix,
        )
        append_figure_index(
            figure_records,
            heatmap_png,
            "rank_heatmap",
            scenario,
            "RMSE",
            "six_methods",
            "all_rates",
            False,
            "Auxiliary ranking heatmap.",
        )

    cross_heatmap_png = figures_dir / "rank_heatmap_scenario_method_rmse.png"
    cross_heatmap_pdf = figures_dir / "rank_heatmap_scenario_method_rmse.pdf"
    x_labels, y_labels, matrix = build_heatmap_matrix(
        ranking_df,
        scenario=None,
        metric="rmse",
        methods=methods,
        rates=rates,
    )
    save_rank_heatmap(
        ranking_df,
        "Cross-scenario RMSE Rank Heatmap",
        cross_heatmap_png,
        cross_heatmap_pdf,
        x_labels,
        y_labels,
        matrix,
    )
    append_figure_index(
        figure_records,
        cross_heatmap_png,
        "rank_heatmap",
        "all_scenarios",
        "RMSE",
        "six_methods",
        "all_rates",
        False,
        "Auxiliary cross-scenario ranking heatmap.",
    )

    figure_index_path = output_root / "figure_index.csv"
    pd.DataFrame(figure_records).to_csv(figure_index_path, index=False, encoding="utf-8-sig")

    audit_payload = {
        "input_sources": {key: str(path) for key, path in source_paths.items()},
        "output_root": str(output_root),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "audits_dir": str(audits_dir),
        "scenarios": SCENARIO_ORDER,
        "missing_rates": rates,
        "methods": methods,
        "read_global_mcar_point_summary": True,
        "read_node_temporal_block_summary": True,
        "read_node_subset_temporal_outage_summary": True,
        "contains_all_missing_rates": True,
        "contains_all_six_methods": True,
        "generated_internal_scenario_figures": True,
        "generated_length_group_figures": True,
        "generated_cross_scenario_figures": True,
        "generated_ranking_heatmaps": True,
        "generated_nonzero_zoom_figures": False,
        "reran_impute": False,
        "regenerated_masks": False,
        "regenerated_missing_datasets": False,
        "generated_imputed_datasets": False,
        "masked_position_imputation_error_only": True,
        "uses_forward_fill_as_baseline": False,
        "generated_relative_to_forward_fill_formal_figures": False,
        "figure_count": len(figure_records),
        "table_outputs": {
            "method_ranking_by_scenario_rate_metric": str(ranking_path),
            "best_method_summary": str(best_summary_path),
            "best_method_by_length_group": str(best_length_path),
            "figure_index": str(figure_index_path),
        },
    }

    audit_json_path = audits_dir / "visualization_comparison_audit.json"
    write_json(audit_json_path, audit_payload)

    audit_md_path = audits_dir / "visualization_comparison_audit_zh.md"
    markdown_lines = [
        "# Comprehensive Visualization Comparison Audit",
        "",
        "## Summary",
        "",
        f"- output_root: `{output_root}`",
        f"- figures_dir: `{figures_dir}`",
        f"- tables_dir: `{tables_dir}`",
        f"- audits_dir: `{audits_dir}`",
        "",
        "## Checks",
        "",
        f"- 已读取 global MCAR point summary: {'是' if audit_payload['read_global_mcar_point_summary'] else '否'}",
        f"- 已读取 node_temporal_block summary: {'是' if audit_payload['read_node_temporal_block_summary'] else '否'}",
        f"- 已读取 node_subset_temporal_outage summary: {'是' if audit_payload['read_node_subset_temporal_outage_summary'] else '否'}",
        f"- 已包含 5%、10%、20%、30%: {'是' if audit_payload['contains_all_missing_rates'] else '否'}",
        f"- 已包含 6 个方法: {'是' if audit_payload['contains_all_six_methods'] else '否'}",
        f"- 已生成每个机制内部图: {'是' if audit_payload['generated_internal_scenario_figures'] else '否'}",
        f"- 已生成 length_group 图: {'是' if audit_payload['generated_length_group_figures'] else '否'}",
        f"- 已生成三机制横向对比图: {'是' if audit_payload['generated_cross_scenario_figures'] else '否'}",
        f"- 未重新运行 impute: {'是' if not audit_payload['reran_impute'] else '否'}",
        f"- 未重新生成 masks / missing_datasets: {'是' if (not audit_payload['regenerated_masks'] and not audit_payload['regenerated_missing_datasets']) else '否'}",
        f"- 未生成 imputed_datasets: {'是' if not audit_payload['generated_imputed_datasets'] else '否'}",
        f"- 图件只代表 masked-position imputation error: {'是' if audit_payload['masked_position_imputation_error_only'] else '否'}",
        f"- 没有把 forward_fill 作为 baseline: {'是' if not audit_payload['uses_forward_fill_as_baseline'] else '否'}",
        f"- 没有生成 relative-to-forward-fill 正式主图: {'是' if not audit_payload['generated_relative_to_forward_fill_formal_figures'] else '否'}",
        "",
        "## Note",
        "",
        "- road_topology_neighbor_fill 表示基于路网拓扑邻接关系的补全，不表示经纬度距离近邻。",
    ]
    write_markdown(audit_md_path, markdown_lines)

    return {
        "figure_records": figure_records,
        "audit_json_path": audit_json_path,
        "audit_md_path": audit_md_path,
        "figure_index_path": figure_index_path,
        "ranking_path": ranking_path,
        "best_summary_path": best_summary_path,
        "best_length_path": best_length_path,
        "zoom_created": {},
        "scenario_zoom_created": False,
    }


def main() -> None:
    args = parse_args()
    rates = parse_rate_list(args.missing_rates)
    methods = parse_method_list(args.methods)

    scenario_data, source_paths = load_all_summaries(args, rates, methods)
    generate_all_outputs(args, scenario_data, source_paths, rates, methods)


if __name__ == "__main__":
    main()
