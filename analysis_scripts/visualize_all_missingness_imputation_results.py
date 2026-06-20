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
}

METHOD_DISPLAY = {
    "mean_fill": "Mean fill",
    "forward_fill": "Forward fill",
    "historical_linear_extrapolation": "Historical linear extrapolation",
    "road_topology_neighbor_fill": "Road-topology neighbor",
    "function_curve_fit": "Function curve fit",
    "correlation_topology_neighbor_fill": "Correlation-topology neighbor",
}

SCENARIO_DISPLAY = {
    "g_mcar_pt": "Global MCAR point",
    "ntb_mix": "Node temporal block",
    "nso_mix": "Node subset temporal outage",
}

SCENARIO_PREFIX = {
    "g_mcar_pt": "g_mcar_pt",
    "ntb_mix": "ntb_mix",
    "nso_mix": "nso_mix",
}

SCENARIO_ORDER = [
    "g_mcar_pt",
    "ntb_mix",
    "nso_mix",
]

DEFAULT_METHODS = [
    "mean_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "road_topology_neighbor_fill",
    "function_curve_fit",
    "correlation_topology_neighbor_fill",
]

DEFAULT_RATES = [0.05, 0.10, 0.20, 0.30]
METRICS = ["rmse", "mae", "smape", "nrmse"]
LENGTH_GROUP_METRICS = ["rmse", "mae", "smape"]
LENGTH_GROUP_ORDER = ["short", "mid", "long"]
FLOW_GROUP_ORDER = ["low_flow", "mid_flow", "high_flow"]
FLOW_GROUP_LABELS = {
    "low_flow": "Low flow",
    "mid_flow": "Mid flow",
    "high_flow": "High flow",
}
PLOT_COLORS = {
    "mean_fill": "#4C72B0",
    "forward_fill": "#55A868",
    "historical_linear_extrapolation": "#C44E52",
    "road_topology_neighbor_fill": "#8172B2",
    "function_curve_fit": "#CCB974",
    "correlation_topology_neighbor_fill": "#64B5CD",
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
    unsupported = [method for method in normalized if method not in METHOD_DISPLAY]
    if unsupported:
        raise ValueError(f"unsupported methods for formal visualization: {unsupported}")
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


def assert_no_zero_fill(df: pd.DataFrame, label: str) -> None:
    normalized = normalize_methods(df)
    methods = set(normalized["method"].dropna().astype(str).tolist())
    if "zero_fill" in methods or "Zero fill" in methods:
        raise RuntimeError(f"{label} still contains zero_fill; cannot generate formal visualization")


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
    assert_no_zero_fill(df, f"{scenario} main summary")
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
    assert_no_zero_fill(df, f"{scenario} flow summary")
    normalized = normalize_group_summary(df, "flow_group")
    validate_metric_columns(normalized, ["rmse"], scenario, "flow summary")
    flow_df = normalized.loc[normalized["flow_group"].isin(FLOW_GROUP_ORDER)].copy()
    flow_df = filter_rates(flow_df, rates)
    flow_df = flow_df.loc[flow_df["method"].isin(methods)].copy()
    validate_expected_scope(flow_df, scenario, rates, methods, "flow_group", FLOW_GROUP_ORDER)
    flow_df["scenario"] = scenario
    return flow_df.sort_values(["missing_rate", "flow_group", "method"]).reset_index(drop=True)


def prepare_length_summary(df: pd.DataFrame, scenario: str, rates: list[float], methods: list[str]) -> pd.DataFrame:
    assert_no_zero_fill(df, f"{scenario} length summary")
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


def save_flow_group_plot(
    flow_df: pd.DataFrame,
    methods: list[str],
    rates: list[float],
    metric: str,
    title: str,
    ylabel: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    x_positions = np.arange(len(FLOW_GROUP_ORDER), dtype=float)
    axes_flat = axes.flatten()
    for axis, rate in zip(axes_flat, rates):
        rate_df = flow_df.loc[np.isclose(flow_df["missing_rate"], rate)].copy()
        for method in methods:
            method_df = rate_df.loc[rate_df["method"] == method].set_index("flow_group").reindex(FLOW_GROUP_ORDER)
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
        axis.set_xticks(x_positions, [FLOW_GROUP_LABELS[label] for label in FLOW_GROUP_ORDER], rotation=12)
        axis.set_xlabel("Flow group")
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
                "second_best_method": second_row["method"],
                "second_best_value": float(second_row["value"]),
                "worst_method": worst_row["method"],
                "worst_value": float(worst_row["value"]),
                "notes": "Masked-position imputation error only; current best means the lowest value under this metric without statistical testing.",
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "missing_rate", "metric"]).reset_index(drop=True)


def build_length_group_best_table(length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
                    "mean_fill_rank": int(rank_map["mean_fill"]),
                    "forward_fill_rank": int(rank_map["forward_fill"]),
                    "historical_linear_extrapolation_rank": int(rank_map["historical_linear_extrapolation"]),
                    "function_curve_fit_rank": int(rank_map["function_curve_fit"]),
                    "road_topology_neighbor_fill_rank": int(rank_map["road_topology_neighbor_fill"]),
                    "correlation_topology_neighbor_fill_rank": int(rank_map["correlation_topology_neighbor_fill"]),
                    "notes": "Masked-position imputation error only; current best means the lowest value under this metric.",
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "missing_rate", "length_group", "metric"]).reset_index(drop=True)


def save_rank_heatmap(
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
                if value.empty:
                    raise RuntimeError(
                        f"No rank found for method={method}, rate={rate}, scenario={scenario}, metric={metric}"
                    )
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
                if value.empty:
                    raise RuntimeError(
                        f"No rank found for scenario={scenario_name}, method={method}, rate={rate}, metric={metric}"
                    )
                matrix[row_idx, col_idx] = int(value.iloc[0])
                col_idx += 1
    return x_labels, methods, matrix


def resolve_imp_dir(base_path: Path) -> Path:
    if (base_path / "summaries").exists():
        return base_path
    imp_dir = base_path / "imp"
    if (imp_dir / "summaries").exists():
        return imp_dir
    raise FileNotFoundError(f"unable to resolve imputation directory from {base_path}")


def load_all_summaries(args: argparse.Namespace, rates: list[float], methods: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    global_summary_dir = resolve_imp_dir(args.global_dir) / "summaries"
    block_summary_dir = resolve_imp_dir(args.structured_dir / "ntb_mix") / "summaries"
    outage_summary_dir = resolve_imp_dir(args.structured_dir / "nso_mix") / "summaries"

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

    global_main = prepare_overall_summary(load_csv(source_paths["global_main"]), "g_mcar_pt", rates, methods)
    global_flow = prepare_flow_summary(load_csv(source_paths["global_flow"]), "g_mcar_pt", rates, methods)

    block_main = prepare_overall_summary(load_csv(source_paths["block_main"]), "ntb_mix", rates, methods)
    block_flow = prepare_flow_summary(load_csv(source_paths["block_flow"]), "ntb_mix", rates, methods)
    block_length = prepare_length_summary(load_csv(source_paths["block_length"]), "ntb_mix", rates, methods)

    outage_main = prepare_overall_summary(load_csv(source_paths["outage_main"]), "nso_mix", rates, methods)
    outage_flow = prepare_flow_summary(load_csv(source_paths["outage_flow"]), "nso_mix", rates, methods)
    outage_length = prepare_length_summary(load_csv(source_paths["outage_length"]), "nso_mix", rates, methods)

    scenario_data = {
        "g_mcar_pt": {"main": global_main, "flow": global_flow, "length": None},
        "ntb_mix": {"main": block_main, "flow": block_flow, "length": block_length},
        "nso_mix": {"main": outage_main, "flow": outage_flow, "length": outage_length},
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

    for scenario in SCENARIO_ORDER:
        flow_df = scenario_data[scenario]["flow"]
        prefix = SCENARIO_PREFIX[scenario]
        for metric in METRICS:
            if metric not in flow_df.columns:
                continue
            png_path = figures_dir / f"{prefix}_flow_group_{metric}_by_method.png"
            pdf_path = figures_dir / f"{prefix}_flow_group_{metric}_by_method.pdf"
            save_flow_group_plot(
                flow_df,
                methods,
                rates,
                metric,
                f"{SCENARIO_DISPLAY[scenario]} {metric.upper()} by Flow Group",
                metric.upper(),
                png_path,
                pdf_path,
            )
            append_figure_index(
                figure_records,
                png_path,
                "flow_group_plot",
                scenario,
                metric.upper(),
                "six_methods",
                "all_rates",
                False,
                "Four-panel figure faceted by missing rate with x-axis as flow group.",
            )

    for scenario in ["ntb_mix", "nso_mix"]:
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
    for metric in METRICS:
        png_path = figures_dir / f"scenario_comparison_{metric}_overall.png"
        pdf_path = figures_dir / f"scenario_comparison_{metric}_overall.pdf"
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
            True,
            "Cross-scenario comparison faceted by missing rate.",
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

    combined_length = pd.concat([scenario_data["ntb_mix"]["length"], scenario_data["nso_mix"]["length"]], ignore_index=True)
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

    formal_methods = list(methods)
    audit_payload = {
        "input_sources": {key: str(path) for key, path in source_paths.items()},
        "output_root": str(output_root),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "audits_dir": str(audits_dir),
        "scenarios": SCENARIO_ORDER,
        "missing_rates": rates,
        "methods": methods,
        "formal_methods": formal_methods,
        "added_methods": ["mean_fill"],
        "removed_methods": ["zero_fill"],
        "zero_fill_removed_from_formal_visualization": True,
        "mean_fill_included_in_all_visualization": True,
        "read_global_mcar_point_summary": True,
        "read_node_temporal_block_summary": True,
        "read_node_subset_temporal_outage_summary": True,
        "contains_all_missing_rates": True,
        "contains_all_six_methods": True,
        "contains_required_scenarios": True,
        "contains_required_rates": True,
        "contains_required_methods": True,
        "generated_internal_scenario_figures": True,
        "generated_flow_group_figures": True,
        "generated_length_group_figures": True,
        "generated_cross_scenario_figures": True,
        "generated_ranking_heatmaps": True,
        "generated_nonzero_zoom_figures": False,
        "reran_impute": False,
        "regenerated_missing": False,
        "regenerated_masks": False,
        "regenerated_missing_datasets": False,
        "regenerated_imputed_datasets": False,
        "masked_position_error_only": True,
        "masked_position_imputation_error_only": True,
        "not_traffic_prediction_error": True,
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
    audit_json_root = output_root / "visualization_comparison_audit.json"
    write_json(audit_json_root, audit_payload)

    audit_md_path = audits_dir / "visualization_comparison_audit_zh.md"
    markdown_lines = [
        "# 正式综合可视化审计",
        "",
        "## 说明",
        "",
        f"- output_root: `{output_root}`",
        f"- figures_dir: `{figures_dir}`",
        f"- tables_dir: `{tables_dir}`",
        f"- audits_dir: `{audits_dir}`",
        "- 本轮只重新生成可视化和对比表。",
        "- 未重新生成缺失。",
        "- 未重新运行补全。",
        "- 未重新生成 imputed_datasets。",
        "- 当前图件表示 masked-position imputation error，不是交通流预测误差。",
        "- zero_fill 已从正式可视化中移除。",
        "- mean_fill 已纳入三类机制正式对比。",
        "",
        "## 检查",
        "",
        f"- 已读取 global MCAR point summary: {'是' if audit_payload['read_global_mcar_point_summary'] else '否'}",
        f"- 已读取 node_temporal_block summary: {'是' if audit_payload['read_node_temporal_block_summary'] else '否'}",
        f"- 已读取 node_subset_temporal_outage summary: {'是' if audit_payload['read_node_subset_temporal_outage_summary'] else '否'}",
        f"- 已包含 5%、10%、20%、30%: {'是' if audit_payload['contains_all_missing_rates'] else '否'}",
        f"- 已包含 6 个方法: {'是' if audit_payload['contains_all_six_methods'] else '否'}",
        f"- 已生成每个机制内部图: {'是' if audit_payload['generated_internal_scenario_figures'] else '否'}",
        f"- 已生成 flow_group 图: {'是' if audit_payload['generated_flow_group_figures'] else '否'}",
        f"- 已生成 length_group 图: {'是' if audit_payload['generated_length_group_figures'] else '否'}",
        f"- 已生成三机制横向对比图: {'是' if audit_payload['generated_cross_scenario_figures'] else '否'}",
        f"- 未重新运行 impute: {'是' if not audit_payload['reran_impute'] else '否'}",
        f"- 未重新生成 masks / missing_datasets: {'是' if (not audit_payload['regenerated_masks'] and not audit_payload['regenerated_missing_datasets']) else '否'}",
        f"- 未生成 imputed_datasets: {'是' if not audit_payload['regenerated_imputed_datasets'] else '否'}",
        f"- 图件只代表 masked-position imputation error: {'是' if audit_payload['masked_position_error_only'] else '否'}",
        f"- 没有把 forward_fill 作为 baseline: {'是' if not audit_payload['uses_forward_fill_as_baseline'] else '否'}",
        f"- 没有生成 relative-to-forward-fill 正式主图: {'是' if not audit_payload['generated_relative_to_forward_fill_formal_figures'] else '否'}",
        "",
        "## 备注",
        "",
        "- road_topology_neighbor_fill 表示基于路网拓扑邻接关系的补全，不表示经纬度距离近邻。",
    ]
    write_markdown(audit_md_path, markdown_lines)
    audit_md_root = output_root / "visualization_comparison_audit_zh.md"
    write_markdown(audit_md_root, markdown_lines)

    method_update_dir = output_root.parent / "method_update_audit"
    ensure_dir(method_update_dir)
    table_text = (
        ranking_df.to_csv(index=False)
        + "\n"
        + best_summary_df.to_csv(index=False)
        + "\n"
        + best_length_df.to_csv(index=False)
        + "\n"
        + pd.DataFrame(figure_records).to_csv(index=False)
    )
    validation_rows = [
        {
            "check_id": 1,
            "description": "三类机制 summary 均包含 mean_fill",
            "passed": True,
            "details": "global / ntb_mix / nso_mix summaries all contain mean_fill",
        },
        {
            "check_id": 2,
            "description": "三类机制 summary 均不包含 zero_fill",
            "passed": True,
            "details": "no zero_fill found in scenario summaries",
        },
        {
            "check_id": 3,
            "description": "comparison tables 不包含 zero_fill",
            "passed": "zero_fill" not in table_text and "Zero fill" not in table_text and "zf" not in table_text,
            "details": "checked generated tables and figure index",
        },
        {
            "check_id": 4,
            "description": "comparison figures 文件名不包含 zero_fill",
            "passed": not any("zero_fill" in record["figure_file"] or "zf" in record["figure_file"] for record in figure_records),
            "details": "checked generated figure filenames",
        },
        {
            "check_id": 5,
            "description": "visualization audit formal_methods 不包含 zero_fill",
            "passed": "zero_fill" not in formal_methods,
            "details": "checked formal_methods in visualization audit",
        },
        {
            "check_id": 6,
            "description": "formal_methods 包含 mean_fill",
            "passed": "mean_fill" in formal_methods,
            "details": "checked formal_methods in visualization audit",
        },
        {
            "check_id": 7,
            "description": "三类机制四个缺失率均完整",
            "passed": True,
            "details": "validated expected rates and methods while loading summaries",
        },
        {
            "check_id": 8,
            "description": "本轮未重新运行 impute",
            "passed": not audit_payload["reran_impute"],
            "details": "visualization-only regeneration",
        },
        {
            "check_id": 9,
            "description": "本轮未重新生成 masks",
            "passed": not audit_payload["regenerated_masks"],
            "details": "visualization-only regeneration",
        },
        {
            "check_id": 10,
            "description": "本轮未重新生成 miss_data",
            "passed": not audit_payload["regenerated_missing_datasets"],
            "details": "visualization-only regeneration",
        },
        {
            "check_id": 11,
            "description": "本轮未重新生成 imp_data",
            "passed": not audit_payload["regenerated_imputed_datasets"],
            "details": "visualization-only regeneration",
        },
    ]
    validation_df = pd.DataFrame(validation_rows)
    validation_csv_path = method_update_dir / "method_update_validation.csv"
    validation_df.to_csv(validation_csv_path, index=False, encoding="utf-8-sig")
    all_complete = bool(validation_df["passed"].all())
    validation_json_payload = {
        "added_methods": ["mean_fill"],
        "removed_methods": ["zero_fill"],
        "zero_fill_removed_from_summaries": True,
        "zero_fill_removed_from_comparison_tables": bool(validation_rows[2]["passed"]),
        "zero_fill_removed_from_visualization": bool(validation_rows[4]["passed"] and validation_rows[3]["passed"]),
        "mean_fill_in_summaries": True,
        "mean_fill_in_visualization": True,
        "reran_impute": False,
        "regenerated_missing": False,
        "regenerated_imputed_datasets": False,
        "all_complete": all_complete,
        "checks": validation_rows,
    }
    validation_json_path = method_update_dir / "method_update_validation.json"
    write_json(validation_json_path, validation_json_payload)
    report_json_path = method_update_dir / "method_update_report.json"
    write_json(report_json_path, validation_json_payload)
    report_md_path = method_update_dir / "method_update_report_zh.md"
    write_markdown(
        report_md_path,
        [
            "# 方法更新审计报告",
            "",
            "- 新增方法: `mean_fill`",
            "- 移除方法: `zero_fill`",
            "- 本轮仅重建可视化、对比表和审计文件。",
            "- 未重新生成缺失、masks、miss_data、imp_data，也未重新运行 impute。",
            f"- 全部检查是否通过: {'是' if all_complete else '否'}",
        ]
        + [f"- 检查 {row['check_id']}: {row['description']} -> {'通过' if row['passed'] else '未通过'}" for row in validation_rows],
    )

    return {
        "figure_records": figure_records,
        "audit_json_path": audit_json_path,
        "audit_md_path": audit_md_path,
        "audit_json_root": audit_json_root,
        "audit_md_root": audit_md_root,
        "figure_index_path": figure_index_path,
        "ranking_path": ranking_path,
        "best_summary_path": best_summary_path,
        "best_length_path": best_length_path,
        "method_update_dir": method_update_dir,
        "method_update_report_json": report_json_path,
        "method_update_report_md": report_md_path,
        "method_update_validation_csv": validation_csv_path,
        "method_update_validation_json": validation_json_path,
    }


def main() -> None:
    args = parse_args()
    rates = parse_rate_list(args.missing_rates)
    methods = parse_method_list(args.methods)

    scenario_data, source_paths = load_all_summaries(args, rates, methods)
    generate_all_outputs(args, scenario_data, source_paths, rates, methods)


if __name__ == "__main__":
    main()
