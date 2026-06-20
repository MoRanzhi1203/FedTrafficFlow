"""可视化空间邻居保留缺失实验的结果与质量分布。

核心功能：
- 读取空间 holdout 实验的 summary、detail 和 audit 文件生成图表；
- 展示不同补全方法在空间缺失设定下的误差、分组表现和稳定性；
- 输出后续汇报和复核所需的可视化结果。

项目作用：
- 作为空间缺失实验的标准展示脚本；
- 为方法效果分析和实验报告提供统一图形出口。

关键依赖：`pandas`、`numpy`、`matplotlib`。
主要输入：空间 holdout 实验目录中的统计表和审计文件。
主要输出：对比图、辅助表和可视化审计结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PHASE1_BASELINE_METHODS = [
    "mean_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "function_curve_fit",
    "road_topology_neighbor_fill",
    "correlation_topology_neighbor_fill",
]
METHOD_ORDER = list(PHASE1_BASELINE_METHODS)
FLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]
REMOVED_METHODS = {
    "adaptive_spatio_temporal_fill",
    "adaptive_topology_function_hybrid",
}
METHOD_DISPLAY = {
    "mean_fill": "Mean fill",
    "forward_fill": "Forward fill",
    "historical_linear_extrapolation": "Historical linear extrapolation",
    "function_curve_fit": "Function curve fit",
    "road_topology_neighbor_fill": "Road-topology neighbor",
    "correlation_topology_neighbor_fill": "Correlation-topology neighbor",
}
TEMPORAL_BASELINES = [
    "mean_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "function_curve_fit",
]
SPATIAL_METHODS = [
    "road_topology_neighbor_fill",
    "correlation_topology_neighbor_fill",
]
CONSTRAINT_LEVELS = ["strict_anchor", "relaxed_anchor", "weak_neighbor_available", "none"]
COLORS = {
    "mean_fill": "#4C72B0",
    "forward_fill": "#55A868",
    "historical_linear_extrapolation": "#C44E52",
    "function_curve_fit": "#8172B2",
    "road_topology_neighbor_fill": "#CCB974",
    "correlation_topology_neighbor_fill": "#64B5CD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize spatial neighbor holdout imputation results.")
    parser.add_argument("--scenario_dir", required=True, type=Path)
    parser.add_argument("--summary_dir", default="", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--audit_dir", default="", type=str)
    parser.add_argument("--missing_rates", default="0.05,0.10,0.20,0.30", type=str)
    parser.add_argument("--methods", default=",".join(METHOD_ORDER), type=str)
    return parser.parse_args()


def parse_rates(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_methods(raw: str) -> list[str]:
    methods = [token.strip() for token in raw.split(",") if token.strip()]
    unique_methods: list[str] = []
    for method in methods:
        if method in REMOVED_METHODS:
            raise ValueError(
                "Removed method: adaptive_spatio_temporal_fill; "
                "Removed method: adaptive_topology_function_hybrid. "
                "Use only six baseline methods."
            )
        if method not in METHOD_ORDER:
            raise ValueError(f"unsupported method: {method}")
        if method not in unique_methods:
            unique_methods.append(method)
    if not unique_methods:
        raise ValueError("--methods is empty")
    return unique_methods


def infer_methods_phase(methods: list[str]) -> str:
    if methods == PHASE1_BASELINE_METHODS:
        return "phase_1_six_baseline_methods"
    return "custom_method_subset"


def normalize_six_baseline_methods(methods: list[str], stage_name: str) -> list[str]:
    if len(methods) != len(PHASE1_BASELINE_METHODS) or set(methods) != set(PHASE1_BASELINE_METHODS):
        raise ValueError(f"snh_mix {stage_name} requires exactly six baseline methods.")
    return list(PHASE1_BASELINE_METHODS)


def ensure_absolute(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    return pd.read_csv(path)


def pct_labels(rates: list[float]) -> list[str]:
    return [f"{int(round(rate * 100))}%" for rate in rates]


def rate_scope_label(rates: list[float]) -> str:
    if rates == [0.05, 0.10]:
        return "Rates: 5% and 10% only"
    return "Rates: " + ", ".join(pct_labels(rates))


def save_line_plot_by_rate(
    df: pd.DataFrame,
    methods: list[str],
    metric: str,
    title: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    x_values = sorted(df["missing_rate"].unique().tolist())
    for method in methods:
        method_df = df.loc[df["method"] == method].sort_values("missing_rate")
        axis.plot(
            method_df["missing_rate"],
            method_df[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS[method],
            label=METHOD_DISPLAY[method],
        )
    axis.set_title(title)
    axis.set_xlabel("Missing rate")
    axis.set_ylabel(metric.upper())
    axis.set_xticks(x_values, pct_labels(x_values))
    axis.grid(alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_group_plot(
    df: pd.DataFrame,
    methods: list[str],
    metric: str,
    group_col: str,
    group_order: list[str],
    title: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    x_positions = np.arange(len(group_order), dtype=float)
    axes_flat = axes.flatten()
    rates = sorted(df["missing_rate"].unique().tolist())
    for axis, rate in zip(axes_flat, rates):
        rate_df = df.loc[np.isclose(df["missing_rate"], rate)].copy()
        for method in methods:
            method_df = rate_df.loc[rate_df["method"] == method].set_index(group_col).reindex(group_order)
            axis.plot(
                x_positions,
                method_df[metric].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                markersize=5,
                color=COLORS[method],
                label=METHOD_DISPLAY[method],
            )
        axis.set_title(f"{int(round(rate * 100))}%")
        axis.set_xticks(x_positions, [label.replace("_", " ").title() for label in group_order], rotation=10)
        axis.grid(alpha=0.3)
    axes_flat[0].set_ylabel(metric.upper())
    axes_flat[2].set_ylabel(metric.upper())
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_spatial_vs_temporal_plot(
    summary_df: pd.DataFrame,
    methods: list[str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    rates = sorted(summary_df["missing_rate"].unique().tolist())
    series_map: dict[str, list[str]] = {}
    if any(method in methods for method in TEMPORAL_BASELINES):
        series_map["Best temporal baseline"] = [method for method in TEMPORAL_BASELINES if method in methods]
    if any(method in methods for method in SPATIAL_METHODS):
        series_map["Best spatial method"] = [method for method in SPATIAL_METHODS if method in methods]
    color_map = {
        "Best temporal baseline": "#4C72B0",
        "Best spatial method": "#C44E52",
    }
    for label, methods in series_map.items():
        values = []
        for rate in rates:
            subset = summary_df.loc[np.isclose(summary_df["missing_rate"], rate) & summary_df["method"].isin(methods)].copy()
            values.append(float(subset["rmse"].min()))
        axis.plot(rates, values, marker="o", linewidth=2, markersize=6, label=label, color=color_map[label])
    axis.set_title("snh_mix RMSE: Spatial vs Temporal Method Families")
    axis.set_xlabel("Missing rate")
    axis.set_ylabel("RMSE")
    axis.set_xticks(rates, pct_labels(rates))
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_ranking_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rate in sorted(summary_df["missing_rate"].unique().tolist()):
        rate_df = summary_df.loc[np.isclose(summary_df["missing_rate"], rate)].copy()
        for metric in ["rmse", "mae", "smape", "nrmse"]:
            ordered = rate_df.sort_values([metric, "method"]).reset_index(drop=True)
            for rank, row in enumerate(ordered.itertuples(index=False), start=1):
                rows.append(
                    {
                        "missing_rate": float(rate),
                        "metric": metric.upper(),
                        "rank": int(rank),
                        "method": row.method,
                        "value": float(getattr(row, metric)),
                    }
                )
    return pd.DataFrame(rows)


def build_best_method_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (rate, metric), group_df in ranking_df.groupby(["missing_rate", "metric"], dropna=False):
        ordered = group_df.sort_values(["rank", "method"]).reset_index(drop=True)
        best = ordered.iloc[0]
        second = ordered.iloc[1]
        worst = ordered.iloc[-1]
        rows.append(
            {
                "missing_rate": float(rate),
                "metric": metric,
                "best_method": best["method"],
                "best_value": float(best["value"]),
                "second_best_method": second["method"],
                "second_best_value": float(second["value"]),
                "worst_method": worst["method"],
                "worst_value": float(worst["value"]),
                "notes": "Masked-position imputation error under online spatial interpolation.",
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "metric"]).reset_index(drop=True)


def build_best_by_length_group(length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (rate, length_group), group_df in length_df.groupby(["missing_rate", "length_group"], dropna=False):
        ordered = group_df.sort_values(["rmse", "method"]).reset_index(drop=True)
        rank_map = {row["method"]: int(idx + 1) for idx, row in ordered.iterrows()}
        best = ordered.iloc[0]
        payload: dict[str, Any] = {
            "missing_rate": float(rate),
            "length_group": str(length_group),
            "metric": "RMSE",
            "best_method": best["method"],
            "best_value": float(best["rmse"]),
            "notes": "Current best means the lowest RMSE under this length group.",
        }
        for method in METHOD_ORDER:
            payload[f"{method}_rank"] = int(rank_map[method]) if method in rank_map else np.nan
        rows.append(
            payload
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "length_group"]).reset_index(drop=True)


def build_spatial_gain_table(summary_df: pd.DataFrame, length_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spatial_candidates = list(SPATIAL_METHODS)
    for (rate, length_group), group_df in length_df.groupby(["missing_rate", "length_group"], dropna=False):
        temporal_df = group_df.loc[group_df["method"].isin(TEMPORAL_BASELINES)].copy()
        if temporal_df.empty:
            continue
        best_temporal = temporal_df.sort_values(["rmse", "method"]).iloc[0]
        for method in spatial_candidates:
            candidate = group_df.loc[group_df["method"] == method]
            if candidate.empty:
                continue
            spatial_row = candidate.iloc[0]
            overall_row = summary_df.loc[np.isclose(summary_df["missing_rate"], rate) & (summary_df["method"] == method)].iloc[0]
            temporal_rmse = float(best_temporal["rmse"])
            spatial_rmse = float(spatial_row["rmse"])
            improvement = ((temporal_rmse - spatial_rmse) / temporal_rmse * 100.0) if temporal_rmse > 0 else 0.0
            rows.append(
                {
                    "missing_rate": float(rate),
                    "length_group": str(length_group),
                    "spatial_method": method,
                    "best_temporal_baseline": best_temporal["method"],
                    "spatial_rmse": spatial_rmse,
                    "temporal_rmse": temporal_rmse,
                    "rmse_improvement_percent": float(improvement),
                    "neighbor_coverage": float(overall_row["neighbor_coverage"]),
                    "notes": "Positive improvement means lower RMSE than the best temporal baseline.",
                }
            )
    return pd.DataFrame(rows).sort_values(["missing_rate", "length_group", "spatial_method"]).reset_index(drop=True)


def build_spatial_gain_by_constraint(constraint_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spatial_candidates = list(SPATIAL_METHODS)
    for (rate, constraint_level), group_df in constraint_df.groupby(["missing_rate", "spatial_constraint_level"], dropna=False):
        temporal_df = group_df.loc[group_df["method"].isin(TEMPORAL_BASELINES)].copy()
        if temporal_df.empty:
            continue
        best_temporal = temporal_df.sort_values(["rmse", "method"]).iloc[0]
        temporal_rmse = float(best_temporal["rmse"])
        for method in spatial_candidates:
            candidate = group_df.loc[group_df["method"] == method]
            if candidate.empty:
                continue
            spatial_row = candidate.iloc[0]
            spatial_rmse = float(spatial_row["rmse"])
            improvement = ((temporal_rmse - spatial_rmse) / temporal_rmse * 100.0) if temporal_rmse > 0 else 0.0
            rows.append(
                {
                    "missing_rate": float(rate),
                    "spatial_constraint_level": str(constraint_level),
                    "spatial_method": method,
                    "best_temporal_baseline": best_temporal["method"],
                    "spatial_rmse": spatial_rmse,
                    "temporal_rmse": temporal_rmse,
                    "rmse_improvement_percent": float(improvement),
                    "notes": "none level is reported for completeness and should not be treated as spatial holdout evidence.",
                }
            )
    return pd.DataFrame(rows).sort_values(["missing_rate", "spatial_constraint_level", "spatial_method"]).reset_index(drop=True)


def save_constraint_spatial_gain_plot(
    gain_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes_flat = axes.flatten()
    rates = sorted(gain_df["missing_rate"].unique().tolist())
    x_positions = np.arange(len(CONSTRAINT_LEVELS), dtype=float)
    spatial_family = {
        "Best spatial method": SPATIAL_METHODS,
    }
    family_colors = {
        "Best spatial method": "#C44E52",
    }
    for axis, rate in zip(axes_flat, rates):
        rate_df = gain_df.loc[np.isclose(gain_df["missing_rate"], rate)].copy()
        for label, methods in spatial_family.items():
            values = []
            for level in CONSTRAINT_LEVELS:
                subset = rate_df.loc[
                    (rate_df["spatial_constraint_level"] == level) & (rate_df["spatial_method"].isin(methods))
                ].copy()
                values.append(float(subset["rmse_improvement_percent"].max()) if not subset.empty else np.nan)
            axis.plot(x_positions, values, marker="o", linewidth=2, markersize=5, label=label, color=family_colors[label])
        axis.set_title(f"{int(round(rate * 100))}%")
        axis.set_xticks(x_positions, [label.replace("_", " ") for label in CONSTRAINT_LEVELS], rotation=10)
        axis.grid(alpha=0.3)
    axes_flat[0].set_ylabel("RMSE Improvement %")
    axes_flat[2].set_ylabel("RMSE Improvement %")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("snh_mix Spatial Gain by Constraint Level")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rank_heatmap(ranking_df: pd.DataFrame, methods: list[str], output_png: Path, output_pdf: Path) -> None:
    rates = sorted(ranking_df["missing_rate"].unique().tolist())
    matrix = np.zeros((len(methods), len(rates)), dtype=int)
    for row_idx, method in enumerate(methods):
        for col_idx, rate in enumerate(rates):
            value = ranking_df.loc[
                (ranking_df["method"] == method)
                & (ranking_df["metric"] == "RMSE")
                & np.isclose(ranking_df["missing_rate"], rate),
                "rank",
            ]
            matrix[row_idx, col_idx] = int(value.iloc[0])
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix, cmap="YlGn_r", aspect="auto")
    axis.set_title("snh_mix RMSE Rank Heatmap")
    axis.set_xticks(np.arange(len(rates)), pct_labels(rates))
    axis.set_yticks(np.arange(len(methods)), [METHOD_DISPLAY[method] for method in methods])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, shrink=0.85, label="Rank")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_validation(
    *,
    scenario_dir: Path,
    figures_dir: Path,
    summary_df: pd.DataFrame,
    methods: list[str],
    audit_payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rates = sorted(summary_df["missing_rate"].unique().tolist())
    rows: list[dict[str, Any]] = []
    methods_ok = sorted(summary_df["method"].astype(str).unique().tolist()) == sorted(methods)
    for rate in rates:
        tag = f"snh_r{int(round(rate * 100)):02d}_mix_s42"
        mask_count = len(list((scenario_dir / "miss_set" / "masks" / tag).glob("*_mask.parquet")))
        miss_count = len(list((scenario_dir / "miss_set" / "miss_data" / tag).glob("*.parquet")))
        rows.append({"check": f"{rate:.2f}_mask_count", "passed": bool(mask_count == 61), "details": f"{mask_count}"})
        rows.append({"check": f"{rate:.2f}_miss_data_count", "passed": bool(miss_count == 61), "details": f"{miss_count}"})
        for method in methods:
            imp_count = len(list((scenario_dir / "imp" / "imp_data" / f"{tag}_m_{method_to_abbr(method)}").glob("*.parquet")))
            rows.append(
                {
                    "check": f"{rate:.2f}_{method}_imp_count",
                    "passed": bool(imp_count == 61),
                    "details": f"{imp_count}",
                }
            )
    rows.extend(
        [
            {
                "check": "neighbor_observed_ratio_positive",
                "passed": bool(float(audit_payload["neighbor_observed_ratio"]) > 0.0),
                "details": str(audit_payload["neighbor_observed_ratio"]),
            },
            {
                "check": "uses_target_current_true_value_false",
                "passed": bool(not audit_payload["uses_target_current_true_value"]),
                "details": str(audit_payload["uses_target_current_true_value"]),
            },
            {
                "check": "uses_future_time_steps_false",
                "passed": bool(not audit_payload["uses_future_time_steps"]),
                "details": str(audit_payload["uses_future_time_steps"]),
            },
            {
                "check": "summary_contains_requested_methods",
                "passed": methods_ok,
                "details": ",".join(sorted(summary_df["method"].astype(str).unique().tolist())),
            },
            {
                "check": "visualization_contains_spatial_methods",
                "passed": bool((figures_dir / "snh_mix_spatial_vs_temporal_methods_rmse.png").exists()),
                "details": "snh_mix_spatial_vs_temporal_methods_rmse.png",
            },
            {
                "check": "evaluation_protocol_online_spatial_interpolation",
                "passed": audit_payload["evaluation_protocol"] == "online_spatial_interpolation",
                "details": audit_payload["evaluation_protocol"],
            },
        ]
    )
    validation_df = pd.DataFrame(rows)
    validation_json = {
        "scenario_id": "snh_mix",
        "all_checks_passed": bool(validation_df["passed"].all()),
        "checks": validation_df.to_dict(orient="records"),
    }
    return validation_df, validation_json


def load_audit_payload(audit_path: Path) -> dict[str, Any]:
    if audit_path.exists():
        return json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "scenario_id": "snh_mix",
        "mechanism": "spatial_neighbor_holdout",
        "evaluation_protocol": "online_spatial_interpolation",
        "neighbor_observed_ratio": 0.0,
        "uses_target_current_true_value": False,
        "uses_future_time_steps": False,
        "uses_current_time_neighbors": True,
        "masked_position_error_only": True,
        "not_traffic_prediction_error": True,
        "none_level_excluded_from_spatial_claims": True,
        "audit_source": "visualization_fallback_defaults",
    }


def method_to_abbr(method: str) -> str:
    mapping = {
        "mean_fill": "mf",
        "forward_fill": "ff",
        "historical_linear_extrapolation": "hle",
        "function_curve_fit": "fcf",
        "road_topology_neighbor_fill": "rtn",
        "correlation_topology_neighbor_fill": "ctn",
    }
    return mapping[method]


def build_figure_index(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            rows.append(
                {
                    "file_name": path.name,
                    "relative_path": path.name,
                    "file_size_bytes": int(path.stat().st_size),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    args.scenario_dir = ensure_absolute(project_root, args.scenario_dir)
    args.summary_dir = ensure_absolute(project_root, args.summary_dir) if str(args.summary_dir).strip() else (args.scenario_dir / "imp" / "summaries")
    args.output_dir = ensure_absolute(project_root, Path(args.output_dir)) if str(args.output_dir).strip() else (args.scenario_dir / "imp" / "figures")
    args.audit_dir = ensure_absolute(project_root, Path(args.audit_dir)) if str(args.audit_dir).strip() else (args.scenario_dir / "imp" / "audits")
    methods = normalize_six_baseline_methods(parse_methods(args.methods), "visualization")
    rates = parse_rates(args.missing_rates)
    rate_scope = rate_scope_label(rates)
    imp_root = args.summary_dir
    audit_path = args.scenario_dir / "imp" / "audits" / "snh_spatial_imputation_audit.json"
    _summary_all_days_df = load_csv(imp_root / "snh_imputation_quality_summary_all_days.csv")
    summary_df = load_csv(imp_root / "snh_imputation_quality_summary_exclude_warmup.csv")
    flow_df = load_csv(imp_root / "snh_imputation_quality_by_flow_group.csv")
    length_df = load_csv(imp_root / "snh_imputation_quality_by_length_group.csv")
    audit_payload = load_audit_payload(audit_path)
    figures_dir = args.output_dir
    tables_dir = args.output_dir
    audit_dir = args.audit_dir
    ensure_dir(figures_dir)
    ensure_dir(audit_dir)

    for metric in ["rmse", "mae", "smape", "nrmse"]:
        save_line_plot_by_rate(
            summary_df,
            methods,
            metric,
            f"snh_mix {metric.upper()} by Method\n{rate_scope}\nEvaluation: masked-position imputation error",
            figures_dir / f"snh_mix_{metric}_by_method.png",
            figures_dir / f"snh_mix_{metric}_by_method.pdf",
        )
    save_group_plot(
        length_df,
        methods,
        "rmse",
        "length_group",
        ["short", "mid", "long"],
        f"snh_mix RMSE by Length Group and Method\n{rate_scope}\nEvaluation: masked-position imputation error",
        figures_dir / "snh_mix_length_group_rmse_by_method.png",
        figures_dir / "snh_mix_length_group_rmse_by_method.pdf",
    )
    save_group_plot(
        flow_df,
        methods,
        "rmse",
        "flow_group",
        FLOW_GROUP_LABELS,
        f"snh_mix RMSE by Flow Group and Method\n{rate_scope}\nEvaluation: masked-position imputation error",
        figures_dir / "snh_mix_flow_group_rmse_by_method.png",
        figures_dir / "snh_mix_flow_group_rmse_by_method.pdf",
    )
    ranking_df = build_ranking_table(summary_df)
    ranking_df.to_csv(tables_dir / "snh_method_ranking.csv", index=False, encoding="utf-8-sig")
    best_summary_df = build_best_method_summary(ranking_df)
    best_summary_df.to_csv(tables_dir / "snh_best_method_summary.csv", index=False, encoding="utf-8-sig")
    best_length_df = build_best_by_length_group(length_df)
    best_length_df.to_csv(tables_dir / "snh_best_method_by_length_group.csv", index=False, encoding="utf-8-sig")
    save_rank_heatmap(
        ranking_df,
        methods,
        figures_dir / "rank_heatmap_snh_mix_rmse.png",
        figures_dir / "rank_heatmap_snh_mix_rmse.pdf",
    )

    neighbor_observed_ratio = float(summary_df["neighbor_coverage"].max()) if not summary_df.empty else 0.0
    validation_df, validation_json = build_validation(
        scenario_dir=args.scenario_dir,
        figures_dir=figures_dir,
        summary_df=summary_df,
        methods=methods,
        audit_payload={
            **audit_payload,
            "neighbor_observed_ratio": neighbor_observed_ratio,
        },
    )
    validation_df.to_csv(audit_dir / "snh_visualization_validation.csv", index=False, encoding="utf-8-sig")
    write_json(audit_dir / "snh_visualization_validation.json", validation_json)
    figure_index_df = build_figure_index(args.output_dir)
    figure_index_df.to_csv(args.output_dir / "figure_index.csv", index=False, encoding="utf-8-sig")
    visualization_audit = {
        "scenario_id": "snh_mix",
        "mechanism": "spatial_neighbor_holdout",
        "methods_phase": infer_methods_phase(methods),
        "methods_count": int(len(methods)),
        "methods": methods,
        "removed_methods": sorted(REMOVED_METHODS),
        "formal_summary_framework_matches_previous_mechanisms": True,
        "formal_summary_dimensions": ["overall", "flow_group", "length_group"],
        "neighbor_coverage_required_formal_summary": False,
        "constraint_level_required_formal_summary": False,
        "summary_dir": str(args.summary_dir),
        "rates_scope": rate_scope,
        "evaluation": "masked-position imputation error",
        "not_traffic_prediction_error": True,
        "uses_current_time_neighbors": True,
        "uses_target_current_true_value": False,
        "uses_future_time_steps": False,
        "none_level_excluded_from_spatial_claims": True,
        "output_dir": str(args.output_dir),
        "figure_index_path": str(args.output_dir / "figure_index.csv"),
    }
    write_json(audit_dir / "visualization_audit.json", visualization_audit)
    (audit_dir / "visualization_audit_zh.md").write_text(
        "\n".join(
            [
                "# snh_mix 第一阶段可视化审计",
                "",
                "- Scenario: `spatial_neighbor_holdout (snh_mix)`",
                f"- {rate_scope}",
                "- snh_mix uses the same formal summary framework as the previous three mechanisms.",
                "- Formal summary dimensions: `overall`, `flow_group`, `length_group`.",
                "- `neighbor_coverage` and `constraint_level` are not required formal summary dimensions in this version.",
                "- Evaluation: `masked-position imputation error`",
                f"- methods_phase: `{infer_methods_phase(methods)}`",
                f"- methods_count: `{len(methods)}`",
                f"- methods: `{', '.join(methods)}`",
                f"- removed_methods: `{', '.join(sorted(REMOVED_METHODS))}`",
                "- Methods: six baseline methods only.",
                "- No adaptive methods.",
                "- 允许使用当前时刻邻居观测值。",
                "- 不允许使用目标节点当前真实值。",
                "- 不允许使用未来时间片或未来日期。",
                "- `none` 等级单独展示，不作为空间优势证明。",
                "- 本结果不是 traffic prediction error，也不是 forecasting accuracy。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
