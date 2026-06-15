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
    "zero_fill": "Zero fill",
    "forward_fill": "Forward fill",
    "historical_linear_extrapolation": "Historical linear extrapolation",
    "road_topology_neighbor_fill": "Road-topology neighbor",
    "function_curve_fit": "Function curve fit",
    "topology_function_hybrid": "Topology-function hybrid",
}

DEFAULT_METHODS = [
    "zero_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "road_topology_neighbor_fill",
    "function_curve_fit",
    "topology_function_hybrid",
]

FLOW_GROUPS = ["low_flow", "mid_flow", "high_flow"]
EXPECTED_RATES = [0.05, 0.10, 0.20, 0.30]
FORMAL_FIGURE_NOTES = "Formal six-method direct comparison"
ZOOM_NOTES = "Zoom view excluding zero fill"
FLOW_GROUP_NOTES = "Flow-group comparison"
HEATMAP_NOTES = "Auxiliary rank visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualization-only comparisons for global missingness imputation summaries."
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        default=Path("results/real_data_global_missingness_setting"),
    )
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=Path("results/real_data_global_missingness_setting/summaries"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("results/real_data_global_missingness_setting/figures"),
    )
    parser.add_argument(
        "--audit_dir",
        type=Path,
        default=Path("results/real_data_global_missingness_setting/audits"),
    )
    parser.add_argument("--missing_rates", type=str, default="0.05,0.10,0.20,0.30")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--main_summary", type=str, default="imputation_quality_summary_exclude_warmup.csv")
    parser.add_argument("--flow_group_summary", type=str, default="imputation_quality_by_flow_group.csv")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_method_list(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    return [METHOD_ALIASES.get(method, method) for method in methods]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required summary file not found: {path}")
    return pd.read_csv(path)


def normalize_methods(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["method"] = normalized["method"].replace(METHOD_ALIASES)
    return normalized


def filter_expected_scope(df: pd.DataFrame, rates: list[float], methods: list[str]) -> pd.DataFrame:
    filtered = df.loc[df["method"].isin(methods)].copy()
    filtered = filtered.loc[np.isclose(filtered["missing_rate"].to_numpy(dtype=float)[:, None], np.array(rates)[None, :]).any(axis=1)]
    return filtered


def validate_main_summary(df: pd.DataFrame, rates: list[float], methods: list[str], expected_file_name: str) -> dict[str, Any]:
    required_columns = {
        "missing_rate",
        "method",
        "flow_group",
        "mae",
        "rmse",
        "exclude_warmup",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"main summary missing required columns: {missing_columns}")

    metric_choice = "smape" if "smape" in df.columns else "mape" if "mape" in df.columns else None
    if metric_choice is None:
        raise RuntimeError("main summary must contain smape or mape")
    if "nrmse" not in df.columns:
        raise RuntimeError("main summary must contain nrmse for NRMSE plotting")

    overall_df = df.loc[df["flow_group"] == "overall"].copy()
    if overall_df.empty:
        raise RuntimeError("main summary does not contain overall rows")

    observed_rates = sorted({round(float(value), 2) for value in overall_df["missing_rate"].unique().tolist()})
    expected_rates = [round(value, 2) for value in rates]
    if observed_rates != expected_rates:
        raise RuntimeError(f"main summary rates mismatch: expected {expected_rates}, got {observed_rates}")

    observed_methods = sorted(overall_df["method"].unique().tolist())
    if sorted(methods) != observed_methods:
        raise RuntimeError(f"main summary methods mismatch: expected {sorted(methods)}, got {observed_methods}")

    method_rate_counts = overall_df.groupby("method")["missing_rate"].nunique()
    incomplete = method_rate_counts.loc[method_rate_counts != len(rates)]
    if not incomplete.empty:
        raise RuntimeError(f"main summary missing rates for methods: {incomplete.to_dict()}")

    uses_exclude_warmup = bool(overall_df["exclude_warmup"].all())
    expected_exclude_file = expected_file_name == "imputation_quality_summary_exclude_warmup.csv"
    if expected_exclude_file and not uses_exclude_warmup:
        raise RuntimeError("main summary is expected to use exclude_warmup rows, but exclude_warmup is not all True")

    if any(np.isclose(overall_df["missing_rate"], 0.0)):
        raise RuntimeError("0% control rows must not be plotted in masked-position error figures")

    return {
        "metric_choice": metric_choice,
        "uses_exclude_warmup": uses_exclude_warmup,
        "observed_rates": observed_rates,
        "observed_methods": observed_methods,
    }


def validate_flow_group_summary(df: pd.DataFrame, rates: list[float], methods: list[str]) -> None:
    required_columns = {"missing_rate", "method", "flow_group", "rmse"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"flow-group summary missing required columns: {missing_columns}")

    subset = df.loc[df["flow_group"].isin(FLOW_GROUPS)].copy()
    if subset.empty:
        raise RuntimeError("flow-group summary does not contain low/mid/high flow rows")

    observed_groups = sorted(subset["flow_group"].unique().tolist())
    if observed_groups != sorted(FLOW_GROUPS):
        raise RuntimeError(f"flow-group summary groups mismatch: expected {sorted(FLOW_GROUPS)}, got {observed_groups}")

    for method in methods:
        method_df = subset.loc[subset["method"] == method]
        if sorted({round(float(value), 2) for value in method_df["missing_rate"].unique().tolist()}) != [round(value, 2) for value in rates]:
            raise RuntimeError(f"flow-group summary missing rates for method: {method}")


def prepare_plot_df(df: pd.DataFrame, rates: list[float], methods: list[str], flow_group: str = "overall") -> pd.DataFrame:
    plot_df = df.loc[df["flow_group"] == flow_group].copy()
    plot_df = plot_df.loc[plot_df["method"].isin(methods)].copy()
    plot_df = plot_df.loc[plot_df["missing_rate"].isin(rates)].copy()
    plot_df["display_method"] = plot_df["method"].map(METHOD_DISPLAY)
    return plot_df.sort_values(["method", "missing_rate"]).reset_index(drop=True)


def rate_tick_labels(rates: list[float]) -> list[str]:
    return [f"{int(round(rate * 100))}%" for rate in rates]


def save_line_plot(
    plot_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_png: Path,
    output_pdf: Path,
    methods: list[str],
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    x_values = sorted(plot_df["missing_rate"].unique().tolist())
    for method in methods:
        method_df = plot_df.loc[plot_df["method"] == method].sort_values("missing_rate")
        axis.plot(
            method_df["missing_rate"],
            method_df[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            label=METHOD_DISPLAY[method],
        )
    axis.set_title(title)
    axis.set_xlabel("Missing rate")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_values, rate_tick_labels(x_values))
    axis.grid(alpha=0.3)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_flow_group_plot(
    flow_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    methods: list[str],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    x_values = sorted(flow_df["missing_rate"].unique().tolist())
    for axis, group_label in zip(axes, FLOW_GROUPS):
        group_df = flow_df.loc[flow_df["flow_group"] == group_label]
        for method in methods:
            method_df = group_df.loc[group_df["method"] == method].sort_values("missing_rate")
            axis.plot(
                method_df["missing_rate"],
                method_df["rmse"],
                marker="o",
                linewidth=2,
                markersize=5,
                label=METHOD_DISPLAY[method],
            )
        axis.set_title(group_label.replace("_", " ").title())
        axis.set_ylabel("RMSE")
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Missing rate")
    axes[-1].set_xticks(x_values, rate_tick_labels(x_values))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("RMSE by method across flow groups", y=0.995)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_ranking_df(summary_df: pd.DataFrame, metrics: list[str], methods: list[str]) -> pd.DataFrame:
    overall_df = summary_df.loc[summary_df["flow_group"] == "overall"].copy()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for rate in sorted(overall_df["missing_rate"].unique().tolist()):
            rate_df = overall_df.loc[overall_df["missing_rate"] == rate, ["method", metric]].copy()
            rate_df = rate_df.rename(columns={metric: "value"}).sort_values("value", ascending=True).reset_index(drop=True)
            rate_df["rank"] = np.arange(1, len(rate_df) + 1)
            for _, row in rate_df.iterrows():
                rows.append(
                    {
                        "missing_rate": float(rate),
                        "metric": metric.upper() if metric != "smape" else "sMAPE",
                        "rank": int(row["rank"]),
                        "method": str(row["method"]),
                        "value": float(row["value"]),
                    }
                )
    ranking_df = pd.DataFrame(rows)
    ranking_df["method"] = pd.Categorical(ranking_df["method"], categories=methods, ordered=True)
    return ranking_df.sort_values(["metric", "missing_rate", "rank"]).reset_index(drop=True)


def save_rank_heatmap(ranking_df: pd.DataFrame, output_png: Path, output_pdf: Path, methods: list[str]) -> None:
    rmse_df = ranking_df.loc[ranking_df["metric"] == "RMSE"].copy()
    pivot = (
        rmse_df.pivot(index="method", columns="missing_rate", values="rank")
        .reindex(index=methods)
        .sort_index(axis=1)
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    heatmap = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    axis.set_title("RMSE rank heatmap by method and missing rate")
    axis.set_xlabel("Missing rate")
    axis.set_ylabel("Method")
    axis.set_xticks(np.arange(len(pivot.columns)), rate_tick_labels(pivot.columns.tolist()))
    axis.set_yticks(np.arange(len(pivot.index)), [METHOD_DISPLAY[method] for method in pivot.index.tolist()])
    for row_index in range(pivot.shape[0]):
        for col_index in range(pivot.shape[1]):
            axis.text(col_index, row_index, str(int(pivot.iat[row_index, col_index])), ha="center", va="center", color="white")
    cbar = fig.colorbar(heatmap, ax=axis)
    cbar.set_label("Rank")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_figure_index(entries: list[dict[str, Any]], figure_dir: Path) -> pd.DataFrame:
    figure_index_df = pd.DataFrame(entries).sort_values("figure_file").reset_index(drop=True)
    figure_index_df.to_csv(figure_dir / "figure_index.csv", index=False, encoding="utf-8-sig")
    return figure_index_df


def build_visualization_audit(
    *,
    args: argparse.Namespace,
    main_summary_path: Path,
    flow_group_summary_path: Path,
    metric_choice: str,
    uses_exclude_warmup: bool,
    figure_index_df: pd.DataFrame,
    generated_figures: dict[str, bool],
    methods: list[str],
    rates: list[float],
) -> dict[str, Any]:
    return {
        "experiment_dir": str(args.experiment_dir.resolve()),
        "summary_dir": str(args.summary_dir.resolve()),
        "figure_dir": str(args.figure_dir.resolve()),
        "audit_dir": str(args.audit_dir.resolve()),
        "data_sources": {
            "main_summary": str(main_summary_path.resolve()),
            "flow_group_summary": str(flow_group_summary_path.resolve()),
        },
        "uses_exclude_warmup_main_summary": uses_exclude_warmup,
        "missing_rates": rates,
        "contains_required_rates": True,
        "contains_required_methods": True,
        "methods": methods,
        "metric_choice_for_percentage_plot": "sMAPE" if metric_choice == "smape" else "MAPE",
        "has_rmse": True,
        "has_mae": True,
        "has_percentage_metric": True,
        "has_nrmse": generated_figures["multirate_nrmse_by_method"],
        "masked_position_error_only": True,
        "includes_zero_percent_control_in_main_plots": False,
        "uses_relative_to_forward_fill_main_plot": False,
        "forward_fill_only_one_method": True,
        "reran_impute": False,
        "regenerated_masks": False,
        "regenerated_missing_datasets": False,
        "regenerated_imputed_datasets": False,
        "generated_figures": generated_figures,
        "figure_index_csv": str((args.figure_dir / "figure_index.csv").resolve()),
        "figure_count": int(len(figure_index_df)),
        "note": "All figures are based on masked-position imputation error, not FedAvg or Independent traffic prediction error.",
    }


def write_visualization_markdown(path: Path, audit_payload: dict[str, Any]) -> None:
    lines = [
        "# 全局缺失值补全结果可视化审计报告",
        "",
        "## 1. 数据源",
        "",
        f"- 主结果 summary: `{audit_payload['data_sources']['main_summary']}`",
        f"- 流量组 summary: `{audit_payload['data_sources']['flow_group_summary']}`",
        f"- 是否使用 exclude_warmup 主结果: `{audit_payload['uses_exclude_warmup_main_summary']}`",
        "",
        "## 2. 校验结论",
        "",
        f"- 是否包含 5%、10%、20%、30%: `{audit_payload['contains_required_rates']}`",
        f"- 是否包含 6 个方法: `{audit_payload['contains_required_methods']}`",
        f"- 是否排除 0% masked-position error: `{not audit_payload['includes_zero_percent_control_in_main_plots']}`",
        f"- 是否未使用 relative-to-forward-fill 正式主图: `{not audit_payload['uses_relative_to_forward_fill_main_plot']}`",
        f"- 是否未重新运行 impute: `{not audit_payload['reran_impute']}`",
        f"- 是否未重新生成 masks: `{not audit_payload['regenerated_masks']}`",
        f"- 是否未重新生成 missing_datasets: `{not audit_payload['regenerated_missing_datasets']}`",
        "",
        "## 3. 图件输出",
        "",
    ]
    for key, value in audit_payload["generated_figures"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 4. 说明",
            "",
            "- 当前所有图件均基于 masked-position imputation error，不代表 FedAvg / Independent 交通流预测性能。",
            "- forward_fill 只是 6 个普通方法之一，不作为正式主图 baseline。",
            "- nonzero zoom 图仅用于观察非 zero_fill 方法间差异，不替代完整六方法主图。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rates = parse_float_list(args.missing_rates)
    methods = parse_method_list(args.methods)

    ensure_dir(args.figure_dir)
    ensure_dir(args.audit_dir)

    main_summary_path = args.summary_dir / args.main_summary
    flow_group_summary_path = args.summary_dir / args.flow_group_summary

    main_summary_df = filter_expected_scope(normalize_methods(load_csv(main_summary_path)), rates, methods)
    flow_group_df = filter_expected_scope(normalize_methods(load_csv(flow_group_summary_path)), rates, methods)

    validation_info = validate_main_summary(main_summary_df, rates, methods, args.main_summary)
    validate_flow_group_summary(flow_group_df, rates, methods)

    overall_df = prepare_plot_df(main_summary_df, rates, methods, flow_group="overall")
    flow_plot_df = flow_group_df.loc[flow_group_df["flow_group"].isin(FLOW_GROUPS)].copy()
    flow_plot_df["display_method"] = flow_plot_df["method"].map(METHOD_DISPLAY)

    save_line_plot(
        overall_df,
        metric="rmse",
        ylabel="RMSE",
        title="RMSE by method across missing rates",
        output_png=args.figure_dir / "multirate_rmse_by_method.png",
        output_pdf=args.figure_dir / "multirate_rmse_by_method.pdf",
        methods=methods,
    )
    save_line_plot(
        overall_df,
        metric="mae",
        ylabel="MAE",
        title="MAE by method across missing rates",
        output_png=args.figure_dir / "multirate_mae_by_method.png",
        output_pdf=args.figure_dir / "multirate_mae_by_method.pdf",
        methods=methods,
    )
    percentage_metric = validation_info["metric_choice"]
    percentage_label = "sMAPE" if percentage_metric == "smape" else "MAPE"
    save_line_plot(
        overall_df,
        metric=percentage_metric,
        ylabel=percentage_label,
        title=f"{percentage_label} by method across missing rates",
        output_png=args.figure_dir / f"multirate_{percentage_metric}_by_method.png",
        output_pdf=args.figure_dir / f"multirate_{percentage_metric}_by_method.pdf",
        methods=methods,
    )
    save_line_plot(
        overall_df,
        metric="nrmse",
        ylabel="NRMSE",
        title="NRMSE by method across missing rates",
        output_png=args.figure_dir / "multirate_nrmse_by_method.png",
        output_pdf=args.figure_dir / "multirate_nrmse_by_method.pdf",
        methods=methods,
    )

    save_line_plot(
        overall_df.loc[overall_df["method"] != "zero_fill"].copy(),
        metric="rmse",
        ylabel="RMSE",
        title="RMSE by method excluding zero fill for readability",
        output_png=args.figure_dir / "multirate_rmse_by_method_nonzero_zoom.png",
        output_pdf=args.figure_dir / "multirate_rmse_by_method_nonzero_zoom.pdf",
        methods=[method for method in methods if method != "zero_fill"],
    )
    save_flow_group_plot(
        flow_plot_df,
        output_png=args.figure_dir / "multirate_flow_group_rmse_by_method.png",
        output_pdf=args.figure_dir / "multirate_flow_group_rmse_by_method.pdf",
        methods=methods,
    )

    ranking_metrics = ["rmse", "mae", percentage_metric]
    ranking_df = build_ranking_df(main_summary_df, ranking_metrics, methods)
    ranking_df.to_csv(args.figure_dir / "method_ranking_by_rate.csv", index=False, encoding="utf-8-sig")
    save_rank_heatmap(
        ranking_df,
        output_png=args.figure_dir / "method_rank_heatmap_rmse.png",
        output_pdf=args.figure_dir / "method_rank_heatmap_rmse.pdf",
        methods=methods,
    )

    figure_index_entries = [
        {
            "figure_file": "multirate_rmse_by_method.png",
            "figure_type": "line",
            "metric": "RMSE",
            "method_scope": "all_6_methods",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": True,
            "notes": FORMAL_FIGURE_NOTES,
        },
        {
            "figure_file": "multirate_mae_by_method.png",
            "figure_type": "line",
            "metric": "MAE",
            "method_scope": "all_6_methods",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": True,
            "notes": FORMAL_FIGURE_NOTES,
        },
        {
            "figure_file": f"multirate_{percentage_metric}_by_method.png",
            "figure_type": "line",
            "metric": "sMAPE" if percentage_metric == "smape" else "MAPE",
            "method_scope": "all_6_methods",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": True,
            "notes": FORMAL_FIGURE_NOTES,
        },
        {
            "figure_file": "multirate_nrmse_by_method.png",
            "figure_type": "line",
            "metric": "NRMSE",
            "method_scope": "all_6_methods",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": True,
            "notes": FORMAL_FIGURE_NOTES,
        },
        {
            "figure_file": "multirate_rmse_by_method_nonzero_zoom.png",
            "figure_type": "line",
            "metric": "RMSE",
            "method_scope": "nonzero_methods",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": False,
            "notes": ZOOM_NOTES,
        },
        {
            "figure_file": "multirate_flow_group_rmse_by_method.png",
            "figure_type": "line",
            "metric": "RMSE",
            "method_scope": "all_6_methods_by_flow_group",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": True,
            "notes": FLOW_GROUP_NOTES,
        },
        {
            "figure_file": "method_rank_heatmap_rmse.png",
            "figure_type": "heatmap",
            "metric": "RMSE",
            "method_scope": "all_6_methods",
            "rate_scope": "5_10_20_30",
            "is_formal_main_figure": False,
            "notes": HEATMAP_NOTES,
        },
    ]
    figure_index_entries.extend(
        [
            {
                "figure_file": "method_ranking_by_rate.csv",
                "figure_type": "table",
                "metric": "RMSE_MAE_" + ("sMAPE" if percentage_metric == "smape" else "MAPE"),
                "method_scope": "all_6_methods",
                "rate_scope": "5_10_20_30",
                "is_formal_main_figure": False,
                "notes": "Auxiliary ranking table",
            },
            {
                "figure_file": "figure_index.csv",
                "figure_type": "index",
                "metric": "multiple",
                "method_scope": "all_outputs",
                "rate_scope": "5_10_20_30",
                "is_formal_main_figure": False,
                "notes": "Visualization output inventory",
            },
        ]
    )
    figure_index_df = build_figure_index(figure_index_entries, args.figure_dir)

    generated_figures = {
        "multirate_rmse_by_method": True,
        "multirate_mae_by_method": True,
        "multirate_smape_or_mape_by_method": True,
        "multirate_nrmse_by_method": True,
        "multirate_rmse_by_method_nonzero_zoom": True,
        "multirate_flow_group_rmse_by_method": True,
        "method_rank_heatmap_rmse": True,
        "figure_index_csv": True,
    }
    audit_payload = build_visualization_audit(
        args=args,
        main_summary_path=main_summary_path,
        flow_group_summary_path=flow_group_summary_path,
        metric_choice=percentage_metric,
        uses_exclude_warmup=validation_info["uses_exclude_warmup"],
        figure_index_df=figure_index_df,
        generated_figures=generated_figures,
        methods=methods,
        rates=rates,
    )
    write_json(args.audit_dir / "visualization_audit.json", audit_payload)
    write_visualization_markdown(args.audit_dir / "visualization_audit_zh.md", audit_payload)


if __name__ == "__main__":
    main()
