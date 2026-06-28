# -*- coding: utf-8 -*-
from __future__ import annotations
"""
CNN 增强实验可视化模块。
只读取 `cfe_core.py` 导出的 CSV 文件并生成图像。
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.ioff()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "simulation_experiments" / "cnn_fed_enhanced_experiments"
DEFAULT_PAPER_READY_DIR = DEFAULT_INPUT_DIR / "paper_ready"
FEDAVG_COLOR = "#0072B2"
BAR_COLOR = "#0072B2"
LINE_COLOR = "#0072B2"
ACCENT_COLOR = "#D55E00"
GRID_ALPHA = 0.35
CATEGORY_PALETTE = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
METHOD_PALETTE = {
    "Independent": "#4C72B0",
    "FedAvg": FEDAVG_COLOR,
    "Proposed": "#55A868",
    "Loss-weighted": "#C44E52",
    "Data-loss weighted": "#8172B3",
}
CLIENT_PALETTE = sns.color_palette("tab10")


def configure_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
    })


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            f"Please run the corresponding *_core.py workflow first."
        )
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_fig(fig, output_dir: Path, filename: str):
    out_path = ensure_dir(output_dir) / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path, pdf_path


def _save_fig_aliases(fig, output_dir: Path, filenames):
    output_dir = ensure_dir(output_dir)
    for filename in filenames:
        out_path = output_dir / filename
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax, rotate: int = 0):
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.8)
    ax.set_axisbelow(True)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _annotate_bar_values(ax, fmt: str = "{:.2f}", offset_ratio: float = 0.015):
    patches = [patch for patch in ax.patches if patch.get_height() == patch.get_height()]
    if not patches:
        return
    max_height = max(patch.get_height() for patch in patches)
    offset = max(max_height * offset_ratio, 0.02)
    for patch in patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_enhanced_dataset_client_timeseries(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_timeseries.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for client_id in sorted(df["client_id"].unique()):
        subset = df[df["client_id"] == client_id]
        ax.plot(subset["time_step"], subset["traffic_flow"], label=f"Client {client_id}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Traffic Flow")
    ax.set_title("Enhanced Dataset Client Time Series")
    ax.legend(ncol=2, fontsize=8)
    _save_fig(fig, output_dir, "enhanced_dataset_client_timeseries.png")


def plot_enhanced_dataset_distribution(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.violinplot(data=df, x="client_id", y="traffic_flow", hue="client_id", palette="tab10", ax=ax, cut=0, legend=False)
    ax.set_title("Enhanced Dataset Distribution")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Traffic Flow")
    _save_fig(fig, output_dir, "enhanced_dataset_distribution.png")


def plot_enhanced_dataset_client_config(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_config.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=df, x="client_id", y="sample_size", hue="client_id", ax=axes[0], palette="tab10", legend=False)
    axes[0].set_title("Sample Size by Client")
    sns.scatterplot(data=df, x="noise_level", y="base_flow", hue="client_id", palette="tab10", s=90, ax=axes[1])
    axes[1].set_title("Noise vs Base Flow")
    _save_fig(fig, output_dir, "enhanced_dataset_client_config.png")


def plot_enhanced_dataset_peak_pattern(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_peak_pattern.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=df, x="hour", y="traffic_flow", hue="client_id", palette="tab10", ax=ax)
    ax.set_title("Peak Pattern by Client")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Traffic Flow")
    _save_fig(fig, output_dir, "enhanced_dataset_peak_pattern.png")


def plot_enhanced_dataset_incident_example(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_incident_example.csv")
    if df.empty:
        raise ValueError("`enhanced_dataset_incident_example.csv` is empty and cannot be visualized.")
    rep_client_id = int(df["client_id"].min())
    rep_df = df[df["client_id"] == rep_client_id]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(rep_df["time_step"], rep_df["traffic_flow"], color="#4C72B0", label="Traffic Flow")
    incident_df = rep_df[rep_df["incident_flag"] == True]
    if not incident_df.empty:
        ax.scatter(incident_df["time_step"], incident_df["traffic_flow"], color="#C44E52", label="Incident", s=20)
    ax.set_title(f"Incident Example (Client {rep_client_id})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Traffic Flow")
    ax.legend()
    _save_fig(fig, output_dir, "enhanced_dataset_incident_example.png")


def plot_enhanced_dataset_client_correlation_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_correlation_matrix.csv")
    matrix = df.pivot(index="source_client", columns="target_client", values="correlation").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Client Correlation Matrix")
    _save_fig(fig, output_dir, "enhanced_dataset_client_correlation_matrix.png")


def plot_enhanced_dataset_node_correlation_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_node_correlation_matrix.csv")
    rep_client_id = int(df["client_id"].min())
    matrix = (
        df[df["client_id"] == rep_client_id]
        .pivot(index="source_node", columns="target_node", values="correlation")
        .sort_index()
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(matrix, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Node Correlation Matrix (Client {rep_client_id})")
    _save_fig(fig, output_dir, "enhanced_dataset_node_correlation_matrix.png")


def plot_main_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_main_metrics.csv")
    pred_df = read_required_csv(input_dir / "cnn_enhanced_main_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, metric_name in enumerate(["rmse", "mae", "mape"]):
        sns.barplot(data=df, x="method", y=metric_name, hue="method", ax=axes[idx], palette=METHOD_PALETTE, legend=False)
        axes[idx].tick_params(axis="x", rotation=12)
        axes[idx].set_title(metric_name.upper())
    _save_fig(fig, output_dir, "cnn_enhanced_main_comparison.png")

    rep_client_id = int(pred_df["client_id"].min())
    rep_df = pred_df[pred_df["client_id"] == rep_client_id]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    gt_df = rep_df[rep_df["method"] == rep_df["method"].iloc[0]]
    ax2.plot(gt_df["sample_id"], gt_df["y_true"], color="black", linestyle="--", linewidth=2, label="Ground Truth")
    for method in rep_df["method"].unique():
        method_df = rep_df[rep_df["method"] == method]
        ax2.plot(method_df["sample_id"], method_df["y_pred"], label=method)
    ax2.set_title(f"Main Prediction Comparison (Client {rep_client_id})")
    ax2.set_xlabel("Sample ID")
    ax2.set_ylabel("Traffic Flow")
    ax2.legend()
    _save_fig(fig2, output_dir, "cnn_enhanced_main_predictions.png")
    raw_df = read_optional_csv(input_dir / "multi_seed_raw_results.csv")
    summary_df = read_optional_csv(input_dir / "multi_seed_summary.csv")
    if raw_df is not None and summary_df is not None:
        plot_multi_seed_metric_summary(summary_df, output_dir)
        plot_multi_seed_metric_boxplot(raw_df, output_dir)
        plot_multi_seed_seed_pairing(raw_df, output_dir)


def plot_multi_seed_metric_summary(summary_df: pd.DataFrame, output_dir: Path):
    target_df = summary_df[summary_df["metric"].isin(["mae", "rmse", "mape"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric_name in zip(axes, ["mae", "rmse", "mape"]):
        metric_df = target_df[target_df["metric"] == metric_name].sort_values("method")
        x_pos = np.arange(len(metric_df))
        ax.bar(
            x_pos,
            metric_df["mean"],
            yerr=metric_df["std"],
            color=[METHOD_PALETTE.get(m, "#4C72B0") for m in metric_df["method"]],
            alpha=0.85,
            capsize=5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_df["method"], rotation=12)
        ax.set_title(f"Multi-seed Mean ± Std: {metric_name.upper()}")
    _save_fig(fig, output_dir, "cnn_enhanced_multi_seed_mean_std.png")


def plot_multi_seed_metric_boxplot(raw_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.boxplot(data=raw_df, x="method", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.stripplot(data=raw_df, x="method", y="rmse", color="black", alpha=0.45, ax=ax)
    ax.set_title("Multi-seed RMSE Distribution")
    _save_fig(fig, output_dir, "cnn_enhanced_multi_seed_rmse_boxplot.png")


def plot_multi_seed_seed_pairing(raw_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(data=raw_df.sort_values("seed"), x="seed", y="rmse", hue="method", style="method", markers=True, dashes=False, palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Per-seed RMSE Comparison")
    _save_fig(fig, output_dir, "cnn_enhanced_multi_seed_seed_pairing.png")


def plot_aggregation_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_aggregation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="method", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax, legend=False)
    ax.set_title("Aggregation Strategy Comparison")
    ax.tick_params(axis="x", rotation=12)
    _save_fig(fig, output_dir, "cnn_enhanced_aggregation.png")


def plot_lambda_sensitivity(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_lambda_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="lambda_value", y="rmse", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_title("Lambda Sensitivity")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("RMSE")
    _save_fig(fig, output_dir, "cnn_enhanced_lambda.png")


def plot_convergence(input_dir: Path, output_dir: Path):
    summary_df = read_optional_csv(input_dir / "multi_seed_convergence_summary.csv")
    raw_df = read_optional_csv(input_dir / "multi_seed_convergence_raw.csv")
    history_df = read_optional_csv(input_dir / "cnn_enhanced_convergence_history.csv")
    if summary_df is not None and not summary_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for method, method_df in summary_df.groupby("method"):
            color = METHOD_PALETTE.get(method, "#4C72B0")
            axes[0].plot(method_df["round"], method_df["avg_train_loss_mean"], color=color, label=method)
            axes[0].fill_between(method_df["round"], method_df["avg_train_loss_mean"] - method_df["avg_train_loss_std"], method_df["avg_train_loss_mean"] + method_df["avg_train_loss_std"], color=color, alpha=0.18)
            axes[1].plot(method_df["round"], method_df["avg_val_rmse_mean"], color=color, label=method)
            axes[1].fill_between(method_df["round"], method_df["avg_val_rmse_mean"] - method_df["avg_val_rmse_std"], method_df["avg_val_rmse_mean"] + method_df["avg_val_rmse_std"], color=color, alpha=0.18)
        axes[0].set_title("Multi-seed Convergence Mean ± Std: Train Loss")
        axes[1].set_title("Multi-seed Convergence Mean ± Std: Validation RMSE")
        axes[0].legend()
        axes[1].legend()
        _save_fig_aliases(fig, output_dir, [
            "cnn_enhanced_convergence.png",
            "cnn_enhanced_multi_seed_convergence_curve.png",
            "convergence_curve.png",
        ])
        return
    df = raw_df if raw_df is not None and not raw_df.empty else history_df
    if df is None or df.empty:
        raise FileNotFoundError("No convergence CSV found. Please run cfe_core.py --workflow convergence first.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=df, x="round", y="avg_train_loss", hue="method", marker="o", ax=axes[0], palette=METHOD_PALETTE)
    axes[0].set_title("Average Training Loss")
    sns.lineplot(data=df, x="round", y="avg_val_rmse", hue="method", marker="s", ax=axes[1], palette=METHOD_PALETTE)
    axes[1].set_title("Average Validation RMSE")
    _save_fig_aliases(fig, output_dir, ["cnn_enhanced_convergence.png", "convergence_curve.png"])


def plot_client_scale(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_client_scale_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="num_clients", y="rmse", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_title("Client Scale Sensitivity")
    _save_fig(fig, output_dir, "cnn_enhanced_client_scale.png")


def plot_noniid(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_noniid_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="noniid_level", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Non-IID Sensitivity")
    _save_fig(fig, output_dir, "cnn_enhanced_noniid.png")


def plot_client_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_client_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="client_id", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Per-client RMSE")
    _save_fig(fig, output_dir, "cnn_enhanced_client_metrics.png")


def plot_peak_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_peak_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=df, x="period", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Peak-period Metrics")
    ax.tick_params(axis="x", rotation=15)
    _save_fig(fig, output_dir, "cnn_enhanced_peak_metrics.png")


def plot_feature_ablation(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_feature_ablation_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="feature_set", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Feature Ablation")
    ax.tick_params(axis="x", rotation=20)
    _save_fig(fig, output_dir, "cnn_enhanced_feature_ablation.png")


def _read_paper_ready_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    df = read_required_csv(input_dir / filename)
    if "method" in df.columns:
        df = df[df["method"] == "FedAvg"].copy()
    if df.empty:
        raise ValueError(f"No FedAvg rows found in {filename}.")
    return df


def plot_paper_ready_noniid(input_dir: Path, output_dir: Path):
    df = _read_paper_ready_csv(input_dir, "cnn_enhanced_noniid_fedavg_only.csv")
    order = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.barplot(
        data=df,
        x="noniid_level",
        y="rmse_mean",
        hue="noniid_level",
        order=order,
        palette=CATEGORY_PALETTE[:3],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("FedAvg under Different Non-IID Levels")
    ax.set_xlabel("Non-IID Level")
    ax.set_ylabel("RMSE")
    _style_axis(ax)
    _annotate_bar_values(ax, fmt="{:.2f}")
    _save_fig(fig, output_dir, "cnn_enhanced_noniid_fedavg_only.png")


def plot_paper_ready_client_scale(input_dir: Path, output_dir: Path):
    df = _read_paper_ready_csv(input_dir, "cnn_enhanced_client_scale_fedavg_only.csv").sort_values("num_clients")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.barplot(
        data=df,
        x="num_clients",
        y="rmse_mean",
        hue="num_clients",
        palette=CATEGORY_PALETTE[: len(df)],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("FedAvg under Different Numbers of Clients")
    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("RMSE")
    _style_axis(ax)
    _annotate_bar_values(ax, fmt="{:.2f}")
    _save_fig(fig, output_dir, "cnn_enhanced_client_scale_fedavg_only.png")


def plot_paper_ready_feature_ablation(input_dir: Path, output_dir: Path):
    df = _read_paper_ready_csv(input_dir, "cnn_enhanced_feature_ablation_fedavg_only.csv")
    feature_order = ["flow_only", "flow_time", "flow_event", "flow_region", "full"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.barplot(
        data=df,
        x="feature_set",
        y="rmse_mean",
        hue="feature_set",
        order=feature_order,
        palette=CATEGORY_PALETTE,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("FedAvg Feature Ablation")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("RMSE")
    _style_axis(ax, rotate=20)
    _annotate_bar_values(ax, fmt="{:.2f}", offset_ratio=0.01)
    _save_fig(fig, output_dir, "cnn_enhanced_feature_ablation_fedavg_only.png")


def run_paper_ready(input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    plot_paper_ready_noniid(input_dir, output_dir)
    plot_paper_ready_client_scale(input_dir, output_dir)
    plot_paper_ready_feature_ablation(input_dir, output_dir)


def run_viz_project(workflow: str, input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    if workflow in ("all", "data_viz"):
        plot_enhanced_dataset_client_timeseries(input_dir, output_dir)
        plot_enhanced_dataset_distribution(input_dir, output_dir)
        plot_enhanced_dataset_client_config(input_dir, output_dir)
        plot_enhanced_dataset_peak_pattern(input_dir, output_dir)
        plot_enhanced_dataset_incident_example(input_dir, output_dir)
        plot_enhanced_dataset_client_correlation_matrix(input_dir, output_dir)
        plot_enhanced_dataset_node_correlation_matrix(input_dir, output_dir)
    if workflow in ("all", "main"):
        plot_main_results(input_dir, output_dir)
    if workflow in ("all", "aggregation"):
        plot_aggregation_results(input_dir, output_dir)
    if workflow in ("all", "lambda"):
        plot_lambda_sensitivity(input_dir, output_dir)
    if workflow in ("all", "convergence"):
        plot_convergence(input_dir, output_dir)
    if workflow in ("all", "client_scale"):
        plot_client_scale(input_dir, output_dir)
    if workflow in ("all", "noniid"):
        plot_noniid(input_dir, output_dir)
    if workflow in ("all", "client_metrics"):
        plot_client_metrics(input_dir, output_dir)
    if workflow in ("all", "peak"):
        plot_peak_metrics(input_dir, output_dir)
    if workflow in ("all", "feature_ablation"):
        plot_feature_ablation(input_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="CNN Enhanced Visualization")
    parser.add_argument(
        "--workflow",
        choices=[
            "all", "data_viz", "main", "aggregation", "lambda", "convergence",
            "client_scale", "noniid", "client_metrics", "peak", "feature_ablation",
        ],
        default="all",
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help="Generate FedAvg-only paper-ready PNG/PDF figures.",
    )
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "paper_ready" if args.paper_ready else input_dir)
    if args.paper_ready:
        run_paper_ready(input_dir / "paper_ready", output_dir)
        return
    run_viz_project(args.workflow, input_dir, output_dir)


if __name__ == "__main__":
    main()
