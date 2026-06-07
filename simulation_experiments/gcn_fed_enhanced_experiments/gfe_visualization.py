# -*- coding: utf-8 -*-
"""
GCN 增强实验可视化模块。
只读取 `gfe_core.py` 导出的 CSV 文件并生成图像。
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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "simulation_experiments" / "gcn_fed_enhanced_experiments"
DEFAULT_PAPER_READY_DIR = DEFAULT_INPUT_DIR / "paper_ready"
FEDAVG_COLOR = "#0072B2"
BAR_COLOR = "#0072B2"
LINE_COLOR = "#0072B2"
ACCENT_COLOR = "#D55E00"
GRID_ALPHA = 0.35
METHOD_PALETTE = {
    "FedAvg": FEDAVG_COLOR,
    "Proposed": "#55A868",
    "Independent": "#4C72B0",
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


def _annotate_bar_values(ax, fmt: str = "{:.2f}", offset_ratio: float = 0.01):
    patches = [patch for patch in ax.patches if patch.get_height() == patch.get_height()]
    if not patches:
        return
    max_height = max(patch.get_height() for patch in patches)
    offset = max(max_height * offset_ratio, 0.03)
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


def _plot_matrix(df: pd.DataFrame, title: str, output_dir: Path, filename: str, cmap="viridis", annot=False):
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    sns.heatmap(df, cmap=cmap, center=0 if cmap == "coolwarm" else None, annot=annot, fmt=".2f", ax=ax)
    ax.set_title(title)
    _save_fig(fig, output_dir, filename)


def plot_enhanced_dataset_client_timeseries(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_timeseries.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=df, x="time_step", y="traffic_flow", hue="client_id", palette="tab10", ax=ax)
    ax.set_title("Enhanced Dataset Client Time Series")
    _save_fig(fig, output_dir, "enhanced_dataset_client_timeseries.png")


def plot_enhanced_dataset_distribution(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(data=df, x="client_id", y="traffic_flow", hue="client_id", palette="tab10", showfliers=False, ax=ax, legend=False)
    ax.set_title("Enhanced Dataset Distribution")
    _save_fig(fig, output_dir, "enhanced_dataset_distribution.png")


def plot_enhanced_dataset_client_config(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_config.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=df, x="client_id", y="sample_size", hue="client_id", palette="tab10", ax=axes[0], legend=False)
    axes[0].set_title("Client Sample Size")
    sns.scatterplot(data=df, x="noise_level", y="base_flow", hue="client_id", palette="tab10", s=90, ax=axes[1])
    axes[1].set_title("Noise Level vs Base Flow")
    _save_fig(fig, output_dir, "enhanced_dataset_client_config.png")


def plot_enhanced_dataset_peak_pattern(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_peak_pattern.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=df, x="hour", y="traffic_flow", hue="client_id", palette="tab10", ax=ax)
    ax.set_title("Peak Pattern by Client")
    _save_fig(fig, output_dir, "enhanced_dataset_peak_pattern.png")


def plot_enhanced_dataset_incident_example(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_incident_example.csv")
    rep_client_id = int(df["client_id"].min())
    rep_df = df[df["client_id"] == rep_client_id]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(rep_df["time_step"], rep_df["traffic_flow"], color="#4C72B0")
    incident_df = rep_df[rep_df["incident_flag"] == True]
    if not incident_df.empty:
        ax.scatter(incident_df["time_step"], incident_df["traffic_flow"], color="#C44E52", s=16)
    ax.set_title(f"Incident Example (Client {rep_client_id})")
    _save_fig(fig, output_dir, "enhanced_dataset_incident_example.png")


def plot_enhanced_dataset_client_correlation_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_correlation_matrix.csv")
    matrix = df.pivot(index="source_client", columns="target_client", values="correlation").sort_index().sort_index(axis=1)
    _plot_matrix(matrix, "Client Correlation Matrix", output_dir, "enhanced_dataset_client_correlation_matrix.png", cmap="coolwarm", annot=True)


def plot_enhanced_dataset_node_correlation_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_node_correlation_matrix.csv")
    rep_client_id = int(df["client_id"].min())
    matrix = (
        df[df["client_id"] == rep_client_id]
        .pivot(index="source_node", columns="target_node", values="correlation")
        .sort_index()
        .sort_index(axis=1)
    )
    _plot_matrix(matrix, f"Node Correlation Matrix (Client {rep_client_id})", output_dir, "enhanced_dataset_node_correlation_matrix.png", cmap="coolwarm")


def plot_gcn_fixed_adjacency_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_fixed_adjacency_matrix.csv")
    _plot_matrix(df, "Fixed Adjacency Matrix", output_dir, "enhanced_gcn_fixed_adjacency.png", cmap="mako", annot=True)


def plot_gcn_dynamic_adjacency_peak(input_dir: Path, output_dir: Path):
    morning = read_required_csv(input_dir / "enhanced_gcn_dynamic_adjacency_morning_peak.csv")
    evening = read_required_csv(input_dir / "enhanced_gcn_dynamic_adjacency_evening_peak.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(morning, cmap="rocket", ax=axes[0])
    axes[0].set_title("Morning Peak Dynamic Adjacency")
    sns.heatmap(evening, cmap="rocket", ax=axes[1])
    axes[1].set_title("Evening Peak Dynamic Adjacency")
    _save_fig(fig, output_dir, "enhanced_gcn_dynamic_peak.png")


def plot_gcn_dynamic_adjacency_offpeak(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_dynamic_adjacency_offpeak.csv")
    _plot_matrix(df, "Off-peak Dynamic Adjacency", output_dir, "enhanced_gcn_dynamic_offpeak.png", cmap="crest")


def plot_gcn_fixed_dynamic_adjacency_comparison(input_dir: Path, output_dir: Path):
    fixed_df = read_required_csv(input_dir / "enhanced_gcn_fixed_adjacency_matrix.csv")
    dynamic_df = read_required_csv(input_dir / "enhanced_gcn_dynamic_adjacency_morning_peak.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(fixed_df, cmap="mako", ax=axes[0])
    axes[0].set_title("Fixed")
    sns.heatmap(dynamic_df, cmap="rocket", ax=axes[1])
    axes[1].set_title("Dynamic Morning Peak")
    _save_fig(fig, output_dir, "enhanced_gcn_fixed_dynamic_comparison.png")


def plot_gcn_functional_similarity_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_functional_similarity_matrix.csv")
    _plot_matrix(df, "Functional Similarity Matrix", output_dir, "enhanced_gcn_functional_similarity.png", cmap="coolwarm")


def plot_gcn_congestion_delay_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_congestion_delay_matrix.csv")
    _plot_matrix(df, "Congestion Delay Matrix", output_dir, "enhanced_gcn_congestion_delay_matrix.png", cmap="YlOrRd", annot=True)


def plot_gcn_congestion_strength_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_congestion_strength_matrix.csv")
    _plot_matrix(df, "Congestion Strength Matrix", output_dir, "enhanced_gcn_congestion_strength_matrix.png", cmap="magma")


def plot_gcn_congestion_delay_distribution(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_congestion_delay.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(data=df[df["source_node"] != df["target_node"]], x="delay_rounds", bins=4, ax=ax, color="#DD8452")
    ax.set_title("Congestion Delay Distribution")
    _save_fig(fig, output_dir, "enhanced_gcn_congestion_delay_distribution.png")


def plot_gcn_congestion_delay_interaction(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_congestion_delay_interaction.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.scatterplot(data=df, x="delay_rounds", y="strength", hue="source_node", palette="tab10", ax=ax)
    ax.set_title("Congestion Delay Interaction")
    _save_fig(fig, output_dir, "enhanced_gcn_congestion_delay_interaction.png")


def plot_gcn_peak_graph_change(input_dir: Path, output_dir: Path):
    morning = read_required_csv(input_dir / "enhanced_gcn_dynamic_adjacency_morning_peak.csv")
    offpeak = read_required_csv(input_dir / "enhanced_gcn_dynamic_adjacency_offpeak.csv")
    diff = morning - offpeak
    _plot_matrix(diff, "Peak Graph Change (Morning - Offpeak)", output_dir, "enhanced_gcn_peak_graph_change.png", cmap="coolwarm")


def plot_gcn_fixed_vs_dynamic_training_comparison(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_fixed_vs_dynamic_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="graph_type", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Fixed vs Dynamic Training Comparison")
    ax.tick_params(axis="x", rotation=15)
    _save_fig(fig, output_dir, "gcn_enhanced_fixed_vs_dynamic.png")


def plot_gcn_congestion_delay_training_comparison(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_congestion_delay_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="graph_type", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Congestion Delay Training Comparison")
    _save_fig(fig, output_dir, "gcn_enhanced_congestion_delay_comp.png")


def plot_main_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_main_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, metric_name in enumerate(["rmse", "mae", "mape"]):
        sns.barplot(data=df, x="method", y=metric_name, hue="method", ax=axes[idx], palette=METHOD_PALETTE, legend=False)
        axes[idx].tick_params(axis="x", rotation=12)
        axes[idx].set_title(metric_name.upper())
    _save_fig(fig, output_dir, "gcn_enhanced_main_results.png")
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
    _save_fig(fig, output_dir, "gcn_enhanced_multi_seed_mean_std.png")


def plot_multi_seed_metric_boxplot(raw_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.boxplot(data=raw_df, x="method", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.stripplot(data=raw_df, x="method", y="rmse", color="black", alpha=0.45, ax=ax)
    ax.set_title("Multi-seed RMSE Distribution")
    _save_fig(fig, output_dir, "gcn_enhanced_multi_seed_rmse_boxplot.png")


def plot_multi_seed_seed_pairing(raw_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(data=raw_df.sort_values("seed"), x="seed", y="rmse", hue="method", style="method", markers=True, dashes=False, palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Per-seed RMSE Comparison")
    _save_fig(fig, output_dir, "gcn_enhanced_multi_seed_seed_pairing.png")


def plot_aggregation_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_aggregation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=df, x="method", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax, legend=False)
    ax.tick_params(axis="x", rotation=12)
    ax.set_title("Aggregation Strategy Comparison")
    _save_fig(fig, output_dir, "gcn_enhanced_aggregation.png")


def plot_lambda_sensitivity(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_lambda_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="lambda_value", y="rmse", hue="method", marker="o", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Lambda Sensitivity")
    _save_fig(fig, output_dir, "gcn_enhanced_lambda.png")


def plot_convergence(input_dir: Path, output_dir: Path):
    summary_df = read_optional_csv(input_dir / "multi_seed_convergence_summary.csv")
    raw_df = read_optional_csv(input_dir / "multi_seed_convergence_raw.csv")
    history_df = read_optional_csv(input_dir / "gcn_enhanced_convergence_history.csv")
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
            "gcn_enhanced_convergence.png",
            "gcn_enhanced_multi_seed_convergence_curve.png",
            "convergence_curve.png",
        ])
        return
    df = raw_df if raw_df is not None and not raw_df.empty else history_df
    if df is None or df.empty:
        raise FileNotFoundError("No convergence CSV found. Please run gfe_core.py --workflow convergence first.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=df, x="round", y="avg_train_loss", hue="method", palette=METHOD_PALETTE, marker="o", ax=axes[0])
    axes[0].set_title("Average Training Loss")
    sns.lineplot(data=df, x="round", y="avg_val_rmse", hue="method", palette=METHOD_PALETTE, marker="s", ax=axes[1])
    axes[1].set_title("Average Validation RMSE")
    _save_fig_aliases(fig, output_dir, ["gcn_enhanced_convergence.png", "convergence_curve.png"])


def plot_client_scale(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_client_scale_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="num_clients", y="rmse", hue="method", palette=METHOD_PALETTE, marker="o", ax=ax)
    ax.set_title("Client Scale Sensitivity")
    _save_fig(fig, output_dir, "gcn_enhanced_client_scale.png")


def plot_noniid(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_noniid_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="noniid_level", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Non-IID Sensitivity")
    _save_fig(fig, output_dir, "gcn_enhanced_noniid.png")


def plot_client_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_client_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="client_id", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Per-client RMSE")
    _save_fig(fig, output_dir, "gcn_enhanced_client_metrics.png")


def plot_peak_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_peak_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=df, x="period", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("Peak-period Metrics")
    _save_fig(fig, output_dir, "gcn_enhanced_peak_metrics.png")


def plot_paper_ready_fixed_vs_dynamic(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_fixed_vs_dynamic_summary.csv")
    df = df[df["method"] == "FedAvg"].copy()
    if df.empty:
        raise ValueError("No FedAvg rows found in gcn_enhanced_fixed_vs_dynamic_summary.csv.")
    print("Single-seed preliminary result; interpret as trend evidence only.")
    print("单种子初步结果，仅作为趋势性证据。")
    graph_order = ["Fixed", "Dynamic-Morning", "Dynamic-Evening", "Dynamic-Offpeak"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.barplot(
        data=df,
        x="graph_type",
        y="mse_mean",
        order=graph_order,
        color=BAR_COLOR,
        ax=ax,
    )
    ax.set_title("FedAvg with Fixed and Dynamic Graph Structures")
    ax.set_xlabel("Graph Setting")
    ax.set_ylabel("MSE")
    _style_axis(ax, rotate=12)
    fig.subplots_adjust(top=0.82)
    _annotate_bar_values(ax, fmt="{:.2f}", offset_ratio=0.003)
    fig.text(
        0.5,
        0.87,
        "Single-seed preliminary result; interpret as trend evidence only.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )
    _save_fig(fig, output_dir, "gcn_fixed_vs_dynamic_fedavg_only.png")


def run_paper_ready(input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    plot_paper_ready_fixed_vs_dynamic(input_dir, output_dir)


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
        plot_gcn_fixed_adjacency_matrix(input_dir, output_dir)
        plot_gcn_dynamic_adjacency_peak(input_dir, output_dir)
        plot_gcn_dynamic_adjacency_offpeak(input_dir, output_dir)
        plot_gcn_fixed_dynamic_adjacency_comparison(input_dir, output_dir)
        plot_gcn_functional_similarity_matrix(input_dir, output_dir)
        plot_gcn_congestion_delay_matrix(input_dir, output_dir)
        plot_gcn_congestion_strength_matrix(input_dir, output_dir)
        plot_gcn_congestion_delay_distribution(input_dir, output_dir)
        plot_gcn_congestion_delay_interaction(input_dir, output_dir)
        plot_gcn_peak_graph_change(input_dir, output_dir)
    if workflow in ("all", "fixed_vs_dynamic"):
        plot_gcn_fixed_vs_dynamic_training_comparison(input_dir, output_dir)
    if workflow in ("all", "congestion_delay"):
        plot_gcn_congestion_delay_training_comparison(input_dir, output_dir)
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


def main():
    parser = argparse.ArgumentParser(description="GCN Enhanced Visualization")
    parser.add_argument(
        "--workflow",
        choices=[
            "all", "data_viz", "fixed_vs_dynamic", "congestion_delay", "main",
            "aggregation", "lambda", "convergence", "client_scale", "noniid",
            "client_metrics", "peak",
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
    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_PAPER_READY_DIR if args.paper_ready else input_dir)
    if args.paper_ready:
        run_paper_ready(input_dir, output_dir)
        return
    run_viz_project(args.workflow, input_dir, output_dir)


if __name__ == "__main__":
    main()
