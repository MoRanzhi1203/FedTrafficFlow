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
import pandas as pd
import seaborn as sns

plt.ioff()

METHOD_PALETTE = {
    "FedAvg": "#DD8452",
    "Proposed": "#55A868",
    "Independent": "#4C72B0",
    "Loss-weighted": "#C44E52",
    "Data-loss weighted": "#8172B3",
}
CLIENT_PALETTE = sns.color_palette("tab10")


def configure_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
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


def _save(fig, output_dir: Path, filename: str):
    out_path = ensure_dir(output_dir) / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_matrix(df: pd.DataFrame, title: str, output_dir: Path, filename: str, cmap="viridis", annot=False):
    fig, ax = plt.subplots(figsize=(6.5, 5.3))
    sns.heatmap(df, cmap=cmap, center=0 if cmap == "coolwarm" else None, annot=annot, fmt=".2f", ax=ax)
    ax.set_title(title)
    _save(fig, output_dir, filename)


def plot_enhanced_dataset_client_timeseries(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_timeseries.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=df, x="time_step", y="traffic_flow", hue="client_id", palette="tab10", ax=ax)
    ax.set_title("Enhanced Dataset Client Time Series")
    _save(fig, output_dir, "enhanced_dataset_client_timeseries.png")


def plot_enhanced_dataset_distribution(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(data=df, x="client_id", y="traffic_flow", hue="client_id", palette="tab10", showfliers=False, ax=ax, legend=False)
    ax.set_title("Enhanced Dataset Distribution")
    _save(fig, output_dir, "enhanced_dataset_distribution.png")


def plot_enhanced_dataset_client_config(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_config.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=df, x="client_id", y="sample_size", hue="client_id", palette="tab10", ax=axes[0], legend=False)
    axes[0].set_title("Client Sample Size")
    sns.scatterplot(data=df, x="noise_level", y="base_flow", hue="client_id", palette="tab10", s=90, ax=axes[1])
    axes[1].set_title("Noise Level vs Base Flow")
    _save(fig, output_dir, "enhanced_dataset_client_config.png")


def plot_enhanced_dataset_peak_pattern(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_peak_pattern.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=df, x="hour", y="traffic_flow", hue="client_id", palette="tab10", ax=ax)
    ax.set_title("Peak Pattern by Client")
    _save(fig, output_dir, "enhanced_dataset_peak_pattern.png")


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
    _save(fig, output_dir, "enhanced_dataset_incident_example.png")


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
    _save(fig, output_dir, "enhanced_gcn_dynamic_peak.png")


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
    _save(fig, output_dir, "enhanced_gcn_fixed_dynamic_comparison.png")


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
    _save(fig, output_dir, "enhanced_gcn_congestion_delay_distribution.png")


def plot_gcn_congestion_delay_interaction(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_gcn_congestion_delay_interaction.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.scatterplot(data=df, x="delay_rounds", y="strength", hue="source_node", palette="tab10", ax=ax)
    ax.set_title("Congestion Delay Interaction")
    _save(fig, output_dir, "enhanced_gcn_congestion_delay_interaction.png")


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
    _save(fig, output_dir, "gcn_enhanced_fixed_vs_dynamic.png")


def plot_gcn_congestion_delay_training_comparison(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_congestion_delay_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="graph_type", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Congestion Delay Training Comparison")
    _save(fig, output_dir, "gcn_enhanced_congestion_delay_comp.png")


def plot_main_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_main_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, metric_name in enumerate(["rmse", "mae", "mape"]):
        sns.barplot(data=df, x="method", y=metric_name, hue="method", ax=axes[idx], palette=METHOD_PALETTE, legend=False)
        axes[idx].tick_params(axis="x", rotation=12)
        axes[idx].set_title(metric_name.upper())
    _save(fig, output_dir, "gcn_enhanced_main_results.png")


def plot_aggregation_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_aggregation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=df, x="method", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax, legend=False)
    ax.tick_params(axis="x", rotation=12)
    ax.set_title("Aggregation Strategy Comparison")
    _save(fig, output_dir, "gcn_enhanced_aggregation.png")


def plot_lambda_sensitivity(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_lambda_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="lambda_value", y="rmse", hue="method", marker="o", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Lambda Sensitivity")
    _save(fig, output_dir, "gcn_enhanced_lambda.png")


def plot_convergence(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_convergence_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=df, x="round", y="avg_train_loss", hue="method", palette=METHOD_PALETTE, marker="o", ax=axes[0])
    axes[0].set_title("Average Training Loss")
    sns.lineplot(data=df, x="round", y="avg_val_rmse", hue="method", palette=METHOD_PALETTE, marker="s", ax=axes[1])
    axes[1].set_title("Average Validation RMSE")
    _save(fig, output_dir, "gcn_enhanced_convergence.png")


def plot_client_scale(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_client_scale_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="num_clients", y="rmse", hue="method", palette=METHOD_PALETTE, marker="o", ax=ax)
    ax.set_title("Client Scale Sensitivity")
    _save(fig, output_dir, "gcn_enhanced_client_scale.png")


def plot_noniid(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_noniid_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="noniid_level", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Non-IID Sensitivity")
    _save(fig, output_dir, "gcn_enhanced_noniid.png")


def plot_client_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_client_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="client_id", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Per-client RMSE")
    _save(fig, output_dir, "gcn_enhanced_client_metrics.png")


def plot_peak_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "gcn_enhanced_peak_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=df, x="period", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("Peak-period Metrics")
    _save(fig, output_dir, "gcn_enhanced_peak_metrics.png")


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
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_viz_project(args.workflow, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
