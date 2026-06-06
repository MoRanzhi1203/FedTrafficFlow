# -*- coding: utf-8 -*-
"""
CNN 增强实验可视化模块。
只读取 `cfe_core.py` 导出的 CSV 文件并生成图像。
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
    "Independent": "#4C72B0",
    "FedAvg": "#DD8452",
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

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


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
    _save(fig, output_dir, "enhanced_dataset_client_timeseries.png")


def plot_enhanced_dataset_distribution(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.violinplot(data=df, x="client_id", y="traffic_flow", hue="client_id", palette="tab10", ax=ax, cut=0, legend=False)
    ax.set_title("Enhanced Dataset Distribution")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Traffic Flow")
    _save(fig, output_dir, "enhanced_dataset_distribution.png")


def plot_enhanced_dataset_client_config(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_config.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=df, x="client_id", y="sample_size", hue="client_id", ax=axes[0], palette="tab10", legend=False)
    axes[0].set_title("Sample Size by Client")
    sns.scatterplot(data=df, x="noise_level", y="base_flow", hue="client_id", palette="tab10", s=90, ax=axes[1])
    axes[1].set_title("Noise vs Base Flow")
    _save(fig, output_dir, "enhanced_dataset_client_config.png")


def plot_enhanced_dataset_peak_pattern(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_peak_pattern.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.lineplot(data=df, x="hour", y="traffic_flow", hue="client_id", palette="tab10", ax=ax)
    ax.set_title("Peak Pattern by Client")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Traffic Flow")
    _save(fig, output_dir, "enhanced_dataset_peak_pattern.png")


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
    _save(fig, output_dir, "enhanced_dataset_incident_example.png")


def plot_enhanced_dataset_client_correlation_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "enhanced_dataset_client_correlation_matrix.csv")
    matrix = df.pivot(index="source_client", columns="target_client", values="correlation").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Client Correlation Matrix")
    _save(fig, output_dir, "enhanced_dataset_client_correlation_matrix.png")


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
    _save(fig, output_dir, "enhanced_dataset_node_correlation_matrix.png")


def plot_main_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_main_metrics.csv")
    pred_df = read_required_csv(input_dir / "cnn_enhanced_main_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, metric_name in enumerate(["rmse", "mae", "mape"]):
        sns.barplot(data=df, x="method", y=metric_name, hue="method", ax=axes[idx], palette=METHOD_PALETTE, legend=False)
        axes[idx].tick_params(axis="x", rotation=12)
        axes[idx].set_title(metric_name.upper())
    _save(fig, output_dir, "cnn_enhanced_main_comparison.png")

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
    _save(fig2, output_dir, "cnn_enhanced_main_predictions.png")


def plot_aggregation_results(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_aggregation_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="method", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax, legend=False)
    ax.set_title("Aggregation Strategy Comparison")
    ax.tick_params(axis="x", rotation=12)
    _save(fig, output_dir, "cnn_enhanced_aggregation.png")


def plot_lambda_sensitivity(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_lambda_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="lambda_value", y="rmse", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_title("Lambda Sensitivity")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("RMSE")
    _save(fig, output_dir, "cnn_enhanced_lambda.png")


def plot_convergence(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_convergence_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=df, x="round", y="avg_train_loss", hue="method", marker="o", ax=axes[0], palette=METHOD_PALETTE)
    axes[0].set_title("Average Training Loss")
    sns.lineplot(data=df, x="round", y="avg_val_rmse", hue="method", marker="s", ax=axes[1], palette=METHOD_PALETTE)
    axes[1].set_title("Average Validation RMSE")
    _save(fig, output_dir, "cnn_enhanced_convergence.png")


def plot_client_scale(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_client_scale_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=df, x="num_clients", y="rmse", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_title("Client Scale Sensitivity")
    _save(fig, output_dir, "cnn_enhanced_client_scale.png")


def plot_noniid(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_noniid_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="noniid_level", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Non-IID Sensitivity")
    _save(fig, output_dir, "cnn_enhanced_noniid.png")


def plot_client_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_client_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="client_id", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Per-client RMSE")
    _save(fig, output_dir, "cnn_enhanced_client_metrics.png")


def plot_peak_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_peak_metrics.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=df, x="period", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Peak-period Metrics")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, output_dir, "cnn_enhanced_peak_metrics.png")


def plot_feature_ablation(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "cnn_enhanced_feature_ablation_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=df, x="feature_set", y="rmse", hue="method", palette=METHOD_PALETTE, ax=ax)
    ax.set_title("Feature Ablation")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir, "cnn_enhanced_feature_ablation.png")


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
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_viz_project(args.workflow, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
