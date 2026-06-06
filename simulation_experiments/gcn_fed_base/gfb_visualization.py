# -*- coding: utf-8 -*-
"""
GCN 基础实验可视化模块。
只读取 `gfb_core.py` 导出的 CSV 文件并生成图像。
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
}
CLIENT_PALETTE = sns.color_palette("tab10")
SPLIT_PALETTE = ["#55A868", "#DD8452", "#C44E52"]


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


def plot_base_dataset_client_timeseries(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "base_dataset_client_timeseries.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for client_id in sorted(df["client_id"].unique()):
        subset = df[df["client_id"] == client_id]
        ax.plot(
            subset["time_step"],
            subset["traffic_flow"],
            label=f"Client {client_id}",
            color=CLIENT_PALETTE[client_id % len(CLIENT_PALETTE)],
        )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Traffic Flow")
    ax.set_title("Per-client Average Traffic Flow")
    ax.legend(ncol=2, fontsize=8)
    out_path = ensure_dir(output_dir) / "base_dataset_client_timeseries.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_base_dataset_node_heatmap(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "base_dataset_node_heatmap.csv")
    rep_client_id = int(df["client_id"].min())
    pivot_df = (
        df[df["client_id"] == rep_client_id]
        .pivot(index="node_id", columns="time_step", values="traffic_flow")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(pivot_df, cmap="viridis", ax=ax, cbar_kws={"label": "Traffic Flow"})
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Node ID")
    ax.set_title(f"Node-Time Traffic Grid (Client {rep_client_id})")
    out_path = ensure_dir(output_dir) / "base_dataset_node_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_base_dataset_client_boxplot(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "base_dataset_client_distribution.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(data=df, x="client_id", y="traffic_flow", hue="client_id", palette="tab10", showfliers=False, ax=ax, legend=False)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Traffic Flow")
    ax.set_title("Client Traffic Distribution")
    out_path = ensure_dir(output_dir) / "base_dataset_client_boxplot.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_base_dataset_split_overview(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "base_dataset_split_overview.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    left = 0
    for idx, row in df.iterrows():
        ax.barh(
            ["Dataset Split"],
            [row["num_samples"]],
            left=[left],
            color=SPLIT_PALETTE[idx % len(SPLIT_PALETTE)],
            label=f"{row['split']} ({row['ratio'] * 100:.0f}%)",
        )
        left += row["num_samples"]
    ax.set_xlabel("Number of Samples")
    ax.set_title("Train / Validation / Test Split")
    ax.legend(loc="upper right")
    out_path = ensure_dir(output_dir) / "base_dataset_split_overview.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_base_dataset_client_sample_size(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "base_dataset_client_sample_size.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=df, x="client_id", y="num_samples", hue="client_id", palette="tab10", ax=ax, legend=False)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Client Sample Sizes")
    for idx, row in df.iterrows():
        ax.text(idx, row["num_samples"] + 2, int(row["num_samples"]), ha="center", fontsize=9)
    out_path = ensure_dir(output_dir) / "base_dataset_client_sample_size.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_base_graph_adjacency_matrix(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "base_graph_adjacency_matrix.csv")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(df, cmap="mako", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Base Graph Adjacency Matrix")
    out_path = ensure_dir(output_dir) / "base_graph_adjacency_matrix.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_main_metrics(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "main_metrics.csv")
    pred_df = read_required_csv(input_dir / "main_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, metric_name in enumerate(["rmse", "mae", "mape"]):
        sns.barplot(data=df, x="method", y=metric_name, hue="method", ax=axes[idx], palette=METHOD_PALETTE, legend=False)
        axes[idx].set_title(metric_name.upper())
        axes[idx].tick_params(axis="x", rotation=12)
    out_path = ensure_dir(output_dir) / "main_metrics_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    rep_client_id = int(pred_df["client_id"].min())
    rep_df = pred_df[pred_df["client_id"] == rep_client_id]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    gt_method = rep_df["method"].iloc[0]
    gt_df = rep_df[rep_df["method"] == gt_method]
    ax2.plot(gt_df["sample_id"], gt_df["y_true"], label="Ground Truth", color="black", linestyle="--", linewidth=2)
    for method in rep_df["method"].unique():
        method_df = rep_df[rep_df["method"] == method]
        ax2.plot(method_df["sample_id"], method_df["y_pred"], label=method)
    ax2.set_xlabel("Sample ID")
    ax2.set_ylabel("Traffic Flow")
    ax2.set_title(f"Prediction Comparison (Client {rep_client_id})")
    ax2.legend()
    out_path2 = output_dir / "main_predictions_comparison.png"
    fig2.savefig(out_path2, bbox_inches="tight")
    plt.close(fig2)


def plot_convergence(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "convergence_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=df, x="round", y="avg_train_loss", hue="method", marker="o", ax=axes[0], palette=METHOD_PALETTE)
    axes[0].set_title("Average Training Loss")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    sns.lineplot(data=df, x="round", y="avg_val_rmse", hue="method", marker="s", ax=axes[1], palette=METHOD_PALETTE)
    axes[1].set_title("Average Validation RMSE")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("RMSE")
    out_path = ensure_dir(output_dir) / "convergence_curve.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def run_viz_project(workflow: str, input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    if workflow in ("all", "data_viz"):
        plot_base_dataset_client_timeseries(input_dir, output_dir)
        plot_base_dataset_node_heatmap(input_dir, output_dir)
        plot_base_dataset_client_boxplot(input_dir, output_dir)
        plot_base_dataset_split_overview(input_dir, output_dir)
        plot_base_dataset_client_sample_size(input_dir, output_dir)
        plot_base_graph_adjacency_matrix(input_dir, output_dir)
    if workflow in ("all", "main"):
        plot_main_metrics(input_dir, output_dir)
    if workflow in ("all", "convergence"):
        plot_convergence(input_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="GCN Base Visualization")
    parser.add_argument("--workflow", choices=["all", "data_viz", "main", "convergence"], default="all")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    run_viz_project(args.workflow, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
