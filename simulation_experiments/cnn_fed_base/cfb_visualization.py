# -*- coding: utf-8 -*-
"""
CNN Federated Base Visualization Module.
Only responsible for reading exported CSV files and generating plots.
"""
import argparse
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.ioff()

# ──────────────────────────────────────────────────────────
# Constants & Style
# ──────────────────────────────────────────────────────────
METHOD_PALETTE = {
    "Independent": "#4C72B0",
    "FedAvg": "#DD8452",
    "Proposed": "#55A868",
}
CLIENT_PALETTE = sns.color_palette("tab10")
SPLIT_PALETTE = ["#55A868", "#DD8452", "#C44E52"]

def configure_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
    })

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. Please run the corresponding *_core.py workflow first."
        )
    return pd.read_csv(path)

# ──────────────────────────────────────────────────────────
# Plotting Functions
# ──────────────────────────────────────────────────────────

def plot_base_dataset_client_timeseries(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "base_dataset_client_timeseries.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for cid in df["client_id"].unique():
        sub = df[df["client_id"] == cid]
        ax.plot(sub["time_step"], sub["traffic_flow"], 
                label=f"Client {cid}", linewidth=2.0, alpha=0.9,
                color=CLIENT_PALETTE[cid % len(CLIENT_PALETTE)])
                
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Traffic Flow")
    ax.set_title("Per-client average traffic flow")
    ax.legend(loc="best", ncol=2, fontsize=8)
    
    out_path = ensure_dir(output_dir) / "base_dataset_client_timeseries.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
    return out_path

def plot_base_dataset_node_heatmap(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "base_dataset_node_heatmap.csv")
    # 默认展示 Client 0
    rep_cid = 0
    df_rep = df[df["client_id"] == rep_cid]
    pivot_df = df_rep.pivot(index="node_id", columns="time_step", values="traffic_flow")
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot_df, ax=ax, cmap="viridis", cbar_kws={"label": "Traffic flow"})
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Node ID")
    ax.set_title(f"Node-time traffic heatmap for client {rep_cid}")
    
    out_path = ensure_dir(output_dir) / "base_dataset_node_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
    return out_path

def plot_base_dataset_client_boxplot(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "base_dataset_client_distribution.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.boxplot(
        data=df,
        x="client_id",
        y="traffic_flow",
        hue="client_id",
        ax=ax,
        palette="tab10",
        showfliers=False,
        linewidth=1.0,
        legend=False,
    )
    
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Traffic flow")
    ax.set_title("Non-IID traffic distribution by client")
    
    out_path = ensure_dir(output_dir) / "base_dataset_client_boxplot.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
    return out_path

def plot_base_dataset_split_overview(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "base_dataset_split_overview.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    left = 0
    for idx, row in df.iterrows():
        label = f"{row['split']} ({row['ratio']*100:.0f}%)"
        ax.barh(["Dataset split"], [row["num_samples"]], left=[left], 
                color=SPLIT_PALETTE[idx % len(SPLIT_PALETTE)], label=label)
        left += row["num_samples"]
        
    ax.set_xlabel("Number of Samples")
    ax.set_title("Train, validation, and test split")
    ax.legend(loc="upper right")
    
    out_path = ensure_dir(output_dir) / "base_dataset_split_overview.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
    return out_path

def plot_base_dataset_client_sample_size(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "base_dataset_client_sample_size.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    sns.barplot(data=df, x="client_id", y="num_samples", hue="client_id", ax=ax, palette="tab10", legend=False)
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample size by client")
    
    for i, v in enumerate(df["num_samples"]):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10)
        
    out_path = ensure_dir(output_dir) / "base_dataset_client_sample_size.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
    return out_path

def plot_main_metrics(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "main_metrics.csv")
    metrics_to_plot = ["rmse", "mae", "mape"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 4.5))
    
    for i, m in enumerate(metrics_to_plot):
        sns.barplot(data=df, x="method", y=m, hue="method", ax=axes[i], palette=METHOD_PALETTE, legend=False)
        axes[i].set_title(f"Method Comparison: {m.upper()}")
        axes[i].tick_params(axis='x', rotation=15)
        
    out_path = ensure_dir(output_dir) / "main_metrics_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    # 额外画预测对比图
    df_pred = load_csv(input_dir, "main_predictions.csv")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    rep_client_id = int(df_pred["client_id"].min())
    df_pred = df_pred[df_pred["client_id"] == rep_client_id]
    first_method = df_pred["method"].unique()[0]
    gt_data = df_pred[df_pred["method"] == first_method]
    ax2.plot(gt_data["sample_id"], gt_data["y_true"], label="Ground Truth", 
             color="black", linewidth=2, linestyle="--")
    
    for method in df_pred["method"].unique():
        method_data = df_pred[df_pred["method"] == method]
        ax2.plot(method_data["sample_id"], method_data["y_pred"], label=f"{method} Pred", alpha=0.8)
        
    ax2.set_title(f"Traffic Flow Prediction Comparison (Client {rep_client_id})")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Traffic Flow")
    ax2.legend()
    
    out_path2 = output_dir / "main_predictions_comparison.png"
    fig2.savefig(out_path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"[viz] Saved: {out_path}, {out_path2}")
    return out_path

def plot_convergence(input_dir: Path, output_dir: Path) -> Path:
    df = load_csv(input_dir, "convergence_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sns.lineplot(data=df, x="round", y="avg_train_loss", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Round")
    ax.set_ylabel("Train Loss")
    ax.set_title("Average Training Loss")

    ax = axes[1]
    sns.lineplot(data=df, x="round", y="avg_val_rmse", hue="method", marker="s", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Average Validation RMSE")
    
    fig.tight_layout()
    out_path = ensure_dir(output_dir) / "convergence_curve.png"
    fig.savefig(out_path, bbox_inches="tight")

    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {out_path}")
    return out_path

# ──────────────────────────────────────────────────────────
# CLI & Main
# ──────────────────────────────────────────────────────────

def run_viz_project(workflow: str, input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    print(f"[cnn_fed_base_viz] workflow={workflow}, input={input_dir}, output={output_dir}")

    if workflow in ("all", "data_viz"):
        plot_base_dataset_client_timeseries(input_dir, output_dir)
        plot_base_dataset_node_heatmap(input_dir, output_dir)
        plot_base_dataset_client_boxplot(input_dir, output_dir)
        plot_base_dataset_split_overview(input_dir, output_dir)
        plot_base_dataset_client_sample_size(input_dir, output_dir)

    if workflow in ("all", "main"):
        plot_main_metrics(input_dir, output_dir)

    if workflow in ("all", "convergence"):
        plot_convergence(input_dir, output_dir)

def main():
    parser = argparse.ArgumentParser(description="CNN Base Visualization")
    parser.add_argument("--workflow", choices=["all", "data_viz", "main", "convergence"], default="all")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    run_viz_project(args.workflow, Path(args.input_dir), Path(args.output_dir))

if __name__ == "__main__":
    main()
