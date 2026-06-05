# -*- coding: utf-8 -*-
"""
联邦鲁棒性可视化模块。
只读取 `fr_core.py` 导出的 CSV 文件并生成图像。
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
}


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


def plot_fed_robustness_communication_cost(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_communication_cost.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=df, x="model_type", y="total_communication_mb", hue="rounds", ax=ax)
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Total Communication (MB)")
    ax.set_title("Communication Cost")
    ax.tick_params(axis="x", rotation=15)
    out_path = ensure_dir(output_dir) / "fed_robustness_communication_cost.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fed_robustness_client_dropout(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_client_dropout_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="dropout_rate", y="rmse_mean", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Client Dropout Robustness")
    out_path = ensure_dir(output_dir) / "fed_robustness_client_dropout.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fed_robustness_communication_delay(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_communication_delay_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="delay_rounds", y="rmse_mean", hue="method", marker="s", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Delay Rounds")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Communication Delay Robustness")
    out_path = ensure_dir(output_dir) / "fed_robustness_communication_delay.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fed_robustness_gradient_noise(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_gradient_noise_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="noise_std", y="rmse_mean", hue="method", marker="^", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Gradient Noise Robustness")
    out_path = ensure_dir(output_dir) / "fed_robustness_gradient_noise.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_viz_project(workflow: str, input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    funcs = {
        "communication_cost": plot_fed_robustness_communication_cost,
        "client_dropout": plot_fed_robustness_client_dropout,
        "communication_delay": plot_fed_robustness_communication_delay,
        "gradient_noise": plot_fed_robustness_gradient_noise,
    }
    steps = [workflow] if workflow != "all" else list(funcs.keys())
    for step in steps:
        funcs[step](input_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Federated Robustness Visualization")
    parser.add_argument(
        "--workflow",
        choices=["all", "communication_cost", "client_dropout", "communication_delay", "gradient_noise"],
        default="all",
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_viz_project(args.workflow, Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
