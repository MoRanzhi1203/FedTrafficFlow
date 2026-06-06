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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "simulation_experiments" / "fed_robustness"
DEFAULT_PAPER_READY_DIR = DEFAULT_INPUT_DIR / "paper_ready"
FEDAVG_COLOR = "#0072B2"
BAR_COLOR = "#0072B2"
LINE_COLOR = "#0072B2"
ACCENT_COLOR = "#D55E00"
GRID_ALPHA = 0.35
METHOD_PALETTE = {
    "FedAvg": FEDAVG_COLOR,
    "Proposed": "#55A868",
}


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


def _save_fig(fig, output_dir: Path, filename: str):
    out_path = ensure_dir(output_dir) / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return out_path, pdf_path


def _style_axis(ax):
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _annotate_line_points(ax, x_values, y_values, fmt: str = "{:.2f}", y_offset: float = 0.035):
    if not y_values:
        return
    y_span = max(y_values) - min(y_values)
    offset = y_span * y_offset if y_span > 0 else 0.03
    for x_value, y_value in zip(x_values, y_values):
        ax.text(x_value, y_value + offset, fmt.format(y_value), ha="center", va="bottom", fontsize=9)


def plot_fed_robustness_communication_cost(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_communication_cost.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=df, x="model_type", y="total_communication_mb", hue="rounds", ax=ax)
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Total Communication (MB)")
    ax.set_title("Communication Cost")
    ax.tick_params(axis="x", rotation=15)
    _save_fig(fig, output_dir, "fed_robustness_communication_cost.png")


def plot_fed_robustness_client_dropout(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_client_dropout_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="dropout_rate", y="rmse_mean", hue="method", marker="o", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Client Dropout Robustness")
    _save_fig(fig, output_dir, "fed_robustness_client_dropout.png")


def plot_fed_robustness_communication_delay(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_communication_delay_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="delay_rounds", y="rmse_mean", hue="method", marker="s", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Delay Rounds")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Communication Delay Robustness")
    _save_fig(fig, output_dir, "fed_robustness_communication_delay.png")


def plot_fed_robustness_gradient_noise(input_dir: Path, output_dir: Path):
    df = read_required_csv(input_dir / "fed_gradient_noise_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="noise_std", y="rmse_mean", hue="method", marker="^", ax=ax, palette=METHOD_PALETTE)
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Gradient Noise Robustness")
    _save_fig(fig, output_dir, "fed_robustness_gradient_noise.png")


def _fedavg_summary(input_dir: Path, filename: str) -> pd.DataFrame:
    df = read_required_csv(input_dir / filename)
    if "method" in df.columns:
        df = df[df["method"] == "FedAvg"].copy()
    if df.empty:
        raise ValueError(f"No FedAvg rows found in {filename}.")
    return df


def plot_paper_ready_client_dropout(input_dir: Path, output_dir: Path):
    df = _fedavg_summary(input_dir, "fed_client_dropout_summary.csv").sort_values("dropout_rate")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.lineplot(
        data=df,
        x="dropout_rate",
        y="rmse_mean",
        marker="o",
        markersize=5,
        linewidth=2.0,
        color=LINE_COLOR,
        ax=ax,
    )
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("RMSE")
    ax.set_title("FedAvg Robustness under Client Dropout")
    _style_axis(ax)
    _annotate_line_points(ax, df["dropout_rate"].tolist(), df["rmse_mean"].tolist())
    _save_fig(fig, output_dir, "fed_robustness_client_dropout_fedavg_only.png")


def plot_paper_ready_communication_delay(input_dir: Path, output_dir: Path):
    df = _fedavg_summary(input_dir, "fed_communication_delay_summary.csv").sort_values("delay_rounds")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.lineplot(
        data=df,
        x="delay_rounds",
        y="rmse_mean",
        marker="o",
        markersize=5,
        linewidth=2.0,
        color=LINE_COLOR,
        ax=ax,
    )
    ax.set_xlabel("Delay Rounds")
    ax.set_ylabel("RMSE")
    ax.set_title("FedAvg Robustness under Communication Delay")
    _style_axis(ax)
    _annotate_line_points(ax, df["delay_rounds"].tolist(), df["rmse_mean"].tolist())
    _save_fig(fig, output_dir, "fed_robustness_communication_delay_fedavg_only.png")


def plot_paper_ready_gradient_noise(input_dir: Path, output_dir: Path):
    df = _fedavg_summary(input_dir, "fed_gradient_noise_summary.csv").sort_values("noise_std")
    print("梯度噪声仅为模拟梯度扰动，不构成正式差分隐私机制。")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(
        data=df,
        x="noise_std",
        y="rmse_mean",
        marker="o",
        markersize=5,
        linewidth=2.0,
        color=LINE_COLOR,
        ax=ax,
    )
    ax.set_xlabel("Noise Std")
    ax.set_ylabel("RMSE")
    ax.set_title("FedAvg Robustness under Simulated Gradient Perturbation")
    _style_axis(ax)
    _annotate_line_points(ax, df["noise_std"].tolist(), df["rmse_mean"].tolist(), y_offset=0.02)
    ax.text(
        0.02,
        0.96,
        "Simulated gradient perturbation;\nnot formal differential privacy.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    _save_fig(fig, output_dir, "fed_robustness_gradient_noise_fedavg_only.png")


def run_paper_ready(input_dir: Path, output_dir: Path):
    configure_plot_style()
    ensure_dir(output_dir)
    plot_paper_ready_client_dropout(input_dir, output_dir)
    plot_paper_ready_communication_delay(input_dir, output_dir)
    plot_paper_ready_gradient_noise(input_dir, output_dir)


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
