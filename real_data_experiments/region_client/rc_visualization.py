"""Visualization entrypoint for the regional-client experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from real_data_experiments.common.io_utils import ensure_dir, resolve_path
from real_data_experiments.common.result_writer import write_csv, write_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regional-client visualization")
    parser.add_argument("--workflow", choices=["all"], default="all")
    parser.add_argument("--input-dir", type=str, default="results/real_data_experiments/region_client_tensor")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _load_csv(input_dir: Path, file_name: str) -> pd.DataFrame:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)


def plot_distribution(summary_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(summary_df["client_id"].astype(str), summary_df["region_count"], color="#4C72B0")
    axes[0].set_title("Active Regions per Client")
    axes[0].set_xlabel("Client ID")
    axes[0].set_ylabel("Region Count")
    axes[1].bar(summary_df["client_id"].astype(str), summary_df["sample_count_estimate"], color="#55A868")
    axes[1].set_title("Estimated Samples per Client")
    axes[1].set_xlabel("Client ID")
    axes[1].set_ylabel("Estimated Samples")
    fig.tight_layout()
    file_name = f"region_client_distribution.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"figure_name": file_name, "title_en": "Regional Client Distribution", "title_zh": "区域客户端分布"}


def plot_non_iid(non_iid_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_df = non_iid_df[non_iid_df["statistic"].isin(["std", "cv"])].copy()
    labels = plot_df["metric"] + "-" + plot_df["statistic"]
    ax.bar(labels, plot_df["value"], color="#C44E52")
    ax.set_title("Cross-Client Non-IID Summary")
    ax.set_xlabel("Statistic")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    file_name = f"region_non_iid_summary.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"figure_name": file_name, "title_en": "Regional Non-IID Summary", "title_zh": "区域客户端 Non-IID 摘要"}


def plot_main_metrics(main_metrics_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> dict[str, str]:
    metrics = ["rmse", "mae", "mape"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(main_metrics_df["method"], main_metrics_df[metric], color=["#4C72B0", "#DD8452"])
        ax.set_title(metric.upper())
        ax.set_xlabel("Method")
        ax.set_ylabel(metric.upper())
    fig.tight_layout()
    file_name = f"region_main_metrics_comparison.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"figure_name": file_name, "title_en": "Regional Main Metrics Comparison", "title_zh": "区域主指标对比"}


def plot_client_rmse(client_metrics_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> dict[str, str]:
    pivot_df = client_metrics_df.pivot(index="client_id", columns="method", values="rmse")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot_df.plot(kind="bar", ax=ax)
    ax.set_title("Client RMSE Comparison")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("RMSE")
    ax.legend(title="Method")
    fig.tight_layout()
    file_name = f"region_client_rmse.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"figure_name": file_name, "title_en": "Regional Client RMSE Comparison", "title_zh": "区域客户端 RMSE 对比"}


def plot_convergence(convergence_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(convergence_df["communication_round"], convergence_df["train_loss"], marker="o", label="Train Loss")
    ax.plot(convergence_df["communication_round"], convergence_df["val_rmse"], marker="s", label="Val RMSE")
    ax.plot(convergence_df["communication_round"], convergence_df["test_rmse"], marker="^", label="Test RMSE")
    ax.set_title("FedAvg Convergence Curve")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Metric Value")
    ax.legend()
    fig.tight_layout()
    file_name = f"region_convergence_curve.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"figure_name": file_name, "title_en": "Regional Convergence Curve", "title_zh": "区域实验收敛曲线"}


def plot_predictions(prediction_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> dict[str, str]:
    sample_df = prediction_df.groupby(["method", "client_id"], group_keys=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 4))
    for (method, client_id), group_df in sample_df.groupby(["method", "client_id"]):
        ax.plot(group_df["sample_index"], group_df["y_true"], linestyle="-", alpha=0.7, label=f"{method}-C{client_id}-True")
        ax.plot(group_df["sample_index"], group_df["y_pred"], linestyle="--", label=f"{method}-C{client_id}-Pred")
    ax.set_title("Prediction vs Truth")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Traffic Flow")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    file_name = f"region_prediction_vs_truth.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"figure_name": file_name, "title_en": "Regional Prediction vs Truth", "title_zh": "区域预测值与真实值对比"}


def generate_figures(input_dir: Path, output_dir: Path, image_format: str, dpi: int) -> None:
    summary_df = _load_csv(input_dir, "client_distribution_summary.csv")
    non_iid_df = _load_csv(input_dir, "non_iid_summary.csv")
    main_metrics_df = _load_csv(input_dir, "main_metrics.csv")
    client_metrics_df = _load_csv(input_dir, "client_metrics.csv")
    convergence_df = _load_csv(input_dir, "convergence_history.csv")
    prediction_df = _load_csv(input_dir, "prediction_samples.csv")

    figure_records = [
        plot_distribution(summary_df, output_dir, image_format, dpi),
        plot_non_iid(non_iid_df, output_dir, image_format, dpi),
        plot_main_metrics(main_metrics_df, output_dir, image_format, dpi),
        plot_client_rmse(client_metrics_df, output_dir, image_format, dpi),
        plot_convergence(convergence_df, output_dir, image_format, dpi),
        plot_predictions(prediction_df, output_dir, image_format, dpi),
    ]
    write_csv(pd.DataFrame(figure_records), output_dir / "figure_index.csv")
    write_text(
        "\n".join(
            [
                "# 图表说明",
                "",
                "- `region_client_distribution`: 展示每个区域客户端包含的 active pooled regions 数量与估计样本量。",
                "- `region_non_iid_summary`: 展示跨客户端的 region_count、sample_count、mean_total_flow 等统计离散程度。",
                "- `region_main_metrics_comparison`: 展示 FedAvg 与 Independent 的总体指标对比。",
                "- `region_client_rmse`: 展示各区域客户端的测试集 RMSE。",
                "- `region_convergence_curve`: 展示 FedAvg 通信轮次上的 train loss、val RMSE 与 test RMSE。",
                "- `region_prediction_vs_truth`: 展示部分样本的预测值与真实值对比。",
            ]
        ),
        output_dir / "figure_notes_zh.md",
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = ensure_dir(args.output_dir) if args.output_dir else input_dir
    generate_figures(input_dir=input_dir, output_dir=output_dir, image_format=args.format, dpi=args.dpi)
    print(f"[region_client_visualization] completed -> {output_dir}")


if __name__ == "__main__":
    main()
