"""Visualization entrypoint for the single-intersection client experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from real_data_experiments.common.io_utils import ensure_dir, resolve_path
from real_data_experiments.common.result_writer import write_csv, write_text


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for visualization."""
    parser = argparse.ArgumentParser(description="Single-intersection visualization")
    parser.add_argument("--workflow", choices=["all"], default="all")
    parser.add_argument("--input-dir", type=str, default="results/real_data_experiments/single_intersection_client_tensor")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _load_csv(input_dir: Path, file_name: str) -> pd.DataFrame:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)


def plot_main_metrics(main_metrics_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot main metric comparison across methods."""
    figure_records: list[dict[str, str]] = []
    metrics = ["rmse", "mae", "mape"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(main_metrics_df["method"], main_metrics_df[metric], color=["#4C72B0", "#DD8452"])
        ax.set_title(metric.upper())
        ax.set_xlabel("Method")
        ax.set_ylabel(metric.upper())
    fig.suptitle("Single-Intersection Main Metrics (Test Set)")
    fig.tight_layout()
    file_name = f"main_metrics_comparison.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "Single-Intersection Main Metrics (Test Set)", "title_zh": "单路口主指标对比（测试集）"})
    return figure_records


def plot_client_metrics(client_metrics_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot client-level RMSE comparison."""
    figure_records: list[dict[str, str]] = []
    pivot_df = client_metrics_df.pivot(index="client_id", columns="method", values="rmse")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot_df.plot(kind="bar", ax=ax)
    ax.set_title("Client RMSE Comparison (Test Set)")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("RMSE")
    ax.legend(title="Method")
    fig.tight_layout()
    file_name = f"client_metrics_comparison.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "Client RMSE Comparison (Test Set)", "title_zh": "客户端 RMSE 对比（测试集）"})
    return figure_records


def plot_convergence(convergence_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot FedAvg convergence history."""
    figure_records: list[dict[str, str]] = []
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(convergence_df["communication_round"], convergence_df["train_loss"], marker="o", label="Train Loss")
    ax1.plot(convergence_df["communication_round"], convergence_df["val_rmse"], marker="s", label="Val RMSE")
    ax1.plot(convergence_df["communication_round"], convergence_df["test_rmse"], marker="^", label="Test RMSE")
    ax1.set_title("FedAvg Convergence Curve")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Metric Value")
    ax1.legend()
    fig.tight_layout()
    file_name = f"convergence_curve.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "FedAvg Convergence Curve", "title_zh": "FedAvg 收敛曲线"})
    return figure_records


def plot_predictions(prediction_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot prediction vs ground truth for a small sample subset."""
    figure_records: list[dict[str, str]] = []
    sample_df = prediction_df.groupby(["method", "client_id"], group_keys=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 4))
    for (method, client_id), group_df in sample_df.groupby(["method", "client_id"]):
        label_pred = f"{method}-C{client_id}-Pred"
        label_true = f"{method}-C{client_id}-True"
        ax.plot(group_df["sample_index"], group_df["y_pred"], linestyle="--", label=label_pred)
        ax.plot(group_df["sample_index"], group_df["y_true"], linestyle="-", alpha=0.7, label=label_true)
    ax.set_title("Prediction vs Ground Truth")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Traffic Flow")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    file_name = f"prediction_vs_ground_truth.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "Prediction vs Ground Truth", "title_zh": "预测值与真实值对比"})
    return figure_records


def generate_figures(input_dir: Path, output_dir: Path, image_format: str, dpi: int) -> None:
    """Generate all single-intersection figures from exported CSV files."""
    main_metrics_df = _load_csv(input_dir, "main_metrics.csv")
    client_metrics_df = _load_csv(input_dir, "client_metrics.csv")
    convergence_df = _load_csv(input_dir, "convergence_history.csv")
    prediction_df = _load_csv(input_dir, "prediction_samples.csv")

    figure_records: list[dict[str, str]] = []
    figure_records.extend(plot_main_metrics(main_metrics_df, output_dir, image_format, dpi))
    figure_records.extend(plot_client_metrics(client_metrics_df, output_dir, image_format, dpi))
    figure_records.extend(plot_convergence(convergence_df, output_dir, image_format, dpi))
    figure_records.extend(plot_predictions(prediction_df, output_dir, image_format, dpi))

    figure_index_df = pd.DataFrame(figure_records)
    write_csv(figure_index_df, output_dir / "figure_index.csv")
    write_text(
        "\n".join(
            [
                "# 图表说明",
                "",
                "- `main_metrics_comparison`: 展示 pooled-grid-region client 上 FedAvg 与 Independent 在测试集上的总体 RMSE/MAE/MAPE 对比。",
                "- `client_metrics_comparison`: 展示各 pooled-grid-region client 测试集 RMSE 的方法对比。",
                "- `convergence_curve`: 展示通信轮次上的 train loss、val RMSE 与 test RMSE。",
                "- `prediction_vs_ground_truth`: 展示部分样本上的预测值与真实值对比曲线。",
            ]
        ),
        output_dir / "figure_notes_zh.md",
    )


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = ensure_dir(args.output_dir) if args.output_dir else input_dir
    generate_figures(input_dir=input_dir, output_dir=output_dir, image_format=args.format, dpi=args.dpi)
    print(f"[single_intersection_visualization] completed -> {output_dir}")


if __name__ == "__main__":
    main()
