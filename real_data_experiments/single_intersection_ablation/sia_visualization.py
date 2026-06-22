"""Visualization entrypoint for the single-intersection ablation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from real_data_experiments.common.io_utils import ensure_dir, resolve_path
from real_data_experiments.common.result_writer import write_csv, write_text


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for ablation visualization."""
    parser = argparse.ArgumentParser(description="Single-intersection ablation visualization")
    parser.add_argument("--workflow", choices=["all"], default="all")
    parser.add_argument("--input-dir", type=str, default="results/real_data_experiments/single_intersection_ablation")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _load_csv(input_dir: Path, file_name: str) -> pd.DataFrame:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)


def plot_ablation_metrics(ablation_metrics_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot test-set RMSE/MAE/MAPE across ablation variants."""
    figure_records: list[dict[str, str]] = []
    metrics = ["rmse", "mae", "mape"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        ax.bar(ablation_metrics_df["variant_label"], ablation_metrics_df[metric], color="#4C72B0")
        ax.set_title(metric.upper())
        ax.set_xlabel("Variant")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Single-Intersection Ablation Metrics (Test Set)")
    fig.tight_layout()
    file_name = f"ablation_metrics_comparison.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "Single-Intersection Ablation Metrics (Test Set)", "title_zh": "单路口消融指标对比（测试集）"})
    return figure_records


def plot_client_rmse(client_metrics_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot client-level RMSE by ablation variant."""
    figure_records: list[dict[str, str]] = []
    pivot_df = client_metrics_df.pivot(index="client_id", columns="variant_label", values="rmse")
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot_df.plot(kind="bar", ax=ax)
    ax.set_title("Client RMSE by Ablation Variant")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("RMSE")
    ax.legend(title="Variant", fontsize=8)
    fig.tight_layout()
    file_name = f"ablation_client_rmse.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "Client RMSE by Ablation Variant", "title_zh": "单路口消融客户端 RMSE 对比"})
    return figure_records


def plot_convergence(convergence_df: pd.DataFrame, output_dir: Path, image_format: str, dpi: int) -> list[dict[str, str]]:
    """Plot test RMSE convergence by ablation variant."""
    figure_records: list[dict[str, str]] = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for variant_label, group_df in convergence_df.groupby("variant_label"):
        ax.plot(group_df["communication_round"], group_df["test_rmse"], marker="o", label=variant_label)
    ax.set_title("Ablation Convergence Curve")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    file_name = f"ablation_convergence_curve.{image_format}"
    fig.savefig(output_dir / file_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figure_records.append({"figure_name": file_name, "title_en": "Ablation Convergence Curve", "title_zh": "单路口消融收敛曲线"})
    return figure_records


def generate_figures(input_dir: Path, output_dir: Path, image_format: str, dpi: int) -> None:
    """Generate all ablation figures from existing CSV files."""
    ablation_metrics_df = _load_csv(input_dir, "ablation_metrics.csv")
    client_metrics_df = _load_csv(input_dir, "ablation_client_metrics.csv")
    convergence_df = _load_csv(input_dir, "convergence_history.csv")

    figure_records: list[dict[str, str]] = []
    figure_records.extend(plot_ablation_metrics(ablation_metrics_df, output_dir, image_format, dpi))
    figure_records.extend(plot_client_rmse(client_metrics_df, output_dir, image_format, dpi))
    figure_records.extend(plot_convergence(convergence_df, output_dir, image_format, dpi))

    write_csv(pd.DataFrame(figure_records), output_dir / "figure_index.csv")
    write_text(
        "\n".join(
            [
                "# 图表说明",
                "",
                "- `ablation_metrics_comparison`: 比较四种结构变体在测试集上的总体误差指标。",
                "- `ablation_client_rmse`: 比较各客户端在不同消融变体下的测试集 RMSE。",
                "- `ablation_convergence_curve`: 比较四种结构变体在通信轮次上的测试集 RMSE 变化。",
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
    print(f"[single_intersection_ablation_visualization] completed -> {output_dir}")


if __name__ == "__main__":
    main()
