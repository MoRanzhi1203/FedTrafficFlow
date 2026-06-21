"""可视化当前网格化流量客户端聚类结果。"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from grid_flow_client_clustering import (
        DEFAULT_OUTPUT_DIR,
        SLOTS_PER_DAY,
        build_grid_region_features,
        load_data_tensor,
    )
    from ccn_region_client_train import DEFAULT_DATASET_PATH
except ImportError:  # pragma: no cover
    from analysis_scripts.federated_learning.grid_flow_client_clustering import (
        DEFAULT_OUTPUT_DIR,
        SLOTS_PER_DAY,
        build_grid_region_features,
        load_data_tensor,
    )
    from analysis_scripts.federated_learning.ccn_region_client_train import DEFAULT_DATASET_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOLED_INPUT = PROJECT_ROOT / "data" / "processed" / "node_flow_grid" / "node_flow_grid_pooled.npy"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"


@dataclass
class VisualizeConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    cluster_dir: Path = DEFAULT_OUTPUT_DIR
    pooled_input: Path = DEFAULT_POOLED_INPUT
    figure_dir: Path = DEFAULT_FIGURE_DIR
    t_in: int = 24
    t_out: int = 1
    seed: int = 42
    verbose: bool = False


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> VisualizeConfig:
    parser = argparse.ArgumentParser(description="可视化网格化流量客户端聚类结果。")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pooled-input", type=Path, default=DEFAULT_POOLED_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--t-in", type=int, default=24)
    parser.add_argument("--t-out", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = VisualizeConfig(
        dataset_path=args.dataset_path.resolve(),
        cluster_dir=args.cluster_dir.resolve(),
        pooled_input=args.pooled_input.resolve(),
        figure_dir=args.figure_dir.resolve(),
        t_in=int(args.t_in),
        t_out=int(args.t_out),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )
    configure_logging(config.verbose)
    configure_fonts()
    return config


def load_cluster_artifacts(cluster_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_path = cluster_dir / "cluster_labels.csv"
    summary_path = cluster_dir / "cluster_summary.csv"
    metric_path = cluster_dir / "k_search_metrics.csv"
    for path in (label_path, summary_path, metric_path):
        if not path.exists():
            raise FileNotFoundError(f"未找到聚类输出文件: {path}")
    return (
        pd.read_csv(label_path),
        pd.read_csv(summary_path),
        pd.read_csv(metric_path),
    )


def infer_grid_shape(pooled_input: Path, num_regions: int) -> tuple[int, int]:
    if pooled_input.exists():
        pooled_data = np.load(pooled_input, allow_pickle=True)
        if len(pooled_data) > 0 and "pooled_grid_tensor" in pooled_data[0]:
            _, height, width = np.asarray(pooled_data[0]["pooled_grid_tensor"]).shape
            if height * width == num_regions:
                return int(height), int(width)

    factor_pairs: list[tuple[int, int]] = []
    for height in range(1, int(np.sqrt(num_regions)) + 1):
        if num_regions % height == 0:
            width = num_regions // height
            factor_pairs.append((height, width))
    if not factor_pairs:
        raise ValueError(f"无法为 region 数 {num_regions} 推断网格形状。")
    return min(factor_pairs, key=lambda pair: abs(pair[0] - pair[1]))


def build_daily_region_profiles(data_tensor: torch.Tensor) -> np.ndarray:
    raw = data_tensor[0].detach().cpu().numpy().astype(np.float64)
    num_regions, total_steps = raw.shape
    if total_steps % SLOTS_PER_DAY != 0:
        raise ValueError(f"时间维 {total_steps} 不能被 {SLOTS_PER_DAY} 整除。")
    num_days = total_steps // SLOTS_PER_DAY
    daily = np.log1p(np.clip(raw, a_min=0.0, a_max=None)).reshape(num_regions, num_days, SLOTS_PER_DAY)
    return np.median(daily, axis=1)


def plot_k_search(metric_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(metric_df["k"], metric_df["silhouette_score"], marker="o", label="Silhouette", color="#4C78A8")
    ax1.plot(metric_df["k"], metric_df["selection_score"], marker="s", label="Selection Score", color="#F58518")
    ax1.set_xlabel("簇数 k")
    ax1.set_ylabel("分数")
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(metric_df["k"], metric_df["max_cluster_ratio"], marker="^", label="Max Cluster Ratio", color="#54A24B")
    ax2.set_ylabel("最大簇占比")
    ax2.set_ylim(0.0, 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    ax1.set_title("K 搜索诊断图")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_size(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x_positions = np.arange(len(summary_df), dtype=float)
    bars = ax.bar(x_positions, summary_df["region_count"], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(summary_df)])
    ax.set_xlabel("客户端 / 簇 ID")
    ax.set_ylabel("区域数")
    ax.set_title("各簇区域数量")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(value)) for value in summary_df["cluster_id"]])
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, count in zip(bars, summary_df["region_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{int(count)}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(features_z: np.ndarray, label_df: pd.DataFrame, output_path: Path, seed: int) -> None:
    pca = PCA(n_components=2, random_state=seed)
    embedding = pca.fit_transform(features_z)
    labels = label_df.sort_values("region_id")["cluster_id"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab10")
    unique_labels = sorted(np.unique(labels).tolist())
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=24,
            alpha=0.8,
            label=f"Cluster {cluster_id}",
            color=cmap(cluster_id % 10),
            edgecolors="none",
        )
    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PCA 1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PCA 2 ({explained[1] * 100:.1f}%)")
    ax.set_title("聚类 PCA 散点图")
    ax.legend()
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grid_cluster_map(label_df: pd.DataFrame, grid_shape: tuple[int, int], output_path: Path) -> None:
    labels = label_df.sort_values("region_id")["cluster_id"].to_numpy(dtype=int)
    cluster_grid = labels.reshape(grid_shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cluster_grid, cmap="tab10", interpolation="nearest", aspect="auto")
    ax.set_title(f"空间网格簇分布图 ({grid_shape[0]} x {grid_shape[1]})")
    ax.set_xlabel("宽度索引")
    ax.set_ylabel("高度索引")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    unique_labels = sorted(np.unique(labels).tolist())
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([f"Cluster {label}" for label in unique_labels])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_daily_profiles(daily_profiles: np.ndarray, label_df: pd.DataFrame, output_path: Path) -> None:
    labels = label_df.sort_values("region_id")["cluster_id"].to_numpy(dtype=int)
    unique_labels = sorted(np.unique(labels).tolist())
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(6 * len(unique_labels), 4.8), sharey=True)
    if len(unique_labels) == 1:
        axes = [axes]

    x = np.arange(SLOTS_PER_DAY, dtype=int)
    cmap = plt.get_cmap("tab10")
    for ax, cluster_id in zip(axes, unique_labels):
        cluster_profiles = daily_profiles[labels == cluster_id]
        q25 = np.quantile(cluster_profiles, 0.25, axis=0)
        median = np.median(cluster_profiles, axis=0)
        q75 = np.quantile(cluster_profiles, 0.75, axis=0)
        ax.fill_between(x, q25, q75, alpha=0.25, color=cmap(cluster_id % 10), label="IQR")
        ax.plot(x, median, color=cmap(cluster_id % 10), linewidth=2.0, label="Median")
        ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("15 分钟时间槽")
        ax.grid(alpha=0.2, linestyle="--")
    axes[0].set_ylabel("log1p(日中位流量)")
    axes[0].legend(loc="upper right")
    fig.suptitle("各簇典型日曲线", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = parse_args()
    config.figure_dir.mkdir(parents=True, exist_ok=True)

    label_df, summary_df, metric_df = load_cluster_artifacts(config.cluster_dir)
    data_tensor = load_data_tensor(config.dataset_path)
    features_z, _ = build_grid_region_features(data_tensor, config.t_in, config.t_out)
    daily_profiles = build_daily_region_profiles(data_tensor)
    grid_shape = infer_grid_shape(config.pooled_input, int(data_tensor.shape[1]))

    plot_k_search(metric_df, config.figure_dir / "k_search_diagnostics.png")
    plot_cluster_size(summary_df, config.figure_dir / "cluster_size_bar.png")
    plot_pca_scatter(features_z, label_df, config.figure_dir / "cluster_pca_scatter.png", config.seed)
    plot_grid_cluster_map(label_df, grid_shape, config.figure_dir / "cluster_grid_map.png")
    plot_cluster_daily_profiles(daily_profiles, label_df, config.figure_dir / "cluster_daily_profiles.png")

    logging.info("聚类可视化已输出到: %s", config.figure_dir)


if __name__ == "__main__":
    main()
