"""基于已输出的拟合曲线与聚类标签，直接展示函数曲线层面的聚类结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis_scripts.compare_date_type_curve_methods import (  # noqa: E402
    CLUSTER_COL,
    DAY_SLOT_COL,
    EPSILON,
    FITTED_FLOW_COL,
    FIG_DPI,
    METHOD_COL,
    NODE_COL,
    OBSERVED_FLOW_COL,
    OUTPUT_ROOT,
    RESIDUAL_COL,
)
from analysis_scripts.fit_node_flow_daily_curve import SLOTS_PER_DAY  # noqa: E402


DEFAULT_METHOD = "M2_shape_normalized_weighted_curve"
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "analysis"
    / "date_type_curve_method_comparison"
    / "function_cluster_visualization"
)
MORNING_SLOT_START = 24
MORNING_SLOT_END = 39
EVENING_SLOT_START = 64
EVENING_SLOT_END = 79
XTICKS = np.array([0, 12, 24, 36, 48, 60, 72, 84, 96], dtype=np.int64)
REQUIRED_FILES = [
    "fitted_curves.parquet",
    "cluster_labels.parquet",
    "cluster_summary.parquet",
    "cluster_centers.parquet",
    "curve_coefficients.parquet",
]
CLUSTER_NAME_MAP = {
    0: "全天平稳型",
    1: "日间活跃型",
    2: "强早晚峰型",
}
CLUSTER_COLOR_MAP = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#B279A2",
    4: "#E45756",
}
INTERPRETATION_MAP = {
    0: "全天流量波动较小，整体形态相对平稳，适合作为稳定型路口代表。",
    1: "白天时段流量更活跃，通常在工作时段形成更明显的平台或单峰。",
    2: "早晚高峰更突出，通勤特征明显，早峰与晚峰存在显著双峰结构。",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于既有拟合函数和聚类标签，输出函数曲线层面的聚类可视化结果。"
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="要可视化的方法名称。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认保存到函数聚类可视化目录。",
    )
    parser.add_argument(
        "--sample-per-cluster",
        type=int,
        default=80,
        help="每个 cluster 在叠加图中的抽样节点数。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="抽样与随机过程使用的随机种子。",
    )
    parser.add_argument(
        "--show-observed",
        action="store_true",
        help="兼容参数；若传入且 --curve-type 为 fitted，则自动切换为 both。",
    )
    parser.add_argument(
        "--curve-type",
        choices=["fitted", "observed", "both"],
        default="fitted",
        help="指定叠加图展示拟合曲线、原始曲线或两者同时展示。",
    )
    return parser.parse_args()


def configure_fonts() -> None:
    """配置中文字体。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_input_files(method_dir: Path) -> None:
    """检查输入 parquet 是否齐全。"""
    missing_files = [name for name in REQUIRED_FILES if not (method_dir / name).exists()]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            f"未找到方法目录 `{method_dir}` 下的结果文件: {missing_text}。"
            "请先运行 compare_date_type_curve_methods.py 生成对应 parquet 输出。"
        )


def resolve_curve_type(args: argparse.Namespace) -> str:
    """兼容 show_observed 与 curve_type 两套参数。"""
    if args.show_observed and args.curve_type == "fitted":
        return "both"
    return str(args.curve_type)


def cluster_display_name(cluster_id: int) -> str:
    """返回聚类展示名称。"""
    base_name = CLUSTER_NAME_MAP.get(int(cluster_id), f"Cluster {int(cluster_id)}")
    return f"Cluster {int(cluster_id)}：{base_name}"


def slot_to_time_label(slot: int) -> str:
    """将 15 分钟粒度的 slot 转换为 HH:MM。"""
    total_minutes = int(slot) * 15
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def normalize_series(values: np.ndarray) -> np.ndarray:
    """将单节点曲线按节点内均值归一化。"""
    mean_value = float(np.mean(values))
    if abs(mean_value) <= EPSILON:
        return np.zeros_like(values, dtype=np.float64)
    return values / mean_value


def prepare_curve_with_label_df(
    fitted_df: pl.DataFrame,
    label_df: pl.DataFrame,
) -> pl.DataFrame:
    """合并曲线与聚类标签，并补充节点内归一化结果。"""
    curve_with_label_df = fitted_df.join(
        label_df.select([NODE_COL, CLUSTER_COL, "R2", "RMSE", "MAE", "平均流量"]),
        on=NODE_COL,
        how="inner",
    )

    norm_df = (
        curve_with_label_df.group_by(NODE_COL)
        .agg([
            pl.col(FITTED_FLOW_COL).mean().alias("_fitted_mean"),
            pl.col(OBSERVED_FLOW_COL).mean().alias("_observed_mean"),
        ])
        .with_columns([
            pl.when(pl.col("_fitted_mean").abs() <= EPSILON)
            .then(1.0)
            .otherwise(pl.col("_fitted_mean"))
            .alias("_fitted_mean"),
            pl.when(pl.col("_observed_mean").abs() <= EPSILON)
            .then(1.0)
            .otherwise(pl.col("_observed_mean"))
            .alias("_observed_mean"),
        ])
    )
    return (
        curve_with_label_df.join(norm_df, on=NODE_COL, how="left")
        .with_columns([
            (pl.col(FITTED_FLOW_COL) / pl.col("_fitted_mean")).alias("_normalized_fitted_flow"),
            (
                pl.col(OBSERVED_FLOW_COL) / pl.col("_observed_mean")
            ).alias("_normalized_observed_flow"),
        ])
        .sort([CLUSTER_COL, NODE_COL, DAY_SLOT_COL])
    )


def build_summary_lookup(summary_df: pl.DataFrame) -> Dict[int, Dict[str, float]]:
    """将 cluster_summary 转为按 cluster_id 索引的字典。"""
    lookup: Dict[int, Dict[str, float]] = {}
    for row in summary_df.to_dicts():
        lookup[int(row[CLUSTER_COL])] = row
    return lookup


def plot_fitted_function_overlay(
    curve_with_label_df: pl.DataFrame,
    center_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_path: Path,
    sample_per_cluster: int,
    random_state: int,
    curve_type: str,
) -> None:
    """绘制每类拟合函数叠加图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(curve_with_label_df.get_column(CLUSTER_COL).unique().to_list())
    summary_lookup = build_summary_lookup(summary_df)
    rng = np.random.default_rng(random_state)
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.5), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_df = curve_with_label_df.filter(pl.col(CLUSTER_COL) == cluster_id)
        node_ids = np.array(cluster_df.get_column(NODE_COL).unique().sort().to_list(), dtype=np.int64)
        sample_size = min(sample_per_cluster, node_ids.size)
        sampled_node_ids = np.sort(rng.choice(node_ids, size=sample_size, replace=False))
        sampled_df = (
            cluster_df.filter(pl.col(NODE_COL).is_in(sampled_node_ids.tolist()))
            .sort([NODE_COL, DAY_SLOT_COL])
        )
        color = CLUSTER_COLOR_MAP.get(int(cluster_id), "#4C78A8")
        for node_df in sampled_df.partition_by(NODE_COL, maintain_order=True):
            x = node_df.get_column(DAY_SLOT_COL).to_numpy()
            fitted_y = node_df.get_column("_normalized_fitted_flow").to_numpy()
            observed_y = node_df.get_column("_normalized_observed_flow").to_numpy()
            if curve_type in {"fitted", "both"}:
                ax.plot(x, fitted_y, color=color, linewidth=0.9, alpha=0.12)
            if curve_type in {"observed", "both"}:
                ax.plot(
                    x,
                    observed_y,
                    color=color,
                    linewidth=0.8,
                    alpha=0.10,
                    linestyle="--" if curve_type == "both" else "-",
                )

        cluster_center_df = (
            center_df.filter(pl.col(CLUSTER_COL) == cluster_id)
            .sort(DAY_SLOT_COL)
        )
        ax.plot(
            cluster_center_df.get_column(DAY_SLOT_COL).to_numpy(),
            cluster_center_df.get_column("类平均归一化流量").to_numpy(),
            color="black",
            linewidth=2.8,
            label="Cluster Center",
            zorder=5,
        )
        summary_row = summary_lookup[int(cluster_id)]
        ax.set_title(
            f"{cluster_display_name(cluster_id)}\n"
            f"节点数 n={int(summary_row['节点数'])} | "
            f"平均R2={float(summary_row['平均R2']):.3f} | "
            f"平均RMSE={float(summary_row['平均RMSE']):.2f}",
            fontsize=11,
        )
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_xticks(XTICKS)
        ax.grid(True, linestyle="--", alpha=0.30)
        ax.set_xlabel("day_slot")

    axes[0].set_ylabel("归一化流量")
    fig.suptitle("每类拟合函数叠加图", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_mean_fitted_vs_center(
    curve_with_label_df: pl.DataFrame,
    center_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """绘制 cluster 平均拟合函数与已有中心曲线的对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_mean_df = (
        curve_with_label_df.group_by([CLUSTER_COL, DAY_SLOT_COL])
        .agg(pl.col("_normalized_fitted_flow").mean().alias("cluster_mean_fitted"))
        .sort([CLUSTER_COL, DAY_SLOT_COL])
    )
    cluster_ids = sorted(cluster_mean_df.get_column(CLUSTER_COL).unique().to_list())
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for cluster_id in cluster_ids:
        mean_df = cluster_mean_df.filter(pl.col(CLUSTER_COL) == cluster_id).sort(DAY_SLOT_COL)
        center_cluster_df = center_df.filter(pl.col(CLUSTER_COL) == cluster_id).sort(DAY_SLOT_COL)
        color = CLUSTER_COLOR_MAP.get(int(cluster_id), "#4C78A8")
        ax.plot(
            mean_df.get_column(DAY_SLOT_COL).to_numpy(),
            mean_df.get_column("cluster_mean_fitted").to_numpy(),
            linewidth=2.2,
            color=color,
            label=f"{cluster_display_name(cluster_id)} 平均拟合",
        )
        ax.plot(
            center_cluster_df.get_column(DAY_SLOT_COL).to_numpy(),
            center_cluster_df.get_column("类平均归一化流量").to_numpy(),
            linewidth=2.0,
            color=color,
            linestyle="--",
            label=f"{cluster_display_name(cluster_id)} 中心曲线",
        )

    ax.set_xlim(0, SLOTS_PER_DAY)
    ax.set_xticks(XTICKS)
    ax.set_xlabel("day_slot")
    ax.set_ylabel("归一化流量")
    ax.set_title("每类平均拟合函数与中心曲线对比")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_representative_fitted_functions(
    curve_with_label_df: pl.DataFrame,
    label_df: pl.DataFrame,
    output_path: Path,
    top_n: int = 5,
) -> None:
    """绘制每类全部节点的归一化拟合函数。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    del label_df, top_n
    cluster_ids = sorted(curve_with_label_df.get_column(CLUSTER_COL).unique().to_list())
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.5), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_curve_df = (
            curve_with_label_df.filter(pl.col(CLUSTER_COL) == cluster_id)
            .sort([NODE_COL, DAY_SLOT_COL])
        )
        node_count = cluster_curve_df.get_column(NODE_COL).n_unique()
        for node_curve_df in cluster_curve_df.partition_by(NODE_COL, maintain_order=True):
            ax.plot(
                node_curve_df.get_column(DAY_SLOT_COL).to_numpy(),
                node_curve_df.get_column("_normalized_fitted_flow").to_numpy(),
                linewidth=0.7,
                alpha=0.12,
                color="#808080",
            )
        ax.set_title(f"{cluster_display_name(cluster_id)}\n全部节点曲线 n={node_count}")
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_xticks(XTICKS)
        ax.set_xlabel("day_slot")
        ax.grid(True, linestyle="--", alpha=0.30)

    axes[0].set_ylabel("归一化 fitted_flow")
    fig.suptitle("每类全部节点拟合函数图", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution_by_cluster(
    curve_with_label_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """绘制按 cluster 对比的残差箱线图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residual_df = curve_with_label_df.select([CLUSTER_COL, RESIDUAL_COL]).drop_nulls()
    lower = float(residual_df.get_column(RESIDUAL_COL).quantile(0.01))
    upper = float(residual_df.get_column(RESIDUAL_COL).quantile(0.99))
    clipped_df = residual_df.with_columns(
        pl.col(RESIDUAL_COL).clip(lower_bound=lower, upper_bound=upper).alias("_residual_clipped")
    )
    cluster_ids = sorted(clipped_df.get_column(CLUSTER_COL).unique().to_list())
    data = [
        clipped_df.filter(pl.col(CLUSTER_COL) == cluster_id)
        .get_column("_residual_clipped")
        .to_numpy()
        for cluster_id in cluster_ids
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=[str(int(cluster_id)) for cluster_id in cluster_ids],
        showfliers=False,
    )
    for patch, cluster_id in zip(box["boxes"], cluster_ids):
        patch.set_facecolor(CLUSTER_COLOR_MAP.get(int(cluster_id), "#4C78A8"))
        patch.set_alpha(0.55)

    ax.set_xlabel("cluster_id")
    ax.set_ylabel("residual")
    ax.set_title("拟合函数残差分布按 cluster 对比")
    ax.grid(True, axis="y", linestyle="--", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_function_cluster_summary(
    curve_with_label_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_path: Path,
    method: str,
) -> None:
    """构建每类函数形态摘要表。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_curve_df = (
        curve_with_label_df.group_by([METHOD_COL, CLUSTER_COL, DAY_SLOT_COL])
        .agg(pl.col("_normalized_fitted_flow").mean().alias("cluster_mean_fitted"))
        .sort([CLUSTER_COL, DAY_SLOT_COL])
    )
    summary_lookup = build_summary_lookup(summary_df)
    records: List[Dict[str, object]] = []

    for cluster_df in cluster_curve_df.partition_by(CLUSTER_COL, maintain_order=True):
        cluster_id = int(cluster_df.item(0, CLUSTER_COL))
        slots = cluster_df.get_column(DAY_SLOT_COL).to_numpy().astype(np.int64)
        values = cluster_df.get_column("cluster_mean_fitted").to_numpy().astype(np.float64)
        peak_idx = int(np.argmax(values))
        valley_idx = int(np.argmin(values))
        peak_slot = int(slots[peak_idx])
        valley_slot = int(slots[valley_idx])
        morning_mask = (slots >= MORNING_SLOT_START) & (slots <= MORNING_SLOT_END)
        evening_mask = (slots >= EVENING_SLOT_START) & (slots <= EVENING_SLOT_END)
        morning_peak = float(np.max(values[morning_mask]))
        evening_peak = float(np.max(values[evening_mask]))
        mean_flow = float(
            curve_with_label_df.filter(pl.col(CLUSTER_COL) == cluster_id)
            .get_column("平均流量")
            .mean()
        )
        summary_row = summary_lookup[cluster_id]
        records.append({
            "method": method,
            "cluster_id": cluster_id,
            "cluster_name": CLUSTER_NAME_MAP.get(cluster_id, f"Cluster {cluster_id}"),
            "cluster_name_source": "manual_label_by_M2_shape",
            "node_count": int(summary_row["节点数"]),
            "mean_R2": float(summary_row["平均R2"]),
            "mean_RMSE": float(summary_row["平均RMSE"]),
            "mean_MAE": float(summary_row["平均MAE"]),
            "mean_flow": mean_flow,
            "peak_slot": peak_slot,
            "peak_time_label": slot_to_time_label(peak_slot),
            "peak_value": float(values[peak_idx]),
            "valley_slot": valley_slot,
            "valley_value": float(values[valley_idx]),
            "peak_valley_diff": float(values[peak_idx] - values[valley_idx]),
            "morning_peak": morning_peak,
            "evening_peak": evening_peak,
            "morning_evening_ratio": float(morning_peak / (evening_peak + EPSILON)),
            "interpretation": INTERPRETATION_MAP.get(
                cluster_id,
                "该类函数形态用于辅助解释聚类结构，建议结合中心曲线图一并阅读。",
            ),
            "interpretation_basis": (
                "基于 M2 聚类中心曲线解释"
                if method == DEFAULT_METHOD
                else "基于 M2 聚类中心曲线解释，其他方法仅供参考"
            ),
        })

    pl.DataFrame(records).sort("cluster_id").write_csv(output_path)


def build_node_function_cluster_labels(
    label_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """导出每个节点的函数聚类标签表。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = (
        label_df.with_columns(
            pl.col(CLUSTER_COL)
            .replace_strict(CLUSTER_NAME_MAP, default=None)
            .fill_null(pl.format("Cluster {}", pl.col(CLUSTER_COL)))
            .alias("cluster_name")
        )
        .select([
            pl.col(NODE_COL).alias("node_id"),
            pl.col(METHOD_COL).alias("method"),
            pl.col(CLUSTER_COL).alias("cluster_id"),
            pl.col("cluster_name"),
            pl.col("R2"),
            pl.col("RMSE"),
            pl.col("MAE"),
            pl.col("平均流量").alias("mean_flow"),
        ])
        .sort(["cluster_id", "node_id"])
    )
    output_df.write_csv(output_path)


def main() -> None:
    """主流程。"""
    args = parse_args()
    configure_fonts()
    curve_type = resolve_curve_type(args)
    method_dir = OUTPUT_ROOT / args.method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_input_files(method_dir)
    fitted_df = pl.read_parquet(method_dir / "fitted_curves.parquet")
    label_df = pl.read_parquet(method_dir / "cluster_labels.parquet")
    summary_df = pl.read_parquet(method_dir / "cluster_summary.parquet")
    center_df = pl.read_parquet(method_dir / "cluster_centers.parquet")
    coeff_df = pl.read_parquet(method_dir / "curve_coefficients.parquet")
    if coeff_df.is_empty():
        raise ValueError(f"`{method_dir / 'curve_coefficients.parquet'}` 为空，无法继续生成函数聚类可视化。")

    curve_with_label_df = prepare_curve_with_label_df(fitted_df=fitted_df, label_df=label_df)

    overlay_path = output_dir / f"{args.method}_fitted_function_overlay.png"
    mean_vs_center_path = output_dir / f"{args.method}_cluster_mean_fitted_vs_center.png"
    representative_path = output_dir / f"{args.method}_representative_fitted_functions.png"
    residual_path = output_dir / f"{args.method}_residual_distribution_by_cluster.png"
    cluster_summary_csv_path = output_dir / f"{args.method}_function_cluster_summary.csv"
    node_labels_csv_path = output_dir / f"{args.method}_node_function_cluster_labels.csv"

    plot_fitted_function_overlay(
        curve_with_label_df=curve_with_label_df,
        center_df=center_df,
        summary_df=summary_df,
        output_path=overlay_path,
        sample_per_cluster=args.sample_per_cluster,
        random_state=args.random_state,
        curve_type=curve_type,
    )
    plot_cluster_mean_fitted_vs_center(
        curve_with_label_df=curve_with_label_df,
        center_df=center_df,
        output_path=mean_vs_center_path,
    )
    plot_representative_fitted_functions(
        curve_with_label_df=curve_with_label_df,
        label_df=label_df,
        output_path=representative_path,
        top_n=5,
    )
    plot_residual_distribution_by_cluster(
        curve_with_label_df=curve_with_label_df,
        output_path=residual_path,
    )
    build_function_cluster_summary(
        curve_with_label_df=curve_with_label_df,
        summary_df=summary_df,
        output_path=cluster_summary_csv_path,
        method=args.method,
    )
    build_node_function_cluster_labels(
        label_df=label_df,
        output_path=node_labels_csv_path,
    )

    print("已完成拟合函数聚类可视化：")
    print(f"- 拟合函数叠加图：{overlay_path}")
    print(f"- 聚类中心函数对比图：{mean_vs_center_path}")
    print(f"- 代表节点函数图：{representative_path}")
    print(f"- 残差分布图：{residual_path}")
    print(f"- 聚类形态摘要表：{cluster_summary_csv_path}")
    print(f"- 节点标签表：{node_labels_csv_path}")
    print("这些结果用于从“函数曲线形态”角度解释聚类，而不是从 PCA 投影角度解释聚类。")


if __name__ == "__main__":
    main()
