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
        default=200,
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
    parser.add_argument(
        "--representative-top-n",
        type=int,
        default=6,
        help="每个 cluster 代表节点曲线数量。",
    )
    parser.add_argument(
        "--overlay-center-type",
        choices=["mean", "median", "saved_center"],
        default="saved_center",
        help="overlay 图中心线类型。",
    )
    parser.add_argument(
        "--main-y-max",
        type=float,
        default=2.2,
        help="主图 y 轴上限；小于等于 0 时回退使用分位数自动计算。",
    )
    parser.add_argument(
        "--diagnostic-y-max",
        type=float,
        default=4.0,
        help="诊断图 y 轴上限；小于等于 0 时回退使用分位数自动计算。",
    )
    parser.add_argument(
        "--all-curves-alpha",
        type=float,
        default=0.035,
        help="全量曲线诊断图中的线条透明度。",
    )
    parser.add_argument(
        "--plot-y-quantile",
        type=float,
        default=0.99,
        help="绘图时 y 轴上限使用的归一化流量分位数。",
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
            (pl.col(RESIDUAL_COL) / pl.col("_observed_mean")).alias("_normalized_residual"),
        ])
        .sort([CLUSTER_COL, NODE_COL, DAY_SLOT_COL])
    )


def build_summary_lookup(summary_df: pl.DataFrame) -> Dict[int, Dict[str, float]]:
    """将 cluster_summary 转为按 cluster_id 索引的字典。"""
    lookup: Dict[int, Dict[str, float]] = {}
    for row in summary_df.to_dicts():
        lookup[int(row[CLUSTER_COL])] = row
    return lookup


def build_cluster_quantile_curves(
    curve_with_label_df: pl.DataFrame,
    value_col: str = "_normalized_fitted_flow",
) -> pl.DataFrame:
    """按 cluster 和 day_slot 计算分位数曲线。"""
    return (
        curve_with_label_df.group_by([CLUSTER_COL, DAY_SLOT_COL])
        .agg([
            pl.col(value_col).quantile(0.10).alias("q10"),
            pl.col(value_col).quantile(0.25).alias("q25"),
            pl.col(value_col).median().alias("median"),
            pl.col(value_col).mean().alias("mean"),
            pl.col(value_col).quantile(0.75).alias("q75"),
            pl.col(value_col).quantile(0.90).alias("q90"),
            pl.len().alias("point_count"),
        ])
        .sort([CLUSTER_COL, DAY_SLOT_COL])
    )


def build_cluster_summary_from_curve_df(curve_with_label_df: pl.DataFrame) -> pl.DataFrame:
    """从曲线表中回收节点级摘要，便于各图共用标题统计。"""
    node_metric_df = (
        curve_with_label_df.select([NODE_COL, CLUSTER_COL, "R2", "RMSE"])
        .unique(subset=[NODE_COL], keep="first")
        .sort([CLUSTER_COL, NODE_COL])
    )
    return (
        node_metric_df.group_by(CLUSTER_COL)
        .agg([
            pl.len().alias("节点数"),
            pl.col("R2").mean().alias("平均R2"),
            pl.col("RMSE").mean().alias("平均RMSE"),
        ])
        .sort(CLUSTER_COL)
    )


def resolve_y_axis_max(
    curve_with_label_df: pl.DataFrame,
    value_col: str,
    plot_y_quantile: float,
    hard_cap: float,
    fixed_y_max: float,
) -> float:
    """按全局分位数为图像设置更稳健的 y 轴上限。"""
    if fixed_y_max > 0:
        return float(fixed_y_max)
    quantile_value = curve_with_label_df.get_column(value_col).quantile(plot_y_quantile)
    if quantile_value is None:
        return hard_cap
    y_max = min(float(quantile_value), hard_cap)
    return max(y_max, 1.0)


def build_cluster_median_curve_lookup(
    curve_with_label_df: pl.DataFrame,
    value_col: str = "_normalized_fitted_flow",
) -> Dict[int, np.ndarray]:
    """返回每个 cluster 的按 day_slot 排序的中位曲线。"""
    median_df = (
        curve_with_label_df.group_by([CLUSTER_COL, DAY_SLOT_COL])
        .agg(pl.col(value_col).median().alias("median"))
        .sort([CLUSTER_COL, DAY_SLOT_COL])
    )
    lookup: Dict[int, np.ndarray] = {}
    for cluster_df in median_df.partition_by(CLUSTER_COL, maintain_order=True):
        lookup[int(cluster_df.item(0, CLUSTER_COL))] = (
            cluster_df.get_column("median").to_numpy().astype(np.float64)
        )
    return lookup


def format_cluster_title(
    cluster_id: int,
    summary_row: Dict[str, float],
    extra_text: str | None = None,
) -> str:
    """统一各图的 cluster 标题样式。"""
    title = (
        f"{cluster_display_name(cluster_id)}\n"
        f"节点数 n={int(summary_row['节点数'])} | "
        f"平均R2={float(summary_row['平均R2']):.3f} | "
        f"平均RMSE={float(summary_row['平均RMSE']):.2f}"
    )
    if extra_text:
        title = f"{title}\n{extra_text}"
    return title


def build_cluster_mean_curve_lookup(
    curve_with_label_df: pl.DataFrame,
    value_col: str = "_normalized_fitted_flow",
) -> Dict[int, np.ndarray]:
    """返回每个 cluster 的按 day_slot 排序的平均曲线。"""
    mean_df = (
        curve_with_label_df.group_by([CLUSTER_COL, DAY_SLOT_COL])
        .agg(pl.col(value_col).mean().alias("mean"))
        .sort([CLUSTER_COL, DAY_SLOT_COL])
    )
    lookup: Dict[int, np.ndarray] = {}
    for cluster_df in mean_df.partition_by(CLUSTER_COL, maintain_order=True):
        lookup[int(cluster_df.item(0, CLUSTER_COL))] = (
            cluster_df.get_column("mean").to_numpy().astype(np.float64)
        )
    return lookup


def build_saved_center_curve_lookup(center_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """返回 parquet 中保存的 cluster center 曲线。"""
    lookup: Dict[int, np.ndarray] = {}
    for cluster_df in center_df.sort([CLUSTER_COL, DAY_SLOT_COL]).partition_by(
        CLUSTER_COL,
        maintain_order=True,
    ):
        lookup[int(cluster_df.item(0, CLUSTER_COL))] = (
            cluster_df.get_column("类平均归一化流量").to_numpy().astype(np.float64)
        )
    return lookup


def select_diverse_representative_nodes(
    cluster_curve_df: pl.DataFrame,
    median_curve: np.ndarray,
    top_n: int,
) -> List[Dict[str, float]]:
    """选择兼具典型性和差异性的代表节点。"""
    candidate_records: List[Dict[str, float]] = []
    for node_curve_df in cluster_curve_df.partition_by(NODE_COL, maintain_order=True):
        node_curve = node_curve_df.get_column("_normalized_fitted_flow").to_numpy().astype(np.float64)
        candidate_records.append({
            NODE_COL: int(node_curve_df.item(0, NODE_COL)),
            "R2": float(node_curve_df.item(0, "R2")),
            "distance": float(np.linalg.norm(node_curve - median_curve)),
        })

    candidate_records.sort(key=lambda item: (item["distance"], -item["R2"], item[NODE_COL]))
    if not candidate_records:
        return []

    top_n = max(int(top_n), 1)
    typical_count = max(1, top_n // 3)
    middle_count = max(1, top_n // 3)
    boundary_count = max(1, top_n - typical_count - middle_count)

    selected_indices: List[int] = []

    def append_near_quantile(start_ratio: float, end_ratio: float, count: int) -> None:
        if count <= 0:
            return
        if len(candidate_records) == 1:
            selected_indices.append(0)
            return
        start_idx = int(round((len(candidate_records) - 1) * start_ratio))
        end_idx = int(round((len(candidate_records) - 1) * end_ratio))
        if end_idx < start_idx:
            end_idx = start_idx
        pool = list(range(start_idx, end_idx + 1))
        if not pool:
            pool = [start_idx]
        if count == 1:
            selected_indices.append(pool[len(pool) // 2])
            return
        for pos in np.linspace(0, len(pool) - 1, num=count):
            selected_indices.append(pool[int(round(pos))])

    append_near_quantile(0.0, 0.20, typical_count)
    append_near_quantile(0.40, 0.60, middle_count)
    append_near_quantile(0.70, 0.90, boundary_count)

    selected_records: List[Dict[str, float]] = []
    seen_node_ids = set()
    for idx in selected_indices:
        record = candidate_records[max(0, min(idx, len(candidate_records) - 1))]
        node_id = int(record[NODE_COL])
        if node_id in seen_node_ids:
            continue
        selected_records.append(record)
        seen_node_ids.add(node_id)

    for record in candidate_records:
        if len(selected_records) >= top_n:
            break
        node_id = int(record[NODE_COL])
        if node_id in seen_node_ids:
            continue
        selected_records.append(record)
        seen_node_ids.add(node_id)

    return selected_records[:top_n]


def build_residual_boxplot(
    curve_with_label_df: pl.DataFrame,
    value_col: str,
    output_path: Path,
    y_label: str,
    title: str,
) -> None:
    """绘制按 cluster 分组的裁剪后残差箱线图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residual_df = curve_with_label_df.select([CLUSTER_COL, value_col]).drop_nulls()
    lower = float(residual_df.get_column(value_col).quantile(0.01))
    upper = float(residual_df.get_column(value_col).quantile(0.99))
    clipped_df = residual_df.with_columns(
        pl.col(value_col).clip(lower_bound=lower, upper_bound=upper).alias("_residual_clipped")
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
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_function_quantile_bands(
    curve_with_label_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_path: Path,
    plot_y_quantile: float = 0.99,
    main_y_max: float = 2.2,
) -> None:
    """绘制每类拟合函数的分位带和中位数曲线。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantile_df = build_cluster_quantile_curves(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
    )
    summary_lookup = build_summary_lookup(summary_df)
    cluster_ids = sorted(quantile_df.get_column(CLUSTER_COL).unique().to_list())
    y_max = resolve_y_axis_max(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
        plot_y_quantile=plot_y_quantile,
        hard_cap=3.0,
        fixed_y_max=main_y_max,
    )
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.8), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for idx, (ax, cluster_id) in enumerate(zip(axes, cluster_ids)):
        cluster_quantile_df = quantile_df.filter(pl.col(CLUSTER_COL) == cluster_id).sort(DAY_SLOT_COL)
        x = cluster_quantile_df.get_column(DAY_SLOT_COL).to_numpy()
        q10 = cluster_quantile_df.get_column("q10").to_numpy()
        q25 = cluster_quantile_df.get_column("q25").to_numpy()
        median = cluster_quantile_df.get_column("median").to_numpy()
        mean = cluster_quantile_df.get_column("mean").to_numpy()
        q75 = cluster_quantile_df.get_column("q75").to_numpy()
        q90 = cluster_quantile_df.get_column("q90").to_numpy()
        color = CLUSTER_COLOR_MAP.get(int(cluster_id), "#4C78A8")

        ax.fill_between(
            x,
            q10,
            q90,
            color="#d9d9d9",
            alpha=0.85,
            label="10%-90%" if idx == 0 else None,
        )
        ax.fill_between(
            x,
            q25,
            q75,
            color="#969696",
            alpha=0.75,
            label="25%-75%" if idx == 0 else None,
        )
        ax.plot(
            x,
            median,
            color="black",
            linewidth=2.8,
            label="Median" if idx == 0 else None,
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label="Mean" if idx == 0 else None,
        )
        ax.set_title(format_cluster_title(cluster_id, summary_lookup[int(cluster_id)]), fontsize=11)
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_ylim(0, y_max)
        ax.set_xticks(XTICKS)
        ax.set_xlabel("day_slot")
        ax.grid(True, linestyle="--", alpha=0.30)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("归一化 fitted_flow")
    fig.suptitle("每类拟合函数分位带图", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_fitted_function_overlay(
    curve_with_label_df: pl.DataFrame,
    center_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_path: Path,
    sample_per_cluster: int,
    random_state: int,
    curve_type: str,
    plot_y_quantile: float = 0.99,
    overlay_center_type: str = "saved_center",
    main_y_max: float = 2.2,
) -> None:
    """绘制每类拟合函数叠加图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(curve_with_label_df.get_column(CLUSTER_COL).unique().to_list())
    summary_lookup = build_summary_lookup(summary_df)
    quantile_df = build_cluster_quantile_curves(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
    )
    rng = np.random.default_rng(random_state)
    y_max = resolve_y_axis_max(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
        plot_y_quantile=plot_y_quantile,
        hard_cap=3.0,
        fixed_y_max=main_y_max,
    )
    mean_lookup = build_cluster_mean_curve_lookup(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
    )
    median_lookup = build_cluster_median_curve_lookup(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
    )
    saved_center_lookup = build_saved_center_curve_lookup(center_df)
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.5), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for idx, (ax, cluster_id) in enumerate(zip(axes, cluster_ids)):
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

        cluster_center_df = center_df.filter(pl.col(CLUSTER_COL) == cluster_id).sort(DAY_SLOT_COL)
        x_center = cluster_center_df.get_column(DAY_SLOT_COL).to_numpy()
        if overlay_center_type == "mean":
            center_line = mean_lookup[int(cluster_id)]
            center_label = "Mean fitted function"
        elif overlay_center_type == "median":
            center_line = median_lookup[int(cluster_id)]
            center_label = "Median fitted function"
        else:
            center_line = saved_center_lookup[int(cluster_id)]
            center_label = "Saved cluster center"
        ax.plot(
            x_center,
            center_line,
            color="black",
            linewidth=2.8,
            label=center_label,
            zorder=5,
        )
        summary_row = summary_lookup[int(cluster_id)]
        ax.set_title(format_cluster_title(cluster_id, summary_row), fontsize=11)
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_ylim(0, y_max)
        ax.set_xticks(XTICKS)
        ax.grid(True, linestyle="--", alpha=0.30)
        ax.set_xlabel("day_slot")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("归一化流量")
    fig.suptitle("每类拟合函数叠加图", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sampled_function_cloud_with_center(
    curve_with_label_df: pl.DataFrame,
    center_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_path: Path,
    sample_per_cluster: int,
    random_state: int,
    main_y_max: float = 2.2,
) -> None:
    """绘制更接近上一版风格的抽样函数云图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(curve_with_label_df.get_column(CLUSTER_COL).unique().to_list())
    summary_lookup = build_summary_lookup(summary_df)
    saved_center_lookup = build_saved_center_curve_lookup(center_df)
    rng = np.random.default_rng(random_state)
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.5), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for idx, (ax, cluster_id) in enumerate(zip(axes, cluster_ids)):
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
            ax.plot(
                node_df.get_column(DAY_SLOT_COL).to_numpy(),
                node_df.get_column("_normalized_fitted_flow").to_numpy(),
                color=color,
                linewidth=0.8,
                alpha=0.08,
            )

        center_x = (
            center_df.filter(pl.col(CLUSTER_COL) == cluster_id)
            .sort(DAY_SLOT_COL)
            .get_column(DAY_SLOT_COL)
            .to_numpy()
        )
        ax.plot(
            center_x,
            saved_center_lookup[int(cluster_id)],
            color="black",
            linewidth=2.8,
            label="Saved cluster center",
            zorder=5,
        )
        ax.set_title(format_cluster_title(cluster_id, summary_lookup[int(cluster_id)]), fontsize=11)
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_ylim(0, main_y_max)
        ax.set_xticks(XTICKS)
        ax.set_xlabel("day_slot")
        ax.grid(True, linestyle="--", alpha=0.30)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("归一化 fitted_flow")
    fig.suptitle("每类抽样拟合函数云图与聚类中心", fontsize=15)
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
    top_n: int = 6,
    plot_y_quantile: float = 0.99,
    main_y_max: float = 2.2,
) -> None:
    """绘制每类兼具典型性和差异性的代表节点拟合函数。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = (
        label_df.group_by(CLUSTER_COL)
        .agg([
            pl.len().alias("节点数"),
            pl.col("R2").mean().alias("平均R2"),
            pl.col("RMSE").mean().alias("平均RMSE"),
        ])
        .sort(CLUSTER_COL)
    )
    summary_lookup = build_summary_lookup(summary_df)
    median_lookup = build_cluster_median_curve_lookup(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
    )
    cluster_ids = sorted(curve_with_label_df.get_column(CLUSTER_COL).unique().to_list())
    y_max = resolve_y_axis_max(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
        plot_y_quantile=plot_y_quantile,
        hard_cap=3.0,
        fixed_y_max=main_y_max,
    )
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.8), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_curve_df = (
            curve_with_label_df.filter(pl.col(CLUSTER_COL) == cluster_id)
            .sort([NODE_COL, DAY_SLOT_COL])
        )
        median_curve = median_lookup[int(cluster_id)]
        selected_records = select_diverse_representative_nodes(
            cluster_curve_df=cluster_curve_df,
            median_curve=median_curve,
            top_n=top_n,
        )
        cmap = plt.get_cmap("tab10", max(len(selected_records), 1))
        for idx, record in enumerate(selected_records):
            node_curve_df = (
                cluster_curve_df.filter(pl.col(NODE_COL) == int(record[NODE_COL]))
                .sort(DAY_SLOT_COL)
            )
            ax.plot(
                node_curve_df.get_column(DAY_SLOT_COL).to_numpy(),
                node_curve_df.get_column("_normalized_fitted_flow").to_numpy(),
                linewidth=1.8,
                alpha=0.95,
                color=cmap(idx),
                label=(
                    f"{int(record[NODE_COL])} | "
                    f"R2={record['R2']:.3f} | "
                    f"d={record['distance']:.3f}"
                ),
            )
        x = np.arange(SLOTS_PER_DAY, dtype=np.int64)
        ax.plot(
            x,
            median_curve,
            color="black",
            linewidth=2.8,
            label="Median fitted function",
            zorder=5,
        )
        ax.set_title(
            format_cluster_title(
                cluster_id,
                summary_lookup[int(cluster_id)],
                extra_text=f"diverse representatives top_n={max(top_n, 1)}",
            ),
            fontsize=11,
        )
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_ylim(0, y_max)
        ax.set_xticks(XTICKS)
        ax.set_xlabel("day_slot")
        ax.grid(True, linestyle="--", alpha=0.30)
        ax.legend(fontsize=7.5, loc="upper right")

    axes[0].set_ylabel("归一化 fitted_flow")
    fig.suptitle("每类代表节点拟合函数图", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_all_fitted_functions_diagnostic(
    curve_with_label_df: pl.DataFrame,
    output_path: Path,
    all_curves_alpha: float = 0.035,
    plot_y_quantile: float = 0.99,
    diagnostic_y_max: float = 4.0,
) -> None:
    """绘制每类全部拟合函数曲线，仅用于诊断。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = build_cluster_summary_from_curve_df(curve_with_label_df)
    summary_lookup = build_summary_lookup(summary_df)
    cluster_ids = sorted(curve_with_label_df.get_column(CLUSTER_COL).unique().to_list())
    y_max = resolve_y_axis_max(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_fitted_flow",
        plot_y_quantile=plot_y_quantile,
        hard_cap=4.0,
        fixed_y_max=diagnostic_y_max,
    )
    fig_width = max(5 * len(cluster_ids), 14)
    fig, axes = plt.subplots(1, len(cluster_ids), figsize=(fig_width, 5.8), sharey=True)
    if len(cluster_ids) == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_curve_df = (
            curve_with_label_df.filter(pl.col(CLUSTER_COL) == cluster_id)
            .sort([NODE_COL, DAY_SLOT_COL])
        )
        for node_curve_df in cluster_curve_df.partition_by(NODE_COL, maintain_order=True):
            ax.plot(
                node_curve_df.get_column(DAY_SLOT_COL).to_numpy(),
                node_curve_df.get_column("_normalized_fitted_flow").to_numpy(),
                linewidth=0.4,
                alpha=all_curves_alpha,
                color="#808080",
            )
        ax.set_title(
            format_cluster_title(
                cluster_id,
                summary_lookup[int(cluster_id)],
                extra_text="全部节点曲线诊断图",
            ),
            fontsize=11,
        )
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_ylim(0, y_max)
        ax.set_xticks(XTICKS)
        ax.set_xlabel("day_slot")
        ax.grid(True, linestyle="--", alpha=0.30)

    axes[0].set_ylabel("归一化 fitted_flow")
    fig.suptitle("每类全部拟合函数诊断图", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution_by_cluster(
    curve_with_label_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """绘制按 cluster 对比的残差箱线图。"""
    build_residual_boxplot(
        curve_with_label_df=curve_with_label_df,
        value_col=RESIDUAL_COL,
        output_path=output_path,
        y_label="residual",
        title="拟合函数残差分布按 cluster 对比",
    )


def plot_normalized_residual_distribution_by_cluster(
    curve_with_label_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """绘制按 cluster 对比的归一化残差箱线图。"""
    build_residual_boxplot(
        curve_with_label_df=curve_with_label_df,
        value_col="_normalized_residual",
        output_path=output_path,
        y_label="normalized residual",
        title="归一化拟合残差分布按 cluster 对比",
    )


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

    sampled_cloud_path = output_dir / f"{args.method}_sampled_function_cloud_with_center.png"
    quantile_band_path = output_dir / f"{args.method}_cluster_function_quantile_bands.png"
    overlay_path = output_dir / f"{args.method}_fitted_function_overlay.png"
    mean_vs_center_path = output_dir / f"{args.method}_cluster_mean_fitted_vs_center.png"
    representative_path = output_dir / f"{args.method}_representative_fitted_functions.png"
    all_curves_diagnostic_path = output_dir / f"{args.method}_all_fitted_functions_diagnostic.png"
    residual_path = output_dir / f"{args.method}_residual_distribution_by_cluster.png"
    normalized_residual_path = (
        output_dir / f"{args.method}_normalized_residual_distribution_by_cluster.png"
    )
    cluster_summary_csv_path = output_dir / f"{args.method}_function_cluster_summary.csv"
    node_labels_csv_path = output_dir / f"{args.method}_node_function_cluster_labels.csv"

    plot_sampled_function_cloud_with_center(
        curve_with_label_df=curve_with_label_df,
        center_df=center_df,
        summary_df=summary_df,
        output_path=sampled_cloud_path,
        sample_per_cluster=args.sample_per_cluster,
        random_state=args.random_state,
        main_y_max=args.main_y_max,
    )
    plot_fitted_function_overlay(
        curve_with_label_df=curve_with_label_df,
        center_df=center_df,
        summary_df=summary_df,
        output_path=overlay_path,
        sample_per_cluster=args.sample_per_cluster,
        random_state=args.random_state,
        curve_type=curve_type,
        plot_y_quantile=args.plot_y_quantile,
        overlay_center_type=args.overlay_center_type,
        main_y_max=args.main_y_max,
    )
    plot_cluster_function_quantile_bands(
        curve_with_label_df=curve_with_label_df,
        summary_df=summary_df,
        output_path=quantile_band_path,
        plot_y_quantile=args.plot_y_quantile,
        main_y_max=args.main_y_max,
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
        top_n=args.representative_top_n,
        plot_y_quantile=args.plot_y_quantile,
        main_y_max=args.main_y_max,
    )
    plot_all_fitted_functions_diagnostic(
        curve_with_label_df=curve_with_label_df,
        output_path=all_curves_diagnostic_path,
        all_curves_alpha=args.all_curves_alpha,
        plot_y_quantile=args.plot_y_quantile,
        diagnostic_y_max=args.diagnostic_y_max,
    )
    plot_residual_distribution_by_cluster(
        curve_with_label_df=curve_with_label_df,
        output_path=residual_path,
    )
    plot_normalized_residual_distribution_by_cluster(
        curve_with_label_df=curve_with_label_df,
        output_path=normalized_residual_path,
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
    print(f"- 抽样函数云图主图：{sampled_cloud_path}")
    print(f"- 技术检查叠加图：{overlay_path}")
    print(f"- 分位带主图：{quantile_band_path}")
    print(f"- 聚类中心函数对比图：{mean_vs_center_path}")
    print(f"- 代表节点函数图：{representative_path}")
    print(f"- 全量曲线诊断图：{all_curves_diagnostic_path}")
    print(f"- 残差分布图：{residual_path}")
    print(f"- 归一化残差分布图：{normalized_residual_path}")
    print(f"- 聚类形态摘要表：{cluster_summary_csv_path}")
    print(f"- 节点标签表：{node_labels_csv_path}")
    print("这些结果用于从“函数曲线形态”角度解释聚类，而不是从 PCA 投影角度解释聚类。")


if __name__ == "__main__":
    main()
