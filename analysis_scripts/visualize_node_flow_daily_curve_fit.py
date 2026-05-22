"""可视化路口节点日内平均车流量曲线及其傅里叶拟合结果。"""

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
FIT_DIR = ROOT_DIR / "data" / "analysis" / "node_flow_curve_fit"

FITTED_CURVE_PATH = FIT_DIR / "node_flow_fitted_daily_curves.parquet"
COEFFICIENT_PATH = FIT_DIR / "node_flow_curve_coefficients.parquet"
OUTPUT_DIR = FIT_DIR / "plots"
CURVE_GRID_PATH = OUTPUT_DIR / "node_flow_daily_curve_fit_samples.png"
METRICS_PATH = OUTPUT_DIR / "node_flow_daily_curve_fit_metrics.png"

NODE_COL = "节点ID"
DAY_SLOT_COL = "日内时间段"
AVG_FLOW_COL = "平均路口车流量"
FITTED_FLOW_COL = "拟合路口车流量"
RESIDUAL_COL = "残差"

TOP_N_NODES = 12
FIG_DPI = 200
SLOTS_PER_DAY = 96


def configure_fonts() -> None:
    """配置中文字体，避免图表中文乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def check_input_files() -> None:
    """检查拟合结果输入文件是否存在。"""
    if not FIT_DIR.exists():
        raise FileNotFoundError(f"拟合结果目录不存在: {FIT_DIR}")
    if not FITTED_CURVE_PATH.exists():
        raise FileNotFoundError(f"未找到拟合曲线结果文件: {FITTED_CURVE_PATH}")
    if not COEFFICIENT_PATH.exists():
        raise FileNotFoundError(f"未找到系数结果文件: {COEFFICIENT_PATH}")


def load_results() -> tuple[pl.DataFrame, pl.DataFrame]:
    """加载拟合曲线结果和系数结果。"""
    fitted_df = pl.read_parquet(FITTED_CURVE_PATH)
    coeff_df = pl.read_parquet(COEFFICIENT_PATH)

    required_fitted_cols = [NODE_COL, DAY_SLOT_COL, AVG_FLOW_COL, FITTED_FLOW_COL, RESIDUAL_COL]
    required_coeff_cols = [NODE_COL, "RMSE", "MAE", "R2", "平均流量", "最大流量", "最小流量"]

    missing_fitted = [col for col in required_fitted_cols if col not in fitted_df.columns]
    missing_coeff = [col for col in required_coeff_cols if col not in coeff_df.columns]

    if missing_fitted:
        raise ValueError(f"拟合曲线结果缺少字段: {missing_fitted}")
    if missing_coeff:
        raise ValueError(f"系数结果缺少字段: {missing_coeff}")

    return fitted_df, coeff_df


def select_nodes_to_plot(coeff_df: pl.DataFrame, top_n: int) -> List[int]:
    """默认选择平均流量最高的一批节点进行绘图。"""
    selected = (
        coeff_df
        .sort(["平均流量", "R2"], descending=[True, True])
        .head(top_n)
        .get_column(NODE_COL)
        .to_list()
    )
    return [int(node_id) for node_id in selected]


def build_grid(n_items: int) -> tuple[int, int]:
    """根据节点数量自动计算子图网格。"""
    if n_items <= 0:
        return 1, 1

    n_cols = min(3, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    return n_rows, n_cols


def plot_metrics_overview(coeff_df: pl.DataFrame, output_path: Path) -> None:
    """绘制全体节点的拟合质量总览图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rmse = coeff_df.get_column("RMSE").to_numpy()
    mae = coeff_df.get_column("MAE").to_numpy()
    r2 = coeff_df.get_column("R2").to_numpy()
    mean_flow = coeff_df.get_column("平均流量").to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(rmse, bins=40, color="#4C78A8", alpha=0.9, edgecolor="white")
    axes[0, 0].set_title("RMSE 分布")
    axes[0, 0].set_xlabel("RMSE")
    axes[0, 0].set_ylabel("节点数")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[0, 1].hist(mae, bins=40, color="#F58518", alpha=0.9, edgecolor="white")
    axes[0, 1].set_title("MAE 分布")
    axes[0, 1].set_xlabel("MAE")
    axes[0, 1].set_ylabel("节点数")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1, 0].hist(r2, bins=40, color="#54A24B", alpha=0.9, edgecolor="white")
    axes[1, 0].set_title("R2 分布")
    axes[1, 0].set_xlabel("R2")
    axes[1, 0].set_ylabel("节点数")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1, 1].scatter(
        mean_flow,
        r2,
        s=10,
        alpha=0.35,
        color="#E45756",
        edgecolors="none",
    )
    axes[1, 1].set_title("平均流量 vs R2")
    axes[1, 1].set_xlabel("平均流量")
    axes[1, 1].set_ylabel("R2")
    axes[1, 1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("路口节点日内曲线拟合质量总览", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def iter_node_curve_frames(fitted_df: pl.DataFrame, node_ids: Sequence[int]) -> Iterable[pl.DataFrame]:
    """按给定节点顺序返回对应曲线数据。"""
    if not node_ids:
        return []

    node_set = set(int(node_id) for node_id in node_ids)
    filtered = (
        fitted_df
        .filter(pl.col(NODE_COL).is_in(node_set))
        .sort([NODE_COL, DAY_SLOT_COL])
    )

    frames = filtered.partition_by(NODE_COL, maintain_order=True)
    frame_map = {int(frame.item(0, NODE_COL)): frame for frame in frames}
    return [frame_map[node_id] for node_id in node_ids if node_id in frame_map]


def plot_sample_node_curves(
    fitted_df: pl.DataFrame,
    coeff_df: pl.DataFrame,
    node_ids: Sequence[int],
    output_path: Path,
) -> None:
    """绘制若干代表节点的平均曲线与拟合曲线对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coeff_map = {
        int(row[NODE_COL]): row
        for row in coeff_df.select([NODE_COL, "RMSE", "MAE", "R2", "平均流量"]).to_dicts()
    }

    node_frames = list(iter_node_curve_frames(fitted_df, node_ids))
    if not node_frames:
        raise ValueError("未找到可用于绘图的节点曲线数据")

    n_rows, n_cols = build_grid(len(node_frames))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 3.8 * n_rows),
        sharex=True,
    )

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = np.array([axes])

    x_ticks = np.arange(0, SLOTS_PER_DAY + 1, 12)

    for ax, node_df in zip(axes_flat, node_frames):
        node_id = int(node_df.item(0, NODE_COL))
        day_slot = node_df.get_column(DAY_SLOT_COL).to_numpy()
        avg_flow = node_df.get_column(AVG_FLOW_COL).to_numpy()
        fitted_flow = node_df.get_column(FITTED_FLOW_COL).to_numpy()
        residual = node_df.get_column(RESIDUAL_COL).to_numpy()
        meta = coeff_map[node_id]

        ax.plot(day_slot, avg_flow, color="#4C78A8", linewidth=1.8, label="平均流量")
        ax.plot(day_slot, fitted_flow, color="#E45756", linewidth=1.6, linestyle="--", label="拟合流量")
        ax.fill_between(day_slot, residual, 0, color="#72B7B2", alpha=0.20, label="残差")

        ax.set_title(
            f"节点 {node_id}\nR2={meta['R2']:.4f}, RMSE={meta['RMSE']:.2f}, 平均流量={meta['平均流量']:.2f}",
            fontsize=10,
        )
        ax.set_xlim(0, SLOTS_PER_DAY - 1)
        ax.set_xticks(x_ticks)
        ax.set_xlabel("日内时间段")
        ax.set_ylabel("车流量")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes_flat[len(node_frames):]:
        ax.axis("off")

    fig.suptitle("代表节点日内平均车流量与傅里叶拟合曲线对比", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """主函数。"""
    configure_fonts()
    check_input_files()

    fitted_df, coeff_df = load_results()
    selected_nodes = select_nodes_to_plot(coeff_df, TOP_N_NODES)

    print(f"拟合曲线记录数: {fitted_df.height}")
    print(f"系数结果记录数: {coeff_df.height}")
    print(f"默认绘图节点数: {len(selected_nodes)}")
    print(f"选中节点ID: {selected_nodes}")

    plot_metrics_overview(coeff_df, METRICS_PATH)
    plot_sample_node_curves(
        fitted_df=fitted_df,
        coeff_df=coeff_df,
        node_ids=selected_nodes,
        output_path=CURVE_GRID_PATH,
    )

    print(f"拟合质量总览图: {METRICS_PATH}")
    print(f"节点曲线对比图: {CURVE_GRID_PATH}")


if __name__ == "__main__":
    main()
