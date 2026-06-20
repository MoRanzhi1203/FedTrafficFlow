# ==========================
# add_p995_to_speed_histogram.py
# ==========================
# -*- coding: utf-8 -*-

"""按速度等级绘制速度直方图，并导出估算得到的 P99.5 速度指标。"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


# ==========================
# 路径配置
# ==========================
BASE_DIR = Path(__file__).resolve().parents[2]
CHUNK_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"

OUTPUT_IMAGE_PATH = OUTPUT_DIR / "speed_histograms_by_class_with_p995_percent.png"
OUTPUT_CSV_PATH = OUTPUT_DIR / "speed_histograms_by_class_p995.csv"


# ==========================
# 参数配置
# ==========================
HIST_MIN_SPEED = 0.0
HIST_MAX_SPEED = 120.0

# 绘图用粗分箱数量
PLOT_BIN_COUNT = 30

# P99.5 估计用细分箱数量
# 120 / 2400 = 0.05
P99_BIN_COUNT = 2400

PERCENTILE = 0.995

# 是否显示每个柱子的百分比标签
# True：信息更完整，但绘图稍慢
# False：图片更简洁，绘图更快
SHOW_BAR_LABELS = True

FIG_DPI = 220


# ==========================
# 绘图样式
# ==========================
def configure_plot_style():
    """
    配置中文字体和 Seaborn 图表风格。
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="Microsoft YaHei",
        rc={
            "axes.unicode_minus": False,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 18,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.45,
        },
    )


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    使用 Polars 流式引擎执行 LazyFrame。
    兼容新旧版本 Polars。
    """
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def format_percent_label(value: float) -> str:
    """
    格式化柱状图百分比标签。
    """
    if value >= 10:
        return f"{value:.1f}%"
    if value >= 1:
        return f"{value:.2f}%"
    return f"{value:.3f}%"


def get_grid_shape(item_count: int):
    """
    根据速度等级数量自动计算子图网格。
    """
    col_count = min(4, max(1, item_count))
    row_count = (item_count + col_count - 1) // col_count
    return row_count, col_count


def check_input_files() -> List[Path]:
    """
    检查 Parquet 分片文件是否存在。
    """
    chunk_files = sorted(CHUNK_DIR.glob("speed_chunk_*.parquet"))

    if not chunk_files:
        raise FileNotFoundError(
            f"未在目录中找到 Parquet 数据分片文件：{CHUNK_DIR}\n"
            f"请确认文件名类似：speed_chunk_000.parquet"
        )

    return chunk_files


# ==========================
# 数据聚合
# ==========================
def build_base_lazy_frame() -> pl.LazyFrame:
    """
    一次性懒加载所有 speed_chunk_*.parquet，并完成基础清洗。

    优化点：
    1. 直接 scan_parquet，不再 scan_csv；
    2. 只读取 平均速度、速度等级 两列；
    3. Parquet 已经是结构化列存格式，不再做 Utf8 -> strip -> cast；
    4. 不 collect 原始大表；
    5. 只在 LazyFrame 中完成空值过滤和速度裁剪。
    """
    parquet_pattern = (CHUNK_DIR / "speed_chunk_*.parquet").as_posix()

    return (
        pl.scan_parquet(parquet_pattern)
        .select([
            pl.col("平均速度").cast(pl.Float64, strict=False),
            pl.col("速度等级").cast(pl.Int64, strict=False),
        ])
        .filter(
            pl.col("平均速度").is_not_null()
            & pl.col("速度等级").is_not_null()
        )
        .with_columns([
            pl.col("平均速度")
            .clip(HIST_MIN_SPEED, HIST_MAX_SPEED - 1e-9)
            .alias("速度裁剪值")
        ])
    )


def aggregate_fine_histogram_bins(base_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    只对原始大数据扫描一次，聚合 P99.5 用的细粒度分箱。

    后续从这个小聚合表推导：
    1. 每个速度等级的样本数；
    2. 绘图用 30 桶百分比分布；
    3. P99.5 估计值。
    """
    p99_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / P99_BIN_COUNT

    if p99_bin_width <= 0:
        raise ValueError("P99.5 分箱宽度无效，请检查速度范围和分箱数量。")

    query = (
        base_lf
        .with_columns([
            (
                ((pl.col("速度裁剪值") - HIST_MIN_SPEED) / p99_bin_width)
                .floor()
                .clip(0, P99_BIN_COUNT - 1)
                .cast(pl.Int32)
            ).alias("p99_bin_index")
        ])
        .group_by(["速度等级", "p99_bin_index"])
        .agg([
            pl.len().alias("频数")
        ])
        .sort(["速度等级", "p99_bin_index"])
    )

    return collect_streaming(query)


def build_histogram_data_from_fine_bins(fine_bin_frame: pl.DataFrame) -> Dict:
    """
    从 2400 个细分箱聚合结果构建绘图和 P99.5 所需数据。

    fine_bin_frame 的规模约为：
    速度等级数量 × 2400
    已经非常小，可以安全地用 Python 字典处理。
    """
    plot_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / PLOT_BIN_COUNT
    p99_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / P99_BIN_COUNT

    if plot_bin_width <= 0 or p99_bin_width <= 0:
        raise ValueError("速度分箱宽度无效，请检查速度范围和分箱数量。")

    speed_classes = (
        fine_bin_frame
        .select("速度等级")
        .unique()
        .sort("速度等级")
        .get_column("速度等级")
        .to_list()
    )

    plot_counts = {
        int(speed_class): [0] * PLOT_BIN_COUNT
        for speed_class in speed_classes
    }

    p99_counts = {
        int(speed_class): [0] * P99_BIN_COUNT
        for speed_class in speed_classes
    }

    sample_sizes = {
        int(speed_class): 0
        for speed_class in speed_classes
    }

    # 如果 2400 能整除 30，则每 80 个细分箱合并为 1 个绘图箱
    if P99_BIN_COUNT % PLOT_BIN_COUNT == 0:
        fine_bins_per_plot_bin = P99_BIN_COUNT // PLOT_BIN_COUNT
    else:
        fine_bins_per_plot_bin = None

    for speed_class, p99_bin_index, count in fine_bin_frame.iter_rows():
        speed_class = int(speed_class)
        p99_bin_index = int(p99_bin_index)
        count = int(count)

        p99_counts[speed_class][p99_bin_index] = count
        sample_sizes[speed_class] += count

        if fine_bins_per_plot_bin is not None:
            plot_bin_index = p99_bin_index // fine_bins_per_plot_bin
        else:
            bin_left_speed = HIST_MIN_SPEED + p99_bin_index * p99_bin_width
            plot_bin_index = int((bin_left_speed - HIST_MIN_SPEED) / plot_bin_width)

        plot_bin_index = max(0, min(PLOT_BIN_COUNT - 1, plot_bin_index))
        plot_counts[speed_class][plot_bin_index] += count

    return {
        "plot_counts": plot_counts,
        "p99_counts": p99_counts,
        "sample_sizes": sample_sizes,
        "plot_bin_width": plot_bin_width,
        "p99_bin_width": p99_bin_width,
    }


def estimate_percentiles_from_bins(
    p99_counts: Dict[int, List[int]],
    sample_sizes: Dict[int, int],
) -> Dict[int, Optional[float]]:
    """
    基于细粒度分箱估计每个速度等级的 P99.5。
    """
    percentile_speeds = {}
    p99_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / P99_BIN_COUNT

    for speed_class in sorted(sample_sizes):
        counts = p99_counts[speed_class]
        total = sample_sizes[speed_class]

        if total <= 0:
            percentile_speeds[speed_class] = None
            continue

        target = total * PERCENTILE
        cumulative = 0
        percentile_speed = HIST_MAX_SPEED

        for index, count in enumerate(counts):
            if count <= 0:
                continue

            next_cumulative = cumulative + count

            if target <= next_cumulative:
                left_edge = HIST_MIN_SPEED + index * p99_bin_width
                inside_ratio = (target - cumulative) / count
                percentile_speed = left_edge + inside_ratio * p99_bin_width
                break

            cumulative = next_cumulative

        percentile_speeds[speed_class] = percentile_speed

    return percentile_speeds


# ==========================
# CSV 输出
# ==========================
def export_percentile_csv(
    output_path: Path,
    percentile_speeds: Dict[int, Optional[float]],
    sample_sizes: Dict[int, int],
):
    """
    导出 P99.5 估计结果 CSV。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p99_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / P99_BIN_COUNT

    rows = []

    for speed_class in sorted(sample_sizes):
        percentile_speed = percentile_speeds[speed_class]

        rows.append(
            {
                "速度等级": speed_class,
                "第99.5百分位速度": ""
                if percentile_speed is None
                else f"{percentile_speed:.6f}",
                "样本数": sample_sizes[speed_class],
                "百分位": PERCENTILE,
                "说明": f"基于{p99_bin_width:.6f}速度分箱估计",
            }
        )

    pl.DataFrame(rows).write_csv(output_path)


# ==========================
# 绘图
# ==========================
def plot_histograms_with_p99(
    histogram_data: Dict,
    percentile_speeds: Dict[int, Optional[float]],
    output_path: Path,
):
    """
    使用 Seaborn 风格绘制各速度等级平均速度百分比分布直方图，并标出 P99.5。
    """
    plot_counts = histogram_data["plot_counts"]
    sample_sizes = histogram_data["sample_sizes"]
    bin_width = histogram_data["plot_bin_width"]

    speed_classes = sorted(plot_counts)
    row_count, col_count = get_grid_shape(len(speed_classes))

    palette = sns.color_palette("Set2", n_colors=max(len(speed_classes), 8))

    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(5.8 * col_count, 4.7 * row_count),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    left_edges = [
        HIST_MIN_SPEED + index * bin_width
        for index in range(PLOT_BIN_COUNT)
    ]

    x_ticks = list(range(0, int(HIST_MAX_SPEED) + 1, 20))

    for index, speed_class in enumerate(speed_classes):
        ax = axes_list[index]

        counts = plot_counts[speed_class]
        total = sample_sizes[speed_class]

        percentages = [
            count * 100.0 / total if total else 0.0
            for count in counts
        ]

        color = palette[index % len(palette)]

        ax.bar(
            left_edges,
            percentages,
            width=bin_width * 0.92,
            align="edge",
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )

        if SHOW_BAR_LABELS:
            max_percent = max(percentages) if percentages else 0
            label_offset = max_percent * 0.012 if max_percent else 0

            for left_edge, percentage in zip(left_edges, percentages):
                if percentage <= 0:
                    continue

                ax.text(
                    left_edge + bin_width / 2,
                    percentage + label_offset,
                    format_percent_label(percentage),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        percentile_speed = percentile_speeds[speed_class]

        if percentile_speed is not None:
            ax.axvline(
                percentile_speed,
                color="black",
                linestyle="--",
                linewidth=1.7,
                alpha=0.95,
            )

            ax.text(
                0.02,
                0.95,
                f"P99.5：{percentile_speed:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "black",
                    "linewidth": 0.8,
                },
            )

        ax.text(
            0.98,
            0.95,
            f"样本数：{sample_sizes[speed_class]:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "none",
            },
        )

        ax.set_title(f"速度等级 {speed_class} 速度频率直方图")
        ax.set_xlabel("平均速度")
        ax.set_ylabel("百分比")
        ax.set_xlim(HIST_MIN_SPEED, HIST_MAX_SPEED)
        ax.set_xticks(x_ticks)

        # 即使 sharex=True，也强制每一行都显示 X 轴刻度值
        ax.tick_params(
            axis="x",
            which="both",
            labelbottom=True,
            bottom=True,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.45,
        )

        sns.despine(ax=ax, top=True, right=True)

    for ax in axes_list[len(speed_classes):]:
        ax.axis("off")

    fig.suptitle(
        "各速度等级平均速度分布直方图（百分比，含 P99.5 垂直线）",
        fontsize=18,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.965])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ==========================
# 主流程
# ==========================
def main():
    configure_plot_style()

    chunk_files = check_input_files()

    print(f"发现 {len(chunk_files)} 个 Parquet 数据分片文件")
    print(f"数据目录：{CHUNK_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")

    print("\n正在构建 Polars 惰性查询...")
    base_lf = build_base_lazy_frame()

    print("正在一次性聚合 P99.5 细粒度速度分箱...")
    fine_bin_frame = aggregate_fine_histogram_bins(base_lf)

    print(f"细粒度分箱聚合结果行数：{fine_bin_frame.height}")

    print("正在从细粒度分箱推导绘图分布和样本数...")
    histogram_data = build_histogram_data_from_fine_bins(fine_bin_frame)

    print("正在估计各速度等级的 P99.5...")
    percentile_speeds = estimate_percentiles_from_bins(
        p99_counts=histogram_data["p99_counts"],
        sample_sizes=histogram_data["sample_sizes"],
    )

    print("正在生成直方图图片...")
    plot_histograms_with_p99(
        histogram_data=histogram_data,
        percentile_speeds=percentile_speeds,
        output_path=OUTPUT_IMAGE_PATH,
    )

    print("正在导出 P99.5 结果 CSV...")
    export_percentile_csv(
        output_path=OUTPUT_CSV_PATH,
        percentile_speeds=percentile_speeds,
        sample_sizes=histogram_data["sample_sizes"],
    )

    print("\n处理完成")
    print(f"输出图片：{OUTPUT_IMAGE_PATH}")
    print(f"输出 CSV：{OUTPUT_CSV_PATH}")

    print("\n各速度等级统计：")

    for speed_class in sorted(histogram_data["sample_sizes"]):
        percentile_speed = percentile_speeds[speed_class]
        speed_text = "N/A" if percentile_speed is None else f"{percentile_speed:.6f}"
        sample_count = histogram_data["sample_sizes"][speed_class]

        print(
            f"速度等级 {speed_class}：P99.5={speed_text}，样本数={sample_count:,}"
        )


if __name__ == "__main__":
    main()
    
