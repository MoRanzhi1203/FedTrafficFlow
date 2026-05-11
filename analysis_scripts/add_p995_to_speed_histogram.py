from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"

OUTPUT_IMAGE_PATH = OUTPUT_DIR / "speed_histograms_by_class_with_p995_percent.png"
OUTPUT_CSV_PATH = OUTPUT_DIR / "speed_histograms_by_class_p995.csv"

HIST_MIN_SPEED = 0.0
HIST_MAX_SPEED = 120.0

PLOT_BIN_COUNT = 30
P99_BIN_COUNT = 2400
PERCENTILE = 0.995


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
        },
    )


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    使用 Polars 流式引擎执行 LazyFrame。
    """
    return lazy_frame.collect(engine="streaming")


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


def build_base_lazy_frame() -> pl.LazyFrame:
    """
    一次性懒加载所有 speed_chunk_*.csv，并完成基础清洗。
    """
    csv_pattern = str(CHUNK_DIR / "speed_chunk_*.csv")

    return (
        pl.scan_csv(
            csv_pattern,
            glob=True,
            infer_schema_length=0,
            ignore_errors=True,
        )
        .select(["平均速度", "速度等级"])
        .with_columns(
            [
                pl.col("平均速度")
                .cast(pl.Utf8)
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .alias("平均速度"),
                pl.col("速度等级")
                .cast(pl.Utf8)
                .str.strip_chars()
                .cast(pl.Int64, strict=False)
                .alias("速度等级"),
            ]
        )
        .filter(
            pl.col("平均速度").is_not_null()
            & pl.col("速度等级").is_not_null()
        )
        .with_columns(
            pl.col("平均速度")
            .clip(HIST_MIN_SPEED, HIST_MAX_SPEED - 1e-9)
            .alias("平均速度")
        )
    )


def aggregate_histogram_bins(
    base_lf: pl.LazyFrame,
    bin_count: int,
    count_column_name: str,
) -> pl.DataFrame:
    """
    按速度等级和分箱编号聚合频数。
    """
    bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / bin_count

    query = (
        base_lf.with_columns(
            (
                ((pl.col("平均速度") - HIST_MIN_SPEED) / bin_width)
                .floor()
                .clip(0, bin_count - 1)
                .cast(pl.Int64)
            ).alias("bin_index")
        )
        .group_by(["速度等级", "bin_index"])
        .agg(pl.len().alias(count_column_name))
        .sort(["速度等级", "bin_index"])
    )

    return collect_streaming(query)


def aggregate_sample_sizes(base_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    统计每个速度等级的样本数。
    """
    query = (
        base_lf.group_by("速度等级")
        .agg(pl.len().alias("样本数"))
        .sort("速度等级")
    )

    return collect_streaming(query)


def build_count_matrix(
    bin_frame: pl.DataFrame,
    speed_classes: List[int],
    bin_count: int,
    count_column_name: str,
) -> Dict[int, List[int]]:
    """
    将 Polars 聚合结果转换为 {速度等级: 分箱频数列表}。
    """
    counts_by_class = {
        speed_class: [0] * bin_count
        for speed_class in speed_classes
    }

    for row in bin_frame.iter_rows(named=True):
        speed_class = row["速度等级"]
        bin_index = row["bin_index"]
        count = row[count_column_name]
        counts_by_class[speed_class][bin_index] = count

    return counts_by_class


def estimate_percentiles_from_bins(
    p99_counts: Dict[int, List[int]],
    sample_sizes: Dict[int, int],
) -> Dict[int, Optional[float]]:
    """
    基于细粒度分箱估计每个速度等级的 P99.5。
    """
    percentile_speeds = {}
    bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / P99_BIN_COUNT

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
                left_edge = HIST_MIN_SPEED + index * bin_width
                inside_ratio = (target - cumulative) / count
                percentile_speed = left_edge + inside_ratio * bin_width
                break

            cumulative = next_cumulative

        percentile_speeds[speed_class] = percentile_speed

    return percentile_speeds


def build_histogram_data() -> Dict:
    """
    构建绘图和 P99.5 估计所需数据。
    """
    plot_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / PLOT_BIN_COUNT
    p99_bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / P99_BIN_COUNT

    if plot_bin_width <= 0 or p99_bin_width <= 0:
        raise ValueError("速度分箱宽度无效，请检查速度范围和分箱数量。")

    print("正在构建 Polars 惰性查询...")

    base_lf = build_base_lazy_frame()

    print("正在统计各速度等级样本数...")
    sample_frame = aggregate_sample_sizes(base_lf)

    sample_sizes = {
        row["速度等级"]: row["样本数"]
        for row in sample_frame.iter_rows(named=True)
    }

    speed_classes = sorted(sample_sizes)

    print("正在聚合绘图用速度直方图分箱...")
    plot_bin_frame = aggregate_histogram_bins(
        base_lf=base_lf,
        bin_count=PLOT_BIN_COUNT,
        count_column_name="频数",
    )

    print("正在聚合 P99.5 估计用细粒度分箱...")
    p99_bin_frame = aggregate_histogram_bins(
        base_lf=base_lf,
        bin_count=P99_BIN_COUNT,
        count_column_name="频数",
    )

    plot_counts = build_count_matrix(
        bin_frame=plot_bin_frame,
        speed_classes=speed_classes,
        bin_count=PLOT_BIN_COUNT,
        count_column_name="频数",
    )

    p99_counts = build_count_matrix(
        bin_frame=p99_bin_frame,
        speed_classes=speed_classes,
        bin_count=P99_BIN_COUNT,
        count_column_name="频数",
    )

    return {
        "plot_counts": plot_counts,
        "p99_counts": p99_counts,
        "sample_sizes": sample_sizes,
        "plot_bin_width": plot_bin_width,
        "p99_bin_width": p99_bin_width,
    }


def export_percentile_csv(
    output_path: Path,
    percentile_speeds: Dict[int, Optional[float]],
    sample_sizes: Dict[int, int],
):
    """
    导出 P99.5 估计结果 CSV。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                "说明": "基于0.05速度分箱估计",
            }
        )

    pl.DataFrame(rows).write_csv(output_path)


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

        # 关键：即使 sharex=True，也强制每一行都显示 X 轴刻度值
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
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_plot_style()

    chunk_files = sorted(CHUNK_DIR.glob("speed_chunk_*.csv"))

    if not chunk_files:
        raise FileNotFoundError(f"未在目录中找到数据分片文件：{CHUNK_DIR}")

    print(f"发现 {len(chunk_files)} 个数据分片文件")

    histogram_data = build_histogram_data()

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

    print("处理完成")
    print(f"输出图片：{OUTPUT_IMAGE_PATH}")
    print(f"输出 CSV：{OUTPUT_CSV_PATH}")

    for speed_class in sorted(histogram_data["sample_sizes"]):
        percentile_speed = percentile_speeds[speed_class]
        speed_text = "N/A" if percentile_speed is None else f"{percentile_speed:.6f}"
        sample_count = histogram_data["sample_sizes"][speed_class]

        print(
            f"速度等级 {speed_class}：P99.5={speed_text}，样本数={sample_count:,}"
        )


if __name__ == "__main__":
    main()