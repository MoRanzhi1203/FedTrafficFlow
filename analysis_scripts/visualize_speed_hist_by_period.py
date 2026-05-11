from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl


BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_SUBDIR = OUTPUT_DIR / "speed_histograms_by_period_by_class"

PERIODS = ["凌晨", "上午", "中午", "下午", "傍晚", "晚上"]

BIN_COUNT = 30
HIST_MIN_SPEED = 0.0
HIST_MAX_SPEED = 120.0

PERIOD_COLORS = {
    "凌晨": "#4C78A8",
    "上午": "#F58518",
    "中午": "#54A24B",
    "下午": "#E45756",
    "傍晚": "#72B7B2",
    "晚上": "#B279A2",
}


def configure_fonts():
    """
    配置中文字体，避免图表标题和标签中文乱码。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def format_count_label(value: int) -> str:
    """
    将频数格式化为更短的显示形式。
    """
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def create_empty_period_counts():
    """
    为每个时段创建固定长度的分箱计数列表。
    """
    return {period: [0] * BIN_COUNT for period in PERIODS}


def get_period_expr():
    """
    根据 slot 生成中文时段表达式。
    """
    return (
        pl.when(pl.col("slot").is_between(0, 19, closed="both"))
        .then(pl.lit("凌晨"))
        .when(pl.col("slot").is_between(20, 39, closed="both"))
        .then(pl.lit("上午"))
        .when(pl.col("slot").is_between(40, 53, closed="both"))
        .then(pl.lit("中午"))
        .when(pl.col("slot").is_between(54, 67, closed="both"))
        .then(pl.lit("下午"))
        .when(pl.col("slot").is_between(68, 79, closed="both"))
        .then(pl.lit("傍晚"))
        .otherwise(pl.lit("晚上"))
    )


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    使用 Polars 流式引擎执行 LazyFrame。
    """
    return lazy_frame.collect(engine="streaming")


def build_histogram_data(chunk_files):
    """
    构建直方图数据。

    优化点：
    1. 使用 Polars 一次性懒加载所有 speed_chunk_*.csv；
    2. 不再逐个 chunk read_csv；
    3. 不再对每个 chunk 的聚合结果做 Python 层合并；
    4. 只读取需要的三列：时间段、平均速度、速度等级；
    5. 在 Polars 内部完成类型转换、过滤、分箱、分组计数。
    """
    bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / BIN_COUNT

    if bin_width <= 0:
        raise ValueError("直方图分箱宽度无效，请检查 HIST_MIN_SPEED、HIST_MAX_SPEED 和 BIN_COUNT。")

    csv_pattern = str(CHUNK_DIR / "speed_chunk_*.csv")

    print("正在使用 Polars 构建直方图聚合查询...")

    query = (
        pl.scan_csv(
            csv_pattern,
            glob=True,
            infer_schema_length=0,
            ignore_errors=True,
        )
        .select(["时间段", "平均速度", "速度等级"])
        .with_columns(
            [
                pl.col("时间段")
                .cast(pl.Utf8)
                .str.strip_chars()
                .cast(pl.Int64, strict=False)
                .alias("时间段"),
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
            pl.col("时间段").is_not_null()
            & pl.col("平均速度").is_not_null()
            & pl.col("速度等级").is_not_null()
        )
        .with_columns(
            [
                (pl.col("时间段") % 96).alias("slot"),
                pl.col("平均速度")
                .clip(HIST_MIN_SPEED, HIST_MAX_SPEED - 1e-9)
                .alias("平均速度"),
            ]
        )
        .with_columns(
            get_period_expr().alias("时段")
        )
        .with_columns(
            (
                ((pl.col("平均速度") - HIST_MIN_SPEED) / bin_width)
                .floor()
                .clip(0, BIN_COUNT - 1)
                .cast(pl.Int64)
            ).alias("bin_index")
        )
        .group_by(["速度等级", "时段", "bin_index"])
        .agg(
            pl.len().alias("频数")
        )
        .sort(["速度等级", "时段", "bin_index"])
    )

    print("正在执行直方图聚合...")

    aggregated = collect_streaming(query)

    counts_by_class = defaultdict(create_empty_period_counts)

    for row in aggregated.iter_rows(named=True):
        speed_class = row["速度等级"]
        period = row["时段"]
        bin_index = row["bin_index"]
        count = row["频数"]

        counts_by_class[speed_class][period][bin_index] = count

    sample_sizes_by_class = {
        speed_class: {
            period: sum(counts_by_period[period])
            for period in PERIODS
        }
        for speed_class, counts_by_period in counts_by_class.items()
    }

    return {
        "speed_min": HIST_MIN_SPEED,
        "speed_max": HIST_MAX_SPEED,
        "bin_width": bin_width,
        "counts_by_class": dict(counts_by_class),
        "sample_sizes_by_class": sample_sizes_by_class,
    }


def plot_histograms_for_class(
    histogram_data,
    speed_class: int,
    output_path: Path,
    left_edges,
):
    """
    为单个速度等级绘制 6 个时段的平均速度频率直方图。
    """
    speed_min = histogram_data["speed_min"]
    speed_max = histogram_data["speed_max"]
    bin_width = histogram_data["bin_width"]

    counts_by_period = histogram_data["counts_by_class"][speed_class]
    sample_sizes = histogram_data["sample_sizes_by_class"][speed_class]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        sharex=True,
        sharey=True,
    )

    for ax, period in zip(axes.flat, PERIODS):
        counts = counts_by_period.get(period, [0] * BIN_COUNT)

        bars = ax.bar(
            left_edges,
            counts,
            width=bin_width,
            align="edge",
            color=PERIOD_COLORS[period],
            edgecolor="white",
            alpha=0.9,
        )

        max_count = max(counts) if counts else 0
        label_offset = max_count * 0.015 if max_count else 0

        for bar, count in zip(bars, counts):
            if count <= 0:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count + label_offset,
                format_count_label(count),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        ax.set_title(f"{period} 速度频率直方图")
        ax.set_xlabel("平均速度")
        ax.set_ylabel("频数")
        ax.set_xlim(speed_min, speed_max)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        ax.text(
            0.98,
            0.95,
            f"样本数：{sample_sizes.get(period, 0):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "none",
            },
        )

    fig.suptitle(
        f"速度等级 {speed_class} 各时段平均速度频率直方图",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_histograms_by_class(histogram_data, output_dir: Path):
    """
    按速度等级批量绘图。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    speed_min = histogram_data["speed_min"]
    bin_width = histogram_data["bin_width"]

    left_edges = [
        speed_min + index * bin_width
        for index in range(BIN_COUNT)
    ]

    output_paths = []

    speed_classes = sorted(histogram_data["counts_by_class"])

    print(f"正在生成 {len(speed_classes)} 张速度等级直方图...")

    for speed_class in speed_classes:
        output_path = output_dir / f"speed_histograms_by_period_class_{speed_class}.png"

        plot_histograms_for_class(
            histogram_data=histogram_data,
            speed_class=speed_class,
            output_path=output_path,
            left_edges=left_edges,
        )

        output_paths.append(output_path)

    return output_paths


def main():
    configure_fonts()

    chunk_files = sorted(CHUNK_DIR.glob("speed_chunk_*.csv"))

    if not chunk_files:
        raise FileNotFoundError(f"未在目录中找到数据分片文件：{CHUNK_DIR}")

    print(f"发现 {len(chunk_files)} 个数据分片文件")

    histogram_data = build_histogram_data(chunk_files)
    output_paths = plot_histograms_by_class(histogram_data, OUTPUT_SUBDIR)

    print("直方图生成完成")
    print(f"输出目录：{OUTPUT_SUBDIR}")

    for output_path in output_paths:
        print(f"输出文件：{output_path}")

    for speed_class in sorted(histogram_data["sample_sizes_by_class"]):
        total_samples = sum(
            histogram_data["sample_sizes_by_class"][speed_class].values()
        )
        print(f"速度等级 {speed_class}：{total_samples:,} 条样本")


if __name__ == "__main__":
    main()