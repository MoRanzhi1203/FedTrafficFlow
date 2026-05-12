# ==========================
# visualize_speed_hist_by_period.py
# ==========================
# -*- coding: utf-8 -*-

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl


# ==========================
# 路径配置
# ==========================
BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_SUBDIR = OUTPUT_DIR / "speed_histograms_by_period_by_class"

# 保存聚合后的直方图计数表，方便复查
HISTOGRAM_TABLE_PATH = OUTPUT_DIR / "speed_histogram_counts_by_period_by_class.csv"


# ==========================
# 直方图配置
# ==========================
PERIODS = ["凌晨", "上午", "中午", "下午", "傍晚", "晚上"]

PERIOD_ID_TO_LABEL = {
    1: "凌晨",
    2: "上午",
    3: "中午",
    4: "下午",
    5: "傍晚",
    6: "晚上",
}

BIN_COUNT = 30
HIST_MIN_SPEED = 0.0
HIST_MAX_SPEED = 120.0

# True：给每根柱子标注频数，图片信息更完整，但绘图稍慢
# False：不标注频数，绘图更快
SHOW_BAR_LABELS = True

# 图片分辨率
FIG_DPI = 200

PERIOD_COLORS = {
    "凌晨": "#4C78A8",
    "上午": "#F58518",
    "中午": "#54A24B",
    "下午": "#E45756",
    "傍晚": "#72B7B2",
    "晚上": "#B279A2",
}


# ==========================
# 基础工具函数
# ==========================
def configure_fonts():
    """
    配置中文字体，避免图表中文乱码。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    使用 Polars 流式引擎执行 LazyFrame。
    兼容新旧版本 Polars。
    """
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


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
    return {
        period: [0] * BIN_COUNT
        for period in PERIODS
    }


def get_period_id_expr() -> pl.Expr:
    """
    根据一天内的 slot 生成时段编号。

    1 凌晨：slot 0 - 19
    2 上午：slot 20 - 39
    3 中午：slot 40 - 53
    4 下午：slot 54 - 67
    5 傍晚：slot 68 - 79
    6 晚上：slot 80 - 95
    """
    return (
        pl.when(pl.col("slot").is_between(0, 19, closed="both"))
        .then(pl.lit(1))
        .when(pl.col("slot").is_between(20, 39, closed="both"))
        .then(pl.lit(2))
        .when(pl.col("slot").is_between(40, 53, closed="both"))
        .then(pl.lit(3))
        .when(pl.col("slot").is_between(54, 67, closed="both"))
        .then(pl.lit(4))
        .when(pl.col("slot").is_between(68, 79, closed="both"))
        .then(pl.lit(5))
        .otherwise(pl.lit(6))
        .cast(pl.Int8)
    )


def get_period_label_expr() -> pl.Expr:
    """
    将时段编号转换为中文时段。

    使用 Polars 原生 replace_strict，避免 map_elements 的性能警告。
    """
    return (
        pl.col("时段编号")
        .cast(pl.Int64)
        .replace_strict(PERIOD_ID_TO_LABEL)
        .cast(pl.Utf8)
    )


def check_input_files():
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
# 构建直方图聚合数据
# ==========================
def build_histogram_aggregation() -> pl.DataFrame:
    """
    构建直方图聚合数据。

    优化点：
    1. 直接 scan_parquet，不再 scan_csv；
    2. 一次性扫描所有 speed_chunk_*.parquet；
    3. 只读取必要列：时间段、平均速度、速度等级；
    4. 大表阶段使用时段编号分组，不使用中文字符串分组；
    5. 在 Polars 内部完成过滤、分箱、分组计数；
    6. 只 collect 聚合后的计数表，不 collect 原始大表。
    """
    bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / BIN_COUNT

    if bin_width <= 0:
        raise ValueError(
            "直方图分箱宽度无效，请检查 HIST_MIN_SPEED、HIST_MAX_SPEED 和 BIN_COUNT。"
        )

    parquet_pattern = (CHUNK_DIR / "speed_chunk_*.parquet").as_posix()

    print("正在使用 Polars 扫描 Parquet 并构建直方图聚合查询...")

    query = (
        pl.scan_parquet(parquet_pattern)
        .select([
            pl.col("时间段").cast(pl.Int64, strict=False),
            pl.col("平均速度").cast(pl.Float64, strict=False),
            pl.col("速度等级").cast(pl.Int64, strict=False),
        ])
        .filter(
            pl.col("时间段").is_not_null()
            & pl.col("平均速度").is_not_null()
            & pl.col("速度等级").is_not_null()
        )
        .with_columns([
            # 一天内第几个 15 分钟时间段：0 - 95
            (pl.col("时间段") % 96)
            .cast(pl.Int64)
            .alias("slot"),

            # 将异常速度限制到直方图区间内
            # 小于 HIST_MIN_SPEED 的归入第 0 箱
            # 大于等于 HIST_MAX_SPEED 的归入最后一箱
            pl.col("平均速度")
            .clip(HIST_MIN_SPEED, HIST_MAX_SPEED - 1e-9)
            .alias("速度裁剪值"),
        ])
        .with_columns([
            get_period_id_expr().alias("时段编号"),

            (
                ((pl.col("速度裁剪值") - HIST_MIN_SPEED) / bin_width)
                .floor()
                .clip(0, BIN_COUNT - 1)
                .cast(pl.Int16)
            ).alias("bin_index"),
        ])
        .group_by(["速度等级", "时段编号", "bin_index"])
        .agg([
            pl.len().alias("频数")
        ])
        .sort(["速度等级", "时段编号", "bin_index"])
    )

    print("正在执行直方图聚合...")

    aggregated = collect_streaming(query)

    return aggregated


def save_histogram_aggregation(aggregated: pl.DataFrame):
    """
    保存聚合后的直方图计数表。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = (
        aggregated
        .with_columns([
            get_period_label_expr().alias("时段")
        ])
        .select([
            "速度等级",
            "时段编号",
            "时段",
            "bin_index",
            "频数",
        ])
        .sort(["速度等级", "时段编号", "bin_index"])
    )

    result.write_csv(HISTOGRAM_TABLE_PATH)

    print(f"直方图聚合计数表已保存：{HISTOGRAM_TABLE_PATH}")


def convert_aggregation_to_plot_data(aggregated: pl.DataFrame) -> dict:
    """
    将聚合后的 DataFrame 转换为绘图字典结构。

    聚合结果很小，这里使用 Python 字典处理即可。
    """
    bin_width = (HIST_MAX_SPEED - HIST_MIN_SPEED) / BIN_COUNT

    counts_by_class = defaultdict(create_empty_period_counts)

    for speed_class, period_id, bin_index, count in aggregated.iter_rows():
        period = PERIOD_ID_TO_LABEL[int(period_id)]
        counts_by_class[int(speed_class)][period][int(bin_index)] = int(count)

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


# ==========================
# 绘图函数
# ==========================
def plot_histograms_for_class(
    histogram_data: dict,
    speed_class: int,
    output_path: Path,
    left_edges: list,
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
            color=PERIOD_COLORS.get(period),
            edgecolor="white",
            alpha=0.9,
        )

        if SHOW_BAR_LABELS:
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
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_histograms_by_class(histogram_data: dict, output_dir: Path):
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


# ==========================
# 主流程
# ==========================
def main():
    configure_fonts()

    chunk_files = check_input_files()

    print(f"发现 {len(chunk_files)} 个 Parquet 数据分片文件")
    print(f"数据目录：{CHUNK_DIR}")
    print(f"输出目录：{OUTPUT_SUBDIR}")

    aggregated = build_histogram_aggregation()

    print(f"直方图聚合结果行数：{aggregated.height}")

    save_histogram_aggregation(aggregated)

    histogram_data = convert_aggregation_to_plot_data(aggregated)

    output_paths = plot_histograms_by_class(
        histogram_data=histogram_data,
        output_dir=OUTPUT_SUBDIR,
    )

    print("\n直方图生成完成")
    print(f"输出目录：{OUTPUT_SUBDIR}")

    for output_path in output_paths:
        print(f"输出文件：{output_path}")

    print("\n各速度等级样本数：")

    for speed_class in sorted(histogram_data["sample_sizes_by_class"]):
        total_samples = sum(
            histogram_data["sample_sizes_by_class"][speed_class].values()
        )
        print(f"速度等级 {speed_class}：{total_samples:,} 条样本")


if __name__ == "__main__":
    main()