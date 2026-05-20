# ==========================
# summarize_speed_stats.py
# ==========================
# -*- coding: utf-8 -*-

"""汇总处理后速度分片中的总体统计和分时段速度统计结果。"""

from datetime import date, timedelta
from pathlib import Path

import polars as pl


BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"

START_DATE = date(2000, 4, 1)

# 一个自然日 = 96 个 15 分钟时间段
SLOTS_PER_DAY = 96

PERIOD_LABEL = {
    1: "凌晨",
    2: "上午",
    3: "中午",
    4: "下午",
    5: "傍晚",
    6: "晚上",
}


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    使用 Polars 流式引擎执行 LazyFrame。
    兼容新旧版本 Polars。
    """
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def build_lazy_frame() -> pl.LazyFrame:
    """
    构建统一 LazyFrame。

    优化点：
    1. 直接 scan_parquet，不再 scan_csv；
    2. 一次性扫描所有 parquet 分片；
    3. 只读取必要列：时间段、平均速度、速度等级；
    4. 根据 时间段 // 96 推日期；
    5. 时段先用数字编码，聚合后再转中文，减少字符串处理开销。
    """
    parquet_pattern = (CHUNK_DIR / "speed_chunk_*.parquet").as_posix()

    lf = (
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
            # 第几天：0 表示 START_DATE 当天
            (pl.col("时间段") // SLOTS_PER_DAY)
            .cast(pl.Int64)
            .alias("day_index"),

            # 一天内第几个 15 分钟时间段：0 - 95
            (pl.col("时间段") % SLOTS_PER_DAY)
            .cast(pl.Int64)
            .alias("slot"),
        ])
        .with_columns([
            # 用数字表示时段，避免在大表中处理中文字符串
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
            .alias("时段排序")
        ])
    )

    return lf


def add_date_and_period_label(daily_result: pl.DataFrame) -> pl.DataFrame:
    """
    聚合完成后，再把 day_index、时段排序 转成 日期、中文时段。
    此时数据量已经很小，用 map_elements 不会造成明显性能压力。
    """
    return (
        daily_result
        .with_columns([
            pl.col("day_index")
            .map_elements(
                lambda x: (START_DATE + timedelta(days=int(x))).strftime("%m-%d"),
                return_dtype=pl.Utf8,
            )
            .alias("日期"),

            pl.col("时段排序")
            .map_elements(
                lambda x: PERIOD_LABEL[int(x)],
                return_dtype=pl.Utf8,
            )
            .alias("时段"),
        ])
        .sort(["day_index", "时段排序", "速度等级"])
        .select([
            "日期",
            "时段",
            "速度等级",
            "最小平均速度",
            "最大平均速度",
            "平均平均速度",
            "记录数",
        ])
    )


def summarize_speed_stats():
    """
    统计：
    1. 各速度等级的总体速度分布；
    2. 每日各时段的速度等级分布。

    关键优化：
    先只扫描原始 parquet 一次，得到 daily_internal_result；
    overall_result 再从 daily_internal_result 汇总得到，避免对大数据重复扫描两遍。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(CHUNK_DIR.glob("speed_chunk_*.parquet"))

    if not chunk_files:
        raise FileNotFoundError(
            f"未在目录中找到 Parquet 数据分片文件：{CHUNK_DIR}\n"
            f"请确认文件名类似：speed_chunk_000.parquet"
        )

    print(f"发现 {len(chunk_files)} 个 Parquet 数据分片文件")
    print(f"数据目录：{CHUNK_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")

    print("\n正在构建惰性查询...")

    lf = build_lazy_frame()

    print("正在统计每日各时段的速度等级分布...")

    # 先做每日分时段聚合
    # 额外保留 _平均速度总和，用于后续准确计算总体平均速度
    daily_internal_query = (
        lf.group_by(["day_index", "时段排序", "速度等级"])
        .agg([
            pl.col("平均速度").min().alias("最小平均速度"),
            pl.col("平均速度").max().alias("最大平均速度"),
            pl.col("平均速度").sum().alias("_平均速度总和"),
            pl.len().alias("记录数"),
        ])
        .with_columns([
            (
                pl.col("_平均速度总和") / pl.col("记录数")
            ).alias("平均平均速度")
        ])
    )

    daily_internal_result = collect_streaming(daily_internal_query)

    print("正在基于每日统计结果生成总体速度等级统计...")

    # 从 daily_internal_result 汇总总体统计
    # 这样不用再次扫描所有 parquet 原始数据
    overall_result = (
        daily_internal_result
        .lazy()
        .group_by("速度等级")
        .agg([
            pl.col("最小平均速度").min().alias("最小平均速度"),
            pl.col("最大平均速度").max().alias("最大平均速度"),
            pl.col("_平均速度总和").sum().alias("_平均速度总和"),
            pl.col("记录数").sum().alias("记录数"),
        ])
        .with_columns([
            (
                pl.col("_平均速度总和") / pl.col("记录数")
            ).alias("平均平均速度")
        ])
        .select([
            "速度等级",
            "最小平均速度",
            "最大平均速度",
            "平均平均速度",
            "记录数",
        ])
        .sort("速度等级")
        .collect()
    )

    daily_result = add_date_and_period_label(daily_internal_result)

    overall_path = OUTPUT_DIR / "speed_class_overall_stats.csv"
    daily_path = OUTPUT_DIR / "speed_class_daily_period_stats.csv"

    print("\n正在写出统计结果...")

    overall_result.write_csv(overall_path)
    daily_result.write_csv(daily_path)

    print("\n统计完成")
    print(f"总体统计行数：{overall_result.height}")
    print(f"每日分时段统计行数：{daily_result.height}")
    print(f"总体统计输出文件：{overall_path}")
    print(f"每日分时段统计输出文件：{daily_path}")


def main():
    summarize_speed_stats()


if __name__ == "__main__":
    main()
