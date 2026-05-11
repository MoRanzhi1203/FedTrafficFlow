from datetime import date, timedelta
from pathlib import Path

import polars as pl


BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis"

START_DATE = date(2000, 4, 1)

PERIOD_ORDER = {
    "凌晨": 1,
    "上午": 2,
    "中午": 3,
    "下午": 4,
    "傍晚": 5,
    "晚上": 6,
}


def collect_streaming(lazy_frame):
    """
    使用 Polars 流式引擎执行 LazyFrame。
    """
    return lazy_frame.collect(engine="streaming")


def build_lazy_frame(chunk_files):
    """
    构建统一的 Polars LazyFrame。

    优化点：
    1. 不在每个 chunk 上单独 collect；
    2. 每个文件只构建 lazy scan；
    3. 提前 select 必要列，减少 CSV 解析压力；
    4. 统一做类型转换、空值过滤和 slot 计算。
    """
    frames = []

    for chunk_file in chunk_files:
        chunk_id = int(chunk_file.stem.split("_")[-1])
        day_label = (START_DATE + timedelta(days=chunk_id)).strftime("%m-%d")

        lf = (
            pl.scan_csv(
                chunk_file,
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
                    pl.lit(day_label).alias("日期"),
                ]
            )
            .filter(
                pl.col("时间段").is_not_null()
                & pl.col("平均速度").is_not_null()
                & pl.col("速度等级").is_not_null()
            )
            .with_columns(
                (pl.col("时间段") % 96).alias("slot")
            )
        )

        frames.append(lf)

    return pl.concat(frames, how="vertical")


def add_period_column(lf):
    """
    根据 slot 添加中文时段列。
    """
    period_expr = (
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

    return lf.with_columns(period_expr.alias("时段"))


def summarize_speed_stats(chunk_files):
    """
    统计速度等级的总体速度分布，以及每日各时段速度等级分布。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"发现 {len(chunk_files)} 个数据分片文件")
    print("正在构建惰性查询...")

    lf = build_lazy_frame(chunk_files)
    lf = add_period_column(lf)

    common_aggs = [
        pl.col("平均速度").min().alias("最小平均速度"),
        pl.col("平均速度").max().alias("最大平均速度"),
        pl.col("平均速度").mean().alias("平均平均速度"),
        pl.len().alias("记录数"),
    ]

    print("正在统计各速度等级的总体速度分布...")

    overall_query = (
        lf.group_by("速度等级")
        .agg(common_aggs)
        .sort("速度等级")
    )

    overall_result = collect_streaming(overall_query)

    print("正在统计每日各时段的速度等级分布...")

    daily_query = (
        lf.group_by(["日期", "时段", "速度等级"])
        .agg(common_aggs)
        .with_columns(
            pl.col("时段")
            .replace(PERIOD_ORDER)
            .cast(pl.Int64)
            .alias("时段排序")
        )
        .sort(["日期", "时段排序", "速度等级"])
        .drop("时段排序")
    )

    daily_result = collect_streaming(daily_query)

    overall_path = OUTPUT_DIR / "speed_class_overall_stats.csv"
    daily_path = OUTPUT_DIR / "speed_class_daily_period_stats.csv"

    overall_result.write_csv(overall_path)
    daily_result.write_csv(daily_path)

    print("统计完成")
    print(f"总体统计行数：{overall_result.height}")
    print(f"每日分时段统计行数：{daily_result.height}")
    print(f"总体统计输出文件：{overall_path}")
    print(f"每日分时段统计输出文件：{daily_path}")


def main():
    chunk_files = sorted(CHUNK_DIR.glob("speed_chunk_*.csv"))

    if not chunk_files:
        raise FileNotFoundError(f"未在目录中找到数据分片文件：{CHUNK_DIR}")

    summarize_speed_stats(chunk_files)


if __name__ == "__main__":
    main()