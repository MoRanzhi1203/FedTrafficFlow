# ==========================
# compute_greenshields_density.py
# ==========================
# -*- coding: utf-8 -*-

from pathlib import Path

import polars as pl


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis" / "density_metrics_chunks"

PARAM_DIR = BASE_DIR / "data" / "params"
SPEED_CLASS_PARAM_PATH = PARAM_DIR / "speed_class_density_params.csv"
CAPACITY_PARAM_PATH = PARAM_DIR / "beijing_capacity_params.csv"


# ============================================================
# 2. 输出设置
# ============================================================

# 推荐 parquet：速度快、体积小、后续 Polars 读取快
# 如果必须输出 CSV，把 OUTPUT_FORMAT 改成 "csv"
OUTPUT_FORMAT = "parquet"

PARQUET_COMPRESSION = "lz4"


# ============================================================
# 3. 通用工具函数
# ============================================================

def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """
    使用 Polars 流式引擎执行 LazyFrame。
    兼容新旧版本 Polars。
    """
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def require_columns(df: pl.DataFrame, required_columns, file_path: Path):
    """
    检查参数表是否包含必要字段。
    """
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"文件缺少必要字段：{file_path}\n"
            f"缺失字段：{missing_columns}\n"
            f"当前字段：{df.columns}"
        )


def clean_float_col(col_name: str):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(col_name)
    )


def clean_int_col(col_name: str):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Int64, strict=False)
        .alias(col_name)
    )


def clean_str_col(col_name: str):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(col_name)
    )


# ============================================================
# 4. 读取参数表
# ============================================================

def load_capacity_param_frame() -> pl.DataFrame:
    """
    读取《北京市公路通行能力研究总报告》表3.2参数表。

    文件路径：
    data/params/beijing_capacity_params.csv
    """

    df = pl.read_csv(CAPACITY_PARAM_PATH)

    require_columns(
        df=df,
        required_columns=[
            "标准速度",
            "通行能力",
            "临界密度",
            "临界速度",
        ],
        file_path=CAPACITY_PARAM_PATH,
    )

    return (
        df
        .with_columns([
            clean_float_col("标准速度"),
            clean_float_col("通行能力"),
            clean_float_col("临界密度"),
            clean_float_col("临界速度"),
        ])
        .filter(
            pl.col("标准速度").is_not_null()
            & pl.col("通行能力").is_not_null()
            & pl.col("临界密度").is_not_null()
            & pl.col("临界速度").is_not_null()
        )
    )


def load_speed_param_frame() -> pl.DataFrame:
    """
    读取速度等级密度参数表。

    文件路径：
    data/params/speed_class_density_params.csv
    """

    df = pl.read_csv(SPEED_CLASS_PARAM_PATH)

    # 兼容前面 P99.5 输出表的字段名
    if "P99.5速度" not in df.columns and "第99.5百分位速度" in df.columns:
        df = df.rename({"第99.5百分位速度": "P99.5速度"})

    require_columns(
        df=df,
        required_columns=[
            "速度等级",
            "P99.5速度",
            "自由流速度",
            "每车道临界密度",
            "速度档位",
            "说明",
        ],
        file_path=SPEED_CLASS_PARAM_PATH,
    )

    return (
        df
        .with_columns([
            clean_int_col("速度等级"),
            clean_float_col("P99.5速度"),
            clean_float_col("自由流速度"),
            clean_float_col("每车道临界密度"),
            clean_str_col("速度档位"),
            clean_str_col("说明"),
        ])
        .filter(
            pl.col("速度等级").is_not_null()
            & pl.col("自由流速度").is_not_null()
            & pl.col("每车道临界密度").is_not_null()
            & (pl.col("自由流速度") > 0)
            & (pl.col("每车道临界密度") > 0)
        )
    )


def build_speed_param_frame() -> pl.DataFrame:
    """
    读取速度等级参数表，并与通行能力参数表按自由流速度匹配。

    输出字段包括：
    - 速度等级
    - P99.5速度
    - 自由流速度
    - 每车道临界密度
    - 速度档位
    - 说明
    - 通行能力
    - 临界密度
    - 临界速度
    """

    speed_param_frame = load_speed_param_frame()
    capacity_param_frame = load_capacity_param_frame()

    # 使用整数 key 匹配，避免 80 和 80.0 这类浮点匹配问题
    speed_param_frame = speed_param_frame.with_columns([
        pl.col("自由流速度")
        .round(0)
        .cast(pl.Int64, strict=False)
        .alias("_速度匹配键")
    ])

    capacity_param_frame = capacity_param_frame.with_columns([
        pl.col("标准速度")
        .round(0)
        .cast(pl.Int64, strict=False)
        .alias("_速度匹配键")
    ])

    joined = (
        speed_param_frame
        .join(
            capacity_param_frame.select([
                "_速度匹配键",
                "通行能力",
                "临界密度",
                "临界速度",
            ]),
            on="_速度匹配键",
            how="left",
        )
        .drop("_速度匹配键")
    )

    missing_capacity = joined.filter(pl.col("通行能力").is_null())

    if missing_capacity.height > 0:
        print("警告：以下速度等级没有匹配到通行能力参数：")
        print(
            missing_capacity.select([
                "速度等级",
                "自由流速度",
                "P99.5速度",
                "每车道临界密度",
            ])
        )

    return joined


# ============================================================
# 5. 输入分块读取与字段清洗
# ============================================================

def find_speed_chunk_files():
    """
    优先读取 parquet 分片。
    如果没有 parquet，才回退查找 csv。
    """

    parquet_files = sorted(INPUT_DIR.glob("speed_chunk_*.parquet"))

    if parquet_files:
        return parquet_files, "parquet"

    csv_files = sorted(INPUT_DIR.glob("speed_chunk_*.csv"))

    if csv_files:
        return csv_files, "csv"

    raise FileNotFoundError(
        f"未在目录中找到速度分块文件：{INPUT_DIR}\n"
        f"期望文件名类似：speed_chunk_000.parquet"
    )


def scan_speed_chunk(chunk_file: Path, input_format: str) -> pl.LazyFrame:
    """
    根据文件格式扫描速度分块。
    """

    if input_format == "parquet":
        return pl.scan_parquet(chunk_file.as_posix())

    if input_format == "csv":
        return pl.scan_csv(
            chunk_file.as_posix(),
            infer_schema_length=0,
            ignore_errors=True,
        )

    raise ValueError(f"不支持的输入格式：{input_format}")


def cast_metric_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    统一转换密度计算所需字段。

    输入分块文件中必须包含：
    - 路段ID
    - 时间段
    - 平均速度
    - 长度
    - 速度等级
    - 车道数
    """

    required_columns = [
        "路段ID",
        "时间段",
        "平均速度",
        "长度",
        "速度等级",
        "车道数",
    ]

    return (
        lf
        .select(required_columns)
        .with_columns([
            # 路段ID 只作为标识符输出，保留字符串更安全
            pl.col("路段ID")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("路段ID"),

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

            pl.col("长度")
            .cast(pl.Utf8)
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
            .alias("长度"),

            pl.col("速度等级")
            .cast(pl.Utf8)
            .str.strip_chars()
            .cast(pl.Int64, strict=False)
            .alias("速度等级"),

            pl.col("车道数")
            .cast(pl.Utf8)
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
            .alias("车道数"),
        ])
    )


# ============================================================
# 6. 密度与流量计算
# ============================================================

def build_density_metrics_query(
    chunk_file: Path,
    input_format: str,
    speed_param_frame: pl.DataFrame,
) -> pl.LazyFrame:
    """
    为单个速度分块文件构建 Greenshields 密度计算查询。

    数学模型：

    1. 速度非负化并截断：
       v = min(max(v_raw, 0), v_f)

    2. 每车道密度：
       d_lane = d_j_lane * (1 - v / v_f)

    3. 路段总密度：
       d_road = d_lane * lane_num

    4. 路段车辆数：
       N = d_road * L

    5. 小时流量：
       q = v * d_road

    6. 15分钟流量：
       Q_15 = q / 4

    注意：
    最终只输出一列 平均速度。
    这列已经是用于计算的速度，即 min(max(原始平均速度, 0), 自由流速度)。
    """

    speed_param_lf = speed_param_frame.lazy()

    raw_lf = scan_speed_chunk(
        chunk_file=chunk_file,
        input_format=input_format,
    )

    base_lf = (
        raw_lf
        .pipe(cast_metric_columns)

        # 只保留关键字段完整、物理含义合理的记录
        .filter(
            pl.col("路段ID").is_not_null()
            & pl.col("时间段").is_not_null()
            & pl.col("平均速度").is_not_null()
            & pl.col("长度").is_not_null()
            & pl.col("速度等级").is_not_null()
            & pl.col("车道数").is_not_null()
            & (pl.col("长度") > 0)
            & (pl.col("车道数") > 0)
        )

        # 按速度等级匹配自由流速度和临界密度参数
        .join(
            speed_param_lf,
            on="速度等级",
            how="left",
        )

        # 严格参数逻辑：没有匹配到参数的记录不参与计算
        .filter(
            pl.col("自由流速度").is_not_null()
            & pl.col("每车道临界密度").is_not_null()
            & pl.col("通行能力").is_not_null()
            & (pl.col("自由流速度") > 0)
            & (pl.col("每车道临界密度") > 0)
        )

        .with_columns([
            # 长度字段已经是 km，因此不做单位换算
            pl.col("长度").alias("路段长度_km"),

            # 只保留一列 平均速度：
            # 平均速度 = min(max(原始平均速度, 0), 自由流速度)
            pl.min_horizontal(
                pl.max_horizontal(
                    pl.col("平均速度"),
                    pl.lit(0.0),
                ),
                pl.col("自由流速度"),
            ).alias("平均速度"),
        ])

        .with_columns([
            # 每车道密度：
            # d_lane = d_j_lane * (1 - v / v_f)
            pl.max_horizontal(
                (
                    pl.col("每车道临界密度")
                    * (
                        1.0
                        - pl.col("平均速度")
                        / pl.col("自由流速度")
                    )
                ),
                pl.lit(0.0),
            ).alias("每车道密度_pcu_per_km"),

            # 路段通行能力：
            # C_road = C_lane * lane_num
            (
                pl.col("通行能力") * pl.col("车道数")
            ).alias("路段通行能力_pcu_per_h"),
        ])

        .with_columns([
            # 路段总密度：
            # d_road = d_lane * lane_num
            (
                pl.col("每车道密度_pcu_per_km")
                * pl.col("车道数")
            ).alias("密度_pcu_per_km"),
        ])

        .with_columns([
            # 路段车辆数：
            # N = d_road * L
            (
                pl.col("密度_pcu_per_km")
                * pl.col("路段长度_km")
            ).alias("车辆数估计"),

            # 小时流量：
            # q = v * d_road
            (
                pl.col("平均速度")
                * pl.col("密度_pcu_per_km")
            ).alias("流量_pcu_per_h"),
        ])

        .with_columns([
            # 15分钟流量：
            # Q_15 = q / 4
            (
                pl.col("流量_pcu_per_h") / 4.0
            ).alias("15分钟流量_pcu"),
        ])
    )

    return (
        base_lf
        .select([
            "路段ID",
            "时间段",
            "速度等级",
            "平均速度",
            "密度_pcu_per_km",
            "车辆数估计",
            "流量_pcu_per_h",
            "15分钟流量_pcu",
        ])
        # 结果文件按时间段升序输出，便于核查和后续按时段分析
        .sort(["时间段", "路段ID", "速度等级"])
    )


# ============================================================
# 7. 结果写出
# ============================================================

def build_output_path(chunk_file: Path) -> Path:
    """
    构造输出文件路径。

    输入：
    speed_chunk_000.parquet

    输出：
    density_chunk_000.parquet
    """

    chunk_suffix = chunk_file.stem.replace("speed_", "")

    if OUTPUT_FORMAT == "parquet":
        return OUTPUT_DIR / f"density_{chunk_suffix}.parquet"

    if OUTPUT_FORMAT == "csv":
        return OUTPUT_DIR / f"density_{chunk_suffix}.csv"

    raise ValueError(f"不支持的输出格式：{OUTPUT_FORMAT}")


def write_lazy_result(lf: pl.LazyFrame, output_path: Path):
    """
    写出 LazyFrame。

    parquet 优先使用 sink_parquet，避免 collect 后占用大量内存。
    csv 使用 collect_streaming 后写出。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FORMAT == "parquet":
        try:
            lf.sink_parquet(
                output_path.as_posix(),
                compression=PARQUET_COMPRESSION,
                statistics=False,
            )
            print(f"输出文件：{output_path}")
            return
        except TypeError:
            # 兼容旧版本 Polars
            result = collect_streaming(lf)
            result.write_parquet(
                output_path,
                compression=PARQUET_COMPRESSION,
                statistics=False,
            )
            print(f"输出文件：{output_path}")
            print(f"输出记录数：{result.height}")
            return

    if OUTPUT_FORMAT == "csv":
        result = collect_streaming(lf)
        result.write_csv(output_path)
        print(f"输出文件：{output_path}")
        print(f"输出记录数：{result.height}")
        return

    raise ValueError(f"不支持的输出格式：{OUTPUT_FORMAT}")


# ============================================================
# 8. 批量处理
# ============================================================

def process_chunk_files():
    """
    遍历速度分块文件，逐块输出密度和流量计算结果。
    """

    chunk_files, input_format = find_speed_chunk_files()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    speed_param_frame = build_speed_param_frame()

    print(f"发现 {len(chunk_files)} 个速度分块文件")
    print(f"输入格式：{input_format}")
    print(f"读取速度等级参数表：{SPEED_CLASS_PARAM_PATH}")
    print(f"读取通行能力参数表：{CAPACITY_PARAM_PATH}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"输出格式：{OUTPUT_FORMAT}")

    print("\n速度等级参数表预览：")
    print(
        speed_param_frame.select([
            "速度等级",
            "P99.5速度",
            "自由流速度",
            "每车道临界密度",
            "通行能力",
            "临界密度",
            "临界速度",
        ])
    )

    for index, chunk_file in enumerate(chunk_files, start=1):
        print(f"\n正在处理 {index}/{len(chunk_files)}：{chunk_file.name}")

        query = build_density_metrics_query(
            chunk_file=chunk_file,
            input_format=input_format,
            speed_param_frame=speed_param_frame,
        )

        output_path = build_output_path(chunk_file)

        write_lazy_result(
            lf=query,
            output_path=output_path,
        )


# ============================================================
# 9. 主函数
# ============================================================

def main():
    process_chunk_files()


if __name__ == "__main__":
    main()
    
