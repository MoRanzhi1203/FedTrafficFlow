# ==========================
# compute_greenshields_density.py
# ==========================
# -*- coding: utf-8 -*-

"""按文档方案计算 Greenshields 密度、流量及相关派生特征。"""

from pathlib import Path

import polars as pl


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "processed" / "speed_data_chunks"
OUTPUT_DIR = BASE_DIR / "data" / "analysis" / "density_metrics_chunks"
PARAM_PATH = BASE_DIR / "data" / "params" / "speed_class_density_params.csv"

OUTPUT_FORMAT = "parquet"
PARQUET_COMPRESSION = "lz4"

REQUIRED_INPUT_COLUMNS = [
    "路段ID",
    "时间段",
    "平均速度",
    "长度",
    "速度等级",
    "车道数",
]

REQUIRED_PARAM_COLUMNS = [
    "速度等级",
    "P99.5速度",
    "自由流速度",
    "每车道临界密度",
]


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """兼容不同 Polars 版本的流式 collect。"""
    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def require_columns(columns: list[str], required_columns: list[str], file_path: Path) -> None:
    """检查输入或参数文件是否包含必需字段。"""
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        raise ValueError(
            f"文件缺少必要字段：{file_path}\n"
            f"缺失字段：{missing_columns}\n"
            f"当前字段：{columns}"
        )


def clean_string_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .alias(column_name)
    )


def clean_float_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(column_name)
    )


def clean_int_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Int64, strict=False)
        .alias(column_name)
    )


def load_speed_param_frame() -> pl.DataFrame:
    """读取速度等级参数，并按文档补齐阻塞密度。"""
    df = pl.read_csv(PARAM_PATH)

    if "P99.5速度" not in df.columns and "第99.5百分位速度" in df.columns:
        df = df.rename({"第99.5百分位速度": "P99.5速度"})

    require_columns(df.columns, REQUIRED_PARAM_COLUMNS, PARAM_PATH)

    param_df = (
        df
        .with_columns([
            clean_int_expr("速度等级"),
            clean_float_expr("P99.5速度"),
            clean_float_expr("自由流速度"),
            clean_float_expr("每车道临界密度"),
        ])
        .filter(
            pl.col("速度等级").is_not_null()
            & pl.col("P99.5速度").is_not_null()
            & pl.col("自由流速度").is_not_null()
            & pl.col("每车道临界密度").is_not_null()
            & (pl.col("自由流速度") > 0)
            & (pl.col("每车道临界密度") > 0)
        )
        .with_columns([
            pl.col("P99.5速度").alias("p995_class"),
            pl.col("自由流速度").alias("v_f"),
            pl.col("每车道临界密度").alias("k_c_lane"),
            (pl.col("每车道临界密度") * 2.0).alias("k_j_lane"),
        ])
        .select([
            "速度等级",
            "p995_class",
            "v_f",
            "k_c_lane",
            "k_j_lane",
        ])
        .sort("速度等级")
    )

    if param_df.height == 0:
        raise ValueError(f"参数表为空或清洗后无有效记录：{PARAM_PATH}")

    return param_df


def find_speed_chunk_files() -> tuple[list[Path], str]:
    """优先读取 Parquet 分片，找不到时回退到 CSV。"""
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
    """根据文件格式扫描速度分片。"""
    if input_format == "parquet":
        return pl.scan_parquet(chunk_file.as_posix())

    if input_format == "csv":
        return pl.scan_csv(
            chunk_file.as_posix(),
            infer_schema_length=0,
            ignore_errors=True,
        )

    raise ValueError(f"不支持的输入格式：{input_format}")


def cast_metric_columns(lf: pl.LazyFrame, chunk_file: Path) -> pl.LazyFrame:
    """统一转换密度计算所需字段。"""
    require_columns(lf.collect_schema().names(), REQUIRED_INPUT_COLUMNS, chunk_file)

    return (
        lf
        .select(REQUIRED_INPUT_COLUMNS)
        .with_columns([
            clean_string_expr("路段ID"),
            clean_int_expr("时间段"),
            clean_float_expr("平均速度"),
            clean_float_expr("长度"),
            clean_int_expr("速度等级"),
            clean_float_expr("车道数"),
        ])
        .with_columns([
            pl.col("车道数")
            .round(0)
            .cast(pl.Int64, strict=False)
            .alias("车道数"),
        ])
    )


def build_density_metrics_query(
    chunk_file: Path,
    input_format: str,
    speed_param_frame: pl.DataFrame,
) -> pl.LazyFrame:
    """按文档公式构建单个分片的密度与流量计算查询。"""
    speed_param_lf = speed_param_frame.lazy()

    raw_lf = scan_speed_chunk(chunk_file=chunk_file, input_format=input_format)

    base_lf = (
        raw_lf
        .pipe(cast_metric_columns, chunk_file=chunk_file)
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
        .join(speed_param_lf, on="速度等级", how="left")
        .filter(
            pl.col("p995_class").is_not_null()
            & pl.col("v_f").is_not_null()
            & pl.col("k_c_lane").is_not_null()
            & pl.col("k_j_lane").is_not_null()
            & (pl.col("v_f") > 0)
            & (pl.col("k_c_lane") > 0)
            & (pl.col("k_j_lane") > 0)
        )
        .with_columns([
            pl.col("速度等级").alias("speedclass"),
            pl.col("平均速度").alias("v_obs"),
            pl.col("长度").alias("segment_length_km"),
            pl.col("车道数").alias("lane_num"),
        ])
        .with_columns([
            pl.min_horizontal(
                pl.max_horizontal(pl.col("v_obs"), pl.lit(0.0)),
                pl.col("v_f"),
            ).alias("v_used"),
            (pl.col("v_obs") > pl.col("p995_class"))
            .cast(pl.Int8)
            .alias("high_speed_outlier"),
            (pl.col("k_j_lane") * pl.col("lane_num")).alias("k_j_road"),
        ])
        .with_columns([
            pl.max_horizontal(
                pl.col("k_j_road") * (1.0 - pl.col("v_used") / pl.col("v_f")),
                pl.lit(0.0),
            ).alias("density_k"),
        ])
        .with_columns([
            (pl.col("density_k") * pl.col("segment_length_km")).alias("vehicle_count_N"),
            (pl.col("v_used") * pl.col("density_k")).alias("flow_q_hour"),
            (pl.col("v_f") * pl.col("k_j_road") / 4.0).alias("q_max"),
        ])
        .with_columns([
            (pl.col("flow_q_hour") / 4.0).alias("flow_Q_15min"),
            (pl.col("density_k") / pl.col("k_j_road")).alias("rho"),
            pl.when(pl.col("q_max") > 0)
            .then(pl.col("flow_q_hour") / pl.col("q_max"))
            .otherwise(pl.lit(0.0))
            .alias("q_tilde"),
        ])
        .with_columns([
            pl.when(pl.col("rho") < 0.3)
            .then(pl.lit(0))
            .when(pl.col("rho") < 0.5)
            .then(pl.lit(1))
            .when(pl.col("rho") < 0.7)
            .then(pl.lit(2))
            .otherwise(pl.lit(3))
            .cast(pl.Int8)
            .alias("traffic_state_S"),
        ])
    )

    return (
        base_lf
        .select([
            "路段ID",
            "时间段",
            "speedclass",
            "v_obs",
            "v_f",
            "v_used",
            "p995_class",
            "high_speed_outlier",
            "lane_num",
            "k_c_lane",
            "k_j_lane",
            "k_j_road",
            "density_k",
            "segment_length_km",
            "vehicle_count_N",
            "flow_q_hour",
            "flow_Q_15min",
            "rho",
            "q_max",
            "q_tilde",
            "traffic_state_S",
        ])
        .sort(["时间段", "路段ID", "speedclass"])
    )


def build_output_path(chunk_file: Path) -> Path:
    """根据输入分片名构造输出路径。"""
    chunk_suffix = chunk_file.stem.replace("speed_", "")

    if OUTPUT_FORMAT == "parquet":
        return OUTPUT_DIR / f"density_{chunk_suffix}.parquet"

    if OUTPUT_FORMAT == "csv":
        return OUTPUT_DIR / f"density_{chunk_suffix}.csv"

    raise ValueError(f"不支持的输出格式：{OUTPUT_FORMAT}")


def write_lazy_result(lf: pl.LazyFrame, output_path: Path) -> None:
    """写出结果，优先使用惰性 sink。"""
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


def process_chunk_files() -> None:
    """批量读取速度分片并输出密度指标分片。"""
    chunk_files, input_format = find_speed_chunk_files()
    speed_param_frame = load_speed_param_frame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"发现 {len(chunk_files)} 个速度分块文件")
    print(f"输入格式：{input_format}")
    print(f"读取参数表：{PARAM_PATH}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"输出格式：{OUTPUT_FORMAT}")

    print("\n参数表预览：")
    print(speed_param_frame)

    for index, chunk_file in enumerate(chunk_files, start=1):
        print(f"\n正在处理 {index}/{len(chunk_files)}：{chunk_file.name}")
        query = build_density_metrics_query(
            chunk_file=chunk_file,
            input_format=input_format,
            speed_param_frame=speed_param_frame,
        )
        write_lazy_result(query, build_output_path(chunk_file))


def main() -> None:
    process_chunk_files()


if __name__ == "__main__":
    main()
    
