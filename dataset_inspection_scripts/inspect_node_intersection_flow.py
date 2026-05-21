"""检查路口流量 Parquet 分片的结构、全局时间段覆盖和排序情况。"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import sys

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"

HEAD_ROWS = 50
TAIL_ROWS = 20
SAMPLE_VALUES_PER_COLUMN = 10
FOLDER_PREVIEW_LIMIT = 10
FILES_TO_INSPECT: Optional[int] = None
TIME_COL = "时间段"
NODE_COL = "节点ID"
EXPECTED_SLOTS_PER_FILE = 96
TIME_ORDER_REPORT_KEYS = [
    "分片序号",
    "最小时间段",
    "最大时间段",
    "唯一时间段数",
    "期望起始时间段",
    "期望结束时间段",
    "时间段覆盖正确",
    "排序正确",
    "错误信息",
]

# 默认关闭高成本的唯一值统计，避免对数百万行浮点列做全量 distinct。
ENABLE_HIGH_COST_UNIQUE_STATS = False
LOW_CARDINALITY_THRESHOLD = 20
LOW_CARDINALITY_CANDIDATE_COLS = [TIME_COL]

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def configure_display() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_fmt_str_lengths(100)
    pl.Config.set_tbl_width_chars(240)


def print_title(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def list_parquet_files(folder_path: Path) -> List[Path]:
    return sorted(folder_path.glob("*.parquet"))


def extract_chunk_index(file_path: Path) -> Optional[int]:
    match = re.search(r"(\d+)$", file_path.stem)
    if match:
        return int(match.group(1))
    return None


def build_error_summary(file_path: Path, error: Exception) -> Dict[str, Any]:
    chunk_index = extract_chunk_index(file_path)
    return {
        "文件名": file_path.name,
        "分片序号": chunk_index,
        "记录数": None,
        "最小时间段": None,
        "最大时间段": None,
        "唯一时间段数": None,
        "期望起始时间段": None if chunk_index is None else chunk_index * EXPECTED_SLOTS_PER_FILE,
        "期望结束时间段": None
        if chunk_index is None
        else chunk_index * EXPECTED_SLOTS_PER_FILE + EXPECTED_SLOTS_PER_FILE - 1,
        "时间段覆盖正确": False,
        "排序正确": False,
        "错误信息": str(error),
    }


def is_lexicographically_sorted(df: pl.DataFrame) -> bool:
    """检查 [时间段, 节点ID] 是否按字典序非降序排列。"""
    if df.is_empty():
        return True

    order_check = (
        df.select([TIME_COL, NODE_COL])
        .with_columns([
            pl.col(TIME_COL).shift(1).alias("_prev_time"),
            pl.col(NODE_COL).shift(1).alias("_prev_node"),
        ])
        .select(
            (
                pl.col("_prev_time").is_null()
                | (pl.col(TIME_COL) > pl.col("_prev_time"))
                | (
                    (pl.col(TIME_COL) == pl.col("_prev_time"))
                    & (pl.col(NODE_COL) >= pl.col("_prev_node"))
                )
            )
            .all()
            .alias("sorted_ok")
        )
    )
    return bool(order_check.item())


def build_evenly_spaced_indices(length: int, limit: int) -> List[int]:
    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [0]

    step = (length - 1) / (limit - 1)
    return [round(index * step) for index in range(limit)]


def sample_non_null_values(series: pl.Series, limit: int) -> List[Any]:
    non_null_series = series.drop_nulls()
    if non_null_series.is_empty():
        return []

    sample_indices = build_evenly_spaced_indices(len(non_null_series), limit)
    return non_null_series.gather(sample_indices).to_list()


def format_sample_value(value: Any) -> str:
    if isinstance(value, float):
        return "{0:.6g}".format(value)
    return str(value)


def format_sample_values(values: List[Any]) -> str:
    if not values:
        return "[]"
    return "[{0}]".format(", ".join(format_sample_value(value) for value in values))


def build_time_order_summary(file_path: Path) -> Dict[str, Any]:
    """生成单个节点流量分片的时间段与排序检查结果。"""
    chunk_index = extract_chunk_index(file_path)

    try:
        time_node_df = pl.read_parquet(file_path, columns=[TIME_COL, NODE_COL])
    except Exception as exc:
        return build_error_summary(file_path, exc)

    required_columns = {TIME_COL, NODE_COL}
    if not required_columns.issubset(set(time_node_df.columns)):
        missing_columns = sorted(required_columns - set(time_node_df.columns))
        return build_error_summary(
            file_path,
            ValueError("缺少必要字段: {0}".format(", ".join(missing_columns))),
        )

    stats_row = time_node_df.select(
        [
            pl.len().alias("rows"),
            pl.col(TIME_COL).min().alias("min_time"),
            pl.col(TIME_COL).max().alias("max_time"),
            pl.col(TIME_COL).n_unique().alias("unique_times"),
        ]
    ).to_dicts()[0]

    if chunk_index is None:
        expected_start = None
        expected_end = None
        expected_range_ok = None
    else:
        expected_start = chunk_index * EXPECTED_SLOTS_PER_FILE
        expected_end = expected_start + EXPECTED_SLOTS_PER_FILE - 1
        expected_range_ok = (
            stats_row["min_time"] == expected_start
            and stats_row["max_time"] == expected_end
            and stats_row["unique_times"] == EXPECTED_SLOTS_PER_FILE
        )

    return {
        "文件名": file_path.name,
        "分片序号": chunk_index,
        "记录数": stats_row["rows"],
        "最小时间段": stats_row["min_time"],
        "最大时间段": stats_row["max_time"],
        "唯一时间段数": stats_row["unique_times"],
        "期望起始时间段": expected_start,
        "期望结束时间段": expected_end,
        "时间段覆盖正确": expected_range_ok,
        "排序正确": is_lexicographically_sorted(time_node_df),
        "错误信息": "",
    }


def build_folder_time_order_summary_df(parquet_files: List[Path]) -> pl.DataFrame:
    return pl.DataFrame([build_time_order_summary(file_path) for file_path in parquet_files])


def build_dtype_df(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({"列名": df.columns, "数据类型": [str(dtype) for dtype in df.dtypes]})


def build_column_stats_df(df: pl.DataFrame) -> pl.DataFrame:
    column_stats = []

    for col in df.columns:
        null_count = df[col].null_count()
        row = {
            "列名": col,
            "数据类型": str(df[col].dtype),
            "空值数": null_count,
            "非空数": df.height - null_count,
            "空值比例": null_count / df.height if df.height > 0 else 0,
        }
        if ENABLE_HIGH_COST_UNIQUE_STATS:
            row["唯一值数"] = df[col].n_unique()
        column_stats.append(row)

    return pl.DataFrame(column_stats)


def build_numeric_stats_df(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    numeric_columns = [col for col in df.columns if df[col].dtype in NUMERIC_DTYPES]
    if not numeric_columns:
        return None

    numeric_stats = []
    for col in numeric_columns:
        numeric_stats.append(
            {
                "列名": col,
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
            }
        )

    return pl.DataFrame(numeric_stats)


def build_sample_df(df: pl.DataFrame) -> pl.DataFrame:
    sample_rows = []

    for col in df.columns:
        sample_values = sample_non_null_values(df.get_column(col), SAMPLE_VALUES_PER_COLUMN)
        sample_rows.append({"列名": col, "样例值": sample_values})

    return pl.DataFrame(
        {
            "列名": [row["列名"] for row in sample_rows],
            "样例值": [format_sample_values(row["样例值"]) for row in sample_rows],
        }
    )


def print_low_cardinality_stats(df: pl.DataFrame) -> None:
    print_title("低唯一值列频数预览")

    preview_columns = [col for col in LOW_CARDINALITY_CANDIDATE_COLS if col in df.columns]
    if not preview_columns:
        print("未配置可预览字段，跳过。")
        return

    found_preview = False
    for col in preview_columns:
        unique_count = df[col].n_unique()
        if unique_count <= LOW_CARDINALITY_THRESHOLD:
            found_preview = True
            print("\n字段:", col)
            print(
                df.group_by(col)
                .agg(pl.len().alias("记录数"))
                .sort(col)
            )
        else:
            print("字段 {0} 唯一值数为 {1}，超过阈值 {2}，跳过频数统计。".format(
                col,
                unique_count,
                LOW_CARDINALITY_THRESHOLD,
            ))

    if not found_preview:
        print("没有需要输出频数预览的低唯一值字段。")


def print_time_order_summary(summary_df: pl.DataFrame) -> None:
    print_title("全量文件时间段覆盖与排序检查")
    print(summary_df)

    print_title("时间段与排序检查汇总")
    total_files = summary_df.height
    coverage_ok_count = summary_df.filter(pl.col("时间段覆盖正确") == True).height
    sorted_ok_count = summary_df.filter(pl.col("排序正确") == True).height
    error_count = summary_df.filter(pl.col("错误信息") != "").height

    print("文件总数:", total_files)
    print("时间段覆盖正确文件数: {0}/{1}".format(coverage_ok_count, total_files))
    print("排序正确文件数: {0}/{1}".format(sorted_ok_count, total_files))
    print("读取失败文件数: {0}".format(error_count))

    abnormal_df = summary_df.filter(
        (pl.col("时间段覆盖正确") != True)
        | (pl.col("排序正确") != True)
        | (pl.col("错误信息") != "")
    )
    if abnormal_df.height == 0:
        print("检查结果: 所有文件的时间段覆盖和排序均正确。")
    else:
        print("检查结果: 以下文件存在异常，请重点复查。")
        print(abnormal_df)


def inspect_parquet_file(
    file_path: Path,
    file_index: int,
    total_files: int,
    time_order_summary: Optional[Dict[str, Any]] = None,
) -> None:
    configure_display()

    print_title("文件 {0}/{1}: {2}".format(file_index, total_files, file_path.name))
    print("文件路径:", file_path)

    try:
        file_size = file_path.stat().st_size
        print("文件大小:", "{0:,} 字节 ({1:.1f} MB)".format(file_size, file_size / 1024 / 1024))
    except OSError as exc:
        print("文件大小: 无法获取 ({0})".format(exc))

    try:
        df = pl.read_parquet(file_path)
    except Exception as exc:
        print("读取失败:", exc)
        return

    print("数据形状:", df.shape)
    print("总行数:", df.height)
    print("总列数:", df.width)

    if time_order_summary is not None:
        print_title("时间段与排序检查")
        for key in TIME_ORDER_REPORT_KEYS:
            print("{0}: {1}".format(key, time_order_summary.get(key)))

    print_title("字段名")
    for i, col in enumerate(df.columns, start=1):
        print("{0:02d}. {1}".format(i, col))

    print_title("数据类型")
    print(build_dtype_df(df))

    print_title("每列空值与非空值统计")
    print(build_column_stats_df(df))

    print_title("数值列描述统计")
    numeric_stats_df = build_numeric_stats_df(df)
    if numeric_stats_df is None:
        print("没有数值列。")
    else:
        print(numeric_stats_df)

    print_title("每列前 {0} 个非空样例值".format(SAMPLE_VALUES_PER_COLUMN))
    print(build_sample_df(df))

    print_low_cardinality_stats(df)

    print_title("前 {0} 行：全部字段".format(HEAD_ROWS))
    print(df.head(HEAD_ROWS))

    print_title("后 {0} 行：全部字段".format(TAIL_ROWS))
    print(df.tail(TAIL_ROWS))

    print_title("查看完成")
    print("文件 {0} 读取和字段检查完成。".format(file_path.name))


def inspect_first_n_files(folder_path: Path, n: Optional[int] = None) -> None:
    parquet_files = list_parquet_files(folder_path)

    print_title("路口流量数据检查")
    print("目标文件夹:", folder_path)
    print("Parquet 文件总数:", len(parquet_files))

    if not parquet_files:
        print("该文件夹下没有找到 parquet 文件。")
        return

    inspect_all_files = n is None or n <= 0 or n >= len(parquet_files)
    files_to_inspect = parquet_files if inspect_all_files else parquet_files[:n]

    if inspect_all_files:
        print("本次检查全部文件")
    else:
        print("本次检查前 {0} 个文件".format(len(files_to_inspect)))

    print_title("前 {0} 个文件列表".format(min(FOLDER_PREVIEW_LIMIT, len(parquet_files))))
    for index, parquet_file in enumerate(parquet_files[:FOLDER_PREVIEW_LIMIT], start=1):
        print("{0:02d}. {1}".format(index, parquet_file.name))

    summary_df = build_folder_time_order_summary_df(parquet_files)
    print_time_order_summary(summary_df)
    summary_map = {row["文件名"]: row for row in summary_df.to_dicts()}

    for i, file_path in enumerate(files_to_inspect, 1):
        inspect_parquet_file(
            file_path,
            i,
            len(files_to_inspect),
            time_order_summary=summary_map.get(file_path.name),
        )

        if i < len(files_to_inspect):
            print("\n" + "#" * 100)
            print("继续检查下一个文件...")
            print("#" * 100)

    print_title("所有文件检查完成")
    print("已成功检查 {0} 个文件".format(len(files_to_inspect)))

    if not inspect_all_files:
        print("提示: 还有 {0} 个文件未检查".format(len(parquet_files) - n))
        print("如需检查更多文件，请增加 FILES_TO_INSPECT 参数")


def main() -> None:
    if not TARGET_DIR.exists():
        print("错误: 目标文件夹不存在: {0}".format(TARGET_DIR))
        print("请确保数据已正确生成")
        sys.exit(1)

    inspect_first_n_files(TARGET_DIR, FILES_TO_INSPECT)


if __name__ == "__main__":
    main()
