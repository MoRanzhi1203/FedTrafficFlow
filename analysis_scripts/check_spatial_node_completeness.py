"""检查道路交通节点时空数据的完整性。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
DEFAULT_INPUT_PATTERN = "node_flow_chunk_*.parquet"
DEFAULT_TOPOLOGY_PATH = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "data" / "analysis" / "node_intersection_flow_check_reports"
)

DEFAULT_NODE_COL = "节点ID"
DEFAULT_TIME_COL = "时间段"
DEFAULT_FLOW_COL = "路口车流量"
DEFAULT_SLOTS_PER_DAY = 96
EXPECTED_FILE_COUNT_HINT = 61

TOPOLOGY_NODE_COLUMN_CANDIDATES = [
    ("起始节点", "终止节点"),
    ("起点节点", "终点节点"),
    ("起始节点ID", "终止节点ID"),
    ("起始节点ID", "结束节点ID"),
    ("起点节点ID", "终点节点ID"),
    ("起点ID", "终点ID"),
    ("起始点ID", "终止点ID"),
    ("from_node", "to_node"),
    ("source_node", "target_node"),
    ("source", "target"),
]

NODE_ID_COL = "node_id"
TIME_INTERNAL_COL = "time"
FLOW_INTERNAL_COL = "flow"
DAY_INDEX_COL = "day_index"
DAY_SLOT_COL = "day_slot"
RAW_COUNT_COL = "_raw_count"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检查道路交通节点时空数据是否存在缺失、重复、非法值和拓扑不一致。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="节点流量 parquet 输入目录。",
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default=DEFAULT_INPUT_PATTERN,
        help="输入 parquet 分片匹配模式。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="完整性检查报告输出目录。",
    )
    parser.add_argument(
        "--topology-path",
        type=Path,
        default=DEFAULT_TOPOLOGY_PATH,
        help="道路网络拓扑文件路径。",
    )
    parser.add_argument(
        "--source-col",
        type=str,
        default=None,
        help="拓扑起点字段名；未传入时自动识别。",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="拓扑终点字段名；未传入时自动识别。",
    )
    parser.add_argument(
        "--node-col",
        type=str,
        default=DEFAULT_NODE_COL,
        help="节点字段名。",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=DEFAULT_TIME_COL,
        help="时间字段名。",
    )
    parser.add_argument(
        "--flow-col",
        type=str,
        default=DEFAULT_FLOW_COL,
        help="流量字段名。",
    )
    parser.add_argument(
        "--slots-per-day",
        type=int,
        default=DEFAULT_SLOTS_PER_DAY,
        help="每日时间段数量，默认 96。",
    )
    parser.add_argument(
        "--write-missing-detail",
        action="store_true",
        default=False,
        help="仅当发现缺失时，按天写出具体缺失 node-slot 明细。",
    )
    return parser.parse_args()


def format_available_columns(columns: Sequence[str]) -> str:
    """格式化字段名列表，便于报错展示。"""
    return ", ".join(columns)


def list_input_files(input_dir: Path, input_pattern: str) -> list[Path]:
    """按输入目录和匹配模式列出分片文件。"""
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    parquet_files = sorted(input_dir.glob(input_pattern))
    if not parquet_files:
        raise FileNotFoundError(f"未找到输入分片: {input_dir / input_pattern}")
    return parquet_files


def ensure_required_columns(
    parquet_files: Sequence[Path],
    node_col: str,
    time_col: str,
    flow_col: str,
) -> None:
    """检查所有输入分片是否包含必需字段。"""
    required_cols = [node_col, time_col, flow_col]
    missing_by_file: list[str] = []

    for file_path in parquet_files:
        actual_columns = list(pl.scan_parquet(file_path).collect_schema().names())
        missing_cols = [col for col in required_cols if col not in actual_columns]
        if missing_cols:
            missing_by_file.append(
                f"{file_path.name}: 缺少 {missing_cols}; 实际字段: {format_available_columns(actual_columns)}"
            )

    if missing_by_file:
        raise ValueError("以下输入分片缺少必需字段:\n" + "\n".join(missing_by_file))


def read_table(table_path: Path) -> pl.DataFrame:
    """读取 csv 或 parquet 表。"""
    if not table_path.exists():
        raise FileNotFoundError(f"拓扑文件不存在: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(table_path)
    if suffix == ".csv":
        return pl.read_csv(table_path)
    raise ValueError(f"不支持的拓扑文件格式: {table_path}")


def infer_topology_node_columns(topology_df: pl.DataFrame) -> tuple[str, str]:
    """自动识别拓扑起止节点字段。"""
    columns = set(topology_df.columns)
    for source_col, target_col in TOPOLOGY_NODE_COLUMN_CANDIDATES:
        if source_col in columns and target_col in columns:
            return source_col, target_col

    raise ValueError(
        "无法自动识别拓扑起止节点字段。\n"
        f"当前 rnsd_processed.csv 字段为：\n{format_available_columns(topology_df.columns)}\n"
        "请显式传入：--source-col xxx --target-col yyy"
    )


def resolve_topology_columns(
    topology_df: pl.DataFrame,
    source_col: str | None,
    target_col: str | None,
) -> tuple[str, str]:
    """优先使用用户显式传入字段，否则自动识别。"""
    if source_col and target_col:
        missing_cols = [
            column
            for column in [source_col, target_col]
            if column not in topology_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"拓扑文件中不存在指定字段: {missing_cols}\n"
                f"当前 rnsd_processed.csv 字段为：\n{format_available_columns(topology_df.columns)}"
            )
        return source_col, target_col

    if source_col or target_col:
        raise ValueError(
            "请同时传入 --source-col 和 --target-col，或都不传让脚本自动识别。"
        )

    return infer_topology_node_columns(topology_df)


def load_topology_node_df(
    topology_df: pl.DataFrame,
    source_col: str,
    target_col: str,
) -> pl.DataFrame:
    """构造拓扑理论节点集合。"""
    topology_node_df = (
        pl.concat(
            [
                topology_df.select(
                    pl.col(source_col).cast(pl.Int64, strict=False).alias(NODE_ID_COL)
                ),
                topology_df.select(
                    pl.col(target_col).cast(pl.Int64, strict=False).alias(NODE_ID_COL)
                ),
            ],
            how="vertical",
        )
        .filter(pl.col(NODE_ID_COL).is_not_null())
        .unique()
        .sort(NODE_ID_COL)
    )
    if topology_node_df.is_empty():
        raise ValueError("拓扑节点集合为空，无法进行完整性检查。")
    return topology_node_df


def read_day_data(
    file_path: Path,
    node_col: str,
    time_col: str,
    flow_col: str,
    slots_per_day: int,
) -> pl.DataFrame:
    """读取单日分片并统一为内部字段。"""
    return (
        pl.read_parquet(file_path, columns=[node_col, time_col, flow_col])
        .select(
            [
                pl.col(node_col).cast(pl.Int64, strict=False).alias(NODE_ID_COL),
                pl.col(time_col).cast(pl.Int64, strict=False).alias(TIME_INTERNAL_COL),
                pl.col(flow_col).cast(pl.Float64, strict=False).alias(FLOW_INTERNAL_COL),
            ]
        )
        .with_columns(
            (pl.col(TIME_INTERNAL_COL) % slots_per_day).cast(pl.Int16).alias(DAY_SLOT_COL)
        )
    )


def invalid_flow_expr() -> pl.Expr:
    """非法流量判定表达式。"""
    return (
        pl.col(FLOW_INTERNAL_COL).is_null()
        | pl.col(FLOW_INTERNAL_COL).is_nan().fill_null(False)
        | (pl.col(FLOW_INTERNAL_COL) < 0).fill_null(False)
    )


def summarize_day(
    day_df: pl.DataFrame,
    day_index: int,
    file_name: str,
    slots_per_day: int,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, set[int]]:
    """统计单日完整性信息。"""
    expected_start_time = day_index * slots_per_day
    expected_end_time = expected_start_time + slots_per_day - 1

    time_values = (
        day_df.select(pl.col(TIME_INTERNAL_COL).drop_nulls().unique().sort())
        .get_column(TIME_INTERNAL_COL)
        .to_list()
    )
    actual_time_set = set(time_values)
    expected_time_set = set(range(expected_start_time, expected_end_time + 1))

    actual_min_time = min(time_values) if time_values else None
    actual_max_time = max(time_values) if time_values else None
    unique_time_count = len(time_values)
    missing_time_slots = len(expected_time_set - actual_time_set)
    extra_time_slots = len(actual_time_set - expected_time_set)
    bad_time_range = (
        actual_min_time != expected_start_time
        or actual_max_time != expected_end_time
        or unique_time_count != slots_per_day
    )

    pair_df = day_df.select([NODE_ID_COL, DAY_SLOT_COL]).drop_nulls().unique()
    observed_unique_pairs = pair_df.height
    raw_records = day_df.height

    invalid_expr = invalid_flow_expr()
    flow_count_row = day_df.select(
        [
            pl.when(pl.col(FLOW_INTERNAL_COL).is_null())
            .then(1)
            .otherwise(0)
            .sum()
            .alias("null_flow_count"),
            pl.when(pl.col(FLOW_INTERNAL_COL).is_nan().fill_null(False))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("nan_flow_count"),
            pl.when((pl.col(FLOW_INTERNAL_COL) < 0).fill_null(False))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("negative_flow_count"),
            pl.when((pl.col(FLOW_INTERNAL_COL) == 0).fill_null(False))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("zero_flow_count"),
            pl.when((pl.col(FLOW_INTERNAL_COL) > 0).fill_null(False))
            .then(1)
            .otherwise(0)
            .sum()
            .alias("positive_flow_count"),
        ]
    ).row(0, named=True)

    valid_numeric_flow_df = day_df.filter(
        pl.col(FLOW_INTERNAL_COL).is_not_null()
        & ~pl.col(FLOW_INTERNAL_COL).is_nan().fill_null(False)
    )
    flow_stat_row = valid_numeric_flow_df.select(
        [
            pl.col(FLOW_INTERNAL_COL).min().alias("min_flow"),
            pl.col(FLOW_INTERNAL_COL).max().alias("max_flow"),
            pl.col(FLOW_INTERNAL_COL).mean().alias("mean_flow"),
        ]
    ).row(0, named=True)

    node_observed_df = (
        pair_df.group_by(NODE_ID_COL)
        .agg(pl.len().alias("observed_unique_count"))
        .sort(NODE_ID_COL)
    )
    node_raw_df = (
        day_df.filter(pl.col(NODE_ID_COL).is_not_null())
        .group_by(NODE_ID_COL)
        .agg(pl.len().alias(RAW_COUNT_COL))
    )
    node_invalid_df = (
        day_df.filter(pl.col(NODE_ID_COL).is_not_null() & invalid_expr)
        .group_by(NODE_ID_COL)
        .agg(pl.len().alias("invalid_flow_count"))
    )
    node_day_summary_df = (
        node_raw_df.join(node_observed_df, on=NODE_ID_COL, how="left")
        .join(node_invalid_df, on=NODE_ID_COL, how="left")
        .with_columns(
            [
                pl.col(RAW_COUNT_COL).fill_null(0).cast(pl.Int64),
                pl.col("observed_unique_count").fill_null(0).cast(pl.Int64),
                pl.col("invalid_flow_count").fill_null(0).cast(pl.Int64),
            ]
        )
        .with_columns(
            (pl.col(RAW_COUNT_COL) - pl.col("observed_unique_count")).alias(
                "duplicate_count"
            )
        )
        .select([NODE_ID_COL, "observed_unique_count", "duplicate_count", "invalid_flow_count"])
        .sort(NODE_ID_COL)
    )

    slot_observed_df = pair_df.group_by(DAY_SLOT_COL).agg(
        pl.len().alias("observed_unique_count")
    )
    slot_invalid_df = (
        day_df.filter(pl.col(DAY_SLOT_COL).is_not_null() & invalid_expr)
        .group_by(DAY_SLOT_COL)
        .agg(pl.len().alias("invalid_flow_count"))
    )
    slot_day_summary_df = (
        slot_observed_df.join(slot_invalid_df, on=DAY_SLOT_COL, how="left")
        .with_columns(
            [
                pl.col("observed_unique_count").fill_null(0).cast(pl.Int64),
                pl.col("invalid_flow_count").fill_null(0).cast(pl.Int64),
            ]
        )
        .select([DAY_SLOT_COL, "observed_unique_count", "invalid_flow_count"])
        .sort(DAY_SLOT_COL)
    )

    observed_node_ids = set(
        day_df.select(pl.col(NODE_ID_COL).drop_nulls().unique())
        .get_column(NODE_ID_COL)
        .to_list()
    )

    day_record = {
        DAY_INDEX_COL: day_index,
        "file_name": file_name,
        "expected_start_time": expected_start_time,
        "expected_end_time": expected_end_time,
        "actual_min_time": actual_min_time,
        "actual_max_time": actual_max_time,
        "unique_time_count": unique_time_count,
        "bad_time_range": bad_time_range,
        "missing_time_slots": missing_time_slots,
        "extra_time_slots": extra_time_slots,
        "raw_records": raw_records,
        "observed_unique_pairs": observed_unique_pairs,
        "null_flow_count": int(flow_count_row["null_flow_count"] or 0),
        "nan_flow_count": int(flow_count_row["nan_flow_count"] or 0),
        "negative_flow_count": int(flow_count_row["negative_flow_count"] or 0),
        "zero_flow_count": int(flow_count_row["zero_flow_count"] or 0),
        "positive_flow_count": int(flow_count_row["positive_flow_count"] or 0),
        "min_flow": flow_stat_row["min_flow"],
        "max_flow": flow_stat_row["max_flow"],
        "mean_flow": flow_stat_row["mean_flow"],
    }
    return day_record, node_day_summary_df, slot_day_summary_df, observed_node_ids


def build_topology_comparison(
    observed_node_df: pl.DataFrame,
    topology_node_df: pl.DataFrame,
) -> pl.DataFrame:
    """输出观测节点集与拓扑节点集对比表。"""
    comparison_df = (
        pl.concat(
            [
                observed_node_df.with_columns(
                    [
                        pl.lit(True).alias("in_observed"),
                        pl.lit(False).alias("in_topology"),
                    ]
                ),
                topology_node_df.with_columns(
                    [
                        pl.lit(False).alias("in_observed"),
                        pl.lit(True).alias("in_topology"),
                    ]
                ),
            ],
            how="vertical",
        )
        .group_by(NODE_ID_COL)
        .agg(
            [
                pl.col("in_observed").any().alias("in_observed"),
                pl.col("in_topology").any().alias("in_topology"),
            ]
        )
        .with_columns(
            pl.when(pl.col("in_observed") & pl.col("in_topology"))
            .then(pl.lit("both"))
            .when(pl.col("in_observed"))
            .then(pl.lit("observed_only"))
            .otherwise(pl.lit("topology_only"))
            .alias("status")
        )
        .sort(NODE_ID_COL)
    )
    return comparison_df


def write_missing_details(
    missing_day_indices: Sequence[int],
    input_files: Sequence[Path],
    observed_node_df: pl.DataFrame,
    node_col: str,
    time_col: str,
    slots_per_day: int,
    output_dir: Path,
) -> None:
    """仅对存在缺失的日期写出具体缺失 node-slot 明细。"""
    if not missing_day_indices:
        return

    missing_dir = output_dir / "missing_details"
    missing_dir.mkdir(parents=True, exist_ok=True)

    slot_df = pl.DataFrame({DAY_SLOT_COL: list(range(slots_per_day))}).with_columns(
        pl.col(DAY_SLOT_COL).cast(pl.Int16)
    )
    expected_pair_df = observed_node_df.join(slot_df, how="cross")

    for day_index in missing_day_indices:
        file_path = input_files[day_index]
        print(f"写出缺失明细 day_index={day_index}: {file_path.name}", flush=True)
        observed_pair_df = (
            pl.read_parquet(file_path, columns=[node_col, time_col])
            .select(
                [
                    pl.col(node_col).cast(pl.Int64, strict=False).alias(NODE_ID_COL),
                    (pl.col(time_col).cast(pl.Int64, strict=False) % slots_per_day)
                    .cast(pl.Int16)
                    .alias(DAY_SLOT_COL),
                ]
            )
            .drop_nulls()
            .unique()
        )
        missing_df = (
            expected_pair_df.join(
                observed_pair_df,
                on=[NODE_ID_COL, DAY_SLOT_COL],
                how="anti",
            )
            .with_columns(
                [
                    pl.lit(day_index).cast(pl.Int32).alias(DAY_INDEX_COL),
                    (
                        pl.lit(day_index * slots_per_day).cast(pl.Int64)
                        + pl.col(DAY_SLOT_COL).cast(pl.Int64)
                    ).alias(TIME_INTERNAL_COL),
                ]
            )
            .select([NODE_ID_COL, DAY_INDEX_COL, DAY_SLOT_COL, TIME_INTERNAL_COL])
            .sort([NODE_ID_COL, DAY_SLOT_COL])
        )
        if not missing_df.is_empty():
            missing_df.write_parquet(
                missing_dir / f"missing_day_{day_index:03d}.parquet",
                compression="snappy",
            )


def main() -> None:
    """主函数。"""
    args = parse_args()
    input_files = list_input_files(args.input_dir, args.input_pattern)
    ensure_required_columns(
        parquet_files=input_files,
        node_col=args.node_col,
        time_col=args.time_col,
        flow_col=args.flow_col,
    )

    topology_df = read_table(args.topology_path)
    source_col, target_col = resolve_topology_columns(
        topology_df=topology_df,
        source_col=args.source_col,
        target_col=args.target_col,
    )
    topology_node_df = load_topology_node_df(
        topology_df=topology_df,
        source_col=source_col,
        target_col=target_col,
    )

    print("=" * 80, flush=True)
    print("道路交通时空数据完整性检查", flush=True)
    print("=" * 80, flush=True)
    print(f"输入目录: {args.input_dir}", flush=True)
    print(f"输入匹配模式: {args.input_pattern}", flush=True)
    print(f"拓扑文件: {args.topology_path}", flush=True)
    print(f"识别到的起点字段: {source_col}", flush=True)
    print(f"识别到的终点字段: {target_col}", flush=True)
    print(f"节点字段: {args.node_col}", flush=True)
    print(f"时间字段: {args.time_col}", flush=True)
    print(f"流量字段: {args.flow_col}", flush=True)
    print("-" * 80, flush=True)

    if len(input_files) != EXPECTED_FILE_COUNT_HINT:
        print(
            f"警告: 当前文件数量不是 {EXPECTED_FILE_COUNT_HINT}，请确认日期范围是否变化。",
            flush=True,
        )

    daily_records: list[dict[str, Any]] = []
    node_summary_frames: list[pl.DataFrame] = []
    slot_summary_frames: list[pl.DataFrame] = []
    observed_node_ids: set[int] = set()

    for day_index, file_path in enumerate(input_files):
        print(f"检查 day_index={day_index}: {file_path.name}", flush=True)
        day_df = read_day_data(
            file_path=file_path,
            node_col=args.node_col,
            time_col=args.time_col,
            flow_col=args.flow_col,
            slots_per_day=args.slots_per_day,
        )
        day_record, node_day_summary_df, slot_day_summary_df, day_node_ids = summarize_day(
            day_df=day_df,
            day_index=day_index,
            file_name=file_path.name,
            slots_per_day=args.slots_per_day,
        )
        daily_records.append(day_record)
        node_summary_frames.append(node_day_summary_df)
        slot_summary_frames.append(slot_day_summary_df)
        observed_node_ids.update(day_node_ids)

    if not observed_node_ids:
        raise ValueError("输入 node_flow 分片中未扫描到任何有效节点。")

    observed_node_df = pl.DataFrame(
        {NODE_ID_COL: sorted(observed_node_ids)},
        schema={NODE_ID_COL: pl.Int64},
    )
    observed_node_count = observed_node_df.height
    file_count = len(input_files)

    daily_summary_df = (
        pl.DataFrame(daily_records)
        .with_columns(
            [
                pl.lit(observed_node_count * args.slots_per_day)
                .cast(pl.Int64)
                .alias("expected_records"),
                (
                    pl.lit(observed_node_count * args.slots_per_day).cast(pl.Int64)
                    - pl.col("observed_unique_pairs").cast(pl.Int64)
                ).alias("missing_records"),
                (
                    pl.col("raw_records").cast(pl.Int64)
                    - pl.col("observed_unique_pairs").cast(pl.Int64)
                ).alias("duplicate_records"),
            ]
        )
        .with_columns(
            (
                (pl.col("missing_records") == 0)
                & (pl.col("duplicate_records") == 0)
                & (pl.col("null_flow_count") == 0)
                & (pl.col("nan_flow_count") == 0)
                & (pl.col("negative_flow_count") == 0)
                & (pl.col("unique_time_count") == args.slots_per_day)
                & (~pl.col("bad_time_range"))
            ).alias("is_day_passed")
        )
        .select(
            [
                DAY_INDEX_COL,
                "file_name",
                "expected_start_time",
                "expected_end_time",
                "actual_min_time",
                "actual_max_time",
                "unique_time_count",
                "bad_time_range",
                "missing_time_slots",
                "extra_time_slots",
                "raw_records",
                "observed_unique_pairs",
                "expected_records",
                "missing_records",
                "duplicate_records",
                "null_flow_count",
                "nan_flow_count",
                "negative_flow_count",
                "zero_flow_count",
                "positive_flow_count",
                "min_flow",
                "max_flow",
                "mean_flow",
                "is_day_passed",
            ]
        )
        .sort(DAY_INDEX_COL)
    )

    node_aggregate_df = (
        pl.concat(node_summary_frames, how="vertical")
        .group_by(NODE_ID_COL)
        .agg(
            [
                pl.col("observed_unique_count").sum().alias("observed_unique_count"),
                pl.col("duplicate_count").sum().alias("duplicate_count"),
                pl.col("invalid_flow_count").sum().alias("invalid_flow_count"),
            ]
        )
    )
    node_summary_df = (
        observed_node_df.join(node_aggregate_df, on=NODE_ID_COL, how="left")
        .with_columns(
            [
                pl.col("observed_unique_count").fill_null(0).cast(pl.Int64),
                pl.col("duplicate_count").fill_null(0).cast(pl.Int64),
                pl.col("invalid_flow_count").fill_null(0).cast(pl.Int64),
                pl.lit(file_count * args.slots_per_day)
                .cast(pl.Int64)
                .alias("expected_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("expected_count") - pl.col("observed_unique_count")).alias(
                    "missing_count"
                ),
                (pl.col("expected_count") - pl.col("observed_unique_count"))
                .truediv(pl.col("expected_count"))
                .alias("missing_ratio"),
            ]
        )
        .with_columns(
            (
                (pl.col("missing_count") == 0)
                & (pl.col("duplicate_count") == 0)
                & (pl.col("invalid_flow_count") == 0)
            ).alias("is_node_passed")
        )
        .select(
            [
                NODE_ID_COL,
                "expected_count",
                "observed_unique_count",
                "missing_count",
                "missing_ratio",
                "duplicate_count",
                "invalid_flow_count",
                "is_node_passed",
            ]
        )
        .sort(NODE_ID_COL)
    )

    slot_aggregate_df = (
        pl.concat(slot_summary_frames, how="vertical")
        .group_by(DAY_SLOT_COL)
        .agg(
            [
                pl.col("observed_unique_count").sum().alias("observed_unique_count"),
                pl.col("invalid_flow_count").sum().alias("invalid_flow_count"),
            ]
        )
    )
    slot_universe_df = pl.DataFrame(
        {DAY_SLOT_COL: list(range(args.slots_per_day))},
        schema={DAY_SLOT_COL: pl.Int16},
    )
    slot_summary_df = (
        slot_universe_df.join(slot_aggregate_df, on=DAY_SLOT_COL, how="left")
        .with_columns(
            [
                pl.col("observed_unique_count").fill_null(0).cast(pl.Int64),
                pl.col("invalid_flow_count").fill_null(0).cast(pl.Int64),
                pl.lit(observed_node_count * file_count)
                .cast(pl.Int64)
                .alias("expected_count"),
                (
                    pl.lit(observed_node_count * file_count).cast(pl.Int64)
                    - pl.col("observed_unique_count")
                ).alias("missing_count"),
            ]
        )
        .with_columns(
            [
                pl.col("missing_count")
                .truediv(pl.col("expected_count"))
                .alias("missing_ratio"),
                (
                    (pl.col("missing_count") == 0)
                    & (pl.col("invalid_flow_count") == 0)
                ).alias("is_slot_passed"),
            ]
        )
        .sort(DAY_SLOT_COL)
    )

    topology_comparison_df = build_topology_comparison(
        observed_node_df=observed_node_df,
        topology_node_df=topology_node_df,
    )
    observed_not_in_topology_count = topology_comparison_df.filter(
        pl.col("status") == "observed_only"
    ).height
    topology_not_in_observed_count = topology_comparison_df.filter(
        pl.col("status") == "topology_only"
    ).height

    total_raw_records = int(daily_summary_df["raw_records"].sum())
    total_expected_records = int(observed_node_count * file_count * args.slots_per_day)
    total_observed_unique_pairs = int(daily_summary_df["observed_unique_pairs"].sum())
    total_missing_records = int(daily_summary_df["missing_records"].sum())
    total_duplicate_records = int(daily_summary_df["duplicate_records"].sum())
    total_null_flow_count = int(daily_summary_df["null_flow_count"].sum())
    total_nan_flow_count = int(daily_summary_df["nan_flow_count"].sum())
    total_negative_flow_count = int(daily_summary_df["negative_flow_count"].sum())
    total_invalid_flow_count = (
        total_null_flow_count + total_nan_flow_count + total_negative_flow_count
    )

    is_passed = (
        total_missing_records == 0
        and total_duplicate_records == 0
        and total_null_flow_count == 0
        and total_nan_flow_count == 0
        and total_negative_flow_count == 0
        and bool(daily_summary_df["unique_time_count"].eq(args.slots_per_day).all())
        and bool((~daily_summary_df["bad_time_range"]).all())
        and observed_not_in_topology_count == 0
    )

    completeness_summary_df = pl.DataFrame(
        {
            "file_count": [file_count],
            "expected_day_count": [file_count],
            "observed_node_count": [observed_node_count],
            "topology_node_count": [topology_node_df.height],
            "observed_not_in_topology_count": [observed_not_in_topology_count],
            "topology_not_in_observed_count": [topology_not_in_observed_count],
            "total_raw_records": [total_raw_records],
            "total_expected_records": [total_expected_records],
            "total_observed_unique_pairs": [total_observed_unique_pairs],
            "total_missing_records": [total_missing_records],
            "total_duplicate_records": [total_duplicate_records],
            "total_null_flow_count": [total_null_flow_count],
            "total_nan_flow_count": [total_nan_flow_count],
            "total_negative_flow_count": [total_negative_flow_count],
            "is_passed": [is_passed],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completeness_summary_df.write_csv(args.output_dir / "completeness_summary.csv")
    daily_summary_df.write_csv(args.output_dir / "daily_completeness_summary.csv")
    node_summary_df.rename({NODE_ID_COL: "node_id"}).write_csv(
        args.output_dir / "node_completeness_summary.csv"
    )
    slot_summary_df.write_csv(args.output_dir / "slot_completeness_summary.csv")
    topology_comparison_df.rename({NODE_ID_COL: "node_id"}).write_csv(
        args.output_dir / "topology_node_comparison.csv"
    )

    missing_day_indices = (
        daily_summary_df.filter(pl.col("missing_records") > 0)
        .get_column(DAY_INDEX_COL)
        .to_list()
    )
    if args.write_missing_detail:
        write_missing_details(
            missing_day_indices=missing_day_indices,
            input_files=input_files,
            observed_node_df=observed_node_df,
            node_col=args.node_col,
            time_col=args.time_col,
            slots_per_day=args.slots_per_day,
            output_dir=args.output_dir,
        )

    print("-" * 80, flush=True)
    print(f"输入分片数量: {file_count}", flush=True)
    print(f"观测节点数: {observed_node_count}", flush=True)
    print(f"拓扑节点数: {topology_node_df.height}", flush=True)
    print(f"总原始记录数: {total_raw_records}", flush=True)
    print(f"理论记录数: {total_expected_records}", flush=True)
    print(f"缺失记录数: {total_missing_records}", flush=True)
    print(f"重复记录数: {total_duplicate_records}", flush=True)
    print(f"非法流量记录数: {total_invalid_flow_count}", flush=True)
    print(f"检查是否通过: {is_passed}", flush=True)
    if topology_not_in_observed_count > 0:
        print(
            f"警告: 拓扑中有 {topology_not_in_observed_count} 个节点从未出现在节点流量数据中。",
            flush=True,
        )
    print(f"报告输出目录: {args.output_dir}", flush=True)
    if is_passed:
        print(
            "检查通过：当前节点流量数据完整，无需空间均值填补。",
            flush=True,
        )
    else:
        print(
            "检查未通过：请查看 reports 中的 daily/node/slot/topology 明细。",
            flush=True,
        )
    print("=" * 80, flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
