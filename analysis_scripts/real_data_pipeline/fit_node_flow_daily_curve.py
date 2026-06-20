"""对每个路口节点的日内平均车流量曲线进行傅里叶拟合。"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
OUTPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_flow_curve_fit"

INPUT_PATTERN = "node_flow_chunk_*.parquet"
FITTED_OUTPUT_PATH = OUTPUT_DIR / "node_flow_fitted_daily_curves.parquet"
COEFF_OUTPUT_PATH = OUTPUT_DIR / "node_flow_curve_coefficients.parquet"

NODE_COL = "节点ID"
TIME_COL = "时间段"
FLOW_COL = "路口车流量"
DAY_SLOT_COL = "日内时间段"
AVG_FLOW_COL = "平均路口车流量"
FITTED_FLOW_COL = "拟合路口车流量"
RESIDUAL_COL = "残差"

REQUIRED_COLUMNS = [NODE_COL, TIME_COL, FLOW_COL]
SLOTS_PER_DAY = 96
DEFAULT_HARMONICS = 8


def build_fourier_design_matrix(day_slots: np.ndarray, harmonics: int) -> np.ndarray:
    """构造傅里叶基函数设计矩阵。"""
    x = day_slots.astype(np.float64) / float(SLOTS_PER_DAY)
    columns = [np.ones_like(x, dtype=np.float64)]

    for h in range(1, harmonics + 1):
        angle = 2.0 * np.pi * h * x
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))

    return np.column_stack(columns)


def ensure_required_columns(parquet_files: List[Path]) -> None:
    """检查所有输入分片是否包含必需字段。"""
    missing_by_file: Dict[str, List[str]] = {}

    for file_path in parquet_files:
        schema_names = list(pl.scan_parquet(file_path).collect_schema().names())
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in schema_names]
        if missing_cols:
            missing_by_file[file_path.name] = missing_cols

    if missing_by_file:
        detail_lines = [
            f"{filename}: 缺少字段 {missing_cols}"
            for filename, missing_cols in missing_by_file.items()
        ]
        raise ValueError("发现缺少必需字段的输入文件：\n" + "\n".join(detail_lines))


def load_and_build_daily_profile(input_dir: Path) -> Tuple[pl.DataFrame, int]:
    """读取所有输入分片，并构建按节点聚合后的 96 点日内平均流量曲线。"""
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    parquet_files = sorted(input_dir.glob(INPUT_PATTERN))
    if not parquet_files:
        raise FileNotFoundError(
            f"在目录中未找到匹配文件: {input_dir / INPUT_PATTERN}"
        )

    ensure_required_columns(parquet_files)

    lazy_frames = []
    for file_path in parquet_files:
        lazy_frames.append(
            pl.scan_parquet(file_path)
            .select([
                pl.col(NODE_COL).cast(pl.Int64, strict=False),
                pl.col(TIME_COL).cast(pl.Int64, strict=False),
                pl.col(FLOW_COL).cast(pl.Float64, strict=False),
            ])
        )

    combined_lf = pl.concat(lazy_frames, how="vertical")

    profile_df = (
        combined_lf
        .filter(
            pl.col(NODE_COL).is_not_null()
            & pl.col(TIME_COL).is_not_null()
            & pl.col(FLOW_COL).is_not_null()
            & (pl.col(FLOW_COL) >= 0)
        )
        .with_columns(
            (pl.col(TIME_COL) % SLOTS_PER_DAY)
            .cast(pl.Int64)
            .alias(DAY_SLOT_COL)
        )
        .group_by([NODE_COL, DAY_SLOT_COL])
        .agg(pl.col(FLOW_COL).mean().alias(AVG_FLOW_COL))
        .sort([NODE_COL, DAY_SLOT_COL])
        .collect()
    )

    return profile_df, len(parquet_files)


def fit_one_node_curve(
    node_id: int,
    node_df: pl.DataFrame,
    harmonics: int,
) -> Tuple[Dict[str, float], pl.DataFrame]:
    """对单个节点的 96 点日内平均流量曲线做傅里叶最小二乘拟合。"""
    sorted_df = node_df.sort(DAY_SLOT_COL)
    day_slots = sorted_df.get_column(DAY_SLOT_COL).to_numpy()

    if len(day_slots) != SLOTS_PER_DAY:
        raise ValueError(f"节点 {node_id} 的日内时间段数量不是 96")

    expected_slots = np.arange(SLOTS_PER_DAY, dtype=np.int64)
    if not np.array_equal(day_slots, expected_slots):
        raise ValueError(f"节点 {node_id} 的日内时间段不完整或未按 0-95 排列")

    observed = sorted_df.get_column(AVG_FLOW_COL).to_numpy().astype(np.float64)
    design_matrix = build_fourier_design_matrix(day_slots, harmonics)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, observed, rcond=None)

    fitted = design_matrix @ coefficients
    fitted = np.clip(fitted, a_min=0.0, a_max=None)
    residual = observed - fitted

    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(observed - observed.mean())))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    coeff_record: Dict[str, float] = {
        NODE_COL: int(node_id),
        "a0": float(coefficients[0]),
    }
    coeff_index = 1
    for h in range(1, harmonics + 1):
        coeff_record[f"a{h}"] = float(coefficients[coeff_index])
        coeff_record[f"b{h}"] = float(coefficients[coeff_index + 1])
        coeff_index += 2

    coeff_record["RMSE"] = rmse
    coeff_record["MAE"] = mae
    coeff_record["R2"] = float(r2)
    coeff_record["平均流量"] = float(np.mean(observed))
    coeff_record["最大流量"] = float(np.max(observed))
    coeff_record["最小流量"] = float(np.min(observed))

    fitted_df = pl.DataFrame({
        NODE_COL: np.full(SLOTS_PER_DAY, int(node_id), dtype=np.int64),
        DAY_SLOT_COL: day_slots.astype(np.int64),
        AVG_FLOW_COL: observed,
        FITTED_FLOW_COL: fitted,
        RESIDUAL_COL: residual,
    })

    return coeff_record, fitted_df


def fit_all_node_curves(
    profile_df: pl.DataFrame,
    harmonics: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, int, int]:
    """批量拟合所有节点的日内平均流量曲线。"""
    if profile_df.is_empty():
        raise ValueError("日内平均曲线结果为空，无法执行拟合")

    fitted_frames: List[pl.DataFrame] = []
    coeff_records: List[Dict[str, float]] = []
    success_count = 0
    skipped_count = 0

    for node_df in profile_df.partition_by(NODE_COL, maintain_order=True):
        node_id = int(node_df.item(0, NODE_COL))
        unique_slots = node_df.get_column(DAY_SLOT_COL).n_unique()

        if unique_slots < SLOTS_PER_DAY or node_df.height < SLOTS_PER_DAY:
            skipped_count += 1
            continue

        try:
            coeff_record, fitted_df = fit_one_node_curve(node_id, node_df, harmonics)
        except ValueError:
            skipped_count += 1
            continue

        coeff_records.append(coeff_record)
        fitted_frames.append(fitted_df)
        success_count += 1

    if not fitted_frames or not coeff_records:
        raise ValueError("没有任何节点成功完成曲线拟合")

    fitted_result = pl.concat(fitted_frames, how="vertical").sort([NODE_COL, DAY_SLOT_COL])
    coeff_result = pl.DataFrame(coeff_records).sort(NODE_COL)

    return fitted_result, coeff_result, success_count, skipped_count


def save_outputs(fitted_df: pl.DataFrame, coeff_df: pl.DataFrame, output_dir: Path) -> None:
    """保存拟合结果和系数结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted_df.write_parquet(FITTED_OUTPUT_PATH, compression="snappy")
    coeff_df.write_parquet(COEFF_OUTPUT_PATH, compression="snappy")


def main() -> None:
    """主函数。"""
    harmonics = DEFAULT_HARMONICS

    print("=" * 80)
    print("路口节点日内车流量傅里叶曲线拟合")
    print("=" * 80)

    profile_df, input_file_count = load_and_build_daily_profile(INPUT_DIR)
    fitted_df, coeff_df, success_count, skipped_count = fit_all_node_curves(
        profile_df=profile_df,
        harmonics=harmonics,
    )
    save_outputs(fitted_df, coeff_df, OUTPUT_DIR)

    print(f"输入文件数量: {input_file_count}")
    print(f"日内平均曲线记录数: {profile_df.height}")
    print(f"成功拟合节点数: {success_count}")
    print(f"跳过节点数: {skipped_count}")
    print(f"拟合曲线输出: {FITTED_OUTPUT_PATH}")
    print(f"傅里叶系数输出: {COEFF_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
