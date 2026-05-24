"""比较不同日期类型曲线构造方法的傅里叶拟合与聚类效果。"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis_scripts.fit_node_flow_daily_curve import (  # noqa: E402
    DAY_SLOT_COL,
    FLOW_COL,
    INPUT_PATTERN,
    NODE_COL,
    SLOTS_PER_DAY,
    TIME_COL,
    build_fourier_design_matrix,
)


INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
OUTPUT_ROOT = ROOT_DIR / "data" / "analysis" / "date_type_curve_method_comparison"
COMPARISON_DIR = OUTPUT_ROOT / "comparison"

METHOD_M0 = "M0_original_fourier"
METHOD_M1 = "M1_three_date_type_curves"
METHOD_M2 = "M2_weighted_single_curve"
METHOD_M3 = "M3_multiplicative_corrected_single_curve"
METHODS = [METHOD_M0, METHOD_M1, METHOD_M2, METHOD_M3]

METHOD_COL = "method"
DATE_TYPE_COL = "date_type"
DAY_INDEX_COL = "day_index"
OBSERVED_FLOW_COL = "observed_flow"
FITTED_FLOW_COL = "fitted_flow"
RESIDUAL_COL = "residual"
AVG_FLOW_COL = "平均路口车流量"
CORRECTED_AVG_FLOW_COL = "平均校正路口车流量"
RAW_CURVE_COL = "raw_curve_flow"
NORMALIZED_FLOW_COL = "normalized_flow"
SAMPLE_COUNT_COL = "样本数"
CLUSTER_COL = "cluster_id"
SELECTION_SCORE_COL = "_selection_score"

FOURIER_HARMONICS = 6
K_VALUES = [3, 4, 5, 6]
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 20
EPSILON = 1e-6
LOW_R2_THRESHOLD = 0.85
PCA_SAMPLE_LIMIT = 6000
SILHOUETTE_SAMPLE_LIMIT = 8000
FIG_DPI = 200

DATE_TYPES = ["workday", "weekend", "holiday"]
HOLIDAY_DATES = {
    "2017-04-02",
    "2017-04-03",
    "2017-04-04",
    "2017-04-29",
    "2017-04-30",
    "2017-05-01",
    "2017-05-28",
    "2017-05-29",
    "2017-05-30",
}
WORKDAY_OVERRIDE_DATES = {"2017-04-01", "2017-05-27"}

MORNING_PEAK_SLOTS = slice(24, 40)  # 06:00-10:00
EVENING_PEAK_SLOTS = slice(64, 80)  # 16:00-20:00


@dataclass
class MethodArtifacts:
    """保存单一方法的关键结果，用于最终比较。"""

    method_name: str
    best_k: int
    fit_summary_df: pl.DataFrame
    cluster_metric_df: pl.DataFrame
    representative_curve_df: pl.DataFrame
    center_df: pl.DataFrame
    standardized_features: np.ndarray
    cluster_labels: np.ndarray


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="比较日期类型处理方案下的路口日内曲线拟合与聚类效果。"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        choices=METHODS,
        help="待运行的方法列表。",
    )
    parser.add_argument(
        "--workday-weight",
        type=float,
        default=0.60,
        help="M2 中工作日权重，默认 0.60。",
    )
    parser.add_argument(
        "--weekend-weight",
        type=float,
        default=0.25,
        help="M2 中普通周末权重，默认 0.25。",
    )
    parser.add_argument(
        "--holiday-weight",
        type=float,
        default=0.15,
        help="M2 中节假日权重，默认 0.15。",
    )
    parser.add_argument(
        "--node-sample-size",
        type=int,
        default=0,
        help="仅抽样部分节点进行验证；0 表示使用全部节点。",
    )
    return parser.parse_args()


def configure_fonts() -> None:
    """配置中文字体，避免图表中文乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    """归一化 M2 使用的日期类型权重。"""
    total_weight = float(sum(raw_weights.values()))
    if total_weight <= 0:
        raise ValueError("M2 的日期类型权重之和必须大于 0")
    return {key: float(value / total_weight) for key, value in raw_weights.items()}


def build_calendar_lookup() -> Tuple[List[str], Dict[int, str], Dict[int, str]]:
    """构建 61 天日期与日期类型映射。"""
    start_date = date(2017, 4, 1)
    all_dates: List[str] = []
    day_index_to_date: Dict[int, str] = {}
    day_index_to_type: Dict[int, str] = {}

    for day_index in range(61):
        current_date = start_date + timedelta(days=day_index)
        date_str = current_date.isoformat()
        all_dates.append(date_str)
        day_index_to_date[day_index] = date_str

        if date_str in HOLIDAY_DATES:
            date_type = "holiday"
        elif date_str in WORKDAY_OVERRIDE_DATES:
            date_type = "workday"
        elif current_date.weekday() >= 5:
            date_type = "weekend"
        else:
            date_type = "workday"
        day_index_to_type[day_index] = date_type

    return all_dates, day_index_to_date, day_index_to_type


def build_date_type_expr(day_index_to_type: Dict[int, str]) -> pl.Expr:
    """构造 Polars 中的日期类型表达式。"""
    holiday_indices = [idx for idx, kind in day_index_to_type.items() if kind == "holiday"]
    weekend_indices = [idx for idx, kind in day_index_to_type.items() if kind == "weekend"]
    return (
        pl.when(pl.col(DAY_INDEX_COL).is_in(holiday_indices))
        .then(pl.lit("holiday"))
        .when(pl.col(DAY_INDEX_COL).is_in(weekend_indices))
        .then(pl.lit("weekend"))
        .otherwise(pl.lit("workday"))
        .alias(DATE_TYPE_COL)
    )


def build_base_lazy_frame(day_index_to_type: Dict[int, str]) -> pl.LazyFrame:
    """读取输入分片，生成带日期类型标记的懒加载数据。"""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    parquet_files = sorted(INPUT_DIR.glob(INPUT_PATTERN))
    if not parquet_files:
        raise FileNotFoundError(f"未找到输入分片: {INPUT_DIR / INPUT_PATTERN}")
    if len(parquet_files) != len(day_index_to_type):
        raise ValueError(
            f"输入分片数量 {len(parquet_files)} 与预期日期数量 {len(day_index_to_type)} 不一致"
        )

    lazy_frames = []
    for day_index, file_path in enumerate(parquet_files):
        date_type = day_index_to_type[day_index]
        lazy_frames.append(
            pl.scan_parquet(file_path)
            .select([
                pl.col(NODE_COL).cast(pl.Int64, strict=False),
                pl.col(TIME_COL).cast(pl.Int64, strict=False),
                pl.col(FLOW_COL).cast(pl.Float64, strict=False),
            ])
            .with_columns([
                pl.lit(day_index).cast(pl.Int32).alias(DAY_INDEX_COL),
                pl.lit(date_type).alias(DATE_TYPE_COL),
            ])
        )
    return (
        pl.concat(lazy_frames, how="vertical")
        .filter(
            pl.col(NODE_COL).is_not_null()
            & pl.col(TIME_COL).is_not_null()
            & pl.col(FLOW_COL).is_not_null()
            & (pl.col(FLOW_COL) >= 0)
        )
        .with_columns([
            (pl.col(TIME_COL) % SLOTS_PER_DAY).cast(pl.Int16).alias(DAY_SLOT_COL),
        ])
    )


def sample_base_lazy_frame(base_lf: pl.LazyFrame, sample_size: int) -> pl.LazyFrame:
    """按固定随机种子抽样节点，便于快速验证脚本流程。"""
    if sample_size <= 0:
        return base_lf

    node_ids = (
        base_lf.select(pl.col(NODE_COL).unique().sort())
        .collect()
        .get_column(NODE_COL)
        .to_numpy()
    )
    if sample_size >= node_ids.shape[0]:
        return base_lf

    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    sampled_node_ids = np.sort(rng.choice(node_ids, size=sample_size, replace=False))
    return base_lf.filter(pl.col(NODE_COL).is_in(sampled_node_ids.tolist()))


def collect_type_profiles(base_lf: pl.LazyFrame) -> pl.DataFrame:
    """构造按节点-日期类型-日内时间段聚合的平均曲线。"""
    return (
        base_lf.group_by([NODE_COL, DATE_TYPE_COL, DAY_SLOT_COL])
        .agg([
            pl.col(FLOW_COL).mean().alias(AVG_FLOW_COL),
            pl.len().alias(SAMPLE_COUNT_COL),
        ])
        .sort([NODE_COL, DATE_TYPE_COL, DAY_SLOT_COL])
        .collect()
    )


def collect_m0_profiles(base_lf: pl.LazyFrame) -> pl.DataFrame:
    """构造 M0 基线方法的单曲线。"""
    return (
        base_lf.group_by([NODE_COL, DAY_SLOT_COL])
        .agg([
            pl.col(FLOW_COL).mean().alias(AVG_FLOW_COL),
            pl.len().alias(SAMPLE_COUNT_COL),
        ])
        .sort([NODE_COL, DAY_SLOT_COL])
        .collect()
    )


def build_m2_profiles_from_type_profiles(
    type_profiles_df: pl.DataFrame,
    weights: Dict[str, float],
) -> pl.DataFrame:
    """由日期类型曲线构造加权单曲线。"""
    weight_df = pl.DataFrame({
        DATE_TYPE_COL: DATE_TYPES,
        "_weight": [weights[date_type] for date_type in DATE_TYPES],
    })
    return (
        type_profiles_df.join(weight_df, on=DATE_TYPE_COL, how="inner")
        .group_by([NODE_COL, DAY_SLOT_COL])
        .agg([
            (pl.col(AVG_FLOW_COL) * pl.col("_weight")).sum().alias("_weighted_sum"),
            pl.col("_weight").sum().alias("_weight_sum"),
            pl.col(SAMPLE_COUNT_COL).sum().alias(SAMPLE_COUNT_COL),
        ])
        .with_columns(
            (pl.col("_weighted_sum") / (pl.col("_weight_sum") + EPSILON)).alias(AVG_FLOW_COL)
        )
        .select([NODE_COL, DAY_SLOT_COL, AVG_FLOW_COL, SAMPLE_COUNT_COL])
        .sort([NODE_COL, DAY_SLOT_COL])
    )


def collect_m3_profiles(base_lf: pl.LazyFrame) -> pl.DataFrame:
    """构造日期类型乘性校正后的单曲线。"""
    global_type_mean_lf = (
        base_lf.group_by([DATE_TYPE_COL, DAY_SLOT_COL])
        .agg(pl.col(FLOW_COL).mean().alias("_type_slot_mean"))
    )
    global_slot_mean_lf = (
        base_lf.group_by(DAY_SLOT_COL)
        .agg(pl.col(FLOW_COL).mean().alias("_overall_slot_mean"))
    )
    factor_lf = (
        global_type_mean_lf.join(global_slot_mean_lf, on=DAY_SLOT_COL, how="inner")
        .with_columns(
            (
                pl.col("_type_slot_mean") / (pl.col("_overall_slot_mean") + EPSILON)
            ).alias("_global_factor")
        )
        .select([DATE_TYPE_COL, DAY_SLOT_COL, "_global_factor"])
    )
    corrected_lf = (
        base_lf.join(factor_lf, on=[DATE_TYPE_COL, DAY_SLOT_COL], how="inner")
        .with_columns(
            (pl.col(FLOW_COL) / (pl.col("_global_factor") + EPSILON)).alias("_corrected_flow")
        )
        .group_by([NODE_COL, DAY_SLOT_COL])
        .agg([
            pl.col("_corrected_flow").mean().alias(CORRECTED_AVG_FLOW_COL),
            pl.len().alias(SAMPLE_COUNT_COL),
        ])
        .sort([NODE_COL, DAY_SLOT_COL])
    )
    return corrected_lf.collect()


def ensure_complete_curve(day_slots: np.ndarray) -> bool:
    """检查曲线是否完整覆盖 96 个日内时间段。"""
    expected_slots = np.arange(SLOTS_PER_DAY, dtype=np.int64)
    return len(day_slots) == SLOTS_PER_DAY and np.array_equal(day_slots, expected_slots)


def compute_shape_metrics(observed: np.ndarray, raw_fitted: np.ndarray) -> Dict[str, float]:
    """计算论文中常用的曲线形态摘要指标。"""
    peak_slot = int(np.argmax(observed))
    peak_value = float(np.max(observed))
    valley_value = float(np.min(observed))
    morning_peak = float(np.max(observed[MORNING_PEAK_SLOTS]))
    evening_peak = float(np.max(observed[EVENING_PEAK_SLOTS]))
    return {
        "峰谷差": float(peak_value - valley_value),
        "峰值时段": peak_slot,
        "早晚峰比": float(morning_peak / (evening_peak + EPSILON)),
        "零流量比例": float(np.mean(observed <= EPSILON)),
        "负拟合点数": int(np.sum(raw_fitted < 0.0)),
    }


def fit_curve_values(observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """对一条 96 点曲线进行傅里叶拟合并计算指标。"""
    day_slots = np.arange(SLOTS_PER_DAY, dtype=np.int64)
    design_matrix = build_fourier_design_matrix(day_slots, FOURIER_HARMONICS)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, observed, rcond=None)
    raw_fitted = design_matrix @ coefficients
    clipped_fitted = np.clip(raw_fitted, a_min=0.0, a_max=None)
    residual = observed - clipped_fitted

    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(observed - observed.mean())))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "平均流量": float(np.mean(observed)),
        "最大流量": float(np.max(observed)),
        "最小流量": float(np.min(observed)),
    }
    metrics.update(compute_shape_metrics(observed, raw_fitted))
    return coefficients, clipped_fitted, residual, metrics


def fit_profile_dataframe(
    profile_df: pl.DataFrame,
    method_name: str,
    observed_col: str,
    include_date_type: bool,
) -> Tuple[pl.DataFrame, pl.DataFrame, int, int]:
    """将日内平均曲线拟合为傅里叶曲线。"""
    group_keys = [NODE_COL] + ([DATE_TYPE_COL] if include_date_type else [])
    coeff_records: List[Dict[str, float]] = []
    fitted_frames: List[pl.DataFrame] = []
    success_count = 0
    skipped_count = 0

    sorted_df = profile_df.sort(group_keys + [DAY_SLOT_COL])
    for curve_df in sorted_df.partition_by(group_keys, maintain_order=True):
        node_id = int(curve_df.item(0, NODE_COL))
        day_slots = curve_df.get_column(DAY_SLOT_COL).to_numpy().astype(np.int64)
        if not ensure_complete_curve(day_slots):
            skipped_count += 1
            continue

        observed = curve_df.get_column(observed_col).to_numpy().astype(np.float64)
        coefficients, fitted_flow, residual, metrics = fit_curve_values(observed)
        coeff_record: Dict[str, float] = {
            NODE_COL: node_id,
            METHOD_COL: method_name,
            "a0": float(coefficients[0]),
        }
        if include_date_type:
            coeff_record[DATE_TYPE_COL] = str(curve_df.item(0, DATE_TYPE_COL))

        coeff_index = 1
        for harmonic in range(1, FOURIER_HARMONICS + 1):
            coeff_record[f"a{harmonic}"] = float(coefficients[coeff_index])
            coeff_record[f"b{harmonic}"] = float(coefficients[coeff_index + 1])
            coeff_index += 2
        coeff_record.update(metrics)
        coeff_records.append(coeff_record)

        fitted_records = {
            NODE_COL: np.full(SLOTS_PER_DAY, node_id, dtype=np.int64),
            METHOD_COL: np.full(SLOTS_PER_DAY, method_name, dtype=object),
            DAY_SLOT_COL: day_slots,
            OBSERVED_FLOW_COL: observed,
            FITTED_FLOW_COL: fitted_flow,
            RESIDUAL_COL: residual,
        }
        if include_date_type:
            fitted_records[DATE_TYPE_COL] = np.full(
                SLOTS_PER_DAY,
                str(curve_df.item(0, DATE_TYPE_COL)),
                dtype=object,
            )
        fitted_frames.append(pl.DataFrame(fitted_records))
        success_count += 1

    if not coeff_records or not fitted_frames:
        raise ValueError(f"{method_name} 没有任何曲线成功拟合")

    coeff_df = pl.DataFrame(coeff_records).sort(group_keys)
    fitted_df = pl.concat(fitted_frames, how="vertical").sort(group_keys + [DAY_SLOT_COL])
    return coeff_df, fitted_df, success_count, skipped_count


def build_single_curve_metadata(coeff_df: pl.DataFrame) -> pl.DataFrame:
    """构建单曲线方法的节点级元数据。"""
    return coeff_df.select([
        NODE_COL,
        METHOD_COL,
        "R2",
        "RMSE",
        "MAE",
        "平均流量",
        "最大流量",
        "最小流量",
    ]).sort(NODE_COL)


def build_m1_node_metadata(coeff_df: pl.DataFrame) -> pl.DataFrame:
    """将 M1 的三条曲线指标汇总为节点级元数据。"""
    return (
        coeff_df.group_by([NODE_COL, METHOD_COL])
        .agg([
            pl.col("R2").mean().alias("R2"),
            pl.col("RMSE").mean().alias("RMSE"),
            pl.col("MAE").mean().alias("MAE"),
            pl.col("平均流量").mean().alias("平均流量"),
            pl.col("最大流量").max().alias("最大流量"),
            pl.col("最小流量").min().alias("最小流量"),
        ])
        .sort(NODE_COL)
    )


def build_representative_curves(
    profile_df: pl.DataFrame,
    method_name: str,
    observed_col: str,
    include_date_type: bool,
) -> pl.DataFrame:
    """构建聚类中心和可视化使用的节点代表曲线。"""
    if include_date_type:
        curve_df = (
            profile_df.group_by([NODE_COL, DAY_SLOT_COL])
            .agg(pl.col(observed_col).mean().alias(RAW_CURVE_COL))
            .sort([NODE_COL, DAY_SLOT_COL])
        )
    else:
        curve_df = (
            profile_df.select([NODE_COL, DAY_SLOT_COL, pl.col(observed_col).alias(RAW_CURVE_COL)])
            .sort([NODE_COL, DAY_SLOT_COL])
        )

    normalized_frames: List[pl.DataFrame] = []
    for node_df in curve_df.partition_by(NODE_COL, maintain_order=True):
        day_slots = node_df.get_column(DAY_SLOT_COL).to_numpy().astype(np.int64)
        if not ensure_complete_curve(day_slots):
            continue
        raw_values = node_df.get_column(RAW_CURVE_COL).to_numpy().astype(np.float64)
        normalized_values = raw_values / (float(np.mean(raw_values)) + EPSILON)
        normalized_frames.append(
            pl.DataFrame({
                NODE_COL: np.full(SLOTS_PER_DAY, int(node_df.item(0, NODE_COL)), dtype=np.int64),
                METHOD_COL: np.full(SLOTS_PER_DAY, method_name, dtype=object),
                DAY_SLOT_COL: day_slots,
                RAW_CURVE_COL: raw_values,
                NORMALIZED_FLOW_COL: normalized_values,
            })
        )
    if not normalized_frames:
        raise ValueError(f"{method_name} 没有可用于聚类中心展示的代表曲线")
    return pl.concat(normalized_frames, how="vertical").sort([NODE_COL, DAY_SLOT_COL])


def build_normalized_shape_features(
    curve_values: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """根据归一化曲线构造聚类使用的形态特征。"""
    normalized_curve = curve_values / (float(np.mean(curve_values)) + EPSILON)
    design_matrix = build_fourier_design_matrix(
        np.arange(SLOTS_PER_DAY, dtype=np.int64),
        FOURIER_HARMONICS,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, normalized_curve, rcond=None)

    morning_peak = float(np.max(normalized_curve[MORNING_PEAK_SLOTS]))
    evening_peak = float(np.max(normalized_curve[EVENING_PEAK_SLOTS]))
    feature_record: Dict[str, float] = {}
    coeff_index = 1
    for harmonic in range(1, FOURIER_HARMONICS + 1):
        feature_record[f"{prefix}a{harmonic}"] = float(coefficients[coeff_index])
        feature_record[f"{prefix}b{harmonic}"] = float(coefficients[coeff_index + 1])
        coeff_index += 2
    feature_record[f"{prefix}峰谷差"] = float(np.max(normalized_curve) - np.min(normalized_curve))
    feature_record[f"{prefix}峰值时段"] = float(np.argmax(normalized_curve))
    feature_record[f"{prefix}早晚峰比"] = float(morning_peak / (evening_peak + EPSILON))
    return feature_record


def build_single_curve_feature_df(
    representative_curve_df: pl.DataFrame,
    method_name: str,
) -> pl.DataFrame:
    """构建单曲线方法的节点级聚类特征。"""
    feature_records: List[Dict[str, float]] = []
    for node_df in representative_curve_df.partition_by(NODE_COL, maintain_order=True):
        raw_curve = node_df.get_column(RAW_CURVE_COL).to_numpy().astype(np.float64)
        record = {NODE_COL: int(node_df.item(0, NODE_COL)), METHOD_COL: method_name}
        record.update(build_normalized_shape_features(raw_curve))
        feature_records.append(record)
    return pl.DataFrame(feature_records).sort(NODE_COL)


def build_m1_feature_df(type_profile_df: pl.DataFrame) -> pl.DataFrame:
    """构建 M1 的三日期类型拼接聚类特征。"""
    feature_by_node: Dict[int, Dict[str, float]] = {}
    presence_by_node: Dict[int, set[str]] = {}
    sorted_df = type_profile_df.sort([NODE_COL, DATE_TYPE_COL, DAY_SLOT_COL])

    for curve_df in sorted_df.partition_by([NODE_COL, DATE_TYPE_COL], maintain_order=True):
        day_slots = curve_df.get_column(DAY_SLOT_COL).to_numpy().astype(np.int64)
        if not ensure_complete_curve(day_slots):
            continue
        node_id = int(curve_df.item(0, NODE_COL))
        date_type = str(curve_df.item(0, DATE_TYPE_COL))
        raw_curve = curve_df.get_column(AVG_FLOW_COL).to_numpy().astype(np.float64)
        feature_record = feature_by_node.setdefault(
            node_id,
            {NODE_COL: node_id, METHOD_COL: METHOD_M1},
        )
        feature_record.update(build_normalized_shape_features(raw_curve, prefix=f"{date_type}_"))
        presence_by_node.setdefault(node_id, set()).add(date_type)

    complete_records = [
        feature_record
        for node_id, feature_record in feature_by_node.items()
        if presence_by_node.get(node_id, set()) == set(DATE_TYPES)
    ]
    if not complete_records:
        raise ValueError("M1 没有任何节点同时具备三类日期的完整 96 点曲线")
    return pl.DataFrame(complete_records).sort(NODE_COL)


def evaluate_kmeans_models(
    standardized_features: np.ndarray,
    k_values: Sequence[int],
) -> Tuple[pl.DataFrame, int]:
    """评估不同 k 下的聚类指标，并用多指标综合规则选择最佳 k。"""
    if standardized_features.shape[0] <= max(k_values):
        raise ValueError("样本数不足，无法评估给定的聚类数范围")

    metric_records: List[Dict[str, float]] = []
    for k in k_values:
        model = KMeans(
            n_clusters=int(k),
            random_state=DEFAULT_RANDOM_STATE,
            n_init=DEFAULT_N_INIT,
        )
        labels = model.fit_predict(standardized_features)
        _, silhouette_vals = sample_silhouette_values(standardized_features, labels)
        counts = np.bincount(labels, minlength=int(k))
        metric_records.append({
            "k": int(k),
            "silhouette_score": float(np.mean(silhouette_vals)),
            "calinski_harabasz_score": float(
                calinski_harabasz_score(standardized_features, labels)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(standardized_features, labels)
            ),
            "inertia": float(model.inertia_),
            "min_cluster_ratio": float(counts.min() / standardized_features.shape[0]),
            "max_cluster_ratio": float(counts.max() / standardized_features.shape[0]),
            "negative_silhouette_ratio": float(np.mean(silhouette_vals < 0.0)),
        })

    metric_df = pl.DataFrame(metric_records).sort("k")
    metric_records = metric_df.to_dicts()

    def normalized_score(values: List[float], higher_is_better: bool) -> List[float]:
        min_value = min(values)
        max_value = max(values)
        if math.isclose(max_value, min_value):
            return [1.0] * len(values)
        scores = [(value - min_value) / (max_value - min_value) for value in values]
        if higher_is_better:
            return scores
        return [1.0 - score for score in scores]

    silhouette_scores = normalized_score(
        [record["silhouette_score"] for record in metric_records],
        higher_is_better=True,
    )
    ch_scores = normalized_score(
        [record["calinski_harabasz_score"] for record in metric_records],
        higher_is_better=True,
    )
    db_scores = normalized_score(
        [record["davies_bouldin_score"] for record in metric_records],
        higher_is_better=False,
    )
    balance_scores = normalized_score(
        [record["max_cluster_ratio"] for record in metric_records],
        higher_is_better=False,
    )
    negative_scores = normalized_score(
        [record["negative_silhouette_ratio"] for record in metric_records],
        higher_is_better=False,
    )
    complexity_scores = normalized_score(
        [float(record["k"]) for record in metric_records],
        higher_is_better=False,
    )

    selection_scores = []
    for idx, record in enumerate(metric_records):
        score = (
            0.30 * silhouette_scores[idx]
            + 0.20 * db_scores[idx]
            + 0.15 * balance_scores[idx]
            + 0.15 * negative_scores[idx]
            + 0.15 * ch_scores[idx]
            + 0.05 * complexity_scores[idx]
        )
        if record["max_cluster_ratio"] > 0.80:
            score -= 0.08
        selection_scores.append(score)

    scored_metric_df = metric_df.with_columns(pl.Series(SELECTION_SCORE_COL, selection_scores))
    best_metric_row = (
        scored_metric_df.sort(
            [
                SELECTION_SCORE_COL,
                "silhouette_score",
                "davies_bouldin_score",
                "negative_silhouette_ratio",
                "k",
            ],
            descending=[True, True, False, False, False],
        )
        .row(0, named=True)
    )
    best_k = int(best_metric_row["k"])
    return metric_df, best_k


def sample_silhouette_values(
    standardized_features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """对 silhouette 相关指标使用固定随机种子抽样，控制实验时间。"""
    sample_size = min(SILHOUETTE_SAMPLE_LIMIT, standardized_features.shape[0])
    if sample_size == standardized_features.shape[0]:
        sample_indices = np.arange(standardized_features.shape[0], dtype=np.int64)
    else:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
        sample_indices = np.sort(
            rng.choice(standardized_features.shape[0], size=sample_size, replace=False)
        )
    sampled_features = standardized_features[sample_indices]
    sampled_labels = labels[sample_indices]
    silhouette_vals = silhouette_samples(sampled_features, sampled_labels)
    return sample_indices, silhouette_vals


def fit_final_kmeans(
    standardized_features: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用最终 k 训练 KMeans 并返回标签和抽样 silhouette 结果。"""
    model = KMeans(
        n_clusters=k,
        random_state=DEFAULT_RANDOM_STATE,
        n_init=DEFAULT_N_INIT,
    )
    labels = model.fit_predict(standardized_features)
    sample_indices, silhouette_vals = sample_silhouette_values(standardized_features, labels)
    return labels, sample_indices, silhouette_vals


def build_cluster_outputs(
    method_name: str,
    metadata_df: pl.DataFrame,
    representative_curve_df: pl.DataFrame,
    cluster_labels: np.ndarray,
    silhouette_sample_indices: np.ndarray,
    silhouette_vals: np.ndarray,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """构建聚类标签、摘要和类中心曲线结果。"""
    label_df = metadata_df.sort(NODE_COL).with_columns(
        pl.Series(CLUSTER_COL, cluster_labels.astype(np.int64))
    ).select([
        NODE_COL,
        METHOD_COL,
        CLUSTER_COL,
        "R2",
        "RMSE",
        "MAE",
        "平均流量",
        "最大流量",
        "最小流量",
    ]).sort([CLUSTER_COL, NODE_COL])

    summary_df = (
        label_df.group_by([METHOD_COL, CLUSTER_COL])
        .agg([
            pl.len().alias("节点数"),
            pl.col("平均流量").mean().alias("平均流量均值"),
            pl.col("R2").mean().alias("平均R2"),
            pl.col("RMSE").mean().alias("平均RMSE"),
            pl.col("MAE").mean().alias("平均MAE"),
        ])
        .sort([METHOD_COL, CLUSTER_COL])
    )
    sampled_cluster_ids = cluster_labels[silhouette_sample_indices]
    sampled_negative_df = (
        pl.DataFrame({
            METHOD_COL: np.full(sampled_cluster_ids.shape[0], method_name, dtype=object),
            CLUSTER_COL: sampled_cluster_ids.astype(np.int64),
            "_silhouette": silhouette_vals.astype(np.float64),
        })
        .group_by([METHOD_COL, CLUSTER_COL])
        .agg((pl.col("_silhouette") < 0.0).mean().alias("negative_silhouette_ratio"))
        .sort([METHOD_COL, CLUSTER_COL])
    )
    summary_df = (
        summary_df.join(sampled_negative_df, on=[METHOD_COL, CLUSTER_COL], how="left")
        .with_columns(pl.col("negative_silhouette_ratio").fill_null(0.0))
    )

    center_df = (
        representative_curve_df.join(
            label_df.select([NODE_COL, CLUSTER_COL]),
            on=NODE_COL,
            how="inner",
        )
        .group_by([METHOD_COL, CLUSTER_COL, DAY_SLOT_COL])
        .agg([
            pl.col(NORMALIZED_FLOW_COL).mean().alias("类平均归一化流量"),
            pl.col(RAW_CURVE_COL).mean().alias("类平均原始流量"),
        ])
        .sort([METHOD_COL, CLUSTER_COL, DAY_SLOT_COL])
    )
    return label_df, summary_df, center_df


def save_method_outputs(
    method_dir: Path,
    profile_df: pl.DataFrame,
    fitted_df: pl.DataFrame,
    coeff_df: pl.DataFrame,
    cluster_metric_df: pl.DataFrame,
    cluster_label_df: pl.DataFrame,
    cluster_summary_df: pl.DataFrame,
    cluster_center_df: pl.DataFrame,
) -> None:
    """保存单个方法的所有 parquet 输出。"""
    method_dir.mkdir(parents=True, exist_ok=True)
    profile_df.write_parquet(method_dir / "daily_profiles.parquet", compression="snappy")
    fitted_df.write_parquet(method_dir / "fitted_curves.parquet", compression="snappy")
    coeff_df.write_parquet(method_dir / "curve_coefficients.parquet", compression="snappy")
    cluster_metric_df.write_parquet(method_dir / "cluster_metrics.parquet", compression="snappy")
    cluster_label_df.write_parquet(method_dir / "cluster_labels.parquet", compression="snappy")
    cluster_summary_df.write_parquet(method_dir / "cluster_summary.parquet", compression="snappy")
    cluster_center_df.write_parquet(method_dir / "cluster_centers.parquet", compression="snappy")


def build_fit_method_summary(method_name: str, node_metadata_df: pl.DataFrame) -> pl.DataFrame:
    """汇总单个方法的拟合质量。"""
    return pl.DataFrame({
        METHOD_COL: [method_name],
        "平均RMSE": [float(node_metadata_df.get_column("RMSE").mean())],
        "平均MAE": [float(node_metadata_df.get_column("MAE").mean())],
        "平均R2": [float(node_metadata_df.get_column("R2").mean())],
        "中位数R2": [float(node_metadata_df.get_column("R2").median())],
        "低R2节点比例": [
            float((node_metadata_df.get_column("R2") < LOW_R2_THRESHOLD).mean())
        ],
    })


def build_cluster_method_summary(
    method_name: str,
    best_k: int,
    cluster_metric_df: pl.DataFrame,
) -> pl.DataFrame:
    """提取方法在最优 k 下的聚类质量摘要。"""
    selected_metric = cluster_metric_df.filter(pl.col("k") == best_k)
    return selected_metric.with_columns(pl.lit(method_name).alias(METHOD_COL)).select([
        METHOD_COL,
        "k",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "inertia",
        "min_cluster_ratio",
        "max_cluster_ratio",
        "negative_silhouette_ratio",
    ])


def build_method_comparison_table(
    fit_summary_df: pl.DataFrame,
    cluster_summary_df: pl.DataFrame,
) -> pl.DataFrame:
    """合并方法级拟合与聚类摘要，生成论文总表。"""
    return (
        fit_summary_df.join(cluster_summary_df, on=METHOD_COL, how="inner")
        .rename({"k": "best_k"})
        .sort(METHOD_COL)
    )


def plot_method_cluster_metric_comparison(
    cluster_summary_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """绘制四种方法在最优 k 下的聚类指标对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = cluster_summary_df.get_column(METHOD_COL).to_list()
    x = np.arange(len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_specs = [
        ("silhouette_score", "Silhouette", "#4C78A8"),
        ("davies_bouldin_score", "Davies-Bouldin", "#E45756"),
        ("max_cluster_ratio", "Max Cluster Ratio", "#54A24B"),
        ("negative_silhouette_ratio", "Negative Silhouette Ratio", "#B279A2"),
    ]

    for ax, (column, title, color) in zip(axes.ravel(), metric_specs):
        values = cluster_summary_df.get_column(column).to_numpy()
        ax.bar(x, values, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("四种方法在最优 k 下的聚类质量对比", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_method_center_curve_comparison(
    artifacts: Sequence[MethodArtifacts],
    output_path: Path,
) -> None:
    """绘制四种方法的最终聚类中心曲线对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

    for ax, artifact in zip(axes.ravel(), artifacts):
        method_center_df = artifact.center_df.sort([CLUSTER_COL, DAY_SLOT_COL])
        for cluster_df in method_center_df.partition_by(CLUSTER_COL, maintain_order=True):
            cluster_id = int(cluster_df.item(0, CLUSTER_COL))
            ax.plot(
                cluster_df.get_column(DAY_SLOT_COL).to_numpy(),
                cluster_df.get_column("类平均归一化流量").to_numpy(),
                linewidth=2.0,
                label=f"Cluster {cluster_id}",
            )
        ax.set_title(f"{artifact.method_name} (k={artifact.best_k})")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0, SLOTS_PER_DAY - 1)
        ax.set_xticks(np.arange(0, SLOTS_PER_DAY + 1, 12))
        ax.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("日内时间段")
    for ax in axes[:, 0]:
        ax.set_ylabel("类平均归一化流量")

    fig.suptitle("四种方法的最终聚类中心曲线对比", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_method_pca_comparison(
    artifacts: Sequence[MethodArtifacts],
    output_path: Path,
) -> None:
    """绘制四种方法的 PCA 聚类散点图对比。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    for ax, artifact in zip(axes.ravel(), artifacts):
        pca = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
        embedding = pca.fit_transform(artifact.standardized_features)
        labels = artifact.cluster_labels
        if embedding.shape[0] > PCA_SAMPLE_LIMIT:
            sampled_indices = np.sort(rng.choice(embedding.shape[0], PCA_SAMPLE_LIMIT, replace=False))
            embedding = embedding[sampled_indices]
            labels = labels[sampled_indices]
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab10",
            s=10,
            alpha=0.65,
            edgecolors="none",
        )
        ax.set_title(f"{artifact.method_name} (k={artifact.best_k})")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, linestyle="--", alpha=0.30)
        legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
        ax.add_artist(legend)

    fig.suptitle("四种方法的 PCA 聚类散点图对比", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def run_method_pipeline(
    method_name: str,
    profile_df: pl.DataFrame,
    observed_col: str,
    include_date_type: bool,
    method_dir: Path,
    feature_builder: str,
) -> MethodArtifacts:
    """执行单个方法的拟合、聚类和输出流程。"""
    coeff_df, fitted_df, success_count, skipped_count = fit_profile_dataframe(
        profile_df=profile_df,
        method_name=method_name,
        observed_col=observed_col,
        include_date_type=include_date_type,
    )
    print(f"{method_name}: 成功拟合曲线数 {success_count}，跳过 {skipped_count}")

    if method_name == METHOD_M1:
        node_metadata_df = build_m1_node_metadata(coeff_df)
        representative_curve_df = build_representative_curves(
            profile_df,
            method_name=method_name,
            observed_col=observed_col,
            include_date_type=True,
        )
        feature_df = build_m1_feature_df(profile_df)
    else:
        node_metadata_df = build_single_curve_metadata(coeff_df)
        representative_curve_df = build_representative_curves(
            profile_df,
            method_name=method_name,
            observed_col=observed_col,
            include_date_type=False,
        )
        feature_df = build_single_curve_feature_df(representative_curve_df, method_name)

    common_node_ids = sorted(
        set(node_metadata_df.get_column(NODE_COL).to_list())
        & set(representative_curve_df.get_column(NODE_COL).unique().to_list())
        & set(feature_df.get_column(NODE_COL).to_list())
    )
    if not common_node_ids:
        raise ValueError(f"{method_name} 没有可用于聚类的公共节点集合")

    node_metadata_df = node_metadata_df.filter(pl.col(NODE_COL).is_in(common_node_ids))
    representative_curve_df = representative_curve_df.filter(pl.col(NODE_COL).is_in(common_node_ids))
    feature_df = feature_df.filter(pl.col(NODE_COL).is_in(common_node_ids)).sort(NODE_COL)
    node_metadata_df = node_metadata_df.sort(NODE_COL)
    representative_curve_df = representative_curve_df.sort([NODE_COL, DAY_SLOT_COL])

    feature_columns = [
        column
        for column in feature_df.columns
        if column not in {NODE_COL, METHOD_COL}
    ]
    feature_matrix = feature_df.select(feature_columns).to_numpy().astype(np.float64)
    standardized_features = StandardScaler().fit_transform(feature_matrix)
    cluster_metric_df, best_k = evaluate_kmeans_models(standardized_features, K_VALUES)
    cluster_metric_df = cluster_metric_df.with_columns(pl.lit(method_name).alias(METHOD_COL)).select([
        METHOD_COL,
        "k",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "inertia",
        "min_cluster_ratio",
        "max_cluster_ratio",
        "negative_silhouette_ratio",
    ])
    cluster_labels, silhouette_sample_indices, silhouette_vals = fit_final_kmeans(
        standardized_features,
        best_k,
    )
    cluster_label_df, cluster_summary_df, center_df = build_cluster_outputs(
        method_name=method_name,
        metadata_df=node_metadata_df,
        representative_curve_df=representative_curve_df,
        cluster_labels=cluster_labels,
        silhouette_sample_indices=silhouette_sample_indices,
        silhouette_vals=silhouette_vals,
    )

    profile_output_df = profile_df.with_columns(pl.lit(method_name).alias(METHOD_COL))
    output_columns = [NODE_COL, METHOD_COL]
    if include_date_type:
        output_columns.append(DATE_TYPE_COL)
    output_columns.extend([DAY_SLOT_COL, observed_col, SAMPLE_COUNT_COL])
    profile_output_df = profile_output_df.select(output_columns).sort(output_columns[:-2] + [DAY_SLOT_COL])

    save_method_outputs(
        method_dir=method_dir,
        profile_df=profile_output_df,
        fitted_df=fitted_df,
        coeff_df=coeff_df,
        cluster_metric_df=cluster_metric_df,
        cluster_label_df=cluster_label_df,
        cluster_summary_df=cluster_summary_df,
        cluster_center_df=center_df,
    )

    return MethodArtifacts(
        method_name=method_name,
        best_k=best_k,
        fit_summary_df=build_fit_method_summary(method_name, node_metadata_df),
        cluster_metric_df=build_cluster_method_summary(method_name, best_k, cluster_metric_df),
        representative_curve_df=representative_curve_df,
        center_df=center_df,
        standardized_features=standardized_features,
        cluster_labels=cluster_labels,
    )


def save_comparison_outputs(artifacts: Sequence[MethodArtifacts]) -> None:
    """保存四种方法的总体比较结果和图片。"""
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    fit_summary_df = pl.concat([artifact.fit_summary_df for artifact in artifacts], how="vertical").sort(METHOD_COL)
    cluster_summary_df = pl.concat(
        [artifact.cluster_metric_df for artifact in artifacts],
        how="vertical",
    ).sort(METHOD_COL)
    comparison_table_df = build_method_comparison_table(fit_summary_df, cluster_summary_df)

    fit_summary_df.write_parquet(
        COMPARISON_DIR / "method_fit_metrics_summary.parquet",
        compression="snappy",
    )
    cluster_summary_df.write_parquet(
        COMPARISON_DIR / "method_cluster_metrics_summary.parquet",
        compression="snappy",
    )
    comparison_table_df.write_csv(COMPARISON_DIR / "method_comparison_table.csv")

    plot_method_cluster_metric_comparison(
        cluster_summary_df,
        COMPARISON_DIR / "method_cluster_metric_comparison.png",
    )
    plot_method_center_curve_comparison(
        artifacts,
        COMPARISON_DIR / "method_center_curve_comparison.png",
    )
    plot_method_pca_comparison(
        artifacts,
        COMPARISON_DIR / "method_pca_comparison.png",
    )


def main() -> None:
    """主函数。"""
    args = parse_args()
    configure_fonts()
    weights = normalize_weights({
        "workday": args.workday_weight,
        "weekend": args.weekend_weight,
        "holiday": args.holiday_weight,
    })
    _, _, day_index_to_type = build_calendar_lookup()

    print("=" * 80)
    print("日期类型曲线构造与聚类方法对比实验")
    print("=" * 80)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"统一傅里叶阶数: {FOURIER_HARMONICS}")
    print(f"KMeans 搜索范围: {K_VALUES}")
    print(f"M2 权重: {weights}")

    base_lf = build_base_lazy_frame(day_index_to_type)
    base_lf = sample_base_lazy_frame(base_lf, args.node_sample_size)
    if args.node_sample_size > 0:
        print(f"节点抽样数: {args.node_sample_size}")
    artifacts: List[MethodArtifacts] = []
    type_profiles_df: pl.DataFrame | None = None

    for method_name in args.methods:
        print("-" * 80)
        print(f"开始处理: {method_name}")
        method_dir = OUTPUT_ROOT / method_name

        if method_name == METHOD_M0:
            profile_df = collect_m0_profiles(base_lf)
            artifacts.append(
                run_method_pipeline(
                    method_name=method_name,
                    profile_df=profile_df,
                    observed_col=AVG_FLOW_COL,
                    include_date_type=False,
                    method_dir=method_dir,
                    feature_builder="single",
                )
            )
        elif method_name == METHOD_M1:
            if type_profiles_df is None:
                type_profiles_df = collect_type_profiles(base_lf)
            artifacts.append(
                run_method_pipeline(
                    method_name=method_name,
                    profile_df=type_profiles_df,
                    observed_col=AVG_FLOW_COL,
                    include_date_type=True,
                    method_dir=method_dir,
                    feature_builder="m1",
                )
            )
        elif method_name == METHOD_M2:
            if type_profiles_df is None:
                type_profiles_df = collect_type_profiles(base_lf)
            profile_df = build_m2_profiles_from_type_profiles(type_profiles_df, weights)
            artifacts.append(
                run_method_pipeline(
                    method_name=method_name,
                    profile_df=profile_df,
                    observed_col=AVG_FLOW_COL,
                    include_date_type=False,
                    method_dir=method_dir,
                    feature_builder="single",
                )
            )
        elif method_name == METHOD_M3:
            profile_df = collect_m3_profiles(base_lf)
            artifacts.append(
                run_method_pipeline(
                    method_name=method_name,
                    profile_df=profile_df,
                    observed_col=CORRECTED_AVG_FLOW_COL,
                    include_date_type=False,
                    method_dir=method_dir,
                    feature_builder="single",
                )
            )
        else:
            raise ValueError(f"未知方法: {method_name}")

    save_comparison_outputs(artifacts)
    print("-" * 80)
    print("全部方法处理完成")
    for artifact in artifacts:
        print(f"{artifact.method_name}: 最优 k = {artifact.best_k}")
    print(f"比较输出目录: {COMPARISON_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
