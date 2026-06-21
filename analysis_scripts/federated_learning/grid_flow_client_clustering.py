"""为网格化流量张量生成客户端聚类结果。

核心目标：
- 读取预处理最终产物 `data/processed/node_flow_grid/node_flow_grid_tensor.pt`；
- 复用当前联邦训练中的区域时序特征提取逻辑；
- 在候选 k 范围内评估 KMeans 聚类质量，自动推荐簇数；
- 输出“每个簇 = 一个客户端”的标签表、摘要表和 JSON 映射。

主要输出目录：
- `data/analysis/federated_learning/grid_client_clustering/`
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples

try:
    from ccn_region_client_train import DEFAULT_DATASET_PATH, extract_region_features
except ImportError:  # pragma: no cover - fallback for module-style execution
    from analysis_scripts.federated_learning.ccn_region_client_train import (
        DEFAULT_DATASET_PATH,
        extract_region_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "federated_learning" / "grid_client_clustering"
SELECTION_SCORE_COL = "selection_score"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 20
SILHOUETTE_SAMPLE_LIMIT = 8000
MIN_HARD_CLUSTER_SIZE = 3
SOFT_MIN_CLUSTER_SIZE = 10
SOFT_MIN_CLUSTER_RATIO = 0.05
HARD_MIN_CLUSTER_RATIO = 0.03
MAX_CLUSTER_RATIO_TARGET = 0.85
MAX_CLUSTER_RATIO_HARD = 0.92
TINY_CLUSTER_PENALTY_WEIGHT = 0.20
DOMINANT_CLUSTER_PENALTY_WEIGHT = 0.35
EPSILON = 1e-12
SLOTS_PER_DAY = 96
FOURIER_HARMONICS = 3
FEATURE_CLIP_LOWER_QUANTILE = 0.01
FEATURE_CLIP_UPPER_QUANTILE = 0.99


@dataclass
class ClusterConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    k_min: int = 2
    k_max: int = 10
    t_in: int = 24
    t_out: int = 1
    seed: int = DEFAULT_RANDOM_STATE
    verbose: bool = False


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> ClusterConfig:
    parser = argparse.ArgumentParser(description="为网格化流量张量生成簇=客户端的聚类结果。")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-min", type=int, default=2, help="候选最小聚类数。")
    parser.add_argument("--k-max", type=int, default=10, help="候选最大聚类数。")
    parser.add_argument("--t-in", type=int, default=24)
    parser.add_argument("--t-out", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = ClusterConfig(
        dataset_path=args.dataset_path.resolve(),
        output_dir=args.output_dir.resolve(),
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        t_in=int(args.t_in),
        t_out=int(args.t_out),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )
    configure_logging(config.verbose)
    validate_config(config)
    return config


def validate_config(config: ClusterConfig) -> None:
    if config.k_min < 2:
        raise ValueError("k_min 必须至少为 2。")
    if config.k_max < config.k_min:
        raise ValueError("k_max 必须大于等于 k_min。")
    if config.t_in <= 0 or config.t_out <= 0:
        raise ValueError("t_in 和 t_out 必须为正数。")


def load_data_tensor(dataset_path: Path) -> torch.Tensor:
    if not dataset_path.exists():
        raise FileNotFoundError(f"未找到最终训练数据集: {dataset_path}")
    tensor = torch.load(dataset_path, map_location="cpu")
    validate_data_tensor(tensor)
    return tensor


def validate_data_tensor(data_tensor: Any) -> None:
    if not isinstance(data_tensor, torch.Tensor):
        raise TypeError("预处理最终输出的数据集必须是 torch.Tensor。")
    if data_tensor.dim() != 3:
        raise ValueError(f"张量维度必须为 3，实际为 {data_tensor.dim()}。")
    channels, num_regions, total_steps = map(int, data_tensor.shape)
    if channels < 1 or num_regions < 2 or total_steps <= 1:
        raise ValueError(f"无效张量形状: {tuple(data_tensor.shape)}")
    if not torch.isfinite(data_tensor).all():
        raise ValueError("张量中存在 NaN 或 Inf。")
    if data_tensor.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"张量 dtype 必须为浮点型，实际为 {data_tensor.dtype}")


def sample_silhouette_values(features_z: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sample_size = min(SILHOUETTE_SAMPLE_LIMIT, features_z.shape[0])
    if sample_size == features_z.shape[0]:
        sample_indices = np.arange(features_z.shape[0], dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        sample_indices = np.sort(rng.choice(features_z.shape[0], size=sample_size, replace=False))
    sampled_features = features_z[sample_indices]
    sampled_labels = labels[sample_indices]
    silhouette_vals = silhouette_samples(sampled_features, sampled_labels)
    return sample_indices, silhouette_vals


def build_fourier_design_matrix(num_slots: int, harmonics: int) -> np.ndarray:
    x = np.arange(num_slots, dtype=np.float64)
    design = [np.ones_like(x)]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * math.pi * harmonic * x / num_slots
        design.append(np.sin(angle))
        design.append(np.cos(angle))
    return np.column_stack(design)


def build_normalized_shape_features(curve_values: np.ndarray, prefix: str) -> dict[str, float]:
    curve_values = np.asarray(curve_values, dtype=np.float64)
    normalized_curve = curve_values / (float(np.mean(curve_values)) + EPSILON)
    design_matrix = build_fourier_design_matrix(SLOTS_PER_DAY, FOURIER_HARMONICS)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, normalized_curve, rcond=None)

    morning_peak = float(np.max(normalized_curve[24:40]))
    evening_peak = float(np.max(normalized_curve[64:84]))
    peak_slot = int(np.argmax(normalized_curve))
    peak_angle = 2.0 * math.pi * peak_slot / SLOTS_PER_DAY
    morning_evening_log_ratio = float(math.log((morning_peak + EPSILON) / (evening_peak + EPSILON)))

    feature_record: dict[str, float] = {}
    coeff_index = 1
    for harmonic in range(1, FOURIER_HARMONICS + 1):
        feature_record[f"{prefix}a{harmonic}"] = float(coefficients[coeff_index])
        feature_record[f"{prefix}b{harmonic}"] = float(coefficients[coeff_index + 1])
        coeff_index += 2
    feature_record[f"{prefix}peak_to_valley"] = float(np.max(normalized_curve) - np.min(normalized_curve))
    feature_record[f"{prefix}peak_slot_sin"] = float(math.sin(peak_angle))
    feature_record[f"{prefix}peak_slot_cos"] = float(math.cos(peak_angle))
    feature_record[f"{prefix}morning_evening_log_ratio"] = morning_evening_log_ratio
    return feature_record


def winsorize_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    if feature_matrix.size == 0:
        return feature_matrix
    lower_bounds = np.nanquantile(feature_matrix, FEATURE_CLIP_LOWER_QUANTILE, axis=0)
    upper_bounds = np.nanquantile(feature_matrix, FEATURE_CLIP_UPPER_QUANTILE, axis=0)
    clipped_matrix = np.clip(feature_matrix, lower_bounds, upper_bounds)
    return np.nan_to_num(clipped_matrix, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        fill_value = float(np.nanmedian(values[finite_mask])) if np.any(finite_mask) else 0.0
        values = np.where(finite_mask, values, fill_value)
    values = np.clip(values, a_min=0.0, a_max=None)
    return np.log1p(values)


def build_grid_region_features(data_tensor: torch.Tensor, t_in: int, t_out: int) -> tuple[np.ndarray, np.ndarray]:
    values = data_tensor.detach().cpu().numpy().astype(np.float64)
    channels, num_regions, total_steps = values.shape
    if total_steps % SLOTS_PER_DAY != 0:
        raise ValueError(f"时间维 {total_steps} 不能被 {SLOTS_PER_DAY} 整除，无法构造完整日曲线。")
    num_days = total_steps // SLOTS_PER_DAY

    feature_records: list[dict[str, float]] = []
    sizes = np.zeros(num_regions, dtype=np.float64)
    for region_id in range(num_regions):
        primary_ts = prepare_series(values[0, region_id, :])
        aux_ts = prepare_series(values[min(1, channels - 1), region_id, :])
        valid_mask = np.isfinite(values[0, region_id, :])
        sizes[region_id] = float(max(0, int(valid_mask.sum()) - t_in - t_out))

        primary_daily = primary_ts.reshape(num_days, SLOTS_PER_DAY)
        aux_daily = aux_ts.reshape(num_days, SLOTS_PER_DAY)
        primary_curve = np.median(primary_daily, axis=0)
        aux_curve = np.median(aux_daily, axis=0)

        daily_mean = primary_daily.mean(axis=1)
        day_type_indicator = np.arange(num_days)
        workday_mask = (day_type_indicator % 7 < 5)
        weekend_mask = ~workday_mask

        record: dict[str, float] = {"region_id": float(region_id)}
        record.update(build_normalized_shape_features(primary_curve, prefix="sum_"))
        record.update(build_normalized_shape_features(aux_curve, prefix="avg_"))
        record["sum_level_mean"] = float(np.mean(primary_ts))
        record["sum_level_std"] = float(np.std(primary_ts))
        record["sum_level_iqr"] = float(np.quantile(primary_ts, 0.75) - np.quantile(primary_ts, 0.25))
        record["avg_level_mean"] = float(np.mean(aux_ts))
        record["avg_level_std"] = float(np.std(aux_ts))
        record["active_ratio"] = float(np.mean(np.expm1(primary_ts) > 0.0))
        record["daily_mean_cv"] = float(np.std(daily_mean) / (np.mean(daily_mean) + EPSILON))
        if np.any(workday_mask) and np.any(weekend_mask):
            record["weekend_workday_gap"] = float(
                np.mean(daily_mean[weekend_mask]) - np.mean(daily_mean[workday_mask])
            )
        else:
            record["weekend_workday_gap"] = 0.0
        feature_records.append(record)

    feature_df = pd.DataFrame(feature_records).sort_values("region_id").reset_index(drop=True)
    feature_matrix = feature_df.drop(columns=["region_id"]).to_numpy(dtype=np.float64)
    feature_matrix = winsorize_feature_matrix(feature_matrix)
    features_z = (feature_matrix - feature_matrix.mean(axis=0, keepdims=True)) / (
        feature_matrix.std(axis=0, keepdims=True) + EPSILON
    )
    if float(np.sum(sizes)) <= 0:
        sizes[:] = float(max(0, total_steps - t_in - t_out))
    return features_z, sizes


def normalized_score(values: list[float], higher_is_better: bool) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if math.isclose(max_value, min_value):
        return [1.0] * len(values)
    scores = [(value - min_value) / (max_value - min_value) for value in values]
    if higher_is_better:
        return scores
    return [1.0 - score for score in scores]


def evaluate_kmeans_models(features_z: np.ndarray, k_values: list[int], seed: int) -> tuple[pd.DataFrame, int]:
    if features_z.shape[0] <= max(k_values):
        raise ValueError("区域数不足，无法评估给定 k 范围。")

    soft_min_required_count = max(
        SOFT_MIN_CLUSTER_SIZE,
        int(math.ceil(SOFT_MIN_CLUSTER_RATIO * features_z.shape[0])),
    )
    hard_min_required_count = max(
        MIN_HARD_CLUSTER_SIZE,
        int(math.ceil(HARD_MIN_CLUSTER_RATIO * features_z.shape[0])),
    )
    metric_records: list[dict[str, float | int | bool]] = []

    for k in k_values:
        model = KMeans(n_clusters=int(k), random_state=seed, n_init=DEFAULT_N_INIT)
        labels = model.fit_predict(features_z)
        _, silhouette_vals = sample_silhouette_values(features_z, labels, seed)
        counts = np.bincount(labels, minlength=int(k))
        min_cluster_count = int(counts.min())
        max_cluster_count = int(counts.max())
        min_cluster_ratio = float(min_cluster_count / features_z.shape[0])
        max_cluster_ratio = float(max_cluster_count / features_z.shape[0])
        is_valid_cluster_size = bool(min_cluster_count >= hard_min_required_count)

        tiny_cluster_penalty = 0.0
        if min_cluster_count < soft_min_required_count:
            tiny_cluster_penalty = TINY_CLUSTER_PENALTY_WEIGHT * (
                (soft_min_required_count - min_cluster_count) / max(soft_min_required_count, 1)
            )

        dominant_cluster_penalty = 0.0
        if max_cluster_ratio > MAX_CLUSTER_RATIO_TARGET:
            dominant_cluster_penalty = DOMINANT_CLUSTER_PENALTY_WEIGHT * (
                (max_cluster_ratio - MAX_CLUSTER_RATIO_TARGET) / max(EPSILON, 1.0 - MAX_CLUSTER_RATIO_TARGET)
            )
        if max_cluster_ratio > MAX_CLUSTER_RATIO_HARD:
            dominant_cluster_penalty += 0.20

        metric_records.append({
            "k": int(k),
            "silhouette_score": float(np.mean(silhouette_vals)),
            "calinski_harabasz_score": float(calinski_harabasz_score(features_z, labels)),
            "davies_bouldin_score": float(davies_bouldin_score(features_z, labels)),
            "inertia": float(model.inertia_),
            "min_cluster_count": min_cluster_count,
            "max_cluster_count": max_cluster_count,
            "hard_min_required_count": hard_min_required_count,
            "soft_min_required_count": soft_min_required_count,
            "min_cluster_ratio": min_cluster_ratio,
            "max_cluster_ratio": max_cluster_ratio,
            "negative_silhouette_ratio": float(np.mean(silhouette_vals < 0.0)),
            "tiny_cluster_penalty": float(tiny_cluster_penalty),
            "dominant_cluster_penalty": float(dominant_cluster_penalty),
            "is_valid_cluster_size": is_valid_cluster_size,
        })

    metric_df = pd.DataFrame(metric_records).sort_values("k").reset_index(drop=True)
    silhouette_scores = normalized_score(metric_df["silhouette_score"].tolist(), higher_is_better=True)
    ch_scores = normalized_score(metric_df["calinski_harabasz_score"].tolist(), higher_is_better=True)
    db_scores = normalized_score(metric_df["davies_bouldin_score"].tolist(), higher_is_better=False)
    balance_scores = normalized_score(metric_df["max_cluster_ratio"].tolist(), higher_is_better=False)
    negative_scores = normalized_score(metric_df["negative_silhouette_ratio"].tolist(), higher_is_better=False)
    complexity_scores = normalized_score(metric_df["k"].astype(float).tolist(), higher_is_better=False)

    selection_scores = []
    for idx, row in metric_df.iterrows():
        score = (
            0.30 * silhouette_scores[idx]
            + 0.20 * db_scores[idx]
            + 0.15 * balance_scores[idx]
            + 0.15 * negative_scores[idx]
            + 0.15 * ch_scores[idx]
            + 0.05 * complexity_scores[idx]
        )
        score -= float(row["tiny_cluster_penalty"])
        score -= float(row["dominant_cluster_penalty"])
        if not bool(row["is_valid_cluster_size"]):
            score = -1e9
        selection_scores.append(score)
    metric_df[SELECTION_SCORE_COL] = selection_scores

    valid_df = metric_df[metric_df["is_valid_cluster_size"]].copy()
    ranking_df = valid_df if not valid_df.empty else metric_df
    best_row = ranking_df.sort_values(
        by=[SELECTION_SCORE_COL, "silhouette_score", "davies_bouldin_score", "negative_silhouette_ratio", "k"],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    best_k = int(best_row["k"])
    return metric_df, best_k


def fit_final_kmeans(features_z: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=int(k), random_state=seed, n_init=DEFAULT_N_INIT)
    labels = model.fit_predict(features_z)
    sample_indices, silhouette_vals = sample_silhouette_values(features_z, labels, seed)
    return labels, model.cluster_centers_, sample_indices, silhouette_vals


def remap_cluster_labels_by_size(cluster_labels: np.ndarray, cluster_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    labels = cluster_labels.astype(np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_pairs = sorted(zip(unique_labels.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    old_to_new = {int(old_label): int(new_label) for new_label, (old_label, _) in enumerate(sorted_pairs)}
    remapped_labels = np.array([old_to_new[int(label)] for label in labels], dtype=np.int64)
    remapped_centers = np.zeros_like(cluster_centers)
    for old_label, new_label in old_to_new.items():
        remapped_centers[new_label] = cluster_centers[old_label]
    return remapped_labels, remapped_centers, old_to_new


def build_cluster_outputs(
    labels: np.ndarray,
    sizes: np.ndarray,
    features_z: np.ndarray,
    silhouette_sample_indices: np.ndarray,
    silhouette_vals: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_df = pd.DataFrame({
        "region_id": np.arange(labels.shape[0], dtype=np.int64),
        "cluster_id": labels.astype(np.int64),
        "client_id": labels.astype(np.int64),
        "estimated_samples": sizes.astype(float),
    }).sort_values(["cluster_id", "region_id"]).reset_index(drop=True)

    sampled_cluster_ids = labels[silhouette_sample_indices]
    sampled_df = pd.DataFrame({
        "cluster_id": sampled_cluster_ids.astype(np.int64),
        "silhouette": silhouette_vals.astype(float),
    })
    silhouette_summary = (
        sampled_df.groupby("cluster_id", as_index=False)
        .agg(
            mean_silhouette=("silhouette", "mean"),
            median_silhouette=("silhouette", "median"),
            negative_silhouette_ratio=("silhouette", lambda s: float(np.mean(np.asarray(s) < 0.0))),
        )
    )

    cluster_feature_df = pd.DataFrame(features_z).assign(cluster_id=labels.astype(np.int64))
    summary_df = (
        label_df.groupby("cluster_id", as_index=False)
        .agg(
            client_id=("client_id", "first"),
            region_count=("region_id", "count"),
            estimated_samples=("estimated_samples", "sum"),
        )
        .merge(silhouette_summary, on="cluster_id", how="left")
    )

    feature_mean_df = cluster_feature_df.groupby("cluster_id", as_index=False).mean()
    feature_mean_df = feature_mean_df.rename(columns={col: f"feature_{col}_mean" for col in feature_mean_df.columns if col != "cluster_id"})
    summary_df = summary_df.merge(feature_mean_df, on="cluster_id", how="left")
    summary_df = summary_df.sort_values(["cluster_id"]).reset_index(drop=True)
    return label_df, summary_df


def build_client_mapping(label_df: pd.DataFrame) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for cluster_id, cluster_df in label_df.groupby("cluster_id"):
        mapping[f"client_{int(cluster_id)}"] = cluster_df["region_id"].astype(int).tolist()
    return mapping


def save_outputs(
    config: ClusterConfig,
    metric_df: pd.DataFrame,
    best_k: int,
    label_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    client_mapping: dict[str, list[int]],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(config.output_dir / "k_search_metrics.csv", index=False, encoding="utf-8-sig")
    label_df.to_csv(config.output_dir / "cluster_labels.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(config.output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "recommended_num_clients": int(best_k),
        "dataset_path": str(config.dataset_path),
        "output_dir": str(config.output_dir),
        "k_range": [int(config.k_min), int(config.k_max)],
        "t_in": int(config.t_in),
        "t_out": int(config.t_out),
        "seed": int(config.seed),
        "client_mapping": client_mapping,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    with (config.output_dir / "recommended_client_partition.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    config = parse_args()
    logging.info("加载最终训练张量: %s", config.dataset_path)
    data_tensor = load_data_tensor(config.dataset_path)
    _, num_regions, total_steps = map(int, data_tensor.shape)
    logging.info("张量形状: %s", tuple(data_tensor.shape))

    k_values = list(range(config.k_min, config.k_max + 1))
    if num_regions <= max(k_values):
        raise ValueError(f"区域数 {num_regions} 不足以评估到 k={max(k_values)}。")

    _legacy_features_z, legacy_sizes = extract_region_features(data_tensor, config.t_in, config.t_out)
    features_z, sizes = build_grid_region_features(data_tensor, config.t_in, config.t_out)
    if np.sum(sizes) <= 0 and np.sum(legacy_sizes) > 0:
        sizes = legacy_sizes
    metric_df, best_k = evaluate_kmeans_models(features_z, k_values, config.seed)
    labels, centers, sample_indices, silhouette_vals = fit_final_kmeans(features_z, best_k, config.seed)
    labels, centers, remap_dict = remap_cluster_labels_by_size(labels, centers)
    label_df, summary_df = build_cluster_outputs(labels, sizes, features_z, sample_indices, silhouette_vals)
    client_mapping = build_client_mapping(label_df)
    save_outputs(config, metric_df, best_k, label_df, summary_df, client_mapping)

    logging.info("推荐簇数/客户端数: %s", best_k)
    logging.info("标签重映射: %s", remap_dict)
    for _, row in summary_df.iterrows():
        logging.info(
            "客户端 %s: 区域数=%s, 估计样本量=%.0f, mean_silhouette=%.4f",
            int(row["client_id"]),
            int(row["region_count"]),
            float(row["estimated_samples"]),
            float(row.get("mean_silhouette", 0.0) if pd.notna(row.get("mean_silhouette", np.nan)) else 0.0),
        )


if __name__ == "__main__":
    main()
