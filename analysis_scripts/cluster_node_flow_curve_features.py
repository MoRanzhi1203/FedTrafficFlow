"""基于日内车流量拟合结果对路口节点曲线形态进行聚类。"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis_scripts.fit_node_flow_daily_curve import (  # noqa: E402
    AVG_FLOW_COL,
    DAY_SLOT_COL,
    NODE_COL,
    SLOTS_PER_DAY,
    build_fourier_design_matrix,
)


FIT_DIR = ROOT_DIR / "data" / "analysis" / "node_flow_curve_fit"
OUTPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_flow_curve_cluster"
PLOT_DIR = OUTPUT_DIR / "plots"

COEFFICIENT_PATH = FIT_DIR / "node_flow_curve_coefficients.parquet"
FITTED_CURVE_PATH = FIT_DIR / "node_flow_fitted_daily_curves.parquet"

LABEL_OUTPUT_PATH = OUTPUT_DIR / "node_flow_curve_cluster_labels.parquet"
METRIC_OUTPUT_PATH = OUTPUT_DIR / "node_flow_curve_cluster_metrics.parquet"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "node_flow_curve_cluster_summary.parquet"
CENTER_OUTPUT_PATH = OUTPUT_DIR / "node_flow_curve_cluster_centers.parquet"

METRIC_PLOT_PATH = PLOT_DIR / "cluster_metric_scores.png"
CENTER_PLOT_PATH = PLOT_DIR / "cluster_center_curves.png"
PCA_PLOT_PATH = PLOT_DIR / "cluster_scatter_pca.png"

NORMALIZED_FLOW_COL = "归一化路口车流量"
CLUSTER_COL = "cluster_id"

DEFAULT_K_VALUES = [3, 4, 5, 6, 7, 8]
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 20
MIN_R2 = 0.85
MEAN_FLOW_QUANTILE = 0.05
EPSILON = 1e-6
FIG_DPI = 200


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对路口节点日内车流量曲线形态进行 KMeans 聚类。"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="待评估的聚类数列表，默认: 3 4 5 6 7 8",
    )
    parser.add_argument(
        "--harmonics",
        type=int,
        default=None,
        help="归一化傅里叶形态特征的谐波数，默认从系数文件自动推断",
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


def check_input_files() -> None:
    """检查输入结果文件是否存在。"""
    if not FIT_DIR.exists():
        raise FileNotFoundError(f"拟合结果目录不存在: {FIT_DIR}")
    if not COEFFICIENT_PATH.exists():
        raise FileNotFoundError(f"未找到系数结果文件: {COEFFICIENT_PATH}")
    if not FITTED_CURVE_PATH.exists():
        raise FileNotFoundError(f"未找到拟合曲线结果文件: {FITTED_CURVE_PATH}")


def validate_required_columns(df: pl.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    """检查 DataFrame 是否包含必需字段。"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{df_name} 缺少字段: {missing_columns}")


def load_inputs() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """读取聚类所需输入结果。"""
    coeff_df = pl.read_parquet(COEFFICIENT_PATH)
    fitted_df = pl.read_parquet(FITTED_CURVE_PATH)

    validate_required_columns(
        coeff_df,
        [NODE_COL, "R2", "RMSE", "MAE", "平均流量", "最大流量", "最小流量", "a0"],
        "node_flow_curve_coefficients.parquet",
    )
    validate_required_columns(
        fitted_df,
        [NODE_COL, DAY_SLOT_COL, AVG_FLOW_COL],
        "node_flow_fitted_daily_curves.parquet",
    )
    return coeff_df, fitted_df


def infer_harmonics_from_coefficients(coeff_df: pl.DataFrame) -> int:
    """从系数字段中推断傅里叶谐波数。"""
    a_orders = set()
    b_orders = set()
    for column_name in coeff_df.columns:
        match = re.fullmatch(r"([ab])(\d+)", column_name)
        if not match:
            continue
        prefix, order_text = match.groups()
        order = int(order_text)
        if order == 0:
            continue
        if prefix == "a":
            a_orders.add(order)
        else:
            b_orders.add(order)

    harmonics = max(a_orders & b_orders) if (a_orders & b_orders) else 0
    if harmonics <= 0:
        raise ValueError("无法从系数文件中推断傅里叶谐波数")
    return harmonics


def select_candidate_nodes(coeff_df: pl.DataFrame) -> Tuple[pl.DataFrame, float]:
    """按拟合质量和低流量阈值筛选候选节点。"""
    mean_flows = coeff_df.get_column("平均流量").to_numpy().astype(np.float64)
    flow_threshold = float(np.quantile(mean_flows, MEAN_FLOW_QUANTILE))

    selected_df = coeff_df.filter(
        (pl.col("R2") >= MIN_R2) & (pl.col("平均流量") > flow_threshold)
    ).sort(NODE_COL)

    if selected_df.is_empty():
        raise ValueError("筛选后没有满足条件的候选节点，请检查阈值设置")

    return selected_df, flow_threshold


def iter_complete_node_curves(
    fitted_df: pl.DataFrame,
    selected_node_ids: Sequence[int],
) -> Iterable[pl.DataFrame]:
    """按节点遍历完整的 96 点曲线。"""
    if not selected_node_ids:
        return []

    expected_slots = np.arange(SLOTS_PER_DAY, dtype=np.int64)
    node_id_list = [int(node_id) for node_id in selected_node_ids]
    filtered_df = (
        fitted_df
        .filter(pl.col(NODE_COL).is_in(node_id_list))
        .sort([NODE_COL, DAY_SLOT_COL])
    )

    for node_df in filtered_df.partition_by(NODE_COL, maintain_order=True):
        day_slots = node_df.get_column(DAY_SLOT_COL).to_numpy()
        if len(day_slots) != SLOTS_PER_DAY:
            continue
        if not np.array_equal(day_slots, expected_slots):
            continue
        yield node_df


def build_feature_dataset(
    coeff_df: pl.DataFrame,
    fitted_df: pl.DataFrame,
    harmonics: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """构建用于聚类的节点特征表、曲线表和特征矩阵。"""
    selected_coeff_df, flow_threshold = select_candidate_nodes(coeff_df)
    selected_node_ids = selected_coeff_df.get_column(NODE_COL).to_list()

    design_matrix = build_fourier_design_matrix(
        np.arange(SLOTS_PER_DAY, dtype=np.int64),
        harmonics,
    )
    feature_records: List[Dict[str, float]] = []
    curve_frames: List[pl.DataFrame] = []
    skipped_incomplete_nodes = 0

    for node_df in iter_complete_node_curves(fitted_df, selected_node_ids):
        node_id = int(node_df.item(0, NODE_COL))
        avg_flow = node_df.get_column(AVG_FLOW_COL).to_numpy().astype(np.float64)
        mean_flow = float(np.mean(avg_flow))
        normalized_flow = avg_flow / (mean_flow + EPSILON)

        coefficients, _, _, _ = np.linalg.lstsq(
            design_matrix,
            normalized_flow,
            rcond=None,
        )

        record: Dict[str, float] = {NODE_COL: node_id}
        coeff_index = 1
        for h in range(1, harmonics + 1):
            record[f"a{h}"] = float(coefficients[coeff_index])
            record[f"b{h}"] = float(coefficients[coeff_index + 1])
            coeff_index += 2
        feature_records.append(record)

        curve_frames.append(
            pl.DataFrame({
                NODE_COL: np.full(SLOTS_PER_DAY, node_id, dtype=np.int64),
                DAY_SLOT_COL: np.arange(SLOTS_PER_DAY, dtype=np.int64),
                AVG_FLOW_COL: avg_flow,
                NORMALIZED_FLOW_COL: normalized_flow,
            })
        )

    built_node_ids = {int(record[NODE_COL]) for record in feature_records}
    skipped_incomplete_nodes = selected_coeff_df.height - len(built_node_ids)

    if not feature_records:
        raise ValueError("没有节点通过完整 96 点曲线检查，无法执行聚类")

    feature_df = pl.DataFrame(feature_records).sort(NODE_COL)
    node_curve_df = pl.concat(curve_frames, how="vertical").sort([NODE_COL, DAY_SLOT_COL])
    built_node_id_list = feature_df.get_column(NODE_COL).to_list()
    selected_metadata_df = (
        selected_coeff_df
        .filter(pl.col(NODE_COL).is_in(built_node_id_list))
        .sort(NODE_COL)
    )

    feature_columns = []
    for h in range(1, harmonics + 1):
        feature_columns.extend([f"a{h}", f"b{h}"])
    feature_matrix = feature_df.select(feature_columns).to_numpy().astype(np.float64)

    print(f"全体节点数: {coeff_df.height}")
    print(f"平均流量 5% 分位数阈值: {flow_threshold:.6f}")
    print(f"通过筛选的候选节点数: {selected_coeff_df.height}")
    print(f"通过完整曲线检查的节点数: {feature_df.height}")
    print(f"因曲线不完整跳过节点数: {skipped_incomplete_nodes}")
    print(f"聚类特征维度: {feature_matrix.shape[1]}")

    return selected_metadata_df, node_curve_df, feature_matrix


def evaluate_kmeans_models(
    standardized_features: np.ndarray,
    k_values: Sequence[int],
) -> Tuple[pl.DataFrame, int]:
    """评估不同聚类数下的 KMeans 表现并返回推荐 k。"""
    n_samples = standardized_features.shape[0]
    valid_k_values = sorted({int(k) for k in k_values if 2 < int(k) < n_samples})
    if not valid_k_values:
        raise ValueError(
            f"可用于聚类的节点数为 {n_samples}，无法评估给定的 k 值列表 {list(k_values)}"
        )

    metric_records: List[Dict[str, float]] = []
    for k in valid_k_values:
        model = KMeans(
            n_clusters=k,
            random_state=DEFAULT_RANDOM_STATE,
            n_init=DEFAULT_N_INIT,
        )
        labels = model.fit_predict(standardized_features)
        metric_records.append({
            "k": int(k),
            "silhouette_score": float(silhouette_score(standardized_features, labels)),
            "calinski_harabasz_score": float(
                calinski_harabasz_score(standardized_features, labels)
            ),
            "davies_bouldin_score": float(
                davies_bouldin_score(standardized_features, labels)
            ),
        })

    metric_df = pl.DataFrame(metric_records).sort("k")
    best_metric_row = (
        metric_df
        .sort(["silhouette_score", "k"], descending=[True, False])
        .row(0, named=True)
    )
    best_k = int(best_metric_row["k"])
    return metric_df, best_k


def fit_final_kmeans(
    standardized_features: np.ndarray,
    best_k: int,
) -> np.ndarray:
    """使用推荐 k 训练最终 KMeans。"""
    model = KMeans(
        n_clusters=best_k,
        random_state=DEFAULT_RANDOM_STATE,
        n_init=DEFAULT_N_INIT,
    )
    return model.fit_predict(standardized_features)


def build_output_tables(
    metadata_df: pl.DataFrame,
    node_curve_df: pl.DataFrame,
    cluster_labels: np.ndarray,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """构造标签表、聚类汇总表和类中心曲线表。"""
    label_df = (
        metadata_df
        .with_columns(pl.Series(CLUSTER_COL, cluster_labels.astype(np.int64)))
        .select([
            NODE_COL,
            CLUSTER_COL,
            "R2",
            "RMSE",
            "MAE",
            "平均流量",
            "最大流量",
            "最小流量",
        ])
        .sort([CLUSTER_COL, NODE_COL])
    )

    summary_df = (
        label_df
        .group_by(CLUSTER_COL)
        .agg([
            pl.len().alias("节点数"),
            pl.col("平均流量").mean().alias("平均流量均值"),
            pl.col("最大流量").mean().alias("最大流量均值"),
            pl.col("R2").mean().alias("平均R2"),
            pl.col("RMSE").mean().alias("平均RMSE"),
            pl.col("MAE").mean().alias("平均MAE"),
        ])
        .sort(CLUSTER_COL)
    )

    center_df = (
        node_curve_df
        .join(label_df.select([NODE_COL, CLUSTER_COL]), on=NODE_COL, how="inner")
        .group_by([CLUSTER_COL, DAY_SLOT_COL])
        .agg([
            pl.col(NORMALIZED_FLOW_COL).mean().alias("类平均归一化流量"),
            pl.col(AVG_FLOW_COL).mean().alias("类平均原始流量"),
        ])
        .sort([CLUSTER_COL, DAY_SLOT_COL])
    )

    return label_df, summary_df, center_df


def save_outputs(
    label_df: pl.DataFrame,
    metric_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    center_df: pl.DataFrame,
) -> None:
    """保存 parquet 输出结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_df.write_parquet(LABEL_OUTPUT_PATH, compression="snappy")
    metric_df.write_parquet(METRIC_OUTPUT_PATH, compression="snappy")
    summary_df.write_parquet(SUMMARY_OUTPUT_PATH, compression="snappy")
    center_df.write_parquet(CENTER_OUTPUT_PATH, compression="snappy")


def plot_cluster_metric_scores(metric_df: pl.DataFrame, output_path: Path) -> None:
    """绘制不同 k 对应的聚类指标曲线。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k = metric_df.get_column("k").to_numpy()
    silhouette = metric_df.get_column("silhouette_score").to_numpy()
    ch_score = metric_df.get_column("calinski_harabasz_score").to_numpy()
    db_score = metric_df.get_column("davies_bouldin_score").to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(k, silhouette, marker="o", color="#4C78A8")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Silhouette Score")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(k, ch_score, marker="o", color="#54A24B")
    axes[1].set_ylabel("Calinski-Harabasz")
    axes[1].set_title("Calinski-Harabasz Score")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(k, db_score, marker="o", color="#E45756")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Davies-Bouldin")
    axes[2].set_title("Davies-Bouldin Score")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("不同聚类数下的聚类评价指标", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_center_curves(center_df: pl.DataFrame, output_path: Path) -> None:
    """绘制每个聚类的类平均归一化曲线。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_frames = center_df.partition_by(CLUSTER_COL, maintain_order=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    for cluster_df in cluster_frames:
        cluster_id = int(cluster_df.item(0, CLUSTER_COL))
        ax.plot(
            cluster_df.get_column(DAY_SLOT_COL).to_numpy(),
            cluster_df.get_column("类平均归一化流量").to_numpy(),
            linewidth=2.0,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlim(0, SLOTS_PER_DAY - 1)
    ax.set_xticks(np.arange(0, SLOTS_PER_DAY + 1, 12))
    ax.set_xlabel("日内时间段")
    ax.set_ylabel("类平均归一化流量")
    ax.set_title("各聚类类别的平均归一化日内流量曲线")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_scatter_pca(
    standardized_features: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
) -> None:
    """使用 PCA 将特征降至二维并绘制聚类散点图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
    embedding = pca.fit_transform(standardized_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=12,
        alpha=0.65,
        edgecolors="none",
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("节点曲线形态聚类 PCA 二维投影")
    ax.grid(True, linestyle="--", alpha=0.30)

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
    )
    ax.add_artist(legend)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """主函数。"""
    args = parse_args()
    configure_fonts()
    check_input_files()

    coeff_df, fitted_df = load_inputs()
    harmonics = args.harmonics or infer_harmonics_from_coefficients(coeff_df)

    print("=" * 80)
    print("路口节点日内车流量曲线形态聚类")
    print("=" * 80)
    print(f"系数输入文件: {COEFFICIENT_PATH}")
    print(f"拟合曲线输入文件: {FITTED_CURVE_PATH}")
    print(f"归一化傅里叶特征谐波数: {harmonics}")

    metadata_df, node_curve_df, feature_matrix = build_feature_dataset(
        coeff_df=coeff_df,
        fitted_df=fitted_df,
        harmonics=harmonics,
    )

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(feature_matrix)
    metric_df, best_k = evaluate_kmeans_models(standardized_features, args.k_values)
    cluster_labels = fit_final_kmeans(standardized_features, best_k)

    label_df, summary_df, center_df = build_output_tables(
        metadata_df=metadata_df,
        node_curve_df=node_curve_df,
        cluster_labels=cluster_labels,
    )
    save_outputs(label_df, metric_df, summary_df, center_df)

    plot_cluster_metric_scores(metric_df, METRIC_PLOT_PATH)
    plot_cluster_center_curves(center_df, CENTER_PLOT_PATH)
    plot_cluster_scatter_pca(standardized_features, cluster_labels, PCA_PLOT_PATH)

    print(f"评估的 k 值: {metric_df.get_column('k').to_list()}")
    print(f"推荐 k: {best_k}")
    print(f"最终参与聚类节点数: {label_df.height}")
    print(f"聚类标签输出: {LABEL_OUTPUT_PATH}")
    print(f"聚类评价输出: {METRIC_OUTPUT_PATH}")
    print(f"聚类汇总输出: {SUMMARY_OUTPUT_PATH}")
    print(f"聚类中心输出: {CENTER_OUTPUT_PATH}")
    print(f"指标图输出: {METRIC_PLOT_PATH}")
    print(f"类中心曲线图输出: {CENTER_PLOT_PATH}")
    print(f"PCA 散点图输出: {PCA_PLOT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
