# -*- coding: utf-8 -*-
"""
GCN 一审增强仿真实验组。

本文件复用 cnn_fed_enhanced_experiments.py 的增强 Non-IID 数据生成框架，
在此基础上增加 GCN 图结构构造与可视化：

1. data_viz:         增强 Non-IID 数据集可视化（与 CNN 增强一致）；
2. fixed_vs_dynamic: 固定邻接 vs 动态邻接 vs 功能相似 vs 拥堵延迟矩阵可视化；
3. congestion_delay: 拥堵传播延迟数据准备与可视化；
4. main / aggregation / lambda / client_scale / noniid / convergence / client_metrics / peak：
   后续批次补充（当前保留接口存根）。

GCN 与 CNN 的区别仅体现在：
- GCN 使用固定邻接矩阵；
- GCN 使用动态邻接矩阵；
- GCN 使用功能相似矩阵；
- GCN 使用拥堵传播延迟矩阵；
- GCN 模型结构为 GCN-BiLSTM-Attention。

原始输入 X 与 CNN 增强实验在同一 seed 下保持一致。
"""

import argparse
import copy
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_cjk_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((fn for fn in _cjk_candidates if fn in _available), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

plt.ioff()

# ══════════════════════════════════════════════════════════════
# 导入 CNN 增强实验的数据生成框架（保证数据一致性）
# ══════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cnn_fed_enhanced_experiments import (
    CLIENT_CONFIGS_BASE,
    generate_traffic_flow,
    sample_distribution_noise,
    build_noniid_client_configs,
    classify_period,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHOD_PALETTE = {
    "Independent": "#4C72B0",
    "FedAvg": "#DD8452",
    "CNN-FedAvg": "#DD8452",
    "GCN-FedAvg": "#DD8452",
    "Proposed": "#55A868",
    "CNN-Proposed": "#55A868",
    "GCN-Proposed": "#55A868",
    "Loss-weighted": "#C44E52",
    "Data-loss weighted": "#8172B3",
    "Similarity-aware": "#937860",
}
CLIENT_PALETTE = sns.color_palette("tab10")

FIGURE_INDEX_ENTRIES = [
    {"figure_file": "enhanced_dataset_client_timeseries.png", "workflow": "data_viz", "figure_type": "line", "description": "Per-client average traffic flow time series for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_distribution_comparison.png", "workflow": "data_viz", "figure_type": "box", "description": "Client-level traffic flow distribution comparison for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_client_config.png", "workflow": "data_viz", "figure_type": "bar", "description": "Client configuration overview for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "enhanced_dataset_peak_pattern.png", "workflow": "data_viz", "figure_type": "line", "description": "Twenty-four-hour peak traffic patterns across enhanced clients.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_incident_example.png", "workflow": "data_viz", "figure_type": "line", "description": "Incident example with shaded disruption periods for the incident-prone client.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "enhanced_dataset_client_correlation_matrix.png", "workflow": "data_viz", "figure_type": "heatmap", "description": "Inter-client traffic correlation matrix for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_node_correlation_matrix.png", "workflow": "data_viz", "figure_type": "heatmap", "description": "Node correlation matrix for a representative enhanced client.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_fixed_adjacency_matrix.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Fixed adjacency matrix used by the enhanced GCN experiment.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_dynamic_adjacency_peak.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Dynamic adjacency matrix during peak traffic.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_dynamic_adjacency_offpeak.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Dynamic adjacency matrix during off-peak traffic.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_fixed_dynamic_adjacency_comparison.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Comparison of fixed and dynamic graph structures across traffic periods.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_functional_similarity_matrix.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Functional similarity matrix estimated from node traffic profiles.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_congestion_delay_matrix.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Congestion propagation delay matrix for the enhanced GCN setting.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_peak_graph_change.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Graph structure changes across off-peak and peak traffic periods.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "gcn_enhanced_fixed_vs_dynamic_comparison.png", "workflow": "fixed_vs_dynamic", "figure_type": "bar", "description": "Performance comparison under fixed, dynamic, and functional graph definitions.", "source_csv": "gcn_enhanced_fixed_vs_dynamic_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_congestion_delay_comparison.png", "workflow": "congestion_delay", "figure_type": "bar", "description": "Performance comparison across congestion-delay-related graph definitions.", "source_csv": "gcn_enhanced_congestion_delay_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_main_rmse_comparison.png", "workflow": "main", "figure_type": "bar", "description": "Main comparison of Independent, GCN-FedAvg, and GCN-Proposed.", "source_csv": "gcn_enhanced_main_metrics_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_aggregation_ablation.png", "workflow": "aggregation", "figure_type": "bar", "description": "Aggregation strategy ablation on RMSE and MAE.", "source_csv": "gcn_enhanced_aggregation_ablation_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_lambda_sensitivity.png", "workflow": "lambda", "figure_type": "line", "description": "Lambda sensitivity analysis for the data-loss weighted GCN aggregation strategy.", "source_csv": "gcn_enhanced_lambda_sensitivity_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_client_scale.png", "workflow": "client_scale", "figure_type": "line", "description": "Client-scale sensitivity analysis for the enhanced GCN setting.", "source_csv": "gcn_enhanced_client_scale_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_noniid_strength.png", "workflow": "noniid", "figure_type": "bar", "description": "Method comparison under different Non-IID strengths for enhanced GCN.", "source_csv": "gcn_enhanced_noniid_strength_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_global_validation_rmse.png", "workflow": "convergence", "figure_type": "line", "description": "Global validation RMSE across communication rounds for enhanced GCN.", "source_csv": "gcn_enhanced_convergence_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_client_training_loss.png", "workflow": "convergence", "figure_type": "line", "description": "Per-client training loss across communication rounds for GCN-FedAvg and GCN-Proposed.", "source_csv": "gcn_enhanced_convergence_round_metrics.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_client_rmse_comparison.png", "workflow": "client_metrics", "figure_type": "bar", "description": "Per-client RMSE comparison across Independent, GCN-FedAvg, and GCN-Proposed.", "source_csv": "gcn_enhanced_client_metrics.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_peak_offpeak_comparison.png", "workflow": "peak", "figure_type": "bar", "description": "Performance comparison across peak, off-peak, and incident periods for enhanced GCN.", "source_csv": "gcn_enhanced_peak_offpeak_summary.csv", "used_in_paper": "recommended"},
]

# 全局常量（与 CNN 增强实验保持一致）
NUM_NODES = 8
SEQ_LEN = 12
PRED_LEN = 1
SEED = 42

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_academic_plot_style() -> None:
    """Configure a unified seaborn style for paper-ready figures."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.autolayout": False,
        },
    )
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path


def export_figure_index(output_dir: Path) -> Path:
    """Export figure metadata for paper curation."""
    return save_dataframe(pd.DataFrame(FIGURE_INDEX_ENTRIES), output_dir, "figure_index.csv")


def plot_heatmap(
    ax,
    matrix,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    cbar_label: str,
    vmin=None,
    vmax=None,
    center=None,
    square: bool = True,
    annot: bool = False,
    fmt: str = ".2f",
    mask=None,
):
    """Draw a consistent academic heatmap."""
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=square,
        annot=annot,
        fmt=fmt,
        mask=mask,
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ══════════════════════════════════════════════════════════════
# 图结构构造函数
# ══════════════════════════════════════════════════════════════

def build_fixed_adjacency(num_nodes: int = NUM_NODES, seed: int = SEED):
    """构建固定邻接矩阵（链式 + 少量跨连接，模拟路网拓扑）。

    结构：节点沿编号排列形成一线型拓扑，相邻节点全连接（权重1.0），
    再加入少量跨连接以模拟交叉口。最终进行对称归一化。

    返回:
        a_norm: 归一化邻接矩阵 (num_nodes, num_nodes)
        a_raw:  原始邻接矩阵
        meta:   dict of graph statistics
    """
    rng = np.random.RandomState(seed)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    # 相邻节点连接（线型路网）
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    # 添加少量跨连接（模拟交叉口）
    cross_edges = min(num_nodes // 2, 3)
    for _ in range(cross_edges):
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u != v and A[u, v] == 0:
            w = 0.3 + 0.4 * rng.rand()
            A[u, v] = w
            A[v, u] = w
    # 自环 + 对称归一化
    A_self = A + np.eye(num_nodes, dtype=np.float32)
    deg = A_self.sum(axis=1)
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_self @ D_inv_sqrt

    degrees = A.sum(axis=1)
    num_edges = int(np.sum(A > 0) / 2)
    density = float(2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0
    meta = {
        "graph_type": "fixed_adjacency",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": float(np.max(degrees)),
        "min_degree": float(np.min(degrees)),
        "density": density,
        "description": "chain_with_cross_connections",
    }
    return A_norm.astype(np.float32), A.astype(np.float32), meta


def _compute_node_similarity(data_chunk):
    """计算节点间 Pearson 相关系数矩阵。

    参数:
        data_chunk: (n_timesteps, n_nodes) 流量数据

    返回:
        sim: (n_nodes, n_nodes) 归一化相似度矩阵
    """
    n_nodes = data_chunk.shape[1]
    sim = np.corrcoef(data_chunk, rowvar=False)  # (n_nodes, n_nodes)
    # 绝对值化（负相关也视为结构相关），阈值过滤弱连接
    sim = np.abs(sim)
    threshold = np.percentile(sim[~np.eye(n_nodes, dtype=bool)], 70)
    sim[sim < threshold] = 0.0
    # 加入自环和归一化
    sim = sim + np.eye(n_nodes)
    deg = sim.sum(axis=1)
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    sim_norm = D_inv_sqrt @ sim @ D_inv_sqrt
    return sim_norm.astype(np.float32), sim.astype(np.float32)


def build_dynamic_adjacency(data, period="off_peak"):
    """构建动态邻接矩阵（基于指定时段节点流量相关性）。

    参数:
        data: (n_timesteps, n_nodes) 来自代表性 client 的完整时间序列
        period: "morning_peak" | "evening_peak" | "off_peak" | "peak" (morning + evening)

    返回:
        a_norm: 动态归一化邻接矩阵
        a_raw:  原始相似度矩阵
        meta:   图统计信息
    """
    n_ts = len(data)
    hours = (np.arange(n_ts) * 24.0 / n_ts) % 24
    # 只取 incident_prob = 0 的 client，incident 单独处理
    # 按 period 取索引
    if period == "morning_peak":
        mask = (hours >= 7) & (hours < 9)
    elif period == "evening_peak":
        mask = (hours >= 17) & (hours < 19)
    elif period == "peak":
        mask = ((hours >= 7) & (hours < 9)) | ((hours >= 17) & (hours < 19))
    else:  # off_peak
        mask = ~(((hours >= 7) & (hours < 9)) | ((hours >= 17) & (hours < 19)))

    if mask.sum() < 5:
        # fallback: 使用全部数据
        mask = np.ones(n_ts, dtype=bool)

    chunk = data[mask]
    a_norm, a_raw = _compute_node_similarity(chunk)

    degrees = (a_raw > 0).sum(axis=1).astype(float)
    # 元素级 degree (只统计非零)
    adj_binary = (a_raw > 0).astype(int)
    np.fill_diagonal(adj_binary, 0)
    num_edges = int(adj_binary.sum() / 2)
    n_nodes = a_raw.shape[0]
    density = float(2 * num_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0

    meta = {
        "graph_type": f"dynamic_{period}",
        "num_nodes": n_nodes,
        "num_edges": num_edges,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": float(np.max(degrees)),
        "min_degree": float(np.min(degrees)),
        "density": density,
        "description": f"correlation_based_dynamic_{period}",
    }
    return a_norm.astype(np.float32), a_raw.astype(np.float32), meta


def build_functional_similarity_matrix(data):
    """构建节点功能相似矩阵（基于完整时间序列 Pearson 相关系数）。

    参数:
        data: (n_timesteps, n_nodes) 完整时间序列

    返回:
        sim_norm: 归一化相似度矩阵
        sim_raw:  原始相似度矩阵
        meta:     统计信息
    """
    n_nodes = data.shape[1]
    sim = np.corrcoef(data, rowvar=False)
    sim_raw = np.abs(sim)  # 绝对值

    # 阈值过滤
    flat = sim_raw[~np.eye(n_nodes, dtype=bool)]
    threshold = np.percentile(flat, 50) if len(flat) > 0 else 0.3
    sim_filtered = sim_raw.copy()
    sim_filtered[sim_filtered < threshold] = 0.0

    # 自环 + 归一化
    sim_self = sim_filtered + np.eye(n_nodes)
    deg = sim_self.sum(axis=1)
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    sim_norm = D_inv_sqrt @ sim_self @ D_inv_sqrt

    degrees = (sim_filtered > 0).sum(axis=1).astype(float)
    adj_b = (sim_filtered > 0).astype(int)
    np.fill_diagonal(adj_b, 0)
    num_edges = int(adj_b.sum() / 2)
    density = float(2 * num_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0

    meta = {
        "graph_type": "functional_similarity",
        "num_nodes": n_nodes,
        "num_edges": num_edges,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": float(np.max(degrees)),
        "min_degree": float(np.min(degrees)),
        "density": density,
        "description": "pearson_correlation_based",
    }
    return sim_norm.astype(np.float32), sim_raw.astype(np.float32), meta


def build_congestion_delay_matrix(data, max_lag=5):
    """构建拥堵传播延迟矩阵。

    对每一对节点 (i -> j) 计算不同 lag 下的互相关，
    取最高相关性对应的 lag 作为延迟步数。

    参数:
        data: (n_timesteps, n_nodes)
        max_lag: 最大延迟步数

    返回:
        delay_matrix: (n_nodes, n_nodes) 延迟步数
        strength_matrix: (n_nodes, n_nodes) 最大相关系数
        meta: 统计信息
    """
    n_nodes = data.shape[1]
    delay_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    strength_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                delay_matrix[i, j] = 0.0
                strength_matrix[i, j] = 1.0
                continue
            best_corr = -1.0
            best_lag = max_lag
            for lag in range(max_lag + 1):
                if lag == 0:
                    corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                else:
                    corr = np.corrcoef(data[:-lag, i], data[lag:, j])[0, 1]
                corr = abs(corr)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            delay_matrix[i, j] = float(best_lag)
            strength_matrix[i, j] = float(best_corr)

    # 阈值：相关性过低的设为 max_lag（无有效传播）
    weak_mask = strength_matrix < 0.3
    delay_matrix[weak_mask] = float(max_lag)
    np.fill_diagonal(delay_matrix, 0)

    meta = {
        "graph_type": "congestion_delay",
        "num_nodes": n_nodes,
        "num_edges": 0,  # 延迟矩阵不是二值邻接
        "avg_degree": float(np.mean(delay_matrix[delay_matrix > 0])) if (delay_matrix > 0).any() else 0,
        "max_degree": float(np.max(delay_matrix)),
        "min_degree": float(np.min(delay_matrix)),
        "density": float(np.mean(strength_matrix > 0.3)),
        "description": "cross_correlation_delay_max_lag_{}".format(max_lag),
    }
    return delay_matrix, strength_matrix, meta


# ══════════════════════════════════════════════════════════════
# 基础数据获取（与 CNN 增强实验 100% 一致）
# ══════════════════════════════════════════════════════════════

def get_enhanced_raw_data():
    """生成与 CNN 增强实验完全相同的原始交通流数据。

    返回:
        raw_signals: list of (n_timesteps, num_nodes) arrays per client
        raw_masks:  list of incident_mask per client
        cfgs:       client configs
    """
    set_global_seed(SEED)
    cfgs = list(CLIENT_CONFIGS_BASE)
    num_nodes = NUM_NODES
    raw_signals, raw_masks = [], []
    for cid, cfg in enumerate(cfgs):
        n_ts = cfg["n_samples"] + 50
        data, mask = generate_traffic_flow(cfg, n_ts, num_nodes, SEED + cid * 100)
        raw_signals.append(data)
        raw_masks.append(mask)
    return raw_signals, raw_masks, cfgs


# ══════════════════════════════════════════════════════════════
# GCN 模型定义
# ══════════════════════════════════════════════════════════════

class AdaptiveSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class SimpleGCNLayer(nn.Module):
    """基础图卷积层 A_norm @ X @ W。"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, a_norm):
        ax = torch.einsum("ij,bjf->bif", a_norm, x)
        return self.lin(ax)


class GCNEncoder(nn.Module):
    """GCN 编码器（固定邻接 + 两层图卷积）。"""
    def __init__(self, k: int, t: int, hidden_dim: int = 64, fixed_adj: np.ndarray = None):
        super().__init__()
        self.k = k
        self.node_proj = nn.Sequential(
            nn.Linear(t, hidden_dim), nn.LayerNorm(hidden_dim), AdaptiveSwish())
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()
        self.a_param = nn.Parameter(torch.randn(k, k) * 0.01)
        if fixed_adj is not None:
            self.register_buffer("fixed_adj",
                                  torch.tensor(fixed_adj.astype(np.float32), dtype=torch.float32))
        else:
            self.fixed_adj = None

    def _normalize_adj(self, a):
        a = torch.relu(a)
        a = a + torch.eye(self.k, device=a.device, dtype=a.dtype)
        deg = a.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        return torch.diag(deg_inv_sqrt) @ a @ torch.diag(deg_inv_sqrt)

    def forward(self, x):
        x = self.node_proj(x)
        a_norm = self.fixed_adj if self.fixed_adj is not None else self._normalize_adj(self.a_param)
        h = self.gcn1(x, a_norm); h = self.norm1(h); h = self.act(h)
        h = self.gcn2(h, a_norm); h = self.norm2(h); h = self.act(h)
        return h.mean(dim=1)


class GCNEnhancedModel(nn.Module):
    """GCN-BiLSTM-Attention 增强模型。"""
    def __init__(self, k: int, t: int, hidden_dim: int = 64, num_heads: int = 4,
                 fixed_adj: np.ndarray = None):
        super().__init__()
        self.gcn_encoder = GCNEncoder(k=k, t=t, hidden_dim=hidden_dim, fixed_adj=fixed_adj)
        self.lstm = nn.LSTM(input_size=k, hidden_size=hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.LayerNorm(32), AdaptiveSwish(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_gcn = self.gcn_encoder(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.lstm_proj(x_lstm.mean(dim=1))
        feat_seq = torch.stack([x_gcn, x_lstm], dim=1)
        attn_out, attn_w = self.multihead_attn(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        return self.regression_head(attn_out.mean(dim=1)), attn_w


# ══════════════════════════════════════════════════════════════
# 数据处理（复用 CNN 增强框架）
# ══════════════════════════════════════════════════════════════

def build_sequences(data, seq_len, pred_len, incident_mask=None):
    """滑动窗口构建监督学习样本，同时返回 target_meta。

    返回:
        X: (N, seq_len, n_nodes)
        y: (N,)
        target_meta: dict with 'target_time_index', 'target_hour', 'target_incident_flag'
    """
    X, y, target_time_index = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        target_idx = i + seq_len + pred_len - 1
        y.append(data[target_idx, 0])
        target_time_index.append(target_idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    t_idx = np.array(target_time_index, dtype=int)
    n_ts = len(data)
    target_hour = ((t_idx * 24.0 / n_ts) % 24).astype(np.float32)
    if incident_mask is not None:
        target_incident_flag = incident_mask[t_idx].astype(bool)
    else:
        target_incident_flag = np.zeros(len(y), dtype=bool)
    meta = {"target_time_index": t_idx, "target_hour": target_hour,
            "target_incident_flag": target_incident_flag}
    return X, y, meta


def classify_period_local(hour, incident_flag):
    """根据真实 hour 和 incident flag 对样本分类。"""
    if incident_flag: return "incident_period"
    if 7 <= hour < 9: return "morning_peak"
    if 17 <= hour < 19: return "evening_peak"
    return "off_peak"


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_client_data(cfgs, num_nodes, seq_len, pred_len, seed):
    """为所有 client 构建 train/val/test DataLoader 及标准化参数。"""
    buffer = seq_len + pred_len + 10
    B = 32
    all_data = []
    for cid, cfg in enumerate(cfgs):
        n_ts = cfg["n_samples"] + buffer
        data, inc_mask = generate_traffic_flow(cfg, n_ts, num_nodes, seed + cid * 100)
        X, y, meta = build_sequences(data, seq_len, pred_len, inc_mask)
        n = len(X); n_train = int(n * 0.70); n_val = int(n * 0.10)
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
        meta_test = {k: v[n_train + n_val:] for k, v in meta.items()}
        x_mean = X_train.mean(axis=(0, 1), keepdims=True)
        x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        y_mean = y_train.mean(); y_std = y_train.std() + 1e-8
        X_train_n = (X_train - x_mean) / x_std; X_val_n = (X_val - x_mean) / x_std
        X_test_n = (X_test - x_mean) / x_std
        y_train_n = (y_train - y_mean) / y_std; y_val_n = (y_val - y_mean) / y_std
        y_test_n = (y_test - y_mean) / y_std
        train_loader = torch.utils.data.DataLoader(
            TimeSeriesDataset(X_train_n, y_train_n), batch_size=B, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            TimeSeriesDataset(X_val_n, y_val_n), batch_size=B, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            TimeSeriesDataset(X_test_n, y_test_n), batch_size=B, shuffle=False)
        all_data.append({"cid": cid, "train_loader": train_loader,
                          "val_loader": val_loader, "test_loader": test_loader,
                          "train_size": len(X_train), "y_mean": y_mean, "y_std": y_std,
                          "meta_test": meta_test})
    return all_data


def compute_metrics(preds, truths):
    mse = float(np.mean((preds - truths) ** 2))
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(preds - truths)))


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.view(-1).float(); b_f = b.view(-1).float()
    dot = float(torch.dot(a_f, b_f))
    na = float(torch.norm(a_f)); nb = float(torch.norm(b_f))
    return max(0.0, dot / (na * nb + 1e-12))


# ══════════════════════════════════════════════════════════════
# 联邦客户端与服务端
# ══════════════════════════════════════════════════════════════

class FederatedClient:
    def __init__(self, client_id, model, train_loader, val_loader, test_loader,
                 criterion, lr=1e-3):
        self.client_id = client_id
        self.model = model.to(DEVICE).float()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for x, y in self.train_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred.view(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item() * x.shape[0]
        return total / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate_loss(self, loader=None):
        if loader is None: loader = self.val_loader
        self.model.eval()
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            total += self.criterion(pred.view(-1), y).item() * x.shape[0]
        return total / len(loader.dataset)

    def train_local(self, epochs=2, global_model=None):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        for _ in range(epochs):
            self.train_epoch()
        val_loss = self.validate_loss()
        return val_loss, copy.deepcopy(self.model.state_dict())

    @torch.no_grad()
    def test_metrics(self, y_mean, y_std):
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            preds.append(pred.view(-1).cpu().numpy())
            truths.append(y.cpu().numpy())
        preds = np.concatenate(preds); truths = np.concatenate(truths)
        preds_raw = preds * y_std + y_mean; truths_raw = truths * y_std + y_mean
        return compute_metrics(preds_raw, truths_raw)

    @torch.no_grad()
    def test_predictions_raw(self, y_mean, y_std):
        """返回 test_loader 上所有样本的逆标准化预测值和真实值。"""
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            preds.append(pred.view(-1).cpu().numpy())
            truths.append(y.cpu().numpy())
        preds = np.concatenate(preds); truths = np.concatenate(truths)
        preds_raw = preds * y_std + y_mean; truths_raw = truths * y_std + y_mean
        return preds_raw, truths_raw

    @torch.no_grad()
    def val_metrics(self, y_mean, y_std):
        """在 val_loader 上计算真实尺度 MSE/RMSE/MAE。"""
        self.model.eval()
        preds, truths = [], []
        for x, y in self.val_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            preds.append(pred.view(-1).cpu().numpy())
            truths.append(y.cpu().numpy())
        preds = np.concatenate(preds); truths = np.concatenate(truths)
        preds_raw = preds * y_std + y_mean; truths_raw = truths * y_std + y_mean
        return compute_metrics(preds_raw, truths_raw)


class FedAvgServer:
    def __init__(self, model, num_clients):
        self.global_model = model.to(DEVICE).float()
        self.num_clients = num_clients
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = sizes

    def aggregate(self, client_weights, client_losses=None):
        w = np.array(self.client_data_sizes, dtype=float) / float(sum(self.client_data_sizes))
        gd = self.global_model.state_dict()
        nd = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gd.items()}
        for key in nd:
            for idx in range(self.num_clients):
                cw = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                nd[key] += cw * torch.tensor(float(w[idx]), device=DEVICE, dtype=torch.float32)
        self.global_model.load_state_dict(nd)
        return self.global_model.state_dict()


class AggregationServer(FedAvgServer):
    def __init__(self, model, num_clients, agg_method="fedavg", lam=0.5, client_data_sizes=None):
        super().__init__(model, num_clients)
        self.agg_method = agg_method
        self.lam = lam

    def aggregate(self, client_weights, client_losses):
        if self.agg_method == "fedavg" or client_losses is None:
            return super().aggregate(client_weights)
        n = np.array(self.client_data_sizes, dtype=float)
        data_w = n / n.sum()
        loss_arr = np.array(client_losses, dtype=float)
        q = 1.0 / (loss_arr + 1e-8); quality_w = q / q.sum()
        if self.agg_method == "loss_weighted":
            w = quality_w
        elif self.agg_method == "data_loss_weighted":
            w = self.lam * data_w + (1.0 - self.lam) * quality_w
        elif self.agg_method == "similarity_aware":
            sim_w = np.ones(self.num_clients) / self.num_clients
            if len(client_weights) > 1:
                flat = []
                for cw in client_weights:
                    flat.append(torch.cat([v.view(-1) for v in cw.values()]))
                sim = np.ones((self.num_clients, self.num_clients))
                for i in range(self.num_clients):
                    for j in range(i + 1, self.num_clients):
                        s = cos_sim(flat[i], flat[j]); sim[i, j] = s; sim[j, i] = s
                sim_scores = sim.mean(axis=1)
                sim_w = sim_scores / (sim_scores.sum() + 1e-8)
            w = 0.3 * data_w + 0.3 * quality_w + 0.4 * sim_w
        elif self.agg_method == "proposed":
            loss_cv = float(np.std(loss_arr)) / (float(np.mean(loss_arr)) + 1e-8)
            dynamic_lam = 1.0 / (1.0 + loss_cv)
            mixed_w = dynamic_lam * data_w + (1.0 - dynamic_lam) * quality_w
            reg_w = np.ones(self.num_clients) / self.num_clients
            w = 0.8 * mixed_w + 0.2 * reg_w
        else:
            w = data_w
        gd = self.global_model.state_dict()
        nd = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gd.items()}
        for key in nd:
            for idx in range(self.num_clients):
                cw = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                nd[key] += cw * torch.tensor(float(w[idx]), device=DEVICE, dtype=torch.float32)
        self.global_model.load_state_dict(nd)
        return self.global_model.state_dict()


# ══════════════════════════════════════════════════════════════
# 统一训练/评估函数
# ══════════════════════════════════════════════════════════════

def run_federated_training(cfgs, graph_type, agg_method, lam=0.5, seed=42,
                           num_nodes=NUM_NODES, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                           comm_rounds=5, local_epochs=2, lr=0.001,
                           record_convergence=False):
    """运行一轮 GCN 联邦训练。

    返回:
        results: list of dict (per-client test metrics)
        convergence: dict (round-level validation data) or None
    """
    set_global_seed(seed)
    client_data = build_client_data(cfgs, num_nodes, seq_len, pred_len, seed)
    nc = len(client_data)
    criterion = nn.MSELoss()
    sizes = [d["train_size"] for d in client_data]

    # 构造邻接矩阵
    raw_signals, _, _ = get_enhanced_raw_data()
    adj = _get_adj_matrix(raw_signals[0], graph_type, num_nodes)

    model_func = lambda: GCNEnhancedModel(k=num_nodes, t=seq_len,
                                           hidden_dim=64, num_heads=4, fixed_adj=adj)

    clients = [
        FederatedClient(d["cid"], model_func(), d["train_loader"],
                        d["val_loader"], d["test_loader"], criterion, lr=lr)
        for d in client_data
    ]
    server = AggregationServer(model_func(), nc, agg_method=agg_method, lam=lam)
    server.set_client_data_sizes(sizes)

    convergence = {"round": [], "avg_train_loss": [], "avg_val_rmse": [], "avg_val_rmse_std": [],
                    "avg_val_mae": [], "avg_val_mae_std": []}
    for cid in range(nc):
        convergence[f"c{cid}_train"] = []
        convergence[f"c{cid}_val_mse"] = []
        convergence[f"c{cid}_val_rmse"] = []
        convergence[f"c{cid}_val_mae"] = []

    for rnd in range(comm_rounds):
        cw_list, cl_list = [], []
        for client in clients:
            cl, cw = client.train_local(epochs=local_epochs,
                                         global_model=server.global_model)
            cw_list.append(cw); cl_list.append(float(cl))
        server.aggregate(cw_list, cl_list)

        if record_convergence:
            val_mses, val_rmses, val_maes = [], [], []
            for cid, client in enumerate(clients):
                client.model.load_state_dict(server.global_model.state_dict())
                cd = client_data[cid]
                mse, rmse, mae = client.val_metrics(cd["y_mean"], cd["y_std"])
                val_mses.append(mse); val_rmses.append(rmse); val_maes.append(mae)
            convergence["round"].append(rnd + 1)
            convergence["avg_train_loss"].append(float(np.mean(cl_list)))
            convergence["avg_val_rmse"].append(float(np.mean(val_rmses)))
            convergence["avg_val_rmse_std"].append(float(np.std(val_rmses, ddof=0)))
            convergence["avg_val_mae"].append(float(np.mean(val_maes)))
            convergence["avg_val_mae_std"].append(float(np.std(val_maes, ddof=0)))
            for cid in range(nc):
                convergence[f"c{cid}_train"].append(cl_list[cid])
                convergence[f"c{cid}_val_mse"].append(val_mses[cid])
                convergence[f"c{cid}_val_rmse"].append(val_rmses[cid])
                convergence[f"c{cid}_val_mae"].append(val_maes[cid])

    results = []
    for cid in range(nc):
        clients[cid].model.load_state_dict(server.global_model.state_dict())
        mse, rmse, mae = clients[cid].test_metrics(
            client_data[cid]["y_mean"], client_data[cid]["y_std"])
        results.append({"client_id": cid, "mse": mse, "rmse": rmse, "mae": mae})

    conv = convergence if record_convergence else None
    return results, conv


def run_independent_training(cfgs, graph_type, seed=42, num_nodes=NUM_NODES,
                             seq_len=SEQ_LEN, pred_len=PRED_LEN,
                             total_epochs=10, lr=0.01):
    """Independent 训练（同一 batch 数据）。"""
    set_global_seed(seed)
    client_data = build_client_data(cfgs, num_nodes, seq_len, pred_len, seed)
    criterion = nn.MSELoss()
    raw_signals, _, _ = get_enhanced_raw_data()
    adj = _get_adj_matrix(raw_signals[0], graph_type, num_nodes)
    results = []
    for d in client_data:
        model = GCNEnhancedModel(k=num_nodes, t=seq_len, hidden_dim=64,
                                  num_heads=4, fixed_adj=adj).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(total_epochs):
            model.train()
            for x, y in d["train_loader"]:
                x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
                opt.zero_grad()
                pred, _ = model(x)
                loss = criterion(pred.view(-1), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y in d["test_loader"]:
                x = x.to(DEVICE).float()
                pred, _ = model(x)
                preds.append(pred.view(-1).cpu().numpy())
                truths.append(y.cpu().numpy())
        preds = np.concatenate(preds); truths = np.concatenate(truths)
        preds_raw = preds * d["y_std"] + d["y_mean"]
        truths_raw = truths * d["y_std"] + d["y_mean"]
        mse, rmse, mae = compute_metrics(preds_raw, truths_raw)
        results.append({"client_id": d["cid"], "mse": mse, "rmse": rmse, "mae": mae})
    return results


def _get_adj_matrix(data, graph_type, num_nodes):
    """根据 graph_type 获取或构造邻接矩阵。"""
    _, A_fixed_norm, _ = build_fixed_adjacency(num_nodes)
    if graph_type == "dynamic_peak_adjacency":
        adj, _, _ = build_dynamic_adjacency(data, "peak")
    elif graph_type == "dynamic_offpeak_adjacency":
        adj, _, _ = build_dynamic_adjacency(data, "off_peak")
    elif graph_type == "functional_similarity_adjacency":
        adj, _, _ = build_functional_similarity_matrix(data)
    elif graph_type == "congestion_delay_adjacency":
        delay_mat, strength_mat, _ = build_congestion_delay_matrix(data, max_lag=5)
        # 将延迟矩阵转为邻接矩阵：延迟小 → 权重高
        max_lag_val = 5.0
        adj = np.maximum(0, 1.0 - delay_mat / max_lag_val) * strength_mat
        adj = adj + np.eye(num_nodes)
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt
        adj = adj.astype(np.float32)
    else:
        adj = A_fixed_norm
    return adj


# ══════════════════════════════════════════════════════════════
# Workflow: data_viz — 增强数据集可视化（与 CNN 一致）
# ══════════════════════════════════════════════════════════════

def run_data_visualization_enhanced(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[data_viz] Enhanced Non-IID Dataset Visualizations (GCN)")
    print("=" * 60)

    raw_signals, raw_masks, cfgs = get_enhanced_raw_data()
    num_clients = len(cfgs)
    ensure_output_dir(output_dir)
    colors = CLIENT_PALETTE[:num_clients]

    # ── 1. client 平均时间序列 ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid in range(num_clients):
        ts = raw_signals[cid].mean(axis=1)[:200]
        sns.lineplot(x=np.arange(len(ts)), y=ts, ax=ax, color=colors[cid], linewidth=2.0, alpha=0.85, label=f"Client {cid}")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Traffic Flow")
    ax.set_title("Per-client average traffic flow")
    ax.legend(fontsize=8, loc="best", ncol=2)
    save_figure(fig, output_dir, "enhanced_dataset_client_timeseries.png")

    # ── 2. 分布对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dist_rows = []
    for cid in range(num_clients):
        sampled_values = raw_signals[cid].ravel()[::10]
        dist_rows.extend({"client": f"Client {cid}", "traffic_flow": float(v)} for v in sampled_values)
    dist_df = pd.DataFrame(dist_rows)
    sns.boxplot(data=dist_df, x="client", y="traffic_flow", ax=axes[0], palette=colors, showfliers=False, linewidth=1.0)
    axes[0].set_title("Non-IID traffic distribution by client")
    axes[0].set_xlabel("Client")
    axes[0].set_ylabel("Traffic flow")
    for cid in range(num_clients):
        sns.histplot(raw_signals[cid].ravel(), bins=35, alpha=0.25, stat="density", element="step", fill=True, color=colors[cid], ax=axes[1], label=f"Client {cid}")
    axes[1].set_title("Traffic flow density comparison")
    axes[1].set_xlabel("Traffic flow"); axes[1].set_ylabel("Density"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_dir, "enhanced_dataset_distribution_comparison.png")

    # ── 3. client 配置概览 ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    config_specs = [
        ([c["n_samples"] for c in cfgs], "Sample size"),
        ([c["noise"] for c in cfgs], "Noise level"),
        ([c["base"] for c in cfgs], "Base traffic flow"),
        ([c.get("incident_prob", 0) for c in cfgs], "Incident probability"),
    ]
    for ax, (values, title) in zip(axes.flatten(), config_specs):
        cfg_df = pd.DataFrame({"client": [f"Client {i}" for i in range(num_clients)], "value": values})
        sns.barplot(data=cfg_df, x="client", y="value", ax=ax, palette=colors, errorbar=None)
        ax.set_title(title)
        ax.set_xlabel("Client")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save_figure(fig, output_dir, "enhanced_dataset_client_config.png")

    # ── 4. 高峰模式 ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid in range(num_clients):
        ts = raw_signals[cid].mean(axis=1)
        sns.lineplot(x=np.arange(24), y=ts[:24], ax=ax, color=colors[cid], linewidth=2.0, alpha=0.85, label=f"Client {cid}")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Traffic Flow")
    ax.set_title("Peak-period traffic patterns")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, output_dir, "enhanced_dataset_peak_pattern.png")

    # ── 5. incident 示例 ──
    incident_cid = 4
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ts = raw_signals[incident_cid].mean(axis=1)
    mask = raw_masks[incident_cid]
    t = np.arange(len(ts))
    sns.lineplot(x=t, y=ts, ax=ax, color=METHOD_PALETTE["FedAvg"], linewidth=2.0, label="Traffic flow")
    in_inc = False; start = 0
    for i in range(len(mask)):
        if mask[i] and not in_inc: start = i; in_inc = True
        elif not mask[i] and in_inc:
            ax.axvspan(start, i, alpha=0.3, color="#e74c3c"); in_inc = False
    if in_inc: ax.axvspan(start, len(mask) - 1, alpha=0.3, color="#e74c3c")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Traffic flow")
    ax.set_title(f"Incident example for client {incident_cid}")
    ax.legend()
    save_figure(fig, output_dir, "enhanced_dataset_incident_example.png")

    # ── 6. client 间相关系数矩阵 ──
    min_len = min(len(s) for s in raw_signals)
    client_ts = np.array([raw_signals[cid].mean(axis=1)[:min_len] for cid in range(num_clients)])
    corr_client = np.corrcoef(client_ts)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, corr_client, title="Inter-client correlation matrix", xlabel="Client ID", ylabel="Client ID", cmap="vlag", cbar_label="Correlation", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f")
    save_figure(fig, output_dir, "enhanced_dataset_client_correlation_matrix.png")

    # ── 7. 节点间相关系数矩阵 ──
    rep_cid = 0
    node_corr = np.corrcoef(raw_signals[rep_cid].T)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, node_corr, title=f"Node correlation matrix for client {rep_cid}", xlabel="Node ID", ylabel="Node ID", cmap="vlag", cbar_label="Correlation", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f")
    save_figure(fig, output_dir, "enhanced_dataset_node_correlation_matrix.png")

    # ── 8. 汇总 CSV ──
    rows = []
    for cid, cfg in enumerate(cfgs):
        ts = raw_signals[cid].ravel()
        rows.append({
            "client_id": cid, "distribution_type": cfg["dist"], "traffic_pattern": cfg["pattern"],
            "sample_size": cfg["n_samples"], "noise_level": cfg["noise"], "base_flow": cfg["base"],
            "morning_mu": cfg["morning_mu"], "evening_mu": cfg["evening_mu"],
            "morning_amp": cfg["morning_amp"], "evening_amp": cfg["evening_amp"],
            "incident_prob": cfg.get("incident_prob", 0),
            "mean_flow": float(np.mean(ts)), "std_flow": float(np.std(ts)),
            "min_flow": float(np.min(ts)), "max_flow": float(np.max(ts)),
        })
    df_sum = pd.DataFrame(rows)
    save_dataframe(df_sum, output_dir, "enhanced_dataset_summary.csv")
    print("[data_viz] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow: fixed_vs_dynamic — 图结构可视化
# ══════════════════════════════════════════════════════════════

def run_fixed_vs_dynamic_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[fixed_vs_dynamic] GCN Graph Structure Visualization")
    print("=" * 60)

    raw_signals, raw_masks, cfgs = get_enhanced_raw_data()
    rep_cid = 0  # 使用第一个 client 作为代表性节点
    data = raw_signals[rep_cid]  # (n_timesteps, num_nodes)
    num_nodes = data.shape[1]
    ensure_output_dir(output_dir)

    # ── 构造各类型邻接矩阵 ──
    _, A_fixed_raw, meta_fixed = build_fixed_adjacency(num_nodes)

    _, A_dyn_morning_raw, meta_morning = build_dynamic_adjacency(data, "morning_peak")
    _, A_dyn_evening_raw, meta_evening = build_dynamic_adjacency(data, "evening_peak")
    _, A_dyn_offpeak_raw, meta_offpeak = build_dynamic_adjacency(data, "off_peak")

    _, A_func_raw, meta_func = build_functional_similarity_matrix(data)

    A_delay_raw, _, meta_delay = build_congestion_delay_matrix(data, max_lag=5)

    # ── 1. 固定邻接矩阵热力图 ──
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, A_fixed_raw, title="Fixed adjacency matrix", xlabel="Node ID", ylabel="Node ID", cmap="mako", cbar_label="Adjacency weight", vmin=0, vmax=1, annot=True, fmt=".2f")
    save_figure(fig, output_dir, "enhanced_gcn_fixed_adjacency_matrix.png")

    # ── 2. 高峰期动态邻接矩阵（早高峰） ──
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, A_dyn_morning_raw, title="Dynamic adjacency during morning peak", xlabel="Node ID", ylabel="Node ID", cmap="viridis", cbar_label="Correlation", vmin=0, vmax=1, annot=True, fmt=".2f")
    save_figure(fig, output_dir, "enhanced_gcn_dynamic_adjacency_peak.png")

    # ── 3. 平峰期动态邻接矩阵 ──
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, A_dyn_offpeak_raw, title="Dynamic adjacency during off-peak", xlabel="Node ID", ylabel="Node ID", cmap="viridis", cbar_label="Correlation", vmin=0, vmax=1, annot=True, fmt=".2f")
    save_figure(fig, output_dir, "enhanced_gcn_dynamic_adjacency_offpeak.png")

    # ── 4. 固定 vs 动态对比图 ──
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    adj_list = [
        (A_fixed_raw, "Fixed Topology", "Blues"),
        (A_dyn_morning_raw, "Dynamic Morning Peak", "Oranges"),
        (A_dyn_evening_raw, "Dynamic Evening Peak", "Reds"),
        (A_dyn_offpeak_raw, "Dynamic Off-Peak", "Greens"),
    ]
    for idx, (mat, title, cmap) in enumerate(adj_list):
        ax = axes[idx]
        plot_heatmap(ax, mat, title=title, xlabel="Node ID", ylabel="Node ID", cmap="mako" if idx == 0 else "viridis", cbar_label="Weight", vmin=0, vmax=1)
    fig.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_fixed_dynamic_adjacency_comparison.png")

    # ── 5. 功能相似矩阵 ──
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_heatmap(ax, A_func_raw, title="Functional similarity matrix", xlabel="Node ID", ylabel="Node ID", cmap="mako", cbar_label="Similarity", vmin=0, vmax=1, annot=True, fmt=".2f")
    save_figure(fig, output_dir, "enhanced_gcn_functional_similarity_matrix.png")

    # ── 6. 拥堵传播延迟矩阵 ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_heatmap(axes[0], A_delay_raw, title="Congestion delay matrix", xlabel="Target Node ID", ylabel="Source Node ID", cmap="rocket_r", cbar_label="Delay (steps)", vmin=0, vmax=5, annot=True, fmt=".0f")
    plot_heatmap(axes[1], A_delay_raw, title="Annotated congestion delay matrix", xlabel="Target Node ID", ylabel="Source Node ID", cmap="rocket_r", cbar_label="Delay (steps)", vmin=0, vmax=5, annot=False, fmt=".0f")
    # 标注 source->target 的方向
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and A_delay_raw[i, j] < 3:
                axes[1].annotate("", xy=(j, i), xytext=(j - 0.15, i - 0.15),
                                 arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    fig.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_congestion_delay_matrix.png")

    # ── 7. 高峰/平峰图结构变化对比 ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    period_mats = [
        (A_dyn_offpeak_raw, "Off-Peak", "Greens"),
        (A_dyn_morning_raw, "Morning Peak (07-09)", "Oranges"),
        (A_dyn_evening_raw, "Evening Peak (17-19)", "Reds"),
    ]
    for idx, (mat, title, cmap) in enumerate(period_mats):
        ax = axes[idx]
        plot_heatmap(ax, mat, title=title, xlabel="Node ID", ylabel="Node ID", cmap="viridis", cbar_label="Weight", vmin=0, vmax=1)
    fig.suptitle("Graph structure changes across traffic periods", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_peak_graph_change.png")

    # ── 8. 图结构汇总 CSV ──
    all_meta = [meta_fixed, meta_morning, meta_evening, meta_offpeak, meta_func, meta_delay]
    df_meta = pd.DataFrame(all_meta)
    save_dataframe(df_meta, output_dir, "enhanced_gcn_graph_summary.csv")
    print("[fixed_vs_dynamic] Graph summary:\n", df_meta.to_string(index=False))

    # ── 训练对比实验 ──
    print("\n[fixed_vs_dynamic] Running training comparison...")
    cfgs = list(CLIENT_CONFIGS_BASE)
    graph_types = ["fixed_adjacency", "dynamic_peak_adjacency", "dynamic_offpeak_adjacency",
                   "functional_similarity_adjacency"]
    all_train_rows = []
    for gt in graph_types:
        print(f"  Graph: {gt}")
        seed = 42
        fed, _ = run_federated_training(cfgs, gt, "fedavg", seed=seed, comm_rounds=5, local_epochs=2)
        prop, _ = run_federated_training(cfgs, gt, "proposed", seed=seed, comm_rounds=5, local_epochs=2)
        for r in fed:
            all_train_rows.append({"seed": seed, "graph_type": gt, "method": "GCN-FedAvg",
                                   "client_id": r["client_id"],
                                   "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
        for r in prop:
            all_train_rows.append({"seed": seed, "graph_type": gt, "method": "GCN-Proposed",
                                   "client_id": r["client_id"],
                                   "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df_train = pd.DataFrame(all_train_rows)
    save_dataframe(df_train, output_dir, "gcn_enhanced_fixed_vs_dynamic_metrics.csv")
    agg_train = df_train.groupby(["graph_type", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg_train, output_dir, "gcn_enhanced_fixed_vs_dynamic_summary.csv")
    print("\n[fixed_vs_dynamic] Training results:\n", agg_train.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    gts_short = {"fixed_adjacency": "Fixed", "dynamic_peak_adjacency": "Dyn-Peak",
                 "dynamic_offpeak_adjacency": "Dyn-Off", "functional_similarity_adjacency": "Func"}
    for gt in graph_types:
        sub = agg_train[agg_train["graph_type"] == gt]
        for method in ["GCN-FedAvg", "GCN-Proposed"]:
            row = sub[sub["method"] == method]
            if len(row) > 0:
                gt_label = gts_short.get(gt, gt)
                # Simple grouped bar
                x_idx = graph_types.index(gt) * 2 + (0 if method == "GCN-FedAvg" else 1)
                color = "#3498db" if method == "GCN-FedAvg" else "#2ecc71"
                axes[0].bar(x_idx, row["rmse_mean"].values[0], color=color, alpha=0.85)
                axes[1].bar(x_idx, row["mae_mean"].values[0], color=color, alpha=0.85)
    for ax in axes:
        ax.set_xticks([i * 2 + 0.5 for i in range(len(graph_types))])
        ax.set_xticklabels([gts_short[gt] for gt in graph_types], fontsize=9)
        ax.set_xlabel("Graph Type")
    axes[0].set_title("RMSE by Graph Type"); axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE by Graph Type"); axes[1].set_ylabel("MAE")
    axes[0].legend(["GCN-FedAvg", "GCN-Proposed"], fontsize=8)
    plt.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_fixed_vs_dynamic_comparison.png")
    print("[fixed_vs_dynamic] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow: congestion_delay — 拥堵传播延迟数据与可视化
# ══════════════════════════════════════════════════════════════

def run_congestion_delay_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[congestion_delay] Congestion Propagation & Delay Visualization")
    print("=" * 60)

    raw_signals, raw_masks, cfgs = get_enhanced_raw_data()
    rep_cid = 0
    data = raw_signals[rep_cid]
    num_nodes = data.shape[1]
    ensure_output_dir(output_dir)

    # 构造延迟矩阵
    delay_mat, strength_mat, meta_delay = build_congestion_delay_matrix(data, max_lag=5)

    # ── 图 1: 拥堵传播延迟矩阵（彩色） ──
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(delay_mat, cmap="YlOrRd", aspect="equal", vmin=0, vmax=5)
    ax.set_title("GCN Enhanced: Congestion Propagation Delay (Client {})".format(rep_cid))
    ax.set_xlabel("Target Node ID"); ax.set_ylabel("Source Node ID")
    for i in range(num_nodes):
        for j in range(num_nodes):
            ax.text(j, i, f"{int(delay_mat[i, j])}", ha="center", va="center",
                    fontsize=8, color="white" if delay_mat[i, j] > 2 else "black")
    plt.colorbar(im, ax=ax, label="Delay (steps)", shrink=0.8)
    save_figure(fig, output_dir, "enhanced_gcn_congestion_delay_matrix.png")

    # ── 图 2: 传播强度矩阵 ──
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(strength_mat, cmap="RdYlGn", aspect="equal", vmin=0, vmax=1)
    ax.set_title("GCN Enhanced: Congestion Propagation Strength (Correlation)")
    ax.set_xlabel("Target Node ID"); ax.set_ylabel("Source Node ID")
    for i in range(num_nodes):
        for j in range(num_nodes):
            ax.text(j, i, f"{strength_mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="|Correlation|", shrink=0.8)
    save_figure(fig, output_dir, "enhanced_gcn_congestion_strength_matrix.png")

    # ── 图 3: 延迟分布直方图 ──
    delays_flat = delay_mat[~np.eye(num_nodes, dtype=bool)].ravel()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delays_flat, bins=np.arange(-0.5, 6.5, 1), edgecolor="white",
            color="#e67e22", alpha=0.8)
    ax.set_xlabel("Delay (steps)"); ax.set_ylabel("Count")
    ax.set_title("GCN Enhanced: Congestion Delay Distribution")
    ax.set_xticks(range(6))
    save_figure(fig, output_dir, "enhanced_gcn_congestion_delay_distribution.png")

    # ── 图 4: 代表性节点对的延迟交互 ──
    # 选取延迟最小的几对节点（非对角）
    delay_off_diag = delay_mat.copy()
    np.fill_diagonal(delay_off_diag, 99)
    flat_idx = np.argsort(delay_off_diag.ravel())
    top_k = min(4, len(flat_idx))

    fig, axes = plt.subplots(top_k, 1, figsize=(14, 3 * top_k))
    if top_k == 1: axes = [axes]
    hours = (np.arange(len(data)) * 24.0 / len(data)) % 24
    for k in range(top_k):
        si, sj = np.unravel_index(flat_idx[k], delay_mat.shape)
        ax = axes[k]
        ax.plot(hours, data[:, si], label=f"Source Node {si}", linewidth=1.5, color="#3498db")
        ax.plot(hours, data[:, sj], label=f"Target Node {sj}", linewidth=1.5, color="#e74c3c")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Traffic Flow")
        lag_val = int(delay_mat[si, sj])
        ax.set_title(f"Node {si} -> Node {sj} (Delay={lag_val} steps, Corr={strength_mat[si, sj]:.3f})")
        ax.legend(fontsize=9)
        # 标注早晚高峰
        ax.axvspan(7, 9, alpha=0.1, color="orange")
        ax.axvspan(17, 19, alpha=0.1, color="orange")
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_congestion_delay_interaction.png")

    # ── 保存 CSV ──
    delay_rows = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                delay_rows.append({
                    "source_node": i, "target_node": j,
                    "delay_steps": int(delay_mat[i, j]),
                    "correlation": float(strength_mat[i, j]),
                })
    df_delay = pd.DataFrame(delay_rows)
    save_dataframe(df_delay, output_dir, "enhanced_gcn_congestion_delay.csv")

    # ── 训练对比：fixed vs functional vs congestion_delay ──
    print("\n[congestion_delay] Running graph-type training comparison...")
    cfgs = list(CLIENT_CONFIGS_BASE)
    graph_types = ["fixed_adjacency", "functional_similarity_adjacency", "congestion_delay_adjacency"]
    seed = 42
    all_rows = []
    for gt in graph_types:
        print(f"  Graph: {gt}")
        # FedAvg
        fed, _ = run_federated_training(cfgs, gt, "fedavg", seed=seed, comm_rounds=5, local_epochs=2)
        # Proposed
        prop, _ = run_federated_training(cfgs, gt, "proposed", seed=seed, comm_rounds=5, local_epochs=2)
        for r in fed:
            all_rows.append({"seed": seed, "graph_type": gt, "method": "GCN-FedAvg",
                             "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
        for r in prop:
            all_rows.append({"seed": seed, "graph_type": gt, "method": "GCN-Proposed",
                             "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df_cd = pd.DataFrame(all_rows)
    save_dataframe(df_cd, output_dir, "gcn_enhanced_congestion_delay_metrics.csv")
    agg_cd = df_cd.groupby(["graph_type", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg_cd, output_dir, "gcn_enhanced_congestion_delay_summary.csv")
    print("\n[congestion_delay] Training results:\n", agg_cd.to_string(index=False))

    # 图：不同 graph_type 的 RMSE 对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gt_short = {"fixed_adjacency": "Fixed", "functional_similarity_adjacency": "Func",
                "congestion_delay_adjacency": "Delay"}
    for gt in graph_types:
        sub = agg_cd[agg_cd["graph_type"] == gt]
        for method, color in [("GCN-FedAvg", "#3498db"), ("GCN-Proposed", "#2ecc71")]:
            row = sub[sub["method"] == method]
            if len(row) > 0:
                x_idx = graph_types.index(gt) * 2 + (0 if method == "GCN-FedAvg" else 1)
                axes[0].bar(x_idx, row["rmse_mean"].values[0], color=color, alpha=0.85)
                axes[1].bar(x_idx, row["mae_mean"].values[0], color=color, alpha=0.85)
    for ax in axes:
        ax.set_xticks([i * 2 + 0.5 for i in range(len(graph_types))])
        ax.set_xticklabels([gt_short[gt] for gt in graph_types], fontsize=9)
        ax.set_xlabel("Graph Type")
    axes[0].set_title("RMSE by Graph Type"); axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE by Graph Type"); axes[1].set_ylabel("MAE")
    axes[0].legend(["GCN-FedAvg", "GCN-Proposed"], fontsize=8)
    plt.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_congestion_delay_comparison.png")
    print("[congestion_delay] Done.\n")


# ══════════════════════════════════════════════════════════════
# 占位 workflow（后续批次补充）
# ══════════════════════════════════════════════════════════════

def run_main_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[main] GCN Enhanced Main Experiment")
    print("=" * 60)
    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)
    seeds = [42, 2024, 2025]
    graph_type = "fixed_adjacency"

    all_rows = []
    for seed in seeds:
        print(f"\n--- Seed = {seed} ---")
        # Independent
        ind = run_independent_training(cfgs, graph_type, seed=seed, total_epochs=10, lr=0.01)
        for r in ind:
            all_rows.append({"seed": seed, "method": "Independent", "graph_type": graph_type,
                             "aggregation_method": "none", "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
        # GCN-FedAvg
        fed, _ = run_federated_training(cfgs, graph_type, "fedavg", seed=seed, comm_rounds=5, local_epochs=2)
        for r in fed:
            all_rows.append({"seed": seed, "method": "GCN-FedAvg", "graph_type": graph_type,
                             "aggregation_method": "fedavg", "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
        # GCN-Proposed
        prop, _ = run_federated_training(cfgs, graph_type, "proposed", seed=seed, comm_rounds=5, local_epochs=2)
        for r in prop:
            all_rows.append({"seed": seed, "method": "GCN-Proposed", "graph_type": graph_type,
                             "aggregation_method": "proposed", "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "gcn_enhanced_main_metrics.csv")
    agg = df.groupby(["method", "graph_type", "aggregation_method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg, output_dir, "gcn_enhanced_main_metrics_summary.csv")
    print("\n[main] Summary:\n", agg.to_string(index=False))

    methods = ["Independent", "GCN-FedAvg", "GCN-Proposed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, metric in enumerate(["rmse", "mae"]):
        plot_df = pd.DataFrame(
            {
                "method": methods,
                "value": [agg[agg["method"] == m][f"{metric}_mean"].values[0] for m in methods],
                "error": [agg[agg["method"] == m][f"{metric}_std"].values[0] for m in methods],
            }
        )
        sns.barplot(data=plot_df, x="method", y="value", order=methods, palette=METHOD_PALETTE, ax=axes[idx], errorbar=None)
        axes[idx].errorbar(np.arange(len(plot_df)), plot_df["value"], yerr=plot_df["error"], fmt="none", ecolor="0.25", elinewidth=1.0, capsize=4)
        axes[idx].set_title(metric.upper())
        axes[idx].set_xlabel("Method")
        axes[idx].set_ylabel(f"{metric.upper()} (lower is better)")
        axes[idx].tick_params(axis="x", rotation=15)
    fig.suptitle("GCN enhanced main comparison", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_main_rmse_comparison.png")
    print("[main] Done.\n")


def run_aggregation_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[aggregation] GCN Aggregation Strategy Ablation")
    print("=" * 60)
    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)
    graph_type = "fixed_adjacency"
    agg_methods = ["fedavg", "loss_weighted", "data_loss_weighted", "similarity_aware", "proposed"]
    agg_labels = ["FedAvg", "Loss-weighted", "Data-loss", "Similarity", "Proposed"]

    all_rows = []
    for seed in [42]:
        print(f"\n--- Seed = {seed} ---")
        for method, label in zip(agg_methods, agg_labels):
            print(f"  [{label}]")
            results, _ = run_federated_training(cfgs, graph_type, method, seed=seed,
                                               comm_rounds=5, local_epochs=2)
            for r in results:
                all_rows.append({"seed": seed, "aggregation_method": label, "graph_type": graph_type,
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "gcn_enhanced_aggregation_ablation.csv")
    agg = df.groupby(["aggregation_method", "graph_type"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg, output_dir, "gcn_enhanced_aggregation_ablation_summary.csv")
    print("\n[aggregation] Summary:\n", agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, metric in enumerate(["rmse", "mae"]):
        plot_df = pd.DataFrame(
            {
                "aggregation_method": agg_labels,
                "value": [agg[agg["aggregation_method"] == l][f"{metric}_mean"].values[0] for l in agg_labels],
                "error": [agg[agg["aggregation_method"] == l][f"{metric}_std"].values[0] for l in agg_labels],
            }
        )
        palette = [METHOD_PALETTE.get(label, CLIENT_PALETTE[i % len(CLIENT_PALETTE)]) for i, label in enumerate(agg_labels)]
        sns.barplot(data=plot_df, x="aggregation_method", y="value", ax=axes[idx], palette=palette, errorbar=None)
        axes[idx].errorbar(np.arange(len(plot_df)), plot_df["value"], yerr=plot_df["error"], fmt="none", ecolor="0.25", elinewidth=1.0, capsize=4)
        axes[idx].set_xticklabels(agg_labels, rotation=20, ha="right", fontsize=8)
        axes[idx].set_title(f"{metric.upper()} by aggregation strategy")
        axes[idx].set_xlabel("Aggregation strategy")
        axes[idx].set_ylabel(f"{metric.upper()} (lower is better)")
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_aggregation_ablation.png")
    print("[aggregation] Done.\n")


def run_lambda_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[lambda] Lambda Sensitivity Analysis (GCN)")
    print("=" * 60)
    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)
    graph_type = "fixed_adjacency"
    lam_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_rows = []
    for seed in [42]:
        print(f"\n--- Seed = {seed} ---")
        for lam in lam_vals:
            print(f"  [lambda={lam:.2f}]")
            results, _ = run_federated_training(cfgs, graph_type, "data_loss_weighted",
                                              lam=lam, seed=seed, comm_rounds=5, local_epochs=2)
            for r in results:
                all_rows.append({"seed": seed, "lambda": lam, "graph_type": graph_type,
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "gcn_enhanced_lambda_sensitivity.csv")
    agg = df.groupby(["lambda", "graph_type"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg, output_dir, "gcn_enhanced_lambda_sensitivity_summary.csv")
    print("\n[lambda] Summary:\n", agg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(lam_vals, agg["rmse_mean"], yerr=agg["rmse_std"], fmt="o-", capsize=5, label="RMSE", linewidth=2, color=METHOD_PALETTE["GCN-FedAvg"])
    ax2 = ax.twinx()
    ax2.errorbar(lam_vals, agg["mae_mean"], yerr=agg["mae_std"], fmt="s-", capsize=5, label="MAE", linewidth=2, color=METHOD_PALETTE["Independent"])
    ax.set_xlabel("Lambda (data_weight fraction)"); ax.set_ylabel("RMSE (lower is better)", color=METHOD_PALETTE["GCN-FedAvg"])
    ax2.set_ylabel("MAE (lower is better)", color=METHOD_PALETTE["Independent"])
    ax.set_title("GCN Data-Loss Weighted: Lambda Sensitivity")
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="center right")
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_lambda_sensitivity.png")
    print("[lambda] Done.\n")


def run_client_scale_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[client_scale] GCN Client Count Sensitivity Experiment")
    print("=" * 60)

    client_nums = [3, 5, 8, 10]
    graph_type = "fixed_adjacency"
    seeds = [42, 2024, 2025]
    ensure_output_dir(output_dir)

    all_rows = []
    for nc in client_nums:
        print(f"\n--- Num Clients = {nc} ---")
        cfgs = build_noniid_client_configs(nc, "medium")
        for seed in seeds:
            print(f"  Seed = {seed}")
            # Independent
            ind = run_independent_training(cfgs, graph_type, seed=seed, total_epochs=10, lr=0.01)
            for r in ind:
                all_rows.append({"seed": seed, "num_clients": nc, "graph_type": graph_type,
                                 "method": "Independent", "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
            # GCN-FedAvg
            fed, _ = run_federated_training(cfgs, graph_type, "fedavg", seed=seed,
                                          comm_rounds=5, local_epochs=2)
            for r in fed:
                all_rows.append({"seed": seed, "num_clients": nc, "graph_type": graph_type,
                                 "method": "GCN-FedAvg", "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
            # GCN-Proposed
            prop, _ = run_federated_training(cfgs, graph_type, "proposed", seed=seed,
                                           comm_rounds=5, local_epochs=2)
            for r in prop:
                all_rows.append({"seed": seed, "num_clients": nc, "graph_type": graph_type,
                                 "method": "GCN-Proposed", "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "gcn_enhanced_client_scale_metrics.csv")
    agg = df.groupby(["num_clients", "graph_type", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg, output_dir, "gcn_enhanced_client_scale_summary.csv")
    print("\n[client_scale] Summary:\n", agg.to_string(index=False))

    methods = ["Independent", "GCN-FedAvg", "GCN-Proposed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        sub = agg[agg["method"] == method].sort_values("num_clients")
        x_pos = np.arange(len(sub))
        axes[0].errorbar(x_pos, sub["rmse_mean"], yerr=sub["rmse_std"],
                         fmt="o-", capsize=5, label=method, linewidth=2, color=METHOD_PALETTE[method])
        axes[1].errorbar(x_pos, sub["mae_mean"], yerr=sub["mae_std"],
                         fmt="s-", capsize=5, label=method, linewidth=2, color=METHOD_PALETTE[method])
    for ax in axes:
        ax.set_xticks(range(len(client_nums)))
        ax.set_xticklabels([str(n) for n in client_nums])
        ax.set_xlabel("Number of Clients"); ax.legend(fontsize=9)
    axes[0].set_title("GCN RMSE vs Client Count"); axes[0].set_ylabel("RMSE")
    axes[1].set_title("GCN MAE vs Client Count"); axes[1].set_ylabel("MAE")
    fig.suptitle("Client scale sensitivity", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_client_scale.png")
    print("[client_scale] Done.\n")


def run_noniid_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[noniid] GCN Non-IID Strength Experiment")
    print("=" * 60)

    levels = ["low", "medium", "high"]
    num_clients = 5
    graph_type = "fixed_adjacency"
    seeds = [42, 2024, 2025]
    ensure_output_dir(output_dir)

    all_rows = []
    for level in levels:
        print(f"\n--- Non-IID Level = {level} ---")
        cfgs = build_noniid_client_configs(num_clients, level)
        for seed in seeds:
            print(f"  Seed = {seed}")
            # Independent
            ind = run_independent_training(cfgs, graph_type, seed=seed, total_epochs=10, lr=0.01)
            for r in ind:
                all_rows.append({"seed": seed, "noniid_level": level, "graph_type": graph_type,
                                 "method": "Independent", "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
            # GCN-FedAvg
            fed, _ = run_federated_training(cfgs, graph_type, "fedavg", seed=seed,
                                          comm_rounds=5, local_epochs=2)
            for r in fed:
                all_rows.append({"seed": seed, "noniid_level": level, "graph_type": graph_type,
                                 "method": "GCN-FedAvg", "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})
            # GCN-Proposed
            prop, _ = run_federated_training(cfgs, graph_type, "proposed", seed=seed,
                                           comm_rounds=5, local_epochs=2)
            for r in prop:
                all_rows.append({"seed": seed, "noniid_level": level, "graph_type": graph_type,
                                 "method": "GCN-Proposed", "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "gcn_enhanced_noniid_strength_metrics.csv")
    agg = df.groupby(["noniid_level", "graph_type", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std")).reset_index()
    save_dataframe(agg, output_dir, "gcn_enhanced_noniid_strength_summary.csv")
    print("\n[noniid] Summary:\n", agg.to_string(index=False))

    level_order = ["low", "medium", "high"]
    methods = ["Independent", "GCN-FedAvg", "GCN-Proposed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    plot_df = agg.copy()
    plot_df["noniid_level"] = pd.Categorical(plot_df["noniid_level"], categories=level_order, ordered=True)
    plot_df = plot_df.sort_values("noniid_level")
    for idx, metric in enumerate(["rmse", "mae"]):
        sns.barplot(data=plot_df, x="noniid_level", y=f"{metric}_mean", hue="method", hue_order=methods, palette=METHOD_PALETTE, ax=axes[idx], errorbar=None)
        axes[idx].set_title(f"{metric.upper()} by Non-IID strength")
        axes[idx].set_xlabel("Non-IID level")
        axes[idx].set_ylabel(f"{metric.upper()} (lower is better)")
        if idx == 0:
            axes[idx].legend(title=None, fontsize=8)
        else:
            axes[idx].get_legend().remove()
    fig.suptitle("Non-IID strength sensitivity", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_noniid_strength.png")
    print("[noniid] Done.\n")


def run_convergence_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[convergence] GCN FedAvg vs Proposed Convergence Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    graph_type = "fixed_adjacency"
    ensure_output_dir(output_dir)
    conv_rounds = 10
    all_round_rows = []

    for method, agg_method in [("GCN-FedAvg", "fedavg"), ("GCN-Proposed", "proposed")]:
        print(f"\n--- {method} Convergence (seed=42) ---")
        set_global_seed(42)
        _, conv = run_federated_training(
            cfgs, graph_type, agg_method, seed=42,
            comm_rounds=conv_rounds, record_convergence=True)

        for r_idx in range(len(conv["round"])):
            rnd = conv["round"][r_idx]
            for cid in range(num_clients):
                all_round_rows.append({
                    "round": rnd, "method": method, "graph_type": graph_type,
                    "client_id": cid,
                    "train_loss": conv[f"c{cid}_train"][r_idx],
                    "val_mse": conv[f"c{cid}_val_mse"][r_idx],
                    "val_rmse": conv[f"c{cid}_val_rmse"][r_idx],
                    "val_mae": conv[f"c{cid}_val_mae"][r_idx],
                })

    df_round = pd.DataFrame(all_round_rows)
    save_dataframe(df_round, output_dir, "gcn_enhanced_convergence_round_metrics.csv")

    agg = df_round.groupby(["round", "method", "graph_type"]).agg(
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_mae_mean=("val_mae", "mean"), val_mae_std=("val_mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "gcn_enhanced_convergence_summary.csv")
    print("\n[convergence] Summary (last round):")
    last = agg[agg["round"] == agg["round"].max()]
    print(last.to_string(index=False))

    # ── 图 1: Global Validation RMSE ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in ["GCN-FedAvg", "GCN-Proposed"]:
        sub = agg[agg["method"] == method]
        color = METHOD_PALETTE[method]
        sns.lineplot(data=sub, x="round", y="val_rmse_mean", marker="o", color=color, linewidth=2.0, ax=ax, label=method)
        ax.fill_between(sub["round"],
                         sub["val_rmse_mean"] - sub["val_rmse_std"],
                         sub["val_rmse_mean"] + sub["val_rmse_std"],
                         alpha=0.15, color=color)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Validation RMSE")
    ax.set_title("Global validation RMSE across communication rounds")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_global_validation_rmse.png")

    # ── 图 2: Client Training Loss ──
    fedavg_sub = df_round[df_round["method"] == "GCN-FedAvg"]
    prop_sub = df_round[df_round["method"] == "GCN-Proposed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cid in range(num_clients):
        cdata = fedavg_sub[fedavg_sub["client_id"] == cid]
        sns.lineplot(data=cdata, x="round", y="train_loss", ax=axes[0], color=CLIENT_PALETTE[cid % len(CLIENT_PALETTE)], linewidth=2.0, alpha=0.8, label=f"C{cid}")
    axes[0].set_xlabel("Communication Round"); axes[0].set_ylabel("Training loss")
    axes[0].set_title("GCN-FedAvg client training loss"); axes[0].legend(fontsize=7)
    for cid in range(num_clients):
        cdata = prop_sub[prop_sub["client_id"] == cid]
        sns.lineplot(data=cdata, x="round", y="train_loss", ax=axes[1], color=CLIENT_PALETTE[cid % len(CLIENT_PALETTE)], linewidth=2.0, alpha=0.8, label=f"C{cid}")
    axes[1].set_xlabel("Communication Round"); axes[1].set_ylabel("Training loss")
    axes[1].set_title("GCN-Proposed client training loss"); axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_client_training_loss.png")

    # ── 图 3: Convergence Overview ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    for method, color in [("GCN-FedAvg", "#3498db"), ("GCN-Proposed", "#2ecc71")]:
        sub = agg[agg["method"] == method]
        ax.plot(sub["round"], sub["val_rmse_mean"], "o-", color=color, linewidth=2, label=method)
        ax.fill_between(sub["round"], sub["val_rmse_mean"] - sub["val_rmse_std"],
                         sub["val_rmse_mean"] + sub["val_rmse_std"], alpha=0.12, color=color)
    ax.set_xlabel("Comm Round"); ax.set_ylabel("Val RMSE")
    ax.set_title("(a) Global Validation RMSE"); ax.legend()

    ax = axes[0, 1]
    for method, color in [("GCN-FedAvg", "#3498db"), ("GCN-Proposed", "#2ecc71")]:
        sub = agg[agg["method"] == method]
        ax.plot(sub["round"], sub["val_mae_mean"], "s--", color=color, linewidth=2, label=method)
    ax.set_xlabel("Comm Round"); ax.set_ylabel("Val MAE")
    ax.set_title("(b) Global Validation MAE"); ax.legend()

    ax = axes[1, 0]
    for cid in range(num_clients):
        cdata = fedavg_sub[fedavg_sub["client_id"] == cid]
        ax.plot(cdata["round"], cdata["train_loss"], "o-", label=f"C{cid}", markersize=3)
    ax.set_xlabel("Comm Round"); ax.set_ylabel("Train Loss")
    ax.set_title("(c) GCN-FedAvg: Client Training Loss"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    for cid in range(num_clients):
        cdata = prop_sub[prop_sub["client_id"] == cid]
        ax.plot(cdata["round"], cdata["train_loss"], "s--", label=f"C{cid}", markersize=3)
    ax.set_xlabel("Comm Round"); ax.set_ylabel("Train Loss")
    ax.set_title("(d) GCN-Proposed: Client Training Loss"); ax.legend(fontsize=7)
    fig.suptitle("GCN Enhanced: Convergence Overview", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_convergence_overview.png")
    print("[convergence] Done.\n")


def run_client_metrics_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[client_metrics] GCN Per-Client Error Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    graph_type = "fixed_adjacency"
    ensure_output_dir(output_dir)
    seed = 42

    set_global_seed(seed)
    # Independent
    print("[Independent]")
    ind_results = run_independent_training(cfgs, graph_type, seed=seed, total_epochs=10, lr=0.01)
    # GCN-FedAvg
    print("[GCN-FedAvg]")
    fed_results, _ = run_federated_training(cfgs, graph_type, "fedavg", seed=seed, comm_rounds=5, local_epochs=2)
    # GCN-Proposed
    print("[GCN-Proposed]")
    prop_results, _ = run_federated_training(cfgs, graph_type, "proposed", seed=seed, comm_rounds=5, local_epochs=2)

    rows = []
    for cid in range(num_clients):
        cfg = cfgs[cid]
        im = ind_results[cid]; fm = fed_results[cid]; pm = prop_results[cid]
        imp_fedavg = (fm["rmse"] - pm["rmse"]) / (fm["rmse"] + 1e-12) * 100
        imp_ind = (im["rmse"] - pm["rmse"]) / (im["rmse"] + 1e-12) * 100

        rows.append({"method": "Independent", "graph_type": graph_type, "client_id": cid,
                     "distribution_type": cfg["dist"], "traffic_pattern": cfg["pattern"],
                     "sample_size": cfg["n_samples"], "noise_level": cfg["noise"],
                     "incident_prob": cfg.get("incident_prob", 0),
                     "mse": im["mse"], "rmse": im["rmse"], "mae": im["mae"],
                     "improvement_over_fedavg_rmse": float("nan"),
                     "improvement_over_independent_rmse": float("nan")})
        rows.append({"method": "GCN-FedAvg", "graph_type": graph_type, "client_id": cid,
                     "distribution_type": cfg["dist"], "traffic_pattern": cfg["pattern"],
                     "sample_size": cfg["n_samples"], "noise_level": cfg["noise"],
                     "incident_prob": cfg.get("incident_prob", 0),
                     "mse": fm["mse"], "rmse": fm["rmse"], "mae": fm["mae"],
                     "improvement_over_fedavg_rmse": float("nan"),
                     "improvement_over_independent_rmse": float("nan")})
        rows.append({"method": "GCN-Proposed", "graph_type": graph_type, "client_id": cid,
                     "distribution_type": cfg["dist"], "traffic_pattern": cfg["pattern"],
                     "sample_size": cfg["n_samples"], "noise_level": cfg["noise"],
                     "incident_prob": cfg.get("incident_prob", 0),
                     "mse": pm["mse"], "rmse": pm["rmse"], "mae": pm["mae"],
                     "improvement_over_fedavg_rmse": round(imp_fedavg, 2),
                     "improvement_over_independent_rmse": round(imp_ind, 2)})

    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "gcn_enhanced_client_metrics.csv")
    print("\n[client_metrics] Per-client results:")
    print(df.to_string(index=False))

    # 图 1: 每个 client 的 RMSE 对比
    methods_data = [("Independent", "#e74c3c", ind_results),
                    ("GCN-FedAvg", "#3498db", fed_results),
                    ("GCN-Proposed", "#2ecc71", prop_results)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rmse_plot_df = df[["method", "client_id", "rmse"]].copy()
    rmse_plot_df["client"] = rmse_plot_df["client_id"].map(lambda cid: f"Client {cid}")
    sns.barplot(data=rmse_plot_df, x="client", y="rmse", hue="method", hue_order=["Independent", "GCN-FedAvg", "GCN-Proposed"], palette=METHOD_PALETTE, ax=ax, errorbar=None)
    ax.set_ylabel("RMSE (lower is better)"); ax.set_xlabel("Client")
    ax.set_title("Per-client RMSE comparison")
    ax.legend(title=None, fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_client_rmse_comparison.png")

    # 图 2: Proposed 改善率
    prop_df = df[df["method"] == "GCN-Proposed"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    imp_df = pd.DataFrame(
        {
            "client": [f"Client {cid}" for cid in prop_df["client_id"]],
            "vs GCN-FedAvg": prop_df["improvement_over_fedavg_rmse"].values,
            "vs Independent": prop_df["improvement_over_independent_rmse"].values,
        }
    ).melt(id_vars="client", var_name="baseline", value_name="improvement")
    sns.barplot(data=imp_df, x="client", y="improvement", hue="baseline", palette={"vs GCN-FedAvg": METHOD_PALETTE["GCN-FedAvg"], "vs Independent": METHOD_PALETTE["GCN-Proposed"]}, ax=ax, errorbar=None)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("RMSE Improvement (%)")
    ax.set_xlabel("Client")
    ax.set_title("RMSE improvement of GCN-Proposed by client")
    ax.legend(title=None)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_client_improvement.png")
    print("[client_metrics] Done.\n")


def run_peak_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[peak] GCN Peak / Off-peak / Incident Period Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    graph_type = "fixed_adjacency"
    ensure_output_dir(output_dir)
    seed = 42

    set_global_seed(seed)
    client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)

    # Independent（使用同一批数据）
    print("[Independent]")
    ind_results = run_independent_training(cfgs, graph_type, seed=seed,
                                            total_epochs=10, lr=0.01)

    # GCN-FedAvg
    print("[GCN-FedAvg]")
    fed_results, _ = run_federated_training(cfgs, graph_type, "fedavg", seed=seed,
                                              comm_rounds=5, local_epochs=2)

    # GCN-Proposed
    print("[GCN-Proposed]")
    prop_results, _ = run_federated_training(cfgs, graph_type, "proposed", seed=seed,
                                               comm_rounds=5, local_epochs=2)

    # Re-train per client to get per-sample preds for peak analysis
    all_rows = []
    for cid in range(num_clients):
        d = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)[cid]
        meta = d["meta_test"]
        hours = meta["target_hour"]
        inc_flags = meta["target_incident_flag"]
        sample_periods = [classify_period_local(h, f) for h, f in zip(hours, inc_flags)]
        n_test = len(sample_periods)

        # Re-train per client to get per-sample preds
        adj = _get_adj_matrix(None, graph_type, NUM_NODES)
        criterion = nn.MSELoss()
        tr_sizes = [build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)[i]["train_size"]
                    for i in range(num_clients)]

        for method_label, agg_method in [("Independent", None), ("GCN-FedAvg", "fedavg"),
                                          ("GCN-Proposed", "proposed")]:
            set_global_seed(seed)
            # Build all clients
            all_cds = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
            if method_label == "Independent":
                model = GCNEnhancedModel(k=NUM_NODES, t=SEQ_LEN, hidden_dim=64,
                                          num_heads=4, fixed_adj=adj).to(DEVICE)
                opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                for _ in range(10):
                    model.train()
                    for x, y in all_cds[cid]["train_loader"]:
                        x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
                        opt.zero_grad()
                        pred, _ = model(x)
                        loss = criterion(pred.view(-1), y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                model.eval()
                p, t = [], []
                for x, y in all_cds[cid]["test_loader"]:
                    x = x.to(DEVICE).float()
                    po, _ = model(x)
                    p.append(po.detach().view(-1).cpu().numpy()); t.append(y.cpu().numpy())
                p = np.concatenate(p); t = np.concatenate(t)
                preds_raw = p * all_cds[cid]["y_std"] + all_cds[cid]["y_mean"]
                truths_raw = t * all_cds[cid]["y_std"] + all_cds[cid]["y_mean"]
            else:
                model_func = lambda: GCNEnhancedModel(k=NUM_NODES, t=SEQ_LEN,
                                                       hidden_dim=64, num_heads=4, fixed_adj=adj)
                server = AggregationServer(model_func(), num_clients, agg_method=agg_method)
                server.set_client_data_sizes(tr_sizes)
                clients = []
                for cid2 in range(num_clients):
                    cd2 = all_cds[cid2]
                    clients.append(FederatedClient(cd2["cid"], model_func(),
                                                    cd2["train_loader"], cd2["val_loader"],
                                                    cd2["test_loader"], criterion))
                for _ in range(5):
                    cw_l, cl_l = [], []
                    for cl in clients:
                        cll, cw = cl.train_local(epochs=2, global_model=server.global_model)
                        cw_l.append(cw); cl_l.append(float(cll))
                    server.aggregate(cw_l, cl_l)
                clients[cid].model.load_state_dict(server.global_model.state_dict())
                preds_raw, truths_raw = clients[cid].test_predictions_raw(
                    all_cds[cid]["y_mean"], all_cds[cid]["y_std"])

            # Accumulate per-sample errors
            n_use = min(len(preds_raw), n_test)
            for i in range(n_use):
                period = sample_periods[i]
                err = preds_raw[i] - truths_raw[i]
                all_rows.append({"method": method_label, "graph_type": graph_type,
                                 "client_id": cid, "period": period,
                                 "mse": float(err ** 2), "rmse": float(abs(err)),
                                 "mae": float(abs(err)), "num_samples": 1})

    df = pd.DataFrame(all_rows)
    # Aggregate
    metrics_rows = []
    for (method, cid, period), grp in df.groupby(["method", "client_id", "period"]):
        mse_val = float(np.mean([r for r in grp["mse"]]))
        metrics_rows.append({"method": method, "graph_type": graph_type, "client_id": cid,
                             "period": period, "mse": mse_val,
                             "rmse": float(np.sqrt(mse_val)),
                             "mae": float(np.mean([r for r in grp["mae"]])),
                             "num_samples": len(grp)})
    df_metrics = pd.DataFrame(metrics_rows)
    save_dataframe(df_metrics, output_dir, "gcn_enhanced_peak_offpeak_metrics.csv")

    agg_sum = df_metrics.groupby(["method", "period"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        total_samples=("num_samples", "sum")).reset_index()
    save_dataframe(agg_sum, output_dir, "gcn_enhanced_peak_offpeak_summary.csv")
    print("\n[peak] Summary:\n", agg_sum.to_string(index=False))

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    methods = ["Independent", "GCN-FedAvg", "GCN-Proposed"]
    period_order = ["morning_peak", "evening_peak", "off_peak", "incident_period"]
    agg_sum["period"] = pd.Categorical(agg_sum["period"], categories=period_order, ordered=True)
    for idx, metric in enumerate(["rmse", "mae"]):
        sns.barplot(data=agg_sum.sort_values("period"), x="period", y=f"{metric}_mean", hue="method", hue_order=methods, palette=METHOD_PALETTE, ax=axes[idx], errorbar=None)
        axes[idx].set_title(f"{metric.upper()} across traffic periods")
        axes[idx].set_xlabel("Traffic period")
        axes[idx].set_ylabel(f"{metric.upper()} (lower is better)")
        axes[idx].tick_params(axis="x", rotation=20)
        if idx == 0:
            axes[idx].legend(title=None, fontsize=8)
        else:
            axes[idx].get_legend().remove()
    fig.suptitle("Peak, off-peak, and incident comparison", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_enhanced_peak_offpeak_comparison.png")
    print("[peak] Done.\n")


# ══════════════════════════════════════════════════════════════
# 工作流调度
# ══════════════════════════════════════════════════════════════

WORKFLOW_MAP = {
    "all": ["data_viz", "main", "fixed_vs_dynamic", "aggregation", "lambda",
            "client_scale", "noniid", "convergence", "client_metrics",
            "peak", "congestion_delay"],
    "data_viz":         ["data_viz"],
    "main":             ["main"],
    "fixed_vs_dynamic": ["fixed_vs_dynamic"],
    "aggregation":      ["aggregation"],
    "lambda":           ["lambda"],
    "client_scale":     ["client_scale"],
    "noniid":           ["noniid"],
    "convergence":      ["convergence"],
    "client_metrics":   ["client_metrics"],
    "peak":             ["peak"],
    "congestion_delay": ["congestion_delay"],
}

WORKFLOW_FUNCTIONS = {
    "data_viz":         run_data_visualization_enhanced,
    "main":             run_main_experiment,
    "fixed_vs_dynamic": run_fixed_vs_dynamic_experiment,
    "aggregation":      run_aggregation_experiment,
    "lambda":           run_lambda_experiment,
    "client_scale":     run_client_scale_experiment,
    "noniid":           run_noniid_experiment,
    "convergence":      run_convergence_experiment,
    "client_metrics":   run_client_metrics_experiment,
    "peak":             run_peak_experiment,
    "congestion_delay": run_congestion_delay_experiment,
}


def run_project(workflow: str, output_dir: Path) -> None:
    configure_academic_plot_style()
    ensure_output_dir(output_dir)
    export_figure_index(output_dir)
    print(f"[gcn_fed_enhanced] workflow={workflow}, device={DEVICE}")
    print(f"[gcn_fed_enhanced] output={output_dir}")

    steps = WORKFLOW_MAP[workflow]
    for step in steps:
        fn = WORKFLOW_FUNCTIONS[step]
        print(f"\n>>> Running step: {step}")
        fn(output_dir)

    print(f"\n[gcn_fed_enhanced] All done. Results in: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="GCN Enhanced Federated Simulation")
    parser.add_argument("--workflow", choices=list(WORKFLOW_MAP.keys()),
                        default="all", help="Workflow to execute (default: all).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    PROJECT_ROOT = SCRIPT_DIR.parent
    RESULTS_ROOT = PROJECT_ROOT / "results"
    SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
    output_dir = SIMULATION_RESULTS_ROOT / "gcn_fed_enhanced"
    _export_communication_cost(output_dir)
    run_project(args.workflow, output_dir)


def _export_communication_cost(output_dir: Path) -> None:
    """自动输出 GCN 模型通信开销估计表。"""
    _, adj, _ = build_fixed_adjacency(NUM_NODES)
    m = GCNEnhancedModel(k=NUM_NODES, t=SEQ_LEN, hidden_dim=64, fixed_adj=adj)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    size_mb = n_params * 4 / (1024 ** 2)
    rows = []
    for nc in [3, 5, 8, 10]:
        for cr in [5, 10, 15]:
            rows.append({"model_type": "GCN", "num_clients": nc,
                         "communication_rounds": cr,
                         "num_parameters": n_params,
                         "parameter_size_mb": round(size_mb, 4),
                         "total_communication_mb": round(2 * nc * size_mb * cr, 2)})
    df = pd.DataFrame(rows)
    path = ensure_output_dir(output_dir) / "gcn_enhanced_communication_cost.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[gcn_fed_enhanced] Communication cost saved: {path}")


if __name__ == "__main__":
    main()
