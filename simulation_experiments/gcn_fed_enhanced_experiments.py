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


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    return path


def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path


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
# Workflow: data_viz — 增强数据集可视化（与 CNN 一致）
# ══════════════════════════════════════════════════════════════

def run_data_visualization_enhanced(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[data_viz] Enhanced Non-IID Dataset Visualizations (GCN)")
    print("=" * 60)

    raw_signals, raw_masks, cfgs = get_enhanced_raw_data()
    num_clients = len(cfgs)
    ensure_output_dir(output_dir)
    colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

    # ── 1. client 平均时间序列 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    for cid in range(num_clients):
        ts = raw_signals[cid].mean(axis=1)[:200]
        ax.plot(ts, color=colors[cid], linewidth=1.5,
                label=f"C{cid} ({cfgs[cid]['pattern']})")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Avg Traffic Flow")
    ax.set_title("Enhanced Dataset (GCN): Per-Client Average Traffic Flow Time Series")
    ax.legend(fontsize=7, loc="upper right")
    save_figure(fig, output_dir, "enhanced_dataset_client_timeseries.png")

    # ── 2. 分布对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    box_data = [raw_signals[cid].ravel() for cid in range(num_clients)]
    bp = axes[0].boxplot(box_data, tick_labels=[f"C{cid}\n{cfgs[cid]['dist']}"
                          for cid in range(num_clients)], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    axes[0].set_title("Traffic Flow Distribution by Client (Boxplot)")
    axes[0].set_ylabel("Traffic Flow")
    for cid in range(num_clients):
        axes[1].hist(raw_signals[cid].ravel(), bins=40, alpha=0.4,
                     color=colors[cid], label=f"C{cid} ({cfgs[cid]['dist']})")
    axes[1].set_title("Traffic Flow Distribution (Histogram)")
    axes[1].set_xlabel("Traffic Flow"); axes[1].legend(fontsize=7)
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_dataset_distribution_comparison.png")

    # ── 3. client 配置概览 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cids = np.arange(num_clients)
    axes[0, 0].bar(cids, [c["n_samples"] for c in cfgs], color=colors)
    axes[0, 0].set_title("Sample Size per Client"); axes[0, 0].set_xticks(cids)
    axes[0, 1].bar(cids, [c["noise"] for c in cfgs], color=colors)
    axes[0, 1].set_title("Noise Level per Client"); axes[0, 1].set_xticks(cids)
    axes[1, 0].bar(cids, [c["base"] for c in cfgs], color=colors)
    axes[1, 0].set_title("Base Flow per Client"); axes[1, 0].set_xticks(cids)
    axes[1, 1].bar(cids, [c.get("incident_prob", 0) for c in cfgs], color=colors)
    axes[1, 1].set_title("Incident Probability per Client"); axes[1, 1].set_xticks(cids)
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_dataset_client_config.png")

    # ── 4. 高峰模式 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    for cid in range(num_clients):
        ts = raw_signals[cid].mean(axis=1)
        ax.plot(ts[:24], "o-", color=colors[cid], linewidth=2, markersize=4,
                label=f"C{cid}: {cfgs[cid]['pattern']}")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Traffic Flow")
    ax.set_title("Enhanced Dataset (GCN): 24-Hour Peak Patterns")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "enhanced_dataset_peak_pattern.png")

    # ── 5. incident 示例 ──
    incident_cid = 4
    fig, ax = plt.subplots(figsize=(14, 5))
    ts = raw_signals[incident_cid].mean(axis=1)
    mask = raw_masks[incident_cid]
    t = np.arange(len(ts))
    ax.plot(t, ts, color="#3498db", linewidth=1, label="Traffic Flow")
    in_inc = False; start = 0
    for i in range(len(mask)):
        if mask[i] and not in_inc: start = i; in_inc = True
        elif not mask[i] and in_inc:
            ax.axvspan(start, i, alpha=0.3, color="#e74c3c"); in_inc = False
    if in_inc: ax.axvspan(start, len(mask) - 1, alpha=0.3, color="#e74c3c")
    ax.set_xlabel("Time Step"); ax.set_ylabel("Avg Traffic Flow")
    ax.set_title(f"Enhanced Dataset (GCN): Client {incident_cid} ({cfgs[incident_cid]['pattern']}) with Incidents")
    ax.legend()
    save_figure(fig, output_dir, "enhanced_dataset_incident_example.png")

    # ── 6. client 间相关系数矩阵 ──
    min_len = min(len(s) for s in raw_signals)
    client_ts = np.array([raw_signals[cid].mean(axis=1)[:min_len] for cid in range(num_clients)])
    corr_client = np.corrcoef(client_ts)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_client, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    for i in range(num_clients):
        for j in range(num_clients):
            ax.text(j, i, f"{corr_client[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Enhanced Dataset (GCN): Inter-Client Correlation Matrix"); plt.colorbar(im, ax=ax)
    save_figure(fig, output_dir, "enhanced_dataset_client_correlation_matrix.png")

    # ── 7. 节点间相关系数矩阵 ──
    rep_cid = 0
    node_corr = np.corrcoef(raw_signals[rep_cid].T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(node_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            ax.text(j, i, f"{node_corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Enhanced Dataset (GCN): Node Correlation Matrix (Client {rep_cid})"); plt.colorbar(im, ax=ax)
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
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(A_fixed_raw, cmap="Blues", aspect="equal", vmin=0, vmax=1)
    ax.set_title("Enhanced GCN: Fixed Adjacency Matrix (Road Network Topology)")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A_fixed_raw[i, j] > 0:
                ax.text(j, i, f"{A_fixed_raw[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Weight")
    save_figure(fig, output_dir, "enhanced_gcn_fixed_adjacency_matrix.png")

    # ── 2. 高峰期动态邻接矩阵（早高峰） ──
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(A_dyn_morning_raw, cmap="Oranges", aspect="equal", vmin=0, vmax=1)
    ax.set_title("Enhanced GCN: Dynamic Adjacency (Morning Peak, 07:00-09:00)")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A_dyn_morning_raw[i, j] > 0:
                ax.text(j, i, f"{A_dyn_morning_raw[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="Correlation")
    save_figure(fig, output_dir, "enhanced_gcn_dynamic_adjacency_peak.png")

    # ── 3. 平峰期动态邻接矩阵 ──
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(A_dyn_offpeak_raw, cmap="Greens", aspect="equal", vmin=0, vmax=1)
    ax.set_title("Enhanced GCN: Dynamic Adjacency (Off-Peak)")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A_dyn_offpeak_raw[i, j] > 0:
                ax.text(j, i, f"{A_dyn_offpeak_raw[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="Correlation")
    save_figure(fig, output_dir, "enhanced_gcn_dynamic_adjacency_offpeak.png")

    # ── 4. 固定 vs 动态对比图 ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    adj_list = [
        (A_fixed_raw, "Fixed Topology", "Blues"),
        (A_dyn_morning_raw, "Dynamic Morning Peak", "Oranges"),
        (A_dyn_evening_raw, "Dynamic Evening Peak", "Reds"),
        (A_dyn_offpeak_raw, "Dynamic Off-Peak", "Greens"),
    ]
    for idx, (mat, title, cmap) in enumerate(adj_list):
        ax = axes[idx]
        im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_fixed_dynamic_adjacency_comparison.png")

    # ── 5. 功能相似矩阵 ──
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(A_func_raw, cmap="Purples", aspect="equal", vmin=0, vmax=1)
    ax.set_title("Enhanced GCN: Functional Similarity Matrix (Pearson Correlation)")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A_func_raw[i, j] > 0.3:
                ax.text(j, i, f"{A_func_raw[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="|Correlation|")
    save_figure(fig, output_dir, "enhanced_gcn_functional_similarity_matrix.png")

    # ── 6. 拥堵传播延迟矩阵 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    im0 = axes[0].imshow(A_delay_raw, cmap="YlOrRd", aspect="equal", vmin=0, vmax=5)
    axes[0].set_title("Enhanced GCN: Congestion Delay Matrix (lag steps)")
    axes[0].set_xlabel("Target Node ID"); axes[0].set_ylabel("Source Node ID")
    plt.colorbar(im0, ax=axes[0], label="Delay (steps)")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A_delay_raw[i, j] < float(5):
                axes[0].text(j, i, f"{A_delay_raw[i, j]:.0f}", ha="center", va="center", fontsize=7)

    im1 = axes[1].imshow(A_delay_raw, cmap="YlOrRd", aspect="equal", vmin=0, vmax=5)
    axes[1].set_title("Enhanced GCN: Congestion Delay (annotated)")
    axes[1].set_xlabel("Target Node ID"); axes[1].set_ylabel("Source Node ID")
    plt.colorbar(im1, ax=axes[1], label="Delay (steps)")
    # 标注 source->target 的方向
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and A_delay_raw[i, j] < 3:
                axes[1].annotate("", xy=(j, i), xytext=(j - 0.15, i - 0.15),
                                 arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_congestion_delay_matrix.png")

    # ── 7. 高峰/平峰图结构变化对比 ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    period_mats = [
        (A_dyn_offpeak_raw, "Off-Peak", "Greens"),
        (A_dyn_morning_raw, "Morning Peak (07-09)", "Oranges"),
        (A_dyn_evening_raw, "Evening Peak (17-19)", "Reds"),
    ]
    for idx, (mat, title, cmap) in enumerate(period_mats):
        ax = axes[idx]
        im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Enhanced GCN: Graph Structure Changes Across Traffic Periods", fontsize=13)
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_gcn_peak_graph_change.png")

    # ── 8. 图结构汇总 CSV ──
    all_meta = [meta_fixed, meta_morning, meta_evening, meta_offpeak, meta_func, meta_delay]
    df_meta = pd.DataFrame(all_meta)
    save_dataframe(df_meta, output_dir, "enhanced_gcn_graph_summary.csv")
    print("[fixed_vs_dynamic] Graph summary:\n", df_meta.to_string(index=False))
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
    print("[congestion_delay] Done.\n")


# ══════════════════════════════════════════════════════════════
# 占位 workflow（后续批次补充）
# ══════════════════════════════════════════════════════════════

def run_main_experiment(output_dir: Path) -> None:
    print("[main] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_aggregation_experiment(output_dir: Path) -> None:
    print("[aggregation] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_lambda_experiment(output_dir: Path) -> None:
    print("[lambda] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_client_scale_experiment(output_dir: Path) -> None:
    print("[client_scale] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_noniid_experiment(output_dir: Path) -> None:
    print("[noniid] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_convergence_experiment(output_dir: Path) -> None:
    print("[convergence] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_client_metrics_experiment(output_dir: Path) -> None:
    print("[client_metrics] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


def run_peak_experiment(output_dir: Path) -> None:
    print("[peak] Placeholder — to be implemented in next batch.")
    ensure_output_dir(output_dir)


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
    ensure_output_dir(output_dir)
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
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
