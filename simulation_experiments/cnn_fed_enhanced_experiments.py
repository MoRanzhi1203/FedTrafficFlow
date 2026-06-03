# -*- coding: utf-8 -*-
"""
CNN/CCN 一审增强仿真实验组。

本文件回应一审意见和会议要求中的各项增强需求：
1. data_viz:   增强 Non-IID 数据集可视化（8 张图 + 1 CSV）；
2. main:       Independent / FedAvg / Proposed 主结果（3 seeds）；
3. aggregation: FedAvg / Loss-weighted / Data-loss / Similarity / Proposed 消融；
4. lambda:     Data-loss weighted 中 λ 参数敏感性；
5. convergence / client_scale / noniid / client_metrics / peak / feature_ablation：
   后续批次补充（当前保留接口存根）。

所有新增函数均写在本文件内部，无外部 utils/config/dataset 文件。
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
# 尝试使用系统中支持中文的字体
_cjk_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((fn for fn in _cjk_candidates if fn in _available), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

plt.ioff()

# ══════════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 增强数据集超参数（与 gcn_fed_enhanced_experiments.py 共享）
NUM_NODES = 8
SEQ_LEN = 12
PRED_LEN = 1
BATCH_SIZE = 32
HIDDEN_DIM = 64
COMM_ROUNDS = 5
LOCAL_EPOCHS = 2
LR = 0.001
SEEDS = [42, 2024, 2025]

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


def compute_metrics(preds, truths):
    mse = float(np.mean((preds - truths) ** 2))
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(preds - truths)))


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个一维张量的余弦相似度。"""
    a_f = a.view(-1).float(); b_f = b.view(-1).float()
    dot = float(torch.dot(a_f, b_f))
    na = float(torch.norm(a_f)); nb = float(torch.norm(b_f))
    return max(0.0, dot / (na * nb + 1e-12))


# ══════════════════════════════════════════════════════════════
# 增强 Non-IID 数据集生成
# ══════════════════════════════════════════════════════════════

CLIENT_CONFIGS_BASE = [
    {"dist": "normal",          "pattern": "平稳通勤型",   "n_samples": 600, "noise": 2.0,
     "base": 100.0, "morning_mu": 8.0, "evening_mu": 18.0,
     "morning_amp": 30.0, "evening_amp": 25.0, "peak_sigma": 0.8,
     "trend": 0.0, "incident_prob": 0.0},
    {"dist": "student-t",       "pattern": "波动型",       "n_samples": 500, "noise": 5.0,
     "base": 80.0, "morning_mu": 7.5, "evening_mu": 17.5,
     "morning_amp": 35.0, "evening_amp": 30.0, "peak_sigma": 1.0,
     "trend": 0.005, "incident_prob": 0.0},
    {"dist": "chi-square",      "pattern": "偏态高流量型", "n_samples": 700, "noise": 8.0,
     "base": 120.0, "morning_mu": 8.5, "evening_mu": 18.5,
     "morning_amp": 25.0, "evening_amp": 20.0, "peak_sigma": 1.2,
     "trend": -0.003, "incident_prob": 0.0},
    {"dist": "gaussian_mixture","pattern": "双峰型",       "n_samples": 550, "noise": 4.0,
     "base": 90.0, "morning_mu": 7.0, "evening_mu": 19.0,
     "morning_amp": 40.0, "evening_amp": 35.0, "peak_sigma": 0.7,
     "trend": 0.002, "incident_prob": 0.0},
    {"dist": "log_normal",      "pattern": "突发拥堵型",   "n_samples": 450, "noise": 6.0,
     "base": 70.0, "morning_mu": 8.2, "evening_mu": 17.8,
     "morning_amp": 28.0, "evening_amp": 22.0, "peak_sigma": 0.9,
     "trend": 0.001, "incident_prob": 0.05},
]


def build_noniid_client_configs(num_clients, noniid_level="medium"):
    """根据 Non-IID 强度构建客户端配置列表。

    low:    客户端差异较小，所有客户端使用 normal 分布，仅噪声和高峰参数略有差异。
    medium: 标准增强配置，5 种不同分布/模式/样本量/噪声。
    high:   更强异质性，更大噪声、更不平衡样本量、更高 incident 概率、更多重尾分布。
    """
    if noniid_level == "low":
        templates = [
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 600, "noise": 2.0,
             "base": 100.0, "morning_mu": 8.0, "evening_mu": 18.0,
             "morning_amp": 30.0, "evening_amp": 25.0, "peak_sigma": 0.9,
             "trend": 0.0, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 580, "noise": 2.5,
             "base": 95.0, "morning_mu": 7.8, "evening_mu": 17.8,
             "morning_amp": 32.0, "evening_amp": 27.0, "peak_sigma": 0.85,
             "trend": 0.001, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 620, "noise": 3.0,
             "base": 105.0, "morning_mu": 8.2, "evening_mu": 18.2,
             "morning_amp": 28.0, "evening_amp": 23.0, "peak_sigma": 0.75,
             "trend": -0.001, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 590, "noise": 2.8,
             "base": 98.0, "morning_mu": 7.9, "evening_mu": 17.9,
             "morning_amp": 31.0, "evening_amp": 26.0, "peak_sigma": 0.88,
             "trend": 0.0005, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 610, "noise": 2.2,
             "base": 102.0, "morning_mu": 8.1, "evening_mu": 18.1,
             "morning_amp": 29.0, "evening_amp": 24.0, "peak_sigma": 0.82,
             "trend": -0.0005, "incident_prob": 0.0},
        ]
    elif noniid_level == "high":
        templates = [
            {"dist": "student-t",      "pattern": "波动型",     "n_samples": 400, "noise": 8.0,
             "base": 75.0, "morning_mu": 6.5, "evening_mu": 16.0,
             "morning_amp": 42.0, "evening_amp": 38.0, "peak_sigma": 1.3,
             "trend": 0.008, "incident_prob": 0.04},
            {"dist": "chi-square",     "pattern": "偏态高流量型", "n_samples": 750, "noise": 12.0,
             "base": 135.0, "morning_mu": 9.5, "evening_mu": 19.5,
             "morning_amp": 20.0, "evening_amp": 16.0, "peak_sigma": 1.5,
             "trend": -0.006, "incident_prob": 0.1},
            {"dist": "gaussian_mixture", "pattern": "双峰型",    "n_samples": 380, "noise": 6.0,
             "base": 85.0, "morning_mu": 6.0, "evening_mu": 20.0,
             "morning_amp": 48.0, "evening_amp": 42.0, "peak_sigma": 0.6,
             "trend": 0.004, "incident_prob": 0.03},
            {"dist": "log_normal",     "pattern": "突发拥堵型",  "n_samples": 300, "noise": 9.0,
             "base": 65.0, "morning_mu": 8.8, "evening_mu": 18.8,
             "morning_amp": 25.0, "evening_amp": 20.0, "peak_sigma": 1.1,
             "trend": 0.002, "incident_prob": 0.12},
            {"dist": "chi-square",     "pattern": "偏态高流量型", "n_samples": 500, "noise": 10.0,
             "base": 110.0, "morning_mu": 8.0, "evening_mu": 17.0,
             "morning_amp": 35.0, "evening_amp": 30.0, "peak_sigma": 1.4,
             "trend": -0.004, "incident_prob": 0.06},
        ]
    else:  # medium
        templates = list(CLIENT_CONFIGS_BASE)

    configs = []
    for cid in range(num_clients):
        tpl = templates[cid % len(templates)].copy()
        # 对重复使用的模板做微小扰动以增加多样性
        if cid >= len(templates):
            jitter = 1.0 + 0.02 * (cid - len(templates)) * ((-1) ** cid)
            tpl["noise"] = tpl["noise"] * jitter
            tpl["n_samples"] = int(tpl["n_samples"] * jitter)
            tpl["base"] = tpl["base"] * jitter
        configs.append(tpl)
    return configs


def sample_distribution_noise(n_timesteps, n_nodes, dist_type, noise_level, seed):
    """按指定分布类型生成噪声。"""
    rng = np.random.RandomState(seed)
    if dist_type == "normal":
        return rng.randn(n_timesteps, n_nodes) * noise_level
    elif dist_type == "student-t":
        return rng.standard_t(df=4, size=(n_timesteps, n_nodes)) * noise_level * 0.7
    elif dist_type == "chi-square":
        return (rng.chisquare(df=3, size=(n_timesteps, n_nodes)) - 3) * noise_level * 0.5
    elif dist_type == "gaussian_mixture":
        mask = rng.rand(n_timesteps, n_nodes) < 0.5
        n1 = rng.randn(n_timesteps, n_nodes) * noise_level * 0.6
        n2 = rng.randn(n_timesteps, n_nodes) * noise_level * 1.5 + 0.5
        return np.where(mask, n1, n2)
    elif dist_type == "log_normal":
        return (rng.lognormal(mean=0, sigma=0.5, size=(n_timesteps, n_nodes)) - 1.5) * noise_level
    return rng.randn(n_timesteps, n_nodes) * noise_level


def generate_traffic_flow(cfg, n_timesteps, n_nodes, seed):
    """生成复杂交通流时间序列。

    返回: data (n_timesteps, n_nodes), incident_mask (n_timesteps,) 或 None
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_timesteps)
    hours = t * 24.0 / n_timesteps

    base_flow = cfg["base"] + cfg["trend"] * t

    morning_peak = cfg["morning_amp"] * np.exp(
        -((hours - cfg["morning_mu"]) ** 2) / (2 * cfg["peak_sigma"] ** 2))
    evening_peak = cfg["evening_amp"] * np.exp(
        -((hours - cfg["evening_mu"]) ** 2) / (2 * cfg["peak_sigma"] ** 2))

    daily_period = (5.0 * np.sin(2 * np.pi * t / (n_timesteps / 2)) +
                    3.0 * np.cos(2 * np.pi * t / (n_timesteps / 4)))

    regional_bias = rng.randn(n_nodes) * 3.0

    incident = np.zeros(n_timesteps)
    incident_mask = np.zeros(n_timesteps, dtype=bool)
    if cfg.get("incident_prob", 0) > 0:
        for i in range(n_timesteps):
            if rng.rand() < cfg["incident_prob"]:
                duration = rng.randint(5, 20)
                end = min(i + duration, n_timesteps)
                incident[i:end] -= rng.uniform(20, 50)
                incident_mask[i:end] = True

    base_signal = base_flow + morning_peak + evening_peak + daily_period + incident

    data = np.zeros((n_timesteps, n_nodes), dtype=np.float32)
    for node in range(n_nodes):
        node_scale = 0.8 + 0.4 * (node / n_nodes)
        data[:, node] = base_signal * node_scale + regional_bias[node]

    noise = sample_distribution_noise(n_timesteps, n_nodes,
                                       cfg["dist"], cfg["noise"], seed + 1000)
    data += noise

    return data.astype(np.float32), incident_mask


def build_sequences(data, seq_len, pred_len, incident_mask=None):
    """滑动窗口构建监督学习样本，同时返回 target_meta。

    返回:
        X: (N, seq_len, n_nodes)
        y: (N,)
        target_meta: dict with 'target_time_index', 'target_hour', 'target_incident_flag'
    """
    X, y = [], []
    target_time_index = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        target_idx = i + seq_len + pred_len - 1
        y.append(data[target_idx, 0])
        target_time_index.append(target_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    t_idx = np.array(target_time_index, dtype=int)

    # hour of day
    n_ts = len(data)
    hours = (t_idx * 24.0 / n_ts) % 24
    target_hour = hours.astype(np.float32)

    # incident flag
    if incident_mask is not None:
        target_incident_flag = incident_mask[t_idx].astype(bool)
    else:
        target_incident_flag = np.zeros(len(y), dtype=bool)

    meta = {
        "target_time_index": t_idx,
        "target_hour": target_hour,
        "target_incident_flag": target_incident_flag,
    }
    return X, y, meta


def classify_period(hour, incident_flag):
    """根据真实 hour 和 incident flag 对样本分类。"""
    if incident_flag:
        return "incident_period"
    if 7 <= hour < 9:
        return "morning_peak"
    if 17 <= hour < 19:
        return "evening_peak"
    return "off_peak"


class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_train_val_test(X, y, train_r=0.70, val_r=0.10):
    n = len(X)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return (X[:n_train], y[:n_train],
            X[n_train:n_train + n_val], y[n_train:n_train + n_val],
            X[n_train + n_val:], y[n_train + n_val:])


def split_with_meta(X, y, meta, train_r=0.70, val_r=0.10):
    """三路划分 X, y, meta 字典（每个 value 是等长数组）。"""
    n = len(X)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, n)

    def _slice(d):
        return {k: v[train_idx] for k, v in d.items()}, \
               {k: v[val_idx] for k, v in d.items()}, \
               {k: v[test_idx] for k, v in d.items()}

    meta_train, meta_val, meta_test = _slice(meta)
    return (X[train_idx], y[train_idx], meta_train,
            X[val_idx], y[val_idx], meta_val,
            X[test_idx], y[test_idx], meta_test)


def build_client_data_shuffled(client_configs, num_nodes, seq_len, pred_len, seed):
    """同 build_client_data 但使用随机打乱划分以保证各 period 均匀分布。"""
    buffer = seq_len + pred_len + 10
    all_data = []
    for cid, cfg in enumerate(client_configs):
        n_timesteps = cfg["n_samples"] + buffer
        data, incident_mask = generate_traffic_flow(cfg, n_timesteps, num_nodes, seed + cid * 100)
        X, y, meta = build_sequences(data, seq_len, pred_len, incident_mask)

        # 随机打乱索引
        rng = np.random.RandomState(seed + cid)
        n = len(X)
        idx = rng.permutation(n)
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        def _idx_slice(d, indices):
            return {k: v[indices] for k, v in d.items()}

        X_train, y_train, meta_train = X[train_idx], y[train_idx], _idx_slice(meta, train_idx)
        X_val, y_val, meta_val = X[val_idx], y[val_idx], _idx_slice(meta, val_idx)
        X_test, y_test, meta_test = X[test_idx], y[test_idx], _idx_slice(meta, test_idx)

        x_mean = X_train.mean(axis=(0, 1), keepdims=True)
        x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8

        X_train_n = (X_train - x_mean) / x_std
        X_val_n = (X_val - x_mean) / x_std
        X_test_n = (X_test - x_mean) / x_std
        y_train_n = (y_train - y_mean) / y_std
        y_val_n = (y_val - y_mean) / y_std
        y_test_n = (y_test - y_mean) / y_std

        train_ds = EnhancedTimeSeriesDataset(X_train_n, y_train_n)
        val_ds = EnhancedTimeSeriesDataset(X_val_n, y_val_n)
        test_ds = EnhancedTimeSeriesDataset(X_test_n, y_test_n)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        all_data.append({
            "cid": cid,
            "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
            "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
            "y_mean": y_mean, "y_std": y_std,
            "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        })
    return all_data


def build_client_data(client_configs, num_nodes, seq_len, pred_len, seed):
    """为所有 client 构建训练/验证/测试数据（顺序划分）及标准化参数。"""
    buffer = seq_len + pred_len + 10
    all_data = []
    for cid, cfg in enumerate(client_configs):
        n_timesteps = cfg["n_samples"] + buffer
        data, incident_mask = generate_traffic_flow(cfg, n_timesteps, num_nodes, seed + cid * 100)
        X, y, meta = build_sequences(data, seq_len, pred_len, incident_mask)
        X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = \
            split_with_meta(X, y, meta)

        x_mean = X_train.mean(axis=(0, 1), keepdims=True)
        x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8

        X_train_n = (X_train - x_mean) / x_std
        X_val_n = (X_val - x_mean) / x_std
        X_test_n = (X_test - x_mean) / x_std
        y_train_n = (y_train - y_mean) / y_std
        y_val_n = (y_val - y_mean) / y_std
        y_test_n = (y_test - y_mean) / y_std

        train_ds = EnhancedTimeSeriesDataset(X_train_n, y_train_n)
        val_ds = EnhancedTimeSeriesDataset(X_val_n, y_val_n)
        test_ds = EnhancedTimeSeriesDataset(X_test_n, y_test_n)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        all_data.append({
            "cid": cid,
            "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
            "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds),
            "y_mean": y_mean, "y_std": y_std,
            "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        })
    return all_data


# ══════════════════════════════════════════════════════════════
# 特征工程：为 feature_ablation 构造不同输入组合
# ══════════════════════════════════════════════════════════════

def build_feature_augmented_data(client_configs, feature_set, num_nodes, seq_len, pred_len, seed):
    """构建指定特征集的 client 数据。

    feature_set 选项:
        flow_only:  K=num_nodes
        flow_time:  K=num_nodes + 2 (sin_hour, cos_hour)
        flow_event: K=num_nodes + 1 (incident_flag)
        flow_region:K=num_nodes + 1 (region_id)
        full:       K=num_nodes + 2 + 1 + 1 = num_nodes + 4
    """
    buffer = seq_len + pred_len + 10
    all_data = []
    for cid, cfg in enumerate(client_configs):
        n_timesteps = cfg["n_samples"] + buffer
        data, incident_mask = generate_traffic_flow(cfg, n_timesteps, num_nodes, seed + cid * 100)
        X_flow, y, meta = build_sequences(data, seq_len, pred_len, incident_mask)

        # 构造额外特征通道（每个通道 shape: (N, seq_len)）
        t_idx_arr = meta["target_time_index"]  # target indices
        seq_t_idx = np.array([np.arange(i, i + seq_len) for i in range(len(X_flow))])  # (N, seq_len)
        n_samples = len(X_flow)

        # 建增强特征 (N, extra_channels, seq_len)
        extra_feats = []
        if feature_set in ("flow_time", "full"):
            hours = (seq_t_idx * 24.0 / n_timesteps) % 24
            sin_h = np.sin(2 * np.pi * hours / 24).astype(np.float32)  # (N, seq_len)
            cos_h = np.cos(2 * np.pi * hours / 24).astype(np.float32)
            extra_feats.append(sin_h)
            extra_feats.append(cos_h)

        if feature_set in ("flow_event", "full"):
            inc_seq = np.zeros((n_samples, seq_len), dtype=np.float32)
            for s in range(n_samples):
                window_slice = slice(s, s + seq_len)
                inc_seq[s] = incident_mask[window_slice].astype(np.float32)
            extra_feats.append(inc_seq)

        if feature_set in ("flow_region", "full"):
            region_id = np.full((n_samples, seq_len), float(cid) / max(len(client_configs) - 1, 1),
                                dtype=np.float32)
            extra_feats.append(region_id)

        if extra_feats:
            extra = np.stack(extra_feats, axis=1)  # (N, C_extra, seq_len)
            extra = extra.transpose(0, 2, 1)        # (N, seq_len, C_extra)
            X_full = np.concatenate([X_flow, extra], axis=2)  # (N, seq_len, K + C_extra)
        else:
            X_full = X_flow

        X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = \
            split_with_meta(X_full, y, meta)

        x_mean = X_train.mean(axis=(0, 1), keepdims=True)
        x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        y_mean = y_train.mean()
        y_std = y_train.std() + 1e-8

        X_train_n = (X_train - x_mean) / x_std
        X_val_n = (X_val - x_mean) / x_std
        X_test_n = (X_test - x_mean) / x_std
        y_train_n = (y_train - y_mean) / y_std
        y_val_n = (y_val - y_mean) / y_std
        y_test_n = (y_test - y_mean) / y_std

        train_ds = EnhancedTimeSeriesDataset(X_train_n, y_train_n)
        val_ds = EnhancedTimeSeriesDataset(X_val_n, y_val_n)
        test_ds = EnhancedTimeSeriesDataset(X_test_n, y_test_n)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 记录用于 infer 的原始 test 数据和标准化参数
        all_data.append({
            "cid": cid,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "y_mean": y_mean,
            "y_std": y_std,
            "X_test": X_test,
            "y_test": y_test,
            "meta_test": meta_test,
            "k_dim": X_full.shape[2],  # 经过 permute(0,2,1) 后成为 channel 维
        })
    return all_data


# ══════════════════════════════════════════════════════════════
# 模型定义（LayerNorm 替代 BatchNorm 以适配 Non-IID）
# ══════════════════════════════════════════════════════════════

class AdaptiveSwish(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class CNNEnhancedModel(nn.Module):
    """CNN-BiLSTM-Attention 模型（LayerNorm 版，适配 Non-IID）。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim), AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim), AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=k, hidden_size=hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.LayerNorm(32),
            AdaptiveSwish(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.lstm_proj(x_lstm.mean(dim=1))
        feat_seq = torch.stack([x_cnn, x_lstm], dim=1)
        attn_out, attn_w = self.multihead_attn(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        x_fused = attn_out.mean(dim=1)
        return self.regression_head(x_fused), attn_w


# ══════════════════════════════════════════════════════════════
# 联邦客户端与服务端
# ══════════════════════════════════════════════════════════════

class FederatedClient:
    """联邦客户端——train_loader / val_loader / test_loader 三段划分。"""

    def __init__(self, client_id, model, train_loader, val_loader, test_loader,
                 criterion, lr=1e-3):
        self.client_id = client_id
        self.model = model.to(DEVICE).float()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred.view(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        avg = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(avg)
        return avg

    @torch.no_grad()
    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            total_loss += self.criterion(pred.view(-1), y).item() * x.shape[0]
        avg = total_loss / len(loader.dataset)
        self.val_losses.append(avg)
        return avg

    @torch.no_grad()
    def validate_metrics(self, y_mean, y_std, loader=None):
        """在指定 loader 上计算真实尺度 MSE/RMSE/MAE。"""
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        preds, truths = [], []
        for x, y in loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            preds.append(pred.view(-1).cpu().numpy())
            truths.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        preds_raw = preds * y_std + y_mean
        truths_raw = truths * y_std + y_mean
        return compute_metrics(preds_raw, truths_raw)

    def train_local(self, epochs=3, global_model=None, verbose=False, prefix="Local"):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        epoch_losses = []
        for ep in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_losses.append((train_loss, val_loss))
            if verbose:
                print(f"  {prefix} Client{self.client_id} ep{ep+1}/{epochs} "
                      f"train={train_loss:.6f} val={val_loss:.6f}")
        return (float(self.train_losses[-1]),
                copy.deepcopy(self.model.state_dict()),
                epoch_losses)

    @torch.no_grad()
    def test_metrics(self):
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            preds.append(pred.view(-1))
            truths.append(y)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        truths = torch.cat(truths, dim=0).cpu().numpy()
        mse, rmse, mae = compute_metrics(preds, truths)
        return {"mse": mse, "rmse": rmse, "mae": mae, "preds": preds, "truths": truths}

    @torch.no_grad()
    def test_loss(self):
        """返回 test_loader 上的 MSE loss。"""
        self.model.eval()
        total_loss = 0.0
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            total_loss += self.criterion(pred.view(-1), y).item() * x.shape[0]
        return total_loss / len(self.test_loader.dataset)


class FedAvgServer:
    """标准样本量加权 FedAvg。"""

    def __init__(self, model, num_clients: int):
        self.global_model = model.to(DEVICE).float()
        self.num_clients = num_clients
        self.round_losses = []
        self.round_val_rmse = []
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = sizes

    def aggregate(self, client_weights, client_losses):
        total_n = float(sum(self.client_data_sizes))
        w = np.array(self.client_data_sizes) / total_n
        global_dict = self.global_model.state_dict()
        new_dict = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_dict.items()}
        for key in new_dict:
            for idx in range(self.num_clients):
                cw = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                new_dict[key] += cw * torch.tensor(float(w[idx]), device=DEVICE, dtype=torch.float32)
        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(client_losses)))
        return self.global_model.state_dict()


class AggregationServer(FedAvgServer):
    """可切换聚合策略的服务端。"""

    def __init__(self, model, num_clients: int, agg_method="fedavg", lam=0.5):
        super().__init__(model, num_clients)
        self.agg_method = agg_method
        self.lam = lam
        self.client_test_losses = None

    def compute_aggregation_weights(self, client_losses, client_weights=None):
        n = np.array(self.client_data_sizes, dtype=float)
        data_w = n / n.sum()

        if self.agg_method == "fedavg":
            return data_w
        elif self.agg_method == "loss_weighted":
            loss_arr = np.array(client_losses, dtype=float)
            q = 1.0 / (loss_arr + 1e-8)
            return q / q.sum()
        elif self.agg_method == "data_loss_weighted":
            loss_arr = np.array(client_losses, dtype=float)
            q = 1.0 / (loss_arr + 1e-8)
            quality_w = q / q.sum()
            return self.lam * data_w + (1.0 - self.lam) * quality_w
        elif self.agg_method == "similarity_aware":
            loss_arr = np.array(client_losses, dtype=float)
            q = 1.0 / (loss_arr + 1e-8)
            quality_w = q / q.sum()
            sim_w = np.ones(self.num_clients, dtype=float) / self.num_clients
            if client_weights is not None and len(client_weights) > 1:
                flat_weights = []
                for cw in client_weights:
                    f = torch.cat([v.view(-1) for v in cw.values()])
                    flat_weights.append(f)
                sim_matrix = np.ones((self.num_clients, self.num_clients))
                for i in range(self.num_clients):
                    for j in range(i + 1, self.num_clients):
                        s = cos_sim(flat_weights[i], flat_weights[j])
                        sim_matrix[i, j] = s
                        sim_matrix[j, i] = s
                sim_scores = sim_matrix.mean(axis=1)
                sim_w = sim_scores / (sim_scores.sum() + 1e-8)
            return 0.3 * data_w + 0.3 * quality_w + 0.4 * sim_w
        elif self.agg_method == "proposed":
            # Proposed: 质量加权 + 动态温度 + 相似性正则
            loss_arr = np.array(client_losses, dtype=float)
            q = 1.0 / (loss_arr + 1e-8)
            quality_w = q / q.sum()
            # 基于 loss 方差动态调整 data/quality 混合比例
            loss_cv = float(np.std(loss_arr)) / (float(np.mean(loss_arr)) + 1e-8)
            # 高异质性时更依赖 quality_weight（避免坏 client 拖累全局）
            dynamic_lam = 1.0 / (1.0 + loss_cv)
            mixed_w = dynamic_lam * data_w + (1.0 - dynamic_lam) * quality_w
            # 再加相似性正则：与均值权重的偏差不能太大
            reg_w = np.ones(self.num_clients) / self.num_clients
            return 0.8 * mixed_w + 0.2 * reg_w
        return data_w

    def aggregate(self, client_weights, client_losses):
        w = self.compute_aggregation_weights(client_losses, client_weights)
        global_dict = self.global_model.state_dict()
        new_dict = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_dict.items()}
        for key in new_dict:
            for idx in range(self.num_clients):
                cw = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                new_dict[key] += cw * torch.tensor(float(w[idx]), device=DEVICE, dtype=torch.float32)
        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(client_losses)))
        return self.global_model.state_dict()


# ══════════════════════════════════════════════════════════════
# 实验执行函数（通用）
# ══════════════════════════════════════════════════════════════

def _make_model(k=NUM_NODES, t=SEQ_LEN, hd=HIDDEN_DIM):
    return CNNEnhancedModel(k=k, t=t, hidden_dim=hd, num_heads=4)


def run_federated_training(client_data, agg_method="fedavg", lam=0.5,
                           comm_rounds=COMM_ROUNDS, local_epochs=LOCAL_EPOCHS,
                           lr=LR, seed=42, verbose=False, record_convergence=False):
    """运行一轮联邦训练并返回结果。

    返回:
        results: list of dict (per-client metrics)
        convergence: dict (round-level data) or None
    """
    set_global_seed(seed)
    num_clients = len(client_data)
    criterion = nn.MSELoss()
    train_sizes = [d["train_size"] for d in client_data]

    fed_clients = [
        FederatedClient(
            d["cid"], _make_model(k=d.get("k_dim", NUM_NODES)), d["train_loader"],
            d["val_loader"], d["test_loader"], criterion, lr=lr)
        for d in client_data
    ]
    server = AggregationServer(_make_model(k=client_data[0].get("k_dim", NUM_NODES)),
                                num_clients, agg_method=agg_method, lam=lam)
    server.set_client_data_sizes(train_sizes)

    convergence = {"round": [], "avg_train_loss": [], "avg_val_rmse": [], "avg_val_rmse_std": [],
                    "avg_val_mae": [], "avg_val_mae_std": []}
    for cid in range(num_clients):
        convergence[f"c{cid}_train"] = []
        convergence[f"c{cid}_val_mse"] = []
        convergence[f"c{cid}_val_rmse"] = []
        convergence[f"c{cid}_val_mae"] = []

    for rnd in range(comm_rounds):
        client_weights, client_losses = [], []
        for client in fed_clients:
            loss, state, _ = client.train_local(
                epochs=local_epochs, global_model=server.global_model, verbose=False)
            client_weights.append(state)
            client_losses.append(loss)
        server.aggregate(client_weights, client_losses)

        if record_convergence:
            val_mses, val_rmses, val_maes = [], [], []
            for cid, client in enumerate(fed_clients):
                client.model.load_state_dict(server.global_model.state_dict())
                cd = client_data[cid]
                mse, rmse, mae = client.validate_metrics(cd["y_mean"], cd["y_std"],
                                                         client.val_loader)
                val_mses.append(mse)
                val_rmses.append(rmse)
                val_maes.append(mae)
            convergence["round"].append(rnd + 1)
            convergence["avg_train_loss"].append(server.round_losses[-1])
            convergence["avg_val_rmse"].append(float(np.mean(val_rmses)))
            convergence["avg_val_rmse_std"].append(float(np.std(val_rmses, ddof=0)))
            convergence["avg_val_mae"].append(float(np.mean(val_maes)))
            convergence["avg_val_mae_std"].append(float(np.std(val_maes, ddof=0)))
            for cid in range(num_clients):
                convergence[f"c{cid}_train"].append(client_losses[cid])
                convergence[f"c{cid}_val_mse"].append(val_mses[cid])
                convergence[f"c{cid}_val_rmse"].append(val_rmses[cid])
                convergence[f"c{cid}_val_mae"].append(val_maes[cid])

        if verbose:
            print(f"  Round {rnd+1}/{comm_rounds} "
                  f"AvgLoss={server.round_losses[-1]:.6f}")

    # 最终评估：全局模型在测试集上
    results = []
    for cid in range(num_clients):
        fed_clients[cid].model.load_state_dict(server.global_model.state_dict())
        metrics = fed_clients[cid].test_metrics()
        # 逆标准化
        preds_raw = metrics["preds"] * client_data[cid]["y_std"] + client_data[cid]["y_mean"]
        truths_raw = metrics["truths"] * client_data[cid]["y_std"] + client_data[cid]["y_mean"]
        mse, rmse, mae = compute_metrics(preds_raw, truths_raw)
        results.append({"client_id": cid, "mse": mse, "rmse": rmse, "mae": mae,
                        "preds": preds_raw[:200], "truths": truths_raw[:200]})

    conv = convergence if record_convergence else None
    return results, conv


def run_independent_training(client_data, total_epochs=30, lr=0.01, seed=42):
    """Independent/Local Training 基线。"""
    set_global_seed(seed)
    criterion = nn.MSELoss()
    results = []
    for d in client_data:
        model = _make_model().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for ep in range(total_epochs):
            model.train()
            tl = 0.0
            for x, y in d["train_loader"]:
                x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
                optimizer.zero_grad()
                pred, _ = model(x)
                loss = criterion(pred.view(-1), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item() * x.shape[0]
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y in d["test_loader"]:
                x = x.to(DEVICE).float()
                pred, _ = model(x)
                preds.append(pred.view(-1).cpu().numpy())
                truths.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        preds_raw = preds * d["y_std"] + d["y_mean"]
        truths_raw = truths * d["y_std"] + d["y_mean"]
        mse, rmse, mae = compute_metrics(preds_raw, truths_raw)
        results.append({"client_id": d["cid"], "mse": mse, "rmse": rmse, "mae": mae,
                        "preds": preds_raw[:200], "truths": truths_raw[:200]})
    return results


# ══════════════════════════════════════════════════════════════
# Workflow: data_viz
# ══════════════════════════════════════════════════════════════

def run_data_visualization_enhanced(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[data_viz] Enhanced Non-IID Dataset Visualizations")
    print("=" * 60)

    set_global_seed(42)
    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    num_nodes = NUM_NODES
    seq_len = SEQ_LEN
    pred_len = PRED_LEN
    ensure_output_dir(output_dir)

    # 为每个 client 生成原始数据（使用更大的时间范围以便可视化）
    raw_signals = []
    raw_masks = []
    for cid, cfg in enumerate(cfgs):
        n_ts = cfg["n_samples"] + 50
        data, mask = generate_traffic_flow(cfg, n_ts, num_nodes, 42 + cid * 100)
        raw_signals.append(data)
        raw_masks.append(mask)

    # ── 1. 每个 client 的平均时间序列 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clients))
    for cid in range(num_clients):
        ts = raw_signals[cid].mean(axis=1)[:200]
        ax.plot(ts, color=colors[cid], linewidth=1.5,
                label=f"C{cid} ({cfgs[cid]['pattern']})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Avg Traffic Flow")
    ax.set_title("Enhanced Dataset: Per-Client Average Traffic Flow Time Series")
    ax.legend(fontsize=7, loc="upper right")
    save_figure(fig, output_dir, "enhanced_dataset_client_timeseries.png")

    # ── 2. 分布对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 箱线图
    box_data = [raw_signals[cid].ravel() for cid in range(num_clients)]
    bp = axes[0].boxplot(box_data, tick_labels=[f"C{cid}\n{cfgs[cid]['dist']}"
                          for cid in range(num_clients)],
                          patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    axes[0].set_title("Traffic Flow Distribution by Client (Boxplot)")
    axes[0].set_ylabel("Traffic Flow")
    # 直方图
    for cid in range(num_clients):
        axes[1].hist(raw_signals[cid].ravel(), bins=40, alpha=0.4,
                     color=colors[cid], label=f"C{cid} ({cfgs[cid]['dist']})")
    axes[1].set_title("Traffic Flow Distribution (Histogram)")
    axes[1].set_xlabel("Traffic Flow")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_dataset_distribution_comparison.png")

    # ── 3. client 配置概览 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cids = np.arange(num_clients)
    # sample size
    axes[0, 0].bar(cids, [c["n_samples"] for c in cfgs], color=colors)
    axes[0, 0].set_title("Sample Size per Client")
    axes[0, 0].set_xticks(cids)
    # noise level
    axes[0, 1].bar(cids, [c["noise"] for c in cfgs], color=colors)
    axes[0, 1].set_title("Noise Level per Client")
    axes[0, 1].set_xticks(cids)
    # base flow
    axes[1, 0].bar(cids, [c["base"] for c in cfgs], color=colors)
    axes[1, 0].set_title("Base Flow per Client")
    axes[1, 0].set_xticks(cids)
    # incident prob
    axes[1, 1].bar(cids, [c.get("incident_prob", 0) for c in cfgs], color=colors)
    axes[1, 1].set_title("Incident Probability per Client")
    axes[1, 1].set_xticks(cids)
    plt.tight_layout()
    save_figure(fig, output_dir, "enhanced_dataset_client_config.png")

    # ── 4. 高峰模式 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    for cid in range(num_clients):
        ts = raw_signals[cid].mean(axis=1)
        ax.plot(ts[:24], "o-", color=colors[cid], linewidth=2, markersize=4,
                label=f"C{cid}: {cfgs[cid]['pattern']}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Traffic Flow")
    ax.set_title("Enhanced Dataset: 24-Hour Peak Patterns")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "enhanced_dataset_peak_pattern.png")

    # ── 5. incident 示例 ──
    incident_cid = 4  # 突发拥堵型
    fig, ax = plt.subplots(figsize=(14, 5))
    ts = raw_signals[incident_cid].mean(axis=1)
    mask = raw_masks[incident_cid]
    t = np.arange(len(ts))
    ax.plot(t, ts, color="#3498db", linewidth=1, label="Traffic Flow")
    # 找到 incident 区间
    in_incident = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_incident:
            start = i; in_incident = True
        elif not mask[i] and in_incident:
            ax.axvspan(start, i, alpha=0.3, color="#e74c3c")
            in_incident = False
    if in_incident:
        ax.axvspan(start, len(mask) - 1, alpha=0.3, color="#e74c3c")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Avg Traffic Flow")
    ax.set_title(f"Enhanced Dataset: Client {incident_cid} ({cfgs[incident_cid]['pattern']}) "
                 "with Incident Periods Shaded")
    ax.legend()
    save_figure(fig, output_dir, "enhanced_dataset_incident_example.png")

    # ── 6. client 间相关系数矩阵 ──
    min_len = min(len(raw_signals[cid]) for cid in range(num_clients))
    client_ts = np.array([raw_signals[cid].mean(axis=1)[:min_len] for cid in range(num_clients)])
    corr_client = np.corrcoef(client_ts)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_client, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(num_clients))
    ax.set_yticks(range(num_clients))
    ax.set_xticklabels([f"C{cid}" for cid in range(num_clients)])
    ax.set_yticklabels([f"C{cid}" for cid in range(num_clients)])
    for i in range(num_clients):
        for j in range(num_clients):
            ax.text(j, i, f"{corr_client[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Enhanced Dataset: Inter-Client Correlation Matrix")
    plt.colorbar(im, ax=ax)
    save_figure(fig, output_dir, "enhanced_dataset_client_correlation_matrix.png")

    # ── 7. 节点间相关系数矩阵 ──
    rep_cid = 0
    node_corr = np.corrcoef(raw_signals[rep_cid].T)  # (n_nodes, n_nodes)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(node_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    for i in range(num_nodes):
        for j in range(num_nodes):
            ax.text(j, i, f"{node_corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Enhanced Dataset: Node Correlation Matrix (Client {rep_cid})")
    ax.set_xlabel("Node ID"); ax.set_ylabel("Node ID")
    plt.colorbar(im, ax=ax)
    save_figure(fig, output_dir, "enhanced_dataset_node_correlation_matrix.png")

    # ── 8. 汇总 CSV ──
    rows = []
    for cid, cfg in enumerate(cfgs):
        ts = raw_signals[cid].ravel()
        rows.append({
            "client_id": cid,
            "distribution_type": cfg["dist"],
            "traffic_pattern": cfg["pattern"],
            "sample_size": cfg["n_samples"],
            "noise_level": cfg["noise"],
            "base_flow": cfg["base"],
            "morning_mu": cfg["morning_mu"],
            "evening_mu": cfg["evening_mu"],
            "morning_amp": cfg["morning_amp"],
            "evening_amp": cfg["evening_amp"],
            "incident_prob": cfg.get("incident_prob", 0),
            "mean_flow": float(np.mean(ts)),
            "std_flow": float(np.std(ts)),
            "min_flow": float(np.min(ts)),
            "max_flow": float(np.max(ts)),
        })
    df_sum = pd.DataFrame(rows)
    save_dataframe(df_sum, output_dir, "enhanced_dataset_summary.csv")
    print("[data_viz] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow: main
# ══════════════════════════════════════════════════════════════

def run_main_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[main] Independent / FedAvg / Proposed Comparison")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)

    all_rows = []
    for seed in SEEDS:
        print(f"\n--- Seed = {seed} ---")
        client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)

        # FedAvg
        fed_results, _ = run_federated_training(client_data, agg_method="fedavg",
                                                 seed=seed, verbose=False)
        for r in fed_results:
            all_rows.append({"seed": seed, "method": "FedAvg",
                             "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

        # Proposed
        prop_results, _ = run_federated_training(client_data, agg_method="proposed",
                                                  seed=seed, verbose=False)
        for r in prop_results:
            all_rows.append({"seed": seed, "method": "Proposed",
                             "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

        # Independent
        ind_results = run_independent_training(client_data, total_epochs=10,
                                                lr=0.01, seed=seed)
        for r in ind_results:
            all_rows.append({"seed": seed, "method": "Independent",
                             "client_id": r["client_id"],
                             "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "cnn_enhanced_main_metrics.csv")

    # 汇总
    agg = df.groupby("method").agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_main_metrics_summary.csv")
    print("\n[main] Summary:\n", agg.to_string(index=False))

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ["Independent", "FedAvg", "Proposed"]
    bar_colors = {"Independent": "#e74c3c", "FedAvg": "#3498db", "Proposed": "#2ecc71"}
    for idx, metric in enumerate(["mse", "rmse", "mae"]):
        ax = axes[idx]
        vals = [agg[agg["method"] == m][f"{metric}_mean"].values[0] for m in methods]
        errs = [agg[agg["method"] == m][f"{metric}_std"].values[0] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, vals, color=[bar_colors[m] for m in methods],
               yerr=errs, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
    fig.suptitle("CNN Enhanced: Main Experiment Results", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_main_comparison.png")
    print("[main] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow: aggregation
# ══════════════════════════════════════════════════════════════

def run_aggregation_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[aggregation] Aggregation Strategy Ablation")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)
    agg_methods = ["fedavg", "loss_weighted", "data_loss_weighted", "proposed"]
    agg_labels = ["FedAvg", "Loss-weighted", "Data-loss weighted", "Proposed"]

    all_rows = []
    for seed in [42]:  # 单 seed 加速
        print(f"\n--- Seed = {seed} ---")
        client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
        for method, label in zip(agg_methods, agg_labels):
            print(f"  [{label}]")
            results, _ = run_federated_training(
                client_data, agg_method=method, seed=seed, verbose=False)
            for r in results:
                all_rows.append({"seed": seed, "aggregation_method": label,
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "cnn_enhanced_aggregation_ablation.csv")
    agg = df.groupby("aggregation_method").agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_aggregation_ablation_summary.csv")
    print("\n[aggregation] Summary:\n", agg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, metric in enumerate(["rmse", "mae"]):
        ax = axes[idx]
        x = np.arange(len(agg_labels))
        vals = [agg[agg["aggregation_method"] == l][f"{metric}_mean"].values[0] for l in agg_labels]
        errs = [agg[agg["aggregation_method"] == l][f"{metric}_std"].values[0] for l in agg_labels]
        ax.bar(x, vals, capsize=5,
               color=plt.cm.viridis(np.linspace(0.1, 0.9, len(agg_labels))))
        ax.set_xticks(x)
        ax.set_xticklabels(agg_labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{metric.upper()} by Aggregation Method")
        ax.set_ylabel(metric.upper())
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_aggregation_ablation.png")
    print("[aggregation] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow: lambda
# ══════════════════════════════════════════════════════════════

def run_lambda_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[lambda] Lambda Sensitivity Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)
    lam_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_rows = []
    for seed in [42]:  # 单 seed 加速
        print(f"\n--- Seed = {seed} ---")
        client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
        for lam in lam_vals:
            print(f"  [lambda={lam:.2f}]")
            results, _ = run_federated_training(
                client_data, agg_method="data_loss_weighted", lam=lam,
                seed=seed, verbose=False)
            for r in results:
                all_rows.append({"seed": seed, "lambda": lam,
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "cnn_enhanced_lambda_sensitivity.csv")
    agg = df.groupby("lambda").agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_lambda_sensitivity_summary.csv")
    print("\n[lambda] Summary:\n", agg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(lam_vals, agg["rmse_mean"], yerr=agg["rmse_std"],
                fmt="o-", capsize=5, label="RMSE", linewidth=2, color="#3498db")
    ax2 = ax.twinx()
    ax2.errorbar(lam_vals, agg["mae_mean"], yerr=agg["mae_std"],
                 fmt="s-", capsize=5, label="MAE", linewidth=2, color="#e74c3c")
    ax.set_xlabel("Lambda (data_weight fraction)")
    ax.set_ylabel("RMSE", color="#3498db")
    ax2.set_ylabel("MAE", color="#e74c3c")
    ax.set_title("Data-Loss Weighted: Lambda Sensitivity")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_lambda_sensitivity.png")
    print("[lambda] Done.\n")


# ══════════════════════════════════════════════════════════════
# 占位 workflow（后续补充）
# ══════════════════════════════════════════════════════════════

def run_convergence_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[convergence] FedAvg vs Proposed Convergence Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    ensure_output_dir(output_dir)

    # 增加通信轮次以观察完整收敛
    conv_rounds = 15
    all_round_rows = []

    for method, agg_method in [("FedAvg", "fedavg"), ("Proposed", "proposed")]:
        print(f"\n--- {method} Convergence (seed=42) ---")
        set_global_seed(42)
        client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, 42)
        _, conv = run_federated_training(
            client_data, agg_method=agg_method, seed=42,
            comm_rounds=conv_rounds, record_convergence=True)

        for r_idx in range(len(conv["round"])):
            rnd = conv["round"][r_idx]
            for cid in range(num_clients):
                all_round_rows.append({
                    "round": rnd,
                    "method": method,
                    "client_id": cid,
                    "train_loss": conv[f"c{cid}_train"][r_idx],
                    "val_mse": conv[f"c{cid}_val_mse"][r_idx],
                    "val_rmse": conv[f"c{cid}_val_rmse"][r_idx],
                    "val_mae": conv[f"c{cid}_val_mae"][r_idx],
                })

    df_round = pd.DataFrame(all_round_rows)
    save_dataframe(df_round, output_dir, "cnn_enhanced_convergence_round_metrics.csv")

    # 汇总
    agg = df_round.groupby(["round", "method"]).agg(
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        val_mae_mean=("val_mae", "mean"), val_mae_std=("val_mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_convergence_summary.csv")
    print("\n[convergence] Summary (last round):")
    last = agg[agg["round"] == agg["round"].max()]
    print(last.to_string(index=False))

    # ── 图 1: Global Validation RMSE ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, color in [("FedAvg", "#3498db"), ("Proposed", "#2ecc71")]:
        sub = agg[agg["method"] == method]
        ax.plot(sub["round"], sub["val_rmse_mean"], "o-",
                color=color, linewidth=2, label=f"{method} val RMSE")
        ax.fill_between(sub["round"],
                         sub["val_rmse_mean"] - sub["val_rmse_std"],
                         sub["val_rmse_mean"] + sub["val_rmse_std"],
                         alpha=0.15, color=color)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Validation RMSE (real scale)")
    ax.set_title("CNN Enhanced: Global Validation RMSE Convergence")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_global_validation_rmse.png")

    # ── 图 2: Client Training Loss ──
    fedavg_sub = df_round[df_round["method"] == "FedAvg"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for cid in range(num_clients):
        cdata = fedavg_sub[fedavg_sub["client_id"] == cid]
        axes[0].plot(cdata["round"], cdata["train_loss"], "o-",
                     label=f"Client {cid}", linewidth=1.5, markersize=3)
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Local Train Loss (MSE)")
    axes[0].set_title("FedAvg: Per-Client Training Loss")
    axes[0].legend(fontsize=7)

    prop_sub = df_round[df_round["method"] == "Proposed"]
    for cid in range(num_clients):
        cdata = prop_sub[prop_sub["client_id"] == cid]
        axes[1].plot(cdata["round"], cdata["train_loss"], "s--",
                     label=f"Client {cid}", linewidth=1.5, markersize=3)
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Local Train Loss (MSE)")
    axes[1].set_title("Proposed: Per-Client Training Loss")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_client_training_loss.png")

    # ── 图 3: Convergence Overview ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # (a) Global val RMSE
    ax = axes[0, 0]
    for method, color in [("FedAvg", "#3498db"), ("Proposed", "#2ecc71")]:
        sub = agg[agg["method"] == method]
        ax.plot(sub["round"], sub["val_rmse_mean"], "o-", color=color, linewidth=2, label=method)
        ax.fill_between(sub["round"],
                         sub["val_rmse_mean"] - sub["val_rmse_std"],
                         sub["val_rmse_mean"] + sub["val_rmse_std"],
                         alpha=0.12, color=color)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Val RMSE")
    ax.set_title("(a) Global Validation RMSE"); ax.legend()

    # (b) Global val MAE
    ax = axes[0, 1]
    for method, color in [("FedAvg", "#3498db"), ("Proposed", "#2ecc71")]:
        sub = agg[agg["method"] == method]
        ax.plot(sub["round"], sub["val_mae_mean"], "s--", color=color, linewidth=2, label=method)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Val MAE")
    ax.set_title("(b) Global Validation MAE"); ax.legend()

    # (c) FedAvg client training loss
    ax = axes[1, 0]
    for cid in range(num_clients):
        cdata = fedavg_sub[fedavg_sub["client_id"] == cid]
        ax.plot(cdata["round"], cdata["train_loss"], "o-", label=f"C{cid}", markersize=3)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Train Loss")
    ax.set_title("(c) FedAvg: Client Training Loss"); ax.legend(fontsize=7)

    # (d) Proposed client training loss
    ax = axes[1, 1]
    for cid in range(num_clients):
        cdata = prop_sub[prop_sub["client_id"] == cid]
        ax.plot(cdata["round"], cdata["train_loss"], "s--", label=f"C{cid}", markersize=3)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Train Loss")
    ax.set_title("(d) Proposed: Client Training Loss"); ax.legend(fontsize=7)

    fig.suptitle("CNN Enhanced: Convergence Overview", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_convergence_overview.png")
    print("[convergence] Done.\n")


def run_client_scale_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[client_scale] Client Count Sensitivity Experiment")
    print("=" * 60)

    client_nums = [3, 5, 8, 10]
    ensure_output_dir(output_dir)

    all_rows = []
    for nc in client_nums:
        print(f"\n--- Num Clients = {nc} ---")
        cfgs = build_noniid_client_configs(nc, "medium")
        for seed in SEEDS:
            print(f"  Seed = {seed}")
            client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)

            # Independent
            ind_results = run_independent_training(client_data, total_epochs=10,
                                                    lr=0.01, seed=seed)
            for r in ind_results:
                all_rows.append({"seed": seed, "num_clients": nc, "method": "Independent",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

            # FedAvg
            fed_results, _ = run_federated_training(client_data, agg_method="fedavg",
                                                     seed=seed, verbose=False)
            for r in fed_results:
                all_rows.append({"seed": seed, "num_clients": nc, "method": "FedAvg",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

            # Proposed
            prop_results, _ = run_federated_training(client_data, agg_method="proposed",
                                                      seed=seed, verbose=False)
            for r in prop_results:
                all_rows.append({"seed": seed, "num_clients": nc, "method": "Proposed",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "cnn_enhanced_client_scale_metrics.csv")

    agg = df.groupby(["num_clients", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_client_scale_summary.csv")
    print("\n[client_scale] Summary:\n", agg.to_string(index=False))

    # 绘制对比图
    methods = ["Independent", "FedAvg", "Proposed"]
    bar_colors = {"Independent": "#e74c3c", "FedAvg": "#3498db", "Proposed": "#2ecc71"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for m_idx, method in enumerate(methods):
        sub = agg[agg["method"] == method].sort_values("num_clients")
        xs = sub["num_clients"].astype(str)
        axes[0].errorbar(range(len(xs)), sub["rmse_mean"], yerr=sub["rmse_std"],
                         fmt="o-", capsize=5, label=method, linewidth=2,
                         color=bar_colors[method])
        axes[1].errorbar(range(len(xs)), sub["mae_mean"], yerr=sub["mae_std"],
                         fmt="s--", capsize=5, label=method, linewidth=2,
                         color=bar_colors[method])
    for ax in axes:
        ax.set_xticks(range(len(client_nums)))
        ax.set_xticklabels([str(n) for n in client_nums])
        ax.set_xlabel("Number of Clients")
        ax.legend(fontsize=9)
    axes[0].set_title("RMSE vs Client Count")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE vs Client Count")
    axes[1].set_ylabel("MAE")
    fig.suptitle("CNN Enhanced: Client Scale Sensitivity", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_client_scale.png")
    print("[client_scale] Done.\n")


def run_noniid_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[noniid] Non-IID Strength Experiment")
    print("=" * 60)

    levels = ["low", "medium", "high"]
    num_clients = 5
    ensure_output_dir(output_dir)

    all_rows = []
    for level in levels:
        print(f"\n--- Non-IID Level = {level} ---")
        cfgs = build_noniid_client_configs(num_clients, level)
        for seed in SEEDS:
            print(f"  Seed = {seed}")
            client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)

            # Independent
            ind_results = run_independent_training(client_data, total_epochs=10,
                                                    lr=0.01, seed=seed)
            for r in ind_results:
                all_rows.append({"seed": seed, "noniid_level": level, "method": "Independent",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

            # FedAvg
            fed_results, _ = run_federated_training(client_data, agg_method="fedavg",
                                                     seed=seed, verbose=False)
            for r in fed_results:
                all_rows.append({"seed": seed, "noniid_level": level, "method": "FedAvg",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

            # Proposed
            prop_results, _ = run_federated_training(client_data, agg_method="proposed",
                                                      seed=seed, verbose=False)
            for r in prop_results:
                all_rows.append({"seed": seed, "noniid_level": level, "method": "Proposed",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "cnn_enhanced_noniid_strength_metrics.csv")

    agg = df.groupby(["noniid_level", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_noniid_strength_summary.csv")
    print("\n[noniid] Summary:\n", agg.to_string(index=False))

    # 绘制对比图 — 横轴固定顺序 low -> medium -> high
    methods = ["Independent", "FedAvg", "Proposed"]
    bar_colors = {"Independent": "#e74c3c", "FedAvg": "#3498db", "Proposed": "#2ecc71"}
    level_order = ["low", "medium", "high"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(methods))
    width = 0.25
    for l_idx, level in enumerate(level_order):
        sub = agg[agg["noniid_level"] == level]
        offset = (l_idx - 1) * width
        rmse_vals = [sub[sub["method"] == m]["rmse_mean"].values[0] for m in methods]
        mae_vals = [sub[sub["method"] == m]["mae_mean"].values[0] for m in methods]
        axes[0].bar(x + offset, rmse_vals, width, label=level, alpha=0.85)
        axes[1].bar(x + offset, mae_vals, width, label=level, alpha=0.85)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_xlabel("Method")
        ax.legend(title="Non-IID Level", fontsize=8)
    axes[0].set_title("RMSE by Non-IID Strength")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE by Non-IID Strength")
    axes[1].set_ylabel("MAE")
    fig.suptitle("CNN Enhanced: Non-IID Strength Sensitivity", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_noniid_strength.png")
    print("[noniid] Done.\n")


def run_client_metrics_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[client_metrics] Per-Client Error Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    ensure_output_dir(output_dir)
    seed = 42

    set_global_seed(seed)
    client_data = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)

    # Independent
    print("\n[Independent]")
    ind_results = run_independent_training(client_data, total_epochs=10, lr=0.01, seed=seed)

    # FedAvg
    print("[FedAvg]")
    fed_results, _ = run_federated_training(
        client_data, agg_method="fedavg", seed=seed, verbose=False)

    # Proposed
    print("[Proposed]")
    prop_results, _ = run_federated_training(
        client_data, agg_method="proposed", seed=seed, verbose=False)

    # 组装详细指标
    rows = []
    for cid in range(num_clients):
        cfg = cfgs[cid]
        im = ind_results[cid]
        fm = fed_results[cid]
        pm = prop_results[cid]

        # Proposed 相对于 FedAvg 的改善率
        imp_fedavg = (fm["rmse"] - pm["rmse"]) / (fm["rmse"] + 1e-12) * 100
        imp_ind = (im["rmse"] - pm["rmse"]) / (im["rmse"] + 1e-12) * 100

        rows.append({
            "method": "Independent", "client_id": cid,
            "distribution_type": cfg["dist"],
            "traffic_pattern": cfg["pattern"],
            "sample_size": cfg["n_samples"], "noise_level": cfg["noise"],
            "incident_prob": cfg.get("incident_prob", 0),
            "mse": im["mse"], "rmse": im["rmse"], "mae": im["mae"],
            "improvement_over_fedavg_rmse": float("nan"),
            "improvement_over_independent_rmse": float("nan"),
        })
        rows.append({
            "method": "FedAvg", "client_id": cid,
            "distribution_type": cfg["dist"],
            "traffic_pattern": cfg["pattern"],
            "sample_size": cfg["n_samples"], "noise_level": cfg["noise"],
            "incident_prob": cfg.get("incident_prob", 0),
            "mse": fm["mse"], "rmse": fm["rmse"], "mae": fm["mae"],
            "improvement_over_fedavg_rmse": float("nan"),
            "improvement_over_independent_rmse": float("nan"),
        })
        rows.append({
            "method": "Proposed", "client_id": cid,
            "distribution_type": cfg["dist"],
            "traffic_pattern": cfg["pattern"],
            "sample_size": cfg["n_samples"], "noise_level": cfg["noise"],
            "incident_prob": cfg.get("incident_prob", 0),
            "mse": pm["mse"], "rmse": pm["rmse"], "mae": pm["mae"],
            "improvement_over_fedavg_rmse": round(imp_fedavg, 2),
            "improvement_over_independent_rmse": round(imp_ind, 2),
        })

    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_client_metrics.csv")
    print("\n[client_metrics] Per-client results:")
    print(df.to_string(index=False))

    # ── 图 1: 每个 client 的 RMSE 对比 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(num_clients)
    width = 0.25
    methods_data = [
        ("Independent", "#e74c3c", ind_results),
        ("FedAvg", "#3498db", fed_results),
        ("Proposed", "#2ecc71", prop_results),
    ]
    for m_idx, (name, color, m_list) in enumerate(methods_data):
        rmse_vals = [r["rmse"] for r in m_list]
        ax.bar(x + (m_idx - 1) * width, rmse_vals, width, label=name, color=color, alpha=0.9)
    ax.set_xticks(x)
    labels = [f"C{cid}\n{cfgs[cid]['dist']}\n{cfgs[cid]['pattern']}" for cid in range(num_clients)]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("CNN Enhanced: Per-Client RMSE Comparison")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_client_rmse_comparison.png")

    # ── 图 2: Proposed 改善率 ──
    prop_df = df[df["method"] == "Proposed"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_clients)
    width = 0.3
    ax.bar(x - width / 2, prop_df["improvement_over_fedavg_rmse"], width,
           label="vs FedAvg", color="#3498db")
    ax.bar(x + width / 2, prop_df["improvement_over_independent_rmse"], width,
           label="vs Independent", color="#2ecc71")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid}\n{cfgs[cid]['pattern']}" for cid in range(num_clients)], fontsize=9)
    ax.set_ylabel("RMSE Improvement (%)")
    ax.set_title("CNN Enhanced: Proposed RMSE Improvement by Client")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_client_improvement.png")
    print("[client_metrics] Done.\n")


def run_peak_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[peak] Peak / Off-peak / Incident Period Analysis")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    num_clients = len(cfgs)
    ensure_output_dir(output_dir)
    seed = 42

    set_global_seed(seed)
    client_data = build_client_data_shuffled(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
    all_rows = []

    # ---- Independent ----
    print("\n[Independent]")
    ind_results = run_independent_training(client_data, total_epochs=10, lr=0.01, seed=seed)

    # ---- FedAvg ----
    print("[FedAvg]")
    fed_results, _ = run_federated_training(client_data, agg_method="fedavg", seed=seed, verbose=False)

    # ---- Proposed ----
    print("[Proposed]")
    prop_results, _ = run_federated_training(client_data, agg_method="proposed", seed=seed, verbose=False)

    # 评估：用 per-sample 真实尺度预测分类到 period
    for method, agg_results in [("Independent", ind_results),
                                  ("FedAvg", fed_results),
                                  ("Proposed", prop_results)]:
        for cid in range(num_clients):
            cd = client_data[cid]
            meta = cd["meta_test"]
            hours = meta["target_hour"]
            inc_flags = meta["target_incident_flag"]
            sample_periods = [classify_period(h, f) for h, f in zip(hours, inc_flags)]

            y_std = cd["y_std"]
            y_mean = cd["y_mean"]
            preds_raw = agg_results[cid]["preds"]
            truths_raw = agg_results[cid]["truths"]
            errors = (preds_raw - truths_raw)

            # 确保长度匹配（preds 可能被截断到 200）
            n_use = min(len(errors), len(sample_periods))
            for i in range(n_use):
                period = sample_periods[i]
                all_rows.append({
                    "method": method, "client_id": cid, "period": period,
                    "mse": float(errors[i] ** 2),
                    "rmse": float(abs(errors[i])),
                    "mae": float(abs(errors[i])),
                    "num_samples": 1,
                })

    df = pd.DataFrame(all_rows)

    # 聚合 metrics
    metrics_rows = []
    for (method, cid, period), grp in df.groupby(["method", "client_id", "period"]):
        mse_val = float(np.mean([r for r in grp["mse"]]))
        rmse_val = float(np.sqrt(mse_val))
        mae_val = float(np.mean([r for r in grp["mae"]]))
        metrics_rows.append({
            "method": method, "client_id": cid, "period": period,
            "mse": mse_val, "rmse": rmse_val, "mae": mae_val,
            "num_samples": len(grp),
        })

    df_metrics = pd.DataFrame(metrics_rows)
    save_dataframe(df_metrics, output_dir, "cnn_enhanced_peak_offpeak_metrics.csv")

    # 汇总
    agg_sum = df_metrics.groupby(["method", "period"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        total_samples=("num_samples", "sum"),
    ).reset_index()
    save_dataframe(agg_sum, output_dir, "cnn_enhanced_peak_offpeak_summary.csv")
    print("\n[peak] Summary:\n", agg_sum.to_string(index=False))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    methods = ["Independent", "FedAvg", "Proposed"]
    bar_colors = {"Independent": "#e74c3c", "FedAvg": "#3498db", "Proposed": "#2ecc71"}
    x = np.arange(len(methods))
    width = 0.2
    period_order = ["morning_peak", "evening_peak", "off_peak", "incident_period"]
    for p_idx, period in enumerate(period_order):
        sub = agg_sum[agg_sum["period"] == period]
        offset = (p_idx - 1.5) * width
        rmse_vals = [sub[sub["method"] == m]["rmse_mean"].values[0] if m in sub["method"].values else np.nan for m in methods]
        mae_vals = [sub[sub["method"] == m]["mae_mean"].values[0] if m in sub["method"].values else np.nan for m in methods]
        axes[0].bar(x + offset, rmse_vals, width, label=period, alpha=0.85)
        axes[1].bar(x + offset, mae_vals, width, label=period, alpha=0.85)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_xlabel("Method")
        ax.legend(fontsize=7, title="Period")
    axes[0].set_title("RMSE by Traffic Period")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE by Traffic Period")
    axes[1].set_ylabel("MAE")
    fig.suptitle("CNN Enhanced: Peak / Off-peak / Incident Analysis", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_peak_offpeak_comparison.png")
    print("[peak] Done.\n")


def run_feature_ablation_experiment(output_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("[feature_ablation] Input Feature Ablation")
    print("=" * 60)

    cfgs = list(CLIENT_CONFIGS_BASE)
    ensure_output_dir(output_dir)
    feature_sets = ["flow_only", "flow_time", "flow_event", "flow_region", "full"]

    all_rows = []
    for fs in feature_sets:
        print(f"\n--- Feature Set: {fs} ---")
        for seed in SEEDS:
            print(f"  Seed = {seed}")
            set_global_seed(seed)
            client_data = build_feature_augmented_data(cfgs, fs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)

            # FedAvg
            fed_results, _ = run_federated_training(client_data, agg_method="fedavg",
                                                     seed=seed, verbose=False)
            for r in fed_results:
                all_rows.append({"seed": seed, "feature_set": fs, "method": "FedAvg",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

            # Proposed
            prop_results, _ = run_federated_training(client_data, agg_method="proposed",
                                                      seed=seed, verbose=False)
            for r in prop_results:
                all_rows.append({"seed": seed, "feature_set": fs, "method": "Proposed",
                                 "client_id": r["client_id"],
                                 "mse": r["mse"], "rmse": r["rmse"], "mae": r["mae"]})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, output_dir, "cnn_enhanced_feature_ablation.csv")

    agg = df.groupby(["feature_set", "method"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()
    save_dataframe(agg, output_dir, "cnn_enhanced_feature_ablation_summary.csv")
    print("\n[feature_ablation] Summary:\n", agg.to_string(index=False))

    # 绘图
    fs_labels = {"flow_only": "Flow Only", "flow_time": "+ Time",
                 "flow_event": "+ Event", "flow_region": "+ Region", "full": "Full"}
    fs_order = list(feature_sets)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for m_idx, method in enumerate(["FedAvg", "Proposed"]):
        sub = agg[agg["method"] == method]
        x = np.arange(len(fs_order))
        rmse_vals = [sub[sub["feature_set"] == fs]["rmse_mean"].values[0] for fs in fs_order]
        mae_vals = [sub[sub["feature_set"] == fs]["mae_mean"].values[0] for fs in fs_order]
        axes[0].plot(x, rmse_vals, "o-", label=method, linewidth=2)
        axes[1].plot(x, mae_vals, "s--", label=method, linewidth=2)
    for ax in axes:
        ax.set_xticks(range(len(fs_order)))
        ax.set_xticklabels([fs_labels[fs] for fs in fs_order], rotation=20, ha="right")
        ax.set_xlabel("Feature Set")
        ax.legend()
    axes[0].set_title("RMSE by Feature Set")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE by Feature Set")
    axes[1].set_ylabel("MAE")
    fig.suptitle("CNN Enhanced: Feature Ablation", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_feature_ablation.png")
    print("[feature_ablation] Done.\n")


# ══════════════════════════════════════════════════════════════
# 工作流调度
# ══════════════════════════════════════════════════════════════

WORKFLOW_MAP = {
    "all": ["data_viz", "main", "aggregation", "lambda",
            "client_scale", "noniid", "convergence",
            "client_metrics", "peak", "feature_ablation"],
    "data_viz":        ["data_viz"],
    "main":            ["main"],
    "aggregation":     ["aggregation"],
    "lambda":          ["lambda"],
    "convergence":     ["convergence"],
    "client_scale":    ["client_scale"],
    "noniid":          ["noniid"],
    "client_metrics":  ["client_metrics"],
    "peak":            ["peak"],
    "feature_ablation":["feature_ablation"],
}

WORKFLOW_FUNCTIONS = {
    "data_viz":        run_data_visualization_enhanced,
    "main":            run_main_experiment,
    "aggregation":     run_aggregation_experiment,
    "lambda":          run_lambda_experiment,
    "convergence":     run_convergence_experiment,
    "client_scale":    run_client_scale_experiment,
    "noniid":          run_noniid_experiment,
    "client_metrics":  run_client_metrics_experiment,
    "peak":            run_peak_experiment,
    "feature_ablation":run_feature_ablation_experiment,
}


def run_project(workflow: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    print(f"[cnn_fed_enhanced] workflow={workflow}, device={DEVICE}")
    print(f"[cnn_fed_enhanced] output={output_dir}")

    steps = WORKFLOW_MAP[workflow]
    for step in steps:
        fn = WORKFLOW_FUNCTIONS[step]
        print(f"\n>>> Running step: {step}")
        fn(output_dir)

    print(f"\n[cnn_fed_enhanced] All done. Results in: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="CNN/CCN Enhanced Federated Simulation")
    parser.add_argument("--workflow", choices=list(WORKFLOW_MAP.keys()),
                        default="all", help="Workflow to execute (default: all).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    output_dir = SIMULATION_RESULTS_ROOT / "cnn_fed_enhanced"
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
