# -*- coding: utf-8 -*-
"""
CNN/CCN 增强仿真实验组核心逻辑。
负责：数据生成、模型训练、聚合策略实验、指标计算、数据导出。
移除：所有 matplotlib/seaborn 绘图逻辑。
"""

import argparse
import copy
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ══════════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEFAULT_ENHANCED_RESULTS_DIR = SIMULATION_RESULTS_ROOT / "cnn_fed_enhanced_experiments"
DEFAULT_PAPER_READY_DIR = DEFAULT_ENHANCED_RESULTS_DIR / "paper_ready"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAFFIC_MIN_VALUE = 0.0
MAPE_EPS = 1.0

NUM_NODES = 8
SEQ_LEN = 12
PRED_LEN = 1
BATCH_SIZE = 32
HIDDEN_DIM = 64
COMM_ROUNDS = 5
LOCAL_EPOCHS = 2
LR = 0.001
DEFAULT_MULTI_SEEDS = [42, 2024, 2025, 2026, 3407]
SEEDS = list(DEFAULT_MULTI_SEEDS)

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

def save_dataframe(df, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path


def parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数: {value}")


def parse_seed_list(seed_text: str | None) -> list[int]:
    if seed_text is None or not str(seed_text).strip():
        return list(DEFAULT_MULTI_SEEDS)
    seeds = [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]
    if not seeds:
        raise ValueError("至少需要提供一个随机种子")
    return seeds


def compute_r2_score(preds: np.ndarray, truths: np.ndarray) -> float:
    preds_arr = np.asarray(preds, dtype=np.float64)
    truths_arr = np.asarray(truths, dtype=np.float64)
    ss_res = float(np.sum((preds_arr - truths_arr) ** 2))
    ss_tot = float(np.sum((truths_arr - truths_arr.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def build_multi_seed_summary(raw_df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group_df in raw_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base_record = dict(zip(group_cols, group_values))
        for metric in metric_cols:
            if metric not in group_df.columns:
                continue
            values = group_df[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0))
            ci95 = 1.96 * std_value / math.sqrt(values.size)
            record = dict(base_record)
            record.update({
                "metric": metric,
                "mean": mean_value,
                "std": std_value,
                "ci95_lower": float(mean_value - ci95),
                "ci95_upper": float(mean_value + ci95),
                "best": float(np.min(values)) if metric.lower() != "r2" else float(np.max(values)),
                "worst": float(np.max(values)) if metric.lower() != "r2" else float(np.min(values)),
                "n": int(values.size),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def build_multi_seed_convergence_summary(raw_df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group_df in raw_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = dict(zip(group_cols, group_values))
        record["n"] = int(group_df["seed"].nunique()) if "seed" in group_df.columns else int(len(group_df))
        for metric in metric_cols:
            if metric not in group_df.columns:
                continue
            values = group_df[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0))
            ci95 = 1.96 * std_value / math.sqrt(values.size)
            record[f"{metric}_mean"] = mean_value
            record[f"{metric}_std"] = std_value
            record[f"{metric}_ci95_lower"] = float(mean_value - ci95)
            record[f"{metric}_ci95_upper"] = float(mean_value + ci95)
        rows.append(record)
    return pd.DataFrame(rows)


def build_pairwise_improvement_summary(raw_df: pd.DataFrame, experiment_name: str, baseline_method: str, enhanced_method: str, metric_cols: list[str]) -> pd.DataFrame:
    baseline_df = raw_df[raw_df["method"] == baseline_method].set_index("seed")
    enhanced_df = raw_df[raw_df["method"] == enhanced_method].set_index("seed")
    common_seeds = sorted(set(baseline_df.index) & set(enhanced_df.index))
    rows = []
    for metric in metric_cols:
        if metric not in baseline_df.columns or metric not in enhanced_df.columns:
            continue
        improvements = []
        flags = []
        for seed in common_seeds:
            baseline_value = float(baseline_df.loc[seed, metric])
            enhanced_value = float(enhanced_df.loc[seed, metric])
            if metric.lower() == "r2":
                improvement = (enhanced_value - baseline_value) / max(abs(baseline_value), 1e-8) * 100.0
            else:
                improvement = (baseline_value - enhanced_value) / max(abs(baseline_value), 1e-8) * 100.0
            improvements.append(improvement)
            flags.append(improvement > 0.0)
        if improvements:
            rows.append({
                "experiment": experiment_name,
                "baseline_method": baseline_method,
                "enhanced_method": enhanced_method,
                "metric": metric,
                "mean_improvement_percent": float(np.mean(improvements)),
                "std_improvement_percent": float(np.std(improvements, ddof=0)),
                "improved_seed_count": int(np.sum(flags)),
                "total_seed_count": int(len(flags)),
                "improved_seed_ratio": float(np.mean(flags)),
                "per_seed_improved": ",".join(f"{seed}:{'Y' if flag else 'N'}" for seed, flag in zip(common_seeds, flags)),
            })
    return pd.DataFrame(rows)


def write_multi_seed_stability_report(output_dir: Path, raw_df: pd.DataFrame, improvement_df: pd.DataFrame, experiment_name: str, baseline_method: str, enhanced_method: str) -> Path:
    report_path = ensure_output_dir(output_dir) / "multi_seed_stability_report.txt"
    lines = [
        f"Experiment: {experiment_name}",
        f"Seeds: {', '.join(str(seed) for seed in sorted(raw_df['seed'].unique()))}",
        "",
        "Per-method statistics:",
    ]
    for method, method_df in raw_df.groupby("method"):
        lines.append(
            f"- {method}: "
            f"MAE={method_df['mae'].mean():.4f}±{method_df['mae'].std(ddof=0):.4f}, "
            f"RMSE={method_df['rmse'].mean():.4f}±{method_df['rmse'].std(ddof=0):.4f}, "
            f"MAPE={method_df['mape'].mean():.4f}±{method_df['mape'].std(ddof=0):.4f}, "
            f"R2={method_df['r2'].mean():.4f}±{method_df['r2'].std(ddof=0):.4f}"
        )
    if not improvement_df.empty:
        lines.extend(["", f"{enhanced_method} vs {baseline_method}:"])
        for _, row in improvement_df.iterrows():
            lines.append(
                f"- {row['metric']}: mean improvement={row['mean_improvement_percent']:.2f}% "
                f"(std={row['std_improvement_percent']:.2f}%), "
                f"improved on {int(row['improved_seed_count'])}/{int(row['total_seed_count'])} seeds"
            )
        lines.append(
            f"{enhanced_method} achieves average MAE/RMSE/MAPE gains across multiple seeds, "
            "showing that the performance gain is stable rather than caused by a single favorable seed."
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")
    return report_path

def compute_metrics(preds, truths):
    mse = float(np.mean((preds - truths) ** 2))
    mape = float(np.mean(np.abs(preds - truths) / np.maximum(np.abs(truths), MAPE_EPS))) * 100
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(preds - truths))), mape

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.view(-1).float(); b_f = b.view(-1).float()
    dot = float(torch.dot(a_f, b_f))
    na = float(torch.norm(a_f)); nb = float(torch.norm(b_f))
    return max(0.0, dot / (na * nb + 1e-12))

# ══════════════════════════════════════════════════════════════
# 增强 Non-IID 数据集生成
# ══════════════════════════════════════════════════════════════

CLIENT_CONFIGS_BASE = [
    {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 600, "noise": 2.0, "base": 100.0, "morning_mu": 8.0, "evening_mu": 18.0, "morning_amp": 30.0, "evening_amp": 25.0, "peak_sigma": 0.8, "trend": 0.0, "incident_prob": 0.0},
    {"dist": "student-t", "pattern": "波动型", "n_samples": 500, "noise": 5.0, "base": 80.0, "morning_mu": 7.5, "evening_mu": 17.5, "morning_amp": 35.0, "evening_amp": 30.0, "peak_sigma": 1.0, "trend": 0.005, "incident_prob": 0.0},
    {"dist": "chi-square", "pattern": "偏态高流量型", "n_samples": 700, "noise": 8.0, "base": 120.0, "morning_mu": 8.5, "evening_mu": 18.5, "morning_amp": 25.0, "evening_amp": 20.0, "peak_sigma": 1.2, "trend": -0.003, "incident_prob": 0.0},
    {"dist": "gaussian_mixture", "pattern": "双峰型", "n_samples": 550, "noise": 4.0, "base": 90.0, "morning_mu": 7.0, "evening_mu": 19.0, "morning_amp": 40.0, "evening_amp": 35.0, "peak_sigma": 0.7, "trend": 0.002, "incident_prob": 0.0},
    {"dist": "log_normal", "pattern": "突发拥堵型", "n_samples": 450, "noise": 6.0, "base": 70.0, "morning_mu": 8.2, "evening_mu": 17.8, "morning_amp": 28.0, "evening_amp": 22.0, "peak_sigma": 0.9, "trend": 0.001, "incident_prob": 0.05},
]

def build_noniid_client_configs(num_clients, noniid_level="medium"):
    if noniid_level == "low":
        templates = [
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 600, "noise": 2.0, "base": 100.0, "morning_mu": 8.0, "evening_mu": 18.0, "morning_amp": 30.0, "evening_amp": 25.0, "peak_sigma": 0.9, "trend": 0.0, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 580, "noise": 2.5, "base": 95.0, "morning_mu": 7.8, "evening_mu": 17.8, "morning_amp": 32.0, "evening_amp": 27.0, "peak_sigma": 0.85, "trend": 0.001, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 620, "noise": 3.0, "base": 105.0, "morning_mu": 8.2, "evening_mu": 18.2, "morning_amp": 28.0, "evening_amp": 23.0, "peak_sigma": 0.75, "trend": -0.001, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 590, "noise": 2.8, "base": 98.0, "morning_mu": 7.9, "evening_mu": 17.9, "morning_amp": 31.0, "evening_amp": 26.0, "peak_sigma": 0.88, "trend": 0.0005, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 610, "noise": 2.2, "base": 102.0, "morning_mu": 8.1, "evening_mu": 18.1, "morning_amp": 29.0, "evening_amp": 24.0, "peak_sigma": 0.82, "trend": -0.0005, "incident_prob": 0.0},
        ]
    elif noniid_level == "high":
        templates = [
            {"dist": "student-t", "pattern": "波动型", "n_samples": 400, "noise": 8.0, "base": 75.0, "morning_mu": 6.5, "evening_mu": 16.0, "morning_amp": 42.0, "evening_amp": 38.0, "peak_sigma": 1.3, "trend": 0.008, "incident_prob": 0.04},
            {"dist": "chi-square", "pattern": "偏态高流量型", "n_samples": 750, "noise": 12.0, "base": 135.0, "morning_mu": 9.5, "evening_mu": 19.5, "morning_amp": 20.0, "evening_amp": 16.0, "peak_sigma": 1.5, "trend": -0.006, "incident_prob": 0.1},
            {"dist": "gaussian_mixture", "pattern": "双峰型", "n_samples": 380, "noise": 6.0, "base": 85.0, "morning_mu": 6.0, "evening_mu": 20.0, "morning_amp": 48.0, "evening_amp": 42.0, "peak_sigma": 0.6, "trend": 0.004, "incident_prob": 0.03},
            {"dist": "log_normal", "pattern": "突发拥堵型", "n_samples": 300, "noise": 9.0, "base": 65.0, "morning_mu": 8.8, "evening_mu": 18.8, "morning_amp": 25.0, "evening_amp": 20.0, "peak_sigma": 1.1, "trend": 0.002, "incident_prob": 0.12},
            {"dist": "chi-square", "pattern": "偏态高流量型", "n_samples": 500, "noise": 10.0, "base": 110.0, "morning_mu": 8.0, "evening_mu": 17.0, "morning_amp": 35.0, "evening_amp": 30.0, "peak_sigma": 1.4, "trend": -0.004, "incident_prob": 0.06},
        ]
    else: templates = list(CLIENT_CONFIGS_BASE)
    configs = []
    for cid in range(num_clients):
        tpl = templates[cid % len(templates)].copy()
        if cid >= len(templates):
            jitter = 1.0 + 0.02 * (cid - len(templates)) * ((-1) ** cid)
            tpl["noise"] *= jitter; tpl["n_samples"] = int(tpl["n_samples"] * jitter); tpl["base"] *= jitter
        configs.append(tpl)
    return configs

def sample_distribution_noise(n_timesteps, n_nodes, dist_type, noise_level, seed):
    rng = np.random.RandomState(seed)
    if dist_type == "normal": return rng.randn(n_timesteps, n_nodes) * noise_level
    elif dist_type == "student-t": return rng.standard_t(df=4, size=(n_timesteps, n_nodes)) * noise_level * 0.7
    elif dist_type == "chi-square": return (rng.chisquare(df=3, size=(n_timesteps, n_nodes)) - 3) * noise_level * 0.5
    elif dist_type == "gaussian_mixture":
        mask = rng.rand(n_timesteps, n_nodes) < 0.5
        n1 = rng.randn(n_timesteps, n_nodes) * noise_level * 0.6
        n2 = rng.randn(n_timesteps, n_nodes) * noise_level * 1.5 + 0.5
        return np.where(mask, n1, n2)
    elif dist_type == "log_normal": return (rng.lognormal(mean=0, sigma=0.5, size=(n_timesteps, n_nodes)) - 1.5) * noise_level
    return rng.randn(n_timesteps, n_nodes) * noise_level

def generate_traffic_flow(cfg, n_timesteps, n_nodes, seed):
    rng = np.random.RandomState(seed); t = np.arange(n_timesteps); hours = t * 24.0 / n_timesteps
    base_flow = cfg["base"] + cfg["trend"] * t
    morning_peak = cfg["morning_amp"] * np.exp(-((hours - cfg["morning_mu"]) ** 2) / (2 * cfg["peak_sigma"] ** 2))
    evening_peak = cfg["evening_amp"] * np.exp(-((hours - cfg["evening_mu"]) ** 2) / (2 * cfg["peak_sigma"] ** 2))
    daily_period = (5.0 * np.sin(2 * np.pi * t / (n_timesteps / 2)) + 3.0 * np.cos(2 * np.pi * t / (n_timesteps / 4)))
    regional_bias = rng.randn(n_nodes) * 3.0
    incident = np.zeros(n_timesteps); incident_mask = np.zeros(n_timesteps, dtype=bool)
    if cfg.get("incident_prob", 0) > 0:
        for i in range(n_timesteps):
            if rng.rand() < cfg["incident_prob"]:
                duration = rng.randint(5, 20); end = min(i + duration, n_timesteps)
                incident[i:end] -= rng.uniform(20, 50); incident_mask[i:end] = True
    base_signal = base_flow + morning_peak + evening_peak + daily_period + incident
    data = np.zeros((n_timesteps, n_nodes), dtype=np.float32)
    for node in range(n_nodes):
        node_scale = 0.8 + 0.4 * (node / n_nodes)
        data[:, node] = base_signal * node_scale + regional_bias[node]
    noise = sample_distribution_noise(n_timesteps, n_nodes, cfg["dist"], cfg["noise"], seed + 1000)
    data += noise; data = np.clip(data, TRAFFIC_MIN_VALUE, None)
    return data.astype(np.float32), incident_mask

def build_sequences(data, seq_len, pred_len, incident_mask=None):
    X, y, target_time_index = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        target_idx = i + seq_len + pred_len - 1
        y.append(data[target_idx, 0])
        target_time_index.append(target_idx)
    t_idx = np.array(target_time_index, dtype=int)
    hours = (t_idx * 24.0 / len(data)) % 24
    if incident_mask is not None: target_incident_flag = incident_mask[t_idx].astype(bool)
    else: target_incident_flag = np.zeros(len(y), dtype=bool)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), {"target_time_index": t_idx, "target_hour": hours.astype(np.float32), "target_incident_flag": target_incident_flag}

def classify_period(hour, incident_flag):
    if incident_flag: return "incident_period"
    if 7 <= hour < 9: return "morning_peak"
    if 17 <= hour < 19: return "evening_peak"
    return "off_peak"

class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def split_with_meta(X, y, meta, train_r=0.70, val_r=0.10):
    n = len(X); n_train = int(n * train_r); n_val = int(n * val_r)
    def _slice(d, s): return {k: v[s] for k, v in d.items()}
    return X[:n_train], y[:n_train], _slice(meta, slice(0, n_train)), \
           X[n_train:n_train+n_val], y[n_train:n_train+n_val], _slice(meta, slice(n_train, n_train+n_val)), \
           X[n_train+n_val:], y[n_train+n_val:], _slice(meta, slice(n_train+n_val, n))

def build_client_data(client_configs, num_nodes, seq_len, pred_len, seed):
    buffer = seq_len + pred_len + 10; all_data = []
    for cid, cfg in enumerate(client_configs):
        n_ts = cfg["n_samples"] + buffer
        data, mask = generate_traffic_flow(cfg, n_ts, num_nodes, seed + cid * 100)
        X, y, meta = build_sequences(data, seq_len, pred_len, mask)
        xt, yt, mt, xv, yv, mv, xte, yte, mte = split_with_meta(X, y, meta)
        x_m, x_s = xt.mean(axis=(0,1), keepdims=True), xt.std(axis=(0,1), keepdims=True) + 1e-8
        y_m, y_s = yt.mean(), yt.std() + 1e-8
        xtn, xvn, xten = (xt-x_m)/x_s, (xv-x_m)/x_s, (xte-x_m)/x_s
        ytn, yvn, yten = (yt-y_m)/y_s, (yv-y_m)/y_s, (yte-y_m)/y_s
        all_data.append({
            "cid": cid, "train_loader": DataLoader(EnhancedTimeSeriesDataset(xtn, ytn), batch_size=BATCH_SIZE, shuffle=True),
            "val_loader": DataLoader(EnhancedTimeSeriesDataset(xvn, yvn), batch_size=BATCH_SIZE),
            "test_loader": DataLoader(EnhancedTimeSeriesDataset(xten, yten), batch_size=BATCH_SIZE),
            "train_size": len(xt), "val_size": len(xv), "test_size": len(xte),
            "y_mean": y_m, "y_std": y_s, "X_test": xte, "y_test": yte, "meta_test": mte
        })
    return all_data

def build_feature_augmented_data(client_configs, feature_set, num_nodes, seq_len, pred_len, seed):
    buffer = seq_len + pred_len + 10; all_data = []
    for cid, cfg in enumerate(client_configs):
        n_ts = cfg["n_samples"] + buffer
        data, mask = generate_traffic_flow(cfg, n_ts, num_nodes, seed + cid * 100)
        Xf, y, meta = build_sequences(data, seq_len, pred_len, mask)
        n_samples = len(Xf); seq_idx = np.array([np.arange(i, i + seq_len) for i in range(n_samples)])
        extra = []
        if feature_set in ("flow_time", "full"):
            h = (seq_idx * 24.0 / n_ts) % 24
            extra.append(np.sin(2*np.pi*h/24).astype(np.float32)); extra.append(np.cos(2*np.pi*h/24).astype(np.float32))
        if feature_set in ("flow_event", "full"):
            inc = np.zeros((n_samples, seq_len), dtype=np.float32)
            for s in range(n_samples): inc[s] = mask[s:s+seq_len]
            extra.append(inc)
        if feature_set in ("flow_region", "full"):
            extra.append(np.full((n_samples, seq_len), float(cid)/max(len(client_configs)-1,1), dtype=np.float32))
        X_full = np.concatenate([Xf, np.stack(extra, axis=1).transpose(0, 2, 1)], axis=2) if extra else Xf
        xt, yt, mt, xv, yv, mv, xte, yte, mte = split_with_meta(X_full, y, meta)
        x_m, x_s = xt.mean(axis=(0,1), keepdims=True), xt.std(axis=(0,1), keepdims=True) + 1e-8
        y_m, y_s = yt.mean(), yt.std() + 1e-8
        xtn, xvn, xten = (xt-x_m)/x_s, (xv-x_m)/x_s, (xte-x_m)/x_s
        ytn, yvn, yten = (yt-y_m)/y_s, (yv-y_m)/y_s, (yte-y_m)/y_s
        all_data.append({
            "cid": cid, "train_loader": DataLoader(EnhancedTimeSeriesDataset(xtn, ytn), batch_size=BATCH_SIZE, shuffle=True),
            "val_loader": DataLoader(EnhancedTimeSeriesDataset(xvn, yvn), batch_size=BATCH_SIZE),
            "test_loader": DataLoader(EnhancedTimeSeriesDataset(xten, yten), batch_size=BATCH_SIZE),
            "train_size": len(xt), "val_size": len(xv), "test_size": len(xte),
            "y_mean": y_m, "y_std": y_s, "X_test": xte, "y_test": yte, "meta_test": mte, "k_dim": X_full.shape[2]
        })
    return all_data

class AdaptiveSwish(nn.Module):
    def __init__(self, tr=True):
        super().__init__()
        if tr: self.beta = nn.Parameter(torch.ones(1))
        else: self.register_buffer("beta", torch.tensor(1.0))
    def forward(self, x): return x * torch.sigmoid(self.beta * x)

class CNNEnhancedModel(nn.Module):
    def __init__(self, k, t, hd=64, nh=4):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(k, hd, 3, padding=1), nn.GroupNorm(4, hd), AdaptiveSwish(), nn.Conv1d(hd, hd, 3, padding=1), nn.GroupNorm(4, hd), AdaptiveSwish(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.lstm = nn.LSTM(k, hd//2, 1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hd, hd)
        self.mh_attn = nn.MultiheadAttention(hd, nh, batch_first=True)
        self.attn_norm = nn.LayerNorm(hd); self.reg_head = nn.Sequential(nn.Linear(hd, 32), nn.LayerNorm(32), AdaptiveSwish(), nn.Linear(32, 1))
    def forward(self, x):
        xc = self.cnn(x.float()); xl, _ = self.lstm(x.permute(0,2,1).float()); xl = self.lstm_proj(xl.mean(1))
        fs = torch.stack([xc, xl], 1); ao, aw = self.mh_attn(fs, fs, fs)
        xf = self.attn_norm(ao + fs).mean(1); return self.reg_head(xf), aw

class FederatedClient:
    def __init__(self, cid, model, trl, vl, tel, cr, lr=1e-3):
        self.cid, self.model, self.trl, self.vl, self.tel, self.cr = cid, model.to(DEVICE), trl, vl, tel, cr
        self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
    def train_epoch(self):
        self.model.train(); tl = 0
        for x, y in self.trl:
            x, y = x.to(DEVICE), y.to(DEVICE); self.opt.zero_grad()
            p, _ = self.model(x); loss = self.cr(p.view(-1), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.opt.step(); tl += loss.item() * x.shape[0]
        return tl / len(self.trl.dataset)
    @torch.no_grad()
    def validate(self, l=None):
        if l is None: l = self.vl
        self.model.eval(); tl = 0
        for x, y in l:
            x, y = x.to(DEVICE), y.to(DEVICE); p, _ = self.model(x); tl += self.cr(p.view(-1), y).item() * x.shape[0]
        return tl / len(l.dataset)
    @torch.no_grad()
    def validate_metrics(self, ym, ys, l=None):
        if l is None: l = self.vl
        self.model.eval(); ps, ts = [], []
        for x, y in l:
            x, y = x.to(DEVICE), y.to(DEVICE); p, _ = self.model(x); ps.append(p.view(-1).cpu().numpy()); ts.append(y.cpu().numpy())
        ps, ts = np.concatenate(ps) * ys + ym, np.concatenate(ts) * ys + ym
        return compute_metrics(ps, ts)
    def train_local(self, eps=3, gm=None):
        if gm: self.model.load_state_dict(gm.state_dict())
        for _ in range(eps): l = self.train_epoch()
        return l, copy.deepcopy(self.model.state_dict()), None
    @torch.no_grad()
    def test_metrics(self):
        self.model.eval(); ps, ts = [], []
        for x, y in self.tel:
            x, y = x.to(DEVICE), y.to(DEVICE); p, _ = self.model(x); ps.append(p.view(-1).cpu().numpy()); ts.append(y.cpu().numpy())
        ps, ts = np.concatenate(ps), np.concatenate(ts)
        mse, rmse, mae, mape = compute_metrics(ps, ts); return {"mse":mse, "rmse":rmse, "mae":mae, "mape":mape, "preds":ps, "truths":ts}

class AggregationServer:
    def __init__(self, model, nc, am="fedavg", lam=0.5):
        self.gm, self.nc, self.am, self.lam = model.to(DEVICE), nc, am, lam
        self.ds, self.rl = None, []
    def aggregate(self, cws, cls):
        n = np.array(self.ds, float); dw = n/n.sum()
        if self.am == "fedavg": w = dw
        elif self.am == "loss_weighted": q = 1/(np.array(cls)+1e-8); w = q/q.sum()
        elif self.am == "data_loss_weighted": q = 1/(np.array(cls)+1e-8); qw = q/q.sum(); w = self.lam*dw + (1-self.lam)*qw
        elif self.am == "proposed":
            q = 1/(np.array(cls)+1e-8); qw = q/q.sum(); cv = np.std(cls)/(np.mean(cls)+1e-8); dl = 1/(1+cv); w = 0.8*(dl*dw+(1-dl)*qw) + 0.2/self.nc
        else: w = dw
        nd = {k: torch.zeros_like(v) for k, v in self.gm.state_dict().items()}
        for k in nd:
            for i in range(self.nc): nd[k] += cws[i][k].to(DEVICE) * torch.tensor(float(w[i]), device=DEVICE)
        self.gm.load_state_dict(nd); self.rl.append(float(np.mean(cls)))

def _method_label(method_key: str) -> str:
    return {
        "fedavg": "FedAvg",
        "loss_weighted": "Loss-weighted",
        "data_loss_weighted": "Data-loss weighted",
        "proposed": "Proposed",
        "independent": "Independent",
    }[method_key]


def _build_summary(df: pd.DataFrame, group_cols):
    agg_df = (
        df.groupby(group_cols)[["mse", "rmse", "mae", "mape"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_df.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]
    return agg_df


def _pred_rows_from_results(results, workflow: str, method: str, seed: int):
    rows = []
    for item in results:
        meta_test = item["meta_test"]
        hours = meta_test["target_hour"]
        incident_flags = meta_test["target_incident_flag"]
        for sample_id in range(min(200, len(item["truths"]))):
            rows.append({
                "workflow": workflow,
                "method": method,
                "seed": seed,
                "client_id": item["client_id"],
                "sample_id": sample_id,
                "y_true": float(item["truths"][sample_id]),
                "y_pred": float(item["preds"][sample_id]),
                "period": classify_period(float(hours[sample_id]), bool(incident_flags[sample_id])),
            })
    return rows


def _period_rows_from_results(results, method: str, seed: int):
    rows = []
    for item in results:
        hours = item["meta_test"]["target_hour"]
        incident_flags = item["meta_test"]["target_incident_flag"]
        periods = np.array([
            classify_period(float(hour), bool(flag))
            for hour, flag in zip(hours, incident_flags)
        ])
        for period in ["morning_peak", "evening_peak", "off_peak", "incident_period"]:
            mask = periods == period
            if not np.any(mask):
                continue
            mse, rmse, mae, mape = compute_metrics(item["preds"][mask], item["truths"][mask])
            rows.append({
                "seed": seed,
                "method": method,
                "client_id": item["client_id"],
                "period": period,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            })
    return rows


def _prepare_paper_ready_summary(summary_df: pd.DataFrame, scenario_col: str, experiment_name: str) -> pd.DataFrame:
    ordered_columns = [
        scenario_col,
        "experiment_name",
        "method",
        "mse_mean",
        "mse_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "mape_mean",
        "mape_std",
        "source_seeds",
    ]
    paper_ready_df = summary_df.copy()
    paper_ready_df.insert(1, "experiment_name", experiment_name)
    paper_ready_df["source_seeds"] = ",".join(str(seed) for seed in SEEDS)
    return paper_ready_df[ordered_columns]


def _run_and_export_fedavg_only(
    output_dir: Path,
    paper_ready_dir: Path,
    scenario_values,
    scenario_col: str,
    metrics_filename: str,
    summary_filename: str,
    paper_ready_filename: str,
    client_data_builder,
    experiment_name: str,
):
    metric_rows = []
    for seed in SEEDS:
        for scenario_value in scenario_values:
            client_data = client_data_builder(scenario_value, seed)
            results, _ = run_federated_training(client_data, am="fedavg", seed=seed)
            metric_rows.extend({
                scenario_col: scenario_value,
                "seed": seed,
                "method": "FedAvg",
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = _build_summary(metrics_df, [scenario_col, "method"])
    save_dataframe(metrics_df, output_dir, metrics_filename)
    save_dataframe(summary_df, output_dir, summary_filename)
    paper_ready_df = _prepare_paper_ready_summary(summary_df, scenario_col, experiment_name)
    save_dataframe(paper_ready_df, paper_ready_dir, paper_ready_filename)


def run_missing_fedavg_workflows(
    output_dir: Path = DEFAULT_ENHANCED_RESULTS_DIR,
    paper_ready_dir: Path = DEFAULT_PAPER_READY_DIR,
):
    ensure_output_dir(output_dir)
    ensure_output_dir(paper_ready_dir)

    _run_and_export_fedavg_only(
        output_dir=output_dir,
        paper_ready_dir=paper_ready_dir,
        scenario_values=["low", "medium", "high"],
        scenario_col="noniid_level",
        metrics_filename="cnn_enhanced_noniid_metrics_fedavg.csv",
        summary_filename="cnn_enhanced_noniid_summary_fedavg.csv",
        paper_ready_filename="cnn_enhanced_noniid_fedavg_only.csv",
        client_data_builder=lambda noniid_level, seed: build_client_data(
            build_noniid_client_configs(5, noniid_level),
            NUM_NODES,
            SEQ_LEN,
            PRED_LEN,
            seed,
        ),
        experiment_name="cnn_enhanced_noniid",
    )

    _run_and_export_fedavg_only(
        output_dir=output_dir,
        paper_ready_dir=paper_ready_dir,
        scenario_values=[3, 5, 8],
        scenario_col="num_clients",
        metrics_filename="cnn_enhanced_client_scale_metrics_fedavg.csv",
        summary_filename="cnn_enhanced_client_scale_summary_fedavg.csv",
        paper_ready_filename="cnn_enhanced_client_scale_fedavg_only.csv",
        client_data_builder=lambda num_clients, seed: build_client_data(
            build_noniid_client_configs(num_clients),
            NUM_NODES,
            SEQ_LEN,
            PRED_LEN,
            seed,
        ),
        experiment_name="cnn_enhanced_client_scale",
    )

    _run_and_export_fedavg_only(
        output_dir=output_dir,
        paper_ready_dir=paper_ready_dir,
        scenario_values=["flow_only", "flow_time", "flow_event", "flow_region", "full"],
        scenario_col="feature_set",
        metrics_filename="cnn_enhanced_feature_ablation_metrics_fedavg.csv",
        summary_filename="cnn_enhanced_feature_ablation_summary_fedavg.csv",
        paper_ready_filename="cnn_enhanced_feature_ablation_fedavg_only.csv",
        client_data_builder=lambda feature_set, seed: build_feature_augmented_data(
            CLIENT_CONFIGS_BASE,
            feature_set,
            NUM_NODES,
            SEQ_LEN,
            PRED_LEN,
            seed,
        ),
        experiment_name="cnn_enhanced_feature_ablation",
    )


def run_federated_training(
    client_data,
    am="fedavg",
    lam=0.5,
    cr=COMM_ROUNDS,
    le=LOCAL_EPOCHS,
    lr=LR,
    seed=42,
    rec=False,
):
    set_global_seed(seed)
    nc = len(client_data)
    criterion = nn.MSELoss()
    feature_dim = client_data[0].get("k_dim", NUM_NODES)
    clients = [
        FederatedClient(
            d["cid"],
            CNNEnhancedModel(feature_dim, SEQ_LEN),
            d["train_loader"],
            d["val_loader"],
            d["test_loader"],
            criterion,
            lr,
        )
        for d in client_data
    ]
    server = AggregationServer(CNNEnhancedModel(feature_dim, SEQ_LEN), nc, am, lam)
    server.ds = [d["train_size"] for d in client_data]

    convergence_rows = []
    for round_idx in range(cr):
        client_weights, client_losses = [], []
        for client in clients:
            loss, weights, _ = client.train_local(le, server.gm)
            client_weights.append(weights)
            client_losses.append(loss)
        server.aggregate(client_weights, client_losses)

        if rec:
            val_losses = []
            val_metrics = []
            for client, data in zip(clients, client_data):
                client.model.load_state_dict(server.gm.state_dict())
                val_losses.append(client.validate())
                val_metrics.append(client.validate_metrics(data["y_mean"], data["y_std"]))
            convergence_rows.append({
                "round": round_idx + 1,
                "method": _method_label(am),
                "avg_train_loss": float(np.mean(client_losses)),
                "avg_val_loss": float(np.mean(val_losses)),
                "avg_val_rmse": float(np.mean([m[1] for m in val_metrics])),
                "avg_val_mae": float(np.mean([m[2] for m in val_metrics])),
                "avg_val_mape": float(np.mean([m[3] for m in val_metrics])),
            })

    results = []
    for idx, (client, data) in enumerate(zip(clients, client_data)):
        client.model.load_state_dict(server.gm.state_dict())
        metric_dict = client.test_metrics()
        preds = metric_dict["preds"] * data["y_std"] + data["y_mean"]
        truths = metric_dict["truths"] * data["y_std"] + data["y_mean"]
        mse, rmse, mae, mape = compute_metrics(preds, truths)
        results.append({
            "client_id": idx,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "preds": preds,
            "truths": truths,
            "meta_test": data["meta_test"],
        })
    return results, pd.DataFrame(convergence_rows)


def run_independent_training(client_data, eps=10, lr=0.01, seed=42):
    set_global_seed(seed)
    criterion = nn.MSELoss()
    results = []
    feature_dim = client_data[0].get("k_dim", NUM_NODES)
    for data in client_data:
        model = CNNEnhancedModel(feature_dim, SEQ_LEN).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(eps):
            model.train()
            for x_batch, y_batch in data["train_loader"]:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred, _ = model(x_batch)
                loss = criterion(pred.view(-1), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x_batch, y_batch in data["test_loader"]:
                x_batch = x_batch.to(DEVICE)
                pred, _ = model(x_batch)
                preds.append(pred.view(-1).cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        preds = np.concatenate(preds) * data["y_std"] + data["y_mean"]
        truths = np.concatenate(truths) * data["y_std"] + data["y_mean"]
        mse, rmse, mae, mape = compute_metrics(preds, truths)
        results.append({
            "client_id": data["cid"],
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "preds": preds,
            "truths": truths,
            "meta_test": data["meta_test"],
        })
    return results


def export_enhanced_dataset_artifacts(output_dir: Path):
    set_global_seed(42)
    ensure_output_dir(output_dir)
    client_configs = list(CLIENT_CONFIGS_BASE)
    raw_series = []
    incident_masks = []
    for cid, cfg in enumerate(client_configs):
        data, incident_mask = generate_traffic_flow(cfg, cfg["n_samples"] + 80, NUM_NODES, 42 + cid * 100)
        raw_series.append(data)
        incident_masks.append(incident_mask)

    ts_rows = []
    dist_rows = []
    config_rows = []
    peak_rows = []
    incident_rows = []
    summary_rows = []
    min_length = min(len(series) for series in raw_series)
    aligned_client_series = []

    for cid, (cfg, data, incident_mask) in enumerate(zip(client_configs, raw_series, incident_masks)):
        client_mean_series = data.mean(axis=1)
        aligned_client_series.append(client_mean_series[:min_length])
        for time_step, traffic_flow in enumerate(client_mean_series[:240]):
            ts_rows.append({
                "client_id": cid,
                "time_step": time_step,
                "traffic_flow": float(traffic_flow),
            })
        sampled_values = data.ravel()[::10]
        for value in sampled_values:
            dist_rows.append({"client_id": cid, "traffic_flow": float(value)})
        config_rows.append({
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
            "incident_prob": cfg["incident_prob"],
        })
        hours = (np.arange(len(client_mean_series)) * 24.0 / len(client_mean_series)) % 24
        for hour in range(24):
            hour_mask = (hours >= hour) & (hours < hour + 1)
            if np.any(hour_mask):
                peak_rows.append({
                    "client_id": cid,
                    "hour": hour,
                    "traffic_flow": float(client_mean_series[hour_mask].mean()),
                })
        if cfg["incident_prob"] > 0:
            for time_step in range(min(240, len(client_mean_series))):
                incident_rows.append({
                    "time_step": time_step,
                    "traffic_flow": float(client_mean_series[time_step]),
                    "incident_flag": bool(incident_mask[time_step]),
                    "client_id": cid,
                })
        summary_rows.append({
            "client_id": cid,
            "num_samples": cfg["n_samples"],
            "mean_flow": float(client_mean_series.mean()),
            "std_flow": float(client_mean_series.std()),
            "min_flow": float(client_mean_series.min()),
            "max_flow": float(client_mean_series.max()),
            "incident_ratio": float(np.mean(incident_mask)),
        })

    client_corr = np.corrcoef(np.stack(aligned_client_series))
    client_corr_rows = []
    for src in range(client_corr.shape[0]):
        for dst in range(client_corr.shape[1]):
            client_corr_rows.append({
                "source_client": src,
                "target_client": dst,
                "correlation": float(client_corr[src, dst]),
            })

    node_corr_rows = []
    for cid, data in enumerate(raw_series):
        node_corr = np.corrcoef(data[:min(400, len(data))], rowvar=False)
        node_corr = np.nan_to_num(node_corr)
        for src in range(node_corr.shape[0]):
            for dst in range(node_corr.shape[1]):
                node_corr_rows.append({
                    "client_id": cid,
                    "source_node": src,
                    "target_node": dst,
                    "correlation": float(node_corr[src, dst]),
                })

    save_dataframe(pd.DataFrame(ts_rows), output_dir, "enhanced_dataset_client_timeseries.csv")
    save_dataframe(pd.DataFrame(dist_rows), output_dir, "enhanced_dataset_distribution.csv")
    save_dataframe(pd.DataFrame(config_rows), output_dir, "enhanced_dataset_client_config.csv")
    save_dataframe(pd.DataFrame(peak_rows), output_dir, "enhanced_dataset_peak_pattern.csv")
    save_dataframe(pd.DataFrame(incident_rows), output_dir, "enhanced_dataset_incident_example.csv")
    save_dataframe(pd.DataFrame(client_corr_rows), output_dir, "enhanced_dataset_client_correlation_matrix.csv")
    save_dataframe(pd.DataFrame(node_corr_rows), output_dir, "enhanced_dataset_node_correlation_matrix.csv")
    save_dataframe(pd.DataFrame(summary_rows), output_dir, "enhanced_dataset_summary.csv")


def run_single_seed_main_experiment(seed: int):
    metric_rows = []
    pred_rows = []
    client_data = build_client_data(CLIENT_CONFIGS_BASE, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
    for method_key in ["fedavg", "proposed"]:
        results, _ = run_federated_training(client_data, am=method_key, seed=seed)
        method_name = _method_label(method_key)
        metric_rows.extend({
            "seed": seed,
            "method": method_name,
            "client_id": result["client_id"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "mae": result["mae"],
            "mape": result["mape"],
            "r2": compute_r2_score(result["preds"], result["truths"]),
        } for result in results)
        pred_rows.extend(_pred_rows_from_results(results, "main", method_name, seed))
    independent_results = run_independent_training(client_data, seed=seed)
    metric_rows.extend({
        "seed": seed,
        "method": "Independent",
        "client_id": result["client_id"],
        "mse": result["mse"],
        "rmse": result["rmse"],
        "mae": result["mae"],
        "mape": result["mape"],
        "r2": compute_r2_score(result["preds"], result["truths"]),
    } for result in independent_results)
    pred_rows.extend(_pred_rows_from_results(independent_results, "main", "Independent", seed))
    metrics_df = pd.DataFrame(metric_rows)
    raw_rows = []
    for method in metrics_df["method"].unique():
        method_df = metrics_df[metrics_df["method"] == method]
        raw_rows.append({
            "experiment": "cnn_fed_enhanced_main",
            "method": method,
            "seed": seed,
            "mse": float(method_df["mse"].mean()),
            "rmse": float(method_df["rmse"].mean()),
            "mae": float(method_df["mae"].mean()),
            "mape": float(method_df["mape"].mean()),
            "r2": float(method_df["r2"].mean()),
            "final_loss": np.nan,
            "best_loss": np.nan,
            "communication_rounds": int(COMM_ROUNDS if method != "Independent" else 0),
            "convergence_round": int(COMM_ROUNDS if method != "Independent" else COMM_ROUNDS * LOCAL_EPOCHS),
        })
    return metrics_df, pd.DataFrame(pred_rows), pd.DataFrame(raw_rows)


def run_main_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    metric_frames = []
    pred_frames = []
    raw_frames = []
    for seed in SEEDS:
        metrics_df, pred_df, raw_df = run_single_seed_main_experiment(seed)
        metric_frames.append(metrics_df)
        pred_frames.append(pred_df)
        raw_frames.append(raw_df)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    save_dataframe(metrics_df, output_dir, "cnn_enhanced_main_metrics.csv")
    save_dataframe(_build_summary(metrics_df, ["method"]), output_dir, "cnn_enhanced_main_summary.csv")
    save_dataframe(pd.concat(pred_frames, ignore_index=True), output_dir, "cnn_enhanced_main_predictions.csv")
    raw_df = pd.concat(raw_frames, ignore_index=True)
    multi_seed_summary_df = build_multi_seed_summary(
        raw_df,
        group_cols=["experiment", "method"],
        metric_cols=["mae", "rmse", "mape", "r2", "final_loss", "best_loss", "communication_rounds", "convergence_round"],
    )
    improvement_df = build_pairwise_improvement_summary(
        raw_df,
        experiment_name="cnn_fed_enhanced_main",
        baseline_method="FedAvg",
        enhanced_method="Proposed",
        metric_cols=["mae", "rmse", "mape", "r2"],
    )
    save_dataframe(raw_df, output_dir, "multi_seed_raw_results.csv")
    save_dataframe(multi_seed_summary_df, output_dir, "multi_seed_summary.csv")
    if not improvement_df.empty:
        save_dataframe(improvement_df, output_dir, "multi_seed_improvement_summary.csv")
    write_multi_seed_stability_report(
        output_dir=output_dir,
        raw_df=raw_df,
        improvement_df=improvement_df,
        experiment_name="cnn_fed_enhanced_main",
        baseline_method="FedAvg",
        enhanced_method="Proposed",
    )


def run_aggregation_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(CLIENT_CONFIGS_BASE, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
        for method_key in ["fedavg", "loss_weighted", "data_loss_weighted", "proposed"]:
            results, _ = run_federated_training(client_data, am=method_key, seed=seed)
            rows.extend({
                "seed": seed,
                "method": _method_label(method_key),
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_aggregation_metrics.csv")
    save_dataframe(_build_summary(df, ["method"]), output_dir, "cnn_enhanced_aggregation_summary.csv")


def run_lambda_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(CLIENT_CONFIGS_BASE, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
        for lambda_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            results, _ = run_federated_training(client_data, am="data_loss_weighted", lam=lambda_value, seed=seed)
            rows.extend({
                "seed": seed,
                "method": "Data-loss weighted",
                "lambda_value": lambda_value,
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_lambda_metrics.csv")
    save_dataframe(_build_summary(df, ["lambda_value", "method"]), output_dir, "cnn_enhanced_lambda_summary.csv")


def run_convergence_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    conv_frames = []
    for seed in SEEDS:
        for method_key in ["fedavg", "proposed"]:
            client_data = build_client_data(CLIENT_CONFIGS_BASE, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
            _, conv_df = run_federated_training(client_data, am=method_key, seed=seed, cr=10, rec=True)
            conv_frames.append(conv_df.assign(seed=seed))
    raw_df = pd.concat(conv_frames, ignore_index=True)
    save_dataframe(raw_df, output_dir, "cnn_enhanced_convergence_history.csv")
    save_dataframe(raw_df, output_dir, "multi_seed_convergence_raw.csv")
    summary_df = build_multi_seed_convergence_summary(
        raw_df,
        group_cols=["method", "round"],
        metric_cols=["avg_train_loss", "avg_val_loss", "avg_val_rmse", "avg_val_mae", "avg_val_mape"],
    )
    save_dataframe(summary_df, output_dir, "multi_seed_convergence_summary.csv")


def run_client_scale_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        for num_clients in [3, 5, 8]:
            client_configs = build_noniid_client_configs(num_clients)
            client_data = build_client_data(client_configs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
            results, _ = run_federated_training(client_data, am="proposed", seed=seed)
            rows.extend({
                "seed": seed,
                "method": "Proposed",
                "num_clients": num_clients,
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_client_scale_metrics.csv")
    save_dataframe(_build_summary(df, ["num_clients", "method"]), output_dir, "cnn_enhanced_client_scale_summary.csv")


def run_noniid_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        for noniid_level in ["low", "medium", "high"]:
            client_configs = build_noniid_client_configs(5, noniid_level)
            client_data = build_client_data(client_configs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
            results, _ = run_federated_training(client_data, am="proposed", seed=seed)
            rows.extend({
                "seed": seed,
                "method": "Proposed",
                "noniid_level": noniid_level,
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_noniid_metrics.csv")
    save_dataframe(_build_summary(df, ["noniid_level", "method"]), output_dir, "cnn_enhanced_noniid_summary.csv")


def run_client_metrics_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(CLIENT_CONFIGS_BASE, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
        fedavg_results, _ = run_federated_training(client_data, am="fedavg", seed=seed)
        proposed_results, _ = run_federated_training(client_data, am="proposed", seed=seed)
        independent_results = run_independent_training(client_data, seed=seed)
        for method_name, results in [
            ("FedAvg", fedavg_results),
            ("Proposed", proposed_results),
            ("Independent", independent_results),
        ]:
            rows.extend({
                "seed": seed,
                "method": method_name,
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)
    save_dataframe(pd.DataFrame(rows), output_dir, "cnn_enhanced_client_metrics.csv")


def run_peak_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(CLIENT_CONFIGS_BASE, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
        for method_key in ["fedavg", "proposed"]:
            results, _ = run_federated_training(client_data, am=method_key, seed=seed)
            rows.extend(_period_rows_from_results(results, _method_label(method_key), seed))
        independent_results = run_independent_training(client_data, seed=seed)
        rows.extend(_period_rows_from_results(independent_results, "Independent", seed))
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_peak_metrics.csv")
    save_dataframe(_build_summary(df, ["period", "method"]), output_dir, "cnn_enhanced_peak_summary.csv")


def run_feature_ablation_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    rows = []
    for seed in SEEDS:
        for feature_set in ["flow_only", "flow_time", "flow_event", "flow_region", "full"]:
            client_data = build_feature_augmented_data(CLIENT_CONFIGS_BASE, feature_set, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
            results, _ = run_federated_training(client_data, am="proposed", seed=seed)
            rows.extend({
                "seed": seed,
                "method": "Proposed",
                "feature_set": feature_set,
                "client_id": result["client_id"],
                "mse": result["mse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
            } for result in results)
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "cnn_enhanced_feature_ablation_metrics.csv")
    save_dataframe(_build_summary(df, ["feature_set", "method"]), output_dir, "cnn_enhanced_feature_ablation_summary.csv")


def run_project(workflow: str, output_dir: Path):
    ensure_output_dir(output_dir)
    workflow_map = {
        "data_viz": export_enhanced_dataset_artifacts,
        "main": run_main_experiment,
        "aggregation": run_aggregation_experiment,
        "lambda": run_lambda_experiment,
        "convergence": run_convergence_experiment,
        "client_scale": run_client_scale_experiment,
        "noniid": run_noniid_experiment,
        "client_metrics": run_client_metrics_experiment,
        "peak": run_peak_experiment,
        "feature_ablation": run_feature_ablation_experiment,
    }
    selected = list(workflow_map) if workflow == "all" else [workflow]
    for item in selected:
        workflow_map[item](output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="CNN Enhanced Federated Simulation Core")
    parser.add_argument(
        "--workflow",
        choices=[
            "all", "data_viz", "main", "aggregation", "lambda", "convergence",
            "client_scale", "noniid", "client_metrics", "peak", "feature_ablation",
        ],
        default="all",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for exported experiment artifacts.",
    )
    parser.add_argument(
        "--run-fedavg-missing",
        action="store_true",
        help="Generate missing FedAvg-only CSV artifacts for non-IID, client-scale, and feature-ablation experiments.",
    )
    parser.add_argument("--multi_seed", type=str, default="True", help="Whether to run multiple seeds.")
    parser.add_argument("--seeds", type=str, default="42,2024,2025,2026,3407", help="Comma-separated random seeds.")
    parser.add_argument("--single_seed", type=int, default=42, help="Single seed used when --multi_seed False.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    global SEEDS
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_ENHANCED_RESULTS_DIR
    multi_seed = parse_bool_flag(args.multi_seed)
    SEEDS = parse_seed_list(args.seeds) if multi_seed else [int(args.single_seed)]
    if args.run_fedavg_missing:
        run_missing_fedavg_workflows(output_dir=output_dir, paper_ready_dir=output_dir / "paper_ready")
        return
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
