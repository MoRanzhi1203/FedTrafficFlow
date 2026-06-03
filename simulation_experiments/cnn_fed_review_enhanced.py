# -*- coding: utf-8 -*-
"""
CNN 联邦仿真增强实验工程。

本文件从 cnn_fed_base.py 结构派生，在保留基础模型、FedAvgServer、
FederatedClient 等核心组件的基础上，扩展以下增强实验：

1. main：复杂 Non-IID 交通流主实验；
2. client_scale：客户端数量实验 (3, 5, 8)；
3. noniid：Non-IID 强度实验 (low, medium, high)；
4. convergence：收敛性分析（全局+客户端本地 loss）；
5. client_metrics：客户端级误差分析；
6. peak：高峰/平峰误差分析；
7. all：依次执行全部增强实验。
"""

import argparse
import copy
import os
import random
import sys
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

plt.ioff()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_NAME = "cnn-fed-review-enhanced"
DEFAULT_OUTPUT_DIR = SIMULATION_RESULTS_ROOT / PROJECT_NAME
INVALID_PATH_CHARS = set('<>:"|?*')


def configure_plot_style() -> None:
    sns.set_theme(
        style="whitegrid", context="notebook", font="DejaVu Sans",
        rc={"axes.unicode_minus": False, "figure.titlesize": 18,
            "axes.titlesize": 16, "axes.labelsize": 13,
            "xtick.labelsize": 11, "ytick.labelsize": 11,
            "legend.fontsize": 11, "legend.title_fontsize": 12},
    )


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


def validate_output_subdir(output_subdir: str) -> None:
    if not output_subdir or not output_subdir.strip():
        raise ValueError("The output directory must not be empty.")
    candidate = Path(output_subdir)
    if candidate.is_absolute():
        raise ValueError("Only relative output subdirectories under results/simulation_experiments are allowed.")
    for part in candidate.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("Path traversal is not allowed in the output directory.")
        if any(char in INVALID_PATH_CHARS for char in part):
            raise ValueError("The output directory contains illegal characters.")
        if part.endswith(" ") or part.endswith("."):
            raise ValueError("Directory names must not end with a space or dot.")


def resolve_output_dir(output_subdir: Optional[str] = None) -> Path:
    ensure_output_dir(SIMULATION_RESULTS_ROOT)
    relative_subdir = output_subdir or PROJECT_NAME
    validate_output_subdir(relative_subdir)
    resolved_output_dir = (SIMULATION_RESULTS_ROOT / relative_subdir).resolve()
    resolved_root = SIMULATION_RESULTS_ROOT.resolve()
    if resolved_output_dir != resolved_root and resolved_root not in resolved_output_dir.parents:
        raise ValueError("The resolved output directory escapes the simulation results root.")
    return ensure_output_dir(resolved_output_dir)


def save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    output_path = ensure_output_dir(output_dir) / file_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path


def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    output_path = ensure_output_dir(output_dir) / file_name
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved table: {output_path}")
    return output_path


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def unpack_model_output(model_output):
    if isinstance(model_output, tuple):
        return model_output
    return model_output, None


class AdaptiveSwish(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# ================================================================
# 模型定义（从 cnn_fed_base.py 保留）
# ================================================================

class CCNOverviewModel(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=k, hidden_size=hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1))

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


class CCNAblationFull(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=k, hidden_size=hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.lstm_proj(x_lstm.mean(dim=1))
        feat_seq = torch.stack([x_cnn, x_lstm], dim=1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


class CCNAblationCNNLSTM(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=k, hidden_size=hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), AdaptiveSwish())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.lstm_proj(x_lstm.mean(dim=1))
        fused = self.fuse(torch.cat([x_cnn, x_lstm], dim=1))
        return self.head(fused), None


class LSTMAttentionHetero(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=k, hidden_size=hidden_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.lstm_proj(x_lstm.mean(dim=1))
        feat_seq = x_lstm.unsqueeze(1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


class CCNAblationCNNAttention(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim), AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        feat_seq = x_cnn.unsqueeze(1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


# ================================================================
# 联邦组件（从 cnn_fed_base.py 保留）
# ================================================================

class FederatedClient:
    def __init__(self, client_id, model, train_loader, test_loader, criterion, lr=1e-3):
        self.client_id = client_id
        self.model = model.to(DEVICE).float()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.9)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float().view(-1)
            self.optimizer.zero_grad()
            pred, _ = unpack_model_output(self.model(x))
            loss = self.criterion(pred.view(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        avg_loss = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float().view(-1)
            pred, _ = unpack_model_output(self.model(x))
            total_loss += self.criterion(pred.view(-1), y).item() * x.shape[0]
        avg_loss = total_loss / len(self.test_loader.dataset)
        self.val_losses.append(avg_loss)
        self.scheduler.step()
        return avg_loss

    def train_local(self, epochs=5, global_model=None, verbose=False, prefix="Local"):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        epoch_losses = []
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_losses.append((train_loss, val_loss))
            if verbose:
                print(f"  {prefix} epoch {epoch + 1}/{epochs}, "
                      f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
        return float(self.train_losses[-1]), copy.deepcopy(self.model.state_dict()), epoch_losses

    @torch.no_grad()
    def test_metrics(self):
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float().view(-1)
            pred, _ = unpack_model_output(self.model(x))
            preds.append(pred.detach().view(-1))
            truths.append(y.view(-1))
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        diff = preds - truths
        mse = float((diff ** 2).mean().item())
        mae = float(diff.abs().mean().item())
        rmse = float(np.sqrt(mse))
        return {"mse": mse, "rmse": rmse, "mae": mae}

    @torch.no_grad()
    def test_predictions(self):
        self.model.eval()
        ap, at = [], []
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = unpack_model_output(self.model(x))
            ap.append(pred.detach().cpu().numpy().reshape(-1))
            at.append(y.cpu().numpy().reshape(-1))
        return np.concatenate(ap), np.concatenate(at)


class IndependentClient(FederatedClient):
    def __init__(self, client_id, model, train_loader, test_loader, criterion):
        super().__init__(client_id, model, train_loader, test_loader, criterion, lr=0.02)

    def train_local(self, epochs=2, verbose=False):
        loss, state, eps = super().train_local(epochs=epochs, global_model=None,
                                                verbose=verbose, prefix="Independent")
        return loss, state


class FedAvgServer:
    """标准样本量加权 FedAvg 服务端。

    聚合公式：
        global_model = sum(n_i / total_n * local_model_i)
    其中 n_i 为客户端 i 的训练样本量，total_n 为所有客户端训练样本量总和。
    """

    def __init__(self, model, num_clients: int):
        self.global_model = model.to(DEVICE).float()
        self.num_clients = num_clients
        self.round_losses = []
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = sizes

    def aggregate(self, client_weights, client_losses):
        total_n = float(sum(self.client_data_sizes))
        weights = np.array(self.client_data_sizes) / total_n

        global_dict = self.global_model.state_dict()
        new_dict = {
            key: torch.zeros_like(value, dtype=torch.float32)
            for key, value in global_dict.items()
        }

        for key in new_dict.keys():
            for idx in range(self.num_clients):
                client_weight = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                new_dict[key] += client_weight * torch.tensor(
                    float(weights[idx]), device=DEVICE, dtype=torch.float32)

        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(client_losses)))
        return self.global_model.state_dict()


# ================================================================
# 增强版：复杂 Non-IID 交通流数据生成
# ================================================================

DIST_TYPES = ["normal", "student-t", "chi-square", "gaussian_mixture", "log_normal"]
TRAFFIC_PATTERNS = ["平稳通勤型", "波动型", "偏态高流量型", "双峰型", "突发拥堵型"]

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


def sample_distribution_noise(n_timesteps, n_nodes, dist_type, noise_level, seed):
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

    公式：traffic(t) = base + morning_peak + evening_peak + daily_period
                     + trend + regional_bias + incident + noise
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_timesteps)
    hours = t * 24.0 / n_timesteps

    base_flow = cfg["base"] + cfg["trend"] * t

    morning_peak = cfg["morning_amp"] * np.exp(
        -((hours - cfg["morning_mu"]) ** 2) / (2 * cfg["peak_sigma"] ** 2))
    evening_peak = cfg["evening_amp"] * np.exp(
        -((hours - cfg["evening_mu"]) ** 2) / (2 * cfg["peak_sigma"] ** 2))

    daily_period = 5.0 * np.sin(2 * np.pi * t / (n_timesteps / 2)) + \
                   3.0 * np.cos(2 * np.pi * t / (n_timesteps / 4))

    regional_bias = rng.randn(n_nodes) * 3.0

    incident = np.zeros(n_timesteps)
    if cfg["incident_prob"] > 0:
        for i in range(n_timesteps):
            if rng.rand() < cfg["incident_prob"]:
                duration = rng.randint(5, 20)
                end = min(i + duration, n_timesteps)
                incident[i:end] -= rng.uniform(20, 50)

    base_signal = base_flow + morning_peak + evening_peak + daily_period + incident

    data = np.zeros((n_timesteps, n_nodes))
    for node in range(n_nodes):
        node_scale = 0.8 + 0.4 * (node / n_nodes)
        data[:, node] = base_signal * node_scale + regional_bias[node]

    noise = sample_distribution_noise(
        n_timesteps, n_nodes, cfg["dist"], cfg["noise"], seed + 1000)
    data += noise

    return data.astype(np.float32)


def build_noniid_client_configs(num_clients, noniid_level="medium"):
    configs = []
    if noniid_level == "low":
        templates = [
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 600, "noise": 2.0,
             "base": 100.0, "morning_mu": 8.0, "evening_mu": 18.0,
             "morning_amp": 30.0, "evening_amp": 25.0, "peak_sigma": 0.8,
             "trend": 0.0, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 580, "noise": 2.5,
             "base": 95.0, "morning_mu": 7.8, "evening_mu": 17.8,
             "morning_amp": 32.0, "evening_amp": 27.0, "peak_sigma": 0.85,
             "trend": 0.001, "incident_prob": 0.0},
            {"dist": "normal", "pattern": "平稳通勤型", "n_samples": 620, "noise": 3.0,
             "base": 105.0, "morning_mu": 8.2, "evening_mu": 18.2,
             "morning_amp": 28.0, "evening_amp": 23.0, "peak_sigma": 0.75,
             "trend": -0.001, "incident_prob": 0.0},
        ]
    elif noniid_level == "high":
        extra = [
            {"dist": "student-t", "pattern": "波动型", "n_samples": 400, "noise": 7.0,
             "base": 75.0, "morning_mu": 7.0, "evening_mu": 17.0,
             "morning_amp": 38.0, "evening_amp": 33.0, "peak_sigma": 1.1,
             "trend": 0.008, "incident_prob": 0.03},
            {"dist": "chi-square", "pattern": "偏态高流量型", "n_samples": 750, "noise": 10.0,
             "base": 130.0, "morning_mu": 9.0, "evening_mu": 19.0,
             "morning_amp": 22.0, "evening_amp": 18.0, "peak_sigma": 1.3,
             "trend": -0.005, "incident_prob": 0.08},
            {"dist": "gaussian_mixture", "pattern": "双峰型", "n_samples": 480, "noise": 5.5,
             "base": 85.0, "morning_mu": 6.5, "evening_mu": 19.5,
             "morning_amp": 45.0, "evening_amp": 38.0, "peak_sigma": 0.6,
             "trend": 0.003, "incident_prob": 0.02},
        ]
        templates = list(CLIENT_CONFIGS_BASE) + extra
    else:
        templates = list(CLIENT_CONFIGS_BASE)

    for cid in range(num_clients):
        cfg = templates[cid % len(templates)].copy()
        configs.append(cfg)
    return configs


def build_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len + pred_len - 1, 0])
    return np.array(X), np.array(y)


class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_metrics(preds, truths):
    mse = np.mean((preds - truths) ** 2)
    return mse, np.sqrt(mse), np.mean(np.abs(preds - truths))


# ================================================================
# 增强版实验执行函数
# ================================================================

def run_fedavg_enhanced(client_configs, num_nodes, seq_len, pred_len,
                        comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed,
                        verbose=False):
    """运行一次样本量加权 FedAvg 实验，返回结果、loss 历史、预测值。"""
    set_global_seed(seed)
    num_clients = len(client_configs)
    buffer = seq_len + pred_len + 10

    clients_data = []
    for cid, cfg in enumerate(client_configs):
        n_timesteps = cfg["n_samples"] + buffer
        data = generate_traffic_flow(cfg, n_timesteps, num_nodes, seed + cid * 100)
        X, y = build_sequences(data, seq_len, pred_len)
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        train_ds = EnhancedTimeSeriesDataset(X[:train_end], y[:train_end])
        val_ds = EnhancedTimeSeriesDataset(X[train_end:val_end], y[train_end:val_end])
        test_ds = EnhancedTimeSeriesDataset(X[val_end:], y[val_end:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        clients_data.append((train_loader, val_loader, test_loader, len(train_ds)))

    criterion = nn.MSELoss()
    make_model = lambda: CCNOverviewModel(k=num_nodes, t=seq_len,
                                          hidden_dim=hidden_dim, num_heads=4)

    fed_clients = [
        FederatedClient(cid, make_model(), clients_data[cid][0],
                        clients_data[cid][1], criterion, lr=lr)
        for cid in range(num_clients)
    ]

    server = FedAvgServer(make_model(), num_clients)
    server.set_client_data_sizes([d[3] for d in clients_data])

    client_train_hist = {cid: [] for cid in range(num_clients)}
    client_epoch_hist = []  # list of (round, cid, local_epoch, train_loss, val_loss)
    for rnd in range(comm_rounds):
        round_states, round_losses = [], []
        for client in fed_clients:
            loss, state, epoch_losses = client.train_local(
                epochs=local_epochs, global_model=server.global_model, verbose=False)
            round_losses.append(loss)
            round_states.append(state)
            client_train_hist[client.client_id].append(loss)
            for le, (tl, vl) in enumerate(epoch_losses):
                client_epoch_hist.append((rnd, client.client_id, le, tl, vl))
        server.aggregate(round_states, round_losses)
        if verbose:
            print(f"    Round {rnd + 1}/{comm_rounds}, "
                  f"Avg Loss: {server.round_losses[-1]:.6f}")

    results = []
    all_preds, all_truths = [], []
    for cid in range(num_clients):
        fed_clients[cid].model.load_state_dict(server.global_model.state_dict())
        preds, truths = fed_clients[cid].test_predictions()
        metrics = fed_clients[cid].test_metrics()
        metrics["client"] = cid
        results.append(metrics)
        all_preds.append(preds)
        all_truths.append(truths)

    return results, server.round_losses, client_train_hist, client_epoch_hist, all_preds, all_truths


def run_independent_enhanced(client_configs, num_nodes, seq_len, pred_len,
                             comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed):
    set_global_seed(seed)
    num_clients = len(client_configs)
    buffer = seq_len + pred_len + 10

    results = []
    criterion = nn.MSELoss()
    for cid, cfg in enumerate(client_configs):
        n_timesteps = cfg["n_samples"] + buffer
        data = generate_traffic_flow(cfg, n_timesteps, num_nodes, seed + cid * 100)
        X, y = build_sequences(data, seq_len, pred_len)
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        train_ds = EnhancedTimeSeriesDataset(X[:train_end], y[:train_end])
        test_ds = EnhancedTimeSeriesDataset(X[val_end:], y[val_end:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = CCNOverviewModel(k=num_nodes, t=seq_len,
                                 hidden_dim=hidden_dim, num_heads=4).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        total_epochs = comm_rounds * local_epochs
        for _ in range(total_epochs):
            model.train()
            for x, y_b in train_loader:
                x, y_b = x.to(DEVICE).float(), y_b.to(DEVICE).float().view(-1)
                optimizer.zero_grad()
                p, _ = model(x)
                loss = criterion(p.view(-1), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            ap, at = [], []
            for x, y_b in test_loader:
                x = x.to(DEVICE).float()
                p, _ = model(x)
                ap.append(p.detach().cpu().numpy().reshape(-1))
                at.append(y_b.cpu().numpy().reshape(-1))
            preds = np.concatenate(ap)
            truths = np.concatenate(at)
            mse, rmse, mae = compute_metrics(preds, truths)
            results.append({"client": cid, "mse": mse, "rmse": rmse, "mae": mae,
                            "preds": preds, "truths": truths})
    ind_preds = [r["preds"] for r in results]
    ind_truths = [r["truths"] for r in results]
    for r in results:
        del r["preds"]
        del r["truths"]
    return results, ind_preds, ind_truths


# ================================================================
# workflow: main
# ================================================================

def run_main_experiment(output_dir: Path):
    print("\n" + "=" * 60)
    print("Workflow: main -- Complex Non-IID Main Experiment")
    print("=" * 60)

    num_clients = 5
    num_nodes = 8
    seq_len = 12
    pred_len = 1
    comm_rounds = 20
    local_epochs = 3
    batch_size = 32
    lr = 0.001
    hidden_dim = 64
    seeds = [42, 2024, 2025]

    all_fedavg, all_independent = [], []

    for seed in seeds:
        print(f"\n--- Seed = {seed} ---")
        cfgs = build_noniid_client_configs(num_clients, "medium")
        fed_results, _, _, _, _, _ = run_fedavg_enhanced(
            cfgs, num_nodes, seq_len, pred_len,
            comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)
        ind_results, _, _ = run_independent_enhanced(
            cfgs, num_nodes, seq_len, pred_len,
            comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)

        for r in fed_results:
            r["method"] = "CNN-FedAvg"; r["seed"] = seed
        for r in ind_results:
            r["method"] = "Independent"; r["seed"] = seed
        all_fedavg.extend(fed_results)
        all_independent.extend(ind_results)

    df_fed = pd.DataFrame(all_fedavg)
    df_ind = pd.DataFrame(all_independent)

    def agg(df, method):
        g = df.groupby("method")
        return {"Method": method,
                "MSE_mean": g["mse"].mean()[method], "MSE_std": g["mse"].std()[method],
                "RMSE_mean": g["rmse"].mean()[method], "RMSE_std": g["rmse"].std()[method],
                "MAE_mean": g["mae"].mean()[method], "MAE_std": g["mae"].std()[method]}

    df_summary = pd.DataFrame([agg(df_fed, "CNN-FedAvg"), agg(df_ind, "Independent")])
    print("\n=== Main Experiment Results (mean ± std across clients & seeds) ===")
    print(df_summary.to_string(index=False))
    save_dataframe(df_summary, output_dir, "cnn_enhanced_main_metrics.csv")

    cfgs = build_noniid_client_configs(num_clients, "medium")
    _, _, _, _, fed_preds, fed_truths = run_fedavg_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, 42)
    _, ind_preds, _ = run_independent_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, 42)

    configure_plot_style()
    fig, axes = plt.subplots(num_clients, 1, figsize=(14, 2.5 * num_clients))
    for cid in range(num_clients):
        n_show = min(100, len(fed_preds[cid]))
        axes[cid].plot(fed_truths[cid][:n_show], "k-", alpha=0.6, linewidth=2, label="Ground Truth")
        axes[cid].plot(fed_preds[cid][:n_show], "b--", alpha=0.7, label="CNN-FedAvg")
        axes[cid].plot(ind_preds[cid][:n_show], "r:", alpha=0.7, label="Independent")
        axes[cid].set_ylabel(f"Client {cid}")
        if cid == 0:
            axes[cid].legend()
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle("CNN Enhanced: Prediction Curve (Main Experiment, Seed=42)")
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_prediction_curve.png")
    return df_summary


# ================================================================
# workflow: client_scale
# ================================================================

def run_client_scale_experiment(output_dir: Path):
    print("\n" + "=" * 60)
    print("Workflow: client_scale -- Client Count Experiment")
    print("=" * 60)

    client_nums = [3, 5, 8]
    num_nodes = 8; seq_len = 12; pred_len = 1
    comm_rounds = 20; local_epochs = 3; batch_size = 32; lr = 0.001; hidden_dim = 64
    seed = 42

    results = []
    for nc in client_nums:
        print(f"\n--- Num Clients = {nc} ---")
        cfgs = build_noniid_client_configs(nc, "medium")
        fed_results, round_losses, _, _, _, _ = run_fedavg_enhanced(
            cfgs, num_nodes, seq_len, pred_len,
            comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)
        mse_vals = [r["mse"] for r in fed_results]
        rmse_vals = [r["rmse"] for r in fed_results]
        mae_vals = [r["mae"] for r in fed_results]
        results.append({
            "Client_Number": nc,
            "MSE_mean": np.mean(mse_vals),
            "RMSE_mean": np.mean(rmse_vals),
            "RMSE_std": np.std(rmse_vals, ddof=0),
            "MAE_mean": np.mean(mae_vals),
            "Final_Global_Loss": round_losses[-1] if round_losses else 0,
        })

    df_scale = pd.DataFrame(results)
    print("\n=== Client Scale Results ===")
    print(df_scale.to_string(index=False))
    save_dataframe(df_scale, output_dir, "cnn_enhanced_client_scale_metrics.csv")

    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = df_scale["Client_Number"].astype(str)
    axes[0].bar(xs, df_scale["RMSE_mean"], color="darkorange")
    axes[0].set_title("RMSE vs Client Count")
    axes[0].set_ylabel("RMSE")
    axes[1].bar(xs, df_scale["Final_Global_Loss"], color="seagreen")
    axes[1].set_title("Final Global Loss vs Client Count")
    axes[1].set_ylabel("Loss")
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_client_scale_figure.png")
    return df_scale


# ================================================================
# workflow: noniid
# ================================================================

def run_noniid_strength_experiment(output_dir: Path):
    print("\n" + "=" * 60)
    print("Workflow: noniid -- Non-IID Strength Experiment")
    print("=" * 60)

    levels = ["low", "medium", "high"]
    num_clients = 5
    num_nodes = 8; seq_len = 12; pred_len = 1
    comm_rounds = 20; local_epochs = 3; batch_size = 32; lr = 0.001; hidden_dim = 64
    seed = 42

    results = []
    for level in levels:
        print(f"\n--- Non-IID Level = {level} ---")
        cfgs = build_noniid_client_configs(num_clients, level)
        fed_results, _, _, _, _, _ = run_fedavg_enhanced(
            cfgs, num_nodes, seq_len, pred_len,
            comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)
        ind_results, _, _ = run_independent_enhanced(
            cfgs, num_nodes, seq_len, pred_len,
            comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)

        for r in fed_results:
            results.append({"Level": level, "Method": "CNN-FedAvg",
                            "MSE": r["mse"], "RMSE": r["rmse"], "MAE": r["mae"]})
        for r in ind_results:
            results.append({"Level": level, "Method": "Independent",
                            "MSE": r["mse"], "RMSE": r["rmse"], "MAE": r["mae"]})

    df_noniid = pd.DataFrame(results)
    df_agg = df_noniid.groupby(["Level", "Method"]).agg(
        MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")).reset_index()

    print("\n=== Non-IID Strength Results ===")
    print(df_agg.to_string(index=False))
    save_dataframe(df_agg, output_dir, "cnn_enhanced_noniid_strength_metrics.csv")

    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(["RMSE_mean", "MAE_mean", "MSE_mean"]):
        for method in ["CNN-FedAvg", "Independent"]:
            sub = df_agg[df_agg["Method"] == method]
            axes[i].plot(sub["Level"], sub[metric], "o-", linewidth=2, label=method)
        axes[i].set_title(metric.replace("_mean", ""))
        axes[i].legend()
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_noniid_strength_figure.png")
    return df_agg


# ================================================================
# workflow: convergence
# ================================================================

def run_convergence_experiment(output_dir: Path):
    print("\n" + "=" * 60)
    print("Workflow: convergence -- Convergence Analysis")
    print("=" * 60)

    num_clients = 5
    num_nodes = 8; seq_len = 12; pred_len = 1
    comm_rounds = 30; local_epochs = 3; batch_size = 32; lr = 0.001; hidden_dim = 64
    seed = 42

    cfgs = build_noniid_client_configs(num_clients, "medium")
    _, global_losses, client_train_hist, client_epoch_hist, _, _ = run_fedavg_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)

    # CSV 1: global round loss
    df_global = pd.DataFrame({
        "Round": np.arange(1, len(global_losses) + 1),
        "Global_Loss": global_losses})
    df_global.to_csv(output_dir / "cnn_enhanced_global_loss.csv", index=False, encoding="utf-8")

    # CSV 2: client round loss
    df_cr = pd.DataFrame(client_train_hist)
    df_cr.columns = [f"Client_{c}" for c in df_cr.columns]
    df_cr.insert(0, "Round", np.arange(1, len(df_cr) + 1))
    df_cr.to_csv(output_dir / "cnn_enhanced_client_round_loss.csv", index=False, encoding="utf-8")

    # CSV 3: client epoch loss
    df_ce = pd.DataFrame(client_epoch_hist,
                         columns=["Round", "Client", "Local_Epoch", "Train_Loss", "Val_Loss"])
    df_ce.to_csv(output_dir / "cnn_enhanced_client_epoch_loss.csv", index=False, encoding="utf-8")

    configure_plot_style()

    # PNG 1: global loss only
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(global_losses) + 1), global_losses, "b-o", linewidth=2)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Average Training Loss")
    ax.set_title("Global Model Convergence (CNN-FedAvg)")
    plt.tight_layout()
    fig.savefig(output_dir / "cnn_enhanced_global_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PNG 2: client epoch-level loss
    fig, ax = plt.subplots(figsize=(12, 6))
    for cid in range(num_clients):
        sub = df_ce[df_ce["Client"] == cid]
        ax.plot(sub["Round"] + sub["Local_Epoch"] / local_epochs,
                sub["Train_Loss"], "o-", markersize=2, alpha=0.7, label=f"Client {cid}")
    ax.set_xlabel("Round (with epoch offset)")
    ax.set_ylabel("Training Loss")
    ax.set_title("Client-Level Local Epoch Loss (CNN-FedAvg)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "cnn_enhanced_client_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PNG 3: convergence overview (global + client round on same figure)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.arange(1, len(global_losses) + 1), global_losses, "b-o", linewidth=2)
    axes[0].set_xlabel("Communication Round"); axes[0].set_ylabel("Average Training Loss")
    axes[0].set_title("Global Convergence")
    for cid in range(num_clients):
        axes[1].plot(np.arange(1, comm_rounds + 1),
                     client_train_hist[cid], "o-", label=f"Client {cid}")
    axes[1].set_xlabel("Communication Round"); axes[1].set_ylabel("Local Training Loss")
    axes[1].set_title("Client Round-Level Loss"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "cnn_enhanced_convergence_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    return df_global
def run_client_metrics_experiment(output_dir: Path):
    print("\n" + "=" * 60)
    print("Workflow: client_metrics -- Per-Client Error Analysis")
    print("=" * 60)

    num_clients = 5
    num_nodes = 8; seq_len = 12; pred_len = 1
    comm_rounds = 20; local_epochs = 3; batch_size = 32; lr = 0.001; hidden_dim = 64
    seed = 42

    cfgs = build_noniid_client_configs(num_clients, "medium")
    fed_results, _, _, _, _, _ = run_fedavg_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)
    ind_results, _, _ = run_independent_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)

    rows = []
    for cid in range(num_clients):
        rows.append({"Client": f"Client {cid}", "Method": "CNN-FedAvg",
                     "MSE": fed_results[cid]["mse"], "RMSE": fed_results[cid]["rmse"],
                     "MAE": fed_results[cid]["mae"],
                     "Distribution_Type": cfgs[cid]["dist"],
                     "Traffic_Pattern": cfgs[cid]["pattern"],
                     "Sample_Size": cfgs[cid]["n_samples"],
                     "Noise_Level": cfgs[cid]["noise"]})
        rows.append({"Client": f"Client {cid}", "Method": "Independent",
                     "MSE": ind_results[cid]["mse"], "RMSE": ind_results[cid]["rmse"],
                     "MAE": ind_results[cid]["mae"],
                     "Distribution_Type": cfgs[cid]["dist"],
                     "Traffic_Pattern": cfgs[cid]["pattern"],
                     "Sample_Size": cfgs[cid]["n_samples"],
                     "Noise_Level": cfgs[cid]["noise"]})

    df_cm = pd.DataFrame(rows)
    print("\n=== Per-Client Metrics ===")
    print(df_cm.to_string(index=False))
    save_dataframe(df_cm, output_dir, "cnn_enhanced_client_metrics.csv")

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(num_clients); w = 0.35
    fed_rmse = [fed_results[c]["rmse"] for c in range(num_clients)]
    ind_rmse = [ind_results[c]["rmse"] for c in range(num_clients)]
    ax.bar(x - w/2, fed_rmse, w, label="CNN-FedAvg", color="steelblue")
    ax.bar(x + w/2, ind_rmse, w, label="Independent", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels([f"Client {i}" for i in range(num_clients)])
    ax.set_ylabel("RMSE"); ax.set_title("Per-Client RMSE Comparison"); ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_client_rmse_figure.png")
    return df_cm


# ================================================================
# workflow: peak
# ================================================================

def run_peak_offpeak_experiment(output_dir: Path):
    print("\n" + "=" * 60)
    print("Workflow: peak -- Peak / Off-Peak Analysis")
    print("=" * 60)

    num_clients = 5
    num_nodes = 8; seq_len = 12; pred_len = 1
    comm_rounds = 20; local_epochs = 3; batch_size = 32; lr = 0.001; hidden_dim = 64
    seed = 42

    cfgs = build_noniid_client_configs(num_clients, "medium")
    hpd = 24
    _, _, _, _, fed_preds, fed_truths = run_fedavg_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)
    _, ind_preds, _ = run_independent_enhanced(
        cfgs, num_nodes, seq_len, pred_len,
        comm_rounds, local_epochs, batch_size, lr, hidden_dim, seed)

    n_preds = len(fed_preds[0])
    hours_per_step = hpd / n_preds

    def classify_period(idx):
        hour = (idx * hours_per_step) % hpd
        if 7 <= hour < 9: return "morning_peak"
        elif 17 <= hour < 19: return "evening_peak"
        else: return "off_peak"

    rows = []
    for cid in range(num_clients):
        for method, preds in [("CNN-FedAvg", fed_preds), ("Independent", ind_preds)]:
            truths = fed_truths[cid]
            n = min(len(preds[cid]), len(truths))
            for period_name in ["morning_peak", "evening_peak", "off_peak"]:
                indices = [i for i in range(n) if classify_period(i) == period_name]
                if len(indices) < 5: continue
                p = preds[cid][indices]; t = truths[indices]
                mse, rmse, mae = compute_metrics(p, t)
                rows.append({"Client": f"Client {cid}", "Method": method,
                             "Period": period_name,
                             "MSE": mse, "RMSE": rmse, "MAE": mae})

    df_peak = pd.DataFrame(rows)
    save_dataframe(df_peak, output_dir, "cnn_enhanced_peak_offpeak_metrics.csv")

    df_agg = df_peak.groupby(["Period", "Method"]).agg(
        RMSE_mean=("RMSE", "mean"), MAE_mean=("MAE", "mean")).reset_index()

    print("\n=== Peak / Off-Peak Analysis ===")
    print(df_agg.to_string(index=False))

    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    periods = ["morning_peak", "evening_peak", "off_peak"]
    x = np.arange(len(periods)); w = 0.35
    for i, (metric, ax) in enumerate(zip(["RMSE_mean", "MAE_mean"], axes)):
        sub_fed = df_agg[df_agg["Method"] == "CNN-FedAvg"].set_index("Period").reindex(periods)
        sub_ind = df_agg[df_agg["Method"] == "Independent"].set_index("Period").reindex(periods)
        ax.bar(x - w/2, sub_fed[metric], w, label="CNN-FedAvg", color="steelblue")
        ax.bar(x + w/2, sub_ind[metric], w, label="Independent", color="darkorange")
        ax.set_xticks(x); ax.set_xticklabels(periods, rotation=15)
        ax.set_title(metric.replace("_mean", "")); ax.legend()
    plt.tight_layout()
    save_figure(fig, output_dir, "cnn_enhanced_peak_offpeak_figure.png")
    return df_agg
def run_project(workflow: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    log_path = output_dir / "cnn_enhanced_run_log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle, redirect_stdout(
        TeeStream(sys.stdout, log_handle)):
        configure_plot_style()
        print(f"[setup] Using device: {DEVICE}")
        print(f"[setup] Writing experiment log: {log_path}")
        print(f"[setup] PROJECT_NAME: {PROJECT_NAME}")

        if workflow in ("all", "main"):
            run_main_experiment(output_dir)
        if workflow in ("all", "client_scale"):
            run_client_scale_experiment(output_dir)
        if workflow in ("all", "noniid"):
            run_noniid_strength_experiment(output_dir)
        if workflow in ("all", "convergence"):
            run_convergence_experiment(output_dir)
        if workflow in ("all", "client_metrics"):
            run_client_metrics_experiment(output_dir)
        if workflow in ("all", "peak"):
            run_peak_offpeak_experiment(output_dir)

        print(f"\n=== All {workflow} workflows completed ===")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="CNN Enhanced Simulation.")
    parser.add_argument(
        "--workflow",
        choices=["all", "main", "client_scale", "noniid", "convergence",
                 "client_metrics", "peak"],
        default="all", help="Workflow to execute.")
    parser.add_argument("--output-dir", default=None,
                        help="Relative subdirectory under results/simulation_experiments.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_project(args.workflow, resolve_output_dir(args.output_dir))


if __name__ == "__main__":
    main()
