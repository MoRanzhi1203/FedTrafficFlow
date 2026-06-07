# -*- coding: utf-8 -*-
"""
CNN/CCN 基础联邦仿真实验。

本文件实现基于 CNN-BiLSTM-Attention 的联邦仿真基础实验，包含：
1. data_viz: 基础数据集可视化（时间序列、热力图、箱线图、划分概览、样本量）；
2. main: Independent / FedAvg 主结果对比（MSE、RMSE、MAE）；
3. convergence: 联邦训练收敛曲线；
4. all: 依次运行上述全部工作流。

与 gcn_fed_base.py 共享相同的数据生成逻辑和随机种子，保证基础对比公平。

主要依赖：PyTorch, NumPy, pandas。
功能：数据生成、模型训练（FedAvg / Independent）、指标计算、数据导出。
导出数据：results/simulation_experiments/cnn_fed_base/
    - base_dataset_*.csv
    - main_metrics.csv
    - convergence_history.csv
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
from torch.utils.data import DataLoader, Dataset, random_split

# ──────────────────────────────────────────────────────────
# 全局路径与设备常量
# ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # core.py is inside sub-package
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAFFIC_MIN_VALUE = 0.0
MAPE_EPS = 1.0

# ──────────────────────────────────────────────────────────
# 基础实验共享超参数（与 gcn_fed_base.py 保持一致）
# ──────────────────────────────────────────────────────────
BASE_SEED = 42
DEFAULT_MULTI_SEEDS = [42, 2024, 2025, 2026, 3407]
BASE_NUM_CLIENTS = 5
BASE_NUM_NODES = 8          # K: 交通传感器/观测节点数
BASE_SEQ_LEN = 24           # T: 时间窗口长度（例如每小时一个点，共一天）
BASE_PRED_LEN = 1           # 预测步长
BASE_SAMPLES_PER_CLIENT = [200, 200, 200, 200, 200]  # 平衡样本
BASE_NOISE = 0.05           # 观测噪声标准差
BASE_TRAIN_RATIO = 0.70
BASE_VAL_RATIO = 0.10
BASE_TEST_RATIO = 0.20
# 联邦训练超参数
FED_ROUNDS = 10
FED_LOCAL_EPOCHS = 3
FED_BATCH_SIZE = 16
FED_HIDDEN_DIM = 64

# ──────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """设置全局随机种子，保证可复现性。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dir(output_dir: Path) -> Path:
    """创建并返回输出目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(df, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path


def parse_bool_flag(value) -> bool:
    """解析命令行中的布尔参数。"""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数: {value}")


def parse_seed_list(seed_text: str | None) -> list[int]:
    """将逗号分隔的 seed 字符串解析为整数列表。"""
    if seed_text is None or not str(seed_text).strip():
        return list(DEFAULT_MULTI_SEEDS)
    seeds = [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]
    if not seeds:
        raise ValueError("至少需要提供一个随机种子")
    return seeds


def compute_r2_score(preds: np.ndarray, truths: np.ndarray) -> float:
    """计算 R2。"""
    preds_arr = np.asarray(preds, dtype=np.float64)
    truths_arr = np.asarray(truths, dtype=np.float64)
    ss_res = float(np.sum((preds_arr - truths_arr) ** 2))
    ss_tot = float(np.sum((truths_arr - truths_arr.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def build_metric_summary_table(
    raw_df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """按实验/方法汇总 mean、std、95% CI、best、worst。"""
    summary_rows = []
    for group_values, group_df in raw_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base_record = dict(zip(group_cols, group_values))
        n = len(group_df)
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
            summary_rows.append(record)
    return pd.DataFrame(summary_rows)


def build_convergence_summary_table(
    raw_df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """按方法与轮次汇总多 seed 收敛统计。"""
    summary_rows = []
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
        summary_rows.append(record)
    return pd.DataFrame(summary_rows)


def build_improvement_rows(
    raw_df: pd.DataFrame,
    experiment_name: str,
    baseline_method: str,
    enhanced_method: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    """计算增强方法相对基线的多 seed 提升率。"""
    baseline_df = raw_df[raw_df["method"] == baseline_method].set_index("seed")
    enhanced_df = raw_df[raw_df["method"] == enhanced_method].set_index("seed")
    common_seeds = sorted(set(baseline_df.index) & set(enhanced_df.index))
    rows = []
    for metric in metric_cols:
        if metric not in baseline_df.columns or metric not in enhanced_df.columns:
            continue
        improvements = []
        improved_flags = []
        for seed in common_seeds:
            baseline_value = float(baseline_df.loc[seed, metric])
            enhanced_value = float(enhanced_df.loc[seed, metric])
            if metric.lower() == "r2":
                improvement = (enhanced_value - baseline_value) / max(abs(baseline_value), 1e-8) * 100.0
            else:
                improvement = (baseline_value - enhanced_value) / max(abs(baseline_value), 1e-8) * 100.0
            improvements.append(improvement)
            improved_flags.append(improvement > 0.0)
        if not improvements:
            continue
        rows.append({
            "experiment": experiment_name,
            "baseline_method": baseline_method,
            "enhanced_method": enhanced_method,
            "metric": metric,
            "mean_improvement_percent": float(np.mean(improvements)),
            "std_improvement_percent": float(np.std(improvements, ddof=0)),
            "improved_seed_count": int(np.sum(improved_flags)),
            "total_seed_count": int(len(improved_flags)),
            "improved_seed_ratio": float(np.mean(improved_flags)),
            "per_seed_improved": ",".join(
                f"{seed}:{'Y' if flag else 'N'}" for seed, flag in zip(common_seeds, improved_flags)
            ),
        })
    return pd.DataFrame(rows)


def write_stability_report(
    output_dir: Path,
    raw_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    experiment_name: str,
    baseline_method: str,
    enhanced_method: str,
) -> Path:
    """输出多 seed 文字稳定性报告。"""
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
        lines.extend([
            "",
            (
                f"{enhanced_method} achieves stable improvement over {baseline_method} across "
                f"{int(improvement_df['total_seed_count'].max())} random seeds, indicating that the "
                "observed gain is not caused by a single favorable seed."
            ),
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")
    return report_path


def generate_base_traffic_data(
    seed: int = BASE_SEED,
    num_clients: int = BASE_NUM_CLIENTS,
    num_nodes: int = BASE_NUM_NODES,
    seq_len: int = BASE_SEQ_LEN,
    pred_len: int = BASE_PRED_LEN,
    samples_per_client: list = None,
    noise: float = BASE_NOISE,
):
    """生成基础交通流仿真数据。

    生成逻辑：
      - 为每个客户端生成一个基础的周期性交通流模式（双峰，模拟早晚高峰）；
      - 不同客户端之间存在受控的轻微差异（相位偏移、幅度缩放、节点偏好）；
      - 数据格式：X shape [num_clients, samples, num_nodes, seq_len]
                  Y shape [num_clients, samples]  (预测下一个时间步的全局流量均值)

    该函数在 cnn_fed_base.py 和 gcn_fed_base.py 中实现完全一致，
    仅在 GCN 文件中额外生成邻接矩阵。

    返回:
        all_X: list of np.ndarray, 每个元素 shape [num_samples_i, num_nodes, seq_len]
        all_Y: list of np.ndarray, 每个元素 shape [num_samples_i,]
        metadata: dict, 包含数据集的汇总信息
    """
    if samples_per_client is None:
        samples_per_client = BASE_SAMPLES_PER_CLIENT

    rng = np.random.RandomState(seed)

    # 时间轴 (0 到 seq_len-1)
    t_axis = np.arange(seq_len)

    # 基础交通流模式：双峰曲线模拟早晚高峰
    base_pattern = (
        0.3 * np.sin(2 * np.pi * t_axis / seq_len)           # 日周期
        + 0.5 * np.exp(-0.5 * ((t_axis - 8) / 2) ** 2)       # 早高峰 ~8h
        + 0.6 * np.exp(-0.5 * ((t_axis - 17) / 2) ** 2)      # 晚高峰 ~17h
        + 0.2 * np.sin(4 * np.pi * t_axis / seq_len + 1.0)   # 半日周期谐波
    )

    all_X = []
    all_Y = []
    metadata = {
        "num_clients": num_clients,
        "num_nodes": num_nodes,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "samples_per_client": samples_per_client,
        "noise": noise,
    }

    for cid in range(num_clients):
        n_samples = samples_per_client[cid]
        # 每个客户端的模式有轻微差异
        phase_shift = 0.05 * cid                      # 相位偏移
        amp_scale = 1.0 + 0.08 * (cid - num_clients // 2)  # 幅度缩放
        # 不同节点对早晚高峰的敏感度不同
        node_sensitivity = 0.7 + 0.3 * np.sin(np.linspace(0, np.pi, num_nodes) + cid * 0.3)

        X_client = np.zeros((n_samples, num_nodes, seq_len), dtype=np.float32)
        Y_client = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            # 每个样本基于基础模式并加入节点级变化和噪声
            sample_noise = rng.randn(num_nodes, seq_len) * noise
            for node in range(num_nodes):
                # 节点级流量 = 敏感度 * 幅度 * 基础模式 + 噪声
                node_flow = (
                    node_sensitivity[node]
                    * amp_scale
                    * (base_pattern + 0.02 * rng.randn(seq_len))  # 样本间微小扰动
                    + sample_noise[node]
                )
                X_client[i, node, :] = node_flow
            X_client[i] = np.clip(X_client[i], TRAFFIC_MIN_VALUE, None)
            

            # 目标值：最后 pred_len 个时间步内所有节点的平均流量
            Y_client[i] = X_client[i, :, -pred_len:].mean()

        all_X.append(X_client)
        all_Y.append(Y_client)

    return all_X, all_Y, metadata


def split_train_val_test(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = BASE_SEED,
    train_ratio: float = BASE_TRAIN_RATIO,
    val_ratio: float = BASE_VAL_RATIO,
):
    """按时间顺序划分训练/验证/测试集。

    返回:
        (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        X[train_idx], Y[train_idx],
        X[val_idx], Y[val_idx],
        X[test_idx], Y[test_idx],
    )


# ──────────────────────────────────────────────────────────
# Dataset 类
# ──────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """交通流数据集封装。"""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ──────────────────────────────────────────────────────────
# 模型定义
# ──────────────────────────────────────────────────────────

class AdaptiveSwish(nn.Module):
    """带可学习系数的 Swish 激活函数。"""

    def __init__(self, trainable: bool = True):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class CNNBaseModel(nn.Module):
    """CNN-BiLSTM-Attention 基础联邦模型。

    结构：
    1. CNN 分支：一维卷积提取局部时间邻域模式；
    2. BiLSTM 分支：捕捉双向时序依赖；
    3. 多头注意力融合两个分支特征。
    """

    def __init__(self, k: int, t: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=k, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(
            input_size=k, hidden_size=hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            AdaptiveSwish(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)

        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)

        feat_seq = torch.stack([x_cnn, x_lstm], dim=1)
        attn_output, attn_weights = self.multihead_attn(feat_seq, feat_seq, feat_seq)
        attn_output = self.attn_norm(attn_output + feat_seq)
        x_fused = attn_output.mean(dim=1)
        return self.regression_head(x_fused), attn_weights


class IndependentBaseModel(nn.Module):
    """独立训练基线模型（简单 MLP）。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k * t, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x), None


# ──────────────────────────────────────────────────────────
# 联邦客户端与服务端
# ──────────────────────────────────────────────────────────

class FederatedClient:
    """联邦客户端。"""

    def __init__(self, client_id, model, train_loader, val_loader, test_loader,
                 criterion, lr: float = 1e-3):
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
            loss = self.criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        avg = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(avg)
        return avg

    @torch.no_grad()
    def validate(self, loader=None):
        """在指定数据加载器上评估损失。"""
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            total_loss += self.criterion(pred.squeeze(), y).item() * x.shape[0]
        avg = total_loss / len(loader.dataset)
        return avg

    def train_local(self, epochs: int = 3, global_model=None, verbose: bool = False,
                    prefix: str = "Local"):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            if verbose:
                print(f"  {prefix} Client{self.client_id} epoch {epoch+1}/{epochs} "
                      f"train={train_loss:.6f} val={val_loss:.6f}")
        return float(self.train_losses[-1]), copy.deepcopy(self.model.state_dict())

    @torch.no_grad()
    def test_metrics(self):
        """计算测试集 MSE、RMSE、MAE、MAPE。"""
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred, _ = self.model(x)
            preds.append(pred.squeeze())
            truths.append(y)
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        diff = preds - truths
        mse = float((diff ** 2).mean().item())
        rmse = float(torch.sqrt((diff ** 2).mean()).item())
        mae = float(diff.abs().mean().item())
        mape = float((diff.abs() / torch.clamp(truths.abs(), min=MAPE_EPS)).mean().item()) * 100
        preds_np = preds.detach().cpu().numpy()
        truths_np = truths.detach().cpu().numpy()
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": compute_r2_score(preds_np, truths_np),
            "preds": preds_np,
            "truths": truths_np,
        }


class IndependentClient(FederatedClient):
    """独立训练客户端（不参与联邦聚合）。"""

    def __init__(self, client_id, model, train_loader, val_loader, test_loader, criterion):
        super().__init__(client_id, model, train_loader, val_loader, test_loader,
                         criterion, lr=0.01)

    def train_local(self, epochs: int = 10, verbose: bool = False):
        return super().train_local(epochs=epochs, global_model=None,
                                   verbose=verbose, prefix="Independent")


class FedAvgServer:
    """FedAvg 服务端：样本量加权聚合。"""

    def __init__(self, model, num_clients: int):
        self.global_model = model.to(DEVICE).float()
        self.num_clients = num_clients
        self.round_losses = []
        self.round_val_losses = []
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = sizes

    def aggregate(self, client_weights, client_losses):
        total_n = float(sum(self.client_data_sizes))
        weights_arr = np.array(self.client_data_sizes) / total_n
        global_dict = self.global_model.state_dict()
        new_dict = {k: torch.zeros_like(v, dtype=torch.float32)
                    for k, v in global_dict.items()}
        for key in new_dict:
            for idx in range(self.num_clients):
                cw = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                new_dict[key] += cw * torch.tensor(float(weights_arr[idx]),
                                                    device=DEVICE, dtype=torch.float32)
        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(client_losses)))
        return self.global_model.state_dict()


# ══════════════════════════════════════════════════════════
# Workflow: data_viz — 导出基础数据集可视化所需数据
# ══════════════════════════════════════════════════════════

def export_base_dataset_artifacts(output_dir: Path) -> None:
    """只导出基础数据集可视化所需的数据文件，不画图。"""
    print("\n" + "=" * 60)
    print("[data_viz] Exporting base dataset artifacts")
    print("=" * 60)

    set_global_seed(BASE_SEED)
    all_X, all_Y, meta = generate_base_traffic_data()

    ensure_output_dir(output_dir)

    # ── 1. 每个 client 的平均交通流时间序列 ──
    ts_rows = []
    for cid in range(meta["num_clients"]):
        ts_mean = all_X[cid].mean(axis=(0, 1))  # shape [seq_len,]
        for t_step, val in enumerate(ts_mean):
            ts_rows.append({"client_id": cid, "time_step": t_step, "traffic_flow": float(val)})
    save_dataframe(pd.DataFrame(ts_rows), output_dir, "base_dataset_client_timeseries.csv")

    # ── 2. 代表性 client 的节点-时间热力图数据 ──
    heatmap_rows = []
    for cid in range(meta["num_clients"]):
        X_c = all_X[cid]  # [samples, nodes, seq_len]
        node_time_matrix = X_c.mean(axis=0)  # [nodes, seq_len]
        for node_id in range(node_time_matrix.shape[0]):
            for t_step in range(node_time_matrix.shape[1]):
                heatmap_rows.append({
                    "client_id": cid,
                    "node_id": node_id,
                    "time_step": t_step,
                    "traffic_flow": float(node_time_matrix[node_id, t_step])
                })
    save_dataframe(pd.DataFrame(heatmap_rows), output_dir, "base_dataset_node_heatmap.csv")

    # ── 3. 不同 client 的流量分布数据 ──
    dist_rows = []
    for cid in range(meta["num_clients"]):
        sampled_values = all_X[cid].ravel()[::8]
        dist_rows.extend(
            {"client_id": cid, "traffic_flow": float(value)}
            for value in sampled_values
        )
    save_dataframe(pd.DataFrame(dist_rows), output_dir, "base_dataset_client_distribution.csv")

    # ── 4. train / val / test 划分概览 ──
    total_samples = sum(meta["samples_per_client"])
    train_n = int(total_samples * BASE_TRAIN_RATIO)
    val_n = int(total_samples * BASE_VAL_RATIO)
    test_n = total_samples - train_n - val_n
    split_rows = [
        {"split": "Train", "num_samples": train_n, "ratio": BASE_TRAIN_RATIO},
        {"split": "Val", "num_samples": val_n, "ratio": BASE_VAL_RATIO},
        {"split": "Test", "num_samples": test_n, "ratio": BASE_TEST_RATIO},
    ]
    save_dataframe(pd.DataFrame(split_rows), output_dir, "base_dataset_split_overview.csv")

    # ── 5. 每个 client 的样本量 ──
    sample_size_rows = [
        {"client_id": i, "num_samples": meta["samples_per_client"][i]}
        for i in range(meta["num_clients"])
    ]
    save_dataframe(pd.DataFrame(sample_size_rows), output_dir, "base_dataset_client_sample_size.csv")

    # ── 6. 数据集汇总 CSV ──
    summary_rows = []
    for cid in range(meta["num_clients"]):
        X_c = all_X[cid]
        Y_c = all_Y[cid]
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test(X_c, Y_c)
        all_vals = X_c.ravel()
        summary_rows.append({
            "client_id": cid,
            "num_samples": len(X_c),
            "num_nodes": meta["num_nodes"],
            "seq_len": meta["seq_len"],
            "pred_len": meta["pred_len"],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "mean_flow": float(np.mean(all_vals)),
            "std_flow": float(np.std(all_vals)),
            "min_flow": float(np.min(all_vals)),
            "max_flow": float(np.max(all_vals)),
        })

    df_summary = pd.DataFrame(summary_rows)
    save_dataframe(df_summary, output_dir, "base_dataset_summary.csv")
    print("[data_viz] Export Done.\n")


# ══════════════════════════════════════════════════════════
# Workflow: main — Independent / FedAvg 主结果对比
# ══════════════════════════════════════════════════════════

def run_single_seed_main_experiment(seed: int):
    """运行单个 seed 的基础实验主结果。"""
    print("\n" + "=" * 60)
    print(f"[main] Running CNN FedAvg base experiment | seed={seed}")
    print("=" * 60)

    set_global_seed(seed)
    all_X, all_Y, _ = generate_base_traffic_data(seed=seed)

    criterion = nn.MSELoss()
    num_clients = BASE_NUM_CLIENTS
    k = BASE_NUM_NODES
    t = BASE_SEQ_LEN

    # 构建数据加载器
    train_loaders, val_loaders, test_loaders = [], [], []
    train_sizes = []
    for cid in range(num_clients):
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test(
            all_X[cid], all_Y[cid], seed=seed + cid
        )
        train_sizes.append(len(X_train))
        train_loaders.append(DataLoader(TrafficDataset(X_train, Y_train),
                                        batch_size=FED_BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(TrafficDataset(X_val, Y_val),
                                       batch_size=FED_BATCH_SIZE, shuffle=False))
        test_loaders.append(DataLoader(TrafficDataset(X_test, Y_test),
                                        batch_size=FED_BATCH_SIZE, shuffle=False))

    # ── FedAvg 训练 ──
    fed_clients = [
        FederatedClient(cid, CNNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM),
                        train_loaders[cid], val_loaders[cid], test_loaders[cid],
                        criterion, lr=1e-3)
        for cid in range(num_clients)
    ]
    server = FedAvgServer(CNNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM), num_clients)
    server.set_client_data_sizes(train_sizes)

    print("\n[FedAvg Training]")
    for rnd in range(FED_ROUNDS):
        client_weights, client_losses = [], []
        for client in fed_clients:
            loss, weights = client.train_local(epochs=FED_LOCAL_EPOCHS,
                                               global_model=server.global_model)
            client_weights.append(weights)
            client_losses.append(loss)
        server.aggregate(client_weights, client_losses)

        # 聚合后在每个客户端的验证集上评估
        val_losses = []
        for client in fed_clients:
            client.model.load_state_dict(server.global_model.state_dict())
            val_losses.append(client.validate(client.val_loader))
        server.round_val_losses.append(float(np.mean(val_losses)))
        print(f"  Round {rnd+1}/{FED_ROUNDS} | Avg Train Loss: {server.round_losses[-1]:.6f} "
              f"| Avg Val Loss: {server.round_val_losses[-1]:.6f}")

    # 记录 FedAvg 最终测试指标
    fed_metrics = []
    for client in fed_clients:
        client.model.load_state_dict(server.global_model.state_dict())
        fed_metrics.append(client.test_metrics())

    # ── Independent 训练 ──
    ind_clients = [
        IndependentClient(cid, IndependentBaseModel(k=k, t=t),
                          train_loaders[cid], val_loaders[cid], test_loaders[cid],
                          criterion)
        for cid in range(num_clients)
    ]

    print("\n[Independent Training]")
    for client in ind_clients:
        loss, _ = client.train_local(epochs=15, verbose=False)
        print(f"  Client {client.client_id} | Final Train Loss: {loss:.6f}")

    ind_metrics = []
    for client in ind_clients:
        ind_metrics.append(client.test_metrics())

    # ── 输出结果 ──
    print("\n===== Results: FedAvg vs Independent =====")
    for cid in range(num_clients):
        fm = fed_metrics[cid]
        im = ind_metrics[cid]
        print(f"Client {cid}:")
        print(f"  FedAvg       - MSE={fm['mse']:.6f} RMSE={fm['rmse']:.6f} MAE={fm['mae']:.6f} MAPE={fm['mape']:.2f}%")
        print(f"  Independent  - MSE={im['mse']:.6f} RMSE={im['rmse']:.6f} MAE={im['mae']:.6f} MAPE={im['mape']:.2f}%")

    # 保存详细指标
    rows = []
    pred_rows = []
    for cid in range(num_clients):
        fm = fed_metrics[cid]
        im = ind_metrics[cid]
        rows.append({
            "seed": seed,
            "method": "FedAvg",
            "client_id": cid,
            "mse": fm["mse"],
            "rmse": fm["rmse"],
            "mape": fm["mape"],
            "mae": fm["mae"],
            "r2": fm["r2"],
        })
        rows.append({
            "seed": seed,
            "method": "Independent",
            "client_id": cid,
            "mse": im["mse"],
            "rmse": im["rmse"],
            "mape": im["mape"],
            "mae": im["mae"],
            "r2": im["r2"],
        })

        for metrics, method_name in (
            (fm, "FedAvg"),
            (im, "Independent"),
        ):
            y_pred = metrics["preds"]
            y_true = metrics["truths"]
            for sample_id in range(min(200, len(y_true))):
                pred_rows.append({
                    "seed": seed,
                    "method": method_name,
                    "client_id": cid,
                    "sample_id": sample_id,
                    "y_true": float(y_true[sample_id]),
                    "y_pred": float(y_pred[sample_id]),
                })

    client_metrics_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    seed_result_rows = []
    for method in ["FedAvg", "Independent"]:
        method_df = client_metrics_df[client_metrics_df["method"] == method]
        seed_result_rows.append({
            "experiment": "cnn_fed_base_main",
            "method": method,
            "seed": seed,
            "mse": float(method_df["mse"].mean()),
            "rmse": float(method_df["rmse"].mean()),
            "mae": float(method_df["mae"].mean()),
            "mape": float(method_df["mape"].mean()),
            "r2": float(method_df["r2"].mean()),
            "final_loss": (
                float(server.round_losses[-1]) if method == "FedAvg" else float(np.mean([client.train_losses[-1] for client in ind_clients]))
            ),
            "best_loss": (
                float(np.min(server.round_val_losses)) if method == "FedAvg" else float(np.mean([min(client.val_losses) for client in ind_clients]))
            ),
            "communication_rounds": int(FED_ROUNDS if method == "FedAvg" else 0),
            "convergence_round": (
                int(np.argmin(server.round_val_losses) + 1) if method == "FedAvg" else int(FED_ROUNDS * FED_LOCAL_EPOCHS)
            ),
        })
    return client_metrics_df, pred_df, pd.DataFrame(seed_result_rows)


def run_main_experiment(output_dir: Path, seeds: list[int], multi_seed: bool = True) -> None:
    """运行基础实验主结果：Independent vs FedAvg。"""
    ensure_output_dir(output_dir)
    target_seeds = list(seeds if multi_seed else seeds[:1])
    metrics_frames = []
    pred_frames = []
    raw_frames = []
    for seed in target_seeds:
        print(f"[main] Running seed {seed} ...")
        metric_df, pred_df, raw_df = run_single_seed_main_experiment(seed)
        metrics_frames.append(metric_df)
        pred_frames.append(pred_df)
        raw_frames.append(raw_df)

    df_metrics = pd.concat(metrics_frames, ignore_index=True)
    df_preds = pd.concat(pred_frames, ignore_index=True)
    df_raw = pd.concat(raw_frames, ignore_index=True)
    save_dataframe(df_metrics, output_dir, "main_metrics.csv")
    summary_df = (
        df_metrics.groupby("method")[["mse", "rmse", "mae", "mape", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]
    save_dataframe(summary_df, output_dir, "main_summary.csv")
    save_dataframe(df_preds, output_dir, "main_predictions.csv")
    multi_seed_summary_df = build_metric_summary_table(
        df_raw,
        group_cols=["experiment", "method"],
        metric_cols=["mae", "rmse", "mape", "r2", "final_loss", "best_loss", "communication_rounds", "convergence_round"],
    )
    improvement_df = build_improvement_rows(
        df_raw,
        experiment_name="cnn_fed_base_main",
        baseline_method="Independent",
        enhanced_method="FedAvg",
        metric_cols=["mae", "rmse", "mape", "r2"],
    )
    save_dataframe(df_raw, output_dir, "multi_seed_raw_results.csv")
    save_dataframe(multi_seed_summary_df, output_dir, "multi_seed_summary.csv")
    if not improvement_df.empty:
        save_dataframe(improvement_df, output_dir, "multi_seed_improvement_summary.csv")
    write_stability_report(
        output_dir=output_dir,
        raw_df=df_raw,
        improvement_df=improvement_df,
        experiment_name="cnn_fed_base_main",
        baseline_method="Independent",
        enhanced_method="FedAvg",
    )
    print("\n[main] Done.\n")


# ══════════════════════════════════════════════════════════
# Workflow: convergence — 联邦训练收敛曲线
# ══════════════════════════════════════════════════════════

def run_single_seed_convergence_experiment(seed: int, convergence_rounds: int = 15) -> pd.DataFrame:
    """输出基础训练收敛曲线。"""
    print("\n" + "=" * 60)
    print(f"[convergence] Running CNN convergence analysis | seed={seed}")
    print("=" * 60)

    set_global_seed(seed)
    all_X, all_Y, _ = generate_base_traffic_data(seed=seed)

    criterion = nn.MSELoss()
    num_clients = BASE_NUM_CLIENTS
    k = BASE_NUM_NODES
    t = BASE_SEQ_LEN
    train_loaders, val_loaders, test_loaders = [], [], []
    train_sizes = []
    for cid in range(num_clients):
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test(
            all_X[cid], all_Y[cid], seed=seed + cid
        )
        train_sizes.append(len(X_train))
        train_loaders.append(DataLoader(TrafficDataset(X_train, Y_train),
                                        batch_size=FED_BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(TrafficDataset(X_val, Y_val),
                                       batch_size=FED_BATCH_SIZE, shuffle=False))
        test_loaders.append(DataLoader(TrafficDataset(X_test, Y_test),
                                        batch_size=FED_BATCH_SIZE, shuffle=False))

    fed_clients = [
        FederatedClient(cid, CNNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM),
                        train_loaders[cid], val_loaders[cid], test_loaders[cid],
                        criterion, lr=1e-3)
        for cid in range(num_clients)
    ]
    server = FedAvgServer(CNNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM), num_clients)
    server.set_client_data_sizes(train_sizes)

    # 记录每轮数据
    round_rows = []

    print("\n[FedAvg Convergence Training]")
    for rnd in range(convergence_rounds):
        client_weights, client_losses = [], []
        for client in fed_clients:
            loss, weights = client.train_local(epochs=FED_LOCAL_EPOCHS,
                                               global_model=server.global_model)
            client_weights.append(weights)
            client_losses.append(loss)
        server.aggregate(client_weights, client_losses)

        # 全局模型在每个客户端验证集上的评估
        val_losses = []
        val_rmses = []
        val_maes = []
        val_mapes = []
        for client in fed_clients:
            client.model.load_state_dict(server.global_model.state_dict())
            val_loss = client.validate(client.val_loader)
            val_losses.append(float(val_loss))
            client.model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for x_val, y_val in client.val_loader:
                    x_val = x_val.to(DEVICE).float()
                    y_val = y_val.to(DEVICE).float()
                    pred, _ = client.model(x_val)
                    preds.append(pred.view(-1).detach().cpu())
                    truths.append(y_val.view(-1).detach().cpu())
            y_pred = torch.cat(preds).numpy()
            y_true = torch.cat(truths).numpy()
            diff = y_pred - y_true
            val_rmses.append(float(np.sqrt(np.mean(diff ** 2))))
            val_maes.append(float(np.mean(np.abs(diff))))
            val_mapes.append(float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), MAPE_EPS)) * 100))

        avg_val_loss = float(np.mean(val_losses))
        avg_val_rmse = float(np.mean(val_rmses))
        avg_val_mae = float(np.mean(val_maes))
        avg_val_mape = float(np.mean(val_mapes))
        server.round_val_losses.append(avg_val_loss)

        round_rows.append({
            "seed": seed,
            "round": rnd + 1,
            "method": "FedAvg",
            "avg_train_loss": float(server.round_losses[-1]),
            "avg_val_loss": avg_val_loss,
            "avg_val_rmse": avg_val_rmse,
            "avg_val_mae": avg_val_mae,
            "avg_val_mape": avg_val_mape,
        })

        print(f"  Round {rnd+1}/{convergence_rounds} | "
              f"Train Loss: {server.round_losses[-1]:.6f} | "
              f"Val RMSE: {avg_val_rmse:.6f}")

    return pd.DataFrame(round_rows)


def run_convergence_experiment(output_dir: Path, seeds: list[int], multi_seed: bool = True) -> None:
    """输出基础训练收敛曲线。"""
    ensure_output_dir(output_dir)
    target_seeds = list(seeds if multi_seed else seeds[:1])
    frames = [run_single_seed_convergence_experiment(seed) for seed in target_seeds]
    df_conv = pd.concat(frames, ignore_index=True)
    save_dataframe(df_conv, output_dir, "convergence_history.csv")
    save_dataframe(df_conv, output_dir, "multi_seed_convergence_raw.csv")
    save_dataframe(
        build_convergence_summary_table(
            df_conv,
            group_cols=["method", "round"],
            metric_cols=["avg_train_loss", "avg_val_loss", "avg_val_rmse"],
        ),
        output_dir,
        "multi_seed_convergence_summary.csv",
    )
    print("[convergence] Done.\n")


# ══════════════════════════════════════════════════════════
# 工作流调度
# ══════════════════════════════════════════════════════════

def run_project(workflow: str, output_dir: Path, seeds: list[int], multi_seed: bool = True) -> None:
    """按工作流执行 CNN 基础实验。"""
    ensure_output_dir(output_dir)
    print(f"[cnn_fed_base] workflow={workflow}, output={output_dir}")
    print(f"[cnn_fed_base] device={DEVICE}")

    if workflow in ("all", "data_viz"):
        export_base_dataset_artifacts(output_dir)

    if workflow in ("all", "main"):
        run_main_experiment(output_dir, seeds=seeds, multi_seed=multi_seed)

    if workflow in ("all", "convergence"):
        run_convergence_experiment(output_dir, seeds=seeds, multi_seed=multi_seed)

    print(f"\n[cnn_fed_base] All done. Results in: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CNN/CCN Base Federated Simulation")
    parser.add_argument(
        "--workflow",
        choices=["all", "data_viz", "main", "convergence"],
        default="all",
        help="Workflow to execute (default: all).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for exported experiment artifacts.",
    )
    parser.add_argument("--multi_seed", type=str, default="True", help="Whether to run multiple seeds.")
    parser.add_argument("--seeds", type=str, default="42,2024,2025,2026,3407", help="Comma-separated random seeds.")
    parser.add_argument("--single_seed", type=int, default=42, help="Single seed used when --multi_seed False.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    """程序主入口。"""
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else SIMULATION_RESULTS_ROOT / "cnn_fed_base"
    multi_seed = parse_bool_flag(args.multi_seed)
    seeds = parse_seed_list(args.seeds) if multi_seed else [int(args.single_seed)]
    run_project(args.workflow, output_dir, seeds=seeds, multi_seed=multi_seed)


if __name__ == "__main__":
    main()
