# -*- coding: utf-8 -*-
"""
GCN 仿真实验独立工程。

本文件实现了基于 GCN（图卷积网络）主干的联邦仿真实验，包含以下内容：
1. 联邦总览实验：比较联邦训练与独立训练在异构客户端上的误差表现；
2. 消融实验：比较 GCN-LSTM-Attention 及其裁剪变体的性能差异；
3. 可视化输出：将指标对比图、消融分析图、指标表格与运行日志统一写入输出目录。

主要依赖库：
- PyTorch：模型定义、训练与联邦聚合；
- NumPy / pandas：数值处理与表格整理；
- matplotlib / seaborn：图表渲染与保存。

输入输出场景：
- 输入：脚本内部生成的异构多客户端合成时序数据；
- 输出：PNG 图像、CSV 指标文件、TXT 运行日志。
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

# 使用无界面后端，避免在命令行或服务器环境中弹出图形窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# 关闭交互模式，统一采用“绘制后直接保存”的工作流。
plt.ioff()

# 当前脚本所在目录。
SCRIPT_DIR = Path(__file__).resolve().parent
# 项目根目录，用于构造统一的结果输出根目录。
PROJECT_ROOT = SCRIPT_DIR.parent
# 统一的结果根目录。
RESULTS_ROOT = PROJECT_ROOT / "results"
# 仿真实验主目录，所有独立仿真实验均在此目录下建立各自子目录。
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
# 自动选择可用计算设备。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 工程名作为统一输出文件名前缀。
PROJECT_NAME = "gcn-base"
# GCN 实验默认输出目录。
DEFAULT_OUTPUT_DIR = SIMULATION_RESULTS_ROOT / PROJECT_NAME
# Windows 常见非法路径字符，用于阻止错误路径写入。
INVALID_PATH_CHARS = set('<>:"|?*')


def configure_plot_style() -> None:
    """配置全局绘图主题与版式参数。"""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="DejaVu Sans",
        rc={
            "axes.unicode_minus": False,
            "figure.titlesize": 18,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
        },
    )


def set_global_seed(seed: int) -> None:
    """设置全局随机种子，提升结果可复现性。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dir(output_dir: Path) -> Path:
    """确保输出目录存在并返回该目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_output_subdir(output_subdir: str) -> None:
    """校验相对子目录参数是否合法。"""
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
            raise ValueError(
                "The output directory contains illegal characters for the current operating system."
            )
        if part.endswith(" ") or part.endswith("."):
            raise ValueError("Directory names must not end with a space or dot.")


def resolve_output_dir(output_subdir: Optional[str] = None) -> Path:
    """解析并创建安全的输出目录。"""
    ensure_output_dir(SIMULATION_RESULTS_ROOT)
    relative_subdir = output_subdir or PROJECT_NAME
    validate_output_subdir(relative_subdir)

    resolved_output_dir = (SIMULATION_RESULTS_ROOT / relative_subdir).resolve()
    resolved_root = SIMULATION_RESULTS_ROOT.resolve()
    if resolved_output_dir != resolved_root and resolved_root not in resolved_output_dir.parents:
        raise ValueError("The resolved output directory escapes the simulation results root.")

    return ensure_output_dir(resolved_output_dir)


def build_output_file_name(
    workflow_name: str,
    artifact_name: str,
    extension: str,
) -> str:
    """构造统一命名规则的输出文件名。"""
    return f"{PROJECT_NAME}_{workflow_name}_{artifact_name}.{extension}"


def save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    """保存图像并关闭图对象。"""
    output_path = ensure_output_dir(output_dir) / file_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path


def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    """保存数据表为 CSV 文件。"""
    output_path = ensure_output_dir(output_dir) / file_name
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved table: {output_path}")
    return output_path


class TeeStream:
    """将标准输出同时写入多个目标流。"""

    def __init__(self, *streams):
        """初始化输出流集合。"""
        self.streams = streams

    def write(self, data: str) -> int:
        """将文本写入所有流对象。"""
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        """刷新所有流对象。"""
        for stream in self.streams:
            stream.flush()


def unpack_model_output(model_output):
    """统一解析模型输出。

    返回:
        `(prediction, attention_weights)` 二元组。
    """
    if isinstance(model_output, tuple):
        return model_output
    return model_output, None


def stability_stats(arr):
    """计算跨客户端稳定性统计量。"""
    arr = np.array(arr, dtype=float)
    std = float(arr.std())
    gap = float(arr.max() - arr.min())
    mean = float(arr.mean())
    cv = float(std / (mean + 1e-12))
    return std, gap, cv


def print_summary_table(results_summary: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """打印并返回消融实验汇总表。"""
    df_sum = (
        pd.DataFrame(results_summary)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("Model")
    )
    df_sum = df_sum[
        [
            "Model",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "mse_mean",
            "mse_std",
        ]
    ]
    print("\n=== Final Test Metrics Summary (mean ± std across clients) ===")
    print(df_sum.to_string(index=False))
    return df_sum


class AdaptiveSwish(nn.Module):
    """带可学习缩放因子的 Swish 激活函数。"""

    def __init__(self, trainable: bool = True):
        """初始化激活函数参数。"""
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        """执行自适应 Swish 非线性变换。"""
        return x * torch.sigmoid(self.beta * x)


class SimpleGCNLayer(nn.Module):
    """基础图卷积层。

    本层先利用归一化邻接矩阵传播节点特征，再通过线性层完成通道映射。
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        """初始化图卷积层。"""
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, a_norm):
        """执行单层图卷积。

        参数:
            x: 形状为 `[B, K, F]` 的节点特征张量。
            a_norm: 形状为 `[K, K]` 的归一化邻接矩阵。
        """
        # `einsum` 实现对每个 batch 共享同一邻接矩阵的消息传播。
        ax = torch.einsum("ij,bjf->bif", a_norm, x)
        return self.lin(ax)


class GCNEncoder(nn.Module):
    """GCN 编码器。

    编码器通过可学习邻接矩阵显式构建节点之间的拓扑关系，再经过两层图卷积
    完成节点状态更新与信息传递，最后对节点维进行池化得到图级表示。
    """

    def __init__(self, k: int, t: int, hidden_dim: int = 128):
        """初始化 GCN 编码器。

        参数:
            k: 节点数或观测变量数。
            t: 每个节点的时间序列长度。
            hidden_dim: 图编码器隐层维度。
        """
        super().__init__()
        self.k = k
        self.t = t
        self.hidden_dim = hidden_dim

        # 先将每个节点长度为 T 的时间序列投影到统一隐藏空间。
        self.node_proj = nn.Sequential(
            nn.Linear(t, hidden_dim),
            nn.LayerNorm(hidden_dim),
            AdaptiveSwish(),
        )
        # 两层图卷积负责执行节点间信息传递与状态迭代更新。
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()
        # 可学习邻接矩阵是拓扑构建核心，用于让模型自动发现变量关联。
        self.a_param = nn.Parameter(torch.randn(k, k) * 0.01)

    def _normalize_adj(self, a):
        """对邻接矩阵执行非负化、自环补充和对称归一化。"""
        a = torch.relu(a)
        a = a + torch.eye(self.k, device=a.device, dtype=a.dtype)
        deg = a.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        return d_inv_sqrt @ a @ d_inv_sqrt

    def forward(self, x):
        """执行 GCN 编码流程。"""
        x = x.to(dtype=torch.float32)
        x = self.node_proj(x)
        a_norm = self._normalize_adj(self.a_param)

        # 第一层图卷积：聚合邻居信息并更新节点状态。
        h = self.gcn1(x, a_norm)
        h = self.norm1(h)
        h = self.act(h)

        # 第二层图卷积：进一步扩展信息传播范围。
        h = self.gcn2(h, a_norm)
        h = self.norm2(h)
        h = self.act(h)
        # 对所有节点做平均池化，得到图级全局表示。
        return h.mean(dim=1)


class WeakModel(nn.Module):
    """独立训练基线模型。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 16):
        """初始化弱基线模型。"""
        super().__init__()
        self.k = k
        self.t = t
        self.simple_extractor = nn.Sequential(
            nn.Linear(k * t, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """执行前向传播，返回预测值与空注意力权重。"""
        x = x.to(dtype=torch.float32)
        batch_size, k, t = x.shape
        x = x.view(batch_size, k * t)
        x = self.simple_extractor(x)
        return self.fc(x), None


class GCNOverviewModel(nn.Module):
    """GCN 总览实验主模型。

    模型由三部分组成：
    1. GCN 分支：基于可学习图拓扑执行节点间消息传递；
    2. BiLSTM 分支：建模全局时序依赖；
    3. 多头注意力融合：自适应整合图结构特征与时序特征。
    """

    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        """初始化 GCN 总览模型。"""
        super().__init__()
        self.gcn_encoder = GCNEncoder(k=k, t=t, hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=k,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """执行 GCN + BiLSTM + Attention 前向传播。"""
        x = x.to(dtype=torch.float32)
        x_gcn = self.gcn_encoder(x)

        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)

        feat_seq = torch.stack([x_gcn, x_lstm], dim=1)
        attn_output, attn_weights = self.multihead_attn(feat_seq, feat_seq, feat_seq)
        attn_output = self.attn_norm(attn_output + feat_seq)
        x_fused = attn_output.mean(dim=1)
        return self.regression_head(x_fused), attn_weights


class LSTMAttentionHetero(nn.Module):
    """移除 GCN 图结构分支后的时序注意力模型。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        """初始化 LSTM-Attention 变体。"""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=k,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """执行仅保留时序分支的前向传播。"""
        x = x.to(dtype=torch.float32)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)
        feat_seq = x_lstm.unsqueeze(1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


class GCNAblationFull(nn.Module):
    """GCN 消融实验中的完整模型。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        """初始化 GCN-LSTM-Attention 完整模型。"""
        super().__init__()
        self.gcn_encoder = GCNEncoder(k=k, t=t, hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=k,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """执行完整消融模型前向传播。"""
        x = x.to(dtype=torch.float32)
        x_gcn = self.gcn_encoder(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)
        feat_seq = torch.stack([x_gcn, x_lstm], dim=1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


class GCNAblationLSTM(nn.Module):
    """移除注意力模块后的 GCN 消融模型。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 128):
        """初始化 GCN-LSTM 变体。"""
        super().__init__()
        self.gcn_encoder = GCNEncoder(k=k, t=t, hidden_dim=hidden_dim)
        self.lstm = nn.LSTM(
            input_size=k,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            AdaptiveSwish(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """执行 GCN-LSTM 变体前向传播。"""
        x = x.to(dtype=torch.float32)
        x_gcn = self.gcn_encoder(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)
        fused = self.fuse(torch.cat([x_gcn, x_lstm], dim=1))
        return self.head(fused), None


class GCNAblationAttention(nn.Module):
    """移除 LSTM 分支后的 GCN-Attention 变体。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        """初始化 GCN-Attention 变体。"""
        super().__init__()
        self.gcn_encoder = GCNEncoder(k=k, t=t, hidden_dim=hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """执行仅保留图结构主干的前向传播。"""
        x = x.to(dtype=torch.float32)
        x_gcn = self.gcn_encoder(x)
        feat_seq = x_gcn.unsqueeze(1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


class OverviewHeterogeneousDataset(Dataset):
    """总览实验使用的异构客户端数据集。"""

    def __init__(
        self,
        client_id: int,
        num_samples: int,
        k: int,
        t: int,
        noise: float = 0.1,
    ):
        """构造异构客户端样本。"""
        self.x = np.random.randn(num_samples, k, t)
        base_feature = self.x[:, :, t // 4 : t * 3 // 4].mean(axis=(1, 2))
        if client_id == 0:
            self.y = (
                0.6 * np.sin(base_feature)
                + 0.4 * np.sin(self.x[:, :, : t // 2].mean(axis=(1, 2)))
                + noise * np.random.randn(num_samples)
            )
        elif client_id == 1:
            self.y = (
                0.6 * np.sin(base_feature)
                + 0.4 * np.cos(self.x[:, :, t // 2 :].mean(axis=(1, 2)))
                + noise * np.random.randn(num_samples)
            )
        else:
            self.y = (
                0.6 * np.sin(base_feature)
                + 0.4 * np.tanh(self.x.max(axis=(1, 2)))
                + noise * np.random.randn(num_samples)
            )

    def __len__(self):
        """返回数据集长度。"""
        return len(self.x)

    def __getitem__(self, idx):
        """返回单个样本及其标签。"""
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class AblationHeterogeneousDataset(Dataset):
    """消融实验使用的异构客户端数据集。"""

    def __init__(
        self,
        client_id: int,
        num_samples: int,
        k: int,
        t: int,
        noise: float = 0.1,
    ):
        """构造消融实验数据。"""
        self.x = np.random.randn(num_samples, k, t).astype(np.float32)
        base_feature = self.x[:, :, t // 4 : t * 3 // 4].mean(axis=(1, 2))
        if client_id == 0:
            y = 0.6 * np.sin(base_feature) + 0.4 * np.sin(
                self.x[:, :, : t // 2].mean(axis=(1, 2))
            )
        elif client_id == 1:
            y = 0.6 * np.sin(base_feature) + 0.4 * np.cos(
                self.x[:, :, t // 2 :].mean(axis=(1, 2))
            )
        else:
            y = 0.6 * np.sin(base_feature) + 0.4 * np.tanh(
                self.x.max(axis=(1, 2))
            )
        y = y + noise * np.random.randn(num_samples).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        """返回数据集长度。"""
        return len(self.x)

    def __getitem__(self, idx):
        """返回单个样本及其标签。"""
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class FederatedClient:
    """联邦客户端封装类。"""

    def __init__(
        self,
        client_id,
        model,
        train_loader,
        test_loader,
        criterion,
        lr: float = 1e-3,
    ):
        """初始化联邦客户端。"""
        self.client_id = client_id
        self.model = model.to(DEVICE).float()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.9,
        )
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """执行单个本地训练 epoch。"""
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().squeeze()
            self.optimizer.zero_grad()
            pred, _ = unpack_model_output(self.model(x))
            loss = self.criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        avg_loss = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """在本地测试集上计算验证损失。"""
        self.model.eval()
        total_loss = 0.0
        for x, y in self.test_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().squeeze()
            pred, _ = unpack_model_output(self.model(x))
            total_loss += self.criterion(pred.squeeze(), y).item() * x.shape[0]
        avg_loss = total_loss / len(self.test_loader.dataset)
        self.val_losses.append(avg_loss)
        self.scheduler.step()
        return avg_loss

    def train_local(
        self,
        epochs: int = 5,
        global_model=None,
        verbose: bool = False,
        prefix: str = "Local",
    ):
        """执行客户端本地训练并返回权重副本。"""
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            if verbose:
                print(
                    f"  {prefix} epoch {epoch + 1}/{epochs}, "
                    f"Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}"
                )
        return float(self.train_losses[-1]), copy.deepcopy(self.model.state_dict())

    @torch.no_grad()
    def test_predictions(self):
        """获取测试集预测结果及注意力统计。"""
        self.model.eval()
        preds, truths, att_weights = [], [], []
        for x, y in self.test_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().squeeze()
            pred, weights = unpack_model_output(self.model(x))
            preds.extend(np.atleast_1d(pred.squeeze().cpu().numpy()).tolist())
            truths.extend(np.atleast_1d(y.cpu().numpy()).tolist())
            if weights is not None:
                att_weights.append(weights.cpu().numpy())

        preds = np.array(preds)
        truths = np.array(truths)
        mse = float(np.mean((preds - truths) ** 2))
        mae = float(np.mean(np.abs(preds - truths)))

        att_mean = None
        if att_weights:
            att_weights = np.concatenate(att_weights, axis=0)
            att_mean = np.mean(att_weights, axis=0)
        return {
            "mse": mse,
            "mae": mae,
            "preds": preds,
            "truths": truths,
            "att_weights": att_mean,
        }

    @torch.no_grad()
    def test_metrics(self):
        """计算测试集上的 MSE、RMSE 与 MAE。"""
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float().squeeze()
            pred, _ = unpack_model_output(self.model(x))
            preds.append(pred.squeeze())
            truths.append(y)
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        diff = preds - truths
        mse = float((diff ** 2).mean().item())
        mae = float(diff.abs().mean().item())
        rmse = float(np.sqrt(mse))
        return {"mse": mse, "rmse": rmse, "mae": mae}


class IndependentClient(FederatedClient):
    """独立训练客户端。"""

    def __init__(self, client_id, model, train_loader, test_loader, criterion):
        """初始化独立训练客户端。"""
        super().__init__(
            client_id,
            model,
            train_loader,
            test_loader,
            criterion,
            lr=0.02,
        )

    def train_local(self, epochs: int = 2, verbose: bool = False):
        """执行独立训练。"""
        return super().train_local(
            epochs=epochs,
            global_model=None,
            verbose=verbose,
            prefix="Independent",
        )


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
                client_weight = client_weights[idx][key].to(
                    DEVICE,
                    dtype=torch.float32,
                )
                new_dict[key] += client_weight * torch.tensor(
                    float(weights[idx]),
                    device=DEVICE,
                    dtype=torch.float32,
                )

        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(client_losses)))
        return self.global_model.state_dict()

def plot_overview_figure(
    fed_metrics,
    weak_metrics,
    server,
    fed_clients,
    output_dir: Path,
    file_name: str,
) -> pd.DataFrame:
    """绘制总览实验图并返回指标表。"""
    client_labels = [f"Client {i}" for i in range(len(fed_metrics))]
    fed_mse = [m["mse"] for m in fed_metrics]
    fed_rmse = [np.sqrt(m["mse"]) for m in fed_metrics]
    fed_mae = [m["mae"] for m in fed_metrics]

    weak_mse = [m["mse"] for m in weak_metrics]
    weak_rmse = [np.sqrt(m["mse"]) for m in weak_metrics]
    weak_mae = [m["mae"] for m in weak_metrics]

    df_metrics = pd.DataFrame(
        {
            "Client": client_labels * 2,
            "Method": ["GCN-FedAvg"] * len(client_labels)
            + ["Independent"] * len(client_labels),
            "MSE": fed_mse + weak_mse,
            "RMSE": fed_rmse + weak_rmse,
            "MAE": fed_mae + weak_mae,
        }
    )
    df_long = df_metrics.melt(
        id_vars=["Client", "Method"],
        value_vars=["MSE", "RMSE", "MAE"],
        var_name="Metric",
        value_name="Value",
    )

    round_axis = np.arange(1, len(server.round_losses) + 1)
    df_global = pd.DataFrame(
        {"Round": round_axis, "AvgTrainLoss": server.round_losses}
    )

    df_client_val = pd.concat(
        [
            pd.DataFrame(
                {
                    "Round": np.arange(1, len(client.val_losses) + 1),
                    "Client": f"Client {client.client_id}",
                    "ValLoss": client.val_losses,
                }
            )
            for client in fed_clients
        ],
        ignore_index=True,
    )

    fed_mse_std, fed_mse_gap, fed_mse_cv = stability_stats(fed_mse)
    weak_mse_std, weak_mse_gap, weak_mse_cv = stability_stats(weak_mse)
    fed_mae_std, fed_mae_gap, fed_mae_cv = stability_stats(fed_mae)
    weak_mae_std, weak_mae_gap, weak_mae_cv = stability_stats(weak_mae)

    df_stability = pd.DataFrame(
        {
            "Statistic": [
                "MSE-STD",
                "MSE-GAP",
                "MSE-CV",
                "MAE-STD",
                "MAE-GAP",
                "MAE-CV",
            ]
            * 2,
            "Value": [
                fed_mse_std,
                fed_mse_gap,
                fed_mse_cv,
                fed_mae_std,
                fed_mae_gap,
                fed_mae_cv,
                weak_mse_std,
                weak_mse_gap,
                weak_mse_cv,
                weak_mae_std,
                weak_mae_gap,
                weak_mae_cv,
            ],
            "Method": ["GCN-FedAvg"] * 6 + ["Independent"] * 6,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    sns.barplot(
        data=df_long[df_long["Metric"] == "MSE"],
        x="Client",
        y="Value",
        hue="Method",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("(a) MSE Comparison")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend(title="Method", loc="lower right", frameon=True)

    sns.barplot(
        data=df_long[df_long["Metric"] == "RMSE"],
        x="Client",
        y="Value",
        hue="Method",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("(b) RMSE Comparison")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].legend(title="Method", loc="lower right", frameon=True)

    sns.barplot(
        data=df_long[df_long["Metric"] == "MAE"],
        x="Client",
        y="Value",
        hue="Method",
        ax=axes[0, 2],
    )
    axes[0, 2].set_title("(c) MAE Comparison")
    axes[0, 2].set_xlabel("")
    axes[0, 2].set_ylabel("MAE")
    axes[0, 2].legend(title="Method", loc="lower right", frameon=True)

    sns.lineplot(
        data=df_global,
        x="Round",
        y="AvgTrainLoss",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("(d) FedAvg Convergence (Global)")
    axes[1, 0].set_xlabel("Communication Round")
    axes[1, 0].set_ylabel("Avg Train Loss")

    sns.lineplot(
        data=df_client_val,
        x="Round",
        y="ValLoss",
        hue="Client",
        marker="o",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("(e) Client Validation Convergence (FedAvg)")
    axes[1, 1].set_xlabel("Communication Round")
    axes[1, 1].set_ylabel("Validation Loss")
    axes[1, 1].legend(title="Client")

    sns.barplot(
        data=df_stability,
        x="Statistic",
        y="Value",
        hue="Method",
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("(f) Cross-Client Error Stability (Dispersion)")
    axes[1, 2].set_xlabel("")
    axes[1, 2].set_ylabel("Value")
    axes[1, 2].tick_params(axis="x", rotation=30)
    axes[1, 2].legend(title="Method")

    plt.tight_layout()
    save_figure(fig, output_dir, file_name)
    return df_metrics


def plot_ablation_figure(
    df_conv: pd.DataFrame,
    df_stab: pd.DataFrame,
    df_delta: pd.DataFrame,
    client_labels,
    rounds: int,
    output_dir: Path,
    file_name: str,
):
    """绘制消融实验图。"""
    heat = df_stab.pivot_table(
        index="Client",
        columns="Model",
        values="rmse",
        aggfunc="mean",
    )
    heat = heat.reindex(index=client_labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    (ax1, ax2), (ax3, ax4) = axes

    for name in df_conv["Model"].unique():
        sub = df_conv[df_conv["Model"] == name].sort_values("Round")
        x = sub["Round"].to_numpy(dtype=int)
        y = sub["TestRMSE_mean"].to_numpy(dtype=float)
        s = sub["TestRMSE_std"].to_numpy(dtype=float)
        ax1.plot(x, y, marker="o", linewidth=2, label=name)
        ax1.fill_between(x, y - s, y + s, alpha=0.15)

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Test RMSE")
    ax1.set_title("(a) Convergence of Test RMSE (mean ± std)")
    ax1.set_xticks(np.arange(1, rounds + 1, dtype=int))
    ax1.legend(frameon=True)

    sns.violinplot(data=df_stab, x="Model", y="rmse", inner=None, cut=0, ax=ax2)
    sns.stripplot(
        data=df_stab,
        x="Model",
        y="rmse",
        color="k",
        size=4,
        alpha=0.6,
        ax=ax2,
    )
    ax2.set_xlabel("Model Variant")
    ax2.set_ylabel("Final Test RMSE")
    ax2.set_title("(b) Client-level Stability (Final RMSE)")
    ax2.tick_params(axis="x", rotation=15)

    if len(df_delta) > 0:
        df_delta_melt = df_delta.melt(
            id_vars=["Model"],
            var_name="Metric",
            value_name="DeltaPercent",
        )
        sns.barplot(
            data=df_delta_melt,
            x="Model",
            y="DeltaPercent",
            hue="Metric",
            ax=ax3,
        )
        ax3.axhline(0, linewidth=1)
        ax3.set_xlabel("Ablation Variant")
        ax3.set_ylabel("Relative Change (%)")
        ax3.set_title("(c) Relative Change Compared with Full Model")
        ax3.tick_params(axis="x", rotation=15)
        ax3.legend(title="Metric", frameon=True)
    else:
        ax3.axis("off")

    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Final Test RMSE"},
        ax=ax4,
    )
    ax4.set_xlabel("Model Variant")
    ax4.set_ylabel("Client")
    ax4.set_title("(d) Client × Model Heatmap (Final RMSE)")
    ax4.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_figure(fig, output_dir, file_name)


def run_fedavg_ablation(
    *,
    workflow_name: str,
    seed: int,
    num_clients: int,
    k: int,
    t: int,
    samples_per_client,
    num_rounds: int,
    local_epochs: int,
    full_name: str,
    variants: "OrderedDict[str, Callable[[], nn.Module]]",
    output_dir: Path,
    figure_name: str,
    metrics_file_name: str,
) -> Dict[str, Dict[str, float]]:
    """执行样本量加权 FedAvg 消融实验。"""
    set_global_seed(seed)
    criterion = nn.MSELoss()
    split_gen = torch.Generator().manual_seed(seed)

    def build_loaders():
        """为每个客户端构建训练集与测试集加载器。"""
        train_loaders, test_loaders = [], []
        for cid in range(num_clients):
            dataset = AblationHeterogeneousDataset(
                client_id=cid,
                num_samples=samples_per_client[cid],
                k=k,
                t=t,
            )
            train_size = int(0.8 * len(dataset))
            train_data, test_data = random_split(
                dataset,
                [train_size, len(dataset) - train_size],
                generator=split_gen,
            )
            loader_gen = torch.Generator().manual_seed(seed + cid)
            train_loader = DataLoader(
                train_data,
                batch_size=8,
                shuffle=True,
                generator=loader_gen,
            )
            test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        return train_loaders, test_loaders

    def eval_global_on_clients(global_model, clients):
        """在所有客户端上评估同一全局模型。"""
        per_client = []
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            per_client.append(client.test_metrics())
        return per_client

    results_client = {}
    results_summary = {}
    histories = {}

    print(f"\n===== {workflow_name} =====")
    for name, ctor in variants.items():
        train_loaders, test_loaders = build_loaders()
        clients = [
            FederatedClient(
                cid,
                ctor(),
                train_loaders[cid],
                test_loaders[cid],
                criterion,
                lr=1e-3,
            )
            for cid in range(num_clients)
        ]
        server = FedAvgServer(ctor(), num_clients)
        server.set_client_data_sizes(samples_per_client)

        hist_train_client, hist_train_mean, hist_train_std = [], [], []
        hist_test_client, hist_test_mean, hist_test_std = [], [], []

        print(f"\nStart FedAvg Training: {name}")
        for rnd in range(num_rounds):
            print(f"  Round {rnd + 1}/{num_rounds}")
            client_weights, client_losses = [], []
            for client in clients:
                loss, weights = client.train_local(
                    epochs=local_epochs,
                    global_model=server.global_model,
                    verbose=False,
                )
                client_weights.append(weights)
                client_losses.append(float(loss))
                print(f"    Client {client.client_id} | Local avg MSE: {loss:.6f}")

            server.aggregate(client_weights, client_losses)
            hist_train_client.append(client_losses)
            hist_train_mean.append(float(np.mean(client_losses)))
            hist_train_std.append(float(np.std(client_losses, ddof=0)))

            per_client_metrics = eval_global_on_clients(server.global_model, clients)
            per_client_rmse = np.array(
                [m["rmse"] for m in per_client_metrics],
                dtype=float,
            )
            hist_test_client.append(per_client_rmse.tolist())
            hist_test_mean.append(float(per_client_rmse.mean()))
            hist_test_std.append(float(per_client_rmse.std(ddof=0)))
            print(
                f"    Global Test RMSE mean: {hist_test_mean[-1]:.6f} "
                f"(std {hist_test_std[-1]:.6f})"
            )

        final_list = eval_global_on_clients(server.global_model, clients)
        df_final = pd.DataFrame(final_list)
        df_final["cid"] = list(range(num_clients))
        df_final = df_final.sort_values("cid").reset_index(drop=True)

        results_client[name] = df_final
        results_summary[name] = {
            "mse_mean": float(df_final["mse"].mean()),
            "mse_std": float(df_final["mse"].std(ddof=0)),
            "rmse_mean": float(df_final["rmse"].mean()),
            "rmse_std": float(df_final["rmse"].std(ddof=0)),
            "mae_mean": float(df_final["mae"].mean()),
            "mae_std": float(df_final["mae"].std(ddof=0)),
        }
        histories[name] = {
            "train_mean": hist_train_mean,
            "train_std": hist_train_std,
            "train_client": hist_train_client,
            "test_mean": hist_test_mean,
            "test_std": hist_test_std,
            "test_client": hist_test_client,
        }

    conv_rows = []
    for name, hist in histories.items():
        for rnd in range(num_rounds):
            conv_rows.append(
                {
                    "Model": name,
                    "Round": rnd + 1,
                    "TestRMSE_mean": hist["test_mean"][rnd],
                    "TestRMSE_std": hist["test_std"][rnd],
                }
            )
    df_conv = pd.DataFrame(conv_rows)

    stab_rows = []
    client_labels = [f"Client {idx}" for idx in range(num_clients)]
    for name, df_pc in results_client.items():
        for _, row in df_pc.iterrows():
            stab_rows.append(
                {
                    "Model": name,
                    "Client": f"Client {int(row['cid'])}",
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "mse": float(row["mse"]),
                }
            )
    df_stab = pd.DataFrame(stab_rows)

    full = results_summary[full_name]
    delta_rows = []
    for name, summary in results_summary.items():
        if name == full_name:
            continue
        delta_rows.append(
            {
                "Model": name,
                "Delta_RMSE_%": (
                    (summary["rmse_mean"] - full["rmse_mean"])
                    / (full["rmse_mean"] + 1e-12)
                    * 100.0
                ),
                "Delta_MAE_%": (
                    (summary["mae_mean"] - full["mae_mean"])
                    / (full["mae_mean"] + 1e-12)
                    * 100.0
                ),
            }
        )
    df_delta = pd.DataFrame(delta_rows)

    plot_ablation_figure(
        df_conv,
        df_stab,
        df_delta,
        client_labels,
        num_rounds,
        output_dir,
        figure_name,
    )
    summary_df = print_summary_table(results_summary)
    save_dataframe(summary_df, output_dir, metrics_file_name)
    return results_summary


def run_overview_experiment(output_dir: Path) -> None:
    """运行 GCN 总览实验。"""
    seed = 42
    num_rounds = 5
    local_epochs = 5
    num_clients = 3
    k, t = 5, 24
    samples_per_client = [50, 80, 120]
    criterion = nn.MSELoss()

    set_global_seed(seed)

    split_gen = torch.Generator()
    split_gen.manual_seed(seed)

    fed_clients = []
    weak_clients = []
    for client_id in range(num_clients):
        dataset = OverviewHeterogeneousDataset(
            client_id=client_id,
            num_samples=samples_per_client[client_id],
            k=k,
            t=t,
        )
        train_size = int(0.8 * len(dataset))
        train_data, test_data = random_split(
            dataset,
            [train_size, len(dataset) - train_size],
            generator=split_gen,
        )

        loader_gen = torch.Generator()
        loader_gen.manual_seed(seed + client_id)
        train_loader = DataLoader(
            train_data,
            batch_size=8,
            shuffle=True,
            generator=loader_gen,
        )
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

        fed_clients.append(
            FederatedClient(
                client_id,
                GCNOverviewModel(k=k, t=t),
                train_loader,
                test_loader,
                criterion,
                lr=1e-3,
            )
        )
        weak_clients.append(
            IndependentClient(
                client_id,
                WeakModel(k=k, t=t),
                train_loader,
                test_loader,
                criterion,
            )
        )

    server = FedAvgServer(GCNOverviewModel(k=k, t=t), num_clients)
    server.set_client_data_sizes(samples_per_client)

    print("\n===== GCN Overview =====")
    for rnd in range(num_rounds):
        print(f"[overview] round {rnd + 1}/{num_rounds}")
        client_weights, client_losses = [], []
        for client in fed_clients:
            loss, weights = client.train_local(
                epochs=local_epochs,
                global_model=server.global_model,
                verbose=False,
            )
            client_weights.append(weights)
            client_losses.append(loss)
            print(f"  Client {client.client_id} | Local avg MSE: {loss:.4f}")
        server.aggregate(client_weights, client_losses)
        print(f"  Round average federated loss: {server.round_losses[-1]:.4f}")

    print("[overview] independent baselines")
    for client in weak_clients:
        loss, _ = client.train_local(epochs=2, verbose=False)
        print(f"  Client {client.client_id} | Independent avg MSE: {loss:.4f}")

    fed_metrics = [client.test_predictions() for client in fed_clients]
    weak_metrics = [client.test_predictions() for client in weak_clients]

    print("\n===== Performance Comparison =====")
    for idx in range(num_clients):
        print(f"Client {idx}:")
        print(
            f"  GCN-FedAvg   - MSE: {fed_metrics[idx]['mse']:.4f}, "
            f"MAE: {fed_metrics[idx]['mae']:.4f}"
        )
        print(
            f"  Independent - MSE: {weak_metrics[idx]['mse']:.4f}, "
            f"MAE: {weak_metrics[idx]['mae']:.4f}"
        )
        if fed_metrics[idx]["att_weights"] is not None:
            print(
                "  Mean attention weight: "
                f"{np.round(fed_metrics[idx]['att_weights'].mean(), 4)}"
            )

    overview_df = plot_overview_figure(
        fed_metrics,
        weak_metrics,
        server,
        fed_clients,
        output_dir,
        build_output_file_name("overview", "figure", "png"),
    )
    save_dataframe(
        overview_df,
        output_dir,
        build_output_file_name("overview", "metrics", "csv"),
    )


def run_ablation_experiment(output_dir: Path) -> Dict[str, Dict[str, float]]:
    """运行 GCN 消融实验。"""
    return run_fedavg_ablation(
        workflow_name="GCN FedAvg Ablation",
        seed=42,
        num_clients=3,
        k=5,
        t=24,
        samples_per_client=[50, 80, 120],
        num_rounds=5,
        local_epochs=5,
        full_name="GCN-LSTM-Attention",
        variants=OrderedDict(
            [
                (
                    "GCN-LSTM-Attention",
                    lambda: GCNAblationFull(k=5, t=24, hidden_dim=128, num_heads=4),
                ),
                (
                    "GCN-LSTM",
                    lambda: GCNAblationLSTM(k=5, t=24, hidden_dim=128),
                ),
                (
                    "LSTM-Attention",
                    lambda: LSTMAttentionHetero(
                        k=5,
                        t=24,
                        hidden_dim=128,
                        num_heads=4,
                    ),
                ),
                (
                    "GCN-Attention",
                    lambda: GCNAblationAttention(
                        k=5,
                        t=24,
                        hidden_dim=128,
                        num_heads=4,
                    ),
                ),
            ]
        ),
        output_dir=output_dir,
        figure_name=build_output_file_name("ablation", "figure", "png"),
        metrics_file_name=build_output_file_name("ablation", "metrics", "csv"),
    )


def run_project(workflow: str, output_dir: Path) -> None:
    """按工作流执行整个 GCN 工程。"""
    ensure_output_dir(output_dir)
    log_path = output_dir / build_output_file_name("run", "log", "txt")
    with log_path.open("w", encoding="utf-8") as log_handle, redirect_stdout(
        TeeStream(sys.stdout, log_handle)
    ):
        configure_plot_style()
        print(f"[setup] Using device: {DEVICE}")
        print(f"[setup] Writing experiment log: {log_path}")

        if workflow in ("all", "overview"):
            run_overview_experiment(output_dir)

        if workflow in ("all", "ablation"):
            run_ablation_experiment(output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Standalone GCN simulation project.")
    parser.add_argument(
        "--workflow",
        choices=["all", "overview", "ablation"],
        default="all",
        help="Workflow to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_NAME,
        help=(
            "Relative subdirectory under results/simulation_experiments used "
            "to save generated outputs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    """程序主入口。"""
    args = parse_args(argv)
    run_project(args.workflow, resolve_output_dir(args.output_dir))


if __name__ == "__main__":
    main()
