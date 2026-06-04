# -*- coding: utf-8 -*-
"""
GCN 基础联邦仿真实验。

本文件实现基于 GCN-BiLSTM-Attention 的联邦仿真基础实验，包含：
1. data_viz: 基础数据集可视化（含图结构可视化）；
2. main: Independent / FedAvg 主结果对比（MSE、RMSE、MAE、MAPE）；
3. convergence: 联邦训练收敛曲线；
4. all: 依次运行上述全部工作流。

与 cnn_fed_base.py 共享相同的数据生成逻辑和随机种子，保证基础对比公平。
区别在于 GCN 使用图卷积代替 CNN 作为空间建模模块，并额外使用邻接矩阵。

主要依赖：PyTorch, NumPy, pandas, matplotlib。
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
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

plt.ioff()

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

METHOD_PALETTE = {
    "Independent": "#4C72B0",
    "FedAvg": "#DD8452",
    "GCN-FedAvg": "#DD8452",
    "GCN-Proposed": "#55A868",
    "Proposed": "#55A868",
    "Loss-weighted": "#C44E52",
    "Data-loss weighted": "#8172B3",
    "Similarity-aware": "#937860",
}
CLIENT_PALETTE = sns.color_palette("tab10")
SPLIT_PALETTE = ["#55A868", "#DD8452", "#C44E52"]


# ──────────────────────────────────────────────────────────
# 基础实验共享超参数（与 cnn_fed_base.py 保持一致）
# ──────────────────────────────────────────────────────────
BASE_SEED = 42
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


def generate_adjacency_matrix(num_nodes: int = BASE_NUM_NODES, seed: int = BASE_SEED):
    """为 GCN 生成固定的归一化邻接矩阵。

    构造一个基于路网拓扑的邻接矩阵：节点按编号顺序排列，相邻节点相连，
    并加入少量随机跨连接模拟交叉口。最终进行对称归一化。

    返回:
        a_norm: np.ndarray, shape [num_nodes, num_nodes], 归一化邻接矩阵
        a_raw: np.ndarray, 原始邻接矩阵（用于可视化）
        graph_meta: dict, 图结构摘要信息
    """
    rng = np.random.RandomState(seed)

    # 初始化邻接矩阵：相邻节点连接（线型拓扑）
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # 添加少量跨连接（模拟交叉口或支路），使图不完全为线型
    cross_edges = min(num_nodes // 2, 3)
    for _ in range(cross_edges):
        u = rng.randint(0, num_nodes)
        v = rng.randint(0, num_nodes)
        if u != v and A[u, v] == 0:
            w = 0.3 + 0.4 * rng.rand()  # 弱连接权重
            A[u, v] = w
            A[v, u] = w

    # 自环
    A_self = A + np.eye(num_nodes, dtype=np.float32)

    # 对称归一化: D^{-1/2} @ A @ D^{-1/2}
    deg = A_self.sum(axis=1)
    deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_self @ D_inv_sqrt

    # 计算图统计信息
    degrees = A.sum(axis=1)
    num_edges = int(np.sum(A > 0) / 2)

    graph_meta = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": float(np.mean(degrees)),
        "max_degree": float(np.max(degrees)),
        "min_degree": float(np.min(degrees)),
        "adjacency_type": "fixed_line_with_cross",
    }

    return A_norm.astype(np.float32), A.astype(np.float32), graph_meta


def split_train_val_test(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = BASE_SEED,
    train_ratio: float = BASE_TRAIN_RATIO,
    val_ratio: float = BASE_VAL_RATIO,
):
    """按随机顺序划分训练/验证/测试集。

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
# GCN 模型组件
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


class SimpleGCNLayer(nn.Module):
    """基础图卷积层：A_norm @ X @ W。"""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, a_norm):
        """x: [B, K, F], a_norm: [K, K]"""
        ax = torch.einsum("ij,bjf->bif", a_norm, x)
        return self.lin(ax)


class GCNEncoder(nn.Module):
    """GCN 编码器：可学习邻接矩阵 + 两层图卷积。"""

    def __init__(self, k: int, t: int, hidden_dim: int = 64, use_fixed_adj: bool = True,
                 fixed_adj: np.ndarray = None):
        super().__init__()
        self.k = k
        self.t = t
        self.hidden_dim = hidden_dim
        self.use_fixed_adj = use_fixed_adj

        # 将每个节点长度为 T 的时间序列投影到隐藏空间
        self.node_proj = nn.Sequential(
            nn.Linear(t, hidden_dim),
            nn.LayerNorm(hidden_dim),
            AdaptiveSwish(),
        )
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()

        # 可学习邻接矩阵参数（用于微调或从头学习）
        self.a_param = nn.Parameter(torch.randn(k, k) * 0.01)

        # 如果提供了固定邻接矩阵，注册为 buffer
        if fixed_adj is not None:
            self.register_buffer("fixed_adj", torch.tensor(fixed_adj, dtype=torch.float32))
        else:
            self.fixed_adj = None

    def _normalize_adj(self, a):
        """对称归一化邻接矩阵。"""
        a = torch.relu(a)
        a = a + torch.eye(self.k, device=a.device, dtype=a.dtype)
        deg = a.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        return d_inv_sqrt @ a @ d_inv_sqrt

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x = self.node_proj(x)

        if self.use_fixed_adj and self.fixed_adj is not None:
            a_norm = self.fixed_adj
        else:
            a_norm = self._normalize_adj(self.a_param)

        h = self.gcn1(x, a_norm)
        h = self.norm1(h)
        h = self.act(h)
        h = self.gcn2(h, a_norm)
        h = self.norm2(h)
        h = self.act(h)
        return h.mean(dim=1)  # [B, hidden_dim]


class GCNBaseModel(nn.Module):
    """GCN-BiLSTM-Attention 基础联邦模型。

    结构：
    1. GCN 分支：基于固定邻接矩阵执行节点间消息传递；
    2. BiLSTM 分支：捕捉双向时序依赖；
    3. 多头注意力融合两个分支特征。
    """

    def __init__(self, k: int, t: int, hidden_dim: int = 64, num_heads: int = 4,
                 fixed_adj: np.ndarray = None):
        super().__init__()
        self.gcn_encoder = GCNEncoder(k=k, t=t, hidden_dim=hidden_dim,
                                       fixed_adj=fixed_adj)
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
        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


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
# Workflow: data_viz — 基础数据集可视化（含图结构）
# ══════════════════════════════════════════════════════════

def run_data_visualization_base(output_dir: Path) -> None:
    """基于基础实验数据生成逻辑，生成可视化图（含 GCN 图结构）。"""
    print("\n" + "=" * 60)
    print("[data_viz] Generating base dataset visualizations (GCN)...")
    print("=" * 60)

    set_global_seed(BASE_SEED)
    all_X, all_Y, meta = generate_base_traffic_data()
    _, A_raw, graph_meta = generate_adjacency_matrix()

    ensure_output_dir(output_dir)

    # ── 1. 每个 client 的平均交通流时间序列 ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid in range(meta["num_clients"]):
        ts_mean = all_X[cid].mean(axis=(0, 1))
        sns.lineplot(
            x=np.arange(len(ts_mean)),
            y=ts_mean,
            ax=ax,
            linewidth=2.0,
            alpha=0.9,
            color=CLIENT_PALETTE[cid % len(CLIENT_PALETTE)],
            label=f"Client {cid}",
        )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Traffic Flow")
    ax.set_title("Per-client average traffic flow")
    ax.legend(loc="best", ncol=2, fontsize=8)
    ax.set_xlim(0, meta["seq_len"] - 1)
    save_figure(fig, output_dir, "base_dataset_client_timeseries.png")

    # ── 2. 代表性 client 的节点-时间热力图 ──
    rep_cid = 0
    X_rep = all_X[rep_cid]
    node_time_matrix = X_rep.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        node_time_matrix,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Traffic flow"},
        xticklabels=1,
        yticklabels=1,
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Node ID")
    ax.set_title(f"Node-time traffic heatmap for client {rep_cid}")
    save_figure(fig, output_dir, "base_dataset_node_heatmap.png")

    # ── 3. 不同 client 的流量分布箱线图 ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dist_rows = []
    for cid in range(meta["num_clients"]):
        sampled_values = all_X[cid].ravel()[::8]
        dist_rows.extend(
            {"client": f"Client {cid}", "traffic_flow": float(value)}
            for value in sampled_values
        )
    df_dist = pd.DataFrame(dist_rows)
    sns.boxplot(
        data=df_dist,
        x="client",
        y="traffic_flow",
        ax=ax,
        palette=CLIENT_PALETTE[: meta["num_clients"]],
        showfliers=False,
        linewidth=1.0,
    )
    ax.set_xlabel("Client")
    ax.set_ylabel("Traffic flow")
    ax.set_title("Non-IID traffic distribution by client")
    save_figure(fig, output_dir, "base_dataset_client_boxplot.png")

    # ── 4. train / val / test 划分概览 ──
    fig, ax = plt.subplots(figsize=(10, 4))
    total_samples = sum(meta["samples_per_client"])
    train_n = int(total_samples * BASE_TRAIN_RATIO)
    val_n = int(total_samples * BASE_VAL_RATIO)
    test_n = total_samples - train_n - val_n
    labels = [f"Train ({BASE_TRAIN_RATIO*100:.0f}%)",
              f"Val ({BASE_VAL_RATIO*100:.0f}%)",
              f"Test ({BASE_TEST_RATIO*100:.0f}%)"]
    sizes = [train_n, val_n, test_n]
    left = 0
    for size, label, color in zip(sizes, labels, SPLIT_PALETTE):
        ax.barh(["Dataset split"], [size], left=[left], color=color, label=label)
        left += size
    ax.set_xlabel("Number of Samples")
    ax.set_title("Train, validation, and test split")
    ax.legend(loc="upper right")
    ax.set_xlim(0, total_samples + 50)
    save_figure(fig, output_dir, "base_dataset_split_overview.png")

    # ── 5. 每个 client 的样本量 ──
    fig, ax = plt.subplots(figsize=(8, 4.8))
    client_ids = [f"Client {i}" for i in range(meta["num_clients"])]
    df_counts = pd.DataFrame({"client": client_ids, "samples": meta["samples_per_client"]})
    sns.barplot(
        data=df_counts,
        x="client",
        y="samples",
        ax=ax,
        palette=CLIENT_PALETTE[: meta["num_clients"]],
    )
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample size by client")
    for i, v in enumerate(meta["samples_per_client"]):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10)
    save_figure(fig, output_dir, "base_dataset_client_sample_size.png")

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

    # ── 7. GCN 邻接矩阵热力图 ──
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        A_raw,
        ax=ax,
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Adjacency weight"},
    )
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Node ID")
    ax.set_title("Fixed adjacency matrix")
    save_figure(fig, output_dir, "base_gcn_adjacency_matrix.png")

    # ── 8. 节点度分布柱状图 ──
    degrees = A_raw.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    df_degree = pd.DataFrame({"node": [f"Node {i}" for i in range(len(degrees))], "degree": degrees})
    sns.barplot(
        data=df_degree,
        x="node",
        y="degree",
        ax=ax,
        color=METHOD_PALETTE["FedAvg"],
        errorbar=None,
    )
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Degree")
    ax.set_title("Node degree distribution")
    ax.set_xticklabels(df_degree["node"], rotation=30, ha="right")
    for i, deg in enumerate(degrees):
        ax.text(i, deg + 0.05, f"{deg:.1f}", ha="center", fontsize=9)
    save_figure(fig, output_dir, "base_gcn_degree_distribution.png")

    # ── 9. 图结构摘要 CSV ──
    df_graph = pd.DataFrame([graph_meta])
    save_dataframe(df_graph, output_dir, "base_gcn_graph_summary.csv")

    print("[data_viz] Dataset summary:\n", df_summary.to_string(index=False))
    print("[data_viz] Graph summary:\n", df_graph.to_string(index=False))
    print("[data_viz] Done.\n")


# ══════════════════════════════════════════════════════════
# Workflow: main — Independent / FedAvg 主结果对比
# ══════════════════════════════════════════════════════════

def run_main_experiment(output_dir: Path) -> None:
    """运行基础实验主结果：Independent vs FedAvg。"""
    print("\n" + "=" * 60)
    print("[main] Running GCN FedAvg base experiment...")
    print("=" * 60)

    set_global_seed(BASE_SEED)
    all_X, all_Y, _ = generate_base_traffic_data()
    A_norm, _, _ = generate_adjacency_matrix()
    ensure_output_dir(output_dir)

    criterion = nn.MSELoss()
    num_clients = BASE_NUM_CLIENTS
    k = BASE_NUM_NODES
    t = BASE_SEQ_LEN

    train_loaders, val_loaders, test_loaders = [], [], []
    train_sizes = []
    for cid in range(num_clients):
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test(
            all_X[cid], all_Y[cid]
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
        FederatedClient(cid, GCNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM,
                                           fixed_adj=A_norm),
                        train_loaders[cid], val_loaders[cid], test_loaders[cid],
                        criterion, lr=1e-3)
        for cid in range(num_clients)
    ]
    server = FedAvgServer(GCNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM,
                                        fixed_adj=A_norm), num_clients)
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

        val_losses = []
        for client in fed_clients:
            client.model.load_state_dict(server.global_model.state_dict())
            val_losses.append(client.validate(client.val_loader))
        server.round_val_losses.append(float(np.mean(val_losses)))
        print(f"  Round {rnd+1}/{FED_ROUNDS} | Avg Train Loss: {server.round_losses[-1]:.6f} "
              f"| Avg Val Loss: {server.round_val_losses[-1]:.6f}")

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

    rows = []
    for cid in range(num_clients):
        fm = fed_metrics[cid]
        im = ind_metrics[cid]
        rows.append({"method": "FedAvg", "client_id": cid,
                     "mse": fm["mse"], "rmse": fm["rmse"], "mape": fm["mape"],
                     "mae": fm["mae"]})
        rows.append({"method": "Independent", "client_id": cid,
                     "mse": im["mse"], "rmse": im["rmse"], "mape": im["mape"],
                     "mae": im["mae"]})
    df_metrics = pd.DataFrame(rows)
    save_dataframe(df_metrics, output_dir, "gcn_base_metrics.csv")

    summary_rows = []
    for method in ["FedAvg", "Independent"]:
        sub = df_metrics[df_metrics["method"] == method]
        summary_rows.append({
            "method": method,
            "mse_mean": float(sub["mse"].mean()),
            "mse_std": float(sub["mse"].std(ddof=0)),
            "rmse_mean": float(sub["rmse"].mean()),
            "rmse_std": float(sub["rmse"].std(ddof=0)),
            "mape_mean": float(sub["mape"].mean()),
            "mae_mean": float(sub["mae"].mean()),
            "mape_std": float(sub["mape"].std(ddof=0)),
            "mae_std": float(sub["mae"].std(ddof=0)),
        })
    df_summary = pd.DataFrame(summary_rows)
    save_dataframe(df_summary, output_dir, "gcn_base_metrics_summary.csv")
    print("\n[main] Summary:\n", df_summary.to_string(index=False))

    # 对比柱状图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    client_labels = [f"Client {i}" for i in range(num_clients)]
    for idx, metric in enumerate(["mse", "rmse", "mae", "mape"]):
        ax = axes[idx]
        plot_df = pd.DataFrame(
            {
                "client": client_labels * 2,
                "method": ["Independent"] * num_clients + ["FedAvg"] * num_clients,
                "value": [ind_metrics[c][metric] for c in range(num_clients)]
                + [fed_metrics[c][metric] for c in range(num_clients)],
            }
        )
        sns.barplot(
            data=plot_df,
            x="client",
            y="value",
            hue="method",
            hue_order=["Independent", "FedAvg"],
            palette=METHOD_PALETTE,
            ax=ax,
            errorbar=None,
        )
        ax.set_xticklabels(client_labels, rotation=30, ha="right")
        ax.set_title(metric.upper())
        ylabel_map = f"{metric.upper()} (%) (lower is better)" if metric == "mape" else f"{metric.upper()} (lower is better)"
        ax.set_ylabel(ylabel_map)
        ax.set_xlabel("Client")
        if idx == 0:
            ax.legend(loc="best", fontsize=8, title=None)
        else:
            ax.get_legend().remove()
    fig.suptitle("GCN base method comparison", fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_base_main_comparison.png")
    print("[main] Done.\n")


# ══════════════════════════════════════════════════════════
# Workflow: convergence — 联邦训练收敛曲线
# ══════════════════════════════════════════════════════════

def run_convergence_experiment(output_dir: Path) -> None:
    """输出基础训练收敛曲线。"""
    print("\n" + "=" * 60)
    print("[convergence] Running GCN convergence analysis...")
    print("=" * 60)

    set_global_seed(BASE_SEED)
    all_X, all_Y, _ = generate_base_traffic_data()
    A_norm, _, _ = generate_adjacency_matrix()
    ensure_output_dir(output_dir)

    criterion = nn.MSELoss()
    num_clients = BASE_NUM_CLIENTS
    k = BASE_NUM_NODES
    t = BASE_SEQ_LEN
    convergence_rounds = 15

    train_loaders, val_loaders, test_loaders = [], [], []
    train_sizes = []
    for cid in range(num_clients):
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test(
            all_X[cid], all_Y[cid]
        )
        train_sizes.append(len(X_train))
        train_loaders.append(DataLoader(TrafficDataset(X_train, Y_train),
                                        batch_size=FED_BATCH_SIZE, shuffle=True))
        val_loaders.append(DataLoader(TrafficDataset(X_val, Y_val),
                                       batch_size=FED_BATCH_SIZE, shuffle=False))
        test_loaders.append(DataLoader(TrafficDataset(X_test, Y_test),
                                        batch_size=FED_BATCH_SIZE, shuffle=False))

    fed_clients = [
        FederatedClient(cid, GCNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM,
                                           fixed_adj=A_norm),
                        train_loaders[cid], val_loaders[cid], test_loaders[cid],
                        criterion, lr=1e-3)
        for cid in range(num_clients)
    ]
    server = FedAvgServer(GCNBaseModel(k=k, t=t, hidden_dim=FED_HIDDEN_DIM,
                                        fixed_adj=A_norm), num_clients)
    server.set_client_data_sizes(train_sizes)

    round_data = {
        "round": [], "avg_train_loss": [], "avg_val_rmse": [],
    }
    for cid in range(num_clients):
        round_data[f"client_{cid}_train_loss"] = []
        round_data[f"client_{cid}_val_rmse"] = []

    print("\n[FedAvg Convergence Training]")
    for rnd in range(convergence_rounds):
        client_weights, client_losses = [], []
        for client in fed_clients:
            loss, weights = client.train_local(epochs=FED_LOCAL_EPOCHS,
                                               global_model=server.global_model)
            client_weights.append(weights)
            client_losses.append(loss)
        server.aggregate(client_weights, client_losses)

        val_rmses = []
        for client in fed_clients:
            client.model.load_state_dict(server.global_model.state_dict())
            val_loss = client.validate(client.val_loader)
            val_rmses.append(float(np.sqrt(val_loss)))

        server.round_val_losses.append(float(np.mean(val_rmses)))

        round_data["round"].append(rnd + 1)
        round_data["avg_train_loss"].append(server.round_losses[-1])
        round_data["avg_val_rmse"].append(server.round_val_losses[-1])
        for cid in range(num_clients):
            round_data[f"client_{cid}_train_loss"].append(client_losses[cid])
            round_data[f"client_{cid}_val_rmse"].append(val_rmses[cid])

        print(f"  Round {rnd+1}/{convergence_rounds} | "
              f"Train Loss: {server.round_losses[-1]:.6f} | "
              f"Val RMSE: {server.round_val_losses[-1]:.6f}")

    df_conv = pd.DataFrame(round_data)
    save_dataframe(df_conv, output_dir, "gcn_base_convergence.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    sns.lineplot(
        x=round_data["round"],
        y=round_data["avg_train_loss"],
        ax=ax,
        marker="o",
        linewidth=2.0,
        color=METHOD_PALETTE["FedAvg"],
        label="Average train loss",
    )
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Training loss (MSE)", color=METHOD_PALETTE["FedAvg"])
    ax.tick_params(axis="y", labelcolor=METHOD_PALETTE["FedAvg"])

    ax2 = ax.twinx()
    sns.lineplot(
        x=round_data["round"],
        y=round_data["avg_val_rmse"],
        ax=ax2,
        marker="s",
        linewidth=2.0,
        color=METHOD_PALETTE["Independent"],
        label="Average validation RMSE",
    )
    ax2.set_ylabel("Validation RMSE", color=METHOD_PALETTE["Independent"])
    ax2.tick_params(axis="y", labelcolor=METHOD_PALETTE["Independent"])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.set_title("Global validation RMSE across communication rounds")

    ax = axes[1]
    for cid in range(num_clients):
        sns.lineplot(
            x=round_data["round"],
            y=round_data[f"client_{cid}_train_loss"],
            ax=ax,
            linewidth=2.0,
            alpha=0.8,
            color=CLIENT_PALETTE[cid % len(CLIENT_PALETTE)],
            label=f"Client {cid}",
        )
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title("Per-client local training loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "gcn_base_convergence.png")
    print("[convergence] Done.\n")


# ══════════════════════════════════════════════════════════
# 工作流调度
# ══════════════════════════════════════════════════════════

def run_project(workflow: str, output_dir: Path) -> None:
    """按工作流执行 GCN 基础实验。"""
    configure_academic_plot_style()
    ensure_output_dir(output_dir)
    export_figure_index(output_dir)
    print(f"[gcn_fed_base] workflow={workflow}, output={output_dir}")
    print(f"[gcn_fed_base] device={DEVICE}")

    if workflow in ("all", "data_viz"):
        run_data_visualization_base(output_dir)

    if workflow in ("all", "main"):
        run_main_experiment(output_dir)

    if workflow in ("all", "convergence"):
        run_convergence_experiment(output_dir)

    print(f"\n[gcn_fed_base] All done. Results in: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="GCN Base Federated Simulation")
    parser.add_argument(
        "--workflow",
        choices=["all", "data_viz", "main", "convergence"],
        default="all",
        help="Workflow to execute (default: all).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    """程序主入口。"""
    args = parse_args(argv)
    output_dir = SIMULATION_RESULTS_ROOT / "gcn_fed_base"
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()

from .visualization import *

