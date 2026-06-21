"""区域客户端正式训练脚本。

核心功能：
- 将区域客户端计算 Notebook 重构为可独立运行的联邦训练模块；
- 保留区域划分、CNN/LSTM/Attention 建模、FedProx + FedAvg 聚合与个性化评估；
- 统一从预处理流水线最后一步输出的标准数据集文件中加载训练张量。

项目作用：
- 为真实交通数据上的区域客户端联邦训练提供工程化实现；
- 与现有 CCN 联邦训练保持相同的 client update 与 server aggregation 风格。

关键依赖：`torch`、`numpy`、`pandas`、`matplotlib`。
主要输入：`data/processed/node_flow_grid/node_flow_grid_tensor.pt`。
主要输出：训练历史、客户端评估汇总、JSON 指标和 PNG 图像。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "node_flow_grid"
DEFAULT_DATASET_FILE = "node_flow_grid_tensor.pt"
DEFAULT_DATASET_PATH = DEFAULT_DATASET_DIR / DEFAULT_DATASET_FILE
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "federated_learning" / "region_client_train"


@dataclass
class TrainConfig:
    seed: int = 15
    num_clients: int = 3
    t_in: int = 24
    t_out: int = 1
    batch_size: int = 128
    rounds: int = 5
    local_epochs: int = 1
    lr: float = 2e-4
    mu: float = 3e-4
    server_lr: float = 0.5
    ft_max_epochs: int = 25
    ft_lr: float = 3e-4
    ft_patience: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    stride: int = 6
    alpha_mse: float = 0.7
    beta_huber: float = 0.3
    hidden_dim: int = 64
    model_kind: str = "full"
    dataset_path: Path = DEFAULT_DATASET_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    show_plot: bool = False
    verbose: bool = False

    @property
    def independent_epochs(self) -> int:
        return self.rounds * self.local_epochs


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="区域客户端正式联邦训练脚本。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="训练结果输出目录。")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--t-in", type=int, default=24)
    parser.add_argument("--t-out", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mu", type=float, default=3e-4)
    parser.add_argument("--server-lr", type=float, default=0.5)
    parser.add_argument("--ft-max-epochs", type=int, default=25)
    parser.add_argument("--ft-lr", type=float, default=3e-4)
    parser.add_argument("--ft-patience", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--alpha-mse", type=float, default=0.7)
    parser.add_argument("--beta-huber", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--show-plot", action="store_true", help="保存图像后同时弹窗显示。")
    parser.add_argument("--verbose", action="store_true", help="输出更详细日志。")
    args = parser.parse_args()

    configure_logging(args.verbose)
    config = TrainConfig(
        num_clients=args.num_clients,
        t_in=args.t_in,
        t_out=args.t_out,
        batch_size=args.batch_size,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        mu=args.mu,
        server_lr=args.server_lr,
        ft_max_epochs=args.ft_max_epochs,
        ft_lr=args.ft_lr,
        ft_patience=args.ft_patience,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stride=args.stride,
        alpha_mse=args.alpha_mse,
        beta_huber=args.beta_huber,
        hidden_dim=args.hidden_dim,
        dataset_path=DEFAULT_DATASET_PATH.resolve(),
        output_dir=args.output_dir.resolve(),
        show_plot=bool(args.show_plot),
        verbose=bool(args.verbose),
    )
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    if config.num_clients <= 0:
        raise ValueError("num_clients 必须为正数。")
    if config.t_in <= 0 or config.t_out <= 0:
        raise ValueError("t_in 和 t_out 必须为正数。")
    if config.batch_size <= 0 or config.rounds <= 0 or config.local_epochs <= 0:
        raise ValueError("batch_size、rounds、local_epochs 必须为正数。")
    if config.lr <= 0 or config.ft_lr <= 0:
        raise ValueError("学习率必须为正数。")
    if config.mu < 0:
        raise ValueError("mu 不能为负数。")
    if not (0.0 < config.server_lr <= 1.0):
        raise ValueError("server_lr 必须位于 (0, 1]。")
    if config.stride <= 0:
        raise ValueError("stride 必须为正数。")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim 必须为正数。")
    if config.dataset_path.resolve() != DEFAULT_DATASET_PATH.resolve():
        raise ValueError(f"联邦训练仅允许读取预处理最终产物: {DEFAULT_DATASET_PATH}")
    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1。")
    if min(config.train_ratio, config.val_ratio, config.test_ratio) <= 0:
        raise ValueError("train/val/test ratio 必须均大于 0。")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("使用设备: %s", device)
    return device


def load_data(config: TrainConfig) -> torch.Tensor:
    dataset_path = config.dataset_path.resolve()
    if not dataset_path.parent.exists():
        raise FileNotFoundError(
            f"未找到预处理输出目录: {dataset_path.parent}。请先执行完整预处理流程，并确认最终数据集已生成。"
        )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"未找到预处理最终数据集文件: {dataset_path}。请先执行预处理最后一步，生成标准化训练数据集。"
        )

    logging.info("从预处理最终数据集加载训练数据: %s", dataset_path)
    data_tensor = torch.load(dataset_path, map_location="cpu")
    validate_data_tensor(data_tensor, config)
    return data_tensor


def validate_data_tensor(data_tensor: Any, config: TrainConfig) -> None:
    if not isinstance(data_tensor, torch.Tensor):
        raise TypeError("预处理最终输出的数据集必须是 torch.Tensor。")
    if data_tensor.dim() != 3:
        raise ValueError(f"预处理数据集的张量维度必须为 3，实际为 {data_tensor.dim()}。")
    channels, num_regions, total_steps = map(int, data_tensor.shape)
    if channels <= 0 or num_regions <= 0:
        raise ValueError(f"预处理数据集形状无效: {tuple(data_tensor.shape)}。")
    if channels < 1:
        raise ValueError("预处理数据集至少需要 1 个特征通道。")
    if num_regions < config.num_clients:
        raise ValueError(f"区域数量 {num_regions} 小于客户端数量 {config.num_clients}，无法完成客户端划分。")
    if total_steps <= config.t_in + config.t_out:
        raise ValueError(
            f"预处理数据集时间维长度不足，至少需要大于 t_in+t_out={config.t_in + config.t_out}，实际为 {total_steps}。"
        )
    if not torch.isfinite(data_tensor).all():
        raise ValueError("预处理数据集中存在 NaN 或 Inf，请先重新执行或检查预处理流程。")
    if data_tensor.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"预处理数据集 dtype 必须为浮点类型，实际为 {data_tensor.dtype}。")


def safe_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_region_features(
    data_tensor: torch.Tensor,
    t_in: int,
    t_out: int,
    season_lags: tuple[int, int] = (24, 48),
) -> tuple[np.ndarray, np.ndarray]:
    values = safe_np(data_tensor)
    _, num_regions, total_steps = values.shape

    features = []
    sizes = np.zeros(num_regions, dtype=float)
    for region_id in range(num_regions):
        series = values[:, region_id, :]
        ts = np.nanmedian(series, axis=0)
        valid_mask = np.isfinite(ts)
        effective_steps = int(valid_mask.sum())
        sizes[region_id] = float(max(0, effective_steps - t_in - t_out))

        miss = float(np.mean(~np.isfinite(ts)))
        if not np.isfinite(ts).all():
            ts = np.where(np.isfinite(ts), ts, np.nanmedian(ts))

        centered = ts - np.mean(ts)
        denom = float(np.sum(centered**2) + 1e-12)
        d1 = np.diff(ts)
        acfs = []
        for lag in season_lags:
            if lag < len(centered):
                acf = float(np.sum(centered[:-lag] * centered[lag:]) / denom)
            else:
                acf = 0.0
            acfs.append(acf)

        fft = np.fft.rfft(centered)
        power = np.abs(fft) ** 2
        total_power = float(np.sum(power) + 1e-12)
        peak_power = float(np.max(power[1:]) if len(power) > 2 else 0.0)
        peak_ratio = float(peak_power / total_power)

        features.append(
            [
                float(np.mean(ts)),
                float(np.std(ts)),
                float(np.quantile(ts, 0.05)),
                float(np.quantile(ts, 0.95)),
                float(np.quantile(ts, 0.75) - np.quantile(ts, 0.25)),
                float(np.mean(np.abs(d1))) if len(d1) else 0.0,
                float(np.std(d1)) if len(d1) else 0.0,
                miss,
                *acfs,
                peak_ratio,
            ]
        )

    feats = np.asarray(features, dtype=float)
    feats_z = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-12)
    if float(np.sum(sizes)) <= 0:
        sizes[:] = float(max(0, total_steps - t_in - t_out))
    return feats_z, sizes


def kmeans_cluster(feats_z: np.ndarray, num_clients: int, seed: int, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if feats_z.shape[0] < num_clients:
        raise ValueError(f"区域数量 {feats_z.shape[0]} 小于客户端数量 {num_clients}。")
    rng = np.random.default_rng(seed)
    centers = feats_z[rng.choice(feats_z.shape[0], size=num_clients, replace=False)]
    for _ in range(iters):
        d2 = ((feats_z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        next_centers = []
        for client_id in range(num_clients):
            idx = np.where(labels == client_id)[0]
            next_centers.append(centers[client_id] if len(idx) == 0 else feats_z[idx].mean(axis=0))
        next_centers = np.vstack(next_centers)
        if np.max(np.abs(next_centers - centers)) < 1e-6:
            break
        centers = next_centers
    return labels, centers


def balance_clusters_by_size(
    labels: np.ndarray,
    feats_z: np.ndarray,
    centers: np.ndarray,
    sizes: np.ndarray,
    num_clients: int,
    max_moves: int = 2000,
) -> np.ndarray:
    labels = labels.copy()
    target = float(np.sum(sizes) / max(num_clients, 1))

    def load(client_id: int) -> float:
        return float(np.sum(sizes[labels == client_id]))

    for _ in range(max_moves):
        loads = np.array([load(cid) for cid in range(num_clients)], dtype=float)
        over = int(np.argmax(loads))
        under = int(np.argmin(loads))
        if loads[over] <= target * 1.05 and loads[under] >= target * 0.95:
            break
        idx_over = np.where(labels == over)[0]
        if len(idx_over) == 0:
            break
        dist_over = ((feats_z[idx_over] - centers[over]) ** 2).sum(axis=1)
        dist_under = ((feats_z[idx_over] - centers[under]) ** 2).sum(axis=1)
        score = (dist_under - dist_over) + 0.05 * (sizes[idx_over] / (target + 1e-12))
        best = idx_over[int(np.argmin(score))]
        labels[best] = under
    return labels


def split_clients(data_tensor: torch.Tensor, config: TrainConfig) -> list[np.ndarray]:
    feats_z, sizes = extract_region_features(data_tensor, config.t_in, config.t_out)
    labels, centers = kmeans_cluster(feats_z, config.num_clients, config.seed)
    balanced = balance_clusters_by_size(labels, feats_z, centers, sizes, config.num_clients)
    region_indices = [np.where(balanced == cid)[0].astype(int) for cid in range(config.num_clients)]
    for cid, region_ids in enumerate(region_indices):
        logging.info(
            "客户端 %s: 区域数=%s, 估计样本量~%s, region_ids(head)=%s",
            cid,
            len(region_ids),
            int(np.sum(sizes[region_ids])),
            region_ids[:10].tolist(),
        )
    return region_indices


class RegionDataset(Dataset):
    def __init__(self, data: torch.Tensor, region_ids: np.ndarray, t_in: int, t_out: int, stride: int) -> None:
        super().__init__()
        self.data = data
        self.region_ids = np.asarray(region_ids, dtype=int)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.stride = int(stride)
        _, _, total_steps = data.shape
        self.valid_t = total_steps - self.t_in - self.t_out
        if self.valid_t <= 0 or len(self.region_ids) == 0:
            self.t_positions = np.array([], dtype=int)
            self.length = 0
        else:
            self.t_positions = np.arange(0, self.valid_t, self.stride, dtype=int)
            self.length = int(len(self.region_ids) * len(self.t_positions))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.length:
            raise IndexError(f"样本索引超出范围: {idx}")
        tlen = len(self.t_positions)
        region_pos = idx // tlen
        t0 = int(self.t_positions[idx % tlen])
        region_id = int(self.region_ids[region_pos])
        x = self.data[:, region_id, t0 : t0 + self.t_in]
        y = self.data[0, region_id, t0 + self.t_in]
        return torch.log1p(x).float(), torch.log1p(y).float()


def split_timewise_indices(ds: RegionDataset, config: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tlen = len(ds.t_positions)
    if tlen <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    def split_counts(n: int) -> tuple[int, int, int]:
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        n_test = n - n_train - n_val
        if n >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            while n_train + n_val + n_test > n:
                if n_train >= max(n_val, n_test) and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
        return n_train, n_val, n_test

    train_idx, val_idx, test_idx = [], [], []
    n_train, n_val, _ = split_counts(tlen)
    for region_pos in range(len(ds.region_ids)):
        start = region_pos * tlen
        region_all = np.arange(start, start + tlen, dtype=int)
        train_idx.extend(region_all[:n_train].tolist())
        val_idx.extend(region_all[n_train : n_train + n_val].tolist())
        test_idx.extend(region_all[n_train + n_val :].tolist())
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class CNNLSTMAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.attn(out)
        return self.fc(out).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)


class LSTMAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        out, _ = self.lstm(x)
        out = self.attn(out)
        return self.fc(out).squeeze(-1)


class CNNAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out = self.attn(x)
        return self.fc(out).squeeze(-1)


def build_model(in_channels: int, model_kind: str = "full", hidden_dim: int = 64) -> nn.Module:
    builders: dict[str, type[nn.Module]] = {
        "full": CNNLSTMAttention,
        "cnn_lstm": CNNLSTM,
        "lstm_attention": LSTMAttention,
        "cnn_attention": CNNAttention,
    }
    if model_kind not in builders:
        raise ValueError(f"不支持的模型类型: {model_kind}")
    return builders[model_kind](in_channels=in_channels, hidden_dim=hidden_dim)


def mixed_raw_loss(pred_log: torch.Tensor, target_log: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    pred_raw = torch.expm1(pred_log)
    target_raw = torch.expm1(target_log)
    mse = nn.functional.mse_loss(pred_raw, target_raw)
    huber = nn.functional.smooth_l1_loss(pred_raw, target_raw)
    return alpha * mse + beta * huber


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        preds.append(torch.expm1(pred).detach().cpu())
        trues.append(torch.expm1(y).detach().cpu())
    if not preds:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    preds_tensor = torch.cat(preds, dim=0)
    trues_tensor = torch.cat(trues, dim=0)
    mse = float(nn.functional.mse_loss(preds_tensor, trues_tensor).item())
    mae = float(nn.functional.l1_loss(preds_tensor, trues_tensor).item())
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae}


def train_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    server_snapshot_weights: dict[str, torch.Tensor],
    use_fedprox: bool = True,
) -> tuple[dict[str, torch.Tensor], float, dict[str, float]]:
    model.load_state_dict(server_snapshot_weights)
    model.train()
    ref_weights = {name: tensor.detach().clone().to(device) for name, tensor in server_snapshot_weights.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    total_loss, steps = 0.0, 0
    for _ in range(config.local_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mixed_raw_loss(pred, y, config.alpha_mse, config.beta_huber)
            if use_fedprox and config.mu > 0:
                prox = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        prox = prox + torch.sum((param - ref_weights[name]) ** 2)
                loss = loss + 0.5 * config.mu * prox
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

    avg_train_loss = total_loss / max(steps, 1)
    val_metrics = evaluate(model, val_loader, device)
    return copy.deepcopy(model.state_dict()), avg_train_loss, val_metrics


def fedavg_weighted(weights: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    avg = copy.deepcopy(weights[0])
    total = float(np.sum(sample_counts)) + 1e-12
    for key in avg.keys():
        if avg[key].dtype in (torch.float16, torch.float32, torch.float64):
            avg[key].zero_()
            for state, n in zip(weights, sample_counts):
                avg[key] += state[key] * (float(n) / total)
        else:
            avg[key] = weights[0][key]
    return avg


def federated_aggregation(
    server_weights: dict[str, torch.Tensor],
    local_weights: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    server_lr: float,
) -> dict[str, torch.Tensor]:
    aggregated = fedavg_weighted(local_weights, sample_counts)
    next_weights = copy.deepcopy(server_weights)
    for key in next_weights.keys():
        if next_weights[key].dtype in (torch.float16, torch.float32, torch.float64):
            next_weights[key] = (1.0 - server_lr) * next_weights[key] + server_lr * aggregated[key]
        else:
            next_weights[key] = aggregated[key]
    return next_weights


def build_client_loaders(
    data_tensor: torch.Tensor,
    region_indices: list[np.ndarray],
    config: TrainConfig,
) -> list[dict[str, Any]]:
    client_payloads = []
    for cid, region_ids in enumerate(region_indices):
        dataset = RegionDataset(data_tensor, region_ids, config.t_in, config.t_out, config.stride)
        if len(dataset) == 0:
            logging.warning("客户端 %s 无有效样本，已跳过。", cid)
            continue
        train_idx, val_idx, test_idx = split_timewise_indices(dataset, config)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            logging.warning("客户端 %s 的 train/val/test 至少有一个为空，已跳过。", cid)
            continue

        payload = {
            "cid": cid,
            "regions": region_ids,
            "train_loader": DataLoader(Subset(dataset, train_idx.tolist()), batch_size=config.batch_size, shuffle=True),
            "val_loader": DataLoader(Subset(dataset, val_idx.tolist()), batch_size=config.batch_size, shuffle=False),
            "test_loader": DataLoader(Subset(dataset, test_idx.tolist()), batch_size=config.batch_size, shuffle=False),
            "train_size": int(len(train_idx)),
        }
        client_payloads.append(payload)
    if not client_payloads:
        raise RuntimeError("没有可用客户端，请检查输入数据与时间窗口配置。")
    return client_payloads


def train_federated_model(
    base_model: nn.Module,
    client_payloads: list[dict[str, Any]],
    device: torch.device,
    config: TrainConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    server_model = copy.deepcopy(base_model).to(device)
    server_weights = copy.deepcopy(server_model.state_dict())
    history_rows = []

    for round_id in range(1, config.rounds + 1):
        local_states = []
        sample_counts = []
        train_losses = []
        val_mses = []

        for payload in client_payloads:
            client_model = copy.deepcopy(server_model).to(device)
            local_state, avg_train_loss, val_metrics = train_client(
                model=client_model,
                train_loader=payload["train_loader"],
                val_loader=payload["val_loader"],
                device=device,
                config=config,
                server_snapshot_weights=server_weights,
                use_fedprox=True,
            )
            local_states.append(local_state)
            sample_counts.append(int(payload["train_size"]))
            train_losses.append(avg_train_loss)
            val_mses.append(val_metrics["mse"])

        server_weights = federated_aggregation(server_weights, local_states, sample_counts, config.server_lr)
        server_model.load_state_dict(server_weights)

        test_metrics = [evaluate(server_model, payload["test_loader"], device) for payload in client_payloads]
        history_rows.append(
            {
                "round": round_id,
                "avg_train_loss": float(np.mean(train_losses)),
                "avg_val_mse": float(np.mean(val_mses)),
                "fed_test_mse_mean": float(np.mean([m["mse"] for m in test_metrics])),
                "fed_test_rmse_mean": float(np.mean([m["rmse"] for m in test_metrics])),
                "fed_test_mae_mean": float(np.mean([m["mae"] for m in test_metrics])),
            }
        )
        logging.info(
            "轮次 %s/%s | AvgTrainLoss=%.6f | FedTestRMSE(mean)=%.6f",
            round_id,
            config.rounds,
            history_rows[-1]["avg_train_loss"],
            history_rows[-1]["fed_test_rmse_mean"],
        )
    return server_model, pd.DataFrame(history_rows)


def train_independent_model(
    in_channels: int,
    train_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> nn.Module:
    model = build_model(in_channels=in_channels, model_kind=config.model_kind, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    for _ in range(config.independent_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mixed_raw_loss(pred, y, config.alpha_mse, config.beta_huber)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def personalize_model(
    server_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> nn.Module:
    model = copy.deepcopy(server_model).to(device)

    if hasattr(model, "cnn") and isinstance(model.cnn, nn.Sequential) and len(model.cnn) >= 3:
        first_conv = model.cnn[0]
        second_conv = model.cnn[2]
        if isinstance(first_conv, nn.Conv1d):
            for param in first_conv.parameters():
                param.requires_grad = False
        if isinstance(second_conv, nn.Conv1d):
            for param in second_conv.parameters():
                param.requires_grad = True

    for module_name in ["lstm", "attn", "fc", "proj"]:
        if hasattr(model, module_name):
            for param in getattr(model, module_name).parameters():
                param.requires_grad = True

    parameter_groups = []
    if hasattr(model, "cnn") and isinstance(model.cnn, nn.Sequential) and len(model.cnn) >= 3 and isinstance(model.cnn[2], nn.Conv1d):
        parameter_groups.append({"params": model.cnn[2].parameters(), "lr": config.ft_lr * 0.3})
    if hasattr(model, "lstm"):
        parameter_groups.append({"params": model.lstm.parameters(), "lr": config.ft_lr})
    head_params = []
    for module_name in ["attn", "fc", "proj"]:
        if hasattr(model, module_name):
            head_params.extend(list(getattr(model, module_name).parameters()))
    if head_params:
        parameter_groups.append({"params": head_params, "lr": config.ft_lr * 2.0})
    if not parameter_groups:
        parameter_groups.append({"params": model.parameters(), "lr": config.ft_lr})

    optimizer = torch.optim.Adam(parameter_groups, weight_decay=1e-5)
    best_weights = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(config.ft_max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mixed_raw_loss(pred, y, config.alpha_mse, config.beta_huber)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_mse = evaluate(model, val_loader, device)["mse"]
        if val_mse < best_val * 0.999:
            best_val = val_mse
            best_weights = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.ft_patience:
                break

    model.load_state_dict(best_weights)
    return model


def save_results(
    config: TrainConfig,
    history_df: pd.DataFrame,
    client_results_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "federated_history.csv"
    client_path = config.output_dir / "client_metrics.csv"
    summary_path = config.output_dir / "summary.json"
    figure_path = config.output_dir / "region_client_train_overview.png"

    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    client_results_df.to_csv(client_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if not history_df.empty:
        axes[0, 0].plot(history_df["round"], history_df["avg_train_loss"], marker="o", label="AvgTrainLoss")
        axes[0, 0].plot(history_df["round"], history_df["fed_test_rmse_mean"], marker="s", label="FedTestRMSE")
        axes[0, 0].set_title("Federated Convergence")
        axes[0, 0].set_xlabel("Round")
        axes[0, 0].legend()

    metric_order = ["mse", "rmse", "mae"]
    method_order = ["federated_personalized", "independent"]
    for ax, metric in zip([axes[0, 1], axes[1, 0], axes[1, 1]], metric_order):
        subset = client_results_df.pivot(index="client", columns="method", values=metric).reindex(columns=method_order)
        subset.plot(kind="bar", ax=ax)
        ax.set_title(metric.upper())
        ax.set_xlabel("Client")
        ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    if config.show_plot:
        plt.show()
    plt.close(fig)

    logging.info("结果已保存到: %s", config.output_dir)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = get_device()
    data_tensor = load_data(config)
    region_indices = split_clients(data_tensor, config)
    client_payloads = build_client_loaders(data_tensor, region_indices, config)

    in_channels = int(data_tensor.shape[0])
    server_model = build_model(in_channels=in_channels, model_kind=config.model_kind, hidden_dim=config.hidden_dim).to(device)
    fed_model, history_df = train_federated_model(server_model, client_payloads, device, config)

    client_rows = []
    for payload in client_payloads:
        cid = int(payload["cid"])
        personalized_model = personalize_model(
            fed_model,
            payload["train_loader"],
            payload["val_loader"],
            device,
            config,
        )
        fed_metrics = evaluate(personalized_model, payload["test_loader"], device)

        independent_model = train_independent_model(in_channels, payload["train_loader"], device, config)
        indep_metrics = evaluate(independent_model, payload["test_loader"], device)

        client_rows.append(
            {
                "client": cid,
                "method": "federated_personalized",
                "regions": int(len(payload["regions"])),
                "train_size": int(payload["train_size"]),
                **fed_metrics,
            }
        )
        client_rows.append(
            {
                "client": cid,
                "method": "independent",
                "regions": int(len(payload["regions"])),
                "train_size": int(payload["train_size"]),
                **indep_metrics,
            }
        )

    client_results_df = pd.DataFrame(client_rows).sort_values(["client", "method"]).reset_index(drop=True)
    summary = {
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
        },
        "data_shape": list(map(int, data_tensor.shape)),
        "client_count": int(len(client_payloads)),
        "federated_personalized_mean": (
            client_results_df.loc[client_results_df["method"] == "federated_personalized", ["mse", "rmse", "mae"]]
            .mean()
            .to_dict()
        ),
        "independent_mean": (
            client_results_df.loc[client_results_df["method"] == "independent", ["mse", "rmse", "mae"]]
            .mean()
            .to_dict()
        ),
    }
    save_results(config, history_df, client_results_df, summary)


if __name__ == "__main__":
    main()
