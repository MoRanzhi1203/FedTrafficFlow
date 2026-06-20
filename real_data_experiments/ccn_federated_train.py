# Requirements:
#   torch
#   numpy
#   pandas
#   matplotlib
#   seaborn
#
# Expected external asset:
#   6.池化网格张量.pt

from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.font_manager import FontProperties
from torch.utils.data import DataLoader, Dataset, Subset


# =====================
# 1. 配置定义
# =====================
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

    @property
    def independent_epochs(self) -> int:
        return self.rounds * self.local_epochs


# =====================
# 2. 基础工具
# =====================
def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_data = project_root / "test" / "6.池化网格张量.pt"
    default_results = script_dir / "results"

    parser = argparse.ArgumentParser(
        description="CCN 联邦训练正式 notebook 转换脚本"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data,
        help="输入张量文件路径，默认兼容原 notebook 所在的 test 目录。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="结果输出目录。",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="保存图件后同时弹出 matplotlib 窗口。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_plot_style() -> None:
    mpl.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="SimHei",
        rc={
            "axes.unicode_minus": False,
            "figure.titlesize": 18,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
        },
    )


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


def load_data_tensor(data_path: Path) -> torch.Tensor:
    if not data_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {data_path}\n"
            "请确认 `6.池化网格张量.pt` 已放置在 test 目录，"
            "或通过 `--data-path` 显式指定。"
        )

    data_tensor = torch.load(data_path, map_location="cpu")
    if not isinstance(data_tensor, torch.Tensor) or data_tensor.dim() != 3:
        raise ValueError("Expect data_tensor shape (N, K, T).")

    print("数据形状:", tuple(data_tensor.shape))
    return data_tensor


def safe_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# =====================
# 3. 客户端划分
# =====================
def extract_region_features(
    data_tensor: torch.Tensor,
    t_in: int = 24,
    t_out: int = 1,
    season_lags: tuple[int, int] = (24, 48),
) -> tuple[np.ndarray, np.ndarray]:
    x = safe_np(data_tensor)
    _, num_regions, total_steps = x.shape

    feats = []
    sizes = np.zeros(num_regions, dtype=float)

    for region_id in range(num_regions):
        series = x[:, region_id, :]
        ts = np.nanmedian(series, axis=0)

        valid_mask = np.isfinite(ts)
        effective_steps = int(valid_mask.sum())
        valid_t = max(0, effective_steps - t_in - t_out)
        sizes[region_id] = float(valid_t)

        miss = float(np.mean(~np.isfinite(ts)))
        if not np.isfinite(ts).all():
            ts = np.where(np.isfinite(ts), ts, np.nanmedian(ts))

        mean = float(np.mean(ts))
        std = float(np.std(ts))
        p05 = float(np.quantile(ts, 0.05))
        p95 = float(np.quantile(ts, 0.95))
        iqr = float(np.quantile(ts, 0.75) - np.quantile(ts, 0.25))

        first_diff = np.diff(ts)
        d1_mean = float(np.mean(np.abs(first_diff))) if len(first_diff) else 0.0
        d1_std = float(np.std(first_diff)) if len(first_diff) else 0.0

        ts_centered = ts - np.mean(ts)
        denom = float(np.sum(ts_centered**2) + 1e-12)
        acfs = []
        for lag in season_lags:
            if lag < len(ts_centered):
                acf = float(
                    np.sum(ts_centered[:-lag] * ts_centered[lag:]) / denom
                )
            else:
                acf = 0.0
            acfs.append(acf)

        fft = np.fft.rfft(ts_centered)
        power = np.abs(fft) ** 2
        total_power = float(np.sum(power) + 1e-12)
        peak_power = float(np.max(power[1:]) if len(power) > 2 else 0.0)
        peak_ratio = float(peak_power / total_power)

        feats.append(
            [
                mean,
                std,
                p05,
                p95,
                iqr,
                d1_mean,
                d1_std,
                miss,
                *acfs,
                peak_ratio,
            ]
        )

    feats = np.asarray(feats, dtype=float)
    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True) + 1e-12
    feats_z = (feats - mu) / sigma

    if float(np.sum(sizes)) <= 0:
        valid_t = max(0, total_steps - t_in - t_out)
        sizes = np.full(num_regions, float(valid_t), dtype=float)

    return feats_z, sizes


def kmeans_cluster(
    feats_z: np.ndarray,
    num_clients: int = 3,
    seed: int = 15,
    iters: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_regions, _ = feats_z.shape
    centers = feats_z[rng.choice(num_regions, size=num_clients, replace=False)]

    for _ in range(iters):
        d2 = ((feats_z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        new_centers = []
        for client_id in range(num_clients):
            idx = np.where(labels == client_id)[0]
            if len(idx) == 0:
                new_centers.append(centers[client_id])
            else:
                new_centers.append(feats_z[idx].mean(axis=0))
        new_centers = np.vstack(new_centers)

        if np.max(np.abs(new_centers - centers)) < 1e-6:
            break
        centers = new_centers

    return labels, centers


def balance_clusters_by_size(
    labels: np.ndarray,
    feats_z: np.ndarray,
    centers: np.ndarray,
    sizes: np.ndarray,
    num_clients: int = 3,
    max_moves: int = 2000,
) -> np.ndarray:
    labels = labels.copy()
    target = float(np.sum(sizes) / num_clients)

    def load(client_id: int) -> float:
        return float(np.sum(sizes[labels == client_id]))

    for _ in range(max_moves):
        loads = np.array([load(client_id) for client_id in range(num_clients)])
        over = int(np.argmax(loads))
        under = int(np.argmin(loads))

        if loads[over] <= target * 1.05 and loads[under] >= target * 0.95:
            break

        idx_over = np.where(labels == over)[0]
        if len(idx_over) == 0:
            break

        dist_over = ((feats_z[idx_over] - centers[over]) ** 2).sum(axis=1)
        dist_under = ((feats_z[idx_over] - centers[under]) ** 2).sum(axis=1)
        delta = dist_under - dist_over

        score = delta + 0.05 * (sizes[idx_over] / (target + 1e-12))
        best = idx_over[int(np.argmin(score))]
        labels[best] = under

    return labels


def build_region_clients_cluster_balanced(
    data_tensor: torch.Tensor,
    config: TrainConfig,
) -> list[np.ndarray]:
    feats_z, sizes = extract_region_features(
        data_tensor,
        t_in=config.t_in,
        t_out=config.t_out,
        season_lags=(24, 48),
    )
    labels, centers = kmeans_cluster(
        feats_z, num_clients=config.num_clients, seed=config.seed
    )
    balanced_labels = balance_clusters_by_size(
        labels,
        feats_z,
        centers,
        sizes,
        num_clients=config.num_clients,
    )
    region_indices = [
        np.where(balanced_labels == client_id)[0].astype(int)
        for client_id in range(config.num_clients)
    ]

    print("\n[Client split summary: cluster + balanced]")
    for client_id in range(config.num_clients):
        ks = region_indices[client_id]
        est = int(np.sum(sizes[ks]))
        print(
            f"Client {client_id}: #regions={len(ks)}, "
            f"est_samples~{est}, region_ids(head)={ks[:10].tolist()}"
        )
    return region_indices


# =====================
# 4. 数据集
# =====================
class RegionDataset(Dataset):
    """
    Each sample:
      x: (N, T_in), using N as channels
      y: scalar predicting channel-0 at next step
    """

    def __init__(
        self,
        data: torch.Tensor,
        region_ids: np.ndarray,
        t_in: int = 24,
        t_out: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data = data
        self.region_ids = np.array(region_ids, dtype=int)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.stride = int(stride)

        self.num_nodes, _, self.total_steps = data.shape
        self.valid_t = self.total_steps - self.t_in - self.t_out
        self.num_regions = len(self.region_ids)

        if self.valid_t <= 0 or self.num_regions <= 0:
            self.t_positions = np.array([], dtype=int)
            self.length = 0
        else:
            self.t_positions = np.arange(0, self.valid_t, self.stride, dtype=int)
            self.length = int(self.num_regions * len(self.t_positions))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tlen = len(self.t_positions)
        k_pos = idx // tlen
        t0 = int(self.t_positions[idx % tlen])
        region_id = int(self.region_ids[k_pos])

        x = self.data[:, region_id, t0 : t0 + self.t_in]
        y = self.data[0, region_id, t0 + self.t_in]

        x = torch.log1p(x).float()
        y = torch.log1p(y).float()
        return x, y


def split_indices(
    n: int,
    seed: int = 15,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


# =====================
# 5. 模型定义
# =====================
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
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
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
        pred = self.fc(out).squeeze(-1)
        return pred


# =====================
# 6. 评估与损失
# =====================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        preds.append(torch.expm1(pred).detach().cpu())
        trues.append(torch.expm1(y).detach().cpu())

    preds = torch.cat(preds, dim=0) if preds else torch.tensor([])
    trues = torch.cat(trues, dim=0) if trues else torch.tensor([])
    if preds.numel() == 0:
        return np.nan, np.nan

    mse = nn.functional.mse_loss(preds, trues).item()
    mae = nn.functional.l1_loss(preds, trues).item()
    return mse, mae


def mixed_raw_loss(
    pred_log: torch.Tensor,
    y_log: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    pred_raw = torch.expm1(pred_log)
    y_raw = torch.expm1(y_log)
    mse = nn.functional.mse_loss(pred_raw, y_raw)
    huber = nn.functional.smooth_l1_loss(pred_raw, y_raw)
    return alpha * mse + beta * huber


def fedavg_weighted(
    weights: list[dict[str, torch.Tensor]],
    ns: list[int],
) -> dict[str, torch.Tensor]:
    avg = copy.deepcopy(weights[0])
    total = float(np.sum(ns)) + 1e-12

    for key in avg.keys():
        if avg[key].dtype in (torch.float16, torch.float32, torch.float64):
            avg[key].zero_()
            for w, n in zip(weights, ns):
                avg[key] += w[key] * (float(n) / total)
        else:
            avg[key] = weights[0][key]
    return avg


# =====================
# 7. 联邦客户端
# =====================
class FedClient:
    def __init__(
        self,
        cid: int,
        base_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: TrainConfig,
    ) -> None:
        self.cid = int(cid)
        self.model = copy.deepcopy(base_model).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.train_losses: list[float] = []
        self.val_mses: list[float] = []

    def train_one_round(
        self,
        server_snapshot_weights: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], float, float]:
        self.model.load_state_dict(server_snapshot_weights)
        self.model.train()

        w0 = {
            key: value.detach().clone().to(self.device)
            for key, value in server_snapshot_weights.items()
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        total_loss, steps = 0.0, 0
        for _ in range(self.config.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)

                loss = mixed_raw_loss(
                    pred,
                    y,
                    alpha=self.config.alpha_mse,
                    beta=self.config.beta_huber,
                )

                prox = 0.0
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        prox = prox + torch.sum((param - w0[name]) ** 2)
                loss = loss + 0.5 * self.config.mu * prox

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1

        avg_train = total_loss / max(steps, 1)
        val_mse, _ = evaluate(self.model, self.val_loader, self.device)

        self.train_losses.append(avg_train)
        self.val_mses.append(val_mse)

        return copy.deepcopy(self.model.state_dict()), avg_train, val_mse


# =====================
# 8. 客户端构建
# =====================
def build_clients(
    data_tensor: torch.Tensor,
    region_indices: list[np.ndarray],
    server_model: nn.Module,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, Any]:
    clients = []
    client_train_loaders = []
    client_val_loaders = []
    client_test_loaders = []
    client_train_sizes = []

    for cid, region in enumerate(region_indices):
        full_ds = RegionDataset(
            data_tensor,
            region,
            t_in=config.t_in,
            t_out=config.t_out,
            stride=config.stride,
        )
        if len(full_ds) <= 0:
            continue

        tr_idx, va_idx, te_idx = split_indices(
            len(full_ds),
            seed=config.seed + cid,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )

        ds_tr = Subset(full_ds, tr_idx.tolist())
        ds_va = Subset(full_ds, va_idx.tolist())
        ds_te = Subset(full_ds, te_idx.tolist())

        tr_loader = DataLoader(
            ds_tr, batch_size=config.batch_size, shuffle=True, drop_last=False
        )
        va_loader = DataLoader(
            ds_va, batch_size=config.batch_size, shuffle=False, drop_last=False
        )
        te_loader = DataLoader(
            ds_te, batch_size=config.batch_size, shuffle=False, drop_last=False
        )

        client = FedClient(
            cid,
            server_model,
            tr_loader,
            va_loader,
            device,
            config,
        )
        clients.append(client)
        client_train_loaders.append(tr_loader)
        client_val_loaders.append(va_loader)
        client_test_loaders.append(te_loader)
        client_train_sizes.append(len(ds_tr))

    print("有效客户端数量:", len(clients))
    if not clients:
        raise RuntimeError("没有有效客户端：请检查 K、T_in、T_out 或数据。")

    return {
        "clients": clients,
        "train_loaders": client_train_loaders,
        "val_loaders": client_val_loaders,
        "test_loaders": client_test_loaders,
        "train_sizes": client_train_sizes,
    }


# =====================
# 9. 联邦训练
# =====================
def train_federated(
    server_model: nn.Module,
    clients: list[FedClient],
    client_test_loaders: list[DataLoader],
    client_train_sizes: list[int],
    device: torch.device,
    config: TrainConfig,
) -> tuple[nn.Module, list[float], list[float]]:
    del device
    server_weights = copy.deepcopy(server_model.state_dict())
    server_round_trainloss = []
    server_round_fed_model_testmse_mean = []

    for round_id in range(config.rounds):
        print(f"\n----- 轮次 {round_id + 1}/{config.rounds} -----")
        server_snapshot = copy.deepcopy(server_weights)
        local_weights, local_train_losses, local_ns = [], [], []

        for idx, client in enumerate(clients):
            weights, train_loss, val_mse = client.train_one_round(server_snapshot)
            local_weights.append(weights)
            local_train_losses.append(train_loss)
            local_ns.append(client_train_sizes[idx])

            print(
                f"客户端 {client.cid} | "
                f"TrainLoss(mixed+prox): {train_loss:.6f} | "
                f"ValMSE(raw): {val_mse:.6f}"
            )

        aggregated = fedavg_weighted(local_weights, local_ns)
        for key in server_weights.keys():
            if server_weights[key].dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                server_weights[key] = (
                    (1 - config.server_lr) * server_weights[key]
                    + config.server_lr * aggregated[key]
                )
            else:
                server_weights[key] = aggregated[key]

        server_model.load_state_dict(server_weights)

        round_avg = float(np.mean(local_train_losses)) if local_train_losses else np.nan
        server_round_trainloss.append(round_avg)

        mses = []
        for te_loader in client_test_loaders:
            mse, _ = evaluate(server_model, te_loader, clients[0].device)
            mses.append(mse)
        mse_mean = float(np.mean(mses)) if mses else np.nan
        server_round_fed_model_testmse_mean.append(mse_mean)

        print(
            f"轮次 {round_id + 1} 聚合完成 | "
            f"AvgTrainLoss: {round_avg:.6f} | "
            f"FedModelTestMSE(mean): {mse_mean:.6f}"
        )

    return server_model, server_round_trainloss, server_round_fed_model_testmse_mean


# =====================
# 10. 独立训练与个性化评估
# =====================
def train_local_from_scratch(
    in_channels: int,
    train_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> nn.Module:
    model = CNNLSTMAttention(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    for _ in range(config.independent_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mixed_raw_loss(
                pred,
                y,
                alpha=config.alpha_mse,
                beta=config.beta_huber,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def finetune_personalized_earlystop(
    server_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> nn.Module:
    model = copy.deepcopy(server_model).to(device)

    for p in model.cnn[0].parameters():
        p.requires_grad = False
    for p in model.cnn[2].parameters():
        p.requires_grad = True

    for p in model.lstm.parameters():
        p.requires_grad = True
    for p in model.attn.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    params_cnn_last = list(model.cnn[2].parameters())
    params_lstm = list(model.lstm.parameters())
    params_head = list(model.attn.parameters()) + list(model.fc.parameters())

    optimizer = torch.optim.Adam(
        [
            {"params": params_cnn_last, "lr": config.ft_lr * 0.3},
            {"params": params_lstm, "lr": config.ft_lr * 1.0},
            {"params": params_head, "lr": config.ft_lr * 2.0},
        ],
        weight_decay=1e-5,
    )

    best_w = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_rounds = 0

    for _ in range(config.ft_max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mixed_raw_loss(
                pred,
                y,
                alpha=config.alpha_mse,
                beta=config.beta_huber,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_mse, _ = evaluate(model, val_loader, device)
        if val_mse < best_val * 0.999:
            best_val = val_mse
            best_w = copy.deepcopy(model.state_dict())
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= config.ft_patience:
                break

    model.load_state_dict(best_w)
    return model


def final_evaluation(
    server_model: nn.Module,
    client_train_loaders: list[DataLoader],
    client_val_loaders: list[DataLoader],
    client_test_loaders: list[DataLoader],
    clients: list[FedClient],
    in_channels: int,
    device: torch.device,
    config: TrainConfig,
) -> pd.DataFrame:
    rows = []
    print("\n===== Final Comparison (TEST split): Federated vs Independent =====")

    for idx, client in enumerate(clients):
        tr_loader = client_train_loaders[idx]
        va_loader = client_val_loaders[idx]
        te_loader = client_test_loaders[idx]

        print(f"\n[Client {client.cid}] Evaluating ...")

        fed_model = finetune_personalized_earlystop(
            server_model,
            tr_loader,
            va_loader,
            device,
            config,
        )
        f_mse, f_mae = evaluate(fed_model, te_loader, device)
        print(f"  Federated | TEST MSE: {f_mse:.6f} | MAE: {f_mae:.6f}")

        ind_model = train_local_from_scratch(
            in_channels,
            tr_loader,
            device,
            config,
        )
        i_mse, i_mae = evaluate(ind_model, te_loader, device)
        print(f"  Independent | TEST MSE: {i_mse:.6f} | MAE: {i_mae:.6f}")

        rows.append(
            {
                "Client": f"Client {client.cid}",
                "Federated_MSE": f_mse,
                "Federated_MAE": f_mae,
                "Independent_MSE": i_mse,
                "Independent_MAE": i_mae,
            }
        )

    df_res = pd.DataFrame(rows)
    print("\n===== Result Table =====")
    print(df_res)
    return df_res


# =====================
# 11. 结果整理与可视化
# =====================
def build_plot_frames(
    df_res: pd.DataFrame,
    clients: list[FedClient],
    server_round_trainloss: list[float],
    server_round_fed_model_testmse_mean: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    client_order = sorted(
        df_res["Client"].unique(),
        key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else s,
    )
    method_order = ["Federated", "Independent"]

    df_mse = pd.concat(
        [
            df_res[["Client", "Federated_MSE"]]
            .rename(columns={"Federated_MSE": "Value"})
            .assign(Method="Federated", Metric="MSE"),
            df_res[["Client", "Independent_MSE"]]
            .rename(columns={"Independent_MSE": "Value"})
            .assign(Method="Independent", Metric="MSE"),
        ],
        ignore_index=True,
    )

    df_mae = pd.concat(
        [
            df_res[["Client", "Federated_MAE"]]
            .rename(columns={"Federated_MAE": "Value"})
            .assign(Method="Federated", Metric="MAE"),
            df_res[["Client", "Independent_MAE"]]
            .rename(columns={"Independent_MAE": "Value"})
            .assign(Method="Independent", Metric="MAE"),
        ],
        ignore_index=True,
    )

    df_rmse = pd.concat(
        [
            df_res[["Client"]].assign(
                Method="Federated",
                Metric="RMSE",
                Value=np.sqrt(df_res["Federated_MSE"].values),
            ),
            df_res[["Client"]].assign(
                Method="Independent",
                Metric="RMSE",
                Value=np.sqrt(df_res["Independent_MSE"].values),
            ),
        ],
        ignore_index=True,
    )

    df_long = pd.concat([df_mse, df_rmse, df_mae], ignore_index=True)
    df_long["Method"] = pd.Categorical(
        df_long["Method"], categories=method_order, ordered=True
    )
    df_long["Client"] = pd.Categorical(
        df_long["Client"], categories=client_order, ordered=True
    )

    df_fed = pd.DataFrame(
        {
            "Round": np.arange(1, len(server_round_trainloss) + 1),
            "AvgTrainLoss": server_round_trainloss,
            "FedModelTestMSE_mean": server_round_fed_model_testmse_mean,
        }
    )

    df_client_val = pd.concat(
        [
            pd.DataFrame(
                {
                    "Round": np.arange(1, len(client.val_mses) + 1),
                    "Client": f"Client {client.cid}",
                    "ValMSE": client.val_mses,
                }
            )
            for client in clients
        ],
        ignore_index=True,
    )
    df_client_val["Client"] = pd.Categorical(
        df_client_val["Client"], categories=client_order, ordered=True
    )

    stability_rows = []
    for metric in ["MSE", "MAE"]:
        for method in method_order:
            sub = df_long[
                (df_long["Metric"] == metric) & (df_long["Method"] == method)
            ].sort_values("Client")
            vals = sub["Value"].to_numpy(dtype=float)
            std = float(np.std(vals)) if vals.size else np.nan
            gap = float(np.max(vals) - np.min(vals)) if vals.size else np.nan
            mean = float(np.mean(vals)) if vals.size else np.nan
            cv = float(std / (mean + 1e-12)) if vals.size else np.nan
            stability_rows.extend(
                [
                    {"Statistic": f"{metric}-STD", "Value": std, "Method": method},
                    {"Statistic": f"{metric}-GAP", "Value": gap, "Method": method},
                    {"Statistic": f"{metric}-CV", "Value": cv, "Method": method},
                ]
            )

    df_stability = pd.DataFrame(stability_rows)
    df_stability["Method"] = pd.Categorical(
        df_stability["Method"], categories=method_order, ordered=True
    )

    return df_long, df_fed, df_client_val, client_order, method_order, df_stability


def create_summary_figure(
    df_long: pd.DataFrame,
    df_fed: pd.DataFrame,
    df_client_val: pd.DataFrame,
    df_stability: pd.DataFrame,
    client_order: list[str],
    rounds: int,
    output_path: Path,
    show_plot: bool,
) -> None:
    round_axis = np.arange(1, rounds + 1)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    sns.barplot(
        data=df_long[df_long["Metric"] == "MSE"],
        x="Client",
        y="Value",
        hue="Method",
        ax=ax1,
    )
    ax1.set_title("(a) MSE Comparison (Regional Clients)")
    ax1.set_xlabel("")
    ax1.set_ylabel("MSE")
    ax1.legend(title="Method", loc="lower right", frameon=True)

    sns.barplot(
        data=df_long[df_long["Metric"] == "RMSE"],
        x="Client",
        y="Value",
        hue="Method",
        ax=ax2,
    )
    ax2.set_title("(b) RMSE Comparison (Regional Clients)")
    ax2.set_xlabel("")
    ax2.set_ylabel("RMSE")
    ax2.legend(title="Method", loc="lower right", frameon=True)

    sns.barplot(
        data=df_long[df_long["Metric"] == "MAE"],
        x="Client",
        y="Value",
        hue="Method",
        ax=ax3,
    )
    ax3.set_title("(c) MAE Comparison (Regional Clients)")
    ax3.set_xlabel("")
    ax3.set_ylabel("MAE")
    ax3.legend(title="Method", loc="lower right", frameon=True)

    ax4.plot(
        df_fed["Round"],
        df_fed["AvgTrainLoss"],
        marker="o",
        linewidth=2,
        label="AvgTrainLoss",
    )
    ax4.set_xlabel("Federated Round")
    ax4.set_ylabel("AvgTrainLoss")
    ax4.set_xticks(round_axis)
    ax4.set_title("(d) Federated Convergence")

    for client_name in client_order:
        sub = df_client_val[df_client_val["Client"] == client_name].sort_values(
            "Round"
        )
        ax5.plot(
            sub["Round"],
            sub["ValMSE"],
            marker="o",
            linewidth=2,
            label=str(client_name),
        )
    ax5.set_title("(e) Client Validation Convergence (Federated)")
    ax5.set_xlabel("Federated Round")
    ax5.set_ylabel("Val MSE")
    ax5.set_xticks(round_axis)
    ax5.legend(title="Client", loc="upper right", frameon=True)

    sns.barplot(
        data=df_stability,
        x="Statistic",
        y="Value",
        hue="Method",
        ax=ax6,
    )
    ax6.set_title("(f) Cross-Client Error Stability (log-scale)")
    ax6.set_xlabel("")
    ax6.set_yscale("log")
    ax6.tick_params(axis="x", rotation=30)
    ax6.legend(title="Method")

    fallback_font = FontProperties(family="DejaVu Sans")
    for label in ax6.get_yticklabels():
        label.set_fontproperties(fallback_font)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"已保存图件: {output_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_outputs(
    df_res: pd.DataFrame,
    df_long: pd.DataFrame,
    df_fed: pd.DataFrame,
    df_client_val: pd.DataFrame,
    df_stability: pd.DataFrame,
    results_dir: Path,
    show_plot: bool,
    rounds: int,
    client_order: list[str],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(results_dir / "ccn_federated_result_table.csv", index=False)
    df_long.to_csv(results_dir / "ccn_federated_metric_long.csv", index=False)
    df_fed.to_csv(results_dir / "ccn_federated_round_history.csv", index=False)
    df_client_val.to_csv(
        results_dir / "ccn_federated_client_val_history.csv", index=False
    )
    df_stability.to_csv(
        results_dir / "ccn_federated_stability_stats.csv", index=False
    )

    create_summary_figure(
        df_long=df_long,
        df_fed=df_fed,
        df_client_val=df_client_val,
        df_stability=df_stability,
        client_order=client_order,
        rounds=rounds,
        output_path=results_dir / "ccn_federated_summary.png",
        show_plot=show_plot,
    )


# =====================
# 12. 主流程
# =====================
def main() -> None:
    args = parse_args()
    config = TrainConfig()

    set_seed(config.seed)
    configure_plot_style()
    device = get_device()
    data_tensor = load_data_tensor(args.data_path)
    num_nodes, _, _ = data_tensor.shape

    region_indices = build_region_clients_cluster_balanced(data_tensor, config)
    server_model = CNNLSTMAttention(in_channels=num_nodes).to(device)

    bundle = build_clients(
        data_tensor=data_tensor,
        region_indices=region_indices,
        server_model=server_model,
        device=device,
        config=config,
    )

    server_model, server_round_trainloss, server_round_fed_model_testmse_mean = (
        train_federated(
            server_model=server_model,
            clients=bundle["clients"],
            client_test_loaders=bundle["test_loaders"],
            client_train_sizes=bundle["train_sizes"],
            device=device,
            config=config,
        )
    )

    df_res = final_evaluation(
        server_model=server_model,
        client_train_loaders=bundle["train_loaders"],
        client_val_loaders=bundle["val_loaders"],
        client_test_loaders=bundle["test_loaders"],
        clients=bundle["clients"],
        in_channels=num_nodes,
        device=device,
        config=config,
    )

    (
        df_long,
        df_fed,
        df_client_val,
        client_order,
        _,
        df_stability,
    ) = build_plot_frames(
        df_res=df_res,
        clients=bundle["clients"],
        server_round_trainloss=server_round_trainloss,
        server_round_fed_model_testmse_mean=server_round_fed_model_testmse_mean,
    )

    save_outputs(
        df_res=df_res,
        df_long=df_long,
        df_fed=df_fed,
        df_client_val=df_client_val,
        df_stability=df_stability,
        results_dir=args.results_dir,
        show_plot=args.show_plot,
        rounds=config.rounds,
        client_order=client_order,
    )


if __name__ == "__main__":
    main()
