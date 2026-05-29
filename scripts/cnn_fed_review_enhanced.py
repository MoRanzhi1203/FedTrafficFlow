"""
CNN + BiLSTM + Multi-Head Attention 联邦交通流预测模型（一审增强版）

在基础版模型架构上增加：
- 复杂 Non-IID 异构仿真数据（多种输入分布 + 交通模式）
- 多种聚合策略（FedAvg / loss_weighted / data_loss_weighted）
- 消融实验（full / no_attention / lstm_only / spatial_only / weak）
- 多随机种子实验
- 客户端掉线模拟
- 可选 DP 高斯噪声
- 通信开销估计

模型架构（保持 CNN + BiLSTM + Multi-Head Attention 基础框架）：
- CNN (Conv1d) 提取空间维度/节点维度特征
- BiLSTM 提取时间序列特征
- Multi-Head Attention 特征融合
- RegressionHead 输出单步交通流预测值
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_HEADS = 4


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="CNN + BiLSTM + Attention 联邦交通流预测（增强版）")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seeds", type=str, default=None,
                       help="逗号分隔的随机种子列表，为 None 时自动生成")
    parser.add_argument("--client-dropout-rate", type=float, default=0.0)
    parser.add_argument("--dp-noise-std", type=float, default=0.0)
    parser.add_argument("--show-plot", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="results/cnn_fed_review_enhanced")
    return parser.parse_args()


# ============================================================
# AdaptiveSwish
# ============================================================
class AdaptiveSwish(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta.to(x.dtype) * x)


# ============================================================
# 模型定义（保持 CNN 基础框架）
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self, K, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=K, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.cnn(x)


class BiLSTMEncoder(nn.Module):
    def __init__(self, K, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=K, hidden_size=hidden_dim // 2, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.proj(x)


class CNN_LSTM_Attention(nn.Module):
    """Full: CNN + BiLSTM + Multi-Head Attention"""
    def __init__(self, K, T, hidden_dim=DEFAULT_HIDDEN_DIM, num_heads=DEFAULT_NUM_HEADS):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.cnn_enc = CNNEncoder(K, hidden_dim)
        self.lstm_enc = BiLSTMEncoder(K, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn_enc(x)
        x_lstm = self.lstm_enc(x)
        feat_seq = torch.stack([x_cnn, x_lstm], dim=1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w


class CNN_LSTM_NoAttention(nn.Module):
    """消融: CNN + LSTM (无 Attention)"""
    def __init__(self, K, T, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.cnn_enc = CNNEncoder(K, hidden_dim)
        self.lstm_enc = BiLSTMEncoder(K, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), AdaptiveSwish(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn_enc(x)
        x_lstm = self.lstm_enc(x)
        fused = self.fuse(torch.cat([x_cnn, x_lstm], dim=1))
        return self.head(fused), None


class LSTM_Only(nn.Module):
    """消融: 仅 LSTM (无 CNN, 无 Attention)"""
    def __init__(self, K, T, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.lstm_enc = BiLSTMEncoder(K, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_lstm = self.lstm_enc(x)
        return self.head(x_lstm), None


class CNN_Only(nn.Module):
    """消融: 仅 CNN (无 LSTM, 无 Attention)"""
    def __init__(self, K, T, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.cnn_enc = CNNEncoder(K, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), AdaptiveSwish(), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn_enc(x)
        return self.head(x_cnn), None


class WeakModel(nn.Module):
    def __init__(self, K, T, hidden_dim=16):
        super().__init__()
        self.extractor = nn.Sequential(nn.Linear(K * T, hidden_dim), nn.ReLU(), nn.Dropout(0.8))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        B, K, T = x.shape
        x = x.reshape(B, K * T)
        return self.fc(self.extractor(x)), None


MODEL_BUILDERS = {
    "full": lambda K, T: CNN_LSTM_Attention(K, T),
    "no_attention": lambda K, T: CNN_LSTM_NoAttention(K, T),
    "lstm_only": lambda K, T: LSTM_Only(K, T),
    "spatial_only": lambda K, T: CNN_Only(K, T),
    "weak": lambda K, T: WeakModel(K, T),
}


# ============================================================
# 复杂 Non-IID 仿真数据集
# ============================================================
def _generate_traffic_pattern(T, pattern):
    t = np.arange(T, dtype=np.float32) / T
    if pattern == "morning_peak":
        return np.exp(-((t - 0.25) / 0.1) ** 2) + 0.3
    elif pattern == "evening_peak":
        return np.exp(-((t - 0.7) / 0.1) ** 2) + 0.3
    elif pattern == "double_peak":
        return np.exp(-((t - 0.25) / 0.1) ** 2) + np.exp(-((t - 0.7) / 0.12) ** 2) + 0.2
    elif pattern == "flat":
        return np.ones(T, dtype=np.float32)
    else:
        return np.ones(T, dtype=np.float32)


class ComplexHeterogeneousDataset(Dataset):
    def __init__(self, client_id, num_samples, K, T, noise=0.1,
                 dist_type="normal", pattern_type="double_peak",
                 missing_rate=0.0, outlier_rate=0.0, seed=0):
        self.K = K
        self.T = T
        rng = np.random.RandomState(seed + client_id * 1000)

        if dist_type == "normal":
            self.X = rng.randn(num_samples, K, T).astype(np.float32)
        elif dist_type == "t":
            self.X = rng.standard_t(df=4, size=(num_samples, K, T)).astype(np.float32) * 0.5
        elif dist_type == "chi2":
            self.X = rng.chisquare(df=4, size=(num_samples, K, T)).astype(np.float32) * 0.3 - 0.5
        elif dist_type == "lognormal":
            self.X = rng.lognormal(mean=0, sigma=0.6, size=(num_samples, K, T)).astype(np.float32) - 0.5
        else:
            self.X = rng.randn(num_samples, K, T).astype(np.float32)

        pattern = _generate_traffic_pattern(T, pattern_type)[np.newaxis, np.newaxis, :]

        base_feature = self.X[:, :, T // 4: T * 3 // 4].mean(axis=(1, 2))
        rel = 0.7 * np.sin(base_feature) + 0.3 * np.cos(self.X.max(axis=(1, 2)))
        if client_id == 0:
            rel += 0.15 * np.tanh(self.X[:, :, : T // 2].mean(axis=(1, 2)))
        elif client_id == 1:
            rel += 0.15 * np.sin(self.X[:, :, T // 2:].mean(axis=(1, 2)))
        else:
            rel += 0.15 * np.cos(self.X.std(axis=(1, 2)))

        self.y = (rel[:, np.newaxis] * pattern.squeeze(axis=1)).mean(axis=1) + noise * rng.randn(num_samples).astype(np.float32)
        self.y = self.y.astype(np.float32)

        if missing_rate > 0:
            mask = rng.rand(*self.X.shape) < missing_rate
            self.X[mask] = 0.0
        if outlier_rate > 0:
            mask = rng.rand(*self.X.shape) < outlier_rate
            self.X[mask] += rng.randn(*self.X.shape)[mask].astype(np.float32) * 5.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================
# DP 工具
# ============================================================
def apply_dp_noise(model, dp_noise_std):
    if dp_noise_std <= 0:
        return
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * dp_noise_std)


# ============================================================
# FedClient (支持掉线)
# ============================================================
class FedClient:
    def __init__(self, client_id, model, train_loader, test_loader, criterion, lr=0.001, device=None):
        self.client_id = client_id
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device).float()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.9)
        self.train_losses = []
        self.val_losses = []
        self.data_size = len(train_loader.dataset)

    def train_epoch(self):
        self.model.train()
        total_loss, n = 0.0, 0
        for x, y in self.train_loader:
            x = x.to(self.device).float()
            y = y.to(self.device).float().squeeze()
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        avg = total_loss / max(1, n)
        self.train_losses.append(avg)
        return avg

    def validate(self):
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float().squeeze()
                pred, _ = self.model(x)
                total_loss += self.criterion(pred.squeeze(), y).item() * x.size(0)
                n += x.size(0)
        avg = total_loss / max(1, n)
        self.val_losses.append(avg)
        self.scheduler.step()
        return avg

    def train(self, epochs=5, global_model=None):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        for epoch in range(epochs):
            self.train_epoch()
            self.validate()
        return self.train_losses[-1] if self.train_losses else float("inf"), copy.deepcopy(self.model.state_dict())

    def test_metrics(self):
        self.model.eval()
        preds, truths = [], []
        for x, y in self.test_loader:
            x = x.to(self.device).float()
            y = y.to(self.device).float().squeeze()
            with torch.no_grad():
                pred, _ = self.model(x)
            preds.append(pred.squeeze().cpu().numpy())
            truths.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        diff = preds - truths
        return {
            "mse": float((diff ** 2).mean()),
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "mae": float(np.abs(diff).mean()),
        }


# ============================================================
# Server (多种聚合策略 + 平滑)
# ============================================================
class Server:
    def __init__(self, model, num_clients, device=None, agg_method="data_loss_weighted", agg_lambda=0.5):
        self.device = device if device is not None else torch.device("cpu")
        self.global_model = model.to(self.device).float()
        self.num_clients = num_clients
        self.agg_method = agg_method
        self.agg_lambda = agg_lambda
        self.round_losses = []
        self.agg_weights_history = []
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = np.array(sizes, dtype=float)

    def aggregate(self, client_weights, client_losses, active_ids):
        global_dict = self.global_model.state_dict()
        n_active = len(active_ids)

        data_sizes = np.array([self.client_data_sizes[i] for i in active_ids], dtype=float)
        losses = np.array([client_losses[i] for i in active_ids], dtype=float)

        if self.agg_method == "fedavg":
            w = data_sizes / (data_sizes.sum() + 1e-12)
        elif self.agg_method == "loss_weighted":
            lw = np.exp(-losses * 2.0)
            w = lw / (lw.sum() + 1e-12)
        elif self.agg_method == "data_loss_weighted":
            d_w = data_sizes / (data_sizes.sum() + 1e-12)
            lw = np.exp(-losses * 2.0)
            l_w = lw / (lw.sum() + 1e-12)
            w = self.agg_lambda * d_w + (1.0 - self.agg_lambda) * l_w
            w = w / (w.sum() + 1e-12)
        else:
            w = np.ones(n_active) / n_active

        self.agg_weights_history.append({int(active_ids[j]): float(w[j]) for j in range(n_active)})

        new_dict = {}
        for k, v in global_dict.items():
            if v.dtype in (torch.float32, torch.float64, torch.float16):
                new_dict[k] = torch.zeros_like(v, dtype=torch.float32)
            else:
                new_dict[k] = v.clone()

        for k in list(new_dict.keys()):
            if global_dict[k].dtype not in (torch.float32, torch.float64, torch.float16):
                continue
            for j, cid in enumerate(active_ids):
                cw = client_weights[cid][k].to(self.device, dtype=torch.float32)
                new_dict[k] += cw * float(w[j])

        for k in list(new_dict.keys()):
            if global_dict[k].dtype in (torch.float32, torch.float64, torch.float16):
                new_dict[k] = 0.9 * global_dict[k].to(torch.float32) + 0.1 * new_dict[k]
            else:
                new_dict[k] = new_dict[k].to(global_dict[k].dtype)

        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(losses)))
        return self.global_model.state_dict()


# ============================================================
# 模型大小计算
# ============================================================
def get_model_size_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


# ============================================================
# 单次运行
# ============================================================
def run_single_experiment(seed, args, model_type, agg_method, dist_types, pattern_types, missing_rates, outlier_rates, verbose=True):
    set_seed(seed)
    device = get_device()
    K, T = args.K, args.T
    num_clients = args.num_clients
    samples_per_client = [50, 80, 120][:num_clients]
    if len(samples_per_client) < num_clients:
        samples_per_client = [50 + i * 20 for i in range(num_clients)]

    criterion = nn.MSELoss()
    g_split = torch.Generator().manual_seed(seed)

    clients = []
    for cid in range(num_clients):
        dataset = ComplexHeterogeneousDataset(
            client_id=cid, num_samples=samples_per_client[cid], K=K, T=T,
            noise=args.noise, dist_type=dist_types[cid % len(dist_types)],
            pattern_type=pattern_types[cid % len(pattern_types)],
            missing_rate=missing_rates[cid % len(missing_rates)],
            outlier_rate=outlier_rates[cid % len(outlier_rates)],
            seed=seed,
        )
        train_size = int(0.8 * len(dataset))
        train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size], generator=g_split)
        g_loader = torch.Generator().manual_seed(seed + cid)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=g_loader)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        model = MODEL_BUILDERS[model_type](K, T)
        clients.append(FedClient(cid, model, train_loader, test_loader, criterion, lr=args.lr, device=device))

    server = Server(MODEL_BUILDERS[model_type](K, T), num_clients, device=device, agg_method=agg_method)
    server.set_client_data_sizes(samples_per_client)
    model_size_bytes = get_model_size_bytes(MODEL_BUILDERS[model_type](K, T))

    total_comm_bytes = 0.0
    client_losses_hist = {cid: [] for cid in range(num_clients)}
    run_log = {"seed": seed, "model_type": model_type, "agg_method": agg_method, "rounds": []}

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"[Seed={seed}] {model_type} + {agg_method}")
        print(f"{'=' * 50}")

    for rnd in range(args.num_rounds):
        if verbose:
            print(f"\n--- Round {rnd + 1}/{args.num_rounds} ---")

        active_ids = [cid for cid in range(num_clients) if random.random() >= args.client_dropout_rate]
        if not active_ids:
            active_ids = [0]

        client_weights = [None] * num_clients
        client_losses = [0.0] * num_clients

        for cid in active_ids:
            loss, weights = clients[cid].train(
                epochs=args.local_epochs,
                global_model=server.global_model,
            )
            apply_dp_noise(clients[cid].model, args.dp_noise_std)
            client_weights[cid] = weights
            client_losses[cid] = loss
            client_losses_hist[cid].append(loss)
            if verbose:
                print(f"  Client {cid}: loss={loss:.6f}")

        server.aggregate(client_weights, client_losses, active_ids)

        n_active = len(active_ids)
        round_comm = 2.0 * n_active * model_size_bytes
        total_comm_bytes += round_comm

        per_client = []
        for cid in range(num_clients):
            clients[cid].model.load_state_dict(server.global_model.state_dict())
            per_client.append(clients[cid].test_metrics())

        run_log["rounds"].append({
            "round": rnd + 1,
            "active_clients": active_ids,
            "n_active": n_active,
            "client_losses": {int(c): float(client_losses[c]) for c in range(num_clients)},
            "round_comm_bytes": round_comm,
            "per_client_metrics": per_client,
            "avg_loss": float(server.round_losses[-1]) if server.round_losses else 0.0,
        })

        if verbose:
            rmses = [m["rmse"] for m in per_client]
            print(f"  Round {rnd + 1} | RMSE mean={np.mean(rmses):.6f}, std={np.std(rmses):.6f}")

    final_metrics = []
    for cid in range(num_clients):
        clients[cid].model.load_state_dict(server.global_model.state_dict())
        final_metrics.append(clients[cid].test_metrics())

    df_final = pd.DataFrame(final_metrics)
    summary = {
        "seed": seed,
        "model_type": model_type,
        "agg_method": agg_method,
        "mse_mean": float(df_final["mse"].mean()),
        "mse_std": float(df_final["mse"].std(ddof=0)),
        "rmse_mean": float(df_final["rmse"].mean()),
        "rmse_std": float(df_final["rmse"].std(ddof=0)),
        "mae_mean": float(df_final["mae"].mean()),
        "mae_std": float(df_final["mae"].std(ddof=0)),
        "total_comm_bytes": total_comm_bytes,
    }
    for cid in range(num_clients):
        summary[f"client_{cid}_rmse"] = float(final_metrics[cid]["rmse"])
        summary[f"client_{cid}_mae"] = float(final_metrics[cid]["mae"])
        summary[f"client_{cid}_mse"] = float(final_metrics[cid]["mse"])

    losses_record = {"client_losses": {int(c): [float(v) for v in vals] for c, vals in client_losses_hist.items()},
                     "round_losses": [float(x) for x in server.round_losses],
                     "agg_weights": server.agg_weights_history}
    return summary, run_log, losses_record


# ============================================================
# 结果可视化（全面优化版）
# ============================================================
MODEL_COLORS = {
    "full": "#2E86AB", "no_attention": "#A23B72",
    "lstm_only": "#F18F01", "spatial_only": "#C73E1D", "weak": "#6A4C93",
}
AGG_COLORS = {
    "fedavg": "#4C9F70", "loss_weighted": "#E76F51",
    "data_loss_weighted": "#457B9D",
}
MODEL_LABELS = {
    "full": "Full (CNN+LSTM+Attn)",
    "no_attention": "No Attention",
    "lstm_only": "LSTM Only",
    "spatial_only": "CNN Only",
    "weak": "Weak Model",
}
AGG_LABELS = {
    "fedavg": "FedAvg",
    "loss_weighted": "Loss-Weighted",
    "data_loss_weighted": "Data+Loss Weighted",
}
CLIENT_COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261"]


def _save_or_show(fig, path, show_plot):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  已保存: {path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def generate_all_figures(all_losses_records, df_per_seed, comm_df, model_types,
                         agg_methods, output_dir, show_plot=False):
    os.makedirs(output_dir, exist_ok=True)

    _fig1_model_performance(df_per_seed, model_types, agg_methods, output_dir, show_plot)
    _fig2_convergence_analysis(all_losses_records, model_types, agg_methods, output_dir, show_plot)
    _fig3_aggregation_strategy(all_losses_records, df_per_seed, comm_df,
                               model_types, agg_methods, output_dir, show_plot)
    _fig4_client_analysis(df_per_seed, model_types, agg_methods, output_dir, show_plot)


def _style_ax(ax, title=None, xlabel=None, ylabel=None, grid=True):
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if grid:
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _fig1_model_performance(df, model_types, agg_methods, output_dir, show_plot):
    best_agg = agg_methods[0]
    df_best = df[df["agg_method"] == best_agg].copy()
    if df_best.empty:
        df_best = df

    agg_order = ["full", "no_attention", "lstm_only", "spatial_only", "weak"]
    order = [m for m in agg_order if m in df_best["model_type"].values]
    df_best["rmse_mean"] = df_best["rmse_mean"].astype(float)
    df_best["mae_mean"] = df_best["mae_mean"].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    (ax0, ax1), (ax2, ax3) = axes

    colors_rmse = [MODEL_COLORS.get(m, "#888888") for m in order]
    bars0 = ax0.bar(order, [float(df_best[df_best["model_type"] == m]["rmse_mean"].iloc[0]) for m in order],
                     color=colors_rmse, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, m in zip(bars0, order):
        val = float(df_best[df_best["model_type"] == m]["rmse_mean"].iloc[0])
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _style_ax(ax0, "Final Test RMSE by Model Type", ylabel="RMSE")
    ax0.set_xticks(range(len(order)))
    ax0.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], rotation=15, ha="right", fontsize=10)

    colors_mae = [MODEL_COLORS.get(m, "#888888") for m in order]
    bars1 = ax1.bar(order, [float(df_best[df_best["model_type"] == m]["mae_mean"].iloc[0]) for m in order],
                     color=colors_mae, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, m in zip(bars1, order):
        val = float(df_best[df_best["model_type"] == m]["mae_mean"].iloc[0])
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    _style_ax(ax1, "Final Test MAE by Model Type", ylabel="MAE")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], rotation=15, ha="right", fontsize=10)

    pivot = df.pivot_table(values="rmse_mean", index="model_type", columns="agg_method", aggfunc="mean")
    pivot = pivot.reindex([m for m in order if m in pivot.index])
    pivot.index = [MODEL_LABELS.get(m, m) for m in pivot.index]
    pivot.columns = [AGG_LABELS.get(c, c) for c in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", linewidths=0.8,
                cbar_kws={"label": "RMSE", "shrink": 0.8}, ax=ax2, annot_kws={"fontsize": 9})
    _style_ax(ax2, "RMSE Heatmap: Model × Aggregation Strategy", grid=False)

    comm_pivot = df.groupby("model_type")["total_comm_bytes"].mean() / (1024 * 1024)
    comm_pivot.index = [MODEL_LABELS.get(m, m) for m in comm_pivot.index]
    rmse_series = df[df["agg_method"] == best_agg].groupby("model_type")["rmse_mean"].mean()
    comm_data = pd.DataFrame({"Communication (MB)": comm_pivot, "RMSE": rmse_series}).dropna()
    for mt, row in comm_data.iterrows():
        color = MODEL_COLORS.get([k for k, v in MODEL_LABELS.items() if v == mt][0]
                                 if mt in MODEL_LABELS.values() else mt, "#888888")
        ax3.scatter(row["Communication (MB)"], row["RMSE"], s=200, c=color,
                     edgecolors="white", linewidth=2, zorder=5)
        ax3.annotate(mt, (row["Communication (MB)"], row["RMSE"]),
                      textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight="bold")
    _style_ax(ax3, "Communication Cost vs RMSE Trade-off",
              xlabel="Communication (MB)", ylabel="RMSE")

    fig.suptitle("Model Performance Analysis — CNN Federated Learning",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, os.path.join(output_dir, "model_performance.png"), show_plot)


def _fig2_convergence_analysis(all_records, model_types, agg_methods, output_dir, show_plot):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    (ax0, ax1), (ax2, ax3) = axes

    for agg in agg_methods:
        subset = [r for r in all_records if r.get("model_type") == "full" and r.get("agg_method") == agg]
        if subset:
            losses_arr = np.array([s["round_losses"] for s in subset])
            mean_l = losses_arr.mean(axis=0)
            std_l = losses_arr.std(axis=0)
            rounds = np.arange(1, len(mean_l) + 1)
            color = AGG_COLORS.get(agg, "#888888")
            ax0.plot(rounds, mean_l, color=color, linewidth=2.5, marker="o", markersize=6,
                      label=AGG_LABELS.get(agg, agg))
            ax0.fill_between(rounds, mean_l - std_l, mean_l + std_l, alpha=0.12, color=color)
    _style_ax(ax0, "Full Model Convergence by Aggregation Strategy",
              xlabel="Federated Round", ylabel="Avg Train Loss")
    ax0.legend(frameon=True, fontsize=10)

    best_agg_per_model = {}
    for mt in model_types:
        best_val = float("inf")
        for agg in agg_methods:
            subset = [r for r in all_records if r.get("model_type") == mt and r.get("agg_method") == agg]
            if subset:
                avg_final = np.mean([np.mean(s["round_losses"][-3:]) if len(s["round_losses"]) >= 3
                                     else np.mean(s["round_losses"]) for s in subset])
                if avg_final < best_val:
                    best_val = avg_final
                    best_agg_per_model[mt] = agg
        if mt not in best_agg_per_model:
            best_agg_per_model[mt] = agg_methods[0]

    for mt in model_types:
        best_agg = best_agg_per_model.get(mt, agg_methods[0])
        subset = [r for r in all_records if r.get("model_type") == mt and r.get("agg_method") == best_agg]
        if subset:
            losses_arr = np.array([s["round_losses"] for s in subset])
            mean_l = losses_arr.mean(axis=0)
            std_l = losses_arr.std(axis=0)
            rounds = np.arange(1, len(mean_l) + 1)
            color = MODEL_COLORS.get(mt, "#888888")
            ax1.plot(rounds, mean_l, color=color, linewidth=2.5, marker="o", markersize=6,
                      label=MODEL_LABELS.get(mt, mt))
            ax1.fill_between(rounds, mean_l - std_l, mean_l + std_l, alpha=0.10, color=color)
    _style_ax(ax1, "Model Ablation Convergence (Best Agg per Model)",
              xlabel="Federated Round", ylabel="Avg Train Loss")
    ax1.legend(frameon=True, fontsize=9)

    for agg in agg_methods:
        subset = [r for r in all_records if r.get("model_type") == "full" and r.get("agg_method") == agg]
        if subset:
            client_data = {}
            for s in subset:
                for cid, vals in s.get("client_losses", {}).items():
                    client_data.setdefault(int(cid), []).append(vals)
            for cid, runs in client_data.items():
                arr = np.array(runs)
                mean_c = arr.mean(axis=0)
                rounds = np.arange(1, len(mean_c) + 1)
                ax2.plot(rounds, mean_c, linewidth=2, marker="s", markersize=5,
                          label=f"{AGG_LABELS.get(agg,agg)}-C{cid}",
                          color=CLIENT_COLORS[cid % len(CLIENT_COLORS)], alpha=0.85)
    _style_ax(ax2, "Per-Client Loss Trajectories (Full Model)",
              xlabel="Federated Round", ylabel="Client Avg Train Loss")
    ax2.legend(frameon=True, fontsize=8, ncol=2)

    rounds_data = []
    for r in all_records:
        if r.get("model_type") == "full" and r.get("agg_method") == agg_methods[0]:
            for rd in r.get("rounds", []):
                metrics = rd.get("per_client_metrics", [])
                rmse_vals = [m["rmse"] for m in metrics if "rmse" in m]
                if rmse_vals:
                    rounds_data.append({"Round": rd["round"], "RMSE": np.mean(rmse_vals),
                                        "Std": np.std(rmse_vals)})
    rounds_date = [rnd for rnd in rounds_data if rnd.get("Round")]
    if rounds_data:
        df_rnd = pd.DataFrame(rounds_data).groupby("Round").agg(
            RMSE_mean=("RMSE", "mean"), RMSE_std=("Std", "mean")).reset_index()
        ax3.plot(df_rnd["Round"], df_rnd["RMSE_mean"], color="#E76F51",
                  linewidth=2.8, marker="D", markersize=7)
        ax3.fill_between(df_rnd["Round"],
                          df_rnd["RMSE_mean"] - df_rnd["RMSE_std"],
                          df_rnd["RMSE_mean"] + df_rnd["RMSE_std"],
                          alpha=0.15, color="#E76F51")
    _style_ax(ax3, "Global Test RMSE Convergence (Full Model)",
              xlabel="Federated Round", ylabel="Test RMSE (Mean ± Std)")

    fig.suptitle("Convergence Analysis — CNN Federated Learning",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, os.path.join(output_dir, "convergence_analysis.png"), show_plot)


def _fig3_aggregation_strategy(all_records, df, comm_df, model_types, agg_methods, output_dir, show_plot):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    (ax0, ax1), (ax2, ax3) = axes

    agg_weight_records = [r for r in all_records if r.get("agg_weights")]
    if agg_weight_records:
        rec = agg_weight_records[0]
        if rec.get("agg_weights"):
            client_ids = sorted(rec["agg_weights"][0].keys())
            rounds_list = np.arange(1, len(rec["agg_weights"]) + 1)
            cum_data = np.zeros(len(rounds_list))
            for j, cid in enumerate(client_ids):
                vals = np.array([rnd_w.get(cid, 0) for rnd_w in rec["agg_weights"]])
                color = CLIENT_COLORS[cid % len(CLIENT_COLORS)]
                ax0.fill_between(rounds_list, cum_data, cum_data + vals, alpha=0.75,
                                  color=color, label=f"Client {cid}")
                ax0.plot(rounds_list, cum_data + vals, color=color, linewidth=1.5)
                cum_data = cum_data + vals
    _style_ax(ax0, "Aggregation Weight Evolution (Stacked Area)",
              xlabel="Federated Round", ylabel="Cumulative Weight")
    ax0.legend(frameon=True, fontsize=10, loc="center left")

    best_agg = agg_methods[0]
    final_losses = []
    for mt in model_types:
        for agg in agg_methods:
            subset = [r for r in all_records if r.get("model_type") == mt and r.get("agg_method") == agg]
            if subset:
                final = [np.mean(s["round_losses"][-3:]) if len(s["round_losses"]) >= 3
                         else np.mean(s["round_losses"]) for s in subset]
                final_losses.append({
                    "Model": MODEL_LABELS.get(mt, mt),
                    "Aggregation": AGG_LABELS.get(agg, agg),
                    "FinalLoss": np.mean(final),
                    "LossStd": np.std(final),
                })
    if final_losses:
        df_final = pd.DataFrame(final_losses)
        pivot = df_final.pivot_table(values="FinalLoss", index="Model", columns="Aggregation", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd_r", linewidths=0.8,
                     cbar_kws={"label": "Final Avg Loss", "shrink": 0.8}, ax=ax1, annot_kws={"fontsize": 9})
    _style_ax(ax1, "Final Loss Heatmap: Model × Aggregation Strategy", grid=False)

    model_comm = comm_df.groupby("model_type")["total_comm_mb"].mean()
    best_agg2 = agg_methods[-1]
    df_b2 = df[df["agg_method"] == best_agg2]
    if df_b2.empty:
        df_b2 = df
    model_rmse = df_b2.groupby("model_type")["rmse_mean"].mean()

    comm_vals = [float(model_comm.get(mt, 0)) for mt in model_types]
    rmse_vals = [float(model_rmse.get(mt, 0)) for mt in model_types]
    sizes = [max(60, 600 / (r + 0.05)) for r in rmse_vals]
    colors_comm = [MODEL_COLORS.get(mt, "#888888") for mt in model_types]

    for i, mt in enumerate(model_types):
        ax2.barh(i, comm_vals[i], color=colors_comm[i], edgecolor="white", linewidth=1.2, height=0.55)
        ax2.text(comm_vals[i] + 0.05, i, f"{comm_vals[i]:.2f} MB",
                  va="center", fontsize=10, fontweight="bold")
    _style_ax(ax2, "Communication Cost by Model",
              xlabel="Communication (MB)", ylabel="")
    ax2.set_yticks(range(len(model_types)))
    ax2.set_yticklabels([MODEL_LABELS.get(mt, mt) for mt in model_types], fontsize=10)

    for i, mt in enumerate(model_types):
        color = colors_comm[i]
        ax3.scatter(comm_vals[i], rmse_vals[i], s=sizes[i], c=color,
                     edgecolors="white", linewidth=2, zorder=5)
        ax3.annotate(MODEL_LABELS.get(mt, mt),
                      (comm_vals[i], rmse_vals[i]),
                      textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight="bold")
    _style_ax(ax3, "Communication vs RMSE (Bubble: 1/RMSE)",
              xlabel="Communication (MB)", ylabel="RMSE")

    fig.suptitle("Aggregation & Communication Analysis — CNN Federated Learning",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, os.path.join(output_dir, "aggregation_analysis.png"), show_plot)


def _fig4_client_analysis(df, model_types, agg_methods, output_dir, show_plot):
    best_agg = agg_methods[0]
    df_best = df[df["agg_method"] == best_agg]

    client_cols_rmse = [c for c in df.columns if c.endswith("_rmse") and c.startswith("client_")]
    if not client_cols_rmse:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax0, ax1 = axes

    agg_order = ["full", "no_attention", "lstm_only", "spatial_only", "weak"]
    order = [m for m in agg_order if m in df_best["model_type"].values]
    n_models = len(order)
    n_clients = len(client_cols_rmse)
    bar_width = 0.7 / n_models
    x = np.arange(n_clients)

    for j, mt in enumerate(order):
        sub = df_best[df_best["model_type"] == mt]
        if sub.empty:
            continue
        vals = [float(sub[c].iloc[0]) for c in client_cols_rmse]
        offset = (j - (n_models - 1) / 2) * bar_width
        color = MODEL_COLORS.get(mt, "#888888")
        ax0.bar(x + offset, vals, bar_width * 0.85, color=color, edgecolor="white",
                 linewidth=0.8, label=MODEL_LABELS.get(mt, mt))
    _style_ax(ax0, "Per-Client RMSE by Model Type",
              xlabel="Client", ylabel="RMSE")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"Client {i}" for i in range(n_clients)])
    ax0.legend(frameon=True, fontsize=9)

    client_data = []
    for j, mt in enumerate(order):
        sub = df_best[df_best["model_type"] == mt]
        if sub.empty:
            continue
        for ci, col in enumerate(client_cols_rmse):
            client_data.append({
                "Model": MODEL_LABELS.get(mt, mt),
                "Client": f"Client {ci}",
                "RMSE": float(sub[col].iloc[0]),
            })
    if client_data:
        df_c = pd.DataFrame(client_data)
        pivot = df_c.pivot_table(values="RMSE", index="Client", columns="Model", aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", linewidths=0.8,
                     cbar_kws={"label": "RMSE", "shrink": 0.8}, ax=ax1, annot_kws={"fontsize": 9})
    _style_ax(ax1, "Client × Model RMSE Heatmap", grid=False)

    fig.suptitle("Client-Level Performance Analysis — CNN Federated Learning",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, os.path.join(output_dir, "client_analysis.png"), show_plot)


# ============================================================
# main
# ============================================================
def main():
    args = parse_args()

    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"hidden_dim ({args.hidden_dim}) 必须能被 num_heads ({args.num_heads}) 整除。"
            f"建议 hidden_dim={args.num_heads * (args.hidden_dim // args.num_heads + 1)}"
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.seeds is None:
        seeds = [42 + i * 137 for i in range(args.num_runs)]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        if len(seeds) < args.num_runs:
            seeds = seeds + [seeds[-1] + i * 137 for i in range(1, args.num_runs - len(seeds) + 1)]

    model_types = ["full", "no_attention", "lstm_only", "spatial_only", "weak"]
    agg_methods = ["fedavg", "loss_weighted", "data_loss_weighted"]
    dist_types = ["normal", "t", "chi2", "lognormal"]
    pattern_types = ["morning_peak", "evening_peak", "double_peak", "flat"]
    missing_rates = [0.0, 0.05, 0.1]
    outlier_rates = [0.0, 0.02, 0.05]

    hyperparams = {
        "K": args.K, "T": args.T, "hidden_dim": args.hidden_dim, "num_heads": args.num_heads,
        "num_clients": args.num_clients, "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs, "batch_size": args.batch_size,
        "lr": args.lr, "noise": args.noise, "num_runs": args.num_runs,
        "client_dropout_rate": args.client_dropout_rate, "dp_noise_std": args.dp_noise_std,
        "model_types": model_types, "agg_methods": agg_methods,
        "dist_types": dist_types, "pattern_types": pattern_types,
        "seeds": seeds,
    }
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)
    print(f"超参数已保存至: {output_dir}/hyperparameters.json")

    sns.set_theme(
        style="whitegrid", context="notebook", font="DejaVu Sans",
        rc={"axes.unicode_minus": False, "figure.titlesize": 16, "axes.titlesize": 14,
            "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 10, "legend.title_fontsize": 11},
    )

    all_summaries = []
    all_run_logs = []
    all_losses_records = []
    comm_rows = []

    total_experiments = len(model_types) * len(agg_methods) * len(seeds)
    exp_idx = 0

    for model_type in model_types:
        for agg_method in agg_methods:
            for seed in seeds:
                exp_idx += 1
                print(f"\n{'#' * 60}")
                print(f"Experiment {exp_idx}/{total_experiments}: {model_type} + {agg_method} + seed={seed}")
                print(f"{'#' * 60}")

                summary, run_log, losses_rec = run_single_experiment(
                    seed=seed, args=args, model_type=model_type, agg_method=agg_method,
                    dist_types=dist_types, pattern_types=pattern_types,
                    missing_rates=missing_rates, outlier_rates=outlier_rates,
                    verbose=True,
                )
                summary["model_type"] = model_type
                summary["agg_method"] = agg_method
                summary["seed"] = seed
                all_summaries.append(summary)
                all_run_logs.append(run_log)
                losses_rec["model_type"] = model_type
                losses_rec["agg_method"] = agg_method
                losses_rec["seed"] = seed
                all_losses_records.append(losses_rec)

                comm_rows.append({
                    "model_type": model_type,
                    "agg_method": agg_method,
                    "seed": seed,
                    "total_comm_bytes": summary["total_comm_bytes"],
                    "total_comm_mb": summary["total_comm_bytes"] / (1024 * 1024),
                })

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(output_dir, "metrics_per_seed.csv"), index=False)
    print(f"\n逐种子指标已保存至: {output_dir}/metrics_per_seed.csv")

    comm_df = pd.DataFrame(comm_rows)
    comm_df.to_csv(os.path.join(output_dir, "communication_cost.csv"), index=False)
    print(f"通信开销已保存至: {output_dir}/communication_cost.csv")

    group_cols = ["model_type", "agg_method"]
    summary_mean = df_summary.groupby(group_cols).agg(
        mse_mean=("mse_mean", "mean"), mse_std=("mse_mean", "std"),
        rmse_mean=("rmse_mean", "mean"), rmse_std=("rmse_mean", "std"),
        mae_mean=("mae_mean", "mean"), mae_std=("mae_mean", "std"),
        comm_mean_mb=("total_comm_bytes", lambda x: x.mean() / (1024 * 1024)),
    ).reset_index()
    summary_mean.to_csv(os.path.join(output_dir, "summary_mean_std.csv"), index=False)
    print(f"汇总统计已保存至: {output_dir}/summary_mean_std.csv")

    with open(os.path.join(output_dir, "run_logs.json"), "w") as f:
        json.dump(all_run_logs, f, indent=2, ensure_ascii=False)
    print(f"运行日志已保存至: {output_dir}/run_logs.json")

    print("\n===== Summary (mean +- std across seeds) =====")
    print(summary_mean.to_string(index=False))

    generate_all_figures(all_losses_records, df_summary, comm_df, model_types,
                         agg_methods, output_dir, show_plot=args.show_plot)

    print("\nDone.")


if __name__ == "__main__":
    main()
