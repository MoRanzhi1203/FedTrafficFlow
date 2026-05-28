"""
GCN + BiLSTM + Multi-Head Attention 联邦交通流预测模型（基础版）

后续接入Q-Traffic或项目真实数据时，仅需实现新的Dataset类替换当前仿真数据集，
模型结构、客户端实现、服务器逻辑及训练流程无需修改。

模型架构：
- GCNEncoder (可学习邻接矩阵 + 图卷积层) 提取空间维度/节点维度特征
- BiLSTM 提取时间序列特征
- GCN 和 LSTM 输出堆叠为 (B, 2, hidden_dim) 的特征序列
- Multi-Head Attention 进行特征融合
- RegressionHead 输出单步交通流预测值

输入特征矩阵 X 维度：(B, K, T)
- B: batch size
- K: 交通节点数、路段特征通道数或函数曲线聚类后的特征单元数
- T: 时间步长（默认24）
"""

import argparse
import json
import os
import random
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


DEFAULT_SEED = 48
DEFAULT_K = 5
DEFAULT_T = 24
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_CLIENTS = 3
DEFAULT_NUM_ROUNDS = 5
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 0.001
DEFAULT_NOISE = 0.1


def set_seed(seed=DEFAULT_SEED):
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
    parser = argparse.ArgumentParser(description="GCN + BiLSTM + Multi-Head Attention 联邦交通流预测（基础版）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--num-rounds", type=int, default=DEFAULT_NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--show-plot", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="results/gcn_fed_base")
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
# SimpleGCNLayer
# ============================================================
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, X, A_norm):
        AX = torch.einsum("ij,bjf->bif", A_norm, X)
        return self.lin(AX)


# ============================================================
# GCNEncoder
# ============================================================
class GCNEncoder(nn.Module):
    def __init__(self, K, T, hidden_dim=128):
        super().__init__()
        self.K = K
        self.T = T
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Sequential(
            nn.Linear(T, hidden_dim),
            nn.LayerNorm(hidden_dim),
            AdaptiveSwish(),
        )

        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()

        self.A_param = nn.Parameter(torch.randn(K, K) * 0.01)

    def _normalize_adj(self, A):
        A = torch.relu(A)
        I = torch.eye(self.K, device=A.device, dtype=A.dtype)
        A = A + I
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        B, K, T = x.shape
        X = self.node_proj(x)
        A_norm = self._normalize_adj(self.A_param)

        H = self.gcn1(X, A_norm)
        H = self.norm1(H)
        H = self.act(H)
        H = self.gcn2(H, A_norm)
        H = self.norm2(H)
        H = self.act(H)

        g = H.mean(dim=1)
        return g


# ============================================================
# GCN + BiLSTM + Multi-Head Attention 模型
# ============================================================
class AttentionFedModel(nn.Module):
    def __init__(self, K, T, hidden_dim=DEFAULT_HIDDEN_DIM, num_heads=DEFAULT_NUM_HEADS):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
            )
        self.K = K
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.gcn_encoder = GCNEncoder(K=K, T=T, hidden_dim=hidden_dim)

        self.lstm = nn.LSTM(
            input_size=K,
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
        x = x.to(dtype=torch.float32)
        B, K, T = x.shape

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


# ============================================================
# WeakModel
# ============================================================
class WeakModel(nn.Module):
    def __init__(self, K, T, hidden_dim=16):
        super().__init__()
        self.K = K
        self.T = T
        self.simple_extractor = nn.Sequential(
            nn.Linear(K * T, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.8),
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        B, K, T = x.shape
        x = x.reshape(B, K * T)
        x = self.simple_extractor(x)
        return self.fc(x), None


# ============================================================
# HeterogeneousDataset
# ============================================================
class HeterogeneousDataset(Dataset):
    def __init__(self, client_id, num_samples, K, T, noise=DEFAULT_NOISE):
        self.X = np.random.randn(num_samples, K, T).astype(np.float32)
        base_feature = self.X[:, :, T // 4: T * 3 // 4].mean(axis=(1, 2))
        if client_id == 0:
            self.y = (
                0.6 * np.sin(base_feature)
                + 0.4 * np.sin(self.X[:, :, : T // 2].mean(axis=(1, 2)))
                + noise * np.random.randn(num_samples)
            )
        elif client_id == 1:
            self.y = (
                0.6 * np.sin(base_feature)
                + 0.4 * np.cos(self.X[:, :, T // 2:].mean(axis=(1, 2)))
                + noise * np.random.randn(num_samples)
            )
        else:
            self.y = (
                0.6 * np.sin(base_feature)
                + 0.4 * np.tanh(self.X.max(axis=(1, 2)))
                + noise * np.random.randn(num_samples)
            )
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ============================================================
# FedClient
# ============================================================
class FedClient:
    def __init__(self, client_id, model, train_loader, test_loader, criterion, lr=DEFAULT_LR, device=None):
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

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x = x.to(self.device).float()
            y = y.to(self.device).float().squeeze()
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        avg_loss = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float().squeeze()
                pred, _ = self.model(x)
                total_loss += self.criterion(pred.squeeze(), y).item() * x.shape[0]
        avg_loss = total_loss / len(self.test_loader.dataset)
        self.val_losses.append(avg_loss)
        self.scheduler.step()
        return avg_loss

    def train(self, epochs=DEFAULT_LOCAL_EPOCHS, global_model=None):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"  Local epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        return self.train_losses[-1], self.model.state_dict()

    def test(self):
        self.model.eval()
        preds, truths, all_attn = [], [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float().squeeze()
                pred, attn = self.model(x)
                preds.append(pred.squeeze().cpu().numpy())
                truths.append(y.cpu().numpy())
                if attn is not None:
                    all_attn.append(attn.cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        mse = float(np.mean((preds - truths) ** 2))
        mae = float(np.mean(np.abs(preds - truths)))
        att_mean = np.concatenate(all_attn, axis=0).mean() if all_attn else None
        return {"mse": mse, "mae": mae, "att_weights": att_mean}


# ============================================================
# WeakClient
# ============================================================
class WeakClient:
    def __init__(self, client_id, model, train_loader, test_loader, criterion, lr=0.02, device=None):
        self.client_id = client_id
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device).float()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x = x.to(self.device).float()
            y = y.to(self.device).float().squeeze()
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred.squeeze(), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.shape[0]
        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float().squeeze()
                pred, _ = self.model(x)
                total_loss += self.criterion(pred.squeeze(), y).item() * x.shape[0]
        return total_loss / len(self.test_loader.dataset)

    def train(self, epochs=2):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"  Independent epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    def test(self):
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float().squeeze()
                pred, _ = self.model(x)
                preds.append(pred.squeeze().cpu().numpy())
                truths.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        mse = float(np.mean((preds - truths) ** 2))
        mae = float(np.mean(np.abs(preds - truths)))
        return {"mse": mse, "mae": mae}


# ============================================================
# Server
# ============================================================
class Server:
    def __init__(self, model, num_clients, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.global_model = model.to(self.device).float()
        self.num_clients = num_clients
        self.round_losses = []
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = sizes

    def aggregate(self, client_weights, client_losses):
        data_weights = np.array(self.client_data_sizes, dtype=float) / float(sum(self.client_data_sizes))
        loss_w = np.exp(-np.array(client_losses, dtype=float) * 2.0)
        loss_weights = loss_w / (loss_w.sum() + 1e-12)

        weights = 0.5 * data_weights + 0.5 * loss_weights
        weights = weights / (weights.sum() + 1e-12)

        global_dict = self.global_model.state_dict()
        new_dict = {}

        for k, v in global_dict.items():
            if v.dtype in (torch.float32, torch.float64, torch.float16):
                new_dict[k] = torch.zeros_like(v, dtype=torch.float32)
            else:
                new_dict[k] = v.clone()

        for k in list(new_dict.keys()):
            if global_dict[k].dtype not in (torch.float32, torch.float64, torch.float16):
                continue
            for i in range(self.num_clients):
                cw = client_weights[i][k].to(self.device, dtype=torch.float32)
                new_dict[k] += cw * float(weights[i])

        for k in list(new_dict.keys()):
            if global_dict[k].dtype in (torch.float32, torch.float64, torch.float16):
                new_dict[k] = 0.9 * global_dict[k].to(torch.float32) + 0.1 * new_dict[k]
            else:
                new_dict[k] = new_dict[k].to(global_dict[k].dtype)

        self.global_model.load_state_dict(new_dict)
        self.round_losses.append(float(np.mean(client_losses)))
        return self.global_model.state_dict()


# ============================================================
# 训练与评估
# ============================================================
def train_federated(fed_clients, server, num_rounds, local_epochs):
    print("===== Federated Training =====")
    for rnd in range(num_rounds):
        print(f"\n----- Round {rnd + 1}/{num_rounds} -----")
        client_weights, client_losses = [], []
        for client in fed_clients:
            print(f"Client {client.client_id} training:")
            loss, weights = client.train(epochs=local_epochs, global_model=server.global_model)
            client_weights.append(weights)
            client_losses.append(loss)
        server.aggregate(client_weights, client_losses)
        print(f"Round average federated loss: {server.round_losses[-1]:.4f}")


def train_independent(weak_clients):
    print("\n===== Independent Training =====")
    for client in weak_clients:
        print(f"\nClient {client.client_id} training:")
        client.train(epochs=2)


def evaluate(fed_clients, weak_clients):
    print("\n===== Performance Comparison =====")
    fed_metrics = [c.test() for c in fed_clients]
    weak_metrics = [c.test() for c in weak_clients]
    num_clients = len(fed_clients)
    for i in range(num_clients):
        print(f"\nClient {i}:")
        print(f"  Federated  - MSE: {fed_metrics[i]['mse']:.4f}, MAE: {fed_metrics[i]['mae']:.4f}")
        print(f"  Independent- MSE: {weak_metrics[i]['mse']:.4f}, MAE: {weak_metrics[i]['mae']:.4f}")
        if fed_metrics[i].get("att_weights") is not None:
            print(f"  Mean attention weight: {np.round(fed_metrics[i]['att_weights'].mean(), 4)}")
    return fed_metrics, weak_metrics


def plot_results(fed_metrics, weak_metrics, fed_clients, server, output_dir, show_plot=False):
    os.makedirs(output_dir, exist_ok=True)
    num_clients = len(fed_metrics)
    client_labels = [f"Client {i}" for i in range(num_clients)]

    fed_mse = [m["mse"] for m in fed_metrics]
    fed_rmse = [np.sqrt(m["mse"]) for m in fed_metrics]
    fed_mae = [m["mae"] for m in fed_metrics]
    weak_mse = [m["mse"] for m in weak_metrics]
    weak_rmse = [np.sqrt(m["mse"]) for m in weak_metrics]
    weak_mae = [m["mae"] for m in weak_metrics]

    df_metrics = pd.DataFrame({
        "Client": client_labels * 2,
        "Method": ["Federated"] * num_clients + ["Independent"] * num_clients,
        "MSE": fed_mse + weak_mse,
        "RMSE": fed_rmse + weak_rmse,
        "MAE": fed_mae + weak_mae,
    })
    df_long = df_metrics.melt(
        id_vars=["Client", "Method"], value_vars=["MSE", "RMSE", "MAE"],
        var_name="Metric", value_name="Value",
    )

    df_global = pd.DataFrame({
        "Round": np.arange(1, len(server.round_losses) + 1),
        "AvgTrainLoss": server.round_losses,
    })

    df_client_val = pd.concat(
        [pd.DataFrame({
            "Round": np.arange(1, len(c.val_losses) + 1),
            "Client": f"Client {c.client_id}",
            "ValLoss": c.val_losses,
        }) for c in fed_clients],
        ignore_index=True,
    )

    def stability_stats(arr):
        arr = np.array(arr, dtype=float)
        std = float(arr.std())
        gap = float(arr.max() - arr.min())
        mean = float(arr.mean())
        cv = float(std / (mean + 1e-12))
        return std, gap, cv

    fed_mse_std, fed_mse_gap, fed_mse_cv = stability_stats(fed_mse)
    weak_mse_std, weak_mse_gap, weak_mse_cv = stability_stats(weak_mse)
    fed_mae_std, fed_mae_gap, fed_mae_cv = stability_stats(fed_mae)
    weak_mae_std, weak_mae_gap, weak_mae_cv = stability_stats(weak_mae)

    df_stability = pd.DataFrame({
        "Statistic": ["MSE-STD", "MSE-GAP", "MSE-CV", "MAE-STD", "MAE-GAP", "MAE-CV"] * 2,
        "Value": [fed_mse_std, fed_mse_gap, fed_mse_cv, fed_mae_std, fed_mae_gap, fed_mae_cv]
        + [weak_mse_std, weak_mse_gap, weak_mse_cv, weak_mae_std, weak_mae_gap, weak_mae_cv],
        "Method": ["Federated"] * 6 + ["Independent"] * 6,
    })

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    sns.barplot(data=df_long[df_long["Metric"] == "MSE"], x="Client", y="Value", hue="Method", ax=axes[0, 0])
    axes[0, 0].set_title("MSE Comparison")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend(title="Method", loc="lower right", frameon=True)

    sns.barplot(data=df_long[df_long["Metric"] == "RMSE"], x="Client", y="Value", hue="Method", ax=axes[0, 1])
    axes[0, 1].set_title("RMSE Comparison")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].legend(title="Method", loc="lower right", frameon=True)

    sns.barplot(data=df_long[df_long["Metric"] == "MAE"], x="Client", y="Value", hue="Method", ax=axes[0, 2])
    axes[0, 2].set_title("MAE Comparison")
    axes[0, 2].set_xlabel("")
    axes[0, 2].set_ylabel("MAE")
    axes[0, 2].legend(title="Method", loc="lower right", frameon=True)

    sns.lineplot(data=df_global, x="Round", y="AvgTrainLoss", marker="o", ax=axes[1, 0])
    axes[1, 0].set_title("Federated Convergence (Global)")
    axes[1, 0].set_xlabel("Federated Round")
    axes[1, 0].set_ylabel("Avg Train Loss")

    sns.lineplot(data=df_client_val, x="Round", y="ValLoss", hue="Client", marker="o", ax=axes[1, 1])
    axes[1, 1].set_title("Client Validation Convergence (Federated)")
    axes[1, 1].set_xlabel("Federated Round")
    axes[1, 1].set_ylabel("Validation Loss")
    axes[1, 1].legend(title="Client")

    sns.barplot(data=df_stability, x="Statistic", y="Value", hue="Method", ax=axes[1, 2])
    axes[1, 2].set_title("Cross-Client Error Stability (Dispersion)")
    axes[1, 2].set_xlabel("")
    axes[1, 2].set_ylabel("Value")
    axes[1, 2].tick_params(axis="x", rotation=30)
    axes[1, 2].legend(title="Method")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "fed_vs_weak_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n对比图已保存至: {fig_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def save_metrics(fed_metrics, weak_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for i in range(len(fed_metrics)):
        results[f"client_{i}"] = {
            "federated": {
                "mse": fed_metrics[i]["mse"],
                "rmse": float(np.sqrt(fed_metrics[i]["mse"])),
                "mae": fed_metrics[i]["mae"],
            },
            "independent": {
                "mse": weak_metrics[i]["mse"],
                "rmse": float(np.sqrt(weak_metrics[i]["mse"])),
                "mae": weak_metrics[i]["mae"],
            },
        }
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"指标已保存至: {json_path}")

    csv_rows = []
    for i in range(len(fed_metrics)):
        csv_rows.append({
            "client": i,
            "fed_mse": fed_metrics[i]["mse"],
            "fed_rmse": float(np.sqrt(fed_metrics[i]["mse"])),
            "fed_mae": fed_metrics[i]["mae"],
            "weak_mse": weak_metrics[i]["mse"],
            "weak_rmse": float(np.sqrt(weak_metrics[i]["mse"])),
            "weak_mae": weak_metrics[i]["mae"],
        })
    csv_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"指标CSV已保存至: {csv_path}")


def main():
    args = parse_args()

    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"hidden_dim ({args.hidden_dim}) 必须能被 num_heads ({args.num_heads}) 整除。"
            f"建议 hidden_dim={args.num_heads * (args.hidden_dim // args.num_heads + 1)}"
        )

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    sns.set_theme(
        style="whitegrid", context="notebook", font="DejaVu Sans",
        rc={"axes.unicode_minus": False, "figure.titlesize": 18, "axes.titlesize": 16,
            "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
            "legend.fontsize": 11, "legend.title_fontsize": 12},
    )

    K, T = args.K, args.T
    num_clients = args.num_clients
    samples_per_client = [50, 80, 120][:num_clients]
    if len(samples_per_client) < num_clients:
        samples_per_client = [50 + i * 20 for i in range(num_clients)]

    criterion = nn.MSELoss()
    g_split = torch.Generator().manual_seed(args.seed)

    fed_clients = []
    weak_clients = []

    for cid in range(num_clients):
        dataset = HeterogeneousDataset(
            client_id=cid,
            num_samples=samples_per_client[cid],
            K=K, T=T, noise=args.noise,
        )
        train_size = int(0.8 * len(dataset))
        train_data, test_data = random_split(
            dataset, [train_size, len(dataset) - train_size], generator=g_split,
        )
        g_loader = torch.Generator().manual_seed(args.seed + cid)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=g_loader)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        fed_clients.append(FedClient(
            cid,
            AttentionFedModel(K=K, T=T, hidden_dim=args.hidden_dim, num_heads=args.num_heads),
            train_loader, test_loader, criterion, lr=args.lr, device=device,
        ))
        weak_clients.append(WeakClient(
            cid, WeakModel(K=K, T=T), train_loader, test_loader, criterion, device=device,
        ))

    server = Server(
        AttentionFedModel(K=K, T=T, hidden_dim=args.hidden_dim, num_heads=args.num_heads),
        num_clients, device=device,
    )
    server.set_client_data_sizes(samples_per_client)

    train_federated(fed_clients, server, args.num_rounds, args.local_epochs)
    train_independent(weak_clients)
    fed_metrics, weak_metrics = evaluate(fed_clients, weak_clients)

    plot_results(fed_metrics, weak_metrics, fed_clients, server, args.output_dir, show_plot=args.show_plot)
    save_metrics(fed_metrics, weak_metrics, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
