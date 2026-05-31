# -*- coding: utf-8 -*-
"""Standalone CCN simulation project with isolated outputs."""

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_ccn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_NAME = "ccn"


def configure_plot_style() -> None:
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


def build_output_file_name(workflow_name: str, artifact_name: str, extension: str) -> str:
    return f"{PROJECT_NAME}_{workflow_name}_{artifact_name}.{extension}"


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


def stability_stats(arr):
    arr = np.array(arr, dtype=float)
    std = float(arr.std())
    gap = float(arr.max() - arr.min())
    mean = float(arr.mean())
    cv = float(std / (mean + 1e-12))
    return std, gap, cv


def print_summary_table(results_summary: Dict[str, Dict[str, float]]) -> pd.DataFrame:
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
    def __init__(self, trainable: bool = True):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class WeakModel(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 16):
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
        x = x.to(dtype=torch.float32)
        batch_size, k, t = x.shape
        x = x.view(batch_size, k * t)
        x = self.simple_extractor(x)
        return self.fc(x), None


class CCNOverviewModel(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
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
class CCNAblationFull(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
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
            input_size=k,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)
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
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
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
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.mean(dim=1)
        x_lstm = self.lstm_proj(x_lstm)
        fused = self.fuse(torch.cat([x_cnn, x_lstm], dim=1))
        return self.head(fused), None


class LSTMAttentionHetero(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=k,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
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


class CCNAblationCNNAttention(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 128, num_heads: int = 4):
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
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x_cnn = self.cnn(x)
        feat_seq = x_cnn.unsqueeze(1)
        attn_out, attn_w = self.mha(feat_seq, feat_seq, feat_seq)
        attn_out = self.attn_norm(attn_out + feat_seq)
        fused = attn_out.mean(dim=1)
        return self.head(fused), attn_w
class OverviewHeterogeneousDataset(Dataset):
    def __init__(self, client_id: int, num_samples: int, k: int, t: int, noise: float = 0.1):
        self.x = np.random.randn(num_samples, k, t)
        base_feature = self.x[:, :, t // 4 : t * 3 // 4].mean(axis=(1, 2))
        if client_id == 0:
            self.y = 0.6 * np.sin(base_feature) + 0.4 * np.sin(self.x[:, :, : t // 2].mean(axis=(1, 2))) + noise * np.random.randn(num_samples)
        elif client_id == 1:
            self.y = 0.6 * np.sin(base_feature) + 0.4 * np.cos(self.x[:, :, t // 2 :].mean(axis=(1, 2))) + noise * np.random.randn(num_samples)
        else:
            self.y = 0.6 * np.sin(base_feature) + 0.4 * np.tanh(self.x.max(axis=(1, 2))) + noise * np.random.randn(num_samples)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class AblationHeterogeneousDataset(Dataset):
    def __init__(self, client_id: int, num_samples: int, k: int, t: int, noise: float = 0.1):
        self.x = np.random.randn(num_samples, k, t).astype(np.float32)
        base_feature = self.x[:, :, t // 4 : t * 3 // 4].mean(axis=(1, 2))
        if client_id == 0:
            y = 0.6 * np.sin(base_feature) + 0.4 * np.sin(self.x[:, :, : t // 2].mean(axis=(1, 2)))
        elif client_id == 1:
            y = 0.6 * np.sin(base_feature) + 0.4 * np.cos(self.x[:, :, t // 2 :].mean(axis=(1, 2)))
        else:
            y = 0.6 * np.sin(base_feature) + 0.4 * np.tanh(self.x.max(axis=(1, 2)))
        y = y + noise * np.random.randn(num_samples).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class FederatedClient:
    def __init__(self, client_id, model, train_loader, test_loader, criterion, lr: float = 1e-3):
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

    def train_local(self, epochs: int = 5, global_model=None, verbose: bool = False, prefix: str = "Local"):
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
    def __init__(self, client_id, model, train_loader, test_loader, criterion):
        super().__init__(client_id, model, train_loader, test_loader, criterion, lr=0.02)

    def train_local(self, epochs: int = 2, verbose: bool = False):
        return super().train_local(epochs=epochs, global_model=None, verbose=verbose, prefix="Independent")


class WeightedFederatedServer:
    def __init__(self, model, num_clients: int):
        self.global_model = model.to(DEVICE).float()
        self.num_clients = num_clients
        self.round_losses = []
        self.client_data_sizes = None

    def set_client_data_sizes(self, sizes):
        self.client_data_sizes = sizes

    def aggregate(self, client_weights, client_losses):
        data_weights = np.array(self.client_data_sizes) / float(sum(self.client_data_sizes))
        loss_weights = np.exp(-np.array(client_losses) * 2.0)
        loss_weights = loss_weights / (loss_weights.sum() + 1e-12)

        weights = 0.5 * data_weights + 0.5 * loss_weights
        weights = weights / (weights.sum() + 1e-12)

        global_dict = self.global_model.state_dict()
        new_dict = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_dict.items()}

        for key in new_dict.keys():
            for idx in range(self.num_clients):
                client_weight = client_weights[idx][key].to(DEVICE, dtype=torch.float32)
                new_dict[key] += client_weight * torch.tensor(float(weights[idx]), device=DEVICE, dtype=torch.float32)

        for key in new_dict.keys():
            new_dict[key] = 0.9 * global_dict[key] + 0.1 * new_dict[key]

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
)-> pd.DataFrame:
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
            "Method": ["Federated"] * len(client_labels) + ["Independent"] * len(client_labels),
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
    df_global = pd.DataFrame({"Round": round_axis, "AvgTrainLoss": server.round_losses})

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
            "Statistic": ["MSE-STD", "MSE-GAP", "MSE-CV", "MAE-STD", "MAE-GAP", "MAE-CV"] * 2,
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
            "Method": ["Federated"] * 6 + ["Independent"] * 6,
        }
    )

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
    heat = df_stab.pivot_table(index="Client", columns="Model", values="rmse", aggfunc="mean")
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

    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Test RMSE")
    ax1.set_title("(a) Convergence of Test RMSE (mean ± std)")
    ax1.set_xticks(np.arange(1, rounds + 1, dtype=int))
    ax1.legend(frameon=True)

    sns.violinplot(data=df_stab, x="Model", y="rmse", inner=None, cut=0, ax=ax2)
    sns.stripplot(data=df_stab, x="Model", y="rmse", color="k", size=4, alpha=0.6, ax=ax2)
    ax2.set_xlabel("Model Variant")
    ax2.set_ylabel("Final Test RMSE")
    ax2.set_title("(b) Client-level Stability (Final RMSE)")
    ax2.tick_params(axis="x", rotation=15)

    if len(df_delta) > 0:
        df_delta_melt = df_delta.melt(id_vars=["Model"], var_name="Metric", value_name="DeltaPercent")
        sns.barplot(data=df_delta_melt, x="Model", y="DeltaPercent", hue="Metric", ax=ax3)
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
def run_weighted_ablation(
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
    set_global_seed(seed)
    criterion = nn.MSELoss()
    split_gen = torch.Generator().manual_seed(seed)

    def build_loaders():
        train_loaders, test_loaders = [], []
        for cid in range(num_clients):
            dataset = AblationHeterogeneousDataset(
                client_id=cid,
                num_samples=samples_per_client[cid],
                k=k,
                t=t,
            )
            train_size = int(0.8 * len(dataset))
            train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size], generator=split_gen)
            loader_gen = torch.Generator().manual_seed(seed + cid)
            train_loader = DataLoader(train_data, batch_size=8, shuffle=True, generator=loader_gen)
            test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        return train_loaders, test_loaders

    def eval_global_on_clients(global_model, clients):
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
            FederatedClient(cid, ctor(), train_loaders[cid], test_loaders[cid], criterion, lr=1e-3)
            for cid in range(num_clients)
        ]
        server = WeightedFederatedServer(ctor(), num_clients)
        server.set_client_data_sizes(samples_per_client)

        hist_train_client, hist_train_mean, hist_train_std = [], [], []
        hist_test_client, hist_test_mean, hist_test_std = [], [], []

        print(f"\nStart Federated Training: {name}")
        for rnd in range(num_rounds):
            print(f"  Round {rnd + 1}/{num_rounds}")
            client_weights, client_losses = [], []
            for client in clients:
                loss, weights = client.train_local(epochs=local_epochs, global_model=server.global_model, verbose=False)
                client_weights.append(weights)
                client_losses.append(float(loss))
                print(f"    Client {client.client_id} | Local avg MSE: {loss:.6f}")

            server.aggregate(client_weights, client_losses)
            hist_train_client.append(client_losses)
            hist_train_mean.append(float(np.mean(client_losses)))
            hist_train_std.append(float(np.std(client_losses, ddof=0)))

            per_client_metrics = eval_global_on_clients(server.global_model, clients)
            per_client_rmse = np.array([m["rmse"] for m in per_client_metrics], dtype=float)
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
                "Delta_RMSE_%": (summary["rmse_mean"] - full["rmse_mean"]) / (full["rmse_mean"] + 1e-12) * 100.0,
                "Delta_MAE_%": (summary["mae_mean"] - full["mae_mean"]) / (full["mae_mean"] + 1e-12) * 100.0,
            }
        )
    df_delta = pd.DataFrame(delta_rows)

    plot_ablation_figure(df_conv, df_stab, df_delta, client_labels, num_rounds, output_dir, figure_name)
    summary_df = print_summary_table(results_summary)
    save_dataframe(summary_df, output_dir, metrics_file_name)
    return results_summary
def run_overview_experiment(output_dir: Path) -> None:
    seed = 15
    num_rounds = 6
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
        dataset = OverviewHeterogeneousDataset(client_id=client_id, num_samples=samples_per_client[client_id], k=k, t=t)
        train_size = int(0.8 * len(dataset))
        train_data, test_data = random_split(dataset, [train_size, len(dataset) - train_size], generator=split_gen)

        loader_gen = torch.Generator()
        loader_gen.manual_seed(seed + client_id)
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True, generator=loader_gen)
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

        fed_clients.append(FederatedClient(client_id, CCNOverviewModel(k=k, t=t), train_loader, test_loader, criterion, lr=1e-3))
        weak_clients.append(IndependentClient(client_id, WeakModel(k=k, t=t), train_loader, test_loader, criterion))

    server = WeightedFederatedServer(CCNOverviewModel(k=k, t=t), num_clients)
    server.set_client_data_sizes(samples_per_client)

    print("\n===== CCN Overview =====")
    for rnd in range(num_rounds):
        print(f"[overview] round {rnd + 1}/{num_rounds}")
        client_weights, client_losses = [], []
        for client in fed_clients:
            loss, weights = client.train_local(epochs=local_epochs, global_model=server.global_model, verbose=False)
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
        print(f"  Federated   - MSE: {fed_metrics[idx]['mse']:.4f}, MAE: {fed_metrics[idx]['mae']:.4f}")
        print(f"  Independent - MSE: {weak_metrics[idx]['mse']:.4f}, MAE: {weak_metrics[idx]['mae']:.4f}")
        if fed_metrics[idx]["att_weights"] is not None:
            print(f"  Mean attention weight: {np.round(fed_metrics[idx]['att_weights'].mean(), 4)}")

    overview_df = plot_overview_figure(
        fed_metrics,
        weak_metrics,
        server,
        fed_clients,
        output_dir,
        build_output_file_name("overview", "figure", "png"),
    )
    save_dataframe(overview_df, output_dir, build_output_file_name("overview", "metrics", "csv"))


def run_ablation_experiment(output_dir: Path) -> Dict[str, Dict[str, float]]:
    return run_weighted_ablation(
        workflow_name="CCN Heterogeneous Ablation",
        seed=15,
        num_clients=3,
        k=5,
        t=24,
        samples_per_client=[50, 80, 120],
        num_rounds=5,
        local_epochs=5,
        full_name="CCN-LSTM-Attention",
        variants=OrderedDict(
            [
                ("CCN-LSTM-Attention", lambda: CCNAblationFull(k=5, t=24, hidden_dim=128, num_heads=4)),
                ("CCN-LSTM", lambda: CCNAblationCNNLSTM(k=5, t=24, hidden_dim=128)),
                ("LSTM-Attention", lambda: LSTMAttentionHetero(k=5, t=24, hidden_dim=128, num_heads=4)),
                ("CCN-Attention", lambda: CCNAblationCNNAttention(k=5, t=24, hidden_dim=128, num_heads=4)),
            ]
        ),
        output_dir=output_dir,
        figure_name=build_output_file_name("ablation", "figure", "png"),
        metrics_file_name=build_output_file_name("ablation", "metrics", "csv"),
    )


def run_project(workflow: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    log_path = output_dir / build_output_file_name("run", "log", "txt")
    with log_path.open("w", encoding="utf-8") as log_handle, redirect_stdout(TeeStream(sys.stdout, log_handle)):
        configure_plot_style()
        print(f"[setup] Using device: {DEVICE}")
        print(f"[setup] Writing experiment log: {log_path}")

        if workflow in ("all", "overview"):
            run_overview_experiment(output_dir)

        if workflow in ("all", "ablation"):
            run_ablation_experiment(output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Standalone CCN simulation project.")
    parser.add_argument(
        "--workflow",
        choices=["all", "overview", "ablation"],
        default="all",
        help="Workflow to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used to save generated figures.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    run_project(args.workflow, Path(args.output_dir))


if __name__ == "__main__":
    main()
