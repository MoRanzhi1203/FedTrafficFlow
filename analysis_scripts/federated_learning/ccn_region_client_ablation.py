"""区域客户端消融实验脚本。

核心功能：
- 将区域客户端消融 Notebook 重构为可独立运行的 Python 模块；
- 在相同客户端划分与数据划分下比较 Full、去 Attention、去 CNN、去 LSTM 四种模型；
- 输出联邦训练收敛历史、客户端稳定性结果和论文风格对比图。

项目作用：
- 为区域客户端模型结构消融提供工程化实现；
- 复用正式训练脚本的数据加载与客户端划分逻辑，保证对比口径一致。

关键依赖：`torch`、`numpy`、`pandas`、`matplotlib`。
主要输入：预处理流水线最后一步输出的标准区域客户端数据集。
主要输出：各模型收敛历史、客户端指标汇总和消融可视化图像。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_scripts.federated_learning.ccn_region_client_train import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_FILE,
    Attention,
    RegionDataset,
    configure_logging,
    get_device,
    load_data as load_preprocessed_dataset,
    set_seed,
    split_clients as shared_split_clients,
    split_timewise_indices,
    validate_data_tensor,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "federated_learning" / "region_client_ablation"


@dataclass
class AblationConfig:
    seed: int = 15
    num_clients: int = 3
    t_in: int = 24
    t_out: int = 1
    batch_size: int = 32
    rounds: int = 5
    local_epochs: int = 5
    lr: float = 5e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stride: int = 1
    hidden_dim: int = 64
    dataset_dir: Path = DEFAULT_DATASET_DIR
    dataset_file: str = DEFAULT_DATASET_FILE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    show_plot: bool = False
    verbose: bool = False


MODEL_SPECS = {
    "full": "Full (CNN+LSTM+Attn)",
    "cnn_lstm": "w/o Attention (CNN+LSTM)",
    "lstm_attention": "w/o CNN (LSTM+Attn)",
    "cnn_attention": "w/o LSTM (CNN+Attn)",
}


def parse_args() -> AblationConfig:
    parser = argparse.ArgumentParser(description="区域客户端消融实验脚本。")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-file", type=str, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--t-in", type=int, default=24)
    parser.add_argument("--t-out", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    config = AblationConfig(
        num_clients=args.num_clients,
        t_in=args.t_in,
        t_out=args.t_out,
        batch_size=args.batch_size,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        dataset_dir=args.dataset_dir.resolve(),
        dataset_file=str(args.dataset_file).strip(),
        output_dir=args.output_dir.resolve(),
        show_plot=bool(args.show_plot),
        verbose=bool(args.verbose),
    )
    validate_config(config)
    return config


def validate_config(config: AblationConfig) -> None:
    if config.num_clients <= 0:
        raise ValueError("num_clients 必须为正数。")
    if min(config.t_in, config.t_out, config.batch_size, config.rounds, config.local_epochs, config.hidden_dim) <= 0:
        raise ValueError("时间窗口、批量大小、轮数、局部 epoch、hidden_dim 必须均为正数。")
    if config.lr <= 0:
        raise ValueError("lr 必须为正数。")
    if config.stride <= 0:
        raise ValueError("stride 必须为正数。")
    if not str(config.dataset_file).strip():
        raise ValueError("dataset_file 不能为空。")
    ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1。")
    if min(config.train_ratio, config.val_ratio, config.test_ratio) <= 0:
        raise ValueError("train/val/test ratio 必须均大于 0。")


def load_data(config: AblationConfig) -> torch.Tensor:
    return load_preprocessed_dataset(config)


def split_clients(data_tensor: torch.Tensor, config: AblationConfig) -> list[np.ndarray]:
    return shared_split_clients(data_tensor, config)


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


def build_model(in_channels: int, model_kind: str, hidden_dim: int = 64) -> nn.Module:
    builders: dict[str, type[nn.Module]] = {
        "full": CNNLSTMAttention,
        "cnn_lstm": CNNLSTM,
        "lstm_attention": LSTMAttention,
        "cnn_attention": CNNAttention,
    }
    if model_kind not in builders:
        raise ValueError(f"不支持的消融模型类型: {model_kind}")
    return builders[model_kind](in_channels=in_channels, hidden_dim=hidden_dim)


def build_client_payloads(
    data_tensor: torch.Tensor,
    region_indices: list[np.ndarray],
    config: AblationConfig,
) -> list[dict[str, Any]]:
    payloads = []
    for cid, region_ids in enumerate(region_indices):
        dataset = RegionDataset(data_tensor, region_ids, config.t_in, config.t_out, config.stride)
        if len(dataset) == 0:
            continue
        train_idx, val_idx, test_idx = split_timewise_indices(dataset, config)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue
        payloads.append(
            {
                "cid": cid,
                "regions": region_ids,
                "train_loader": DataLoader(Subset(dataset, train_idx.tolist()), batch_size=config.batch_size, shuffle=True),
                "val_loader": DataLoader(Subset(dataset, val_idx.tolist()), batch_size=config.batch_size, shuffle=False),
                "test_loader": DataLoader(Subset(dataset, test_idx.tolist()), batch_size=config.batch_size, shuffle=False),
                "train_size": int(len(train_idx)),
            }
        )
    if not payloads:
        raise RuntimeError("没有可用于消融实验的客户端。")
    return payloads


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
    device: torch.device,
    config: AblationConfig,
    server_snapshot_weights: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], float]:
    model.load_state_dict(server_snapshot_weights)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.SmoothL1Loss()

    total_loss, steps = 0.0, 0
    for _ in range(config.local_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
    return copy.deepcopy(model.state_dict()), total_loss / max(steps, 1)


def federated_aggregation(local_weights: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    avg = copy.deepcopy(local_weights[0])
    for key in avg.keys():
        if avg[key].dtype in (torch.float16, torch.float32, torch.float64):
            avg[key].zero_()
            for state in local_weights:
                avg[key] += state[key]
            avg[key] /= float(len(local_weights))
        else:
            avg[key] = local_weights[0][key]
    return avg


def run_single_model(
    model_kind: str,
    display_name: str,
    in_channels: int,
    client_payloads: list[dict[str, Any]],
    device: torch.device,
    config: AblationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    server_model = build_model(in_channels=in_channels, model_kind=model_kind, hidden_dim=config.hidden_dim).to(device)
    server_state = copy.deepcopy(server_model.state_dict())
    history_rows = []

    for round_id in range(1, config.rounds + 1):
        local_states = []
        train_losses = []
        test_metrics = []
        for payload in client_payloads:
            local_model = build_model(in_channels=in_channels, model_kind=model_kind, hidden_dim=config.hidden_dim).to(device)
            local_state, train_loss = train_client(local_model, payload["train_loader"], device, config, server_state)
            local_states.append(local_state)
            train_losses.append(train_loss)
        server_state = federated_aggregation(local_states)
        server_model.load_state_dict(server_state)
        for payload in client_payloads:
            test_metrics.append(evaluate(server_model, payload["test_loader"], device))
        history_rows.append(
            {
                "Model": display_name,
                "Round": round_id,
                "TrainLoss_mean": float(np.mean(train_losses)),
                "TestRMSE_mean": float(np.mean([m["rmse"] for m in test_metrics])),
                "TestRMSE_std": float(np.std([m["rmse"] for m in test_metrics], ddof=0)),
            }
        )
        logging.info(
            "%s | round %s/%s | TrainLoss=%.6f | TestRMSE(mean)=%.6f",
            display_name,
            round_id,
            config.rounds,
            history_rows[-1]["TrainLoss_mean"],
            history_rows[-1]["TestRMSE_mean"],
        )

    client_rows = []
    for payload in client_payloads:
        metrics = evaluate(server_model, payload["test_loader"], device)
        client_rows.append(
            {
                "Model": display_name,
                "Client": f"Client {int(payload['cid'])}",
                "cid": int(payload["cid"]),
                "regions": int(len(payload["regions"])),
                **metrics,
            }
        )

    client_df = pd.DataFrame(client_rows).sort_values("cid").reset_index(drop=True)
    summary = {
        "mse_mean": float(client_df["mse"].mean()),
        "rmse_mean": float(client_df["rmse"].mean()),
        "rmse_std": float(client_df["rmse"].std(ddof=0)),
        "mae_mean": float(client_df["mae"].mean()),
        "mae_std": float(client_df["mae"].std(ddof=0)),
    }
    return pd.DataFrame(history_rows), client_df, summary


def save_results(
    config: AblationConfig,
    history_df: pd.DataFrame,
    client_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "ablation_history.csv"
    client_path = config.output_dir / "ablation_client_metrics.csv"
    summary_path = config.output_dir / "ablation_summary.csv"
    json_path = config.output_dir / "ablation_summary.json"
    figure_path = config.output_dir / "ccn_region_client_ablation.png"

    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    client_df.to_csv(client_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    json_path.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.ravel()

    for name in history_df["Model"].unique():
        sub = history_df.loc[history_df["Model"] == name].sort_values("Round")
        ax1.plot(sub["Round"], sub["TestRMSE_mean"], marker="o", linewidth=2, label=name)
    ax1.set_title("(a) Convergence of Test RMSE")
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Test RMSE")
    ax1.set_yscale("log")
    ax1.legend(frameon=True)

    plot_df = client_df.copy()
    model_order = summary_df.sort_values("rmse_mean")["Model"].tolist()
    positions = np.arange(len(model_order))
    for idx, model_name in enumerate(model_order):
        sub = plot_df.loc[plot_df["Model"] == model_name, "rmse"].to_numpy(dtype=float)
        ax2.scatter(np.full_like(sub, idx, dtype=float), sub, alpha=0.75, s=35)
    ax2.boxplot(
        [plot_df.loc[plot_df["Model"] == model_name, "rmse"].to_numpy(dtype=float) for model_name in model_order],
        positions=positions,
        widths=0.6,
    )
    ax2.set_xticks(positions)
    ax2.set_xticklabels(model_order, rotation=20, ha="right")
    ax2.set_title("(b) Client-level Stability")
    ax2.set_ylabel("RMSE")
    ax2.set_yscale("log")

    heat_df = client_df.pivot_table(index="Client", columns="Model", values="rmse", aggfunc="mean").reindex(columns=model_order)
    im = ax3.imshow(heat_df.to_numpy(dtype=float), aspect="auto")
    ax3.set_title("(c) RMSE Heatmap")
    ax3.set_xticks(np.arange(len(heat_df.columns)))
    ax3.set_xticklabels(heat_df.columns, rotation=20, ha="right")
    ax3.set_yticks(np.arange(len(heat_df.index)))
    ax3.set_yticklabels(heat_df.index)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    full_name = MODEL_SPECS["full"]
    full_rmse = float(summary_df.loc[summary_df["Model"] == full_name, "rmse_mean"].iloc[0])
    delta_df = summary_df.loc[summary_df["Model"] != full_name, ["Model", "rmse_mean"]].copy()
    delta_df["delta_rmse_pct"] = (delta_df["rmse_mean"] - full_rmse) / (full_rmse + 1e-12) * 100.0
    ax4.bar(delta_df["Model"], delta_df["delta_rmse_pct"])
    ax4.set_title("(d) Delta RMSE vs Full")
    ax4.set_ylabel("Delta RMSE (%)")
    ax4.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    if config.show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = get_device()
    data_tensor = load_data(config)
    region_indices = split_clients(data_tensor, config)
    client_payloads = build_client_payloads(data_tensor, region_indices, config)

    in_channels = int(data_tensor.shape[0])
    history_frames = []
    client_frames = []
    summary_rows = []
    for model_kind, display_name in MODEL_SPECS.items():
        history_df, client_df, summary = run_single_model(
            model_kind=model_kind,
            display_name=display_name,
            in_channels=in_channels,
            client_payloads=client_payloads,
            device=device,
            config=config,
        )
        history_frames.append(history_df)
        client_frames.append(client_df)
        summary_rows.append({"Model": display_name, **summary})

    full_history_df = pd.concat(history_frames, ignore_index=True)
    full_client_df = pd.concat(client_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("rmse_mean").reset_index(drop=True)

    save_results(config, full_history_df, full_client_df, summary_df)
    config_path = config.output_dir / "ablation_config.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_dir": str(config.dataset_dir),
                "output_dir": str(config.output_dir),
                "data_shape": list(map(int, data_tensor.shape)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
