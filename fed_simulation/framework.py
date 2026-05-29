from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from .config import ExperimentConfig
from .datasets import ComplexHeterogeneousDataset

try:
    import psutil
except ImportError:
    psutil = None


MODEL_COLORS = {
    "full": "#2E86AB",
    "no_attention": "#A23B72",
    "lstm_only": "#F18F01",
    "spatial_only": "#C73E1D",
    "weak": "#6A4C93",
}
AGG_COLORS = {
    "fedavg": "#4C9F70",
    "loss_weighted": "#E76F51",
    "data_loss_weighted": "#457B9D",
    "trimmed_loss_weighted": "#6D597A",
}
LOGGER = logging.getLogger("fed_simulation")
ModelBuilder = Callable[[int, int, int, int], nn.Module]


def build_common_parser(description: str, default_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--client-dropout-rate", type=float, default=0.0)
    parser.add_argument("--dp-noise-std", type=float, default=0.0)
    parser.add_argument("--show-plot", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-pin-memory", action="store_true", default=False)
    parser.add_argument("--disable-persistent-workers", action="store_true", default=False)
    parser.add_argument("--disable-amp", action="store_true", default=False)
    parser.add_argument("--disable-compile", action="store_true", default=False)
    parser.add_argument("--robust-loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--agg-lambda", type=float, default=0.5)
    parser.add_argument("--server-momentum", type=float, default=0.1)
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--save-round-snapshots", action="store_true", default=False)
    return parser


def setup_logging(output_dir: str, log_level: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    LOGGER.propagate = False
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(output_dir) / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_system_info() -> dict:
    info = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if psutil is not None:
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["system_memory_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return info


def build_loss_fn(config: ExperimentConfig) -> nn.Module:
    if config.robust_loss == "huber":
        return nn.HuberLoss(delta=config.huber_delta)
    return nn.MSELoss()


def get_model_size_bytes(model: nn.Module) -> int:
    return sum(param.numel() * param.element_size() for param in model.parameters())


def create_dataloader(dataset, batch_size: int, shuffle: bool, seed: int, config: ExperimentConfig) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": torch.Generator().manual_seed(seed),
        "num_workers": max(config.num_workers, 0),
        "pin_memory": bool(config.pin_memory and torch.cuda.is_available()),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = bool(config.persistent_workers)
        kwargs["prefetch_factor"] = max(2, config.prefetch_factor)
    return DataLoader(**kwargs)


def _autocast_context(device: torch.device, enable_amp: bool):
    if enable_amp and device.type == "cuda" and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _resource_snapshot(device: torch.device) -> dict:
    snapshot = {"cpu_mem_gb": 0.0, "gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0}
    if psutil is not None:
        snapshot["cpu_mem_gb"] = round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), 4)
    if device.type == "cuda":
        snapshot["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 4)
        snapshot["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 4)
    return snapshot


class FedClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device).float()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.9)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=(config.enable_amp and device.type == "cuda"))
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.data_size = len(train_loader.dataset)

    def maybe_compile(self) -> None:
        if not self.config.enable_compile or not hasattr(torch, "compile"):
            return
        try:
            self.model = torch.compile(self.model)
        except Exception as exc:
            LOGGER.warning("Client %s 启用 torch.compile 失败，回退 eager 模式: %s", self.client_id, exc)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss, total_samples = 0.0, 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            if self.config.max_train_batches and batch_idx >= self.config.max_train_batches:
                break
            x = x.to(self.device, non_blocking=True).float()
            y = y.to(self.device, non_blocking=True).float().squeeze()
            self.optimizer.zero_grad(set_to_none=True)
            with _autocast_context(self.device, self.config.enable_amp):
                pred, _ = self.model(x)
                loss = self.criterion(pred.squeeze(), y)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        avg_loss = total_loss / max(1, total_samples)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self) -> float:
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                if self.config.max_eval_batches and batch_idx >= self.config.max_eval_batches:
                    break
                x = x.to(self.device, non_blocking=True).float()
                y = y.to(self.device, non_blocking=True).float().squeeze()
                with _autocast_context(self.device, self.config.enable_amp):
                    pred, _ = self.model(x)
                    loss = self.criterion(pred.squeeze(), y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        avg_loss = total_loss / max(1, total_samples)
        self.val_losses.append(avg_loss)
        self.scheduler.step()
        return avg_loss

    def train(self, epochs: int, global_model: nn.Module | None) -> tuple[float, dict[str, torch.Tensor]]:
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        for _ in range(epochs):
            self.train_epoch()
            self.validate()
        final_loss = self.train_losses[-1] if self.train_losses else float("inf")
        return final_loss, {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}

    def test_metrics(self) -> dict[str, float]:
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                if self.config.max_eval_batches and batch_idx >= self.config.max_eval_batches:
                    break
                x = x.to(self.device, non_blocking=True).float()
                y = y.to(self.device, non_blocking=True).float().squeeze()
                pred, _ = self.model(x)
                preds.append(pred.squeeze().cpu().numpy())
                truths.append(y.cpu().numpy())
        preds_arr = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
        truths_arr = np.concatenate(truths) if truths else np.array([], dtype=np.float32)
        diff = preds_arr - truths_arr
        denom = np.maximum(np.abs(truths_arr), 1e-6)
        return {
            "mse": float(np.mean(diff ** 2)) if diff.size else 0.0,
            "rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
            "mae": float(np.mean(np.abs(diff))) if diff.size else 0.0,
            "mape": float(np.mean(np.abs(diff) / denom)) if diff.size else 0.0,
        }


class Server:
    def __init__(self, model: nn.Module, agg_method: str, config: ExperimentConfig, device: torch.device) -> None:
        self.global_model = model.to(device).float()
        self.agg_method = agg_method
        self.config = config
        self.device = device
        self.round_losses: list[float] = []
        self.agg_weights_history: list[dict[int, float]] = []
        self.client_data_sizes: np.ndarray | None = None

    def set_client_data_sizes(self, sizes: list[int]) -> None:
        self.client_data_sizes = np.array(sizes, dtype=float)

    def _weights(self, losses: np.ndarray, active_ids: list[int]) -> np.ndarray:
        if self.client_data_sizes is None:
            data_weights = np.ones(len(active_ids), dtype=float) / max(1, len(active_ids))
        else:
            active_sizes = np.array([self.client_data_sizes[idx] for idx in active_ids], dtype=float)
            data_weights = active_sizes / (active_sizes.sum() + 1e-12)
        loss_weights = np.exp(-losses * 2.0)
        loss_weights = loss_weights / (loss_weights.sum() + 1e-12)

        if self.agg_method == "fedavg":
            weights = data_weights
        elif self.agg_method == "loss_weighted":
            weights = loss_weights
        elif self.agg_method == "trimmed_loss_weighted":
            keep_mask = np.ones(len(losses), dtype=bool)
            trim_count = min(int(len(losses) * self.config.trim_ratio), max(0, len(losses) - 1))
            if trim_count > 0:
                worst_indices = np.argsort(losses)[-trim_count:]
                keep_mask[worst_indices] = False
            trimmed = np.where(keep_mask, loss_weights, 0.0)
            weights = trimmed / (trimmed.sum() + 1e-12)
        else:
            weights = self.config.agg_lambda * data_weights + (1.0 - self.config.agg_lambda) * loss_weights
            weights = weights / (weights.sum() + 1e-12)
        return weights

    def aggregate(self, client_weights: list[dict[str, torch.Tensor] | None], client_losses: list[float], active_ids: list[int]) -> None:
        state_dict = self.global_model.state_dict()
        losses = np.array([client_losses[idx] for idx in active_ids], dtype=float)
        weights = self._weights(losses, active_ids)
        self.agg_weights_history.append({int(active_ids[idx]): float(weights[idx]) for idx in range(len(active_ids))})

        merged = {}
        for name, tensor in state_dict.items():
            if tensor.dtype in (torch.float16, torch.float32, torch.float64):
                merged[name] = torch.zeros_like(tensor, dtype=torch.float32)
            else:
                merged[name] = tensor.clone()

        for name, tensor in state_dict.items():
            if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
                continue
            for idx, client_id in enumerate(active_ids):
                client_state = client_weights[client_id]
                assert client_state is not None
                merged[name] += client_state[name].to(self.device, dtype=torch.float32) * float(weights[idx])

        momentum = self.config.server_momentum
        for name, tensor in state_dict.items():
            if tensor.dtype in (torch.float16, torch.float32, torch.float64):
                merged[name] = ((1.0 - momentum) * tensor.to(torch.float32) + momentum * merged[name]).to(tensor.dtype)

        self.global_model.load_state_dict(merged)
        self.round_losses.append(float(losses.mean()) if losses.size else 0.0)


def _build_clients(
    seed: int,
    config: ExperimentConfig,
    model_type: str,
    model_builders: dict[str, ModelBuilder],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[list[FedClient], list[int]]:
    sample_sizes = [50, 80, 120][: config.num_clients]
    if len(sample_sizes) < config.num_clients:
        sample_sizes = [50 + 20 * i for i in range(config.num_clients)]

    split_generator = torch.Generator().manual_seed(seed)
    clients = []
    for client_id in range(config.num_clients):
        dataset = ComplexHeterogeneousDataset(
            client_id=client_id,
            num_samples=sample_sizes[client_id],
            K=config.K,
            T=config.T,
            noise=config.noise,
            dist_type=config.dist_types[client_id % len(config.dist_types)],
            pattern_type=config.pattern_types[client_id % len(config.pattern_types)],
            missing_rate=config.missing_rates[client_id % len(config.missing_rates)],
            outlier_rate=config.outlier_rates[client_id % len(config.outlier_rates)],
            seed=seed,
        )
        train_size = int(len(dataset) * config.train_split)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size], generator=split_generator)
        train_loader = create_dataloader(train_data, config.batch_size, True, seed + client_id, config)
        test_loader = create_dataloader(test_data, config.batch_size, False, seed + 1000 + client_id, config)
        model = model_builders[model_type](config.K, config.T, config.hidden_dim, config.num_heads)
        client = FedClient(client_id, model, train_loader, test_loader, criterion, config, device)
        if config.enable_compile:
            client.maybe_compile()
        clients.append(client)
    return clients, sample_sizes


def run_single_experiment(
    seed: int,
    config: ExperimentConfig,
    model_type: str,
    agg_method: str,
    model_builders: dict[str, ModelBuilder],
) -> tuple[dict, dict, dict]:
    set_seed(seed)
    device = get_device()
    criterion = build_loss_fn(config)
    clients, sample_sizes = _build_clients(seed, config, model_type, model_builders, criterion, device)

    server_model = model_builders[model_type](config.K, config.T, config.hidden_dim, config.num_heads)
    server = Server(server_model, agg_method, config, device)
    server.set_client_data_sizes(sample_sizes)
    model_size_bytes = get_model_size_bytes(server_model)

    total_comm_bytes = 0.0
    client_losses_hist = {cid: [] for cid in range(config.num_clients)}
    run_log = {
        "seed": seed,
        "model_type": model_type,
        "agg_method": agg_method,
        "system_info": collect_system_info(),
        "rounds": [],
    }

    for round_idx in range(config.num_rounds):
        round_start = time.perf_counter()
        active_ids = [cid for cid in range(config.num_clients) if random.random() >= config.client_dropout_rate]
        if not active_ids:
            active_ids = [0]

        client_weights: list[dict[str, torch.Tensor] | None] = [None] * config.num_clients
        client_losses = [0.0] * config.num_clients
        local_times = []

        for client_id in active_ids:
            local_start = time.perf_counter()
            loss, weights = clients[client_id].train(config.local_epochs, server.global_model)
            if config.dp_noise_std > 0:
                with torch.no_grad():
                    for parameter in clients[client_id].model.parameters():
                        parameter.add_(torch.randn_like(parameter) * config.dp_noise_std)
            client_weights[client_id] = weights
            client_losses[client_id] = loss
            client_losses_hist[client_id].append(loss)
            local_times.append(time.perf_counter() - local_start)

        server.aggregate(client_weights, client_losses, active_ids)

        per_client_metrics = []
        for client_id in range(config.num_clients):
            clients[client_id].model.load_state_dict(server.global_model.state_dict())
            per_client_metrics.append(clients[client_id].test_metrics())

        round_comm = 2.0 * len(active_ids) * model_size_bytes
        total_comm_bytes += round_comm
        round_time = time.perf_counter() - round_start
        resource = _resource_snapshot(device)
        run_log["rounds"].append(
            {
                "round": round_idx + 1,
                "active_clients": active_ids,
                "n_active": len(active_ids),
                "client_losses": {str(idx): float(client_losses[idx]) for idx in range(config.num_clients)},
                "per_client_metrics": per_client_metrics,
                "round_comm_bytes": round_comm,
                "avg_loss": float(server.round_losses[-1]) if server.round_losses else 0.0,
                "round_time_sec": round_time,
                "mean_local_train_time_sec": float(np.mean(local_times)) if local_times else 0.0,
                **resource,
            }
        )

    final_metrics = []
    for client_id in range(config.num_clients):
        clients[client_id].model.load_state_dict(server.global_model.state_dict())
        final_metrics.append(clients[client_id].test_metrics())

    final_df = pd.DataFrame(final_metrics)
    round_df = pd.DataFrame(run_log["rounds"])
    summary = {
        "seed": seed,
        "model_type": model_type,
        "agg_method": agg_method,
        "mse_mean": float(final_df["mse"].mean()),
        "mse_std": float(final_df["mse"].std(ddof=0)),
        "rmse_mean": float(final_df["rmse"].mean()),
        "rmse_std": float(final_df["rmse"].std(ddof=0)),
        "mae_mean": float(final_df["mae"].mean()),
        "mae_std": float(final_df["mae"].std(ddof=0)),
        "mape_mean": float(final_df["mape"].mean()),
        "mape_std": float(final_df["mape"].std(ddof=0)),
        "total_comm_bytes": total_comm_bytes,
        "avg_round_time_sec": float(round_df["round_time_sec"].mean()) if not round_df.empty else 0.0,
        "peak_cpu_mem_gb": float(round_df["cpu_mem_gb"].max()) if not round_df.empty else 0.0,
        "peak_gpu_allocated_mb": float(round_df["gpu_allocated_mb"].max()) if not round_df.empty else 0.0,
    }
    for client_id, metric in enumerate(final_metrics):
        summary[f"client_{client_id}_rmse"] = float(metric["rmse"])
        summary[f"client_{client_id}_mae"] = float(metric["mae"])
        summary[f"client_{client_id}_mse"] = float(metric["mse"])

    loss_record = {
        "client_losses": {str(cid): [float(value) for value in values] for cid, values in client_losses_hist.items()},
        "round_losses": [float(value) for value in server.round_losses],
        "agg_weights": server.agg_weights_history,
        "model_type": model_type,
        "agg_method": agg_method,
        "seed": seed,
    }
    return summary, run_log, loss_record


def _style_ax(ax, title=None, xlabel=None, ylabel=None, grid=True) -> None:
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, path: Path, show_plot: bool) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def generate_figures(df_summary: pd.DataFrame, losses_records: list[dict], model_labels: dict[str, str], output_dir: Path, show_plot: bool) -> None:
    if df_summary.empty:
        return
    grouped = df_summary.groupby(["model_type", "agg_method"])[["rmse_mean", "avg_round_time_sec"]].mean().reset_index()
    best_rows = grouped.sort_values("rmse_mean").groupby("model_type", as_index=False).first()
    order = [name for name in ["full", "no_attention", "lstm_only", "spatial_only", "weak"] if name in best_rows["model_type"].tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    (ax0, ax1), (ax2, ax3) = axes

    ax0.bar(order, [float(best_rows.loc[best_rows["model_type"] == name, "rmse_mean"].iloc[0]) for name in order], color=[MODEL_COLORS.get(name, "#888888") for name in order])
    _style_ax(ax0, "最优聚合下的 RMSE", ylabel="RMSE")
    ax0.set_xticks(range(len(order)))
    ax0.set_xticklabels([model_labels.get(name, name) for name in order], rotation=15, ha="right")

    ax1.bar(order, [float(best_rows.loc[best_rows["model_type"] == name, "avg_round_time_sec"].iloc[0]) for name in order], color=[MODEL_COLORS.get(name, "#888888") for name in order])
    _style_ax(ax1, "最优聚合下的轮次耗时", ylabel="Seconds")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([model_labels.get(name, name) for name in order], rotation=15, ha="right")

    rmse_pivot = grouped.pivot_table(values="rmse_mean", index="model_type", columns="agg_method", aggfunc="mean")
    rmse_pivot = rmse_pivot.reindex(index=order)
    rmse_pivot.index = [model_labels.get(name, name) for name in rmse_pivot.index]
    sns.heatmap(rmse_pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", linewidths=0.5, ax=ax2)
    _style_ax(ax2, "模型-聚合 RMSE 热力图", grid=False)

    round_rows = []
    for record in losses_records:
        for idx, loss in enumerate(record.get("round_losses", []), start=1):
            round_rows.append({"round": idx, "loss": loss, "model_type": record["model_type"], "agg_method": record["agg_method"]})
    if round_rows:
        round_df = pd.DataFrame(round_rows)
        full_df = round_df[round_df["model_type"] == "full"]
        for agg_method, agg_df in full_df.groupby("agg_method"):
            ax3.plot(agg_df["round"], agg_df["loss"], marker="o", linewidth=2, color=AGG_COLORS.get(agg_method, "#888888"), label=agg_method)
        ax3.legend(frameon=True)
    _style_ax(ax3, "Full 模型收敛曲线", xlabel="Round", ylabel="Loss")

    plt.tight_layout()
    _save_figure(fig, output_dir / "optimization_dashboard.png", show_plot)


def write_reports(output_dir: Path, config: ExperimentConfig, summary_mean: pd.DataFrame, system_info: dict, model_name: str) -> None:
    best_row = summary_mean.sort_values("rmse_mean").iloc[0].to_dict() if not summary_mean.empty else {}
    report = [
        "# 联邦仿真实验优化效果报告",
        "",
        "## 1. 现状诊断",
        "",
        "- 当前 CNN 与 GCN 增强脚本中存在大量重复代码，模型、训练、聚合、绘图与结果保存强耦合。",
        "- 旧实现缺少资源监控、轮次耗时、部署依赖固化和异常日志，不利于定位性能瓶颈。",
        "- 本地环境缺少 `torch` 时无法直接复现实验，说明部署链路需要标准化。",
        "",
        "## 2. 本轮优化内容",
        "",
        "- 抽取 `fed_simulation` 共享框架，统一联邦训练、聚合、结果输出和图表生成。",
        "- 增加稳健损失、AMP、`torch.compile`、DataLoader 并行参数与裁剪聚合策略。",
        "- 增加轮次耗时、CPU/GPU 内存、通信量等可量化指标。",
        "- 将 `CNN` / `GCN` 主脚本重构为薄入口，降低后续维护成本。",
        "",
        "## 3. 当前最优结果",
        "",
        f"- 模型族: `{model_name}`",
        f"- 最优模型: `{best_row.get('model_type', 'N/A')}`",
        f"- 最优聚合: `{best_row.get('agg_method', 'N/A')}`",
        f"- RMSE: `{best_row.get('rmse_mean', 'N/A')}`",
        f"- MAE: `{best_row.get('mae_mean', 'N/A')}`",
        f"- 平均轮次耗时: `{best_row.get('avg_round_time_sec', 'N/A')}` 秒",
        "",
        "## 4. 环境信息",
        "",
        f"- Torch: `{system_info.get('torch_version', 'unknown')}`",
        f"- CUDA 可用: `{system_info.get('cuda_available', False)}`",
        f"- 客户端数: `{config.num_clients}`",
        f"- 联邦轮数: `{config.num_rounds}`",
        f"- 本地训练轮数: `{config.local_epochs}`",
        "",
    ]
    (output_dir / "optimization_report.md").write_text("\n".join(report), encoding="utf-8")


def run_experiment_suite(config: ExperimentConfig, model_name: str, model_builders: dict[str, ModelBuilder], model_labels: dict[str, str]) -> None:
    config.validate()
    config.ensure_seeds()
    setup_logging(config.output_dir, config.log_level)
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    system_info = collect_system_info()
    (output_dir / "hyperparameters.json").write_text(json.dumps(config.to_serializable_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "system_info.json").write_text(json.dumps(system_info, indent=2, ensure_ascii=False), encoding="utf-8")

    summaries = []
    run_logs = []
    losses_records = []
    total = len(config.model_types) * len(config.agg_methods) * len(config.seeds)
    current = 0

    for model_type in config.model_types:
        for agg_method in config.agg_methods:
            for seed in config.seeds:
                current += 1
                LOGGER.info("实验 %s/%s | model=%s | agg=%s | seed=%s", current, total, model_type, agg_method, seed)
                summary, run_log, loss_record = run_single_experiment(seed, config, model_type, agg_method, model_builders)
                summaries.append(summary)
                run_logs.append(run_log)
                losses_records.append(loss_record)

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(output_dir / "metrics_per_seed.csv", index=False)
    summary_mean = (
        df_summary.groupby(["model_type", "agg_method"])
        .agg(
            mse_mean=("mse_mean", "mean"),
            mse_std=("mse_mean", "std"),
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            mape_mean=("mape_mean", "mean"),
            mape_std=("mape_mean", "std"),
            avg_round_time_sec=("avg_round_time_sec", "mean"),
            peak_cpu_mem_gb=("peak_cpu_mem_gb", "max"),
            peak_gpu_allocated_mb=("peak_gpu_allocated_mb", "max"),
            total_comm_mb=("total_comm_bytes", lambda values: values.mean() / (1024 * 1024)),
        )
        .reset_index()
    )
    summary_mean.to_csv(output_dir / "summary_mean_std.csv", index=False)
    (output_dir / "run_logs.json").write_text(json.dumps(run_logs, indent=2, ensure_ascii=False), encoding="utf-8")

    generate_figures(df_summary, losses_records, model_labels, output_dir, config.show_plot)
    write_reports(output_dir, config, summary_mean, system_info, model_name)
