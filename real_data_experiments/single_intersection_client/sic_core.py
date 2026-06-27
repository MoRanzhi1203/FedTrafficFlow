"""Single-intersection real-data federated experiment with standard FedAvg."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from real_data_experiments.common.data_splits import temporal_split_indices
from real_data_experiments.common.fedavg import fedavg_aggregate
from real_data_experiments.common.io_utils import read_node_flow_frame, resolve_path, select_top_nodes_by_activity
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, resolve_default_device, set_global_seed
from real_data_experiments.common.tensor_dataset import (
    GridTensorWindowDataset,
    build_time_split_bounds,
    get_region_usage_summary,
    load_grid_tensor_bundle,
    select_region_clients,
)
from real_data_experiments.single_intersection_client.sic_config import ExperimentConfig, build_arg_parser, config_from_args


class IntersectionDataset(Dataset):
    """Legacy parquet-direct sliding-window dataset for one node series."""

    def __init__(self, series: np.ndarray, sequence_length: int, prediction_horizon: int) -> None:
        self.series = np.asarray(series, dtype=np.float32).reshape(-1)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.window_count = len(self.series) - sequence_length - prediction_horizon + 1
        if self.window_count <= 0:
            raise ValueError("Series is too short for the requested sequence_length and prediction_horizon.")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.sequence_length
        target_index = end + self.prediction_horizon - 1
        features = self.series[start:end]
        target = self.series[target_index]
        return torch.tensor(features[None, :], dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


class Attention(nn.Module):
    """Simple temporal attention over LSTM outputs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(hidden_states).squeeze(-1), dim=1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)


class CNNLSTMAttentionRegressor(nn.Module):
    """CNN + LSTM + Attention regressor that supports 1 or more input channels."""

    def __init__(self, input_channels: int = 1, hidden_dim: int = 32, prediction_horizon: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.head = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        outputs, _ = self.lstm(encoded.transpose(1, 2))
        context = self.attention(outputs)
        return self.head(context)


@dataclass
class ClientData:
    """Per-client datasets and loaders."""

    client_id: int
    entity_id: int
    entity_kind: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split_metadata: dict[str, object]
    entity_metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class TargetScaler:
    """Train-split target normalization stats shared across experiment 1 clients."""

    mean: float
    std: float

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def denormalize_numpy(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}


class TargetNormalizedDataset(Dataset):
    """Wrap a dataset so that only the regression target is z-score normalized."""

    def __init__(self, base_dataset: Dataset, scaler: TargetScaler) -> None:
        self.base_dataset = base_dataset
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, target = self.base_dataset[index]
        return features, self.scaler.normalize_tensor(target.to(dtype=torch.float32))


def resolve_input_channels(config: ExperimentConfig) -> int:
    """Resolve the effective model input channel count for the chosen data mode."""
    return len(config.use_channels) if config.data_mode == "tensor" else 1


def build_single_intersection_matrix(config: ExperimentConfig, node_ids: list[int]) -> pd.DataFrame:
    """Load requested node records and pivot them into a time-indexed matrix."""
    frame = read_node_flow_frame(
        input_dir=config.input_path,
        target_col=config.target_column,
        node_ids=node_ids,
        max_chunks=config.max_chunks,
    )
    matrix = frame.pivot(index="时间段", columns="节点ID", values=config.target_column).sort_index().fillna(0.0)
    return matrix.reindex(columns=node_ids, fill_value=0.0)


def choose_node_ids(config: ExperimentConfig) -> list[int]:
    """Choose deterministic node ids for the legacy parquet fallback."""
    if config.selected_clients:
        return [int(node_id) for node_id in config.selected_clients[: config.num_clients]]
    bootstrap = read_node_flow_frame(
        input_dir=config.input_path,
        target_col=config.target_column,
        node_ids=None,
        max_chunks=1,
    )
    return select_top_nodes_by_activity(bootstrap, num_clients=config.num_clients, target_col=config.target_column)


def _make_entity_record(client: ClientData) -> dict[str, object]:
    """Build a shared entity record for metrics and prediction exports."""
    record: dict[str, object] = {
        "client_id": client.client_id,
        "entity_kind": client.entity_kind,
        "entity_id": client.entity_id,
        "train_samples": len(client.train_loader.dataset),
        "val_samples": len(client.val_loader.dataset),
        "test_samples": len(client.test_loader.dataset),
    }
    if client.entity_kind == "region":
        record["region_id"] = client.entity_id
    if client.entity_kind == "node":
        record["node_id"] = client.entity_id
    for key, value in client.entity_metadata.items():
        if key != "client_id":
            record[key] = value
    return record


def build_parquet_client_data(config: ExperimentConfig) -> tuple[list[ClientData], dict[str, object], pd.DataFrame | None]:
    """Construct legacy parquet-direct client datasets from node-level time series."""
    node_ids = choose_node_ids(config)
    matrix = build_single_intersection_matrix(config, node_ids)
    clients: list[ClientData] = []
    split_summary: dict[str, object] = {
        "data_mode": "parquet",
        "legacy_fallback": True,
        "split_strategy": "temporal_contiguous_over_window_index",
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": 1.0 - config.train_ratio - config.val_ratio,
        "sequence_length": config.sequence_length,
        "prediction_horizon": config.prediction_horizon,
        "num_time_steps": int(matrix.shape[0]),
        "selected_node_ids": [int(col) for col in matrix.columns.tolist()],
        "resolved_input_path": str(resolve_path(config.input_path)),
        "max_chunks": config.max_chunks,
        "clients": [],
    }

    for client_id, node_id in enumerate(matrix.columns.tolist()):
        dataset = IntersectionDataset(
            series=matrix[node_id].to_numpy(dtype=np.float32),
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
        )
        train_idx, val_idx, test_idx, metadata = temporal_split_indices(
            total_size=len(dataset),
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
        )
        client = ClientData(
            client_id=client_id,
            entity_id=int(node_id),
            entity_kind="node",
            train_loader=DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=False),
            val_loader=DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False),
            test_loader=DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=config.batch_size, shuffle=False),
            split_metadata=metadata,
            entity_metadata={
                "node_id": int(node_id),
                "mean_total_flow": float(matrix[node_id].mean()),
            },
        )
        clients.append(client)
        split_summary["clients"].append({"client_id": client_id, "node_id": int(node_id), **metadata})

    return clients, split_summary, None


def build_tensor_client_data(config: ExperimentConfig) -> tuple[list[ClientData], dict[str, object], pd.DataFrame]:
    """Construct tensor-only client datasets from the formal pooled-grid tensor."""
    bundle = load_grid_tensor_bundle(config.tensor_path, config.regions_path)
    selected_regions_df = select_region_clients(
        bundle=bundle,
        num_clients=config.num_clients,
        selected_region_ids=config.selected_clients,
        target_channel=config.target_channel,
        use_active_regions_only=config.use_active_regions_only,
    )
    time_bounds = build_time_split_bounds(
        time_count=int(bundle.tensor.shape[2]),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    region_usage = get_region_usage_summary(bundle.regions_df)
    split_summary: dict[str, object] = {
        "data_mode": "tensor",
        "legacy_fallback": False,
        "tensor_path": str(resolve_path(config.tensor_path)),
        "regions_path": str(resolve_path(config.regions_path)),
        "tensor_shape": list(bundle.tensor.shape),
        "sequence_length": config.sequence_length,
        "prediction_horizon": config.prediction_horizon,
        "use_channels": list(config.use_channels),
        "target_channel": int(config.target_channel),
        "use_active_regions_only": bool(config.use_active_regions_only),
        "total_region_count": region_usage["total_region_count"],
        "active_region_count": region_usage["active_region_count"],
        "used_region_count": int(len(selected_regions_df)),
        "selected_region_ids": [int(region_id) for region_id in selected_regions_df["region_id"].tolist()],
        "split_strategy": "temporal_contiguous_by_target_time",
        **time_bounds,
        "clients": [],
    }

    clients: list[ClientData] = []
    for row in selected_regions_df.to_dict(orient="records"):
        region_id = int(row["region_id"])
        train_dataset = GridTensorWindowDataset(
            tensor=bundle.tensor,
            region_id=region_id,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["train_start"]),
            end_time=int(time_bounds["train_end"]),
        )
        val_dataset = GridTensorWindowDataset(
            tensor=bundle.tensor,
            region_id=region_id,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["val_start"]),
            end_time=int(time_bounds["val_end"]),
        )
        test_dataset = GridTensorWindowDataset(
            tensor=bundle.tensor,
            region_id=region_id,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["test_start"]),
            end_time=int(time_bounds["test_end"]),
        )
        client = ClientData(
            client_id=int(row["client_id"]),
            entity_id=region_id,
            entity_kind="region",
            train_loader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False),
            val_loader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
            test_loader=DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
            split_metadata={
                "train": train_dataset.describe(),
                "val": val_dataset.describe(),
                "test": test_dataset.describe(),
            },
            entity_metadata=row,
        )
        clients.append(client)
        split_summary["clients"].append(
            {
                "client_id": int(row["client_id"]),
                "region_id": region_id,
                "pooled_row": int(row["pooled_row"]),
                "pooled_col": int(row["pooled_col"]),
                "source_node_count": int(row["source_node_count"]),
                "mean_total_flow": float(row["mean_total_flow"]),
                "train": train_dataset.describe(),
                "val": val_dataset.describe(),
                "test": test_dataset.describe(),
            }
        )

    return clients, split_summary, selected_regions_df


def build_client_data(config: ExperimentConfig) -> tuple[list[ClientData], dict[str, object], pd.DataFrame | None]:
    """Dispatch between tensor-only formal input and parquet legacy fallback."""
    if config.data_mode == "tensor":
        return build_tensor_client_data(config)
    if config.data_mode == "parquet":
        return build_parquet_client_data(config)
    raise ValueError(f"Unsupported data_mode: {config.data_mode}")


def _collect_target_array(dataset: Dataset) -> np.ndarray:
    """Materialize one dataset's scalar targets into a contiguous NumPy array."""
    targets = [float(dataset[index][1].reshape(-1)[0].item()) for index in range(len(dataset))]
    return np.asarray(targets, dtype=np.float64)


def fit_target_scaler(clients: list[ClientData], eps: float = 1e-6) -> TargetScaler:
    """Fit one global train-target scaler so FedAvg and Independent share the same output scale."""
    if not clients:
        raise ValueError("clients must not be empty when fitting target normalization.")
    train_targets = np.concatenate([_collect_target_array(client.train_loader.dataset) for client in clients])
    std = float(np.std(train_targets, ddof=0))
    return TargetScaler(mean=float(np.mean(train_targets)), std=max(std, float(eps)))


def apply_target_normalization(clients: list[ClientData], scaler: TargetScaler) -> None:
    """Replace train loaders with target-normalized variants while keeping eval loaders raw."""
    for client in clients:
        train_dataset = TargetNormalizedDataset(client.train_loader.dataset, scaler)
        client.train_loader = DataLoader(train_dataset, batch_size=client.train_loader.batch_size, shuffle=False)


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    learning_rate: float,
    local_epochs: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """Train a local model from its current parameters."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    epoch_losses: list[float] = []
    for _ in range(local_epochs):
        batch_losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
    return copy.deepcopy(model.state_dict()), float(np.mean(epoch_losses)) if epoch_losses else 0.0


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_scaler: TargetScaler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets from a data loader."""
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            outputs = model(features)
            if target_scaler is not None:
                outputs = target_scaler.denormalize_tensor(outputs)
            outputs = outputs.cpu().numpy().reshape(-1)
            preds.append(outputs)
            truths.append(targets.numpy().reshape(-1))
    if not preds:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.concatenate(truths).astype(np.float64), np.concatenate(preds).astype(np.float64)


def evaluate_client_model(
    model: nn.Module,
    client: ClientData,
    device: str,
    target_scaler: TargetScaler | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a model on one client's test split and return metrics plus prediction samples."""
    y_true, y_pred = collect_predictions(model, client.test_loader, device, target_scaler=target_scaler)
    metrics: dict[str, Any] = compute_regression_metrics(y_true, y_pred)
    metrics.update(_make_entity_record(client))
    prediction_df = pd.DataFrame(
        {
            **{key: value for key, value in _make_entity_record(client).items() if key not in {"train_samples", "val_samples", "test_samples"}},
            "sample_index": np.arange(len(y_true), dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return metrics, prediction_df


def evaluate_round(
    global_model: nn.Module,
    clients: list[ClientData],
    device: str,
    target_scaler: TargetScaler | None = None,
) -> dict[str, float]:
    """Evaluate the current global model on validation and test splits across clients."""
    val_rmses: list[float] = []
    val_maes: list[float] = []
    test_rmses: list[float] = []
    for client in clients:
        val_true, val_pred = collect_predictions(global_model, client.val_loader, device, target_scaler=target_scaler)
        test_true, test_pred = collect_predictions(global_model, client.test_loader, device, target_scaler=target_scaler)
        val_metrics = compute_regression_metrics(val_true, val_pred)
        test_metrics = compute_regression_metrics(test_true, test_pred)
        val_rmses.append(val_metrics["rmse"])
        val_maes.append(val_metrics["mae"])
        test_rmses.append(test_metrics["rmse"])
    return {
        "val_rmse": float(np.mean(val_rmses)),
        "val_rmse_std": float(np.std(val_rmses, ddof=0)),
        "val_mae": float(np.mean(val_maes)),
        "test_rmse": float(np.mean(test_rmses)),
        "test_rmse_std": float(np.std(test_rmses, ddof=0)),
    }


def run_fedavg_experiment(
    config: ExperimentConfig,
    clients: list[ClientData],
    device: str,
    target_scaler: TargetScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run standard FedAvg across single-intersection clients."""
    input_channels = resolve_input_channels(config)
    global_model = CNNLSTMAttentionRegressor(
        input_channels=input_channels,
        prediction_horizon=config.prediction_horizon,
    ).to(device)
    round_history: list[dict[str, float]] = []
    for round_idx in range(1, config.communication_rounds + 1):
        local_state_dicts: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        train_losses: list[float] = []
        for client in clients:
            local_model = CNNLSTMAttentionRegressor(
                input_channels=input_channels,
                prediction_horizon=config.prediction_horizon,
            ).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            state_dict, train_loss = train_local_model(
                local_model,
                client.train_loader,
                device=device,
                learning_rate=config.learning_rate,
                local_epochs=config.local_epochs,
            )
            local_state_dicts.append(state_dict)
            sample_counts.append(len(client.train_loader.dataset))
            train_losses.append(train_loss)

        global_model.load_state_dict(fedavg_aggregate(local_state_dicts, sample_counts))
        history_record = {"method": "FedAvg", "communication_round": round_idx, "train_loss": float(np.mean(train_losses))}
        history_record.update(evaluate_round(global_model, clients, device, target_scaler=target_scaler))
        round_history.append(history_record)

    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        metrics, prediction_df = evaluate_client_model(global_model, client, device, target_scaler=target_scaler)
        client_rows.append({"method": "FedAvg", **metrics})
        prediction_frames.append(prediction_df.assign(method="FedAvg"))
    return pd.DataFrame(client_rows), pd.DataFrame(round_history), pd.concat(prediction_frames, ignore_index=True)


def run_independent_experiment(
    config: ExperimentConfig,
    clients: list[ClientData],
    device: str,
    target_scaler: TargetScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the independent baseline with the same data splits."""
    total_epochs = config.independent_total_epochs or (config.communication_rounds * config.local_epochs)
    input_channels = resolve_input_channels(config)
    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        model = CNNLSTMAttentionRegressor(
            input_channels=input_channels,
            prediction_horizon=config.prediction_horizon,
        ).to(device)
        train_local_model(
            model,
            client.train_loader,
            device=device,
            learning_rate=config.learning_rate,
            local_epochs=total_epochs,
        )
        metrics, prediction_df = evaluate_client_model(model, client, device, target_scaler=target_scaler)
        client_rows.append({"method": "Independent", **metrics})
        prediction_frames.append(prediction_df.assign(method="Independent"))
    return pd.DataFrame(client_rows), pd.concat(prediction_frames, ignore_index=True)


def limit_prediction_samples(prediction_df: pd.DataFrame, total_limit: int) -> pd.DataFrame:
    """Keep a balanced prediction sample subset so both methods remain visible in exports."""
    if total_limit <= 0 or prediction_df.empty or "method" not in prediction_df.columns:
        return prediction_df.copy()
    method_groups = list(prediction_df.groupby("method", sort=False))
    base_quota = max(total_limit // len(method_groups), 1)
    remainder = max(total_limit - base_quota * len(method_groups), 0)
    sampled_frames: list[pd.DataFrame] = []
    for index, (_, group_df) in enumerate(method_groups):
        quota = base_quota + (1 if index < remainder else 0)
        sampled_frames.append(group_df.head(quota))
    return pd.concat(sampled_frames, ignore_index=True)


def export_results(
    config: ExperimentConfig,
    output_dir: Path,
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    selected_regions_df: pd.DataFrame | None,
    fed_client_df: pd.DataFrame,
    ind_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    target_scaler: TargetScaler | None = None,
) -> None:
    """Write experiment artifacts to disk."""
    client_metrics_df = pd.concat([fed_client_df, ind_client_df], ignore_index=True)
    main_metrics_df = (
        client_metrics_df.groupby("method", as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values("method")
        .reset_index(drop=True)
    )
    main_summary_df = summarize_metric_frame(client_metrics_df, group_cols=["method"])

    write_json(config.to_dict(), output_dir / "run_config.json")
    write_text(
        "python -m real_data_experiments.single_intersection_client.sic_core " + " ".join(sys.argv[1:]),
        output_dir / "run_commands.txt",
    )
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    if selected_regions_df is not None and "region_id" in selected_regions_df.columns:
        write_csv(selected_regions_df, output_dir / "selected_regions.csv")
    write_csv(main_metrics_df, output_dir / "main_metrics.csv")
    write_json(main_metrics_df.to_dict(orient="records"), output_dir / "main_metrics.json")
    write_csv(main_summary_df, output_dir / "main_summary.csv")
    write_json(main_summary_df.to_dict(orient="records"), output_dir / "main_summary.json")
    write_csv(client_metrics_df, output_dir / "client_metrics.csv")
    write_json(client_metrics_df.to_dict(orient="records"), output_dir / "client_metrics.json")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_json(convergence_df.to_dict(orient="records"), output_dir / "convergence_history.json")
    sampled_prediction_df = limit_prediction_samples(prediction_df, config.prediction_sample_limit)
    write_csv(sampled_prediction_df, output_dir / "prediction_samples.csv")
    write_json(sampled_prediction_df.to_dict(orient="records"), output_dir / "prediction_samples.json")
    if target_scaler is not None:
        write_json(target_scaler.to_dict(), output_dir / "target_scaler.json")
    note_lines = [
        "# 单路口客户端实验说明",
        "",
        "- 当前主线方法始终为标准样本量加权 FedAvg。",
        "- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。",
        "- 当前客户端表示 pooled-grid-region client，每个客户端对应一个 active pooled region。",
        "- 数据划分按 target time 的时间顺序执行，不使用随机切分。",
        "- 训练阶段对目标值使用 train-split 统计量做 z-score normalization，评估与导出预测时反归一化回原始尺度。",
        "- `parquet-direct` 仅保留为 legacy fallback，不作为正式默认结果入口。",
    ]
    if config.data_mode == "parquet":
        note_lines.append("- 本次运行使用了 legacy parquet fallback，仅用于兼容旧 smoke test。")
    write_text("\n".join(note_lines), output_dir / "experiment_notes_zh.md")


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run the single-intersection experiment and export reproducible outputs."""
    output_dir = prepare_output_dir(config.output_dir)
    device = resolve_default_device(config.device)
    set_global_seed(config.seed)
    start_time = datetime.now().isoformat(timespec="seconds")

    clients, split_summary, selected_regions_df = build_client_data(config)
    target_scaler: TargetScaler | None = None
    if config.target_normalization:
        target_scaler = fit_target_scaler(clients, eps=config.target_normalization_eps)
        apply_target_normalization(clients, target_scaler)
        split_summary["target_normalization"] = {
            "enabled": True,
            "mean": target_scaler.mean,
            "std": target_scaler.std,
        }
    else:
        split_summary["target_normalization"] = {"enabled": False}
    fed_client_df, convergence_df, fed_prediction_df = run_fedavg_experiment(
        config,
        clients,
        device,
        target_scaler=target_scaler,
    )
    ind_client_df, ind_prediction_df = run_independent_experiment(
        config,
        clients,
        device,
        target_scaler=target_scaler,
    )
    prediction_df = pd.concat([fed_prediction_df, ind_prediction_df], ignore_index=True)

    environment_summary = build_environment_summary(device)
    environment_summary["seed"] = config.seed
    environment_summary["start_time"] = start_time
    environment_summary["end_time"] = datetime.now().isoformat(timespec="seconds")
    environment_summary["data_mode"] = config.data_mode

    export_results(
        config=config,
        output_dir=output_dir,
        environment_summary=environment_summary,
        split_summary=split_summary,
        selected_regions_df=selected_regions_df,
        fed_client_df=fed_client_df,
        ind_client_df=ind_client_df,
        convergence_df=convergence_df,
        prediction_df=prediction_df,
        target_scaler=target_scaler,
    )
    selected_ids = []
    if selected_regions_df is not None:
        selected_ids = selected_regions_df["region_id"].tolist() if "region_id" in selected_regions_df.columns else []
    elif "selected_node_ids" in split_summary:
        selected_ids = split_summary["selected_node_ids"]
    return {
        "output_dir": str(output_dir),
        "selected_ids": selected_ids,
        "client_metrics_rows": int(len(fed_client_df) + len(ind_client_df)),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[single_intersection_client] completed -> {result['output_dir']}")
    print(f"[selected_ids] {result['selected_ids']}")


if __name__ == "__main__":
    main()
