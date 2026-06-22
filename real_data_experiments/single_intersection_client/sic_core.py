"""Single-intersection real-data federated experiment with standard FedAvg."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from real_data_experiments.common.data_splits import temporal_split_indices
from real_data_experiments.common.fedavg import fedavg_aggregate
from real_data_experiments.common.io_utils import read_node_flow_frame, resolve_path, select_top_nodes_by_activity
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, resolve_default_device, set_global_seed
from real_data_experiments.single_intersection_client.sic_config import ExperimentConfig, build_arg_parser, config_from_args


class IntersectionDataset(Dataset):
    """Sliding-window dataset for a single node client."""

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
    """Notebook-aligned CNN + LSTM + Attention regressor."""

    def __init__(self, hidden_dim: int = 32, prediction_horizon: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.head = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        lstm_in = encoded.transpose(1, 2)
        outputs, _ = self.lstm(lstm_in)
        context = self.attention(outputs)
        return self.head(context)


@dataclass
class ClientData:
    """Per-client datasets and loaders."""

    client_id: int
    node_id: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split_metadata: dict[str, object]


def build_single_intersection_matrix(config: ExperimentConfig, node_ids: list[int]) -> pd.DataFrame:
    """Load requested node records and pivot them into a time-indexed matrix."""
    frame = read_node_flow_frame(
        input_dir=config.input_path,
        target_col=config.target_column,
        node_ids=node_ids,
        max_chunks=config.max_chunks,
    )
    matrix = (
        frame.pivot(index="时间段", columns="节点ID", values=config.target_column)
        .sort_index()
        .fillna(0.0)
    )
    matrix = matrix.reindex(columns=node_ids, fill_value=0.0)
    return matrix


def choose_node_ids(config: ExperimentConfig) -> list[int]:
    """Choose deterministic node ids for the experiment."""
    if config.selected_clients:
        return [int(node_id) for node_id in config.selected_clients[: config.num_clients]]
    bootstrap = read_node_flow_frame(
        input_dir=config.input_path,
        target_col=config.target_column,
        node_ids=None,
        max_chunks=1,
    )
    return select_top_nodes_by_activity(bootstrap, num_clients=config.num_clients, target_col=config.target_column)


def build_client_data(config: ExperimentConfig, matrix: pd.DataFrame, device: str) -> tuple[list[ClientData], dict[str, object]]:
    """Construct client datasets and split metadata from the time-indexed matrix."""
    clients: list[ClientData] = []
    split_summary: dict[str, object] = {
        "split_strategy": "temporal_contiguous",
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": 1.0 - config.train_ratio - config.val_ratio,
        "sequence_length": config.sequence_length,
        "prediction_horizon": config.prediction_horizon,
        "num_time_steps": int(matrix.shape[0]),
        "selected_node_ids": [int(col) for col in matrix.columns.tolist()],
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
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)
        clients.append(
            ClientData(
                client_id=client_id,
                node_id=int(node_id),
                train_loader=DataLoader(train_subset, batch_size=config.batch_size, shuffle=False),
                val_loader=DataLoader(val_subset, batch_size=config.batch_size, shuffle=False),
                test_loader=DataLoader(test_subset, batch_size=config.batch_size, shuffle=False),
                split_metadata=metadata,
            )
        )
        split_summary["clients"].append(
            {
                "client_id": client_id,
                "node_id": int(node_id),
                **metadata,
            }
        )
    return clients, split_summary


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


def collect_predictions(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets from a data loader."""
    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            outputs = model(features).cpu().numpy().reshape(-1)
            preds.append(outputs)
            truths.append(targets.numpy().reshape(-1))
    if not preds:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.concatenate(truths).astype(np.float64), np.concatenate(preds).astype(np.float64)


def evaluate_client_model(model: nn.Module, client: ClientData, device: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a model on one client's test split and return metrics plus prediction samples."""
    y_true, y_pred = collect_predictions(model, client.test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics.update(
        {
            "client_id": client.client_id,
            "node_id": client.node_id,
            "train_samples": len(client.train_loader.dataset),
            "val_samples": len(client.val_loader.dataset),
            "test_samples": len(client.test_loader.dataset),
        }
    )
    prediction_df = pd.DataFrame(
        {
            "client_id": client.client_id,
            "node_id": client.node_id,
            "sample_index": np.arange(len(y_true), dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return metrics, prediction_df


def evaluate_round(global_model: nn.Module, clients: list[ClientData], device: str) -> dict[str, float]:
    """Evaluate the current global model on validation and test splits across clients."""
    val_rmses: list[float] = []
    val_maes: list[float] = []
    test_rmses: list[float] = []
    for client in clients:
        val_true, val_pred = collect_predictions(global_model, client.val_loader, device)
        test_true, test_pred = collect_predictions(global_model, client.test_loader, device)
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


def run_fedavg_experiment(config: ExperimentConfig, clients: list[ClientData], device: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run standard FedAvg across single-intersection clients."""
    global_model = CNNLSTMAttentionRegressor(prediction_horizon=config.prediction_horizon).to(device)
    round_history: list[dict[str, float]] = []
    for round_idx in range(1, config.communication_rounds + 1):
        local_state_dicts: list[dict[str, torch.Tensor]] = []
        sample_counts: list[int] = []
        train_losses: list[float] = []
        for client in clients:
            local_model = CNNLSTMAttentionRegressor(prediction_horizon=config.prediction_horizon).to(device)
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
        history_record.update(evaluate_round(global_model, clients, device))
        round_history.append(history_record)

    client_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        metrics, prediction_df = evaluate_client_model(global_model, client, device)
        client_rows.append({"method": "FedAvg", **metrics})
        prediction_frames.append(prediction_df.assign(method="FedAvg"))
    return pd.DataFrame(client_rows), pd.DataFrame(round_history), pd.concat(prediction_frames, ignore_index=True)


def run_independent_experiment(config: ExperimentConfig, clients: list[ClientData], device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the independent baseline with the same data splits."""
    total_epochs = config.independent_total_epochs or (config.communication_rounds * config.local_epochs)
    client_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        model = CNNLSTMAttentionRegressor(prediction_horizon=config.prediction_horizon).to(device)
        train_local_model(
            model,
            client.train_loader,
            device=device,
            learning_rate=config.learning_rate,
            local_epochs=total_epochs,
        )
        metrics, prediction_df = evaluate_client_model(model, client, device)
        client_rows.append({"method": "Independent", **metrics})
        prediction_frames.append(prediction_df.assign(method="Independent"))
    return pd.DataFrame(client_rows), pd.concat(prediction_frames, ignore_index=True)


def export_results(
    config: ExperimentConfig,
    output_dir: Path,
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    fed_client_df: pd.DataFrame,
    ind_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
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
    write_text("python -m real_data_experiments.single_intersection_client.sic_core " + " ".join(sys.argv[1:]), output_dir / "run_commands.txt")
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    write_csv(main_metrics_df, output_dir / "main_metrics.csv")
    write_json(main_metrics_df.to_dict(orient="records"), output_dir / "main_metrics.json")
    write_csv(main_summary_df, output_dir / "main_summary.csv")
    write_json(main_summary_df.to_dict(orient="records"), output_dir / "main_summary.json")
    write_csv(client_metrics_df, output_dir / "client_metrics.csv")
    write_json(client_metrics_df.to_dict(orient="records"), output_dir / "client_metrics.json")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_json(convergence_df.to_dict(orient="records"), output_dir / "convergence_history.json")
    write_csv(prediction_df.head(config.prediction_sample_limit), output_dir / "prediction_samples.csv")
    write_json(prediction_df.head(config.prediction_sample_limit).to_dict(orient="records"), output_dir / "prediction_samples.json")
    write_text(
        "\n".join(
            [
                "# 单路口客户端实验说明",
                "",
                "- 本次迁移默认主方法为标准样本量加权 FedAvg。",
                "- 数据入口为 data/analysis/node_intersection_flow_parquet，而非 notebook 中缺失的 6.池化网格张量.pt。",
                "- 数据划分采用时间顺序 train/val/test，不使用随机打乱。",
                "- 当前最小交付版本先完成单路口主实验主线与指标导出。",
            ]
        ),
        output_dir / "experiment_notes_zh.md",
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run the single-intersection experiment and export reproducible outputs."""
    output_dir = prepare_output_dir(config.output_dir)
    device = resolve_default_device(config.device)
    set_global_seed(config.seed)
    start_time = datetime.now().isoformat(timespec="seconds")

    node_ids = choose_node_ids(config)
    matrix = build_single_intersection_matrix(config, node_ids)
    clients, split_summary = build_client_data(config, matrix, device)
    split_summary["resolved_input_path"] = str(resolve_path(config.input_path))
    split_summary["max_chunks"] = config.max_chunks

    fed_client_df, convergence_df, fed_prediction_df = run_fedavg_experiment(config, clients, device)
    ind_client_df, ind_prediction_df = run_independent_experiment(config, clients, device)
    prediction_df = pd.concat([fed_prediction_df, ind_prediction_df], ignore_index=True)

    environment_summary = build_environment_summary(device)
    environment_summary["seed"] = config.seed
    environment_summary["start_time"] = start_time
    environment_summary["end_time"] = datetime.now().isoformat(timespec="seconds")

    export_results(
        config=config,
        output_dir=output_dir,
        environment_summary=environment_summary,
        split_summary=split_summary,
        fed_client_df=fed_client_df,
        ind_client_df=ind_client_df,
        convergence_df=convergence_df,
        prediction_df=prediction_df,
    )
    return {
        "output_dir": str(output_dir),
        "selected_node_ids": node_ids,
        "client_metrics_rows": int(len(fed_client_df) + len(ind_client_df)),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[single_intersection_client] completed -> {result['output_dir']}")
    print(f"[selected_node_ids] {result['selected_node_ids']}")


if __name__ == "__main__":
    main()
