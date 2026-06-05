# -*- coding: utf-8 -*-
"""
GCN 基础联邦仿真实验核心逻辑。
负责基础数据导出、GCN 联邦训练、Independent 基线、预测结果与收敛过程导出。
"""

import argparse
import copy
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAFFIC_MIN_VALUE = 0.0
MAPE_EPS = 1.0

BASE_SEED = 42
BASE_NUM_CLIENTS = 5
BASE_NUM_NODES = 8
BASE_SEQ_LEN = 24
BASE_PRED_LEN = 1
BASE_SAMPLES_PER_CLIENT = [200, 200, 200, 200, 200]
BASE_NOISE = 0.05
BASE_TRAIN_RATIO = 0.70
BASE_VAL_RATIO = 0.10
FED_ROUNDS = 10
FED_LOCAL_EPOCHS = 3
FED_BATCH_SIZE = 16
FED_HIDDEN_DIM = 64


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


def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path


def compute_metrics(preds: np.ndarray, truths: np.ndarray):
    diff = preds - truths
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(truths), MAPE_EPS))) * 100.0
    return mse, rmse, mae, mape


def generate_base_traffic_data(seed: int = BASE_SEED):
    rng = np.random.RandomState(seed)
    t_axis = np.arange(BASE_SEQ_LEN)
    base_pattern = (
        0.3 * np.sin(2 * np.pi * t_axis / BASE_SEQ_LEN)
        + 0.5 * np.exp(-0.5 * ((t_axis - 8) / 2) ** 2)
        + 0.6 * np.exp(-0.5 * ((t_axis - 17) / 2) ** 2)
        + 0.2 * np.sin(4 * np.pi * t_axis / BASE_SEQ_LEN + 1.0)
    )
    all_X, all_Y = [], []
    metadata = {
        "num_clients": BASE_NUM_CLIENTS,
        "num_nodes": BASE_NUM_NODES,
        "seq_len": BASE_SEQ_LEN,
        "pred_len": BASE_PRED_LEN,
        "samples_per_client": BASE_SAMPLES_PER_CLIENT,
    }
    for cid in range(BASE_NUM_CLIENTS):
        n_samples = BASE_SAMPLES_PER_CLIENT[cid]
        amp_scale = 1.0 + 0.08 * (cid - BASE_NUM_CLIENTS // 2)
        node_sensitivity = 0.7 + 0.3 * np.sin(np.linspace(0, np.pi, BASE_NUM_NODES) + cid * 0.3)
        x_client = np.zeros((n_samples, BASE_NUM_NODES, BASE_SEQ_LEN), dtype=np.float32)
        y_client = np.zeros(n_samples, dtype=np.float32)
        for sample_idx in range(n_samples):
            sample_noise = rng.randn(BASE_NUM_NODES, BASE_SEQ_LEN) * BASE_NOISE
            for node_idx in range(BASE_NUM_NODES):
                node_flow = (
                    node_sensitivity[node_idx]
                    * amp_scale
                    * (base_pattern + 0.02 * rng.randn(BASE_SEQ_LEN))
                    + sample_noise[node_idx]
                )
                x_client[sample_idx, node_idx, :] = node_flow
            x_client[sample_idx] = np.clip(x_client[sample_idx], TRAFFIC_MIN_VALUE, None)
            y_client[sample_idx] = x_client[sample_idx, :, -BASE_PRED_LEN:].mean()
        all_X.append(x_client)
        all_Y.append(y_client)
    return all_X, all_Y, metadata


def generate_adjacency_matrix(num_nodes: int = BASE_NUM_NODES, seed: int = BASE_SEED):
    rng = np.random.RandomState(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node_idx in range(num_nodes - 1):
        adj[node_idx, node_idx + 1] = 1.0
        adj[node_idx + 1, node_idx] = 1.0
    for _ in range(min(num_nodes // 2, 3)):
        src = rng.randint(0, num_nodes)
        dst = rng.randint(0, num_nodes)
        if src != dst and adj[src, dst] == 0:
            weight = 0.3 + 0.4 * rng.rand()
            adj[src, dst] = weight
            adj[dst, src] = weight
    adj_with_self = adj + np.eye(num_nodes, dtype=np.float32)
    degree = adj_with_self.sum(axis=1)
    degree_inv_sqrt = np.power(degree + 1e-12, -0.5)
    norm = np.diag(degree_inv_sqrt) @ adj_with_self @ np.diag(degree_inv_sqrt)
    graph_meta = {
        "num_nodes": num_nodes,
        "num_edges": int(np.sum(adj > 0) / 2),
        "density": float(np.sum(adj > 0) / max(num_nodes * (num_nodes - 1), 1)),
        "avg_degree": float((adj > 0).sum(axis=1).mean()),
    }
    return norm.astype(np.float32), adj.astype(np.float32), graph_meta


def split_train_val_test(x: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    n_train = int(len(indices) * BASE_TRAIN_RATIO)
    n_val = int(len(indices) * BASE_VAL_RATIO)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return (
        x[train_idx], y[train_idx],
        x[val_idx], y[val_idx],
        x[test_idx], y[test_idx],
    )


class TrafficDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class AdaptiveSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        aggregated = torch.einsum("ij,bjf->bif", adj_norm, x)
        return self.linear(aggregated)


class GCNEncoder(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, adj_norm: np.ndarray):
        super().__init__()
        self.register_buffer("adj_norm", torch.tensor(adj_norm, dtype=torch.float32))
        self.node_proj = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            AdaptiveSwish(),
        )
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()

    def forward(self, x):
        hidden = self.node_proj(x)
        hidden = self.act(self.norm1(self.gcn1(hidden, self.adj_norm)))
        hidden = self.act(self.norm2(self.gcn2(hidden, self.adj_norm)))
        return hidden.mean(dim=1)


class GCNBaseModel(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int, hidden_dim: int, adj_norm: np.ndarray):
        super().__init__()
        self.encoder = GCNEncoder(seq_len, hidden_dim, adj_norm)
        self.lstm = nn.LSTM(num_nodes, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            AdaptiveSwish(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))
        temporal = self.lstm_proj(lstm_out.mean(dim=1))
        fused = torch.stack([encoded, temporal], dim=1)
        attn_out, attn_weights = self.attn(fused, fused, fused)
        return self.head(self.attn_norm(attn_out + fused).mean(dim=1)), attn_weights


class IndependentBaseModel(nn.Module):
    def __init__(self, num_nodes: int, seq_len: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1)), None


class FederatedClient:
    def __init__(self, client_id, model, train_loader, val_loader, test_loader, criterion, lr: float = 1e-3):
        self.client_id = client_id
        self.model = model.to(DEVICE).float()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()
            self.optimizer.zero_grad()
            pred, _ = self.model(x_batch)
            loss = self.criterion(pred.view(-1), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * x_batch.shape[0]
        return total_loss / max(len(self.train_loader.dataset), 1)

    @torch.no_grad()
    def validate(self, loader=None):
        loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()
            pred, _ = self.model(x_batch)
            total_loss += self.criterion(pred.view(-1), y_batch).item() * x_batch.shape[0]
        return total_loss / max(len(loader.dataset), 1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        preds, truths = [], []
        for x_batch, y_batch in self.test_loader:
            x_batch = x_batch.to(DEVICE).float()
            pred, _ = self.model(x_batch)
            preds.append(pred.view(-1).detach().cpu().numpy())
            truths.append(y_batch.detach().cpu().numpy())
        preds_arr = np.concatenate(preds)
        truths_arr = np.concatenate(truths)
        mse, rmse, mae, mape = compute_metrics(preds_arr, truths_arr)
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "preds": preds_arr,
            "truths": truths_arr,
        }

    def train_local(self, epochs: int, global_model=None):
        if global_model is not None:
            self.model.load_state_dict(global_model.state_dict())
        train_loss = 0.0
        for _ in range(epochs):
            train_loss = self.train_epoch()
        return train_loss, copy.deepcopy(self.model.state_dict())


class FedAvgServer:
    def __init__(self, model, client_sizes):
        self.global_model = model.to(DEVICE).float()
        self.client_sizes = client_sizes
        self.round_losses = []

    def aggregate(self, client_weights, client_losses):
        total_size = float(sum(self.client_sizes))
        new_state = OrderedDict()
        for key in client_weights[0]:
            new_state[key] = sum(
                weights[key].to(DEVICE).float() * (self.client_sizes[idx] / total_size)
                for idx, weights in enumerate(client_weights)
            )
        self.global_model.load_state_dict(new_state)
        self.round_losses.append(float(np.mean(client_losses)))


def build_client_splits(seed: int):
    all_x, all_y, metadata = generate_base_traffic_data(seed)
    client_splits = []
    split_overview = {"train": 0, "val": 0, "test": 0}
    for cid in range(BASE_NUM_CLIENTS):
        x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(all_x[cid], all_y[cid], seed + cid)
        split_overview["train"] += len(x_train)
        split_overview["val"] += len(x_val)
        split_overview["test"] += len(x_test)
        client_splits.append({
            "client_id": cid,
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "train_loader": DataLoader(TrafficDataset(x_train, y_train), batch_size=FED_BATCH_SIZE, shuffle=True),
            "val_loader": DataLoader(TrafficDataset(x_val, y_val), batch_size=FED_BATCH_SIZE, shuffle=False),
            "test_loader": DataLoader(TrafficDataset(x_test, y_test), batch_size=FED_BATCH_SIZE, shuffle=False),
        })
    return client_splits, metadata, split_overview, all_x


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("method")[["mse", "rmse", "mae", "mape"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def export_base_dataset_artifacts(output_dir: Path) -> None:
    client_splits, metadata, split_overview, all_x = build_client_splits(BASE_SEED)
    _, raw_adj, graph_meta = generate_adjacency_matrix()
    sequence_rows = []
    grid_rows = []
    distribution_rows = []
    sample_rows = []
    summary_rows = []
    for client in client_splits:
        cid = client["client_id"]
        sequence = all_x[cid].mean(axis=(0, 1))
        for time_step, traffic_flow in enumerate(sequence):
            sequence_rows.append({
                "client_id": cid,
                "time_step": time_step,
                "traffic_flow": float(traffic_flow),
            })
        client_grid = all_x[cid].mean(axis=0)
        for node_id in range(client_grid.shape[0]):
            for time_step in range(client_grid.shape[1]):
                grid_rows.append({
                    "client_id": cid,
                    "node_id": node_id,
                    "time_step": time_step,
                    "traffic_flow": float(client_grid[node_id, time_step]),
                })
        for traffic_flow in all_x[cid].ravel()[::8]:
            distribution_rows.append({"client_id": cid, "traffic_flow": float(traffic_flow)})
        sample_rows.append({"client_id": cid, "num_samples": int(len(all_x[cid]))})
        summary_rows.append({
            "client_id": cid,
            "num_samples": int(len(all_x[cid])),
            "mean_flow": float(all_x[cid].mean()),
            "std_flow": float(all_x[cid].std()),
            "min_flow": float(all_x[cid].min()),
            "max_flow": float(all_x[cid].max()),
        })
    split_rows = []
    total_samples = sum(split_overview.values())
    for split_name in ["train", "val", "test"]:
        split_rows.append({
            "split": split_name,
            "num_samples": split_overview[split_name],
            "ratio": split_overview[split_name] / total_samples,
        })
    save_dataframe(pd.DataFrame(sequence_rows), output_dir, "base_dataset_client_timeseries.csv")
    save_dataframe(pd.DataFrame(grid_rows), output_dir, "base_dataset_node_heatmap.csv")
    save_dataframe(pd.DataFrame(distribution_rows), output_dir, "base_dataset_client_distribution.csv")
    save_dataframe(pd.DataFrame(split_rows), output_dir, "base_dataset_split_overview.csv")
    save_dataframe(pd.DataFrame(sample_rows), output_dir, "base_dataset_client_sample_size.csv")
    save_dataframe(pd.DataFrame(summary_rows), output_dir, "base_dataset_summary.csv")
    save_dataframe(pd.DataFrame(raw_adj), output_dir, "base_graph_adjacency_matrix.csv")
    save_dataframe(pd.DataFrame([graph_meta]), output_dir, "base_graph_summary.csv")


def run_federated_training(seed: int, record_convergence: bool = False):
    set_global_seed(seed)
    client_splits, _, _, _ = build_client_splits(seed)
    adj_norm, _, _ = generate_adjacency_matrix(seed=seed)
    criterion = nn.MSELoss()
    clients = [
        FederatedClient(
            client["client_id"],
            GCNBaseModel(BASE_NUM_NODES, BASE_SEQ_LEN, FED_HIDDEN_DIM, adj_norm),
            client["train_loader"],
            client["val_loader"],
            client["test_loader"],
            criterion,
            lr=1e-3,
        )
        for client in client_splits
    ]
    server = FedAvgServer(
        GCNBaseModel(BASE_NUM_NODES, BASE_SEQ_LEN, FED_HIDDEN_DIM, adj_norm),
        [len(client["x_train"]) for client in client_splits],
    )
    convergence_rows = []
    for round_idx in range(FED_ROUNDS):
        client_weights = []
        client_losses = []
        val_losses = []
        val_metric_rows = []
        for client in clients:
            train_loss, local_state = client.train_local(FED_LOCAL_EPOCHS, server.global_model)
            client_weights.append(local_state)
            client_losses.append(train_loss)
        server.aggregate(client_weights, client_losses)
        if record_convergence:
            for client in clients:
                client.model.load_state_dict(server.global_model.state_dict())
                val_losses.append(client.validate())
                client_eval = client.evaluate()
                val_metric_rows.append(client_eval)
            convergence_rows.append({
                "round": round_idx + 1,
                "method": "FedAvg",
                "avg_train_loss": float(np.mean(client_losses)),
                "avg_val_loss": float(np.mean(val_losses)),
                "avg_val_rmse": float(np.mean([row["rmse"] for row in val_metric_rows])),
                "avg_val_mae": float(np.mean([row["mae"] for row in val_metric_rows])),
                "avg_val_mape": float(np.mean([row["mape"] for row in val_metric_rows])),
            })
    metric_rows = []
    pred_rows = []
    for client, split in zip(clients, client_splits):
        client.model.load_state_dict(server.global_model.state_dict())
        metrics = client.evaluate()
        metric_rows.append({
            "method": "FedAvg",
            "client_id": split["client_id"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
        })
        for sample_id in range(min(200, len(metrics["preds"]))):
            pred_rows.append({
                "method": "FedAvg",
                "client_id": split["client_id"],
                "sample_id": sample_id,
                "y_true": float(metrics["truths"][sample_id]),
                "y_pred": float(metrics["preds"][sample_id]),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), pd.DataFrame(convergence_rows)


def run_independent_training(seed: int):
    set_global_seed(seed)
    client_splits, _, _, _ = build_client_splits(seed)
    criterion = nn.MSELoss()
    metric_rows = []
    pred_rows = []
    for client in client_splits:
        model = IndependentBaseModel(BASE_NUM_NODES, BASE_SEQ_LEN).to(DEVICE).float()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(FED_ROUNDS * FED_LOCAL_EPOCHS):
            model.train()
            for x_batch, y_batch in client["train_loader"]:
                x_batch = x_batch.to(DEVICE).float()
                y_batch = y_batch.to(DEVICE).float()
                optimizer.zero_grad()
                pred, _ = model(x_batch)
                loss = criterion(pred.view(-1), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x_batch, y_batch in client["test_loader"]:
                pred, _ = model(x_batch.to(DEVICE).float())
                preds.append(pred.view(-1).detach().cpu().numpy())
                truths.append(y_batch.detach().cpu().numpy())
        preds_arr = np.concatenate(preds)
        truths_arr = np.concatenate(truths)
        mse, rmse, mae, mape = compute_metrics(preds_arr, truths_arr)
        metric_rows.append({
            "method": "Independent",
            "client_id": client["client_id"],
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
        })
        for sample_id in range(min(200, len(preds_arr))):
            pred_rows.append({
                "method": "Independent",
                "client_id": client["client_id"],
                "sample_id": sample_id,
                "y_true": float(truths_arr[sample_id]),
                "y_pred": float(preds_arr[sample_id]),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def run_main_experiment(output_dir: Path) -> None:
    fed_metrics, fed_preds, _ = run_federated_training(BASE_SEED, record_convergence=False)
    independent_metrics, independent_preds = run_independent_training(BASE_SEED)
    metrics_df = pd.concat([independent_metrics, fed_metrics], ignore_index=True)
    preds_df = pd.concat([independent_preds, fed_preds], ignore_index=True)
    save_dataframe(metrics_df, output_dir, "main_metrics.csv")
    save_dataframe(build_summary(metrics_df), output_dir, "main_summary.csv")
    save_dataframe(preds_df, output_dir, "main_predictions.csv")


def run_convergence_experiment(output_dir: Path) -> None:
    _, _, convergence_df = run_federated_training(BASE_SEED, record_convergence=True)
    save_dataframe(convergence_df, output_dir, "convergence_history.csv")


def run_project(workflow: str, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    if workflow in ("all", "data_viz"):
        export_base_dataset_artifacts(output_dir)
    if workflow in ("all", "main"):
        run_main_experiment(output_dir)
    if workflow in ("all", "convergence"):
        run_convergence_experiment(output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="GCN Base Federated Simulation Core")
    parser.add_argument("--workflow", choices=["all", "data_viz", "main", "convergence"], default="all")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for exported experiment artifacts.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else SIMULATION_RESULTS_ROOT / "gcn_fed_base"
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
