# -*- coding: utf-8 -*-
"""
联邦鲁棒性实验核心逻辑。
负责真实通信开销统计、客户端掉线、通信延迟、梯度噪声实验与结果导出。
"""

import argparse
import copy
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation_experiments.cnn_fed_base.cfb_core import CNNBaseModel
from simulation_experiments.cnn_fed_enhanced_experiments import cfe_core
from simulation_experiments.gcn_fed_base.gfb_core import GCNBaseModel, generate_adjacency_matrix
from simulation_experiments.gcn_fed_enhanced_experiments.gfe_core import GCNEnhancedModel

RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEVICE = cfe_core.DEVICE

SEEDS = [42, 2024, 2025]
ROBUST_COMM_ROUNDS = 3
ROBUST_LOCAL_EPOCHS = 1
DROPOUT_RATES = [0.0, 0.2, 0.4]
DELAY_ROUNDS = [0, 1, 2]
NOISE_STDS = [0.0, 0.02, 0.05]


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


def count_model_stats(model: nn.Module):
    num_parameters = sum(param.numel() for param in model.parameters())
    parameter_size_mb = num_parameters * 4 / (1024 ** 2)
    return num_parameters, parameter_size_mb


def build_client_data(seed: int):
    return cfe_core.build_client_data(
        list(cfe_core.CLIENT_CONFIGS_BASE),
        cfe_core.NUM_NODES,
        cfe_core.SEQ_LEN,
        cfe_core.PRED_LEN,
        seed,
    )


def build_clients(client_data, lr=cfe_core.LR):
    criterion = nn.MSELoss()
    feature_dim = client_data[0].get("k_dim", cfe_core.NUM_NODES)
    return [
        cfe_core.FederatedClient(
            item["cid"],
            cfe_core.CNNEnhancedModel(feature_dim, cfe_core.SEQ_LEN),
            item["train_loader"],
            item["val_loader"],
            item["test_loader"],
            criterion,
            lr,
        )
        for item in client_data
    ]


def compute_method_weights(method_key: str, losses, data_sizes):
    data_sizes = np.array(data_sizes, dtype=float)
    data_weights = data_sizes / np.maximum(data_sizes.sum(), 1e-8)
    if method_key == "fedavg":
        return data_weights
    inv_loss = 1.0 / (np.array(losses, dtype=float) + 1e-8)
    loss_weights = inv_loss / np.maximum(inv_loss.sum(), 1e-8)
    if method_key == "proposed":
        coeff_var = float(np.std(losses) / (np.mean(losses) + 1e-8))
        data_loss_mix = 1.0 / (1.0 + coeff_var)
        weights = 0.8 * (data_loss_mix * data_weights + (1.0 - data_loss_mix) * loss_weights) + 0.2 / len(losses)
        return weights / np.maximum(weights.sum(), 1e-8)
    return data_weights


def aggregate_states(global_model: nn.Module, state_dicts, losses, data_sizes, method_key: str):
    weights = compute_method_weights(method_key, losses, data_sizes)
    new_state = OrderedDict()
    reference = global_model.state_dict()
    for key in reference:
        new_state[key] = sum(
            state_dicts[idx][key].to(DEVICE).float() * float(weights[idx])
            for idx in range(len(state_dicts))
        )
    return new_state


def add_weight_noise(state_dict, noise_std: float):
    if noise_std <= 0:
        return state_dict
    noisy_state = OrderedDict()
    for key, value in state_dict.items():
        if value.dtype.is_floating_point:
            noisy_state[key] = value + torch.randn_like(value) * noise_std
        else:
            noisy_state[key] = value
    return noisy_state


def evaluate_clients(clients, client_data, method_name: str, seed: int, extra_field: str, extra_value):
    rows = []
    for client, item in zip(clients, client_data):
        metrics = client.test_metrics()
        preds = metrics["preds"] * item["y_std"] + item["y_mean"]
        truths = metrics["truths"] * item["y_std"] + item["y_mean"]
        mse, rmse, mae, mape = cfe_core.compute_metrics(preds, truths)
        rows.append({
            "seed": seed,
            "method": method_name,
            "client_id": item["cid"],
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            extra_field: extra_value,
        })
    return rows


def build_summary(df: pd.DataFrame, group_cols):
    summary = (
        df.groupby(group_cols)[["rmse", "mae", "mape"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def run_dropout_delay_noise_training(seed: int, method_key: str, dropout_rate=0.0, delay_rounds=0, noise_std=0.0):
    set_global_seed(seed)
    rng = np.random.RandomState(seed + int(dropout_rate * 100) + delay_rounds * 10 + int(noise_std * 1000))
    client_data = build_client_data(seed)
    clients = build_clients(client_data)
    feature_dim = client_data[0].get("k_dim", cfe_core.NUM_NODES)
    global_model = cfe_core.CNNEnhancedModel(feature_dim, cfe_core.SEQ_LEN).to(DEVICE)
    state_history = [copy.deepcopy(global_model.state_dict())]
    for _ in range(ROBUST_COMM_ROUNDS):
        active_indices = [idx for idx in range(len(clients)) if rng.rand() >= dropout_rate]
        if not active_indices:
            active_indices = [int(rng.randint(0, len(clients)))]
        delayed_count = 0 if delay_rounds <= 0 else max(1, len(active_indices) // 3)
        delayed_indices = set(rng.choice(active_indices, size=delayed_count, replace=False).tolist()) if delayed_count else set()
        local_states = []
        local_losses = []
        local_sizes = []
        for idx in active_indices:
            if idx in delayed_indices and delay_rounds > 0:
                history_idx = max(0, len(state_history) - 1 - delay_rounds)
                stale_model = cfe_core.CNNEnhancedModel(feature_dim, cfe_core.SEQ_LEN).to(DEVICE)
                stale_model.load_state_dict(state_history[history_idx])
                reference_model = stale_model
            else:
                reference_model = global_model
            train_loss, local_state, _ = clients[idx].train_local(ROBUST_LOCAL_EPOCHS, reference_model)
            local_states.append(add_weight_noise(local_state, noise_std))
            local_losses.append(train_loss)
            local_sizes.append(client_data[idx]["train_size"])
        aggregated_state = aggregate_states(global_model, local_states, local_losses, local_sizes, method_key)
        global_model.load_state_dict(aggregated_state)
        state_history.append(copy.deepcopy(aggregated_state))
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
    return clients, client_data


def run_communication_cost_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    adj_norm, _, _ = generate_adjacency_matrix()
    model_specs = [
        ("CNN-Base", CNNBaseModel(cfe_core.NUM_NODES, cfe_core.SEQ_LEN)),
        ("CNN-Enhanced", cfe_core.CNNEnhancedModel(cfe_core.NUM_NODES, cfe_core.SEQ_LEN)),
        ("GCN-Base", GCNBaseModel(cfe_core.NUM_NODES, cfe_core.SEQ_LEN, 64, adj_norm)),
        ("GCN-Enhanced", GCNEnhancedModel(cfe_core.NUM_NODES, cfe_core.SEQ_LEN, 64, adj_norm)),
    ]
    rows = []
    for model_type, model in model_specs:
        num_parameters, parameter_size_mb = count_model_stats(model)
        for num_clients in [3, 5, 8]:
            for rounds in [3, 5]:
                rows.append({
                    "model_type": model_type,
                    "num_clients": num_clients,
                    "rounds": rounds,
                    "num_parameters": num_parameters,
                    "parameter_size_mb": parameter_size_mb,
                    "total_communication_mb": parameter_size_mb * num_clients * rounds * 2,
                })
    save_dataframe(pd.DataFrame(rows), output_dir, "fed_communication_cost.csv")


def run_client_dropout_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    metric_rows = []
    for seed in SEEDS:
        for dropout_rate in DROPOUT_RATES:
            for method_key, method_name in [("fedavg", "FedAvg"), ("proposed", "Proposed")]:
                clients, client_data = run_dropout_delay_noise_training(
                    seed=seed,
                    method_key=method_key,
                    dropout_rate=dropout_rate,
                    delay_rounds=0,
                    noise_std=0.0,
                )
                metric_rows.extend(evaluate_clients(clients, client_data, method_name, seed, "dropout_rate", dropout_rate))
    metrics_df = pd.DataFrame(metric_rows)
    save_dataframe(metrics_df, output_dir, "fed_client_dropout_metrics.csv")
    save_dataframe(build_summary(metrics_df, ["dropout_rate", "method"]), output_dir, "fed_client_dropout_summary.csv")


def run_communication_delay_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    metric_rows = []
    for seed in SEEDS:
        for delay_rounds in DELAY_ROUNDS:
            for method_key, method_name in [("fedavg", "FedAvg"), ("proposed", "Proposed")]:
                clients, client_data = run_dropout_delay_noise_training(
                    seed=seed,
                    method_key=method_key,
                    dropout_rate=0.0,
                    delay_rounds=delay_rounds,
                    noise_std=0.0,
                )
                metric_rows.extend(evaluate_clients(clients, client_data, method_name, seed, "delay_rounds", delay_rounds))
    metrics_df = pd.DataFrame(metric_rows)
    save_dataframe(metrics_df, output_dir, "fed_communication_delay_metrics.csv")
    save_dataframe(build_summary(metrics_df, ["delay_rounds", "method"]), output_dir, "fed_communication_delay_summary.csv")


def run_gradient_noise_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    metric_rows = []
    for seed in SEEDS:
        for noise_std in NOISE_STDS:
            for method_key, method_name in [("fedavg", "FedAvg"), ("proposed", "Proposed")]:
                clients, client_data = run_dropout_delay_noise_training(
                    seed=seed,
                    method_key=method_key,
                    dropout_rate=0.0,
                    delay_rounds=0,
                    noise_std=noise_std,
                )
                metric_rows.extend(evaluate_clients(clients, client_data, method_name, seed, "noise_std", noise_std))
    metrics_df = pd.DataFrame(metric_rows)
    save_dataframe(metrics_df, output_dir, "fed_gradient_noise_metrics.csv")
    save_dataframe(build_summary(metrics_df, ["noise_std", "method"]), output_dir, "fed_gradient_noise_summary.csv")


def run_project(workflow: str, output_dir: Path):
    ensure_output_dir(output_dir)
    workflow_map = {
        "communication_cost": run_communication_cost_experiment,
        "client_dropout": run_client_dropout_experiment,
        "communication_delay": run_communication_delay_experiment,
        "gradient_noise": run_gradient_noise_experiment,
    }
    selected = list(workflow_map) if workflow == "all" else [workflow]
    for item in selected:
        workflow_map[item](output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Federated Robustness Core")
    parser.add_argument(
        "--workflow",
        choices=["all", "communication_cost", "client_dropout", "communication_delay", "gradient_noise"],
        default="all",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for exported experiment artifacts.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else SIMULATION_RESULTS_ROOT / "fed_robustness"
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
