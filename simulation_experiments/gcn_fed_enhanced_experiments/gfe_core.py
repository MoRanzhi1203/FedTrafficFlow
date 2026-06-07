# -*- coding: utf-8 -*-
"""
GCN 增强联邦仿真实验核心逻辑。
负责：增强数据导出、真实图结构构建、GCN 联邦训练、指标计算与 CSV 导出。
"""

import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation_experiments.cnn_fed_enhanced_experiments import cfe_core

RESULTS_ROOT = PROJECT_ROOT / "results"
SIMULATION_RESULTS_ROOT = RESULTS_ROOT / "simulation_experiments"
DEVICE = cfe_core.DEVICE

NUM_NODES = cfe_core.NUM_NODES
SEQ_LEN = cfe_core.SEQ_LEN
PRED_LEN = cfe_core.PRED_LEN
HIDDEN_DIM = 64
COMM_ROUNDS = 4
LOCAL_EPOCHS = 1
LR = 0.001
DEFAULT_MULTI_SEEDS = [42, 2024, 2025, 2026, 3407]
SEEDS = list(DEFAULT_MULTI_SEEDS)


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


def parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"无法解析布尔参数: {value}")


def parse_seed_list(seed_text: str | None) -> list[int]:
    if seed_text is None or not str(seed_text).strip():
        return list(DEFAULT_MULTI_SEEDS)
    seeds = [int(part.strip()) for part in str(seed_text).split(",") if part.strip()]
    if not seeds:
        raise ValueError("至少需要提供一个随机种子")
    return seeds


def compute_r2_score(preds: np.ndarray, truths: np.ndarray) -> float:
    preds_arr = np.asarray(preds, dtype=np.float64)
    truths_arr = np.asarray(truths, dtype=np.float64)
    ss_res = float(np.sum((preds_arr - truths_arr) ** 2))
    ss_tot = float(np.sum((truths_arr - truths_arr.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def build_multi_seed_summary(raw_df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group_df in raw_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base_record = dict(zip(group_cols, group_values))
        for metric in metric_cols:
            if metric not in group_df.columns:
                continue
            values = group_df[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0))
            ci95 = 1.96 * std_value / math.sqrt(values.size)
            record = dict(base_record)
            record.update({
                "metric": metric,
                "mean": mean_value,
                "std": std_value,
                "ci95_lower": float(mean_value - ci95),
                "ci95_upper": float(mean_value + ci95),
                "best": float(np.min(values)) if metric.lower() != "r2" else float(np.max(values)),
                "worst": float(np.max(values)) if metric.lower() != "r2" else float(np.min(values)),
                "n": int(values.size),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def build_multi_seed_convergence_summary(raw_df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group_df in raw_df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = dict(zip(group_cols, group_values))
        record["n"] = int(group_df["seed"].nunique()) if "seed" in group_df.columns else int(len(group_df))
        for metric in metric_cols:
            if metric not in group_df.columns:
                continue
            values = group_df[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0))
            ci95 = 1.96 * std_value / math.sqrt(values.size)
            record[f"{metric}_mean"] = mean_value
            record[f"{metric}_std"] = std_value
            record[f"{metric}_ci95_lower"] = float(mean_value - ci95)
            record[f"{metric}_ci95_upper"] = float(mean_value + ci95)
        rows.append(record)
    return pd.DataFrame(rows)


def build_pairwise_improvement_summary(raw_df: pd.DataFrame, experiment_name: str, baseline_method: str, enhanced_method: str, metric_cols: list[str]) -> pd.DataFrame:
    baseline_df = raw_df[raw_df["method"] == baseline_method].set_index("seed")
    enhanced_df = raw_df[raw_df["method"] == enhanced_method].set_index("seed")
    common_seeds = sorted(set(baseline_df.index) & set(enhanced_df.index))
    rows = []
    for metric in metric_cols:
        if metric not in baseline_df.columns or metric not in enhanced_df.columns:
            continue
        improvements = []
        flags = []
        for seed in common_seeds:
            baseline_value = float(baseline_df.loc[seed, metric])
            enhanced_value = float(enhanced_df.loc[seed, metric])
            if metric.lower() == "r2":
                improvement = (enhanced_value - baseline_value) / max(abs(baseline_value), 1e-8) * 100.0
            else:
                improvement = (baseline_value - enhanced_value) / max(abs(baseline_value), 1e-8) * 100.0
            improvements.append(improvement)
            flags.append(improvement > 0.0)
        if improvements:
            rows.append({
                "experiment": experiment_name,
                "baseline_method": baseline_method,
                "enhanced_method": enhanced_method,
                "metric": metric,
                "mean_improvement_percent": float(np.mean(improvements)),
                "std_improvement_percent": float(np.std(improvements, ddof=0)),
                "improved_seed_count": int(np.sum(flags)),
                "total_seed_count": int(len(flags)),
                "improved_seed_ratio": float(np.mean(flags)),
                "per_seed_improved": ",".join(f"{seed}:{'Y' if flag else 'N'}" for seed, flag in zip(common_seeds, flags)),
            })
    return pd.DataFrame(rows)


def write_multi_seed_stability_report(output_dir: Path, raw_df: pd.DataFrame, improvement_df: pd.DataFrame, experiment_name: str, baseline_method: str, enhanced_method: str) -> Path:
    report_path = ensure_output_dir(output_dir) / "multi_seed_stability_report.txt"
    lines = [
        f"Experiment: {experiment_name}",
        f"Seeds: {', '.join(str(seed) for seed in sorted(raw_df['seed'].unique()))}",
        "",
        "Per-method statistics:",
    ]
    for method, method_df in raw_df.groupby("method"):
        lines.append(
            f"- {method}: "
            f"MAE={method_df['mae'].mean():.4f}±{method_df['mae'].std(ddof=0):.4f}, "
            f"RMSE={method_df['rmse'].mean():.4f}±{method_df['rmse'].std(ddof=0):.4f}, "
            f"MAPE={method_df['mape'].mean():.4f}±{method_df['mape'].std(ddof=0):.4f}, "
            f"R2={method_df['r2'].mean():.4f}±{method_df['r2'].std(ddof=0):.4f}"
        )
    if not improvement_df.empty:
        lines.extend(["", f"{enhanced_method} vs {baseline_method}:"])
        for _, row in improvement_df.iterrows():
            lines.append(
                f"- {row['metric']}: mean improvement={row['mean_improvement_percent']:.2f}% "
                f"(std={row['std_improvement_percent']:.2f}%), "
                f"improved on {int(row['improved_seed_count'])}/{int(row['total_seed_count'])} seeds"
            )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")
    return report_path


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float32)
    adj_with_self = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = adj_with_self.sum(axis=1)
    degree_inv_sqrt = np.power(degree + 1e-12, -0.5)
    return np.diag(degree_inv_sqrt) @ adj_with_self @ np.diag(degree_inv_sqrt)


class AdaptiveSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.linear(torch.einsum("ij,bjf->bif", adj, x))


class GCNEnhancedModel(nn.Module):
    def __init__(self, k: int, t: int, hidden_dim: int = 64, fixed_adj=None):
        super().__init__()
        self.node_proj = nn.Linear(t, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_norm1 = nn.LayerNorm(hidden_dim)
        self.gcn_norm2 = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(k, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            AdaptiveSwish(),
            nn.Linear(32, 1),
        )
        base_adj = np.eye(k, dtype=np.float32) if fixed_adj is None else fixed_adj
        self.register_buffer("adj", torch.tensor(base_adj, dtype=torch.float32))

    def forward(self, x):
        node_hidden = self.act(self.node_proj(x.float()))
        graph_hidden = self.act(self.gcn_norm1(self.gcn1(node_hidden, self.adj)))
        graph_hidden = self.act(self.gcn_norm2(self.gcn2(graph_hidden, self.adj))).mean(dim=1)
        temporal_hidden, _ = self.lstm(x.permute(0, 2, 1).float())
        temporal_hidden = self.lstm_proj(temporal_hidden.mean(dim=1))
        fused = torch.stack([graph_hidden, temporal_hidden], dim=1)
        attn_out, attn_weights = self.attn(fused, fused, fused)
        return self.head(self.attn_norm(attn_out + fused).mean(dim=1)), attn_weights


def build_summary(df: pd.DataFrame, group_cols):
    summary = (
        df.groupby(group_cols)[["mse", "rmse", "mae", "mape"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def build_raw_client_series(client_configs, seed: int):
    raw_series = []
    incident_masks = []
    for cid, cfg in enumerate(client_configs):
        n_timesteps = cfg["n_samples"] + SEQ_LEN + PRED_LEN + 10
        series, incident_mask = cfe_core.generate_traffic_flow(cfg, n_timesteps, NUM_NODES, seed + cid * 100)
        raw_series.append(series)
        incident_masks.append(incident_mask)
    return raw_series, incident_masks


def build_fixed_adjacency(num_nodes: int = NUM_NODES):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for node_idx in range(num_nodes - 1):
        adj[node_idx, node_idx + 1] = 1.0
        adj[node_idx + 1, node_idx] = 1.0
    return normalize_adjacency(adj), adj


def build_period_mask(hours: np.ndarray, period: str):
    if period == "morning_peak":
        return (hours >= 7.0) & (hours < 9.0)
    if period == "evening_peak":
        return (hours >= 17.0) & (hours < 19.0)
    return ((hours < 7.0) | ((hours >= 9.0) & (hours < 17.0)) | (hours >= 19.0))


def build_dynamic_adjacency(raw_series, period: str):
    selected_rows = []
    for series in raw_series:
        hours = (np.arange(len(series)) * 24.0 / len(series)) % 24
        period_mask = build_period_mask(hours, period)
        selected_rows.append(series[period_mask])
    stacked = np.concatenate([rows for rows in selected_rows if len(rows) > 0], axis=0)
    corr = np.corrcoef(stacked, rowvar=False)
    corr = np.nan_to_num(corr)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    return normalize_adjacency(corr), corr.astype(np.float32)


def build_functional_similarity(raw_series):
    stacked = np.concatenate(raw_series, axis=0)
    corr = np.corrcoef(stacked, rowvar=False)
    corr = np.nan_to_num(corr)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    return normalize_adjacency(corr), corr.astype(np.float32)


def build_congestion_delay_matrices(raw_series):
    stacked = np.concatenate(raw_series, axis=0)
    delay_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    strength_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for src in range(NUM_NODES):
        for dst in range(NUM_NODES):
            if src == dst:
                continue
            best_strength = 0.0
            best_delay = 0
            src_signal = stacked[:, src]
            dst_signal = stacked[:, dst]
            for delay in range(4):
                if delay == 0:
                    corr = np.corrcoef(src_signal, dst_signal)[0, 1]
                else:
                    corr = np.corrcoef(src_signal[:-delay], dst_signal[delay:])[0, 1]
                corr = float(np.nan_to_num(corr))
                if abs(corr) > abs(best_strength):
                    best_strength = corr
                    best_delay = delay
            delay_matrix[src, dst] = best_delay
            strength_matrix[src, dst] = max(0.0, best_strength)
    return delay_matrix, strength_matrix, normalize_adjacency(strength_matrix)


def build_graph_bundle(client_configs, seed: int):
    raw_series, _ = build_raw_client_series(client_configs, seed)
    fixed_norm, fixed_raw = build_fixed_adjacency()
    morning_norm, morning_raw = build_dynamic_adjacency(raw_series, "morning_peak")
    evening_norm, evening_raw = build_dynamic_adjacency(raw_series, "evening_peak")
    offpeak_norm, offpeak_raw = build_dynamic_adjacency(raw_series, "off_peak")
    functional_norm, functional_raw = build_functional_similarity(raw_series)
    delay_matrix, strength_matrix, congestion_norm = build_congestion_delay_matrices(raw_series)
    return {
        "raw_series": raw_series,
        "fixed": (fixed_norm, fixed_raw),
        "dynamic_morning_peak": (morning_norm, morning_raw),
        "dynamic_evening_peak": (evening_norm, evening_raw),
        "dynamic_offpeak": (offpeak_norm, offpeak_raw),
        "functional_similarity": (functional_norm, functional_raw),
        "congestion_delay": (congestion_norm, delay_matrix),
        "congestion_strength": (congestion_norm, strength_matrix),
    }


def export_graph_artifacts(output_dir: Path, client_configs, seed: int):
    graphs = build_graph_bundle(client_configs, seed)
    save_dataframe(pd.DataFrame(graphs["fixed"][1]), output_dir, "enhanced_gcn_fixed_adjacency_matrix.csv")
    save_dataframe(pd.DataFrame(graphs["dynamic_morning_peak"][1]), output_dir, "enhanced_gcn_dynamic_adjacency_morning_peak.csv")
    save_dataframe(pd.DataFrame(graphs["dynamic_evening_peak"][1]), output_dir, "enhanced_gcn_dynamic_adjacency_evening_peak.csv")
    save_dataframe(pd.DataFrame(graphs["dynamic_offpeak"][1]), output_dir, "enhanced_gcn_dynamic_adjacency_offpeak.csv")
    save_dataframe(pd.DataFrame(graphs["functional_similarity"][1]), output_dir, "enhanced_gcn_functional_similarity_matrix.csv")
    save_dataframe(pd.DataFrame(graphs["congestion_delay"][1]), output_dir, "enhanced_gcn_congestion_delay_matrix.csv")
    save_dataframe(pd.DataFrame(graphs["congestion_strength"][1]), output_dir, "enhanced_gcn_congestion_strength_matrix.csv")

    delay_rows = []
    interaction_rows = []
    delay_matrix = graphs["congestion_delay"][1]
    strength_matrix = graphs["congestion_strength"][1]
    for src in range(NUM_NODES):
        for dst in range(NUM_NODES):
            delay_rows.append({
                "source_node": src,
                "target_node": dst,
                "delay_rounds": float(delay_matrix[src, dst]),
                "strength": float(strength_matrix[src, dst]),
            })
            if src != dst:
                interaction_rows.append({
                    "source_node": src,
                    "target_node": dst,
                    "delay_rounds": float(delay_matrix[src, dst]),
                    "strength": float(strength_matrix[src, dst]),
                })
    save_dataframe(pd.DataFrame(delay_rows), output_dir, "enhanced_gcn_congestion_delay.csv")
    save_dataframe(pd.DataFrame(interaction_rows), output_dir, "enhanced_gcn_congestion_delay_interaction.csv")

    summary_rows = []
    for graph_name, graph_value in graphs.items():
        if graph_name == "raw_series":
            continue
        _, matrix = graph_value
        summary_rows.append({
            "graph_type": graph_name,
            "num_nodes": matrix.shape[0],
            "mean_weight": float(np.mean(matrix)),
            "max_weight": float(np.max(matrix)),
            "density": float(np.mean(matrix > 0)),
        })
    save_dataframe(pd.DataFrame(summary_rows), output_dir, "enhanced_gcn_graph_summary.csv")
    return graphs


def build_client_data(client_configs, seed: int):
    return cfe_core.build_client_data(client_configs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)


def method_label(method_key: str) -> str:
    return {
        "fedavg": "FedAvg",
        "loss_weighted": "Loss-weighted",
        "data_loss_weighted": "Data-loss weighted",
        "proposed": "Proposed",
        "independent": "Independent",
    }[method_key]


def run_federated_training(client_data, adjacency, am="fedavg", lam=0.5, cr=COMM_ROUNDS, le=LOCAL_EPOCHS, seed=42, rec=False):
    set_global_seed(seed)
    criterion = nn.MSELoss()
    num_clients = len(client_data)
    clients = [
        cfe_core.FederatedClient(
            item["cid"],
            GCNEnhancedModel(NUM_NODES, SEQ_LEN, HIDDEN_DIM, adjacency),
            item["train_loader"],
            item["val_loader"],
            item["test_loader"],
            criterion,
            LR,
        )
        for item in client_data
    ]
    server = cfe_core.AggregationServer(GCNEnhancedModel(NUM_NODES, SEQ_LEN, HIDDEN_DIM, adjacency), num_clients, am, lam)
    server.ds = [item["train_size"] for item in client_data]

    convergence_rows = []
    for round_idx in range(cr):
        client_weights = []
        client_losses = []
        for client in clients:
            train_loss, weights, _ = client.train_local(le, server.gm)
            client_weights.append(weights)
            client_losses.append(train_loss)
        server.aggregate(client_weights, client_losses)
        if rec:
            val_losses = []
            val_metrics = []
            for client, item in zip(clients, client_data):
                client.model.load_state_dict(server.gm.state_dict())
                val_losses.append(client.validate())
                val_metrics.append(client.validate_metrics(item["y_mean"], item["y_std"]))
            convergence_rows.append({
                "round": round_idx + 1,
                "method": method_label(am),
                "avg_train_loss": float(np.mean(client_losses)),
                "avg_val_loss": float(np.mean(val_losses)),
                "avg_val_rmse": float(np.mean([m[1] for m in val_metrics])),
                "avg_val_mae": float(np.mean([m[2] for m in val_metrics])),
                "avg_val_mape": float(np.mean([m[3] for m in val_metrics])),
            })

    results = []
    for client, item in zip(clients, client_data):
        client.model.load_state_dict(server.gm.state_dict())
        metrics = client.test_metrics()
        preds = metrics["preds"] * item["y_std"] + item["y_mean"]
        truths = metrics["truths"] * item["y_std"] + item["y_mean"]
        mse, rmse, mae, mape = cfe_core.compute_metrics(preds, truths)
        results.append({
            "client_id": item["cid"],
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "preds": preds,
            "truths": truths,
            "meta_test": item["meta_test"],
        })
    return results, pd.DataFrame(convergence_rows)


def run_independent_training(client_data, adjacency, seed: int):
    set_global_seed(seed)
    criterion = nn.MSELoss()
    rows = []
    for item in client_data:
        model = GCNEnhancedModel(NUM_NODES, SEQ_LEN, HIDDEN_DIM, adjacency).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(COMM_ROUNDS * LOCAL_EPOCHS):
            model.train()
            for x_batch, y_batch in item["train_loader"]:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred, _ = model(x_batch)
                loss = criterion(pred.view(-1), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for x_batch, y_batch in item["test_loader"]:
                pred, _ = model(x_batch.to(DEVICE))
                preds.append(pred.view(-1).cpu().numpy())
                truths.append(y_batch.cpu().numpy())
        preds = np.concatenate(preds) * item["y_std"] + item["y_mean"]
        truths = np.concatenate(truths) * item["y_std"] + item["y_mean"]
        mse, rmse, mae, mape = cfe_core.compute_metrics(preds, truths)
        rows.append({
            "client_id": item["cid"],
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "preds": preds,
            "truths": truths,
            "meta_test": item["meta_test"],
        })
    return rows


def period_rows(results, method: str, seed: int):
    rows = []
    for item in results:
        hours = item["meta_test"]["target_hour"]
        incident_flags = item["meta_test"]["target_incident_flag"]
        periods = np.array([
            cfe_core.classify_period(float(hour), bool(flag))
            for hour, flag in zip(hours, incident_flags)
        ])
        for period in ["morning_peak", "evening_peak", "off_peak", "incident_period"]:
            mask = periods == period
            if not np.any(mask):
                continue
            mse, rmse, mae, mape = cfe_core.compute_metrics(item["preds"][mask], item["truths"][mask])
            rows.append({
                "seed": seed,
                "method": method,
                "client_id": item["client_id"],
                "period": period,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            })
    return rows


def export_dataset_and_graphs(output_dir: Path):
    cfe_core.export_enhanced_dataset_artifacts(output_dir)
    export_graph_artifacts(output_dir, list(cfe_core.CLIENT_CONFIGS_BASE), 42)


def run_fixed_vs_dynamic_experiment(output_dir: Path):
    graphs = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)
    client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), 42)
    rows = []
    graph_map = {
        "Fixed": graphs["fixed"][0],
        "Dynamic-Morning": graphs["dynamic_morning_peak"][0],
        "Dynamic-Evening": graphs["dynamic_evening_peak"][0],
        "Dynamic-Offpeak": graphs["dynamic_offpeak"][0],
    }
    for graph_type, adjacency in graph_map.items():
        for method_key in ["fedavg", "proposed"]:
            results, _ = run_federated_training(client_data, adjacency, am=method_key, seed=42)
            for item in results:
                rows.append({
                    "graph_type": graph_type,
                    "method": method_label(method_key),
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "gcn_enhanced_fixed_vs_dynamic_metrics.csv")
    save_dataframe(build_summary(df, ["graph_type", "method"]), output_dir, "gcn_enhanced_fixed_vs_dynamic_summary.csv")


def run_congestion_delay_experiment(output_dir: Path):
    graphs = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)
    client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), 42)
    rows = []
    graph_map = {
        "Functional similarity": graphs["functional_similarity"][0],
        "Congestion delay": graphs["congestion_strength"][0],
    }
    for graph_type, adjacency in graph_map.items():
        for method_key in ["fedavg", "proposed"]:
            results, _ = run_federated_training(client_data, adjacency, am=method_key, seed=42)
            for item in results:
                rows.append({
                    "graph_type": graph_type,
                    "method": method_label(method_key),
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "gcn_enhanced_congestion_delay_metrics.csv")
    save_dataframe(build_summary(df, ["graph_type", "method"]), output_dir, "gcn_enhanced_congestion_delay_summary.csv")


def run_single_seed_main_experiment(seed: int):
    graph_bundle = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
    adjacency = graph_bundle["fixed"][0]
    rows = []
    client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
    for method_key in ["fedavg", "proposed"]:
        results, _ = run_federated_training(client_data, adjacency, am=method_key, seed=seed)
        for item in results:
            rows.append({
                "seed": seed,
                "method": method_label(method_key),
                "client_id": item["client_id"],
                "mse": item["mse"],
                "rmse": item["rmse"],
                "mae": item["mae"],
                "mape": item["mape"],
                "r2": compute_r2_score(item["preds"], item["truths"]),
            })
    for item in run_independent_training(client_data, adjacency, seed):
        rows.append({
            "seed": seed,
            "method": "Independent",
            "client_id": item["client_id"],
            "mse": item["mse"],
            "rmse": item["rmse"],
            "mae": item["mae"],
            "mape": item["mape"],
            "r2": compute_r2_score(item["preds"], item["truths"]),
        })
    metrics_df = pd.DataFrame(rows)
    raw_rows = []
    for method in metrics_df["method"].unique():
        method_df = metrics_df[metrics_df["method"] == method]
        raw_rows.append({
            "experiment": "gcn_fed_enhanced_main",
            "method": method,
            "seed": seed,
            "mse": float(method_df["mse"].mean()),
            "rmse": float(method_df["rmse"].mean()),
            "mae": float(method_df["mae"].mean()),
            "mape": float(method_df["mape"].mean()),
            "r2": float(method_df["r2"].mean()),
            "final_loss": np.nan,
            "best_loss": np.nan,
            "communication_rounds": int(COMM_ROUNDS if method != "Independent" else 0),
            "convergence_round": int(COMM_ROUNDS if method != "Independent" else COMM_ROUNDS * LOCAL_EPOCHS),
        })
    return metrics_df, pd.DataFrame(raw_rows)


def run_main_experiment(output_dir: Path):
    frames = []
    raw_frames = []
    for seed in SEEDS:
        metrics_df, raw_df = run_single_seed_main_experiment(seed)
        frames.append(metrics_df)
        raw_frames.append(raw_df)
    df = pd.concat(frames, ignore_index=True)
    save_dataframe(df, output_dir, "gcn_enhanced_main_metrics.csv")
    save_dataframe(build_summary(df, ["method"]), output_dir, "gcn_enhanced_main_summary.csv")
    raw_df = pd.concat(raw_frames, ignore_index=True)
    multi_seed_summary_df = build_multi_seed_summary(
        raw_df,
        group_cols=["experiment", "method"],
        metric_cols=["mae", "rmse", "mape", "r2", "final_loss", "best_loss", "communication_rounds", "convergence_round"],
    )
    improvement_df = build_pairwise_improvement_summary(
        raw_df,
        experiment_name="gcn_fed_enhanced_main",
        baseline_method="FedAvg",
        enhanced_method="Proposed",
        metric_cols=["mae", "rmse", "mape", "r2"],
    )
    save_dataframe(raw_df, output_dir, "multi_seed_raw_results.csv")
    save_dataframe(multi_seed_summary_df, output_dir, "multi_seed_summary.csv")
    if not improvement_df.empty:
        save_dataframe(improvement_df, output_dir, "multi_seed_improvement_summary.csv")
    write_multi_seed_stability_report(
        output_dir=output_dir,
        raw_df=raw_df,
        improvement_df=improvement_df,
        experiment_name="gcn_fed_enhanced_main",
        baseline_method="FedAvg",
        enhanced_method="Proposed",
    )


def run_aggregation_experiment(output_dir: Path):
    graph_bundle = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)
    adjacency = graph_bundle["fixed"][0]
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
        for method_key in ["fedavg", "loss_weighted", "data_loss_weighted", "proposed"]:
            results, _ = run_federated_training(client_data, adjacency, am=method_key, seed=seed)
            for item in results:
                rows.append({
                    "seed": seed,
                    "method": method_label(method_key),
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "gcn_enhanced_aggregation_metrics.csv")
    save_dataframe(build_summary(df, ["method"]), output_dir, "gcn_enhanced_aggregation_summary.csv")


def run_lambda_experiment(output_dir: Path):
    graph_bundle = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)
    adjacency = graph_bundle["fixed"][0]
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
        for lambda_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            results, _ = run_federated_training(client_data, adjacency, am="data_loss_weighted", lam=lambda_value, seed=seed)
            for item in results:
                rows.append({
                    "seed": seed,
                    "method": "Data-loss weighted",
                    "lambda_value": lambda_value,
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    df = pd.DataFrame(rows)
    save_dataframe(df, output_dir, "gcn_enhanced_lambda_metrics.csv")
    save_dataframe(build_summary(df, ["lambda_value", "method"]), output_dir, "gcn_enhanced_lambda_summary.csv")


def run_convergence_experiment(output_dir: Path):
    frames = []
    for seed in SEEDS:
        graph_bundle = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
        adjacency = graph_bundle["fixed"][0]
        for method_key in ["fedavg", "proposed"]:
            client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
            _, conv_df = run_federated_training(client_data, adjacency, am=method_key, seed=seed, rec=True)
            frames.append(conv_df.assign(seed=seed))
    raw_df = pd.concat(frames, ignore_index=True)
    save_dataframe(raw_df, output_dir, "gcn_enhanced_convergence_history.csv")
    save_dataframe(raw_df, output_dir, "multi_seed_convergence_raw.csv")
    summary_df = build_multi_seed_convergence_summary(
        raw_df,
        group_cols=["method", "round"],
        metric_cols=["avg_train_loss", "avg_val_loss", "avg_val_rmse", "avg_val_mae", "avg_val_mape"],
    )
    save_dataframe(summary_df, output_dir, "multi_seed_convergence_summary.csv")


def run_client_scale_experiment(output_dir: Path):
    rows = []
    base_graph = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)["fixed"][0]
    for seed in SEEDS:
        for num_clients in [3, 5, 8]:
            client_configs = cfe_core.build_noniid_client_configs(num_clients)
            client_data = build_client_data(client_configs, seed)
            results, _ = run_federated_training(client_data, base_graph, am="proposed", seed=seed)
            for item in results:
                rows.append({
                    "seed": seed,
                    "method": "Proposed",
                    "num_clients": num_clients,
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    save_dataframe(pd.DataFrame(rows), output_dir, "gcn_enhanced_client_scale_metrics.csv")


def run_noniid_experiment(output_dir: Path):
    rows = []
    base_graph = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)["fixed"][0]
    for seed in SEEDS:
        for noniid_level in ["low", "medium", "high"]:
            client_configs = cfe_core.build_noniid_client_configs(5, noniid_level)
            client_data = build_client_data(client_configs, seed)
            results, _ = run_federated_training(client_data, base_graph, am="proposed", seed=seed)
            for item in results:
                rows.append({
                    "seed": seed,
                    "method": "Proposed",
                    "noniid_level": noniid_level,
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    save_dataframe(pd.DataFrame(rows), output_dir, "gcn_enhanced_noniid_metrics.csv")


def run_client_metrics_experiment(output_dir: Path):
    base_graph = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)["fixed"][0]
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
        fedavg_results, _ = run_federated_training(client_data, base_graph, am="fedavg", seed=seed)
        proposed_results, _ = run_federated_training(client_data, base_graph, am="proposed", seed=seed)
        independent_results = run_independent_training(client_data, base_graph, seed)
        for method, results in [("FedAvg", fedavg_results), ("Proposed", proposed_results), ("Independent", independent_results)]:
            for item in results:
                rows.append({
                    "seed": seed,
                    "method": method,
                    "client_id": item["client_id"],
                    "mse": item["mse"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "mape": item["mape"],
                })
    save_dataframe(pd.DataFrame(rows), output_dir, "gcn_enhanced_client_metrics.csv")


def run_peak_experiment(output_dir: Path):
    base_graph = build_graph_bundle(list(cfe_core.CLIENT_CONFIGS_BASE), 42)["fixed"][0]
    rows = []
    for seed in SEEDS:
        client_data = build_client_data(list(cfe_core.CLIENT_CONFIGS_BASE), seed)
        for method_key in ["fedavg", "proposed"]:
            results, _ = run_federated_training(client_data, base_graph, am=method_key, seed=seed)
            rows.extend(period_rows(results, method_label(method_key), seed))
        rows.extend(period_rows(run_independent_training(client_data, base_graph, seed), "Independent", seed))
    save_dataframe(pd.DataFrame(rows), output_dir, "gcn_enhanced_peak_metrics.csv")


def run_project(workflow: str, output_dir: Path):
    ensure_output_dir(output_dir)
    workflow_map = {
        "data_viz": export_dataset_and_graphs,
        "fixed_vs_dynamic": run_fixed_vs_dynamic_experiment,
        "congestion_delay": run_congestion_delay_experiment,
        "main": run_main_experiment,
        "aggregation": run_aggregation_experiment,
        "lambda": run_lambda_experiment,
        "convergence": run_convergence_experiment,
        "client_scale": run_client_scale_experiment,
        "noniid": run_noniid_experiment,
        "client_metrics": run_client_metrics_experiment,
        "peak": run_peak_experiment,
    }
    selected = list(workflow_map) if workflow == "all" else [workflow]
    for item in selected:
        workflow_map[item](output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="GCN Enhanced Core")
    parser.add_argument(
        "--workflow",
        choices=[
            "all", "data_viz", "fixed_vs_dynamic", "congestion_delay", "main",
            "aggregation", "lambda", "convergence", "client_scale", "noniid",
            "client_metrics", "peak",
        ],
        default="all",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for exported experiment artifacts.")
    parser.add_argument("--multi_seed", type=str, default="True", help="Whether to run multiple seeds.")
    parser.add_argument("--seeds", type=str, default="42,2024,2025,2026,3407", help="Comma-separated random seeds.")
    parser.add_argument("--single_seed", type=int, default=42, help="Single seed used when --multi_seed False.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    global SEEDS
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else SIMULATION_RESULTS_ROOT / "gcn_fed_enhanced_experiments"
    multi_seed = parse_bool_flag(args.multi_seed)
    SEEDS = parse_seed_list(args.seeds) if multi_seed else [int(args.single_seed)]
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
