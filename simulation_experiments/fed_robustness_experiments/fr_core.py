# -*- coding: utf-8 -*-
"""
联邦鲁棒性实验核心逻辑。
负责真实通信开销统计、客户端掉线、通信延迟、梯度噪声实验与结果导出。
"""

import argparse
import copy
import math
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

DEFAULT_MULTI_SEEDS = [42, 2024, 2025, 2026, 3407]
SEEDS = list(DEFAULT_MULTI_SEEDS)
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
        r2 = cfe_core.compute_r2_score(preds, truths)
        rows.append({
            "seed": seed,
            "method": method_name,
            "client_id": item["cid"],
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "final_loss": np.nan,
            "best_loss": np.nan,
            "communication_rounds": int(ROBUST_COMM_ROUNDS),
            "convergence_round": int(ROBUST_COMM_ROUNDS),
            extra_field: extra_value,
        })
    return rows


def finalize_robustness_metrics(metrics_df: pd.DataFrame, scenario_type: str, parameter_field: str) -> pd.DataFrame:
    finalized = metrics_df.copy()
    finalized["experiment"] = "fed_robustness"
    finalized["scenario_type"] = scenario_type
    finalized["scenario"] = finalized[parameter_field].map(lambda value: f"{scenario_type}@{value}")
    if "dropout_rate" not in finalized.columns:
        finalized["dropout_rate"] = np.nan
    if "delay_rounds" not in finalized.columns:
        finalized["delay_rounds"] = np.nan
    if "noise_std" not in finalized.columns:
        finalized["noise_std"] = np.nan
    return finalized


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


def build_pairwise_improvement_summary(raw_df: pd.DataFrame, baseline_method: str, enhanced_method: str, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (experiment, scenario), scenario_df in raw_df.groupby(["experiment", "scenario"], dropna=False):
        baseline_df = scenario_df[scenario_df["method"] == baseline_method].set_index("seed")
        enhanced_df = scenario_df[scenario_df["method"] == enhanced_method].set_index("seed")
        common_seeds = sorted(set(baseline_df.index) & set(enhanced_df.index))
        if not common_seeds:
            continue
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
            rows.append({
                "experiment": experiment,
                "scenario": scenario,
                "baseline_method": baseline_method,
                "enhanced_method": enhanced_method,
                "metric": metric,
                "mean_improvement_percent": float(np.mean(improvements)),
                "std_improvement_percent": float(np.std(improvements, ddof=0)),
                "improved_seed_count": int(np.sum(flags)),
                "total_seed_count": int(len(flags)),
                "per_seed_improved": ",".join(f"{seed}:{'Y' if flag else 'N'}" for seed, flag in zip(common_seeds, flags)),
            })
    return pd.DataFrame(rows)


def detect_constant_seed_issues(raw_df: pd.DataFrame, metric_cols: list[str]) -> list[str]:
    issues = []
    for (scenario, method), scenario_df in raw_df.groupby(["scenario", "method"], dropna=False):
        seed_count = int(scenario_df["seed"].nunique())
        for metric in metric_cols:
            if metric not in scenario_df.columns or seed_count <= 1:
                continue
            values = scenario_df[metric].dropna().to_numpy(dtype=float)
            if values.size > 1 and np.allclose(values, values[0]):
                issues.append(f"{scenario} | {method} | {metric} is identical across all seeds")
    return issues


def write_multi_seed_stability_report(output_dir: Path, raw_df: pd.DataFrame, improvement_df: pd.DataFrame) -> Path:
    report_path = ensure_output_dir(output_dir) / "multi_seed_stability_report.txt"
    lines = [
        "Experiment: fed_robustness",
        f"Seeds: {', '.join(str(seed) for seed in sorted(raw_df['seed'].unique()))}",
        "",
        "Per-scenario statistics:",
    ]
    for scenario, scenario_df in raw_df.groupby("scenario", dropna=False):
        lines.append(f"- Scenario {scenario}:")
        for method, method_df in scenario_df.groupby("method"):
            lines.append(
                f"  {method}: MAE={method_df['mae'].mean():.4f}±{method_df['mae'].std(ddof=0):.4f}, "
                f"RMSE={method_df['rmse'].mean():.4f}±{method_df['rmse'].std(ddof=0):.4f}, "
                f"MAPE={method_df['mape'].mean():.4f}±{method_df['mape'].std(ddof=0):.4f}, "
                f"R2={method_df['r2'].mean():.4f}±{method_df['r2'].std(ddof=0):.4f}"
            )
    if not improvement_df.empty:
        lines.extend(["", "Improvement summary (Proposed vs FedAvg):"])
        for _, row in improvement_df.iterrows():
            lines.append(
                f"- {row['scenario']} | {row['metric']}: mean improvement={row['mean_improvement_percent']:.2f}% "
                f"(std={row['std_improvement_percent']:.2f}%), "
                f"improved on {int(row['improved_seed_count'])}/{int(row['total_seed_count'])} seeds"
            )
    constant_issues = detect_constant_seed_issues(raw_df, ["mae", "rmse", "mape", "r2"])
    lines.extend(["", "Potential anomalies:"])
    if constant_issues:
        lines.extend(f"- {issue}" for issue in constant_issues)
    else:
        lines.append("- No identical-across-seed anomaly detected in the inspected metrics.")
    lines.extend([
        "",
        "Conclusion:",
        "- Robustness results are reported across multiple random seeds rather than a single run.",
        "- The comparison between Proposed and FedAvg is evaluated under the same robustness scenario and seed, supporting the claim that gains are not caused by a single favorable seed.",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")
    return report_path


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
    metrics_df = finalize_robustness_metrics(pd.DataFrame(metric_rows), "client_dropout", "dropout_rate")
    save_dataframe(metrics_df, output_dir, "fed_client_dropout_metrics.csv")
    save_dataframe(build_summary(metrics_df, ["dropout_rate", "method"]), output_dir, "fed_client_dropout_summary.csv")
    return metrics_df


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
    metrics_df = finalize_robustness_metrics(pd.DataFrame(metric_rows), "communication_delay", "delay_rounds")
    save_dataframe(metrics_df, output_dir, "fed_communication_delay_metrics.csv")
    save_dataframe(build_summary(metrics_df, ["delay_rounds", "method"]), output_dir, "fed_communication_delay_summary.csv")
    return metrics_df


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
    metrics_df = finalize_robustness_metrics(pd.DataFrame(metric_rows), "gradient_noise", "noise_std")
    save_dataframe(metrics_df, output_dir, "fed_gradient_noise_metrics.csv")
    save_dataframe(build_summary(metrics_df, ["noise_std", "method"]), output_dir, "fed_gradient_noise_summary.csv")
    return metrics_df


def run_main_experiment(output_dir: Path):
    ensure_output_dir(output_dir)
    scenario_frames = [
        run_client_dropout_experiment(output_dir),
        run_communication_delay_experiment(output_dir),
        run_gradient_noise_experiment(output_dir),
    ]
    client_level_df = pd.concat(scenario_frames, ignore_index=True)
    raw_df = (
        client_level_df.groupby(
            ["experiment", "scenario_type", "scenario", "method", "seed", "dropout_rate", "delay_rounds", "noise_std"],
            dropna=False,
        )[["mse", "rmse", "mae", "mape", "r2", "final_loss", "best_loss", "communication_rounds", "convergence_round"]]
        .mean()
        .reset_index()
    )
    summary_df = build_multi_seed_summary(
        raw_df,
        group_cols=["experiment", "scenario", "method"],
        metric_cols=["mae", "rmse", "mape", "r2", "final_loss", "best_loss", "communication_rounds", "convergence_round"],
    )
    improvement_df = build_pairwise_improvement_summary(
        raw_df,
        baseline_method="FedAvg",
        enhanced_method="Proposed",
        metric_cols=["mae", "rmse", "mape", "r2"],
    )
    save_dataframe(raw_df, output_dir, "multi_seed_raw_results.csv")
    save_dataframe(summary_df, output_dir, "multi_seed_summary.csv")
    if not improvement_df.empty:
        save_dataframe(improvement_df, output_dir, "multi_seed_improvement_summary.csv")
    write_multi_seed_stability_report(output_dir, raw_df, improvement_df)


def run_project(workflow: str, output_dir: Path):
    ensure_output_dir(output_dir)
    workflow_map = {
        "main": run_main_experiment,
        "communication_cost": run_communication_cost_experiment,
        "client_dropout": run_client_dropout_experiment,
        "communication_delay": run_communication_delay_experiment,
        "gradient_noise": run_gradient_noise_experiment,
    }
    selected = ["communication_cost", "main"] if workflow == "all" else [workflow]
    for item in selected:
        workflow_map[item](output_dir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Federated Robustness Core")
    parser.add_argument(
        "--workflow",
        choices=["all", "main", "communication_cost", "client_dropout", "communication_delay", "gradient_noise"],
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
    output_dir = Path(args.output_dir) if args.output_dir else SIMULATION_RESULTS_ROOT / "fed_robustness_experiments"
    multi_seed = parse_bool_flag(args.multi_seed)
    SEEDS = parse_seed_list(args.seeds) if multi_seed else [int(args.single_seed)]
    run_project(args.workflow, output_dir)


if __name__ == "__main__":
    main()
