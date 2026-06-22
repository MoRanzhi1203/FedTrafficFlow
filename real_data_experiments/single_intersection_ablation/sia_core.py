"""Single-intersection ablation experiment under standard FedAvg."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from real_data_experiments.common.client import FedClient
from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, resolve_default_device, set_global_seed
from real_data_experiments.common.trainer import run_federated_rounds
from real_data_experiments.single_intersection_ablation.sia_config import (
    DEFAULT_VARIANTS,
    ExperimentConfig,
    build_arg_parser,
    config_from_args,
)
from real_data_experiments.single_intersection_client.sic_core import (
    Attention,
    ClientData,
    build_client_data,
    build_single_intersection_matrix,
    choose_node_ids,
    collect_predictions,
)


VARIANT_LABELS = {
    "full": "Full",
    "without_attention": "Without Attention",
    "without_cnn": "Without CNN / Spatial Encoder",
    "without_lstm": "Without LSTM",
}


class SingleIntersectionAblationModel(nn.Module):
    """Configurable CNN/LSTM/Attention ablation model for a single node."""

    def __init__(
        self,
        variant: str,
        hidden_dim: int = 32,
        prediction_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.use_cnn = variant in {"full", "without_attention", "without_lstm"}
        self.use_lstm = variant in {"full", "without_attention", "without_cnn"}
        self.use_attention = variant in {"full", "without_cnn", "without_lstm"}

        if self.use_cnn:
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            feature_dim = 32
        else:
            self.encoder = None
            feature_dim = 1

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
            post_dim = hidden_dim
        else:
            self.lstm = None
            post_dim = feature_dim

        self.attention = Attention(post_dim) if self.use_attention else None
        self.head = nn.Linear(post_dim, prediction_horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            sequence = self.encoder(inputs).transpose(1, 2)
        else:
            sequence = inputs.transpose(1, 2)

        if self.use_lstm:
            sequence, _ = self.lstm(sequence)

        if self.use_attention and self.attention is not None:
            features = self.attention(sequence)
        else:
            features = sequence[:, -1, :]

        return self.head(features)


def build_model_fn(variant: str, prediction_horizon: int) -> callable:
    """Build a zero-argument model factory for one ablation variant."""
    return lambda: SingleIntersectionAblationModel(variant=variant, prediction_horizon=prediction_horizon)


def evaluate_round(model: nn.Module, clients: list[ClientData], device: str) -> dict[str, float]:
    """Evaluate validation/test metrics for one federated round."""
    val_rmses: list[float] = []
    val_maes: list[float] = []
    test_rmses: list[float] = []
    for client in clients:
        val_true, val_pred = collect_predictions(model, client.val_loader, device)
        test_true, test_pred = collect_predictions(model, client.test_loader, device)
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


def evaluate_variant(
    model: nn.Module,
    clients: list[ClientData],
    device: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one ablation variant on all client test splits."""
    client_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    for client in clients:
        y_true, y_pred = collect_predictions(model, client.test_loader, device)
        metrics = compute_regression_metrics(y_true, y_pred)
        client_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "client_id": client.client_id,
                "node_id": client.node_id,
                "train_samples": len(client.train_loader.dataset),
                "val_samples": len(client.val_loader.dataset),
                "test_samples": len(client.test_loader.dataset),
                **metrics,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "client_id": client.client_id,
                    "node_id": client.node_id,
                    "sample_index": np.arange(len(y_true), dtype=int),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
    return pd.DataFrame(client_rows), pd.concat(prediction_rows, ignore_index=True)


def run_variant(config: ExperimentConfig, clients: list[ClientData], device: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one ablation variant using standard FedAvg only."""
    model_fn = build_model_fn(variant, config.prediction_horizon)
    global_model = model_fn().to(device)
    criterion = nn.MSELoss()
    fed_clients = [
        FedClient(
            client_id=client.client_id,
            model_fn=model_fn,
            train_loader=client.train_loader,
            device=device,
            learning_rate=config.learning_rate,
            local_epochs=config.local_epochs,
            criterion=criterion,
        )
        for client in clients
    ]
    trained_model, history = run_federated_rounds(
        global_model=global_model,
        clients=fed_clients,
        communication_rounds=config.communication_rounds,
        evaluate_fn=lambda model: evaluate_round(model, clients, device),
    )
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        history_df.insert(0, "variant", variant)
        history_df.insert(1, "variant_label", VARIANT_LABELS[variant])
    client_metrics_df, prediction_df = evaluate_variant(trained_model, clients, device, variant)
    return client_metrics_df, history_df, prediction_df


def export_results(
    config: ExperimentConfig,
    output_dir: Path,
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    ablation_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> None:
    """Write ablation experiment artifacts."""
    ablation_metrics_df = (
        ablation_client_df.groupby(["variant", "variant_label"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["variant"])
        .reset_index(drop=True)
    )
    ablation_summary_df = summarize_metric_frame(ablation_client_df, group_cols=["variant", "variant_label"])

    write_json(config.to_dict(), output_dir / "run_config.json")
    write_text("python -m real_data_experiments.single_intersection_ablation.sia_core " + " ".join(sys.argv[1:]), output_dir / "run_commands.txt")
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    write_csv(ablation_metrics_df, output_dir / "ablation_metrics.csv")
    write_csv(ablation_summary_df, output_dir / "ablation_summary.csv")
    write_csv(ablation_client_df, output_dir / "ablation_client_metrics.csv")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_csv(prediction_df.head(400), output_dir / "prediction_samples.csv")
    write_text(
        "\n".join(
            [
                "# 单路口消融实验说明",
                "",
                "- 本实验仅比较模型结构变体，不改变标准样本量加权 FedAvg 聚合。",
                "- 数据入口与单路口主实验一致，均使用 data/analysis/node_intersection_flow_parquet。",
                "- 数据划分为时间顺序 train/val/test，不复用训练集、验证集与测试集。",
            ]
        ),
        output_dir / "experiment_notes_zh.md",
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run all configured single-intersection ablation variants."""
    output_dir = prepare_output_dir(config.output_dir)
    device = resolve_default_device(config.device)
    set_global_seed(config.seed)
    start_time = datetime.now().isoformat(timespec="seconds")

    node_ids = choose_node_ids(config)
    matrix = build_single_intersection_matrix(config, node_ids)
    clients, split_summary = build_client_data(config, matrix, device)
    split_summary["resolved_input_path"] = str(resolve_path(config.input_path))
    split_summary["max_chunks"] = config.max_chunks
    split_summary["variants"] = [VARIANT_LABELS[name] for name in (config.variants or DEFAULT_VARIANTS)]

    variant_names = config.variants or DEFAULT_VARIANTS
    all_client_rows: list[pd.DataFrame] = []
    all_history_rows: list[pd.DataFrame] = []
    all_prediction_rows: list[pd.DataFrame] = []
    for variant in variant_names:
        client_df, history_df, prediction_df = run_variant(config, clients, device, variant)
        all_client_rows.append(client_df)
        all_history_rows.append(history_df)
        all_prediction_rows.append(prediction_df)

    ablation_client_df = pd.concat(all_client_rows, ignore_index=True)
    convergence_df = pd.concat(all_history_rows, ignore_index=True)
    prediction_df = pd.concat(all_prediction_rows, ignore_index=True)

    environment_summary = build_environment_summary(device)
    environment_summary["seed"] = config.seed
    environment_summary["start_time"] = start_time
    environment_summary["end_time"] = datetime.now().isoformat(timespec="seconds")

    export_results(
        config=config,
        output_dir=output_dir,
        environment_summary=environment_summary,
        split_summary=split_summary,
        ablation_client_df=ablation_client_df,
        convergence_df=convergence_df,
        prediction_df=prediction_df,
    )
    return {
        "output_dir": str(output_dir),
        "variants": [VARIANT_LABELS[name] for name in variant_names],
        "selected_node_ids": node_ids,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[single_intersection_ablation] completed -> {result['output_dir']}")
    print(f"[variants] {result['variants']}")


if __name__ == "__main__":
    main()
