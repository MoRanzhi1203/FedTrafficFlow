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
from real_data_experiments.common.device_utils import DeviceResolution, resolve_device
from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, set_global_seed
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
    InputScaler,
    TargetScaler,
    apply_dataset_normalization,
    build_client_data,
    collect_predictions,
    fit_input_scaler,
    fit_target_scaler,
)


VARIANT_LABELS = {
    "full": "Full",
    "without_attention": "Without Attention",
    "without_cnn": "Without CNN / Spatial Encoder",
    "without_lstm": "Without LSTM",
}


def build_run_config_payload(config: ExperimentConfig, device_info: DeviceResolution) -> dict[str, object]:
    """Build a run_config payload that records requested and actual device metadata."""
    payload = config.to_dict()
    payload.update(
        {
            "device": device_info.actual_device,
            "requested_device": device_info.requested_device,
            "actual_device": device_info.actual_device,
            "cuda_available": device_info.cuda_available,
            "cuda_device_name": device_info.cuda_device_name,
            "device_fallback_reason": device_info.fallback_reason,
        }
    )
    return payload


class SingleIntersectionAblationModel(nn.Module):
    """Configurable CNN/LSTM/Attention ablation model for a single node."""

    def __init__(
        self,
        variant: str,
        input_channels: int,
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
                nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            feature_dim = 32
        else:
            self.encoder = None
            feature_dim = input_channels

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


def build_model_fn(variant: str, input_channels: int, prediction_horizon: int) -> callable:
    """Build a zero-argument model factory for one ablation variant."""
    return lambda: SingleIntersectionAblationModel(
        variant=variant,
        input_channels=input_channels,
        prediction_horizon=prediction_horizon,
    )


def resolve_input_channels(config: ExperimentConfig) -> int:
    """Resolve model input channels for tensor vs parquet modes."""
    return len(config.use_channels) if config.data_mode == "tensor" else 1


def evaluate_round(model: nn.Module, clients: list[ClientData], device: str, target_scaler: TargetScaler | None = None) -> dict[str, float]:
    """Evaluate validation/test metrics for one federated round."""
    val_rmses: list[float] = []
    val_maes: list[float] = []
    test_rmses: list[float] = []
    for client in clients:
        val_true, val_pred = collect_predictions(model, client.val_loader, device, target_scaler=target_scaler)
        test_true, test_pred = collect_predictions(model, client.test_loader, device, target_scaler=target_scaler)
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
    target_scaler: TargetScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one ablation variant on all client test splits."""
    client_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    for client in clients:
        y_true, y_pred = collect_predictions(model, client.test_loader, device, target_scaler=target_scaler)
        metrics = compute_regression_metrics(y_true, y_pred)
        entity_columns = {
            "client_id": client.client_id,
            "entity_kind": client.entity_kind,
            "entity_id": client.entity_id,
            "train_samples": len(client.train_loader.dataset),
            "val_samples": len(client.val_loader.dataset),
            "test_samples": len(client.test_loader.dataset),
        }
        if client.entity_kind == "region":
            entity_columns["region_id"] = client.entity_id
        if client.entity_kind == "node":
            entity_columns["node_id"] = client.entity_id
        for key, value in client.entity_metadata.items():
            if key != "client_id":
                entity_columns[key] = value
        client_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                **entity_columns,
                **metrics,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "client_id": client.client_id,
                    "entity_kind": client.entity_kind,
                    "entity_id": client.entity_id,
                    "region_id": client.entity_id if client.entity_kind == "region" else None,
                    "node_id": client.entity_id if client.entity_kind == "node" else None,
                    "sample_index": np.arange(len(y_true), dtype=int),
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
    return pd.DataFrame(client_rows), pd.concat(prediction_rows, ignore_index=True)


def run_variant(config: ExperimentConfig, clients: list[ClientData], device: str, variant: str, target_scaler: TargetScaler | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one ablation variant using standard FedAvg only."""
    model_fn = build_model_fn(variant, resolve_input_channels(config), config.prediction_horizon)
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
        evaluate_fn=lambda model: evaluate_round(model, clients, device, target_scaler=target_scaler),
    )
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        history_df.insert(0, "variant", variant)
        history_df.insert(1, "variant_label", VARIANT_LABELS[variant])
    client_metrics_df, prediction_df = evaluate_variant(trained_model, clients, device, variant, target_scaler=target_scaler)
    return client_metrics_df, history_df, prediction_df


def limit_prediction_samples_by_variant(prediction_df: pd.DataFrame, total_limit: int = 400) -> pd.DataFrame:
    """Keep a balanced prediction sample subset so every variant is represented."""
    if total_limit <= 0 or prediction_df.empty or "variant" not in prediction_df.columns:
        return prediction_df.copy()
    variant_groups = list(prediction_df.groupby("variant", sort=False))
    base_quota = max(total_limit // len(variant_groups), 1)
    remainder = max(total_limit - base_quota * len(variant_groups), 0)
    sampled_frames: list[pd.DataFrame] = []
    for index, (_, group_df) in enumerate(variant_groups):
        quota = base_quota + (1 if index < remainder else 0)
        sampled_frames.append(group_df.head(quota))
    return pd.concat(sampled_frames, ignore_index=True)


def export_results(
    config: ExperimentConfig,
    output_dir: Path,
    run_config_payload: dict[str, object],
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    selected_regions_df: pd.DataFrame | None,
    ablation_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    input_scaler: InputScaler | None = None,
    target_scaler: TargetScaler | None = None,
) -> None:
    """Write ablation experiment artifacts."""
    ablation_metrics_df = (
        ablation_client_df.groupby(["variant", "variant_label"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["variant"])
        .reset_index(drop=True)
    )
    ablation_summary_df = summarize_metric_frame(ablation_client_df, group_cols=["variant", "variant_label"])

    write_json(run_config_payload, output_dir / "run_config.json")
    write_text("python -m real_data_experiments.single_intersection_ablation.sia_core " + " ".join(sys.argv[1:]), output_dir / "run_commands.txt")
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    if selected_regions_df is not None and "region_id" in selected_regions_df.columns:
        write_csv(selected_regions_df, output_dir / "selected_regions.csv")
    write_csv(ablation_metrics_df, output_dir / "ablation_metrics.csv")
    write_csv(ablation_summary_df, output_dir / "ablation_summary.csv")
    write_csv(ablation_client_df, output_dir / "ablation_client_metrics.csv")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_csv(limit_prediction_samples_by_variant(prediction_df), output_dir / "prediction_samples.csv")
    if input_scaler is not None:
        write_json(input_scaler.to_dict(), output_dir / "input_scaler.json")
    if target_scaler is not None:
        write_json(target_scaler.to_dict(), output_dir / "target_scaler.json")
    write_text(
        "\n".join(
            [
                "# 单路口消融实验说明",
                "",
                "- 本实验仅比较模型结构变体，不改变标准样本量加权 FedAvg 聚合。",
                "- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。",
                "- 当前客户端表示 pooled-grid-region client，并默认仅使用 active regions。",
                "- 数据划分为按 target time 的时间顺序 train/val/test，不复用训练集、验证集与测试集。",
                "- 输入与目标归一化使用与实验 1 一致的 train-split 统计量 z-score normalization。",
                "- 评估时对预测值执行反归一化回原始尺度。",
            ]
        ),
        output_dir / "experiment_notes_zh.md",
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run all configured single-intersection ablation variants."""
    output_dir = prepare_output_dir(config.output_dir)
    device_info = resolve_device(config.device)
    device = device_info.actual_device
    print(
        "[device] "
        f"requested={device_info.requested_device}, actual={device_info.actual_device}, "
        f"cuda_available={device_info.cuda_available}, cuda_device_name={device_info.cuda_device_name}, "
        f"fallback_reason={device_info.fallback_reason}",
        flush=True,
    )
    set_global_seed(config.seed)
    start_time = datetime.now().isoformat(timespec="seconds")

    clients, split_summary, selected_regions_df = build_client_data(config)
    split_summary["variants"] = [VARIANT_LABELS[name] for name in (config.variants or DEFAULT_VARIANTS)]

    # --- normalization: reuse experiment 1 scaler pipeline ---
    input_scaler: InputScaler | None = None
    target_scaler: TargetScaler | None = None
    if config.input_normalization:
        input_scaler = fit_input_scaler(clients, eps=config.input_normalization_eps)
        split_summary["input_normalization"] = {
            "enabled": True,
            **input_scaler.to_dict(),
        }
    else:
        split_summary["input_normalization"] = {"enabled": False}
    if config.target_normalization:
        target_scaler = fit_target_scaler(clients, eps=config.target_normalization_eps)
        split_summary["target_normalization"] = {
            "enabled": True,
            "mean": target_scaler.mean,
            "std": target_scaler.std,
        }
    else:
        split_summary["target_normalization"] = {"enabled": False}
    apply_dataset_normalization(clients, input_scaler=input_scaler, target_scaler=target_scaler)

    variant_names = config.variants or DEFAULT_VARIANTS
    all_client_rows: list[pd.DataFrame] = []
    all_history_rows: list[pd.DataFrame] = []
    all_prediction_rows: list[pd.DataFrame] = []
    for variant in variant_names:
        client_df, history_df, prediction_df = run_variant(config, clients, device, variant, target_scaler=target_scaler)
        all_client_rows.append(client_df)
        all_history_rows.append(history_df)
        all_prediction_rows.append(prediction_df)

    ablation_client_df = pd.concat(all_client_rows, ignore_index=True)
    convergence_df = pd.concat(all_history_rows, ignore_index=True)
    prediction_df = pd.concat(all_prediction_rows, ignore_index=True)

    run_config_payload = build_run_config_payload(config, device_info)
    environment_summary = build_environment_summary(device)
    environment_summary["seed"] = config.seed
    environment_summary["start_time"] = start_time
    environment_summary["end_time"] = datetime.now().isoformat(timespec="seconds")
    environment_summary["data_mode"] = config.data_mode
    environment_summary["requested_device"] = device_info.requested_device
    environment_summary["actual_device"] = device_info.actual_device
    environment_summary["cuda_device_name"] = device_info.cuda_device_name
    environment_summary["device_fallback_reason"] = device_info.fallback_reason

    export_results(
        config=config,
        output_dir=output_dir,
        run_config_payload=run_config_payload,
        environment_summary=environment_summary,
        split_summary=split_summary,
        selected_regions_df=selected_regions_df,
        ablation_client_df=ablation_client_df,
        convergence_df=convergence_df,
        prediction_df=prediction_df,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    return {
        "output_dir": str(output_dir),
        "variants": [VARIANT_LABELS[name] for name in variant_names],
        "selected_ids": (
            selected_regions_df["region_id"].tolist()
            if selected_regions_df is not None and "region_id" in selected_regions_df.columns
            else split_summary.get("selected_node_ids", [])
        ),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[single_intersection_ablation] completed -> {result['output_dir']}")
    print(f"[variants] {result['variants']}")
    print(f"[selected_ids] {result['selected_ids']}")


if __name__ == "__main__":
    main()
