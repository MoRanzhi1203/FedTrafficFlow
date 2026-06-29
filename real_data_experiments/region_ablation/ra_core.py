"""Regional ablation experiment with tensor-only input and standard FedAvg."""

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
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, set_global_seed
from real_data_experiments.common.trainer import run_federated_rounds
from real_data_experiments.region_ablation.ra_config import (
    DEFAULT_VARIANTS,
    ExperimentConfig,
    build_arg_parser,
    config_from_args,
)
from real_data_experiments.region_client.rc_core import (
    RegionClientData,
    _make_client_record,
    build_region_client_data,
    fit_rc_input_scaler,
    fit_rc_target_scaler,
)
from real_data_experiments.single_intersection_client.sic_core import (
    Attention,
    InputScaler,
    TargetScaler,
    apply_dataset_normalization,
    collect_predictions,
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


class RegionAblationModel(nn.Module):
    """Configurable regional ablation model."""

    def __init__(self, variant: str, input_channels: int, hidden_dim: int = 32, prediction_horizon: int = 1) -> None:
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


def _resolve_input_channels(config: ExperimentConfig) -> int:
    return len(config.use_channels)


def build_model_fn(variant: str, input_channels: int, prediction_horizon: int) -> callable:
    return lambda: RegionAblationModel(
        variant=variant,
        input_channels=input_channels,
        prediction_horizon=prediction_horizon,
    )


def evaluate_round(model: nn.Module, clients: list[RegionClientData], device: str, target_scaler: TargetScaler | None = None) -> dict[str, float]:
    val_rmses: list[float] = []
    test_rmses: list[float] = []
    for client in clients:
        val_true, val_pred = collect_predictions(model, client.val_loader, device, target_scaler=target_scaler)
        test_true, test_pred = collect_predictions(model, client.test_loader, device, target_scaler=target_scaler)
        val_metrics = compute_regression_metrics(val_true, val_pred)
        test_metrics = compute_regression_metrics(test_true, test_pred)
        val_rmses.append(val_metrics["rmse"])
        test_rmses.append(test_metrics["rmse"])
    return {
        "val_rmse": float(np.mean(val_rmses)),
        "val_rmse_std": float(np.std(val_rmses, ddof=0)),
        "test_rmse": float(np.mean(test_rmses)),
        "test_rmse_std": float(np.std(test_rmses, ddof=0)),
    }


def evaluate_variant(
    model: nn.Module,
    clients: list[RegionClientData],
    device: str,
    variant: str,
    target_scaler: TargetScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client_rows: list[dict[str, object]] = []
    for client in clients:
        y_true, y_pred = collect_predictions(model, client.test_loader, device, target_scaler=target_scaler)
        metrics = compute_regression_metrics(y_true, y_pred)
        client_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                **_make_client_record(client),
                **metrics,
            }
        )
    return pd.DataFrame(client_rows), pd.DataFrame()


def run_variant(
    config: ExperimentConfig,
    clients: list[RegionClientData],
    device: str,
    variant: str,
    target_scaler: TargetScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_channels = _resolve_input_channels(config)
    model_fn = build_model_fn(variant, input_channels, config.prediction_horizon)
    criterion = nn.MSELoss()
    global_model = model_fn().to(device)
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
    client_metrics_df, _ = evaluate_variant(trained_model, clients, device, variant, target_scaler=target_scaler)
    return client_metrics_df, history_df


def export_results(
    config: ExperimentConfig,
    output_dir: Path,
    run_config_payload: dict[str, object],
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    partition_assignment_df: pd.DataFrame,
    client_distribution_summary_df: pd.DataFrame,
    non_iid_summary_df: pd.DataFrame,
    ablation_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> None:
    ablation_metrics_df = (
        ablation_client_df.groupby(["variant", "variant_label"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["variant"])
        .reset_index(drop=True)
    )
    ablation_summary_df = summarize_metric_frame(ablation_client_df, group_cols=["variant", "variant_label"])

    write_json(run_config_payload, output_dir / "run_config.json")
    write_text("python -m real_data_experiments.region_ablation.ra_core " + " ".join(sys.argv[1:]), output_dir / "run_commands.txt")
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    write_csv(partition_assignment_df, output_dir / "region_assignment.csv")
    write_csv(client_distribution_summary_df, output_dir / "client_distribution_summary.csv")
    write_csv(non_iid_summary_df, output_dir / "non_iid_summary.csv")
    write_csv(ablation_metrics_df, output_dir / "ablation_metrics.csv")
    write_csv(ablation_summary_df, output_dir / "ablation_summary.csv")
    write_csv(ablation_client_df, output_dir / "ablation_client_metrics.csv")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_text(
        "\n".join(
            [
                "# 区域消融实验说明",
                "",
                "- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。",
                "- 当前 `region client` 表示一组 pooled grid regions。",
                "- 当前消融实验仅比较模型结构变体，不改变标准样本量加权 FedAvg。",
                "- 当前默认划分方法为 `spatial_block`。",
                "- 数据划分按 target time 的时间顺序执行，不使用随机切分。",
            ]
        ),
        output_dir / "experiment_notes_zh.md",
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
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

    clients, split_summary, partition_result = build_region_client_data(config)  # type: ignore[arg-type]

    # --- Scaler fitting and normalization ---
    input_scaler = fit_rc_input_scaler(clients) if config.input_normalization else None
    target_scaler = fit_rc_target_scaler(clients) if config.target_normalization else None
    if input_scaler is not None or target_scaler is not None:
        apply_dataset_normalization(clients, input_scaler=input_scaler, target_scaler=target_scaler)
    if input_scaler is not None:
        split_summary["input_normalization"] = {"enabled": True, **input_scaler.to_dict()}
    else:
        split_summary["input_normalization"] = {"enabled": False}
    if target_scaler is not None:
        split_summary["target_normalization"] = {"enabled": True, **target_scaler.to_dict()}
    else:
        split_summary["target_normalization"] = {"enabled": False}

    selected_variants = list(config.variants or DEFAULT_VARIANTS)
    ablation_frames: list[pd.DataFrame] = []
    convergence_frames: list[pd.DataFrame] = []
    for variant in selected_variants:
        client_df, history_df = run_variant(config, clients, device, variant, target_scaler=target_scaler)
        ablation_frames.append(client_df)
        convergence_frames.append(history_df)

    ablation_client_df = pd.concat(ablation_frames, ignore_index=True)
    convergence_df = pd.concat(convergence_frames, ignore_index=True)

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
        partition_assignment_df=partition_result.assignment_df,
        client_distribution_summary_df=partition_result.client_distribution_summary_df,
        non_iid_summary_df=partition_result.non_iid_summary_df,
        ablation_client_df=ablation_client_df,
        convergence_df=convergence_df,
    )
    return {
        "output_dir": str(output_dir),
        "variant_count": int(len(selected_variants)),
        "num_clients": int(len(clients)),
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[region_ablation] completed -> {result['output_dir']}")
    print(f"[variant_count] {result['variant_count']}")
    print(f"[num_clients] {result['num_clients']}")


if __name__ == "__main__":
    main()
