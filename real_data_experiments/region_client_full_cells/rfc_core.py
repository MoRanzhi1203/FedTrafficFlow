"""Full-cells region-client diagnostic experiment with standard FedAvg."""

from __future__ import annotations

import sys
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from real_data_experiments.common.client import FedClient
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, resolve_default_device, set_global_seed
from real_data_experiments.common.trainer import run_federated_rounds
from real_data_experiments.region_client_full_cells.rfc_config import ExperimentConfig, build_arg_parser, config_from_args
from real_data_experiments.region_client_full_cells.rfc_dataset import RFCClientData, build_full_cells_client_data
from real_data_experiments.single_intersection_client.sic_core import (
    CNNLSTMAttentionRegressor,
    InputScaler,
    TargetScaler,
    apply_dataset_normalization,
    collect_predictions,
    fit_input_scaler,
    fit_target_scaler,
    train_local_model,
)


# #region debug-point A:emit
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: dict[str, Any] | None = None) -> None:
    env_path = ".dbg/rfc-smoke-stall.env"
    server_url = "http://127.0.0.1:7777/event"
    session_id = "rfc-smoke-stall"
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as env_file:
                for line in env_file:
                    line = line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        server_url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
            "ts": int(time.time() * 1000),
        }
        urllib.request.urlopen(
            urllib.request.Request(
                server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass


# #endregion


def _make_client_record(client: RFCClientData) -> dict[str, object]:
    record: dict[str, object] = {
        "client_id": int(client.client_id),
        "entity_kind": client.entity_kind,
        "entity_id": int(client.entity_id),
        "cell_count": int(len(client.cell_ids)),
        "cell_ids": ",".join(str(cell_id) for cell_id in client.cell_ids),
        "train_samples": int(len(client.train_loader.dataset)),
        "val_samples": int(len(client.val_loader.dataset)),
        "test_samples": int(len(client.test_loader.dataset)),
    }
    for key, value in client.client_metadata.items():
        if key not in {"client_id", "cell_ids"}:
            record[key] = value
    return record


def _accumulate_region_window_input_stats(dataset) -> tuple[torch.Tensor, torch.Tensor, int]:
    first_start = int(dataset.first_target_time - dataset.target_offset)
    last_end = int(dataset.last_target_time - dataset.target_offset + dataset.input_length)
    window_source = dataset.tensor[dataset.use_channels][:, dataset.region_ids, first_start:last_end].to(dtype=torch.float64)
    window_view = window_source.unfold(dimension=2, size=dataset.input_length, step=1)
    channel_sum = window_view.sum(dim=(1, 2, 3))
    channel_sq_sum = torch.square(window_view).sum(dim=(1, 2, 3))
    element_count = int(len(dataset.region_ids) * dataset.per_region_window_count * dataset.input_length)
    return channel_sum, channel_sq_sum, element_count


def _collect_region_window_targets(dataset) -> np.ndarray:
    target_slice = dataset.tensor[
        dataset.target_channel,
        dataset.region_ids,
        dataset.first_target_time : dataset.last_target_time + 1,
    ].to(dtype=torch.float64)
    return target_slice.reshape(-1).detach().cpu().numpy()


def fit_rfc_input_scaler(clients: list[RFCClientData], eps: float = 1e-6) -> InputScaler:
    if not clients or any(client.raw_train_dataset is None for client in clients):
        return fit_input_scaler(clients, eps=eps)
    global_sum: torch.Tensor | None = None
    global_sq_sum: torch.Tensor | None = None
    total_count = 0
    for client in clients:
        input_sum, input_sq_sum, element_count = _accumulate_region_window_input_stats(client.raw_train_dataset)
        global_sum = input_sum if global_sum is None else global_sum + input_sum
        global_sq_sum = input_sq_sum if global_sq_sum is None else global_sq_sum + input_sq_sum
        total_count += element_count
    if global_sum is None or global_sq_sum is None or total_count <= 0:
        raise ValueError("No train data available for input normalization.")
    mean = (global_sum / float(total_count)).to(dtype=torch.float32).view(-1, 1)
    variance = (global_sq_sum / float(total_count)) - torch.square(global_sum / float(total_count))
    std = torch.sqrt(torch.clamp(variance, min=float(eps))).to(dtype=torch.float32).view(-1, 1)
    return InputScaler(mean=mean, std=std)


def fit_rfc_target_scaler(clients: list[RFCClientData], eps: float = 1e-6) -> TargetScaler:
    if not clients or any(client.raw_train_dataset is None for client in clients):
        return fit_target_scaler(clients, eps=eps)
    train_targets = np.concatenate([_collect_region_window_targets(client.raw_train_dataset) for client in clients])
    std = float(np.std(train_targets, ddof=0))
    target_mean = float(np.mean(train_targets))
    target_std = max(std, float(eps))
    return TargetScaler(mean=target_mean, std=target_std)


def evaluate_round(
    global_model: nn.Module,
    clients: list[RFCClientData],
    device: str,
    target_scaler=None,
) -> dict[str, float]:
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


def evaluate_client_model(
    model: nn.Module,
    client: RFCClientData,
    device: str,
    method: str,
    target_scaler=None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true, y_pred = collect_predictions(model, client.test_loader, device, target_scaler=target_scaler)
    metrics: dict[str, Any] = compute_regression_metrics(y_true, y_pred)
    metrics.update(_make_client_record(client))
    prediction_df = pd.DataFrame(
        {
            **{key: value for key, value in _make_client_record(client).items() if key not in {"train_samples", "val_samples", "test_samples"}},
            "method": method,
            "sample_index": np.arange(len(y_true), dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return metrics, prediction_df


def run_fedavg_experiment(
    config: ExperimentConfig,
    clients: list[RFCClientData],
    device: str,
    target_scaler=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_channels = len(config.use_channels)
    model_fn = lambda: CNNLSTMAttentionRegressor(
        input_channels=input_channels,
        prediction_horizon=config.prediction_horizon,
    )
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
    convergence_df = pd.DataFrame(history)
    if not convergence_df.empty:
        convergence_df.insert(0, "method", "FedAvg")

    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        metrics, prediction_df = evaluate_client_model(
            trained_model,
            client,
            device=device,
            method="FedAvg",
            target_scaler=target_scaler,
        )
        client_rows.append({"method": "FedAvg", **metrics})
        prediction_frames.append(prediction_df)
    return pd.DataFrame(client_rows), convergence_df, pd.concat(prediction_frames, ignore_index=True)


def run_independent_experiment(
    config: ExperimentConfig,
    clients: list[RFCClientData],
    device: str,
    target_scaler=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_epochs = config.independent_total_epochs or (config.communication_rounds * config.local_epochs)
    input_channels = len(config.use_channels)
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
            show_progress=config.show_progress,
            progress_interval=config.progress_interval,
            progress_prefix=f"RFC Independent c{client.client_id}",
        )
        metrics, prediction_df = evaluate_client_model(
            model,
            client,
            device=device,
            method="Independent",
            target_scaler=target_scaler,
        )
        client_rows.append({"method": "Independent", **metrics})
        prediction_frames.append(prediction_df)
    return pd.DataFrame(client_rows), pd.concat(prediction_frames, ignore_index=True)


def evaluate_naive_last_value(config: ExperimentConfig, clients: list[RFCClientData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_channel_index = list(config.use_channels).index(config.target_channel)
    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        if client.raw_test_dataset is None:
            raise ValueError("raw_test_dataset is required for NaiveLastValue evaluation.")
        y_true: list[float] = []
        y_pred: list[float] = []
        for index in range(len(client.raw_test_dataset)):
            features, target = client.raw_test_dataset[index]
            y_true.append(float(target.view(-1)[0].item()))
            y_pred.append(float(features[target_channel_index, -1].item()))
        y_true_array = np.asarray(y_true, dtype=np.float64)
        y_pred_array = np.asarray(y_pred, dtype=np.float64)
        metrics: dict[str, Any] = compute_regression_metrics(y_true_array, y_pred_array)
        metrics.update(_make_client_record(client))
        client_rows.append({"method": "NaiveLastValue", **metrics})
        prediction_frames.append(
            pd.DataFrame(
                {
                    **{key: value for key, value in _make_client_record(client).items() if key not in {"train_samples", "val_samples", "test_samples"}},
                    "method": "NaiveLastValue",
                    "sample_index": np.arange(len(y_true_array), dtype=int),
                    "y_true": y_true_array,
                    "y_pred": y_pred_array,
                }
            )
        )
    return pd.DataFrame(client_rows), pd.concat(prediction_frames, ignore_index=True)


def limit_prediction_samples(prediction_df: pd.DataFrame, total_limit: int) -> pd.DataFrame:
    if total_limit <= 0 or prediction_df.empty or "method" not in prediction_df.columns:
        return prediction_df.copy()
    groups = list(prediction_df.groupby("method", sort=False))
    quota = max(total_limit // max(len(groups), 1), 1)
    sampled = [group.head(quota) for _, group in groups]
    return pd.concat(sampled, ignore_index=True)


def export_results(
    config: ExperimentConfig,
    output_dir: Path,
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    client_membership: dict[str, object],
    client_distribution_df: pd.DataFrame,
    fed_client_df: pd.DataFrame,
    ind_client_df: pd.DataFrame,
    naive_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    input_scaler=None,
    target_scaler=None,
) -> None:
    client_metrics_df = pd.concat([fed_client_df, ind_client_df, naive_client_df], ignore_index=True)
    main_metrics_df = (
        client_metrics_df.groupby("method", as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values("method")
        .reset_index(drop=True)
    )
    main_summary_df = summarize_metric_frame(client_metrics_df, group_cols=["method"])

    write_json(config.to_dict(), output_dir / "run_config.json")
    write_text("python -m real_data_experiments.region_client_full_cells.rfc_core " + " ".join(sys.argv[1:]), output_dir / "run_commands.txt")
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    write_json(client_membership, output_dir / "client_membership.json")
    write_csv(client_distribution_df, output_dir / "client_distribution_summary.csv")
    write_csv(main_metrics_df, output_dir / "main_metrics.csv")
    write_csv(main_summary_df, output_dir / "main_summary.csv")
    write_csv(client_metrics_df, output_dir / "client_metrics.csv")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_csv(limit_prediction_samples(prediction_df, config.prediction_sample_limit), output_dir / "prediction_samples.csv")
    if input_scaler is not None:
        write_json(input_scaler.to_dict(), output_dir / "input_scaler.json")
    if target_scaler is not None:
        write_json(target_scaler.to_dict(), output_dir / "target_scaler.json")


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    # #region debug-point A:run-start
    _debug_emit(
        "A",
        "rfc_core.run_experiment:start",
        "run_experiment entered",
        {
            "output_dir": config.output_dir,
            "partition_file": config.partition_file,
            "rounds": config.communication_rounds,
            "local_epochs": config.local_epochs,
            "device": config.device,
        },
    )
    # #endregion
    output_dir = prepare_output_dir(config.output_dir)
    device = resolve_default_device(config.device)
    set_global_seed(config.seed)
    start_time = datetime.now().isoformat(timespec="seconds")

    build_start = time.perf_counter()
    # #region debug-point B:dataset-build-start
    _debug_emit("B", "rfc_core.run_experiment:before_build_full_cells_client_data", "building client datasets")
    # #endregion
    clients, split_summary, client_membership, client_distribution_df = build_full_cells_client_data(config)
    # #region debug-point B:dataset-build-end
    _debug_emit(
        "B",
        "rfc_core.run_experiment:after_build_full_cells_client_data",
        "client datasets built",
        {
            "elapsed_sec": round(time.perf_counter() - build_start, 3),
            "num_clients": len(clients),
            "used_region_count": int(sum(len(client.cell_ids) for client in clients)),
        },
    )
    # #endregion
    scaler_start = time.perf_counter()
    # #region debug-point C:scaler-start
    _debug_emit("C", "rfc_core.run_experiment:before_scalers", "fitting scalers")
    # #endregion
    input_scaler = fit_rfc_input_scaler(clients, eps=config.input_normalization_eps) if config.input_normalization else None
    target_scaler = fit_rfc_target_scaler(clients, eps=config.target_normalization_eps) if config.target_normalization else None
    # #region debug-point C:scaler-end
    _debug_emit(
        "C",
        "rfc_core.run_experiment:after_scalers",
        "scalers fitted",
        {
            "elapsed_sec": round(time.perf_counter() - scaler_start, 3),
            "input_scaler": bool(input_scaler),
            "target_scaler": bool(target_scaler),
        },
    )
    # #endregion
    split_summary["input_normalization"] = {"enabled": bool(input_scaler), **(input_scaler.to_dict() if input_scaler is not None else {})}
    split_summary["target_normalization"] = (
        {"enabled": True, "mean": target_scaler.mean, "std": target_scaler.std}
        if target_scaler is not None
        else {"enabled": False}
    )
    apply_dataset_normalization(clients, input_scaler=input_scaler, target_scaler=target_scaler)

    fedavg_start = time.perf_counter()
    # #region debug-point D:fedavg-start
    _debug_emit("D", "rfc_core.run_experiment:before_fedavg", "starting FedAvg")
    # #endregion
    fed_client_df, convergence_df, fed_prediction_df = run_fedavg_experiment(config, clients, device=device, target_scaler=target_scaler)
    # #region debug-point D:fedavg-end
    _debug_emit(
        "D",
        "rfc_core.run_experiment:after_fedavg",
        "FedAvg completed",
        {"elapsed_sec": round(time.perf_counter() - fedavg_start, 3), "rows": int(len(fed_client_df))},
    )
    # #endregion
    independent_start = time.perf_counter()
    # #region debug-point E:independent-start
    _debug_emit("E", "rfc_core.run_experiment:before_independent", "starting Independent")
    # #endregion
    ind_client_df, ind_prediction_df = run_independent_experiment(config, clients, device=device, target_scaler=target_scaler)
    # #region debug-point E:independent-end
    _debug_emit(
        "E",
        "rfc_core.run_experiment:after_independent",
        "Independent completed",
        {"elapsed_sec": round(time.perf_counter() - independent_start, 3), "rows": int(len(ind_client_df))},
    )
    # #endregion
    naive_start = time.perf_counter()
    # #region debug-point F:naive-start
    _debug_emit("F", "rfc_core.run_experiment:before_naive", "starting NaiveLastValue")
    # #endregion
    naive_client_df, naive_prediction_df = evaluate_naive_last_value(config, clients)
    # #region debug-point F:naive-end
    _debug_emit(
        "F",
        "rfc_core.run_experiment:after_naive",
        "NaiveLastValue completed",
        {"elapsed_sec": round(time.perf_counter() - naive_start, 3), "rows": int(len(naive_client_df))},
    )
    # #endregion
    prediction_df = pd.concat([fed_prediction_df, ind_prediction_df, naive_prediction_df], ignore_index=True)

    environment_summary = build_environment_summary(device)
    environment_summary["seed"] = config.seed
    environment_summary["start_time"] = start_time
    environment_summary["end_time"] = datetime.now().isoformat(timespec="seconds")

    # #region debug-point G:export-start
    _debug_emit("G", "rfc_core.run_experiment:before_export", "exporting results", {"prediction_rows": int(len(prediction_df))})
    # #endregion
    export_results(
        config=config,
        output_dir=output_dir,
        environment_summary=environment_summary,
        split_summary=split_summary,
        client_membership=client_membership,
        client_distribution_df=client_distribution_df,
        fed_client_df=fed_client_df,
        ind_client_df=ind_client_df,
        naive_client_df=naive_client_df,
        convergence_df=convergence_df,
        prediction_df=prediction_df,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )
    # #region debug-point G:export-end
    _debug_emit("G", "rfc_core.run_experiment:after_export", "results exported", {"output_dir": str(output_dir)})
    # #endregion
    return {
        "output_dir": str(output_dir),
        "num_clients": int(len(clients)),
        "used_region_count": int(sum(len(client.cell_ids) for client in clients)),
        "partition_mode": client_membership["partition_mode"],
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[rfc] completed -> {result['output_dir']}")
    print(f"[partition_mode] {result['partition_mode']}")
    print(f"[num_clients] {result['num_clients']}")
    print(f"[used_region_count] {result['used_region_count']}")


if __name__ == "__main__":
    main()
