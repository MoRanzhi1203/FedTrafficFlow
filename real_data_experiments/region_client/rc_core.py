"""Regional-client real-data experiment with tensor-only input and standard FedAvg."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from real_data_experiments.common.client import FedClient
from real_data_experiments.common.device_utils import DeviceResolution, resolve_device
from real_data_experiments.common.metrics import METRIC_COLUMNS, compute_regression_metrics, summarize_metric_frame
from real_data_experiments.common.region_partition import RegionPartitionResult, assign_region_clients
from real_data_experiments.common.region_tensor_dataset import RegionClientWindowDataset
from real_data_experiments.common.result_writer import prepare_output_dir, write_csv, write_json, write_text
from real_data_experiments.common.seed import build_environment_summary, set_global_seed
from real_data_experiments.common.tensor_dataset import (
    build_time_split_bounds,
    get_region_usage_summary,
    load_grid_tensor_bundle,
)
from real_data_experiments.common.trainer import run_federated_rounds
from real_data_experiments.region_client.rc_config import ExperimentConfig, build_arg_parser, config_from_args
from real_data_experiments.single_intersection_client.sic_core import CNNLSTMAttentionRegressor, collect_predictions


@dataclass
class RegionClientData:
    """Per-client datasets and loaders for a multi-region client."""

    client_id: int
    region_ids: list[int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split_metadata: dict[str, object]
    client_metadata: dict[str, object] = field(default_factory=dict)


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


def _resolve_input_channels(config: ExperimentConfig) -> int:
    return len(config.use_channels)


def _maybe_cap_dataset(dataset: RegionClientWindowDataset, max_samples: int | None) -> RegionClientWindowDataset | Subset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(int(max_samples))))


def _describe_dataset(dataset: RegionClientWindowDataset | Subset) -> dict[str, object]:
    if hasattr(dataset, "describe"):
        return dataset.describe()  # type: ignore[return-value]
    base = dataset.dataset.describe()  # type: ignore[union-attr]
    base["sample_count"] = int(len(dataset))
    base["is_capped_subset"] = True
    return base


def _make_client_record(client: RegionClientData) -> dict[str, object]:
    record: dict[str, object] = {
        "client_id": int(client.client_id),
        "entity_kind": "region_group",
        "region_ids": ",".join(str(region_id) for region_id in client.region_ids),
        "region_count": int(len(client.region_ids)),
        "train_samples": int(len(client.train_loader.dataset)),
        "val_samples": int(len(client.val_loader.dataset)),
        "test_samples": int(len(client.test_loader.dataset)),
    }
    for key, value in client.client_metadata.items():
        if key not in {"client_id", "region_ids"}:
            record[key] = value
    return record


def build_region_client_data(
    config: ExperimentConfig,
) -> tuple[list[RegionClientData], dict[str, object], RegionPartitionResult]:
    """Construct tensor-only regional clients and split summaries."""
    import time

    if config.data_mode != "tensor":
        raise ValueError("region_client currently supports tensor-only input.")

    t0 = time.time()
    bundle = load_grid_tensor_bundle(config.tensor_path, config.regions_path)
    print(f"  [build_rc] load_grid_tensor_bundle took {time.time()-t0:.1f}s", flush=True)
    t_part = time.time()
    try:
        partition_result = assign_region_clients(
            bundle=bundle,
            num_clients=config.num_clients,
            partition_method=config.partition_method,
            use_active_regions_only=config.use_active_regions_only,
            target_channel=config.target_channel,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            seed=config.seed,
        )
    except ImportError as exc:
        if config.partition_method != "flow_kmeans":
            raise
        print(f"[region_client] {exc}")
        print("[region_client] falling back to partition_method=spatial_block")
        partition_result = assign_region_clients(
            bundle=bundle,
            num_clients=config.num_clients,
            partition_method="spatial_block",
            use_active_regions_only=config.use_active_regions_only,
            target_channel=config.target_channel,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            seed=config.seed,
        )
    print(f"  [build_rc] assign_region_clients took {time.time()-t_part:.1f}s", flush=True)

    time_bounds = build_time_split_bounds(
        time_count=int(bundle.tensor.shape[2]),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    region_usage = get_region_usage_summary(bundle.regions_df)

    clients: list[RegionClientData] = []
    client_region_counts: dict[str, int] = {}
    client_sample_counts: dict[str, int] = {}
    split_clients: list[dict[str, object]] = []

    assignment_df = partition_result.assignment_df.copy()
    summary_df = partition_result.client_distribution_summary_df.copy()
    t_ds = time.time()
    for row in summary_df.to_dict(orient="records"):
        client_id = int(row["client_id"])
        region_ids = partition_result.client_region_ids[client_id]
        t_c = time.time()
        train_dataset = RegionClientWindowDataset(
            tensor=bundle.tensor,
            region_ids=region_ids,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["train_start"]),
            end_time=int(time_bounds["train_end"]),
        )
        val_dataset = RegionClientWindowDataset(
            tensor=bundle.tensor,
            region_ids=region_ids,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["val_start"]),
            end_time=int(time_bounds["val_end"]),
        )
        test_dataset = RegionClientWindowDataset(
            tensor=bundle.tensor,
            region_ids=region_ids,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["test_start"]),
            end_time=int(time_bounds["test_end"]),
        )
        train_dataset = _maybe_cap_dataset(train_dataset, config.max_samples_per_client_split)
        val_dataset = _maybe_cap_dataset(val_dataset, config.max_samples_per_client_split)
        test_dataset = _maybe_cap_dataset(test_dataset, config.max_samples_per_client_split)
        client = RegionClientData(
            client_id=client_id,
            region_ids=region_ids,
            train_loader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False),
            val_loader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
            test_loader=DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
            split_metadata={
                "train": _describe_dataset(train_dataset),
                "val": _describe_dataset(val_dataset),
                "test": _describe_dataset(test_dataset),
            },
            client_metadata=row,
        )
        print(f"  [build_rc] client {client_id}: {len(region_ids)} regions, {len(train_dataset)} train samples, dataset build took {time.time()-t_c:.1f}s", flush=True)
        clients.append(client)
        client_region_counts[str(client_id)] = int(len(region_ids))
        client_sample_counts[str(client_id)] = int(len(train_dataset))
        split_clients.append(
            {
                "client_id": client_id,
                "region_ids": list(region_ids),
                "region_count": int(len(region_ids)),
                "train": _describe_dataset(train_dataset),
                "val": _describe_dataset(val_dataset),
                "test": _describe_dataset(test_dataset),
            }
        )

    split_summary: dict[str, object] = {
        "data_mode": "tensor",
        "tensor_path": str(Path(config.tensor_path)),
        "regions_path": str(Path(config.regions_path)),
        "tensor_shape": list(bundle.tensor.shape),
        "sequence_length": int(config.sequence_length),
        "prediction_horizon": int(config.prediction_horizon),
        "use_channels": list(config.use_channels),
        "target_channel": int(config.target_channel),
        "total_region_count": int(region_usage["total_region_count"]),
        "active_region_count": int(region_usage["active_region_count"]),
        "used_region_count": int(len(assignment_df)),
        "num_clients": int(len(clients)),
        "partition_method": partition_result.resolved_partition_method,
        "requested_partition_method": partition_result.requested_partition_method,
        "use_active_regions_only": bool(config.use_active_regions_only),
        "split_strategy": "temporal_contiguous_by_target_time",
        "train_end": int(time_bounds["train_end"]),
        "val_end": int(time_bounds["val_end"]),
        "test_end": int(time_bounds["test_end"]),
        "train_start": int(time_bounds["train_start"]),
        "val_start": int(time_bounds["val_start"]),
        "test_start": int(time_bounds["test_start"]),
        "client_region_counts": client_region_counts,
        "client_sample_counts": client_sample_counts,
        "clients": split_clients,
    }
    return clients, split_summary, partition_result


def evaluate_round(global_model: nn.Module, clients: list[RegionClientData], device: str) -> dict[str, float]:
    """Evaluate the current global model across all region clients."""

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


def evaluate_client_model(model: nn.Module, client: RegionClientData, device: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one client on the test split."""

    y_true, y_pred = collect_predictions(model, client.test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics.update(_make_client_record(client))
    prediction_df = pd.DataFrame(
        {
            **{key: value for key, value in _make_client_record(client).items() if key not in {"train_samples", "val_samples", "test_samples"}},
            "sample_index": np.arange(len(y_true), dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    return metrics, prediction_df


def run_fedavg_experiment(
    config: ExperimentConfig,
    clients: list[RegionClientData],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run standard sample-size weighted FedAvg."""

    input_channels = _resolve_input_channels(config)
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
        evaluate_fn=lambda model: evaluate_round(model, clients, device),
    )
    convergence_df = pd.DataFrame(history)
    if not convergence_df.empty:
        convergence_df.insert(0, "method", "FedAvg")

    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        metrics, prediction_df = evaluate_client_model(trained_model, client, device)
        client_rows.append({"method": "FedAvg", **metrics})
        prediction_frames.append(prediction_df.assign(method="FedAvg"))
    return pd.DataFrame(client_rows), convergence_df, pd.concat(prediction_frames, ignore_index=True)


def _train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    learning_rate: float,
    local_epochs: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(local_epochs):
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()


def run_independent_experiment(
    config: ExperimentConfig,
    clients: list[RegionClientData],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run independent per-client training without federated aggregation."""

    total_epochs = config.independent_total_epochs or (config.communication_rounds * config.local_epochs)
    input_channels = _resolve_input_channels(config)
    client_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for client in clients:
        model = CNNLSTMAttentionRegressor(
            input_channels=input_channels,
            prediction_horizon=config.prediction_horizon,
        ).to(device)
        _train_local_model(
            model=model,
            train_loader=client.train_loader,
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
    run_config_payload: dict[str, object],
    environment_summary: dict[str, object],
    split_summary: dict[str, object],
    partition_result: RegionPartitionResult,
    fed_client_df: pd.DataFrame,
    ind_client_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> None:
    """Write experiment artifacts."""

    client_metrics_df = pd.concat([fed_client_df, ind_client_df], ignore_index=True)
    main_metrics_df = (
        client_metrics_df.groupby("method", as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values("method")
        .reset_index(drop=True)
    )
    main_summary_df = summarize_metric_frame(client_metrics_df, group_cols=["method"])

    write_json(run_config_payload, output_dir / "run_config.json")
    write_text("python -m real_data_experiments.region_client.rc_core " + " ".join(sys.argv[1:]), output_dir / "run_commands.txt")
    write_json(environment_summary, output_dir / "environment_summary.json")
    write_json(split_summary, output_dir / "split_summary.json")
    write_csv(partition_result.assignment_df, output_dir / "region_assignment.csv")
    write_csv(partition_result.client_distribution_summary_df, output_dir / "client_distribution_summary.csv")
    write_csv(partition_result.non_iid_summary_df, output_dir / "non_iid_summary.csv")
    write_csv(main_metrics_df, output_dir / "main_metrics.csv")
    write_csv(main_summary_df, output_dir / "main_summary.csv")
    write_csv(client_metrics_df, output_dir / "client_metrics.csv")
    write_csv(convergence_df, output_dir / "convergence_history.csv")
    write_csv(prediction_df.head(config.prediction_sample_limit), output_dir / "prediction_samples.csv")
    write_text(
        "\n".join(
            [
                "# 区域客户端实验说明",
                "",
                "- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。",
                "- 当前 `region client` 表示一组 pooled grid regions，而不是单个原始路口节点。",
                "- 当前默认划分方法为 `spatial_block`；`flow_kmeans` 仅为可选区域划分方法。",
                "- 当前主线方法始终为标准样本量加权 FedAvg。",
                "- 当前保留 Independent baseline 作为对比方法。",
                "- 数据划分按 target time 连续执行，不进行随机切分。",
            ]
        ),
        output_dir / "experiment_notes_zh.md",
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run the regional-client experiment."""

    import time
    t0 = time.time()
    output_dir = prepare_output_dir(config.output_dir)
    print(f"[stage] prepare_output_dir took {time.time()-t0:.1f}s", flush=True)
    
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

    t1 = time.time()
    print(f"[stage] build_region_client_data starting...", flush=True)
    clients, split_summary, partition_result = build_region_client_data(config)
    print(f"[stage] build_region_client_data took {time.time()-t1:.1f}s, num_clients={len(clients)}", flush=True)
    fed_client_df, convergence_df, fed_prediction_df = run_fedavg_experiment(config, clients, device)
    ind_client_df, ind_prediction_df = run_independent_experiment(config, clients, device)
    prediction_df = pd.concat([fed_prediction_df, ind_prediction_df], ignore_index=True)

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
        partition_result=partition_result,
        fed_client_df=fed_client_df,
        ind_client_df=ind_client_df,
        convergence_df=convergence_df,
        prediction_df=prediction_df,
    )
    return {
        "output_dir": str(output_dir),
        "num_clients": int(len(clients)),
        "used_region_count": int(len(partition_result.assignment_df)),
        "partition_method": partition_result.resolved_partition_method,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_experiment(config)
    print(f"[region_client] completed -> {result['output_dir']}")
    print(f"[partition_method] {result['partition_method']}")
    print(f"[num_clients] {result['num_clients']}")
    print(f"[used_region_count] {result['used_region_count']}")


if __name__ == "__main__":
    main()
