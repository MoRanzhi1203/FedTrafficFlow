"""Dataset and split builders for full-cells region-client experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

from real_data_experiments.common.region_tensor_dataset import RegionClientWindowDataset
from real_data_experiments.common.tensor_dataset import build_time_split_bounds, get_region_usage_summary, load_grid_tensor_bundle
from real_data_experiments.region_client_full_cells.rfc_config import ExperimentConfig


@dataclass
class RFCClientData:
    """Per-client datasets and loaders for full-cells region-client experiments."""

    client_id: int
    entity_id: int
    entity_kind: str
    cell_ids: list[int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split_metadata: dict[str, object]
    client_metadata: dict[str, object] = field(default_factory=dict)
    raw_train_dataset: RegionClientWindowDataset | None = None
    raw_val_dataset: RegionClientWindowDataset | None = None
    raw_test_dataset: RegionClientWindowDataset | None = None


def load_partition_payload(partition_file: str | Path) -> dict[str, Any]:
    path = Path(partition_file)
    return json.loads(path.read_text(encoding="utf-8"))


def _describe_dataset(dataset: RegionClientWindowDataset) -> dict[str, object]:
    return dataset.describe()


def _make_client_distribution_frame(partition_payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in partition_payload["clients"]:
        rows.append(
            {
                "client_id": int(row["client_id"]),
                "cell_count": int(row["cell_count"]),
                "source_node_count_sum": int(row["source_node_count_sum"]),
                "mean_total_flow_mean": float(row["mean_total_flow_mean"]),
                "mean_total_flow_sum": float(row["mean_total_flow_sum"]),
                "flow_cv_mean": float(row["flow_cv_mean"]),
                "lag1_autocorr_mean": float(row["lag1_autocorr_mean"]),
                "internal_mean_pairwise_corr": row["internal_mean_pairwise_corr"],
                "cell_ids": ",".join(str(cell_id) for cell_id in row["cell_ids"]),
            }
        )
    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


def build_full_cells_client_data(
    config: ExperimentConfig,
) -> tuple[list[RFCClientData], dict[str, object], dict[str, object], pd.DataFrame]:
    """Construct client loaders from a pre-generated partition file."""

    bundle = load_grid_tensor_bundle(config.tensor_path, config.regions_path)
    partition_payload = load_partition_payload(config.partition_file)
    time_bounds = build_time_split_bounds(
        time_count=int(bundle.tensor.shape[2]),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    region_usage = get_region_usage_summary(bundle.regions_df)
    distribution_df = _make_client_distribution_frame(partition_payload)

    clients: list[RFCClientData] = []
    split_clients: list[dict[str, object]] = []
    client_membership = {
        "partition_mode": partition_payload["partition_mode"],
        "num_clients": int(partition_payload["num_clients"]),
        "partition_file": str(Path(config.partition_file)),
        "client_membership": partition_payload["cell_membership"],
    }

    for client_row in partition_payload["clients"]:
        client_id = int(client_row["client_id"])
        cell_ids = [int(cell_id) for cell_id in client_row["cell_ids"]]
        raw_train_dataset = RegionClientWindowDataset(
            tensor=bundle.tensor,
            region_ids=cell_ids,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["train_start"]),
            end_time=int(time_bounds["train_end"]),
        )
        raw_val_dataset = RegionClientWindowDataset(
            tensor=bundle.tensor,
            region_ids=cell_ids,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["val_start"]),
            end_time=int(time_bounds["val_end"]),
        )
        raw_test_dataset = RegionClientWindowDataset(
            tensor=bundle.tensor,
            region_ids=cell_ids,
            input_length=config.sequence_length,
            horizon=config.prediction_horizon,
            target_channel=config.target_channel,
            use_channels=config.use_channels,
            start_time=int(time_bounds["test_start"]),
            end_time=int(time_bounds["test_end"]),
        )
        client = RFCClientData(
            client_id=client_id,
            entity_id=client_id,
            entity_kind="region_full_cells_client",
            cell_ids=cell_ids,
            train_loader=DataLoader(raw_train_dataset, batch_size=config.batch_size, shuffle=False),
            val_loader=DataLoader(raw_val_dataset, batch_size=config.batch_size, shuffle=False),
            test_loader=DataLoader(raw_test_dataset, batch_size=config.batch_size, shuffle=False),
            split_metadata={
                "train": _describe_dataset(raw_train_dataset),
                "val": _describe_dataset(raw_val_dataset),
                "test": _describe_dataset(raw_test_dataset),
            },
            client_metadata={
                "client_id": client_id,
                "cell_count": int(client_row["cell_count"]),
                "cell_ids": ",".join(str(cell_id) for cell_id in cell_ids),
                "source_node_count_sum": int(client_row["source_node_count_sum"]),
                "mean_total_flow_mean": float(client_row["mean_total_flow_mean"]),
                "flow_cv_mean": float(client_row["flow_cv_mean"]),
                "lag1_autocorr_mean": float(client_row["lag1_autocorr_mean"]),
                "internal_mean_pairwise_corr": client_row["internal_mean_pairwise_corr"],
            },
            raw_train_dataset=raw_train_dataset,
            raw_val_dataset=raw_val_dataset,
            raw_test_dataset=raw_test_dataset,
        )
        clients.append(client)
        split_clients.append(
            {
                "client_id": client_id,
                "cell_ids": cell_ids,
                "cell_count": len(cell_ids),
                "train": _describe_dataset(raw_train_dataset),
                "val": _describe_dataset(raw_val_dataset),
                "test": _describe_dataset(raw_test_dataset),
            }
        )

    split_summary: dict[str, object] = {
        "tensor_path": str(Path(config.tensor_path)),
        "regions_path": str(Path(config.regions_path)),
        "partition_file": str(Path(config.partition_file)),
        "partition_mode": partition_payload["partition_mode"],
        "cluster_procedure": partition_payload["cluster_procedure"],
        "num_clients": int(partition_payload["num_clients"]),
        "total_region_count": int(region_usage["total_region_count"]),
        "active_region_count": int(region_usage["active_region_count"]),
        "used_region_count": int(partition_payload["valid_cell_count"]),
        "sequence_length": int(config.sequence_length),
        "prediction_horizon": int(config.prediction_horizon),
        "use_channels": list(config.use_channels),
        "target_channel": int(config.target_channel),
        "split_strategy": "temporal_contiguous_by_target_time",
        **time_bounds,
        "clients": split_clients,
    }
    return clients, split_summary, client_membership, distribution_df

