"""Tensor-only dataset helpers for pooled-grid traffic-flow experiments."""

from __future__ import annotations

from real_data_experiments.common.data_splits import validate_split_ratios

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from .io_utils import resolve_path


@dataclass
class GridTensorBundle:
    """Loaded tensor-only input bundle."""

    tensor: torch.Tensor
    regions_df: pd.DataFrame
    tensor_metadata: dict[str, Any]
    grid_metadata: dict[str, Any]


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    """Normalize CSV boolean-like values to actual bool."""
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _default_metadata_path(base_path: Path, file_name: str) -> Path:
    """Resolve a sibling metadata file for a tensor artifact."""
    return base_path.with_name(file_name)


def load_grid_tensor_bundle(
    tensor_path: str | Path,
    regions_path: str | Path,
    tensor_metadata_path: str | Path | None = None,
    grid_metadata_path: str | Path | None = None,
) -> GridTensorBundle:
    """Load and validate the formal pooled-grid tensor input bundle."""
    resolved_tensor_path = resolve_path(tensor_path)
    resolved_regions_path = resolve_path(regions_path)
    resolved_tensor_metadata_path = (
        resolve_path(tensor_metadata_path)
        if tensor_metadata_path is not None
        else _default_metadata_path(resolved_tensor_path, "node_flow_grid_tensor_metadata.json")
    )
    resolved_grid_metadata_path = (
        resolve_path(grid_metadata_path)
        if grid_metadata_path is not None
        else _default_metadata_path(resolved_tensor_path, "node_flow_grid_metadata.json")
    )

    if not resolved_tensor_path.exists():
        raise FileNotFoundError(f"Tensor path does not exist: {resolved_tensor_path}")
    if not resolved_regions_path.exists():
        raise FileNotFoundError(f"Regions path does not exist: {resolved_regions_path}")
    if not resolved_tensor_metadata_path.exists():
        raise FileNotFoundError(f"Tensor metadata path does not exist: {resolved_tensor_metadata_path}")
    if not resolved_grid_metadata_path.exists():
        raise FileNotFoundError(f"Grid metadata path does not exist: {resolved_grid_metadata_path}")

    tensor = torch.load(resolved_tensor_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Loaded object is not a torch.Tensor: {type(tensor)!r}")
    tensor = tensor.detach().clone().to(dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Grid tensor must have rank 3 (C,R,T). Got shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Grid tensor contains NaN or Inf values.")
    if tensor.shape[0] < 1 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
        raise ValueError(f"Invalid tensor shape: {tuple(tensor.shape)}")

    regions_df = pd.read_csv(resolved_regions_path)
    if "region_id" not in regions_df.columns:
        raise ValueError(f"`region_id` column is required in regions file: {resolved_regions_path}")

    tensor_metadata = json.loads(resolved_tensor_metadata_path.read_text(encoding="utf-8"))
    grid_metadata = json.loads(resolved_grid_metadata_path.read_text(encoding="utf-8"))

    expected_tensor_shape = tuple(int(value) for value in tensor_metadata.get("tensor_shape", []))
    if expected_tensor_shape and tuple(tensor.shape) != expected_tensor_shape:
        raise ValueError(
            "Tensor shape mismatch between file and metadata: "
            f"{tuple(tensor.shape)} vs {expected_tensor_shape}"
        )
    expected_region_count = int(tensor_metadata.get("region_count", tensor.shape[1]))
    if len(regions_df) != expected_region_count:
        raise ValueError(
            f"Region sidecar row count {len(regions_df)} does not match metadata region_count {expected_region_count}"
        )

    if "is_active_region" in regions_df.columns:
        regions_df["is_active_region"] = _normalize_bool_series(regions_df["is_active_region"])
    elif "source_node_count" in regions_df.columns:
        regions_df["is_active_region"] = regions_df["source_node_count"].fillna(0).astype(float) > 0
    else:
        raise ValueError("Region sidecar must include `is_active_region` or `source_node_count`.")

    regions_df["region_id"] = regions_df["region_id"].astype(int)
    if "source_node_count" in regions_df.columns:
        regions_df["source_node_count"] = pd.to_numeric(regions_df["source_node_count"], errors="coerce").fillna(0).astype(int)

    return GridTensorBundle(
        tensor=tensor,
        regions_df=regions_df.sort_values("region_id").reset_index(drop=True),
        tensor_metadata=tensor_metadata,
        grid_metadata=grid_metadata,
    )


def get_region_usage_summary(regions_df: pd.DataFrame) -> dict[str, int]:
    """Return total/active region counts from the sidecar."""
    total_region_count = int(len(regions_df))
    active_region_count = int(regions_df["is_active_region"].sum())
    return {
        "total_region_count": total_region_count,
        "active_region_count": active_region_count,
        "empty_region_count": int(total_region_count - active_region_count),
    }


def select_region_clients(
    bundle: GridTensorBundle,
    num_clients: int,
    selected_region_ids: Iterable[int] | None = None,
    target_channel: int = 0,
    use_active_regions_only: bool = True,
) -> pd.DataFrame:
    """Select deterministic pooled-grid-region clients from the formal tensor."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    candidate_df = bundle.regions_df.copy()
    if use_active_regions_only:
        candidate_df = candidate_df[candidate_df["is_active_region"]].copy()
    if candidate_df.empty:
        raise ValueError("No candidate regions remain after applying active-region filtering.")

    mean_total_flow = bundle.tensor[target_channel].mean(dim=1).cpu().numpy()
    candidate_df["mean_total_flow"] = candidate_df["region_id"].map(lambda region_id: float(mean_total_flow[int(region_id)]))

    if selected_region_ids is not None:
        ordered_region_ids = [int(region_id) for region_id in selected_region_ids]
        selected_df = candidate_df[candidate_df["region_id"].isin(ordered_region_ids)].copy()
        if len(selected_df) != len(ordered_region_ids):
            available_ids = set(candidate_df["region_id"].tolist())
            missing_ids = [region_id for region_id in ordered_region_ids if region_id not in available_ids]
            raise ValueError(f"Requested region ids are not available under current filters: {missing_ids}")
        selected_df["selection_order"] = selected_df["region_id"].map({region_id: index for index, region_id in enumerate(ordered_region_ids)})
        selected_df = selected_df.sort_values("selection_order").drop(columns=["selection_order"])
    else:
        selected_df = (
            candidate_df.sort_values(["mean_total_flow", "region_id"], ascending=[False, True])
            .head(num_clients)
            .copy()
        )

    if len(selected_df) < num_clients and selected_region_ids is None:
        raise ValueError(
            f"Requested {num_clients} clients but only {len(selected_df)} candidate regions are available."
        )

    selected_df = selected_df.head(num_clients).reset_index(drop=True)
    selected_df.insert(0, "client_id", list(range(len(selected_df))))
    preferred_columns = [
        "client_id",
        "region_id",
        "pooled_row",
        "pooled_col",
        "centroid_lon",
        "centroid_lat",
        "source_node_count",
        "mean_total_flow",
        "is_active_region",
    ]
    available_columns = [column for column in preferred_columns if column in selected_df.columns]
    return selected_df[available_columns].copy()


def build_time_split_bounds(time_count: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, int | float]:
    """Build contiguous time-index split bounds on the raw tensor timeline."""
    if time_count <= 0:
        raise ValueError("time_count must be positive.")
    split_name = validate_split_ratios(train_ratio, val_ratio)

    train_end = int(time_count * train_ratio)
    val_end = int(time_count * (train_ratio + val_ratio))
    train_end = min(max(train_end, 1), time_count - 2)
    val_end = min(max(val_end, train_end + 1), time_count - 1)
    return {
        "train_start": 0,
        "train_end": int(train_end),
        "val_start": int(train_end),
        "val_end": int(val_end),
        "test_start": int(val_end),
        "test_end": int(time_count),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "time_count": int(time_count),
        "split_strategy": "temporal_contiguous_by_target_time",
        "split_name": split_name,
    }


class GridTensorWindowDataset(Dataset):
    """Sliding-window dataset over one pooled-grid region from a formal tensor input."""

    def __init__(
        self,
        tensor: torch.Tensor,
        region_id: int,
        input_length: int = 12,
        horizon: int = 1,
        target_channel: int = 0,
        use_channels: Iterable[int] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor shape (C,R,T), got {tuple(tensor.shape)}")
        if input_length <= 0 or horizon <= 0:
            raise ValueError("input_length and horizon must be positive.")

        self.tensor = tensor.detach().clone().to(dtype=torch.float32)
        self.region_id = int(region_id)
        self.input_length = int(input_length)
        self.horizon = int(horizon)
        self.target_channel = int(target_channel)
        self.use_channels = list(use_channels) if use_channels is not None else list(range(int(tensor.shape[0])))

        if self.region_id < 0 or self.region_id >= int(tensor.shape[1]):
            raise IndexError(f"region_id {self.region_id} is out of bounds for region axis size {tensor.shape[1]}")
        if any(channel < 0 or channel >= int(tensor.shape[0]) for channel in self.use_channels):
            raise IndexError(f"use_channels contains invalid channel indices: {self.use_channels}")
        if self.target_channel < 0 or self.target_channel >= int(tensor.shape[0]):
            raise IndexError(f"target_channel {self.target_channel} is out of bounds for channel axis size {tensor.shape[0]}")

        self.time_count = int(tensor.shape[2])
        self.start_time = 0 if start_time is None else int(start_time)
        self.end_time = self.time_count if end_time is None else int(end_time)
        if self.start_time < 0 or self.end_time > self.time_count or self.start_time >= self.end_time:
            raise ValueError(f"Invalid time bounds: start={self.start_time}, end={self.end_time}, total={self.time_count}")

        target_offset = self.input_length + self.horizon - 1
        min_start = max(0, self.start_time - target_offset)
        max_start = min(self.time_count - target_offset - 1, self.end_time - 1 - target_offset)
        if max_start < min_start:
            raise ValueError(
                "No valid windows for the requested split bounds: "
                f"region_id={self.region_id}, start_time={self.start_time}, end_time={self.end_time}, "
                f"input_length={self.input_length}, horizon={self.horizon}"
            )
        self.window_starts = list(range(min_start, max_start + 1))

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.window_starts[index]
        end = start + self.input_length
        target_time = end + self.horizon - 1
        features = self.tensor[self.use_channels, self.region_id, start:end]
        target = self.tensor[self.target_channel, self.region_id, target_time]
        return features.to(dtype=torch.float32), torch.tensor([float(target.item())], dtype=torch.float32)

    def describe(self) -> dict[str, Any]:
        """Return split-specific metadata for auditing."""
        first_start = int(self.window_starts[0])
        last_start = int(self.window_starts[-1])
        target_offset = self.input_length + self.horizon - 1
        return {
            "region_id": self.region_id,
            "input_length": self.input_length,
            "horizon": self.horizon,
            "target_channel": self.target_channel,
            "use_channels": list(self.use_channels),
            "sample_count": len(self),
            "window_start_min": first_start,
            "window_start_max": last_start,
            "target_time_min": int(first_start + target_offset),
            "target_time_max": int(last_start + target_offset),
            "split_start_time": self.start_time,
            "split_end_time": self.end_time,
        }
