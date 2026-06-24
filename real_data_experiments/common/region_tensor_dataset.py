"""Tensor-only datasets for multi-region client experiments."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


class RegionClientWindowDataset(Dataset):
    """Sliding-window dataset over one client that owns multiple pooled-grid regions."""

    def __init__(
        self,
        tensor: torch.Tensor,
        region_ids: Iterable[int],
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
        self.region_ids = [int(region_id) for region_id in region_ids]
        self.input_length = int(input_length)
        self.horizon = int(horizon)
        self.target_channel = int(target_channel)
        self.use_channels = list(use_channels) if use_channels is not None else list(range(int(tensor.shape[0])))
        self.time_count = int(tensor.shape[2])
        self.start_time = 0 if start_time is None else int(start_time)
        self.end_time = self.time_count if end_time is None else int(end_time)

        if not self.region_ids:
            raise ValueError("region_ids must contain at least one pooled-grid region.")
        if any(region_id < 0 or region_id >= int(tensor.shape[1]) for region_id in self.region_ids):
            raise IndexError(f"region_ids contains out-of-bounds values: {self.region_ids}")
        if any(channel < 0 or channel >= int(tensor.shape[0]) for channel in self.use_channels):
            raise IndexError(f"use_channels contains invalid channel indices: {self.use_channels}")
        if self.target_channel < 0 or self.target_channel >= int(tensor.shape[0]):
            raise IndexError(f"target_channel {self.target_channel} is out of bounds for channel axis size {tensor.shape[0]}")
        if self.start_time < 0 or self.end_time > self.time_count or self.start_time >= self.end_time:
            raise ValueError(f"Invalid time bounds: start={self.start_time}, end={self.end_time}, total={self.time_count}")

        self.target_offset = self.input_length + self.horizon - 1
        self.first_target_time = max(self.start_time, self.target_offset)
        self.last_target_time = min(self.end_time - 1, self.time_count - 1)
        if self.last_target_time < self.first_target_time:
            raise ValueError(
                "No valid windows for the requested split bounds: "
                f"start_time={self.start_time}, end_time={self.end_time}, input_length={self.input_length}, horizon={self.horizon}"
            )
        self.per_region_window_count = int(self.last_target_time - self.first_target_time + 1)
        self.total_sample_count = int(len(self.region_ids) * self.per_region_window_count)

    def __len__(self) -> int:
        return self.total_sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        region_pos = int(index // self.per_region_window_count)
        target_pos = int(index % self.per_region_window_count)
        region_id = self.region_ids[region_pos]
        target_time = self.first_target_time + target_pos
        start = int(target_time - self.target_offset)
        end = int(start + self.input_length)
        features = self.tensor[self.use_channels, region_id, start:end]
        target = self.tensor[self.target_channel, region_id, target_time]
        return features.to(dtype=torch.float32), torch.tensor([float(target.item())], dtype=torch.float32)

    def describe(self) -> dict[str, Any]:
        """Return split-specific metadata for auditing."""

        return {
            "region_count": int(len(self.region_ids)),
            "region_ids": list(self.region_ids),
            "input_length": int(self.input_length),
            "horizon": int(self.horizon),
            "target_channel": int(self.target_channel),
            "use_channels": list(self.use_channels),
            "sample_count": int(len(self)),
            "per_region_window_count": int(self.per_region_window_count),
            "target_time_min": int(self.first_target_time),
            "target_time_max": int(self.last_target_time),
            "split_start_time": int(self.start_time),
            "split_end_time": int(self.end_time),
        }
