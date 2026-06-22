"""Temporal split helpers for real-data traffic-flow experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Subset


def temporal_split_indices(
    total_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    """Return contiguous train/val/test indices without shuffling."""
    if total_size <= 0:
        raise ValueError("total_size must be positive.")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Require train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    n_train = int(total_size * train_ratio)
    n_val = int(total_size * val_ratio)
    n_test = total_size - n_train - n_val

    n_train = max(1, n_train)
    n_val = max(1, n_val) if val_ratio > 0 else 0
    n_test = max(1, total_size - n_train - n_val)

    while n_train + n_val + n_test > total_size:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        else:
            n_test -= 1

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_train + n_val + n_test))
    metadata = {
        "total_size": int(total_size),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "split_strategy": "temporal_contiguous",
    }
    return train_idx, val_idx, test_idx, metadata


def _slice_data(data: Any, indices: list[int]) -> Any:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices].copy()
    if isinstance(data, np.ndarray):
        return data[indices].copy()
    if isinstance(data, tuple):
        return tuple(data[i] for i in indices)
    if isinstance(data, list):
        return [data[i] for i in indices]
    if hasattr(data, "__len__") and hasattr(data, "__getitem__") and not isinstance(data, Sequence):
        return Subset(data, indices)
    if hasattr(data, "__len__") and hasattr(data, "__getitem__"):
        return [data[i] for i in indices]
    raise TypeError(f"Unsupported data type for temporal split: {type(data)!r}")


def temporal_train_val_test_split(
    data: Any,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Split data into contiguous train/val/test subsets and return metadata."""
    train_idx, val_idx, test_idx, metadata = temporal_split_indices(len(data), train_ratio, val_ratio)
    train_data = _slice_data(data, train_idx)
    val_data = _slice_data(data, val_idx)
    test_data = _slice_data(data, test_idx)
    return train_data, val_data, test_data, metadata
