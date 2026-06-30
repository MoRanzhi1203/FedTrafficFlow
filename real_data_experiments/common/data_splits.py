"""Temporal split helpers for real-data traffic-flow experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Subset


def validate_split_ratios(train_ratio: float, val_ratio: float) -> str:
    """Validate split ratios and return a canonical split_name string.

    Returns:
        split_name: e.g. "chronological_80_10_10"

    Raises:
        ValueError: if ratios are invalid.
    """
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError(
            f"train_ratio and val_ratio must be positive, "
            f"got train_ratio={train_ratio}, val_ratio={val_ratio}"
        )
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, "
            f"got {train_ratio + val_ratio}"
        )
    test_ratio = 1.0 - train_ratio - val_ratio
    split_name = (
        f"chronological_{round(train_ratio * 100)}_"
        f"{round(val_ratio * 100)}_{round(test_ratio * 100)}"
    )
    return split_name


def temporal_split_indices(
    total_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    """Return contiguous train/val/test indices without shuffling."""
    if total_size <= 0:
        raise ValueError("total_size must be positive.")
    split_name = validate_split_ratios(train_ratio, val_ratio)

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
        "split_name": split_name,
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
