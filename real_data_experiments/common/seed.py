"""Seed and environment utilities for real-data experiments."""

from __future__ import annotations

import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_environment_summary(device: str) -> dict[str, Any]:
    """Collect a lightweight environment summary for result auditing."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def resolve_default_device(requested_device: str | None = None) -> str:
    """Resolve a usable device string from CLI input."""
    if requested_device and requested_device != "auto":
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_parent_dir(path: Path) -> Path:
    """Ensure the parent directory of a file path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
