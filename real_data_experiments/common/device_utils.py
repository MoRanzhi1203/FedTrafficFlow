from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceResolution:
    requested_device: str
    actual_device: str
    cuda_available: bool
    cuda_device_name: Optional[str]
    fallback_reason: Optional[str]


def resolve_device(requested_device: str = "cuda") -> DeviceResolution:
    """Resolve requested device for experiment runtime.

    Rules:
    - "cuda", "gpu", and "auto" prefer CUDA when available.
    - "cpu" always uses CPU.
    - If CUDA is requested but unavailable, fallback to CPU.
    - Return metadata for run_config.json.
    """

    raw_requested = requested_device or "cuda"
    normalized = raw_requested.strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized not in {"cuda", "cpu", "auto"}:
        raise ValueError(
            f"Unsupported device: {requested_device!r}. "
            "Use one of: cuda, gpu, cpu, auto."
        )

    try:
        import torch
    except Exception as exc:
        fallback_reason = f"torch import failed: {exc}"
        return DeviceResolution(
            requested_device=raw_requested,
            actual_device="cpu",
            cuda_available=False,
            cuda_device_name=None,
            fallback_reason=None if normalized == "cpu" else fallback_reason,
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None

    if normalized == "cpu":
        return DeviceResolution(
            requested_device=raw_requested,
            actual_device="cpu",
            cuda_available=cuda_available,
            cuda_device_name=cuda_device_name,
            fallback_reason=None,
        )

    if normalized in {"cuda", "auto"} and cuda_available:
        return DeviceResolution(
            requested_device=raw_requested,
            actual_device="cuda",
            cuda_available=True,
            cuda_device_name=cuda_device_name,
            fallback_reason=None,
        )

    return DeviceResolution(
        requested_device=raw_requested,
        actual_device="cpu",
        cuda_available=False,
        cuda_device_name=None,
        fallback_reason="CUDA requested but torch.cuda.is_available() is False",
    )
