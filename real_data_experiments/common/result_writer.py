"""Result writing helpers for reproducible experiment outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import ensure_dir, resolve_path


def _normalize_payload(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_payload(value) for value in payload]
    return payload


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Ensure an experiment output directory exists."""
    return ensure_dir(output_dir)


def write_json(payload: Any, output_path: str | Path) -> Path:
    """Write JSON with UTF-8 encoding."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_payload(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_text(text: str, output_path: str | Path) -> Path:
    """Write a text artifact."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a CSV artifact."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path
