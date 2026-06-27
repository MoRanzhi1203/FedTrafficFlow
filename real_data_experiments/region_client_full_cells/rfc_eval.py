"""Read-only evaluation helpers for region-full-cells reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_main_metrics(result_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(result_dir) / "main_metrics.csv")


def load_client_metrics(result_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(result_dir) / "client_metrics.csv")


def load_run_config(result_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(result_dir) / "run_config.json")


def load_split_summary(result_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(result_dir) / "split_summary.json")

