"""I/O helpers for real-data traffic-flow experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str | Path) -> Path:
    """Resolve project-relative paths to absolute paths."""
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_dir(path_text: str | Path) -> Path:
    """Create a directory if needed and return the absolute path."""
    path = resolve_path(path_text)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_node_flow_frame(
    input_dir: str | Path,
    target_col: str = "路口车流量",
    node_ids: Iterable[int] | None = None,
    max_chunks: int | None = None,
) -> pd.DataFrame:
    """Load selected node-level flow records from parquet chunks."""
    root = resolve_path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    chunk_paths = sorted(root.glob("node_flow_chunk_*.parquet"))
    if not chunk_paths:
        raise FileNotFoundError(f"No node_flow_chunk_*.parquet files found in {root}")
    if max_chunks is not None:
        chunk_paths = chunk_paths[:max_chunks]

    columns = ["节点ID", "时间段", target_col]
    selected_nodes = set(int(node_id) for node_id in node_ids) if node_ids is not None else None
    frames: list[pd.DataFrame] = []
    for chunk_path in chunk_paths:
        frame = pd.read_parquet(chunk_path, columns=columns)
        if selected_nodes is not None:
            frame = frame[frame["节点ID"].isin(selected_nodes)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ValueError("No records were loaded for the requested node selection.")

    merged = pd.concat(frames, ignore_index=True)
    merged["节点ID"] = merged["节点ID"].astype("int64")
    merged["时间段"] = merged["时间段"].astype("int64")
    merged[target_col] = pd.to_numeric(merged[target_col], errors="coerce").fillna(0.0)
    return merged.sort_values(["时间段", "节点ID"]).reset_index(drop=True)


def select_top_nodes_by_activity(frame: pd.DataFrame, num_clients: int, target_col: str = "路口车流量") -> list[int]:
    """Select nodes with the highest average target value for deterministic client construction."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    ranking = (
        frame.groupby("节点ID", as_index=False)[target_col]
        .mean()
        .sort_values([target_col, "节点ID"], ascending=[False, True])
    )
    return [int(node_id) for node_id in ranking["节点ID"].head(num_clients).tolist()]
