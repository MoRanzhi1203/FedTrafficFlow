"""Convert pooled node-flow grids into the formal tensor-only training input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "processed" / "node_flow_grid" / "node_flow_grid_pooled.npy"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "node_flow_grid"
DEFAULT_OUTPUT_NAME = "node_flow_grid_tensor.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description="Convert pooled node-flow grids to a PyTorch tensor.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--regions-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    return parser


def ensure_exists(path: Path, label: str) -> None:
    """Raise a readable error for missing inputs."""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    metadata_path = args.metadata_path if args.metadata_path is not None else args.input_path.with_name("node_flow_grid_metadata.json")
    regions_path = args.regions_path if args.regions_path is not None else args.input_path.with_name("node_flow_grid_regions.csv")
    ensure_exists(args.input_path, "Input pooled grid")
    ensure_exists(metadata_path, "Grid metadata")
    ensure_exists(regions_path, "Region sidecar")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pooled_np = np.load(args.input_path).astype(np.float32, copy=False)
    if pooled_np.ndim != 4:
        raise ValueError(f"Expected pooled grid rank 4, got shape {pooled_np.shape}")

    grid_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    regions_df = pd.read_csv(regions_path)
    time_count, channel_count, pooled_rows, pooled_cols = pooled_np.shape
    region_count = pooled_rows * pooled_cols

    tensor_np = pooled_np.transpose(1, 2, 3, 0).reshape(channel_count, region_count, time_count).astype(np.float32, copy=False)
    nan_count = int(np.isnan(tensor_np).sum())
    inf_count = int(np.isinf(tensor_np).sum())
    if tensor_np.ndim != 3:
        raise ValueError(f"Expected tensor rank 3, got shape {tensor_np.shape}")
    if tensor_np.dtype != np.float32:
        raise ValueError(f"Expected tensor dtype float32, got {tensor_np.dtype}")
    if nan_count or inf_count:
        raise ValueError(f"Tensor contains invalid values: nan={nan_count}, inf={inf_count}")
    if channel_count < 1 or region_count <= 0 or time_count <= 0:
        raise ValueError(
            f"Invalid tensor shape after reshape: channels={channel_count}, regions={region_count}, time={time_count}"
        )
    if len(regions_df) != region_count:
        raise ValueError(
            f"Region sidecar row count {len(regions_df)} does not match pooled region count {region_count}"
        )
    metadata_region_count = grid_metadata.get("region_count")
    if metadata_region_count is not None and int(metadata_region_count) != region_count:
        raise ValueError(f"Metadata region_count {metadata_region_count} does not match inferred region count {region_count}")

    tensor = torch.from_numpy(tensor_np)
    output_path = args.output_dir / args.output_name
    torch.save(tensor, output_path)

    tensor_metadata = {
        "input_path": str(args.input_path),
        "metadata_path": str(metadata_path),
        "regions_path": str(regions_path),
        "output_path": str(output_path),
        "input_shape": list(pooled_np.shape),
        "tensor_shape": list(tensor_np.shape),
        "tensor_shape_meaning": ["channel", "region", "time"],
        "tensor_transform": "pooled_np.transpose(1, 2, 3, 0).reshape(C, H * W, T)",
        "dtype": str(tensor_np.dtype),
        "channel_count": int(channel_count),
        "region_count": int(region_count),
        "time_count": int(time_count),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "grid_metadata_time_count": grid_metadata.get("time_count"),
        "formal_tensor_name": args.output_name,
        "legacy_notebook_name_note": "Historical notebook used `6.池化网格张量.pt` as a temporary name; current engineering code does not generate it and only writes `node_flow_grid_tensor.pt`.",
    }
    metadata_path = args.output_dir / "node_flow_grid_tensor_metadata.json"
    metadata_path.write_text(json.dumps(tensor_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[tensorize] tensor saved to: {output_path}")
    print(f"[tensorize] metadata saved to: {metadata_path}")
    print(f"[tensorize] tensor shape={tuple(tensor.shape)}, dtype={tensor.dtype}, finite={torch.isfinite(tensor).all().item()}")


if __name__ == "__main__":
    main()
