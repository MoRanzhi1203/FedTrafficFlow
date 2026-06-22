"""Build grid-based node-flow tensors from node intersection flow parquet files.

This script is the formal Python replacement for the gridification and pooling
logic previously kept in `test/预处理5.ipynb`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
DEFAULT_TOPOLOGY_PATH = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"
DEFAULT_LINK_GPS_PATH = ROOT_DIR / "data" / "processed" / "link_gps_processed.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "node_flow_grid"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description="Gridify node flow parquet and export pooled tensors.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--topology-path", type=Path, default=DEFAULT_TOPOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-resolution", type=float, default=0.009)
    parser.add_argument("--pool-kernel", type=int, default=2)
    parser.add_argument("--pool-stride", type=int, default=2)
    parser.add_argument("--target-flow-col", type=str, default="路口车流量")
    parser.add_argument("--node-col", type=str, default="节点ID")
    parser.add_argument("--time-col", type=str, default="时间段")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--link-gps-path", type=Path, default=DEFAULT_LINK_GPS_PATH)
    return parser


def ensure_exists(path: Path, path_label: str) -> None:
    """Raise a readable error when an expected path is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path_label} does not exist: {path}")


def list_parquet_files(input_dir: Path, max_chunks: int | None) -> list[Path]:
    """List chunk parquet files in sorted order."""
    ensure_exists(input_dir, "Input directory")
    files = sorted(input_dir.glob("node_flow_chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {input_dir}")
    if max_chunks is not None:
        files = files[:max_chunks]
    return files


def find_direct_coordinate_mapping(columns: Iterable[str]) -> dict[str, str] | None:
    """Detect direct node-id and coordinate columns from topology data."""
    column_set = set(columns)
    candidates = [
        {
            "start_node": "起始节点ID",
            "end_node": "结束节点ID",
            "start_lon": "起始节点经度",
            "start_lat": "起始节点纬度",
            "end_lon": "结束节点经度",
            "end_lat": "结束节点纬度",
        },
        {
            "start_node": "起点节点ID",
            "end_node": "终点节点ID",
            "start_lon": "起点经度",
            "start_lat": "起点纬度",
            "end_lon": "终点经度",
            "end_lat": "终点纬度",
        },
        {
            "start_node": "起始节点ID",
            "end_node": "结束节点ID",
            "start_lon": "start_lon",
            "start_lat": "start_lat",
            "end_lon": "end_lon",
            "end_lat": "end_lat",
        },
        {
            "start_node": "from_node",
            "end_node": "to_node",
            "start_lon": "from_lon",
            "start_lat": "from_lat",
            "end_lon": "to_lon",
            "end_lat": "to_lat",
        },
        {
            "start_node": "source_node",
            "end_node": "target_node",
            "start_lon": "source_lon",
            "start_lat": "source_lat",
            "end_lon": "target_lon",
            "end_lat": "target_lat",
        },
        {
            "start_node": "source",
            "end_node": "target",
            "start_lon": "source_lon",
            "start_lat": "source_lat",
            "end_lon": "target_lon",
            "end_lat": "target_lat",
        },
    ]
    for mapping in candidates:
        if all(field in column_set for field in mapping.values()):
            return mapping
    return None


def are_coordinates_plausible(node_coords: pd.DataFrame) -> bool:
    """Check whether coordinates look like real lon/lat rather than local offsets."""
    if node_coords.empty:
        return False
    lon = node_coords["lon"].astype(float)
    lat = node_coords["lat"].astype(float)
    if lon.abs().max() < 1 or lat.abs().max() < 1:
        return False
    if not lon.between(70, 140).all():
        return False
    if not lat.between(10, 60).all():
        return False
    return True


def build_node_coords_from_direct_columns(topology_df: pd.DataFrame) -> pd.DataFrame:
    """Build node coordinates directly from start/end coordinate columns."""
    mapping = find_direct_coordinate_mapping(topology_df.columns)
    if mapping is None:
        raise ValueError(
            "Unable to detect node coordinate columns in topology file. "
            f"Actual columns: {list(topology_df.columns)}"
        )
    start_df = topology_df[[mapping["start_node"], mapping["start_lon"], mapping["start_lat"]]].rename(
        columns={
            mapping["start_node"]: "node_id",
            mapping["start_lon"]: "lon",
            mapping["start_lat"]: "lat",
        }
    )
    end_df = topology_df[[mapping["end_node"], mapping["end_lon"], mapping["end_lat"]]].rename(
        columns={
            mapping["end_node"]: "node_id",
            mapping["end_lon"]: "lon",
            mapping["end_lat"]: "lat",
        }
    )
    node_coords = pd.concat([start_df, end_df], ignore_index=True)
    node_coords["node_id"] = pd.to_numeric(node_coords["node_id"], errors="coerce")
    node_coords["lon"] = pd.to_numeric(node_coords["lon"], errors="coerce")
    node_coords["lat"] = pd.to_numeric(node_coords["lat"], errors="coerce")
    node_coords = node_coords.dropna(subset=["node_id", "lon", "lat"])
    node_coords["node_id"] = node_coords["node_id"].astype(np.int64)
    node_coords = node_coords.groupby("node_id", as_index=False)[["lon", "lat"]].mean()
    return node_coords


def build_node_coords_from_link_centers(topology_df: pd.DataFrame, link_gps_path: Path) -> pd.DataFrame:
    """Rebuild node coordinates from link center lon/lat plus direction and length."""
    ensure_exists(link_gps_path, "Link GPS path")
    link_gps = pd.read_csv(link_gps_path)
    if "路段ID" not in topology_df.columns:
        raise ValueError(f"Topology file missing '路段ID'. Actual columns: {list(topology_df.columns)}")
    if not {"路段ID", "经度", "纬度"}.issubset(link_gps.columns):
        raise ValueError(f"link_gps_processed.csv columns are invalid: {list(link_gps.columns)}")

    merged_df = topology_df.merge(link_gps[["路段ID", "经度", "纬度"]], on="路段ID", how="left")
    required_columns = {"起始节点ID", "结束节点ID", "长度", "方向", "经度", "纬度"}
    if not required_columns.issubset(merged_df.columns):
        missing = sorted(required_columns.difference(merged_df.columns))
        raise ValueError(f"Topology + link GPS merge missing columns: {missing}")

    lat = pd.to_numeric(merged_df["纬度"], errors="coerce").to_numpy()
    lon = pd.to_numeric(merged_df["经度"], errors="coerce").to_numpy()
    distance = pd.to_numeric(merged_df["长度"], errors="coerce").to_numpy()
    direction = pd.to_numeric(merged_df["方向"], errors="coerce").to_numpy()
    lat_rad = np.radians(lat)
    half_distance = distance / 2.0

    # Keep the same directional convention as the historical preprocessing script.
    start_lat = np.where(direction == 1, lat + half_distance / 111.32, np.where(direction == 2, lat - half_distance / 111.32, lat))
    start_lon = np.where(
        direction == 3,
        lon + half_distance / (111.32 * np.cos(lat_rad)),
        np.where(direction == 4, lon - half_distance / (111.32 * np.cos(lat_rad)), lon),
    )
    reverse_direction = np.where(direction == 1, 2, np.where(direction == 2, 1, np.where(direction == 3, 4, np.where(direction == 4, 3, direction))))
    end_lat = np.where(reverse_direction == 1, lat + half_distance / 111.32, np.where(reverse_direction == 2, lat - half_distance / 111.32, lat))
    end_lon = np.where(
        reverse_direction == 3,
        lon + half_distance / (111.32 * np.cos(lat_rad)),
        np.where(reverse_direction == 4, lon - half_distance / (111.32 * np.cos(lat_rad)), lon),
    )

    start_df = pd.DataFrame(
        {
            "node_id": pd.to_numeric(merged_df["起始节点ID"], errors="coerce"),
            "lon": start_lon,
            "lat": start_lat,
        }
    )
    end_df = pd.DataFrame(
        {
            "node_id": pd.to_numeric(merged_df["结束节点ID"], errors="coerce"),
            "lon": end_lon,
            "lat": end_lat,
        }
    )
    node_coords = pd.concat([start_df, end_df], ignore_index=True).dropna(subset=["node_id", "lon", "lat"])
    node_coords["node_id"] = node_coords["node_id"].astype(np.int64)
    node_coords = node_coords.groupby("node_id", as_index=False)[["lon", "lat"]].mean()
    return node_coords


def build_node_coordinate_table(topology_path: Path, link_gps_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the final node coordinate table with coordinate-source audit info."""
    ensure_exists(topology_path, "Topology path")
    topology_df = pd.read_csv(topology_path)
    direct_coords = build_node_coords_from_direct_columns(topology_df)
    direct_plausible = are_coordinates_plausible(direct_coords)

    if direct_plausible:
        node_coords = direct_coords
        coordinate_source = "topology_direct"
    else:
        node_coords = build_node_coords_from_link_centers(topology_df, link_gps_path)
        coordinate_source = "topology_plus_link_gps_reconstructed"

    audit = {
        "coordinate_source": coordinate_source,
        "topology_columns": list(topology_df.columns),
        "direct_coordinate_plausible": bool(direct_plausible),
        "node_coordinate_range": {
            "lon_min": float(node_coords["lon"].min()),
            "lon_max": float(node_coords["lon"].max()),
            "lat_min": float(node_coords["lat"].min()),
            "lat_max": float(node_coords["lat"].max()),
        },
        "node_count_with_coordinates": int(node_coords["node_id"].nunique()),
    }
    return node_coords, audit


def infer_grid_indices(node_coords: pd.DataFrame, grid_resolution: float) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Assign each node to a raw spatial grid cell."""
    lon_min = float(node_coords["lon"].min())
    lon_max = float(node_coords["lon"].max())
    lat_min = float(node_coords["lat"].min())
    lat_max = float(node_coords["lat"].max())
    width = max(lon_max - lon_min, grid_resolution)
    height = max(lat_max - lat_min, grid_resolution)
    grid_cols = int(np.floor(width / grid_resolution)) + 1
    grid_rows = int(np.floor(height / grid_resolution)) + 1

    enriched = node_coords.copy()
    enriched["grid_col"] = np.floor((enriched["lon"] - lon_min) / grid_resolution).astype(int).clip(0, grid_cols - 1)
    enriched["grid_row"] = np.floor((enriched["lat"] - lat_min) / grid_resolution).astype(int).clip(0, grid_rows - 1)

    grid_meta = {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
    }
    return enriched, grid_meta


def build_node_count_matrix(node_grid_df: pd.DataFrame, grid_rows: int, grid_cols: int) -> np.ndarray:
    """Build a raw grid matrix of unique node counts per cell."""
    node_count_df = node_grid_df.groupby(["grid_row", "grid_col"], as_index=False)["node_id"].nunique()
    matrix = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    matrix[node_count_df["grid_row"].to_numpy(), node_count_df["grid_col"].to_numpy()] = node_count_df["node_id"].to_numpy(dtype=np.int32)
    return matrix


def build_grids_from_parquet(
    parquet_files: list[Path],
    node_grid_df: pd.DataFrame,
    node_col: str,
    time_col: str,
    target_flow_col: str,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Aggregate chunked node-flow parquet data into dense T,C,H,W arrays."""
    time_matrices: list[np.ndarray] = []
    time_values: list[int] = []
    flow_node_ids: set[int] = set()
    missing_coordinate_node_ids: set[int] = set()
    coord_lookup = node_grid_df.rename(columns={"node_id": node_col})

    for file_index, parquet_path in enumerate(parquet_files, start=1):
        print(f"[gridify] reading chunk {file_index}/{len(parquet_files)}: {parquet_path.name}")
        chunk_df = pd.read_parquet(parquet_path, columns=[node_col, time_col, target_flow_col])
        for required in [node_col, time_col, target_flow_col]:
            if required not in chunk_df.columns:
                raise ValueError(f"{parquet_path.name} missing required column '{required}'. Actual columns: {list(chunk_df.columns)}")

        chunk_df[node_col] = pd.to_numeric(chunk_df[node_col], errors="coerce")
        chunk_df[time_col] = pd.to_numeric(chunk_df[time_col], errors="coerce")
        chunk_df[target_flow_col] = pd.to_numeric(chunk_df[target_flow_col], errors="coerce")
        chunk_df = chunk_df.dropna(subset=[node_col, time_col, target_flow_col])
        chunk_df[node_col] = chunk_df[node_col].astype(np.int64)
        chunk_df[time_col] = chunk_df[time_col].astype(np.int64)
        flow_node_ids.update(chunk_df[node_col].unique().tolist())

        merged_df = chunk_df.merge(coord_lookup[[node_col, "grid_row", "grid_col"]], on=node_col, how="left")
        missing_coordinate_node_ids.update(merged_df.loc[merged_df["grid_row"].isna(), node_col].unique().tolist())
        merged_df = merged_df.dropna(subset=["grid_row", "grid_col"])
        merged_df["grid_row"] = merged_df["grid_row"].astype(int)
        merged_df["grid_col"] = merged_df["grid_col"].astype(int)

        grouped_df = (
            merged_df.groupby([time_col, "grid_row", "grid_col"], as_index=False)[target_flow_col]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "total_flow", "count": "node_count"})
        )

        for current_time in sorted(grouped_df[time_col].unique().tolist()):
            time_group = grouped_df[grouped_df[time_col] == current_time]
            total_flow = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            counts = np.zeros((grid_rows, grid_cols), dtype=np.float32)
            row_idx = time_group["grid_row"].to_numpy(dtype=np.int64)
            col_idx = time_group["grid_col"].to_numpy(dtype=np.int64)
            total_flow[row_idx, col_idx] = time_group["total_flow"].to_numpy(dtype=np.float32)
            counts[row_idx, col_idx] = time_group["node_count"].to_numpy(dtype=np.float32)
            mean_flow = np.divide(total_flow, counts, out=np.zeros_like(total_flow), where=counts > 0)
            time_matrices.append(np.stack([total_flow, mean_flow], axis=0))
            time_values.append(int(current_time))

    if not time_matrices:
        raise ValueError("No grid matrices were created from the provided parquet files.")

    raw_grid = np.stack(time_matrices, axis=0).astype(np.float32)
    raw_grid = raw_grid[np.argsort(np.asarray(time_values))]
    sorted_time_values = sorted(time_values)
    audit = {
        "time_min": int(min(sorted_time_values)),
        "time_max": int(max(sorted_time_values)),
        "time_count": int(len(sorted_time_values)),
        "node_count_with_flow": int(len(flow_node_ids)),
        "missing_coordinate_node_count": int(len(missing_coordinate_node_ids)),
        "missing_coordinate_node_ids_sample": sorted(list(missing_coordinate_node_ids))[:20],
    }
    return raw_grid, audit


def average_pool_raw_grid(raw_grid: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    """Average-pool only the spatial dimensions H/W."""
    raw_tensor = torch.from_numpy(raw_grid)
    pooled_tensor = F.avg_pool2d(raw_tensor, kernel_size=kernel, stride=stride)
    return pooled_tensor.numpy().astype(np.float32)


def build_region_sidecar(
    pooled_shape: tuple[int, int, int, int],
    grid_resolution: float,
    pool_kernel: int,
    pool_stride: int,
    lon_min: float,
    lat_min: float,
    raw_node_count_matrix: np.ndarray,
) -> pd.DataFrame:
    """Build pooled region metadata."""
    _, _, pooled_rows, pooled_cols = pooled_shape
    rows: list[dict[str, float | int]] = []
    region_id = 0
    raw_rows, raw_cols = raw_node_count_matrix.shape

    for pooled_row in range(pooled_rows):
        for pooled_col in range(pooled_cols):
            raw_row_start = pooled_row * pool_stride
            raw_col_start = pooled_col * pool_stride
            raw_row_end = min(raw_row_start + pool_kernel, raw_rows)
            raw_col_end = min(raw_col_start + pool_kernel, raw_cols)
            raw_row_indices = np.arange(raw_row_start, raw_row_end)
            raw_col_indices = np.arange(raw_col_start, raw_col_end)
            grid_row = int(raw_row_indices.mean()) if len(raw_row_indices) else raw_row_start
            grid_col = int(raw_col_indices.mean()) if len(raw_col_indices) else raw_col_start
            centroid_lat = lat_min + (grid_row + 0.5) * grid_resolution
            centroid_lon = lon_min + (grid_col + 0.5) * grid_resolution
            source_node_count = int(raw_node_count_matrix[raw_row_start:raw_row_end, raw_col_start:raw_col_end].sum())
            rows.append(
                {
                    "region_id": region_id,
                    "grid_row": grid_row,
                    "grid_col": grid_col,
                    "pooled_row": pooled_row,
                    "pooled_col": pooled_col,
                    "centroid_lon": centroid_lon,
                    "centroid_lat": centroid_lat,
                    "source_node_count": source_node_count,
                }
            )
            region_id += 1
    return pd.DataFrame(rows)


def count_invalid_values(array: np.ndarray) -> tuple[int, int]:
    """Count NaN and Inf values in a NumPy array."""
    return int(np.isnan(array).sum()), int(np.isinf(array).sum())


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    parquet_files = list_parquet_files(args.input_dir, args.max_chunks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    node_coords, coordinate_audit = build_node_coordinate_table(args.topology_path, args.link_gps_path)
    node_grid_df, grid_meta = infer_grid_indices(node_coords, args.grid_resolution)
    raw_node_count_matrix = build_node_count_matrix(node_grid_df, grid_meta["grid_rows"], grid_meta["grid_cols"])

    raw_grid, flow_audit = build_grids_from_parquet(
        parquet_files=parquet_files,
        node_grid_df=node_grid_df,
        node_col=args.node_col,
        time_col=args.time_col,
        target_flow_col=args.target_flow_col,
        grid_rows=grid_meta["grid_rows"],
        grid_cols=grid_meta["grid_cols"],
    )
    pooled_grid = average_pool_raw_grid(raw_grid, kernel=args.pool_kernel, stride=args.pool_stride)

    raw_nan_count, raw_inf_count = count_invalid_values(raw_grid)
    pooled_nan_count, pooled_inf_count = count_invalid_values(pooled_grid)
    if raw_nan_count or raw_inf_count:
        raise ValueError(f"Raw grid contains invalid values: nan={raw_nan_count}, inf={raw_inf_count}")
    if pooled_nan_count or pooled_inf_count:
        raise ValueError(f"Pooled grid contains invalid values: nan={pooled_nan_count}, inf={pooled_inf_count}")

    regions_df = build_region_sidecar(
        pooled_shape=tuple(pooled_grid.shape),
        grid_resolution=args.grid_resolution,
        pool_kernel=args.pool_kernel,
        pool_stride=args.pool_stride,
        lon_min=grid_meta["lon_min"],
        lat_min=grid_meta["lat_min"],
        raw_node_count_matrix=raw_node_count_matrix,
    )

    raw_path = args.output_dir / "node_flow_grid_2ch.npy"
    pooled_path = args.output_dir / "node_flow_grid_pooled.npy"
    regions_path = args.output_dir / "node_flow_grid_regions.csv"
    metadata_path = args.output_dir / "node_flow_grid_metadata.json"

    np.save(raw_path, raw_grid)
    np.save(pooled_path, pooled_grid)
    regions_df.to_csv(regions_path, index=False)

    metadata = {
        "input_dir": str(args.input_dir),
        "topology_path": str(args.topology_path),
        "link_gps_path": str(args.link_gps_path),
        "output_dir": str(args.output_dir),
        "grid_resolution": args.grid_resolution,
        "pool_kernel": args.pool_kernel,
        "pool_stride": args.pool_stride,
        "target_flow_col": args.target_flow_col,
        "node_col": args.node_col,
        "time_col": args.time_col,
        "raw_shape": list(raw_grid.shape),
        "raw_shape_meaning": ["time", "channel", "grid_row", "grid_col"],
        "pooled_shape": list(pooled_grid.shape),
        "pooled_shape_meaning": ["time", "channel", "pooled_row", "pooled_col"],
        "time_min": flow_audit["time_min"],
        "time_max": flow_audit["time_max"],
        "time_count": flow_audit["time_count"],
        "node_count_with_flow": flow_audit["node_count_with_flow"],
        "node_count_with_coordinates": coordinate_audit["node_count_with_coordinates"],
        "missing_coordinate_node_count": flow_audit["missing_coordinate_node_count"],
        "missing_coordinate_node_ids_sample": flow_audit["missing_coordinate_node_ids_sample"],
        "grid_rows": grid_meta["grid_rows"],
        "grid_cols": grid_meta["grid_cols"],
        "pooled_region_count": int(regions_df.shape[0]),
        "nan_count_raw": raw_nan_count,
        "inf_count_raw": raw_inf_count,
        "nan_count_pooled": pooled_nan_count,
        "inf_count_pooled": pooled_inf_count,
        "is_smoke_test": args.max_chunks is not None,
        "max_chunks": args.max_chunks,
        "coordinate_audit": coordinate_audit,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gridify] raw grid saved to: {raw_path}")
    print(f"[gridify] pooled grid saved to: {pooled_path}")
    print(f"[gridify] region sidecar saved to: {regions_path}")
    print(f"[gridify] metadata saved to: {metadata_path}")
    print(f"[gridify] raw shape={tuple(raw_grid.shape)}, pooled shape={tuple(pooled_grid.shape)}")


if __name__ == "__main__":
    main()
