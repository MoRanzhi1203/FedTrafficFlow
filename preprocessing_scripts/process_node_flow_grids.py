from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "node_flow_grid"
DEFAULT_TOPOLOGY_FILE = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"

REQUIRED_NODE_COLUMNS = {"节点ID", "节点经度", "节点纬度"}
REQUIRED_TRAFFIC_COLUMNS = {"节点ID", "时间段", "路口车流量"}
REQUIRED_TOPOLOGY_COLUMNS = {"起始节点ID", "结束节点ID", "start_lat", "start_lon", "end_lat", "end_lon"}


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    topology_file: Path
    raw_output: Path
    pooled_output: Path
    grid_resolution: float = 0.009
    target_grid_size: int = 64
    pool_kernel_size: int = 2
    pool_stride: int = 2
    pool_padding: int = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="将节点车流量 parquet 分片处理为网格双通道张量与池化张量。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="中间 parquet 输入目录，默认使用 data/analysis/node_intersection_flow_parquet。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出数据目录，默认使用 data/processed/node_flow_grid。",
    )
    parser.add_argument(
        "--topology-file",
        type=Path,
        default=DEFAULT_TOPOLOGY_FILE,
        help="用于恢复节点坐标的拓扑 CSV，默认使用 data/processed/rnsd_processed.csv。",
    )
    parser.add_argument("--raw-output", type=Path, default=None, help="原始双通道网格张量输出 .npy 路径。")
    parser.add_argument("--pooled-output", type=Path, default=None, help="池化后网格张量输出 .npy 路径。")
    parser.add_argument("--grid-resolution", type=float, default=0.009, help="网格分辨率。")
    parser.add_argument("--target-grid-size", type=int, default=64, help="自动分辨率回退时的最大网格边长目标。")
    parser.add_argument("--pool-kernel-size", type=int, default=2, help="最大池化窗口大小。")
    parser.add_argument("--pool-stride", type=int, default=2, help="最大池化步幅。")
    parser.add_argument("--pool-padding", type=int, default=0, help="最大池化 padding。")
    parser.add_argument("--verbose", action="store_true", help="输出更详细的日志。")
    args = parser.parse_args()

    configure_logging(args.verbose)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    return Config(
        input_dir=input_dir,
        output_dir=output_dir,
        topology_file=args.topology_file.resolve(),
        raw_output=(args.raw_output or (output_dir / "node_flow_grid_2ch.npy")).resolve(),
        pooled_output=(args.pooled_output or (output_dir / "node_flow_grid_pooled.npy")).resolve(),
        grid_resolution=args.grid_resolution,
        target_grid_size=args.target_grid_size,
        pool_kernel_size=args.pool_kernel_size,
        pool_stride=args.pool_stride,
        pool_padding=args.pool_padding,
    )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def validate_columns(df: pd.DataFrame, required_columns: set[str], file_label: str) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise KeyError(f"{file_label} 缺少必要字段: {missing_text}")


def build_node_coordinates(topology_df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(topology_df, REQUIRED_TOPOLOGY_COLUMNS, "rnsd_processed.csv")

    start_nodes = topology_df.loc[:, ["起始节点ID", "start_lon", "start_lat"]].rename(
        columns={"起始节点ID": "节点ID", "start_lon": "节点经度", "start_lat": "节点纬度"}
    )
    end_nodes = topology_df.loc[:, ["结束节点ID", "end_lon", "end_lat"]].rename(
        columns={"结束节点ID": "节点ID", "end_lon": "节点经度", "end_lat": "节点纬度"}
    )
    nodes_df = pd.concat([start_nodes, end_nodes], axis=0, ignore_index=True)
    validate_columns(nodes_df, REQUIRED_NODE_COLUMNS, "node_coordinates")
    return nodes_df


def load_data(config: Config) -> tuple[pd.DataFrame, list[Path]]:
    if not config.input_dir.exists():
        raise FileNotFoundError(f"未找到中间 parquet 目录: {config.input_dir}")
    if not config.topology_file.exists():
        raise FileNotFoundError(f"未找到拓扑坐标文件: {config.topology_file}")

    topology_df = pd.read_csv(config.topology_file)
    nodes_df = build_node_coordinates(topology_df)

    traffic_files = sorted(config.input_dir.glob("node_flow_chunk_*.parquet"))
    if not traffic_files:
        traffic_files = sorted(config.input_dir.glob("*.parquet"))
    if not traffic_files:
        raise FileNotFoundError(f"{config.input_dir} 下未找到任何 parquet 文件。")

    logging.info("已恢复节点坐标，共 %s 行；待处理 parquet 分块 %s 个。", len(nodes_df), len(traffic_files))
    return nodes_df, traffic_files


def preprocess(nodes_df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, tuple[int, int]]:
    clean_nodes = nodes_df.loc[:, ["节点ID", "节点经度", "节点纬度"]].copy()
    clean_nodes["节点ID"] = pd.to_numeric(clean_nodes["节点ID"], errors="coerce")
    clean_nodes["节点经度"] = pd.to_numeric(clean_nodes["节点经度"], errors="coerce")
    clean_nodes["节点纬度"] = pd.to_numeric(clean_nodes["节点纬度"], errors="coerce")
    clean_nodes = clean_nodes.dropna(subset=["节点ID", "节点经度", "节点纬度"])
    clean_nodes = clean_nodes[(clean_nodes["节点经度"] != 0) | (clean_nodes["节点纬度"] != 0)]
    clean_nodes["节点ID"] = clean_nodes["节点ID"].astype(np.int64)
    clean_nodes = (
        clean_nodes.groupby("节点ID", as_index=False)[["节点经度", "节点纬度"]]
        .mean()
        .sort_values("节点ID")
        .reset_index(drop=True)
    )

    if clean_nodes.empty:
        raise ValueError("节点经纬度数据为空，无法继续网格化。")

    lon_min = clean_nodes["节点经度"].min()
    lon_max = clean_nodes["节点经度"].max()
    lat_min = clean_nodes["节点纬度"].min()
    lat_max = clean_nodes["节点纬度"].max()

    lon_span = float(lon_max - lon_min)
    lat_span = float(lat_max - lat_min)
    auto_resolution = max(
        max(lon_span, lat_span) / max(config.target_grid_size - 1, 1),
        1e-6,
    )
    resolution = float(config.grid_resolution)
    lon_grid_count = int(np.floor(lon_span / resolution)) + 1
    lat_grid_count = int(np.floor(lat_span / resolution)) + 1
    if min(lon_grid_count, lat_grid_count) < max(config.pool_kernel_size, 2):
        logging.warning(
            "当前 grid_resolution=%.6f 生成的网格仅为 %s x %s，无法完成池化；已自动回退为 %.6f。",
            resolution,
            lon_grid_count,
            lat_grid_count,
            auto_resolution,
        )
        resolution = auto_resolution
        lon_grid_count = int(np.floor(lon_span / resolution)) + 1
        lat_grid_count = int(np.floor(lat_span / resolution)) + 1
    grid_shape = (lon_grid_count, lat_grid_count)

    lon_grid = np.floor((clean_nodes["节点经度"] - lon_min) / resolution).astype(np.int32)
    lat_grid = np.floor((clean_nodes["节点纬度"] - lat_min) / resolution).astype(np.int32)
    clean_nodes["经度网格"] = np.clip(lon_grid, 0, lon_grid_count - 1)
    clean_nodes["纬度网格"] = np.clip(lat_grid, 0, lat_grid_count - 1)

    logging.info(
        "网格化完成，分辨率 %.6f，网格大小为 %s x %s。",
        resolution,
        lon_grid_count,
        lat_grid_count,
    )
    return clean_nodes, grid_shape


def clean_traffic_data(traffic_df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    validate_columns(traffic_df, REQUIRED_TRAFFIC_COLUMNS, file_path.name)
    clean_df = traffic_df.loc[:, ["节点ID", "时间段", "路口车流量"]].copy()
    clean_df["节点ID"] = pd.to_numeric(clean_df["节点ID"], errors="coerce")
    clean_df["时间段"] = pd.to_numeric(clean_df["时间段"], errors="coerce")
    clean_df["路口车流量"] = pd.to_numeric(clean_df["路口车流量"], errors="coerce")
    clean_df = clean_df.dropna(subset=["节点ID", "时间段", "路口车流量"])
    clean_df["节点ID"] = clean_df["节点ID"].astype(np.int64)
    clean_df["时间段"] = clean_df["时间段"].astype(np.int64)
    clean_df["路口车流量"] = clean_df["路口车流量"].astype(np.float32)
    return clean_df


def build_features(
    prepared_nodes: pd.DataFrame,
    traffic_files: Sequence[Path],
    grid_shape: tuple[int, int],
) -> list[dict[str, np.ndarray | int]]:
    all_grid_data: list[dict[str, np.ndarray | int]] = []
    counter = 0

    for traffic_file in traffic_files:
        traffic_df = clean_traffic_data(pd.read_parquet(traffic_file), traffic_file)
        merged_df = prepared_nodes.merge(traffic_df, on="节点ID", how="inner")
        if merged_df.empty:
            logging.warning("%s 合并后为空，已跳过。", traffic_file.name)
            continue

        for _, group_df in merged_df.groupby("时间段", sort=True):
            grid_traffic_sum = np.zeros(grid_shape, dtype=np.float32)
            grid_counts = np.zeros(grid_shape, dtype=np.float32)

            lon_idx = group_df["经度网格"].to_numpy(dtype=np.intp, copy=False)
            lat_idx = group_df["纬度网格"].to_numpy(dtype=np.intp, copy=False)
            flow_values = group_df["路口车流量"].to_numpy(dtype=np.float32, copy=False)

            np.add.at(grid_traffic_sum, (lon_idx, lat_idx), flow_values)
            np.add.at(grid_counts, (lon_idx, lat_idx), 1.0)

            grid_traffic_avg = np.divide(
                grid_traffic_sum,
                grid_counts,
                out=np.zeros_like(grid_traffic_sum),
                where=grid_counts > 0,
            )
            grid_tensor = np.stack([grid_traffic_sum, grid_traffic_avg], axis=0).astype(np.float32)
            all_grid_data.append({"index": counter, "grid_tensor": grid_tensor})
            counter += 1

        logging.info("处理完成: %s，累计时间步 %s。", traffic_file.name, counter)

    if not all_grid_data:
        raise ValueError("未生成任何网格张量，请检查输入数据是否可正确合并。")

    return all_grid_data


def generate_output(
    all_grid_data: Sequence[dict[str, np.ndarray | int]],
    config: Config,
) -> list[dict[str, np.ndarray | int]]:
    pooled_data: list[dict[str, np.ndarray | int]] = []

    for entry in all_grid_data:
        grid_tensor = np.asarray(entry["grid_tensor"], dtype=np.float32)
        tensor = torch.from_numpy(grid_tensor).unsqueeze(0)
        pooled_tensor = F.max_pool2d(
            tensor,
            kernel_size=config.pool_kernel_size,
            stride=config.pool_stride,
            padding=config.pool_padding,
        ).squeeze(0)
        pooled_data.append(
            {
                "index": int(entry["index"]),
                "pooled_grid_tensor": pooled_tensor.cpu().numpy().astype(np.float32, copy=False),
            }
        )

    logging.info("池化完成，输出张量数量 %s。", len(pooled_data))
    return pooled_data


def save_result(
    all_grid_data: Sequence[dict[str, np.ndarray | int]],
    pooled_data: Sequence[dict[str, np.ndarray | int]],
    config: Config,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.raw_output, np.array(list(all_grid_data), dtype=object), allow_pickle=True)
    np.save(config.pooled_output, np.array(list(pooled_data), dtype=object), allow_pickle=True)

    logging.info("原始双通道网格张量已保存至: %s", config.raw_output)
    logging.info("池化后的网格张量已保存至: %s", config.pooled_output)


def main() -> None:
    config = parse_args()
    nodes_df, traffic_files = load_data(config)
    prepared_nodes, grid_shape = preprocess(nodes_df, config)
    all_grid_data = build_features(prepared_nodes, traffic_files, grid_shape)
    pooled_data = generate_output(all_grid_data, config)
    save_result(all_grid_data, pooled_data, config)


if __name__ == "__main__":
    main()
