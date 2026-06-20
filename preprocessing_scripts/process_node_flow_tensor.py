from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "processed" / "node_flow_grid"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "node_flow_grid"


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    pooled_input: Path
    output_path: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="将池化后的路口车流量网格结果整理为 PyTorch 张量。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="输入目录，默认使用 data/processed/node_flow_grid。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录，默认使用 data/processed/node_flow_grid。",
    )
    parser.add_argument("--pooled-input", type=Path, default=None, help="池化结果 .npy 路径。")
    parser.add_argument("--output-path", type=Path, default=None, help="输出 .pt 路径。")
    parser.add_argument("--verbose", action="store_true", help="输出更详细的日志。")
    args = parser.parse_args()

    configure_logging(args.verbose)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    return Config(
        input_dir=input_dir,
        output_dir=output_dir,
        pooled_input=(args.pooled_input or (input_dir / "node_flow_grid_pooled.npy")).resolve(),
        output_path=(args.output_path or (output_dir / "node_flow_grid_tensor.pt")).resolve(),
    )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_data(config: Config) -> np.ndarray:
    if not config.pooled_input.exists():
        raise FileNotFoundError(f"未找到池化输入文件: {config.pooled_input}")

    pooled_data = np.load(config.pooled_input, allow_pickle=True)
    if len(pooled_data) == 0:
        raise ValueError("池化输入文件为空，无法继续处理。")

    logging.info("已加载池化输入，共 %s 个时间步。", len(pooled_data))
    return pooled_data


def process(pooled_data: Sequence[object]) -> list[dict[str, np.ndarray | int]]:
    normalized_entries: list[dict[str, np.ndarray | int]] = []

    for position, raw_entry in enumerate(pooled_data):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"第 {position} 个条目不是字典结构。")
        if "pooled_grid_tensor" not in raw_entry:
            raise KeyError(f"第 {position} 个条目缺少 pooled_grid_tensor 字段。")

        tensor_array = np.asarray(raw_entry["pooled_grid_tensor"], dtype=np.float32)
        if tensor_array.ndim != 3:
            raise ValueError(f"第 {position} 个 pooled_grid_tensor 维度应为 3，实际为 {tensor_array.ndim}。")

        normalized_entries.append(
            {
                "index": int(raw_entry.get("index", position)),
                "pooled_grid_tensor": tensor_array,
            }
        )

    normalized_entries.sort(key=lambda item: int(item["index"]))
    logging.info("池化条目清洗完成，已按 index 排序。")
    return normalized_entries


def generate_output(normalized_entries: Sequence[dict[str, np.ndarray | int]]) -> torch.Tensor:
    tensor_list = [np.asarray(entry["pooled_grid_tensor"], dtype=np.float32) for entry in normalized_entries]
    first_shape = tensor_list[0].shape

    for idx, tensor_array in enumerate(tensor_list[1:], start=1):
        if tensor_array.shape != first_shape:
            raise ValueError(f"第 {idx} 个张量形状为 {tensor_array.shape}，与首个张量 {first_shape} 不一致。")

    stacked = np.stack(tensor_list, axis=0)
    time_steps, channels, height, width = stacked.shape

    # 输出形状为 (通道数, 展平空间节点数, 时间步数)。
    channel_major = np.transpose(stacked, (1, 2, 3, 0)).reshape(channels, height * width, time_steps)
    output_tensor = torch.tensor(channel_major, dtype=torch.float32)

    logging.info(
        "张量生成完成，原始堆叠形状 %s，输出张量形状 %s。",
        stacked.shape,
        tuple(output_tensor.shape),
    )
    return output_tensor


def save_result(output_tensor: torch.Tensor, config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output_tensor, config.output_path)
    logging.info("PyTorch 张量已保存至: %s", config.output_path)


def main() -> None:
    config = parse_args()
    pooled_data = load_data(config)
    normalized_entries = process(pooled_data)
    output_tensor = generate_output(normalized_entries)
    save_result(output_tensor, config)


if __name__ == "__main__":
    main()
