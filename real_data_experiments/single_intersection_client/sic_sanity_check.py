"""Sanity checks for experiment 1: grid_cell main full."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from real_data_experiments.common.metrics import compute_regression_metrics
from real_data_experiments.common.tensor_dataset import (
    GridTensorWindowDataset,
    build_time_split_bounds,
    load_grid_tensor_bundle,
)
from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.result_writer import write_text
from real_data_experiments.single_intersection_client.sic_config import parse_selected_clients


@dataclass
class SplitArrays:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity check for experiment 1")
    parser.add_argument(
        "--tensor-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt",
    )
    parser.add_argument(
        "--regions-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv",
    )
    parser.add_argument("--selected-clients", type=str, required=True)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--use-channels", type=str, default="0,1")
    parser.add_argument("--near-zero-threshold", type=float, default=1.0)
    parser.add_argument("--prediction-samples", type=str, default="")
    parser.add_argument(
        "--output-report",
        type=str,
        default="real_data_experiments/experiment1_sanity_check_report_zh.md",
    )
    return parser


def parse_use_channels(raw_text: str) -> list[int]:
    return [int(part.strip()) for part in raw_text.split(",") if part.strip()]


def describe_array(values: np.ndarray, near_zero_threshold: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "zero_ratio": float(np.mean(arr == 0.0)),
        "near_zero_ratio": float(np.mean(np.abs(arr) <= near_zero_threshold)),
    }


def format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"- {name}: count={int(stats['count'])}, min={stats['min']:.6f}, max={stats['max']:.6f}, "
        f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"zero_ratio={stats['zero_ratio']:.6f}, near_zero_ratio={stats['near_zero_ratio']:.6f}"
    )


def collect_targets(dataset: GridTensorWindowDataset) -> np.ndarray:
    return np.asarray(
        [float(dataset[index][1].reshape(-1)[0].item()) for index in range(len(dataset))],
        dtype=np.float64,
    )


def build_split_arrays(
    tensor,
    region_id: int,
    sequence_length: int,
    prediction_horizon: int,
    target_channel: int,
    use_channels: list[int],
    time_bounds: dict[str, int | float],
) -> SplitArrays:
    train_dataset = GridTensorWindowDataset(
        tensor=tensor,
        region_id=region_id,
        input_length=sequence_length,
        horizon=prediction_horizon,
        target_channel=target_channel,
        use_channels=use_channels,
        start_time=int(time_bounds["train_start"]),
        end_time=int(time_bounds["train_end"]),
    )
    val_dataset = GridTensorWindowDataset(
        tensor=tensor,
        region_id=region_id,
        input_length=sequence_length,
        horizon=prediction_horizon,
        target_channel=target_channel,
        use_channels=use_channels,
        start_time=int(time_bounds["val_start"]),
        end_time=int(time_bounds["val_end"]),
    )
    test_dataset = GridTensorWindowDataset(
        tensor=tensor,
        region_id=region_id,
        input_length=sequence_length,
        horizon=prediction_horizon,
        target_channel=target_channel,
        use_channels=use_channels,
        start_time=int(time_bounds["test_start"]),
        end_time=int(time_bounds["test_end"]),
    )
    return SplitArrays(
        train=collect_targets(train_dataset),
        val=collect_targets(val_dataset),
        test=collect_targets(test_dataset),
    )


def compute_last_value_baseline(
    tensor,
    region_id: int,
    sequence_length: int,
    prediction_horizon: int,
    target_channel: int,
    use_channels: list[int],
    time_bounds: dict[str, int | float],
) -> dict[str, float]:
    dataset = GridTensorWindowDataset(
        tensor=tensor,
        region_id=region_id,
        input_length=sequence_length,
        horizon=prediction_horizon,
        target_channel=target_channel,
        use_channels=use_channels,
        start_time=int(time_bounds["test_start"]),
        end_time=int(time_bounds["test_end"]),
    )
    target_channel_index = use_channels.index(target_channel)
    y_true: list[float] = []
    y_pred: list[float] = []
    for index in range(len(dataset)):
        features, target = dataset[index]
        y_true.append(float(target.reshape(-1)[0].item()))
        y_pred.append(float(features[target_channel_index, -1].item()))
    return compute_regression_metrics(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64))


def load_existing_method_metrics(prediction_samples_path: Path | None) -> pd.DataFrame:
    if prediction_samples_path is None:
        return pd.DataFrame()
    client_metrics_path = prediction_samples_path.parent / "client_metrics.csv"
    if not client_metrics_path.exists():
        return pd.DataFrame()
    client_metrics_df = pd.read_csv(client_metrics_path)
    metric_columns = ["mse", "rmse", "mae", "mape", "smape", "r2"]
    return (
        client_metrics_df.groupby("method", as_index=False)[metric_columns]
        .mean()
        .sort_values("method")
        .reset_index(drop=True)
    )


def load_prediction_sample_stats(prediction_samples_path: Path | None) -> list[str]:
    if prediction_samples_path is None or not prediction_samples_path.exists():
        return ["- 未提供可读取的 prediction_samples.csv。"]
    prediction_df = pd.read_csv(prediction_samples_path)
    lines = [
        f"- prediction_samples rows={len(prediction_df)}, methods={sorted(prediction_df['method'].unique().tolist())}",
    ]
    for method, group_df in prediction_df.groupby("method", sort=False):
        y_true = group_df["y_true"].to_numpy(dtype=np.float64)
        y_pred = group_df["y_pred"].to_numpy(dtype=np.float64)
        error = y_pred - y_true
        lines.append(format_stats(f"{method} y_true", describe_array(y_true, near_zero_threshold=1.0)))
        lines.append(format_stats(f"{method} y_pred", describe_array(y_pred, near_zero_threshold=1.0)))
        lines.append(format_stats(f"{method} error", describe_array(error, near_zero_threshold=1.0)))
    return lines


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    selected_clients = parse_selected_clients(args.selected_clients)
    if not selected_clients:
        raise ValueError("--selected-clients must not be empty.")
    use_channels = parse_use_channels(args.use_channels)
    if args.target_channel not in use_channels:
        raise ValueError(f"target_channel {args.target_channel} must be included in use_channels {use_channels}.")

    prediction_samples_path = resolve_path(args.prediction_samples) if args.prediction_samples else None
    output_report_path = resolve_path(args.output_report)

    bundle = load_grid_tensor_bundle(args.tensor_path, args.regions_path)
    time_bounds = build_time_split_bounds(
        time_count=int(bundle.tensor.shape[2]),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    lines: list[str] = [
        "# 实验 1 Sanity Check 报告",
        "",
        "## 1. 配置",
        "",
        f"- selected_clients: {selected_clients}",
        f"- tensor_path: `{resolve_path(args.tensor_path)}`",
        f"- regions_path: `{resolve_path(args.regions_path)}`",
        f"- sequence_length: {args.sequence_length}",
        f"- prediction_horizon: {args.prediction_horizon}",
        f"- use_channels: {use_channels}",
        f"- target_channel: {args.target_channel}",
        f"- split_bounds: train=[{time_bounds['train_start']}, {time_bounds['train_end']}), val=[{time_bounds['val_start']}, {time_bounds['val_end']}), test=[{time_bounds['test_start']}, {time_bounds['test_end']})",
        "",
        "## 2. 每个 selected client 的原始序列与标签统计",
        "",
    ]

    naive_rows: list[dict[str, float | int]] = []
    for region_id in selected_clients:
        region_series = bundle.tensor[args.target_channel, int(region_id)].cpu().numpy().astype(np.float64)
        split_arrays = build_split_arrays(
            tensor=bundle.tensor,
            region_id=int(region_id),
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            target_channel=args.target_channel,
            use_channels=use_channels,
            time_bounds=time_bounds,
        )
        lines.append(f"### region_id={region_id}")
        lines.append(format_stats("series", describe_array(region_series, args.near_zero_threshold)))
        lines.append(format_stats("y_train", describe_array(split_arrays.train, args.near_zero_threshold)))
        lines.append(format_stats("y_val", describe_array(split_arrays.val, args.near_zero_threshold)))
        lines.append(format_stats("y_test", describe_array(split_arrays.test, args.near_zero_threshold)))
        lines.append("")

        metrics = compute_last_value_baseline(
            tensor=bundle.tensor,
            region_id=int(region_id),
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            target_channel=args.target_channel,
            use_channels=use_channels,
            time_bounds=time_bounds,
        )
        naive_rows.append({"region_id": int(region_id), **metrics})

    naive_df = pd.DataFrame(naive_rows)
    naive_mean_df = naive_df.drop(columns=["region_id"]).mean(axis=0).to_frame().T
    naive_mean_df.insert(0, "method", "NaiveLastValue")

    lines.extend(
        [
            "## 3. 历史 prediction_samples 审计",
            "",
            *load_prediction_sample_stats(prediction_samples_path),
            "",
            "## 4. 现有方法与 naive baseline 对比",
            "",
        ]
    )

    existing_metrics_df = load_existing_method_metrics(prediction_samples_path)
    if not existing_metrics_df.empty:
        compare_df = pd.concat([existing_metrics_df, naive_mean_df], ignore_index=True)
        lines.append(compare_df.to_markdown(index=False))
    else:
        lines.append("- 未找到可读取的 client_metrics.csv，只输出 naive baseline。")
        lines.append("")
        lines.append(naive_mean_df.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## 5. 结论",
            "",
            "- 如果 `y_true` 为百万级而模型 `y_pred` 接近小常数，则优先排查目标归一化与尺度一致性。",
            "- 如果 naive baseline 明显优于现有方法，则说明当前训练或标签口径存在明显问题，不能直接进入正式重跑。",
        ]
    )

    write_text("\n".join(lines), output_report_path)
    print(f"[sanity_check_report] {output_report_path}")
    if not existing_metrics_df.empty:
        print(compare_df.to_string(index=False))
    else:
        print(naive_mean_df.to_string(index=False))


if __name__ == "__main__":
    main()
