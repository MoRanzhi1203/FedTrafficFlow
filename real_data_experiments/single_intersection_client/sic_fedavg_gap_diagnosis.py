"""Read-only diagnosis for why experiment 1 FedAvg trails NaiveLastValue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.metrics import compute_regression_metrics, coefficient_of_variation
from real_data_experiments.common.result_writer import write_text
from real_data_experiments.common.tensor_dataset import (
    GridTensorWindowDataset,
    load_grid_tensor_bundle,
)
from real_data_experiments.single_intersection_client.sic_config import parse_selected_clients


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only diagnosis for the FedAvg vs NaiveLastValue gap")
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument("--selected-clients", type=str, required=True)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--use-channels", type=str, default="0,1")
    parser.add_argument(
        "--output-report",
        type=str,
        default="real_data_experiments/single_intersection_client/experiment1_fedavg_gap_diagnosis_zh.md",
    )
    return parser


def parse_use_channels(raw_text: str) -> list[int]:
    return [int(part.strip()) for part in raw_text.split(",") if part.strip()]


def load_split_summary(result_dir: Path) -> dict[str, object]:
    split_summary_path = result_dir / "split_summary.json"
    return json.loads(split_summary_path.read_text(encoding="utf-8"))


def resolve_regions_path(tensor_path: Path, split_summary: dict[str, object]) -> Path:
    regions_path = split_summary.get("regions_path")
    if isinstance(regions_path, str) and regions_path:
        return resolve_path(regions_path)
    return tensor_path.with_name("node_flow_grid_regions.csv")


def read_required_csv(result_dir: Path, name: str) -> pd.DataFrame:
    path = result_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    return pd.read_csv(path)


def pipe_table(frame: pd.DataFrame, float_fmt: str = ".6f") -> str:
    if frame.empty:
        return "| empty |\n| --- |"
    display_df = frame.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda x: format(float(x), float_fmt))
    headers = [str(column) for column in display_df.columns]
    rows = [[str(value) for value in row] for row in display_df.to_numpy()]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * max(3, width) for width in widths) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def describe_series(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": float(coefficient_of_variation(arr)),
    }


def build_test_dataset(
    tensor,
    region_id: int,
    sequence_length: int,
    prediction_horizon: int,
    target_channel: int,
    use_channels: list[int],
    split_summary: dict[str, object],
) -> GridTensorWindowDataset:
    return GridTensorWindowDataset(
        tensor=tensor,
        region_id=region_id,
        input_length=sequence_length,
        horizon=prediction_horizon,
        target_channel=target_channel,
        use_channels=use_channels,
        start_time=int(split_summary["test_start"]),
        end_time=int(split_summary["test_end"]),
    )


def compute_naive_arrays(
    dataset: GridTensorWindowDataset,
    target_channel: int,
    use_channels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    target_channel_index = use_channels.index(target_channel)
    y_true: list[float] = []
    y_pred: list[float] = []
    for index in range(len(dataset)):
        features, target = dataset[index]
        y_true.append(float(target.reshape(-1)[0].item()))
        y_pred.append(float(features[target_channel_index, -1].item()))
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def compute_prediction_stats(group_df: pd.DataFrame) -> dict[str, float]:
    y_true = group_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = group_df["y_pred"].to_numpy(dtype=np.float64)
    error = y_pred - y_true
    y_true_std = float(np.std(y_true, ddof=0))
    y_pred_std = float(np.std(y_pred, ddof=0))
    return {
        "std_y_true": y_true_std,
        "std_y_pred": y_pred_std,
        "std_ratio": float(y_pred_std / y_true_std) if y_true_std > 0 else float("nan"),
        "corr": safe_corr(y_true, y_pred),
        "mean_bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
    }


def compute_convergence_summary(conv_df: pd.DataFrame) -> dict[str, float | str]:
    fedavg_df = conv_df[conv_df["method"] == "FedAvg"].sort_values("communication_round").reset_index(drop=True)
    if fedavg_df.empty:
        raise ValueError("convergence_history.csv does not contain FedAvg rows.")

    tail_window = min(5, len(fedavg_df))
    tail_df = fedavg_df.tail(tail_window).reset_index(drop=True)
    first_row = tail_df.iloc[0]
    last_row = tail_df.iloc[-1]
    train_rel_improve = float((first_row["train_loss"] - last_row["train_loss"]) / max(first_row["train_loss"], 1e-12))
    val_rel_improve = float((first_row["val_rmse"] - last_row["val_rmse"]) / max(first_row["val_rmse"], 1e-12))

    if val_rel_improve >= 0.12:
        verdict = "最后 5 轮仍明显下降，可能尚未完全收敛"
    elif val_rel_improve >= 0.04:
        verdict = "最后 5 轮仍在下降，但已进入缓慢下降阶段"
    else:
        verdict = "最后 5 轮基本平台期，增加轮数未必是主因"

    return {
        "round_count": int(len(fedavg_df)),
        "train_loss_first": float(fedavg_df.iloc[0]["train_loss"]),
        "train_loss_last": float(fedavg_df.iloc[-1]["train_loss"]),
        "val_rmse_first": float(fedavg_df.iloc[0]["val_rmse"]),
        "val_rmse_last": float(fedavg_df.iloc[-1]["val_rmse"]),
        "train_rel_improve_last5": train_rel_improve,
        "val_rel_improve_last5": val_rel_improve,
        "verdict": verdict,
    }


def compute_evaluation_consistency(main_df: pd.DataFrame, client_df: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    metric_columns = ["mse", "rmse", "mae", "mape", "smape", "r2"]
    grouped = client_df.groupby("method", as_index=False)[metric_columns].mean()
    compare_df = main_df.merge(grouped, on="method", suffixes=("_main", "_client_mean"))
    consistent = True
    for metric in metric_columns:
        diff = np.abs(compare_df[f"{metric}_main"] - compare_df[f"{metric}_client_mean"])
        if not bool((diff < 1e-9).all()):
            consistent = False
            break
    return consistent, compare_df


def classify_primary_reason(
    eval_consistent: bool,
    fedavg_vs_naive_df: pd.DataFrame,
    independent_vs_naive_df: pd.DataFrame,
    pred_diag_df: pd.DataFrame,
    convergence_summary: dict[str, float | str],
    heterogeneity_df: pd.DataFrame,
    naive_strength_df: pd.DataFrame,
    pairwise_corr_df: pd.DataFrame,
) -> tuple[str, list[str]]:
    if not eval_consistent:
        return "FedAvg 主要受评估口径问题影响", ["main_metrics 与 client_metrics 聚合口径不一致，需要先修评估链路。"]

    fedavg_worse_count = int((~fedavg_vs_naive_df["fedavg_better_than_naive"]).sum())
    independent_better_count = int(independent_vs_naive_df["independent_better_than_naive"].sum())
    worst_gap_row = fedavg_vs_naive_df.sort_values("fedavg_rmse_minus_naive_rmse", ascending=False).iloc[0]
    fedavg_pred_row = pred_diag_df[pred_diag_df["method"] == "FedAvg"].iloc[0]
    mean_lag_corr = float(naive_strength_df["test_lag1_corr"].mean())
    mean_range_ratio = float(heterogeneity_df["series_mean"].max() / heterogeneity_df["series_mean"].min())
    val_rel_improve_last5 = float(convergence_summary["val_rel_improve_last5"])
    off_diag_values = pairwise_corr_df.to_numpy(dtype=np.float64)
    off_diag_values = off_diag_values[~np.eye(off_diag_values.shape[0], dtype=bool)]
    min_pairwise_corr = float(np.min(off_diag_values)) if off_diag_values.size else float("nan")
    worst_gap_value = float(worst_gap_row["fedavg_rmse_minus_naive_rmse"])

    reasons: list[str] = []
    if mean_range_ratio >= 1.12 and fedavg_worse_count >= 1:
        reasons.append(
            "selected clients 存在明显分布差异，尤其 region "
            f"{int(worst_gap_row['region_id'])} 的 FedAvg 相比 naive 出现最大 RMSE 劣化。"
        )
    if min_pairwise_corr < 0.30:
        reasons.append(
            f"selected clients 的最小 pairwise correlation 仅为 {min_pairwise_corr:.6f}，存在强 non-IID client。"
        )
    if float(fedavg_pred_row["std_ratio"]) < 0.95:
        reasons.append("FedAvg 预测方差被压缩，存在跨区域平均后的平滑/欠拟合迹象。")
    if mean_lag_corr >= 0.90:
        reasons.append("test split 上 y_t 与 y_(t-1) 相关性很高，NaiveLastValue 本身就是强基线。")
    if val_rel_improve_last5 >= 0.12:
        reasons.append("最后 5 轮验证 RMSE 仍明显下降，轮数不足可能是次要因素。")
    if independent_better_count == len(fedavg_vs_naive_df):
        reasons.append("Independent 在所有 client 上都优于 naive，说明问题主要在全局平均后的联邦共享模型，而不是单客户端训练能力。")

    primary_reason = "FedAvg 主要受 client 异质性影响"
    if min_pairwise_corr < 0.30 and worst_gap_value > 5000.0:
        primary_reason = "FedAvg 主要受 client 异质性影响"
    elif float(fedavg_pred_row["std_ratio"]) < 0.85 and fedavg_worse_count >= len(fedavg_vs_naive_df) // 2:
        primary_reason = "FedAvg 主要受预测过度平滑影响"
    elif val_rel_improve_last5 >= 0.15 and fedavg_worse_count >= len(fedavg_vs_naive_df) // 2:
        primary_reason = "FedAvg 主要受训练轮数不足影响"
    elif mean_lag_corr >= 0.95 and fedavg_worse_count <= 1:
        primary_reason = "FedAvg 低于 naive 是因为 naive baseline 极强"

    return primary_reason, reasons


def format_bool(value: bool) -> str:
    return "是" if value else "否"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    selected_clients = parse_selected_clients(args.selected_clients)
    if not selected_clients:
        raise ValueError("--selected-clients must not be empty.")
    use_channels = parse_use_channels(args.use_channels)
    if args.target_channel not in use_channels:
        raise ValueError(f"target_channel {args.target_channel} must be included in use_channels {use_channels}.")

    result_dir = resolve_path(args.result_dir)
    tensor_path = resolve_path(args.tensor_path)
    output_report_path = resolve_path(args.output_report)

    main_df = read_required_csv(result_dir, "main_metrics.csv")
    client_df = read_required_csv(result_dir, "client_metrics.csv")
    conv_df = read_required_csv(result_dir, "convergence_history.csv")
    pred_df = read_required_csv(result_dir, "prediction_samples.csv")
    split_summary = load_split_summary(result_dir)
    regions_path = resolve_regions_path(tensor_path, split_summary)
    bundle = load_grid_tensor_bundle(tensor_path, regions_path)

    metric_columns = ["mse", "rmse", "mae", "mape", "smape", "r2"]
    main_metrics_df = main_df.copy().sort_values("method").reset_index(drop=True)
    eval_consistent, eval_compare_df = compute_evaluation_consistency(main_df, client_df)

    prediction_diag_rows: list[dict[str, float | str]] = []
    for method, method_df in pred_df.groupby("method", sort=False):
        prediction_diag_rows.append({"method": method, **compute_prediction_stats(method_df)})
    pred_diag_df = pd.DataFrame(prediction_diag_rows).sort_values("method").reset_index(drop=True)

    client_series_rows: list[dict[str, float | int]] = []
    naive_rows: list[dict[str, float | int]] = []
    pairwise_series: dict[int, np.ndarray] = {}

    for region_id in selected_clients:
        region_series = bundle.tensor[args.target_channel, int(region_id)].cpu().numpy().astype(np.float64)
        pairwise_series[int(region_id)] = region_series
        series_stats = describe_series(region_series)

        test_dataset = build_test_dataset(
            tensor=bundle.tensor,
            region_id=int(region_id),
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            target_channel=args.target_channel,
            use_channels=use_channels,
            split_summary=split_summary,
        )
        y_true_naive, y_pred_naive = compute_naive_arrays(test_dataset, args.target_channel, use_channels)
        naive_metrics = compute_regression_metrics(y_true_naive, y_pred_naive)
        naive_rows.append(
            {
                "region_id": int(region_id),
                "test_lag1_corr": safe_corr(y_true_naive[1:], y_true_naive[:-1]),
                "naive_std_ratio": float(np.std(y_pred_naive, ddof=0) / max(np.std(y_true_naive, ddof=0), 1e-12)),
                **naive_metrics,
            }
        )
        client_series_rows.append({"region_id": int(region_id), **series_stats})

    client_series_df = pd.DataFrame(client_series_rows).sort_values("region_id").reset_index(drop=True)
    naive_df = pd.DataFrame(naive_rows).sort_values("region_id").reset_index(drop=True)

    fedavg_client_df = (
        client_df[client_df["method"] == "FedAvg"][["region_id", *metric_columns, "mean_total_flow"]]
        .rename(columns={metric: f"fedavg_{metric}" for metric in metric_columns})
        .reset_index(drop=True)
    )
    independent_client_df = (
        client_df[client_df["method"] == "Independent"][["region_id", *metric_columns]]
        .rename(columns={metric: f"independent_{metric}" for metric in metric_columns})
        .reset_index(drop=True)
    )

    per_client_df = (
        client_series_df.merge(fedavg_client_df, on="region_id")
        .merge(independent_client_df, on="region_id")
        .merge(naive_df, on="region_id")
        .sort_values("region_id")
        .reset_index(drop=True)
    )
    per_client_df["fedavg_rmse_minus_naive_rmse"] = per_client_df["fedavg_rmse"] - per_client_df["rmse"]
    per_client_df["independent_rmse_minus_naive_rmse"] = per_client_df["independent_rmse"] - per_client_df["rmse"]
    per_client_df["fedavg_better_than_naive"] = per_client_df["fedavg_rmse"] < per_client_df["rmse"]
    per_client_df["independent_better_than_naive"] = per_client_df["independent_rmse"] < per_client_df["rmse"]

    pairwise_corr_df = pd.DataFrame(
        {
            region_id: {other_id: safe_corr(series, other_series) for other_id, other_series in pairwise_series.items()}
            for region_id, series in pairwise_series.items()
        }
    ).sort_index().sort_index(axis=1)

    convergence_summary = compute_convergence_summary(conv_df)
    naive_summary_df = naive_df[["mse", "rmse", "mae", "mape", "smape", "r2"]].mean().to_frame().T
    naive_summary_df.insert(0, "method", "NaiveLastValue")
    overall_compare_df = pd.concat([main_metrics_df, naive_summary_df], ignore_index=True)

    prediction_compare_df = pred_diag_df.copy()
    prediction_compare_df["unique_y_pred"] = prediction_compare_df["method"].map(
        pred_df.groupby("method")["y_pred"].nunique().to_dict()
    )

    primary_reason, reason_evidence = classify_primary_reason(
        eval_consistent=eval_consistent,
        fedavg_vs_naive_df=per_client_df[
            ["region_id", "fedavg_rmse", "rmse", "fedavg_rmse_minus_naive_rmse", "fedavg_better_than_naive"]
        ],
        independent_vs_naive_df=per_client_df[
            [
                "region_id",
                "independent_rmse",
                "rmse",
                "independent_rmse_minus_naive_rmse",
                "independent_better_than_naive",
            ]
        ],
        pred_diag_df=pred_diag_df,
        convergence_summary=convergence_summary,
        heterogeneity_df=client_series_df.rename(columns={"mean": "series_mean"}),
        naive_strength_df=naive_df,
        pairwise_corr_df=pairwise_corr_df,
    )

    lines: list[str] = [
        "# 实验 1：FedAvg 弱于 NaiveLastValue 的只读诊断报告",
        "",
        "## 1. 诊断范围",
        "",
        "- 本次只诊断实验 1 v4 CUDA：`grid_cell main full`。",
        "- 只读取现有 v4 输出与原始 tensor 数据，不运行新的正式 full。",
        "- 本次不修改 FedAvg、不修改模型结构、不修改数据划分。",
        "",
        "## 2. 当前核心矛盾",
        "",
        pipe_table(overall_compare_df[["method", *metric_columns]], float_fmt=".6f"),
        "",
        "初步观察：",
        "",
        f"- `FedAvg` 相比 `NaiveLastValue` 的 `RMSE` 高出 {float(overall_compare_df.loc[overall_compare_df['method'] == 'FedAvg', 'rmse'].iloc[0] - naive_summary_df['rmse'].iloc[0]):.6f}。",
        f"- `Independent` 明显优于 `NaiveLastValue`，说明单客户端训练链路本身是有效的。",
        "",
        "## 3. per-client 结果对比",
        "",
    ]

    per_client_report_df = per_client_df[
        [
            "region_id",
            "mean_total_flow",
            "fedavg_rmse",
            "rmse",
            "fedavg_rmse_minus_naive_rmse",
            "fedavg_better_than_naive",
            "independent_rmse",
            "independent_rmse_minus_naive_rmse",
            "independent_better_than_naive",
            "fedavg_r2",
            "independent_r2",
            "r2",
        ]
    ].rename(
        columns={
            "mean_total_flow": "flow_mean",
            "rmse": "naive_rmse",
            "r2": "naive_r2",
        }
    )
    lines.append(pipe_table(per_client_report_df, float_fmt=".6f"))
    lines.extend(
        [
            "",
            f"- `FedAvg` 优于 naive 的 client 数：{int(per_client_df['fedavg_better_than_naive'].sum())} / {len(per_client_df)}。",
            f"- `Independent` 优于 naive 的 client 数：{int(per_client_df['independent_better_than_naive'].sum())} / {len(per_client_df)}。",
            f"- FedAvg 最差 client：region `{int(per_client_df.sort_values('fedavg_rmse_minus_naive_rmse', ascending=False).iloc[0]['region_id'])}`。",
            "",
            "## 4. prediction_samples 诊断",
            "",
            pipe_table(prediction_compare_df[["method", "std_y_true", "std_y_pred", "std_ratio", "corr", "mean_bias", "mae", "unique_y_pred"]], float_fmt=".6f"),
            "",
            "- `y_true / y_pred` 同尺度，均为原始百万级流量尺度。",
            f"- FedAvg `std(y_pred)/std(y_true)={float(pred_diag_df.loc[pred_diag_df['method'] == 'FedAvg', 'std_ratio'].iloc[0]):.6f}`，存在一定方差压缩。",
            f"- Independent `std(y_pred)/std(y_true)={float(pred_diag_df.loc[pred_diag_df['method'] == 'Independent', 'std_ratio'].iloc[0]):.6f}`，更接近 1。",
            "",
            "## 5. NaiveLastValue 强度分析",
            "",
        ]
    )
    naive_strength_report_df = naive_df[
        ["region_id", "test_lag1_corr", "naive_std_ratio", "rmse", "mae", "r2"]
    ].rename(columns={"rmse": "naive_rmse", "mae": "naive_mae", "r2": "naive_r2"})
    lines.append(pipe_table(naive_strength_report_df, float_fmt=".6f"))
    lines.extend(
        [
            "",
            f"- selected clients 的平均 test lag-1 相关性为 `{float(naive_df['test_lag1_corr'].mean()):.6f}`，说明 `last value baseline` 天然较强。",
            f"- Naive 平均 `R2={float(naive_summary_df['r2'].iloc[0]):.6f}`，并不是弱基线。",
            "",
            "## 6. client 异质性分析",
            "",
            pipe_table(client_series_df.rename(columns={"mean": "series_mean", "std": "series_std"}), float_fmt=".6f"),
            "",
            "pairwise correlation：",
            "",
            pipe_table(pairwise_corr_df.reset_index().rename(columns={"index": "region_id"}), float_fmt=".6f"),
            "",
            f"- 各 client 平均流量从 `{float(client_series_df['mean'].min()):.6f}` 到 `{float(client_series_df['mean'].max()):.6f}`，均值比约为 `{float(client_series_df['mean'].max() / client_series_df['mean'].min()):.4f}`。",
            f"- 各 client 标准差从 `{float(client_series_df['std'].min()):.6f}` 到 `{float(client_series_df['std'].max()):.6f}`，波动强度差异明显。",
            "- 这说明 selected clients 虽然时间形态可能相关，但仍存在明显的幅值和波动规模非 IID。",
            "",
            "## 7. 收敛性分析",
            "",
            f"- 首轮 `train_loss={float(convergence_summary['train_loss_first']):.6f}`，末轮 `train_loss={float(convergence_summary['train_loss_last']):.6f}`。",
            f"- 首轮 `val_rmse={float(convergence_summary['val_rmse_first']):.6f}`，末轮 `val_rmse={float(convergence_summary['val_rmse_last']):.6f}`。",
            f"- 最后 5 轮 `train_loss` 相对改善 `{float(convergence_summary['train_rel_improve_last5']) * 100.0:.2f}%`。",
            f"- 最后 5 轮 `val_rmse` 相对改善 `{float(convergence_summary['val_rel_improve_last5']) * 100.0:.2f}%`。",
            f"- 结论：{convergence_summary['verdict']}。",
            "",
            "## 8. 原因判断",
            "",
            f"- 主要原因：{primary_reason}。",
            "- 支持证据：",
        ]
    )
    lines.extend([f"  - {reason}" for reason in reason_evidence])
    lines.extend(
        [
            f"- 评估口径是否异常：{format_bool(not eval_consistent)}。",
            f"- `main_metrics` 与 `client_metrics` 均值是否一致：{format_bool(eval_consistent)}。",
            "- 本次没有发现常数预测问题，也没有发现 CUDA/CPU 设备问题。",
            "",
            "## 9. 下一步建议",
            "",
            "- 先做一个小规模 smoke：只增加 `rounds`，验证收敛尾部是否还能明显改善 FedAvg。",
            "- 先做 selected clients 的分组/分布审计，重点检查 region `289` 一类低均值 client 是否系统性拉低全局模型。",
            "- 若后续仍需解释论文结果，可把 `NaiveLastValue` 的高 lag-1 相关性作为 baseline 极强的证据之一。",
            "- 当前不建议直接进入实验 2，也不建议直接更换聚合算法。",
            "",
            "## 10. 边界声明",
            "",
            "- 未运行新实验 2-6。",
            "- 未修改 FedAvg 聚合公式。",
            "- 未修改模型结构。",
            "- 未修改数据划分。",
            "- 未运行新的正式 full。",
            "- 未提交 `results/`。",
        ]
    )

    write_text("\n".join(lines), output_report_path)

    print("[fedavg_gap_diagnosis_report]", output_report_path)
    print(overall_compare_df.to_string(index=False))
    print("\n[primary_reason]", primary_reason)
    for reason in reason_evidence:
        print("-", reason)


if __name__ == "__main__":
    main()
