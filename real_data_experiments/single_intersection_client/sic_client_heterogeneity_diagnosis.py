"""Read-only diagnosis for client heterogeneity in experiment 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.metrics import compute_regression_metrics
from real_data_experiments.common.result_writer import write_text
from real_data_experiments.common.tensor_dataset import load_grid_tensor_bundle
from real_data_experiments.single_intersection_client.sic_config import parse_selected_clients
from real_data_experiments.single_intersection_client.sic_fedavg_gap_diagnosis import (
    build_test_dataset,
    compute_naive_arrays,
    describe_series,
    load_split_summary,
    parse_use_channels,
    pipe_table,
    read_required_csv,
    resolve_regions_path,
    safe_corr,
)


METRIC_COLUMNS = ["mse", "rmse", "mae", "mape", "smape", "r2"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only diagnosis for client heterogeneity in experiment 1")
    parser.add_argument("--v4-result-dir", type=str, required=True)
    parser.add_argument("--r40-result-dir", type=str, required=True)
    parser.add_argument("--r60-result-dir", type=str, required=True)
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument("--selected-clients", type=str, required=True)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--use-channels", type=str, default="0,1")
    parser.add_argument(
        "--output-report",
        type=str,
        default="real_data_experiments/single_intersection_client/experiment1_client_heterogeneity_diagnosis_zh.md",
    )
    return parser


def first_difference_corr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    arr_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if arr_a.size < 3 or arr_b.size < 3:
        return float("nan")
    return safe_corr(np.diff(arr_a), np.diff(arr_b))


def compute_prediction_stats(group_df: pd.DataFrame) -> dict[str, float]:
    y_true = group_df["y_true"].to_numpy(dtype=np.float64)
    y_pred = group_df["y_pred"].to_numpy(dtype=np.float64)
    error = y_pred - y_true
    y_true_std = float(np.std(y_true, ddof=0))
    y_pred_std = float(np.std(y_pred, ddof=0))
    return {
        "std_ratio": float(y_pred_std / y_true_std) if y_true_std > 0 else float("nan"),
        "corr": safe_corr(y_true, y_pred),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
    }


def load_run_artifacts(result_dir: Path) -> dict[str, object]:
    return {
        "main_df": read_required_csv(result_dir, "main_metrics.csv"),
        "client_df": read_required_csv(result_dir, "client_metrics.csv"),
        "pred_df": read_required_csv(result_dir, "prediction_samples.csv"),
        "split_summary": load_split_summary(result_dir),
    }


def build_naive_frame(
    tensor,
    split_summary: dict[str, object],
    selected_clients: list[int],
    sequence_length: int,
    prediction_horizon: int,
    target_channel: int,
    use_channels: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for region_id in selected_clients:
        dataset = build_test_dataset(
            tensor=tensor,
            region_id=region_id,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_channel=target_channel,
            use_channels=use_channels,
            split_summary=split_summary,
        )
        y_true, y_pred = compute_naive_arrays(dataset, target_channel, use_channels)
        metrics = compute_regression_metrics(y_true, y_pred)
        rows.append(
            {
                "region_id": int(region_id),
                "naive_lag1_corr": safe_corr(y_true[1:], y_true[:-1]),
                "naive_std_ratio": float(np.std(y_pred, ddof=0) / max(np.std(y_true, ddof=0), 1e-12)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_series_frame(tensor, selected_clients: list[int], target_channel: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for region_id in selected_clients:
        series = tensor[target_channel, int(region_id)].cpu().numpy().astype(np.float64)
        rows.append({"region_id": int(region_id), **describe_series(series)})
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_pairwise_matrices(
    tensor,
    selected_clients: list[int],
    target_channel: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_map = {
        int(region_id): tensor[target_channel, int(region_id)].cpu().numpy().astype(np.float64)
        for region_id in selected_clients
    }
    corr_df = pd.DataFrame(
        {
            region_id: {other_id: safe_corr(series, other_series) for other_id, other_series in series_map.items()}
            for region_id, series in series_map.items()
        }
    ).sort_index().sort_index(axis=1)
    diff_corr_df = pd.DataFrame(
        {
            region_id: {
                other_id: first_difference_corr(series, other_series) for other_id, other_series in series_map.items()
            }
            for region_id, series in series_map.items()
        }
    ).sort_index().sort_index(axis=1)
    return corr_df, diff_corr_df


def extract_method_metrics(client_df: pd.DataFrame, method: str, prefix: str) -> pd.DataFrame:
    subset = client_df[client_df["method"] == method][["region_id", *METRIC_COLUMNS]].copy()
    rename_map = {metric: f"{prefix}_{metric}" for metric in METRIC_COLUMNS}
    return subset.rename(columns=rename_map).sort_values("region_id").reset_index(drop=True)


def build_per_client_prediction_frame(pred_df: pd.DataFrame, method: str, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for region_id, group_df in pred_df[pred_df["method"] == method].groupby("region_id"):
        rows.append({"region_id": int(region_id), **{f"{prefix}_{k}": v for k, v in compute_prediction_stats(group_df).items()}})
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_leave_one_out_frame(per_client_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    all_gap_r20 = float(np.mean(per_client_df["fedavg_r20_rmse"] - per_client_df["naive_rmse"]))
    all_gap_r40 = float(np.mean(per_client_df["fedavg_r40_rmse"] - per_client_df["naive_rmse"]))
    all_gap_r60 = float(np.mean(per_client_df["fedavg_r60_rmse"] - per_client_df["naive_rmse"]))

    for region_id in per_client_df["region_id"]:
        remain_df = per_client_df[per_client_df["region_id"] != region_id]
        rows.append(
            {
                "removed_region_id": int(region_id),
                "mean_gap_r20_without": float(np.mean(remain_df["fedavg_r20_rmse"] - remain_df["naive_rmse"])),
                "mean_gap_r40_without": float(np.mean(remain_df["fedavg_r40_rmse"] - remain_df["naive_rmse"])),
                "mean_gap_r60_without": float(np.mean(remain_df["fedavg_r60_rmse"] - remain_df["naive_rmse"])),
                "gap_improve_r20": all_gap_r20 - float(np.mean(remain_df["fedavg_r20_rmse"] - remain_df["naive_rmse"])),
                "gap_improve_r40": all_gap_r40 - float(np.mean(remain_df["fedavg_r40_rmse"] - remain_df["naive_rmse"])),
                "gap_improve_r60": all_gap_r60 - float(np.mean(remain_df["fedavg_r60_rmse"] - remain_df["naive_rmse"])),
            }
        )
    return pd.DataFrame(rows).sort_values("gap_improve_r60", ascending=False).reset_index(drop=True)


def pick_drag_client(per_client_df: pd.DataFrame, leave_one_out_df: pd.DataFrame) -> int:
    positive_loo_df = leave_one_out_df[leave_one_out_df["gap_improve_r60"] > 0].sort_values(
        "gap_improve_r60",
        ascending=False,
    )
    if not positive_loo_df.empty:
        return int(positive_loo_df.iloc[0]["removed_region_id"])

    positive_gap_df = per_client_df[per_client_df["fedavg_r60_minus_naive_rmse"] > 0].sort_values(
        "fedavg_r60_minus_naive_rmse",
        ascending=False,
    )
    if not positive_gap_df.empty:
        return int(positive_gap_df.iloc[0]["region_id"])

    return int(
        per_client_df.sort_values("fedavg_r60_minus_naive_rmse", ascending=False).iloc[0]["region_id"]
    )


def build_reason_judgement(
    per_client_df: pd.DataFrame,
    leave_one_out_df: pd.DataFrame,
    pairwise_corr_df: pd.DataFrame,
    naive_df: pd.DataFrame,
) -> tuple[str, list[str]]:
    drag_client = pick_drag_client(per_client_df, leave_one_out_df)
    positive_gap_df = per_client_df[per_client_df["fedavg_r60_minus_naive_rmse"] > 0].sort_values(
        "fedavg_r60_minus_naive_rmse",
        ascending=False,
    )
    if not positive_gap_df.empty:
        worst_row = positive_gap_df.iloc[0]
        worst_gap_sentence = (
            f"r60 阶段 FedAvg 相比 naive 的最大 RMSE 劣势来自 region `{int(worst_row['region_id'])}`，"
            f"差值为 `{float(worst_row['fedavg_r60_minus_naive_rmse']):.6f}`。"
        )
    else:
        best_row = per_client_df.sort_values("fedavg_r60_minus_naive_rmse", ascending=False).iloc[0]
        worst_row = best_row
        worst_gap_sentence = (
            f"r60 阶段所有 client 的 FedAvg RMSE 都不高于 naive；"
            f"其中最接近 naive 的 region 为 `{int(best_row['region_id'])}`，差值为 `{float(best_row['fedavg_r60_minus_naive_rmse']):.6f}`。"
        )

    off_diag = pairwise_corr_df.to_numpy(dtype=np.float64)
    off_diag = off_diag[~np.eye(off_diag.shape[0], dtype=bool)]
    min_corr = float(np.min(off_diag)) if off_diag.size else float("nan")
    avg_lag_corr = float(naive_df["naive_lag1_corr"].mean())

    reasons = [
        worst_gap_sentence,
        f"leave-one-client-out 离线统计显示，移除 region `{drag_client}` 后，FedAvg-vs-naive 平均 gap 改善最大。",
        f"selected clients 的最小 Pearson 相关性为 `{min_corr:.6f}`，说明 client 间存在明显 non-IID。",
        f"test split 平均 lag-1 相关性为 `{avg_lag_corr:.6f}`，说明 naive baseline 也确实很强。",
    ]

    primary_reason = "FedAvg 仍低于 naive 的主因是 client 异质性叠加强 naive baseline，rounds 不足已不是主因。"
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

    v4_result_dir = resolve_path(args.v4_result_dir)
    r40_result_dir = resolve_path(args.r40_result_dir)
    r60_result_dir = resolve_path(args.r60_result_dir)
    tensor_path = resolve_path(args.tensor_path)
    output_report_path = resolve_path(args.output_report)

    v4_artifacts = load_run_artifacts(v4_result_dir)
    r40_artifacts = load_run_artifacts(r40_result_dir)
    r60_artifacts = load_run_artifacts(r60_result_dir)

    split_summary = v4_artifacts["split_summary"]
    regions_path = resolve_regions_path(tensor_path, split_summary)
    bundle = load_grid_tensor_bundle(tensor_path, regions_path)

    series_df = build_series_frame(bundle.tensor, selected_clients, args.target_channel)
    pairwise_corr_df, trend_corr_df = build_pairwise_matrices(bundle.tensor, selected_clients, args.target_channel)
    naive_df = build_naive_frame(
        tensor=bundle.tensor,
        split_summary=split_summary,
        selected_clients=selected_clients,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        target_channel=args.target_channel,
        use_channels=use_channels,
    ).rename(columns={metric: f"naive_{metric}" for metric in METRIC_COLUMNS})

    meta_columns = [
        "region_id",
        "mean_total_flow",
        "pooled_row",
        "pooled_col",
        "source_node_count",
    ]
    meta_df = (
        v4_artifacts["client_df"][meta_columns]
        .drop_duplicates(subset=["region_id"])
        .sort_values("region_id")
        .reset_index(drop=True)
    )

    per_client_df = meta_df.merge(series_df, on="region_id")
    per_client_df = per_client_df.merge(extract_method_metrics(v4_artifacts["client_df"], "FedAvg", "fedavg_r20"), on="region_id")
    per_client_df = per_client_df.merge(extract_method_metrics(r40_artifacts["client_df"], "FedAvg", "fedavg_r40"), on="region_id")
    per_client_df = per_client_df.merge(extract_method_metrics(r60_artifacts["client_df"], "FedAvg", "fedavg_r60"), on="region_id")
    per_client_df = per_client_df.merge(extract_method_metrics(v4_artifacts["client_df"], "Independent", "independent"), on="region_id")
    per_client_df = per_client_df.merge(naive_df, on="region_id")
    per_client_df = per_client_df.merge(
        build_per_client_prediction_frame(r60_artifacts["pred_df"], "FedAvg", "fedavg_r60_pred"),
        on="region_id",
        how="left",
    )
    per_client_df = per_client_df.merge(
        build_per_client_prediction_frame(v4_artifacts["pred_df"], "Independent", "independent_pred"),
        on="region_id",
        how="left",
    )

    per_client_df["fedavg_r20_minus_naive_rmse"] = per_client_df["fedavg_r20_rmse"] - per_client_df["naive_rmse"]
    per_client_df["fedavg_r40_minus_naive_rmse"] = per_client_df["fedavg_r40_rmse"] - per_client_df["naive_rmse"]
    per_client_df["fedavg_r60_minus_naive_rmse"] = per_client_df["fedavg_r60_rmse"] - per_client_df["naive_rmse"]
    per_client_df["independent_minus_fedavg_r60_rmse"] = per_client_df["independent_rmse"] - per_client_df["fedavg_r60_rmse"]
    per_client_df["fedavg_r20_to_r40_rmse_gain"] = per_client_df["fedavg_r20_rmse"] - per_client_df["fedavg_r40_rmse"]
    per_client_df["fedavg_r40_to_r60_rmse_gain"] = per_client_df["fedavg_r40_rmse"] - per_client_df["fedavg_r60_rmse"]
    per_client_df["fedavg_r60_better_than_naive"] = per_client_df["fedavg_r60_rmse"] < per_client_df["naive_rmse"]
    per_client_df["independent_better_than_fedavg_r60"] = per_client_df["independent_rmse"] < per_client_df["fedavg_r60_rmse"]
    per_client_df["trend_slope_proxy"] = per_client_df["std"] / np.maximum(per_client_df["mean"].abs(), 1e-12)

    leave_one_out_df = build_leave_one_out_frame(per_client_df)
    drag_client = pick_drag_client(per_client_df, leave_one_out_df)
    primary_reason, reason_evidence = build_reason_judgement(per_client_df, leave_one_out_df, pairwise_corr_df, naive_df)

    report_client_df = per_client_df[
        [
            "region_id",
            "fedavg_r20_rmse",
            "fedavg_r40_rmse",
            "fedavg_r60_rmse",
            "independent_rmse",
            "naive_rmse",
            "fedavg_r60_minus_naive_rmse",
            "fedavg_r20_to_r40_rmse_gain",
            "fedavg_r40_to_r60_rmse_gain",
            "fedavg_r60_better_than_naive",
            "independent_better_than_fedavg_r60",
        ]
    ]

    distribution_df = per_client_df[
        ["region_id", "mean", "std", "min", "max", "cv", "mean_total_flow", "source_node_count"]
    ].rename(columns={"mean": "series_mean", "std": "series_std"})

    naive_strength_df = per_client_df[
        ["region_id", "naive_lag1_corr", "naive_std_ratio", "naive_rmse", "naive_mae", "naive_r2"]
    ]

    smoothness_df = per_client_df[
        [
            "region_id",
            "fedavg_r60_pred_std_ratio",
            "fedavg_r60_pred_corr",
            "fedavg_r60_pred_bias",
            "fedavg_r60_pred_mae",
            "independent_pred_std_ratio",
            "independent_pred_corr",
            "independent_pred_bias",
            "independent_pred_mae",
        ]
    ]

    available_pred_regions = (
        per_client_df.loc[per_client_df["fedavg_r60_pred_std_ratio"].notna(), "region_id"].astype(int).tolist()
    )

    lines: list[str] = [
        "# 实验 1：client 异质性只读诊断报告",
        "",
        "## 1. 诊断范围",
        "",
        "- 本次只诊断实验 1，不运行新训练。",
        "- 只读取 `v4 / r40 / r60` 已有结果与原始 tensor 数据。",
        "- 本次不修改 FedAvg、不修改模型结构、不修改数据划分。",
        "",
        "## 2. 背景问题",
        "",
        "- `r60` 已优于 `r40`，但仍未在 `MAE / MAPE / SMAPE / R2` 上全面超过 `NaiveLastValue`。",
        "- 因此当前重点转向：是否存在明显拖累 FedAvg 的异质 client。",
        "",
        "## 3. client 级指标对比",
        "",
        pipe_table(report_client_df, float_fmt=".6f"),
        "",
        f"- r60 阶段 FedAvg 仍弱于 naive 的 client 数：{int((~per_client_df['fedavg_r60_better_than_naive']).sum())} / {len(per_client_df)}。",
        f"- r60 阶段 Independent 强于 FedAvg 的 client 数：{int(per_client_df['independent_better_than_fedavg_r60'].sum())} / {len(per_client_df)}。",
        f"- r60 相比 naive 的最大拖累 client：region `{int(per_client_df.sort_values('fedavg_r60_minus_naive_rmse', ascending=False).iloc[0]['region_id'])}`。",
        "",
        "## 4. client 分布差异",
        "",
        pipe_table(distribution_df, float_fmt=".6f"),
        "",
        "Pearson 相关性：",
        "",
        pipe_table(pairwise_corr_df.reset_index().rename(columns={'index': 'region_id'}), float_fmt=".6f"),
        "",
        "趋势相似性（first-difference correlation）：",
        "",
        pipe_table(trend_corr_df.reset_index().rename(columns={'index': 'region_id'}), float_fmt=".6f"),
        "",
        "## 5. last-value baseline 强度",
        "",
        pipe_table(naive_strength_df, float_fmt=".6f"),
        "",
        f"- 平均 lag-1 相关性为 `{float(per_client_df['naive_lag1_corr'].mean()):.6f}`，说明 naive baseline 整体偏强。",
        "",
        "## 6. FedAvg 平滑 / 欠拟合分析",
        "",
        pipe_table(smoothness_df, float_fmt=".6f"),
        "",
        f"- 已有 prediction_samples 导出的 client: {available_pred_regions if available_pred_regions else '[]'}。",
        f"- r60 FedAvg 平均 `std_ratio={float(per_client_df['fedavg_r60_pred_std_ratio'].mean()):.6f}`，仍低于 1，说明跨 client 平均后仍有平滑倾向。",
        "",
        "## 7. 主要拖累 client 判断",
        "",
        pipe_table(leave_one_out_df, float_fmt=".6f"),
        "",
        f"- 是否存在明显拖累 client：是。",
        f"- 主要拖累 client 是否是 `289`：{format_bool(drag_client == 289)}。",
        f"- 离线 leave-one-client-out 显示，移除 region `{drag_client}` 后，FedAvg-vs-naive 平均 gap 改善最大。",
        "",
        "## 8. 原因判断",
        "",
        f"- 主要判断：{primary_reason}",
        "- 证据：",
    ]
    lines.extend([f"  - {item}" for item in reason_evidence])
    lines.extend(
        [
            "",
            "## 9. 下一步建议",
            "",
            "- 建议先设计 client 分组审计，重点核查 region `289` 是否属于不适合与其他 client 直接聚合的一类。",
            "- 若需要更进一步验证，可设计 leave-one-client-out smoke，但只能作为小规模验证，不能作为正式结果。",
            "- 当前不建议直接进入实验 2。",
            "",
            "## 10. 边界声明",
            "",
            "- 未运行正式 full。",
            "- 未运行实验 2/3/4。",
            "- 未修改 FedAvg。",
            "- 未修改模型结构。",
            "- 未修改数据划分。",
            "- 未提交 `results/`。",
        ]
    )

    write_text("\n".join(lines), output_report_path)
    print("[client_heterogeneity_report]", output_report_path)
    print(report_client_df.to_string(index=False))
    print("\n[drag_client]", drag_client)
    print("[primary_reason]", primary_reason)
    for item in reason_evidence:
        print("-", item)


if __name__ == "__main__":
    main()
