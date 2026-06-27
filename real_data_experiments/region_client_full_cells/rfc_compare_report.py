"""Compare full-cells region-client smoke runs against the current single-grid baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.region_client_full_cells.rfc_eval import (
    load_client_metrics,
    load_main_metrics,
    load_run_config,
    load_split_summary,
)
from real_data_experiments.region_client_full_cells.rfc_report import fmt, pipe_table
from real_data_experiments.single_intersection_client.sic_fedavg_metric_optimization_summary import load_run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the full-cells region-client smoke report.")
    parser.add_argument("--inventory-csv", type=str, required=True)
    parser.add_argument("--single-grid-dir", type=str, required=True)
    parser.add_argument("--spatial-dir", type=str, required=True)
    parser.add_argument("--similarity-dir", type=str, required=True)
    parser.add_argument("--spatial-partition", type=str, required=True)
    parser.add_argument("--similarity-partition", type=str, required=True)
    parser.add_argument("--output-report", type=str, required=True)
    return parser


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_full_cells_run(result_dir: str | Path) -> dict[str, Any]:
    result_path = Path(result_dir)
    if not result_path.exists():
        return {"exists": False, "name": result_path.name}
    return {
        "exists": True,
        "name": result_path.name,
        "run_config": load_run_config(result_path),
        "split_summary": load_split_summary(result_path),
        "main_metrics": load_main_metrics(result_path),
        "client_metrics": load_client_metrics(result_path),
    }


def get_method_row(main_df: pd.DataFrame, method: str) -> dict[str, float] | None:
    subset = main_df[main_df["method"] == method]
    if subset.empty:
        return None
    return subset.iloc[0].to_dict()


def compare_against_naive(fedavg_row: dict[str, float] | None, naive_row: dict[str, float] | None) -> dict[str, str]:
    if fedavg_row is None or naive_row is None:
        return {metric: "未运行 / 目录不存在" for metric in ["rmse", "mae", "mape", "smape", "r2", "all"]}
    result = {
        "rmse": "是" if float(fedavg_row["rmse"]) < float(naive_row["rmse"]) else "否",
        "mae": "是" if float(fedavg_row["mae"]) < float(naive_row["mae"]) else "否",
        "mape": "是" if float(fedavg_row["mape"]) < float(naive_row["mape"]) else "否",
        "smape": "是" if float(fedavg_row["smape"]) < float(naive_row["smape"]) else "否",
        "r2": "是" if float(fedavg_row["r2"]) > float(naive_row["r2"]) else "否",
    }
    result["all"] = "是" if all(result[key] == "是" for key in ["rmse", "mae", "mape", "smape", "r2"]) else "否"
    return result


def summarize_partition(partition_payload: dict[str, Any]) -> list[list[object]]:
    rows = []
    for row in partition_payload["clients"]:
        rows.append(
            [
                int(row["client_id"]),
                int(row["cell_count"]),
                int(row["source_node_count_sum"]),
                f"{float(row['mean_total_flow_mean']):.3f}",
                int(row["train_samples_estimate"]),
                int(row["val_samples_estimate"]),
                int(row["test_samples_estimate"]),
                f"{float(row['internal_mean_pairwise_corr']):.6f}" if row["internal_mean_pairwise_corr"] is not None else "nan",
            ]
        )
    return rows


def drag_client_summary(client_df: pd.DataFrame) -> str:
    if client_df.empty:
        return "未运行 / 目录不存在"
    pivot_df = (
        client_df[client_df["method"].isin(["FedAvg", "NaiveLastValue"])]
        .pivot_table(index="client_id", columns="method", values="rmse")
        .reset_index()
    )
    if "FedAvg" not in pivot_df.columns or "NaiveLastValue" not in pivot_df.columns:
        return "未运行 / 目录不存在"
    pivot_df["gap"] = pivot_df["FedAvg"] - pivot_df["NaiveLastValue"]
    row = pivot_df.sort_values("gap", ascending=False).iloc[0]
    return f"client {int(row['client_id'])} (FedAvg_minus_Naive_RMSE={float(row['gap']):.3f})"


def render_report(
    inventory_df: pd.DataFrame,
    baseline_summary: dict[str, Any],
    spatial_run: dict[str, Any],
    similarity_run: dict[str, Any],
    spatial_partition: dict[str, Any],
    similarity_partition: dict[str, Any],
) -> str:
    valid_df = inventory_df[inventory_df["is_valid_cell"]].copy()
    baseline_fed = baseline_summary["fedavg_metrics"]
    baseline_ind = baseline_summary["independent_metrics"]
    baseline_naive = baseline_summary["naive_metrics"]
    spatial_fed = get_method_row(spatial_run["main_metrics"], "FedAvg") if spatial_run["exists"] else None
    spatial_ind = get_method_row(spatial_run["main_metrics"], "Independent") if spatial_run["exists"] else None
    spatial_naive = get_method_row(spatial_run["main_metrics"], "NaiveLastValue") if spatial_run["exists"] else None
    similarity_fed = get_method_row(similarity_run["main_metrics"], "FedAvg") if similarity_run["exists"] else None
    similarity_ind = get_method_row(similarity_run["main_metrics"], "Independent") if similarity_run["exists"] else None
    similarity_naive = get_method_row(similarity_run["main_metrics"], "NaiveLastValue") if similarity_run["exists"] else None

    spatial_cmp = compare_against_naive(spatial_fed, spatial_naive)
    similarity_cmp = compare_against_naive(similarity_fed, similarity_naive)
    baseline_cmp = compare_against_naive(baseline_fed, baseline_naive)

    spatial_homogeneity = float(pd.DataFrame(spatial_partition["clients"])["internal_mean_pairwise_corr"].mean())
    similarity_homogeneity = float(pd.DataFrame(similarity_partition["clients"])["internal_mean_pairwise_corr"].mean())

    table_rows = [
        [
            "single_grid_k5_best",
            fmt(baseline_fed["mse"]),
            fmt(baseline_fed["rmse"]),
            fmt(baseline_fed["mae"]),
            fmt(baseline_fed["mape"]),
            fmt(baseline_fed["smape"]),
            fmt(baseline_fed["r2"]),
            fmt(baseline_ind["rmse"]),
            fmt(baseline_naive["rmse"]),
        ],
        [
            "full_cells_spatial_k5",
            fmt(spatial_fed["mse"] if spatial_fed else None),
            fmt(spatial_fed["rmse"] if spatial_fed else None),
            fmt(spatial_fed["mae"] if spatial_fed else None),
            fmt(spatial_fed["mape"] if spatial_fed else None),
            fmt(spatial_fed["smape"] if spatial_fed else None),
            fmt(spatial_fed["r2"] if spatial_fed else None),
            fmt(spatial_ind["rmse"] if spatial_ind else None),
            fmt(spatial_naive["rmse"] if spatial_naive else None),
        ],
        [
            "full_cells_similarity_k5",
            fmt(similarity_fed["mse"] if similarity_fed else None),
            fmt(similarity_fed["rmse"] if similarity_fed else None),
            fmt(similarity_fed["mae"] if similarity_fed else None),
            fmt(similarity_fed["mape"] if similarity_fed else None),
            fmt(similarity_fed["smape"] if similarity_fed else None),
            fmt(similarity_fed["r2"] if similarity_fed else None),
            fmt(similarity_ind["rmse"] if similarity_ind else None),
            fmt(similarity_naive["rmse"] if similarity_naive else None),
        ],
    ]

    better_org = "similarity_k5" if similarity_homogeneity > spatial_homogeneity else "spatial_k5"
    lines = [
        "# 全量 grid cells 多客户端组织实验 smoke 报告",
        "",
        "## 1. 实验目的",
        "",
        "本实验用于替代“单个 grid cell = 一个 client”的组织方式，构造“多个 grid cells = 一个 client”的 full-cells region/cluster client 设置。",
        "",
        "## 2. 与当前 K=5 single grid-cell 实验的区别",
        "",
        "- 当前实验 1 是 5 个单 grid-cell client；",
        "- 本实验使用全部有效 grid cells；",
        "- 本实验每个 client 包含多个 grid cells；",
        "- 本实验包含 spatial partition 和 similarity partition 两种方式；",
        "- 本实验不是删除 289，而是改变 client 组织方式。",
        "",
        "## 3. 全量有效 cells 统计",
        "",
        f"- total grid cells: `{int(len(inventory_df))}`",
        f"- valid grid cells: `{int(len(valid_df))}`",
        f"- invalid / empty cells: `{int((~inventory_df['is_valid_cell']).sum())}`",
        f"- mean source_node_count: `{valid_df['source_node_count'].mean():.3f}`",
        f"- mean_total_flow mean: `{valid_df['mean_total_flow'].mean():.3f}`",
        f"- flow_cv mean: `{valid_df['flow_cv'].mean():.6f}`",
        f"- lag1_autocorr mean: `{valid_df['lag1_autocorr'].mean():.6f}`",
        "",
        "## 4. client 分组方案",
        "",
        "### spatial_k5",
        "",
        pipe_table(
            ["client_id", "cell_count", "source_node_count", "mean_total_flow", "train_samples", "val_samples", "test_samples", "内部平均相关性"],
            summarize_partition(spatial_partition),
        ),
        "",
        "### similarity_k5",
        "",
        pipe_table(
            ["client_id", "cell_count", "source_node_count", "mean_total_flow", "train_samples", "val_samples", "test_samples", "内部平均相关性"],
            summarize_partition(similarity_partition),
        ),
        "",
        f"- similarity partition 的 cluster procedure: `{similarity_partition['cluster_procedure']}`",
        "",
        "## 5. 主指标对比",
        "",
        pipe_table(
            ["setting", "FedAvg_MSE", "FedAvg_RMSE", "FedAvg_MAE", "FedAvg_MAPE", "FedAvg_SMAPE", "FedAvg_R2", "Independent_RMSE", "Naive_RMSE"],
            table_rows,
        ),
        "",
        "## 6. 是否全面超过 NaiveLastValue",
        "",
        pipe_table(
            ["setting", "RMSE优于naive", "MAE优于naive", "MAPE优于naive", "SMAPE优于naive", "R2优于naive", "全面超过"],
            [
                ["single_grid_k5_best", baseline_cmp["rmse"], baseline_cmp["mae"], baseline_cmp["mape"], baseline_cmp["smape"], baseline_cmp["r2"], baseline_cmp["all"]],
                ["full_cells_spatial_k5", spatial_cmp["rmse"], spatial_cmp["mae"], spatial_cmp["mape"], spatial_cmp["smape"], spatial_cmp["r2"], spatial_cmp["all"]],
                ["full_cells_similarity_k5", similarity_cmp["rmse"], similarity_cmp["mae"], similarity_cmp["mape"], similarity_cmp["smape"], similarity_cmp["r2"], similarity_cmp["all"]],
            ],
        ),
        "",
        "## 7. per-client 诊断",
        "",
        f"- spatial_k5 最大拖累 client: {drag_client_summary(spatial_run['client_metrics']) if spatial_run['exists'] else '未运行 / 目录不存在'}",
        f"- similarity_k5 最大拖累 client: {drag_client_summary(similarity_run['client_metrics']) if similarity_run['exists'] else '未运行 / 目录不存在'}",
        f"- similarity partition 是否降低 client 间异质性：{'是' if similarity_homogeneity > spatial_homogeneity else '否'}，平均内部相关性 `{similarity_homogeneity:.6f}` vs spatial `{spatial_homogeneity:.6f}`。",
        "- spatial partition 保留更直接的空间解释性，因为 client 仍对应连续或近连续的 pooled-grid block。",
        f"- 当前更适合论文主线叙事的方式：`{better_org}`。",
        "",
        "## 8. 结论",
        "",
        f"- 换 client 组织方式是否有效：{'部分有效' if any(value == '是' for value in [spatial_cmp['rmse'], similarity_cmp['rmse'], spatial_cmp['mae'], similarity_cmp['mae']]) else '当前未见明显改善'}。",
        f"- spatial 和 similarity 哪个更好：`{'similarity_k5' if (similarity_fed and spatial_fed and float(similarity_fed['rmse']) <= float(spatial_fed['rmse'])) else 'spatial_k5'}`。",
        f"- 是否优于当前 K=5 single grid-cell：{'是' if (similarity_fed and float(similarity_fed['rmse']) < float(baseline_fed['rmse'])) or (spatial_fed and float(spatial_fed['rmse']) < float(baseline_fed['rmse'])) else '否'}。",
        "- 是否建议作为后续正式实验：仅当该组织方式在多指标上稳定优于 single-grid 与 naive 时才建议。",
        "- 是否需要 K=8/K=10 扩展：若 k5 smoke 中某一方式明显更优，则建议扩展。",
        "",
        "## 9. 下一步建议",
        "",
        f"- {'如果 similarity_k5 明显最好：扩展 K=8/K=10。' if similarity_homogeneity > spatial_homogeneity else '如果 spatial_k5 更稳：补空间地图和正式复跑。'}",
        "",
        "## 10. 边界声明",
        "",
        "本阶段新增的是全量有效 grid cells 多客户端组织 diagnostic/smoke；未修改 FedAvg 聚合公式，未修改模型结构，未修改数据划分，未删除 NaiveLastValue，未删除或替换 289，未提交 results。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()
    inventory_df = pd.read_csv(resolve_path(args.inventory_csv))
    baseline_summary = load_run_summary(resolve_path(args.single_grid_dir))
    spatial_run = load_full_cells_run(resolve_path(args.spatial_dir))
    similarity_run = load_full_cells_run(resolve_path(args.similarity_dir))
    spatial_partition = read_json(resolve_path(args.spatial_partition))
    similarity_partition = read_json(resolve_path(args.similarity_partition))
    report = render_report(
        inventory_df=inventory_df,
        baseline_summary=baseline_summary,
        spatial_run=spatial_run,
        similarity_run=similarity_run,
        spatial_partition=spatial_partition,
        similarity_partition=similarity_partition,
    )
    output_path = resolve_path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[output_report] {output_path}")


if __name__ == "__main__":
    main()

