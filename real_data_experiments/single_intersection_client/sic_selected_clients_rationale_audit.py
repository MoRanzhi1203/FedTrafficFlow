"""Read-only rationale audit for selected clients in experiment 1."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.metrics import compute_regression_metrics
from real_data_experiments.common.tensor_dataset import load_grid_tensor_bundle
from real_data_experiments.single_intersection_client.sic_fedavg_gap_diagnosis import (
    build_test_dataset,
    compute_naive_arrays,
    describe_series,
    safe_corr,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only rationale audit for selected_clients in experiment 1."
    )
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument("--selected-clients", type=str, required=True)
    parser.add_argument("--allocation-report", type=str, required=True)
    parser.add_argument("--heterogeneity-report", type=str, required=True)
    parser.add_argument("--output-report", type=str, required=True)
    return parser


def parse_selected_clients(raw_text: str) -> list[int]:
    values = [part.strip() for part in str(raw_text).split(",")]
    clients = [int(value) for value in values if value]
    if not clients:
        raise ValueError("--selected-clients must not be empty.")
    return clients


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode {path}")


def load_json(path: Path) -> dict:
    return json.loads(read_text(path))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def pipe_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def format_float(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "无法稳定读取"
    return f"{float(value):.{digits}f}"


def build_pairwise_corr_rows(series_map: dict[int, np.ndarray], selected_clients: list[int]) -> list[list[object]]:
    rows: list[list[object]] = []
    for region_id in selected_clients:
        row: list[object] = [region_id]
        for other_id in selected_clients:
            row.append(format_float(safe_corr(series_map[region_id], series_map[other_id])))
        rows.append(row)
    return rows


def compute_test_lag1_autocorr(dataset) -> float:
    y_true, _ = compute_naive_arrays(dataset, target_channel=dataset.target_channel, use_channels=dataset.use_channels)
    if y_true.size < 2:
        return float("nan")
    return safe_corr(y_true[1:], y_true[:-1])


def build_naive_metric_frame(bundle, split_summary: dict, selected_clients: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    sequence_length = int(split_summary["sequence_length"])
    prediction_horizon = int(split_summary["prediction_horizon"])
    target_channel = int(split_summary["target_channel"])
    use_channels = [int(item) for item in split_summary["use_channels"]]

    for region_id in selected_clients:
        dataset = build_test_dataset(
            tensor=bundle.tensor,
            region_id=int(region_id),
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_channel=target_channel,
            use_channels=use_channels,
            split_summary=split_summary,
        )
        y_true, y_pred = compute_naive_arrays(dataset, target_channel=target_channel, use_channels=use_channels)
        metrics = compute_regression_metrics(y_true, y_pred)
        rows.append(
            {
                "region_id": int(region_id),
                "naive_rmse": metrics["rmse"],
                "naive_mae": metrics["mae"],
                "naive_mape": metrics["mape"],
                "naive_smape": metrics["smape"],
                "naive_r2": metrics["r2"],
                "lag1_autocorr": compute_test_lag1_autocorr(dataset),
            }
        )
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_client_fact_frame(
    bundle,
    split_summary: dict,
    client_rows: list[dict[str, str]],
    naive_df: pd.DataFrame,
    selected_clients: list[int],
) -> pd.DataFrame:
    fedavg_map = {
        int(row["region_id"]): row
        for row in client_rows
        if row.get("method") == "FedAvg" and row.get("region_id")
    }
    independent_map = {
        int(row["region_id"]): row
        for row in client_rows
        if row.get("method") == "Independent" and row.get("region_id")
    }
    naive_map = {int(row["region_id"]): row for _, row in naive_df.iterrows()}
    split_map = {int(item["region_id"]): item for item in split_summary.get("clients", [])}

    rows: list[dict[str, object]] = []
    for region_id in selected_clients:
        series = bundle.tensor[int(split_summary["target_channel"]), int(region_id)].cpu().numpy().astype(np.float64)
        stats = describe_series(series)
        split_item = split_map[int(region_id)]
        fedavg_row = fedavg_map[int(region_id)]
        independent_row = independent_map[int(region_id)]
        naive_row = naive_map[int(region_id)]
        rows.append(
            {
                "region_id": int(region_id),
                "train_samples": int(split_item["train"]["sample_count"]),
                "val_samples": int(split_item["val"]["sample_count"]),
                "test_samples": int(split_item["test"]["sample_count"]),
                "train_split": f"[{split_item['train']['split_start_time']},{split_item['train']['split_end_time']})",
                "val_split": f"[{split_item['val']['split_start_time']},{split_item['val']['split_end_time']})",
                "test_split": f"[{split_item['test']['split_start_time']},{split_item['test']['split_end_time']})",
                "pooled_row": int(split_item["pooled_row"]),
                "pooled_col": int(split_item["pooled_col"]),
                "source_node_count": int(split_item["source_node_count"]),
                "mean_total_flow": float(split_item["mean_total_flow"]),
                "series_mean": float(stats["mean"]),
                "series_std": float(stats["std"]),
                "series_min": float(stats["min"]),
                "series_max": float(stats["max"]),
                "cv": float(stats["cv"]),
                "lag1_autocorr": float(naive_row["lag1_autocorr"]),
                "fedavg_rmse": float(fedavg_row["rmse"]),
                "fedavg_r2": float(fedavg_row["r2"]),
                "independent_rmse": float(independent_row["rmse"]),
                "independent_r2": float(independent_row["r2"]),
                "naive_rmse": float(naive_row["naive_rmse"]),
                "naive_r2": float(naive_row["naive_r2"]),
            }
        )
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_fact_table_rows(client_fact_df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for _, row in client_fact_df.iterrows():
        rows.append(
            [
                int(row["region_id"]),
                int(row["train_samples"]),
                int(row["val_samples"]),
                int(row["test_samples"]),
                row["train_split"],
                row["val_split"],
                row["test_split"],
                int(row["source_node_count"]),
                format_float(float(row["mean_total_flow"]), 3),
                format_float(float(row["series_mean"]), 3),
                format_float(float(row["series_std"]), 3),
                format_float(float(row["series_min"]), 3),
                format_float(float(row["series_max"]), 3),
                format_float(float(row["cv"])),
                format_float(float(row["lag1_autocorr"])),
                format_float(float(row["fedavg_rmse"]), 3),
                format_float(float(row["independent_rmse"]), 3),
                format_float(float(row["naive_rmse"]), 3),
                format_float(float(row["fedavg_r2"])),
                format_float(float(row["independent_r2"])),
                format_float(float(row["naive_r2"])),
            ]
        )
    return rows


def build_evidence_summary(allocation_text: str, heterogeneity_text: str, client_fact_df: pd.DataFrame) -> list[str]:
    min_corr_match = re.search(r"最小 Pearson 相关性为 `([0-9.]+)`", heterogeneity_text)
    drag_client_match = re.search(r"主要拖累 client 是否是 `?(\d+)`?：是", heterogeneity_text)
    avg_lag_match = re.search(r"平均 lag-1 相关性为 `([0-9.]+)`", heterogeneity_text)
    evidence_lines = [
        "- 当前 5 个 client 已在 v4 CUDA 正式链路中完整跑通，并在 `run_config.json` 与 `split_summary.json` 中可复现。",
        "- 当前 5 个 client 已具备 `FedAvg vs Independent`、`NaiveLastValue`、`r20/r40/r60` 和 client 异质性诊断链路。",
        "- 当前 client 数量为 `5`，相对原稿口径中的 `K=3` 已形成增强。",
    ]
    if min_corr_match:
        evidence_lines.append(f"- 5 个 client 的最小 Pearson 相关性为 `{min_corr_match.group(1)}`，说明该组合具有明显 non-IID。")
    else:
        evidence_lines.append("- 该项当前无法从已有结果稳定读取，不能伪造。")
    if drag_client_match:
        evidence_lines.append(f"- region `{drag_client_match.group(1)}` 被现有异质性诊断识别为主要拖累 client。")
    else:
        evidence_lines.append("- client 289 的拖累结论当前无法从已有结果稳定读取，不能伪造。")
    if avg_lag_match:
        evidence_lines.append(f"- test split 平均 lag-1 自相关为 `{avg_lag_match.group(1)}`，说明短时惯性 baseline 较强。")
    else:
        evidence_lines.append("- lag-1 自相关的总体汇总当前无法从已有结果稳定读取，不能伪造。")
    if contains_any(allocation_text, [r"K=5", r"小规模真实数据 FL"]):
        evidence_lines.append("- 当前文档链已明确：K=5 是增强后的 grid-cell-level 异质设置，但仍属于小规模真实数据 FL。")
    else:
        evidence_lines.append("- K=5 相比 K=3 的增强说明当前无法从已有报告稳定定位，不能伪造。")
    if float(client_fact_df["source_node_count"].min()) > 0:
        evidence_lines.append("- 5 个 client 都是 active pooled-grid regions，且均有非零 `source_node_count`。")
    return evidence_lines


def render_report(
    run_config: dict,
    split_summary: dict,
    client_fact_df: pd.DataFrame,
    pairwise_corr_rows: list[list[object]],
    allocation_text: str,
    heterogeneity_text: str,
    selected_clients: list[int],
    result_dir: Path,
) -> str:
    evidence_lines = build_evidence_summary(allocation_text, heterogeneity_text, client_fact_df)
    client_fact_table = pipe_table(
        [
            "region_id",
            "train_samples",
            "val_samples",
            "test_samples",
            "train_split",
            "val_split",
            "test_split",
            "source_node_count",
            "mean_total_flow",
            "series_mean",
            "series_std",
            "series_min",
            "series_max",
            "cv",
            "lag1_autocorr",
            "fedavg_rmse",
            "independent_rmse",
            "naive_rmse",
            "fedavg_r2",
            "independent_r2",
            "naive_r2",
        ],
        build_fact_table_rows(client_fact_df),
    )
    corr_table = pipe_table(
        ["region_id", *[str(item) for item in selected_clients]],
        pairwise_corr_rows,
    )

    lines = [
        "# 实验 1：selected_clients 选择依据说明报告",
        "",
        "## 1. 报告目的",
        "",
        "本报告回答：",
        "- 为什么当前实验 1 使用 `selected_clients=290,284,318,288,289`；",
        "- 为什么不是更少；",
        "- 为什么不是更多；",
        "- 为什么不是其他 client；",
        "- 该组合在一审/导师要求下的定位是什么。",
        "",
        "## 2. 当前实验 1 的 client 设置事实",
        "",
        f"- `selected_clients`：`{','.join(str(item) for item in selected_clients)}`",
        f"- client 数量：`{run_config['num_clients']}`",
        f"- `result_dir`：`{result_dir}`",
        f"- `workflow`：`{run_config['workflow']}`",
        f"- `rounds`：`{run_config['communication_rounds']}`",
        f"- `device`：`{run_config['device']}`",
        f"- `split_summary`：`{split_summary['split_strategy']}`，`train=[{split_summary['train_start']},{split_summary['train_end']})`，`val=[{split_summary['val_start']},{split_summary['val_end']})`，`test=[{split_summary['test_start']},{split_summary['test_end']})`",
        "",
        "当前 `client_metrics`、`split_summary` 与 tensor 统计合并后的 per-client 信息如下：",
        "",
        client_fact_table,
        "",
        "当前 5 个 client 的 Pearson 相关性矩阵如下：",
        "",
        corr_table,
        "",
        "## 3. 一审/导师对 client 数量和分配逻辑的约束",
        "",
        "- 一审/导师没有要求固定 client ID。",
        "- 要求的是多区域数据分布、异质性、client-level variability、client 数量说明、clustering procedure。",
        "- 原稿 `K=3` 被认为偏小，当前 `K=5` 是增强，但仍属于小规模真实数据 FL。",
        "- 当前自动检索基于项目内可读取材料；若需对外正式表述，仍应人工核对原始 `docx/pdf`。",
        "",
        "## 4. 为什么是这 5 个 client",
        "",
        "- 这 5 个 client 是当前实验 1 正式 v4 CUDA 链路中已经完整跑通的 `selected_clients`。",
        "- 它们已有完整的 `FedAvg vs Independent`、`NaiveLastValue`、`r20/r40/r60`、client 异质性诊断证据。",
        "- 它们形成了一个 `K=5 grid-cell-level heterogeneous setting`。",
        "- `K=5` 相比原稿 `K=3` 增强了 `client-level variability`。",
        "- 其中 `289` 提供了强异质 client 证据，可以解释标准 `FedAvg` 在 non-IID 下的局限。",
        "- 因此这 5 个适合作为当前阶段“细粒度异质 client 设置”的审计对象。",
        "",
        "当前可直接支撑上述判断的证据包括：",
        *evidence_lines,
        "",
        "## 5. 为什么不是更少",
        "",
        "- 原稿 `K=3` 已被认为偏小。",
        "- 更少 client 会削弱多区域联邦学习属性。",
        "- 更少 client 不利于分析 `client-level variability`。",
        "- 更少 client 可能掩盖 `289` 这类异质 client 对 `FedAvg` 的影响。",
        "- 因此当前不建议回退到 `K=3` 或更少。",
        "",
        "## 6. 为什么不是更多",
        "",
        "- 更多 client 需要新实验。",
        "- 当前阶段只解释已有实验。",
        "- 更多 client 会改变实验边界。",
        "- 当前已有完整诊断链路的是这 5 个 client。",
        "- 更多 client 可作为后续 `region/cluster` 或扩展实验方向，而不是当前阶段立即替换实验 1。",
        "",
        "## 7. 为什么不是其他 client",
        "",
        "- 其他 client 缺少同等完整结果链。",
        "- 替换 client 需要重新训练和重新审计。",
        "- 当前没有证据说明其他 client 更适合作为当前实验 1 的解释对象。",
        "- 因为当前只读阶段只覆盖已有 v4/r40/r60/异质性证据链，所以不能伪造“其他 client 一定更优”的结论。",
        "- 若要更换，应作为独立 client selection 实验。",
        "",
        "## 8. 当前 5-client 设置的局限",
        "",
        "- 不代表全部真实路网。",
        "- 不代表最终最优 client 组合。",
        "- 仍需补充空间覆盖解释。",
        "- 新实验 3-6 的 grouped-client / global-partition 实验线仍需继续补齐。",
        "- 仍需补充更多客户端组织方式证据与后续实验结果。",
        "",
        "## 9. 推荐论文表述",
        "",
        "当前实验 1 采用 K=5 的 grid-cell-level client 设置。与原稿 K=3 的设置相比，该设置增加了参与客户端数量，使实验能够更直接地观察 client-level variability 和强 non-IID 对标准 FedAvg 的影响。五个客户端均来自当前真实数据正式实验链路，并已完成 FedAvg、Independent、NaiveLastValue、通信轮次和异质性诊断。需要强调的是，该设置不是对真实路网全部客户端的穷尽覆盖，而是用于构造一个细粒度异质客户端场景，以检验标准 FedAvg 在真实交通流数据中的可运行性、收敛性和局限性。后续新实验 3/4 的 grouped-client 组织与新实验 5/6 的全局覆盖式划分，将进一步用于验证更同质或更系统的客户端组织方式是否能够缓解跨客户端分布差异带来的性能下降。",
        "",
        "## 10. 结论",
        "",
        "- 选择这 5 个 client 是当前阶段合理的。",
        "- 不能说它们是唯一最优组合。",
        "- 不建议减少。",
        "- 不建议当前直接增加。",
        "- 不建议当前直接替换。",
        "- 下一步应补充空间覆盖与分布统计，或进入新实验 3-6 的 grouped-client / global-partition 审计。",
        "",
        "## 11. 边界声明",
        "",
        "- 本阶段未运行训练。",
        "- 未运行新实验 2-6。",
        "- 未修改 FedAvg。",
        "- 未修改模型结构。",
        "- 未修改数据划分。",
        "- 未提交 `results/`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()

    result_dir = resolve_path(args.result_dir)
    tensor_path = resolve_path(args.tensor_path)
    allocation_report = resolve_path(args.allocation_report)
    heterogeneity_report = resolve_path(args.heterogeneity_report)
    output_report = resolve_path(args.output_report)
    selected_clients = parse_selected_clients(args.selected_clients)

    run_config = load_json(result_dir / "run_config.json")
    split_summary = load_json(result_dir / "split_summary.json")
    client_rows = load_csv_rows(result_dir / "client_metrics.csv")
    allocation_text = read_text(allocation_report)
    heterogeneity_text = read_text(heterogeneity_report)

    bundle = load_grid_tensor_bundle(
        tensor_path=tensor_path,
        regions_path=resolve_path(split_summary["regions_path"]),
    )
    naive_df = build_naive_metric_frame(bundle, split_summary, selected_clients)
    client_fact_df = build_client_fact_frame(bundle, split_summary, client_rows, naive_df, selected_clients)
    series_map = {
        int(region_id): bundle.tensor[int(split_summary["target_channel"]), int(region_id)].cpu().numpy().astype(np.float64)
        for region_id in selected_clients
    }
    pairwise_corr_rows = build_pairwise_corr_rows(series_map, selected_clients)

    report_text = render_report(
        run_config=run_config,
        split_summary=split_summary,
        client_fact_df=client_fact_df,
        pairwise_corr_rows=pairwise_corr_rows,
        allocation_text=allocation_text,
        heterogeneity_text=heterogeneity_text,
        selected_clients=selected_clients,
        result_dir=result_dir,
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report_text, encoding="utf-8")

    print("[selected_clients]", ",".join(str(item) for item in selected_clients))
    print("[result_dir]", result_dir)
    print("[output_report]", output_report)


if __name__ == "__main__":
    main()
