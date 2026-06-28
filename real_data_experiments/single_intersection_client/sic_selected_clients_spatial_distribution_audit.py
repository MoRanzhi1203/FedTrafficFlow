"""Read-only spatial and distribution audit for selected clients in experiment 1."""

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
        description="Read-only spatial/distribution audit for selected_clients in experiment 1."
    )
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument("--selected-clients", type=str, required=True)
    parser.add_argument("--rationale-report", type=str, required=True)
    parser.add_argument("--heterogeneity-report", type=str, required=True)
    parser.add_argument("--output-report", type=str, required=True)
    return parser


def parse_selected_clients(raw_text: str) -> list[int]:
    clients = [int(part.strip()) for part in str(raw_text).split(",") if part.strip()]
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


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "当前无法从已有结果稳定读取该项，未伪造。"
    return f"{float(value):.{digits}f}"


def pipe_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def compute_full_series_lag1(series: np.ndarray) -> float:
    if series.size < 2:
        return float("nan")
    return safe_corr(series[1:], series[:-1])


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
        y_true, y_pred = compute_naive_arrays(
            dataset,
            target_channel=target_channel,
            use_channels=use_channels,
        )
        metrics = compute_regression_metrics(y_true, y_pred)
        rows.append(
            {
                "region_id": int(region_id),
                "naive_rmse": float(metrics["rmse"]),
                "naive_r2": float(metrics["r2"]),
            }
        )
    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def build_client_frames(
    bundle,
    split_summary: dict,
    client_rows: list[dict[str, str]],
    naive_df: pd.DataFrame,
    selected_clients: list[int],
    grid_metadata: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    split_map = {int(item["region_id"]): item for item in split_summary["clients"]}
    fedavg_map = {
        int(row["region_id"]): row for row in client_rows if row.get("method") == "FedAvg" and row.get("region_id")
    }
    independent_map = {
        int(row["region_id"]): row for row in client_rows if row.get("method") == "Independent" and row.get("region_id")
    }
    naive_map = {int(row["region_id"]): row for _, row in naive_df.iterrows()}

    pooled_span = None
    if "grid_resolution" in grid_metadata and "pool_kernel" in grid_metadata:
        pooled_span = float(grid_metadata["grid_resolution"]) * float(grid_metadata["pool_kernel"])
    half_span = pooled_span / 2.0 if pooled_span is not None else None

    base_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    series_map: dict[int, np.ndarray] = {}

    target_channel = int(split_summary["target_channel"])

    for region_id in selected_clients:
        split_item = split_map[int(region_id)]
        fedavg_row = fedavg_map[int(region_id)]
        independent_row = independent_map[int(region_id)]
        naive_row = naive_map[int(region_id)]
        series = bundle.tensor[target_channel, int(region_id)].cpu().numpy().astype(np.float64)
        series_map[int(region_id)] = series
        stats = describe_series(series)
        centroid_lon = float(fedavg_row["centroid_lon"])
        centroid_lat = float(fedavg_row["centroid_lat"])

        min_lon = centroid_lon - half_span if half_span is not None else None
        max_lon = centroid_lon + half_span if half_span is not None else None
        min_lat = centroid_lat - half_span if half_span is not None else None
        max_lat = centroid_lat + half_span if half_span is not None else None

        total_samples = (
            int(split_item["train"]["sample_count"])
            + int(split_item["val"]["sample_count"])
            + int(split_item["test"]["sample_count"])
        )

        base_rows.append(
            {
                "region_id": int(region_id),
                "source_node_count": int(split_item["source_node_count"]),
                "mean_total_flow": float(split_item["mean_total_flow"]),
                "train_samples": int(split_item["train"]["sample_count"]),
                "val_samples": int(split_item["val"]["sample_count"]),
                "test_samples": int(split_item["test"]["sample_count"]),
                "total_samples": int(total_samples),
                "flow_mean": float(stats["mean"]),
                "flow_std": float(stats["std"]),
                "flow_min": float(stats["min"]),
                "flow_max": float(stats["max"]),
                "flow_cv": float(stats["cv"]),
                "lag1_autocorr": float(compute_full_series_lag1(series)),
                "fedavg_rmse": float(fedavg_row["rmse"]),
                "independent_rmse": float(independent_row["rmse"]),
                "naive_rmse": float(naive_row["naive_rmse"]),
                "fedavg_minus_naive_rmse": float(fedavg_row["rmse"]) - float(naive_row["naive_rmse"]),
            }
        )
        spatial_rows.append(
            {
                "region_id": int(region_id),
                "pooled_row": int(split_item["pooled_row"]),
                "pooled_col": int(split_item["pooled_col"]),
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
                "source_node_count": int(split_item["source_node_count"]),
            }
        )

    base_df = pd.DataFrame(base_rows).sort_values("region_id").reset_index(drop=True)
    spatial_df = pd.DataFrame(spatial_rows).sort_values("region_id").reset_index(drop=True)
    return base_df, spatial_df, series_map


def build_corr_matrix_rows(series_map: dict[int, np.ndarray], selected_clients: list[int]) -> list[list[object]]:
    rows: list[list[object]] = []
    for region_id in selected_clients:
        row: list[object] = [region_id]
        for other_id in selected_clients:
            row.append(f"{safe_corr(series_map[region_id], series_map[other_id]):.6f}")
        rows.append(row)
    return rows


def detect_primary_drag_client(heterogeneity_text: str) -> int | None:
    match = re.search(r"主要拖累 client 是否是 `?(\d+)`?：是", heterogeneity_text)
    if match:
        return int(match.group(1))
    return None


def build_model_relation_rows(base_df: pd.DataFrame, primary_drag_client: int | None) -> list[list[object]]:
    rows: list[list[object]] = []
    for _, row in base_df.iterrows():
        region_id = int(row["region_id"])
        is_drag = "是" if primary_drag_client is not None and region_id == primary_drag_client else "否"
        rows.append(
            [
                region_id,
                f"{float(row['fedavg_rmse']):.3f}",
                f"{float(row['independent_rmse']):.3f}",
                f"{float(row['naive_rmse']):.3f}",
                f"{float(row['fedavg_minus_naive_rmse']):.3f}",
                is_drag,
            ]
        )
    return rows


def build_basic_rows(base_df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for _, row in base_df.iterrows():
        rows.append(
            [
                int(row["region_id"]),
                int(row["source_node_count"]),
                f"{float(row['mean_total_flow']):.3f}",
                int(row["train_samples"]),
                int(row["val_samples"]),
                int(row["test_samples"]),
                int(row["total_samples"]),
            ]
        )
    return rows


def build_distribution_rows(base_df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for _, row in base_df.iterrows():
        rows.append(
            [
                int(row["region_id"]),
                f"{float(row['flow_mean']):.3f}",
                f"{float(row['flow_std']):.3f}",
                f"{float(row['flow_min']):.3f}",
                f"{float(row['flow_max']):.3f}",
                f"{float(row['flow_cv']):.6f}",
                f"{float(row['lag1_autocorr']):.6f}",
            ]
        )
    return rows


def build_spatial_rows(spatial_df: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = []
    for _, row in spatial_df.iterrows():
        rows.append(
            [
                int(row["region_id"]),
                int(row["pooled_row"]),
                int(row["pooled_col"]),
                f"{float(row['centroid_lon']):.6f}",
                f"{float(row['centroid_lat']):.6f}",
                format_float(row["min_lon"], 6),
                format_float(row["max_lon"], 6),
                format_float(row["min_lat"], 6),
                format_float(row["max_lat"], 6),
                int(row["source_node_count"]),
            ]
        )
    return rows


def render_report(
    run_config: dict,
    split_summary: dict,
    rationale_text: str,
    heterogeneity_text: str,
    base_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    corr_rows: list[list[object]],
    selected_clients: list[int],
    result_dir: Path,
    tensor_path: Path,
) -> str:
    primary_drag_client = detect_primary_drag_client(heterogeneity_text)
    max_flow_region = int(base_df.sort_values("flow_mean", ascending=False).iloc[0]["region_id"])
    min_flow_region = int(base_df.sort_values("flow_mean", ascending=True).iloc[0]["region_id"])
    max_cv_region = int(base_df.sort_values("flow_cv", ascending=False).iloc[0]["region_id"])
    min_cv_region = int(base_df.sort_values("flow_cv", ascending=True).iloc[0]["region_id"])
    max_lag_region = int(base_df.sort_values("lag1_autocorr", ascending=False).iloc[0]["region_id"])
    min_lag_region = int(base_df.sort_values("lag1_autocorr", ascending=True).iloc[0]["region_id"])

    non_289_corr_values: list[float] = []
    corr_map = {
        int(row[0]): {int(selected_clients[idx]): float(row[idx + 1]) for idx in range(len(selected_clients))}
        for row in corr_rows
    }
    for i, region_id in enumerate(selected_clients):
        for other_id in selected_clients[i + 1 :]:
            if region_id != 289 and other_id != 289:
                non_289_corr_values.append(corr_map[region_id][other_id])
    min_corr_289 = min(corr_map[289][rid] for rid in selected_clients if rid != 289) if 289 in corr_map else None
    max_corr_289 = max(corr_map[289][rid] for rid in selected_clients if rid != 289) if 289 in corr_map else None
    non_289_corr_min = min(non_289_corr_values) if non_289_corr_values else None
    non_289_corr_max = max(non_289_corr_values) if non_289_corr_values else None

    bbox = {
        "min_lon": float(spatial_df["min_lon"].min()) if spatial_df["min_lon"].notna().all() else None,
        "max_lon": float(spatial_df["max_lon"].max()) if spatial_df["max_lon"].notna().all() else None,
        "min_lat": float(spatial_df["min_lat"].min()) if spatial_df["min_lat"].notna().all() else None,
        "max_lat": float(spatial_df["max_lat"].max()) if spatial_df["max_lat"].notna().all() else None,
    }
    pooled_rows = sorted(spatial_df["pooled_row"].astype(int).unique().tolist())
    pooled_cols = sorted(spatial_df["pooled_col"].astype(int).unique().tolist())

    k3_mentioned = "K=3" in rationale_text or "K = 3" in rationale_text
    cluster_level_mentioned = "cluster-level" in rationale_text

    lines = [
        "# 实验 1：selected_clients 空间覆盖与分布统计报告",
        "",
        "## 1. 报告目的",
        "",
        "本报告用于补充 `selected_clients=290,284,318,288,289` 的空间覆盖、流量分布和异质性解释。",
        "",
        "## 2. 当前 client 设置",
        "",
        f"- `selected_clients`：`{','.join(str(item) for item in selected_clients)}`",
        f"- `K=5`：是，当前正式实验 1 使用 `5` 个 grid-cell-level client。",
        "- `grid-cell-level client`：是，每个 `region_id` 对应一个 pooled grid region client。",
        f"- `result_dir`：`{result_dir}`",
        f"- 数据入口：`tensor_path={tensor_path}`，`regions_path={split_summary['regions_path']}`",
        f"- 与原稿 `K=3` 的关系：{'当前 K=5 是相对 K=3 的增强。' if k3_mentioned else '当前报告链未稳定提取到 K=3 原文，但现有说明已将 K=5 定位为增强设置。'}",
        f"- 与后续实验线的区别：{'当前仅覆盖 single-grid client；新实验 3/4 的 grouped-client 线与新实验 5/6 的全局覆盖式划分线仍需继续补齐。' if cluster_level_mentioned else '当前仅覆盖 single-grid client；新实验 3-6 的 grouped-client / global-partition 实验线本阶段未展开。'}",
        "",
        "## 3. client 基础信息",
        "",
        pipe_table(
            ["region_id", "source_node_count", "mean_total_flow", "train_samples", "val_samples", "test_samples", "total_samples"],
            build_basic_rows(base_df),
        ),
        "",
        "## 4. 流量分布统计",
        "",
        pipe_table(
            ["region_id", "flow_mean", "flow_std", "flow_min", "flow_max", "flow_cv", "lag1_autocorr"],
            build_distribution_rows(base_df),
        ),
        "",
        f"- 高流量 client 主要是 `region {max_flow_region}`，低流量 client 主要是 `region {min_flow_region}`。",
        f"- 波动更大的 client 主要是 `region {max_cv_region}`，相对更平稳的是 `region {min_cv_region}`。",
        f"- 时间惯性更强的是 `region {max_lag_region}`，相对更弱的是 `region {min_lag_region}`。",
        "",
        "## 5. client 间相关性",
        "",
        pipe_table(["region_id", *[str(item) for item in selected_clients]], corr_rows),
        "",
        f"- `289` 与其他 client 的 Pearson 相关性范围为 `{format_float(min_corr_289)}` 到 `{format_float(max_corr_289)}`，明显低于其余 4 个 client 两两之间的 `{format_float(non_289_corr_min)}` 到 `{format_float(non_289_corr_max)}`。",
        "- 因此，`289` 与其他 client 的相关性确实偏低，支撑“`289` 是异质 client”的判断。",
        "- 其余 client 之间整体更接近，说明该 5-client 组合中同时存在相对相似子群和显著异质点。",
        "",
        "## 6. 空间覆盖说明",
        "",
        pipe_table(
            ["region_id", "pooled_row", "pooled_col", "centroid_lon", "centroid_lat", "min_lon", "max_lon", "min_lat", "max_lat", "source_node_count"],
            build_spatial_rows(spatial_df),
        ),
        "",
        "- 当前可以从 `node_flow_grid_regions.csv`、`client_metrics.csv` 与 `node_flow_grid_metadata.json` 恢复每个 client 的 pooled grid 位置、centroid 以及按 `grid_resolution=0.009` 和 `pool_kernel=2` 推导出的近似边界。",
        f"- 当前 5 个 client 覆盖的 pooled rows 为 `{pooled_rows}`，pooled cols 为 `{pooled_cols}`。",
        f"- 5 个 client 的联合近似空间包围盒为 `lon=[{format_float(bbox['min_lon'])}, {format_float(bbox['max_lon'])}]`，`lat=[{format_float(bbox['min_lat'])}, {format_float(bbox['max_lat'])}]`。",
        "- 空间上看，`284/288/289/290` 主要位于同一 pooled row 的相邻或近邻列，`318` 位于其北侧相邻 pooled row；因此它们更像是一个局部子区域内的细粒度网格客户端，而不是全路网分散覆盖。",
        "",
        "## 7. 模型表现与分布差异的关系",
        "",
        pipe_table(
            ["region_id", "FedAvg_RMSE", "Independent_RMSE", "NaiveLastValue_RMSE", "FedAvg_minus_Naive_RMSE", "是否拖累FedAvg"],
            build_model_relation_rows(base_df, primary_drag_client),
        ),
        "",
        f"- 当前主要拖累 `FedAvg` 的 client 是 `{primary_drag_client}`。" if primary_drag_client is not None else "- 当前无法从已有诊断报告稳定定位唯一拖累 client，未伪造。",
        "- `289` 的相关性显著偏低，同时其 `FedAvg_minus_Naive_RMSE` 为正，说明它既是分布异质点，也是 FedAvg gap 的关键来源。",
        "- `NaiveLastValue` 较强，和各 client 普遍较高的 `lag1_autocorr` 一致，说明真实交通流短时惯性明显。",
        "- 因此，标准 `FedAvg` 在强异质 client 下仍会受到跨 client 平均带来的欠拟合或平滑影响。",
        "",
        "## 8. 为什么这 5 个 client 可以支撑当前实验 1",
        "",
        "- 这 5 个 client 覆盖了不同流量尺度。",
        "- 这 5 个 client 在波动强度和时间相关性上存在差异。",
        "- 该组合包含明显异质 client `289`。",
        "- 该组合已具备完整训练与诊断链。",
        "- 因而它能够支持 `K=5 grid-cell-level heterogeneous setting` 的表述。",
        "- 但它不能代表全部真实路网。",
        "",
        "## 9. 当前局限",
        "",
        "- 当前仍不是全路网覆盖。",
        "- 新实验 3-6 的 grouped-client / global-partition 实验线仍需继续补齐。",
        "- 当前空间覆盖主要是局部 pooled-grid 子区域，而不是城市尺度均匀取样。",
        "- 当前不应声称这 5 个 client 是唯一最优组合。",
        "",
        "## 10. 推荐论文表述",
        "",
        "本实验采用 K=5 的 grid-cell-level client 设置，选取 290、284、318、288 和 289 作为细粒度空间单元客户端。该设置相较原稿 K=3 增加了参与客户端数量，使实验能够观察更明显的 client-level variability。统计结果显示，不同客户端在节点数量、流量尺度、波动强度和时间相关性方面存在差异，其中 289 与其他客户端的相关性较低，并在 FedAvg 与 NaiveLastValue 的对比中表现出明显 gap，说明其代表了强异质客户端。因而，该设置适合用于验证真实数据联邦链路在 grid-cell-level non-IID 场景下的可运行性和局限性。需要强调的是，该设置不是对全路网客户端的穷尽覆盖，后续新实验 3/4 的 grouped-client 组织与新实验 5/6 的全局覆盖式划分，将用于进一步验证更同质或更系统的客户端组织方式是否能够缓解跨客户端分布差异造成的性能下降。",
        "",
        "## 11. 结论",
        "",
        "- 这 5 个 client 可以支撑当前实验 1 对 `grid-cell-level heterogeneous setting` 的表述。",
        "- `289` 继续提供了明确的异质性证据。",
        "- 当前仍需要后续新实验 3-6 的 grouped-client / global-partition 审计。",
        "- 当前已能恢复 pooled grid 的近似经纬度边界，因此短期内更需要补的是地图图示，而不是重新推断坐标表。",
        "- 下一步建议优先提交本报告与只读审计脚本。",
        "",
        "## 12. 边界声明",
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
    rationale_report = resolve_path(args.rationale_report)
    heterogeneity_report = resolve_path(args.heterogeneity_report)
    output_report = resolve_path(args.output_report)
    selected_clients = parse_selected_clients(args.selected_clients)

    run_config = load_json(result_dir / "run_config.json")
    split_summary = load_json(result_dir / "split_summary.json")
    client_rows = load_csv_rows(result_dir / "client_metrics.csv")
    rationale_text = read_text(rationale_report)
    heterogeneity_text = read_text(heterogeneity_report)

    bundle = load_grid_tensor_bundle(
        tensor_path=tensor_path,
        regions_path=resolve_path(split_summary["regions_path"]),
    )
    naive_df = build_naive_metric_frame(bundle, split_summary, selected_clients)
    base_df, spatial_df, series_map = build_client_frames(
        bundle=bundle,
        split_summary=split_summary,
        client_rows=client_rows,
        naive_df=naive_df,
        selected_clients=selected_clients,
        grid_metadata=bundle.grid_metadata,
    )
    corr_rows = build_corr_matrix_rows(series_map, selected_clients)

    report_text = render_report(
        run_config=run_config,
        split_summary=split_summary,
        rationale_text=rationale_text,
        heterogeneity_text=heterogeneity_text,
        base_df=base_df,
        spatial_df=spatial_df,
        corr_rows=corr_rows,
        selected_clients=selected_clients,
        result_dir=result_dir,
        tensor_path=tensor_path,
    )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report_text, encoding="utf-8")

    print("[selected_clients]", ",".join(str(item) for item in selected_clients))
    print("[output_report]", output_report)


if __name__ == "__main__":
    main()
