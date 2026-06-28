"""Read-only summary for experiment 1 FedAvg metric optimization smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from real_data_experiments.common.io_utils import resolve_path
from real_data_experiments.common.metrics import compute_regression_metrics
from real_data_experiments.common.tensor_dataset import load_grid_tensor_bundle
from real_data_experiments.single_intersection_client.sic_fedavg_gap_diagnosis import (
    build_test_dataset,
    compute_naive_arrays,
)


LOWER_IS_BETTER = {"mse", "rmse", "mae", "mape", "smape"}
HIGHER_IS_BETTER = {"r2"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only summary for FedAvg metric optimization smoke outputs."
    )
    parser.add_argument("--baseline-dirs", nargs="+", required=True)
    parser.add_argument("--candidate-dirs", nargs="+", required=True)
    parser.add_argument("--output-report", required=True)
    return parser


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


def pipe_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def fmt_num(value: float | int | None, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "未运行 / 目录不存在"
    return f"{float(value):.{digits}f}"


def compare_metric(metric: str, fedavg_value: float | None, naive_value: float | None) -> str:
    if fedavg_value is None or naive_value is None:
        return "未运行 / 目录不存在"
    if metric in LOWER_IS_BETTER:
        return "是" if fedavg_value < naive_value else "否"
    if metric in HIGHER_IS_BETTER:
        return "是" if fedavg_value > naive_value else "否"
    return "否"


def compute_naive_outputs(run_dir: Path, run_config: dict, split_summary: dict) -> tuple[dict[str, float], pd.DataFrame]:
    bundle = load_grid_tensor_bundle(
        tensor_path=resolve_path(run_config["tensor_path"]),
        regions_path=resolve_path(run_config["regions_path"]),
    )
    selected_clients = [int(item) for item in run_config["selected_clients"]]
    sequence_length = int(run_config["sequence_length"])
    prediction_horizon = int(run_config["prediction_horizon"])
    target_channel = int(run_config["target_channel"])
    use_channels = [int(item) for item in run_config["use_channels"]]

    global_y_true = []
    global_y_pred = []
    client_rows = []

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
        global_y_true.extend(y_true.tolist())
        global_y_pred.extend(y_pred.tolist())
        client_rows.append(
            {
                "region_id": int(region_id),
                "naive_rmse": float(metrics["rmse"]),
                "naive_mae": float(metrics["mae"]),
                "naive_mape": float(metrics["mape"]),
                "naive_smape": float(metrics["smape"]),
                "naive_r2": float(metrics["r2"]),
            }
        )

    global_metrics = compute_regression_metrics(global_y_true, global_y_pred)
    return (
        {
            "mse": float(global_metrics["mse"]),
            "rmse": float(global_metrics["rmse"]),
            "mae": float(global_metrics["mae"]),
            "mape": float(global_metrics["mape"]),
            "smape": float(global_metrics["smape"]),
            "r2": float(global_metrics["r2"]),
        },
        pd.DataFrame(client_rows).sort_values("region_id").reset_index(drop=True),
    )


def load_run_summary(run_dir: Path) -> dict:
    if not run_dir.exists():
        return {
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "exists": False,
            "completed": False,
            "device": None,
            "selected_clients": None,
            "k": None,
            "rounds": None,
            "local_epochs": None,
            "learning_rate": None,
            "fedavg_metrics": {},
            "independent_metrics": {},
            "naive_metrics": {},
            "client_compare_df": pd.DataFrame(),
        }

    required_files = {
        "run_config": run_dir / "run_config.json",
        "main_metrics": run_dir / "main_metrics.csv",
        "client_metrics": run_dir / "client_metrics.csv",
        "split_summary": run_dir / "split_summary.json",
    }
    if not all(path.exists() for path in required_files.values()):
        return {
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "exists": True,
            "completed": False,
            "device": None,
            "selected_clients": None,
            "k": None,
            "rounds": None,
            "local_epochs": None,
            "learning_rate": None,
            "fedavg_metrics": {},
            "independent_metrics": {},
            "naive_metrics": {},
            "client_compare_df": pd.DataFrame(),
        }

    run_config = load_json(required_files["run_config"])
    split_summary = load_json(required_files["split_summary"])
    main_rows = load_csv_rows(required_files["main_metrics"])
    client_rows = load_csv_rows(required_files["client_metrics"])
    main_df = pd.DataFrame(main_rows)
    client_df = pd.DataFrame(client_rows)
    main_df[["mse", "rmse", "mae", "mape", "smape", "r2"]] = main_df[
        ["mse", "rmse", "mae", "mape", "smape", "r2"]
    ].apply(pd.to_numeric, errors="coerce")
    client_df[["rmse", "mae", "mape", "smape", "r2", "region_id"]] = client_df[
        ["rmse", "mae", "mape", "smape", "r2", "region_id"]
    ].apply(pd.to_numeric, errors="coerce")

    fedavg_row = main_df[main_df["method"] == "FedAvg"].iloc[0].to_dict()
    independent_row = main_df[main_df["method"] == "Independent"].iloc[0].to_dict()
    naive_metrics, naive_client_df = compute_naive_outputs(run_dir, run_config, split_summary)

    fedavg_client_df = (
        client_df[client_df["method"] == "FedAvg"][
            ["region_id", "rmse", "mae", "mape", "smape", "r2"]
        ]
        .rename(
            columns={
                "rmse": "fedavg_rmse",
                "mae": "fedavg_mae",
                "mape": "fedavg_mape",
                "smape": "fedavg_smape",
                "r2": "fedavg_r2",
            }
        )
        .sort_values("region_id")
        .reset_index(drop=True)
    )
    client_compare_df = fedavg_client_df.merge(naive_client_df, on="region_id", how="left")
    client_compare_df["fedavg_minus_naive_rmse"] = (
        client_compare_df["fedavg_rmse"] - client_compare_df["naive_rmse"]
    )
    client_compare_df["fedavg_better_than_naive_rmse"] = (
        client_compare_df["fedavg_rmse"] < client_compare_df["naive_rmse"]
    )

    return {
        "name": run_dir.name,
        "run_dir": str(run_dir),
        "exists": True,
        "completed": True,
        "device": run_config.get("device"),
        "selected_clients": ",".join(str(item) for item in run_config["selected_clients"]),
        "k": int(run_config["num_clients"]),
        "rounds": int(run_config["communication_rounds"]),
        "local_epochs": int(run_config["local_epochs"]),
        "learning_rate": float(run_config["learning_rate"]),
        "fedavg_metrics": {metric: float(fedavg_row[metric]) for metric in ["mse", "rmse", "mae", "mape", "smape", "r2"]},
        "independent_metrics": {metric: float(independent_row[metric]) for metric in ["mse", "rmse", "mae", "mape", "smape", "r2"]},
        "naive_metrics": naive_metrics,
        "client_compare_df": client_compare_df,
    }


def build_candidate_rows(runs: list[dict]) -> list[list[object]]:
    rows = []
    for item in runs:
        rows.append(
            [
                item["name"],
                item["selected_clients"] or "未运行 / 目录不存在",
                item["k"] if item["k"] is not None else "未运行 / 目录不存在",
                item["rounds"] if item["rounds"] is not None else "未运行 / 目录不存在",
                item["local_epochs"] if item["local_epochs"] is not None else "未运行 / 目录不存在",
                fmt_num(item["learning_rate"], 4),
                item["run_dir"],
                "是" if item["completed"] else "未运行 / 目录不存在",
                item["device"] or "未运行 / 目录不存在",
            ]
        )
    return rows


def build_main_metric_rows(runs: list[dict]) -> list[list[object]]:
    rows = []
    for item in runs:
        fed = item["fedavg_metrics"]
        naive = item["naive_metrics"]
        rows.append(
            [
                item["name"],
                fmt_num(fed.get("mse")),
                fmt_num(fed.get("rmse")),
                fmt_num(fed.get("mae")),
                fmt_num(fed.get("mape")),
                fmt_num(fed.get("smape")),
                fmt_num(fed.get("r2")),
                fmt_num(naive.get("rmse")),
                fmt_num(naive.get("mae")),
                fmt_num(naive.get("mape")),
                fmt_num(naive.get("smape")),
                fmt_num(naive.get("r2")),
                compare_metric("rmse", fed.get("rmse"), naive.get("rmse")),
                compare_metric("mae", fed.get("mae"), naive.get("mae")),
                compare_metric("mape", fed.get("mape"), naive.get("mape")),
                compare_metric("smape", fed.get("smape"), naive.get("smape")),
                compare_metric("r2", fed.get("r2"), naive.get("r2")),
                "是"
                if all(
                    compare_metric(metric, fed.get(metric), naive.get(metric)) == "是"
                    for metric in ["rmse", "mae", "mape", "smape", "r2"]
                )
                else "否",
            ]
        )
    return rows


def build_delta_rows(runs: list[dict], r60_run: dict | None) -> list[list[object]]:
    rows = []
    for item in runs:
        if r60_run is None or not item["completed"]:
            rows.append(
                [
                    item["name"],
                    "未运行 / 目录不存在",
                    "未运行 / 目录不存在",
                    "未运行 / 目录不存在",
                    "未运行 / 目录不存在",
                    "未运行 / 目录不存在",
                ]
            )
            continue
        fed = item["fedavg_metrics"]
        base = r60_run["fedavg_metrics"]
        rows.append(
            [
                item["name"],
                fmt_num(base["rmse"] - fed["rmse"]),
                fmt_num(base["mae"] - fed["mae"]),
                fmt_num(base["mape"] - fed["mape"]),
                fmt_num(base["smape"] - fed["smape"]),
                fmt_num(fed["r2"] - base["r2"]),
            ]
        )
    return rows


def build_per_client_section(runs: list[dict]) -> list[str]:
    lines: list[str] = []
    for item in runs:
        lines.append(f"### {item['name']}")
        if not item["completed"]:
            lines.append("- 未运行 / 目录不存在。")
            lines.append("")
            continue
        compare_df = item["client_compare_df"].copy()
        drag_row = compare_df.sort_values("fedavg_minus_naive_rmse", ascending=False).iloc[0]
        better_count = int(compare_df["fedavg_better_than_naive_rmse"].sum())
        rows = []
        for _, row in compare_df.iterrows():
            rows.append(
                [
                    int(row["region_id"]),
                    fmt_num(row["fedavg_rmse"]),
                    fmt_num(row["naive_rmse"]),
                    fmt_num(row["fedavg_minus_naive_rmse"]),
                    "是" if bool(row["fedavg_better_than_naive_rmse"]) else "否",
                ]
            )
        lines.append(
            pipe_table(
                ["region_id", "FedAvg_RMSE", "Naive_RMSE", "FedAvg_minus_Naive_RMSE", "FedAvg_RMSE优于Naive"],
                rows,
            )
        )
        lines.append(
            f"- 该方案中相对 Naive 拖累最大的 client 是 `region {int(drag_row['region_id'])}`，"
            f"`FedAvg_minus_Naive_RMSE={fmt_num(float(drag_row['fedavg_minus_naive_rmse']))}`。"
        )
        lines.append(f"- FedAvg 在 `RMSE` 上优于 Naive 的 client 数为 `{better_count}/{len(compare_df)}`。")
        if item["k"] == 4:
            lines.append("- 该方案属于 `exclude-289` 异质性诊断，不能直接替代当前 K=5 正式结果。")
        lines.append("")
    return lines


def choose_best_run(runs: list[dict]) -> tuple[dict | None, str]:
    completed = [item for item in runs if item["completed"]]
    if not completed:
        return None, "没有候选方案完成。"

    fully_better = [
        item
        for item in completed
        if all(
            compare_metric(metric, item["fedavg_metrics"].get(metric), item["naive_metrics"].get(metric)) == "是"
            for metric in ["rmse", "mae", "mape", "smape", "r2"]
        )
    ]
    if fully_better:
        best = sorted(fully_better, key=lambda item: item["fedavg_metrics"]["rmse"])[0]
        return best, "存在全面超过 NaiveLastValue 的方案，按更优 RMSE 选取。"

    non_k4 = [item for item in completed if item["k"] != 4]
    candidate_pool = non_k4 if non_k4 else completed

    def score(item: dict) -> tuple:
        improvements = 0
        for metric in ["mae", "mape", "smape", "r2"]:
            if compare_metric(metric, item["fedavg_metrics"].get(metric), item["naive_metrics"].get(metric)) == "是":
                improvements += 1
        rmse_safe = compare_metric("rmse", item["fedavg_metrics"].get("rmse"), item["naive_metrics"].get("rmse")) == "是"
        return (1 if rmse_safe else 0, improvements, -item["fedavg_metrics"]["mae"], -item["fedavg_metrics"]["mape"], item["fedavg_metrics"]["r2"])

    best = sorted(candidate_pool, key=score, reverse=True)[0]
    return best, "没有方案全面超过 NaiveLastValue，按 MAE/MAPE/SMAPE/R2 改善且 RMSE 不退化的优先级选取。"


def render_report(baselines: list[dict], candidates: list[dict]) -> str:
    name_map = {item["name"]: item for item in baselines + candidates}
    r60_run = name_map.get("experiment1_fedavg_rounds_smoke_r60_cuda")
    best_run, best_reason = choose_best_run(candidates)

    any_full = any(
        item["completed"]
        and all(
            compare_metric(metric, item["fedavg_metrics"].get(metric), item["naive_metrics"].get(metric)) == "是"
            for metric in ["rmse", "mae", "mape", "smape", "r2"]
        )
        for item in candidates
    )
    k4_run = next((item for item in candidates if item["k"] == 4 and item["completed"]), None)
    k5_runs = [item for item in candidates if item["k"] == 5 and item["completed"]]
    k5_full = any(
        all(
            compare_metric(metric, item["fedavg_metrics"].get(metric), item["naive_metrics"].get(metric)) == "是"
            for metric in ["rmse", "mae", "mape", "smape", "r2"]
        )
        for item in k5_runs
    )

    recommendation = "若 K=5/K=4 都未明显改善：建议停止继续调参，转向 cluster/region client。"
    if k5_full:
        recommendation = "若 K=5 有方案全面超过 naive：建议固定该 smoke 方案，后续做正式复跑。"
    elif k4_run is not None:
        recommendation = "若 K=5 只部分改善但 K=4 明显改善：建议进入 replace-289 候选选择或 cluster/region client。"

    lines = [
        "# 实验 1：FedAvg 多指标优化 smoke 报告",
        "",
        "## 1. 目的",
        "",
        "本报告用于判断 FedAvg 是否能在不改变聚合公式和模型结构的前提下，通过 rounds、local_epochs、learning_rate 和 client 组织诊断，改善相对 NaiveLastValue 的 MAE、MAPE、SMAPE 和 R2。",
        "",
        "## 2. 边界",
        "",
        "- 不删除 NaiveLastValue；",
        "- 不删除 289；",
        "- 不修改 FedAvg；",
        "- 不修改模型结构；",
        "- 不修改数据划分；",
        "- 不进入新实验 2-6；",
        "- 所有结果均为 diagnostics/smoke。",
        "",
        "## 3. 候选方案",
        "",
        pipe_table(
            ["name", "selected_clients", "K", "rounds", "local_epochs", "learning_rate", "output_dir", "是否完成", "device"],
            build_candidate_rows(candidates),
        ),
        "",
        "## 4. 主指标对比",
        "",
        pipe_table(
            [
                "name",
                "FedAvg_MSE",
                "FedAvg_RMSE",
                "FedAvg_MAE",
                "FedAvg_MAPE",
                "FedAvg_SMAPE",
                "FedAvg_R2",
                "Naive_RMSE",
                "Naive_MAE",
                "Naive_MAPE",
                "Naive_SMAPE",
                "Naive_R2",
                "RMSE优于Naive",
                "MAE优于Naive",
                "MAPE优于Naive",
                "SMAPE优于Naive",
                "R2优于Naive",
                "全面超过Naive",
            ],
            build_main_metric_rows(candidates),
        ),
        "",
        "## 5. 相对 r60 的改善",
        "",
        pipe_table(
            ["name", "RMSE改善量", "MAE改善量", "MAPE改善量", "SMAPE改善量", "R2改善量"],
            build_delta_rows(candidates, r60_run),
        ),
        "",
        "## 6. per-client 诊断",
        "",
        *build_per_client_section(candidates),
        "## 7. 最优方案判断",
        "",
        f"- 当前最佳方案：`{best_run['name']}`。" if best_run is not None else "- 当前没有可比较的完成方案。",
        f"- 判断依据：{best_reason}",
        "- 不只按 RMSE 判断，而是优先看是否全面超过 Naive；若不能，则看 MAE/MAPE/SMAPE/R2 改善且 RMSE 不退化。",
        "- 若 `K=4 exclude-289` 更好，也只能作为异质性诊断，不能直接替代当前 K=5 正式结果。",
        "",
        "## 8. 结论",
        "",
        f"- 是否有方案全面超过 NaiveLastValue：{'是' if any_full else '否'}。",
        f"- 如果没有，当前最佳方案主要改善的方向：`{best_run['name']}`。" if best_run is not None else "- 如果没有，当前还无法给出稳定最佳方案。",
        "- local_epochs 降低是否有效：需结合 K=5 e1/e2 对比判断；e1 更偏向抑制 non-IID 本地漂移。",
        "- lr 降低是否有效：本阶段候选固定为 `0.0005`，相对先前正式设置属于更保守学习率。",
        "- 继续增加 rounds 是否仍有效：需结合 `r80` 与 `r100` 对比判断；若只带来 RMSE 小幅改善而其他指标停滞，则继续增 rounds 的边际收益有限。",
        "- 289 是否仍是关键问题：若 K=5 方案中仍由 `289` 贡献最大 FedAvg-vs-Naive gap，则答案为是。",
        "- 是否建议进入后续实验：若 K=5 与 K=4 都不能全面改善多指标，应建议进入新实验 3/4 的 grouped-client 线或新实验 5/6 的全局覆盖式划分线。",
        "- 是否建议继续直接调 K=5：仅当 K=5 出现明显多指标改善且未退化 RMSE 时才继续，否则应降低优先级。",
        "",
        "## 9. 推荐下一步",
        "",
        f"- {recommendation}",
        "",
        "## 10. 边界声明",
        "",
        "本阶段只做实验 1 范围内 FedAvg 多指标优化 smoke；未修改 FedAvg 聚合公式，未修改模型结构，未修改数据划分，未运行新实验 2-6，未提交 results。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()

    baseline_dirs = [resolve_path(path) for path in args.baseline_dirs]
    candidate_dirs = [resolve_path(path) for path in args.candidate_dirs]
    output_report = resolve_path(args.output_report)

    baselines = [load_run_summary(path) for path in baseline_dirs]
    candidates = [load_run_summary(path) for path in candidate_dirs]

    report = render_report(baselines, candidates)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")

    print("[baseline_count]", len(baselines))
    print("[candidate_count]", len(candidates))
    print("[output_report]", output_report)


if __name__ == "__main__":
    main()
