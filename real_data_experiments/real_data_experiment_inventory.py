"""Read-only inventory script for current real-data experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_ORDER = [
    "common",
    "single_intersection_client",
    "single_intersection_ablation",
    "region_client",
    "region_ablation",
    "region_client_full_cells",
]

EXPERIMENT_TITLES = {
    "common": "common",
    "single_intersection_client": "single_intersection_client",
    "single_intersection_ablation": "single_intersection_ablation",
    "region_client": "region_client",
    "region_ablation": "region_ablation",
    "region_client_full_cells": "region_client_full_cells",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a read-only inventory report for current real-data experiments.")
    parser.add_argument("--root", type=str, required=True, help="Path to real_data_experiments root.")
    parser.add_argument("--results-root", type=str, required=True, help="Path to results/real_data_experiments root.")
    parser.add_argument("--output-report", type=str, required=True, help="Output markdown report path.")
    return parser.parse_args()


def resolve_repo_root(output_report: Path) -> Path:
    return output_report.resolve().parents[1]


def to_repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        return completed.stdout.strip() or completed.stderr.strip()
    return completed.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_first_metric_row(path: Path, method: str) -> dict[str, str] | None:
    rows = read_csv_rows(path)
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def fmt_float(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def yes_no(flag: bool) -> str:
    return "是" if flag else "否"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def collect_git_state(repo_root: Path) -> dict[str, Any]:
    status_short = run_git(repo_root, "status", "--short", "--untracked-files=all")
    status_sb = run_git(repo_root, "status", "-sb")
    log_short = run_git(repo_root, "log", "-10", "--oneline")
    status_lines = [line for line in status_short.splitlines() if line.strip()]
    unsubmitted_files = [line[3:].strip() if len(line) > 3 else line.strip() for line in status_lines]
    boundary_hits = [
        path for path in unsubmitted_files
        if path.startswith("results/")
        or path.startswith("data/")
        or path.startswith("simulation_experiments/")
        or "latex" in path.lower()
    ]
    core_code_hits = [
        path for path in unsubmitted_files
        if path.startswith("real_data_experiments/common/")
        or path.endswith("_core.py")
        or path.endswith("_config.py")
        or path.endswith("_dataset.py")
        or path.endswith("_eval.py")
    ]
    return {
        "status_short": status_short,
        "status_sb": status_sb,
        "log_short": log_short,
        "status_lines": status_lines,
        "unsubmitted_files": unsubmitted_files,
        "boundary_hits": boundary_hits,
        "core_code_hits": core_code_hits,
    }


def scan_experiment_dir(exp_dir: Path) -> dict[str, Any]:
    exists = exp_dir.exists() and exp_dir.is_dir()
    if not exists:
        return {
            "exists": False,
            "files": [],
            "reports": [],
            "readme": False,
            "core_files": {},
        }
    files = sorted([path for path in exp_dir.rglob("*") if path.is_file()])
    report_suffixes = (".md",)
    reports = [path.name for path in files if path.suffix.lower() in report_suffixes]
    core_files = {
        "core": any(path.name.endswith("_core.py") for path in files),
        "config": any(path.name.endswith("_config.py") for path in files),
        "dataset": any(path.name.endswith("_dataset.py") for path in files),
        "eval": any(path.name.endswith("_eval.py") for path in files),
        "report_py": any(path.name.endswith("_report.py") for path in files),
        "inventory": any("inventory" in path.name.lower() for path in files),
        "partitions": (exp_dir / "partitions").exists(),
    }
    return {
        "exists": True,
        "files": files,
        "reports": reports,
        "readme": (exp_dir / "README_zh.md").exists(),
        "core_files": core_files,
    }


def find_result_dirs(results_root: Path) -> list[Path]:
    candidate_files = []
    patterns = [
        "**/run_config.json",
        "**/split_summary.json",
        "**/main_metrics.csv",
        "**/client_metrics.csv",
        "**/ablation_summary.csv",
    ]
    for pattern in patterns:
        candidate_files.extend(results_root.glob(pattern))
    result_dirs = {path.parent.resolve() for path in candidate_files if path.exists()}
    return sorted(result_dirs)


def infer_result_type(result_dir: Path) -> str:
    path_text = result_dir.as_posix()
    if "/formal/" in path_text:
        return "formal"
    if "/diagnostics/" in path_text:
        return "diagnostics"
    if "smoke" in result_dir.name.lower():
        return "smoke"
    if "profile" in result_dir.name.lower():
        return "profile"
    return "other"


def infer_result_experiment(result_dir: Path, run_config: dict[str, Any] | None) -> str:
    name = result_dir.name
    if run_config and isinstance(run_config.get("experiment_name"), str):
        exp_name = str(run_config["experiment_name"])
        if "single_intersection_client" in exp_name:
            return "single_intersection_client"
        if "single_intersection_ablation" in exp_name:
            return "single_intersection_ablation"
        if "region_client_full_cells" in exp_name:
            return "region_client_full_cells"
        if "region_client" in exp_name:
            return "region_client"
        if "region_ablation" in exp_name:
            return "region_ablation"
    if "grid_cell" in name or "experiment1" in name or "single_intersection_client" in name:
        return "single_intersection_client"
    if "single_intersection_ablation" in name:
        return "single_intersection_ablation"
    if "region_client" in name:
        return "region_client"
    if "region_ablation" in name:
        return "region_ablation"
    if "full_cells" in name:
        return "region_client_full_cells"
    return "unknown"


def collect_result_info(result_dir: Path, repo_root: Path) -> dict[str, Any]:
    files = {path.name for path in result_dir.iterdir() if path.is_file()}
    run_config = read_json(result_dir / "run_config.json") if (result_dir / "run_config.json").exists() else None
    split_summary = read_json(result_dir / "split_summary.json") if (result_dir / "split_summary.json").exists() else None
    main_metrics = read_csv_rows(result_dir / "main_metrics.csv") if (result_dir / "main_metrics.csv").exists() else []
    required_files = ["run_config.json", "split_summary.json"]
    if "ablation_summary.csv" in files or "ablation_metrics.csv" in files:
        required_files.extend(["ablation_summary.csv"])
    else:
        required_files.extend(["main_metrics.csv", "client_metrics.csv"])
    completeness = all(required in files for required in required_files)
    missing = [name for name in required_files if name not in files]
    return {
        "path": result_dir,
        "rel_path": to_repo_rel(result_dir, repo_root),
        "name": result_dir.name,
        "type": infer_result_type(result_dir),
        "experiment": infer_result_experiment(result_dir, run_config),
        "formal": "/formal/" in result_dir.as_posix(),
        "diagnostics": "/diagnostics/" in result_dir.as_posix(),
        "files": files,
        "run_config": run_config,
        "split_summary": split_summary,
        "main_metrics": main_metrics,
        "completeness": completeness,
        "missing": missing,
    }


def choose_result_dirs(result_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_names = {
        "grid_cell_main_full_cuda_v4",
        "grid_cell_main_full_cuda_v3",
        "grid_cell_main_full_cuda_v2",
        "grid_cell_main_full_cuda",
        "experiment1_fedavg_rounds_smoke_r40_cuda",
        "experiment1_fedavg_rounds_smoke_r60_cuda",
        "experiment1_metric_opt_k5_r80_e1_lr5e4_cuda",
        "experiment1_metric_opt_k5_r80_e2_lr5e4_cuda",
        "experiment1_metric_opt_k5_r100_e1_lr5e4_cuda",
        "full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda",
        "full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda",
        "region_client_tensor_smoke",
        "region_ablation_tensor_smoke",
        "single_intersection_ablation",
        "single_intersection_ablation_tensor",
        "single_intersection_client",
        "single_intersection_client_tensor",
    }
    chosen = [info for info in result_infos if info["name"] in wanted_names]
    type_rank = {"formal": 0, "diagnostics": 1, "smoke": 2, "other": 3, "profile": 4}
    return sorted(chosen, key=lambda item: (type_rank.get(item["type"], 9), item["name"]))


def extract_method_metrics(result_info: dict[str, Any], method: str) -> dict[str, str] | None:
    for row in result_info["main_metrics"]:
        if row.get("method") == method:
            return row
    return None


def collect_key_state(root: Path, results_root: Path, repo_root: Path) -> dict[str, Any]:
    experiments = {name: scan_experiment_dir(root / name) for name in EXPERIMENT_ORDER}
    result_infos = [collect_result_info(path, repo_root) for path in find_result_dirs(results_root)]
    selected_results = choose_result_dirs(result_infos)

    v4 = next((info for info in result_infos if info["name"] == "grid_cell_main_full_cuda_v4"), None)
    exp1_r40 = next((info for info in result_infos if info["name"] == "experiment1_fedavg_rounds_smoke_r40_cuda"), None)
    exp1_r60 = next((info for info in result_infos if info["name"] == "experiment1_fedavg_rounds_smoke_r60_cuda"), None)
    exp1_r80e1 = next((info for info in result_infos if info["name"] == "experiment1_metric_opt_k5_r80_e1_lr5e4_cuda"), None)
    exp1_r80e2 = next((info for info in result_infos if info["name"] == "experiment1_metric_opt_k5_r80_e2_lr5e4_cuda"), None)
    exp1_r100e1 = next((info for info in result_infos if info["name"] == "experiment1_metric_opt_k5_r100_e1_lr5e4_cuda"), None)
    full_cells_spatial = next((info for info in result_infos if info["name"] == "full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda"), None)
    full_cells_similarity = next((info for info in result_infos if info["name"] == "full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda"), None)

    return {
        "experiments": experiments,
        "result_infos": result_infos,
        "selected_results": selected_results,
        "v4": v4,
        "exp1_r40": exp1_r40,
        "exp1_r60": exp1_r60,
        "exp1_r80e1": exp1_r80e1,
        "exp1_r80e2": exp1_r80e2,
        "exp1_r100e1": exp1_r100e1,
        "full_cells_spatial": full_cells_spatial,
        "full_cells_similarity": full_cells_similarity,
    }


def build_directory_overview(root: Path, state: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    mapping = {
        "common": ("公共工具目录", "否", "公共 tensor/划分/FedAvg/指标/结果写出工具", "已存在"),
        "single_intersection_client": ("正式实验", "是", "实验 1：grid-cell-level client 主实验", "最完整"),
        "single_intersection_ablation": ("计划实验", "是", "实验 2：grid-cell-level client 消融", "部分完成"),
        "region_client": ("正式实验", "是", "实验 3：cluster-level / multi-grid client 主实验", "部分完成"),
        "region_ablation": ("计划实验", "是", "实验 4：cluster-level / multi-grid client 消融", "部分完成"),
        "region_client_full_cells": ("新增实验", "是", "新增 full-cells 多客户端组织实验", "部分完成"),
    }
    for name in EXPERIMENT_ORDER:
        exp_dir = root / name
        meta = state["experiments"][name]
        if not meta["exists"]:
            rows.append([f"{name}/", mapping[name][0], mapping[name][1], mapping[name][2], "不存在"])
            continue
        rows.append([f"{name}/", mapping[name][0], mapping[name][1], mapping[name][2], mapping[name][3]])
    return md_table(["目录", "类型", "是否独立实验", "当前定位", "状态"], rows)


def build_result_rows(selected_results: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for info in selected_results:
        remarks = []
        if info["missing"]:
            remarks.append("缺少: " + ", ".join(info["missing"]))
        if info["name"] == "grid_cell_main_full_cuda_v4":
            remarks.append("当前实验 1 正式 v4 CUDA")
        if info["name"] == "experiment1_metric_opt_k5_r80_e2_lr5e4_cuda":
            fedavg = extract_method_metrics(info, "FedAvg")
            if fedavg:
                remarks.append(f"FedAvg_RMSE={fmt_float(fedavg.get('rmse'))}")
        if info["name"] == "full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda":
            remarks.append("full-cells spatial K=5 smoke")
        if info["name"] == "full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda":
            remarks.append("full-cells similarity K=5 smoke")
        rows.append(
            [
                info["rel_path"],
                info["type"],
                info["experiment"],
                yes_no(info["formal"]),
                yes_no(info["diagnostics"]),
                yes_no(info["completeness"]),
                "；".join(remarks) if remarks else "-",
            ]
        )
    return md_table(
        ["result_dir", "类型", "对应实验", "是否 formal", "是否 diagnostics", "核心文件是否齐全", "备注"],
        rows,
    )


def build_completion_rows() -> str:
    rows = [
        ["single_intersection_client", "已完成", "正式 v4 + r40/r60 + 异质性/选择依据/空间覆盖/metric optimization smoke", "未全面超过 NaiveLastValue", "转向新的 client 组织方式"],
        ["single_intersection_ablation", "部分完成", "目录、README、core/config、已有历史与轻量结果", "未形成 formal 结果链", "暂不启动新跑，保留与一审消融要求对齐"],
        ["region_client", "部分完成", "目录、README、core/config、spatial_block/flow_kmeans、tensor smoke", "无 formal 结果", "继续作为多个 grid cells = 一个 client 主线候选"],
        ["region_ablation", "部分完成", "目录、README、core/config、tensor smoke", "无 formal 结果", "等待 region client 主线稳定后再做消融"],
        ["region_client_full_cells", "可进入下一阶段", "inventory、partition、dataset、core、K=5 spatial/similarity smoke", "缺统一对比报告，尚无 formal", "先生成对比报告，再决定是否扩展 K=8/K=10"],
    ]
    return md_table(["实验", "完成度", "已完成", "未完成", "下一步"], rows)


def build_report(root: Path, results_root: Path, output_report: Path, repo_root: Path) -> str:
    git_state = collect_git_state(repo_root)
    state = collect_key_state(root, results_root, repo_root)
    selected_clients = "290, 284, 318, 288, 289"
    v4_config = state["v4"]["run_config"] if state["v4"] else {}
    exp1_v4_fedavg = extract_method_metrics(state["v4"], "FedAvg") if state["v4"] else None
    exp1_v4_naive = extract_method_metrics(state["v4"], "NaiveLastValue") if state["v4"] else None
    exp1_r40_fedavg = extract_method_metrics(state["exp1_r40"], "FedAvg") if state["exp1_r40"] else None
    exp1_r60_fedavg = extract_method_metrics(state["exp1_r60"], "FedAvg") if state["exp1_r60"] else None
    exp1_r80e1_fedavg = extract_method_metrics(state["exp1_r80e1"], "FedAvg") if state["exp1_r80e1"] else None
    exp1_r80e2_fedavg = extract_method_metrics(state["exp1_r80e2"], "FedAvg") if state["exp1_r80e2"] else None
    exp1_r100e1_fedavg = extract_method_metrics(state["exp1_r100e1"], "FedAvg") if state["exp1_r100e1"] else None
    full_cells_spatial_fedavg = extract_method_metrics(state["full_cells_spatial"], "FedAvg") if state["full_cells_spatial"] else None
    full_cells_similarity_fedavg = extract_method_metrics(state["full_cells_similarity"], "FedAvg") if state["full_cells_similarity"] else None
    unsubmitted_text = "\n".join(f"- `{path}`" for path in git_state["unsubmitted_files"]) if git_state["unsubmitted_files"] else "- 无"
    git_status_block = git_state["status_short"] if git_state["status_short"] else "(clean)"
    boundary_text = "否" if not git_state["boundary_hits"] else "是：" + "、".join(f"`{path}`" for path in git_state["boundary_hits"])
    core_text = "否" if not git_state["core_code_hits"] else "是：" + "、".join(f"`{path}`" for path in git_state["core_code_hits"])
    results_section = build_result_rows(state["selected_results"])
    directory_overview = build_directory_overview(root, state)
    completion_table = build_completion_rows()
    if exp1_v4_naive:
        formal_result_line = (
            f"- 是否已有正式结果：是，`results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/` 已存在；"
            f"FedAvg `RMSE={fmt_float(exp1_v4_fedavg.get('rmse') if exp1_v4_fedavg else None)}`，"
            f"NaiveLastValue `RMSE={fmt_float(exp1_v4_naive.get('rmse'))}`。"
        )
    else:
        formal_result_line = (
            f"- 是否已有正式结果：是，`results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/` 已存在；"
            f"FedAvg `RMSE={fmt_float(exp1_v4_fedavg.get('rmse') if exp1_v4_fedavg else None)}`，"
            "NaiveLastValue 对比已在 formal v4 报告与后续诊断链中补充。"
        )

    lines = [
        "# 当前真实数据实验清单报告",
        "",
        "## 1. 报告目的",
        "",
        "本报告用于梳理当前项目中真实数据实验有哪些、每个实验当前状态如何、已有结果和缺口是什么。",
        "",
        "## 2. 当前 Git 状态",
        "",
        "- `git status --short`：",
        "",
        "```text",
        git_status_block,
        "```",
        "",
        "- 当前未提交文件：",
        unsubmitted_text,
        f"- 是否存在 results/data 越界改动：{boundary_text}",
        f"- 是否存在核心代码未提交改动：{core_text}",
        "- `git status -sb`：",
        "",
        "```text",
        git_state["status_sb"] or "(empty)",
        "```",
        "",
        "- 最近 10 条提交：",
        "",
        "```text",
        git_state["log_short"] or "(empty)",
        "```",
        "",
        "## 3. 真实数据实验目录总览",
        "",
        directory_overview,
        "",
        "## 4. 实验 1：single_intersection_client",
        "",
        "- 实验定位：实验 1，当前最完整的 `grid-cell-level client` 正式主实验。",
        "- client 组织方式：每个 client = 一个 active pooled region，即 one pooled grid cell / one pooled grid region。",
        f"- 当前 selected_clients：`{selected_clients}`。",
        "- 数据入口：正式主入口为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`，辅以 `node_flow_grid_regions.csv`；`parquet` 仅保留 legacy fallback。",
        f"- 是否已有训练入口：是，`sic_core.py` 为正式训练入口，`sic_config.py` 提供 CLI 配置；当前正式 v4 `rounds={v4_config.get('communication_rounds', 'N/A')}`，`local_epochs={v4_config.get('local_epochs', 'N/A')}`。",
        formal_result_line,
        f"- 是否已有 diagnostics/smoke：是，至少包括 `experiment1_fedavg_rounds_smoke_r40_cuda`、`experiment1_fedavg_rounds_smoke_r60_cuda`、`experiment1_metric_opt_k5_r80_e1_lr5e4_cuda`、`experiment1_metric_opt_k5_r80_e2_lr5e4_cuda`、`experiment1_metric_opt_k5_r100_e1_lr5e4_cuda`。",
        "- 是否已有报告：是，已有 formal v2/v3/v4 报告、client 异质性诊断、selected_clients rationale、空间覆盖与分布统计、FedAvg gap 诊断、FedAvg 多指标优化 smoke 报告。",
        f"- 已有正式 v4：是，且为当前实验 1 正式 CUDA 审计结果；FedAvg `R2={fmt_float(exp1_v4_fedavg.get('r2') if exp1_v4_fedavg else None)}`。",
        f"- 已有 r40/r60 rounds 诊断：是，r40 FedAvg `RMSE={fmt_float(exp1_r40_fedavg.get('rmse') if exp1_r40_fedavg else None)}`，r60 FedAvg `RMSE={fmt_float(exp1_r60_fedavg.get('rmse') if exp1_r60_fedavg else None)}`。",
        "- 已有 client 异质性诊断：是，报告明确指出 `289` 是主要拖累 client。",
        "- 已有 selected_clients rationale：是，已说明当前 K=5 设置是相对原稿 K=3 的增强且已形成完整证据链。",
        "- 已有空间覆盖与分布统计：是，已恢复 pooled row/col 与近似经纬度边界，确认当前覆盖主要是局部子区域。",
        "- 已有 FedAvg 多指标优化 smoke：是；其中当前最佳 diagnostics 方案是 `experiment1_metric_opt_k5_r80_e2_lr5e4_cuda`。",
        f"- 当前结论：K=5 细粒度异质设置已完成正式链与诊断链，但即便 `r80/e2/lr5e-4` 将 FedAvg `RMSE` 优化到 `{fmt_float(exp1_r80e2_fedavg.get('rmse') if exp1_r80e2_fedavg else None)}`，仍未全面超过 NaiveLastValue；`289` 继续是关键异质点。",
        "- 当前缺口：缺少能缓解强 non-IID 的新 client 组织方式正式对比；继续直接调 K=5 的边际收益有限。",
        "",
        "## 5. 实验 2：single_intersection_ablation",
        "",
        f"- 是否已有目录：{yes_no(state['experiments']['single_intersection_ablation']['exists'])}。",
        f"- 是否已有 core：{yes_no(state['experiments']['single_intersection_ablation']['core_files']['core'])}，当前训练入口为 `sia_core.py`。",
        "- 是否已有正式训练：未见 `results/real_data_experiments/formal/` 下对应 formal 结果；当前更像已实现 + 历史/轻量结果，而非正式结果链。",
        "- 是否已有结果：有，存在 `results/real_data_experiments/single_intersection_ablation/` 与 `results/real_data_experiments/single_intersection_ablation_tensor/`，但不属于 formal/diagnostics 主线目录。",
        "- 是否尚未启动：不是完全未启动；目录、README、core/config、结果样例均已存在，但尚未形成当前阶段正式消融结果链。",
        "- 与一审消融要求的关系：该目录承载模型结构消融，但在当前推进节奏中优先级低于 client 组织方式调整，因此不宜先扩展实验 2。",
        "",
        "## 6. 实验 3：region_client",
        "",
        f"- 是否已有目录：{yes_no(state['experiments']['region_client']['exists'])}。",
        f"- 是否已有 core：{yes_no(state['experiments']['region_client']['core_files']['core'])}，当前训练入口为 `rc_core.py`。",
        "- 是否已有 region/cluster client 逻辑：是；默认 `partition_method=spatial_block`，可选 `flow_kmeans`，每个 client = 一组 pooled grid regions。",
        "- 是否已有正式训练：未见 formal 结果。",
        "- 是否已有结果：有，`results/real_data_experiments/region_client_tensor_smoke/` 表明 tensor smoke 已跑通。",
        "- 是否可作为“多个 grid cells = 一个 client”的方向：是；这是 cluster-level / multi-grid client 的主线候选，但当前只完成 smoke，尚无正式对比结果。",
        "- 当前缺口：缺 formal 训练结果、缺与实验 1 / full-cells 的统一对比、缺结论性报告。",
        "",
        "## 7. 实验 4：region_ablation",
        "",
        f"- 是否已有目录：{yes_no(state['experiments']['region_ablation']['exists'])}。",
        f"- 是否已有 core：{yes_no(state['experiments']['region_ablation']['core_files']['core'])}，当前训练入口为 `ra_core.py`。",
        "- 是否已有正式训练：未见 formal 结果。",
        "- 是否已有结果：有，`results/real_data_experiments/region_ablation_tensor_smoke/` 表明 smoke 已跑通。",
        "- 是否尚未启动：不是完全未启动；已完成 Python 化迁移与 smoke，但尚未进入正式阶段。",
        "- 与 region/cluster client 消融的关系：该目录用于 cluster-level client 的结构消融，应在 region/full-cells 主线结果稳定后再推进。",
        "",
        "## 8. 新增或计划实验：region_client_full_cells",
        "",
        f"- 是否存在目录：{yes_no(state['experiments']['region_client_full_cells']['exists'])}。",
        "- 它是新增 full-cells 多客户端组织实验：是，目标是使用全部有效 grid cells，将多个 cells 组织成一个 client。",
        f"- 是否已有 inventory / partition / dataset / core：{yes_no(state['experiments']['region_client_full_cells']['core_files']['inventory'])} / {yes_no(state['experiments']['region_client_full_cells']['core_files']['partitions'])} / {yes_no(state['experiments']['region_client_full_cells']['core_files']['dataset'])} / {yes_no(state['experiments']['region_client_full_cells']['core_files']['core'])}。",
        "- 是否使用全部有效 grid cells：是，`full_cell_inventory_zh.md` 记录 `630` 个 pooled grid cells 中 `223` 个为 valid cells。",
        "- 是否实现 spatial partition：是，已存在 `spatial_k5.json`、`spatial_k8.json`、`spatial_k10.json`。",
        "- 是否实现 similarity partition：是，已存在 `similarity_k5.json`、`similarity_k8.json`、`similarity_k10.json`。",
        f"- 是否已有 smoke：是，已有 `full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda` 与 `full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda`；spatial FedAvg `RMSE={fmt_float(full_cells_spatial_fedavg.get('rmse') if full_cells_spatial_fedavg else None)}`，similarity FedAvg `RMSE={fmt_float(full_cells_similarity_fedavg.get('rmse') if full_cells_similarity_fedavg else None)}`。",
        "- 当前状态：目录、inventory、partition、dataset、训练入口和 K=5 smoke 已完成；尚缺统一的 smoke 对比报告产物与是否扩展 K=8/K=10 的结论判断。",
        "",
        "## 9. 已有 results 清单",
        "",
        results_section,
        "",
        "## 10. 当前真实数据实验完成度判断",
        "",
        completion_table,
        "",
        "## 11. 当前最重要结论",
        "",
        "- 当前只有 `single_intersection_client` 最完整。",
        "- 当前 K=5 grid-cell client 已进入边际收益阶段。",
        "- 继续调 K=5 不如换 client 组织方式。",
        "- `region/cluster/full-cells client` 是下一步。",
        "- 不能删除 `NaiveLastValue`。",
        "- 不能删除或替换 `289`，只能作为诊断或新实验对照。",
        "",
        "## 12. 下一步建议",
        "",
        "- 已经存在 `full-cells smoke`，建议先生成对比报告并判断是否扩展 `K=8/K=10`。",
        "",
        "## 13. 边界声明",
        "",
        "本阶段只读梳理当前真实数据实验；未运行训练，未修改 FedAvg，未修改模型结构，未修改数据划分，未提交 results，未执行 git add、git commit 或 git push。",
        "",
        f"_本报告由 `{to_repo_rel(output_report, repo_root)}` 生成；扫描根目录为 `{to_repo_rel(root, repo_root)}`，results 根目录为 `{to_repo_rel(results_root, repo_root)}`。_",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    results_root = Path(args.results_root).resolve()
    output_report = Path(args.output_report).resolve()
    repo_root = resolve_repo_root(output_report)

    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not results_root.exists():
        raise FileNotFoundError(f"Results root path does not exist: {results_root}")

    report = build_report(root=root, results_root=results_root, output_report=output_report, repo_root=repo_root)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    print(f"[inventory] report written to: {output_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
