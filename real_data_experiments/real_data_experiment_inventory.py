"""Generate a read-only inventory report for the re-numbered real-data experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DIRECTORY_ORDER = [
    "common",
    "single_intersection_client",
    "single_intersection_ablation",
    "region_client_full_cells",
    "region_client",
    "region_ablation",
]

DIRECTORY_LABELS = {
    "common": "公共模块",
    "single_intersection_client": "新实验 1",
    "single_intersection_ablation": "新实验 2",
    "region_client_full_cells": "新实验 3 / 4",
    "region_client": "新实验 5",
    "region_ablation": "新实验 6",
}

NEW_EXPERIMENTS = [
    {
        "id": 1,
        "title_en": "single grid client comparison",
        "title_zh": "单个网格作为单个客户端的对比实验",
        "directory": "single_intersection_client",
        "old_mapping": "原实验 1 -> 新实验 1",
        "definition": "client_i = grid_cell_i；一个 client 只能对应一个 grid cell，不能把多个 grid cells 合并进同一个 client。",
    },
    {
        "id": 2,
        "title_en": "single grid client ablation",
        "title_zh": "单个网格作为单个客户端的消融实验",
        "directory": "single_intersection_ablation",
        "old_mapping": "原实验 2 -> 新实验 2",
        "definition": "沿用实验 1 的单网格客户端定义，只比较结构消融，不改 FedAvg，不改模型主线。",
    },
    {
        "id": 3,
        "title_en": "similar grid group client comparison",
        "title_zh": "多个相似网格合并为一个客户端的对比实验",
        "directory": "region_client_full_cells",
        "old_mapping": "原实验 5 -> 新实验 3",
        "definition": "client_k = {grid_cell_a, grid_cell_b, ...}；重点是把若干相似 grid cells 合并为一个 client。",
    },
    {
        "id": 4,
        "title_en": "similar grid group client ablation",
        "title_zh": "多个相似网格合并为一个客户端的消融实验",
        "directory": "region_client_full_cells",
        "old_mapping": "原实验 5 的消融补齐 -> 新实验 4",
        "definition": "沿用新实验 3 的客户端组织方式补齐消融；本阶段先补文档、inventory 与结果归属说明。",
    },
    {
        "id": 5,
        "title_en": "global similarity partition comparison",
        "title_zh": "全局所有网格按相似度划分为客户端的对比实验",
        "directory": "region_client",
        "old_mapping": "原实验 3 -> 新实验 5",
        "definition": "All grid cells are partitioned into K non-overlapping clients，且每个 grid cell 只能属于一个 client。",
    },
    {
        "id": 6,
        "title_en": "global similarity partition ablation",
        "title_zh": "全局所有网格按相似度划分为客户端的消融实验",
        "directory": "region_ablation",
        "old_mapping": "原实验 4 -> 新实验 6",
        "definition": "沿用新实验 5 的全局覆盖式客户端划分，只做结构消融，不修改聚合公式。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a read-only inventory report for the re-numbered real-data experiments.")
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
    return (completed.stdout.strip() or completed.stderr.strip()).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_method_row(path: Path, method: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    for row in read_csv_rows(path):
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


def parse_status_path(line: str) -> str:
    stripped = line.rstrip()
    if len(stripped) >= 4 and stripped[2] == " ":
        return stripped[3:].strip()
    parts = stripped.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else stripped.strip()


def scan_experiment_dir(exp_dir: Path) -> dict[str, Any]:
    exists = exp_dir.exists() and exp_dir.is_dir()
    if not exists:
        return {"exists": False, "file_count": 0, "readme": False}
    files = [path for path in exp_dir.rglob("*") if path.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "readme": (exp_dir / "README_zh.md").exists(),
    }


def find_result_dirs(results_root: Path) -> list[Path]:
    candidate_files: list[Path] = []
    patterns = [
        "**/run_config.json",
        "**/split_summary.json",
        "**/main_metrics.csv",
        "**/client_metrics.csv",
        "**/ablation_summary.csv",
        "**/ablation_metrics.csv",
    ]
    for pattern in patterns:
        candidate_files.extend(results_root.glob(pattern))
    return sorted({path.parent.resolve() for path in candidate_files if path.exists()})


def infer_result_type(result_dir: Path) -> str:
    text = result_dir.as_posix().lower()
    if "/formal/" in text:
        return "formal"
    if "/diagnostics/" in text:
        return "diagnostics"
    if "smoke" in result_dir.name.lower():
        return "smoke"
    return "other"


def infer_result_experiment(result_dir: Path, run_config: dict[str, Any] | None) -> str:
    name = result_dir.name.lower()
    exp_name = str(run_config.get("experiment_name", "")).lower() if isinstance(run_config, dict) else ""
    joined = f"{name} {exp_name}"
    if "single_intersection_client" in joined or "grid_cell" in joined or "experiment1" in joined:
        return "single_intersection_client"
    if "single_intersection_ablation" in joined:
        return "single_intersection_ablation"
    if "region_client_full_cells" in joined or "full_cells" in joined:
        return "region_client_full_cells"
    if "region_ablation" in joined:
        return "region_ablation"
    if "region_client" in joined:
        return "region_client"
    return "unknown"


def collect_result_info(result_dir: Path, repo_root: Path) -> dict[str, Any]:
    run_config = read_json(result_dir / "run_config.json") if (result_dir / "run_config.json").exists() else None
    return {
        "path": result_dir,
        "rel_path": to_repo_rel(result_dir, repo_root),
        "name": result_dir.name,
        "type": infer_result_type(result_dir),
        "experiment": infer_result_experiment(result_dir, run_config),
    }


def choose_result_dirs(result_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "grid_cell_main_full_cuda_v4",
        "single_intersection_ablation",
        "single_intersection_ablation_tensor",
        "full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda",
        "full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda",
        "region_client_tensor_smoke",
        "region_ablation_tensor_smoke",
        "experiment1_metric_opt_k5_r80_e2_lr5e4_cuda",
        "experiment1_fedavg_rounds_smoke_r60_cuda",
        "experiment1_fedavg_rounds_smoke_r40_cuda",
    }
    rank = {"formal": 0, "diagnostics": 1, "smoke": 2, "other": 3}
    return sorted(
        [info for info in result_infos if info["name"] in wanted],
        key=lambda item: (rank.get(item["type"], 9), item["name"]),
    )


def build_numbering_overview() -> str:
    rows = []
    for experiment in NEW_EXPERIMENTS:
        rows.append(
            [
                f"实验 {experiment['id']}",
                experiment["title_en"],
                experiment["title_zh"],
                f"`real_data_experiments/{experiment['directory']}/`",
                experiment["old_mapping"],
            ]
        )
    return md_table(["新编号", "英文名称", "实验含义", "对应目录", "旧新映射"], rows)


def build_legacy_mapping_table() -> str:
    rows = [
        ["原实验 1", "新实验 1", "单个网格作为单个客户端的对比实验"],
        ["原实验 2", "新实验 2", "单个网格作为单个客户端的消融实验"],
        ["原实验 5", "新实验 3", "多个相似网格合并为一个客户端的对比实验"],
        ["原实验 5 的消融补齐", "新实验 4", "多个相似网格合并为一个客户端的消融实验"],
        ["原实验 3", "新实验 5", "全局所有网格按相似度划分为客户端的对比实验"],
        ["原实验 4", "新实验 6", "全局所有网格按相似度划分为客户端的消融实验"],
    ]
    return md_table(["旧编号", "新编号", "含义"], rows)


def build_client_boundary_section() -> str:
    return "\n".join(
        [
            "### 第一类：单个网格作为单个客户端",
            "",
            "- 定义：`client_i = grid_cell_i`。",
            "- 对应：新实验 1、新实验 2。",
            "- 边界：不能把多个 grid cells 合并进同一个 client。",
            "",
            "### 第二类：多个相似网格合并为一个客户端",
            "",
            "- 定义：`client_k = {grid_cell_a, grid_cell_b, grid_cell_c, ...}`。",
            "- 对应：新实验 3、新实验 4。",
            "- 边界：重点是把若干相似 grid cells 合并成一个 client，不把它写成全局覆盖式完整划分。",
            "",
            "### 第三类：全局所有网格按相似度划分为客户端",
            "",
            "- 定义：`All grid cells are partitioned into K non-overlapping clients.`",
            "- 对应：新实验 5、新实验 6。",
            "- 边界：强调全局覆盖式划分，且 `client_i and client_j are non-overlapping when i != j.`",
        ]
    )


def build_directory_table(root: Path) -> str:
    meanings = {
        "common": "公共张量、划分、FedAvg、指标与结果写出工具",
        "single_intersection_client": "新实验 1：single grid client comparison",
        "single_intersection_ablation": "新实验 2：single grid client ablation",
        "region_client_full_cells": "新实验 3 / 4：similar grid group client comparison / ablation",
        "region_client": "新实验 5：global similarity partition comparison",
        "region_ablation": "新实验 6：global similarity partition ablation",
    }
    rows = []
    for name in DIRECTORY_ORDER:
        meta = scan_experiment_dir(root / name)
        rows.append(
            [
                f"`real_data_experiments/{name}/`",
                DIRECTORY_LABELS[name],
                meanings[name],
                yes_no(meta["exists"]),
                yes_no(meta["readme"]),
                str(meta["file_count"]),
            ]
        )
    return md_table(["目录", "当前归属", "重构后含义", "是否存在", "是否有 README", "文件数"], rows)


def build_result_mapping_rows(selected_results: list[dict[str, Any]]) -> str:
    rows = []
    for info in selected_results:
        name = info["name"]
        if name == "grid_cell_main_full_cuda_v4":
            mapped = "新实验 1"
            note = "归入新实验 1 的正式结果。"
        elif name in {"single_intersection_ablation", "single_intersection_ablation_tensor"}:
            mapped = "新实验 2"
            note = "归入新实验 2 的历史/轻量消融结果。"
        elif name == "full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda":
            mapped = "新实验 3"
            note = "归入新实验 3 的 similarity diagnostic/smoke 结果。"
        elif name == "full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda":
            mapped = "新实验 3 / 4 所在线的辅助诊断"
            note = "保留为 grouped-client 目录下的 spatial 辅助对照，旧路径不移动。"
        elif name == "region_client_tensor_smoke":
            mapped = "新实验 5"
            note = "若其客户端逻辑为全局覆盖式划分，则归入新实验 5 的 smoke。"
        elif name == "region_ablation_tensor_smoke":
            mapped = "新实验 6"
            note = "若其客户端逻辑为全局覆盖式消融，则归入新实验 6 的 smoke。"
        else:
            mapped = "新实验 1"
            note = "归入新实验 1 的 diagnostics/smoke 结果。"
        rows.append([f"`{info['rel_path']}`", mapped, info["type"], note])
    return md_table(["结果路径", "归属新实验", "类型", "说明"], rows)


def build_experiment_sections(results_root: Path, result_infos: list[dict[str, Any]]) -> str:
    exp1_v4 = next((info for info in result_infos if info["name"] == "grid_cell_main_full_cuda_v4"), None)
    exp1_v4_fed = read_method_row(exp1_v4["path"] / "main_metrics.csv", "FedAvg") if exp1_v4 else None
    exp1_v4_naive = read_method_row(exp1_v4["path"] / "main_metrics.csv", "NaiveLastValue") if exp1_v4 else None
    exp1_r40_fed = read_method_row(results_root / "diagnostics" / "experiment1_fedavg_rounds_smoke_r40_cuda" / "main_metrics.csv", "FedAvg")
    exp1_r60_fed = read_method_row(results_root / "diagnostics" / "experiment1_fedavg_rounds_smoke_r60_cuda" / "main_metrics.csv", "FedAvg")
    exp1_best_fed = read_method_row(results_root / "diagnostics" / "experiment1_metric_opt_k5_r80_e2_lr5e4_cuda" / "main_metrics.csv", "FedAvg")
    exp3_similarity_fed = read_method_row(results_root / "diagnostics" / "full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda" / "main_metrics.csv", "FedAvg")
    exp3_spatial_fed = read_method_row(results_root / "diagnostics" / "full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda" / "main_metrics.csv", "FedAvg")
    exp5_smoke_fed = read_method_row(results_root / "region_client_tensor_smoke" / "main_metrics.csv", "FedAvg")
    exp6_has_ablation = (results_root / "region_ablation_tensor_smoke" / "ablation_metrics.csv").exists()

    return "\n".join(
        [
            "## 6. 新实验 1-6 当前目录与状态",
            "",
            "### 新实验 1：single grid client comparison",
            "",
            "- 对应目录：`real_data_experiments/single_intersection_client/`。",
            "- 旧新映射：原实验 1 -> 新实验 1。",
            "- 实验含义：单个网格作为单个客户端的对比实验。",
            "- 客户端边界：`client_i = grid_cell_i`。",
            "- 正式结果：`results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/`。",
            f"- formal v4 指标：FedAvg `RMSE={fmt_float(exp1_v4_fed.get('rmse') if exp1_v4_fed else None)}`，NaiveLastValue `RMSE={fmt_float(exp1_v4_naive.get('rmse') if exp1_v4_naive else None)}`。",
            f"- 诊断链：r40 FedAvg `RMSE={fmt_float(exp1_r40_fed.get('rmse') if exp1_r40_fed else None)}`；r60 FedAvg `RMSE={fmt_float(exp1_r60_fed.get('rmse') if exp1_r60_fed else None)}`；最佳 diagnostics 方案 `r80/e2/lr5e-4` 的 FedAvg `RMSE={fmt_float(exp1_best_fed.get('rmse') if exp1_best_fed else None)}`。",
            "",
            "### 新实验 2：single grid client ablation",
            "",
            "- 对应目录：`real_data_experiments/single_intersection_ablation/`。",
            "- 旧新映射：原实验 2 -> 新实验 2。",
            "- 实验含义：单个网格作为单个客户端的消融实验。",
            "- 结果归属：`results/real_data_experiments/single_intersection_ablation/` 与 `results/real_data_experiments/single_intersection_ablation_tensor/`。",
            "- 当前状态：目录、README、core/config 和历史结果存在，但未新增 formal 结果链。",
            "",
            "### 新实验 3：similar grid group client comparison",
            "",
            "- 对应目录：`real_data_experiments/region_client_full_cells/`。",
            "- 旧新映射：原实验 5 -> 新实验 3。",
            "- 实验含义：多个相似网格合并为一个客户端的对比实验。",
            f"- 现有 similarity smoke：FedAvg `RMSE={fmt_float(exp3_similarity_fed.get('rmse') if exp3_similarity_fed else None)}`。",
            f"- 辅助 spatial 对照：FedAvg `RMSE={fmt_float(exp3_spatial_fed.get('rmse') if exp3_spatial_fed else None)}`；保留旧路径，只作为 grouped-client 线路的辅助诊断。",
            "",
            "### 新实验 4：similar grid group client ablation",
            "",
            "- 对应目录：仍暂挂在 `real_data_experiments/region_client_full_cells/`。",
            "- 旧新映射：原实验 5 的消融补齐 -> 新实验 4。",
            "- 当前状态：本阶段先补编号、README、inventory 与结果归属说明；尚未新增独立 ablation 训练入口。",
            "",
            "### 新实验 5：global similarity partition comparison",
            "",
            "- 对应目录：`real_data_experiments/region_client/`。",
            "- 旧新映射：原实验 3 -> 新实验 5。",
            "- 实验含义：全局所有网格按相似度划分为客户端的对比实验。",
            f"- smoke 结果：`results/real_data_experiments/region_client_tensor_smoke/`，FedAvg `RMSE={fmt_float(exp5_smoke_fed.get('rmse') if exp5_smoke_fed else None)}`。",
            "",
            "### 新实验 6：global similarity partition ablation",
            "",
            "- 对应目录：`real_data_experiments/region_ablation/`。",
            "- 旧新映射：原实验 4 -> 新实验 6。",
            "- 实验含义：全局所有网格按相似度划分为客户端的消融实验。",
            f"- smoke 文件 `ablation_metrics.csv` 是否存在：{yes_no(exp6_has_ablation)}。",
        ]
    )


def build_report(root: Path, results_root: Path, output_report: Path, repo_root: Path) -> str:
    status_short = run_git(repo_root, "status", "--short", "--untracked-files=all") or "(clean)"
    status_sb = run_git(repo_root, "status", "-sb") or "(empty)"
    status_lines = [line for line in status_short.splitlines() if line.strip()]
    unsubmitted = [parse_status_path(line) for line in status_lines]
    result_infos = [collect_result_info(path, repo_root) for path in find_result_dirs(results_root)]
    selected_results = choose_result_dirs(result_infos)
    unsubmitted_text = "\n".join(f"- `{path}`" for path in unsubmitted) if unsubmitted else "- 无"

    lines = [
        "# 当前真实数据实验清单报告",
        "",
        "## 1. 报告目的",
        "",
        "本报告按新的真实数据实验 1-6 编号重新梳理目录、实验含义、旧新编号关系与结果路径引用。",
        "",
        "## 2. 当前 Git 状态",
        "",
        "- `git status --short --untracked-files=all`：",
        "",
        "```text",
        status_short,
        "```",
        "",
        "- `git status -sb`：",
        "",
        "```text",
        status_sb,
        "```",
        "",
        "- 当前未提交/未跟踪文件：",
        unsubmitted_text,
        "",
        "## 3. 新实验 1-6 编号总览",
        "",
        build_numbering_overview(),
        "",
        "## 4. 旧编号到新编号转换表",
        "",
        build_legacy_mapping_table(),
        "",
        "## 5. 三类客户端组织方式边界",
        "",
        build_client_boundary_section(),
        "",
        "## 6. 目录对应关系",
        "",
        build_directory_table(root),
        "",
        build_experiment_sections(results_root, result_infos),
        "",
        "## 7. 结果目录引用与新编号归属",
        "",
        build_result_mapping_rows(selected_results),
        "",
        "## 8. 本次重构结论",
        "",
        "- `single_intersection_client/` 固定对应新实验 1。",
        "- `single_intersection_ablation/` 固定对应新实验 2。",
        "- `region_client_full_cells/` 在最小改动方案下承接新实验 3，并为新实验 4 预留同目录补齐位。",
        "- `region_client/` 固定对应新实验 5。",
        "- `region_ablation/` 固定对应新实验 6。",
        "- 旧 results 不删除、不移动；只在文档中新增新编号对应关系。",
        "",
        "## 9. 边界声明",
        "",
        "- 未运行训练。",
        "- 未修改 FedAvg 聚合公式。",
        "- 未修改模型结构。",
        "- 未修改正式数据入口。",
        "- 未生成 `6.池化网格张量.pt`。",
        "- 未删除 `NaiveLastValue`。",
        "- 未删除或替换 `289`。",
        "- 未改动 `results/` 中已有结果文件，只更新文档中的引用与归属说明。",
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
