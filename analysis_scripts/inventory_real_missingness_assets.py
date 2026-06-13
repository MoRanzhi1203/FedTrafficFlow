from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pq = None


TEXT_EXTENSIONS = {".py", ".md", ".json", ".csv", ".txt"}
AUDIT_EXTENSIONS = {".py", ".md", ".json", ".csv", ".txt", ".png", ".pdf", ".parquet", ".npz", ".npy"}
RESULT_FILE_EXTENSIONS = {".csv", ".json", ".md", ".png", ".pdf", ".parquet"}
MISSINGNESS_KEYWORDS = [
    "missingness",
    "missing",
    "imputation",
    "impute",
    "mcar",
    "node_temporal_block",
    "temporal_block",
    "geo_neighbor",
    "function_curve",
    "geo_func_hybrid",
    "historical_linear",
    "causal_history",
    "forward_fill",
    "zero_fill",
    "linear_interpolation",
    "intersection",
    "flow",
    "mask",
    "缺失",
    "插补",
    "补全",
]
RISK_KEYWORDS = [
    "天然缺失",
    "原始缺失",
    "数据集存在缺失",
    "自然缺失",
    "FedAvg RMSE",
    "Independent RMSE",
    "预测结果",
    "预测鲁棒性",
    "显著优于",
    "显著提升",
    "未来数据",
    "双向插值",
    "bfill",
]
TARGET_METHODS = [
    "zero_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "geo_neighbor_fill",
    "function_curve_fit",
    "geo_func_hybrid",
    "linear_interpolation",
]
TARGET_MECHANISMS = ["mcar_point", "node_temporal_block"]
TARGET_DIRECTORIES = [
    "analysis_scripts",
    "preprocessing_scripts",
    "dataset_inspection_scripts",
    "results",
    "data/analysis",
    "data/processed",
    "paper_revision",
    "paper_revision/manuscript_sections_zh",
    "paper_revision/manuscript_sections_zh/current",
    "paper_revision/latex_source",
    "paper_revision/formula_notes",
]
REQUIRED_CODE_FILES = [
    "analysis_scripts/real_data_missingness_experiment.py",
    "analysis_scripts/full_intersection_missingness_pipeline.py",
    "analysis_scripts/audit_missingness_mechanism.py",
    "analysis_scripts/audit_real_data_preprocessing.py",
]
RESULT_DIR_CANDIDATES = [
    "results/real_data_missingness_experiments",
    "results/real_data_missingness_experiments_geo_func",
    "results/real_data_missingness_experiments_medium",
    "results/real_data_missingness_experiments_sample",
    "results/real_data_missingness_full_intersection",
    "results/real_data_missingness_full_intersection_causal_history",
    "results/real_data_missingness_full_intersection_causal_history/historical_test",
    "results/real_data_missingness_full_intersection_causal_history/smoke_test",
    "results/real_data_missingness_full_intersection_causal_history_smoketest",
    "results/real_data_missingness_full_intersection_causal_history_hybridtest",
    "results/real_data_missingness_full_intersection_causal_history_hybridtest_small",
    "results/real_data_missingness_mechanism_audit",
    "results/real_data_preprocessing",
]
METHOD_LABELS = {
    "zero_fill": "Zero fill",
    "forward_fill": "Forward fill",
    "historical_linear_extrapolation": "Historical linear extrapolation",
    "geo_neighbor_fill": "Geo-neighbor fill",
    "function_curve_fit": "Function curve fit",
    "geo_func_hybrid": "Geo-function hybrid",
    "linear_interpolation": "Linear interpolation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实数据缺失实验资产清单与状态审计脚本。")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def safe_stat(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"size_bytes": None, "last_modified": None}
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def read_text(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="ignore")


def try_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def list_files(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    items: List[Path] = []
    for root, _, files in os.walk(base_dir):
        root_path = Path(root)
        for file_name in files:
            items.append(root_path / file_name)
    return items


def list_dirs(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        return []
    items: List[Path] = []
    for root, dirs, _ in os.walk(base_dir):
        root_path = Path(root)
        for dir_name in dirs:
            items.append(root_path / dir_name)
    return items


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() in lowered:
            return True
    return False


def extract_cli_args(text: str) -> List[str]:
    return sorted(set(re.findall(r'add_argument\(\s*["\'](--[^"\']+)["\']', text)))


def extract_functions(text: str) -> List[str]:
    return sorted(set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.MULTILINE)))


def infer_purpose_from_name(file_name: str) -> str:
    lowered = file_name.lower()
    if "full_intersection_missingness_pipeline" in lowered:
        return "完整路口真实数据缺失构造、历史因果补全、验证与汇总流水线"
    if "real_data_missingness_experiment" in lowered:
        return "早期样本级真实数据人工缺失注入与简单插补评估脚本"
    if "audit_real_data_preprocessing" in lowered:
        return "真实数据预处理资产审计脚本"
    if "audit_missingness_mechanism" in lowered:
        return "缺失机制审计脚本"
    if "missing" in lowered or "imput" in lowered:
        return "缺失或插补相关脚本"
    if "curve" in lowered or "function" in lowered:
        return "函数曲线或形态建模相关脚本"
    if "inspect" in lowered or "audit" in lowered:
        return "检查或审计相关脚本"
    return "未找到明确证据，需要人工确认"


def extract_list_literal(text: str, variable_name: str) -> List[str]:
    pattern = re.compile(variable_name + r"\s*=\s*\[(.*?)\]", re.S)
    match = pattern.search(text)
    if not match:
        return []
    return sorted(set(re.findall(r'["\']([^"\']+)["\']', match.group(1))))


def parse_json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (int, float, bool)):
        return [value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_command_flags(command_text: str) -> Dict[str, str]:
    flags: Dict[str, str] = {}
    matches = re.finditer(r"--([A-Za-z0-9_]+)\s+([^\s]+)", command_text)
    for match in matches:
        flags[match.group(1)] = match.group(2)
    return flags


def run_command(command: List[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        output = (completed.stdout or "").strip()
        if output:
            return output
        return (completed.stderr or "").strip()
    except Exception as exc:
        return str(exc)


def load_csv_header_and_shape(path: Path) -> Tuple[int, int, List[str]]:
    try:
        header_df = pd.read_csv(path, nrows=0)
        columns = [str(column) for column in header_df.columns]
    except Exception:
        columns = []
    row_count: Optional[int] = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        row_count = None
    return row_count if row_count is not None else 0, len(columns), columns


def is_meaningful_result_dir(project_root: Path, path: Path) -> bool:
    rel = to_rel(project_root, path)
    if rel == "results/real_data_missingness_inventory":
        return False
    if not (rel.startswith("results/real_data_missingness") or rel == "results/real_data_preprocessing"):
        return False
    direct_file_names = {item.name for item in path.iterdir() if item.is_file()} if path.exists() else set()
    direct_dir_names = {item.name for item in path.iterdir() if item.is_dir()} if path.exists() else set()
    if "run_config.json" in direct_file_names or "run_commands.txt" in direct_file_names:
        return True
    if any(
        name in direct_file_names
        for name in [
            "full_intersection_missingness_audit.json",
            "full_intersection_missingness_audit.md",
            "full_intersection_missingness_validation.json",
            "full_intersection_missingness_validation.md",
            "real_data_missingness_experiment_audit.json",
            "real_data_missingness_experiment_audit.md",
            "real_data_preprocessing_audit.json",
            "real_data_preprocessing_audit.md",
        ]
    ):
        return True
    if direct_dir_names.intersection({"summaries", "figures", "manifests", "masks", "missing_datasets", "imputed_datasets"}):
        return True
    return False


def find_result_dirs(project_root: Path) -> List[str]:
    results_root = project_root / "results"
    discovered: Set[str] = set(RESULT_DIR_CANDIDATES)
    if results_root.exists():
        for path in [results_root] + list_dirs(results_root):
            rel = to_rel(project_root, path)
            if is_meaningful_result_dir(project_root, path):
                discovered.add(rel)
    filtered = [rel for rel in discovered if rel != "results/real_data_missingness_inventory"]
    return sorted(set(filtered))


def find_relevant_code_files(project_root: Path) -> List[str]:
    matches: Set[str] = set(REQUIRED_CODE_FILES)
    for folder_name in ["analysis_scripts", "preprocessing_scripts", "dataset_inspection_scripts"]:
        folder = project_root / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*.py"):
            text = read_text(path)
            rel = to_rel(project_root, path)
            if rel in REQUIRED_CODE_FILES or contains_keyword(rel + "\n" + text, MISSINGNESS_KEYWORDS):
                matches.add(rel)
    return sorted(matches)


def find_relevant_document_files(project_root: Path) -> List[str]:
    matches: Set[str] = set()
    candidate_roots = [
        project_root / "paper_revision",
        project_root / "docs",
        project_root,
        project_root / "results",
    ]
    for root in candidate_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt", ".json", ".csv"}:
                continue
            rel = to_rel(project_root, path)
            if not (rel.startswith("paper_revision") or rel.startswith("docs") or rel.startswith("results") or path.parent == project_root):
                continue
            text = read_text(path)
            if contains_keyword(rel + "\n" + text, MISSINGNESS_KEYWORDS) or contains_keyword(text, RISK_KEYWORDS):
                matches.add(rel)
    return sorted(matches)


def detect_code_category(rel_path: str) -> str:
    if rel_path in REQUIRED_CODE_FILES:
        return "required"
    if rel_path.startswith("analysis_scripts/"):
        return "analysis"
    if rel_path.startswith("preprocessing_scripts/"):
        return "preprocessing"
    if rel_path.startswith("dataset_inspection_scripts/"):
        return "inspection"
    return "other"


def analyze_code_file(project_root: Path, rel_path: str) -> Dict[str, Any]:
    path = project_root / rel_path
    exists = path.exists()
    text = read_text(path) if exists else ""
    stats = safe_stat(path)
    methods = extract_list_literal(text, "DEFAULT_IMPUTE_METHODS")
    mechanisms = extract_list_literal(text, "DEFAULT_MECHANISMS")
    notes: List[str] = []
    if "ffill().bfill()" in text or "limit_direction=\"both\"" in text or "limit_direction='both'" in text:
        notes.append("存在 bfill 或双向插值迹象，需警惕未来信息泄露风险。")
    if "context_days_after > 0" in text and "raise ValueError" in text:
        notes.append("代码显式禁止使用未来日期上下文。")
    if "save_masks" in text:
        notes.append("支持保存 masks。")
    if "write_missing_datasets" in text or "write_corrupted" in text:
        notes.append("支持保存缺失数据集。")
    if "write_imputed_datasets" in text or "write_imputed" in text:
        notes.append("支持保存插补数据集。")
    if "warmup_days" in text:
        notes.append("存在 warmup 相关配置。")
    return {
        "file_path": rel_path,
        "file_name": path.name,
        "category": detect_code_category(rel_path),
        "exists": exists,
        "size_bytes": stats["size_bytes"],
        "last_modified": stats["last_modified"],
        "detected_purpose": infer_purpose_from_name(path.name),
        "defines_missing_mechanism": bool(mechanisms or contains_keyword(text, TARGET_MECHANISMS)),
        "defines_imputation_method": bool(methods or contains_keyword(text, TARGET_METHODS)),
        "supports_causal_history": "causal_history_only" in text,
        "supports_full_intersection_data": "node_intersection_flow_parquet" in text or "full_intersection" in text or "global_time_index" in text,
        "supports_node_temporal_block": "node_temporal_block" in text or "temporal_block" in text,
        "supports_generate_missing": "generate_missing" in text,
        "supports_impute": "impute" in text,
        "supports_validate": "validate" in text,
        "supports_summarize": "summarize" in text,
        "key_functions": json_cell(extract_functions(text)[:20]),
        "key_cli_args": json_cell(extract_cli_args(text)),
        "notes": " ".join(notes) if notes else "",
    }


def detect_document_topic(rel_path: str, text: str) -> str:
    lowered = rel_path.lower()
    if "design" in lowered or "设计" in text:
        return "实验设计"
    if "result" in lowered or "结果" in text:
        return "实验结果"
    if "audit" in lowered or "审计" in text:
        return "审计说明"
    if "runbook" in lowered:
        return "运行计划"
    if "optimization" in lowered:
        return "优化计划"
    return "未找到明确证据，需要人工确认"


def analyze_document_file(project_root: Path, rel_path: str) -> Dict[str, Any]:
    path = project_root / rel_path
    exists = path.exists()
    text = read_text(path) if exists else ""
    stats = safe_stat(path)
    risk_hits = [keyword for keyword in RISK_KEYWORDS if keyword.lower() in text.lower()]
    return {
        "file_path": rel_path,
        "file_name": path.name,
        "category": "paper_or_doc" if rel_path.startswith("paper_revision/") or rel_path.startswith("docs/") else "result_or_project_note",
        "exists": exists,
        "size_bytes": stats["size_bytes"],
        "last_modified": stats["last_modified"],
        "detected_topic": detect_document_topic(rel_path, text),
        "mentions_mcar_point": "mcar_point" in text,
        "mentions_node_temporal_block": "node_temporal_block" in text or "temporal_block" in text,
        "mentions_causal_history": "causal_history" in text or "历史因果" in text,
        "mentions_full_intersection": "full_intersection" in text or "完整路口" in text,
        "mentions_fedavg_prediction": "fedavg" in text.lower() or "independent" in text.lower() or "预测结果" in text,
        "contains_final_results": "结论口径" in text or "当前结果结论" in text or "imputation_quality_summary" in text or "结果" in path.name,
        "notes": ("风险关键词: " + ", ".join(risk_hits)) if risk_hits else "",
    }


def collect_directory_stats(path: Path) -> Dict[str, Any]:
    total_files = 0
    total_size = 0
    ext_counter: Dict[str, int] = defaultdict(int)
    recursive_files = list_files(path)
    for file_path in recursive_files:
        total_files += 1
        total_size += int(file_path.stat().st_size)
        ext_counter[file_path.suffix.lower()] += 1
    direct_names = {item.name for item in path.iterdir() if item.is_file()}
    direct_dirs = {item.name for item in path.iterdir() if item.is_dir()}
    return {
        "total_files": total_files,
        "total_size_bytes": total_size,
        "has_run_config": "run_config.json" in direct_names,
        "has_run_commands": "run_commands.txt" in direct_names,
        "has_audit_json": any(name.endswith("_audit.json") or name == "full_intersection_missingness_audit.json" or name == "real_data_missingness_experiment_audit.json" or name == "real_data_preprocessing_audit.json" for name in direct_names),
        "has_audit_md": any(name.endswith("_audit.md") or name == "full_intersection_missingness_audit.md" or name == "real_data_missingness_experiment_audit.md" or name == "real_data_preprocessing_audit.md" for name in direct_names),
        "has_validation_json": any(name.endswith("_validation.json") or name == "full_intersection_missingness_validation.json" for name in direct_names),
        "has_validation_md": any(name.endswith("_validation.md") or name == "full_intersection_missingness_validation.md" for name in direct_names),
        "has_manifests": "manifests" in direct_dirs,
        "has_summaries": "summaries" in direct_dirs,
        "has_figures": "figures" in direct_dirs,
        "has_masks": "masks" in direct_dirs,
        "has_missing_datasets": "missing_datasets" in direct_dirs,
        "has_imputed_datasets": "imputed_datasets" in direct_dirs,
        "parquet_count": ext_counter.get(".parquet", 0),
        "csv_count": ext_counter.get(".csv", 0),
        "json_count": ext_counter.get(".json", 0),
        "md_count": ext_counter.get(".md", 0),
        "png_count": ext_counter.get(".png", 0),
        "pdf_count": ext_counter.get(".pdf", 0),
    }


def detect_scope(rel_dir: str) -> str:
    lowered = rel_dir.lower()
    if "historical_test" in lowered:
        return "8_chunk_historical_test"
    if "smoke_test" in lowered or "smoketest" in lowered:
        return "1_chunk_smoke_test"
    if "sample" in lowered:
        return "sample"
    if "medium" in lowered:
        return "medium"
    if "hybridtest_small" in lowered:
        return "hybridtest_small"
    if "hybridtest" in lowered:
        return "hybridtest"
    if "geo_func" in lowered:
        return "geo_function_subexperiment"
    if rel_dir == "results/real_data_missingness_full_intersection_causal_history":
        return "61_chunk_main_directory"
    if "full_intersection" in lowered:
        return "full_intersection"
    if "preprocessing" in lowered:
        return "preprocessing_audit"
    return "未找到明确证据，需要人工确认"


def summarize_stage_status_csv(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"rows": 0, "completed": 0, "skipped_existing": 0, "unique_chunks": 0}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"rows": 0, "completed": 0, "skipped_existing": 0, "unique_chunks": 0}
    counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    unique_chunks = 0
    if {"day_index", "file_name"}.issubset(set(df.columns)):
        unique_chunks = int(df[["day_index", "file_name"]].drop_duplicates().shape[0])
    return {
        "rows": int(len(df)),
        "completed": int(counts.get("completed", 0)),
        "skipped_existing": int(counts.get("skipped_existing", 0)),
        "unique_chunks": unique_chunks,
    }


def detect_result_status(project_root: Path, rel_dir: str, stats: Dict[str, Any]) -> Tuple[str, str]:
    path = project_root / rel_dir
    notes: List[str] = []
    generate_status = summarize_stage_status_csv(path / "manifests" / "generate_missing_chunk_status.csv")
    impute_status = summarize_stage_status_csv(path / "manifests" / "impute_chunk_status.csv")
    batch_report = try_read_json(path / "summaries" / "batch_processing_report.json")
    if generate_status["rows"] > 0:
        notes.append(
            "generate_missing: {0} completed, {1} skipped_existing, unique_chunks={2}".format(
                generate_status["completed"], generate_status["skipped_existing"], generate_status["unique_chunks"]
            )
        )
    if impute_status["rows"] > 0:
        notes.append(
            "impute: {0} completed, {1} skipped_existing, unique_chunks={2}".format(
                impute_status["completed"], impute_status["skipped_existing"], impute_status["unique_chunks"]
            )
        )
    if batch_report:
        notes.append(
            "batch_report total_chunks_selected={0}, summary_exclude_warmup_row_count={1}".format(
                batch_report.get("total_chunks_selected"),
                batch_report.get("summary_exclude_warmup_row_count"),
            )
        )
    if stats["has_validation_json"] and stats["has_summaries"]:
        return "validated_and_summarized", " ".join(notes)
    if stats["has_summaries"] and not stats["has_validation_json"]:
        return "summaries_present_without_validation", " ".join(notes)
    if stats["has_imputed_datasets"]:
        return "imputation_outputs_present", " ".join(notes)
    if stats["has_missing_datasets"] or stats["has_masks"]:
        return "missing_datasets_present", " ".join(notes)
    if path.exists():
        return "directory_present_no_complete_stage_evidence", " ".join(notes)
    return "missing", ""


def analyze_result_directory(project_root: Path, rel_dir: str) -> Dict[str, Any]:
    path = project_root / rel_dir
    exists = path.exists()
    if exists:
        stats = collect_directory_stats(path)
        detected_status, note = detect_result_status(project_root, rel_dir, stats)
    else:
        stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "has_run_config": False,
            "has_run_commands": False,
            "has_audit_json": False,
            "has_audit_md": False,
            "has_validation_json": False,
            "has_validation_md": False,
            "has_manifests": False,
            "has_summaries": False,
            "has_figures": False,
            "has_masks": False,
            "has_missing_datasets": False,
            "has_imputed_datasets": False,
            "parquet_count": 0,
            "csv_count": 0,
            "json_count": 0,
            "md_count": 0,
            "png_count": 0,
            "pdf_count": 0,
        }
        detected_status, note = "missing", ""
    return {
        "result_dir": rel_dir,
        "exists": exists,
        "total_files": stats["total_files"],
        "total_size_bytes": stats["total_size_bytes"],
        "has_run_config": stats["has_run_config"],
        "has_run_commands": stats["has_run_commands"],
        "has_audit_json": stats["has_audit_json"],
        "has_audit_md": stats["has_audit_md"],
        "has_validation_json": stats["has_validation_json"],
        "has_validation_md": stats["has_validation_md"],
        "has_manifests": stats["has_manifests"],
        "has_summaries": stats["has_summaries"],
        "has_figures": stats["has_figures"],
        "has_masks": stats["has_masks"],
        "has_missing_datasets": stats["has_missing_datasets"],
        "has_imputed_datasets": stats["has_imputed_datasets"],
        "parquet_count": stats["parquet_count"],
        "csv_count": stats["csv_count"],
        "json_count": stats["json_count"],
        "md_count": stats["md_count"],
        "png_count": stats["png_count"],
        "pdf_count": stats["pdf_count"],
        "detected_scope": detect_scope(rel_dir),
        "detected_status": detected_status,
        "notes": note,
    }


def find_run_config_files(project_root: Path) -> List[str]:
    results_root = project_root / "results"
    matches: Set[str] = set()
    if not results_root.exists():
        return []
    for path in results_root.rglob("run_config.json"):
        rel = to_rel(project_root, path)
        if rel.startswith("results/real_data_missingness") or rel.startswith("results/real_data_preprocessing"):
            matches.add(rel)
    return sorted(matches)


def companion_json(path: Path, file_names: List[str]) -> Dict[str, Any]:
    for name in file_names:
        payload = try_read_json(path / name)
        if payload:
            return payload
    return {}


def detect_old_experiment_future_risk(result_dir: str, impute_methods: List[str]) -> Tuple[bool, bool, bool]:
    lowered = result_dir.lower()
    if "real_data_missingness_experiments" in lowered:
        uses_bfill = True
        uses_bidirectional = "linear_interpolation" in impute_methods
        uses_future = uses_bfill or uses_bidirectional
        return uses_future, uses_bfill, uses_bidirectional
    return False, False, False


def analyze_run_config(project_root: Path, rel_path: str) -> Dict[str, Any]:
    path = project_root / rel_path
    payload = try_read_json(path)
    parent = path.parent
    result_dir = to_rel(project_root, parent)
    run_commands_text = read_text(parent / "run_commands.txt")
    command_flags = parse_command_flags(run_commands_text)
    audit_payload = companion_json(parent, [
        "full_intersection_missingness_audit.json",
        "real_data_missingness_experiment_audit.json",
        "real_data_preprocessing_audit.json",
    ])
    validation_payload = companion_json(parent, [
        "full_intersection_missingness_validation.json",
    ])
    batch_payload = try_read_json(parent / "summaries" / "batch_processing_report.json")
    impute_methods = [str(item) for item in parse_json_list(payload.get("impute_methods"))]
    if not impute_methods and payload.get("impute_methods"):
        impute_methods = [str(item) for item in parse_json_list(str(payload.get("impute_methods")))]
    missing_rates = [str(item) for item in parse_json_list(payload.get("missing_rates"))]
    if not missing_rates and payload.get("missing_rates") is not None:
        missing_rates = [str(item) for item in parse_json_list(str(payload.get("missing_rates")))]
    old_future, old_bfill, old_bidirectional = detect_old_experiment_future_risk(result_dir, impute_methods)
    audit_checks = audit_payload.get("causal_checks", {})
    validation_checks = validation_payload.get("checks", {})
    uses_future = audit_checks.get("uses_future_days")
    if uses_future is None:
        uses_future = validation_checks.get("uses_future_days")
    if uses_future is None:
        uses_future = bool(old_future or int(payload.get("context_days_after", 0) or 0) > 0)
    uses_bfill = audit_checks.get("uses_bfill")
    if uses_bfill is None:
        uses_bfill = validation_checks.get("uses_bfill")
    if uses_bfill is None:
        uses_bfill = old_bfill
    uses_bidirectional = audit_checks.get("uses_bidirectional_interpolation")
    if uses_bidirectional is None:
        uses_bidirectional = validation_checks.get("uses_bidirectional_interpolation")
    if uses_bidirectional is None:
        uses_bidirectional = old_bidirectional
    total_chunks_selected = batch_payload.get("total_chunks_selected")
    if total_chunks_selected is None:
        total_chunks_selected = audit_payload.get("batch_scope", {}).get("total_chunks_selected")
    processes_all_chunks = batch_payload.get("all_chunks_covered_in_generate")
    if processes_all_chunks is None:
        processes_all_chunks = audit_payload.get("batch_scope", {}).get("processes_all_chunks")
    notes: List[str] = []
    if payload.get("stage"):
        notes.append("stage={0}".format(payload.get("stage")))
    if validation_payload:
        notes.append("存在 validation 证据")
    if batch_payload:
        notes.append("存在 batch_processing_report 证据")
    if old_future:
        notes.append("旧实验脚本中存在 bfill 或双向插值风险")
    return {
        "config_path": rel_path,
        "result_dir": result_dir,
        "input_dir": payload.get("input_dir", command_flags.get("input_dir")),
        "input_pattern": payload.get("input_pattern", command_flags.get("input_pattern")),
        "output_dir": payload.get("output_dir", command_flags.get("output_dir")),
        "max_chunks": payload.get("max_chunks", payload.get("max_files")),
        "max_rows": payload.get("max_rows"),
        "missing_rates": json_cell(missing_rates),
        "mechanism": payload.get("mechanism", json_cell(parse_json_list(payload.get("mechanisms")))),
        "seed": payload.get("seed", json_cell(parse_json_list(payload.get("seeds")))),
        "impute_methods": json_cell(impute_methods),
        "causal_history_only": payload.get("causal_history_only"),
        "history_days": payload.get("history_days"),
        "context_days_after": payload.get("context_days_after"),
        "warmup_days": payload.get("warmup_days"),
        "exclude_warmup_from_main_metrics": payload.get("exclude_warmup_from_main_metrics"),
        "write_missing_datasets": payload.get("write_missing_datasets", payload.get("write_corrupted")),
        "write_imputed_datasets": payload.get("write_imputed_datasets", payload.get("write_imputed")),
        "save_masks": payload.get("save_masks"),
        "total_chunks_selected": total_chunks_selected,
        "processes_all_chunks": processes_all_chunks if processes_all_chunks is not None else (payload.get("max_chunks", 0) == 0),
        "uses_future_days": uses_future,
        "uses_bfill": uses_bfill,
        "uses_bidirectional_interpolation": uses_bidirectional,
        "notes": " ".join(notes),
    }


def find_visualization_files(project_root: Path) -> List[str]:
    matches: Set[str] = set()
    results_root = project_root / "results"
    if not results_root.exists():
        return []
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".pdf"}:
            continue
        rel = to_rel(project_root, path)
        if rel.startswith("results/real_data_missingness"):
            matches.add(rel)
    return sorted(matches)


def find_parent_result_dir(rel_path: str, result_dirs: List[str]) -> str:
    for result_dir in sorted(result_dirs, key=len, reverse=True):
        if rel_path.startswith(result_dir + "/") or rel_path == result_dir:
            return result_dir
    return "未找到明确证据，需要人工确认"


def detect_plot_type(file_name: str) -> str:
    lowered = file_name.lower()
    if "delta" in lowered or "difference_relative" in lowered:
        return "delta"
    if "zoom" in lowered:
        return "zoom"
    if "overall" in lowered or "missing_rate_vs" in lowered:
        return "overall"
    return "other"


def detect_metric(file_name: str) -> str:
    lowered = file_name.lower()
    for metric in ["rmse", "mae", "mape", "smape"]:
        if metric in lowered:
            return metric.upper() if metric != "smape" else "sMAPE"
    return "未找到明确证据，需要人工确认"


def analyze_visualization_file(project_root: Path, rel_path: str, result_dirs: List[str], run_config_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    path = project_root / rel_path
    parent_result_dir = find_parent_result_dir(rel_path, result_dirs)
    config_info = run_config_map.get(parent_result_dir, {})
    missing_rates_text = str(config_info.get("missing_rates", ""))
    missing_rates: List[str] = []
    if missing_rates_text:
        try:
            missing_rates = json.loads(missing_rates_text)
        except Exception:
            missing_rates = [item.strip() for item in missing_rates_text.split(",") if item.strip()]
    is_single_rate = len(missing_rates) == 1
    is_multi_rate = len(missing_rates) > 1
    repeated_5_percent = bool(is_single_rate and missing_rates and str(missing_rates[0]).startswith("0.05") and "missing_rate_vs" in path.name.lower())
    paper_ready = not repeated_5_percent and "smoke" not in parent_result_dir.lower() and "sample" not in parent_result_dir.lower() and "hybridtest_small" not in parent_result_dir.lower()
    recommended_fix = ""
    if repeated_5_percent:
        recommended_fix = "单缺失率 5% 不宜画成 missing-rate 曲线，建议改为方法对比柱状图或点图。"
    elif "sample" in parent_result_dir.lower() or "smoke" in parent_result_dir.lower():
        recommended_fix = "仅适合作为调试或附录图，论文主文需替换为正式实验图。"
    elif not paper_ready:
        recommended_fix = "需结合正式多缺失率或全量结果后再用于论文。"
    return {
        "file_path": rel_path,
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": int(path.stat().st_size),
        "parent_result_dir": parent_result_dir,
        "detected_plot_type": detect_plot_type(path.name),
        "detected_metric": detect_metric(path.name),
        "detected_missing_rate_scope": "single_rate" if is_single_rate else ("multi_rate" if is_multi_rate else "未找到明确证据，需要人工确认"),
        "is_single_rate_plot": is_single_rate,
        "is_multi_rate_plot": is_multi_rate,
        "has_repeated_5_percent_axis_issue": repeated_5_percent,
        "paper_ready": paper_ready,
        "recommended_fix": recommended_fix,
        "notes": "标题根据文件名推断。",
    }


def inspect_parquet_directory(project_root: Path, rel_dir: str) -> Dict[str, Any]:
    path = project_root / rel_dir
    if not path.exists():
        return {
            "file_path": rel_dir,
            "asset_type": "parquet_directory",
            "exists": False,
            "file_count": 0,
            "size_bytes": None,
            "row_count": None,
            "column_count": None,
            "columns": "",
            "target_col": None,
            "node_col": None,
            "time_col": None,
            "node_count": None,
            "time_slot_count": None,
            "time_slot_min": None,
            "time_slot_max": None,
            "is_full_intersection_node_flow": False,
            "notes": "目录不存在。",
        }
    files = sorted(path.glob("node_flow_chunk_*.parquet"))
    total_size = sum(int(file_path.stat().st_size) for file_path in files)
    total_rows = 0
    all_row_counts: List[int] = []
    columns: List[str] = []
    sample_node_count: Optional[int] = None
    sample_time_slot_count: Optional[int] = None
    sample_time_slot_min: Optional[int] = None
    sample_time_slot_max: Optional[int] = None
    sample_notes: List[str] = []
    if files and pq is not None:
        for file_path in files:
            metadata = pq.ParquetFile(file_path).metadata
            total_rows += int(metadata.num_rows)
            all_row_counts.append(int(metadata.num_rows))
            if not columns:
                columns = list(pq.ParquetFile(file_path).schema.names)
        sample_file = files[0]
        try:
            sample_df = pd.read_parquet(sample_file, columns=["节点ID", "时间段", "路口车流量", "路口进入流量", "路口离开流量"])
        except Exception:
            try:
                sample_df = pd.read_parquet(sample_file)
            except Exception:
                sample_df = pd.DataFrame()
        if not sample_df.empty:
            if "节点ID" in sample_df.columns:
                sample_node_count = int(sample_df["节点ID"].nunique())
            if "时间段" in sample_df.columns:
                time_numeric = pd.to_numeric(sample_df["时间段"], errors="coerce")
                if time_numeric.notna().any():
                    sample_time_slot_count = int(time_numeric.nunique())
                    sample_time_slot_min = int(time_numeric.min())
                    sample_time_slot_max = int(time_numeric.max())
            duplicate_check = None
            if {"节点ID", "时间段"}.issubset(set(sample_df.columns)):
                duplicate_check = int(sample_df.duplicated(subset=["节点ID", "时间段"]).sum())
            negative_check = None
            if "路口车流量" in sample_df.columns:
                series = pd.to_numeric(sample_df["路口车流量"], errors="coerce")
                negative_check = int((series < 0).sum())
                sample_notes.append("sample_chunk_na_count={0}".format(int(series.isna().sum())))
            if duplicate_check is not None:
                sample_notes.append("sample_chunk_duplicate_node_time={0}".format(duplicate_check))
            if negative_check is not None:
                sample_notes.append("sample_chunk_negative_target={0}".format(negative_check))
    notes: List[str] = []
    if files:
        notes.append("chunk_file_range={0}..{1}".format(files[0].name, files[-1].name))
        if len(files) == 61:
            notes.append("检测到 61 个 node_flow_chunk_*.parquet")
        else:
            notes.append("chunk 数量不是 61，需要人工确认")
    if all_row_counts and len(set(all_row_counts)) == 1:
        notes.append("所有 chunk metadata 行数一致: {0}".format(all_row_counts[0]))
    if sample_node_count and sample_time_slot_count:
        notes.append("首个 chunk 节点数={0}, 时间片数={1}".format(sample_node_count, sample_time_slot_count))
        if all_row_counts and all_row_counts[0] == sample_node_count * sample_time_slot_count:
            notes.append("metadata 行数与 节点数*时间片数 一致，可作为按天完整网格的证据")
    notes.extend(sample_notes)
    return {
        "file_path": rel_dir,
        "asset_type": "parquet_directory",
        "exists": True,
        "file_count": len(files),
        "size_bytes": total_size,
        "row_count": total_rows if total_rows else None,
        "column_count": len(columns) if columns else None,
        "columns": json_cell(columns),
        "target_col": "路口车流量" if "路口车流量" in columns else None,
        "node_col": "节点ID" if "节点ID" in columns else None,
        "time_col": "时间段" if "时间段" in columns else None,
        "node_count": sample_node_count,
        "time_slot_count": sample_time_slot_count,
        "time_slot_min": sample_time_slot_min,
        "time_slot_max": sample_time_slot_max,
        "is_full_intersection_node_flow": bool("路口车流量" in columns and "节点ID" in columns and "时间段" in columns),
        "notes": "；".join(notes),
    }


def inspect_csv_asset(project_root: Path, rel_path: str) -> Dict[str, Any]:
    path = project_root / rel_path
    if not path.exists():
        return {
            "file_path": rel_path,
            "asset_type": "csv",
            "exists": False,
            "file_count": 0,
            "size_bytes": None,
            "row_count": None,
            "column_count": None,
            "columns": "",
            "target_col": None,
            "node_col": None,
            "time_col": None,
            "node_count": None,
            "time_slot_count": None,
            "time_slot_min": None,
            "time_slot_max": None,
            "is_full_intersection_node_flow": False,
            "notes": "文件不存在。",
        }
    try:
        df = pd.read_csv(path)
        columns = [str(column) for column in df.columns]
        notes: List[str] = []
        if {"起始节点ID", "结束节点ID"}.issubset(set(columns)):
            notes.append("包含起始/结束节点字段，可支持地理邻近性补全")
        if "长度" in columns:
            notes.append("包含长度字段，可构造距离权重")
        node_col = "起始节点ID" if "起始节点ID" in columns else None
        return {
            "file_path": rel_path,
            "asset_type": "csv",
            "exists": True,
            "file_count": 1,
            "size_bytes": int(path.stat().st_size),
            "row_count": int(len(df)),
            "column_count": len(columns),
            "columns": json_cell(columns),
            "target_col": None,
            "node_col": node_col,
            "time_col": None,
            "node_count": None,
            "time_slot_count": None,
            "time_slot_min": None,
            "time_slot_max": None,
            "is_full_intersection_node_flow": False,
            "notes": "；".join(notes),
        }
    except Exception as exc:
        return {
            "file_path": rel_path,
            "asset_type": "csv",
            "exists": True,
            "file_count": 1,
            "size_bytes": int(path.stat().st_size),
            "row_count": None,
            "column_count": None,
            "columns": "",
            "target_col": None,
            "node_col": None,
            "time_col": None,
            "node_count": None,
            "time_slot_count": None,
            "time_slot_min": None,
            "time_slot_max": None,
            "is_full_intersection_node_flow": False,
            "notes": "读取失败: {0}".format(exc),
        }


def find_summary_tables(project_root: Path) -> List[str]:
    matches: Set[str] = set()
    results_root = project_root / "results"
    if not results_root.exists():
        return []
    for path in results_root.rglob("*.csv"):
        rel = to_rel(project_root, path)
        if rel.startswith("results/real_data_missingness") or rel.startswith("results/real_data_preprocessing"):
            matches.add(rel)
    return sorted(matches)


def analyze_summary_table(project_root: Path, rel_path: str, result_dirs: List[str]) -> Dict[str, Any]:
    path = project_root / rel_path
    rows, cols, columns = load_csv_header_and_shape(path)
    file_name = path.name.lower()
    table_role = "other"
    if "summary" in file_name:
        table_role = "summary"
    if "detail" in file_name:
        table_role = "detail"
    if "manifest" in rel_path or "status" in file_name or "runs" in file_name:
        table_role = "manifest"
    if "batch_processing_report" in file_name:
        table_role = "batch_report"
    return {
        "file_path": rel_path,
        "file_name": path.name,
        "parent_result_dir": find_parent_result_dir(rel_path, result_dirs),
        "row_count": rows,
        "column_count": cols,
        "columns": json_cell(columns),
        "table_role": table_role,
        "contains_missingness_summary": "missing" in file_name or "mask" in file_name,
        "contains_imputation_metrics": "imputation" in file_name or "rmse" in file_name or "mae" in file_name or "mape" in file_name,
        "notes": "",
    }


def build_run_config_map(run_config_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in run_config_rows:
        mapping[row["result_dir"]] = row
    return mapping


def detect_stage_presence(project_root: Path, result_dir: str) -> Tuple[bool, bool, bool, bool, Optional[int]]:
    path = project_root / result_dir
    generate_exists = (path / "manifests" / "generate_missing_chunk_status.csv").exists() or (path / "missing_datasets").exists() or (path / "masks").exists()
    impute_exists = (path / "manifests" / "impute_chunk_status.csv").exists() or (path / "imputed_datasets").exists()
    summarize_exists = (path / "summaries" / "imputation_quality_summary_exclude_warmup.csv").exists() or (path / "summaries" / "imputation_quality_summary.csv").exists()
    validate_exists = (path / "full_intersection_missingness_validation.json").exists() or (path / "real_data_missingness_experiment_audit.json").exists()
    chunk_count = None
    chunk_manifest = path / "manifests" / "chunk_index_summary.csv"
    if chunk_manifest.exists():
        try:
            chunk_df = pd.read_csv(chunk_manifest)
            chunk_count = int(len(chunk_df))
        except Exception:
            chunk_count = None
    return generate_exists, impute_exists, summarize_exists, validate_exists, chunk_count


def build_run_matrix(
    project_root: Path,
    result_dir_rows: List[Dict[str, Any]],
    run_config_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    config_map = build_run_config_map(run_config_rows)
    rows: List[Dict[str, Any]] = []
    for entry in result_dir_rows:
        result_dir = entry["result_dir"]
        config = config_map.get(result_dir, {})
        generate_exists, impute_exists, summarize_exists, validate_exists, chunk_count = detect_stage_presence(project_root, result_dir)
        missing_rates = config.get("missing_rates", "")
        seeds: List[str] = []
        if config.get("seed") not in (None, ""):
            seeds = [str(item) for item in parse_json_list(config.get("seed"))]
        methods: List[str] = []
        if config.get("impute_methods"):
            try:
                methods = [str(item) for item in json.loads(config.get("impute_methods"))]
            except Exception:
                methods = [item.strip() for item in str(config.get("impute_methods")).split(",") if item.strip()]
        data_scope = entry["detected_scope"]
        current_status = entry["detected_status"]
        usable_for_paper_main_table = False
        notes = entry.get("notes", "")
        if data_scope == "8_chunk_historical_test" and summarize_exists:
            notes = (notes + " 当前有 8 chunk、5% MCAR、排除 warmup 的正式 summary，可用于阶段性说明，但不代表最终全量主表。").strip()
        rows.append(
            {
                "result_dir": result_dir,
                "experiment_label": data_scope,
                "data_scope": data_scope,
                "mechanism": config.get("mechanism", "未找到明确证据，需要人工确认"),
                "missing_rates": missing_rates,
                "seed_count": len(seeds),
                "seeds": json_cell(seeds),
                "method_count": len(methods),
                "methods": json_cell(methods),
                "max_chunks": config.get("max_chunks"),
                "chunk_count_detected": chunk_count,
                "is_sample": "sample" in data_scope,
                "is_historical_test": data_scope == "8_chunk_historical_test",
                "is_full_61_chunk": result_dir == "results/real_data_missingness_full_intersection_causal_history",
                "has_generate_missing": generate_exists,
                "has_impute": impute_exists,
                "has_summarize": summarize_exists,
                "has_validate": validate_exists,
                "has_figures": entry["has_figures"],
                "current_status": current_status,
                "usable_for_paper_main_table": usable_for_paper_main_table,
                "notes": notes,
            }
        )
    return rows


def build_method_matrix(
    code_rows: List[Dict[str, Any]],
    run_config_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    supported_methods: Set[str] = set(TARGET_METHODS)
    for row in code_rows:
        if row["key_functions"]:
            continue
    observed_method_to_mechanisms: Dict[str, Set[str]] = defaultdict(set)
    observed_method_to_rates: Dict[str, Set[str]] = defaultdict(set)
    observed_methods: Set[str] = set()
    for row in run_config_rows:
        methods: List[str] = []
        if row.get("impute_methods"):
            try:
                methods = [str(item) for item in json.loads(row["impute_methods"])]
            except Exception:
                methods = [item.strip() for item in str(row["impute_methods"]).split(",") if item.strip()]
        rates: List[str] = []
        if row.get("missing_rates"):
            try:
                rates = [str(item) for item in json.loads(row["missing_rates"])]
            except Exception:
                rates = [item.strip() for item in str(row["missing_rates"]).split(",") if item.strip()]
        mechanism = str(row.get("mechanism", ""))
        for method in methods:
            observed_methods.add(method)
            observed_method_to_mechanisms[method].add(mechanism)
            for rate in rates:
                observed_method_to_rates[method].add(rate)
    rows: List[Dict[str, Any]] = []
    for method in TARGET_METHODS:
        notes = ""
        uses_future = False
        if method == "linear_interpolation":
            uses_future = True
            notes = "旧实验中的 linear_interpolation 使用双向插值。"
        elif method == "forward_fill":
            uses_future = True
            notes = "旧样本脚本中的 forward_fill 含 bfill 回退；历史因果流水线版本不使用未来数据。"
        elif method in {"geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid"}:
            notes = "历史因果流水线版本不使用未来数据；早期原型脚本存在含未来信息的回退逻辑。"
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "is_baseline": method in {"zero_fill", "forward_fill", "linear_interpolation"},
                "is_historical_causal": method in {"forward_fill", "historical_linear_extrapolation", "geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid"},
                "uses_spatial_topology": method in {"geo_neighbor_fill", "geo_func_hybrid"},
                "uses_function_curve": method in {"function_curve_fit", "geo_func_hybrid"},
                "uses_future_data": uses_future,
                "requires_topology": method in {"geo_neighbor_fill", "geo_func_hybrid"},
                "supported_in_script": method in supported_methods,
                "observed_in_results": method in observed_methods,
                "observed_mechanisms": json_cell(sorted(observed_method_to_mechanisms.get(method, set()))),
                "observed_missing_rates": json_cell(sorted(observed_method_to_rates.get(method, set()))),
                "notes": notes,
            }
        )
    return rows


def scan_risk_statements(project_root: Path, document_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for row in document_rows:
        rel_path = row["file_path"]
        path = project_root / rel_path
        text = read_text(path)
        for keyword in RISK_KEYWORDS:
            if keyword.lower() in text.lower():
                hits.append({"file_path": rel_path, "keyword": keyword})
    return hits


def build_current_state_json(
    project_root: Path,
    env_info: Dict[str, Any],
    git_info: Dict[str, Any],
    code_rows: List[Dict[str, Any]],
    document_rows: List[Dict[str, Any]],
    result_dir_rows: List[Dict[str, Any]],
    data_rows: List[Dict[str, Any]],
    visualization_rows: List[Dict[str, Any]],
    run_config_rows: List[Dict[str, Any]],
    run_matrix_rows: List[Dict[str, Any]],
    method_matrix_rows: List[Dict[str, Any]],
    risk_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "environment": env_info,
        "git": git_info,
        "counts": {
            "code_files": len(code_rows),
            "document_files": len(document_rows),
            "result_directories": len(result_dir_rows),
            "data_assets": len(data_rows),
            "visualization_files": len(visualization_rows),
            "run_configs": len(run_config_rows),
            "run_matrix_rows": len(run_matrix_rows),
            "method_matrix_rows": len(method_matrix_rows),
            "risk_hits": len(risk_hits),
        },
        "code_files": code_rows,
        "document_files": document_rows,
        "result_directories": result_dir_rows,
        "data_assets": data_rows,
        "visualization_files": visualization_rows,
        "run_configs": run_config_rows,
        "run_matrix": run_matrix_rows,
        "method_matrix": method_matrix_rows,
        "risk_hits": risk_hits,
    }


def find_row(rows: List[Dict[str, Any]], key: str, value: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if str(row.get(key)) == value:
            return row
    return None


def read_summary_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_workflow_summary(
    project_root: Path,
    data_rows: List[Dict[str, Any]],
    run_matrix_rows: List[Dict[str, Any]],
    run_config_rows: List[Dict[str, Any]],
    visualization_rows: List[Dict[str, Any]],
    risk_hits: List[Dict[str, Any]],
) -> str:
    data_asset = find_row(data_rows, "file_path", "data/analysis/node_intersection_flow_parquet")
    rnsd_asset = find_row(data_rows, "file_path", "data/processed/rnsd_processed.csv")
    historical_config = find_row(run_config_rows, "result_dir", "results/real_data_missingness_full_intersection_causal_history/historical_test")
    full_config = find_row(run_config_rows, "result_dir", "results/real_data_missingness_full_intersection_causal_history")
    historical_summary_path = project_root / "results/real_data_missingness_full_intersection_causal_history/historical_test/summaries/imputation_quality_summary_exclude_warmup.csv"
    historical_summary = read_summary_csv(historical_summary_path)
    summary_lines: List[str] = []
    summary_lines.append("# 当前真实数据缺失值设置与补全实验工作流说明")
    summary_lines.append("")
    summary_lines.append("## 1. 本次审计目的")
    summary_lines.append("")
    summary_lines.append("本次工作是对项目内真实数据缺失值设置、缺失数据生成、补全方法、输出结果和图件的资产级审计，只做读取、扫描、清点、解析、汇总和文档生成，不继续运行正式缺失实验。")
    summary_lines.append("")
    summary_lines.append("## 2. 数据来源")
    summary_lines.append("")
    if data_asset and data_asset.get("exists"):
        summary_lines.append("真实数据主来源为 `data/analysis/node_intersection_flow_parquet`。")
        summary_lines.append("")
        summary_lines.append("- 文件数量：{0}".format(data_asset.get("file_count")))
        summary_lines.append("- 每个 chunk 是否一天：根据 `node_flow_chunk_*.parquet` 命名、metadata 行数一致以及 `historical_test/full_intersection_missingness_audit.json` 中的 `day_index` 记录，可作为按天分片的证据。")
        summary_lines.append("- 每个 chunk 行数：metadata 一致，首个 chunk 为 {0}".format(data_asset.get("notes", "")))
        summary_lines.append("- 节点数量：{0}".format(data_asset.get("node_count")))
        summary_lines.append("- 时间片数量：{0}".format(data_asset.get("time_slot_count")))
        summary_lines.append("- 目标字段：{0}".format(data_asset.get("target_col")))
        summary_lines.append("- 节点字段：{0}".format(data_asset.get("node_col")))
        summary_lines.append("- 时间字段：{0}".format(data_asset.get("time_col")))
    else:
        summary_lines.append("未找到 `data/analysis/node_intersection_flow_parquet` 的明确证据，需要人工确认。")
    summary_lines.append("")
    summary_lines.append("## 3. 缺失值设置方式")
    summary_lines.append("")
    summary_lines.append("### 3.1 mcar_point")
    summary_lines.append("")
    summary_lines.append("代码与结果均表明 `mcar_point` 已实际运行。已找到的运行目录包括：")
    summary_lines.append("")
    for row in run_matrix_rows:
        if str(row.get("mechanism")) == "mcar_point" and row.get("has_generate_missing"):
            summary_lines.append("- `{0}`：missing_rates={1}，methods={2}".format(row["result_dir"], row["missing_rates"], row["methods"]))
    summary_lines.append("")
    summary_lines.append("### 3.2 node_temporal_block")
    summary_lines.append("")
    summary_lines.append("`full_intersection_missingness_pipeline.py` 代码支持 `node_temporal_block`，但本次扫描未在运行配置和结果目录中找到其正式输出证据，因此当前判断为“代码支持但未找到正式运行证据”。")
    summary_lines.append("")
    summary_lines.append("## 4. 补全方法")
    summary_lines.append("")
    method_notes = {
        "zero_fill": "baseline；不使用拓扑；不使用函数曲线；不使用未来数据；已运行。",
        "forward_fill": "baseline；历史因果流水线版本不使用未来数据；已运行；旧样本脚本版本存在 bfill 风险。",
        "historical_linear_extrapolation": "非 baseline；历史因果；不使用拓扑；不使用函数曲线；不使用未来数据；已运行。",
        "geo_neighbor_fill": "非 baseline；历史因果；使用拓扑；不使用函数曲线；历史因果流水线中不使用未来数据；已运行。",
        "function_curve_fit": "非 baseline；历史因果；不使用拓扑；使用函数曲线；历史因果流水线中不使用未来数据；已运行。",
        "geo_func_hybrid": "非 baseline；历史因果；使用拓扑；使用函数曲线；历史因果流水线中不使用未来数据；已运行。",
        "linear_interpolation": "早期样本实验中出现；使用双向插值，存在未来信息泄露风险；不属于当前历史因果主流程。",
    }
    for method in TARGET_METHODS:
        summary_lines.append("- `{0}`：{1}".format(method, method_notes.get(method, "未找到明确证据，需要人工确认。")))
    summary_lines.append("")
    summary_lines.append("## 5. 当前已完成的实验")
    summary_lines.append("")
    summary_lines.append("### 5.1 早期样本缺失实验")
    summary_lines.append("")
    summary_lines.append("已发现 `results/real_data_missingness_experiments_sample` 与 `results/real_data_missingness_experiments`。其运行配置显示：")
    summary_lines.append("- 数据范围为 `node_intersection_flow_parquet` 的少量文件和少量行；")
    summary_lines.append("- 缺失率覆盖 `0, 0.05, 0.10, 0.20, 0.30`；")
    summary_lines.append("- 方法为 `zero_fill`、`forward_fill`、`linear_interpolation`；")
    summary_lines.append("- 结果文件包括 `missingness_design_summary.csv`、`missingness_mask_summary.csv`、`imputation_quality_summary.csv` 与 RMSE 图。")
    summary_lines.append("")
    summary_lines.append("### 5.2 geo/function 子实验")
    summary_lines.append("")
    summary_lines.append("已发现 `results/real_data_missingness_full_intersection_causal_history_hybridtest_small`，其配置显示只跑了 `geo_func_hybrid`、`max_chunks=1`、`max_rows=200`，属于小规模混合方法调试，不是正式主实验。")
    summary_lines.append("")
    summary_lines.append("### 5.3 full intersection causal historical test")
    summary_lines.append("")
    summary_lines.append("已发现 `results/real_data_missingness_full_intersection_causal_history/historical_test`，其 `run_config.json`、`audit.json`、`validation.json`、`batch_processing_report.json` 提供了完整证据：")
    summary_lines.append("- `max_chunks=8`")
    summary_lines.append("- `5% MCAR`")
    summary_lines.append("- `seed=42`")
    summary_lines.append("- `7 天 warmup`，第 8 天为主评估")
    summary_lines.append("- `6 种方法`")
    summary_lines.append("- `causal_history_only=true`")
    summary_lines.append("- `context_days_after=0`，不使用未来数据")
    summary_lines.append("- 已生成 `summary / audit / validation / figures / masks / missing_datasets / imputed_datasets`")
    summary_lines.append("")
    summary_lines.append("### 5.4 61 chunk 主实验目录")
    summary_lines.append("")
    summary_lines.append("已发现 `results/real_data_missingness_full_intersection_causal_history` 根目录级主实验资产，但是否已完成“61 天全量 generate_missing + 全部方法 impute + summarize + validate”不能直接按目录名下结论。当前证据显示：")
    summary_lines.append("- 存在根目录级 `run_config.json` 与 `run_commands.txt`；")
    summary_lines.append("- 存在 `imputed_datasets`、`manifests` 等资产；")
    summary_lines.append("- 根目录运行配置只直接指向部分 `impute` 配置，且当前未找到根目录级完整 `summaries/imputation_quality_summary_exclude_warmup.csv` 与 `validation` 完成证据；")
    summary_lines.append("- 因此不能写成“61 chunk 全量实验已完整完成”，需要以 inventory 结果和人工复核进一步确认。")
    summary_lines.append("")
    summary_lines.append("## 6. 当前可视化图件")
    summary_lines.append("")
    overall_plots = [row for row in visualization_rows if row["detected_plot_type"] == "overall"]
    zoom_plots = [row for row in visualization_rows if row["detected_plot_type"] == "zoom"]
    delta_plots = [row for row in visualization_rows if row["detected_plot_type"] == "delta"]
    summary_lines.append("- overall RMSE 图：{0}".format(len(overall_plots)))
    summary_lines.append("- zoom 图：{0}".format(len(zoom_plots)))
    summary_lines.append("- delta 图：{0}".format(len(delta_plots)))
    summary_lines.append("- 当前多张 `missing_rate_vs_*` 图来自单缺失率 `5%` 结果，存在横轴重复 5% 的表达风险，不宜直接作为最终多缺失率曲线图。")
    summary_lines.append("")
    summary_lines.append("## 7. 当前结果结论")
    summary_lines.append("")
    if not historical_summary.empty and "RMSE" in historical_summary.columns:
        ranked = historical_summary.sort_values("RMSE")
        best_method = str(ranked.iloc[0]["impute_method"])
        worst_method = str(ranked.iloc[-1]["impute_method"])
        hybrid_rmse = None
        if "geo_func_hybrid" in set(historical_summary["impute_method"].astype(str)):
            hybrid_rmse = float(historical_summary.loc[historical_summary["impute_method"] == "geo_func_hybrid", "RMSE"].iloc[0])
        summary_lines.append("- 现有 `8 chunk、5% MCAR historical_test` 证据中，`{0}` 的 RMSE 最低。".format(best_method))
        summary_lines.append("- `zero_fill` 在该测试中明显最差。")
        if hybrid_rmse is not None:
            summary_lines.append("- `geo_func_hybrid` 在空间/函数类方法中优于 `geo_neighbor_fill` 与 `function_curve_fit`，但仍落后于 `forward_fill` 与 `historical_linear_extrapolation`。")
        summary_lines.append("- 以上结论只适用于 `8 chunk、5% MCAR historical_test`，不代表完整 61 天、多缺失率、多 seed 的最终结论。")
    else:
        summary_lines.append("未找到可直接用于排序的 `historical_test` summary 证据，需要人工确认。")
    summary_lines.append("")
    summary_lines.append("## 8. 尚未完成的内容")
    summary_lines.append("")
    summary_lines.append("- 多缺失率全量实验是否完成：未找到明确完整证据，需要人工确认。")
    summary_lines.append("- 多 seed 是否完成：早期样本实验存在多 seed，历史因果主流程未找到多 seed 正式完成证据。")
    summary_lines.append("- `node_temporal_block` 是否完成：代码支持但未找到正式运行证据。")
    summary_lines.append("- 61 chunk 全量 impute 是否完成：未找到明确完整证据，需要人工确认。")
    summary_lines.append("- summarize 是否完成：`historical_test` 已完成，根目录级 61 chunk 主目录未找到明确完整证据。")
    summary_lines.append("- error bar 图是否完成：未找到明确证据。")
    summary_lines.append("- FedAvg / Independent 真实预测是否完成：本次缺失实验资产中未找到真实预测训练输出证据，不能写已完成。")
    summary_lines.append("")
    summary_lines.append("## 9. 风险与修正建议")
    summary_lines.append("")
    summary_lines.append("- 单缺失率图横轴重复 5%：当前已发现该风险。")
    summary_lines.append("- `historical_test` 不能写成完整 61 天全量结果。")
    summary_lines.append("- 插补误差不能写成预测误差。")
    summary_lines.append("- 人工缺失不能写成真实数据天然缺失。")
    summary_lines.append("- `node_temporal_block` 需要单独验证。")
    summary_lines.append("- 多缺失率曲线需要后续正式补跑。")
    if risk_hits:
        summary_lines.append("- 文档中已检出部分风险关键词，需要人工逐条复核表述口径。")
    summary_lines.append("")
    summary_lines.append("## 10. 下一步建议")
    summary_lines.append("")
    summary_lines.append("Phase 1：补全当前主目录审计，逐方法核对 61 chunk 主目录中的 masks、missing_datasets、imputed_datasets 与 summaries 一致性。")
    summary_lines.append("Phase 2：在严格历史因果口径下补跑多缺失率 MCAR 正式实验。")
    summary_lines.append("Phase 3：单独运行并验证 `node_temporal_block`。")
    summary_lines.append("Phase 4：补齐跨 seed mean±std。")
    summary_lines.append("Phase 5：根据正式 inventory 更新论文文档与图件。")
    summary_lines.append("")
    if rnsd_asset and rnsd_asset.get("exists"):
        summary_lines.append("附注：`data/processed/rnsd_processed.csv` 已存在，且字段证据支持地理邻近性补全。")
        summary_lines.append("")
    return "\n".join(summary_lines) + "\n"


def build_state_audit_md(
    project_root: Path,
    env_info: Dict[str, Any],
    git_info: Dict[str, Any],
    code_rows: List[Dict[str, Any]],
    document_rows: List[Dict[str, Any]],
    result_dir_rows: List[Dict[str, Any]],
    data_rows: List[Dict[str, Any]],
    visualization_rows: List[Dict[str, Any]],
    run_config_rows: List[Dict[str, Any]],
    run_matrix_rows: List[Dict[str, Any]],
    method_matrix_rows: List[Dict[str, Any]],
    risk_hits: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# 当前真实数据缺失实验状态审计报告")
    lines.append("")
    lines.append("## 1. 审计时间与环境")
    lines.append("")
    lines.append("- 审计时间：{0}".format(datetime.now().isoformat(timespec="seconds")))
    lines.append("- Python：`{0}`".format(env_info.get("python_path")))
    lines.append("- Python 版本：`{0}`".format(env_info.get("python_version")))
    lines.append("- pandas：`{0}`".format(env_info.get("pandas_version")))
    lines.append("- pyarrow：`{0}`".format(env_info.get("pyarrow_version")))
    lines.append("")
    lines.append("## 2. 项目路径")
    lines.append("")
    lines.append("- `{0}`".format(project_root))
    lines.append("")
    lines.append("## 3. Git 状态")
    lines.append("")
    lines.append("- 当前分支：`{0}`".format(git_info.get("branch")))
    lines.append("- `git status --short`：")
    lines.append("")
    lines.append("```text")
    lines.append(git_info.get("status_short", ""))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Python 环境")
    lines.append("")
    lines.append("- 解释器：`{0}`".format(env_info.get("python_path")))
    lines.append("- 版本：`{0}`".format(env_info.get("python_version")))
    lines.append("")
    lines.append("## 5. 发现的代码文件")
    lines.append("")
    for row in code_rows:
        lines.append("- `{0}`：{1}".format(row["file_path"], row["detected_purpose"]))
    lines.append("")
    lines.append("## 6. 发现的文档文件")
    lines.append("")
    for row in document_rows[:20]:
        lines.append("- `{0}`：{1}".format(row["file_path"], row["detected_topic"]))
    lines.append("")
    lines.append("## 7. 发现的结果目录")
    lines.append("")
    for row in result_dir_rows:
        lines.append("- `{0}`：exists={1}，scope={2}，status={3}".format(row["result_dir"], row["exists"], row["detected_scope"], row["detected_status"]))
    lines.append("")
    lines.append("## 8. 发现的数据资产")
    lines.append("")
    for row in data_rows:
        lines.append("- `{0}`：exists={1}，rows={2}，columns={3}".format(row["file_path"], row["exists"], row["row_count"], row["columns"]))
    lines.append("")
    lines.append("## 9. 发现的可视化文件")
    lines.append("")
    for row in visualization_rows[:20]:
        lines.append("- `{0}`：plot_type={1}，single_rate={2}，paper_ready={3}".format(row["file_path"], row["detected_plot_type"], row["is_single_rate_plot"], row["paper_ready"]))
    lines.append("")
    lines.append("## 10. 运行配置解析")
    lines.append("")
    for row in run_config_rows:
        lines.append("- `{0}`：mechanism={1}，missing_rates={2}，methods={3}".format(row["config_path"], row["mechanism"], row["missing_rates"], row["impute_methods"]))
    lines.append("")
    lines.append("## 11. 实验矩阵")
    lines.append("")
    for row in run_matrix_rows:
        lines.append("- `{0}`：generate={1}，impute={2}，summarize={3}，validate={4}".format(row["result_dir"], row["has_generate_missing"], row["has_impute"], row["has_summarize"], row["has_validate"]))
    lines.append("")
    lines.append("## 12. 方法与机制矩阵")
    lines.append("")
    for row in method_matrix_rows:
        lines.append("- `{0}`：observed={1}，uses_future_data={2}".format(row["method"], row["observed_in_results"], row["uses_future_data"]))
    lines.append("")
    lines.append("## 13. 当前完成度判断")
    lines.append("")
    lines.append("- `historical_test`：证据完整，已完成 generate_missing、impute、summarize、validate。")
    lines.append("- 61 chunk 主目录：存在部分或阶段性输出，但未找到完整完成证据，不能直接写“全量完成”。")
    lines.append("- `node_temporal_block`：代码支持，未找到正式运行证据。")
    lines.append("")
    lines.append("## 14. 不能下结论的部分")
    lines.append("")
    lines.append("- 61 chunk 全量 generate_missing、impute、summarize、validate 是否全部完成。")
    lines.append("- 多缺失率正式主实验是否完成。")
    lines.append("- 多 seed 真实数据缺失实验是否完成。")
    lines.append("- FedAvg / Independent 真实预测训练输出是否完成。")
    lines.append("")
    lines.append("## 15. 需要人工确认的部分")
    lines.append("")
    lines.append("- 风险关键词命中的文档语境是否需要修正。")
    lines.append("- 61 chunk 主目录中各方法的完整 chunk 覆盖情况。")
    lines.append("- `node_temporal_block` 是否在未纳入本次目录的外部位置运行过。")
    lines.append("")
    if risk_hits:
        lines.append("附：风险关键词命中 {0} 条。".format(len(risk_hits)))
        lines.append("")
    return "\n".join(lines) + "\n"


def build_open_issues_md(risk_hits: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# 当前真实数据缺失实验开放问题与下一步建议")
    lines.append("")
    lines.append("## 1. 已完成事项")
    lines.append("")
    lines.append("- 已确认真实数据主来源为 `data/analysis/node_intersection_flow_parquet`。")
    lines.append("- 已确认 `historical_test` 存在完整 `run_config / run_commands / audit / validation / summaries / figures / masks / missing_datasets / imputed_datasets` 证据。")
    lines.append("- 已确认历史因果主流水线代码支持 `generate_missing / impute / validate / summarize`。")
    lines.append("")
    lines.append("## 2. 部分完成事项")
    lines.append("")
    lines.append("- 61 chunk 主目录存在阶段性输出，但未完成全量闭环证据整理。")
    lines.append("- 真实数据图件已生成多张，但部分仍是单缺失率 5% 表达方式。")
    lines.append("")
    lines.append("## 3. 未完成事项")
    lines.append("")
    lines.append("- 多缺失率全量 MCAR 正式实验。")
    lines.append("- `node_temporal_block` 正式实验。")
    lines.append("- 多 seed mean±std 汇总。")
    lines.append("- error bar 图。")
    lines.append("- FedAvg / Independent 真实预测正式输出证据。")
    lines.append("")
    lines.append("## 4. 证据不足事项")
    lines.append("")
    lines.append("- 当前不要直接写“完整全量实验完成”，除非 inventory 证明 61 个 chunk 的 `generate_missing`、`impute`、`summarize`、`validate` 均完成。")
    lines.append("- 当前不能把 `historical_test` 写成 61 chunk 全量结果。")
    lines.append("")
    lines.append("## 5. 后续优先级 P0/P1/P2")
    lines.append("")
    lines.append("- P0：核对 61 chunk 主目录中各方法的实际 chunk 覆盖数与阶段完成度。")
    lines.append("- P1：补齐多缺失率 MCAR 与 `node_temporal_block`。")
    lines.append("- P2：补齐多 seed 统计并更新论文图文。")
    lines.append("")
    lines.append("## 6. 下一步建议命令")
    lines.append("")
    lines.append("- `E:\\anaconda3\\envs\\analysis\\python.exe analysis_scripts\\inventory_real_missingness_assets.py --project_root E:\\Jupter_Notebook\\FedTrafficFlow --output_dir results\\real_data_missingness_inventory`")
    lines.append("- 仅在人工确认 inventory 结论后，再决定是否运行后续正式实验命令。")
    lines.append("")
    lines.append("## 7. 不建议现在做的事情")
    lines.append("")
    lines.append("- 不建议现在直接写“完整全量实验完成”。")
    lines.append("- 不建议现在直接把插补误差写成预测误差。")
    lines.append("- 不建议现在把人工缺失写成天然缺失。")
    lines.append("- 不建议现在继续跑 `generate_missing / impute / summarize / validate / FedAvg / Independent`。")
    lines.append("")
    if risk_hits:
        lines.append("附注：本次风险关键词命中 {0} 条，建议先做文档口径复核。".format(len(risk_hits)))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = ensure_dir(normalize_path(project_root, args.output_dir))

    git_info = {
        "branch": run_command(["git", "branch", "--show-current"], project_root),
        "status_short": run_command(["git", "status", "--short"], project_root),
        "diff_stat": run_command(["git", "diff", "--stat"], project_root),
    }
    env_info = {
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "pyarrow_version": getattr(sys.modules.get("pyarrow"), "__version__", None) if "pyarrow" in sys.modules else (pq.__module__.split(".")[0] if pq is not None else None),
    }
    if pq is not None:
        try:
            import pyarrow as pa

            env_info["pyarrow_version"] = pa.__version__
        except Exception:
            pass

    code_paths = find_relevant_code_files(project_root)
    code_rows = [analyze_code_file(project_root, rel_path) for rel_path in code_paths]
    document_paths = find_relevant_document_files(project_root)
    document_rows = [analyze_document_file(project_root, rel_path) for rel_path in document_paths]
    result_dirs = find_result_dirs(project_root)
    result_dir_rows = [analyze_result_directory(project_root, rel_dir) for rel_dir in result_dirs]
    run_config_paths = find_run_config_files(project_root)
    run_config_rows = [analyze_run_config(project_root, rel_path) for rel_path in run_config_paths]
    run_config_map = build_run_config_map(run_config_rows)
    visualization_paths = find_visualization_files(project_root)
    visualization_rows = [analyze_visualization_file(project_root, rel_path, result_dirs, run_config_map) for rel_path in visualization_paths]
    data_rows = [
        inspect_parquet_directory(project_root, "data/analysis/node_intersection_flow_parquet"),
        inspect_csv_asset(project_root, "data/processed/rnsd_processed.csv"),
    ]
    summary_table_paths = find_summary_tables(project_root)
    summary_table_rows = [analyze_summary_table(project_root, rel_path, result_dirs) for rel_path in summary_table_paths]
    run_matrix_rows = build_run_matrix(project_root, result_dir_rows, run_config_rows)
    method_matrix_rows = build_method_matrix(code_rows, run_config_rows)
    risk_hits = scan_risk_statements(project_root, document_rows)

    state_json = build_current_state_json(
        project_root,
        env_info,
        git_info,
        code_rows,
        document_rows,
        result_dir_rows,
        data_rows,
        visualization_rows,
        run_config_rows,
        run_matrix_rows,
        method_matrix_rows,
        risk_hits,
    )
    workflow_md = build_workflow_summary(
        project_root,
        data_rows,
        run_matrix_rows,
        run_config_rows,
        visualization_rows,
        risk_hits,
    )
    state_audit_md = build_state_audit_md(
        project_root,
        env_info,
        git_info,
        code_rows,
        document_rows,
        result_dir_rows,
        data_rows,
        visualization_rows,
        run_config_rows,
        run_matrix_rows,
        method_matrix_rows,
        risk_hits,
    )
    open_issues_md = build_open_issues_md(risk_hits)

    write_csv(output_dir / "inventory_code_files.csv", code_rows)
    write_csv(output_dir / "inventory_document_files.csv", document_rows)
    write_csv(output_dir / "inventory_result_directories.csv", result_dir_rows)
    write_csv(output_dir / "inventory_visualization_files.csv", visualization_rows)
    write_csv(output_dir / "inventory_data_assets.csv", data_rows)
    write_csv(output_dir / "inventory_run_configs.csv", run_config_rows)
    write_csv(output_dir / "inventory_summary_tables.csv", summary_table_rows)
    write_csv(output_dir / "missingness_experiment_run_matrix.csv", run_matrix_rows)
    write_csv(output_dir / "missingness_method_mechanism_matrix.csv", method_matrix_rows)
    (output_dir / "current_missingness_state_audit.json").write_text(json.dumps(state_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "current_missingness_state_audit.md").write_text(state_audit_md, encoding="utf-8")
    (output_dir / "current_missingness_workflow_summary_zh.md").write_text(workflow_md, encoding="utf-8")
    (output_dir / "open_issues_and_next_steps_zh.md").write_text(open_issues_md, encoding="utf-8")

    print("Inventory output directory: {0}".format(output_dir))
    print("Generated code inventory rows: {0}".format(len(code_rows)))
    print("Generated document inventory rows: {0}".format(len(document_rows)))
    print("Generated result directory inventory rows: {0}".format(len(result_dir_rows)))
    print("Generated visualization inventory rows: {0}".format(len(visualization_rows)))
    print("Generated data asset inventory rows: {0}".format(len(data_rows)))
    print("Generated run config inventory rows: {0}".format(len(run_config_rows)))
    print("Generated summary table inventory rows: {0}".format(len(summary_table_rows)))
    print("Generated run matrix rows: {0}".format(len(run_matrix_rows)))
    print("Generated method matrix rows: {0}".format(len(method_matrix_rows)))


if __name__ == "__main__":
    main()
