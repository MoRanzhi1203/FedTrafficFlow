"""清点、清理并校验缺失实验中的中间补全数据目录。

核心功能：
- 扫描各类 missingness 实验目录，统计可清理中间产物及磁盘占用；
- 在 dry-run 或确认删除模式下清理 `imputed_datasets`、`imp_data` 等中间目录；
- 输出 CSV、JSON、Markdown 报告，供归档、留存和清理审计使用。

项目作用：
- 为缺失实验结果整理和磁盘治理提供统一入口；
- 在不影响正式汇总、审计和图像产物的前提下回收中间文件空间。

关键依赖：`pandas`、`pathlib`、`git` 命令行。
主要输入：实验根目录、场景列表、执行阶段、删除开关。
主要输出：清单表、校验报告，以及可选的目录删除结果。
"""

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_SCENARIOS = ["g_mcar_pt", "ntb_mix", "nso_mix"]
DELETE_DIR_NAMES = ["imp_data", "imputed_datasets"]
METHOD_CODE_MAP = {
    "ff": "forward_fill",
    "fcf": "function_curve_fit",
    "hle": "historical_linear_extrapolation",
    "mf": "mean_fill",
    "rtn": "road_topology_neighbor_fill",
    "ctn": "correlation_topology_neighbor_fill",
    "zf": "zero_fill",
    "gnf": "geo_neighbor_fill",
}
LEGACY_METHODS = set(["zero_fill", "geo_neighbor_fill"])
SUMMARY_EXTENSIONS = set([".csv"])
AUDIT_EXTENSIONS = set([".json", ".md"])
FIGURE_EXTENSIONS = set([".png", ".pdf"])
MANIFEST_EXTENSIONS = set([".csv", ".json"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory and cleanup imputed dataset directories.")
    parser.add_argument("--stage", choices=["inventory", "cleanup", "validate", "all"], default="inventory")
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS), type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--dry_run", default="true", type=str)
    parser.add_argument("--confirm_delete", default="false", type=str)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in ["1", "true", "yes", "y", "on"]:
        return True
    if text in ["0", "false", "no", "n", "off"]:
        return False
    raise ValueError("invalid boolean value: %s" % value)


def parse_scenarios(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, List[str], bool, bool]:
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    scenarios = parse_scenarios(args.scenarios)
    dry_run = parse_bool(args.dry_run)
    confirm_delete = parse_bool(args.confirm_delete)
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root_dir}")
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if (args.stage in ["cleanup", "all"]) and (not dry_run) and (not confirm_delete):
        raise ValueError("set --confirm_delete true when running destructive cleanup with dry_run=false")
    return root_dir, output_dir, scenarios, dry_run, confirm_delete


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    df = pd.DataFrame(list(rows))
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_missing_root(scenario_dir: Path) -> Optional[Path]:
    for name in ["miss_set", "missingness_setting"]:
        candidate = scenario_dir / name
        if candidate.exists():
            return candidate
    return None


def detect_imputation_root(scenario_dir: Path) -> Optional[Path]:
    for name in ["imp", "imputation"]:
        candidate = scenario_dir / name
        if candidate.exists():
            return candidate
    return None


def detect_first_existing_dir(base_dir: Optional[Path], names: Sequence[str]) -> Optional[Path]:
    if base_dir is None:
        return None
    for name in names:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def list_rate_dirs(base_dir: Optional[Path]) -> List[Path]:
    if base_dir is None or not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir()])


def count_parquet_files(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.parquet"))


def total_parquet_size(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*.parquet") if item.is_file())


def total_file_size(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def list_files_by_extension(path: Optional[Path], extensions: Sequence[str]) -> List[Path]:
    if path is None or not path.exists():
        return []
    normalized = set([item.lower() for item in extensions])
    return sorted([item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in normalized])


def rate_value_from_tag(rate_tag: str) -> Optional[float]:
    match = re.search(r"r(\d{2})", rate_tag)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def normalize_rates_from_config(config: Dict[str, Any]) -> List[float]:
    values = config.get("effective_missing_rates_for_dataset_generation")
    if values is None:
        values = config.get("missing_rates", [])
    normalized = []
    for item in values:
        try:
            value = float(item)
        except Exception:
            continue
        if value > 0:
            normalized.append(value)
    return sorted(set(normalized))


def detect_mechanism(scenario_id: str, run_config: Dict[str, Any]) -> str:
    if "mechanism" in run_config:
        return str(run_config["mechanism"])
    if scenario_id == "g_mcar_pt":
        return "mcar_point"
    if scenario_id == "ntb_mix":
        return "node_temporal_block"
    if scenario_id == "nso_mix":
        return "node_subset_temporal_outage"
    return "unknown"


def parse_imputed_dir_name(dir_name: str) -> Tuple[Optional[str], str, str]:
    if "_m_" in dir_name:
        left, code = dir_name.rsplit("_m_", 1)
        method = METHOD_CODE_MAP.get(code, code)
        method_kind = "legacy" if method in LEGACY_METHODS else "formal"
        return left, method, method_kind
    return None, dir_name, "unknown"


def scenario_expected_delete_dirs(scenario_dir: Path, imp_root: Optional[Path]) -> List[Path]:
    candidates = []
    if imp_root is not None:
        for name in DELETE_DIR_NAMES:
            candidates.append(imp_root / name)
    for name in DELETE_DIR_NAMES:
        candidates.append(scenario_dir / name)
    unique_paths = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def scenario_candidate_delete_dirs(scenario_dir: Path, imp_root: Optional[Path]) -> List[Path]:
    return [path for path in scenario_expected_delete_dirs(scenario_dir, imp_root) if path.exists()]


def collect_imputed_dataset_rows(scenario_id: str, scenario_dir: Path, delete_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = []
    for delete_dir in delete_dirs:
        child_dirs = sorted([path for path in delete_dir.iterdir() if path.is_dir()]) if delete_dir.exists() else []
        if not child_dirs and delete_dir.exists():
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_dir": str(scenario_dir),
                    "container_dir": str(delete_dir),
                    "dataset_dir_name": delete_dir.name,
                    "rate_tag": None,
                    "missing_rate": None,
                    "method": delete_dir.name,
                    "method_kind": "unknown",
                    "imputed_dataset_dir": str(delete_dir),
                    "parquet_file_count": count_parquet_files(delete_dir),
                    "total_size_bytes": total_parquet_size(delete_dir),
                    "total_size_gb": round(total_parquet_size(delete_dir) / float(1 << 30), 6),
                }
            )
            continue
        for child_dir in child_dirs:
            rate_tag, method, method_kind = parse_imputed_dir_name(child_dir.name)
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_dir": str(scenario_dir),
                    "container_dir": str(delete_dir),
                    "dataset_dir_name": child_dir.name,
                    "rate_tag": rate_tag,
                    "missing_rate": rate_value_from_tag(rate_tag or ""),
                    "method": method,
                    "method_kind": method_kind,
                    "imputed_dataset_dir": str(child_dir),
                    "parquet_file_count": count_parquet_files(child_dir),
                    "total_size_bytes": total_parquet_size(child_dir),
                    "total_size_gb": round(total_parquet_size(child_dir) / float(1 << 30), 6),
                }
            )
    return rows


def collect_imputed_dataset_rows_from_manifests(
    scenario_id: str,
    scenario_dir: Path,
    imp_root: Optional[Path],
    expected_delete_dirs: Sequence[Path],
) -> List[Dict[str, Any]]:
    if imp_root is None:
        return []
    manifest_dir = imp_root / "manifests"
    if not manifest_dir.exists():
        return []
    normalized_targets = []
    for target in expected_delete_dirs:
        normalized_targets.append((str(target.resolve()).lower().replace("/", "\\"), target))
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for manifest_file in sorted(manifest_dir.glob("*.csv")):
        try:
            df = pd.read_csv(manifest_file, usecols=["output_path", "file_size_bytes"])
        except Exception:
            continue
        if "output_path" not in df.columns or "file_size_bytes" not in df.columns:
            continue
        for record in df.to_dict(orient="records"):
            output_path = str(record.get("output_path", "") or "")
            normalized_output_path = output_path.lower().replace("/", "\\")
            if not normalized_output_path.endswith(".parquet"):
                continue
            matched_target: Optional[Path] = None
            for normalized_target, target in normalized_targets:
                if normalized_output_path.startswith(normalized_target + "\\"):
                    matched_target = target
                    break
            if matched_target is None:
                continue
            dataset_dir = Path(output_path).parent
            dataset_dir_name = dataset_dir.name
            key = (str(matched_target), str(dataset_dir))
            if key not in grouped:
                rate_tag, method, method_kind = parse_imputed_dir_name(dataset_dir_name)
                grouped[key] = {
                    "scenario_id": scenario_id,
                    "scenario_dir": str(scenario_dir),
                    "container_dir": str(matched_target),
                    "dataset_dir_name": dataset_dir_name,
                    "rate_tag": rate_tag,
                    "missing_rate": rate_value_from_tag(rate_tag or ""),
                    "method": method,
                    "method_kind": method_kind,
                    "imputed_dataset_dir": str(dataset_dir),
                    "parquet_file_count": 0,
                    "total_size_bytes": 0,
                }
            try:
                file_size_bytes = int(float(record.get("file_size_bytes") or 0))
            except Exception:
                file_size_bytes = 0
            grouped[key]["parquet_file_count"] += 1
            grouped[key]["total_size_bytes"] += file_size_bytes
    rows = []
    for item in grouped.values():
        item["total_size_gb"] = round(item["total_size_bytes"] / float(1 << 30), 6)
        rows.append(item)
    return sorted(rows, key=lambda row: (row["container_dir"], row["dataset_dir_name"]))


def collect_missing_rate_rows(scenario_id: str, masks_root: Optional[Path], miss_data_root: Optional[Path]) -> List[Dict[str, Any]]:
    rate_tags = sorted(set([item.name for item in list_rate_dirs(masks_root)] + [item.name for item in list_rate_dirs(miss_data_root)]))
    rows = []
    for rate_tag in rate_tags:
        mask_dir = masks_root / rate_tag if masks_root is not None else None
        miss_dir = miss_data_root / rate_tag if miss_data_root is not None else None
        mask_count = count_parquet_files(mask_dir)
        miss_count = count_parquet_files(miss_dir)
        mask_size = total_parquet_size(mask_dir)
        miss_size = total_parquet_size(miss_dir)
        rows.append(
            {
                "scenario_id": scenario_id,
                "rate_tag": rate_tag,
                "missing_rate": rate_value_from_tag(rate_tag),
                "mask_file_count": mask_count,
                "miss_data_file_count": miss_count,
                "mask_total_size_bytes": mask_size,
                "mask_total_size_gb": round(mask_size / float(1 << 30), 6),
                "miss_data_total_size_bytes": miss_size,
                "miss_data_total_size_gb": round(miss_size / float(1 << 30), 6),
                "mask_has_61_files": bool(mask_count == 61),
                "miss_data_has_61_files": bool(miss_count == 61),
            }
        )
    return rows


def collect_summary_index_rows(scenario_id: str, scenario_dir: Path, summary_dir: Optional[Path]) -> List[Dict[str, Any]]:
    rows = []
    if summary_dir is None or not summary_dir.exists():
        return rows
    for summary_file in sorted(summary_dir.glob("*.csv")):
        try:
            df = pd.read_csv(summary_file)
        except Exception:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_dir": str(scenario_dir),
                    "source_file": str(summary_file),
                    "file_name": summary_file.name,
                    "record_type": "file_index",
                    "row_count": None,
                    "missing_rate": None,
                    "method": None,
                    "MAE": None,
                    "RMSE": None,
                    "MAPE": None,
                    "sMAPE": None,
                    "NRMSE": None,
                    "missing_count": None,
                    "valid_eval_count": None,
                }
            )
            continue
        core_columns = []
        for candidate in ["missing_rate", "method", "mae", "rmse", "mape", "smape", "nrmse", "missing_count", "valid_eval_count"]:
            if candidate in df.columns:
                core_columns.append(candidate)
        if core_columns:
            extract = pd.DataFrame()
            extract["scenario_id"] = scenario_id
            extract["scenario_dir"] = str(scenario_dir)
            extract["source_file"] = str(summary_file)
            extract["file_name"] = summary_file.name
            extract["record_type"] = "metric_row"
            extract["row_count"] = len(df)
            extract["missing_rate"] = df["missing_rate"] if "missing_rate" in df.columns else None
            extract["method"] = df["method"] if "method" in df.columns else None
            extract["MAE"] = df["mae"] if "mae" in df.columns else None
            extract["RMSE"] = df["rmse"] if "rmse" in df.columns else None
            extract["MAPE"] = df["mape"] if "mape" in df.columns else None
            extract["sMAPE"] = df["smape"] if "smape" in df.columns else None
            extract["NRMSE"] = df["nrmse"] if "nrmse" in df.columns else None
            extract["missing_count"] = df["missing_count"] if "missing_count" in df.columns else None
            extract["valid_eval_count"] = df["valid_eval_count"] if "valid_eval_count" in df.columns else None
            rows.extend(extract.to_dict(orient="records"))
        else:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_dir": str(scenario_dir),
                    "source_file": str(summary_file),
                    "file_name": summary_file.name,
                    "record_type": "file_index",
                    "row_count": int(len(df)),
                    "missing_rate": None,
                    "method": None,
                    "MAE": None,
                    "RMSE": None,
                    "MAPE": None,
                    "sMAPE": None,
                    "NRMSE": None,
                    "missing_count": None,
                    "valid_eval_count": None,
                }
            )
    return rows


def collect_disk_space() -> Dict[str, float]:
    output = {}
    for drive in ["C:/", "D:/", "E:/"]:
        path = Path(drive)
        if path.exists():
            output[path.drive.rstrip(":")] = round(shutil.disk_usage(str(path)).free / float(1 << 30), 3)
    return output


def git_cached_files(repo_root: Path) -> List[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in [0, 1]:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def scenario_inventory(scenario_id: str, root_dir: Path) -> Dict[str, Any]:
    scenario_dir = root_dir / "scenarios" / scenario_id
    miss_root = detect_missing_root(scenario_dir)
    imp_root = detect_imputation_root(scenario_dir)
    masks_root = detect_first_existing_dir(miss_root, ["masks"])
    miss_data_root = detect_first_existing_dir(miss_root, ["miss_data", "missing_datasets"])
    summary_dir = detect_first_existing_dir(imp_root, ["summaries"])
    audit_dir = detect_first_existing_dir(imp_root, ["audits"])
    figure_dir = detect_first_existing_dir(imp_root, ["figures"])
    manifest_dir = detect_first_existing_dir(imp_root, ["manifests"])
    run_config_path = miss_root / "run_config.json" if miss_root is not None else None
    run_config = safe_load_json(run_config_path) if run_config_path is not None else {}
    missing_rate_rows = collect_missing_rate_rows(scenario_id, masks_root, miss_data_root)
    expected_delete_dirs = scenario_expected_delete_dirs(scenario_dir, imp_root)
    delete_dirs = scenario_candidate_delete_dirs(scenario_dir, imp_root)
    imputed_rows = collect_imputed_dataset_rows(scenario_id, scenario_dir, delete_dirs)
    if not imputed_rows:
        imputed_rows = collect_imputed_dataset_rows_from_manifests(scenario_id, scenario_dir, imp_root, expected_delete_dirs)
    rate_tags = [row["rate_tag"] for row in missing_rate_rows]
    scenario_row = {
        "scenario_id": scenario_id,
        "scenario_dir": str(scenario_dir),
        "mechanism": detect_mechanism(scenario_id, run_config),
        "missing_rates": ",".join(["%.2f" % value for value in normalize_rates_from_config(run_config)]),
        "rate_tags": ",".join(rate_tags),
        "mask_file_count": int(sum(row["mask_file_count"] for row in missing_rate_rows)),
        "miss_data_file_count": int(sum(row["miss_data_file_count"] for row in missing_rate_rows)),
        "imputed_data_file_count_before_cleanup": int(sum(row["parquet_file_count"] for row in imputed_rows)),
        "summary_file_count": len(list_files_by_extension(summary_dir, SUMMARY_EXTENSIONS)),
        "audit_file_count": len(list_files_by_extension(audit_dir, AUDIT_EXTENSIONS)),
        "figure_file_count": len(list_files_by_extension(figure_dir, FIGURE_EXTENSIONS)),
        "manifest_file_count": len(list_files_by_extension(manifest_dir, MANIFEST_EXTENSIONS)),
        "masks_root": str(masks_root) if masks_root is not None else "",
        "miss_data_root": str(miss_data_root) if miss_data_root is not None else "",
        "imputation_root": str(imp_root) if imp_root is not None else "",
    }
    return {
        "scenario_row": scenario_row,
        "missing_rate_rows": missing_rate_rows,
        "imputed_rows": imputed_rows,
        "summary_rows": collect_summary_index_rows(scenario_id, scenario_dir, summary_dir),
        "delete_dirs": [str(path) for path in delete_dirs],
        "expected_delete_dirs": [str(path) for path in expected_delete_dirs],
        "preserve": {
            "masks_exists": bool(masks_root is not None and masks_root.exists()),
            "miss_data_exists": bool(miss_data_root is not None and miss_data_root.exists()),
            "summaries_exists": bool(summary_dir is not None and summary_dir.exists()),
            "audits_exists": bool(audit_dir is not None and audit_dir.exists()),
            "figures_exists": bool(figure_dir is not None and figure_dir.exists()),
        },
    }


def build_deleted_rows(delete_targets: Sequence[Path], dry_run: bool) -> List[Dict[str, Any]]:
    rows = []
    for target in delete_targets:
        rows.append(
            {
                "target_path": str(target),
                "exists_before_cleanup": bool(target.exists()),
                "total_size_bytes": total_file_size(target),
                "total_size_gb": round(total_file_size(target) / float(1 << 30), 6),
                "delete_mode": "planned" if dry_run else "ready_for_delete",
            }
        )
    return rows


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "inventory_csv": output_dir / "three_scenarios_cleanup_inventory.csv",
        "inventory_json": output_dir / "three_scenarios_cleanup_inventory.json",
        "imputed_size_csv": output_dir / "imputed_dataset_size_before_cleanup.csv",
        "retention_csv": output_dir / "missing_dataset_retention_check.csv",
        "summary_index_csv": output_dir / "imputation_summary_file_index.csv",
        "deleted_paths_csv": output_dir / "cleanup_deleted_paths.csv",
        "validation_json": output_dir / "cleanup_validation.json",
        "report_md": output_dir / "cleanup_report_zh.md",
    }


def load_existing_deleted_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return df.to_dict(orient="records")


def deleted_rows_have_size(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(float(row.get("total_size_bytes", 0) or 0) > 0 for row in rows)


def normalize_deleted_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        size_bytes = int(float(row.get("total_size_bytes", 0) or 0))
        exists_before_cleanup = str(row.get("exists_before_cleanup", "")).strip().lower() in ["true", "1", "yes"] or size_bytes > 0
        if not exists_before_cleanup and size_bytes <= 0:
            continue
        normalized.append(
            {
                **row,
                "exists_before_cleanup": exists_before_cleanup,
                "total_size_bytes": size_bytes,
                "total_size_gb": round(float(row.get("total_size_gb", 0.0) or 0.0), 6) if size_bytes <= 0 else round(size_bytes / float(1 << 30), 6),
            }
        )
    return normalized


def reconstruct_deleted_rows_from_manifests(delete_targets: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = []
    for target in delete_targets:
        manifest_dir = target.parent / "manifests"
        total_size_bytes = 0
        matched_files = 0
        normalized_target = str(target.resolve()).lower().replace("/", "\\")
        if manifest_dir.exists():
            for manifest_file in sorted(manifest_dir.glob("*.csv")):
                try:
                    df = pd.read_csv(manifest_file, usecols=["output_path", "file_size_bytes"])
                except Exception:
                    continue
                if "output_path" not in df.columns or "file_size_bytes" not in df.columns:
                    continue
                paths = df["output_path"].astype(str).str.lower().str.replace("/", "\\", regex=False)
                matched = df.loc[paths.str.startswith(normalized_target + "\\"), "file_size_bytes"]
                if not matched.empty:
                    numeric = pd.to_numeric(matched, errors="coerce").fillna(0)
                    total_size_bytes += int(numeric.sum())
                    matched_files += int((numeric > 0).sum())
        if total_size_bytes > 0 or target.exists():
            rows.append(
                {
                    "target_path": str(target),
                    "exists_before_cleanup": bool(total_size_bytes > 0 or target.exists()),
                    "total_size_bytes": int(total_size_bytes),
                    "total_size_gb": round(total_size_bytes / float(1 << 30), 6),
                    "delete_mode": "deleted_reconstructed_from_manifests" if total_size_bytes > 0 else "missing_or_not_applicable",
                    "matched_parquet_files": matched_files,
                }
            )
    return rows


def write_report(
    output_path: Path,
    stage: str,
    scenarios: Sequence[str],
    scenario_rows: Sequence[Dict[str, Any]],
    deleted_rows: Sequence[Dict[str, Any]],
    free_before: Dict[str, float],
    free_after: Dict[str, float],
    validation_payload: Optional[Dict[str, Any]],
    cleanup_started_at: str,
    cleanup_finished_at: str,
) -> None:
    deleted_size_gb = round(sum(float(row.get("total_size_gb", 0.0) or 0.0) for row in deleted_rows if row.get("exists_before_cleanup")), 6)
    lines = [
        "# 前三个机制补全数据集清理报告",
        "",
        "- stage: `%s`" % stage,
        "- scenarios: `%s`" % ",".join(scenarios),
        "- cleanup_started_at: `%s`" % cleanup_started_at,
        "- cleanup_finished_at: `%s`" % cleanup_finished_at,
        "- free_space_before: `%s`" % json.dumps(free_before, ensure_ascii=False),
        "- free_space_after: `%s`" % json.dumps(free_after, ensure_ascii=False),
        "- deleted_size_gb: `%s`" % deleted_size_gb,
        "- deleted_path_count: `%s`" % len([row for row in deleted_rows if row.get("exists_before_cleanup")]),
        "",
        "## 处理对象",
        "",
    ]
    for row in scenario_rows:
        lines.append("- `%s`: mechanism=`%s`, mask_file_count=`%s`, miss_data_file_count=`%s`, imputed_data_file_count_before_cleanup=`%s`" % (
            row["scenario_id"],
            row["mechanism"],
            row["mask_file_count"],
            row["miss_data_file_count"],
            row["imputed_data_file_count_before_cleanup"],
        ))
    lines.extend(["", "## 待删或已删目录", ""])
    for row in deleted_rows:
        lines.append("- `%s`: exists_before_cleanup=`%s`, total_size_gb=`%s`, delete_mode=`%s`" % (
            row["target_path"],
            row["exists_before_cleanup"],
            row["total_size_gb"],
            row["delete_mode"],
        ))
    if validation_payload is not None:
        lines.extend(["", "## 验证结果", ""])
        lines.append("- missing_datasets_preserved: `%s`" % validation_payload.get("missing_datasets_preserved"))
        lines.append("- masks_preserved: `%s`" % validation_payload.get("masks_preserved"))
        lines.append("- imputed_datasets_deleted: `%s`" % validation_payload.get("imputed_datasets_deleted"))
        lines.append("- summaries_preserved: `%s`" % validation_payload.get("summaries_preserved"))
        lines.append("- audits_preserved: `%s`" % validation_payload.get("audits_preserved"))
        lines.append("- figures_preserved: `%s`" % validation_payload.get("figures_preserved"))
        lines.append("- comparison_preserved: `%s`" % validation_payload.get("comparison_preserved"))
        lines.append("- no_parquet_staged: `%s`" % validation_payload.get("no_parquet_staged"))
        lines.append("- all_complete: `%s`" % validation_payload.get("all_complete"))
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_inventory(args: argparse.Namespace, repo_root: Path, root_dir: Path, output_dir: Path) -> Dict[str, Any]:
    scenarios = parse_scenarios(args.scenarios)
    scenario_payloads = [scenario_inventory(scenario_id, root_dir) for scenario_id in scenarios]
    scenario_rows = [item["scenario_row"] for item in scenario_payloads]
    missing_rows = []
    imputed_rows = []
    summary_rows = []
    delete_targets = []
    expected_delete_targets = []
    for payload in scenario_payloads:
        missing_rows.extend(payload["missing_rate_rows"])
        imputed_rows.extend(payload["imputed_rows"])
        summary_rows.extend(payload["summary_rows"])
        delete_targets.extend([Path(item) for item in payload["delete_dirs"]])
        expected_delete_targets.extend([Path(item) for item in payload["expected_delete_dirs"]])
    free_before = collect_disk_space()
    deleted_rows = build_deleted_rows(delete_targets, dry_run=parse_bool(args.dry_run))
    paths = output_paths(output_dir)
    write_csv(paths["inventory_csv"], scenario_rows)
    write_json(
        paths["inventory_json"],
        {
            "scenarios": scenarios,
            "generated_at": timestamp_now(),
            "scenario_inventory": scenario_rows,
            "missing_rate_inventory": missing_rows,
            "imputed_dataset_inventory": imputed_rows,
            "delete_targets": [str(path) for path in delete_targets],
            "expected_delete_targets": [str(path) for path in expected_delete_targets],
        },
    )
    write_csv(paths["imputed_size_csv"], imputed_rows)
    write_csv(paths["retention_csv"], missing_rows)
    write_csv(paths["summary_index_csv"], summary_rows)
    write_csv(paths["deleted_paths_csv"], deleted_rows)
    write_report(
        output_path=paths["report_md"],
        stage="inventory",
        scenarios=scenarios,
        scenario_rows=scenario_rows,
        deleted_rows=deleted_rows,
        free_before=free_before,
        free_after=free_before,
        validation_payload=None,
        cleanup_started_at=timestamp_now(),
        cleanup_finished_at=timestamp_now(),
    )
    return {
        "scenarios": scenarios,
        "scenario_payloads": scenario_payloads,
        "scenario_rows": scenario_rows,
        "missing_rows": missing_rows,
        "imputed_rows": imputed_rows,
        "delete_targets": delete_targets,
        "expected_delete_targets": expected_delete_targets,
        "free_before": free_before,
        "paths": paths,
    }


def execute_cleanup(args: argparse.Namespace, delete_targets: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    dry_run = parse_bool(args.dry_run)
    confirm_delete = parse_bool(args.confirm_delete)
    free_before = collect_disk_space()
    deleted_rows = []
    for target in delete_targets:
        row = {
            "target_path": str(target),
            "exists_before_cleanup": bool(target.exists()),
            "total_size_bytes": total_file_size(target),
            "total_size_gb": round(total_file_size(target) / float(1 << 30), 6),
            "delete_mode": "planned" if dry_run or not confirm_delete else "deleted",
        }
        if target.exists() and (not dry_run) and confirm_delete:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        deleted_rows.append(row)
    free_after = collect_disk_space()
    return deleted_rows, free_before, free_after


def run_validate(scenarios: Sequence[str], repo_root: Path, root_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scenario_payloads = [scenario_inventory(scenario_id, root_dir) for scenario_id in scenarios]
    validation_rows = []
    for payload in scenario_payloads:
        scenario_id = payload["scenario_row"]["scenario_id"]
        delete_dirs_deleted = True
        for delete_dir in payload["delete_dirs"]:
            if Path(delete_dir).exists():
                delete_dirs_deleted = False
                break
        for row in payload["missing_rate_rows"]:
            validation_rows.append(
                {
                    "scenario_id": scenario_id,
                    "rate_tag": row["rate_tag"],
                    "missing_rate": row["missing_rate"],
                    "mask_file_count": row["mask_file_count"],
                    "miss_data_file_count": row["miss_data_file_count"],
                    "mask_has_61_files": row["mask_has_61_files"],
                    "miss_data_has_61_files": row["miss_data_has_61_files"],
                    "imputed_datasets_deleted": delete_dirs_deleted,
                    "masks_preserved": payload["preserve"]["masks_exists"],
                    "missing_datasets_preserved": payload["preserve"]["miss_data_exists"],
                    "summaries_preserved": payload["preserve"]["summaries_exists"],
                    "audits_preserved": payload["preserve"]["audits_exists"],
                    "figures_preserved": payload["preserve"]["figures_exists"],
                }
            )
    cached_files = git_cached_files(repo_root)
    no_parquet_staged = not any(item.lower().endswith(".parquet") for item in cached_files)
    comparison_targets = [root_dir / "comparison", root_dir / "comparison_spatial_extension"]
    comparison_preserved = True
    for target in comparison_targets:
        if target.exists() and not target.is_dir():
            comparison_preserved = False
    payload = {
        "scenarios": list(scenarios),
        "formal_snh_temp_excluded": bool(set(scenarios) == set(DEFAULT_SCENARIOS)),
        "missing_datasets_preserved": bool(all(row["missing_datasets_preserved"] for row in validation_rows)),
        "masks_preserved": bool(all(row["masks_preserved"] for row in validation_rows)),
        "imputed_datasets_deleted": bool(all(row["imputed_datasets_deleted"] for row in validation_rows)),
        "summaries_preserved": bool(all(row["summaries_preserved"] for row in validation_rows)),
        "audits_preserved": bool(all(row["audits_preserved"] for row in validation_rows)),
        "figures_preserved": bool(all(row["figures_preserved"] for row in validation_rows)),
        "comparison_preserved": bool(comparison_preserved),
        "no_parquet_staged": bool(no_parquet_staged),
    }
    payload["all_complete"] = bool(all(payload.values())) if payload else False
    return validation_rows, payload


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    root_dir, output_dir, scenarios, dry_run, _confirm_delete = validate_args(args)
    args.scenarios = ",".join(scenarios)
    ensure_dir(output_dir)
    preexisting_deleted_rows = normalize_deleted_rows(load_existing_deleted_rows(output_paths(output_dir)["deleted_paths_csv"]))
    cleanup_started_at = timestamp_now()
    inventory = run_inventory(args, repo_root, root_dir, output_dir)
    deleted_rows = build_deleted_rows(inventory["delete_targets"], dry_run=dry_run)
    free_before = inventory["free_before"]
    free_after = free_before
    if args.stage in ["cleanup", "all"]:
        deleted_rows, free_before, free_after = execute_cleanup(args, inventory["delete_targets"])
        write_csv(inventory["paths"]["deleted_paths_csv"], deleted_rows)
    validation_payload = None
    if args.stage in ["validate", "all", "cleanup"]:
        validation_rows, validation_payload = run_validate(inventory["scenarios"], repo_root, root_dir)
        write_csv(inventory["paths"]["retention_csv"], validation_rows)
        write_json(inventory["paths"]["validation_json"], validation_payload)
    elif args.stage == "inventory":
        validation_rows = inventory["missing_rows"]
        write_csv(inventory["paths"]["retention_csv"], validation_rows)
    if args.stage == "validate":
        if preexisting_deleted_rows:
            deleted_rows = preexisting_deleted_rows
            write_csv(inventory["paths"]["deleted_paths_csv"], deleted_rows)
        if not deleted_rows_have_size(deleted_rows):
            deleted_rows = reconstruct_deleted_rows_from_manifests(inventory["expected_delete_targets"])
            write_csv(inventory["paths"]["deleted_paths_csv"], deleted_rows)
    cleanup_finished_at = timestamp_now()
    write_report(
        output_path=inventory["paths"]["report_md"],
        stage=args.stage,
        scenarios=inventory["scenarios"],
        scenario_rows=inventory["scenario_rows"],
        deleted_rows=deleted_rows,
        free_before=free_before,
        free_after=free_after,
        validation_payload=validation_payload,
        cleanup_started_at=cleanup_started_at,
        cleanup_finished_at=cleanup_finished_at,
    )


if __name__ == "__main__":
    main()
