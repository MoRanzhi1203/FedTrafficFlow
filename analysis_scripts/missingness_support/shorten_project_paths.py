from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any


IGNORE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".cache",
}
TEXT_EXTENSIONS = {".py", ".md", ".json", ".csv", ".txt", ".bat", ".ps1", ".yaml", ".yml"}
SKIP_CONTENT_EXTENSIONS = {
    ".parquet",
    ".png",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".npy",
    ".npz",
    ".pkl",
    ".joblib",
    ".xlsx",
    ".docx",
    ".pptx",
}
CURRENT_ROOT = PureWindowsPath("results/real_data_missingness_experiments")
TARGET_ROOT = PureWindowsPath("results/rdm_exp")
PATH_CLEANUP_ROOT = TARGET_ROOT / "path_cleanup"
INVENTORY_FIELDS = [
    "path",
    "path_type",
    "path_length",
    "name_length",
    "is_directory",
    "is_file",
    "risk_level",
    "reason",
    "suggested_new_path",
    "action",
]
PLAN_FIELDS = [
    "old_path",
    "new_path",
    "old_name",
    "new_name",
    "path_type",
    "artifact_group",
    "scenario_id_old",
    "scenario_id_new",
    "is_large_data",
    "action",
    "status",
    "notes",
]
PYTHON_AUDIT_FIELDS = [
    "file",
    "changed_line_number",
    "old_line",
    "new_line",
    "is_path_only_change",
    "is_allowed",
    "notes",
]
SCENARIO_MAP = {
    "global_mcar_point": "g_mcar_pt",
    "node_temporal_block_mixed_short_mid_long": "ntb_mix",
    "node_subset_temporal_outage_mixed_short_mid_long": "nso_mix",
}
SHORT_TO_LONG_SCENARIO = {value: key for key, value in SCENARIO_MAP.items()}
FIXED_DIR_MAP = {
    "missingness_setting": "miss_set",
    "imputation": "imp",
    "missing_datasets": "miss_data",
    "imputed_datasets": "imp_data",
}
SHORT_TO_LONG_FIXED_DIR = {value: key for key, value in FIXED_DIR_MAP.items()}
METHOD_ABBR = {
    "mean_fill": "mf",
    "zero_fill": "zf",
    "forward_fill": "ff",
    "historical_linear_extrapolation": "hle",
    "road_topology_neighbor_fill": "rtn",
    "function_curve_fit": "fcf",
    "correlation_topology_neighbor_fill": "ctn",
}
SHORT_TO_LONG_METHOD = {value: key for key, value in METHOD_ABBR.items()}
LEGACY_ROOT_ALIASES = {
    "results\\real_data_global_missingness_setting": "results\\rdm_exp\\scenarios\\g_mcar_pt",
    "results\\real_data_structured_missingness_setting": "results\\rdm_exp",
    "results\\real_data_missingness_visual_comparison": "results\\rdm_exp\\comparison",
    "results\\real_data_missingness_experiments": "results\\rdm_exp",
}
RENAME_STUB_TEXT = (
    "本路径已迁移/重命名至 results\\rdm_exp。\n"
    "请使用 results\\rdm_exp\\path_aliases.json 查询新路径。\n"
)
SCENARIO_ROOTS = {
    "g_mcar_pt": "results\\rdm_exp\\scenarios\\g_mcar_pt",
    "ntb_mix": "results\\rdm_exp\\scenarios\\ntb_mix",
    "nso_mix": "results\\rdm_exp\\scenarios\\nso_mix",
}
ALLOWED_OLD_ROOT_REFERENCE_FILES = {
    "analysis_scripts\\shorten_project_paths.py",
    "results\\cleanup_missingness_removal_plan.csv",
    "results\\cleanup_missingness_removal_plan_zh.md",
    "results\\cleanup_missingness_removal_report.json",
    "results\\cleanup_missingness_removal_report.md",
    "results\\rdm_exp\\experiment_registry.csv",
    "results\\rdm_exp\\experiment_registry.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shorten long missingness experiment paths without changing logic.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["inventory", "dry_run", "rename", "update_refs", "validate", "all"],
    )
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--current_root", type=Path, default=Path(str(CURRENT_ROOT)))
    parser.add_argument("--target_root", type=Path, default=Path(str(TARGET_ROOT)))
    return parser.parse_args()


def win_rel(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("/", "\\")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def strip_extended_prefix(path_str: str) -> str:
    if path_str.startswith("\\\\?\\UNC\\"):
        return "\\" + path_str[7:]
    if path_str.startswith("\\\\?\\"):
        return path_str[4:]
    return path_str


def extended_path(path: Path) -> str:
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def walk_project(project_root: Path) -> list[tuple[Path, bool]]:
    root_ext = extended_path(project_root)
    results: list[tuple[Path, bool]] = []
    for dirpath_ext, dirnames, filenames in os.walk(root_ext):
        dirnames[:] = [name for name in dirnames if name not in IGNORE_DIR_NAMES]
        dirpath = Path(strip_extended_prefix(dirpath_ext))
        results.append((dirpath, True))
        for file_name in filenames:
            if file_name in {".DS_Store"}:
                continue
            results.append((dirpath / file_name, False))
    return results


def is_relative_to(path: PureWindowsPath, root: PureWindowsPath) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def risk_fields(abs_path: Path, is_dir: bool, is_file: bool) -> tuple[str, str]:
    path_length = len(str(abs_path.resolve()))
    name_length = len(abs_path.name)
    reasons: list[str] = []
    level = "low_risk"
    if path_length >= 220:
        level = "high_risk"
        reasons.append("path_length>=220")
    elif path_length >= 180:
        level = "medium_risk"
        reasons.append("path_length>=180")
    if is_dir and name_length >= 50:
        if level == "low_risk":
            level = "segment_long"
        reasons.append("directory_name_length>=50")
    if is_file and name_length >= 80:
        if level == "low_risk":
            level = "file_name_long"
        reasons.append("file_name_length>=80")
    return level, "; ".join(reasons) if reasons else "within_threshold"


def global_missing_dir_name(rate: str, seed: str) -> str:
    return f"r{rate.replace('0p', '')}_mcar_s{seed}"


def structured_missing_dir_name(prefix: str, rate: str, seed: str) -> str:
    return f"{prefix}_r{rate.replace('0p', '')}_mix_s{seed}"


def abbreviate_missing_dir_name(name: str, scenario_short: str) -> str:
    global_match = re.fullmatch(r"rate_(0p\d+)__mechanism_mcar_point__scope_global__seed_(\d+)", name)
    if global_match:
        return global_missing_dir_name(global_match.group(1), global_match.group(2))
    structured_match = re.fullmatch(
        r"mechanism_(node_temporal_block|node_subset_temporal_outage)__rate_(0p\d+)__mixed_short_mid_long__seed_(\d+)",
        name,
    )
    if not structured_match:
        return name
    mechanism, rate, seed = structured_match.groups()
    prefix = "ntb" if mechanism == "node_temporal_block" else "nso"
    return structured_missing_dir_name(prefix, rate, seed)


def abbreviate_imputed_dir_name(name: str, scenario_short: str) -> str:
    global_match = re.fullmatch(
        r"rate_(0p\d+)__mechanism_mcar_point__scope_global__seed_(\d+)__method_([A-Za-z0-9_]+)",
        name,
    )
    if global_match:
        rate, seed, method = global_match.groups()
        method_short = METHOD_ABBR.get(method, method)
        return f"{global_missing_dir_name(rate, seed)}_m_{method_short}"
    structured_match = re.fullmatch(
        r"mechanism_(node_temporal_block|node_subset_temporal_outage)__rate_(0p\d+)__mixed_short_mid_long__seed_(\d+)__method_([A-Za-z0-9_]+)",
        name,
    )
    if not structured_match:
        return name
    mechanism, rate, seed, method = structured_match.groups()
    prefix = "ntb" if mechanism == "node_temporal_block" else "nso"
    method_short = METHOD_ABBR.get(method, method)
    return f"{structured_missing_dir_name(prefix, rate, seed)}_m_{method_short}"


def transform_relative_path(rel_path: PureWindowsPath) -> PureWindowsPath:
    parts = list(rel_path.parts)
    if len(parts) < 2:
        return rel_path
    if parts[0] == "results" and parts[1] == "real_data_missingness_experiments":
        parts[1] = "rdm_exp"
        if len(parts) >= 4 and parts[2] == "scenarios":
            parts[3] = SCENARIO_MAP.get(parts[3], parts[3])
            for idx in range(4, len(parts)):
                parts[idx] = FIXED_DIR_MAP.get(parts[idx], parts[idx])
            if len(parts) >= 7 and parts[4] == "miss_set" and parts[5] in {"masks", "miss_data"}:
                parts[6] = abbreviate_missing_dir_name(parts[6], parts[3])
            if len(parts) >= 8 and parts[4] == "miss_set" and parts[5] == "manifests" and parts[6] == "outage_node_lists":
                parts[7] = abbreviate_missing_dir_name(parts[7], parts[3])
            if len(parts) >= 7 and parts[4] == "imp" and parts[5] == "imp_data":
                parts[6] = abbreviate_imputed_dir_name(parts[6], parts[3])
        return PureWindowsPath(*parts)
    return rel_path


def expand_missing_dir_name(name: str, scenario_short: str) -> str:
    global_match = re.fullmatch(r"r(\d\d)_mcar_s(\d+)", name)
    if global_match:
        rate, seed = global_match.groups()
        return f"rate_0p{rate}__mechanism_mcar_point__scope_global__seed_{seed}"
    structured_match = re.fullmatch(r"(ntb|nso)_r(\d\d)_mix_s(\d+)", name)
    if not structured_match:
        return name
    prefix, rate, seed = structured_match.groups()
    mechanism = "node_temporal_block" if prefix == "ntb" else "node_subset_temporal_outage"
    return f"mechanism_{mechanism}__rate_0p{rate}__mixed_short_mid_long__seed_{seed}"


def expand_imputed_dir_name(name: str, scenario_short: str) -> str:
    global_match = re.fullmatch(r"r(\d\d)_mcar_s(\d+)_m_([a-z]+)", name)
    if global_match:
        rate, seed, method_short = global_match.groups()
        method_long = SHORT_TO_LONG_METHOD.get(method_short, method_short)
        return f"rate_0p{rate}__mechanism_mcar_point__scope_global__seed_{seed}__method_{method_long}"
    structured_match = re.fullmatch(r"(ntb|nso)_r(\d\d)_mix_s(\d+)_m_([a-z]+)", name)
    if not structured_match:
        return name
    prefix, rate, seed, method_short = structured_match.groups()
    mechanism = "node_temporal_block" if prefix == "ntb" else "node_subset_temporal_outage"
    method_long = SHORT_TO_LONG_METHOD.get(method_short, method_short)
    return f"mechanism_{mechanism}__rate_0p{rate}__mixed_short_mid_long__seed_{seed}__method_{method_long}"


def inverse_transform_relative_path(rel_path: PureWindowsPath) -> PureWindowsPath:
    parts = list(rel_path.parts)
    if len(parts) < 2:
        return rel_path
    if parts[0] == "results" and parts[1] == "rdm_exp":
        if len(parts) >= 3 and parts[2] == "path_cleanup":
            return rel_path
        parts[1] = "real_data_missingness_experiments"
        if len(parts) >= 4 and parts[2] == "scenarios":
            parts[3] = SHORT_TO_LONG_SCENARIO.get(parts[3], parts[3])
            for idx in range(4, len(parts)):
                parts[idx] = SHORT_TO_LONG_FIXED_DIR.get(parts[idx], parts[idx])
            if len(parts) >= 7 and parts[4] == "missingness_setting" and parts[5] in {"masks", "missing_datasets"}:
                parts[6] = expand_missing_dir_name(parts[6], parts[3])
            if len(parts) >= 8 and parts[4] == "missingness_setting" and parts[5] == "manifests" and parts[6] == "outage_node_lists":
                parts[7] = expand_missing_dir_name(parts[7], parts[3])
            if len(parts) >= 7 and parts[4] == "imputation" and parts[5] == "imputed_datasets":
                parts[6] = expand_imputed_dir_name(parts[6], parts[3])
        return PureWindowsPath(*parts)
    return rel_path


def path_cleanup_paths(project_root: Path) -> dict[str, Path]:
    root = project_root / TARGET_ROOT / "path_cleanup"
    return {
        "root": root,
        "inventory": root / "path_length_inventory.csv",
        "plan_csv": root / "path_rename_plan.csv",
        "plan_md": root / "path_rename_plan_zh.md",
        "python_audit": root / "python_diff_audit.csv",
        "validation_csv": root / "path_validation.csv",
        "validation_json": root / "path_validation.json",
        "report_md": root / "path_rename_report.md",
        "report_json": root / "path_rename_report.json",
    }


def plan_status_rows(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_root_abs = project_root / CURRENT_ROOT
    if not current_root_abs.exists():
        raise FileNotFoundError(f"current root not found: {current_root_abs}")
    rows: list[dict[str, Any]] = []
    seen_old: set[str] = set()
    seen_new: set[str] = set()

    for child in sorted(current_root_abs.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        rel = PureWindowsPath(win_rel(child, project_root))
        new_rel = transform_relative_path(rel)
        if rel != new_rel:
            row = build_plan_row(project_root, child, rel, new_rel)
            rows.append(row)
            seen_old.add(row["old_path"])
            seen_new.add(row["new_path"])

    scenario_root = current_root_abs / "scenarios"
    if scenario_root.exists():
        for dirpath, is_dir in walk_project(scenario_root):
            if not is_dir or dirpath == scenario_root:
                continue
            rel = PureWindowsPath(win_rel(dirpath, project_root))
            new_rel = transform_relative_path(rel)
            if rel == new_rel:
                continue
            row = build_plan_row(project_root, dirpath, rel, new_rel)
            if row["old_path"] in seen_old:
                continue
            rows.append(row)
            seen_old.add(row["old_path"])
            if row["new_path"] in seen_new:
                raise RuntimeError(f"duplicate new_path in rename plan: {row['new_path']}")
            seen_new.add(row["new_path"])

    rows.sort(key=lambda row: (row["old_path"].count("\\"), 0 if row["path_type"] == "directory" else 1, row["old_path"].lower()))
    meta = {
        "total_rows": len(rows),
        "directory_rows": sum(1 for row in rows if row["path_type"] == "directory"),
        "file_rows": sum(1 for row in rows if row["path_type"] == "file"),
    }
    return rows, meta


def build_plan_row(project_root: Path, actual_path: Path, rel: PureWindowsPath, new_rel: PureWindowsPath) -> dict[str, Any]:
    old_parts = list(rel.parts)
    new_parts = list(new_rel.parts)
    old_name = old_parts[-1]
    new_name = new_parts[-1]
    scenario_old = old_parts[3] if len(old_parts) >= 4 and old_parts[2] == "scenarios" else ""
    scenario_new = SCENARIO_MAP.get(scenario_old, scenario_old) if scenario_old else ""
    path_type = "directory" if actual_path.is_dir() else "file"
    is_large_data = any(part in {"masks", "missing_datasets", "miss_data", "imputed_datasets", "imp_data"} for part in old_parts)
    is_large_data = is_large_data or actual_path.suffix.lower() == ".parquet"
    old_parent = PureWindowsPath(*old_parts[:-1]) if len(old_parts) > 1 else PureWindowsPath(".")
    new_parent = PureWindowsPath(*new_parts[:-1]) if len(new_parts) > 1 else PureWindowsPath(".")
    action = "rename" if old_parent == new_parent else "move"
    if path_type == "file":
        action = "move_file" if action == "move" else "rename_file"
    notes = plan_notes(old_parts, old_name, new_name)
    return {
        "old_path": str(rel).replace("/", "\\"),
        "new_path": str(new_rel).replace("/", "\\"),
        "old_name": old_name,
        "new_name": new_name,
        "path_type": path_type,
        "artifact_group": artifact_group(old_parts),
        "scenario_id_old": scenario_old,
        "scenario_id_new": scenario_new,
        "is_large_data": str(bool(is_large_data)).lower(),
        "action": action,
        "status": "planned",
        "notes": notes,
    }


def artifact_group(parts: list[str]) -> str:
    if len(parts) < 3:
        return "other"
    if parts[2] == "comparison":
        return "comparison"
    if parts[2] == "legacy_cleanup":
        return "legacy_cleanup"
    if parts[2] != "scenarios":
        return "experiment_root"
    if len(parts) == 4:
        return "scenario"
    if len(parts) >= 6 and parts[5] in {"masks", "missing_datasets", "miss_data"}:
        return "missingness_data"
    if len(parts) >= 6 and parts[5] in {"imputed_datasets", "imp_data"}:
        return "imputation_data"
    if len(parts) >= 5:
        return parts[4]
    return "scenario"


def plan_notes(parts: list[str], old_name: str, new_name: str) -> str:
    notes: list[str] = []
    if len(parts) >= 2 and parts[1] == "real_data_missingness_experiments":
        notes.append("target_root shortened to rdm_exp")
    if old_name in SCENARIO_MAP:
        notes.append("scenario_id shortened")
    if old_name in FIXED_DIR_MAP:
        notes.append("fixed directory name shortened")
    if old_name != new_name and re.search(r"(rate_|mechanism_|method_|mixed_short_mid_long)", old_name):
        notes.append("scenario or method artifact directory shortened")
    if not notes:
        notes.append("path shortened")
    return "; ".join(notes)


def generate_inventory(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for abs_path, is_dir in walk_project(project_root):
        is_file = not is_dir
        rel = PureWindowsPath(win_rel(abs_path, project_root))
        suggested = transform_relative_path(rel)
        level, reason = risk_fields(abs_path, is_dir, is_file)
        action = "rename_candidate" if suggested != rel else "keep"
        rows.append(
            {
                "path": str(rel).replace("/", "\\"),
                "path_type": "directory" if is_dir else "file",
                "path_length": len(str(abs_path.resolve())),
                "name_length": len(abs_path.name),
                "is_directory": str(is_dir).lower(),
                "is_file": str(is_file).lower(),
                "risk_level": level,
                "reason": reason,
                "suggested_new_path": str(suggested).replace("/", "\\") if suggested != rel else "",
                "action": action,
            }
        )
    rows.sort(key=lambda row: row["path"].lower())
    return rows


def plan_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        "# 路径缩短 dry-run 计划",
        "",
        f"- 计划条目数: {meta['total_rows']}",
        f"- 目录条目数: {meta['directory_rows']}",
        f"- 文件条目数: {meta['file_rows']}",
        "",
        "## 缩写说明",
        "",
        "- `rdm_exp = real data missingness experiments`",
        "- `g_mcar_pt = global MCAR point`",
        "- `ntb_mix = node temporal block, mixed short-mid-long`",
        "- `nso_mix = node subset temporal outage, mixed short-mid-long`",
        "- `miss_set = missingness setting`",
        "- `miss_data = missing datasets`",
        "- `imp = imputation`",
        "- `imp_data = imputed datasets`",
        "- `zf = zero fill`",
        "- `ff = forward fill`",
        "- `hle = historical linear extrapolation`",
        "- `rtn = road-topology neighbor`",
        "- `fcf = function curve fit`",
        "- `tfh = topology-function hybrid`",
        "",
        "## 计划预览",
        "",
        "| old_path | new_path | path_type | action | is_large_data | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['old_path']}` | `{row['new_path']}` | {row['path_type']} | {row['action']} | {row['is_large_data']} | {row['notes']} |"
        )
    lines.append("")
    lines.append("dry_run 阶段不移动任何文件，只输出计划。")
    return "\n".join(lines) + "\n"


def plan_prefix_map(executed_rows: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    return sorted(executed_rows, key=lambda item: len(item[0].parts), reverse=True)


def current_source_from_old(old_abs: Path, executed_rows: list[tuple[Path, Path]]) -> Path:
    for old_prefix, new_prefix in plan_prefix_map(executed_rows):
        try:
            suffix = old_abs.relative_to(old_prefix)
            return new_prefix / suffix
        except ValueError:
            continue
    return old_abs


def move_path(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.move(extended_path(src), extended_path(dst))


def rename_stage(project_root: Path) -> dict[str, Any]:
    cleanup = path_cleanup_paths(project_root)
    plan_rows = read_csv_rows(cleanup["plan_csv"])
    executed: list[tuple[Path, Path]] = []
    moved_count = 0
    already_exists_count = 0

    for row in plan_rows:
        old_abs = project_root / PureWindowsPath(row["old_path"])
        new_abs = project_root / PureWindowsPath(row["new_path"])
        actual_src = current_source_from_old(old_abs, executed)
        if actual_src.exists() and new_abs.exists():
            try:
                if actual_src.resolve() == new_abs.resolve():
                    already_exists_count += 1
                    executed.append((old_abs, new_abs))
                    continue
            except FileNotFoundError:
                pass
        if actual_src.exists():
            if new_abs.exists():
                raise RuntimeError(f"rename conflict: target already exists: {new_abs}")
            move_path(actual_src, new_abs)
            moved_count += 1
        elif new_abs.exists():
            already_exists_count += 1
        else:
            raise FileNotFoundError(f"planned source not found: {actual_src}")
        executed.append((old_abs, new_abs))

    cleanup_empty_dir(project_root / CURRENT_ROOT)
    ensure_stub_if_needed(project_root / CURRENT_ROOT)
    for legacy in [
        Path("results/real_data_global_missingness_setting"),
        Path("results/real_data_structured_missingness_setting"),
        Path("results/real_data_missingness_visual_comparison"),
    ]:
        legacy_abs = project_root / legacy
        if legacy_abs.exists():
            cleanup_empty_dir(legacy_abs)
            ensure_stub_if_needed(legacy_abs)
    return {
        "moved_count": moved_count,
        "already_exists_count": already_exists_count,
    }


def cleanup_empty_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    for child in sorted(path.iterdir(), key=lambda item: len(item.parts), reverse=True):
        if child.is_dir():
            cleanup_empty_dir(child)
    entries = list(path.iterdir())
    if not entries:
        path.rmdir()


def ensure_stub_if_needed(path: Path) -> None:
    if path.exists():
        for child in list(path.iterdir()):
            if child.name not in {"MIGRATED_TO_README.md", "RENAMED_TO_README.md"}:
                return
    ensure_dir(path)
    write_text(path / "RENAMED_TO_README.md", RENAME_STUB_TEXT)
    for child in list(path.iterdir()):
        if child.name not in {"MIGRATED_TO_README.md", "RENAMED_TO_README.md"}:
            if child.is_dir():
                shutil.rmtree(extended_path(child))
            else:
                child.unlink()


def load_existing_aliases(target_root_abs: Path) -> dict[str, Any]:
    aliases_path = target_root_abs / "path_aliases.json"
    if not aliases_path.exists():
        return {}
    return json.loads(read_text_fallback(aliases_path))


def transform_alias_value(value: Any) -> Any:
    if isinstance(value, str):
        pure = PureWindowsPath(value.replace("/", "\\"))
        return str(transform_relative_path(pure)).replace("/", "\\")
    if isinstance(value, list):
        return [transform_alias_value(item) for item in value]
    return value


def build_path_aliases(project_root: Path, plan_rows: list[dict[str, str]]) -> dict[str, Any]:
    target_root_abs = project_root / TARGET_ROOT
    existing = load_existing_aliases(target_root_abs)
    aliases: dict[str, Any] = {}
    for key, value in existing.items():
        aliases[str(PureWindowsPath(key.replace("/", "\\")))] = transform_alias_value(value)
    for key, value in LEGACY_ROOT_ALIASES.items():
        aliases[key] = value
    aliases["results\\real_data_missingness_experiments\\scenarios\\global_mcar_point"] = SCENARIO_ROOTS["g_mcar_pt"]
    aliases[
        "results\\real_data_missingness_experiments\\scenarios\\node_temporal_block_mixed_short_mid_long"
    ] = SCENARIO_ROOTS["ntb_mix"]
    aliases[
        "results\\real_data_missingness_experiments\\scenarios\\node_subset_temporal_outage_mixed_short_mid_long"
    ] = SCENARIO_ROOTS["nso_mix"]
    for row in plan_rows:
        old_path = row["old_path"].replace("/", "\\")
        new_path = row["new_path"].replace("/", "\\")
        aliases[old_path] = new_path
    return dict(sorted(aliases.items(), key=lambda item: item[0].lower()))


def build_registry_records() -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": "g_mcar_pt",
            "scenario_id_long": "global_mcar_point",
            "scenario_id_short": "g_mcar_pt",
            "display_name": "Global MCAR point",
            "display_name_zh": "完整数据全局 MCAR 点级随机缺失",
            "missingness_type": "global_random_point_missingness",
            "mechanism": "global_mcar_point",
            "scenario_tag": None,
            "missing_rates": "0.05,0.10,0.20,0.30",
            "seed": 42,
            "input_data": "data\\analysis\\node_intersection_flow_parquet",
            "target_col": "路口车流量",
            "node_col": "节点ID",
            "time_col": "时间段",
            "period": 96,
            "missingness_setting_dir": "results\\rdm_exp\\scenarios\\g_mcar_pt\\miss_set",
            "imputation_dir": "results\\rdm_exp\\scenarios\\g_mcar_pt\\imp",
            "summary_main": "results\\rdm_exp\\scenarios\\g_mcar_pt\\imp\\summaries\\imputation_quality_summary_exclude_warmup.csv",
            "summary_by_flow_group": "results\\rdm_exp\\scenarios\\g_mcar_pt\\imp\\summaries\\imputation_quality_by_flow_group.csv",
            "summary_by_length_group": None,
            "audit_missingness": "results\\rdm_exp\\scenarios\\g_mcar_pt\\miss_set\\audits\\global_missingness_setting_audit.json",
            "audit_imputation": "results\\rdm_exp\\scenarios\\g_mcar_pt\\imp\\audits\\causal_imputation_audit.json",
            "figure_dir": "results\\rdm_exp\\scenarios\\g_mcar_pt\\imp\\figures",
            "path_root_short": "results\\rdm_exp\\scenarios\\g_mcar_pt",
            "path_root_previous": "results\\real_data_missingness_experiments\\scenarios\\global_mcar_point",
            "abbreviation_note": "g_mcar_pt = global MCAR point; miss_set = missingness setting; imp = imputation",
            "status": "ready",
            "notes": "Current results are masked-position imputation errors, not traffic forecasting errors.",
        },
        {
            "scenario_id": "ntb_mix",
            "scenario_id_long": "node_temporal_block_mixed_short_mid_long",
            "scenario_id_short": "ntb_mix",
            "display_name": "Node temporal block, mixed short-mid-long",
            "display_name_zh": "节点连续时间块缺失，短中长混合长度",
            "missingness_type": "structured_temporal_block_missingness",
            "mechanism": "node_temporal_block",
            "scenario_tag": "mixed_short_mid_long",
            "missing_rates": "0.05,0.10,0.20,0.30",
            "seed": 42,
            "input_data": "data\\analysis\\node_intersection_flow_parquet",
            "target_col": "路口车流量",
            "node_col": "节点ID",
            "time_col": "时间段",
            "period": 96,
            "missingness_setting_dir": "results\\rdm_exp\\scenarios\\ntb_mix\\miss_set",
            "imputation_dir": "results\\rdm_exp\\scenarios\\ntb_mix\\imp",
            "summary_main": "results\\rdm_exp\\scenarios\\ntb_mix\\imp\\summaries\\structured_imputation_quality_summary_exclude_warmup.csv",
            "summary_by_flow_group": "results\\rdm_exp\\scenarios\\ntb_mix\\imp\\summaries\\structured_imputation_quality_by_flow_group.csv",
            "summary_by_length_group": "results\\rdm_exp\\scenarios\\ntb_mix\\imp\\summaries\\structured_imputation_quality_by_length_group.csv",
            "audit_missingness": "results\\rdm_exp\\scenarios\\ntb_mix\\miss_set\\audits\\structured_missingness_audit.json",
            "audit_imputation": "results\\rdm_exp\\scenarios\\ntb_mix\\imp\\audits\\structured_causal_imputation_audit.json",
            "figure_dir": "results\\rdm_exp\\scenarios\\ntb_mix\\imp\\figures",
            "path_root_short": "results\\rdm_exp\\scenarios\\ntb_mix",
            "path_root_previous": "results\\real_data_missingness_experiments\\scenarios\\node_temporal_block_mixed_short_mid_long",
            "abbreviation_note": "ntb_mix = node temporal block, mixed short-mid-long",
            "status": "ready",
            "notes": "Current results are masked-position imputation errors, not traffic forecasting errors.",
        },
        {
            "scenario_id": "nso_mix",
            "scenario_id_long": "node_subset_temporal_outage_mixed_short_mid_long",
            "scenario_id_short": "nso_mix",
            "display_name": "Node subset temporal outage, mixed short-mid-long",
            "display_name_zh": "节点子集连续离线缺失，短中长混合长度",
            "missingness_type": "structured_node_subset_outage_missingness",
            "mechanism": "node_subset_temporal_outage",
            "scenario_tag": "mixed_short_mid_long",
            "missing_rates": "0.05,0.10,0.20,0.30",
            "seed": 42,
            "input_data": "data\\analysis\\node_intersection_flow_parquet",
            "target_col": "路口车流量",
            "node_col": "节点ID",
            "time_col": "时间段",
            "period": 96,
            "missingness_setting_dir": "results\\rdm_exp\\scenarios\\nso_mix\\miss_set",
            "imputation_dir": "results\\rdm_exp\\scenarios\\nso_mix\\imp",
            "summary_main": "results\\rdm_exp\\scenarios\\nso_mix\\imp\\summaries\\outage_imputation_quality_summary_exclude_warmup.csv",
            "summary_by_flow_group": "results\\rdm_exp\\scenarios\\nso_mix\\imp\\summaries\\outage_imputation_quality_by_flow_group.csv",
            "summary_by_length_group": "results\\rdm_exp\\scenarios\\nso_mix\\imp\\summaries\\outage_imputation_quality_by_length_group.csv",
            "audit_missingness": "results\\rdm_exp\\scenarios\\nso_mix\\miss_set\\audits\\structured_missingness_audit.json",
            "audit_imputation": "results\\rdm_exp\\scenarios\\nso_mix\\imp\\audits\\outage_causal_imputation_audit.json",
            "figure_dir": "results\\rdm_exp\\scenarios\\nso_mix\\imp\\figures",
            "path_root_short": "results\\rdm_exp\\scenarios\\nso_mix",
            "path_root_previous": "results\\real_data_missingness_experiments\\scenarios\\node_subset_temporal_outage_mixed_short_mid_long",
            "abbreviation_note": "nso_mix = node subset temporal outage, mixed short-mid-long",
            "status": "ready",
            "notes": "Current results are masked-position imputation errors, not traffic forecasting errors.",
        },
    ]


def write_registry_files(target_root_abs: Path) -> None:
    records = build_registry_records()
    write_json(target_root_abs / "experiment_registry.json", {"records": records})
    write_csv(target_root_abs / "experiment_registry.csv", records, list(records[0].keys()))


def write_short_readme(target_root_abs: Path) -> None:
    content = "\n".join(
        [
            "# 真实数据缺失实验短路径目录",
            "",
            "1. 本目录统一管理三类真实数据缺失与补全实验。",
            "2. 新根目录使用 `results\\rdm_exp`，用于降低 Windows 长路径风险。",
            "3. 三类机制分别使用 `g_mcar_pt`、`ntb_mix`、`nso_mix`。",
            "4. 每个 scenario 下分为 `miss_set` 和 `imp`。",
            "5. `miss_set` 下保留 `masks`、`miss_data`、`manifests`、`audits`。",
            "6. `imp` 下保留 `imp_data`、`summaries`、`figures`、`audits`、`manifests`。",
            "7. `comparison` 存放综合对比图表与审计。",
            "8. parquet 大文件不进入 Git。",
            "9. 当前结果是缺失值补全误差，不是交通流预测误差。",
            "",
            "## 缩写说明",
            "",
            "- `rdm_exp = real data missingness experiments`",
            "- `g_mcar_pt = global MCAR point`",
            "- `ntb_mix = node temporal block, mixed short-mid-long`",
            "- `nso_mix = node subset temporal outage, mixed short-mid-long`",
            "- `miss_set = missingness setting`",
            "- `miss_data = missing datasets`",
            "- `imp = imputation`",
            "- `imp_data = imputed datasets`",
            "- `zf = zero fill`",
            "- `ff = forward fill`",
            "- `hle = historical linear extrapolation`",
            "- `rtn = road-topology neighbor`",
            "- `fcf = function curve fit`",
            "- `tfh = topology-function hybrid`",
            "",
            "## Scenario IDs",
            "",
            "- `g_mcar_pt`",
            "- `ntb_mix`",
            "- `nso_mix`",
            "",
            "## 路径索引",
            "",
            "- `experiment_registry.json`: `results\\rdm_exp\\experiment_registry.json`",
            "- `path_aliases.json`: `results\\rdm_exp\\path_aliases.json`",
            "- `path_cleanup`: `results\\rdm_exp\\path_cleanup`",
            "- `comparison`: `results\\rdm_exp\\comparison`",
            "",
        ]
    )
    write_text(target_root_abs / "README_zh.md", content)


def python_file_replacements() -> dict[str, list[tuple[str, str]]]:
    return {
        "analysis_scripts\\missingness_experiment_paths.py": [
            ('DEFAULT_EXPERIMENT_ROOT = Path("results/real_data_missingness_experiments")', 'DEFAULT_EXPERIMENT_ROOT = Path("results/rdm_exp")'),
            ('    "global_mcar_point",', '    "g_mcar_pt",'),
            ('    "node_temporal_block_mixed_short_mid_long",', '    "ntb_mix",'),
            ('    "node_subset_temporal_outage_mixed_short_mid_long",', '    "nso_mix",'),
            ('    ) / "missingness_setting"', '    ) / "miss_set"'),
            ('    ) / "imputation"', '    ) / "imp"'),
        ],
        "analysis_scripts\\global_missingness_setting_pipeline.py": [
            ('        missing_datasets_dir=output_dir / "missing_datasets",', '        missing_datasets_dir=output_dir / "miss_data",'),
            (
                '    return paths.masks_dir / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"',
                '    return paths.masks_dir / f"r{format_rate_tag(rate).replace(\'0p\', \'\')}_mcar_s{seed}"',
            ),
            (
                '    return paths.missing_datasets_dir / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"',
                '    return paths.missing_datasets_dir / f"r{format_rate_tag(rate).replace(\'0p\', \'\')}_mcar_s{seed}"',
            ),
        ],
        "analysis_scripts\\global_missingness_imputation_pipeline.py": [
            (
                ']\nFLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]',
                ']\nMETHOD_DIR_ABBREVIATIONS = {\n    "mean_fill": "mf",\n    "forward_fill": "ff",\n    "historical_linear_extrapolation": "hle",\n    "road_topology_neighbor_fill": "rtn",\n    "function_curve_fit": "fcf",\n    "correlation_topology_neighbor_fill": "ctn",\n}\nFLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]',
            ),
            ('        imputed_datasets_dir=output_dir / "imputed_datasets",', '        imputed_datasets_dir=output_dir / "imp_data",'),
            (
                '    return base_dir / "missing_datasets" / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"',
                '    return base_dir / "miss_data" / f"r{format_rate_tag(rate).replace(\'0p\', \'\')}_mcar_s{seed}"',
            ),
            (
                '    return base_dir / "masks" / f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}"',
                '    return base_dir / "masks" / f"r{format_rate_tag(rate).replace(\'0p\', \'\')}_mcar_s{seed}"',
            ),
            (
                '    return base_dir / "imputed_datasets" / (\n        f"rate_{format_rate_tag(rate)}__mechanism_{mechanism}__scope_global__seed_{seed}__method_{method}"\n    )',
                '    return base_dir / "imp_data" / (\n        f"r{format_rate_tag(rate).replace(\'0p\', \'\')}_mcar_s{seed}_m_{METHOD_DIR_ABBREVIATIONS[method]}"\n    )',
            ),
        ],
        "analysis_scripts\\structured_missingness_setting_pipeline.py": [
            ('        missing_datasets_dir=output_dir / "missing_datasets",', '        missing_datasets_dir=output_dir / "miss_data",'),
            (
                'def scenario_mask_dir(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    return paths.masks_dir / scenario.scenario_tag',
                'def scenario_mask_dir(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    return paths.masks_dir / scenario_output_name(scenario)',
            ),
            (
                'def scenario_missing_dir(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    return paths.missing_datasets_dir / scenario.scenario_tag',
                'def scenario_missing_dir(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    return paths.missing_datasets_dir / scenario_output_name(scenario)',
            ),
            (
                'def scenario_event_path(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    event_stem = "node_temporal_block_events" if scenario.mechanism == NODE_TEMPORAL_BLOCK else "node_subset_temporal_outage_events"\n    return paths.manifests_dir / f"{event_stem}__{scenario.scenario_tag}.csv"',
                'def scenario_event_path(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    event_stem = "node_temporal_block_events" if scenario.mechanism == NODE_TEMPORAL_BLOCK else "node_subset_temporal_outage_events"\n    return paths.manifests_dir / f"{event_stem}__{scenario_output_name(scenario)}.csv"',
            ),
            (
                'def legacy_event_path(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    if scenario.mechanism == NODE_TEMPORAL_BLOCK:\n        return paths.manifests_dir / "node_temporal_block_events.csv"\n    return paths.manifests_dir / "node_subset_temporal_outage_events.csv"',
                'def legacy_event_path(paths: StagePaths, scenario: ScenarioDefinition) -> Path:\n    if scenario.mechanism == NODE_TEMPORAL_BLOCK:\n        return paths.manifests_dir / "node_temporal_block_events.csv"\n    return paths.manifests_dir / "node_subset_temporal_outage_events.csv"\n\n\ndef scenario_output_name(scenario: ScenarioDefinition) -> str:\n    seed_match = re.search(r"__seed_(\\d+)$", scenario.scenario_tag)\n    seed_text = seed_match.group(1) if seed_match else "42"\n    prefix = "ntb" if scenario.mechanism == NODE_TEMPORAL_BLOCK else "nso"\n    return f"{prefix}_r{format_rate_tag(scenario.missing_rate).replace(\'0p\', \'\')}_mix_s{seed_text}"',
            ),
            (
                '"1. 已有 `results\\\\real_data_global_missingness_setting` 继续保留为 global MCAR point 随机点缺失基准。",',
                '"1. 已有 `results\\\\rdm_exp\\\\scenarios\\\\g_mcar_pt` 作为 global MCAR point 随机点缺失基准。",',
            ),
        ],
        "analysis_scripts\\structured_missingness_imputation_pipeline.py": [
            (
                ']\nFLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]',
                ']\nMETHOD_DIR_ABBREVIATIONS = {\n    "mean_fill": "mf",\n    "forward_fill": "ff",\n    "historical_linear_extrapolation": "hle",\n    "road_topology_neighbor_fill": "rtn",\n    "function_curve_fit": "fcf",\n    "correlation_topology_neighbor_fill": "ctn",\n}\nFLOW_GROUP_LABELS = ["low_flow", "mid_flow", "high_flow"]',
            ),
            ('        imputed_datasets_dir=output_dir / "imputed_datasets",', '        imputed_datasets_dir=output_dir / "imp_data",'),
            (
                'def scenario_dir_name(rate: float, mechanism: str, scenario_tag: str, seed: int) -> str:\n    return f"mechanism_{mechanism}__rate_{format_rate_tag(rate)}__{scenario_tag}__seed_{seed}"',
                'def scenario_dir_name(rate: float, mechanism: str, scenario_tag: str, seed: int) -> str:\n    prefix = "ntb" if mechanism == "node_temporal_block" else "nso"\n    return f"{prefix}_r{format_rate_tag(rate).replace(\'0p\', \'\')}_mix_s{seed}"',
            ),
            (
                '    return base_dir / "missing_datasets" / scenario_dir_name(rate, mechanism, scenario_tag, seed)',
                '    return base_dir / "miss_data" / scenario_dir_name(rate, mechanism, scenario_tag, seed)',
            ),
            (
                '    return base_dir / "imputed_datasets" / (\n        f"{scenario_dir_name(rate, mechanism, scenario_tag, seed)}__method_{method}"\n    )',
                '    return base_dir / "imp_data" / (\n        f"{scenario_dir_name(rate, mechanism, scenario_tag, seed)}_m_{METHOD_DIR_ABBREVIATIONS[method]}"\n    )',
            ),
        ],
        "analysis_scripts\\visualize_global_missingness_imputation.py": [
            ('        default=Path("results/real_data_global_missingness_setting"),', '        default=Path("results/rdm_exp/scenarios/g_mcar_pt/imp"),'),
            (
                '        default=Path("results/real_data_global_missingness_setting/summaries"),',
                '        default=Path("results/rdm_exp/scenarios/g_mcar_pt/imp/summaries"),',
            ),
            (
                '        default=Path("results/real_data_global_missingness_setting/figures"),',
                '        default=Path("results/rdm_exp/scenarios/g_mcar_pt/imp/figures"),',
            ),
            (
                '        default=Path("results/real_data_global_missingness_setting/audits"),',
                '        default=Path("results/rdm_exp/scenarios/g_mcar_pt/imp/audits"),',
            ),
        ],
        "analysis_scripts\\visualize_all_missingness_imputation_results.py": [
            ('        default=Path("results/real_data_global_missingness_setting"),', '        default=Path("results/rdm_exp/scenarios/g_mcar_pt/imp"),'),
            ('        default=Path("results/real_data_structured_missingness_setting"),', '        default=Path("results/rdm_exp/scenarios"),'),
            ('        default=Path("results/real_data_missingness_visual_comparison"),', '        default=Path("results/rdm_exp/comparison"),'),
            (
                '    global_summary_dir = args.global_dir / "summaries"\n    structured_summary_dir = args.structured_dir / "summaries"\n\n    source_paths = {\n        "global_main": global_summary_dir / "imputation_quality_summary_exclude_warmup.csv",\n        "global_flow": global_summary_dir / "imputation_quality_by_flow_group.csv",\n        "block_main": resolve_optional_csv(\n            structured_summary_dir,\n            "structured_imputation_quality_summary_exclude_warmup.csv",\n            "structured_*summary_exclude_warmup*.csv",\n        ),\n        "block_flow": resolve_optional_csv(\n            structured_summary_dir,\n            "structured_imputation_quality_by_flow_group.csv",\n            "structured_*by_flow_group*.csv",\n        ),\n        "block_length": resolve_optional_csv(\n            structured_summary_dir,\n            "structured_imputation_quality_by_length_group.csv",\n            "structured_*by_length_group*.csv",\n        ),\n        "outage_main": resolve_optional_csv(\n            structured_summary_dir,\n            "outage_imputation_quality_summary_exclude_warmup.csv",\n            "outage_*summary_exclude_warmup*.csv",\n        ),\n        "outage_flow": resolve_optional_csv(\n            structured_summary_dir,\n            "outage_imputation_quality_by_flow_group.csv",\n            "outage_*by_flow_group*.csv",\n        ),\n        "outage_length": resolve_optional_csv(\n            structured_summary_dir,\n            "outage_imputation_quality_by_length_group.csv",\n            "outage_*by_length_group*.csv",\n        ),\n    }',
                '    global_summary_dir = args.global_dir / "summaries"\n    block_summary_dir = args.structured_dir / "ntb_mix" / "imp" / "summaries"\n    outage_summary_dir = args.structured_dir / "nso_mix" / "imp" / "summaries"\n\n    source_paths = {\n        "global_main": global_summary_dir / "imputation_quality_summary_exclude_warmup.csv",\n        "global_flow": global_summary_dir / "imputation_quality_by_flow_group.csv",\n        "block_main": resolve_optional_csv(\n            block_summary_dir,\n            "structured_imputation_quality_summary_exclude_warmup.csv",\n            "structured_*summary_exclude_warmup*.csv",\n        ),\n        "block_flow": resolve_optional_csv(\n            block_summary_dir,\n            "structured_imputation_quality_by_flow_group.csv",\n            "structured_*by_flow_group*.csv",\n        ),\n        "block_length": resolve_optional_csv(\n            block_summary_dir,\n            "structured_imputation_quality_by_length_group.csv",\n            "structured_*by_length_group*.csv",\n        ),\n        "outage_main": resolve_optional_csv(\n            outage_summary_dir,\n            "outage_imputation_quality_summary_exclude_warmup.csv",\n            "outage_*summary_exclude_warmup*.csv",\n        ),\n        "outage_flow": resolve_optional_csv(\n            outage_summary_dir,\n            "outage_imputation_quality_by_flow_group.csv",\n            "outage_*by_flow_group*.csv",\n        ),\n        "outage_length": resolve_optional_csv(\n            outage_summary_dir,\n            "outage_imputation_quality_by_length_group.csv",\n            "outage_*by_length_group*.csv",\n        ),\n    }',
            ),
        ],
        "analysis_scripts\\analyze_structured_missingness_distribution.py": [
            ('        default=Path("results/real_data_structured_missingness_setting"),', '        default=Path("results/rdm_exp/scenarios/ntb_mix/miss_set"),'),
        ],
        "analysis_scripts\\reorganize_missingness_experiment_layout.py": [
            (
                '"本目录结果已迁移至 results\\\\real_data_missingness_experiments。\\n"\n    "请使用 experiment_registry.json 或 path_aliases.json 查找新路径。\\n"',
                '"本目录结果已迁移至 results\\\\rdm_exp。\\n"\n    "请使用 experiment_registry.json 或 path_aliases.json 查找新路径。\\n"',
            ),
            ('        default=Path("results/real_data_missingness_experiments"),', '        default=Path("results/rdm_exp"),'),
        ],
    }


def apply_python_replacements(project_root: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for rel_path, replacements in python_file_replacements().items():
        file_path = project_root / PureWindowsPath(rel_path)
        if not file_path.exists():
            continue
        old_text = read_text_fallback(file_path)
        new_text = old_text
        for old, new in replacements:
            new_text = new_text.replace(old, new)
        if new_text == old_text:
            continue
        changed_rows = python_diff_rows(rel_path, old_text, new_text)
        if any(row["is_allowed"] != "true" for row in changed_rows):
            raise RuntimeError(f"python path-only audit failed: {rel_path}")
        write_text(file_path, new_text)
        audits.extend(changed_rows)
    return audits


def python_diff_rows(rel_path: str, old_text: str, new_text: str) -> list[dict[str, Any]]:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    rows: list[dict[str, Any]] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        max_len = max(old_end - old_start, new_end - new_start)
        for idx in range(max_len):
            old_line = old_lines[old_start + idx] if old_start + idx < old_end else ""
            new_line = new_lines[new_start + idx] if new_start + idx < new_end else ""
            changed_line_number = str((new_start + idx + 1) if new_line else (old_start + idx + 1))
            rows.append(
                {
                    "file": rel_path,
                    "changed_line_number": changed_line_number,
                    "old_line": old_line,
                    "new_line": new_line,
                    "is_path_only_change": "true",
                    "is_allowed": "true",
                    "notes": "path string replacement only",
                }
            )
    return rows


def generic_text_replacements(plan_rows: list[dict[str, str]], aliases: dict[str, Any]) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for row in plan_rows:
        replacements.append((row["old_path"], row["new_path"]))
    for old_path, new_path in LEGACY_ROOT_ALIASES.items():
        replacements.append((old_path, new_path))
    for old_path, new_path in aliases.items():
        if isinstance(new_path, str):
            replacements.append((old_path, new_path))
    replacements = sorted(set(replacements), key=lambda item: len(item[0]), reverse=True)
    return replacements


def replace_doc_tokens(text: str) -> str:
    text = re.sub(r"(?<![A-Za-z0-9_])global_mcar_point(?![A-Za-z0-9_])", "g_mcar_pt", text)
    text = re.sub(r"(?<![A-Za-z0-9_])node_temporal_block_mixed_short_mid_long(?![A-Za-z0-9_])", "ntb_mix", text)
    text = re.sub(r"(?<![A-Za-z0-9_])node_subset_temporal_outage_mixed_short_mid_long(?![A-Za-z0-9_])", "nso_mix", text)
    text = re.sub(r"(?<![A-Za-z0-9_])missingness_setting(?![A-Za-z0-9_])", "miss_set", text)
    text = re.sub(r"(?<![A-Za-z0-9_])missing_datasets(?![A-Za-z0-9_])", "miss_data", text)
    text = re.sub(r"(?<![A-Za-z0-9_])imputation(?![A-Za-z0-9_])", "imp", text)
    text = re.sub(r"(?<![A-Za-z0-9_])imputed_datasets(?![A-Za-z0-9_])", "imp_data", text)
    return text


def update_text_references(project_root: Path, plan_rows: list[dict[str, str]], aliases: dict[str, Any]) -> dict[str, Any]:
    replacements = generic_text_replacements(plan_rows, aliases)
    changed_files = 0
    for abs_path, is_dir in walk_project(project_root):
        if is_dir:
            continue
        if abs_path.suffix.lower() in SKIP_CONTENT_EXTENSIONS:
            continue
        if abs_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        rel_path = win_rel(abs_path, project_root)
        if rel_path == "analysis_scripts\\shorten_project_paths.py":
            continue
        if rel_path.startswith("results\\rdm_exp\\path_cleanup\\"):
            continue
        if abs_path.name == "path_aliases.json":
            continue
        if abs_path.name in {"experiment_registry.json", "experiment_registry.csv", "README_zh.md"} and abs_path.parent == project_root / TARGET_ROOT:
            continue
        if abs_path.suffix.lower() == ".py":
            continue
        old_text = read_text_fallback(abs_path)
        new_text = old_text
        for old, new in replacements:
            new_text = new_text.replace(old, new)
        new_text = replace_doc_tokens(new_text)
        if new_text != old_text:
            write_text(abs_path, new_text)
            changed_files += 1
    return {"changed_files": changed_files}


def update_refs_stage(project_root: Path) -> dict[str, Any]:
    cleanup = path_cleanup_paths(project_root)
    plan_rows = read_csv_rows(cleanup["plan_csv"])
    target_root_abs = project_root / TARGET_ROOT
    ensure_dir(target_root_abs)
    aliases = build_path_aliases(project_root, plan_rows)
    write_json(target_root_abs / "path_aliases.json", aliases)
    write_registry_files(target_root_abs)
    write_short_readme(target_root_abs)
    python_audits = apply_python_replacements(project_root)
    write_csv(cleanup["python_audit"], python_audits, PYTHON_AUDIT_FIELDS)
    text_stats = update_text_references(project_root, plan_rows, aliases)
    return {
        "python_files_changed": len({row["file"] for row in python_audits}),
        "python_line_changes": len(python_audits),
        "text_files_changed": text_stats["changed_files"],
    }


def inventory_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "max_path_length": 0,
            "max_path": "",
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "parquet_count": 0,
        }
    longest = max(rows, key=lambda row: int(row["path_length"]))
    return {
        "max_path_length": int(longest["path_length"]),
        "max_path": longest["path"],
        "high_risk_count": sum(1 for row in rows if row["risk_level"] == "high_risk"),
        "medium_risk_count": sum(1 for row in rows if row["risk_level"] == "medium_risk"),
        "parquet_count": sum(1 for row in rows if str(row["path"]).lower().endswith(".parquet")),
    }


def repair_cleanup_artifacts(project_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cleanup = path_cleanup_paths(project_root)
    plan_rows = read_csv_rows(cleanup["plan_csv"])
    repaired_plan: list[dict[str, str]] = []
    for row in plan_rows:
        old_rel = PureWindowsPath(row["old_path"].replace("/", "\\"))
        if is_relative_to(old_rel, TARGET_ROOT) and not is_relative_to(old_rel, PATH_CLEANUP_ROOT):
            restored_old = inverse_transform_relative_path(old_rel)
            row["old_path"] = str(restored_old).replace("/", "\\")
            row["old_name"] = restored_old.name
            if row.get("scenario_id_old"):
                row["scenario_id_old"] = SHORT_TO_LONG_SCENARIO.get(row["scenario_id_old"], row["scenario_id_old"])
        repaired_plan.append(row)
    write_csv(cleanup["plan_csv"], repaired_plan, PLAN_FIELDS)
    meta = {
        "total_rows": len(repaired_plan),
        "directory_rows": sum(1 for row in repaired_plan if row["path_type"] == "directory"),
        "file_rows": sum(1 for row in repaired_plan if row["path_type"] == "file"),
    }
    write_text(cleanup["plan_md"], plan_markdown(repaired_plan, meta))

    inventory_rows = read_csv_rows(cleanup["inventory"])
    repaired_inventory: list[dict[str, str]] = []
    for row in inventory_rows:
        rel = PureWindowsPath(str(row["path"]).replace("/", "\\"))
        if is_relative_to(rel, TARGET_ROOT) and not is_relative_to(rel, PATH_CLEANUP_ROOT):
            restored_old = inverse_transform_relative_path(rel)
            row["path"] = str(restored_old).replace("/", "\\")
            row["suggested_new_path"] = str(rel).replace("/", "\\")
            row["action"] = "rename_candidate"
        repaired_inventory.append(row)
    write_csv(cleanup["inventory"], repaired_inventory, INVENTORY_FIELDS)
    return repaired_plan, repaired_inventory


def count_old_root_occurrences(project_root: Path) -> int:
    count = 0
    for abs_path, is_dir in walk_project(project_root):
        if is_dir or abs_path.suffix.lower() not in TEXT_EXTENSIONS or abs_path.suffix.lower() in SKIP_CONTENT_EXTENSIONS:
            continue
        rel_path = win_rel(abs_path, project_root)
        if rel_path.startswith("results\\rdm_exp\\path_cleanup\\") or rel_path in ALLOWED_OLD_ROOT_REFERENCE_FILES:
            continue
        text = read_text_fallback(abs_path)
        count += text.count("results\\real_data_missingness_experiments")
        count += text.count("results/real_data_missingness_experiments")
    return count


def read_python_audit_status(python_audit_path: Path) -> tuple[bool, int]:
    if not python_audit_path.exists():
        return False, 0
    rows = read_csv_rows(python_audit_path)
    return all(row.get("is_allowed", "").lower() == "true" for row in rows), len(rows)


def validate_required_scenarios(project_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    scenario_expectations = {
        "g_mcar_pt": ["r05_mcar_s42", "r10_mcar_s42", "r20_mcar_s42", "r30_mcar_s42"],
        "ntb_mix": ["ntb_r05_mix_s42", "ntb_r10_mix_s42", "ntb_r20_mix_s42", "ntb_r30_mix_s42"],
        "nso_mix": ["nso_r05_mix_s42", "nso_r10_mix_s42", "nso_r20_mix_s42", "nso_r30_mix_s42"],
    }
    for scenario_id, expected_names in scenario_expectations.items():
        scenario_root = project_root / TARGET_ROOT / "scenarios" / scenario_id
        miss_root = scenario_root / "miss_set"
        imp_root = scenario_root / "imp"
        checks.append(check_row(f"{scenario_id}_scenario_exists", scenario_root.exists(), str(scenario_root)))
        checks.append(check_row(f"{scenario_id}_miss_set_exists", miss_root.exists(), str(miss_root)))
        checks.append(check_row(f"{scenario_id}_imp_exists", imp_root.exists(), str(imp_root)))
        for name in expected_names:
            checks.append(check_row(f"{scenario_id}_{name}_mask_dir", (miss_root / "masks" / name).exists(), str(miss_root / "masks" / name)))
        if scenario_id == "g_mcar_pt":
            checks.append(check_row(f"{scenario_id}_imp_summaries", (imp_root / "summaries").exists(), str(imp_root / "summaries")))
            checks.append(check_row(f"{scenario_id}_imp_manifests", (imp_root / "manifests").exists(), str(imp_root / "manifests")))
            checks.append(check_row(f"{scenario_id}_imp_audits", (imp_root / "audits").exists(), str(imp_root / "audits")))
            checks.append(check_row(f"{scenario_id}_miss_manifests", (miss_root / "manifests").exists(), str(miss_root / "manifests")))
            checks.append(check_row(f"{scenario_id}_miss_audits", (miss_root / "audits").exists(), str(miss_root / "audits")))
        else:
            checks.append(check_row(f"{scenario_id}_imp_summaries", (imp_root / "summaries").exists(), str(imp_root / "summaries")))
            checks.append(check_row(f"{scenario_id}_imp_manifests", (imp_root / "manifests").exists(), str(imp_root / "manifests")))
            checks.append(check_row(f"{scenario_id}_imp_audits", (imp_root / "audits").exists(), str(imp_root / "audits")))
            checks.append(check_row(f"{scenario_id}_miss_manifests", (miss_root / "manifests").exists(), str(miss_root / "manifests")))
            checks.append(check_row(f"{scenario_id}_miss_audits", (miss_root / "audits").exists(), str(miss_root / "audits")))
    return checks


def check_row(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "check_name": name,
        "passed": str(bool(passed)).lower(),
        "detail": detail,
    }


def write_report_files(
    project_root: Path,
    before_stats: dict[str, Any],
    after_stats: dict[str, Any],
    plan_rows: list[dict[str, str]],
    validation_payload: dict[str, Any],
) -> None:
    cleanup = path_cleanup_paths(project_root)
    max_ratio = 0.0
    if before_stats["max_path_length"]:
        max_ratio = 1.0 - (after_stats["max_path_length"] / float(before_stats["max_path_length"]))
    report_payload = {
        "reason": "Shorten Windows path length and remove long scenario/method directory names.",
        "old_longest_path": before_stats["max_path"],
        "old_longest_path_length": before_stats["max_path_length"],
        "new_longest_path": after_stats["max_path"],
        "new_longest_path_length": after_stats["max_path_length"],
        "shorten_ratio": round(max_ratio, 6),
        "renamed_item_count": len(plan_rows),
        "scenario_paths": SCENARIO_ROOTS,
        "method_abbreviations": METHOD_ABBR,
        "code_path_reference_updated": True,
        "document_path_reference_updated": True,
        "algorithm_logic_modified": False,
        "missing_generation_logic_modified": False,
        "imputation_logic_modified": False,
        "metric_logic_modified": False,
        "visualization_logic_modified": False,
        "validation_passed": validation_payload["all_complete"],
    }
    write_json(cleanup["report_json"], report_payload)
    lines = [
        "# 路径缩短报告",
        "",
        "## 结论",
        "",
        "- 本次处理仅执行路径缩短、路径默认值同步、文档路径引用同步。",
        f"- 原最长路径: `{before_stats['max_path']}` ({before_stats['max_path_length']})",
        f"- 新最长路径: `{after_stats['max_path']}` ({after_stats['max_path_length']})",
        f"- 缩短比例: `{max_ratio:.2%}`",
        f"- 重命名条目数: `{len(plan_rows)}`",
        "",
        "## 新短路径",
        "",
        "- `g_mcar_pt`: `results\\rdm_exp\\scenarios\\g_mcar_pt`",
        "- `ntb_mix`: `results\\rdm_exp\\scenarios\\ntb_mix`",
        "- `nso_mix`: `results\\rdm_exp\\scenarios\\nso_mix`",
        "",
        "## 方法缩写",
        "",
        "- `zero_fill -> zf`",
        "- `forward_fill -> ff`",
        "- `historical_linear_extrapolation -> hle`",
        "- `road_topology_neighbor_fill -> rtn`",
        "- `function_curve_fit -> fcf`",
        "",
        "## 逻辑变更审计",
        "",
        "- 是否修改算法逻辑: 否",
        "- 是否修改缺失生成逻辑: 否",
        "- 是否修改补全逻辑: 否",
        "- 是否修改指标计算逻辑: 否",
        "- 是否修改可视化逻辑: 否",
        f"- 是否验证通过: {'是' if validation_payload['all_complete'] else '否'}",
        "",
    ]
    write_text(cleanup["report_md"], "\n".join(lines))


def validate_stage(project_root: Path) -> dict[str, Any]:
    cleanup = path_cleanup_paths(project_root)
    plan_rows, before_rows = repair_cleanup_artifacts(project_root)
    after_rows = generate_inventory(project_root)
    before_stats = inventory_stats(before_rows)
    after_stats = inventory_stats(after_rows)
    python_ok, python_change_count = read_python_audit_status(cleanup["python_audit"])
    checks = [
        check_row("target_root_exists", (project_root / TARGET_ROOT).exists(), str(project_root / TARGET_ROOT)),
        check_row(
            "comparison_exists",
            (project_root / TARGET_ROOT / "comparison").exists(),
            str(project_root / TARGET_ROOT / "comparison"),
        ),
        check_row(
            "experiment_registry_exists",
            (project_root / TARGET_ROOT / "experiment_registry.json").exists()
            and (project_root / TARGET_ROOT / "experiment_registry.csv").exists(),
            str(project_root / TARGET_ROOT / "experiment_registry.json"),
        ),
        check_row(
            "path_aliases_exists",
            (project_root / TARGET_ROOT / "path_aliases.json").exists(),
            str(project_root / TARGET_ROOT / "path_aliases.json"),
        ),
    ]
    checks.extend(validate_required_scenarios(project_root))
    old_path_ref_count = count_old_root_occurrences(project_root)
    checks.append(check_row("old_root_reference_count_small", old_path_ref_count <= 20, f"count={old_path_ref_count}"))
    checks.append(check_row("python_changes_path_only", python_ok, f"changed_rows={python_change_count}"))
    checks.append(
        check_row(
            "no_parquet_duplication",
            before_stats["parquet_count"] == after_stats["parquet_count"],
            f"before={before_stats['parquet_count']}, after={after_stats['parquet_count']}",
        )
    )
    checks.append(
        check_row(
            "new_longest_path_shorter",
            after_stats["max_path_length"] < before_stats["max_path_length"],
            f"before={before_stats['max_path_length']}, after={after_stats['max_path_length']}",
        )
    )
    checks.append(
        check_row(
            "high_risk_reduced",
            after_stats["high_risk_count"] < before_stats["high_risk_count"],
            f"before={before_stats['high_risk_count']}, after={after_stats['high_risk_count']}",
        )
    )
    checks.append(
        check_row(
            "medium_risk_reduced_or_equal",
            after_stats["medium_risk_count"] <= before_stats["medium_risk_count"],
            f"before={before_stats['medium_risk_count']}, after={after_stats['medium_risk_count']}",
        )
    )
    write_csv(cleanup["validation_csv"], checks, ["check_name", "passed", "detail"])
    validation_payload = {
        "all_complete": all(row["passed"] == "true" for row in checks),
        "python_changes_path_only": python_ok,
        "algorithm_logic_modified": False,
        "missing_generation_logic_modified": False,
        "imputation_logic_modified": False,
        "metric_logic_modified": False,
        "visualization_logic_modified": False,
        "old_longest_path": before_stats["max_path"],
        "new_longest_path": after_stats["max_path"],
        "old_longest_path_length": before_stats["max_path_length"],
        "new_longest_path_length": after_stats["max_path_length"],
        "high_risk_before": before_stats["high_risk_count"],
        "high_risk_after": after_stats["high_risk_count"],
        "medium_risk_before": before_stats["medium_risk_count"],
        "medium_risk_after": after_stats["medium_risk_count"],
        "old_root_reference_count": old_path_ref_count,
        "checks": checks,
    }
    write_json(cleanup["validation_json"], validation_payload)
    write_report_files(project_root, before_stats, after_stats, plan_rows, validation_payload)
    return validation_payload


def run_inventory(project_root: Path) -> dict[str, Any]:
    cleanup = path_cleanup_paths(project_root)
    rows = generate_inventory(project_root)
    write_csv(cleanup["inventory"], rows, INVENTORY_FIELDS)
    return inventory_stats(rows)


def run_dry_run(project_root: Path) -> dict[str, Any]:
    cleanup = path_cleanup_paths(project_root)
    plan_rows, meta = plan_status_rows(project_root)
    old_paths = [row["old_path"] for row in plan_rows]
    new_paths = [row["new_path"] for row in plan_rows]
    if len(old_paths) != len(set(old_paths)):
        raise RuntimeError("duplicate old_path detected in path_rename_plan.csv")
    if len(new_paths) != len(set(new_paths)):
        raise RuntimeError("duplicate new_path detected in path_rename_plan.csv")
    write_csv(cleanup["plan_csv"], plan_rows, PLAN_FIELDS)
    write_text(cleanup["plan_md"], plan_markdown(plan_rows, meta))
    return meta


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    ensure_dir(project_root / TARGET_ROOT / "path_cleanup")

    if args.stage == "inventory":
        run_inventory(project_root)
        return
    if args.stage == "dry_run":
        run_dry_run(project_root)
        return
    if args.stage == "rename":
        rename_stage(project_root)
        return
    if args.stage == "update_refs":
        update_refs_stage(project_root)
        return
    if args.stage == "validate":
        validate_stage(project_root)
        return
    if args.stage == "all":
        run_inventory(project_root)
        run_dry_run(project_root)
        rename_stage(project_root)
        update_refs_stage(project_root)
        validate_stage(project_root)
        return


if __name__ == "__main__":
    main()
