from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from analysis_scripts.missingness_experiment_paths import SCENARIO_IDS
except ImportError:
    from missingness_experiment_paths import SCENARIO_IDS


GLOBAL_SCENARIO_ID = "global_mcar_point"
BLOCK_SCENARIO_ID = "node_temporal_block_mixed_short_mid_long"
OUTAGE_SCENARIO_ID = "node_subset_temporal_outage_mixed_short_mid_long"

SCENARIO_META = {
    GLOBAL_SCENARIO_ID: {
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
    },
    BLOCK_SCENARIO_ID: {
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
    },
    OUTAGE_SCENARIO_ID: {
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
    },
}

EXPECTED_RATES = ["0p05", "0p10", "0p20", "0p30"]
README_TEXT = (
    "本目录结果已迁移至 results\\rdm_exp。\n"
    "请使用 experiment_registry.json 或 path_aliases.json 查找新路径。\n"
)

GLOBAL_MISSINGNESS_MANIFESTS = {
    "generate_missing_chunk_status.csv",
    "global_eligible_chunk_counts.csv",
    "global_eligible_summary.json",
    "global_missing_allocation.csv",
}
GLOBAL_IMPUTATION_MANIFESTS = {
    "flow_group_thresholds.json",
    "imputation_input_check.csv",
    "imputation_input_check.json",
    "imputed_chunk_runtime_state.jsonl",
    "imputed_chunk_status.csv",
    "imputed_resume_scan.csv",
}
GLOBAL_MISSINGNESS_AUDITS = {
    "global_missingness_setting_audit.json",
    "global_missingness_setting_audit_zh.md",
    "global_missingness_rate_by_chunk.csv",
}
GLOBAL_IMPUTATION_AUDITS = {
    "causal_imputation_audit.json",
    "causal_imputation_audit_zh.md",
    "visualization_audit.json",
    "visualization_audit_zh.md",
}

STRUCTURED_SHARED_MISSINGNESS_AUDITS = {
    "structured_missingness_audit.json",
    "structured_missingness_audit_zh.md",
    "structured_missingness_day_profile.csv",
    "structured_missingness_distribution_report.json",
    "structured_missingness_distribution_report_zh.md",
    "structured_missingness_distribution_summary.csv",
    "structured_missingness_rate_by_chunk.csv",
    "structured_missingness_time_slot_profile.csv",
    "structured_missingness_top_nodes.csv",
}
STRUCTURED_SHARED_MISSINGNESS_MANIFESTS = {
    "global_time_index_summary.json",
    "node_index.csv",
    "structured_missing_chunk_runtime_state.jsonl",
    "structured_missing_chunk_status.csv",
    "structured_missing_scenario_summary.csv",
    "structured_prepare_chunk_summary.csv",
}
STRUCTURED_SHARED_MISSINGNESS_FIGURES = {
    "structured_missingness_length_group_ratio.png",
    "structured_missingness_observed_rate.png",
    "structured_missingness_time_slot_heatmap.png",
}
STRUCTURED_SHARED_MISC = {
    "structured_missingness_design_zh.md",
}
STRUCTURED_BLOCK_IMPUTATION_MANIFESTS = {
    "structured_flow_group_thresholds.json",
    "structured_imputation_input_check.csv",
    "structured_imputation_input_check.json",
    "structured_imputed_chunk_runtime_state.jsonl",
    "structured_imputed_chunk_status.csv",
    "structured_imputed_resume_scan.csv",
}
STRUCTURED_OUTAGE_IMPUTATION_MANIFESTS = {
    "structured_flow_group_thresholds.json",
    "outage_imputation_input_check.csv",
    "outage_imputation_input_check.json",
    "outage_imputed_chunk_runtime_state.jsonl",
    "outage_imputed_chunk_status.csv",
    "outage_imputed_resume_scan.csv",
}
STRUCTURED_BLOCK_SUMMARIES = {
    "structured_imputation_quality_detail.csv",
    "structured_imputation_quality_summary_all_days.csv",
    "structured_imputation_quality_summary_exclude_warmup.csv",
    "structured_imputation_quality_by_flow_group.csv",
    "structured_imputation_quality_by_length_group.csv",
}
STRUCTURED_OUTAGE_SUMMARIES = {
    "outage_imputation_quality_detail.csv",
    "outage_imputation_quality_summary_all_days.csv",
    "outage_imputation_quality_summary_exclude_warmup.csv",
    "outage_imputation_quality_by_flow_group.csv",
    "outage_imputation_quality_by_length_group.csv",
}
STRUCTURED_BLOCK_AUDITS = {
    "structured_causal_imputation_audit.json",
    "structured_causal_imputation_audit_zh.md",
}
STRUCTURED_OUTAGE_AUDITS = {
    "outage_causal_imputation_audit.json",
    "outage_causal_imputation_audit_zh.md",
}
STRUCTURED_OUTAGE_MISSINGNESS_MANIFESTS = {
    "node_subset_temporal_outage_events.csv",
    "outage_node_lists",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize completed missingness experiment results into a unified layout."
    )
    parser.add_argument(
        "--stage",
        choices=["inventory", "dry_run", "migrate", "validate", "all"],
        default="dry_run",
    )
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--target_root",
        type=Path,
        default=Path("results/rdm_exp"),
    )
    parser.add_argument(
        "--global_source",
        type=Path,
        default=Path("results/real_data_global_missingness_setting"),
    )
    parser.add_argument(
        "--structured_source",
        type=Path,
        default=Path("results/real_data_structured_missingness_setting"),
    )
    parser.add_argument(
        "--comparison_source",
        type=Path,
        default=Path("results/real_data_missingness_visual_comparison"),
    )
    return parser.parse_args()


def normalize_relative(path: Path, project_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return str(rel).replace("/", "\\")
    except Exception:
        return str(path).replace("/", "\\")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def scenario_root(target_root: Path, scenario_id: str) -> Path:
    return target_root / "scenarios" / scenario_id


def missingness_root(target_root: Path, scenario_id: str) -> Path:
    return scenario_root(target_root, scenario_id) / "missingness_setting"


def imputation_root(target_root: Path, scenario_id: str) -> Path:
    return scenario_root(target_root, scenario_id) / "imputation"


def add_plan_item(
    items: list[dict[str, Any]],
    aliases: dict[str, Any],
    project_root: Path,
    *,
    source_path: Path,
    target_path: Path,
    scenario_id: str,
    artifact_group: str,
    artifact_type: str,
    is_large_data: bool,
    action: str,
    notes: str,
) -> None:
    source_rel = normalize_relative(source_path, project_root)
    target_rel = normalize_relative(target_path, project_root)
    status = "planned" if source_path.exists() else "source_missing"
    items.append(
        {
            "source_path": source_rel,
            "target_path": target_rel,
            "scenario_id": scenario_id,
            "artifact_group": artifact_group,
            "artifact_type": artifact_type,
            "is_large_data": bool(is_large_data),
            "action": action,
            "status": status,
            "notes": notes,
        }
    )
    existing = aliases.get(source_rel)
    if existing is None:
        aliases[source_rel] = target_rel
    elif isinstance(existing, list):
        if target_rel not in existing:
            existing.append(target_rel)
    elif existing != target_rel:
        aliases[source_rel] = [existing, target_rel]


def add_root_aliases(project_root: Path, aliases: dict[str, Any], source_root: Path, target_path: Path) -> None:
    source_rel = normalize_relative(source_root, project_root)
    target_rel = normalize_relative(target_path, project_root)
    aliases[source_rel] = target_rel


def scan_immediate_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([item for item in path.iterdir() if item.is_dir()], key=lambda item: item.name)


def scan_immediate_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([item for item in path.iterdir() if item.is_file()], key=lambda item: item.name)


def build_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    project_root = args.project_root.resolve()
    target_root = (project_root / args.target_root).resolve()
    global_source = (project_root / args.global_source).resolve()
    structured_source = (project_root / args.structured_source).resolve()
    comparison_source = (project_root / args.comparison_source).resolve()

    items: list[dict[str, Any]] = []
    aliases: dict[str, Any] = {}

    add_root_aliases(project_root, aliases, global_source, scenario_root(target_root, GLOBAL_SCENARIO_ID))
    add_root_aliases(project_root, aliases, structured_source, target_root / "scenarios")
    add_root_aliases(project_root, aliases, comparison_source, target_root / "comparison")

    build_global_plan(items, aliases, project_root, target_root, global_source)
    build_structured_plan(items, aliases, project_root, target_root, structured_source)
    build_comparison_plan(items, aliases, project_root, target_root, comparison_source)

    source_roots = {
        "global_source": global_source,
        "structured_source": structured_source,
        "comparison_source": comparison_source,
        "target_root": target_root,
    }
    summary = {
        "total_items": len(items),
        "planned_moves": sum(1 for item in items if item["action"] == "move"),
        "planned_copies": sum(1 for item in items if item["action"] == "copy"),
        "missing_sources": sum(1 for item in items if item["status"] == "source_missing"),
    }
    return items, aliases, {"roots": source_roots, "summary": summary}


def build_global_plan(
    items: list[dict[str, Any]],
    aliases: dict[str, Any],
    project_root: Path,
    target_root: Path,
    source_root: Path,
) -> None:
    scenario_id = GLOBAL_SCENARIO_ID
    missing_root = missingness_root(target_root, scenario_id)
    imp_root = imputation_root(target_root, scenario_id)

    for artifact_dir, group, artifact_type in [
        ("masks", "missingness_setting", "mask"),
        ("missing_datasets", "missingness_setting", "missing_dataset"),
        ("imputed_datasets", "imputation", "imputed_dataset"),
    ]:
        for item in scan_immediate_dirs(source_root / artifact_dir):
            if group == "missingness_setting":
                target_path = missing_root / artifact_dir / item.name
            else:
                target_path = imp_root / artifact_dir / item.name
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=target_path,
                scenario_id=scenario_id,
                artifact_group=group,
                artifact_type=artifact_type,
                is_large_data=True,
                action="move",
                notes="Move large scenario directory without copying parquet files.",
            )

    for item in scan_immediate_files(source_root / "manifests"):
        if item.name in GLOBAL_MISSINGNESS_MANIFESTS:
            group = "missingness_setting"
            target_path = missing_root / "manifests" / item.name
            notes = "Global missingness-setting manifest."
        elif item.name in GLOBAL_IMPUTATION_MANIFESTS:
            group = "imputation"
            target_path = imp_root / "manifests" / item.name
            notes = "Global imputation manifest."
        else:
            group = "legacy"
            target_path = target_root / "legacy" / "global_mcar_point" / "manifests" / item.name
            notes = "Unclassified manifest retained under legacy review path."
        add_plan_item(
            items,
            aliases,
            project_root,
            source_path=item,
            target_path=target_path,
            scenario_id=scenario_id,
            artifact_group=group,
            artifact_type="manifest",
            is_large_data=False,
            action="move",
            notes=notes,
        )

    for item in scan_immediate_files(source_root / "audits"):
        if item.name in GLOBAL_MISSINGNESS_AUDITS:
            group = "missingness_setting"
            target_path = missing_root / "audits" / item.name
        elif item.name in GLOBAL_IMPUTATION_AUDITS:
            group = "imputation"
            target_path = imp_root / "audits" / item.name
        else:
            group = "legacy"
            target_path = target_root / "legacy" / "global_mcar_point" / "audits" / item.name
        add_plan_item(
            items,
            aliases,
            project_root,
            source_path=item,
            target_path=target_path,
            scenario_id=scenario_id,
            artifact_group=group,
            artifact_type="audit",
            is_large_data=False,
            action="move",
            notes="Global audit file.",
        )

    for folder_name, target_path, artifact_group, artifact_type in [
        ("summaries", imp_root / "summaries", "imputation", "summary"),
        ("figures", imp_root / "figures", "imputation", "figure"),
    ]:
        for item in scan_immediate_files(source_root / folder_name):
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=target_path / item.name,
                scenario_id=scenario_id,
                artifact_group=artifact_group,
                artifact_type=artifact_type,
                is_large_data=False,
                action="move",
                notes=f"Global {folder_name} artifact.",
            )

    for file_name, target_path, group, artifact_type in [
        ("run_config.json", missing_root / "run_config.json", "missingness_setting", "run_config"),
        ("run_commands.txt", missing_root / "run_commands.txt", "missingness_setting", "run_commands"),
        ("run_config_imputation.json", imp_root / "run_config_imputation.json", "imputation", "run_config"),
        ("run_commands_imputation.txt", imp_root / "run_commands_imputation.txt", "imputation", "run_commands"),
    ]:
        add_plan_item(
            items,
            aliases,
            project_root,
            source_path=source_root / file_name,
            target_path=target_path,
            scenario_id=scenario_id,
            artifact_group=group,
            artifact_type=artifact_type,
            is_large_data=False,
            action="move",
            notes="Global run configuration artifact.",
        )


def build_structured_plan(
    items: list[dict[str, Any]],
    aliases: dict[str, Any],
    project_root: Path,
    target_root: Path,
    source_root: Path,
) -> None:
    block_missing = missingness_root(target_root, BLOCK_SCENARIO_ID)
    block_imp = imputation_root(target_root, BLOCK_SCENARIO_ID)
    outage_missing = missingness_root(target_root, OUTAGE_SCENARIO_ID)
    outage_imp = imputation_root(target_root, OUTAGE_SCENARIO_ID)

    for artifact_dir, artifact_type in [
        ("masks", "mask"),
        ("missing_datasets", "missing_dataset"),
        ("imputed_datasets", "imputed_dataset"),
    ]:
        for item in scan_immediate_dirs(source_root / artifact_dir):
            if item.name.startswith("mechanism_node_temporal_block__rate_"):
                scenario_id = BLOCK_SCENARIO_ID
                target_base = block_missing if artifact_dir != "imputed_datasets" else block_imp
            elif item.name.startswith("mechanism_node_subset_temporal_outage__rate_"):
                scenario_id = OUTAGE_SCENARIO_ID
                target_base = outage_missing if artifact_dir != "imputed_datasets" else outage_imp
            else:
                continue
            target_path = target_base / artifact_dir / item.name
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=target_path,
                scenario_id=scenario_id,
                artifact_group="missingness_setting" if artifact_dir != "imputed_datasets" else "imputation",
                artifact_type=artifact_type,
                is_large_data=True,
                action="move",
                notes="Move structured scenario directory without copying parquet files.",
            )

    for item in scan_immediate_files(source_root / "summaries"):
        if item.name in STRUCTURED_BLOCK_SUMMARIES:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=block_imp / "summaries" / item.name,
                scenario_id=BLOCK_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="summary",
                is_large_data=False,
                action="move",
                notes="Block imputation summary.",
            )
        elif item.name in STRUCTURED_OUTAGE_SUMMARIES:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=outage_imp / "summaries" / item.name,
                scenario_id=OUTAGE_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="summary",
                is_large_data=False,
                action="move",
                notes="Outage imputation summary.",
            )

    for item in scan_immediate_files(source_root / "audits"):
        if item.name in STRUCTURED_SHARED_MISSINGNESS_AUDITS:
            for scenario_id, target_path in [
                (BLOCK_SCENARIO_ID, block_missing / "audits" / item.name),
                (OUTAGE_SCENARIO_ID, outage_missing / "audits" / item.name),
            ]:
                add_plan_item(
                    items,
                    aliases,
                    project_root,
                    source_path=item,
                    target_path=target_path,
                    scenario_id=scenario_id,
                    artifact_group="missingness_setting",
                    artifact_type="audit",
                    is_large_data=False,
                    action="copy",
                    notes="Shared structured missingness audit copied to both structured scenarios.",
                )
        elif item.name in STRUCTURED_BLOCK_AUDITS:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=block_imp / "audits" / item.name,
                scenario_id=BLOCK_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="audit",
                is_large_data=False,
                action="move",
                notes="Block imputation audit.",
            )
        elif item.name in STRUCTURED_OUTAGE_AUDITS:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=outage_imp / "audits" / item.name,
                scenario_id=OUTAGE_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="audit",
                is_large_data=False,
                action="move",
                notes="Outage imputation audit.",
            )

    for item in scan_immediate_files(source_root / "manifests"):
        if item.name in STRUCTURED_SHARED_MISSINGNESS_MANIFESTS:
            for scenario_id, target_path in [
                (BLOCK_SCENARIO_ID, block_missing / "manifests" / item.name),
                (OUTAGE_SCENARIO_ID, outage_missing / "manifests" / item.name),
            ]:
                add_plan_item(
                    items,
                    aliases,
                    project_root,
                    source_path=item,
                    target_path=target_path,
                    scenario_id=scenario_id,
                    artifact_group="missingness_setting",
                    artifact_type="manifest",
                    is_large_data=False,
                    action="copy",
                    notes="Shared structured missingness manifest copied to both structured scenarios.",
                )
        elif item.name in STRUCTURED_BLOCK_IMPUTATION_MANIFESTS:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=block_imp / "manifests" / item.name,
                scenario_id=BLOCK_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="manifest",
                is_large_data=False,
                action="move",
                notes="Block imputation manifest.",
            )
        elif item.name in STRUCTURED_OUTAGE_IMPUTATION_MANIFESTS:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=outage_imp / "manifests" / item.name,
                scenario_id=OUTAGE_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="manifest",
                is_large_data=False,
                action="move",
                notes="Outage imputation manifest.",
            )

    outage_node_lists_dir = source_root / "manifests" / "outage_node_lists"
    if outage_node_lists_dir.exists():
        add_plan_item(
            items,
            aliases,
            project_root,
            source_path=outage_node_lists_dir,
            target_path=outage_missing / "manifests" / "outage_node_lists",
            scenario_id=OUTAGE_SCENARIO_ID,
            artifact_group="missingness_setting",
            artifact_type="manifest",
            is_large_data=False,
            action="move",
            notes="Outage-specific event node lists.",
        )

    for item in scan_immediate_files(source_root / "figures"):
        if item.name.startswith("structured_"):
            if item.name in STRUCTURED_SHARED_MISSINGNESS_FIGURES:
                for scenario_id, target_path in [
                    (BLOCK_SCENARIO_ID, block_missing / "audits" / item.name),
                    (OUTAGE_SCENARIO_ID, outage_missing / "audits" / item.name),
                ]:
                    add_plan_item(
                        items,
                        aliases,
                        project_root,
                        source_path=item,
                        target_path=target_path,
                        scenario_id=scenario_id,
                        artifact_group="missingness_setting",
                        artifact_type="audit",
                        is_large_data=False,
                        action="copy",
                        notes="Shared structured missingness figure copied to both structured scenarios.",
                    )
            else:
                add_plan_item(
                    items,
                    aliases,
                    project_root,
                    source_path=item,
                    target_path=block_imp / "figures" / item.name,
                    scenario_id=BLOCK_SCENARIO_ID,
                    artifact_group="imputation",
                    artifact_type="figure",
                    is_large_data=False,
                    action="move",
                    notes="Block imputation figure.",
                )
        elif item.name.startswith("outage_"):
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=outage_imp / "figures" / item.name,
                scenario_id=OUTAGE_SCENARIO_ID,
                artifact_group="imputation",
                artifact_type="figure",
                is_large_data=False,
                action="move",
                notes="Outage imputation figure.",
            )

    for item_name in STRUCTURED_SHARED_MISC:
        item = source_root / item_name
        for scenario_id, target_path in [
            (BLOCK_SCENARIO_ID, block_missing / "audits" / item.name),
            (OUTAGE_SCENARIO_ID, outage_missing / "audits" / item.name),
        ]:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=target_path,
                scenario_id=scenario_id,
                artifact_group="missingness_setting",
                artifact_type="readme",
                is_large_data=False,
                action="copy",
                notes="Shared structured design/readme copied to both structured scenarios.",
            )

    for file_name in ["run_config.json", "run_commands.txt"]:
        item = source_root / file_name
        artifact_type = "run_config" if file_name.endswith(".json") else "run_commands"
        for scenario_id, target_path in [
            (BLOCK_SCENARIO_ID, block_missing / file_name),
            (OUTAGE_SCENARIO_ID, outage_missing / file_name),
        ]:
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=target_path,
                scenario_id=scenario_id,
                artifact_group="missingness_setting",
                artifact_type=artifact_type,
                is_large_data=False,
                action="copy",
                notes="Shared structured missingness run artifact copied to both structured scenarios.",
            )

    for file_name, scenario_id, target_path in [
        ("run_config_imputation.json", BLOCK_SCENARIO_ID, block_imp / "run_config_imputation.json"),
        ("run_commands_imputation.txt", BLOCK_SCENARIO_ID, block_imp / "run_commands_imputation.txt"),
        ("run_config_imputation_outage.json", OUTAGE_SCENARIO_ID, outage_imp / "run_config_imputation.json"),
        ("run_commands_imputation_outage.txt", OUTAGE_SCENARIO_ID, outage_imp / "run_commands_imputation.txt"),
    ]:
        artifact_type = "run_config" if file_name.endswith(".json") else "run_commands"
        add_plan_item(
            items,
            aliases,
            project_root,
            source_path=source_root / file_name,
            target_path=target_path,
            scenario_id=scenario_id,
            artifact_group="imputation",
            artifact_type=artifact_type,
            is_large_data=False,
            action="move",
            notes="Structured imputation run artifact.",
        )


def build_comparison_plan(
    items: list[dict[str, Any]],
    aliases: dict[str, Any],
    project_root: Path,
    target_root: Path,
    source_root: Path,
) -> None:
    comparison_target = target_root / "comparison"
    for file_name, artifact_type, source_path, target_path, action in [
        (
            "figure_index.csv",
            "table",
            source_root / "figure_index.csv",
            comparison_target / "figure_index.csv",
            "move",
        ),
        (
            "visualization_comparison_audit.json",
            "audit",
            source_root / "audits" / "visualization_comparison_audit.json",
            comparison_target / "visualization_comparison_audit.json",
            "copy",
        ),
        (
            "visualization_comparison_audit_zh.md",
            "audit",
            source_root / "audits" / "visualization_comparison_audit_zh.md",
            comparison_target / "visualization_comparison_audit_zh.md",
            "copy",
        ),
    ]:
        add_plan_item(
            items,
            aliases,
            project_root,
            source_path=source_path,
            target_path=target_path,
            scenario_id="comparison",
            artifact_group="comparison",
            artifact_type=artifact_type,
            is_large_data=False,
            action=action,
            notes="Comparison root artifact.",
        )
    for folder_name, artifact_type in [
        ("figures", "figure"),
        ("tables", "table"),
        ("audits", "audit"),
    ]:
        for item in scan_immediate_files(source_root / folder_name):
            add_plan_item(
                items,
                aliases,
                project_root,
                source_path=item,
                target_path=comparison_target / folder_name / item.name,
                scenario_id="comparison",
                artifact_group="comparison",
                artifact_type=artifact_type,
                is_large_data=False,
                action="move",
                notes="Comparison artifact.",
            )


def generate_registry(project_root: Path, target_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario_id in SCENARIO_IDS:
        meta = SCENARIO_META[scenario_id]
        scenario_dir = scenario_root(target_root, scenario_id)
        missing_dir = missingness_root(target_root, scenario_id)
        imp_dir = imputation_root(target_root, scenario_id)
        if scenario_id == GLOBAL_SCENARIO_ID:
            summary_main = imp_dir / "summaries" / "imputation_quality_summary_exclude_warmup.csv"
            summary_by_flow = imp_dir / "summaries" / "imputation_quality_by_flow_group.csv"
            summary_by_length = None
            audit_missingness = missing_dir / "audits" / "global_missingness_setting_audit.json"
            audit_imputation = imp_dir / "audits" / "causal_imputation_audit.json"
        elif scenario_id == BLOCK_SCENARIO_ID:
            summary_main = imp_dir / "summaries" / "structured_imputation_quality_summary_exclude_warmup.csv"
            summary_by_flow = imp_dir / "summaries" / "structured_imputation_quality_by_flow_group.csv"
            summary_by_length = imp_dir / "summaries" / "structured_imputation_quality_by_length_group.csv"
            audit_missingness = missing_dir / "audits" / "structured_missingness_audit.json"
            audit_imputation = imp_dir / "audits" / "structured_causal_imputation_audit.json"
        else:
            summary_main = imp_dir / "summaries" / "outage_imputation_quality_summary_exclude_warmup.csv"
            summary_by_flow = imp_dir / "summaries" / "outage_imputation_quality_by_flow_group.csv"
            summary_by_length = imp_dir / "summaries" / "outage_imputation_quality_by_length_group.csv"
            audit_missingness = missing_dir / "audits" / "structured_missingness_audit.json"
            audit_imputation = imp_dir / "audits" / "outage_causal_imputation_audit.json"
        record = {
            "scenario_id": scenario_id,
            "display_name": meta["display_name"],
            "display_name_zh": meta["display_name_zh"],
            "missingness_type": meta["missingness_type"],
            "mechanism": meta["mechanism"],
            "scenario_tag": meta["scenario_tag"],
            "missing_rates": meta["missing_rates"],
            "seed": meta["seed"],
            "input_data": meta["input_data"],
            "target_col": meta["target_col"],
            "node_col": meta["node_col"],
            "time_col": meta["time_col"],
            "period": meta["period"],
            "missingness_setting_dir": normalize_relative(missing_dir, project_root),
            "imputation_dir": normalize_relative(imp_dir, project_root),
            "summary_main": normalize_relative(summary_main, project_root),
            "summary_by_flow_group": normalize_relative(summary_by_flow, project_root),
            "summary_by_length_group": None if summary_by_length is None else normalize_relative(summary_by_length, project_root),
            "audit_missingness": normalize_relative(audit_missingness, project_root),
            "audit_imputation": normalize_relative(audit_imputation, project_root),
            "figure_dir": normalize_relative(imp_dir / "figures", project_root),
            "status": "ready" if scenario_dir.exists() else "planned",
            "notes": "Current results are masked-position imputation errors, not traffic forecasting errors.",
        }
        records.append(record)
    return records


def write_registry(project_root: Path, target_root: Path) -> list[dict[str, Any]]:
    records = generate_registry(project_root, target_root)
    registry_df = pd.DataFrame(records)
    registry_csv_path = target_root / "experiment_registry.csv"
    registry_json_path = target_root / "experiment_registry.json"
    ensure_dir(target_root)
    registry_df.to_csv(registry_csv_path, index=False, encoding="utf-8-sig")
    write_json(registry_json_path, {"records": records})
    return records


def write_readme(project_root: Path, target_root: Path) -> None:
    readme_path = target_root / "README_zh.md"
    lines = [
        "# 真实数据缺失实验统一结果目录",
        "",
        "1. 本目录统一管理三类真实数据缺失与补全实验。",
        "2. global_mcar_point 是完整数据全局 MCAR 点级随机缺失。",
        "3. node_temporal_block_mixed_short_mid_long 是单节点连续时间块缺失，长度为 short/mid/long 混合。",
        "4. node_subset_temporal_outage_mixed_short_mid_long 是节点子集连续离线缺失，长度为 short/mid/long 混合。",
        "5. 每个 scenario 下分为 missingness_setting 和 imputation。",
        "6. missingness_setting 存放 masks、missing_datasets、缺失设置 audit 和 manifest。",
        "7. imputation 存放 imputed_datasets、summary、figures、补全 audit 和 manifest。",
        "8. comparison 存放三类机制综合对比图。",
        "9. parquet 大文件不进入 Git。",
        "10. 当前结果是缺失值补全误差，不是交通流预测误差。",
        "",
        "## Scenario IDs",
        "",
        f"- `{GLOBAL_SCENARIO_ID}`",
        f"- `{BLOCK_SCENARIO_ID}`",
        f"- `{OUTAGE_SCENARIO_ID}`",
        "",
        "## 路径索引",
        "",
        f"- `experiment_registry.json`: `{normalize_relative(target_root / 'experiment_registry.json', project_root)}`",
        f"- `path_aliases.json`: `{normalize_relative(target_root / 'path_aliases.json', project_root)}`",
        f"- `comparison`: `{normalize_relative(target_root / 'comparison', project_root)}`",
    ]
    write_markdown(readme_path, lines)


def execute_action(source_path: Path, target_path: Path, action: str) -> str:
    if action not in {"move", "copy"}:
        return "skipped_invalid_action"
    if target_path.exists():
        return "target_exists"
    if (
        action == "copy"
        and not source_path.exists()
        and target_path.parent.name == "comparison"
        and target_path.parent.parent.name == "real_data_missingness_experiments"
    ):
        alias_source = target_path.parent / "audits" / target_path.name
        if alias_source.exists():
            ensure_dir(target_path.parent)
            shutil.copy2(alias_source, target_path)
            return "copied"
    if not source_path.exists():
        return "source_missing"
    ensure_dir(target_path.parent)
    if action == "move":
        shutil.move(str(source_path), str(target_path))
        return "moved"
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    else:
        shutil.copy2(source_path, target_path)
    return "copied"


def write_plan_outputs(
    project_root: Path,
    target_root: Path,
    plan_items: list[dict[str, Any]],
    report_status: str,
) -> None:
    ensure_dir(target_root)
    plan_df = pd.DataFrame(plan_items)
    plan_csv_path = target_root / "layout_migration_plan.csv"
    plan_md_path = target_root / "layout_migration_plan_zh.md"
    plan_df.to_csv(plan_csv_path, index=False, encoding="utf-8-sig")

    counter = Counter(item["action"] for item in plan_items)
    status_counter = Counter(item["status"] for item in plan_items)
    lines = [
        "# 缺失实验结果目录迁移计划",
        "",
        f"- 计划状态：`{report_status}`",
        f"- 计划条目数：`{len(plan_items)}`",
        f"- move 条目数：`{counter.get('move', 0)}`",
        f"- copy 条目数：`{counter.get('copy', 0)}`",
        f"- source_missing 条目数：`{status_counter.get('source_missing', 0)}`",
        "",
        "## 说明",
        "",
        "- `is_large_data = true` 代表 masks、missing_datasets、imputed_datasets 或 parquet 大文件目录。",
        "- dry_run 阶段只生成计划，不实际移动文件。",
        "- shared structured 审计与 manifest 会复制到 block/outage 两个 scenario 中。",
    ]
    write_markdown(plan_md_path, lines)


def write_migration_report(
    project_root: Path,
    target_root: Path,
    plan_items: list[dict[str, Any]],
    validation_summary: dict[str, Any] | None,
    source_roots: dict[str, Path],
) -> None:
    moved = [item for item in plan_items if item["status"] in {"moved", "copied", "already_migrated", "target_exists"}]
    skipped = [item for item in plan_items if item["status"] in {"source_missing", "skipped_invalid_action"}]
    large_items = [item for item in moved if item["is_large_data"]]
    old_dirs = {
        key: normalize_relative(path, project_root)
        for key, path in source_roots.items()
        if key.endswith("_source")
    }
    payload = {
        "purpose": "Reorganize completed missingness experiment results under a unified root for writing, visualization lookup, and Git hygiene.",
        "old_directories": old_dirs,
        "new_root": normalize_relative(target_root, project_root),
        "scenario_ids": SCENARIO_IDS,
        "moved_or_resolved_items": len(moved),
        "large_data_items": len(large_items),
        "skipped_items": len(skipped),
        "kept_legacy_readmes": True,
        "generated_experiment_registry": (target_root / "experiment_registry.json").exists(),
        "generated_path_aliases": (target_root / "path_aliases.json").exists(),
        "validation_passed": None if validation_summary is None else bool(validation_summary.get("all_complete")),
        "follow_up_recommendation": "Use results\\real_data_missingness_experiments as the primary root for future lookup and lightweight Git operations.",
        "shared_structured_note": "Shared structured missingness audits/manifests were copied into both structured scenarios because the original source covered the whole structured root.",
        "missing_or_skipped_items": skipped,
    }
    write_json(target_root / "layout_migration_report.json", payload)

    lines = [
        "# 缺失实验结果目录迁移报告",
        "",
        "## 迁移目的",
        "",
        "- 统一整理三类缺失机制的结果目录，便于论文写作、可视化检索和 Git 管理。",
        "",
        "## 旧目录",
        "",
    ]
    for label, path in old_dirs.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## 新目录",
            "",
            f"- `{normalize_relative(target_root, project_root)}`",
            "",
            "## 三类机制的新 scenario_id",
            "",
            f"- `{GLOBAL_SCENARIO_ID}`",
            f"- `{BLOCK_SCENARIO_ID}`",
            f"- `{OUTAGE_SCENARIO_ID}`",
            "",
            "## 迁移结果",
            "",
            f"- 已移动或已解析条目数：`{len(moved)}`",
            f"- 大体积 parquet 相关目录条目数：`{len(large_items)}`",
            f"- 缺失或跳过条目数：`{len(skipped)}`",
            "",
            "## 说明",
            "",
            "- 大体积 parquet 目录只在文件系统层面移动，不应提交到 Git。",
            "- shared structured 审计与 manifest 采用复制方式放入 block/outage 两个 scenario。",
            "- 旧目录保留 `MIGRATED_TO_README.md` 指向新根目录。",
            f"- 是否生成 experiment_registry：`{'是' if payload['generated_experiment_registry'] else '否'}`",
            f"- 是否生成 path_aliases：`{'是' if payload['generated_path_aliases'] else '否'}`",
            f"- 是否验证通过：`{'是' if payload['validation_passed'] else '否' if payload['validation_passed'] is not None else '未执行 validate'}`",
            "",
            "## 建议",
            "",
            "- 后续脚本、可视化和论文表格优先使用新 root 或通过 `experiment_registry.json` / `path_aliases.json` 查找路径。",
        ]
    )
    write_markdown(target_root / "layout_migration_report.md", lines)


def build_validation_rows(project_root: Path, target_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(
        scenario_id: str,
        artifact_group: str,
        artifact_type: str,
        expected_count: int,
        observed_count: int,
        path: Path,
        notes: str,
    ) -> None:
        rows.append(
            {
                "scenario_id": scenario_id,
                "artifact_group": artifact_group,
                "artifact_type": artifact_type,
                "expected_count": expected_count,
                "observed_count": observed_count,
                "is_complete": bool(observed_count >= expected_count),
                "path": normalize_relative(path, project_root),
                "notes": notes,
            }
        )

    for scenario_id in SCENARIO_IDS:
        miss_root = missingness_root(target_root, scenario_id)
        imp_root = imputation_root(target_root, scenario_id)
        add_row(scenario_id, "missingness_setting", "directory", 1, int(miss_root.exists()), miss_root, "Scenario missingness_setting directory.")
        add_row(scenario_id, "imputation", "directory", 1, int(imp_root.exists()), imp_root, "Scenario imputation directory.")

        if scenario_id == GLOBAL_SCENARIO_ID:
            summary_main = imp_root / "summaries" / "imputation_quality_summary_exclude_warmup.csv"
            summary_flow = imp_root / "summaries" / "imputation_quality_by_flow_group.csv"
            summary_length = None
            mask_root = miss_root / "masks"
            missing_root_dir = miss_root / "missing_datasets"
            imputed_root_dir = imp_root / "imputed_datasets"
            rate_dirs_masks = scan_immediate_dirs(mask_root)
            rate_dirs_missing = scan_immediate_dirs(missing_root_dir)
            add_row(scenario_id, "missingness_setting", "mask_rate_dirs", 4, len(rate_dirs_masks), mask_root, "Expected 4 global mask rate directories.")
            add_row(scenario_id, "missingness_setting", "missing_dataset_rate_dirs", 4, len(rate_dirs_missing), missing_root_dir, "Expected 4 global missing dataset rate directories.")
            for rate_code in EXPECTED_RATES:
                mask_dir = next((item for item in rate_dirs_masks if f"rate_{rate_code}" in item.name), mask_root / f"rate_{rate_code}")
                miss_dir = next((item for item in rate_dirs_missing if f"rate_{rate_code}" in item.name), missing_root_dir / f"rate_{rate_code}")
                imputed_rate_dirs = [item for item in scan_immediate_dirs(imputed_root_dir) if f"rate_{rate_code}" in item.name]
                add_row(scenario_id, "missingness_setting", f"mask_chunks_{rate_code}", 61, len(list(mask_dir.glob("*.parquet"))), mask_dir, "Expected 61 mask parquet chunks.")
                add_row(scenario_id, "missingness_setting", f"missing_dataset_chunks_{rate_code}", 61, len(list(miss_dir.glob("*.parquet"))), miss_dir, "Expected 61 missing-dataset parquet chunks.")
                add_row(scenario_id, "imputation", f"imputed_rate_method_dirs_{rate_code}", 6, len(imputed_rate_dirs), imputed_root_dir, "Expected 6 method directories per rate.")
                for method_dir in imputed_rate_dirs:
                    add_row(
                        scenario_id,
                        "imputation",
                        f"imputed_chunks_{rate_code}_{method_dir.name}",
                        61,
                        len(list(method_dir.glob("*.parquet"))),
                        method_dir,
                        "Expected 61 imputed parquet chunks per method.",
                    )
        else:
            if scenario_id == BLOCK_SCENARIO_ID:
                prefix = "mechanism_node_temporal_block__rate_"
                summary_main = imp_root / "summaries" / "structured_imputation_quality_summary_exclude_warmup.csv"
                summary_flow = imp_root / "summaries" / "structured_imputation_quality_by_flow_group.csv"
                summary_length = imp_root / "summaries" / "structured_imputation_quality_by_length_group.csv"
            else:
                prefix = "mechanism_node_subset_temporal_outage__rate_"
                summary_main = imp_root / "summaries" / "outage_imputation_quality_summary_exclude_warmup.csv"
                summary_flow = imp_root / "summaries" / "outage_imputation_quality_by_flow_group.csv"
                summary_length = imp_root / "summaries" / "outage_imputation_quality_by_length_group.csv"
            mask_root = miss_root / "masks"
            missing_root_dir = miss_root / "missing_datasets"
            imputed_root_dir = imp_root / "imputed_datasets"
            rate_dirs_masks = [item for item in scan_immediate_dirs(mask_root) if item.name.startswith(prefix)]
            rate_dirs_missing = [item for item in scan_immediate_dirs(missing_root_dir) if item.name.startswith(prefix)]
            rate_dirs_imputed = [item for item in scan_immediate_dirs(imputed_root_dir) if item.name.startswith(prefix)]
            add_row(scenario_id, "missingness_setting", "mask_rate_dirs", 4, len({dir_name.split("__seed_")[0] for dir_name in [item.name for item in rate_dirs_masks]}), mask_root, "Expected 4 structured mask rate directories.")
            add_row(scenario_id, "missingness_setting", "missing_dataset_rate_dirs", 4, len({dir_name.split("__seed_")[0] for dir_name in [item.name for item in rate_dirs_missing]}), missing_root_dir, "Expected 4 structured missing-dataset rate directories.")
            add_row(scenario_id, "imputation", "imputed_rate_method_dirs", 24, len(rate_dirs_imputed), imputed_root_dir, "Expected 24 structured imputed method directories.")
            for rate_code in EXPECTED_RATES:
                mask_dir = next((item for item in rate_dirs_masks if f"rate_{rate_code}" in item.name), mask_root / f"{prefix}{rate_code}")
                miss_dir = next((item for item in rate_dirs_missing if f"rate_{rate_code}" in item.name), missing_root_dir / f"{prefix}{rate_code}")
                imputed_for_rate = [item for item in rate_dirs_imputed if f"rate_{rate_code}" in item.name]
                add_row(scenario_id, "missingness_setting", f"mask_chunks_{rate_code}", 61, len(list(mask_dir.glob("*.parquet"))), mask_dir, "Expected 61 mask parquet chunks.")
                add_row(scenario_id, "missingness_setting", f"missing_dataset_chunks_{rate_code}", 61, len(list(miss_dir.glob("*.parquet"))), miss_dir, "Expected 61 missing-dataset parquet chunks.")
                add_row(scenario_id, "imputation", f"imputed_rate_method_dirs_{rate_code}", 6, len(imputed_for_rate), imputed_root_dir, "Expected 6 imputed method directories per rate.")
                for method_dir in imputed_for_rate:
                    add_row(
                        scenario_id,
                        "imputation",
                        f"imputed_chunks_{rate_code}_{method_dir.name}",
                        61,
                        len(list(method_dir.glob("*.parquet"))),
                        method_dir,
                        "Expected 61 imputed parquet chunks per method.",
                    )

        add_row(scenario_id, "imputation", "summary_main", 1, int(summary_main.exists()), summary_main, "Main summary file must exist.")
        add_row(scenario_id, "imputation", "summary_by_flow_group", 1, int(summary_flow.exists()), summary_flow, "Flow-group summary file must exist.")
        if summary_length is not None:
            add_row(scenario_id, "imputation", "summary_by_length_group", 1, int(summary_length.exists()), summary_length, "Length-group summary file must exist.")

    comparison_dir = target_root / "comparison"
    add_row("comparison", "comparison", "directory", 1, int(comparison_dir.exists()), comparison_dir, "Comparison root must exist.")
    add_row(
        "comparison",
        "comparison",
        "figure_index",
        1,
        int((comparison_dir / "figure_index.csv").exists()),
        comparison_dir / "figure_index.csv",
        "Comparison figure_index.csv must exist.",
    )
    add_row(
        "shared",
        "shared",
        "path_aliases",
        1,
        int((target_root / "path_aliases.json").exists()),
        target_root / "path_aliases.json",
        "path_aliases.json must exist.",
    )
    add_row(
        "shared",
        "shared",
        "readme",
        1,
        int((target_root / "README_zh.md").exists()),
        target_root / "README_zh.md",
        "README_zh.md must exist.",
    )
    all_complete = all(bool(row["is_complete"]) for row in rows)
    summary = {"all_complete": all_complete, "row_count": len(rows)}
    return rows, summary


def write_validation_outputs(project_root: Path, target_root: Path) -> dict[str, Any]:
    rows, summary = build_validation_rows(project_root, target_root)
    validation_df = pd.DataFrame(rows)
    validation_df.to_csv(target_root / "layout_validation.csv", index=False, encoding="utf-8-sig")
    write_json(target_root / "layout_validation.json", {"summary": summary, "rows": rows})
    return summary


def write_path_aliases(target_root: Path, aliases: dict[str, Any]) -> None:
    write_json(target_root / "path_aliases.json", aliases)


def create_legacy_readmes(project_root: Path, target_root: Path, source_roots: dict[str, Path]) -> None:
    for key, source_root in source_roots.items():
        if not key.endswith("_source"):
            continue
        if source_root.exists():
            write_text(source_root / "MIGRATED_TO_README.md", README_TEXT)


def run_inventory(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan_items, aliases, metadata = build_plan(args)
    target_root = metadata["roots"]["target_root"]
    project_root = args.project_root.resolve()
    ensure_dir(target_root)
    write_path_aliases(target_root, aliases)
    write_registry(project_root, target_root)
    write_readme(project_root, target_root)
    return plan_items, aliases, metadata


def run_dry_run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan_items, aliases, metadata = run_inventory(args)
    write_plan_outputs(args.project_root.resolve(), metadata["roots"]["target_root"], plan_items, "dry_run")
    write_migration_report(
        args.project_root.resolve(),
        metadata["roots"]["target_root"],
        plan_items,
        validation_summary=None,
        source_roots=metadata["roots"],
    )
    return plan_items, aliases, metadata


def run_migrate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan_items, aliases, metadata = run_inventory(args)
    for item in plan_items:
        source_path = args.project_root.resolve() / Path(item["source_path"])
        target_path = args.project_root.resolve() / Path(item["target_path"])
        item["status"] = execute_action(source_path, target_path, item["action"])
    create_legacy_readmes(args.project_root.resolve(), metadata["roots"]["target_root"], metadata["roots"])
    write_path_aliases(metadata["roots"]["target_root"], aliases)
    write_registry(args.project_root.resolve(), metadata["roots"]["target_root"])
    write_readme(args.project_root.resolve(), metadata["roots"]["target_root"])
    write_plan_outputs(args.project_root.resolve(), metadata["roots"]["target_root"], plan_items, "migrated")
    return plan_items, aliases, metadata


def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    target_root = (project_root / args.target_root).resolve()
    return write_validation_outputs(project_root, target_root)


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    if not args.target_root.is_absolute():
        args.target_root = args.project_root / args.target_root
    if not args.global_source.is_absolute():
        args.global_source = args.project_root / args.global_source
    if not args.structured_source.is_absolute():
        args.structured_source = args.project_root / args.structured_source
    if not args.comparison_source.is_absolute():
        args.comparison_source = args.project_root / args.comparison_source

    if args.stage == "inventory":
        run_inventory(args)
        return
    if args.stage == "dry_run":
        run_dry_run(args)
        return
    if args.stage == "migrate":
        plan_items, _, metadata = run_migrate(args)
        write_migration_report(
            args.project_root.resolve(),
            metadata["roots"]["target_root"],
            plan_items,
            validation_summary=None,
            source_roots=metadata["roots"],
        )
        return
    if args.stage == "validate":
        run_validate(args)
        return
    if args.stage == "all":
        plan_items, _, metadata = run_dry_run(args)
        plan_items, _, metadata = run_migrate(args)
        validation_summary = run_validate(args)
        write_migration_report(
            args.project_root.resolve(),
            metadata["roots"]["target_root"],
            plan_items,
            validation_summary=validation_summary,
            source_roots=metadata["roots"],
        )


if __name__ == "__main__":
    main()
