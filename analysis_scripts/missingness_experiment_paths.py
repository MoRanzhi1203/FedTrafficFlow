from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENT_ROOT = Path("results/real_data_missingness_experiments")
SCENARIO_IDS = [
    "global_mcar_point",
    "node_temporal_block_mixed_short_mid_long",
    "node_subset_temporal_outage_mixed_short_mid_long",
]


def _project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_root(project_root: Path | str | None = None) -> Path:
    if project_root is None:
        return _project_root_from_file()
    return Path(project_root).resolve()


def _ensure_experiment_root(
    experiment_root: Path | str | None = None,
    project_root: Path | str | None = None,
) -> Path:
    if experiment_root is not None:
        return Path(experiment_root).resolve()
    return (_ensure_project_root(project_root) / DEFAULT_EXPERIMENT_ROOT).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required json file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_experiment_root(
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path:
    return _ensure_experiment_root(experiment_root=experiment_root, project_root=project_root)


def get_scenario_dir(
    scenario_id: str,
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path:
    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"unsupported scenario_id: {scenario_id}")
    root = get_experiment_root(project_root=project_root, experiment_root=experiment_root)
    return root / "scenarios" / scenario_id


def get_missingness_setting_dir(
    scenario_id: str,
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path:
    return get_scenario_dir(
        scenario_id=scenario_id,
        project_root=project_root,
        experiment_root=experiment_root,
    ) / "missingness_setting"


def get_imputation_dir(
    scenario_id: str,
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path:
    return get_scenario_dir(
        scenario_id=scenario_id,
        project_root=project_root,
        experiment_root=experiment_root,
    ) / "imputation"


def get_comparison_dir(
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path:
    root = get_experiment_root(project_root=project_root, experiment_root=experiment_root)
    return root / "comparison"


def load_experiment_registry(
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    root = get_experiment_root(project_root=project_root, experiment_root=experiment_root)
    registry_path = root / "experiment_registry.json"
    payload = _load_json(registry_path)
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    else:
        records = payload
    if not isinstance(records, list):
        raise RuntimeError(f"experiment registry has unexpected shape: {registry_path}")
    return records


def get_summary_path(
    scenario_id: str,
    summary_type: str,
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path | None:
    records = load_experiment_registry(project_root=project_root, experiment_root=experiment_root)
    match = next((item for item in records if item.get("scenario_id") == scenario_id), None)
    if match is None:
        raise KeyError(f"scenario_id not found in experiment registry: {scenario_id}")
    summary_key = {
        "main": "summary_main",
        "by_flow_group": "summary_by_flow_group",
        "by_length_group": "summary_by_length_group",
    }.get(summary_type, summary_type)
    value = match.get(summary_key)
    if value in {None, "", "null"}:
        return None
    root = _ensure_project_root(project_root)
    return (root / Path(value)).resolve()


def load_path_aliases(
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> dict[str, str]:
    root = get_experiment_root(project_root=project_root, experiment_root=experiment_root)
    return _load_json(root / "path_aliases.json")


def resolve_legacy_path(
    old_path: str | Path,
    project_root: Path | str | None = None,
    experiment_root: Path | str | None = None,
) -> Path:
    root = _ensure_project_root(project_root)
    aliases = load_path_aliases(project_root=project_root, experiment_root=experiment_root)
    raw_old = str(old_path).replace("/", "\\")
    mapped = aliases.get(raw_old)
    if mapped is None:
        try:
            relative_old = str(Path(old_path).resolve().relative_to(root)).replace("/", "\\")
        except Exception:
            relative_old = raw_old
        mapped = aliases.get(relative_old, raw_old)
    mapped_path = Path(mapped)
    if mapped_path.is_absolute():
        return mapped_path.resolve()
    return (root / mapped_path).resolve()
