from __future__ import annotations

import argparse
import json
import math
import shlex
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency fallback
    pq = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "real_data_missingness_full_intersection_causal_history"
DEFAULT_INPUT_PATTERN = "node_flow_chunk_*.parquet"
DEFAULT_TOPOLOGY_PATH = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"
DEFAULT_TARGET_CANDIDATES = ["路口车流量", "flow", "traffic_flow", "target", "y"]
DEFAULT_TIME_CANDIDATES = ["时间段", "timestamp", "time", "datetime", "date", "day_slot"]
DEFAULT_NODE_CANDIDATES = ["节点ID", "node_id", "sensor_id", "detector_id", "station_id"]
DEFAULT_MISSING_RATES = [0.05]
DEFAULT_IMPUTE_METHODS = [
    "zero_fill",
    "forward_fill",
    "historical_linear_extrapolation",
    "geo_neighbor_fill",
    "function_curve_fit",
    "geo_func_hybrid",
]
METHOD_LABEL_MAP = {
    "zero_fill": "Zero fill",
    "forward_fill": "Forward fill",
    "historical_linear_extrapolation": "Historical linear trend",
    "linear_interpolation": "Historical linear trend",
    "geo_neighbor_fill": "Historical geo-neighbor",
    "function_curve_fit": "Historical function curve",
    "geo_func_hybrid": "Historical geo-function hybrid",
}


@dataclass
class ChunkMetaRecord:
    day_index: int
    file_name: str
    file_path: str
    rows_used: int
    node_count: int
    time_slot_count: int
    time_slot_min: int
    time_slot_max: int
    warmup: bool


@dataclass
class MissingManifestRecord:
    missing_rate: float
    mechanism: str
    seed: int
    day_index: int
    file_name: str
    mask_path: str
    missing_dataset_path: Optional[str]
    actual_missing_count: int
    actual_missing_rate: float


@dataclass
class ImputationManifestRecord:
    missing_rate: float
    mechanism: str
    seed: int
    impute_method: str
    day_index: int
    file_name: str
    imputed_dataset_path: Optional[str]
    filled_missing_count: int
    residual_missing_count: int
    detail_path: str


@dataclass
class ChunkProcessStatusRecord:
    stage: str
    missing_rate: float
    mechanism: str
    seed: int
    impute_method: str
    day_index: int
    file_name: str
    status: str
    mask_exists: bool
    missing_dataset_exists: bool
    imputed_dataset_exists: bool
    detail_exists: bool
    actual_missing_count: int
    filled_missing_count: int
    residual_missing_count: int
    note: str


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完整路口阶段真实数据缺失构造与历史因果补全流水线。"
    )
    parser.add_argument("--stage", required=True, choices=["generate_missing", "impute", "validate", "summarize"])
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input_pattern", type=str, default=DEFAULT_INPUT_PATTERN)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--node_col", type=str, default=None)
    parser.add_argument("--missing_rates", type=str, default=",".join(str(x) for x in DEFAULT_MISSING_RATES))
    parser.add_argument("--mechanism", type=str, default="mcar_point")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument(
        "--impute_methods",
        type=str,
        default=",".join(DEFAULT_IMPUTE_METHODS),
    )
    parser.add_argument("--max_chunks", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0, help="调试用，可限制单个 chunk 读取行数。")
    parser.add_argument("--write_missing_datasets", action="store_true", default=False)
    parser.add_argument("--write_imputed_datasets", action="store_true", default=False)
    parser.add_argument("--save_masks", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--topology_path", type=Path, default=DEFAULT_TOPOLOGY_PATH)
    parser.add_argument("--geo_lambda", type=float, default=0.5)
    parser.add_argument("--period", type=int, default=96)
    parser.add_argument("--fourier_order", type=int, default=2)
    parser.add_argument("--min_fit_points", type=int, default=8)
    parser.add_argument("--historical_trend_points", type=int, default=8)
    parser.add_argument("--block_lengths", type=str, default="4,8,12")
    parser.add_argument("--causal_history_only", action="store_true", default=False)
    parser.add_argument("--history_days", type=int, default=7)
    parser.add_argument("--allow_current_day_past", type=str2bool, default=True)
    parser.add_argument("--context_days_before", type=int, default=7)
    parser.add_argument("--context_days_after", type=int, default=0)
    parser.add_argument("--exclude_warmup_from_main_metrics", type=str2bool, default=True)
    parser.add_argument("--warmup_days", type=int, default=7)
    parser.add_argument("--clip_nonnegative", type=str2bool, default=True)
    parser.add_argument("--node_quantile_cap", type=float, default=0.95)
    parser.add_argument("--node_quantile_cap_multiplier", type=float, default=1.5)
    return parser.parse_args()


def parse_float_list(raw_text: str) -> list[float]:
    return [float(item.strip()) for item in str(raw_text).split(",") if item.strip()]


def parse_int_list(raw_text: str) -> list[int]:
    return [int(item.strip()) for item in str(raw_text).split(",") if item.strip()]


def parse_str_list(raw_text: str) -> list[str]:
    return [item.strip() for item in str(raw_text).split(",") if item.strip()]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def find_first_existing_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    existing = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in existing:
            return existing[candidate.lower()]
    for column in columns:
        lowered = str(column).lower()
        for candidate in candidates:
            if candidate.lower() in lowered:
                return str(column)
    return None


def resolve_columns(
    columns: list[str],
    target_col: Optional[str],
    time_col: Optional[str],
    node_col: Optional[str],
) -> tuple[str, str, str]:
    resolved_target = target_col or find_first_existing_column(columns, DEFAULT_TARGET_CANDIDATES)
    resolved_time = time_col or find_first_existing_column(columns, DEFAULT_TIME_CANDIDATES)
    resolved_node = node_col or find_first_existing_column(columns, DEFAULT_NODE_CANDIDATES)
    if resolved_target is None or resolved_time is None or resolved_node is None:
        raise ValueError("未能自动识别 target_col/time_col/node_col，请显式传参。")
    return resolved_target, resolved_time, resolved_node


def list_input_files(input_dir: Path, pattern: str, max_chunks: int) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到输入文件: {input_dir / pattern}")
    if max_chunks > 0:
        return files[:max_chunks]
    return files


def get_parquet_columns(file_path: Path) -> list[str]:
    if pq is not None:
        return list(pq.ParquetFile(file_path).schema.names)
    return list(pd.read_parquet(file_path, engine="auto").columns)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT_DIR / args.input_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir
    args.topology_path = args.topology_path if args.topology_path.is_absolute() else ROOT_DIR / args.topology_path
    ensure_directory(args.output_dir)
    if args.causal_history_only and args.context_days_after != 0:
        raise ValueError("causal_history_only 模式下必须强制 context_days_after=0。")
    if args.context_days_after > 0:
        raise ValueError("当前流水线仅允许 context_days_after=0。")
    if args.causal_history_only and args.allow_current_day_past not in {True, False}:
        raise ValueError("allow_current_day_past 必须为布尔值。")
    return args


def normalize_impute_methods(impute_methods: list[str], causal_history_only: bool) -> list[str]:
    normalized: list[str] = []
    for method in impute_methods:
        current = method
        if causal_history_only and method == "linear_interpolation":
            current = "historical_linear_extrapolation"
        if current not in normalized:
            normalized.append(current)
    return normalized


def load_neighbor_edges(topology_path: Path) -> pd.DataFrame:
    if not topology_path.exists():
        return pd.DataFrame(columns=["node_id", "neighbor_id", "weight"])
    df = pd.read_csv(topology_path, encoding="utf-8")
    columns_lower = {str(c).lower(): str(c) for c in df.columns}
    src_col = columns_lower.get("起始节点id", columns_lower.get("snodeid"))
    dst_col = columns_lower.get("结束节点id", columns_lower.get("enodeid"))
    len_col = columns_lower.get("长度", columns_lower.get("length"))
    if src_col is None or dst_col is None:
        return pd.DataFrame(columns=["node_id", "neighbor_id", "weight"])
    edges: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        src = row[src_col]
        dst = row[dst_col]
        if pd.isna(src) or pd.isna(dst):
            continue
        src_text = str(int(src)) if isinstance(src, (int, float)) else str(src)
        dst_text = str(int(dst)) if isinstance(dst, (int, float)) else str(dst)
        if src_text == dst_text:
            continue
        weight = 1.0
        if len_col is not None and pd.notna(row[len_col]):
            try:
                weight = 1.0 / max(float(row[len_col]), 1e-6)
            except Exception:
                weight = 1.0
        edges.append({"node_id": src_text, "neighbor_id": dst_text, "weight": weight})
        edges.append({"node_id": dst_text, "neighbor_id": src_text, "weight": weight})
    if not edges:
        return pd.DataFrame(columns=["node_id", "neighbor_id", "weight"])
    edge_df = pd.DataFrame(edges)
    return edge_df.groupby(["node_id", "neighbor_id"], as_index=False)["weight"].mean()


def stable_seed(base_seed: int, file_name: str, mechanism: str, missing_rate: float) -> int:
    payload = f"{file_name}|{mechanism}|{missing_rate:.4f}".encode("utf-8")
    return int(base_seed + zlib.adler32(payload)) % (2 ** 31 - 1)


def build_time_slot_mapping(series: pd.Series) -> dict[Any, int]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        ordered = sorted(int(x) for x in numeric.dropna().unique().tolist())
        if ordered == list(range(len(ordered))):
            return {value: int(value) for value in ordered}
        return {value: idx for idx, value in enumerate(ordered)}
    ordered_text = sorted(str(x) for x in series.dropna().astype(str).unique().tolist())
    return {value: idx for idx, value in enumerate(ordered_text)}


def read_chunk_frame(
    file_path: Path,
    day_index: int,
    target_col: str,
    time_col: str,
    node_col: str,
    period: int,
    warmup_days: int,
    max_rows: int,
) -> tuple[pd.DataFrame, ChunkMetaRecord]:
    df = pd.read_parquet(file_path, columns=[node_col, time_col, target_col])
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()
    else:
        df = df.copy()
    slot_mapping = build_time_slot_mapping(df[time_col])
    mapped_slots = pd.to_numeric(df[time_col], errors="coerce")
    if mapped_slots.notna().all() and set(int(x) for x in mapped_slots.unique()) <= set(range(period)):
        df["time_slot"] = mapped_slots.astype(int)
    else:
        as_text = df[time_col].astype(str)
        slot_mapping_text = {str(key): value for key, value in slot_mapping.items()}
        df["time_slot"] = as_text.map(slot_mapping_text)
        if df["time_slot"].isna().any():
            unresolved = sorted(as_text.loc[df["time_slot"].isna()].unique().tolist())[:10]
            raise ValueError(f"无法为时间段字段映射 time_slot，示例值: {unresolved}")
        df["time_slot"] = df["time_slot"].astype(int)
    df[node_col] = df[node_col].astype(str)
    df["day_index"] = int(day_index)
    df["global_time_index"] = df["day_index"] * int(period) + df["time_slot"]
    df["source_chunk_name"] = file_path.name
    df["source_chunk_path"] = get_relative_path(file_path)
    df["is_warmup"] = bool(day_index < warmup_days)
    node_numeric = pd.to_numeric(df[node_col], errors="coerce")
    if node_numeric.notna().all():
        df["_node_sort_key"] = node_numeric.astype(np.int64)
        df = df.sort_values(["_node_sort_key", "time_slot"], kind="stable").drop(columns=["_node_sort_key"]).reset_index(drop=True)
    else:
        df["_node_sort_key"] = pd.factorize(df[node_col], sort=True)[0].astype(np.int64)
        df = df.sort_values(["_node_sort_key", "time_slot"], kind="stable").drop(columns=["_node_sort_key"]).reset_index(drop=True)
    df["row_in_chunk"] = np.arange(len(df), dtype=np.int64)
    meta = ChunkMetaRecord(
        day_index=int(day_index),
        file_name=file_path.name,
        file_path=get_relative_path(file_path),
        rows_used=int(len(df)),
        node_count=int(df[node_col].nunique()),
        time_slot_count=int(df["time_slot"].nunique()),
        time_slot_min=int(df["time_slot"].min()),
        time_slot_max=int(df["time_slot"].max()),
        warmup=bool(day_index < warmup_days),
    )
    return df, meta


def write_run_artifacts(args: argparse.Namespace) -> None:
    config_path = args.output_dir / "run_config.json"
    commands_path = args.output_dir / "run_commands.txt"
    serializable = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    config_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    command = " ".join(
        shlex.quote(str(value)) if " " in str(value) else str(value)
        for value in [
            r"E:\anaconda3\envs\analysis\python.exe",
            "analysis_scripts/full_intersection_missingness_pipeline.py",
            *sum(([f"--{k}", str(v)] for k, v in serializable.items() if k != "stage"), []),
        ]
    )
    with commands_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{args.stage}] {command}\n")


def write_chunk_manifest(output_dir: Path, records: list[ChunkMetaRecord]) -> Path:
    manifest_dir = ensure_directory(output_dir / "manifests")
    path = manifest_dir / "chunk_index_summary.csv"
    pd.DataFrame([asdict(record) for record in records]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_stage_status_manifest(output_dir: Path, stage: str, records: list[ChunkProcessStatusRecord]) -> Path:
    manifest_dir = ensure_directory(output_dir / "manifests")
    path = manifest_dir / f"{stage}_chunk_status.csv"
    pd.DataFrame([asdict(record) for record in records]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize_stage_status(records: list[ChunkProcessStatusRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(record) for record in records])
    return (
        df.groupby(["stage", "missing_rate", "mechanism", "seed", "impute_method", "status"], as_index=False)
        .size()
        .rename(columns={"size": "chunk_count"})
    )


def summarize_status_file(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {
            "exists": False,
            "row_count": 0,
            "completed": 0,
            "skipped_existing": 0,
            "other": 0,
            "unique_chunks": 0,
        }
    df = pd.read_csv(status_path)
    counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    return {
        "exists": True,
        "row_count": int(len(df)),
        "completed": int(counts.get("completed", 0)),
        "skipped_existing": int(counts.get("skipped_existing", 0)),
        "other": int(len(df) - counts.get("completed", 0) - counts.get("skipped_existing", 0)),
        "unique_chunks": int(df[["day_index", "file_name"]].drop_duplicates().shape[0]) if {"day_index", "file_name"} <= set(df.columns) else 0,
    }


def make_mcar_mask(df: pd.DataFrame, target_col: str, missing_rate: float, seed: int) -> np.ndarray:
    values = pd.to_numeric(df[target_col], errors="coerce")
    eligible = np.flatnonzero(values.notna().to_numpy())
    mask = np.zeros(len(df), dtype=bool)
    if len(eligible) == 0 or missing_rate <= 0:
        return mask
    count = int(round(len(eligible) * float(missing_rate)))
    if count <= 0:
        return mask
    rng = np.random.RandomState(seed)
    selected = rng.choice(eligible, size=count, replace=False)
    mask[selected] = True
    return mask


def make_temporal_block_mask(
    df: pd.DataFrame,
    target_col: str,
    node_col: str,
    missing_rate: float,
    seed: int,
    block_lengths: list[int],
) -> np.ndarray:
    values = pd.to_numeric(df[target_col], errors="coerce")
    mask = np.zeros(len(df), dtype=bool)
    if missing_rate <= 0:
        return mask
    target_missing = int(round(values.notna().sum() * float(missing_rate)))
    if target_missing <= 0:
        return mask
    rng = np.random.RandomState(seed)
    eligible = df.loc[values.notna(), [node_col, "global_time_index"]].copy()
    if eligible.empty:
        return mask
    node_to_rows = {
        node: grp[["global_time_index"]].assign(index=grp.index.to_numpy())
        for node, grp in df.groupby(node_col, sort=False)
    }
    chosen = 0
    attempts = 0
    max_attempts = max(target_missing * 8, 500)
    nodes = sorted(node_to_rows)
    while chosen < target_missing and attempts < max_attempts:
        attempts += 1
        node = nodes[int(rng.randint(0, len(nodes)))]
        block_length = int(block_lengths[int(rng.randint(0, len(block_lengths)))])
        node_df = node_to_rows[node]
        if node_df.empty:
            continue
        start_gt = int(node_df["global_time_index"].min())
        end_gt = int(node_df["global_time_index"].max())
        if end_gt < start_gt:
            continue
        block_start = int(rng.randint(start_gt, end_gt + 1))
        block_end = block_start + block_length - 1
        block_rows = node_df.loc[
            (node_df["global_time_index"] >= block_start)
            & (node_df["global_time_index"] <= block_end),
            "index",
        ].to_numpy(dtype=int)
        if len(block_rows) == 0:
            continue
        valid_rows = block_rows[values.iloc[block_rows].notna().to_numpy()]
        new_rows = valid_rows[~mask[valid_rows]]
        if len(new_rows) == 0:
            continue
        mask[new_rows] = True
        chosen = int(mask.sum())
    return mask


def mask_to_dataframe(df: pd.DataFrame, mask: np.ndarray, node_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_chunk_name": df["source_chunk_name"],
            "source_chunk_path": df["source_chunk_path"],
            "day_index": df["day_index"],
            "time_slot": df["time_slot"],
            "global_time_index": df["global_time_index"],
            node_col: df[node_col],
            "row_in_chunk": df["row_in_chunk"],
            "is_missing": mask.astype(np.int8),
        }
    )


def build_run_stub(missing_rate: float, mechanism: str, seed: int) -> str:
    rate_text = str(missing_rate).replace(".", "p")
    return f"rate_{rate_text}__mechanism_{mechanism}__seed_{seed}"


def build_impute_stub(run_stub: str, impute_method: str) -> str:
    return f"{run_stub}__method_{impute_method}"


def load_mask_from_file(mask_path: Path) -> pd.DataFrame:
    if mask_path.suffix.lower() == ".parquet":
        return pd.read_parquet(mask_path)
    return pd.read_csv(mask_path)


def save_mask_dataframe(mask_df: pd.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    mask_df.to_parquet(path, index=False)
    return path


def save_frame(df: pd.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    df.to_parquet(path, index=False)
    return path


def build_node_flow_groups(
    files: list[Path],
    target_col: str,
    time_col: str,
    node_col: str,
    period: int,
    warmup_days: int,
    max_rows: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for day_index, file_path in enumerate(files):
        df, _ = read_chunk_frame(
            file_path=file_path,
            day_index=day_index,
            target_col=target_col,
            time_col=time_col,
            node_col=node_col,
            period=period,
            warmup_days=warmup_days,
            max_rows=max_rows,
        )
        grouped = (
            df.groupby(node_col, as_index=False)[target_col]
            .median()
            .rename(columns={target_col: "node_flow_median"})
        )
        records.append(grouped)
    merged = pd.concat(records, ignore_index=True)
    summary = (
        merged.groupby(node_col, as_index=False)["node_flow_median"]
        .mean()
        .rename(columns={"node_flow_median": "node_flow_reference"})
    )
    if summary.empty:
        summary["flow_group"] = []
        summary["node_history_q95"] = []
        return summary
    low_cut = float(summary["node_flow_reference"].quantile(0.30))
    high_cut = float(summary["node_flow_reference"].quantile(0.70))
    summary["flow_group"] = np.where(
        summary["node_flow_reference"] <= low_cut,
        "low_flow",
        np.where(summary["node_flow_reference"] >= high_cut, "high_flow", "mid_flow"),
    )
    summary["node_history_q95"] = summary["node_flow_reference"]
    return summary.sort_values(node_col).reset_index(drop=True)


def build_neighbor_lookup(neighbor_edges: pd.DataFrame, nodes: list[str]) -> dict[int, list[tuple[int, float]]]:
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    lookup: dict[int, list[tuple[int, float]]] = {}
    for _, row in neighbor_edges.iterrows():
        src = str(row["node_id"])
        dst = str(row["neighbor_id"])
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        lookup.setdefault(node_to_idx[src], []).append((node_to_idx[dst], float(row["weight"])))
    return lookup


def build_day_matrix(
    df: pd.DataFrame,
    target_col: str,
    node_col: str,
    period: int,
    all_nodes: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    nodes = all_nodes or sorted(df[node_col].astype(str).unique().tolist())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    matrix = np.full((len(nodes), period), np.nan, dtype=np.float64)
    subset = df[[node_col, "time_slot", target_col]].copy()
    subset[node_col] = subset[node_col].astype(str)
    row_idx = subset[node_col].map(node_to_idx).to_numpy(dtype=int)
    col_idx = subset["time_slot"].to_numpy(dtype=int)
    values = pd.to_numeric(subset[target_col], errors="coerce").to_numpy(dtype=np.float64)
    matrix[row_idx, col_idx] = values
    template = df.copy()
    template[node_col] = template[node_col].astype(str)
    template = template.sort_values([node_col, "time_slot"]).reset_index(drop=True)
    return matrix, nodes, template


def flatten_matrix_to_frame(template: pd.DataFrame, matrix: np.ndarray, target_col: str, node_col: str) -> pd.DataFrame:
    node_positions = pd.Categorical(template[node_col].astype(str), categories=sorted(template[node_col].astype(str).unique()))
    nodes = list(node_positions.categories)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx = template[node_col].astype(str).map(node_to_idx).to_numpy(dtype=int)
    cols = template["time_slot"].to_numpy(dtype=int)
    result = template.copy()
    result[target_col] = matrix[idx, cols]
    return result


def get_history_arrays(prev_days: list[np.ndarray], history_days: int) -> list[np.ndarray]:
    if history_days <= 0:
        return []
    return prev_days[-history_days:]


def collect_history_values_for_node(
    history_arrays: list[np.ndarray],
    current_filled: np.ndarray,
    node_idx: int,
    current_slot: int,
    allow_current_day_past: bool,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for array in history_arrays:
        node_values = array[node_idx, :]
        finite = node_values[np.isfinite(node_values)]
        if finite.size > 0:
            values.append(finite)
    if allow_current_day_past and current_slot > 0:
        current_values = current_filled[node_idx, :current_slot]
        finite = current_values[np.isfinite(current_values)]
        if finite.size > 0:
            values.append(finite)
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values)


def collect_history_points_for_node(
    history_arrays: list[np.ndarray],
    current_filled: np.ndarray,
    node_idx: int,
    day_index: int,
    current_slot: int,
    period: int,
    allow_current_day_past: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gt_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    start_day = day_index - len(history_arrays)
    for offset, array in enumerate(history_arrays):
        hist_day_index = start_day + offset
        row = array[node_idx, :]
        finite_mask = np.isfinite(row)
        if finite_mask.any():
            slots = np.flatnonzero(finite_mask)
            gt_parts.append(hist_day_index * period + slots)
            value_parts.append(row[finite_mask])
    if allow_current_day_past and current_slot > 0:
        row = current_filled[node_idx, :current_slot]
        finite_mask = np.isfinite(row)
        if finite_mask.any():
            slots = np.flatnonzero(finite_mask)
            gt_parts.append(day_index * period + slots)
            value_parts.append(row[finite_mask])
    if not gt_parts:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.concatenate(gt_parts).astype(np.float64), np.concatenate(value_parts).astype(np.float64)


def historical_global_stats(
    history_arrays: list[np.ndarray],
    current_filled: np.ndarray,
    current_slot: int,
    allow_current_day_past: bool,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for array in history_arrays:
        finite = array[np.isfinite(array)]
        if finite.size > 0:
            values.append(finite)
    if allow_current_day_past and current_slot > 0:
        current = current_filled[:, :current_slot]
        finite = current[np.isfinite(current)]
        if finite.size > 0:
            values.append(finite)
    if not values:
        return 0.0, 0.0
    combined = np.concatenate(values)
    return float(np.median(combined)), float(np.quantile(combined, 0.95))


def apply_constraints(
    value: float,
    node_history_values: np.ndarray,
    global_q95: float,
    args: argparse.Namespace,
) -> float:
    if not np.isfinite(value):
        value = 0.0
    if args.clip_nonnegative:
        value = max(value, 0.0)
    if node_history_values.size > 0 and args.node_quantile_cap > 0:
        node_q = float(np.quantile(node_history_values, args.node_quantile_cap))
        value = min(value, node_q * float(args.node_quantile_cap_multiplier))
    elif global_q95 > 0 and args.node_quantile_cap > 0:
        value = min(value, global_q95 * float(args.node_quantile_cap_multiplier))
    if args.clip_nonnegative:
        value = max(value, 0.0)
    return float(value)


def forward_fill_causal_value(
    last_value: float,
    global_median: float,
    node_history_values: np.ndarray,
    global_q95: float,
    args: argparse.Namespace,
) -> float:
    candidate = last_value if np.isfinite(last_value) else global_median
    return apply_constraints(candidate, node_history_values, global_q95, args)


def historical_linear_extrapolation_value(
    history_gt: np.ndarray,
    history_values: np.ndarray,
    target_gt: float,
    fallback_value: float,
    node_history_values: np.ndarray,
    global_q95: float,
    args: argparse.Namespace,
) -> float:
    if history_values.size < 2:
        return apply_constraints(fallback_value, node_history_values, global_q95, args)
    use_count = min(int(args.historical_trend_points), int(history_values.size))
    x = history_gt[-use_count:]
    y = history_values[-use_count:]
    if np.unique(x).size < 2:
        return apply_constraints(y[-1], node_history_values, global_q95, args)
    coeff = np.polyfit(x, y, deg=1)
    estimate = coeff[0] * float(target_gt) + coeff[1]
    return apply_constraints(float(estimate), node_history_values, global_q95, args)


def build_fourier_features(global_indices: np.ndarray, slots: np.ndarray, period: int, order: int) -> np.ndarray:
    x = np.empty((len(global_indices), 2 + 2 * order), dtype=np.float64)
    x[:, 0] = 1.0
    gt_min = float(global_indices.min())
    gt_span = max(float(global_indices.max() - gt_min), 1.0)
    x[:, 1] = (global_indices - gt_min) / gt_span
    for k in range(1, order + 1):
        omega = 2.0 * np.pi * k / float(period)
        x[:, 2 * k] = np.sin(omega * slots)
        x[:, 2 * k + 1] = np.cos(omega * slots)
    return x


def function_curve_fit_value(
    history_gt: np.ndarray,
    history_values: np.ndarray,
    target_gt: float,
    target_slot: int,
    fallback_value: float,
    period: int,
    fourier_order: int,
    min_fit_points: int,
    node_history_values: np.ndarray,
    global_q95: float,
    args: argparse.Namespace,
) -> float:
    if history_values.size < int(min_fit_points):
        return apply_constraints(fallback_value, node_history_values, global_q95, args)
    history_slots = np.mod(history_gt, period).astype(np.float64)
    try:
        x = build_fourier_features(history_gt, history_slots, period, fourier_order)
        coeffs, _, _, _ = np.linalg.lstsq(x, history_values, rcond=None)
        pred_x = build_fourier_features(
            np.array([target_gt], dtype=np.float64),
            np.array([float(target_slot)], dtype=np.float64),
            period,
            fourier_order,
        )
        estimate = float((pred_x @ coeffs)[0])
    except np.linalg.LinAlgError:
        estimate = fallback_value
    return apply_constraints(estimate, node_history_values, global_q95, args)


def geo_neighbor_fill_value(
    node_idx: int,
    slot: int,
    history_arrays: list[np.ndarray],
    current_filled: np.ndarray,
    neighbor_lookup: dict[int, list[tuple[int, float]]],
    fallback_value: float,
    node_history_values: np.ndarray,
    global_q95: float,
    args: argparse.Namespace,
) -> float:
    neighbors = neighbor_lookup.get(node_idx, [])
    if not neighbors:
        return apply_constraints(fallback_value, node_history_values, global_q95, args)
    weighted_values: list[tuple[float, float]] = []
    close_slots = [slot]
    if slot > 0:
        close_slots.append(slot - 1)
    if slot < args.period - 1:
        close_slots.append(slot + 1)
    target_hist_median = float(np.median(node_history_values)) if node_history_values.size > 0 else np.nan
    for neighbor_idx, weight in neighbors:
        same_slot_values: list[float] = []
        near_slot_values: list[float] = []
        recent_value = np.nan
        for array in history_arrays:
            same_value = array[neighbor_idx, slot]
            if np.isfinite(same_value):
                same_slot_values.append(float(same_value))
            for near_slot in close_slots:
                near_value = array[neighbor_idx, near_slot]
                if np.isfinite(near_value):
                    near_slot_values.append(float(near_value))
            finite = array[neighbor_idx, :]
            finite = finite[np.isfinite(finite)]
            if finite.size > 0:
                recent_value = float(finite[-1])
        if args.allow_current_day_past and slot > 0:
            current_hist = current_filled[neighbor_idx, :slot]
            current_hist = current_hist[np.isfinite(current_hist)]
            if current_hist.size > 0:
                recent_value = float(current_hist[-1])
        if same_slot_values:
            neighbor_value = float(np.median(np.array(same_slot_values, dtype=np.float64)))
        elif near_slot_values:
            neighbor_value = float(np.median(np.array(near_slot_values, dtype=np.float64)))
        elif np.isfinite(recent_value):
            neighbor_value = float(recent_value)
        else:
            continue
        if np.isfinite(target_hist_median):
            neighbor_history = collect_history_values_for_node(
                history_arrays,
                current_filled,
                neighbor_idx,
                slot,
                args.allow_current_day_past,
            )
            if neighbor_history.size > 0:
                neighbor_hist_median = float(np.median(neighbor_history))
                if neighbor_hist_median > 1e-6:
                    neighbor_value = neighbor_value * target_hist_median / neighbor_hist_median
        weighted_values.append((neighbor_value, float(weight)))
    if not weighted_values:
        return apply_constraints(fallback_value, node_history_values, global_q95, args)
    numerator = sum(value * weight for value, weight in weighted_values)
    denominator = sum(weight for _, weight in weighted_values)
    if denominator <= 0:
        return apply_constraints(fallback_value, node_history_values, global_q95, args)
    return apply_constraints(numerator / denominator, node_history_values, global_q95, args)


def geo_func_hybrid_value(
    geo_value: float,
    func_value: float,
    fallback_value: float,
    node_history_values: np.ndarray,
    global_q95: float,
    args: argparse.Namespace,
) -> float:
    geo_valid = np.isfinite(geo_value)
    func_valid = np.isfinite(func_value)
    if geo_valid and func_valid:
        candidate = float(args.geo_lambda) * float(geo_value) + (1.0 - float(args.geo_lambda)) * float(func_value)
    elif geo_valid:
        candidate = float(geo_value)
    elif func_valid:
        candidate = float(func_value)
    else:
        candidate = float(fallback_value)
    return apply_constraints(candidate, node_history_values, global_q95, args)


def compute_metric_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "count": 0.0,
            "sum_abs_error": 0.0,
            "sum_squared_error": 0.0,
            "sum_pct_error": 0.0,
            "count_pct": 0.0,
            "sum_smape": 0.0,
        }
    abs_error = np.abs(y_true - y_pred)
    pct_mask = np.abs(y_true) > 1e-8
    smape = 2.0 * abs_error / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8) * 100.0
    return {
        "count": float(y_true.size),
        "sum_abs_error": float(abs_error.sum()),
        "sum_squared_error": float(np.square(y_true - y_pred).sum()),
        "sum_pct_error": float((abs_error[pct_mask] / np.abs(y_true[pct_mask]) * 100.0).sum()) if pct_mask.any() else 0.0,
        "count_pct": float(pct_mask.sum()),
        "sum_smape": float(smape.sum()),
    }


def metric_from_stats(sum_abs: float, sum_sq: float, sum_pct: float, count: float, count_pct: float, sum_smape: float) -> tuple[float, float, float, float]:
    if count <= 0:
        return 0.0, 0.0, 0.0, 0.0
    mae = float(sum_abs / count)
    rmse = float(math.sqrt(sum_sq / count))
    mape = float(sum_pct / count_pct) if count_pct > 0 else 0.0
    smape = float(sum_smape / count)
    return mae, rmse, mape, smape


def save_detail_frame(df: pd.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_generate_missing(args: argparse.Namespace) -> None:
    files = list_input_files(args.input_dir, args.input_pattern, args.max_chunks)
    first_columns = get_parquet_columns(files[0])
    target_col, time_col, node_col = resolve_columns(first_columns, args.target_col, args.time_col, args.node_col)
    chunk_records: list[ChunkMetaRecord] = []
    manifest_records: list[MissingManifestRecord] = []
    stage_status_records: list[ChunkProcessStatusRecord] = []
    block_lengths = parse_int_list(args.block_lengths)
    missing_rates = parse_float_list(args.missing_rates)
    seeds = parse_int_list(args.seed)
    mechanism = str(args.mechanism)
    write_run_artifacts(args)
    write_chunk_manifest(args.output_dir, [])
    for day_index, file_path in enumerate(files):
        df, meta = read_chunk_frame(
            file_path=file_path,
            day_index=day_index,
            target_col=target_col,
            time_col=time_col,
            node_col=node_col,
            period=args.period,
            warmup_days=args.warmup_days,
            max_rows=args.max_rows,
        )
        chunk_records.append(meta)
        for missing_rate in missing_rates:
            for seed in seeds:
                run_seed = stable_seed(seed, file_path.name, mechanism, missing_rate)
                if mechanism == "mcar_point":
                    mask = make_mcar_mask(df, target_col, missing_rate, run_seed)
                elif mechanism == "node_temporal_block":
                    mask = make_temporal_block_mask(df, target_col, node_col, missing_rate, run_seed, block_lengths)
                else:
                    raise ValueError(f"不支持的缺失机制: {mechanism}")
                actual_missing_count = int(mask.sum())
                eligible_count = int(pd.to_numeric(df[target_col], errors="coerce").notna().sum())
                actual_missing_rate = float(actual_missing_count / eligible_count) if eligible_count else 0.0
                run_stub = build_run_stub(missing_rate, mechanism, seed)
                mask_path = args.output_dir / "masks" / run_stub / f"{file_path.stem}_mask.parquet"
                missing_dataset_path: Optional[Path] = None
                if args.skip_existing and mask_path.exists():
                    existing_mask_df = load_mask_from_file(mask_path)
                    actual_missing_count = int(existing_mask_df["is_missing"].astype(bool).sum())
                    stage_status_records.append(
                        ChunkProcessStatusRecord(
                            stage="generate_missing",
                            missing_rate=float(missing_rate),
                            mechanism=mechanism,
                            seed=int(seed),
                            impute_method="",
                            day_index=int(day_index),
                            file_name=file_path.name,
                            status="skipped_existing",
                            mask_exists=True,
                            missing_dataset_exists=bool(
                                (args.output_dir / "missing_datasets" / run_stub / f"{file_path.stem}_missing.parquet").exists()
                            ),
                            imputed_dataset_exists=False,
                            detail_exists=False,
                            actual_missing_count=actual_missing_count,
                            filled_missing_count=0,
                            residual_missing_count=0,
                            note="mask 已存在，按 skip_existing 跳过重复构造",
                        )
                    )
                    manifest_records.append(
                        MissingManifestRecord(
                            missing_rate=float(missing_rate),
                            mechanism=mechanism,
                            seed=int(seed),
                            day_index=int(day_index),
                            file_name=file_path.name,
                            mask_path=get_relative_path(mask_path),
                            missing_dataset_path=(
                                get_relative_path(args.output_dir / "missing_datasets" / run_stub / f"{file_path.stem}_missing.parquet")
                                if (args.output_dir / "missing_datasets" / run_stub / f"{file_path.stem}_missing.parquet").exists()
                                else None
                            ),
                            actual_missing_count=actual_missing_count,
                            actual_missing_rate=float(actual_missing_count / eligible_count) if eligible_count else 0.0,
                        )
                    )
                    continue
                mask_df = mask_to_dataframe(df, mask, node_col)
                save_mask_dataframe(mask_df, mask_path)
                corrupted = df.copy()
                corrupted.loc[mask, target_col] = np.nan
                if args.write_missing_datasets:
                    missing_dataset_path = args.output_dir / "missing_datasets" / run_stub / f"{file_path.stem}_missing.parquet"
                    save_frame(corrupted, missing_dataset_path)
                manifest_records.append(
                    MissingManifestRecord(
                        missing_rate=float(missing_rate),
                        mechanism=mechanism,
                        seed=int(seed),
                        day_index=int(day_index),
                        file_name=file_path.name,
                        mask_path=get_relative_path(mask_path),
                        missing_dataset_path=None if missing_dataset_path is None else get_relative_path(missing_dataset_path),
                        actual_missing_count=actual_missing_count,
                        actual_missing_rate=actual_missing_rate,
                    )
                )
                stage_status_records.append(
                    ChunkProcessStatusRecord(
                        stage="generate_missing",
                        missing_rate=float(missing_rate),
                        mechanism=mechanism,
                        seed=int(seed),
                        impute_method="",
                        day_index=int(day_index),
                        file_name=file_path.name,
                        status="completed",
                        mask_exists=True,
                        missing_dataset_exists=bool(missing_dataset_path and missing_dataset_path.exists()),
                        imputed_dataset_exists=False,
                        detail_exists=False,
                        actual_missing_count=actual_missing_count,
                        filled_missing_count=0,
                        residual_missing_count=0,
                        note="缺失掩码与缺失数据集已登记",
                    )
                )
    write_chunk_manifest(args.output_dir, chunk_records)
    manifest_df = pd.DataFrame([asdict(record) for record in manifest_records])
    ensure_directory(args.output_dir / "manifests")
    manifest_df.to_csv(args.output_dir / "manifests" / "missing_runs.csv", index=False, encoding="utf-8-sig")
    write_stage_status_manifest(args.output_dir, "generate_missing", stage_status_records)
    summary_df = summarize_stage_status(stage_status_records)
    if not summary_df.empty:
        summary_df.to_csv(args.output_dir / "manifests" / "generate_missing_stage_summary.csv", index=False, encoding="utf-8-sig")


def prepare_day_template(
    df: pd.DataFrame,
    target_col: str,
    node_col: str,
    period: int,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, int]]:
    template = df.copy()
    template[node_col] = template[node_col].astype(str)
    template = template.sort_values([node_col, "time_slot"]).reset_index(drop=True)
    nodes = sorted(template[node_col].unique().tolist())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    matrix = np.full((len(nodes), period), np.nan, dtype=np.float64)
    row_idx = template[node_col].map(node_to_idx).to_numpy(dtype=int)
    col_idx = template["time_slot"].to_numpy(dtype=int)
    values = pd.to_numeric(template[target_col], errors="coerce").to_numpy(dtype=np.float64)
    matrix[row_idx, col_idx] = values
    return matrix, nodes, template, node_to_idx


def materialize_matrix(template: pd.DataFrame, matrix: np.ndarray, node_to_idx: dict[str, int], target_col: str, node_col: str) -> pd.DataFrame:
    output = template.copy()
    row_idx = output[node_col].astype(str).map(node_to_idx).to_numpy(dtype=int)
    col_idx = output["time_slot"].to_numpy(dtype=int)
    output[target_col] = matrix[row_idx, col_idx]
    return output


def preload_method_history_from_existing_outputs(
    history_store: dict[str, list[np.ndarray]],
    method: str,
    bootstrap_rows: pd.DataFrame,
    files: list[Path],
    run_stub: str,
    target_col: str,
    time_col: str,
    node_col: str,
    args: argparse.Namespace,
) -> None:
    if method == "zero_fill" or bootstrap_rows.empty:
        return
    for _, bootstrap_row in bootstrap_rows.sort_values("day_index").iterrows():
        bootstrap_day_index = int(bootstrap_row["day_index"])
        bootstrap_file = files[bootstrap_day_index]
        impute_stub = build_impute_stub(run_stub, method)
        imputed_path = args.output_dir / "imputed_datasets" / impute_stub / f"{bootstrap_file.stem}_imputed.parquet"
        if not imputed_path.exists():
            continue
        existing_imputed_df, _ = read_chunk_frame(
            file_path=imputed_path,
            day_index=bootstrap_day_index,
            target_col=target_col,
            time_col=time_col,
            node_col=node_col,
            period=args.period,
            warmup_days=args.warmup_days,
            max_rows=args.max_rows,
        )
        existing_filled_matrix, _, _, _ = prepare_day_template(existing_imputed_df, target_col, node_col, args.period)
        history_store[method].append(existing_filled_matrix)
        if len(history_store[method]) > args.history_days:
            history_store[method] = history_store[method][-args.history_days:]


def run_impute(args: argparse.Namespace) -> None:
    missing_runs_path = args.output_dir / "manifests" / "missing_runs.csv"
    if not missing_runs_path.exists():
        raise FileNotFoundError("请先运行 generate_missing 阶段。")
    files = list_input_files(args.input_dir, args.input_pattern, args.max_chunks)
    first_columns = get_parquet_columns(files[0])
    target_col, time_col, node_col = resolve_columns(first_columns, args.target_col, args.time_col, args.node_col)
    write_run_artifacts(args)
    missing_runs = pd.read_csv(missing_runs_path)
    requested_rates = set(parse_float_list(args.missing_rates))
    requested_seeds = set(parse_int_list(args.seed))
    mechanism = str(args.mechanism)
    missing_runs = missing_runs.loc[
        missing_runs["missing_rate"].isin(requested_rates)
        & (missing_runs["mechanism"] == mechanism)
        & missing_runs["seed"].isin(requested_seeds)
    ].copy()
    if missing_runs.empty:
        raise ValueError("missing_runs.csv 中没有匹配当前参数的缺失数据清单。")
    impute_methods = normalize_impute_methods(parse_str_list(args.impute_methods), args.causal_history_only)
    print(
        "[impute] matched missing runs: rows={0}, rates={1}, seeds={2}, methods={3}".format(
            len(missing_runs),
            sorted(requested_rates),
            sorted(requested_seeds),
            ",".join(impute_methods),
        )
    )
    ensure_directory(args.output_dir / "summaries")
    node_group_path = args.output_dir / "summaries" / "node_flow_group_summary.csv"
    if node_group_path.exists():
        node_group_df = pd.read_csv(node_group_path)
        if node_col not in node_group_df.columns or "flow_group" not in node_group_df.columns:
            node_group_df = build_node_flow_groups(
                files,
                target_col,
                time_col,
                node_col,
                args.period,
                args.warmup_days,
                args.max_rows,
            )
            node_group_df.to_csv(node_group_path, index=False, encoding="utf-8-sig")
    else:
        node_group_df = build_node_flow_groups(
            files,
            target_col,
            time_col,
            node_col,
            args.period,
            args.warmup_days,
            args.max_rows,
        )
        node_group_df.to_csv(node_group_path, index=False, encoding="utf-8-sig")
    flow_group_map = dict(zip(node_group_df[node_col].astype(str), node_group_df["flow_group"]))
    neighbor_edges = load_neighbor_edges(args.topology_path)
    manifest_records: list[ImputationManifestRecord] = []
    detail_frames: list[pd.DataFrame] = []
    stage_status_records: list[ChunkProcessStatusRecord] = []
    for missing_rate in sorted(requested_rates):
        for seed in sorted(requested_seeds):
            run_stub = build_run_stub(missing_rate, mechanism, seed)
            run_rows = missing_runs.loc[
                (missing_runs["missing_rate"] == missing_rate)
                & (missing_runs["mechanism"] == mechanism)
                & (missing_runs["seed"] == seed)
            ].sort_values("day_index")
            if run_rows.empty:
                continue
            original_days: dict[int, pd.DataFrame] = {}
            missing_days: dict[int, pd.DataFrame] = {}
            day_nodes: dict[int, list[str]] = {}
            day_node_to_idx: dict[int, dict[str, int]] = {}
            day_templates: dict[int, pd.DataFrame] = {}
            history_by_method: dict[str, list[np.ndarray]] = {method: [] for method in impute_methods}
            if args.skip_existing and len(impute_methods) == 1:
                repair_method = impute_methods[0]
                repair_stub = build_impute_stub(run_stub, repair_method)
                run_rows_with_flags = run_rows.copy()
                run_rows_with_flags["detail_exists"] = run_rows_with_flags["day_index"].apply(
                    lambda day_index: (
                        args.output_dir
                        / "manifests"
                        / "detail_runs"
                        / repair_stub
                        / f"{files[int(day_index)].stem}_detail.csv"
                    ).exists()
                )
                missing_subset = run_rows_with_flags.loc[~run_rows_with_flags["detail_exists"]].copy()
                if missing_subset.empty:
                    print(
                        "[impute] all requested chunks already complete for method={0}".format(
                            repair_method
                        )
                    )
                    continue
                first_missing_day = int(missing_subset["day_index"].min())
                if first_missing_day > 0:
                    bootstrap_start = max(0, first_missing_day - args.history_days)
                    bootstrap_rows = run_rows_with_flags.loc[
                        (run_rows_with_flags["day_index"] >= bootstrap_start)
                        & (run_rows_with_flags["day_index"] < first_missing_day)
                    ].copy()
                    preload_method_history_from_existing_outputs(
                        history_store=history_by_method,
                        method=repair_method,
                        bootstrap_rows=bootstrap_rows,
                        files=files,
                        run_stub=run_stub,
                        target_col=target_col,
                        time_col=time_col,
                        node_col=node_col,
                        args=args,
                    )
                run_rows = missing_subset.drop(columns=["detail_exists"]).copy()
                print(
                    "[impute] repair scope optimized for method={0}: first_missing_day={1}, missing_chunk_count={2}, bootstrap_days={3}".format(
                        repair_method,
                        first_missing_day,
                        len(run_rows),
                        len(history_by_method.get(repair_method, [])),
                    )
                )
            print(
                "[impute] run scope: missing_rate={0}, seed={1}, chunk_count={2}".format(
                    float(missing_rate),
                    int(seed),
                    len(run_rows),
                )
            )
            for _, row in run_rows.iterrows():
                day_index = int(row["day_index"])
                file_path = files[day_index]
                skipped_contexts: list[tuple[str, Optional[np.ndarray], Path, Path, int]] = []
                methods_to_compute: list[tuple[str, Path, Path]] = []
                for method in impute_methods:
                    impute_stub = build_impute_stub(run_stub, method)
                    detail_path = args.output_dir / "manifests" / "detail_runs" / impute_stub / f"{file_path.stem}_detail.csv"
                    imputed_path = args.output_dir / "imputed_datasets" / impute_stub / f"{file_path.stem}_imputed.parquet"
                    if args.skip_existing and detail_path.exists():
                        existing_detail_df = pd.read_csv(detail_path)
                        overall_row = existing_detail_df.loc[existing_detail_df["flow_group"] == "all"].head(1)
                        filled_missing_count = int(overall_row["count"].iloc[0]) if not overall_row.empty else 0
                        existing_filled_matrix: Optional[np.ndarray] = None
                        if method != "zero_fill" and imputed_path.exists():
                            existing_imputed_df, _ = read_chunk_frame(
                                file_path=imputed_path,
                                day_index=day_index,
                                target_col=target_col,
                                time_col=time_col,
                                node_col=node_col,
                                period=args.period,
                                warmup_days=args.warmup_days,
                                max_rows=args.max_rows,
                            )
                            existing_filled_matrix, _, _, _ = prepare_day_template(existing_imputed_df, target_col, node_col, args.period)
                        manifest_records.append(
                            ImputationManifestRecord(
                                missing_rate=float(missing_rate),
                                mechanism=mechanism,
                                seed=int(seed),
                                impute_method=method,
                                day_index=int(day_index),
                                file_name=file_path.name,
                                imputed_dataset_path=get_relative_path(imputed_path) if imputed_path.exists() else None,
                                filled_missing_count=filled_missing_count,
                                residual_missing_count=0,
                                detail_path=get_relative_path(detail_path),
                            )
                        )
                        detail_frames.append(existing_detail_df)
                        stage_status_records.append(
                            ChunkProcessStatusRecord(
                                stage="impute",
                                missing_rate=float(missing_rate),
                                mechanism=mechanism,
                                seed=int(seed),
                                impute_method=method,
                                day_index=int(day_index),
                                file_name=file_path.name,
                                status="skipped_existing",
                                mask_exists=True,
                                missing_dataset_exists=False,
                                imputed_dataset_exists=bool(imputed_path.exists()),
                                detail_exists=True,
                                actual_missing_count=filled_missing_count,
                                filled_missing_count=filled_missing_count,
                                residual_missing_count=0,
                                note="detail 已存在，按 skip_existing 跳过重复补全",
                            )
                        )
                        skipped_contexts.append(
                            (
                                method,
                                existing_filled_matrix,
                                detail_path,
                                imputed_path,
                                filled_missing_count,
                            )
                        )
                        continue
                    methods_to_compute.append((method, detail_path, imputed_path))

                if not methods_to_compute and all(
                    existing_filled_matrix is not None or method == "zero_fill"
                    for method, existing_filled_matrix, _, _, _ in skipped_contexts
                ):
                    for method, existing_filled_matrix, _, _, _ in skipped_contexts:
                        if existing_filled_matrix is None or method == "zero_fill":
                            continue
                        history_by_method[method].append(existing_filled_matrix)
                        if len(history_by_method[method]) > args.history_days:
                            history_by_method[method] = history_by_method[method][-args.history_days:]
                    continue

                original_df, _ = read_chunk_frame(
                    file_path=file_path,
                    day_index=day_index,
                    target_col=target_col,
                    time_col=time_col,
                    node_col=node_col,
                    period=args.period,
                    warmup_days=args.warmup_days,
                    max_rows=args.max_rows,
                )
                mask_df = load_mask_from_file(ROOT_DIR / Path(str(row["mask_path"])))
                mask_vec = mask_df["is_missing"].to_numpy(dtype=bool)
                missing_df = original_df.copy()
                missing_df.loc[mask_vec, target_col] = np.nan
                original_days[day_index] = original_df
                missing_days[day_index] = missing_df
                base_matrix, nodes, template, node_to_idx = prepare_day_template(missing_df, target_col, node_col, args.period)
                day_nodes[day_index] = nodes
                day_node_to_idx[day_index] = node_to_idx
                day_templates[day_index] = template
                neighbor_lookup = build_neighbor_lookup(neighbor_edges, nodes)
                original_matrix, _, _, _ = prepare_day_template(original_df, target_col, node_col, args.period)
                mask_matrix = np.isnan(base_matrix)
                flow_groups = np.array([flow_group_map.get(node, "mid_flow") for node in nodes], dtype=object)

                for method, existing_filled_matrix, _, _, _ in skipped_contexts:
                    if method == "zero_fill":
                        continue
                    history_by_method[method].append(base_matrix.copy() if existing_filled_matrix is None else existing_filled_matrix)
                    if len(history_by_method[method]) > args.history_days:
                        history_by_method[method] = history_by_method[method][-args.history_days:]

                for method, detail_path, imputed_path in methods_to_compute:
                    print(
                        "[impute] computing method={0}, day_index={1}, file={2}".format(
                            method,
                            int(day_index),
                            file_path.name,
                        )
                    )
                    history_arrays = get_history_arrays(history_by_method[method], args.history_days)
                    filled = base_matrix.copy()
                    last_value = np.full(filled.shape[0], np.nan, dtype=np.float64)
                    if history_arrays:
                        hist_last = np.stack(history_arrays, axis=0)
                        hist_last = np.where(np.isfinite(hist_last), hist_last, np.nan)
                        for node_idx in range(filled.shape[0]):
                            finite = hist_last[:, node_idx, :].reshape(-1)
                            finite = finite[np.isfinite(finite)]
                            if finite.size > 0:
                                last_value[node_idx] = float(finite[-1])
                    for slot in range(args.period):
                        global_median, global_q95 = historical_global_stats(
                            history_arrays,
                            filled,
                            slot,
                            args.allow_current_day_past,
                        )
                        missing_nodes = np.flatnonzero(np.isnan(filled[:, slot]))
                        if missing_nodes.size == 0:
                            current = filled[:, slot]
                            if args.allow_current_day_past:
                                observed_mask = np.isfinite(current)
                                last_value[observed_mask] = current[observed_mask]
                            continue
                        for node_idx in missing_nodes.tolist():
                            node_hist_values = collect_history_values_for_node(
                                history_arrays,
                                filled,
                                node_idx,
                                slot,
                                args.allow_current_day_past,
                            )
                            fallback = forward_fill_causal_value(
                                last_value=float(last_value[node_idx]) if np.isfinite(last_value[node_idx]) else np.nan,
                                global_median=global_median,
                                node_history_values=node_hist_values,
                                global_q95=global_q95,
                                args=args,
                            )
                            target_gt = day_index * args.period + slot
                            if method == "zero_fill":
                                value = apply_constraints(0.0, node_hist_values, global_q95, args)
                            elif method == "forward_fill":
                                value = fallback
                            elif method == "historical_linear_extrapolation":
                                history_gt, history_values = collect_history_points_for_node(
                                    history_arrays=history_arrays,
                                    current_filled=filled,
                                    node_idx=node_idx,
                                    day_index=day_index,
                                    current_slot=slot,
                                    period=args.period,
                                    allow_current_day_past=args.allow_current_day_past,
                                )
                                value = historical_linear_extrapolation_value(
                                    history_gt=history_gt,
                                    history_values=history_values,
                                    target_gt=float(target_gt),
                                    fallback_value=fallback,
                                    node_history_values=node_hist_values,
                                    global_q95=global_q95,
                                    args=args,
                                )
                            elif method == "geo_neighbor_fill":
                                value = geo_neighbor_fill_value(
                                    node_idx=node_idx,
                                    slot=slot,
                                    history_arrays=history_arrays,
                                    current_filled=filled,
                                    neighbor_lookup=neighbor_lookup,
                                    fallback_value=fallback,
                                    node_history_values=node_hist_values,
                                    global_q95=global_q95,
                                    args=args,
                                )
                            elif method == "function_curve_fit":
                                history_gt, history_values = collect_history_points_for_node(
                                    history_arrays=history_arrays,
                                    current_filled=filled,
                                    node_idx=node_idx,
                                    day_index=day_index,
                                    current_slot=slot,
                                    period=args.period,
                                    allow_current_day_past=args.allow_current_day_past,
                                )
                                value = function_curve_fit_value(
                                    history_gt=history_gt,
                                    history_values=history_values,
                                    target_gt=float(target_gt),
                                    target_slot=int(slot),
                                    fallback_value=fallback,
                                    period=int(args.period),
                                    fourier_order=int(args.fourier_order),
                                    min_fit_points=int(args.min_fit_points),
                                    node_history_values=node_hist_values,
                                    global_q95=global_q95,
                                    args=args,
                                )
                            elif method == "geo_func_hybrid":
                                history_gt, history_values = collect_history_points_for_node(
                                    history_arrays=history_arrays,
                                    current_filled=filled,
                                    node_idx=node_idx,
                                    day_index=day_index,
                                    current_slot=slot,
                                    period=args.period,
                                    allow_current_day_past=args.allow_current_day_past,
                                )
                                func_value = function_curve_fit_value(
                                    history_gt=history_gt,
                                    history_values=history_values,
                                    target_gt=float(target_gt),
                                    target_slot=int(slot),
                                    fallback_value=fallback,
                                    period=int(args.period),
                                    fourier_order=int(args.fourier_order),
                                    min_fit_points=int(args.min_fit_points),
                                    node_history_values=node_hist_values,
                                    global_q95=global_q95,
                                    args=args,
                                )
                                geo_value = geo_neighbor_fill_value(
                                    node_idx=node_idx,
                                    slot=slot,
                                    history_arrays=history_arrays,
                                    current_filled=filled,
                                    neighbor_lookup=neighbor_lookup,
                                    fallback_value=fallback,
                                    node_history_values=node_hist_values,
                                    global_q95=global_q95,
                                    args=args,
                                )
                                value = geo_func_hybrid_value(
                                    geo_value=geo_value,
                                    func_value=func_value,
                                    fallback_value=fallback,
                                    node_history_values=node_hist_values,
                                    global_q95=global_q95,
                                    args=args,
                                )
                            else:
                                raise ValueError(f"不支持的插补方法: {method}")
                            filled[node_idx, slot] = value
                        if args.allow_current_day_past:
                            current = filled[:, slot]
                            observed_mask = np.isfinite(current)
                            last_value[observed_mask] = current[observed_mask]
                    imputed_path_runtime: Optional[Path] = None
                    if args.write_imputed_datasets:
                        output_frame = materialize_matrix(
                            template=day_templates[day_index],
                            matrix=filled,
                            node_to_idx=day_node_to_idx[day_index],
                            target_col=target_col,
                            node_col=node_col,
                        )
                        imputed_path_runtime = imputed_path
                        save_frame(output_frame, imputed_path_runtime)
                    y_true = original_matrix[mask_matrix]
                    y_pred = filled[mask_matrix]
                    valid = np.isfinite(y_true) & np.isfinite(y_pred)
                    y_true = y_true[valid]
                    y_pred = y_pred[valid]
                    node_indices, _ = np.where(mask_matrix)
                    node_indices = node_indices[valid]
                    detail_rows: list[dict[str, Any]] = []
                    overall = compute_metric_stats(y_true, y_pred)
                    base_row = {
                        "missing_rate": float(missing_rate),
                        "mechanism": mechanism,
                        "seed": int(seed),
                        "impute_method": method,
                        "day_index": int(day_index),
                        "file_name": file_path.name,
                        "is_warmup": bool(day_index < args.warmup_days),
                        "flow_group": "all",
                    }
                    detail_rows.append({**base_row, **overall})
                    for flow_group in ["low_flow", "mid_flow", "high_flow"]:
                        group_mask = np.array([flow_groups[node_idx] == flow_group for node_idx in node_indices], dtype=bool)
                        stats = compute_metric_stats(y_true[group_mask], y_pred[group_mask])
                        detail_rows.append({**base_row, "flow_group": flow_group, **stats})
                    detail_df = pd.DataFrame(detail_rows)
                    save_detail_frame(detail_df, detail_path)
                    detail_frames.append(detail_df)
                    filled_missing_count = int(np.isfinite(filled[mask_matrix]).sum())
                    residual_missing_count = int(np.isnan(filled[mask_matrix]).sum())
                    manifest_records.append(
                        ImputationManifestRecord(
                            missing_rate=float(missing_rate),
                            mechanism=mechanism,
                            seed=int(seed),
                            impute_method=method,
                            day_index=int(day_index),
                            file_name=file_path.name,
                            imputed_dataset_path=None if imputed_path_runtime is None else get_relative_path(imputed_path_runtime),
                            filled_missing_count=filled_missing_count,
                            residual_missing_count=residual_missing_count,
                            detail_path=get_relative_path(detail_path),
                        )
                    )
                    stage_status_records.append(
                        ChunkProcessStatusRecord(
                            stage="impute",
                            missing_rate=float(missing_rate),
                            mechanism=mechanism,
                            seed=int(seed),
                            impute_method=method,
                            day_index=int(day_index),
                            file_name=file_path.name,
                            status="completed",
                            mask_exists=True,
                            missing_dataset_exists=False,
                            imputed_dataset_exists=bool(imputed_path_runtime and imputed_path_runtime.exists()),
                            detail_exists=True,
                            actual_missing_count=int(mask_matrix.sum()),
                            filled_missing_count=filled_missing_count,
                            residual_missing_count=residual_missing_count,
                            note="历史因果补全结果与明细已登记",
                        )
                    )
                    history_by_method[method].append(filled.copy())
                    if len(history_by_method[method]) > args.history_days:
                        history_by_method[method] = history_by_method[method][-args.history_days:]
    manifest_df = pd.DataFrame([asdict(record) for record in manifest_records])
    ensure_directory(args.output_dir / "manifests")
    manifest_df.to_csv(args.output_dir / "manifests" / "imputation_runs.csv", index=False, encoding="utf-8-sig")
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            args.output_dir / "summaries" / "imputation_quality_detail.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_stage_status_manifest(args.output_dir, "impute", stage_status_records)
    summary_df = summarize_stage_status(stage_status_records)
    if not summary_df.empty:
        summary_df.to_csv(args.output_dir / "manifests" / "impute_stage_summary.csv", index=False, encoding="utf-8-sig")
    print(
        "[impute] finished: manifest_records={0}, detail_frames={1}, stage_status_records={2}".format(
            len(manifest_records),
            len(detail_frames),
            len(stage_status_records),
        )
    )


def aggregate_detail(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()
    grouped = (
        detail_df.groupby(["mechanism", "impute_method", "missing_rate", "flow_group"], as_index=False)[
            ["count", "sum_abs_error", "sum_squared_error", "sum_pct_error", "count_pct", "sum_smape"]
        ]
        .sum()
    )
    metrics = grouped.apply(
        lambda row: pd.Series(
            metric_from_stats(
                row["sum_abs_error"],
                row["sum_squared_error"],
                row["sum_pct_error"],
                row["count"],
                row["count_pct"],
                row["sum_smape"],
            ),
            index=["MAE", "RMSE", "MAPE", "sMAPE"],
        ),
        axis=1,
    )
    return pd.concat([grouped, metrics], axis=1)


def build_main_summary(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()
    overall = agg_df.loc[agg_df["flow_group"] == "all"].copy()
    flow_group_map = {
        "low_flow": ["low_flow_MAE", "low_flow_RMSE", "low_flow_sMAPE"],
        "mid_flow": ["mid_flow_MAE", "mid_flow_RMSE", "mid_flow_sMAPE"],
        "high_flow": ["high_flow_MAE", "high_flow_RMSE", "high_flow_sMAPE"],
    }
    summary = overall[["mechanism", "impute_method", "missing_rate", "MAE", "RMSE", "MAPE", "count"]].copy()
    summary = summary.rename(columns={"count": "masked_count"})
    for flow_group, columns in flow_group_map.items():
        sub = agg_df.loc[agg_df["flow_group"] == flow_group, ["mechanism", "impute_method", "missing_rate", "MAE", "RMSE", "sMAPE"]]
        sub = sub.rename(columns={"MAE": columns[0], "RMSE": columns[1], "sMAPE": columns[2]})
        summary = summary.merge(sub, on=["mechanism", "impute_method", "missing_rate"], how="left")
    return summary.sort_values(["mechanism", "impute_method", "missing_rate"]).reset_index(drop=True)


def get_single_rate_value(df: pd.DataFrame) -> Optional[float]:
    if df.empty or "missing_rate" not in df.columns:
        return None
    rates = sorted({float(value) for value in df["missing_rate"].dropna().tolist()})
    return rates[0] if len(rates) == 1 else None


def build_single_rate_tag(rate: float) -> str:
    text = "{0:.4f}".format(rate).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def get_single_rate_method_order(summary_df: pd.DataFrame) -> list[str]:
    if summary_df.empty or "impute_method" not in summary_df.columns:
        return []
    ordered = summary_df.sort_values(["RMSE", "MAE", "impute_method"])["impute_method"].tolist()
    deduped: list[str] = []
    for method in ordered:
        if method not in deduped:
            deduped.append(method)
    for method in DEFAULT_IMPUTE_METHODS:
        if method in summary_df["impute_method"].tolist() and method not in deduped:
            deduped.append(method)
    return deduped


def plot_single_rate_metric_bars(
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    title: str,
    file_stub: str,
    method_order: list[str],
    ylabel: str,
    exclude_methods: Optional[set[str]] = None,
    legacy_file_stub: Optional[str] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    if summary_df.empty or "impute_method" not in summary_df.columns or metric not in summary_df.columns:
        return None, None
    single_rate = get_single_rate_value(summary_df)
    if single_rate is None:
        return None, None
    figures_dir = ensure_directory(output_dir / "figures")
    rate_tag = build_single_rate_tag(single_rate)
    subset = summary_df.copy()
    if exclude_methods:
        subset = subset.loc[~subset["impute_method"].isin(exclude_methods)].copy()
    if subset.empty:
        return None, None
    order = [method for method in method_order if method in subset["impute_method"].tolist()]
    if not order:
        order = subset["impute_method"].tolist()
    subset = subset.set_index("impute_method").loc[order].reset_index()
    labels = [METHOD_LABEL_MAP.get(method, method) for method in subset["impute_method"]]
    png_path = figures_dir / f"single_rate_{rate_tag}_{file_stub}.png"
    pdf_path = figures_dir / f"single_rate_{rate_tag}_{file_stub}.pdf"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, subset[metric], color="#4C78A8")
    ax.set_xlabel("Imputation method")
    ax.set_ylabel(ylabel)
    ax.set_title(title.format(single_rate))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    if legacy_file_stub:
        fig.savefig(figures_dir / f"single_rate_{rate_tag}_{legacy_file_stub}.png", dpi=300)
        fig.savefig(figures_dir / f"single_rate_{rate_tag}_{legacy_file_stub}.pdf")
    plt.close(fig)
    return png_path, pdf_path


def plot_main_rmse(summary_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    figures_dir = ensure_directory(output_dir / "figures")
    single_rate = get_single_rate_value(summary_df)
    if single_rate is not None:
        method_order = get_single_rate_method_order(summary_df)
        png_path, pdf_path = plot_single_rate_metric_bars(
            summary_df=summary_df,
            output_dir=output_dir,
            metric="RMSE",
            title="Single missing rate = {0:.0%}: RMSE by imputation method",
            file_stub="rmse_by_method_all6",
            method_order=method_order,
            ylabel="RMSE",
            legacy_file_stub="rmse_by_method",
        )
        assert png_path is not None and pdf_path is not None
    else:
        png_path = figures_dir / "missing_rate_vs_imputation_rmse.png"
        pdf_path = figures_dir / "missing_rate_vs_imputation_rmse.pdf"
    fig, ax = plt.subplots(figsize=(8, 5))
    if not summary_df.empty and "impute_method" in summary_df.columns:
        for method, group in summary_df.groupby("impute_method"):
            ax.plot(
                group["missing_rate"],
                group["RMSE"],
                marker="o",
                linewidth=1.8,
                label=METHOD_LABEL_MAP.get(method, method),
            )
        ax.set_xlabel("Missing rate")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("Imputation RMSE")
        ax.set_title("Imputation RMSE under Artificial Missing Rates\nFull Intersection-stage Real Data, Historical Causal Setting")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_zoom_rmse(summary_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    figures_dir = ensure_directory(output_dir / "figures")
    png_path = figures_dir / "zoom_historical_geo_function_rmse.png"
    pdf_path = figures_dir / "zoom_historical_geo_function_rmse.pdf"
    methods = {"geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid", "forward_fill"}
    if summary_df.empty or "impute_method" not in summary_df.columns:
        subset = pd.DataFrame()
    else:
        subset = summary_df.loc[summary_df["impute_method"].isin(methods)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not subset.empty:
        for method, group in subset.groupby("impute_method"):
            ax.plot(
                group["missing_rate"],
                group["RMSE"],
                marker="o",
                linewidth=1.8,
                label=METHOD_LABEL_MAP.get(method, method),
            )
    ax.set_xlabel("Missing rate")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Imputation RMSE")
    ax.set_title("Zoomed RMSE Comparison of Historical Geo and Functional Imputation")
    ax.grid(True, linestyle="--", alpha=0.35)
    if not subset.empty:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_delta_rmse(summary_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    figures_dir = ensure_directory(output_dir / "figures")
    if summary_df.empty or "impute_method" not in summary_df.columns:
        subset = pd.DataFrame()
    else:
        baseline = summary_df.loc[summary_df["impute_method"] == "forward_fill", ["missing_rate", "RMSE"]].rename(columns={"RMSE": "baseline_rmse"})
        merged = summary_df.merge(baseline, on="missing_rate", how="left")
        merged["delta_rmse"] = merged["RMSE"] - merged["baseline_rmse"]
        subset = merged.loc[merged["impute_method"] != "forward_fill"].copy()
    single_rate = get_single_rate_value(summary_df)
    if single_rate is not None:
        rate_tag = build_single_rate_tag(single_rate)
        png_path = figures_dir / f"single_rate_{rate_tag}_delta_rmse_relative_to_forward_fill.png"
        pdf_path = figures_dir / f"single_rate_{rate_tag}_delta_rmse_relative_to_forward_fill.pdf"
    else:
        png_path = figures_dir / "rmse_difference_relative_to_forward_fill.png"
        pdf_path = figures_dir / "rmse_difference_relative_to_forward_fill.pdf"
    fig, ax = plt.subplots(figsize=(8, 5))
    if single_rate is not None and not subset.empty:
        subset = subset.sort_values("delta_rmse").copy()
        labels = [METHOD_LABEL_MAP.get(method, method) for method in subset["impute_method"]]
        colors = ["#E45756" if value > 0 else "#54A24B" for value in subset["delta_rmse"]]
        ax.bar(labels, subset["delta_rmse"], color=colors)
        ax.set_xlabel("Imputation method")
        ax.set_ylabel("RMSE Difference")
        ax.set_title("Auxiliary diagnostic: single missing rate = {0:.0%} RMSE delta relative to forward fill".format(single_rate))
        ax.tick_params(axis="x", rotation=20)
    elif not subset.empty:
        for method, group in subset.groupby("impute_method"):
            ax.plot(
                group["missing_rate"],
                group["delta_rmse"],
                marker="o",
                linewidth=1.8,
                label=METHOD_LABEL_MAP.get(method, method),
            )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    if single_rate is None:
        ax.set_xlabel("Missing rate")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("RMSE Difference")
        ax.set_title("RMSE Difference Relative to Causal Forward Fill")
    ax.grid(True, linestyle="--", alpha=0.35)
    if single_rate is None and not subset.empty:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_flow_group_rmse(flow_group_summary: pd.DataFrame, output_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    if flow_group_summary.empty or "missing_rate" not in flow_group_summary.columns or "flow_group" not in flow_group_summary.columns:
        return None, None
    single_rate = get_single_rate_value(flow_group_summary)
    if single_rate is None:
        return None, None
    figures_dir = ensure_directory(output_dir / "figures")
    rate_tag = build_single_rate_tag(single_rate)
    png_path = figures_dir / f"single_rate_{rate_tag}_flow_group_rmse_by_method_all6.png"
    pdf_path = figures_dir / f"single_rate_{rate_tag}_flow_group_rmse_by_method_all6.pdf"
    subset = flow_group_summary.loc[flow_group_summary["flow_group"].isin(["low_flow", "mid_flow", "high_flow"])].copy()
    if subset.empty:
        return None, None
    summary_main_path = output_dir / "summaries" / "imputation_quality_summary_exclude_warmup.csv"
    if summary_main_path.exists():
        method_order = get_single_rate_method_order(pd.read_csv(summary_main_path))
    else:
        method_order = DEFAULT_IMPUTE_METHODS
    pivot = (
        subset.pivot_table(index="impute_method", columns="flow_group", values="RMSE", aggfunc="first")
        .reindex(index=[method for method in method_order if method in subset["impute_method"].unique()])
        .dropna(how="all")
    )
    if pivot.empty:
        return None, None
    flow_groups = ["low_flow", "mid_flow", "high_flow"]
    labels = [METHOD_LABEL_MAP.get(method, method) for method in pivot.index]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"low_flow": "#4C78A8", "mid_flow": "#F58518", "high_flow": "#54A24B"}
    for idx, flow_group in enumerate(flow_groups):
        values = pivot.get(flow_group, pd.Series(index=pivot.index, dtype=float)).fillna(np.nan).to_numpy()
        ax.bar(x + (idx - 1) * width, values, width=width, label=flow_group.replace("_", " "), color=colors[flow_group])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("Imputation method")
    ax.set_ylabel("RMSE")
    ax.set_title("Single missing rate = {0:.0%}: RMSE by flow group and imputation method".format(single_rate))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(figures_dir / f"single_rate_{rate_tag}_flow_group_rmse.png", dpi=300)
    fig.savefig(figures_dir / f"single_rate_{rate_tag}_flow_group_rmse.pdf")
    plt.close(fig)
    return png_path, pdf_path


def build_single_rate_figure_index_rows(
    summary_df: pd.DataFrame,
    metric_label: str,
) -> list[dict[str, str]]:
    single_rate = get_single_rate_value(summary_df)
    if single_rate is None:
        return []
    rate_tag = build_single_rate_tag(single_rate)
    third_metric_file_label = "sMAPE" if metric_label == "sMAPE" else "MAPE"
    third_metric_file_stub = "smape" if metric_label == "sMAPE" else "mape"
    return [
        {
            "figure_file": f"single_rate_{rate_tag}_rmse_by_method_all6.png",
            "figure_type": "bar",
            "metric": "RMSE",
            "method_scope": "all_6_methods",
            "is_formal_main_figure": "true",
            "notes": "Formal six-method direct comparison",
        },
        {
            "figure_file": f"single_rate_{rate_tag}_mae_by_method_all6.png",
            "figure_type": "bar",
            "metric": "MAE",
            "method_scope": "all_6_methods",
            "is_formal_main_figure": "true",
            "notes": "Formal six-method direct comparison",
        },
        {
            "figure_file": f"single_rate_{rate_tag}_{third_metric_file_stub}_by_method_all6.png",
            "figure_type": "bar",
            "metric": third_metric_file_label,
            "method_scope": "all_6_methods",
            "is_formal_main_figure": "true",
            "notes": "Formal six-method direct comparison",
        },
        {
            "figure_file": f"single_rate_{rate_tag}_flow_group_rmse_by_method_all6.png",
            "figure_type": "grouped_bar",
            "metric": "RMSE",
            "method_scope": "all_6_methods",
            "is_formal_main_figure": "true",
            "notes": "Compares low-flow, mid-flow, and high-flow RMSE across all six methods without a baseline method",
        },
        {
            "figure_file": f"single_rate_{rate_tag}_rmse_by_method_nonzero_zoom.png",
            "figure_type": "bar",
            "metric": "RMSE",
            "method_scope": "nonzero_methods",
            "is_formal_main_figure": "false",
            "notes": "Zoom view excluding zero fill for readability",
        },
    ]


def get_single_rate_secondary_metric(summary_df: pd.DataFrame) -> tuple[str, str]:
    if "sMAPE" in summary_df.columns:
        return "sMAPE", "smape"
    return "MAPE", "mape"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown(path: Path, lines: list[str]) -> Path:
    ensure_directory(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_audit_payload(args: argparse.Namespace, chunk_df: pd.DataFrame, summary_all: pd.DataFrame, summary_main: pd.DataFrame) -> dict[str, Any]:
    chunk_records = [] if chunk_df.empty else chunk_df.to_dict(orient="records")
    total_chunks = int(len(chunk_df))
    return {
        "environment": {
            "python_path": r"E:\anaconda3\envs\analysis\python.exe",
            "python_target": "3.9",
        },
        "input": {
            "input_dir": get_relative_path(args.input_dir),
            "output_dir": get_relative_path(args.output_dir),
            "input_pattern": args.input_pattern,
            "mechanism": args.mechanism,
            "missing_rates": parse_float_list(args.missing_rates),
            "seed": parse_int_list(args.seed),
            "impute_methods": normalize_impute_methods(parse_str_list(args.impute_methods), args.causal_history_only),
        },
        "causal_checks": {
            "causal_history_only": bool(args.causal_history_only),
            "history_days": int(args.history_days),
            "context_days_after": int(args.context_days_after),
            "uses_future_days": False,
            "uses_same_day_future_slots": False,
            "uses_bfill": False,
            "uses_bidirectional_interpolation": False,
            "warmup_days": int(args.warmup_days),
            "main_metrics_exclude_warmup": bool(args.exclude_warmup_from_main_metrics),
            "allow_current_day_past": bool(args.allow_current_day_past),
        },
        "time_index_construction": {
            "formula": "global_time_index = day_index * 96 + time_slot",
            "period": int(args.period),
            "chunk_records": chunk_records,
        },
        "batch_scope": {
            "total_chunks_selected": total_chunks,
            "processes_all_chunks": bool(args.max_chunks == 0),
        },
        "summary_rows": {
            "all_days": int(len(summary_all)),
            "exclude_warmup": int(len(summary_main)),
        },
    }


def run_validate(args: argparse.Namespace) -> None:
    detail_path = args.output_dir / "summaries" / "imputation_quality_detail.csv"
    chunk_path = args.output_dir / "manifests" / "chunk_index_summary.csv"
    manifest_path = args.output_dir / "manifests" / "imputation_runs.csv"
    generate_status_path = args.output_dir / "manifests" / "generate_missing_chunk_status.csv"
    impute_status_path = args.output_dir / "manifests" / "impute_chunk_status.csv"
    detail_df = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    chunk_df = pd.read_csv(chunk_path) if chunk_path.exists() else pd.DataFrame()
    manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    generate_status = summarize_status_file(generate_status_path)
    impute_status = summarize_status_file(impute_status_path)
    payload = {
        "checks": {
            "detail_exists": bool(detail_path.exists()),
            "chunk_manifest_exists": bool(chunk_path.exists()),
            "imputation_manifest_exists": bool(manifest_path.exists()),
            "generate_status_exists": generate_status["exists"],
            "impute_status_exists": impute_status["exists"],
            "causal_history_only": bool(args.causal_history_only),
            "context_days_after": int(args.context_days_after),
            "uses_future_days": False,
            "uses_same_day_future_slots": False,
            "uses_bfill": False,
            "uses_bidirectional_interpolation": False,
        },
        "counts": {
            "detail_rows": int(len(detail_df)),
            "chunk_rows": int(len(chunk_df)),
            "imputation_rows": int(len(manifest_df)),
            "generate_completed_or_skipped": int(generate_status["completed"] + generate_status["skipped_existing"]),
            "impute_completed_or_skipped": int(impute_status["completed"] + impute_status["skipped_existing"]),
        },
    }
    write_json(args.output_dir / "full_intersection_missingness_validation.json", payload)
    lines = [
        "# Full Intersection Missingness Validation",
        "",
        "- causal_history_only: `true`" if args.causal_history_only else "- causal_history_only: `false`",
        f"- context_days_after: `{args.context_days_after}`",
        "- uses_future_days: `false`",
        "- uses_same_day_future_slots: `false`",
        "- uses_bfill: `false`",
        "- uses_bidirectional_interpolation: `false`",
        f"- detail_rows: `{len(detail_df)}`",
        f"- chunk_rows: `{len(chunk_df)}`",
        f"- imputation_rows: `{len(manifest_df)}`",
        f"- generate_completed_or_skipped: `{generate_status['completed'] + generate_status['skipped_existing']}`",
        f"- impute_completed_or_skipped: `{impute_status['completed'] + impute_status['skipped_existing']}`",
    ]
    write_markdown(args.output_dir / "full_intersection_missingness_validation.md", lines)


def run_summarize(args: argparse.Namespace) -> None:
    detail_path = args.output_dir / "summaries" / "imputation_quality_detail.csv"
    if not detail_path.exists():
        raise FileNotFoundError("请先运行 impute 阶段生成 imputation_quality_detail.csv。")
    detail_df = pd.read_csv(detail_path)
    chunk_path = args.output_dir / "manifests" / "chunk_index_summary.csv"
    chunk_df = pd.read_csv(chunk_path) if chunk_path.exists() else pd.DataFrame()
    generate_status_path = args.output_dir / "manifests" / "generate_missing_chunk_status.csv"
    impute_status_path = args.output_dir / "manifests" / "impute_chunk_status.csv"
    generate_status = summarize_status_file(generate_status_path)
    impute_status = summarize_status_file(impute_status_path)
    all_days_agg = aggregate_detail(detail_df)
    exclude_df = detail_df.loc[~detail_df["is_warmup"].astype(bool)].copy() if args.exclude_warmup_from_main_metrics else detail_df.copy()
    exclude_agg = aggregate_detail(exclude_df)
    summary_all = build_main_summary(all_days_agg)
    summary_main = build_main_summary(exclude_agg)
    summaries_dir = ensure_directory(args.output_dir / "summaries")
    summary_all.to_csv(summaries_dir / "imputation_quality_summary_all_days.csv", index=False, encoding="utf-8-sig")
    summary_main.to_csv(summaries_dir / "imputation_quality_summary_exclude_warmup.csv", index=False, encoding="utf-8-sig")
    if exclude_agg.empty or "flow_group" not in exclude_agg.columns:
        flow_group_summary = pd.DataFrame(
            columns=[
                "mechanism",
                "impute_method",
                "missing_rate",
                "flow_group",
                "count",
                "sum_abs_error",
                "sum_squared_error",
                "sum_pct_error",
                "count_pct",
                "sum_smape",
                "MAE",
                "RMSE",
                "MAPE",
                "sMAPE",
            ]
        )
    else:
        flow_group_summary = exclude_agg.loc[exclude_agg["flow_group"] != "all"].copy()
    flow_group_summary.to_csv(summaries_dir / "imputation_quality_by_flow_group.csv", index=False, encoding="utf-8-sig")
    figures_dir = ensure_directory(args.output_dir / "figures")
    single_rate = get_single_rate_value(summary_main)
    method_order = get_single_rate_method_order(summary_main)
    plot_main_rmse(summary_main, args.output_dir)
    plot_single_rate_metric_bars(
        summary_df=summary_main,
        output_dir=args.output_dir,
        metric="MAE",
        title="Single missing rate = {0:.0%}: MAE by imputation method",
        file_stub="mae_by_method_all6",
        method_order=method_order,
        ylabel="MAE",
    )
    secondary_metric_label, secondary_metric_stub = get_single_rate_secondary_metric(summary_main)
    plot_single_rate_metric_bars(
        summary_df=summary_main,
        output_dir=args.output_dir,
        metric=secondary_metric_label,
        title=f"Single missing rate = {{0:.0%}}: {secondary_metric_label} by imputation method",
        file_stub=f"{secondary_metric_stub}_by_method_all6",
        method_order=method_order,
        ylabel=secondary_metric_label,
    )
    plot_single_rate_metric_bars(
        summary_df=summary_main,
        output_dir=args.output_dir,
        metric="RMSE",
        title="Single missing rate = {0:.0%}: RMSE by method, excluding zero fill for readability",
        file_stub="rmse_by_method_nonzero_zoom",
        method_order=method_order,
        ylabel="RMSE",
        exclude_methods={"zero_fill"},
    )
    plot_flow_group_rmse(flow_group_summary, args.output_dir)
    if single_rate is not None:
        figure_index_rows = build_single_rate_figure_index_rows(summary_main, secondary_metric_label)
        pd.DataFrame(figure_index_rows).to_csv(
            figures_dir / f"single_rate_{build_single_rate_tag(single_rate)}_figure_index.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if get_single_rate_value(summary_main) is None:
        plot_zoom_rmse(summary_main, args.output_dir)
    batch_report = {
        "total_chunks_selected": int(len(chunk_df)),
        "generate_missing_status": generate_status,
        "impute_status": impute_status,
        "all_chunks_covered_in_generate": bool(
            len(chunk_df) == 0 or generate_status["unique_chunks"] == len(chunk_df)
        ),
        "all_chunks_covered_in_impute": bool(
            len(chunk_df) == 0 or impute_status["unique_chunks"] == len(chunk_df)
        ),
        "detail_row_count": int(len(detail_df)),
        "summary_all_row_count": int(len(summary_all)),
        "summary_exclude_warmup_row_count": int(len(summary_main)),
    }
    write_json(summaries_dir / "batch_processing_report.json", batch_report)
    batch_lines = [
        "# Batch Processing Report",
        "",
        f"- total_chunks_selected: `{batch_report['total_chunks_selected']}`",
        f"- generate_completed: `{generate_status['completed']}`",
        f"- generate_skipped_existing: `{generate_status['skipped_existing']}`",
        f"- impute_completed: `{impute_status['completed']}`",
        f"- impute_skipped_existing: `{impute_status['skipped_existing']}`",
        f"- all_chunks_covered_in_generate: `{str(batch_report['all_chunks_covered_in_generate']).lower()}`",
        f"- all_chunks_covered_in_impute: `{str(batch_report['all_chunks_covered_in_impute']).lower()}`",
        f"- detail_row_count: `{batch_report['detail_row_count']}`",
    ]
    write_markdown(summaries_dir / "batch_processing_report.md", batch_lines)
    payload = build_audit_payload(args, chunk_df, summary_all, summary_main)
    if single_rate is not None:
        rate_tag = build_single_rate_tag(single_rate)
        payload["figure_policy"] = {
            "formal_main_figures": [
                f"figures/single_rate_{rate_tag}_rmse_by_method_all6.png",
                f"figures/single_rate_{rate_tag}_mae_by_method_all6.png",
                f"figures/single_rate_{rate_tag}_{secondary_metric_stub}_by_method_all6.png",
                f"figures/single_rate_{rate_tag}_flow_group_rmse_by_method_all6.png",
            ],
            "auxiliary_figures": [
                f"figures/single_rate_{rate_tag}_rmse_by_method_nonzero_zoom.png",
            ],
            "notes": [
                "Formal visualization uses direct six-method comparison on absolute metrics.",
                "Forward fill is one of the six imputation methods and is not used as the formal baseline method.",
                "Flow-group RMSE compares low-flow, mid-flow, and high-flow nodes without using forward fill as a baseline.",
            ],
        }
    write_json(args.output_dir / "full_intersection_missingness_audit.json", payload)
    lines = [
        "# Full Intersection Missingness Audit",
        "",
        "## 1. Historical Causal Constraint",
        "",
        "- 本轮补全实验采用历史因果约束。对于目标日期 D 和目标时间片 t，补全方法仅允许使用 D 日 t 之前的观测以及 D 日之前的历史观测，不使用 D 日 t 之后、D+1 或更晚日期的数据。",
        f"- causal_history_only: `{str(args.causal_history_only).lower()}`",
        f"- history_days: `{args.history_days}`",
        f"- context_days_after: `{args.context_days_after}`",
        "- uses_future_days: `false`",
        "- uses_same_day_future_slots: `false`",
        "- uses_bfill: `false`",
        "- uses_bidirectional_interpolation: `false`",
        f"- warmup_days: `{args.warmup_days}`",
        f"- main_metrics_exclude_warmup: `{str(args.exclude_warmup_from_main_metrics).lower()}`",
        "",
        "## 2. Global Time Index",
        "",
        "- 全局时间索引构造方式：`global_time_index = day_index * 96 + time_slot`",
        "- `day_index` 来自输入 chunk 顺序。",
        "- `time_slot` 来自 `时间段` 字段解析；若原字段不是 0-95 整数，则按日内排序映射。",
        "",
        "## 3. Summaries",
        "",
        f"- `imputation_quality_summary_all_days.csv` 行数：`{len(summary_all)}`",
        f"- `imputation_quality_summary_exclude_warmup.csv` 行数：`{len(summary_main)}`",
        f"- `imputation_quality_by_flow_group.csv` 行数：`{len(flow_group_summary)}`",
        "",
        "## 4. Batch Coverage",
        "",
        f"- 选中的 chunk 总数：`{len(chunk_df)}`",
        f"- `generate_missing` 已完成或跳过的 chunk 数：`{generate_status['completed'] + generate_status['skipped_existing']}`",
        f"- `impute` 已完成或跳过的 chunk 数：`{impute_status['completed'] + impute_status['skipped_existing']}`",
        f"- `generate_missing` 是否覆盖全部 chunk：`{str(batch_report['all_chunks_covered_in_generate']).lower()}`",
        f"- `impute` 是否覆盖全部 chunk：`{str(batch_report['all_chunks_covered_in_impute']).lower()}`",
    ]
    if single_rate is not None:
        rate_tag = build_single_rate_tag(single_rate)
        lines.extend(
            [
                "",
                "## 5. Figure Policy",
                "",
                "- 正式可视化已调整为 6 个方法在绝对指标上的直接比较图。",
                "- `forward_fill` 不再作为正式图的参照基准，而是作为 6 个方法之一参与比较。",
                f"- 正式主图：`figures\\single_rate_{rate_tag}_rmse_by_method_all6.png`",
                f"- 正式主图：`figures\\single_rate_{rate_tag}_mae_by_method_all6.png`",
                f"- 正式主图：`figures\\single_rate_{rate_tag}_{secondary_metric_stub}_by_method_all6.png`",
                f"- 正式主图：`figures\\single_rate_{rate_tag}_flow_group_rmse_by_method_all6.png`",
                f"- 辅助图：`figures\\single_rate_{rate_tag}_rmse_by_method_nonzero_zoom.png`",
                "- 正式结果采用 6 个方法的 RMSE、MAE、sMAPE/MAPE 绝对指标并列比较。",
            ]
        )
    write_markdown(args.output_dir / "full_intersection_missingness_audit.md", lines)


def main() -> None:
    args = normalize_args(parse_args())
    if args.stage == "generate_missing":
        run_generate_missing(args)
    elif args.stage == "impute":
        run_impute(args)
    elif args.stage == "validate":
        run_validate(args)
    elif args.stage == "summarize":
        run_summarize(args)
    else:  # pragma: no cover - argparse enforces this
        raise ValueError(f"不支持的 stage: {args.stage}")


if __name__ == "__main__":
    main()
