from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency fallback
    pq = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "real_data_missingness_experiments_geo_func"
DEFAULT_INPUT_PATTERN = "node_flow_chunk_*.parquet"
DEFAULT_TOPOLOGY_PATH = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"
DEFAULT_TARGET_CANDIDATES = ["路口车流量", "flow", "traffic_flow", "target", "y"]
DEFAULT_TIME_CANDIDATES = ["时间段", "timestamp", "time", "datetime", "date", "day_slot"]
DEFAULT_NODE_CANDIDATES = ["节点ID", "node_id", "sensor_id", "detector_id", "station_id"]
DEFAULT_MISSING_RATES = [0.0, 0.05, 0.10]
DEFAULT_MECHANISMS = ["mcar_point"]
DEFAULT_SEEDS = [42]
DEFAULT_IMPUTE_METHODS = ["zero_fill", "forward_fill", "linear_interpolation"]
DEFAULT_MAX_FILES = 2
DEFAULT_MAX_ROWS = 500


@dataclass
class ExperimentDesignRecord:
    file_name: str
    file_path: str
    rows_used: int
    mechanism: str
    missing_rate: float
    seed: int
    target_col: str
    time_col: Optional[str]
    node_col: Optional[str]
    mask_path: Optional[str]


@dataclass
class MaskSummaryRecord:
    file_name: str
    file_path: str
    rows_used: int
    mechanism: str
    missing_rate: float
    seed: int
    target_col: str
    time_col: Optional[str]
    node_col: Optional[str]
    requested_missing_count: int
    actual_missing_count: int
    actual_missing_rate: float
    mask_path: Optional[str]
    corrupted_path: Optional[str]


@dataclass
class ImputationQualityRecord:
    file_name: str
    file_path: str
    rows_used: int
    mechanism: str
    missing_rate: float
    seed: int
    impute_method: str
    target_col: str
    time_col: Optional[str]
    node_col: Optional[str]
    masked_count: int
    actual_missing_rate: float
    imputation_mae: Optional[float]
    imputation_rmse: Optional[float]
    imputation_mape: Optional[float]
    output_path: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在完整真实数据基础上人为注入缺失值，并评估简单插补策略。"
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input_pattern", type=str, default=DEFAULT_INPUT_PATTERN)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--node_col", type=str, default=None)
    parser.add_argument("--missing_rates", type=str, default=",".join(str(x) for x in DEFAULT_MISSING_RATES))
    parser.add_argument("--mechanisms", type=str, default=",".join(DEFAULT_MECHANISMS))
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--impute_methods", type=str, default=",".join(DEFAULT_IMPUTE_METHODS))
    parser.add_argument("--max_files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="每个输入分片最多读取的行数；设为 0 或负数表示不限制。",
    )
    parser.add_argument("--write_corrupted", action="store_true", default=False)
    parser.add_argument("--write_imputed", action="store_true", default=False)
    parser.add_argument("--save_masks", action="store_true", default=False)
    # 地理邻近性与函数曲线拟合相关参数
    parser.add_argument(
        "--topology_path",
        type=Path,
        default=DEFAULT_TOPOLOGY_PATH,
        help="用于地理邻近性填补的路网拓扑文件。",
    )
    parser.add_argument(
        "--geo_lambda",
        type=float,
        default=0.5,
        help="geo_func_hybrid 中地理邻近性填补结果的权重。",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=96,
        help="函数曲线拟合中的周期长度。真实数据每日 96 个时间片。",
    )
    parser.add_argument(
        "--fourier_order",
        type=int,
        default=2,
        help="函数曲线拟合使用的傅里叶阶数。",
    )
    parser.add_argument(
        "--min_fit_points",
        type=int,
        default=8,
        help="函数曲线拟合时每个节点至少需要的非缺失观测点数。",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="head",
        choices=["head", "node_subset"],
        help="样本抽取方式。函数曲线拟合推荐 node_subset。",
    )
    parser.add_argument(
        "--sample_nodes",
        type=int,
        default=200,
        help="node_subset 模式下抽取的节点数量。",
    )
    return parser.parse_args()


def parse_float_list(raw_text: str) -> list[float]:
    return [float(item.strip()) for item in raw_text.split(",") if item.strip()]


def parse_int_list(raw_text: str) -> list[int]:
    return [int(item.strip()) for item in raw_text.split(",") if item.strip()]


def parse_str_list(raw_text: str) -> list[str]:
    return [item.strip() for item in raw_text.split(",") if item.strip()]


def find_first_existing_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    existing = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        match = existing.get(candidate.lower())
        if match is not None:
            return match
    for column in columns:
        col_text = str(column).lower()
        for candidate in candidates:
            if candidate.lower() in col_text:
                return str(column)
    return None


def detect_target_column(columns: Iterable[str]) -> Optional[str]:
    return find_first_existing_column(columns, DEFAULT_TARGET_CANDIDATES)


def detect_time_column(columns: Iterable[str]) -> Optional[str]:
    return find_first_existing_column(columns, DEFAULT_TIME_CANDIDATES)


def detect_node_column(columns: Iterable[str]) -> Optional[str]:
    return find_first_existing_column(columns, DEFAULT_NODE_CANDIDATES)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def list_input_files(input_dir: Path, pattern: str, max_files: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到输入文件: {input_dir / pattern}")
    if max_files > 0:
        return files[:max_files]
    return files


def get_parquet_columns(file_path: Path) -> list[str]:
    if pq is not None:
        return list(pq.ParquetFile(file_path).schema.names)
    df = pd.read_parquet(file_path)
    return list(df.columns)


def resolve_columns(
    available_columns: list[str],
    target_col: Optional[str],
    time_col: Optional[str],
    node_col: Optional[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    resolved_target = target_col or detect_target_column(available_columns)
    resolved_time = time_col or detect_time_column(available_columns)
    resolved_node = node_col or detect_node_column(available_columns)

    if resolved_target is None:
        raise ValueError(
            "未能自动识别目标字段，请显式传入 `--target_col`。当前字段为: "
            + ", ".join(available_columns)
        )
    return resolved_target, resolved_time, resolved_node


def read_input_frame(
    file_path: Path,
    target_col: str,
    time_col: Optional[str],
    node_col: Optional[str],
    max_rows: Optional[int],
    sample_mode: str = "head",
    sample_nodes: int = 200,
) -> pd.DataFrame:
    columns = [target_col]
    if time_col:
        columns.append(time_col)
    if node_col and node_col not in columns:
        columns.append(node_col)

    df = pd.read_parquet(file_path, columns=columns)

    if sample_mode == "node_subset" and node_col is not None:
        # 从节点列表中抽取 sample_nodes 个节点，保留这些节点的全部记录
        all_nodes = df[node_col].unique()
        if len(all_nodes) > sample_nodes:
            rng = np.random.RandomState(42)
            selected = rng.choice(all_nodes, size=sample_nodes, replace=False)
            df = df[df[node_col].isin(selected)].copy()
        # 如果节点数不足，保留全部
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows).copy()
    else:
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows).copy()
        else:
            df = df.copy()

    if time_col and node_col:
        df = df.sort_values([node_col, time_col]).reset_index(drop=True)
    elif time_col:
        df = df.sort_values([time_col]).reset_index(drop=True)
    elif node_col:
        df = df.sort_values([node_col]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 地理邻近性填补：拓扑加载
# ---------------------------------------------------------------------------

def load_neighbor_edges(topology_path: Path) -> pd.DataFrame:
    """从路网拓扑文件加载节点邻接关系。

    优先识别字段：起始节点ID, 结束节点ID, 长度。
    兼容字段：snodeid/enodeid/length, start_node/end_node/length,
             source/target/distance, from/to/dist。
    输出 DataFrame 包含 node_id, neighbor_id, weight 三列。
    """
    edge_candidates = [
        (["起始节点ID", "结束节点ID", "长度"], "length"),
        (["snodeid", "enodeid", "length"], "length"),
        (["start_node", "end_node", "length"], "length"),
        (["source", "target", "distance"], "distance"),
        (["from", "to", "dist"], "dist"),
        (["起始节点ID", "结束节点ID"], None),
        (["snodeid", "enodeid"], None),
        (["start_node", "end_node"], None),
        (["source", "target"], None),
        (["from", "to"], None),
    ]

    if not topology_path.exists():
        print(f"[WARN] 拓扑文件不存在: {topology_path}，地理填补将回退。")
        return pd.DataFrame(columns=["node_id", "neighbor_id", "weight"])

    df = pd.read_csv(topology_path, encoding="utf-8")
    columns_lower = {c.lower(): c for c in df.columns}

    src_col = None
    dst_col = None
    len_col = None
    has_length = False

    for sc in ["起始节点ID", "snodeid", "start_node", "source", "from"]:
        if sc.lower() in columns_lower:
            src_col = columns_lower[sc.lower()]
            break
    for dc in ["结束节点ID", "enodeid", "end_node", "target", "to"]:
        if dc.lower() in columns_lower:
            dst_col = columns_lower[dc.lower()]
            break
    for lc in ["长度", "length", "distance", "dist"]:
        if lc.lower() in columns_lower:
            len_col = columns_lower[lc.lower()]
            has_length = True
            break

    if src_col is None or dst_col is None:
        print(f"[WARN] 拓扑文件中未找到起始/结束节点字段，可用字段: {list(df.columns)}")
        return pd.DataFrame(columns=["node_id", "neighbor_id", "weight"])

    edges_list: list = []
    for _, row in df.iterrows():
        a = row[src_col]
        b = row[dst_col]
        if pd.isna(a) or pd.isna(b):
            continue
        a = str(int(a)) if isinstance(a, (int, float)) else str(a)
        b = str(int(b)) if isinstance(b, (int, float)) else str(b)
        if a == b:
            continue
        w = 1.0
        if has_length and len_col is not None:
            val = row[len_col]
            if pd.notna(val) and float(val) > 0:
                w = 1.0 / max(float(val), 1e-6)
        edges_list.append({"node_id": a, "neighbor_id": b, "weight": w})
        edges_list.append({"node_id": b, "neighbor_id": a, "weight": w})

    if not edges_list:
        return pd.DataFrame(columns=["node_id", "neighbor_id", "weight"])

    edges_df = pd.DataFrame(edges_list)
    edges_df = edges_df.groupby(["node_id", "neighbor_id"], as_index=False)["weight"].mean()
    print(f"[INFO] 已加载 {len(edges_df)} 条邻接边，覆盖 {edges_df['node_id'].nunique()} 个节点。")
    return edges_df


# ---------------------------------------------------------------------------
# 傅里叶基函数矩阵
# ---------------------------------------------------------------------------

def build_fourier_design_matrix(
    time_values: np.ndarray,
    period: int,
    order: int,
) -> np.ndarray:
    """构造傅里叶设计矩阵。

    基函数包括：1, t_norm, sin(2pi*k*t/P), cos(2pi*k*t/P), k=1..order。
    """
    n = len(time_values)
    cols = 2 + 2 * order  # constant + trend + sin/cos pairs
    X = np.empty((n, cols), dtype=np.float64)
    P = float(period)
    t = time_values.astype(np.float64)
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1.0)
    X[:, 0] = 1.0
    X[:, 1] = t_norm
    for k in range(1, order + 1):
        omega = 2.0 * np.pi * k / P
        X[:, 2 * k] = np.sin(omega * t)
        X[:, 2 * k + 1] = np.cos(omega * t)
    return X


# ---------------------------------------------------------------------------
# 地理邻近性填补
# ---------------------------------------------------------------------------

def geo_neighbor_impute(
    corrupted: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    node_col: Optional[str],
    neighbor_edges: pd.DataFrame,
) -> pd.Series:
    """使用地理邻近性填补缺失值。

    对缺失位置 (i, t)，使用节点 i 的邻居在时间 t 的非缺失观测的
    距离加权均值进行填补。若无可用邻居，回退至 forward_fill；若仍失败，
    回退至全局中位数。
    """
    series = corrupted[target_col].copy()
    missing_mask = series.isna()
    if not missing_mask.any():
        return series

    if neighbor_edges.empty or node_col is None:
        fallback_df = corrupted.copy()
        fallback_df["__ro__"] = np.arange(len(fallback_df))
        if node_col and time_col:
            fallback_df = fallback_df.sort_values([node_col, time_col, "__ro__"]).reset_index(drop=True)
        elif time_col:
            fallback_df = fallback_df.sort_values([time_col, "__ro__"]).reset_index(drop=True)
        fallback_df[target_col] = fallback_df.groupby(node_col, sort=False)[target_col].transform(
            lambda s: s.ffill().bfill()
        ) if node_col else fallback_df[target_col].ffill().bfill()
        fallback_df[target_col] = fallback_df[target_col].fillna(
            float(pd.to_numeric(corrupted[target_col], errors="coerce").median())
        )
        fb = fallback_df.sort_values("__ro__")[target_col].values
        fb_series = pd.Series(fb, index=series.index)
        series[missing_mask] = fb_series[missing_mask]
        return series

    neighbor_weight: dict = {}
    for _, row in neighbor_edges.iterrows():
        nid = str(row["node_id"])
        nbr = str(row["neighbor_id"])
        w = float(row["weight"])
        neighbor_weight.setdefault(nid, {})[nbr] = w

    global_median = float(pd.to_numeric(series, errors="coerce").median())
    if pd.isna(global_median):
        global_median = 0.0

    time_to_obs = {}
    if time_col is not None:
        for idx, row_data in corrupted.iterrows():
            if pd.notna(row_data[target_col]):
                t = row_data[time_col]
                n = str(row_data[node_col]) if node_col else "__all__"
                time_to_obs.setdefault(t, {})[n] = float(row_data[target_col])

    filled_count = 0
    for idx in series[missing_mask].index:
        node = str(corrupted.loc[idx, node_col]) if node_col else "__all__"
        t = corrupted.loc[idx, time_col] if time_col else None

        geo_val = None
        if node in neighbor_weight and time_col is not None and t is not None and t in time_to_obs:
            neighbors = neighbor_weight[node]
            weighted_sum = 0.0
            weight_sum = 0.0
            for nbr, w in neighbors.items():
                if nbr in time_to_obs[t]:
                    weighted_sum += w * time_to_obs[t][nbr]
                    weight_sum += w
            if weight_sum > 0:
                geo_val = float(weighted_sum / weight_sum)
                filled_count += 1

        if geo_val is not None and np.isfinite(geo_val):
            series.iloc[series.index.get_loc(idx)] = geo_val

    still_missing = series.isna()
    if still_missing.any():
        fb = corrupted.copy()
        fb["__ro__"] = np.arange(len(fb))
        if node_col and time_col:
            fb = fb.sort_values([node_col, time_col, "__ro__"]).reset_index(drop=True)
        elif time_col:
            fb = fb.sort_values([time_col, "__ro__"]).reset_index(drop=True)
        if node_col:
            fb[target_col] = fb.groupby(node_col, sort=False)[target_col].transform(
                lambda s: s.ffill().bfill()
            )
        else:
            fb[target_col] = fb[target_col].ffill().bfill()
        fb[target_col] = fb[target_col].fillna(float(global_median))
        fb_series = pd.Series(fb.sort_values("__ro__")[target_col].values, index=series.index)
        series[still_missing] = fb_series[still_missing]

    return series


# ---------------------------------------------------------------------------
# 函数曲线拟合填补
# ---------------------------------------------------------------------------

def function_curve_fit_impute(
    corrupted: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    node_col: Optional[str],
    period: int = 96,
    fourier_order: int = 2,
    min_fit_points: int = 8,
) -> pd.Series:
    """使用函数曲线拟合填补缺失值。

    对每个节点的非缺失时间序列进行傅里叶基函数最小二乘拟合，
    在缺失位置用拟合曲线重构。拟合点数不足的节点回退到 linear_interpolation。
    """
    series = corrupted[target_col].copy()
    missing_mask = series.isna()
    if not missing_mask.any():
        return series

    global_median = float(pd.to_numeric(corrupted[target_col], errors="coerce").median())
    if pd.isna(global_median):
        global_median = 0.0

    if node_col is None or time_col is None:
        fb = corrupted.copy()
        fb["__ro__"] = np.arange(len(fb))
        fb[target_col] = fb[target_col].interpolate(method="linear", limit_direction="both")
        fb[target_col] = fb[target_col].ffill().bfill().fillna(float(global_median))
        series = pd.Series(fb.sort_values("__ro__")[target_col].values, index=series.index)
        return series

    time_series = corrupted[time_col]
    try:
        time_numeric = pd.to_numeric(time_series, errors="coerce")
    except Exception:
        time_codes = pd.Categorical(time_series).codes
        time_numeric = pd.Series(time_codes, index=time_series.index, dtype=float)

    node_str = corrupted[node_col].astype(str)
    filled = series.copy()
    fitted_count = 0
    skipped_count = 0

    for node, grp_idx in corrupted.groupby(node_str).groups.items():
        idx_list = list(grp_idx)
        obs_mask = series.loc[idx_list].notna()
        n_obs = int(obs_mask.sum())

        if n_obs < min_fit_points:
            skipped_count += 1
            continue

        t_vals = time_numeric.loc[idx_list].to_numpy(dtype=float)
        y_vals = series.loc[idx_list].to_numpy(dtype=float)
        t_obs = t_vals[obs_mask]
        y_obs = y_vals[obs_mask]

        try:
            X = build_fourier_design_matrix(t_obs, period, fourier_order)
            coeffs, _, _, _ = np.linalg.lstsq(X, y_obs, rcond=None)
            X_all = build_fourier_design_matrix(t_vals, period, fourier_order)
            y_fit = X_all @ coeffs
        except np.linalg.LinAlgError:
            continue

        miss_in_node = ~obs_mask
        if miss_in_node.any():
            miss_positions = list(miss_in_node[miss_in_node].index)
            for pos, fit_val in zip(miss_positions, y_fit[miss_in_node]):
                filled.loc[pos] = float(fit_val)
            fitted_count += 1

    still_missing = filled.isna()
    if still_missing.any():
        fb = corrupted.copy()
        fb["__ro__"] = np.arange(len(fb))
        if node_col and time_col:
            fb = fb.sort_values([node_col, time_col, "__ro__"]).reset_index(drop=True)
            fb[target_col] = fb.groupby(node_col, sort=False)[target_col].transform(
                lambda s: s.interpolate(method="linear", limit_direction="both")
            )
            fb[target_col] = fb[target_col].ffill().bfill()
        else:
            fb[target_col] = fb[target_col].interpolate(method="linear", limit_direction="both")
            fb[target_col] = fb[target_col].ffill().bfill()
        fb[target_col] = fb[target_col].fillna(float(global_median))
        fb_series = pd.Series(fb.sort_values("__ro__")[target_col].values, index=filled.index)
        filled[still_missing] = fb_series[still_missing]

    return filled


# ---------------------------------------------------------------------------
# 地理-函数混合填补
# ---------------------------------------------------------------------------

def geo_func_hybrid_impute(
    corrupted: pd.DataFrame,
    target_col: str,
    time_col: Optional[str],
    node_col: Optional[str],
    neighbor_edges: pd.DataFrame,
    geo_lambda: float,
    period: int,
    fourier_order: int,
    min_fit_points: int,
) -> pd.Series:
    """地理邻近性与函数曲线拟合混合填补。

    对原始缺失位置：
    - 两者都有效：加权融合 filled = lambda * geo + (1-lambda) * func
    - 仅地理有效：使用地理填补值
    - 仅函数有效：使用函数拟合值
    - 均无效：回退到 forward_fill 和全局中位数
    """
    original = corrupted[target_col].copy()
    missing_mask = original.isna()
    if not missing_mask.any():
        return original

    geo_series = geo_neighbor_impute(corrupted, target_col, time_col, node_col, neighbor_edges)
    func_series = function_curve_fit_impute(
        corrupted, target_col, time_col, node_col, period, fourier_order, min_fit_points
    )

    global_median = float(pd.to_numeric(corrupted[target_col], errors="coerce").median())
    if pd.isna(global_median):
        global_median = 0.0

    result = original.copy()
    for idx in result[missing_mask].index:
        geo_val = geo_series.loc[idx]
        func_val = func_series.loc[idx]
        geo_valid = pd.notna(geo_val)
        func_valid = pd.notna(func_val)

        if geo_valid and func_valid:
            result.loc[idx] = float(geo_lambda * float(geo_val) + (1.0 - geo_lambda) * float(func_val))
        elif geo_valid:
            result.loc[idx] = float(geo_val)
        elif func_valid:
            result.loc[idx] = float(func_val)

    still_missing = result.isna()
    if still_missing.any():
        fb = corrupted.copy()
        fb["__ro__"] = np.arange(len(fb))
        if node_col and time_col:
            fb = fb.sort_values([node_col, time_col, "__ro__"]).reset_index(drop=True)
        elif time_col:
            fb = fb.sort_values([time_col, "__ro__"]).reset_index(drop=True)
        if node_col:
            fb[target_col] = fb.groupby(node_col, sort=False)[target_col].transform(
                lambda s: s.ffill().bfill()
            )
        else:
            fb[target_col] = fb[target_col].ffill().bfill()
        fb[target_col] = fb[target_col].fillna(float(global_median))
        fb_series = pd.Series(fb.sort_values("__ro__")[target_col].values, index=result.index)
        result[still_missing] = fb_series[still_missing]

    return result

def stable_seed(base_seed: int, file_name: str, mechanism: str, missing_rate: float) -> int:
    payload = f"{file_name}|{mechanism}|{missing_rate:.4f}".encode("utf-8")
    return int(base_seed + zlib.adler32(payload)) % (2 ** 31 - 1)


def make_mask(
    df: pd.DataFrame,
    target_col: Optional[str],
    missing_rate: float,
    mechanism: str,
    seed: int,
    time_col: Optional[str],
    node_col: Optional[str],
) -> np.ndarray:
    if target_col is None:
        raise ValueError("target_col 不能为空。")
    if mechanism != "mcar_point":
        raise ValueError(f"当前仅支持 `mcar_point`，收到: {mechanism}")

    values = df[target_col]
    eligible_indices = np.flatnonzero(values.notna().to_numpy())
    mask = np.zeros(len(df), dtype=bool)
    if len(eligible_indices) == 0 or missing_rate <= 0:
        return mask

    missing_count = int(round(len(eligible_indices) * float(missing_rate)))
    if missing_count <= 0:
        return mask

    rng = np.random.RandomState(seed)
    selected = rng.choice(eligible_indices, size=missing_count, replace=False)
    mask[selected] = True
    return mask


def apply_mask(df: pd.DataFrame, target_col: str, mask: np.ndarray) -> pd.DataFrame:
    corrupted = df.copy()
    corrupted.loc[mask, target_col] = np.nan
    return corrupted


def restore_original_order(df: pd.DataFrame, working: pd.DataFrame) -> pd.DataFrame:
    restored = working.sort_values("__row_order__").drop(columns=["__row_order__"])
    restored.index = df.index
    return restored


def impute_target(
    df: pd.DataFrame,
    target_col: Optional[str],
    impute_method: str,
    time_col: Optional[str],
    node_col: Optional[str],
    neighbor_edges: Optional[pd.DataFrame] = None,
    geo_lambda: float = 0.5,
    period: int = 96,
    fourier_order: int = 2,
    min_fit_points: int = 8,
) -> pd.DataFrame:
    if target_col is None:
        raise ValueError("target_col 不能为空。")

    # 新方法：直接返回填补后的 Series，不经过原有 sort/restore 流程
    if impute_method == "geo_neighbor_fill":
        if neighbor_edges is None or neighbor_edges.empty:
            # 回退到 forward_fill
            filled_series = impute_target(
                df, target_col, "forward_fill", time_col, node_col
            )[target_col]
        else:
            filled_series = geo_neighbor_impute(
                df, target_col, time_col, node_col, neighbor_edges
            )
        result = df.copy()
        result[target_col] = filled_series
        return result

    if impute_method == "function_curve_fit":
        filled_series = function_curve_fit_impute(
            df, target_col, time_col, node_col,
            period=period, fourier_order=fourier_order, min_fit_points=min_fit_points,
        )
        result = df.copy()
        result[target_col] = filled_series
        return result

    if impute_method == "geo_func_hybrid":
        filled_series = geo_func_hybrid_impute(
            df, target_col, time_col, node_col,
            neighbor_edges=neighbor_edges if neighbor_edges is not None else pd.DataFrame(),
            geo_lambda=geo_lambda, period=period,
            fourier_order=fourier_order, min_fit_points=min_fit_points,
        )
        result = df.copy()
        result[target_col] = filled_series
        return result

    if impute_method == "zero_fill":
        working = df.copy()
        working[target_col] = working[target_col].fillna(0.0)
        return working

    fallback_value = pd.to_numeric(df[target_col], errors="coerce").dropna().median()
    if pd.isna(fallback_value):
        fallback_value = 0.0

    working = df.copy()
    working["__row_order__"] = np.arange(len(working))

    sort_columns: list[str] = []
    if node_col:
        sort_columns.append(node_col)
    if time_col:
        sort_columns.append(time_col)
    if sort_columns:
        working = working.sort_values(sort_columns + ["__row_order__"]).reset_index(drop=True)

    if impute_method == "forward_fill":
        if node_col:
            working[target_col] = (
                working.groupby(node_col, sort=False)[target_col]
                .transform(lambda s: s.ffill().bfill())
            )
        else:
            working[target_col] = working[target_col].ffill().bfill()
    elif impute_method == "linear_interpolation":
        if node_col:
            working[target_col] = (
                working.groupby(node_col, sort=False)[target_col]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )
            working[target_col] = working[target_col].ffill().bfill()
        else:
            working[target_col] = working[target_col].interpolate(
                method="linear",
                limit_direction="both",
            )
            working[target_col] = working[target_col].ffill().bfill()
    else:
        raise ValueError(f"不支持的插补方法: {impute_method}")

    working[target_col] = working[target_col].fillna(float(fallback_value))
    return restore_original_order(df, working)


def compute_imputation_metrics(
    original_values: pd.Series,
    imputed_values: pd.Series,
    mask: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if int(mask.sum()) == 0:
        return 0.0, 0.0, 0.0

    y_true = pd.to_numeric(original_values[mask], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(imputed_values[mask], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return None, None, None

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    abs_error = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    non_zero = np.abs(y_true) > 1e-8
    if non_zero.any():
        mape = float(np.mean(abs_error[non_zero] / np.abs(y_true[non_zero])) * 100.0)
    else:
        mape = 0.0
    return mae, rmse, mape


def write_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def aggregate_actual_missing_rates(mask_df: pd.DataFrame) -> pd.DataFrame:
    if mask_df.empty:
        return pd.DataFrame()
    return (
        mask_df.groupby(["mechanism", "missing_rate"], as_index=False)["actual_missing_rate"]
        .mean()
        .sort_values(["mechanism", "missing_rate"])
    )


def aggregate_imputation_quality(quality_df: pd.DataFrame) -> pd.DataFrame:
    if quality_df.empty:
        return pd.DataFrame()
    return (
        quality_df.groupby(["mechanism", "impute_method", "missing_rate"], as_index=False)[
            ["imputation_mae", "imputation_rmse", "imputation_mape"]
        ]
        .mean()
        .sort_values(["mechanism", "impute_method", "missing_rate"])
    )


def write_json_audit(
    output_path: Path,
    args: argparse.Namespace,
    selected_files: list[Path],
    detected_columns: dict[str, Optional[str]],
    design_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> Path:
    audit = {
        "environment": {
            "python_requirement": "Python 3.9.23 (analysis)",
            "python_path": r"E:\anaconda3\envs\analysis\python.exe",
            "python39_compatible": True,
        },
        "input": {
            "input_dir": get_relative_path(args.input_dir),
            "output_dir": get_relative_path(args.output_dir),
            "input_pattern": args.input_pattern,
            "selected_files": [get_relative_path(path) for path in selected_files],
            "max_files": args.max_files,
            "max_rows": args.max_rows,
            "mechanisms": parse_str_list(args.mechanisms),
            "missing_rates": parse_float_list(args.missing_rates),
            "seeds": parse_int_list(args.seeds),
            "impute_methods": parse_str_list(args.impute_methods),
        },
        "detected_columns": detected_columns,
        "summary": {
            "design_rows": int(len(design_df)),
            "mask_rows": int(len(mask_df)),
            "quality_rows": int(len(quality_df)),
            "actual_missing_rate_mean": (
                None if mask_df.empty else float(mask_df["actual_missing_rate"].mean())
            ),
            "imputation_rmse_mean": (
                None
                if quality_df.empty
                else float(pd.to_numeric(quality_df["imputation_rmse"], errors="coerce").mean())
            ),
        },
        "aggregates": {
            "actual_missing_rate": aggregate_actual_missing_rates(mask_df).to_dict(orient="records"),
            "imputation_quality": aggregate_imputation_quality(quality_df).to_dict(orient="records"),
        },
    }
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_audit_markdown(
    output_path: Path,
    args: argparse.Namespace,
    selected_files: list[Path],
    detected_columns: dict[str, Optional[str]],
    design_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> Path:
    agg_mask = aggregate_actual_missing_rates(mask_df)
    agg_quality = aggregate_imputation_quality(quality_df)

    lines: list[str] = []
    lines.append("# Real Data Missingness Experiment Audit")
    lines.append("")
    lines.append("## 1. Environment")
    lines.append("")
    lines.append("- Python path: `E:\\anaconda3\\envs\\analysis\\python.exe`")
    lines.append("- Python version target: `3.9.23`")
    lines.append("- Compatibility note: script avoids Python 3.10+ union syntax and avoids optional Markdown table dependencies.")
    lines.append("")
    lines.append("## 2. Inputs")
    lines.append("")
    lines.append(f"- Input directory: `{get_relative_path(args.input_dir)}`")
    lines.append(f"- Output directory: `{get_relative_path(args.output_dir)}`")
    lines.append(f"- Selected file count: `{len(selected_files)}`")
    lines.append(f"- Max files: `{args.max_files}`")
    lines.append(f"- Max rows per file: `{args.max_rows}`")
    lines.append(f"- Mechanisms: `{', '.join(parse_str_list(args.mechanisms))}`")
    lines.append(f"- Missing rates: `{', '.join(str(x) for x in parse_float_list(args.missing_rates))}`")
    lines.append(f"- Seeds: `{', '.join(str(x) for x in parse_int_list(args.seeds))}`")
    lines.append(f"- Impute methods: `{', '.join(parse_str_list(args.impute_methods))}`")
    lines.append("")
    lines.append("## 3. Detected Columns")
    lines.append("")
    for key, value in detected_columns.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## 4. Actual Missing Rate Summary")
    lines.append("")
    if agg_mask.empty:
        lines.append("- No mask summary generated.")
    else:
        lines.append("```text")
        lines.append(agg_mask.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## 5. Imputation Quality Summary")
    lines.append("")
    if agg_quality.empty:
        lines.append("- No imputation quality summary generated.")
    else:
        lines.append("```text")
        lines.append(agg_quality.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("## 6. Output Files")
    lines.append("")
    lines.append(f"- Design summary rows: `{len(design_df)}`")
    lines.append(f"- Mask summary rows: `{len(mask_df)}`")
    lines.append(f"- Quality summary rows: `{len(quality_df)}`")
    lines.append("- Figure: `figures/missing_rate_vs_imputation_rmse.png`")
    lines.append("- Figure: `figures/missing_rate_vs_imputation_rmse.pdf`")
    lines.append("")

    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# 图例标签映射
METHOD_LABEL_MAP = {
    "zero_fill": "Zero fill",
    "forward_fill": "Forward fill",
    "linear_interpolation": "Linear interpolation",
    "geo_neighbor_fill": "Geo-neighbor",
    "function_curve_fit": "Function curve",
    "geo_func_hybrid": "Geo-function hybrid",
}


def plot_missing_rate_vs_rmse(quality_df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    figures_dir = ensure_directory(output_dir / "figures")
    png_path = figures_dir / "missing_rate_vs_imputation_rmse.png"
    pdf_path = figures_dir / "missing_rate_vs_imputation_rmse.pdf"

    agg = aggregate_imputation_quality(quality_df)
    plt.figure(figsize=(8, 5))
    if not agg.empty:
        for (mechanism, impute_method), group in agg.groupby(["mechanism", "impute_method"]):
            plt.plot(
                group["missing_rate"],
                group["imputation_rmse"],
                marker="o",
                linewidth=1.8,
                label=f"{mechanism}-{impute_method}",
            )
    plt.xlabel("Missing rate")
    # 横轴显示百分比
    ax = plt.gca()
    ax.set_xticklabels([f"{int(x*100)}%" for x in ax.get_xticks()])
    plt.ylabel("Imputation RMSE")
    plt.title("Imputation RMSE under Artificial Missing Rates\nReal Data (Geo + Func)")
    plt.grid(True, linestyle="--", alpha=0.35)
    if not agg.empty:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    return png_path, pdf_path


def maybe_write_mask(mask: np.ndarray, output_dir: Path, file_stub: str) -> Path:
    path = ensure_directory(output_dir / "masks") / f"{file_stub}_mask.npy"
    np.save(path, mask)
    return path


def maybe_write_dataframe(df: pd.DataFrame, output_dir: Path, subdir: str, file_stub: str) -> Path:
    path = ensure_directory(output_dir / subdir) / f"{file_stub}.parquet"
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT_DIR / args.input_dir
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir
    ensure_directory(args.output_dir)

    # 加载路网拓扑
    topology_path = args.topology_path
    if not topology_path.is_absolute():
        topology_path = ROOT_DIR / topology_path
    print(f"[INFO] 加载拓扑文件: {topology_path}")
    neighbor_edges = load_neighbor_edges(topology_path)

    summaries_dir = ensure_directory(args.output_dir / "summaries")

    selected_files = list_input_files(args.input_dir, args.input_pattern, args.max_files)
    first_columns = get_parquet_columns(selected_files[0])
    target_col, time_col, node_col = resolve_columns(first_columns, args.target_col, args.time_col, args.node_col)
    detected_columns = {
        "target_col": target_col,
        "time_col": time_col,
        "node_col": node_col,
    }

    missing_rates = parse_float_list(args.missing_rates)
    mechanisms = parse_str_list(args.mechanisms)
    seeds = parse_int_list(args.seeds)
    impute_methods = parse_str_list(args.impute_methods)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    design_records: list[ExperimentDesignRecord] = []
    mask_records: list[MaskSummaryRecord] = []
    quality_records: list[ImputationQualityRecord] = []

    for file_path in selected_files:
        file_columns = get_parquet_columns(file_path)
        file_target_col, file_time_col, file_node_col = resolve_columns(
            file_columns,
            target_col,
            time_col,
            node_col,
        )
        df = read_input_frame(file_path, file_target_col, file_time_col, file_node_col, max_rows=max_rows)
        if df.empty:
            continue

        eligible_count = int(df[file_target_col].notna().sum())
        rel_file_path = get_relative_path(file_path)
        file_name = file_path.name

        for mechanism in mechanisms:
            for missing_rate in missing_rates:
                for seed in seeds:
                    run_seed = stable_seed(seed, file_name, mechanism, missing_rate)
                    mask = make_mask(
                        df=df,
                        target_col=file_target_col,
                        missing_rate=missing_rate,
                        mechanism=mechanism,
                        seed=run_seed,
                        time_col=file_time_col,
                        node_col=file_node_col,
                    )
                    corrupted = apply_mask(df, file_target_col, mask)
                    actual_missing_count = int(mask.sum())
                    actual_missing_rate = float(actual_missing_count / eligible_count) if eligible_count else 0.0
                    requested_missing_count = int(round(eligible_count * missing_rate))
                    file_stub = f"{file_path.stem}_seed{seed}_{mechanism}_{str(missing_rate).replace('.', 'p')}"

                    mask_path: Optional[str] = None
                    corrupted_path: Optional[str] = None
                    if args.save_masks:
                        mask_path = get_relative_path(maybe_write_mask(mask, args.output_dir, file_stub))
                    if args.write_corrupted:
                        corrupted_path = get_relative_path(
                            maybe_write_dataframe(corrupted, args.output_dir, "corrupted", f"{file_stub}_corrupted")
                        )

                    design_records.append(
                        ExperimentDesignRecord(
                            file_name=file_name,
                            file_path=rel_file_path,
                            rows_used=int(len(df)),
                            mechanism=mechanism,
                            missing_rate=float(missing_rate),
                            seed=int(seed),
                            target_col=file_target_col,
                            time_col=file_time_col,
                            node_col=file_node_col,
                            mask_path=mask_path,
                        )
                    )
                    mask_records.append(
                        MaskSummaryRecord(
                            file_name=file_name,
                            file_path=rel_file_path,
                            rows_used=int(len(df)),
                            mechanism=mechanism,
                            missing_rate=float(missing_rate),
                            seed=int(seed),
                            target_col=file_target_col,
                            time_col=file_time_col,
                            node_col=file_node_col,
                            requested_missing_count=requested_missing_count,
                            actual_missing_count=actual_missing_count,
                            actual_missing_rate=actual_missing_rate,
                            mask_path=mask_path,
                            corrupted_path=corrupted_path,
                        )
                    )

                    for impute_method in impute_methods:
                        if actual_missing_count == 0:
                            mae, rmse, mape = 0.0, 0.0, 0.0
                            imputed = corrupted
                        else:
                            imputed = impute_target(
                                df=corrupted,
                                target_col=file_target_col,
                                impute_method=impute_method,
                                time_col=file_time_col,
                                node_col=file_node_col,
                                neighbor_edges=neighbor_edges,
                                geo_lambda=args.geo_lambda,
                                period=args.period,
                                fourier_order=args.fourier_order,
                                min_fit_points=args.min_fit_points,
                            )
                            mae, rmse, mape = compute_imputation_metrics(
                                original_values=df[file_target_col],
                                imputed_values=imputed[file_target_col],
                                mask=mask,
                            )
                        output_path: Optional[str] = None
                        if args.write_imputed:
                            output_path = get_relative_path(
                                maybe_write_dataframe(imputed, args.output_dir, "imputed", f"{file_stub}_{impute_method}")
                            )
                        quality_records.append(
                            ImputationQualityRecord(
                                file_name=file_name,
                                file_path=rel_file_path,
                                rows_used=int(len(df)),
                                mechanism=mechanism,
                                missing_rate=float(missing_rate),
                                seed=int(seed),
                                impute_method=impute_method,
                                target_col=file_target_col,
                                time_col=file_time_col,
                                node_col=file_node_col,
                                masked_count=actual_missing_count,
                                actual_missing_rate=actual_missing_rate,
                                imputation_mae=mae,
                                imputation_rmse=rmse,
                                imputation_mape=mape,
                                output_path=output_path,
                            )
                        )

    design_df = pd.DataFrame([asdict(record) for record in design_records])
    mask_df = pd.DataFrame([asdict(record) for record in mask_records])
    quality_df = pd.DataFrame([asdict(record) for record in quality_records])

    design_path = write_dataframe(design_df, summaries_dir / "missingness_design_summary.csv")
    mask_path = write_dataframe(mask_df, summaries_dir / "missingness_mask_summary.csv")
    quality_path = write_dataframe(quality_df, summaries_dir / "imputation_quality_summary.csv")
    png_path, pdf_path = plot_missing_rate_vs_rmse(quality_df, args.output_dir)
    audit_md_path = write_audit_markdown(
        args.output_dir / "real_data_missingness_experiment_audit.md",
        args,
        selected_files,
        detected_columns,
        design_df,
        mask_df,
        quality_df,
    )
    audit_json_path = write_json_audit(
        args.output_dir / "real_data_missingness_experiment_audit.json",
        args,
        selected_files,
        detected_columns,
        design_df,
        mask_df,
        quality_df,
    )

    print(f"Design summary: {design_path}")
    print(f"Mask summary: {mask_path}")
    print(f"Imputation quality summary: {quality_path}")
    print(f"Audit markdown: {audit_md_path}")
    print(f"Audit json: {audit_json_path}")
    print(f"Figure PNG: {png_path}")
    print(f"Figure PDF: {pdf_path}")


if __name__ == "__main__":
    main()
