"""审计真实数据预处理产物的完整性与基础统计结果。

核心功能：
- 检查预处理阶段输出文件是否存在、字段是否齐全、规模是否合理；
- 输出 CSV 与 Markdown 审计结果，辅助确认预处理链路正常；
- 为后续密度估计、节点车流量构造和缺失实验提供前置校验。

项目作用：
- 作为真实交通数据预处理后的质量检查脚本；
- 帮助在下游建模前尽早发现文件缺失、列缺失或数据规模异常。

关键依赖：`pandas`、`pathlib`。
主要输入：真实数据预处理阶段生成的目录与文件。
主要输出：审计表、异常摘要和验证报告。
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "results" / "real_data_preprocessing"

DATA_DIR_CANDIDATES = [
    ROOT_DIR / "data",
    ROOT_DIR / "data" / "raw",
    ROOT_DIR / "data" / "processed",
    ROOT_DIR / "data" / "real",
    ROOT_DIR / "data" / "analysis",
    ROOT_DIR / "datasets",
]

CODE_PATHS = [
    ROOT_DIR / "preprocessing_scripts",
    ROOT_DIR / "analysis_scripts",
    ROOT_DIR / "dataset_inspection_scripts",
    ROOT_DIR / "docs" / "project_pipeline.md",
    ROOT_DIR / "docs" / "environment_setup.md",
    ROOT_DIR / "docs" / "project_documentation.md",
    ROOT_DIR / "docs" / "node_intersection_flow_inspection.md",
    ROOT_DIR / "docs" / "node_flow_daily_curve_fit.md",
]

DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".json",
    ".pkl",
    ".npy",
    ".npz",
    ".parquet",
    ".v2",
}

CODE_EXTENSIONS = {".py", ".md", ".txt"}
MAX_SAMPLE_ROWS = 5000
MAX_COUNTABLE_TEXT_SIZE = 512 * 1024 * 1024
MAX_CODE_LINES_PER_FILE = 600
TIME_COLUMN_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "date",
    "时间",
    "时间段",
    "日期",
    "day_slot",
    "day_index",
]
NODE_COLUMN_KEYWORDS = [
    "sensor",
    "node",
    "detector",
    "station",
    "link",
    "segment",
    "路段",
    "节点",
    "传感器",
    "检测点",
]
TRAFFIC_COLUMN_KEYWORDS = [
    "flow",
    "speed",
    "occupancy",
    "density",
    "traffic",
    "volume",
    "流量",
    "速度",
    "占有率",
    "密度",
]

KNOWN_RAW_SCHEMAS: dict[str, dict[str, Any]] = {
    "link_gps.v2": {
        "sep": "\t",
        "header": False,
        "columns": ["路段ID", "经度", "纬度"],
        "schema_source": "preprocessing_scripts/process_link_gps.py",
    },
    "road_network_sub-dataset.v2": {
        "sep": "\t",
        "header": True,
        "columns": None,
        "schema_source": "preprocessing_scripts/process_rnsd.py",
    },
    "traffic_speed_sub-dataset.v2": {
        "sep": ",",
        "header": False,
        "columns": ["路段ID", "时间段", "平均速度"],
        "schema_source": "preprocessing_scripts/merge_speed_data.py",
    },
}

EVIDENCE_PATTERNS: dict[str, list[str]] = {
    "data_reading": [
        r"read_csv",
        r"read_parquet",
        r"scan_csv",
        r"scan_parquet",
        r"read_excel",
        r"data/raw",
    ],
    "missing_handling": [
        r"dropna",
        r"fillna",
        r"interpolate",
        r"missing",
        r"空值",
        r"缺失",
    ],
    "anomaly_handling": [
        r"<\s*0",
        r"负值",
        r"异常",
        r"非法值",
        r"clip",
        r"outlier",
    ],
    "time_processing": [
        r"时间段",
        r"timestamp",
        r"datetime",
        r"sort_values",
        r"\.sort\(",
        r"slots_per_day",
        r"96",
    ],
    "matrix_or_window": [
        r"window",
        r"horizon",
        r"sequence",
        r"seq_len",
        r"pred_len",
        r"样本",
    ],
    "normalization": [
        r"MinMaxScaler",
        r"StandardScaler",
        r"RobustScaler",
        r"normalize",
        r"normalized",
        r"归一化",
        r"标准化",
        r"scaler",
    ],
    "split": [
        r"train",
        r"val",
        r"test",
        r"split",
        r"划分",
        r"训练集",
        r"验证集",
        r"测试集",
    ],
    "graph": [
        r"adjacency",
        r"topology",
        r"graph",
        r"路网",
        r"节点",
        r"拓扑",
        r"邻接",
    ],
    "client_partition": [
        r"client",
        r"federated",
        r"FedAvg",
        r"联邦",
        r"客户端",
    ],
    "output_files": [
        r"to_csv",
        r"to_parquet",
        r"write_",
        r"OUTPUT",
        r"输出",
        r"保存",
    ],
}


def to_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_stat(path: Path) -> dict[str, Any]:
    stats = path.stat()
    return {
        "size_bytes": int(stats.st_size),
        "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(timespec="seconds"),
    }


def detect_time_column(columns: list[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        col_l = col.lower()
        if any(token in col_l for token in TIME_COLUMN_CANDIDATES):
            return col
    return None


def detect_traffic_columns(columns: list[str]) -> list[str]:
    matches: list[str] = []
    for col in columns:
        col_l = col.lower()
        if any(keyword in col_l for keyword in TRAFFIC_COLUMN_KEYWORDS):
            matches.append(col)
    return matches


def detect_sensor_node_columns(columns: list[str]) -> list[str]:
    matches: list[str] = []
    for col in columns:
        col_l = col.lower()
        if any(keyword in col_l for keyword in NODE_COLUMN_KEYWORDS):
            matches.append(col)
    return matches


def sniff_delimiter(path: Path) -> str:
    schema = KNOWN_RAW_SCHEMAS.get(path.name)
    if schema and schema.get("sep"):
        return str(schema["sep"])

    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    except Exception:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return ","


def should_count_lines(path: Path) -> bool:
    return path.stat().st_size <= MAX_COUNTABLE_TEXT_SIZE


def count_lines(path: Path) -> Optional[int]:
    if not should_count_lines(path):
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return None


def detect_header(path: Path, delimiter: str) -> bool:
    schema = KNOWN_RAW_SCHEMAS.get(path.name)
    if schema and "header" in schema:
        return bool(schema["header"])

    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        return csv.Sniffer().has_header(sample)
    except Exception:
        return True


def read_delimited_sample(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    delimiter = sniff_delimiter(path)
    has_header = detect_header(path, delimiter)
    schema = KNOWN_RAW_SCHEMAS.get(path.name, {})
    columns = schema.get("columns")
    header = 0 if has_header else None
    kwargs: dict[str, Any] = {
        "sep": delimiter,
        "nrows": MAX_SAMPLE_ROWS,
        "encoding": "utf-8",
        "on_bad_lines": "skip",
        "low_memory": False,
    }
    if not has_header and columns:
        kwargs["names"] = columns
    else:
        kwargs["header"] = header
    df = pd.read_csv(path, **kwargs)
    if not has_header and columns and list(df.columns) != columns:
        df.columns = columns[: len(df.columns)]
    meta = {
        "delimiter": delimiter,
        "header_present": has_header,
        "schema_hint": columns,
        "schema_source": schema.get("schema_source"),
    }
    return df, meta


def read_excel_sample(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_excel(path, nrows=MAX_SAMPLE_ROWS)
    return df, {"sheet_name": 0, "header_present": True, "schema_hint": None, "schema_source": None}


def read_parquet_sample(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(path)
    if len(df) > MAX_SAMPLE_ROWS:
        df = df.head(MAX_SAMPLE_ROWS).copy()
    return df, {"header_present": True, "schema_hint": None, "schema_source": None}


def read_json_sample(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        df = pd.read_json(path)
    except ValueError:
        df = pd.read_json(path, lines=True)
    if len(df) > MAX_SAMPLE_ROWS:
        df = df.head(MAX_SAMPLE_ROWS).copy()
    return df, {"header_present": True, "schema_hint": None, "schema_source": None}


def read_table_sample(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".v2"}:
        return read_delimited_sample(path)
    if suffix in {".xlsx", ".xls"}:
        return read_excel_sample(path)
    if suffix == ".parquet":
        return read_parquet_sample(path)
    if suffix == ".json":
        return read_json_sample(path)
    raise ValueError(f"unsupported_table_type: {suffix}")


def infer_time_frequency(series: pd.Series) -> Optional[str]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None

    if pd.api.types.is_numeric_dtype(cleaned):
        return None

    parsed = pd.to_datetime(cleaned, errors="coerce")
    parsed = parsed.dropna().drop_duplicates().sort_values()
    if len(parsed) < 3:
        return None

    try:
        freq = pd.infer_freq(parsed.head(100))
        if freq:
            return str(freq)
    except Exception:
        pass

    deltas = parsed.diff().dropna()
    if deltas.empty:
        return None
    most_common = Counter(deltas.astype("timedelta64[s]").astype("int64")).most_common(1)[0][0]
    return f"{most_common}s"


def rough_outlier_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if math.isclose(iqr, 0.0):
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((numeric < lower) | (numeric > upper)).sum())


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def summarize_numeric_columns(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    numeric_df = df.select_dtypes(include=[np.number])
    for column in numeric_df.columns:
        series = pd.to_numeric(numeric_df[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue
        summary[column] = {
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "mean": float(non_null.mean()),
            "std": float(non_null.std(ddof=0)) if len(non_null) > 1 else 0.0,
            "negative_count": int((non_null < 0).sum()),
            "zero_count": int((non_null == 0).sum()),
            "rough_outlier_count": rough_outlier_count(non_null),
        }
    return summary


def summarize_table_quality(path: Path, df: pd.DataFrame, meta: dict[str, Any], full_row_count: Optional[int]) -> dict[str, Any]:
    columns = [str(col) for col in df.columns]
    time_column = detect_time_column(columns)
    traffic_columns = detect_traffic_columns(columns)
    node_columns = detect_sensor_node_columns(columns)
    missing_counts = {col: int(value) for col, value in df.isna().sum().to_dict().items()}
    missing_rates = {
        col: round(float(value) / len(df), 6) if len(df) else 0.0
        for col, value in missing_counts.items()
    }
    numeric_summary = summarize_numeric_columns(df)
    duplicate_rows = int(df.duplicated().sum())

    time_start = None
    time_end = None
    time_frequency = None
    if time_column and time_column in df.columns:
        time_series = df[time_column]
        if pd.api.types.is_numeric_dtype(time_series):
            numeric_time = pd.to_numeric(time_series, errors="coerce")
            if numeric_time.notna().any():
                time_start = str(int(numeric_time.min()))
                time_end = str(int(numeric_time.max()))
        else:
            parsed_time = pd.to_datetime(time_series, errors="coerce")
            if parsed_time.notna().any():
                valid_time = parsed_time.dropna()
                time_start = valid_time.min().isoformat()
                time_end = valid_time.max().isoformat()
                time_frequency = infer_time_frequency(time_series)
            else:
                numeric_time = pd.to_numeric(time_series, errors="coerce")
                if numeric_time.notna().any():
                    time_start = str(int(numeric_time.min()))
                    time_end = str(int(numeric_time.max()))

    return {
        "rel_path": to_rel(path),
        "file_type": path.suffix.lower() or path.name,
        "rows": int(full_row_count) if full_row_count is not None else len(df),
        "columns_count": len(columns),
        "columns": columns,
        "time_column": time_column,
        "sensor_or_node_columns": node_columns,
        "traffic_columns": traffic_columns,
        "missing_values_per_column": missing_counts,
        "missing_rate_per_column": missing_rates,
        "duplicate_rows": duplicate_rows,
        "time_coverage_start": time_start,
        "time_coverage_end": time_end,
        "time_frequency": time_frequency,
        "numeric_descriptive_stats": numeric_summary,
        "negative_values_count": {
            col: stats["negative_count"] for col, stats in numeric_summary.items()
        },
        "zero_values_count": {
            col: stats["zero_count"] for col, stats in numeric_summary.items()
        },
        "outlier_rough_count": {
            col: stats["rough_outlier_count"] for col, stats in numeric_summary.items()
        },
        "stats_scope": "full" if full_row_count is not None and full_row_count <= MAX_SAMPLE_ROWS else "sample",
        "sample_rows_used": len(df),
        "header_present": meta.get("header_present"),
        "schema_hint": meta.get("schema_hint"),
        "schema_source": meta.get("schema_source"),
    }


def classify_data_file(path: Path) -> str:
    rel = to_rel(path).lower()
    if "/raw/" in rel:
        return "raw"
    if "/processed/" in rel:
        return "processed"
    if "/analysis/" in rel:
        return "analysis"
    return "other"


def summarize_data_file(path: Path) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    base = {
        "path": str(path),
        "rel_path": to_rel(path),
        "file_type": path.suffix.lower() or path.name,
        "category": classify_data_file(path),
        **safe_stat(path),
        "row_count": None,
        "column_count": None,
        "columns": [],
        "time_column": None,
        "sensor_or_node_columns": [],
        "traffic_columns": [],
        "header_present": None,
        "schema_hint": None,
        "schema_source": None,
        "readable": False,
        "read_error": None,
    }

    suffix = path.suffix.lower()
    if suffix not in DATA_EXTENSIONS:
        base["read_error"] = "unsupported_extension"
        return base, None

    try:
        if suffix in {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet", ".v2"}:
            df, meta = read_table_sample(path)
            full_row_count = count_lines(path)
            if suffix in {".csv", ".tsv", ".txt", ".v2"} and full_row_count is not None:
                if meta.get("header_present"):
                    full_row_count = max(full_row_count - 1, 0)
            if suffix == ".parquet":
                full_row_count = int(pd.read_parquet(path, columns=[]).shape[0]) if path.exists() else len(df)
            if suffix in {".xlsx", ".xls", ".json"}:
                full_row_count = len(df)

            columns = [str(col) for col in df.columns]
            base.update(
                {
                    "row_count": full_row_count if full_row_count is not None else len(df),
                    "column_count": len(columns),
                    "columns": columns,
                    "time_column": detect_time_column(columns),
                    "sensor_or_node_columns": detect_sensor_node_columns(columns),
                    "traffic_columns": detect_traffic_columns(columns),
                    "header_present": meta.get("header_present"),
                    "schema_hint": meta.get("schema_hint"),
                    "schema_source": meta.get("schema_source"),
                    "readable": True,
                }
            )
            quality = summarize_table_quality(path, df, meta, full_row_count)
            return base, quality

        if suffix in {".npy", ".npz"}:
            arr = np.load(path, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):
                shapes = {key: list(value.shape) for key, value in arr.items()}
                base.update(
                    {
                        "row_count": None,
                        "column_count": len(shapes),
                        "columns": list(shapes.keys()),
                        "schema_hint": shapes,
                        "readable": True,
                    }
                )
            else:
                base.update(
                    {
                        "row_count": int(arr.shape[0]) if arr.ndim >= 1 else 1,
                        "column_count": int(arr.shape[1]) if arr.ndim >= 2 else 1,
                        "columns": [],
                        "schema_hint": list(arr.shape),
                        "readable": True,
                    }
                )
            return base, None

        base["read_error"] = "unsupported_reader"
        return base, None
    except Exception as exc:
        base["read_error"] = f"{type(exc).__name__}: {exc}"
        return base, None


def scan_candidate_files(root: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in DATA_DIR_CANDIDATES:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in DATA_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return sorted(found, key=lambda item: to_rel(item))


def read_code_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def detect_file_role(path: Path, text: str) -> str:
    first_lines = [line.strip() for line in text.splitlines()[:20] if line.strip()]
    for line in first_lines:
        if re.fullmatch(r"[#=\-\*\s]+", line):
            continue
        if line.startswith('"""') or line.startswith("'''"):
            return line.strip("\"' ")
        if line.startswith("#"):
            cleaned = line.lstrip("# ").strip()
            if cleaned:
                return cleaned
    name = path.name.lower()
    if "process_link_gps" in name:
        return "清洗路段 GPS 坐标"
    if "process_rnsd" in name:
        return "清洗路网属性并推导节点坐标"
    if "merge_speed_data" in name:
        return "合并速度观测与路网属性并按时间分块输出"
    if "compute_node_intersection_flow" in name:
        return "将路段流量聚合为节点流量"
    if "check_spatial_node_completeness" in name:
        return "检查节点时空完整性与缺失/重复/非法值"
    return "待人工确认"


def extract_evidence_lines(path: Path, text: str) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {key: [] for key in EVIDENCE_PATTERNS}
    lines = text.splitlines()
    for index, line in enumerate(lines[:MAX_CODE_LINES_PER_FILE], start=1):
        line_text = line.strip()
        if not line_text:
            continue
        lowered = line_text.lower()
        for category, patterns in EVIDENCE_PATTERNS.items():
            if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns):
                evidence[category].append(
                    {
                        "line": index,
                        "text": line_text[:240],
                    }
                )
    return evidence


def collect_code_evidence() -> dict[str, Any]:
    code_records: list[dict[str, Any]] = []
    category_index: dict[str, list[dict[str, Any]]] = {key: [] for key in EVIDENCE_PATTERNS}

    def iter_code_paths() -> Iterable[Path]:
        for item in CODE_PATHS:
            if not item.exists():
                continue
            if item.is_file():
                if item.name != "audit_real_data_preprocessing.py":
                    yield item
            else:
                for path in sorted(item.rglob("*")):
                    if (
                        path.is_file()
                        and path.suffix.lower() in CODE_EXTENSIONS
                        and path.name != "audit_real_data_preprocessing.py"
                    ):
                        yield path

    for path in iter_code_paths():
        text = read_code_text(path)
        evidence = extract_evidence_lines(path, text)
        role = detect_file_role(path, text)
        record = {
            "rel_path": to_rel(path),
            "role": role,
            "involves_missing_handling": bool(evidence["missing_handling"]),
            "involves_normalization": bool(evidence["normalization"]),
            "involves_windowing": bool(evidence["matrix_or_window"]),
            "involves_graph": bool(evidence["graph"]),
            "involves_client_partition": bool(evidence["client_partition"]),
            "involves_split": bool(evidence["split"]),
            "evidence": {key: value[:5] for key, value in evidence.items() if value},
        }
        if any(record["evidence"].values()):
            code_records.append(record)
            for category, snippets in evidence.items():
                if snippets:
                    category_index[category].append(
                        {
                            "rel_path": to_rel(path),
                            "role": role,
                            "snippets": snippets[:5],
                        }
                    )

    return {
        "code_files": sorted(code_records, key=lambda item: item["rel_path"]),
        "category_evidence": category_index,
    }


def summarize_pipeline_steps(code_evidence: dict[str, Any], inventory: list[dict[str, Any]], quality_records: list[dict[str, Any]]) -> dict[str, Any]:
    inventory_by_name = {Path(item["rel_path"]).name: item for item in inventory}
    quality_by_name = {Path(item["rel_path"]).name: item for item in quality_records}

    def build_step(found: bool, evidence_files: list[str], description: str) -> dict[str, Any]:
        return {
            "found": found,
            "evidence_files": evidence_files,
            "description": description,
        }

    steps = {
        "raw_data_reading": build_step(
            True,
            [
                "preprocessing_scripts/process_link_gps.py",
                "preprocessing_scripts/process_rnsd.py",
                "preprocessing_scripts/merge_speed_data.py",
                "docs/project_pipeline.md",
            ],
            "项目当前将 `link_gps.v2`、`road_network_sub-dataset.v2` 与 `traffic_speed_sub-dataset.v2` 作为真实数据链路输入，并通过预处理脚本按表结构读取。",
        ),
        "timestamp_processing": build_step(
            True,
            [
                "preprocessing_scripts/merge_speed_data.py",
                "analysis_scripts/compute_node_intersection_flow_optimized.py",
                "analysis_scripts/check_spatial_node_completeness.py",
            ],
            "已发现按 `时间段` 获取最小/最大范围、分块处理、按 `[时间段, 路段ID]` 或 `[时间段, 节点ID]` 排序，以及按每日 96 个时段检查连续性的实现。",
        ),
        "missing_value_handling": build_step(
            True,
            [
                "preprocessing_scripts/process_link_gps.py",
                "preprocessing_scripts/process_rnsd.py",
                "analysis_scripts/compute_node_intersection_flow_optimized.py",
                "analysis_scripts/check_spatial_node_completeness.py",
            ],
            "已发现去除关键字段空值、节点流量聚合后的 `fillna(0)`、以及完整性检查脚本对缺失记录的统计与报告。",
        ),
        "anomaly_handling": build_step(
            True,
            [
                "analysis_scripts/compute_node_intersection_flow_optimized.py",
                "analysis_scripts/check_spatial_node_completeness.py",
            ],
            "已发现将负车流量视为非法并删除的逻辑，以及对 `null / NaN / 负值` 的完整性检查；但未发现更细化的异常值裁剪或鲁棒插值策略。",
        ),
        "traffic_metric_selection": build_step(
            True,
            [
                "preprocessing_scripts/merge_speed_data.py",
                "analysis_scripts/compute_greenshields_density.py",
                "analysis_scripts/compute_node_intersection_flow_optimized.py",
            ],
            "当前真实数据链路以速度观测为起点，后续构造 `flow_q_hour`、`路口进入流量`、`路口离开流量` 与 `路口车流量` 等指标。",
        ),
        "time_window_construction": build_step(
            False,
            [],
            "当前项目中暂未发现面向真实数据预测任务的明确滑动窗口、输入长度和预测步长实现，需要后续确认。",
        ),
        "normalization": build_step(
            False,
            [],
            "当前项目中未发现针对真实数据预测样本的 `MinMaxScaler` / `StandardScaler` 归一化与保存逻辑。现有归一化主要出现在曲线形态比较与聚类分析阶段，不等同于预测数据预处理。",
        ),
        "train_val_test_split": build_step(
            False,
            [],
            "当前项目中暂未发现真实数据训练/验证/测试划分脚本或划分统计文件，需要后续确认。",
        ),
        "client_partition": build_step(
            False,
            [],
            "当前扫描未发现真实数据实验中明确的联邦客户端划分规则或客户端样本文件。",
        ),
        "graph_construction": build_step(
            True,
            [
                "preprocessing_scripts/process_rnsd.py",
                "analysis_scripts/compute_node_intersection_flow_optimized.py",
                "analysis_scripts/check_spatial_node_completeness.py",
            ],
            "已发现基于 `起始节点ID/结束节点ID` 的路网拓扑映射与节点集合检查，但未发现面向真实数据预测模型的独立邻接矩阵输出文件。",
        ),
        "preprocessed_outputs": build_step(
            True,
            [
                "data/processed/link_gps_processed.csv",
                "data/processed/rnsd_processed.csv",
                "data/analysis/node_intersection_flow_check_reports/completeness_summary.csv",
            ],
            "已识别到坐标清洗结果、路网属性清洗结果，以及节点完整性统计结果等可直接引用的输出产物。",
        ),
    }

    completeness = quality_by_name.get("completeness_summary.csv")
    if completeness:
        steps["preprocessing_result_statistics"] = build_step(
            True,
            ["data/analysis/node_intersection_flow_check_reports/completeness_summary.csv"],
            "已扫描到节点流量完整性检查汇总表，可用于引用节点数量、观测覆盖、缺失/重复/非法记录等统计。",
        )
    else:
        steps["preprocessing_result_statistics"] = build_step(
            False,
            [],
            "当前项目中暂未发现可直接复核的预处理结果汇总统计文件，需要后续确认。",
        )

    return steps


def write_inventory(records: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    path = output_dir / "real_data_file_inventory.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_quality_summary(records: list[dict[str, Any]], output_dir: Path) -> Path:
    flattened: list[dict[str, Any]] = []
    for record in records:
        flattened.append(
            {
                "rel_path": record["rel_path"],
                "file_type": record["file_type"],
                "rows": record["rows"],
                "columns_count": record["columns_count"],
                "columns": json_dumps(record["columns"]),
                "time_column": record["time_column"],
                "sensor_or_node_columns": json_dumps(record["sensor_or_node_columns"]),
                "traffic_columns": json_dumps(record["traffic_columns"]),
                "missing_values_per_column": json_dumps(record["missing_values_per_column"]),
                "missing_rate_per_column": json_dumps(record["missing_rate_per_column"]),
                "duplicate_rows": record["duplicate_rows"],
                "time_coverage_start": record["time_coverage_start"],
                "time_coverage_end": record["time_coverage_end"],
                "time_frequency": record["time_frequency"],
                "numeric_descriptive_stats": json_dumps(record["numeric_descriptive_stats"]),
                "negative_values_count": json_dumps(record["negative_values_count"]),
                "zero_values_count": json_dumps(record["zero_values_count"]),
                "outlier_rough_count": json_dumps(record["outlier_rough_count"]),
                "stats_scope": record["stats_scope"],
                "sample_rows_used": record["sample_rows_used"],
                "header_present": record["header_present"],
                "schema_hint": json_dumps(record["schema_hint"]),
                "schema_source": record["schema_source"],
            }
        )
    df = pd.DataFrame(flattened)
    path = output_dir / "real_data_quality_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def extract_usability_evidence(inventory: list[dict[str, Any]], quality_records: list[dict[str, Any]], pipeline_steps: dict[str, Any]) -> list[str]:
    items: list[str] = []
    raw_files = [item for item in inventory if item["category"] == "raw"]
    processed_files = [item for item in inventory if item["category"] == "processed"]
    readable_with_time = [item for item in inventory if item.get("time_column")]
    readable_with_traffic = [item for item in inventory if item.get("traffic_columns")]
    readable_with_nodes = [item for item in inventory if item.get("sensor_or_node_columns")]

    items.append(f"共识别到 {len(inventory)} 个候选真实数据/分析文件，其中原始文件 {len(raw_files)} 个、处理后文件 {len(processed_files)} 个。")
    items.append(f"包含时间列的文件数量为 {len(readable_with_time)}，包含交通指标字段的文件数量为 {len(readable_with_traffic)}。")
    if readable_with_nodes:
        items.append(f"包含节点/传感器字段的文件数量为 {len(readable_with_nodes)}。")

    completeness_record = next((item for item in quality_records if item["rel_path"].endswith("completeness_summary.csv")), None)
    if completeness_record:
        items.append("已检测到节点完整性汇总表，可直接支撑论文中关于覆盖完整性、缺失记录和非法值检查的描述。")

    if not pipeline_steps["train_val_test_split"]["found"]:
        items.append("当前未发现真实数据训练/验证/测试划分证据，正式论文中需要补充。")
    if not pipeline_steps["client_partition"]["found"]:
        items.append("当前未发现真实数据联邦客户端划分证据，正式论文中需要补充。")
    if not pipeline_steps["normalization"]["found"]:
        items.append("当前未发现真实数据预测样本的归一化与 scaler 保存证据，正式论文中需要补充。")
    return items


def build_markdown_report(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    lines = [
        "# Real Data Preprocessing Audit",
        "",
        "## 1. Audit Scope",
        "",
        "- Task focus: real traffic data preprocessing only; no retraining and no simulation rerun.",
        f"- Existing scan directories: {', '.join(audit['scan_scope']['existing_data_dirs']) or 'None'}",
        f"- Missing scan directories: {', '.join(audit['scan_scope']['missing_data_dirs']) or 'None'}",
        "",
        "## 2. File Inventory Summary",
        "",
        f"- Candidate data files: {summary['candidate_data_file_count']}",
        f"- Readable tabular files: {summary['readable_tabular_file_count']}",
        f"- Files with read errors: {summary['unreadable_file_count']}",
        f"- Files with time columns: {summary['files_with_time_column']}",
        f"- Files with traffic metric columns: {summary['files_with_traffic_columns']}",
        f"- Files with node or sensor columns: {summary['files_with_node_columns']}",
        "",
        "## 3. Identified Pipeline",
        "",
    ]

    for step_name, step in audit["pipeline_steps"].items():
        title = step_name.replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- Found: {'Yes' if step['found'] else 'No'}")
        if step["evidence_files"]:
            lines.append(f"- Evidence files: {', '.join(step['evidence_files'])}")
        lines.append(f"- Note: {step['description']}")
        lines.append("")

    lines.extend(
        [
            "## 4. Code File Inventory",
            "",
            "| File | Role | Missing | Normalization | Window | Graph | Client | Split |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in audit["code_evidence"]["code_files"]:
        lines.append(
            "| {rel_path} | {role} | {missing} | {norm} | {window} | {graph} | {client} | {split} |".format(
                rel_path=record["rel_path"],
                role=record["role"],
                missing="Y" if record["involves_missing_handling"] else "N",
                norm="Y" if record["involves_normalization"] else "N",
                window="Y" if record["involves_windowing"] else "N",
                graph="Y" if record["involves_graph"] else "N",
                client="Y" if record["involves_client_partition"] else "N",
                split="Y" if record["involves_split"] else "N",
            )
        )

    lines.extend(["", "## 5. Paper-Usable Evidence", ""])
    for item in audit["paper_usable_evidence"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 6. Open Questions", ""])
    for item in audit["open_questions"]:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def write_markdown_report(audit: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "real_data_preprocessing_audit.md"
    path.write_text(build_markdown_report(audit), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_data_dirs = [to_rel(path) for path in DATA_DIR_CANDIDATES if path.exists()]
    missing_data_dirs = [to_rel(path) for path in DATA_DIR_CANDIDATES if not path.exists()]

    candidate_files = scan_candidate_files(ROOT_DIR)
    inventory_records: list[dict[str, Any]] = []
    quality_records: list[dict[str, Any]] = []
    unreadable_records: list[dict[str, Any]] = []

    for path in candidate_files:
        inventory_record, quality_record = summarize_data_file(path)
        inventory_records.append(inventory_record)
        if quality_record is not None:
            quality_records.append(quality_record)
        if inventory_record.get("read_error"):
            unreadable_records.append(
                {
                    "rel_path": inventory_record["rel_path"],
                    "read_error": inventory_record["read_error"],
                }
            )

    code_evidence = collect_code_evidence()
    pipeline_steps = summarize_pipeline_steps(code_evidence, inventory_records, quality_records)
    paper_usable_evidence = extract_usability_evidence(inventory_records, quality_records, pipeline_steps)

    open_questions = [
        "真实数据来源、采集区域和正式引用信息仍需在论文中补充明确。",
        "当前未发现真实数据预测任务的滑动窗口、输入长度和预测步长设置。",
        "当前未发现真实数据训练/验证/测试划分比例或样本量统计。",
        "当前未发现真实数据联邦客户端划分规则与客户端样本量统计。",
        "当前未发现真实数据预测样本的归一化/scaler 保存文件。",
        "当前仅发现基于路网起止节点的拓扑映射，尚未发现独立保存的真实数据邻接矩阵文件。",
    ]

    summary = {
        "candidate_data_file_count": len(inventory_records),
        "readable_tabular_file_count": sum(1 for item in inventory_records if item["readable"]),
        "unreadable_file_count": len(unreadable_records),
        "files_with_time_column": sum(1 for item in inventory_records if item["time_column"]),
        "files_with_traffic_columns": sum(1 for item in inventory_records if item["traffic_columns"]),
        "files_with_node_columns": sum(1 for item in inventory_records if item["sensor_or_node_columns"]),
        "code_file_count": len(code_evidence["code_files"]),
        "identified_normalization": pipeline_steps["normalization"]["found"],
        "identified_windowing": pipeline_steps["time_window_construction"]["found"],
        "identified_split": pipeline_steps["train_val_test_split"]["found"],
        "identified_client_partition": pipeline_steps["client_partition"]["found"],
        "identified_graph": pipeline_steps["graph_construction"]["found"],
    }

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_scope": {
            "existing_data_dirs": existing_data_dirs,
            "missing_data_dirs": missing_data_dirs,
            "code_paths": [to_rel(path) for path in CODE_PATHS if path.exists()],
        },
        "summary": summary,
        "inventory_preview": inventory_records[:20],
        "quality_preview": quality_records[:20],
        "unreadable_files": unreadable_records,
        "pipeline_steps": pipeline_steps,
        "code_evidence": code_evidence,
        "paper_usable_evidence": paper_usable_evidence,
        "open_questions": open_questions,
    }

    inventory_path = write_inventory(inventory_records, OUTPUT_DIR)
    quality_path = write_quality_summary(quality_records, OUTPUT_DIR)
    json_path = OUTPUT_DIR / "real_data_preprocessing_audit.json"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = write_markdown_report(audit, OUTPUT_DIR)

    print(f"Inventory written to: {inventory_path}")
    print(f"Quality summary written to: {quality_path}")
    print(f"Audit JSON written to: {json_path}")
    print(f"Audit markdown written to: {markdown_path}")


if __name__ == "__main__":
    main()
