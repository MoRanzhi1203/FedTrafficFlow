"""Profile real-data tensor experiments without changing the formal training logic.

This wrapper only records wall-clock time, hardware snapshots, and split scale
summaries around existing experiment entrypoints. It is not the formal training
driver and does not generate official paper results. By default it writes
artifacts under ``results/real_data_experiments/compute_time_profile/``.

GPU profiling should only be interpreted as real measurements when running in a
CUDA-enabled PyTorch environment.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ImportError:  # pragma: no cover - environment dependent
    psutil = None  # type: ignore[assignment]

from real_data_experiments.common.result_writer import write_csv, write_json, write_text
from real_data_experiments.common.tensor_dataset import build_time_split_bounds, get_region_usage_summary, load_grid_tensor_bundle
from real_data_experiments.region_ablation.ra_config import DEFAULT_VARIANTS as RA_DEFAULT_VARIANTS
from real_data_experiments.region_ablation.ra_config import ExperimentConfig as RegionAblationConfig
from real_data_experiments.region_ablation.ra_core import run_experiment as run_region_ablation_experiment
from real_data_experiments.region_client.rc_config import ExperimentConfig as RegionClientConfig
from real_data_experiments.region_client.rc_core import build_region_client_data, run_experiment as run_region_client_experiment
from real_data_experiments.single_intersection_ablation.sia_config import DEFAULT_VARIANTS as SIA_DEFAULT_VARIANTS
from real_data_experiments.single_intersection_ablation.sia_config import ExperimentConfig as SingleIntersectionAblationConfig
from real_data_experiments.single_intersection_ablation.sia_core import run_experiment as run_single_intersection_ablation_experiment
from real_data_experiments.single_intersection_client.sic_config import ExperimentConfig as SingleIntersectionConfig
from real_data_experiments.single_intersection_client.sic_core import run_experiment as run_single_intersection_experiment


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "real_data_experiments" / "compute_time_profile"
DEFAULT_TENSOR_PATH = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"
DEFAULT_REGIONS_PATH = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv"
GRID_SELECTED_CLIENTS = [290, 284, 318]
GRID_FULL_SELECTED_CLIENTS = [290, 284, 318, 288, 289]
DEFAULT_MAX_SAMPLES_PER_CLIENT_SPLIT = 2048

PROFILE_CSV_NAMES = {
    ("grid_cell", "main", "cuda"): "profile_grid_cell_main_cuda.csv",
    ("grid_cell", "main", "cpu"): "profile_grid_cell_main_cpu.csv",
    ("grid_cell", "ablation", "cuda"): "profile_grid_cell_ablation_cuda.csv",
    ("grid_cell", "ablation", "cpu"): "profile_grid_cell_ablation_cpu.csv",
    ("cluster", "main", "cuda"): "profile_cluster_main_cuda.csv",
    ("cluster", "main", "cpu"): "profile_cluster_main_cpu.csv",
    ("cluster", "ablation", "cuda"): "profile_cluster_ablation_cuda.csv",
    ("cluster", "ablation", "cpu"): "profile_cluster_ablation_cpu.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile real-data tensor experiments")
    parser.add_argument("--setting", choices=["grid_cell", "cluster"], help="Experiment family to profile.")
    parser.add_argument("--task", choices=["main", "ablation"], help="Experiment task to profile.")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device mode to profile.")
    parser.add_argument("--run-all", action="store_true", help="Run the full GPU/CPU profiling matrix.")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--partition-method", choices=["spatial_block", "flow_kmeans"], default="spatial_block")
    parser.add_argument("--selected-clients", type=str, default="290,284,318")
    parser.add_argument("--max-samples-per-client-split", type=int, default=DEFAULT_MAX_SAMPLES_PER_CLIENT_SPLIT)
    parser.add_argument("--tensor-path", type=str, default=DEFAULT_TENSOR_PATH)
    parser.add_argument("--regions-path", type=str, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_selected_clients(raw_text: str) -> list[int]:
    return [int(part.strip()) for part in raw_text.split(",") if part.strip()]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_mb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1024.0 / 1024.0, 3)


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    return f"{hours:.2f} h"


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return None
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return output.strip() or None


def detect_hardware_summary() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_mem_gb = None
    device_name = "cpu"
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

    ram_total_gb = None
    if psutil is not None:
        ram_total_gb = round(psutil.virtual_memory().total / 1024**3, 2)

    return {
        "generated_at": now_iso(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device_name": device_name,
        "gpu_mem_total_GB": gpu_mem_gb,
        "system_ram_total_GB": ram_total_gb,
        "nvidia_smi": run_nvidia_smi(),
    }


def detect_dataset_scale_summary(
    tensor_path: str,
    regions_path: str,
    sequence_length: int,
    horizon: int,
) -> dict[str, Any]:
    bundle = load_grid_tensor_bundle(tensor_path=tensor_path, regions_path=regions_path)
    tensor = bundle.tensor
    usage = get_region_usage_summary(bundle.regions_df)
    split_bounds = build_time_split_bounds(time_count=int(tensor.shape[2]))
    return {
        "tensor_shape": [int(v) for v in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "C": int(tensor.shape[0]),
        "R": int(tensor.shape[1]),
        "T": int(tensor.shape[2]),
        "total_region_count": int(usage["total_region_count"]),
        "active_region_count": int(usage["active_region_count"]),
        "sequence_length": int(sequence_length),
        "horizon": int(horizon),
        "train_end": int(split_bounds["train_end"]),
        "val_end": int(split_bounds["val_end"]),
        "test_end": int(split_bounds["test_end"]),
        "estimated_tensor_memory_MB": round(int(tensor.shape[0]) * int(tensor.shape[1]) * int(tensor.shape[2]) * 4 / 1024 / 1024, 3),
    }


@contextlib.contextmanager
def temporary_argv(argv: list[str]) -> Any:
    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original_argv


def current_process_memory_mb() -> float | None:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 3)


def load_split_details(output_dir: Path) -> dict[str, Any]:
    split_summary = safe_read_json(output_dir / "split_summary.json")
    used_region_count = split_summary.get("used_region_count")
    client_region_counts = split_summary.get("client_region_counts")
    client_sample_counts = split_summary.get("client_sample_counts")
    total_train_samples = None

    if isinstance(client_sample_counts, dict):
        total_train_samples = int(sum(int(value) for value in client_sample_counts.values()))
    elif isinstance(split_summary.get("clients"), list):
        client_region_counts = client_region_counts or {}
        client_sample_counts = client_sample_counts or {}
        total_train_samples = 0
        for item in split_summary["clients"]:
            client_id = str(item.get("client_id"))
            if "region_count" in item:
                client_region_counts[client_id] = int(item["region_count"])
            elif "train" in item and isinstance(item["train"], dict) and "region_ids" in item:
                client_region_counts[client_id] = int(len(item["region_ids"]))

            train_value = None
            if isinstance(item.get("train"), dict):
                train_value = item["train"].get("sample_count")
            elif "train_size" in item:
                train_value = item["train_size"]
            if train_value is not None:
                client_sample_counts[client_id] = int(train_value)
                total_train_samples += int(train_value)
        if total_train_samples == 0:
            total_train_samples = None

    convergence_df = read_csv_frame(output_dir / "convergence_history.csv")
    round_count = None
    if not convergence_df.empty:
        if "variant" in convergence_df.columns and "communication_round" in convergence_df.columns:
            round_count = int(convergence_df[["variant", "communication_round"]].drop_duplicates().shape[0])
        elif "communication_round" in convergence_df.columns:
            round_count = int(convergence_df["communication_round"].nunique())
        else:
            round_count = int(len(convergence_df))

    return {
        "used_region_count": used_region_count,
        "client_region_counts": client_region_counts,
        "client_sample_counts": client_sample_counts,
        "total_train_samples": total_train_samples,
        "round_count": round_count,
    }


def build_profile_output_dir(output_root: Path, setting: str, task: str, device: str, rounds: int, local_epochs: int) -> Path:
    prefix = "gpu" if device == "cuda" else "cpu"
    middle = "grid_cell" if setting == "grid_cell" else "cluster"
    return output_root / f"profile_{prefix}_{middle}_{task}_r{rounds}e{local_epochs}"


def build_profile_csv_path(output_root: Path, setting: str, task: str, device: str) -> Path:
    return output_root / PROFILE_CSV_NAMES[(setting, task, device)]


def build_module_command(
    setting: str,
    task: str,
    device: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    if setting == "grid_cell" and task == "main":
        module = "real_data_experiments.single_intersection_client.sic_core"
        parts = [
            "--workflow",
            "all",
            "--data-mode",
            "tensor",
            "--num-clients",
            str(args.num_clients),
            "--selected-clients",
            ",".join(str(v) for v in parse_selected_clients(args.selected_clients)),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            device,
            "--output-dir",
            str(output_dir),
        ]
        return module, parts
    if setting == "grid_cell" and task == "ablation":
        module = "real_data_experiments.single_intersection_ablation.sia_core"
        parts = [
            "--workflow",
            "all",
            "--data-mode",
            "tensor",
            "--num-clients",
            str(args.num_clients),
            "--selected-clients",
            ",".join(str(v) for v in parse_selected_clients(args.selected_clients)),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            device,
            "--output-dir",
            str(output_dir),
        ]
        return module, parts
    if setting == "cluster" and task == "main":
        module = "real_data_experiments.region_client.rc_core"
        parts = [
            "--workflow",
            "all",
            "--data-mode",
            "tensor",
            "--partition-method",
            args.partition_method,
            "--num-clients",
            str(args.num_clients),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            device,
            "--max-samples-per-client-split",
            str(args.max_samples_per_client_split),
            "--output-dir",
            str(output_dir),
        ]
        return module, parts
    if setting == "cluster" and task == "ablation":
        module = "real_data_experiments.region_ablation.ra_core"
        parts = [
            "--workflow",
            "all",
            "--data-mode",
            "tensor",
            "--partition-method",
            args.partition_method,
            "--num-clients",
            str(args.num_clients),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--learning-rate",
            str(args.learning_rate),
            "--device",
            device,
            "--max-samples-per-client-split",
            str(args.max_samples_per_client_split),
            "--output-dir",
            str(output_dir),
        ]
        return module, parts
    raise ValueError(f"Unsupported profile target: setting={setting}, task={task}, device={device}")


def build_runtime_config(
    setting: str,
    task: str,
    device: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Any:
    if setting == "grid_cell" and task == "main":
        return SingleIntersectionConfig(
            workflow="all",
            data_mode="tensor",
            tensor_path=args.tensor_path,
            regions_path=args.regions_path,
            output_dir=str(output_dir),
            num_clients=args.num_clients,
            selected_clients=parse_selected_clients(args.selected_clients),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            communication_rounds=args.rounds,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            device=device,
        )
    if setting == "grid_cell" and task == "ablation":
        return SingleIntersectionAblationConfig(
            workflow="all",
            data_mode="tensor",
            tensor_path=args.tensor_path,
            regions_path=args.regions_path,
            output_dir=str(output_dir),
            num_clients=args.num_clients,
            selected_clients=parse_selected_clients(args.selected_clients),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            communication_rounds=args.rounds,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            device=device,
            variants=list(SIA_DEFAULT_VARIANTS),
        )
    if setting == "cluster" and task == "main":
        return RegionClientConfig(
            workflow="all",
            data_mode="tensor",
            tensor_path=args.tensor_path,
            regions_path=args.regions_path,
            output_dir=str(output_dir),
            partition_method=args.partition_method,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            communication_rounds=args.rounds,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            device=device,
            max_samples_per_client_split=(None if int(args.max_samples_per_client_split) <= 0 else int(args.max_samples_per_client_split)),
        )
    if setting == "cluster" and task == "ablation":
        return RegionAblationConfig(
            workflow="all",
            data_mode="tensor",
            tensor_path=args.tensor_path,
            regions_path=args.regions_path,
            output_dir=str(output_dir),
            partition_method=args.partition_method,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            communication_rounds=args.rounds,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            device=device,
            variants=list(RA_DEFAULT_VARIANTS),
            max_samples_per_client_split=(None if int(args.max_samples_per_client_split) <= 0 else int(args.max_samples_per_client_split)),
        )
    raise ValueError(f"Unsupported profile target: setting={setting}, task={task}, device={device}")


def dispatch_experiment(setting: str, task: str, config: Any, module: str, cli_parts: list[str]) -> dict[str, Any]:
    argv = [module, *cli_parts]
    if setting == "grid_cell" and task == "main":
        with temporary_argv(argv):
            return run_single_intersection_experiment(config)
    if setting == "grid_cell" and task == "ablation":
        with temporary_argv(argv):
            return run_single_intersection_ablation_experiment(config)
    if setting == "cluster" and task == "main":
        with temporary_argv(argv):
            return run_region_client_experiment(config)
    if setting == "cluster" and task == "ablation":
        with temporary_argv(argv):
            return run_region_ablation_experiment(config)
    raise ValueError(f"Unsupported dispatch target: setting={setting}, task={task}")


def collect_cluster_full_sample_counts(args: argparse.Namespace) -> dict[str, Any]:
    config = RegionClientConfig(
        workflow="all",
        data_mode="tensor",
        tensor_path=args.tensor_path,
        regions_path=args.regions_path,
        output_dir=str(DEFAULT_OUTPUT_ROOT / "_tmp_cluster_full_scale_unused"),
        partition_method=args.partition_method,
        num_clients=3,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_epochs=1,
        communication_rounds=1,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        device="cpu",
        max_samples_per_client_split=None,
    )
    clients, split_summary, partition_result = build_region_client_data(config)
    total_train_samples = int(sum(len(client.train_loader.dataset) for client in clients))
    return {
        "full_train_samples": total_train_samples,
        "full_client_sample_counts": split_summary.get("client_sample_counts"),
        "full_client_region_counts": split_summary.get("client_region_counts"),
        "used_region_count": int(len(partition_result.assignment_df)),
    }


def estimate_time_row(
    experiment: str,
    device: str,
    measured_row: dict[str, Any] | None,
    target_rounds: int,
    target_local_epochs: int,
    sample_scale: float,
    notes: str,
) -> dict[str, Any]:
    if not measured_row:
        return {
            "experiment": experiment,
            "device": device,
            "status": "missing_profile",
            "measured_wall_time_sec": None,
            "measured_rounds": None,
            "measured_local_epochs": None,
            "measured_total_train_samples": None,
            "target_rounds": target_rounds,
            "target_local_epochs": target_local_epochs,
            "sample_scale": round(sample_scale, 6),
            "estimated_time_sec": None,
            "estimated_time_min": None,
            "estimated_time_hr": None,
            "notes": notes,
        }

    status = measured_row.get("status", "ok")
    if status != "ok":
        return {
            "experiment": experiment,
            "device": device,
            "status": status,
            "measured_wall_time_sec": measured_row.get("wall_time_sec"),
            "measured_rounds": measured_row.get("rounds"),
            "measured_local_epochs": measured_row.get("local_epochs"),
            "measured_total_train_samples": measured_row.get("total_train_samples"),
            "target_rounds": target_rounds,
            "target_local_epochs": target_local_epochs,
            "sample_scale": round(sample_scale, 6),
            "estimated_time_sec": None,
            "estimated_time_min": None,
            "estimated_time_hr": None,
            "notes": notes,
        }

    measured_wall_time = float(measured_row["wall_time_sec"])
    measured_rounds = max(int(measured_row["rounds"]), 1)
    measured_local_epochs = max(int(measured_row["local_epochs"]), 1)
    estimated_time_sec = measured_wall_time * (target_rounds / measured_rounds) * (target_local_epochs / measured_local_epochs) * sample_scale
    return {
        "experiment": experiment,
        "device": device,
        "status": status,
        "measured_wall_time_sec": round(measured_wall_time, 6),
        "measured_rounds": measured_rounds,
        "measured_local_epochs": measured_local_epochs,
        "measured_total_train_samples": measured_row.get("total_train_samples"),
        "target_rounds": target_rounds,
        "target_local_epochs": target_local_epochs,
        "sample_scale": round(sample_scale, 6),
        "estimated_time_sec": round(estimated_time_sec, 6),
        "estimated_time_min": round(estimated_time_sec / 60.0, 6),
        "estimated_time_hr": round(estimated_time_sec / 3600.0, 6),
        "notes": notes,
    }


def build_estimation_summary(profile_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    row_map = {(row["setting"], row["task"], row["device"]): row for row in profile_rows}
    cluster_full_scale = collect_cluster_full_sample_counts(args)

    measured_cluster_profile_samples = row_map.get(("cluster", "main", "cpu"), {}).get("total_train_samples") or row_map.get(("cluster", "main", "cuda"), {}).get("total_train_samples")
    if measured_cluster_profile_samples:
        cluster_without_cap_scale = float(cluster_full_scale["full_train_samples"]) / float(measured_cluster_profile_samples)
    else:
        cluster_without_cap_scale = 1.0

    rows: list[dict[str, Any]] = []
    for device in ("cuda", "cpu"):
        grid_main = row_map.get(("grid_cell", "main", device))
        grid_ablation = row_map.get(("grid_cell", "ablation", device))
        cluster_main = row_map.get(("cluster", "main", device))
        cluster_ablation = row_map.get(("cluster", "ablation", device))

        rows.append(
            estimate_time_row(
                experiment="grid_cell_main_quick",
                device=device,
                measured_row=grid_main,
                target_rounds=5,
                target_local_epochs=3,
                sample_scale=3 / 3,
                notes="Grid quick: 3 clients, 5 rounds, 3 local epochs.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="grid_cell_main_full",
                device=device,
                measured_row=grid_main,
                target_rounds=20,
                target_local_epochs=3,
                sample_scale=5 / 3,
                notes="Grid full: extrapolated from 3 clients to 5 clients.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="grid_cell_ablation_full",
                device=device,
                measured_row=grid_ablation,
                target_rounds=20,
                target_local_epochs=3,
                sample_scale=5 / 3,
                notes="Grid ablation full: measured run already includes 4 variants.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="cluster_main_quick_with_cap",
                device=device,
                measured_row=cluster_main,
                target_rounds=5,
                target_local_epochs=3,
                sample_scale=1.0,
                notes=f"Cluster quick with sample cap={args.max_samples_per_client_split}.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="cluster_main_quick",
                device=device,
                measured_row=cluster_main,
                target_rounds=5,
                target_local_epochs=3,
                sample_scale=cluster_without_cap_scale,
                notes="Cluster quick without cap, extrapolated by full/profiled train samples.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="cluster_main_full_with_cap",
                device=device,
                measured_row=cluster_main,
                target_rounds=20,
                target_local_epochs=3,
                sample_scale=1.0,
                notes=f"Cluster full with sample cap={args.max_samples_per_client_split}.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="cluster_main_full_without_cap",
                device=device,
                measured_row=cluster_main,
                target_rounds=20,
                target_local_epochs=3,
                sample_scale=cluster_without_cap_scale,
                notes="Cluster full without cap, extrapolated by full/profiled train samples.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="cluster_ablation_full_with_cap",
                device=device,
                measured_row=cluster_ablation,
                target_rounds=20,
                target_local_epochs=3,
                sample_scale=1.0,
                notes=f"Cluster ablation full with sample cap={args.max_samples_per_client_split}; measured run already includes 4 variants.",
            )
        )
        rows.append(
            estimate_time_row(
                experiment="cluster_ablation_full_without_cap",
                device=device,
                measured_row=cluster_ablation,
                target_rounds=20,
                target_local_epochs=3,
                sample_scale=cluster_without_cap_scale,
                notes="Cluster ablation full without cap; measured run already includes 4 variants.",
            )
        )

    for row in rows:
        row["cluster_full_train_samples"] = cluster_full_scale["full_train_samples"]
        row["cluster_full_client_sample_counts"] = json.dumps(cluster_full_scale["full_client_sample_counts"], ensure_ascii=False)
        row["cluster_full_client_region_counts"] = json.dumps(cluster_full_scale["full_client_region_counts"], ensure_ascii=False)

    return rows


def write_status_csv(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    write_csv(frame, path)


def profile_one(setting: str, task: str, device: str, args: argparse.Namespace, hardware_summary: dict[str, Any]) -> dict[str, Any]:
    output_root = ensure_dir(Path(args.output_dir))
    run_output_dir = build_profile_output_dir(output_root, setting, task, device, args.rounds, args.local_epochs)
    profile_csv_path = build_profile_csv_path(output_root, setting, task, device)
    module, cli_parts = build_module_command(setting, task, device, run_output_dir, args)
    command_text = f"python -m {module} {' '.join(cli_parts)}"

    row: dict[str, Any] = {
        "start_time": now_iso(),
        "end_time": None,
        "wall_time_sec": None,
        "device": device,
        "setting": setting,
        "task": task,
        "num_clients": int(args.num_clients),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "prediction_horizon": int(args.prediction_horizon),
        "output_dir": str(run_output_dir),
        "exit_status": None,
        "status": "pending",
        "command": command_text,
        "gpu_total_memory_MB": round(float(hardware_summary["gpu_mem_total_GB"]) * 1024, 3) if hardware_summary.get("gpu_mem_total_GB") else None,
        "gpu_max_allocated_MB": None,
        "gpu_max_reserved_MB": None,
        "nvidia_smi_before": None,
        "nvidia_smi_after": None,
        "system_ram_total_GB": hardware_summary.get("system_ram_total_GB"),
        "process_memory_before_MB": None,
        "process_memory_after_MB": None,
        "used_region_count": None,
        "client_region_counts": None,
        "client_sample_counts": None,
        "total_train_samples": None,
        "round_count": None,
        "avg_time_per_round_sec": None,
        "error_message": None,
    }

    if device == "cuda" and not bool(hardware_summary.get("cuda_available")):
        row["end_time"] = now_iso()
        row["exit_status"] = "skipped"
        row["status"] = "cuda_unavailable"
        row["error_message"] = "Current PyTorch environment does not detect CUDA."
        write_status_csv(profile_csv_path, row)
        return row

    if run_output_dir.exists():
        # Keep existing artifacts if the same path already exists and gets overwritten by the current run.
        pass

    row["process_memory_before_MB"] = current_process_memory_mb()
    if device == "cuda":
        row["nvidia_smi_before"] = run_nvidia_smi()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_counter = perf_counter()
    try:
        config = build_runtime_config(setting, task, device, run_output_dir, args)
        dispatch_experiment(setting, task, config, module, cli_parts)
        row["exit_status"] = "success"
        row["status"] = "ok"
    except Exception as exc:  # pragma: no cover - runtime dependent
        row["exit_status"] = "failed"
        row["status"] = "failed"
        row["error_message"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        error_path = run_output_dir / "profile_error.txt"
        write_text("".join(traceback.format_exc()), error_path)
    finally:
        if device == "cuda" and bool(hardware_summary.get("cuda_available")):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            row["gpu_max_allocated_MB"] = to_mb(torch.cuda.max_memory_allocated())
            row["gpu_max_reserved_MB"] = to_mb(torch.cuda.max_memory_reserved())
            row["nvidia_smi_after"] = run_nvidia_smi()
            torch.cuda.empty_cache()
        gc.collect()

    row["wall_time_sec"] = round(perf_counter() - start_counter, 6)
    row["end_time"] = now_iso()
    row["process_memory_after_MB"] = current_process_memory_mb()

    split_details = load_split_details(run_output_dir)
    row["used_region_count"] = split_details["used_region_count"]
    row["client_region_counts"] = json.dumps(split_details["client_region_counts"], ensure_ascii=False) if split_details["client_region_counts"] is not None else None
    row["client_sample_counts"] = json.dumps(split_details["client_sample_counts"], ensure_ascii=False) if split_details["client_sample_counts"] is not None else None
    row["total_train_samples"] = split_details["total_train_samples"]
    row["round_count"] = split_details["round_count"]
    if split_details["round_count"]:
        row["avg_time_per_round_sec"] = round(float(row["wall_time_sec"]) / float(split_details["round_count"]), 6)

    write_status_csv(profile_csv_path, row)
    return row


def resolve_jobs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.run_all:
        return [
            ("grid_cell", "main", "cuda"),
            ("grid_cell", "ablation", "cuda"),
            ("cluster", "main", "cuda"),
            ("cluster", "ablation", "cuda"),
            ("grid_cell", "main", "cpu"),
            ("grid_cell", "ablation", "cpu"),
            ("cluster", "main", "cpu"),
            ("cluster", "ablation", "cpu"),
        ]
    if not args.setting or not args.task or not args.device:
        raise ValueError("Specify --run-all or provide --setting, --task, and --device.")
    return [(args.setting, args.task, args.device)]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_dir))
    hardware_summary = detect_hardware_summary()
    dataset_scale_summary = detect_dataset_scale_summary(
        tensor_path=args.tensor_path,
        regions_path=args.regions_path,
        sequence_length=args.sequence_length,
        horizon=args.prediction_horizon,
    )

    write_json(hardware_summary, output_root / "hardware_summary.json")
    write_json(dataset_scale_summary, output_root / "dataset_scale_summary.json")

    profile_rows: list[dict[str, Any]] = []
    for setting, task, device in resolve_jobs(args):
        print(f"[profiling] setting={setting} task={task} device={device}")
        profile_rows.append(profile_one(setting, task, device, args, hardware_summary))

    estimation_rows = build_estimation_summary(profile_rows, args)
    write_csv(pd.DataFrame(estimation_rows), output_root / "compute_time_estimation_summary.csv")
    write_json(
        {
            "generated_at": now_iso(),
            "profile_rows": profile_rows,
            "estimation_rows": estimation_rows,
        },
        output_root / "profiling_summary.json",
    )

    print(f"[profiling] completed -> {output_root}")


if __name__ == "__main__":
    main()
