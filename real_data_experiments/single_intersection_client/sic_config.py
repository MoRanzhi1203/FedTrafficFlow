"""Configuration utilities for the single-intersection client experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

from real_data_experiments.common.data_splits import validate_split_ratios


@dataclass
class ExperimentConfig:
    """Runtime configuration for the single-intersection client experiment."""

    experiment_name: str = "single_intersection_client_tensor"
    seed: int = 42
    data_mode: str = "tensor"
    input_path: str = "data/analysis/node_intersection_flow_parquet"
    tensor_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"
    regions_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv"
    output_dir: str = "results/real_data_experiments/single_intersection_client_tensor"
    num_clients: int = 5
    selected_clients: list[int] | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_name: str = ""
    batch_size: int = 64
    learning_rate: float = 1e-3
    local_epochs: int = 2
    communication_rounds: int = 3
    sequence_length: int = 12
    prediction_horizon: int = 1
    model_name: str = "cnn_lstm_attention"
    device: str = "cuda"
    target_column: str = "路口车流量"
    max_chunks: int | None = 7
    prediction_sample_limit: int = 200
    workflow: str = "all"
    independent_total_epochs: int | None = None
    use_active_regions_only: bool = True
    use_channels: list[int] = field(default_factory=lambda: [0, 1])
    target_channel: int = 0
    input_normalization: bool = True
    input_normalization_eps: float = 1e-6
    target_normalization: bool = True
    target_normalization_eps: float = 1e-6
    show_progress: bool = True
    progress_interval: int = 20
    model_variant: str = "baseline"

    # Calendar Feature FedAvg
    enable_calendar_profile_baseline: bool = False
    enable_calendar_feature_fedavg: bool = False
    calendar_features_path: str = "data/external/calendar/calendar_features_15min_2017_04_01_to_2017_05_31.csv"
    calendar_feature_columns: tuple[str, ...] = (
        "sin_time_of_day",
        "cos_time_of_day",
        "sin_day_of_week",
        "cos_day_of_week",
        "is_holiday",
        "is_weekend",
        "is_effective_workday",
        "is_adjusted_workday",
        "days_to_nearest_holiday",
    )
    calendar_feature_mode: str = "target_time"

    calendar_feature_sets: tuple[str, ...] = ("time_only", "holiday_only", "full")
    calendar_gate_init: float = -5.0
    calendar_fusion: str = "residual_gate"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly config dict."""
        return asdict(self)


def parse_selected_clients(raw_text: str | None) -> list[int] | None:
    """Parse a comma-separated client list from CLI."""
    if raw_text is None or not raw_text.strip():
        return None
    return [int(part.strip()) for part in raw_text.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the experiment."""
    parser = argparse.ArgumentParser(description="Single-intersection client experiment")
    parser.add_argument("--workflow", choices=["all", "train", "evaluate", "export"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-mode", choices=["tensor", "parquet"], default="tensor")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "gpu", "cpu", "auto"],
        help="Runtime device. Default: cuda. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--input-path", type=str, default="data/analysis/node_intersection_flow_parquet")
    parser.add_argument("--tensor-path", type=str, default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt")
    parser.add_argument("--regions-path", type=str, default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv")
    parser.add_argument("--output-dir", type=str, default="results/real_data_experiments/single_intersection_client_tensor")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--selected-clients", type=str, default="")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--target-column", type=str, default="路口车流量")
    parser.add_argument("--max-chunks", type=int, default=7)
    parser.add_argument("--disable-input-normalization", action="store_true")
    parser.add_argument("--input-normalization-eps", type=float, default=1e-6)
    parser.add_argument("--disable-target-normalization", action="store_true")
    parser.add_argument("--target-normalization-eps", type=float, default=1e-6)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--hide-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=True)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--model-variant", type=str, default="baseline", choices=["baseline", "legacy_ipynb"])
    parser.add_argument("--enable-calendar-profile-baseline", dest="enable_calendar_profile_baseline", action="store_true")
    parser.add_argument("--no-calendar-profile-baseline", dest="enable_calendar_profile_baseline", action="store_false")
    parser.set_defaults(enable_calendar_profile_baseline=False)
    parser.add_argument("--enable-calendar-feature-fedavg", dest="enable_calendar_feature_fedavg", action="store_true")
    parser.add_argument("--no-calendar-feature-fedavg", dest="enable_calendar_feature_fedavg", action="store_false")
    parser.set_defaults(enable_calendar_feature_fedavg=False)
    parser.add_argument("--calendar-features-path", type=str, default="data/external/calendar/calendar_features_15min_2017_04_01_to_2017_05_31.csv")
    parser.add_argument("--calendar-feature-columns", type=str, default="")
    parser.add_argument("--calendar-feature-mode", type=str, default="target_time")
    parser.add_argument("--calendar-feature-sets", type=str, default="time_only,holiday_only,full")
    parser.add_argument("--calendar-gate-init", type=float, default=-5.0)
    parser.add_argument("--calendar-fusion", type=str, default="residual_gate", choices=["residual_gate"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser


def _parse_calendar_columns(raw_text: str) -> tuple[str, ...]:
    """Parse comma-separated calendar column names from CLI."""
    if not raw_text or not raw_text.strip():
        return ()
    return tuple(col.strip() for col in raw_text.split(",") if col.strip())


def _parse_calendar_feature_sets(raw_text: str) -> tuple[str, ...]:
    if not raw_text or not raw_text.strip():
        return ("time_only", "holiday_only", "full")
    valid = {"time_only", "holiday_only", "full"}
    values = tuple(part.strip() for part in raw_text.split(",") if part.strip())
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise ValueError(f"Invalid calendar feature sets: {invalid}. valid={sorted(valid)}")
    return values


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Construct an ExperimentConfig from parsed CLI arguments."""
    split_name = validate_split_ratios(args.train_ratio, args.val_ratio)
    return ExperimentConfig(
        workflow=args.workflow,
        seed=args.seed,
        data_mode=args.data_mode,
        input_path=args.input_path,
        tensor_path=args.tensor_path,
        regions_path=args.regions_path,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        selected_clients=parse_selected_clients(args.selected_clients),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_name=split_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        communication_rounds=args.rounds,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        device=args.device,
        target_column=args.target_column,
        max_chunks=args.max_chunks,
        input_normalization=not args.disable_input_normalization,
        input_normalization_eps=args.input_normalization_eps,
        target_normalization=not args.disable_target_normalization,
        target_normalization_eps=args.target_normalization_eps,
        show_progress=args.show_progress,
        progress_interval=args.progress_interval,
        model_variant=args.model_variant,
        enable_calendar_profile_baseline=args.enable_calendar_profile_baseline,
        enable_calendar_feature_fedavg=args.enable_calendar_feature_fedavg,
        calendar_features_path=args.calendar_features_path,
        calendar_feature_columns=_parse_calendar_columns(args.calendar_feature_columns) or ExperimentConfig.calendar_feature_columns,
        calendar_feature_mode=args.calendar_feature_mode,
        calendar_feature_sets=_parse_calendar_feature_sets(args.calendar_feature_sets),
        calendar_gate_init=args.calendar_gate_init,
        calendar_fusion=args.calendar_fusion,
    )
