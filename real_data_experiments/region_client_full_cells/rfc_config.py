"""Configuration utilities for the full-cells region-client experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    """Runtime configuration for region-full-cells diagnostics."""

    experiment_name: str = "region_client_full_cells"
    workflow: str = "all"
    seed: int = 2026
    tensor_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"
    regions_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv"
    partition_file: str = "real_data_experiments/region_client_full_cells/partitions/spatial_k5.json"
    output_dir: str = "results/real_data_experiments/diagnostics/full_cells_spatial_k5_r80_e2_lr5e4_cuda"
    batch_size: int = 32
    learning_rate: float = 5e-4
    local_epochs: int = 2
    communication_rounds: int = 80
    sequence_length: int = 12
    prediction_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    device: str = "cuda"
    use_channels: list[int] = field(default_factory=lambda: [0, 1])
    target_channel: int = 0
    prediction_sample_limit: int = 500
    independent_total_epochs: int | None = None
    input_normalization: bool = True
    target_normalization: bool = True
    input_normalization_eps: float = 1e-6
    target_normalization_eps: float = 1e-6
    show_progress: bool = True
    progress_interval: int = 20

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the region-full-cells experiment."""

    parser = argparse.ArgumentParser(description="Full-cells region-client diagnostic experiment")
    parser.add_argument("--workflow", choices=["all"], default="all")
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument(
        "--regions-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv",
    )
    parser.add_argument("--partition-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--prediction-sample-limit", type=int, default=500)
    parser.add_argument("--show-progress", action="store_true", default=False)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Construct ExperimentConfig from parsed CLI arguments."""

    return ExperimentConfig(
        workflow=args.workflow,
        seed=args.seed,
        tensor_path=args.tensor_path,
        regions_path=args.regions_path,
        partition_file=args.partition_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        communication_rounds=args.rounds,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        device=args.device,
        prediction_sample_limit=args.prediction_sample_limit,
        show_progress=bool(args.show_progress),
    )

