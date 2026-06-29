"""Configuration utilities for the regional-client experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    """Runtime configuration for the regional-client experiment."""

    experiment_name: str = "region_client_tensor"
    seed: int = 42
    data_mode: str = "tensor"
    tensor_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"
    regions_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv"
    output_dir: str = "results/real_data_experiments/region_client_tensor"
    partition_method: str = "spatial_block"
    num_clients: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    learning_rate: float = 1e-3
    local_epochs: int = 1
    communication_rounds: int = 1
    sequence_length: int = 12
    prediction_horizon: int = 1
    device: str = "cuda"
    workflow: str = "all"
    prediction_sample_limit: int = 400
    independent_total_epochs: int | None = None
    max_samples_per_client_split: int | None = None
    use_active_regions_only: bool = True
    use_channels: list[int] = field(default_factory=lambda: [0, 1])
    target_channel: int = 0
    input_normalization: bool = True
    target_normalization: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the regional-client experiment."""

    parser = argparse.ArgumentParser(description="Regional-client experiment")
    parser.add_argument("--workflow", choices=["all", "train", "evaluate", "export"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-mode", choices=["tensor"], default="tensor")
    parser.add_argument("--tensor-path", type=str, default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt")
    parser.add_argument("--regions-path", type=str, default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv")
    parser.add_argument("--partition-method", choices=["spatial_block", "flow_kmeans"], default="spatial_block")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "gpu", "cpu", "auto"],
        help="Runtime device. Default: cuda. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--output-dir", type=str, default="results/real_data_experiments/region_client_tensor")
    parser.add_argument("--max-samples-per-client-split", type=int, default=0)
    parser.add_argument("--input-normalization", dest="input_normalization", action="store_true", default=True)
    parser.add_argument("--no-input-normalization", dest="input_normalization", action="store_false")
    parser.add_argument("--target-normalization", dest="target_normalization", action="store_true", default=True)
    parser.add_argument("--no-target-normalization", dest="target_normalization", action="store_false")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Construct ExperimentConfig from parsed CLI arguments."""

    return ExperimentConfig(
        workflow=args.workflow,
        seed=args.seed,
        data_mode=args.data_mode,
        tensor_path=args.tensor_path,
        regions_path=args.regions_path,
        output_dir=args.output_dir,
        partition_method=args.partition_method,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        communication_rounds=args.rounds,
        sequence_length=args.sequence_length,
        device=args.device,
        max_samples_per_client_split=(None if int(args.max_samples_per_client_split) <= 0 else int(args.max_samples_per_client_split)),
        input_normalization=args.input_normalization,
        target_normalization=args.target_normalization,
    )
