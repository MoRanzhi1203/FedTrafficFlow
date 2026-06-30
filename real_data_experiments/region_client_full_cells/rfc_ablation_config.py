"""Configuration utilities for the RFC ablation experiment (Experiment 4).

Reuses Exp3 (rfc_core) client construction with similarity_k5 full-cells clients,
and applies Exp6-style ablation variants.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

from real_data_experiments.common.data_splits import validate_split_ratios


DEFAULT_VARIANTS = [
    "full",
    "without_attention",
    "without_cnn",
    "without_lstm",
]


@dataclass
class ExperimentConfig:
    """Runtime configuration for the RFC ablation experiment (Exp4)."""

    experiment_name: str = "rfc_ablation"
    workflow: str = "all"
    seed: int = 2026
    tensor_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"
    regions_path: str = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv"
    partition_file: str = "real_data_experiments/region_client_full_cells/partitions/similarity_k5.json"
    output_dir: str = "results/real_data_experiments/diagnostic/exp4_rfc_ablation_similarity_k5"
    batch_size: int = 32
    learning_rate: float = 1e-3
    local_epochs: int = 1
    communication_rounds: int = 1
    sequence_length: int = 12
    prediction_horizon: int = 1
    prediction_sample_limit: int = 500
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_name: str = ""
    device: str = "cuda"
    use_channels: list[int] = field(default_factory=lambda: [0, 1])
    target_channel: int = 0
    variants: list[str] | None = None
    max_samples_per_client_split: int | None = None
    input_normalization: bool = True
    target_normalization: bool = True
    input_normalization_eps: float = 1e-6
    target_normalization_eps: float = 1e-6
    show_progress: bool = True
    progress_interval: int = 20

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["variants"] = list(self.variants or DEFAULT_VARIANTS)
        payload["experiment_id"] = "exp4_rfc_ablation"
        payload["client_setting"] = "similarity_full_cells"
        return payload


def parse_variants(raw_text: str | None) -> list[str] | None:
    if raw_text is None or not raw_text.strip():
        return None
    return [part.strip() for part in raw_text.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the RFC ablation experiment (Exp4)."""

    parser = argparse.ArgumentParser(
        description="RFC ablation experiment (Exp4) — similarity_k5 full-cells clients with model structure ablation"
    )
    parser.add_argument("--workflow", choices=["all"], default="all")
    parser.add_argument("--tensor-path", type=str, required=True)
    parser.add_argument(
        "--regions-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv",
    )
    parser.add_argument(
        "--partition-file",
        type=str,
        default="real_data_experiments/region_client_full_cells/partitions/similarity_k5.json",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--variants", type=str, default="")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "gpu", "cpu", "auto"],
        help="Runtime device. Default: cuda. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples-per-client-split", type=int, default=0)
    parser.add_argument("--input-normalization", dest="input_normalization", action="store_true", default=True)
    parser.add_argument("--no-input-normalization", dest="input_normalization", action="store_false")
    parser.add_argument("--target-normalization", dest="target_normalization", action="store_true", default=True)
    parser.add_argument("--no-target-normalization", dest="target_normalization", action="store_false")
    parser.add_argument("--show-progress", action="store_true", default=False)
    parser.add_argument("--progress-interval", type=int, default=20)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Construct ExperimentConfig from parsed CLI arguments."""
    split_name = validate_split_ratios(args.train_ratio, args.val_ratio)

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
        split_name=split_name,
        device=args.device,
        variants=parse_variants(args.variants),
        max_samples_per_client_split=(None if int(args.max_samples_per_client_split) <= 0 else int(args.max_samples_per_client_split)),
        input_normalization=args.input_normalization,
        target_normalization=args.target_normalization,
        show_progress=bool(args.show_progress),
        progress_interval=int(args.progress_interval),
    )
