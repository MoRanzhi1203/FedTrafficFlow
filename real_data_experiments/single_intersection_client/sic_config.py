"""Configuration utilities for the single-intersection client experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Runtime configuration for the single-intersection client experiment."""

    experiment_name: str = "single_intersection_client"
    seed: int = 42
    input_path: str = "data/analysis/node_intersection_flow_parquet"
    output_dir: str = "results/real_data_experiments/single_intersection_client"
    num_clients: int = 5
    selected_clients: list[int] | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    learning_rate: float = 1e-3
    local_epochs: int = 2
    communication_rounds: int = 3
    sequence_length: int = 24
    prediction_horizon: int = 1
    model_name: str = "cnn_lstm_attention"
    device: str = "auto"
    target_column: str = "路口车流量"
    max_chunks: int | None = 7
    prediction_sample_limit: int = 200
    workflow: str = "all"
    independent_total_epochs: int | None = None

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
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-path", type=str, default="data/analysis/node_intersection_flow_parquet")
    parser.add_argument("--output-dir", type=str, default="results/real_data_experiments/single_intersection_client")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--selected-clients", type=str, default="")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    parser.add_argument("--target-column", type=str, default="路口车流量")
    parser.add_argument("--max-chunks", type=int, default=7)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Construct an ExperimentConfig from parsed CLI arguments."""
    return ExperimentConfig(
        workflow=args.workflow,
        seed=args.seed,
        input_path=args.input_path,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        selected_clients=parse_selected_clients(args.selected_clients),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        communication_rounds=args.rounds,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        device=args.device,
        target_column=args.target_column,
        max_chunks=args.max_chunks,
    )
