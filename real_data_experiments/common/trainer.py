"""Reusable federated-training helpers for real-data experiments."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch

from .client import FedClient
from .fedavg import fedavg_aggregate


def run_federated_rounds(
    global_model: torch.nn.Module,
    clients: list[FedClient],
    communication_rounds: int,
    evaluate_fn: Callable[[torch.nn.Module], dict[str, float]] | None = None,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """Run standard FedAvg rounds and optionally collect round-level metrics."""
    history: list[dict[str, float]] = []
    global_state = copy.deepcopy(global_model.state_dict())

    for round_idx in range(1, communication_rounds + 1):
        local_results = [client.train(global_state) for client in clients]
        aggregated_state = fedavg_aggregate(
            [result.state_dict for result in local_results],
            [result.sample_count for result in local_results],
        )
        global_model.load_state_dict(aggregated_state)
        global_state = copy.deepcopy(aggregated_state)

        record = {
            "communication_round": round_idx,
            "train_loss": float(np.mean([result.train_loss for result in local_results])),
        }
        if evaluate_fn is not None:
            record.update(evaluate_fn(global_model))
        history.append(record)

    return global_model, history
