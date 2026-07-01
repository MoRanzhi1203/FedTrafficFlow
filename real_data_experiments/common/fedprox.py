"""FedProx support for real-data experiments.

Provides FedProx-aware training with proximal regularization term.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class FedProxResult:
    """Result of a single FedProx local training episode."""
    client_id: int
    sample_count: int
    state_dict: dict[str, torch.Tensor]
    train_loss: float
    proximal_loss: float


def train_client_fedprox(
    model: nn.Module,
    global_state_dict: dict[str, torch.Tensor],
    train_loader: DataLoader,
    device: str,
    learning_rate: float,
    local_epochs: int,
    mu: float,
    criterion: nn.Module | None = None,
) -> tuple[dict[str, torch.Tensor], int, float, float]:
    """Train one client with FedProx proximal term.

    FedProx objective:
        L_total = L_pred(w) + (mu/2) * ||w - w_global||^2

    Args:
        model: client model (already loaded with global_state_dict)
        global_state_dict: frozen copy of global model params
        train_loader: local training data
        device: 'cuda' or 'cpu'
        learning_rate: optimizer learning rate
        local_epochs: number of local epochs
        mu: proximal regularization coefficient
        criterion: loss function (defaults to MSELoss)

    Returns:
        (state_dict, sample_count, train_loss, proximal_loss)
    """
    if criterion is None:
        criterion = nn.MSELoss()

    model.load_state_dict(copy.deepcopy(global_state_dict))
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sample_count = len(train_loader.dataset)
    total_pred_loss = 0.0
    total_prox_loss = 0.0
    num_batches = 0

    for _ in range(local_epochs):
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(features)
            pred_loss = criterion(preds, targets)

            # FedProx proximal term: (mu/2) * ||w - w_global||^2
            prox_term = 0.0
            for name, param in model.named_parameters():
                global_param = global_state_dict[name].to(device)
                prox_term += torch.sum((param - global_param) ** 2)
            prox_loss = (mu / 2.0) * prox_term

            total_loss = pred_loss + prox_loss
            total_loss.backward()
            optimizer.step()

            total_pred_loss += float(pred_loss.item())
            total_prox_loss += float(prox_loss.item())
            num_batches += 1

    avg_pred_loss = total_pred_loss / max(num_batches, 1)
    avg_prox_loss = total_prox_loss / max(num_batches, 1)

    return copy.deepcopy(model.state_dict()), sample_count, avg_pred_loss, avg_prox_loss


def run_fedprox_rounds(
    model_factory,
    clients: list,
    communication_rounds: int,
    mu: float,
    device: str,
    learning_rate: float = 1e-3,
    local_epochs: int = 1,
    evaluate_fn=None,
) -> tuple[nn.Module, list[dict]]:
    """Run FedProx federated training for multiple rounds.

    Args:
        model_factory: callable that returns a fresh model instance
        clients: list of objects with train_loader and client_id attributes
        communication_rounds: number of rounds
        mu: proximal coefficient
        device: device string
        learning_rate: local optimizer lr
        local_epochs: local epochs per round
        evaluate_fn: optional round eval callback

    Returns:
        (trained_global_model, history_list)
    """
    from .fedavg import fedavg_aggregate

    history: list[dict] = []
    global_model = model_factory().to(device)
    global_state = copy.deepcopy(global_model.state_dict())

    for round_idx in range(1, communication_rounds + 1):
        local_states = []
        sample_counts = []
        round_losses = []
        round_prox_losses = []

        for client in clients:
            model = model_factory().to(device)
            global_state_device = {k: v.detach().to(device) for k, v in global_state.items()}
            state_dict, n_samples, pred_loss, prox_loss = train_client_fedprox(
                model=model,
                global_state_dict=global_state_device,
                train_loader=client.train_loader,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                mu=mu,
            )
            local_states.append(state_dict)
            sample_counts.append(n_samples)
            round_losses.append(pred_loss)
            round_prox_losses.append(prox_loss)

        global_state = fedavg_aggregate(local_states, sample_counts)

        record = {
            "communication_round": round_idx,
            "train_loss": float(sum(round_losses) / max(len(round_losses), 1)),
            "prox_loss": float(sum(round_prox_losses) / max(len(round_prox_losses), 1)),
            "method": "FedProx",
            "mu": mu,
        }
        if evaluate_fn is not None:
            record.update(evaluate_fn(global_model))
        history.append(record)

    global_model.load_state_dict(global_state)
    return global_model, history
