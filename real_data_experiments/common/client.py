"""Generic client-side helpers for real-data federated training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader


@dataclass
class LocalTrainingResult:
    """Container for a single client's local training result."""

    client_id: int
    sample_count: int
    state_dict: dict[str, torch.Tensor]
    train_loss: float


class FedClient:
    """A lightweight reusable local-training wrapper."""

    def __init__(
        self,
        client_id: int,
        model_fn: Callable[[], torch.nn.Module],
        train_loader: DataLoader,
        device: str,
        learning_rate: float,
        local_epochs: int,
        criterion: torch.nn.Module,
    ) -> None:
        self.client_id = client_id
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.criterion = criterion

    def train(self, global_state_dict: dict[str, torch.Tensor]) -> LocalTrainingResult:
        """Train a local model initialized from the current global state."""
        model = self.model_fn().to(self.device)
        model.load_state_dict(copy.deepcopy(global_state_dict))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()

        epoch_losses: list[float] = []
        for _ in range(self.local_epochs):
            batch_losses: list[float] = []
            for features, targets in self.train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                predictions = model(features)
                loss = self.criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            epoch_losses.append(float(sum(batch_losses) / max(len(batch_losses), 1)))

        return LocalTrainingResult(
            client_id=self.client_id,
            sample_count=len(self.train_loader.dataset),
            state_dict=copy.deepcopy(model.state_dict()),
            train_loss=float(sum(epoch_losses) / max(len(epoch_losses), 1)),
        )
