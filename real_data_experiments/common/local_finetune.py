"""Local fine-tuning support for real-data experiments.

Provides LocalFT (local fine-tuning) on top of a global model.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HEAD_KEYWORDS = ("head", "regressor", "fc", "output")


def local_finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int = 3,
    lr: float = 5e-4,
    head_only: bool = False,
    criterion: nn.Module | None = None,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    target_scaler=None,
) -> dict:
    """Fine-tune a global model on local client data (LocalFT)."""
    if criterion is None:
        criterion = nn.MSELoss()

    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    model_copy.train()

    from real_data_experiments.common.metrics import compute_regression_metrics

    pre_metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    num_test_samples = 0
    if test_loader is not None:
        all_true, all_pred = [], []
        model_copy.eval()
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                preds = model_copy(features).cpu().numpy()
                all_pred.append(preds.reshape(-1))
                all_true.append(targets.numpy().reshape(-1))
        if all_true:
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            if target_scaler is not None:
                y_true = target_scaler.inverse_transform(y_true)
                y_pred = target_scaler.inverse_transform(y_pred)
            pre_metrics = compute_regression_metrics(y_true, y_pred)
            num_test_samples = len(y_true)

    if head_only:
        any_trainable = False
        for name, param in model_copy.named_parameters():
            trainable = any(k in name.lower() for k in HEAD_KEYWORDS)
            param.requires_grad = trainable
            any_trainable = any_trainable or trainable
        if not any_trainable:
            raise ValueError("head_only=True but no head/regressor/fc/output parameters found")

    optimizer = torch.optim.Adam([p for p in model_copy.parameters() if p.requires_grad], lr=lr)
    model_copy.train()
    for _ in range(epochs):
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model_copy(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

    post_metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    if test_loader is not None:
        all_true, all_pred = [], []
        model_copy.eval()
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                preds = model_copy(features).cpu().numpy()
                all_pred.append(preds.reshape(-1))
                all_true.append(targets.numpy().reshape(-1))
        if all_true:
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            if target_scaler is not None:
                y_true = target_scaler.inverse_transform(y_true)
                y_pred = target_scaler.inverse_transform(y_pred)
            post_metrics = compute_regression_metrics(y_true, y_pred)

    return {
        "rmse_before_ft": pre_metrics["rmse"],
        "rmse_after_ft": post_metrics["rmse"],
        "mae_before_ft": pre_metrics["mae"],
        "mae_after_ft": post_metrics["mae"],
        "r2_before_ft": pre_metrics["r2"],
        "r2_after_ft": post_metrics["r2"],
        "local_ft_gain_rmse": pre_metrics.get("rmse", float("nan")) - post_metrics.get("rmse", float("nan")),
        "num_test_samples": num_test_samples if num_test_samples > 0 else (len(y_true) if test_loader and all_true else 0),
    }


def local_finetune_all_clients(
    global_model: nn.Module,
    train_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    device: str,
    epochs: int = 3,
    lr: float = 5e-4,
    head_only: bool = False,
    target_scaler=None,
) -> list[dict]:
    """Run LocalFT for each client independently."""
    results = []
    for i, (train_ldr, test_ldr) in enumerate(zip(train_loaders, test_loaders)):
        result = local_finetune_model(
            model=global_model,
            train_loader=train_ldr,
            device=device,
            epochs=epochs,
            lr=lr,
            head_only=head_only,
            test_loader=test_ldr,
            target_scaler=target_scaler,
        )
        result["client_id"] = i
        results.append(result)
    return results
