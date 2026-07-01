"""Generic federated mechanism evaluation.

Provides a reusable function to run multiple FL mechanisms against
the same client data, producing unified metrics.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from real_data_experiments.common.fedavg import fedavg_aggregate
from real_data_experiments.common.fedprox import train_client_fedprox
from real_data_experiments.common.local_finetune import local_finetune_model
from real_data_experiments.common.metrics import compute_regression_metrics


def evaluate_mechanisms(
    client_data_list: list[Any],
    model_factory: Callable[[], nn.Module],
    device: str,
    config: dict | None = None,
    methods: list[str] | None = None,
    target_scaler=None,
    target_channel_index: int = 0,
    allow_scaled_naive_last_value: bool = False,
    calendar_utils_fn: Callable | None = None,
) -> dict[str, pd.DataFrame]:
    """Run multiple federated mechanisms and return unified results.

    Args:
        client_data_list: list of objects with .train_loader, .test_loader, .client_id
        model_factory: callable that returns a fresh model instance
        device: 'cuda' or 'cpu'
        config: dict with lr, local_epochs, rounds, batch_size, etc.
        methods: list of method names to run
        target_scaler: optional target scaler
        calendar_utils_fn: optional function for CalendarProfileNaive

    Returns:
        dict mapping method name to metrics DataFrame
    """
    if config is None:
        config = {}
    if methods is None:
        methods = ["FedAvg"]

    lr = config.get("lr", 1e-3)
    local_epochs = config.get("local_epochs", 1)
    rounds = config.get("rounds", 1)
    mu = config.get("fedprox_mu", 0.001)
    local_ft_epochs = config.get("local_ft_epochs", 3)
    local_ft_lr = config.get("local_ft_lr", 5e-4)

    criterion = nn.MSELoss()
    all_results: dict[str, pd.DataFrame] = {}

    def _collect_metrics(model, data_list, method_name):
        rows = []
        for client_data in data_list:
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for features, targets in client_data.test_loader:
                    features = features.to(device)
                    preds = model(features).cpu().numpy()
                    y_pred.append(preds.reshape(-1))
                    y_true.append(targets.numpy().reshape(-1))
            if y_true:
                yt = np.concatenate(y_true)
                yp = np.concatenate(y_pred)
                if target_scaler is not None:
                    yt = target_scaler.inverse_transform(yt)
                    yp = target_scaler.inverse_transform(yp)
                metrics = compute_regression_metrics(yt, yp)
                metrics["client_id"] = client_data.client_id
                metrics["method"] = method_name
                metrics["num_test_samples"] = len(yt)
                rows.append(metrics)
        return pd.DataFrame(rows)

    # --- FedAvg ---
    if "FedAvg" in methods:
        global_model = model_factory().to(device)
        global_state = copy.deepcopy(global_model.state_dict())
        for _ in range(rounds):
            local_states, sample_counts = [], []
            for client_data in client_data_list:
                model = model_factory().to(device)
                model.load_state_dict(copy.deepcopy(global_state))
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                model.train()
                for _ in range(local_epochs):
                    for features, targets in client_data.train_loader:
                        features, targets = features.to(device), targets.to(device)
                        opt.zero_grad()
                        loss = criterion(model(features), targets)
                        loss.backward()
                        opt.step()
                local_states.append(copy.deepcopy(model.state_dict()))
                sample_counts.append(len(client_data.train_loader.dataset))
            global_state = fedavg_aggregate(local_states, sample_counts)
        global_model.load_state_dict(global_state)
        all_results["FedAvg"] = _collect_metrics(global_model, client_data_list, "FedAvg")

        # --- FedAvg+LocalFT ---
        if "FedAvg+LocalFT" in methods:
            ft_model = copy.deepcopy(global_model).to(device)
            ft_rows = []
            for client_data in client_data_list:
                result = local_finetune_model(
                    ft_model, client_data.train_loader, device,
                    epochs=local_ft_epochs, lr=local_ft_lr,
                    test_loader=client_data.test_loader,
                    target_scaler=target_scaler,
                )
                result["client_id"] = client_data.client_id
                result["method"] = "FedAvg+LocalFT"
                ft_rows.append(result)
            ft_df = pd.DataFrame(ft_rows)
            if not ft_df.empty:
                ft_df["num_test_samples"] = ft_df.get("num_test_samples", 0)
            all_results["FedAvg+LocalFT"] = ft_df

    # --- FedProx ---
    if "FedProx" in methods:
        global_model = model_factory().to(device)
        global_state = copy.deepcopy(global_model.state_dict())
        for _ in range(rounds):
            local_states, sample_counts = [], []
            for client_data in client_data_list:
                model = model_factory().to(device)
                state_dict, n_samples, _, _ = train_client_fedprox(
                    model, global_state, client_data.train_loader,
                    device, lr, local_epochs, mu, criterion,
                )
                local_states.append(state_dict)
                sample_counts.append(n_samples)
            global_state = fedavg_aggregate(local_states, sample_counts)
        global_model.load_state_dict(global_state)
        all_results["FedProx"] = _collect_metrics(global_model, client_data_list, "FedProx")

        # --- FedProx+LocalFT ---
        if "FedProx+LocalFT" in methods:
            ft_model = copy.deepcopy(global_model).to(device)
            ft_rows = []
            for client_data in client_data_list:
                result = local_finetune_model(
                    ft_model, client_data.train_loader, device,
                    epochs=local_ft_epochs, lr=local_ft_lr,
                    test_loader=client_data.test_loader,
                    target_scaler=target_scaler,
                )
                result["client_id"] = client_data.client_id
                result["method"] = "FedProx+LocalFT"
                ft_rows.append(result)
            ft_df = pd.DataFrame(ft_rows)
            if not ft_df.empty:
                ft_df["num_test_samples"] = ft_df.get("num_test_samples", 0)
            all_results["FedProx+LocalFT"] = ft_df

    # --- Independent ---
    if "Independent" in methods:
        ind_rows = []
        for client_data in client_data_list:
            model = model_factory().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            total_epochs = rounds * local_epochs
            for _ in range(total_epochs):
                for features, targets in client_data.train_loader:
                    features, targets = features.to(device), targets.to(device)
                    opt.zero_grad()
                    loss = criterion(model(features), targets)
                    loss.backward()
                    opt.step()
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for features, targets in client_data.test_loader:
                    features = features.to(device)
                    preds = model(features).cpu().numpy()
                    y_pred.append(preds.reshape(-1))
                    y_true.append(targets.numpy().reshape(-1))
            if y_true:
                yt = np.concatenate(y_true)
                yp = np.concatenate(y_pred)
                if target_scaler is not None:
                    yt = target_scaler.inverse_transform(yt)
                    yp = target_scaler.inverse_transform(yp)
                m = compute_regression_metrics(yt, yp)
                m["client_id"] = client_data.client_id
                m["method"] = "Independent"
                m["num_test_samples"] = len(yt)
                ind_rows.append(m)
        all_results["Independent"] = pd.DataFrame(ind_rows)

    # --- NaiveLastValue ---
    if "NaiveLastValue" in methods:
        if not allow_scaled_naive_last_value:
            nlv_rows = []
            for client_data in client_data_list:
                nlv_rows.append({
                    "client_id": client_data.client_id,
                    "method": "NaiveLastValue",
                    "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                    "not_applicable_reason": "raw last-value anchor unavailable in common mechanism_eval; use experiment-specific evaluator",
                })
            all_results["NaiveLastValue"] = pd.DataFrame(nlv_rows)
        else:
            nlv_rows = []
            for client_data in client_data_list:
                y_true, y_pred = [], []
                for features, targets in client_data.test_loader:
                    y_true.append(targets.numpy().reshape(-1))
                    last_inp = features[:, target_channel_index, -1].detach().cpu().numpy().reshape(-1)
                    y_pred.append(last_inp)
                if y_true:
                    yt = np.concatenate(y_true)
                    yp = np.concatenate(y_pred)
                    if target_scaler is not None:
                        yt = target_scaler.inverse_transform(yt)
                        yp = target_scaler.inverse_transform(yp)
                    m = compute_regression_metrics(yt, yp)
                    m["client_id"] = client_data.client_id
                    m["method"] = "NaiveLastValue"
                    m["num_test_samples"] = len(yt)
                    nlv_rows.append(m)
            all_results["NaiveLastValue"] = pd.DataFrame(nlv_rows)

    return all_results


def _localft_result_to_metric_row(result: dict, client_id: int, method: str) -> dict:
    return {
        "client_id": client_id,
        "method": method,
        "rmse": result.get("rmse_after_ft", float("nan")),
        "mae": result.get("mae_after_ft", float("nan")),
        "r2": result.get("r2_after_ft", float("nan")),
        "rmse_before_ft": result.get("rmse_before_ft", float("nan")),
        "mae_before_ft": result.get("mae_before_ft", float("nan")),
        "r2_before_ft": result.get("r2_before_ft", float("nan")),
        "local_ft_gain_rmse": result.get("local_ft_gain_rmse", float("nan")),
        "num_test_samples": result.get("num_test_samples", 0),
    }
