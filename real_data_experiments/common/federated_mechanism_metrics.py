"""Federated mechanism advantage statistics."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def build_federated_mechanism_advantage(main_metrics_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("GlobalInitGain", "RandomInit+LocalFT", "FedAvg+LocalFT", "联邦共享初始化是否有效"),
        ("PersonalizationGain", "FedAvg", "FedAvg+LocalFT", "本地微调是否提升纯 FedAvg"),
        ("FedProxGain", "FedAvg", "FedProx", "FedProx non-IID 约束是否优于 FedAvg"),
        ("FedProxPersonalizationGain", "FedProx", "FedProx+LocalFT", "FedProx 之后本地微调是否进一步提升"),
        ("CentralizedGap", "CentralizedUpperBound", "FedAvg+LocalFT", "联邦个性化相比集中式上界的差距"),
        ("NaiveRobustGain", "NaiveLastValue", "FedAvg+LocalFT", "联邦个性化相比简单 last-value 的鲁棒性"),
        ("IndependentGap", "Independent", "FedAvg+LocalFT", "联邦个性化相比独立训练的差距"),
        ("CalendarPersonalizationGain", "CalendarFeatureFedAvg-Full", "CalendarFeatureFedAvg-Full+LocalFT", "Calendar 模型本地微调效果"),
    ]
    rows = []
    metrics = main_metrics_df.set_index("method")
    for metric_name, baseline, federated, interpretation in comparisons:
        if baseline not in metrics.index or federated not in metrics.index:
            continue
        rmse_b = float(metrics.loc[baseline, "rmse"])
        rmse_f = float(metrics.loc[federated, "rmse"])
        absolute_gain = rmse_b - rmse_f
        relative_gain = (absolute_gain / rmse_b) * 100.0 if rmse_b != 0 else 0.0
        rows.append({
            "metric_name": metric_name,
            "baseline_method": baseline,
            "federated_method": federated,
            "rmse_baseline": rmse_b,
            "rmse_federated": rmse_f,
            "absolute_gain": absolute_gain,
            "relative_gain_percent": relative_gain,
            "is_improvement": absolute_gain > 0,
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def build_client_win_rate(client_metrics_df: pd.DataFrame, main_metrics_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("RandomInit+LocalFT", "FedAvg+LocalFT"),
        ("FedAvg", "FedAvg+LocalFT"),
        ("FedAvg", "FedProx"),
        ("FedProx", "FedProx+LocalFT"),
        ("CentralizedUpperBound", "FedAvg+LocalFT"),
        ("NaiveLastValue", "FedAvg+LocalFT"),
        ("Independent", "FedAvg+LocalFT"),
        ("CalendarFeatureFedAvg-Full", "CalendarFeatureFedAvg-Full+LocalFT"),
    ]
    rows = []
    for baseline, federated in comparisons:
        base = client_metrics_df[client_metrics_df["method"] == baseline]
        fed = client_metrics_df[client_metrics_df["method"] == federated]
        if base.empty or fed.empty:
            continue
        base_rmse = base.set_index("client_id")["rmse"]
        fed_rmse = fed.set_index("client_id")["rmse"]
        common = base_rmse.index.intersection(fed_rmse.index)
        if len(common) == 0:
            continue
        wins = int((fed_rmse[common] < base_rmse[common]).sum())
        rows.append({
            "baseline_method": baseline,
            "federated_method": federated,
            "metric": "rmse",
            "wins": wins,
            "total_clients": len(common),
            "win_rate": wins / len(common),
            "mean_baseline": float(base_rmse[common].mean()),
            "mean_federated": float(fed_rmse[common].mean()),
            "mean_absolute_gain": float((base_rmse[common] - fed_rmse[common]).mean()),
        })
    return pd.DataFrame(rows)
