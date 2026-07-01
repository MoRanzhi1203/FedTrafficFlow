"""Estimate communication cost for real-data federated training.

Computes parameter count, per-round upload/download, and total communication
for different model configurations used in real experiments 1-6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CNNLSTMAttentionRegressor(nn.Module):
    """Recreated model architecture matching sic_core.py."""

    def __init__(self, input_channels: int = 2, hidden_dim: int = 32, prediction_horizon: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attn_query = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, x):
        x = self.encoder(x).transpose(1, 2)
        x, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn_query(x).squeeze(-1), dim=1)
        x = torch.sum(x * attn_w.unsqueeze(-1), dim=1)
        return self.head(x)


class CalendarFeatureCNN(nn.Module):
    """Calendar feature version: traffic branch + calendar branch with residual gate."""

    def __init__(self, input_channels: int = 2, calendar_dim: int = 9, hidden_dim: int = 32, prediction_horizon: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.cal_encoder = nn.Sequential(
            nn.Linear(calendar_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        self.head = nn.Linear(hidden_dim, prediction_horizon)
        # approximate param count for attention (not fully implemented here)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_cost(num_params: int, num_clients: int, rounds: int, fp_bytes: int = 4) -> dict[str, Any]:
    param_bytes = num_params * fp_bytes
    param_mb = param_bytes / (1024 * 1024)
    return {
        "num_parameters": num_params,
        "parameter_bytes_fp32": param_bytes,
        "parameter_mb_fp32": round(param_mb, 4),
        "upload_per_client_per_round_mb": round(param_mb, 4),
        "download_per_client_per_round_mb": round(param_mb, 4),
        "total_upload_per_round_mb": round(param_mb * num_clients, 4),
        "total_download_per_round_mb": round(param_mb * num_clients, 4),
        f"total_upload_r{rounds}_mb": round(param_mb * num_clients * rounds, 4),
        f"total_download_r{rounds}_mb": round(param_mb * num_clients * rounds, 4),
    }


def main():
    models = {
        "CNN-LSTM-Attention (baseline)": CNNLSTMAttentionRegressor(input_channels=2),
        "CNN-LSTM-Attention (1ch)": CNNLSTMAttentionRegressor(input_channels=1),
        "CalendarFeatureCNN": CalendarFeatureCNN(input_channels=2, calendar_dim=9),
    }

    configs = [
        {"name": "Exp1 (K=5, R=20)", "num_clients": 5, "rounds": 20},
        {"name": "Exp3 (K=5, R=20)", "num_clients": 5, "rounds": 20},
        {"name": "Exp5 spatial (K=3, R=20)", "num_clients": 3, "rounds": 20},
        {"name": "Exp5 spatial (K=10, R=20)", "num_clients": 10, "rounds": 20},
        {"name": "Exp1 diagnostic (K=5, R=5)", "num_clients": 5, "rounds": 5},
    ]

    results = []
    for model_name, model in models.items():
        params = count_parameters(model)
        for cfg in configs:
            cost = estimate_cost(params, cfg["num_clients"], cfg["rounds"])
            cost["model"] = model_name
            cost["config"] = cfg["name"]
            results.append(cost)

    output_dir = Path("results/real_data_experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "communication_cost_estimate.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Communication Cost Estimate ===")
    print(f"{'Model':<35} {'Config':<30} {'Params':>8} {'MB/Client/Round':>15} {'Total MB(R20)':>15}")
    print("-" * 105)
    for r in results:
        params = r["num_parameters"]
        mb_per = r["upload_per_client_per_round_mb"]
        total_key = f"total_upload_r{r['config'].split('R=')[-1].split(')')[0] if 'R=' in r['config'] else '20'}_mb"
        total = r.get("total_upload_r20_mb", r.get("total_upload_r5_mb", 0))
        print(f"{r['model']:<35} {r['config']:<30} {params:>8} {mb_per:>15.4f} {total:>15.4f}")

    print(f"\nOutput: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
