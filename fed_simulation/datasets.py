from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_traffic_pattern(T: int, pattern: str) -> np.ndarray:
    t = np.arange(T, dtype=np.float32) / max(T, 1)
    if pattern == "morning_peak":
        return np.exp(-((t - 0.25) / 0.1) ** 2) + 0.3
    if pattern == "evening_peak":
        return np.exp(-((t - 0.7) / 0.1) ** 2) + 0.3
    if pattern == "double_peak":
        return np.exp(-((t - 0.25) / 0.1) ** 2) + np.exp(-((t - 0.7) / 0.12) ** 2) + 0.2
    return np.ones(T, dtype=np.float32)


class ComplexHeterogeneousDataset(Dataset):
    def __init__(
        self,
        client_id: int,
        num_samples: int,
        K: int,
        T: int,
        noise: float = 0.1,
        dist_type: str = "normal",
        pattern_type: str = "double_peak",
        missing_rate: float = 0.0,
        outlier_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.K = K
        self.T = T
        rng = np.random.RandomState(seed + client_id * 1000)

        if dist_type == "normal":
            self.X = rng.randn(num_samples, K, T).astype(np.float32)
        elif dist_type == "t":
            self.X = rng.standard_t(df=4, size=(num_samples, K, T)).astype(np.float32) * 0.5
        elif dist_type == "chi2":
            self.X = rng.chisquare(df=4, size=(num_samples, K, T)).astype(np.float32) * 0.3 - 0.5
        elif dist_type == "lognormal":
            self.X = rng.lognormal(mean=0, sigma=0.6, size=(num_samples, K, T)).astype(np.float32) - 0.5
        else:
            self.X = rng.randn(num_samples, K, T).astype(np.float32)

        pattern = generate_traffic_pattern(T, pattern_type)
        pattern = pattern[np.newaxis, np.newaxis, :]

        base_feature = self.X[:, :, T // 4 : T * 3 // 4].mean(axis=(1, 2))
        rel = 0.7 * np.sin(base_feature) + 0.3 * np.cos(self.X.max(axis=(1, 2)))
        if client_id == 0:
            rel += 0.15 * np.tanh(self.X[:, :, : T // 2].mean(axis=(1, 2)))
        elif client_id == 1:
            rel += 0.15 * np.sin(self.X[:, :, T // 2 :].mean(axis=(1, 2)))
        else:
            rel += 0.15 * np.cos(self.X.std(axis=(1, 2)))

        target = (rel[:, np.newaxis] * pattern.squeeze(axis=1)).mean(axis=1)
        self.y = (target + noise * rng.randn(num_samples).astype(np.float32)).astype(np.float32)

        if missing_rate > 0:
            missing_mask = rng.rand(*self.X.shape) < missing_rate
            self.X[missing_mask] = 0.0
        if outlier_rate > 0:
            outlier_mask = rng.rand(*self.X.shape) < outlier_rate
            noise_tensor = rng.randn(*self.X.shape).astype(np.float32) * 5.0
            self.X[outlier_mask] += noise_tensor[outlier_mask]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
