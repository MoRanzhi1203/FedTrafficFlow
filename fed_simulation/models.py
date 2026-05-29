from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveSwish(nn.Module):
    def __init__(self, trainable: bool = True) -> None:
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones(1, dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta.to(x.dtype) * x)


class CNNEncoder(nn.Module):
    def __init__(self, K: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=K, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            AdaptiveSwish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.einsum("ij,bjf->bif", A_norm, X))


class GCNEncoder(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int) -> None:
        super().__init__()
        self.K = K
        self.node_proj = nn.Sequential(nn.Linear(T, hidden_dim), nn.LayerNorm(hidden_dim), AdaptiveSwish())
        self.gcn1 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = AdaptiveSwish()
        self.A_param = nn.Parameter(torch.randn(K, K) * 0.01)

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        A = torch.relu(A)
        I = torch.eye(self.K, device=A.device, dtype=A.dtype)
        A = A + I
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = self.node_proj(x)
        A_norm = self._normalize_adj(self.A_param)
        H = self.act(self.norm1(self.gcn1(X, A_norm)))
        H = self.act(self.norm2(self.gcn2(H, A_norm)))
        return H.mean(dim=1)


class BiLSTMEncoder(nn.Module):
    def __init__(self, K: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=K,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.proj(x.mean(dim=1))


class FusionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            AdaptiveSwish(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CNNLSTMAttention(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cnn = CNNEncoder(K, hidden_dim)
        self.lstm = BiLSTMEncoder(K, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        feat = torch.stack([self.cnn(x), self.lstm(x)], dim=1)
        attn_out, attn_w = self.mha(feat, feat, feat)
        fused = self.norm(attn_out + feat).mean(dim=1)
        return self.head(fused), attn_w


class GCNLSTMAttention(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.gcn = GCNEncoder(K, T, hidden_dim)
        self.lstm = BiLSTMEncoder(K, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        feat = torch.stack([self.gcn(x), self.lstm(x)], dim=1)
        attn_out, attn_w = self.mha(feat, feat, feat)
        fused = self.norm(attn_out + feat).mean(dim=1)
        return self.head(fused), attn_w


class CNNLSTMNoAttention(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cnn = CNNEncoder(K, hidden_dim)
        self.lstm = BiLSTMEncoder(K, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), AdaptiveSwish())
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        fused = self.fuse(torch.cat([self.cnn(x), self.lstm(x)], dim=1))
        return self.head(fused), None


class GCNLSTMNoAttention(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.gcn = GCNEncoder(K, T, hidden_dim)
        self.lstm = BiLSTMEncoder(K, hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), AdaptiveSwish())
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        fused = self.fuse(torch.cat([self.gcn(x), self.lstm(x)], dim=1))
        return self.head(fused), None


class LSTMOnly(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.lstm = BiLSTMEncoder(K, hidden_dim)
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        return self.head(self.lstm(x)), None


class CNNOnly(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cnn = CNNEncoder(K, hidden_dim)
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        return self.head(self.cnn(x)), None


class GCNOnly(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.gcn = GCNEncoder(K, T, hidden_dim)
        self.head = FusionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x.float()
        return self.head(self.gcn(x)), None


class WeakModel(nn.Module):
    def __init__(self, K: int, T: int, hidden_dim: int = 16, num_heads: int = 1) -> None:
        super().__init__()
        self.extractor = nn.Sequential(nn.Linear(K * T, hidden_dim), nn.ReLU(), nn.Dropout(0.5))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        x = x.float().reshape(x.size(0), -1)
        return self.fc(self.extractor(x)), None


CNN_MODEL_BUILDERS = {
    "full": lambda K, T, hidden_dim, num_heads: CNNLSTMAttention(K, T, hidden_dim, num_heads),
    "no_attention": lambda K, T, hidden_dim, num_heads: CNNLSTMNoAttention(K, T, hidden_dim, num_heads),
    "lstm_only": lambda K, T, hidden_dim, num_heads: LSTMOnly(K, T, hidden_dim, num_heads),
    "spatial_only": lambda K, T, hidden_dim, num_heads: CNNOnly(K, T, hidden_dim, num_heads),
    "weak": lambda K, T, hidden_dim, num_heads: WeakModel(K, T, 16, num_heads),
}

GCN_MODEL_BUILDERS = {
    "full": lambda K, T, hidden_dim, num_heads: GCNLSTMAttention(K, T, hidden_dim, num_heads),
    "no_attention": lambda K, T, hidden_dim, num_heads: GCNLSTMNoAttention(K, T, hidden_dim, num_heads),
    "lstm_only": lambda K, T, hidden_dim, num_heads: LSTMOnly(K, T, hidden_dim, num_heads),
    "spatial_only": lambda K, T, hidden_dim, num_heads: GCNOnly(K, T, hidden_dim, num_heads),
    "weak": lambda K, T, hidden_dim, num_heads: WeakModel(K, T, 16, num_heads),
}

CNN_MODEL_LABELS = {
    "full": "Full (CNN+LSTM+Attn)",
    "no_attention": "No Attention",
    "lstm_only": "LSTM Only",
    "spatial_only": "CNN Only",
    "weak": "Weak Model",
}

GCN_MODEL_LABELS = {
    "full": "Full (GCN+LSTM+Attn)",
    "no_attention": "No Attention",
    "lstm_only": "LSTM Only",
    "spatial_only": "GCN Only",
    "weak": "Weak Model",
}
