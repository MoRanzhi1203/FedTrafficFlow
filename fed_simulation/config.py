from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    K: int = 5
    T: int = 24
    hidden_dim: int = 128
    num_heads: int = 4
    num_clients: int = 3
    num_rounds: int = 5
    local_epochs: int = 5
    batch_size: int = 8
    lr: float = 0.001
    noise: float = 0.1
    num_runs: int = 3
    seeds: list[int] = field(default_factory=list)
    client_dropout_rate: float = 0.0
    dp_noise_std: float = 0.0
    show_plot: bool = False
    output_dir: str = ""
    model_name: str = "cnn"
    enable_amp: bool = True
    enable_compile: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    train_split: float = 0.8
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    robust_loss: str = "huber"
    huber_delta: float = 1.0
    agg_lambda: float = 0.5
    server_momentum: float = 0.1
    trim_ratio: float = 0.1
    log_level: str = "INFO"
    max_train_batches: int = 0
    max_eval_batches: int = 0
    save_round_snapshots: bool = False

    model_types: list[str] = field(
        default_factory=lambda: ["full", "no_attention", "lstm_only", "spatial_only", "weak"]
    )
    agg_methods: list[str] = field(
        default_factory=lambda: ["fedavg", "loss_weighted", "data_loss_weighted", "trimmed_loss_weighted"]
    )
    dist_types: list[str] = field(default_factory=lambda: ["normal", "t", "chi2", "lognormal"])
    pattern_types: list[str] = field(
        default_factory=lambda: ["morning_peak", "evening_peak", "double_peak", "flat"]
    )
    missing_rates: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1])
    outlier_rates: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05])

    def validate(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split 必须位于 (0, 1) 区间")
        if self.num_clients <= 0 or self.num_rounds <= 0 or self.local_epochs <= 0:
            raise ValueError("num_clients、num_rounds、local_epochs 必须为正整数")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def ensure_seeds(self) -> list[int]:
        if self.seeds:
            if len(self.seeds) < self.num_runs:
                last_seed = self.seeds[-1]
                for index in range(self.num_runs - len(self.seeds)):
                    self.seeds.append(last_seed + (index + 1) * 137)
        else:
            self.seeds = [42 + i * 137 for i in range(self.num_runs)]
        return self.seeds

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


def namespace_to_config(namespace: Any, model_name: str, default_output_dir: str) -> ExperimentConfig:
    seeds_raw = getattr(namespace, "seeds", None)
    seeds = []
    if seeds_raw:
        seeds = [int(item.strip()) for item in seeds_raw.split(",") if item.strip()]

    config = ExperimentConfig(
        K=namespace.K,
        T=namespace.T,
        hidden_dim=namespace.hidden_dim,
        num_heads=namespace.num_heads,
        num_clients=namespace.num_clients,
        num_rounds=namespace.num_rounds,
        local_epochs=namespace.local_epochs,
        batch_size=namespace.batch_size,
        lr=namespace.lr,
        noise=namespace.noise,
        num_runs=namespace.num_runs,
        seeds=seeds,
        client_dropout_rate=namespace.client_dropout_rate,
        dp_noise_std=namespace.dp_noise_std,
        show_plot=namespace.show_plot,
        output_dir=namespace.output_dir or default_output_dir,
        model_name=model_name,
        enable_amp=not getattr(namespace, "disable_amp", False),
        enable_compile=not getattr(namespace, "disable_compile", False),
        num_workers=namespace.num_workers,
        pin_memory=not getattr(namespace, "disable_pin_memory", False),
        persistent_workers=not getattr(namespace, "disable_persistent_workers", False),
        prefetch_factor=namespace.prefetch_factor,
        robust_loss=namespace.robust_loss,
        huber_delta=namespace.huber_delta,
        agg_lambda=namespace.agg_lambda,
        server_momentum=namespace.server_momentum,
        trim_ratio=namespace.trim_ratio,
        log_level=namespace.log_level,
        max_train_batches=namespace.max_train_batches,
        max_eval_batches=namespace.max_eval_batches,
        save_round_snapshots=namespace.save_round_snapshots,
    )
    config.validate()
    config.ensure_seeds()
    return config
