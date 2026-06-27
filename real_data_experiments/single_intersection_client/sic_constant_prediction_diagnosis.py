"""Runtime diagnosis for experiment 1 constant-prediction behavior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from real_data_experiments.common.result_writer import write_text
from real_data_experiments.single_intersection_client.sic_config import ExperimentConfig, parse_selected_clients
from real_data_experiments.single_intersection_client.sic_core import (
    CNNLSTMAttentionRegressor,
    apply_dataset_normalization,
    build_client_data,
    fit_target_scaler,
    resolve_input_channels,
)


@dataclass
class ArrayStats:
    min: float
    max: float
    mean: float
    std: float

    def to_markdown(self, name: str) -> str:
        return f"- {name}: min={self.min:.6f}, max={self.max:.6f}, mean={self.mean:.6f}, std={self.std:.6f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose constant predictions for experiment 1")
    parser.add_argument(
        "--tensor-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt",
    )
    parser.add_argument(
        "--regions-path",
        type=str,
        default="data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv",
    )
    parser.add_argument("--selected-clients", type=str, default="290,284,318,288,289")
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mini-batch-steps", type=int, default=5)
    parser.add_argument(
        "--output-report",
        type=str,
        default="real_data_experiments/single_intersection_client/experiment1_constant_prediction_diagnosis_zh.md",
    )
    return parser


def to_stats(values: np.ndarray) -> ArrayStats:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return ArrayStats(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=0)),
    )


def collect_targets(dataset, limit: int | None = None) -> np.ndarray:
    count = len(dataset) if limit is None else min(len(dataset), limit)
    return np.asarray([float(dataset[index][1].reshape(-1)[0].item()) for index in range(count)], dtype=np.float64)


def collect_feature_stats(dataset, limit: int = 256) -> ArrayStats:
    count = min(len(dataset), limit)
    features = [dataset[index][0].numpy() for index in range(count)]
    stacked = np.stack(features, axis=0)
    return to_stats(stacked)


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2).item())
    return total ** 0.5


def compute_update_norm(before_state: dict[str, torch.Tensor], after_state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, after_tensor in after_state.items():
        diff = after_tensor.detach() - before_state[key]
        total += float(torch.sum(diff ** 2).item())
    return total ** 0.5


def main() -> None:
    args = build_arg_parser().parse_args()
    selected_clients = parse_selected_clients(args.selected_clients)
    if not selected_clients:
        raise ValueError("--selected-clients must not be empty.")

    config = ExperimentConfig(
        data_mode="tensor",
        tensor_path=args.tensor_path,
        regions_path=args.regions_path,
        num_clients=len(selected_clients),
        selected_clients=selected_clients,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    clients, split_summary, _ = build_client_data(config)
    scaler = fit_target_scaler(clients, eps=config.target_normalization_eps)

    lines: list[str] = [
        "# 实验 1 常数预测诊断报告",
        "",
        "## 1. 诊断配置",
        "",
        f"- device: {device}",
        f"- selected_clients: {selected_clients}",
        f"- sequence_length: {args.sequence_length}",
        f"- batch_size: {args.batch_size}",
        f"- learning_rate: {args.learning_rate}",
        f"- target_scaler.mean: {scaler.mean:.6f}",
        f"- target_scaler.std: {scaler.std:.6f}",
        "",
        "## 2. 数据集与目标统计",
        "",
    ]

    raw_clients = clients
    for client in raw_clients:
        record = f"client_id={client.client_id}, region_id={client.entity_id}"
        raw_y_train = collect_targets(client.train_loader.dataset)
        raw_y_val = collect_targets(client.val_loader.dataset)
        raw_y_test = collect_targets(client.test_loader.dataset)
        lines.append(f"### {record}")
        lines.append(f"- X train shape: {tuple(client.train_loader.dataset[0][0].shape)}")
        lines.append(f"- y train shape: {tuple(client.train_loader.dataset[0][1].shape)}")
        lines.append(f"- train sample count: {len(client.train_loader.dataset)}")
        lines.append(f"- val sample count: {len(client.val_loader.dataset)}")
        lines.append(f"- test sample count: {len(client.test_loader.dataset)}")
        lines.append(collect_feature_stats(client.train_loader.dataset).to_markdown("X_train"))
        lines.append(to_stats(raw_y_train).to_markdown("y_train"))
        lines.append(to_stats(raw_y_val).to_markdown("y_val"))
        lines.append(to_stats(raw_y_test).to_markdown("y_test"))
        y_train_norm = (raw_y_train - scaler.mean) / scaler.std
        y_val_norm = (raw_y_val - scaler.mean) / scaler.std
        y_test_norm = (raw_y_test - scaler.mean) / scaler.std
        lines.append(to_stats(y_train_norm).to_markdown("y_train_norm"))
        lines.append(to_stats(y_val_norm).to_markdown("y_val_norm"))
        lines.append(to_stats(y_test_norm).to_markdown("y_test_norm"))
        lines.append("")

    apply_dataset_normalization(clients, target_scaler=scaler)
    diagnosed_client = clients[0]
    features, targets = next(iter(diagnosed_client.train_loader))
    model = CNNLSTMAttentionRegressor(
        input_channels=resolve_input_channels(config),
        prediction_horizon=config.prediction_horizon,
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    features = features.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        predictions = model(features)
        denorm_predictions = predictions * scaler.std + scaler.mean

    lines.extend(
        [
            "## 3. 单 batch 前向检查",
            "",
            f"- batch x shape: {tuple(features.shape)}",
            f"- batch y shape: {tuple(targets.shape)}",
            f"- batch pred shape: {tuple(predictions.shape)}",
            to_stats(features.detach().cpu().numpy()).to_markdown("batch_x"),
            to_stats(targets.detach().cpu().numpy()).to_markdown("batch_y_norm"),
            to_stats(predictions.detach().cpu().numpy()).to_markdown("batch_pred_norm_before_train"),
            to_stats(denorm_predictions.detach().cpu().numpy()).to_markdown("batch_pred_denorm_before_train"),
            f"- single batch loss before train: {float(criterion(predictions, targets).item()):.6f}",
            "",
            "## 4. mini-batch 更新检查",
            "",
        ]
    )

    step_rows: list[str] = []
    for step in range(args.mini_batch_steps):
        before_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.train()
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()
        after_state = model.state_dict()
        update_norm = compute_update_norm(before_state, after_state)
        with torch.no_grad():
            predictions_after = model(features)
            denorm_after = predictions_after * scaler.std + scaler.mean
        step_rows.append(
            "| {step} | {loss:.6f} | {grad_norm:.6f} | {update_norm:.6f} | {pred_mean:.6f} | {pred_std:.6f} | {pred_denorm_mean:.6f} | {pred_denorm_std:.6f} |".format(
                step=step + 1,
                loss=float(loss.item()),
                grad_norm=grad_norm,
                update_norm=update_norm,
                pred_mean=float(predictions_after.mean().item()),
                pred_std=float(predictions_after.std().item()),
                pred_denorm_mean=float(denorm_after.mean().item()),
                pred_denorm_std=float(denorm_after.std().item()),
            )
        )

    lines.extend(
        [
            "| step | loss | grad_norm | update_norm | pred_mean_norm | pred_std_norm | pred_mean_denorm | pred_std_denorm |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            *step_rows,
            "",
            "## 5. 初步诊断结论",
            "",
            "- 若 `grad_norm` 和 `update_norm` 持续大于 0，则说明训练循环与参数更新并未完全失效。",
            "- 若 `pred_std_denorm` 始终很小，而 `batch_x` 量级远大于 `y_norm`，则说明模型更可能在大尺度输入上退化为近常数输出。",
            "- 本报告只提供运行时证据，不直接修改正式训练逻辑。",
        ]
    )

    write_text("\n".join(lines), args.output_report)
    print(f"[constant_prediction_diagnosis_report] {args.output_report}")
    for row in step_rows:
        print(row)


if __name__ == "__main__":
    main()
