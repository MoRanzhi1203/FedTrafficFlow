from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation_experiments.cnn_fed_base import cfb_core
from simulation_experiments.cnn_fed_enhanced_experiments import cfe_core
OUTPUT_DIR = PROJECT_ROOT / "results" / "simulation_experiments" / "dataset_overview"
DEPRECATED_DIR = OUTPUT_DIR / "deprecated"
README_NAME = "simulation_dataset_construction_overview_readme.md"
LEGACY_PNG = "simulation_dataset_construction_overview.png"
LEGACY_PDF = "simulation_dataset_construction_overview.pdf"
BASE_PNG = "base_simulation_dataset_overview.png"
BASE_PDF = "base_simulation_dataset_overview.pdf"
ENHANCED_PNG = "enhanced_simulation_noniid_overview.png"
ENHANCED_PDF = "enhanced_simulation_noniid_overview.pdf"

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
GRID_COLOR = "#DDDDDD"
TEXT_COLOR = "#333333"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.titlesize": 16,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#BBBBBB",
            "axes.linewidth": 0.8,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def move_legacy_overview_outputs(output_dir: Path) -> None:
    DEPRECATED_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in [LEGACY_PNG, LEGACY_PDF]:
        source = output_dir / file_name
        if not source.exists():
            continue
        target = DEPRECATED_DIR / file_name
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))


def _add_bar_labels(ax, bars, values: list[float]) -> None:
    upper = ax.get_ylim()[1]
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + upper * 0.02,
            f"{value:g}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=9,
        )


def _smooth_curve(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _configure_line_panel(ax, legend_items: int) -> None:
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    if legend_items <= 0:
        return
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9,
        ncol=1,
        handlelength=2.4,
    )
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_edgecolor("none")


def _prepare_base_data() -> dict[str, object]:
    all_x, all_y, meta = cfb_core.generate_base_traffic_data(seed=cfb_core.BASE_SEED)
    mean_series = [_smooth_curve(client_x.mean(axis=(0, 1)), window=3) for client_x in all_x]
    summary_df = cfb_core.build_base_dataset_client_summary(all_x, all_y, meta["client_configs"])
    _, audit_stats = cfb_core.build_base_weak_heterogeneity_report(summary_df)
    audit_stats["flow_mean_cv"] = cfb_core.coefficient_of_variation(summary_df["flow_mean"].to_numpy())
    audit_stats["noise_std_range"] = float(summary_df["noise_std"].max() - summary_df["noise_std"].min())
    return {
        "meta": meta,
        "mean_series": mean_series,
        "targets": all_y,
        "summary_df": summary_df,
        "audit_stats": audit_stats,
    }


def _prepare_enhanced_data() -> dict[str, object]:
    common_hours = np.linspace(0.0, 24.0, 160)
    target_distributions = []
    resampled_series = []
    sample_sizes = []
    noises = []
    peak_amplitudes = []
    incident_probs = []

    for cid, cfg in enumerate(cfe_core.CLIENT_CONFIGS_BASE):
        n_timesteps = cfg["n_samples"] + cfe_core.SEQ_LEN + cfe_core.PRED_LEN + 10
        data, incident_mask = cfe_core.generate_traffic_flow(cfg, n_timesteps, cfe_core.NUM_NODES, 42 + cid * 100)
        _, targets, _ = cfe_core.build_sequences(data, cfe_core.SEQ_LEN, cfe_core.PRED_LEN, incident_mask)

        mean_series = data.mean(axis=1)
        hours = np.arange(len(mean_series), dtype=float) * 24.0 / len(mean_series)
        resampled = np.interp(common_hours, hours, mean_series)
        kernel = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
        kernel /= kernel.sum()
        resampled = np.convolve(resampled, kernel, mode="same")
        resampled = _smooth_curve(resampled, window=7)

        target_distributions.append(np.clip(targets, 0.0, None))
        resampled_series.append(resampled)
        sample_sizes.append(int(cfg["n_samples"]))
        noises.append(float(cfg["noise"]))
        peak_amplitudes.append(float(cfg["morning_amp"] + cfg["evening_amp"]))
        incident_probs.append(float(cfg["incident_prob"]) * 100.0)

    return {
        "hours": common_hours,
        "target_distributions": target_distributions,
        "resampled_series": resampled_series,
        "sample_sizes": sample_sizes,
        "noises": noises,
        "peak_amplitudes": peak_amplitudes,
        "incident_probs": incident_probs,
    }


def _draw_driver_heatmap(
    fig,
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    strength_label: str = "Relative strength",
) -> None:
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(strength_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def _draw_base_driver_heatmap(fig, ax, base_data: dict[str, object]) -> None:
    summary_df = base_data["summary_df"]
    meta = base_data["meta"]
    sample_sizes = np.asarray(meta["samples_per_client"], dtype=float)
    flow_levels = summary_df["target_mean"].to_numpy(dtype=float)
    peak_amplitudes = np.asarray([cfg["peak_amp"] for cfg in meta["client_configs"]], dtype=float)
    noise_levels = summary_df["noise_std"].to_numpy(dtype=float)

    raw_matrix = np.vstack([sample_sizes, flow_levels, peak_amplitudes, noise_levels])
    row_min = raw_matrix.min(axis=1, keepdims=True)
    row_max = raw_matrix.max(axis=1, keepdims=True)
    normalized = (raw_matrix - row_min) / np.maximum(row_max - row_min, 1e-8)
    softened = 0.22 + 0.38 * normalized

    _draw_driver_heatmap(
        fig,
        ax,
        softened,
        ["sample size", "flow level", "peak amplitude", "noise level"],
        [f"Client {idx}" for idx in range(1, len(sample_sizes) + 1)],
        "(d) Base Weak-Heterogeneity Drivers",
    )


def generate_base_dataset_overview(output_dir: Path) -> None:
    base_data = _prepare_base_data()
    meta = base_data["meta"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Base Synthetic Traffic Dataset", y=0.97, fontweight="bold")

    clients = [f"Client {idx}" for idx in range(1, meta["num_clients"] + 1)]
    sample_sizes = list(meta["samples_per_client"])
    bars = axes[0, 0].bar(clients, sample_sizes, color=COLORS, width=0.65, edgecolor="white")
    axes[0, 0].set_title("(a) Mild Client Sample-size Imbalance", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].set_ylim(0, max(sample_sizes) * 1.25)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].set_axisbelow(True)
    _add_bar_labels(axes[0, 0], bars, sample_sizes)

    time_axis = np.arange(len(base_data["mean_series"][0]))
    base_labels = [f"Client {idx}" for idx in range(1, len(base_data["mean_series"]) + 1)]
    for cid, series in enumerate(base_data["mean_series"]):
        axes[0, 1].plot(
            time_axis,
            series,
            lw=1.9,
            alpha=0.95,
            color=COLORS[cid],
            label=base_labels[cid],
        )
    overall_mean = np.mean(np.vstack(base_data["mean_series"]), axis=0)
    axes[0, 1].plot(
        time_axis,
        overall_mean,
        color="#222222",
        lw=1.8,
        ls="--",
        alpha=0.75,
        label="Overall mean",
    )
    axes[0, 1].set_title("(b) Mean Traffic Trajectories", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Average Flow")
    y_min = min(float(np.min(series)) for series in base_data["mean_series"])
    y_max = max(float(np.max(series)) for series in base_data["mean_series"])
    pad = max((y_max - y_min) * 0.16, 0.06)
    axes[0, 1].set_ylim(max(0.0, y_min - pad), y_max + pad)
    axes[0, 1].set_xlim(time_axis[0], time_axis[-1])
    _configure_line_panel(axes[0, 1], legend_items=len(base_labels) + 1)

    target_data = [np.asarray(targets, dtype=float) for targets in base_data["targets"]]
    box = axes[1, 0].boxplot(target_data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.40)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.4)
    axes[1, 0].set_title("(c) Target-flow Distributions", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Client")
    axes[1, 0].set_ylabel("Target Flow")
    axes[1, 0].set_xticks(range(1, len(target_data) + 1))
    axes[1, 0].set_xticklabels([f"Client {idx}" for idx in range(1, len(target_data) + 1)])
    axes[1, 0].grid(alpha=0.25)

    _draw_base_driver_heatmap(fig, axes[1, 1], base_data)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / BASE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / BASE_PDF, bbox_inches="tight")
    plt.close(fig)


def generate_enhanced_noniid_overview(output_dir: Path) -> None:
    enhanced_data = _prepare_enhanced_data()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Enhanced Synthetic Dataset Non-IID Overview", y=0.97, fontweight="bold")

    clients = [f"Client {idx}" for idx in range(1, 6)]
    bars = axes[0, 0].bar(clients, enhanced_data["sample_sizes"], color=COLORS, width=0.65, edgecolor="white")
    axes[0, 0].set_title("(a) Imbalanced Client Sample Sizes", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].set_ylim(0, max(enhanced_data["sample_sizes"]) * 1.25)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].set_axisbelow(True)
    _add_bar_labels(axes[0, 0], bars, enhanced_data["sample_sizes"])

    enhanced_labels = [f"Client {idx}" for idx in range(1, len(enhanced_data["resampled_series"]) + 1)]
    for cid, series in enumerate(enhanced_data["resampled_series"]):
        axes[0, 1].plot(
            enhanced_data["hours"],
            series,
            lw=1.9,
            alpha=0.95,
            color=COLORS[cid],
            label=enhanced_labels[cid],
        )
    axes[0, 1].set_title("(b) Mean Daily Traffic Patterns", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Hour of Day")
    axes[0, 1].set_ylabel("Average Flow")
    axes[0, 1].set_xlim(0, 24)
    axes[0, 1].set_xticks([0, 6, 12, 18, 24])
    enhanced_y_min = min(float(np.min(series)) for series in enhanced_data["resampled_series"])
    enhanced_y_max = max(float(np.max(series)) for series in enhanced_data["resampled_series"])
    enhanced_pad = max((enhanced_y_max - enhanced_y_min) * 0.14, 0.25)
    axes[0, 1].set_ylim(max(0.0, enhanced_y_min - enhanced_pad), enhanced_y_max + enhanced_pad)
    axes[0, 1].set_xlim(0, 24)
    _configure_line_panel(axes[0, 1], legend_items=len(enhanced_labels))

    box = axes[1, 0].boxplot(
        [np.asarray(targets, dtype=float) for targets in enhanced_data["target_distributions"]],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.40)
        patch.set_edgecolor(color)
    for median in box["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.4)
    axes[1, 0].set_title("(c) Target-flow Distribution Shift", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Client")
    axes[1, 0].set_ylabel("Target Flow")
    axes[1, 0].set_xticks(range(1, 6))
    axes[1, 0].set_xticklabels(clients)
    axes[1, 0].grid(alpha=0.25)
    all_targets = np.concatenate([np.asarray(targets, dtype=float) for targets in enhanced_data["target_distributions"]])
    y_low = np.percentile(all_targets, 1.0)
    y_high = np.percentile(all_targets, 99.0)
    axes[1, 0].set_ylim(max(0.0, y_low - 0.05 * (y_high - y_low)), y_high + 0.08 * (y_high - y_low))

    driver_matrix = np.vstack(
        [
            np.asarray(enhanced_data["sample_sizes"], dtype=float),
            np.asarray(enhanced_data["noises"], dtype=float),
            np.asarray(enhanced_data["peak_amplitudes"], dtype=float),
            np.asarray(enhanced_data["incident_probs"], dtype=float),
        ]
    )
    row_min = driver_matrix.min(axis=1, keepdims=True)
    row_max = driver_matrix.max(axis=1, keepdims=True)
    normalized = (driver_matrix - row_min) / np.maximum(row_max - row_min, 1e-8)
    _draw_driver_heatmap(
        fig,
        axes[1, 1],
        normalized,
        ["sample size", "noise level", "peak amplitude", "event probability"],
        clients,
        "(d) Joint Non-IID Drivers",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / ENHANCED_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / ENHANCED_PDF, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path) -> None:
    readme = f"""# 仿真数据集构造总览图说明

旧图处理：
- 已弃用旧组合图 `simulation_dataset_construction_overview.png`
- 已弃用旧组合图 `simulation_dataset_construction_overview.pdf`
- 旧图已移动至 `deprecated/` 目录，仅保留历史记录，不再用于正文

当前推荐图件：
- {BASE_PNG}
- {BASE_PDF}
- {ENHANCED_PNG}
- {ENHANCED_PDF}

图件使用位置：
- `{BASE_PNG}` 用于“仿真数据构造、客户端划分与 Non-IID 设置”小节中的基础仿真数据集说明图
- `{ENHANCED_PNG}` 用于同一小节中的增强仿真数据集 Non-IID 说明图

数据来源说明：
- 基础仿真图基于 `simulation_experiments/cnn_fed_base/cfb_core.py` 中的 `generate_base_traffic_data()` 生成，并结合基础客户端配置构造弱异质性驱动因素热力图
- 增强仿真图基于 `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py` 中的 `CLIENT_CONFIGS_BASE`、`generate_traffic_flow()` 与 `build_sequences()` 生成
- 本脚本仅用于生成数据说明图，不触发模型训练，不修改已有实验 CSV

说明：
- 基础图强调轻度样本量不平衡、受控弱异质性与弱异质性驱动因素热力图
- 增强图强调样本量不平衡、目标值分布差异、峰型差异以及噪声/事件扰动共同构成的 Non-IID 来源
- 图中不涉及 Proposed、Loss-weighted 或 Data-loss weighted
"""
    (output_dir / README_NAME).write_text(readme, encoding="utf-8")


def main() -> None:
    _configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    move_legacy_overview_outputs(OUTPUT_DIR)
    generate_base_dataset_overview(OUTPUT_DIR)
    generate_enhanced_noniid_overview(OUTPUT_DIR)
    write_readme(OUTPUT_DIR)


if __name__ == "__main__":
    main()
