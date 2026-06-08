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
from simulation_experiments.gcn_fed_base import gfb_core


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
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def _prepare_base_data() -> dict[str, object]:
    all_x, all_y, meta = cfb_core.generate_base_traffic_data(seed=cfb_core.BASE_SEED)
    _, raw_adj, graph_meta = gfb_core.generate_adjacency_matrix(seed=gfb_core.BASE_SEED)
    mean_series = [client_x.mean(axis=(0, 1)) for client_x in all_x]
    sample_history = all_x[0][0].mean(axis=0)
    sample_target = float(all_y[0][0])
    return {
        "meta": meta,
        "mean_series": mean_series,
        "sample_history": sample_history,
        "sample_target": sample_target,
        "raw_adj": raw_adj,
        "graph_meta": graph_meta,
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

        target_distributions.append(targets)
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


def generate_base_dataset_overview(output_dir: Path) -> None:
    base_data = _prepare_base_data()
    meta = base_data["meta"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle("Base Synthetic Dataset Overview", y=0.97, fontweight="bold")

    clients = [f"Client {idx}" for idx in range(1, meta["num_clients"] + 1)]
    sample_sizes = list(meta["samples_per_client"])
    bars = axes[0, 0].bar(clients, sample_sizes, color=COLORS[0], width=0.65, edgecolor="white")
    axes[0, 0].set_title("(a) Balanced Client Sample Sizes", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].set_ylim(0, max(sample_sizes) * 1.25)
    axes[0, 0].grid(axis="y", alpha=0.7)
    axes[0, 0].set_axisbelow(True)
    _add_bar_labels(axes[0, 0], bars, sample_sizes)

    for cid, series in enumerate(base_data["mean_series"]):
        axes[0, 1].plot(np.arange(len(series)), series, label=f"Client {cid + 1}", lw=1.8, color=COLORS[cid])
    axes[0, 1].set_title("(b) Mean Traffic Trajectories", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Average Flow")
    axes[0, 1].grid(alpha=0.7)
    axes[0, 1].legend(ncol=2, fontsize=8, loc="upper left")

    history = np.asarray(base_data["sample_history"], dtype=float)
    history_steps = np.arange(len(history))
    forecast_step = len(history)
    axes[1, 0].plot(history_steps, history, color=COLORS[0], lw=2.0, label="Input window")
    axes[1, 0].scatter([forecast_step], [base_data["sample_target"]], color=COLORS[1], s=60, zorder=3, label="Prediction target")
    axes[1, 0].plot([history_steps[-1], forecast_step], [history[-1], base_data["sample_target"]], ls="--", lw=1.2, color=COLORS[1])
    axes[1, 0].axvspan(-0.5, len(history) - 0.5, color=COLORS[0], alpha=0.08)
    axes[1, 0].axvspan(len(history) - 0.5, forecast_step + 0.5, color=COLORS[1], alpha=0.08)
    axes[1, 0].set_title("(c) Input Window and Forecast Horizon", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Mean Node Flow")
    axes[1, 0].grid(alpha=0.7)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].text(
        0.02,
        0.05,
        "24-step input -> 1-step forecast\n8 traffic nodes, 70/10/20 split",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=TEXT_COLOR,
    )

    im = axes[1, 1].imshow(base_data["raw_adj"], cmap="Blues", aspect="auto")
    axes[1, 1].set_title("(d) Base 8-node Adjacency Structure", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Target Node")
    axes[1, 1].set_ylabel("Source Node")
    axes[1, 1].set_xticks(range(meta["num_nodes"]))
    axes[1, 1].set_xticklabels(range(1, meta["num_nodes"] + 1))
    axes[1, 1].set_yticks(range(meta["num_nodes"]))
    axes[1, 1].set_yticklabels(range(1, meta["num_nodes"] + 1))
    axes[1, 1].text(
        0.02,
        -0.18,
        f"Edges={base_data['graph_meta']['num_edges']}, density={base_data['graph_meta']['density']:.3f}",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
    )
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / BASE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / BASE_PDF, bbox_inches="tight")
    plt.close(fig)


def _plot_distribution_lines(ax, target_distributions: list[np.ndarray]) -> None:
    all_targets = np.concatenate(target_distributions)
    bins = np.linspace(float(np.min(all_targets)), float(np.max(all_targets)), 45)
    for cid, targets in enumerate(target_distributions):
        hist, edges = np.histogram(targets, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, lw=1.8, color=COLORS[cid], label=f"Client {cid + 1}")
        ax.fill_between(centers, hist, alpha=0.10, color=COLORS[cid])


def generate_enhanced_noniid_overview(output_dir: Path) -> None:
    enhanced_data = _prepare_enhanced_data()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("white")
    fig.suptitle("Enhanced Synthetic Dataset Non-IID Overview", y=0.97, fontweight="bold")

    clients = [f"Client {idx}" for idx in range(1, 6)]
    bars = axes[0, 0].bar(clients, enhanced_data["sample_sizes"], color=COLORS, width=0.65, edgecolor="white")
    axes[0, 0].set_title("(a) Imbalanced Client Sample Sizes", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].set_ylim(0, max(enhanced_data["sample_sizes"]) * 1.25)
    axes[0, 0].grid(axis="y", alpha=0.7)
    axes[0, 0].set_axisbelow(True)
    _add_bar_labels(axes[0, 0], bars, enhanced_data["sample_sizes"])

    _plot_distribution_lines(axes[0, 1], enhanced_data["target_distributions"])
    axes[0, 1].set_title("(b) Target-flow Distribution Shift", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Target Flow")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].grid(alpha=0.7)
    axes[0, 1].legend(ncol=2, fontsize=8, loc="upper right")

    for cid, series in enumerate(enhanced_data["resampled_series"]):
        axes[1, 0].plot(enhanced_data["hours"], series, lw=1.8, color=COLORS[cid], label=f"Client {cid + 1}")
    axes[1, 0].set_title("(c) Mean Daily Traffic Patterns", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("Hour of Day")
    axes[1, 0].set_ylabel("Average Flow")
    axes[1, 0].set_xlim(0, 24)
    axes[1, 0].set_xticks([0, 6, 12, 18, 24])
    axes[1, 0].grid(alpha=0.7)
    axes[1, 0].legend(ncol=2, fontsize=8, loc="upper left")

    x = np.arange(len(clients))
    width = 0.34
    peak_scaled = np.asarray(enhanced_data["peak_amplitudes"], dtype=float) / 10.0
    noise_bars = axes[1, 1].bar(x - width / 2, enhanced_data["noises"], width, color="#72B7B2", label="Noise std")
    peak_bars = axes[1, 1].bar(x + width / 2, peak_scaled, width, color="#FF9DA6", label="Peak amplitude / 10")
    axes[1, 1].set_title("(d) Joint Non-IID Drivers", loc="left", fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(clients)
    axes[1, 1].set_ylabel("Noise / scaled peak amplitude")
    axes[1, 1].grid(axis="y", alpha=0.7)
    axes[1, 1].set_axisbelow(True)
    ax2 = axes[1, 1].twinx()
    incident_line = ax2.plot(x, enhanced_data["incident_probs"], color="#333333", marker="o", lw=1.6, label="Incident probability (%)")
    ax2.set_ylabel("Incident probability (%)")
    for bar_set, values in [(noise_bars, enhanced_data["noises"]), (peak_bars, peak_scaled)]:
        for bar, value in zip(bar_set, values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + axes[1, 1].get_ylim()[1] * 0.02,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )
    handles1, labels1 = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(handles1 + incident_line, labels1 + labels2, loc="upper left", fontsize=8)

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
- 基础仿真图基于 `simulation_experiments/cnn_fed_base/cfb_core.py` 中的 `generate_base_traffic_data()` 生成，并结合 `simulation_experiments/gcn_fed_base/gfb_core.py` 中的基础邻接矩阵
- 增强仿真图基于 `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py` 中的 `CLIENT_CONFIGS_BASE`、`generate_traffic_flow()` 与 `build_sequences()` 生成
- 本脚本仅用于生成数据说明图，不触发模型训练，不修改已有实验 CSV

说明：
- 基础图强调受控、均衡的客户端设置与基础路网结构
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
