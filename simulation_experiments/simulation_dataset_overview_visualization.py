from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation_experiments.cnn_fed_base import cfb_core
from simulation_experiments.cnn_fed_enhanced_experiments import cfe_core


OUTPUT_DIR = PROJECT_ROOT / "results" / "simulation_experiments" / "dataset_overview"
PNG_NAME = "simulation_dataset_construction_overview.png"
PDF_NAME = "simulation_dataset_construction_overview.pdf"
README_NAME = "simulation_dataset_construction_overview_readme.md"

BASE_COLOR = "#4C78A8"
ENHANCED_COLOR = "#F58518"
ACCENT_GREEN = "#54A24B"
ACCENT_PURPLE = "#B279A2"
TEXT_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 15,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#BBBBBB",
            "axes.linewidth": 0.8,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def get_base_dataset_config() -> dict[str, object]:
    return {
        "num_clients": cfb_core.BASE_NUM_CLIENTS,
        "num_nodes": cfb_core.BASE_NUM_NODES,
        "seq_len": cfb_core.BASE_SEQ_LEN,
        "pred_len": cfb_core.BASE_PRED_LEN,
        "samples_per_client": list(cfb_core.BASE_SAMPLES_PER_CLIENT),
        "noise_std": cfb_core.BASE_NOISE,
        "split": (
            cfb_core.BASE_TRAIN_RATIO,
            cfb_core.BASE_VAL_RATIO,
            cfb_core.BASE_TEST_RATIO,
        ),
    }


def get_enhanced_dataset_config() -> dict[str, object]:
    client_configs = list(cfe_core.CLIENT_CONFIGS_BASE)
    return {
        "num_clients": len(client_configs),
        "num_nodes": cfe_core.NUM_NODES,
        "seq_len": cfe_core.SEQ_LEN,
        "pred_len": cfe_core.PRED_LEN,
        "samples_per_client": [int(item["n_samples"]) for item in client_configs],
        "distributions": [str(item["dist"]) for item in client_configs],
        "split": (0.70, 0.10, 0.20),
    }


def _draw_panel_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=10, fontweight="bold")


def _draw_bar_panel(ax, values: list[int], color: str, title: str, ylabel: str) -> None:
    clients = [f"Client {idx}" for idx in range(1, len(values) + 1)]
    bars = ax.bar(clients, values, color=color, width=0.65, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    _draw_panel_title(ax, title)
    upper = max(values) * 1.2
    ax.set_ylim(0, upper)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + upper * 0.025,
            f"{value}",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=9,
        )


def _draw_window_panel(ax, seq_len: int, pred_len: int, num_nodes: int, title: str) -> None:
    _draw_panel_title(ax, title)
    ax.set_xlim(0, seq_len + pred_len + 4)
    ax.set_ylim(0, 4)
    ax.axis("off")

    for idx in range(seq_len):
        rect = Rectangle((idx + 0.3, 1.8), 0.8, 0.6, facecolor=BASE_COLOR, alpha=0.85, edgecolor="white")
        ax.add_patch(rect)
    for idx in range(pred_len):
        rect = Rectangle((seq_len + idx + 1.3, 1.8), 0.8, 0.6, facecolor=ENHANCED_COLOR, alpha=0.9, edgecolor="white")
        ax.add_patch(rect)

    ax.annotate(
        "",
        xy=(seq_len + 1.2, 2.1),
        xytext=(seq_len - 0.2, 2.1),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": TEXT_COLOR},
    )
    ax.text(seq_len / 2.0 + 0.3, 2.75, f"{seq_len} historical steps", ha="center", color=TEXT_COLOR, fontsize=10)
    ax.text(seq_len + 1.7, 2.75, f"predict {pred_len} step", ha="left", color=TEXT_COLOR, fontsize=10)
    ax.text(0.3, 1.2, "Input window", ha="left", color=TEXT_COLOR, fontsize=9)
    ax.text(seq_len + 1.3, 1.2, "Forecast", ha="left", color=TEXT_COLOR, fontsize=9)
    ax.text(seq_len / 2.0 + 0.4, 0.55, f"Nodes = {num_nodes}", ha="center", color=ACCENT_PURPLE, fontsize=10)


def _draw_summary_box(ax, title: str, lines: list[str], facecolor: str) -> None:
    _draw_panel_title(ax, title)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.06, 0.10),
        0.88,
        0.78,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=facecolor,
        facecolor=facecolor,
        alpha=0.14,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    for idx, line in enumerate(lines):
        ax.text(
            0.12,
            0.78 - idx * 0.14,
            line,
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=TEXT_COLOR,
            fontsize=10,
        )


def _draw_distribution_panel(ax, labels: list[str], title: str) -> None:
    _draw_panel_title(ax, title)
    ax.axis("off")
    y_positions = np.linspace(0.82, 0.18, len(labels))
    colors = [BASE_COLOR, ENHANCED_COLOR, ACCENT_GREEN, ACCENT_PURPLE, "#E45756"]
    for idx, (label, ypos) in enumerate(zip(labels, y_positions, strict=True)):
        marker_x = 0.12
        text_x = 0.20
        ax.scatter([marker_x], [ypos], s=160, color=colors[idx % len(colors)], transform=ax.transAxes, clip_on=False)
        ax.text(
            text_x,
            ypos,
            f"Client {idx + 1}: {label}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            color=TEXT_COLOR,
        )


def write_readme(output_dir: Path, base_cfg: dict[str, object], enhanced_cfg: dict[str, object]) -> None:
    readme = f"""# 仿真数据集构造总览图说明

本图用于论文“仿真数据构造、客户端划分与 Non-IID 设置”小节。

图件：
- {PNG_NAME}
- {PDF_NAME}

图件内容：
- 第一行展示基础仿真数据集：{base_cfg["num_clients"]} 个客户端、{base_cfg["num_nodes"]} 个节点、每客户端 {base_cfg["samples_per_client"][0]} 个样本、输入窗口 {base_cfg["seq_len"]}、预测步长 {base_cfg["pred_len"]}。
- 第二行展示增强仿真数据集：5 个客户端、样本量 600/500/700/550/450、分布族 normal/student-t/chi-square/gaussian_mixture/log_normal，以及样本量、分布族、噪声、高峰和事件扰动构成的联合 Non-IID 设置。

说明：
- 本图是实验设计说明图，不是模型性能结果图。
- 本图不涉及 Proposed、Loss-weighted 或 Data-loss weighted。
- 本图用于帮助读者理解基础仿真数据集和增强仿真数据集的差异。
"""
    (output_dir / README_NAME).write_text(readme, encoding="utf-8")


def create_dataset_overview_figure(output_dir: Path) -> None:
    _configure_style()
    base_cfg = get_base_dataset_config()
    enhanced_cfg = get_enhanced_dataset_config()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle("Synthetic Traffic Dataset Construction and Non-IID Settings", y=0.98, fontweight="bold")

    _draw_bar_panel(
        axes[0, 0],
        base_cfg["samples_per_client"],
        BASE_COLOR,
        "(a) Base: Client Sizes",
        "Samples",
    )
    _draw_window_panel(
        axes[0, 1],
        int(base_cfg["seq_len"]),
        int(base_cfg["pred_len"]),
        int(base_cfg["num_nodes"]),
        "(b) Base: Window and Horizon",
    )
    _draw_summary_box(
        axes[0, 2],
        "(c) Base: Controlled Setting",
        [
            "5 clients",
            "8 traffic nodes",
            "200 samples per client",
            "70% / 10% / 20% split",
            "noise std = 0.05",
        ],
        BASE_COLOR,
    )

    _draw_bar_panel(
        axes[1, 0],
        enhanced_cfg["samples_per_client"],
        ENHANCED_COLOR,
        "(d) Enhanced: Client Imbalance",
        "Samples",
    )
    _draw_distribution_panel(
        axes[1, 1],
        [
            "normal",
            "student-t",
            "chi-square",
            "gaussian mixture",
            "log-normal",
        ],
        "(e) Enhanced: Distribution Shift",
    )
    _draw_summary_box(
        axes[1, 2],
        "(f) Enhanced: Joint Non-IID",
        [
            "sample-size imbalance",
            "distribution-family shift",
            "noise-level variation",
            "peak-pattern variation",
            "event perturbations",
        ],
        ENHANCED_COLOR,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / PNG_NAME, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / PDF_NAME, bbox_inches="tight")
    plt.close(fig)

    write_readme(output_dir, base_cfg, enhanced_cfg)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_dataset_overview_figure(OUTPUT_DIR)


if __name__ == "__main__":
    main()
