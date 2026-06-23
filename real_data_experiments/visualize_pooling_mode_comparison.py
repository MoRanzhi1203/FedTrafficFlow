"""Visualize pooling-mode comparison results from existing smoke-test outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


POOL_MODES = ["avg", "max", "sum_mean"]
SMOKE_DIR_MAP = {
    "avg": "smoke_avg",
    "max": "smoke_max",
    "sum_mean": "smoke_sum_mean",
}
MODE_COLORS = {
    "avg": "#4C72B0",
    "max": "#DD8452",
    "sum_mean": "#55A868",
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize pooling-mode comparison results.")
    parser.add_argument("--input-root", type=Path, default=Path("data/processed/node_flow_grid"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pooling_mode_comparison"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--peak-time-index", type=int, default=None)
    return parser


def ensure_exists(path: Path, label: str) -> None:
    """Raise readable errors for missing paths."""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def load_mode_bundle(input_root: Path, mode: str) -> dict[str, object]:
    """Load one smoke directory into memory."""
    smoke_dir = input_root / SMOKE_DIR_MAP[mode]
    ensure_exists(smoke_dir, f"Smoke directory for {mode}")
    metadata = json.loads((smoke_dir / "node_flow_grid_metadata.json").read_text(encoding="utf-8"))
    tensor_metadata = json.loads((smoke_dir / "node_flow_grid_tensor_metadata.json").read_text(encoding="utf-8"))
    regions_df = pd.read_csv(smoke_dir / "node_flow_grid_regions.csv")
    flow_audit_df = pd.read_csv(smoke_dir / "node_flow_grid_flow_audit.csv")
    pooled_np = np.load(smoke_dir / "node_flow_grid_pooled.npy")
    tensor = torch.load(smoke_dir / "node_flow_grid_tensor.pt", map_location="cpu")
    return {
        "mode": mode,
        "smoke_dir": smoke_dir,
        "metadata": metadata,
        "tensor_metadata": tensor_metadata,
        "regions_df": regions_df,
        "flow_audit_df": flow_audit_df,
        "pooled_np": pooled_np,
        "tensor": tensor,
    }


def choose_peak_time_index(bundles: dict[str, dict[str, object]], explicit_index: int | None) -> int:
    """Choose a representative peak time index."""
    if explicit_index is not None:
        return explicit_index
    reference_df = bundles["sum_mean"]["flow_audit_df"]
    peak_row = reference_df["raw_input_total_flow"].idxmax()
    return int(reference_df.loc[peak_row, "time_index"])


def save_flow_difference_bar(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Save mean/max relative-difference bar chart."""
    modes = list(bundles.keys())
    mean_values = [bundles[mode]["flow_audit_df"]["relative_difference_grid_vs_pooled"].mean() for mode in modes]
    max_values = [bundles[mode]["flow_audit_df"]["relative_difference_grid_vs_pooled"].max() for mode in modes]
    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, mean_values, width=width, color=[MODE_COLORS[mode] for mode in modes], label="Mean")
    ax.bar(x + width / 2, max_values, width=width, color=[MODE_COLORS[mode] for mode in modes], alpha=0.45, label="Max")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Relative Difference")
    ax.set_title("Grid-to-Pooled Relative Difference by Pool Mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_flow_difference_bar.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_active_region_bar(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Save active/empty region count bar chart."""
    modes = list(bundles.keys())
    active_values = [int(bundles[mode]["metadata"]["active_region_count"]) for mode in modes]
    empty_values = [int(bundles[mode]["metadata"]["empty_region_count"]) for mode in modes]
    x = np.arange(len(modes))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, active_values, color="#55A868", label="Active")
    ax.bar(x, empty_values, bottom=active_values, color="#C44E52", label="Empty")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Region Count")
    ax.set_title("Active vs Empty Regions by Pool Mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_active_region_bar.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_shape_summary_table(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Render shape summary as a table figure."""
    rows = []
    for mode, bundle in bundles.items():
        rows.append(
            [
                mode,
                str(tuple(bundle["metadata"]["raw_shape"])),
                str(tuple(bundle["metadata"]["pooled_shape"])),
                str(tuple(bundle["tensor_metadata"]["tensor_shape"])),
            ]
        )
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["pool_mode", "raw_shape", "pooled_shape", "tensor_shape"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Pooling Mode Shape Summary", pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_shape_summary_table.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_total_flow_timeseries(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Save raw total-flow timeseries comparison."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    reference_df = bundles["sum_mean"]["flow_audit_df"]
    ax.plot(reference_df["time_index"], reference_df["raw_input_total_flow"], color="black", linewidth=2, label="Raw Input Total Flow")
    for mode, bundle in bundles.items():
        ax.plot(
            bundle["flow_audit_df"]["time_index"],
            bundle["flow_audit_df"]["grid_total_flow_channel0_sum"],
            color=MODE_COLORS[mode],
            linestyle="--",
            label=f"{mode} Grid Channel0 Sum",
        )
    ax.set_title("Raw vs Grid Total Flow Timeseries")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Flow")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_total_flow_timeseries.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pooled_total_flow_timeseries(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Save pooled channel-0 total-flow timeseries comparison."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    reference_df = bundles["sum_mean"]["flow_audit_df"]
    ax.plot(reference_df["time_index"], reference_df["grid_total_flow_channel0_sum"], color="black", linewidth=2, label="Grid Channel0 Sum")
    for mode, bundle in bundles.items():
        ax.plot(
            bundle["flow_audit_df"]["time_index"],
            bundle["flow_audit_df"]["pooled_total_flow_channel0_sum"],
            color=MODE_COLORS[mode],
            label=f"{mode} Pooled Channel0 Sum",
        )
    ax.set_title("Grid vs Pooled Channel0 Total Flow Timeseries")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Flow")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_pooled_total_flow_timeseries.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_relative_difference_timeseries(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Save relative-difference timeseries."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for mode, bundle in bundles.items():
        axes[0].plot(
            bundle["flow_audit_df"]["time_index"],
            bundle["flow_audit_df"]["relative_difference_raw_vs_grid"],
            color=MODE_COLORS[mode],
            label=mode,
        )
        axes[1].plot(
            bundle["flow_audit_df"]["time_index"],
            bundle["flow_audit_df"]["relative_difference_grid_vs_pooled"],
            color=MODE_COLORS[mode],
            label=mode,
        )
    axes[0].set_title("Raw vs Grid Relative Difference")
    axes[0].set_ylabel("Relative Difference")
    axes[1].set_title("Grid vs Pooled Relative Difference")
    axes[1].set_ylabel("Relative Difference")
    axes[1].set_xlabel("Time Index")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_relative_difference_timeseries.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_heatmaps(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int, time_index: int, peak_time_index: int) -> None:
    """Save pooled channel-0 heatmaps for representative times."""
    for mode, bundle in bundles.items():
        pooled_np = bundle["pooled_np"]
        for index_value, suffix in [(time_index, "time0"), (peak_time_index, "time_peak")]:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            heatmap = pooled_np[index_value, 0, :, :]
            image = ax.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
            ax.set_title(f"{mode} pooled channel0 heatmap (time_index={index_value})")
            ax.set_xlabel("col=lon")
            ax.set_ylabel("row=lat")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / f"pool_mode_heatmap_{mode}_{suffix}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def save_region_activity_map(bundles: dict[str, dict[str, object]], output_dir: Path, dpi: int) -> None:
    """Save pooled-region activity map."""
    modes = list(bundles.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, mode in zip(axes, modes):
        regions_df = bundles[mode]["regions_df"]
        active_df = regions_df[regions_df["is_active_region"] == True]
        empty_df = regions_df[regions_df["is_active_region"] == False]
        ax.scatter(empty_df["pooled_col"], empty_df["pooled_row"], s=12, c="#C44E52", alpha=0.5, label="Empty")
        ax.scatter(active_df["pooled_col"], active_df["pooled_row"], s=18, c="#55A868", alpha=0.8, label="Active")
        meta = bundles[mode]["metadata"]
        ax.set_title(f"{mode}\nactive={meta['active_region_count']}, empty={meta['empty_region_count']}")
        ax.set_xlabel("pooled_col")
        ax.set_ylabel("pooled_row")
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Pooled Region Activity Map (630 Regions)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_dir / "pool_mode_region_activity_map.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_selection_summary(output_dir: Path, dpi: int) -> None:
    """Save one text-based summary figure."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    lines = [
        "Pooling Mode Selection Summary",
        "",
        "avg: smooths both channels, but channel 0 total flow is not preserved.",
        "max: best matches the historical notebook (max_pool2d), but physical meaning is weaker.",
        "sum_mean: channel 0 uses sum pooling and channel 1 uses average pooling.",
        "sum_mean: best preserves traffic-flow scale and is easiest to explain in the paper.",
        "",
        "Formal choice: sum_mean",
        "Historical reproduction baseline: max",
        "avg is not selected as the formal default.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(output_dir / "pool_mode_selection_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {mode: load_mode_bundle(args.input_root, mode) for mode in POOL_MODES}
    peak_time_index = choose_peak_time_index(bundles, args.peak_time_index)

    save_flow_difference_bar(bundles, args.output_dir, args.dpi)
    save_active_region_bar(bundles, args.output_dir, args.dpi)
    save_shape_summary_table(bundles, args.output_dir, args.dpi)
    save_total_flow_timeseries(bundles, args.output_dir, args.dpi)
    save_pooled_total_flow_timeseries(bundles, args.output_dir, args.dpi)
    save_relative_difference_timeseries(bundles, args.output_dir, args.dpi)
    save_heatmaps(bundles, args.output_dir, args.dpi, args.time_index, peak_time_index)
    save_region_activity_map(bundles, args.output_dir, args.dpi)
    save_selection_summary(args.output_dir, args.dpi)

    print(f"[pooling_mode_visualization] completed -> {args.output_dir}")
    print(f"[peak_time_index] {peak_time_index}")


if __name__ == "__main__":
    main()
