# -*- coding: utf-8 -*-
"""Pure visualization module."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_cjk_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((fn for fn in _cjk_candidates if fn in _available), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
plt.ioff()

def ensure_output_dir(d):
    d = Path(d) if not isinstance(d, Path) else d
    d.mkdir(parents=True, exist_ok=True)
    return d

FIGURE_INDEX_ENTRIES = [
    {
        "figure_file": "base_dataset_client_timeseries.png",
        "workflow": "data_viz",
        "figure_type": "line",
        "description": "Per-client average traffic flow time series for the base dataset.",
        "source_csv": "base_dataset_summary.csv",
        "used_in_paper": "recommended",
    },
    {
        "figure_file": "base_dataset_node_heatmap.png",
        "workflow": "data_viz",
        "figure_type": "heatmap",
        "description": "Node-time traffic flow heatmap for a representative base client.",
        "source_csv": "base_dataset_summary.csv",
        "used_in_paper": "recommended",
    },
    {
        "figure_file": "base_dataset_client_boxplot.png",
        "workflow": "data_viz",
        "figure_type": "box",
        "description": "Traffic flow distribution comparison across clients in the base dataset.",
        "source_csv": "base_dataset_summary.csv",
        "used_in_paper": "recommended",
    },
    {
        "figure_file": "base_dataset_split_overview.png",
        "workflow": "data_viz",
        "figure_type": "bar",
        "description": "Train, validation, and test split overview for the base dataset.",
        "source_csv": "base_dataset_summary.csv",
        "used_in_paper": "yes",
    },
    {
        "figure_file": "base_dataset_client_sample_size.png",
        "workflow": "data_viz",
        "figure_type": "bar",
        "description": "Sample size comparison across clients in the base dataset.",
        "source_csv": "base_dataset_summary.csv",
        "used_in_paper": "yes",
    },
    {
        "figure_file": "base_gcn_adjacency_matrix.png",
        "workflow": "data_viz",
        "figure_type": "heatmap",
        "description": "Fixed adjacency matrix used by the base GCN experiment.",
        "source_csv": "base_gcn_graph_summary.csv",
        "used_in_paper": "recommended",
    },
    {
        "figure_file": "base_gcn_degree_distribution.png",
        "workflow": "data_viz",
        "figure_type": "bar",
        "description": "Node degree distribution for the base GCN graph.",
        "source_csv": "base_gcn_graph_summary.csv",
        "used_in_paper": "yes",
    },
    {
        "figure_file": "gcn_base_main_comparison.png",
        "workflow": "main",
        "figure_type": "bar",
        "description": "Client-level MSE, RMSE, and MAE comparison between Independent and FedAvg.",
        "source_csv": "gcn_base_metrics_summary.csv",
        "used_in_paper": "recommended",
    },
    {
        "figure_file": "gcn_base_convergence.png",
        "workflow": "convergence",
        "figure_type": "line",
        "description": "Global validation RMSE and local training loss across communication rounds.",
        "source_csv": "gcn_base_convergence.csv",
        "used_in_paper": "recommended",
    },
]

def configure_academic_plot_style() -> None:
    """Configure a unified seaborn style for paper-ready figures."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.autolayout": False,
        },
    )
    plt.rcParams["font.family"] = "DejaVu Sans"



def save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    """保存图像并关闭图对象。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path



def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    """保存 DataFrame 为 CSV。"""
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path



def export_figure_index(output_dir: Path) -> Path:
    """Export figure metadata for paper curation."""
    return save_dataframe(pd.DataFrame(FIGURE_INDEX_ENTRIES), output_dir, "figure_index.csv")


# ──────────────────────────────────────────────────────────
# 基础数据集生成（与 cnn_fed_base.py 完全一致）
# ──────────────────────────────────────────────────────────

