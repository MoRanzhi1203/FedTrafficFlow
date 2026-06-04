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
    {"figure_file": "enhanced_dataset_client_timeseries.png", "workflow": "data_viz", "figure_type": "line", "description": "Per-client average traffic flow time series for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_distribution_comparison.png", "workflow": "data_viz", "figure_type": "box", "description": "Client-level traffic flow distribution comparison for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_client_config.png", "workflow": "data_viz", "figure_type": "bar", "description": "Client configuration overview for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "enhanced_dataset_peak_pattern.png", "workflow": "data_viz", "figure_type": "line", "description": "Twenty-four-hour peak traffic patterns across enhanced clients.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_incident_example.png", "workflow": "data_viz", "figure_type": "line", "description": "Incident example with shaded disruption periods for the incident-prone client.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "enhanced_dataset_client_correlation_matrix.png", "workflow": "data_viz", "figure_type": "heatmap", "description": "Inter-client traffic correlation matrix for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_node_correlation_matrix.png", "workflow": "data_viz", "figure_type": "heatmap", "description": "Node correlation matrix for a representative enhanced client.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_fixed_adjacency_matrix.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Fixed adjacency matrix used by the enhanced GCN experiment.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_dynamic_adjacency_peak.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Dynamic adjacency matrix during peak traffic.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_dynamic_adjacency_offpeak.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Dynamic adjacency matrix during off-peak traffic.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_fixed_dynamic_adjacency_comparison.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Comparison of fixed and dynamic graph structures across traffic periods.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_functional_similarity_matrix.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Functional similarity matrix estimated from node traffic profiles.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_congestion_delay_matrix.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Congestion propagation delay matrix for the enhanced GCN setting.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_gcn_peak_graph_change.png", "workflow": "fixed_vs_dynamic", "figure_type": "heatmap", "description": "Graph structure changes across off-peak and peak traffic periods.", "source_csv": "enhanced_gcn_graph_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "gcn_enhanced_fixed_vs_dynamic_comparison.png", "workflow": "fixed_vs_dynamic", "figure_type": "bar", "description": "Performance comparison under fixed, dynamic, and functional graph definitions measured by RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_fixed_vs_dynamic_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_congestion_delay_comparison.png", "workflow": "congestion_delay", "figure_type": "bar", "description": "Performance comparison across congestion-delay-related graph definitions measured by RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_congestion_delay_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_main_rmse_comparison.png", "workflow": "main", "figure_type": "bar", "description": "Main comparison of Independent, GCN-FedAvg, and GCN-Proposed using RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_main_metrics_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_aggregation_ablation.png", "workflow": "aggregation", "figure_type": "bar", "description": "Aggregation strategy ablation on RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_aggregation_ablation_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_lambda_sensitivity.png", "workflow": "lambda", "figure_type": "line", "description": "Lambda sensitivity analysis for the data-loss weighted GCN aggregation strategy.", "source_csv": "gcn_enhanced_lambda_sensitivity_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_client_scale.png", "workflow": "client_scale", "figure_type": "line", "description": "Client-scale sensitivity analysis for the enhanced GCN setting measured by RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_client_scale_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_noniid_strength.png", "workflow": "noniid", "figure_type": "bar", "description": "Method comparison under different Non-IID strengths for enhanced GCN measured by RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_noniid_strength_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_global_validation_rmse.png", "workflow": "convergence", "figure_type": "line", "description": "Global validation RMSE across communication rounds for enhanced GCN.", "source_csv": "gcn_enhanced_convergence_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_client_training_loss.png", "workflow": "convergence", "figure_type": "line", "description": "Per-client training loss across communication rounds for GCN-FedAvg and GCN-Proposed.", "source_csv": "gcn_enhanced_convergence_round_metrics.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_client_rmse_comparison.png", "workflow": "client_metrics", "figure_type": "bar", "description": "Per-client RMSE comparison across Independent, GCN-FedAvg, and GCN-Proposed.", "source_csv": "gcn_enhanced_client_metrics.csv", "used_in_paper": "recommended"},
    {"figure_file": "gcn_enhanced_peak_offpeak_comparison.png", "workflow": "peak", "figure_type": "bar", "description": "Performance comparison across peak, off-peak, and incident periods for enhanced GCN measured by RMSE, MAE, and MAPE.", "source_csv": "gcn_enhanced_peak_offpeak_summary.csv", "used_in_paper": "recommended"},
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
    plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]



def save_figure(fig: plt.Figure, output_dir: Path, file_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path



def save_dataframe(df: pd.DataFrame, output_dir: Path, file_name: str) -> Path:
    path = ensure_output_dir(output_dir) / file_name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")
    return path



def export_figure_index(output_dir: Path) -> Path:
    """Export figure metadata for paper curation."""
    return save_dataframe(pd.DataFrame(FIGURE_INDEX_ENTRIES), output_dir, "figure_index.csv")


