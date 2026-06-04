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
    {"figure_file": "enhanced_dataset_client_config.png", "workflow": "data_viz", "figure_type": "bar", "description": "Client configuration overview for sample size, noise, base flow, and incident probability.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "enhanced_dataset_peak_pattern.png", "workflow": "data_viz", "figure_type": "line", "description": "Twenty-four-hour peak traffic patterns across enhanced clients.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_incident_example.png", "workflow": "data_viz", "figure_type": "line", "description": "Incident example with shaded disruption periods for the incident-prone client.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "enhanced_dataset_client_correlation_matrix.png", "workflow": "data_viz", "figure_type": "heatmap", "description": "Inter-client traffic correlation matrix for the enhanced dataset.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "enhanced_dataset_node_correlation_matrix.png", "workflow": "data_viz", "figure_type": "heatmap", "description": "Node correlation matrix for a representative enhanced client.", "source_csv": "enhanced_dataset_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_main_rmse_comparison.png", "workflow": "main", "figure_type": "bar", "description": "Main comparison of Independent, FedAvg, and Proposed using MSE, RMSE, MAE, and MAPE.", "source_csv": "cnn_enhanced_main_metrics_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_aggregation_ablation.png", "workflow": "aggregation", "figure_type": "bar", "description": "Aggregation strategy ablation on RMSE, MAE, and MAPE.", "source_csv": "cnn_enhanced_aggregation_ablation_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_lambda_sensitivity.png", "workflow": "lambda", "figure_type": "line", "description": "Lambda sensitivity analysis for the data-loss weighted aggregation strategy.", "source_csv": "cnn_enhanced_lambda_sensitivity_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_global_validation_rmse.png", "workflow": "convergence", "figure_type": "line", "description": "Global validation RMSE across communication rounds.", "source_csv": "cnn_enhanced_convergence_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_client_training_loss.png", "workflow": "convergence", "figure_type": "line", "description": "Per-client training loss across communication rounds for FedAvg and Proposed.", "source_csv": "cnn_enhanced_convergence_round_metrics.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_convergence_overview.png", "workflow": "convergence", "figure_type": "line", "description": "Combined convergence overview including validation metrics and per-client losses.", "source_csv": "cnn_enhanced_convergence_summary.csv", "used_in_paper": "yes"},
    {"figure_file": "cnn_enhanced_client_scale.png", "workflow": "client_scale", "figure_type": "line", "description": "Client-scale sensitivity analysis for RMSE, MAE, and MAPE.", "source_csv": "cnn_enhanced_client_scale_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_noniid_strength.png", "workflow": "noniid", "figure_type": "bar", "description": "Method comparison under different Non-IID strengths measured by RMSE, MAE, and MAPE.", "source_csv": "cnn_enhanced_noniid_strength_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_client_rmse_comparison.png", "workflow": "client_metrics", "figure_type": "bar", "description": "Per-client RMSE comparison across Independent, FedAvg, and Proposed.", "source_csv": "cnn_enhanced_client_metrics.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_client_improvement.png", "workflow": "client_metrics", "figure_type": "bar", "description": "RMSE improvement of Proposed over baselines for each client.", "source_csv": "cnn_enhanced_client_metrics.csv", "used_in_paper": "yes"},
    {"figure_file": "cnn_enhanced_peak_offpeak_comparison.png", "workflow": "peak", "figure_type": "bar", "description": "Performance comparison across peak, off-peak, and incident periods measured by RMSE, MAE, and MAPE.", "source_csv": "cnn_enhanced_peak_offpeak_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "cnn_enhanced_feature_ablation.png", "workflow": "feature_ablation", "figure_type": "bar", "description": "Feature ablation comparison for FedAvg and Proposed measured by RMSE, MAE, and MAPE.", "source_csv": "cnn_enhanced_feature_ablation_summary.csv", "used_in_paper": "recommended"},
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


