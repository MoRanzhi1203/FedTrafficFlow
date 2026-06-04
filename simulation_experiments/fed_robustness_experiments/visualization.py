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
    {"figure_file": "fed_robustness_communication_cost.png", "workflow": "communication_cost", "figure_type": "line", "description": "Communication cost comparison under different client counts and communication rounds.", "source_csv": "fed_communication_cost.csv", "used_in_paper": "recommended"},
    {"figure_file": "fed_robustness_client_dropout.png", "workflow": "client_dropout", "figure_type": "line", "description": "Robustness to client dropout measured by RMSE, MAE, and MAPE.", "source_csv": "fed_client_dropout_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "fed_robustness_communication_delay.png", "workflow": "communication_delay", "figure_type": "line", "description": "Robustness to communication delay measured by RMSE, MAE, and MAPE.", "source_csv": "fed_communication_delay_summary.csv", "used_in_paper": "recommended"},
    {"figure_file": "fed_robustness_gradient_noise.png", "workflow": "gradient_noise", "figure_type": "line", "description": "Sensitivity to Gaussian parameter perturbation (lightweight simulation, NOT formal DP) measured by RMSE, MAE, and MAPE.", "source_csv": "fed_gradient_noise_summary.csv", "used_in_paper": "recommended"},
]

def configure_academic_plot_style() -> None:
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


def save_dataframe(df: pd.DataFrame, d: Path, n: str) -> Path:
    p = ensure_output_dir(d) / n; df.to_csv(p, index=False, encoding="utf-8")
    print(f"[saved] {p}"); return p


def save_figure(fig, d: Path, n: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / n
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig); print(f"Saved figure: {p}"); return p


def export_figure_index(output_dir: Path) -> Path:
    return save_dataframe(pd.DataFrame(FIGURE_INDEX_ENTRIES), output_dir, "figure_index.csv")

