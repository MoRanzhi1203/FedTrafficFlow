# FedTrafficFlow

## Overview

This repository focuses on real traffic data preprocessing, traffic speed analysis,
Greenshields-based density and flow estimation, node-level flow construction,
intra-day curve fitting, and curve-cluster visualization.

## Directory Layout

```text
FedTrafficFlow/
|- analysis_scripts/
|- dataset_inspection_scripts/
|- docs/
|- data/
|  |- analysis/
|  |- params/
|  |- processed/
|  `- raw/
|- preprocessing_scripts/
|- README.md
`- requirements.txt
```

## Main Workflow

```bash
python preprocessing_scripts/process_link_gps.py
python preprocessing_scripts/process_rnsd.py
python preprocessing_scripts/merge_speed_data.py
python analysis_scripts/summarize_speed_stats.py
python analysis_scripts/visualize_speed_hist_by_period.py
python analysis_scripts/add_p995_to_speed_histogram.py
python analysis_scripts/compute_greenshields_density.py
python analysis_scripts/compute_node_intersection_flow_optimized.py
python analysis_scripts/check_spatial_node_completeness.py
python analysis_scripts/fit_node_flow_daily_curve.py
python analysis_scripts/compare_node_flow_fourier_orders.py
python analysis_scripts/compare_date_type_curve_methods.py
python analysis_scripts/visualize_fitted_function_clusters.py --method M2_shape_normalized_weighted_curve
python analysis_scripts/visualize_node_flow_daily_curve_fit.py
```

## Core Outputs

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`
- `data/processed/speed_data_chunks/`
- `data/analysis/density_metrics_chunks/`
- `data/analysis/node_intersection_flow_parquet/`
- `data/analysis/node_flow_curve_fit/`
- `data/analysis/date_type_curve_method_comparison/`

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Current requirements cover:

- `numpy`
- `pandas`
- `polars`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pyarrow`
- `psutil`
- `pytest`

For detailed documentation, see files under `docs/`.
