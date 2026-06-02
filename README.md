# FedTrafficFlow

## Overview

This repository focuses on two main areas:

1. **Real Traffic Data Pipeline** — preprocessing, speed analysis, Greenshields-based density/flow estimation, node-level flow construction, intra-day curve fitting, and curve-cluster visualization.
2. **Federated Simulation Experiments** — CNN + BiLSTM + Attention / GCN + BiLSTM + Attention based FedAvg traffic flow prediction using synthetic heterogeneous client data.

## Directory Layout

```text
FedTrafficFlow/
 analysis_scripts/                     # 真实交通数据分析、密度计算、节点曲线拟合与聚类
 dataset_inspection_scripts/           # 数据结构检查、顺序检查、样例查看
 docs/                                 # 项目说明文档
 simulation_experiments/               # 联邦仿真实验脚本
   cnn_fed_base.py                     #   CNN + BiLSTM + Attention 基础仿真
   gcn_fed_base.py                     #   GCN + BiLSTM + Attention 基础仿真 (图结构)
 data/
   raw/                               # 原始数据
   processed/                         # 预处理后的标准化数据与速度分块
   params/                            # 参数表
   analysis/                          # 分析结果与中间产物
 preprocessing_scripts/                # 原始路网与速度数据预处理
 results/
   simulation_experiments/
     cnn/                             # CCN/C 仿真结果 (PNG/CSV/TXT)
     gcn/                             # GCN 仿真结果 (PNG/CSV/TXT)
 test/                                # Jupyter Notebook 实验文件
 README.md
 requirements.txt
 .gitignore
```

## Main Workflow

### Real Data Pipeline

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

### Federated Simulation Experiments

```powershell
conda activate analysis
cd simulation_experiments
python cnn_fed_base.py --workflow all
python gcn_fed_base.py --workflow all
```

Workflow options: `all` (default, overview + ablation), `overview`, `ablation`.

**Aggregation method**: Standard sample-weighted FedAvg.

```text
global_model = sum(n_i / total_n * local_model_i)
```

**Model architecture**: CNN/GCN + BiLSTM + MultiheadAttention + LayerNorm + AdaptiveSwish activation.

See `docs/simulation_experiments.md` for detailed documentation.

## Core Outputs

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`
- `data/processed/speed_data_chunks/`
- `data/analysis/density_metrics_chunks/`
- `data/analysis/node_intersection_flow_parquet/`
- `data/analysis/node_flow_curve_fit/`
- `data/analysis/date_type_curve_method_comparison/`
- `results/simulation_experiments/cnn/` (CCN/C simulation overview + ablation)
- `results/simulation_experiments/gcn/` (GCN simulation overview + ablation)

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Simulation experiments additionally require PyTorch:

```bash
pip install torch
```

Current verified environment: conda `analysis` (Python 3.9.23, torch 2.8.0+cpu).

For detailed documentation, see files under `docs/`.
