# FedTrafficFlow 项目进展分析报告

> 更新日期：2026-06-15

## 1. 项目总览

本项目当前保留两条核心主线：

| 主线 | 说明 | 状态 |
|------|------|------|
| 仿真实验（Simulation） | CNN / GCN 联邦交通流预测仿真实验 | 已完成 |
| 真实数据主线（Real Data） | 真实数据预处理、分析与后续联邦预测准备 | 进行中 |

## 2. 仿真实验状态

仿真实验相关代码与结果保留，核心范围包括：

- `simulation_experiments/cnn_fed_base.py`
- `simulation_experiments/gcn_fed_base.py`
- `simulation_experiments/cnn_fed_enhanced_experiments.py`
- `simulation_experiments/gcn_fed_enhanced_experiments.py`
- `simulation_experiments/fed_robustness_experiments.py`

已保留的仿真实验内容包括：

- FedAvg 与 Independent 对比；
- CNN / GCN 两类模型；
- 图结构、聚合策略、鲁棒性等仿真实验；
- 论文仿真实验章节与图表资产。

## 3. 真实数据主线状态

### 3.1 已完成内容

- 原始真实数据清洗；
- 路网属性清洗；
- 速度数据合并与分块；
- Greenshields 密度计算；
- 路口节点车流量构造；
- 节点日曲线拟合与日期类型分析；
- 真实数据预处理审计。

### 3.2 当前保留的数据与产物

- `data/processed/rnsd_processed.csv`
- `data/processed/speed_data_chunks/`
- `data/analysis/node_intersection_flow_parquet/`
- `data/analysis/node_flow_curve_fit/`
- `data/analysis/date_type_curve_method_comparison/`
- `results/real_data_preprocessing/`

## 4. 当前后续任务

当前重点转向真实数据联邦预测主任务，建议顺序为：

1. 真实数据滑动窗口构造；
2. 预测任务归一化与 scaler 保存；
3. 训练 / 验证 / 测试划分；
4. 联邦客户端划分；
5. 真实路网邻接矩阵构造；
6. FedAvg / Independent 真实数据训练入口。

## 5. 保留文件范围

### 保留的数据

- `data/raw/`
- `data/processed/`
- `data/analysis/node_intersection_flow_parquet/`

### 保留的真实数据分析脚本

- `analysis_scripts/audit_real_data_preprocessing.py`
- `analysis_scripts/compute_greenshields_density.py`
- `analysis_scripts/compute_node_intersection_flow_optimized.py`
- `analysis_scripts/fit_node_flow_daily_curve.py`
- `analysis_scripts/compare_node_flow_fourier_orders.py`
- `analysis_scripts/compare_date_type_curve_methods.py`
- `analysis_scripts/visualize_fitted_function_clusters.py`

### 保留的仿真实验脚本

- `simulation_experiments/`

## 6. 总结

当前项目定位为：

- 保留真实数据预处理与分析资产；
- 保留仿真实验与联邦学习主线；
- 后续围绕真实数据联邦预测训练继续推进。
