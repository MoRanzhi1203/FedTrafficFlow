# FedTrafficFlow 项目文档

## 1. 项目概述

### 1.1 项目定位

`FedTrafficFlow` 当前聚焦于真实交通路网与速度观测数据的数据预处理、统计分析、密度与流量推导、节点流量构造、日内曲线拟合以及聚类解释。

### 1.2 研究目标

项目围绕以下问题展开：

1. 如何将多源原始路网与速度观测数据清洗为结构统一、可复用、可分块处理的中间数据资产。
2. 如何基于速度观测与经验参数，利用 Greenshields 模型推导路段密度、车辆数和流量。
3. 如何从路段流量进一步聚合为节点级路口流量，并构造节点日内平均曲线。
4. 如何使用傅里叶函数对节点日内流量曲线进行紧凑表达，并从曲线形态上做聚类解释。

### 1.3 应用场景

- 城市道路交通运行状态分析。
- 路段速度到路口流量的推导与中间变量构造。
- 路口日内车流规律提取与模式聚类。
- 后续交通预测建模与特征工程准备。

### 1.4 核心解决的问题

- 通过按时间段分块的 Parquet 方案降低大体量速度数据处理压力。
- 通过参数表和 Greenshields 关系式把速度观测转化为可分析的流量指标。
- 通过显式区分进入流量、离开流量和综合流量提升节点定义清晰度。
- 通过傅里叶拟合和聚类特征构造压缩高维节点日内曲线。

## 2. 环境依赖说明

### 2.1 操作系统与运行环境

项目代码主要面向本地 Python 环境运行，当前仓库与脚本写法在 Windows 环境下已经验证可用。由于多数脚本使用 `Path(__file__).resolve().parents[1]` 推导项目根目录，因此只要目录结构保持不变，Windows、Linux 和 macOS 均可迁移。

### 2.2 推荐 Python 版本

建议使用：

- Python `3.10` 或 `3.11`

### 2.3 仓库实际使用到的第三方库

| 类别 | 依赖 |
| --- | --- |
| 数据处理 | `numpy`、`pandas`、`polars` |
| Parquet 支持 | `pyarrow` |
| 绘图分析 | `matplotlib`、`seaborn` |
| 机器学习 | `scikit-learn` |
| 深度学习 | `torch` (PyTorch) |
| 资源监控 | `psutil` |
| 测试 | `pytest` |

### 2.4 可复现环境建议

建议创建如下最小复现环境：

```text
Python 3.11
numpy
pandas
polars
matplotlib
seaborn
scikit-learn
pyarrow
psutil
pytest
```

### 2.5 安装方式

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 代码架构说明

### 3.1 顶层目录结构

```text
FedTrafficFlow/
 analysis_scripts/                # 真实交通数据分析、密度计算、节点曲线拟合与聚类
 dataset_inspection_scripts/      # 数据结构检查、顺序检查、样例查看
 docs/                            # 项目说明文档
 simulation_experiments/          # 联邦仿真实验脚本 (CNN/GCN + BiLSTM + Attention)
 results/                         # 仿真实验结果输出 (PNG, CSV, TXT)
 data/
   raw/                          # 原始数据
   processed/                    # 预处理后的标准化数据与速度分块
   params/                       # 参数表
   analysis/                     # 分析结果与中间产物
 preprocessing_scripts/           # 原始路网与速度数据预处理
 test/                           # Jupyter Notebook 实验文件
 README.md
 requirements.txt
```

### 3.2 模块划分

#### A. 预处理模块

- `process_link_gps.py`：清洗路段 GPS 坐标。
- `process_rnsd.py`：清洗路网结构属性并推导起终点坐标。
- `merge_speed_data.py`：将原始速度观测与路网静态属性关联，输出速度分块。

#### B. 分析计算模块

- `summarize_speed_stats.py`：速度整体统计、日期-时段统计。
- `visualize_speed_hist_by_period.py`：各速度等级的分时段直方图。
- `add_p995_to_speed_histogram.py`：估计速度等级的 `P99.5`。
- `compute_greenshields_density.py`：从速度推导密度、车辆数和流量。
- `compute_node_intersection_flow_optimized.py`：路段流量聚合为节点流量。
- `fit_node_flow_daily_curve.py`：节点 96 点日内曲线傅里叶拟合。
- `compare_node_flow_fourier_orders.py`：比较不同傅里叶阶数。
- `compare_date_type_curve_methods.py`：比较 `M0/M1/M2/M3` 日期类型方法并做聚类。
- `visualize_fitted_function_clusters.py`：对已有聚类结果做函数层面的解释性可视化。
- `visualize_node_flow_daily_curve_fit.py`：拟合质量分布与代表节点曲线展示。

#### C. 检查模块

- `inspect_speed_data_chunks.py`
- `inspect_density_metrics_chunks.py`
- `check_density_time_order.py`
- `inspect_node_intersection_flow.py`
- `inspect_road_directionality.py`
- `check_spatial_node_completeness.py`

这些脚本不产生主流程结果，但用于保障主流程输入正确。

#### D. 仿真实验模块

**文件位置**：`simulation_experiments/`

**核心脚本**：

- `cnn_fed_base.py`：CNN + BiLSTM + Attention 联邦仿真实验。
- `gcn_fed_base.py`：GCN + BiLSTM + Attention 联邦仿真实验（链式图结构）。

**仿真内容**：

1. **总览实验 (overview)**：比较 CCN-FedAvg / GCN-FedAvg 与 Independent Training 在异构客户端上的性能差异。
2. **消融实验 (ablation)**：比较完整模型与移除 Attention / 移除 LSTM / 移除空间编码器的变体性能。

**联邦聚合方式**：标准样本量加权 FedAvg。

```text
global_model = sum(n_i / total_n * local_model_i)
```

**模型架构**：

- CNN 分支：Conv1d×2 + BN + AdaptiveSwish + AdaptiveAvgPool1d
- GCN 分支：GCNLayer×2 + LayerNorm + AdaptiveSwish（可学习邻接矩阵）
- BiLSTM 分支 + MultiheadAttention(4 head) + 残差连接
- 回归头：Linear → LayerNorm → AdaptiveSwish → Dropout → Linear

**输出位置**：`results/simulation_experiments/cnn/` 和 `results/simulation_experiments/gcn/`

**详细文档**：参见 `docs/simulation_experiments.md`。

### 3.3 模块调用逻辑与数据流转

```text
link_gps.v2
  + road_network_sub-dataset.v2
  + traffic_speed_sub-dataset.v2
        |
        v
process_link_gps.py
process_rnsd.py
merge_speed_data.py
        |
        v
speed_data_chunks/
        |
        +--> summarize_speed_stats.py
        +--> visualize_speed_hist_by_period.py
        +--> add_p995_to_speed_histogram.py
        +--> compute_greenshields_density.py
                    |
                    v
          density_metrics_chunks/
                    |
                    v
compute_node_intersection_flow_optimized.py
                    |
                    v
node_intersection_flow_parquet/
                    |
                    +--> check_spatial_node_completeness.py
                    +--> fit_node_flow_daily_curve.py
                    |         |
                    |         +--> compare_node_flow_fourier_orders.py
                    |         +--> visualize_node_flow_daily_curve_fit.py
                    |
                    +--> compare_date_type_curve_methods.py
                               |
                               +--> visualize_fitted_function_clusters.py
```

## 4. 核心实现逻辑

### 4.1 原始速度数据与路网属性的分块合并

核心脚本：`preprocessing_scripts/merge_speed_data.py`

实现目标：

- 将原始速度观测表与静态路网属性按 `路段ID` 关联。
- 按时间段区间切分为多个 Parquet 文件。
- 控制单次处理数据量，避免一次性载入全量大表。

### 4.2 基于 Greenshields 模型的密度与流量推导

核心脚本：`analysis_scripts/compute_greenshields_density.py`

实现目标：

- 结合经验参数，将路段平均速度转换为密度、车辆数、小时流率与 15 分钟交通量。

### 4.3 路段流量聚合为节点流量

核心脚本：`analysis_scripts/compute_node_intersection_flow_optimized.py`

实现目标：

- 把每条路段的流量聚合为节点级进入流量、离开流量和综合流量。

### 4.4 节点日内平均曲线的傅里叶拟合

核心脚本：`analysis_scripts/fit_node_flow_daily_curve.py`

实现目标：

- 对每个节点构造 96 个 15 分钟点的平均日内曲线。
- 使用傅里叶基函数做最小二乘拟合。

### 4.5 日期类型方法比较与聚类解释

核心脚本：

- `analysis_scripts/compare_date_type_curve_methods.py`
- `analysis_scripts/visualize_fitted_function_clusters.py`

实现目标：

- 比较不同日期类型曲线构造方法。
- 对节点曲线形态进行聚类与可视化解释。

## 5. 使用操作指南

### 5.1 数据准备

将原始数据放入：

- `data/raw/link_gps.v2`
- `data/raw/road_network_sub-dataset.v2`
- `data/raw/traffic_speed_sub-dataset.v2`

### 5.2 真实交通分析链路执行顺序

```powershell
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

### 5.3 仿真实验执行

激活环境并运行：

```powershell
conda activate analysis
cd simulation_experiments
python cnn_fed_base.py --workflow all
python gcn_fed_base.py --workflow all
```

workflow 选项：`all`（默认，overview + ablation）、`overview`、`ablation`。

输出：
- `results/simulation_experiments/cnn/`：CNN 仿真 PNG / CSV / TXT
- `results/simulation_experiments/gcn/`：GCN 仿真 PNG / CSV / TXT

详细文档：`docs/simulation_experiments.md`

### 5.4 推荐的快速验证方式

如果只想快速验证流程是否通畅，至少执行到：

```powershell
python preprocessing_scripts/process_link_gps.py
python preprocessing_scripts/process_rnsd.py
python preprocessing_scripts/merge_speed_data.py
python analysis_scripts/compute_greenshields_density.py
python analysis_scripts/compute_node_intersection_flow_optimized.py
python analysis_scripts/check_spatial_node_completeness.py
```

## 6. 结果输出说明

### 6.1 预处理输出

| 路径 | 格式 | 含义 |
| --- | --- | --- |
| `data/processed/link_gps_processed.csv` | CSV | 清洗后的路段 GPS 坐标 |
| `data/processed/rnsd_processed.csv` | CSV | 清洗后的路网属性与节点映射 |
| `data/processed/speed_data_chunks/speed_chunk_*.parquet` | Parquet | 按时间段分块的速度-路网融合结果 |

### 6.2 分析输出

| 路径 | 格式 | 含义 |
| --- | --- | --- |
| `data/analysis/speed_class_daily_period_stats.csv` | CSV | 按日期-时段-速度等级的聚合统计 |
| `data/analysis/density_metrics_chunks/density_chunk_*.parquet` | Parquet | 密度、车辆数和流量等派生结果 |
| `data/analysis/node_intersection_flow_parquet/node_flow_chunk_*.parquet` | Parquet | 节点级进入流量、离开流量和综合流量 |
| `data/analysis/node_flow_curve_fit/` | Directory | 傅里叶拟合结果与可视化产物 |
| `data/analysis/date_type_curve_method_comparison/` | Directory | 日期类型方法比较与聚类结果 |

## 7. 维护与扩展指引

- 更新参数表时，同时同步检查 `README.md` 与 `docs/parameter_files.md`。
- 新增分析结果目录时，补充 `docs/` 中对应说明。
- 若修改字段名，优先检查所有下游脚本的字段依赖。
- 大体量中间产物不建议纳入普通 Git 提交。
- 若要提升可复现性，可逐步为主流程脚本补充更稳定的 CLI 参数接口。

## 8. 新接入人员推荐路径

1. 先读 `README.md` 与本文件，理解项目主流程。
2. 运行预处理脚本，确认 `processed/` 数据生成成功。
3. 运行 `compute_greenshields_density.py` 和 `compute_node_intersection_flow_optimized.py`。
4. 运行 `check_spatial_node_completeness.py` 确认节点流量数据可用。
5. 运行 `fit_node_flow_daily_curve.py` 与 `compare_date_type_curve_methods.py`。
6. 最后运行可视化脚本，完成结果解释与图表整理。
