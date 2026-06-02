# 环境安装与运行说明

## 概述

本文档说明当前项目的推荐运行环境、核心依赖、安装方式、执行顺序与常见注意事项。

## 1. 推荐 Python 环境

建议使用：

- Python `3.9` - `3.11`（当前项目验证环境：Python 3.9.23, conda env `analysis`）

推荐原因：

- `polars`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch`（仿真实验需要）

这些依赖在 Python 3.9 - 3.11 下兼容性通常更稳定。

## 2. 核心依赖

根据当前项目脚本的实际导入情况，建议至少安装以下包：

- `numpy`
- `pandas`
- `polars`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pyarrow`
- `psutil`
- `pytest`
- `torch`（联邦仿真实验需要）

补充说明：

- `pyarrow` 用于 `pandas.read_parquet()` 和 `DataFrame.to_parquet()`。
- `psutil` 用于节点流量聚合脚本中的性能监控输出。
- `scikit-learn` 用于日期类型方法比较中的聚类分析。
- `torch` 用于仿真实验中的 CNN/GCN + BiLSTM + Attention 模型训练。

## 3. 推荐安装方式

### 方式一：使用 `venv`

在项目根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 方式二：使用 conda

创建环境：

```bash
conda create -n fedtrafficflow python=3.11
```

激活环境：

```bash
conda activate fedtrafficflow
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 当前项目使用的 conda 环境

当前项目已在 `analysis` conda 环境下验证通过：

```powershell
conda activate analysis
python --version     # Python 3.9.23
```

该环境包含完整依赖：

- torch 2.8.0+cpu
- numpy 2.0.2
- pandas 2.3.3
- matplotlib 3.9.4
- sklearn 1.6.1
- seaborn

## 4. 仿真实验环境要求

仿真实验（`simulation_experiments/`）需要额外安装 PyTorch：

```bash
pip install torch
```

验证安装：

```python
import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

仿真实验脚本已内置自动 GPU/CPU 判断逻辑。

## 5. 数据准备

运行脚本前，建议先确认以下数据位置：

### 原始输入

- `data/raw/link_gps.v2`
- `data/raw/road_network_sub-dataset.v2`
- `data/raw/traffic_speed_sub-dataset.v2`

### 参数文件

- `data/params/speed_class_density_params.csv`

可选参考参数文件：

- `data/params/beijing_capacity_params.csv`

补充说明：

- `traffic_speed_sub-dataset.v2` 文件体积较大，默认不纳入版本管理。
- 若缺少该文件，后续速度分片、密度分片和节点流量分片都无法生成。

## 6. 推荐执行顺序

主流程建议按以下顺序运行：

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

## 7. 各阶段最小依赖关系

### 预处理阶段

需要先准备：

- `link_gps.v2`
- `road_network_sub-dataset.v2`
- `traffic_speed_sub-dataset.v2`

输出：

- `link_gps_processed.csv`
- `rnsd_processed.csv`
- `speed_data_chunks/`

### 速度分析阶段

需要：

- `speed_data_chunks/`

输出：

- 速度总体统计
- 分时段统计
- 直方图图片
- P99.5 统计结果

### 密度与流量阶段

需要：

- `speed_data_chunks/`
- `speed_class_density_params.csv`

输出：

- `density_metrics_chunks/`

### 节点流量与曲线阶段

需要：

- `density_metrics_chunks/`
- `rnsd_processed.csv`

输出：

- `node_intersection_flow_parquet/`
- `node_intersection_flow_check_reports/`
- `node_flow_curve_fit/`
- `date_type_curve_method_comparison/`
- `date_type_curve_method_comparison/function_cluster_visualization/`

## 8. 运行注意事项

### 路径说明

当前主要脚本已统一使用相对脚本位置推导项目根目录的方式定位数据目录，因此：

- 推荐在项目根目录执行。
- 即使从其他目录调用，只要文件结构保持不变，核心脚本通常也能正确解析路径。

### 磁盘空间

项目会生成较大的中间结果，尤其是：

- `data/processed/speed_data_chunks/`
- `data/analysis/density_metrics_chunks/`
- `data/analysis/node_intersection_flow_parquet/`
- `data/analysis/date_type_curve_method_comparison/`

建议预留充足磁盘空间。

### 内存与执行时间

以下脚本资源消耗相对较大：

- `preprocessing_scripts/merge_speed_data.py`
- `analysis_scripts/compute_greenshields_density.py`
- `analysis_scripts/compute_node_intersection_flow_optimized.py`
- `analysis_scripts/compare_date_type_curve_methods.py`
- `analysis_scripts/visualize_fitted_function_clusters.py`

建议：

- 在本地有足够内存的环境中运行。
- 优先使用 SSD。
- 在试验阶段先用抽样参数或先只跑部分流程。

## 9. 常见问题

### 1. 找不到 `traffic_speed_sub-dataset.v2`

原因：

- 大文件默认被 `.gitignore` 忽略。
- 本地尚未放入 `data/raw/`。

解决：

- 将原始文件放到 `data/raw/traffic_speed_sub-dataset.v2`。

### 2. 找不到 `speed_chunk_000.parquet`

原因：

- 尚未执行 `merge_speed_data.py`。
- 或输出目录被清空。

解决：

- 先执行 `preprocessing_scripts/merge_speed_data.py`。

### 3. 找不到参数文件 `speed_class_density_params.csv`

原因：

- 参数文件未准备。
- 文件字段不符合当前脚本要求。

解决：

- 检查 `data/params/speed_class_density_params.csv`。
- 确认至少包含 `速度等级`、`P99.5速度`、`自由流速度`、`每车道临界密度`。

### 4. `pandas` 读写 parquet 报错

原因：

- 本机未安装 `pyarrow`。

解决：

```bash
pip install pyarrow
```

### 5. 节点流量或日期类型对比阶段运行太慢

原因：

- 输入分片数量多。
- 需要做大量聚合、拟合和聚类。

解决建议：

- 先运行检查脚本确认输入是否正确。
- 推荐使用 `analysis_scripts/check_spatial_node_completeness.py` 作为节点流量检查入口。
- 对 `compare_date_type_curve_methods.py` 使用 `--node-sample-size`。

示例：

```bash
python analysis_scripts/compare_date_type_curve_methods.py --node-sample-size 1000
```

## 10. 后续建议

- 当前已提供 `requirements.txt`，后续可进一步按环境锁定版本号。
- 若项目继续扩展，可再补 `Makefile`、PowerShell 脚本或批处理脚本。
- 若需要完整复现与交付，建议补充数据获取方式与版本信息。

## 相关文档

- [README.md](file:///e:/Jupter_Notebook/FedTrafficFlow/README.md)
- [project_pipeline.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/project_pipeline.md)
- [project_documentation.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/project_documentation.md)
- [parameter_files.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/parameter_files.md)
- [greenshields_speed_density_scheme.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/greenshields_speed_density_scheme.md)
- [function_cluster_visualization.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/function_cluster_visualization.md)
