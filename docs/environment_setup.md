# 环境安装与运行说明

## 概述

本文档说明当前项目的推荐运行环境、核心依赖、安装方式、执行顺序与常见注意事项。

当前仓库中还没有 `requirements.txt` 或 `pyproject.toml`，因此这里采用“根据实际脚本导入推导出的依赖清单”进行说明。

## 1. 推荐 Python 环境

建议使用：

- Python `3.10` 或 `3.11`

推荐原因：

- `polars`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

这些依赖在 Python 3.10/3.11 下兼容性通常更稳定。

## 2. 核心依赖

根据当前项目脚本的实际导入情况，建议至少安装以下包：

- `polars`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pyarrow`
- `psutil`

补充说明：

- `pyarrow` 用于 `pandas.read_parquet()` 和 `DataFrame.to_parquet()`
- `psutil` 用于节点流量聚合脚本中的性能监控输出

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
pip install polars pandas numpy matplotlib seaborn scikit-learn pyarrow psutil
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
pip install polars pandas numpy matplotlib seaborn scikit-learn pyarrow psutil
```

## 4. 数据准备

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

- `traffic_speed_sub-dataset.v2` 文件体积较大，默认不纳入版本管理
- 若缺少该文件，后续速度分片、密度分片和节点流量分片都无法生成

## 5. 推荐执行顺序

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
python analysis_scripts/fit_node_flow_daily_curve.py
python analysis_scripts/compare_node_flow_fourier_orders.py
python analysis_scripts/compare_date_type_curve_methods.py
python analysis_scripts/visualize_node_flow_daily_curve_fit.py
```

## 6. 各阶段最小依赖关系

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
- `node_flow_curve_fit/`
- `date_type_curve_method_comparison/`

## 7. 运行注意事项

### 路径说明

当前主要脚本已统一使用“相对脚本位置推导项目根目录”的方式定位数据目录，因此：

- 推荐在项目根目录执行
- 即使从其他目录调用，只要文件结构保持不变，核心脚本通常也能正确解析路径

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

建议：

- 在本地有足够内存的环境中运行
- 优先使用 SSD
- 在试验阶段先用抽样参数或先只跑部分流程

### 中文字体

若绘图时出现中文乱码，通常是本机缺少合适中文字体。

当前脚本中已优先尝试：

- `Microsoft YaHei`
- `SimHei`
- `Noto Sans CJK SC`
- `Arial Unicode MS`

Windows 环境下一般可直接使用。

## 8. 常见问题

### 1. 找不到 `traffic_speed_sub-dataset.v2`

原因：

- 大文件默认被 `.gitignore` 忽略
- 本地尚未放入 `data/raw/`

解决：

- 将原始文件放到 `data/raw/traffic_speed_sub-dataset.v2`

### 2. 找不到 `speed_chunk_000.parquet`

原因：

- 尚未执行 `merge_speed_data.py`
- 或输出目录被清空

解决：

- 先执行 `preprocessing_scripts/merge_speed_data.py`

### 3. 找不到参数文件 `speed_class_density_params.csv`

原因：

- 参数文件未准备
- 文件字段不符合当前脚本要求

解决：

- 检查 `data/params/speed_class_density_params.csv`
- 确认至少包含：
  - `速度等级`
  - `P99.5速度`
  - `自由流速度`
  - `每车道临界密度`

### 4. `pandas` 读写 parquet 报错

原因：

- 本机未安装 `pyarrow`

解决：

```bash
pip install pyarrow
```

### 5. 节点流量或日期类型对比阶段运行太慢

原因：

- 输入分片数量多
- 需要做大量聚合、拟合和聚类

解决建议：

- 先运行检查脚本确认输入是否正确
- 对 `compare_date_type_curve_methods.py` 使用 `--node-sample-size`

示例：

```bash
python analysis_scripts/compare_date_type_curve_methods.py --node-sample-size 1000
```

## 9. 后续建议

- 后续可以把本文档中的依赖清单固化为 `requirements.txt`
- 若项目继续扩展，可再补 `Makefile`、PowerShell 脚本或批处理脚本
- 若需要完整复现与交付，建议补充数据获取方式与版本信息

## 相关文档

- [README.md](file:///e:/Jupter_Notebook/FedTrafficFlow/README.md)
- [project_pipeline.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/project_pipeline.md)
- [parameter_files.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/parameter_files.md)
- [greenshields_speed_density_scheme.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/greenshields_speed_density_scheme.md)
