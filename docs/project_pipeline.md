# 项目处理流程总览

## 概述

本文档从“原始数据输入”到“分析结果输出”梳理整个项目的数据处理链路，便于快速理解各脚本之间的先后依赖关系。

当前项目的主流程可以概括为：

```text
原始路网与坐标数据
-> 基础预处理
-> 速度与路网属性合并
-> 速度统计与可视化
-> Greenshields 密度与流量计算
-> 路段流量聚合为节点流量
-> 节点日内平均曲线拟合
-> 拟合阶数与日期类型方法比较
```

## 1. 原始输入

当前项目主要使用以下原始数据：

- `data/raw/link_gps.v2`
- `data/raw/road_network_sub-dataset.v2`
- `data/raw/traffic_speed_sub-dataset.v2`

其中：

- `link_gps.v2` 提供路段坐标信息
- `road_network_sub-dataset.v2` 提供路网属性、方向和节点信息
- `traffic_speed_sub-dataset.v2` 提供大规模交通速度观测

## 2. 基础预处理

### 2.1 路段坐标清洗

脚本：

- `preprocessing_scripts/process_link_gps.py`

输入：

- `data/raw/link_gps.v2`

输出：

- `data/processed/link_gps_processed.csv`

主要作用：

- 去重
- 去除关键字段空值
- 统一字段类型

### 2.2 路网属性清洗

脚本：

- `preprocessing_scripts/process_rnsd.py`

输入：

- `data/raw/road_network_sub-dataset.v2`

输出：

- `data/processed/rnsd_processed.csv`

主要作用：

- 英文字段重命名为中文字段
- 清洗缺失和重复数据
- 推导起始节点、结束节点对应的坐标信息

## 3. 速度数据与路网属性合并

脚本：

- `preprocessing_scripts/merge_speed_data.py`

输入：

- `data/raw/traffic_speed_sub-dataset.v2`
- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`

输出：

- `data/processed/speed_data_chunks/`

输出文件命名模式：

- `speed_chunk_000.parquet`
- `speed_chunk_001.parquet`
- ...

主要作用：

- 将速度观测与路段静态属性进行关联
- 按时间段区间分块生成结构化 Parquet 文件
- 作为后续所有分析脚本的统一输入

## 4. 速度统计与可视化

### 4.1 总体与分时段统计

脚本：

- `analysis_scripts/summarize_speed_stats.py`

输入：

- `data/processed/speed_data_chunks/`

输出：

- `data/analysis/speed_class_overall_stats.csv`
- `data/analysis/speed_class_daily_period_stats.csv`

作用：

- 汇总不同速度等级的总体速度统计
- 统计日期-时段-速度等级层面的聚合结果

### 4.2 分时段直方图

脚本：

- `analysis_scripts/visualize_speed_hist_by_period.py`

输入：

- `data/processed/speed_data_chunks/`

输出：

- `data/analysis/speed_histogram_counts_by_period_by_class.csv`
- `data/analysis/speed_histograms_by_period_by_class/`

作用：

- 按速度等级和时段绘制速度分布直方图

### 4.3 P99.5 统计

脚本：

- `analysis_scripts/add_p995_to_speed_histogram.py`

输入：

- `data/processed/speed_data_chunks/`

输出：

- `data/analysis/speed_histograms_by_class_p995.csv`
- `data/analysis/speed_histograms_by_class_with_p995_percent.png`

作用：

- 估计各速度等级的 P99.5
- 为后续参数表整理提供统计依据

## 5. 密度与流量计算

脚本：

- `analysis_scripts/compute_greenshields_density.py`

输入：

- `data/processed/speed_data_chunks/`
- `data/params/speed_class_density_params.csv`

输出：

- `data/analysis/density_metrics_chunks/`

输出文件命名模式：

- `density_chunk_000.parquet`
- `density_chunk_001.parquet`
- ...

作用：

- 将速度观测转换为密度、车辆数、小时流率、15 分钟交通量等派生物理量

补充说明：

- 参数文件来源与角色见 [parameter_files.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/parameter_files.md)
- 数学公式与建模规则见 [greenshields_speed_density_scheme.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/greenshields_speed_density_scheme.md)

## 6. 路段流量聚合为节点流量

脚本：

- `analysis_scripts/compute_node_intersection_flow_optimized.py`

输入：

- `data/analysis/density_metrics_chunks/`
- `data/processed/rnsd_processed.csv`

输出：

- `data/analysis/node_intersection_flow_parquet/`

输出文件命名模式：

- `node_flow_chunk_000.parquet`
- `node_flow_chunk_001.parquet`
- ...

作用：

- 以路段起止节点为基础
- 分别聚合进入流量和离开流量
- 生成节点级综合车流量结果

更多检查说明见：

- [node_intersection_flow_inspection.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/node_intersection_flow_inspection.md)

## 7. 节点日内平均曲线拟合

### 7.1 正式拟合流程

脚本：

- `analysis_scripts/fit_node_flow_daily_curve.py`

输入：

- `data/analysis/node_intersection_flow_parquet/`

输出：

- `data/analysis/node_flow_curve_fit/node_flow_fitted_daily_curves.parquet`
- `data/analysis/node_flow_curve_fit/node_flow_curve_coefficients.parquet`

作用：

- 构造节点级 96 点日内平均流量曲线
- 对每个节点做傅里叶拟合

### 7.2 阶数比较

脚本：

- `analysis_scripts/compare_node_flow_fourier_orders.py`

输出：

- `data/analysis/node_flow_curve_fit/node_flow_fourier_order_comparison.json`

作用：

- 比较不同傅里叶阶数的拟合质量

### 7.3 可视化

脚本：

- `analysis_scripts/visualize_node_flow_daily_curve_fit.py`

输出：

- `data/analysis/node_flow_curve_fit/plots/`

更多说明见：

- [node_flow_daily_curve_fit.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/node_flow_daily_curve_fit.md)

## 8. 日期类型方法比较

脚本：

- `analysis_scripts/compare_date_type_curve_methods.py`

输入：

- `data/analysis/node_intersection_flow_parquet/`

输出：

- `data/analysis/date_type_curve_method_comparison/`

作用：

- 基于 `workday / weekend / holiday` 三类日期构造不同方法
- 比较 `M0 / M1 / M2 / M3` 四种处理方案的拟合与聚类表现

更多说明见：

- [date_type_curve_method_comparison.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/date_type_curve_method_comparison.md)

## 9. 检查脚本位置

当前项目还提供多类检查脚本：

- `dataset_inspection_scripts/inspect_speed_data_chunks.py`
- `dataset_inspection_scripts/inspect_density_metrics_chunks.py`
- `dataset_inspection_scripts/check_density_time_order.py`
- `dataset_inspection_scripts/inspect_node_intersection_flow.py`
- `dataset_inspection_scripts/inspect_road_directionality.py`

这些脚本不参与主流程产出，但用于：

- 核验字段结构
- 检查分片顺序
- 抽样查看样例值
- 研究路网方向性

## 10. 推荐执行顺序

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

## 11. 使用建议

- 若只想复现主流程，优先执行到 `fit_node_flow_daily_curve.py`
- 若只想验证日期类型方法实验，主流程至少需要先跑到 `node_intersection_flow_parquet/`
- 若本地磁盘空间有限，可优先保留脚本和轻量级 CSV 结果，重型分块结果可按需重建
