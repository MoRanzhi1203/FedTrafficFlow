# FedTrafficFlow

## 项目当前进度

当前仓库已完成交通路网基础数据预处理，并新增了基于速度分片结果的统计分析与可视化脚本，可直接用于后续建模、数据理解与联邦学习实验设计。

已完成内容：

- 完成 `link_gps` 原始数据清洗与格式化。
- 完成 `road_network_sub-dataset` 路网结构数据清洗、字段重命名与节点坐标推导。
- 完成交通速度数据与路网信息的分块合并脚本。
- 产出可复用的处理后 CSV 文件，供后续训练、分析与联邦切分使用。
- 新增速度等级总体统计、按日期时段统计、分时段直方图与 P99.5 可视化分析结果。
- 新增基于 Greenshields 模型的密度、车辆数与流量计算脚本及查询脚本。
- 新增路段流量聚合为路口节点流量的脚本、校验脚本和检查文档。
- 新增路口节点日内平均车流量的傅里叶曲线拟合、阶数比较、日期类型方法对比脚本及相关说明。
- 新增基于既有聚类结果的函数曲线可视化脚本，可直接输出分位带图、抽样函数云图、代表节点图与残差诊断图。

## 当前目录结构

```text
FedTrafficFlow/
├─ analysis_scripts/
│  ├─ add_p995_to_speed_histogram.py
│  ├─ compare_date_type_curve_methods.py
│  ├─ check_spatial_node_completeness.py
│  ├─ compute_greenshields_density.py
│  ├─ compare_node_flow_fourier_orders.py
│  ├─ compute_node_intersection_flow_optimized.py
│  ├─ fit_node_flow_daily_curve.py
│  ├─ summarize_speed_stats.py
│  ├─ visualize_fitted_function_clusters.py
│  ├─ visualize_node_flow_daily_curve_fit.py
│  ├─ visualize_speed_hist_by_period.py
├─ dataset_inspection_scripts/
│  ├─ check_density_time_order.py
│  ├─ inspect_density_metrics_chunks.py
│  ├─ inspect_node_intersection_flow.py
│  ├─ inspect_road_directionality.py
│  └─ inspect_speed_data_chunks.py
├─ docs/
│  ├─ date_type_curve_method_comparison.md
│  ├─ environment_setup.md
│  ├─ greenshields_speed_density_scheme.md
│  ├─ function_cluster_visualization.md
│  ├─ node_flow_daily_curve_fit.md
│  ├─ node_intersection_flow_inspection.md
│  ├─ parameter_files.md
│  └─ project_pipeline.md
├─ data/
│  ├─ analysis/
│  │  ├─ speed_histogram_counts_by_period_by_class.csv
│  │  ├─ speed_histograms_by_class_p995.csv
│  │  ├─ speed_histograms_by_class_with_p995_percent.png
│  │  └─ speed_histograms_by_period_by_class/
│  ├─ params/
│  │  ├─ beijing_capacity_params.csv
│  │  └─ speed_class_density_params.csv
│  ├─ raw/
│  │  ├─ link_gps.v2
│  │  └─ road_network_sub-dataset.v2
│  └─ processed/
│     ├─ link_gps_processed.csv
│     ├─ rnsd_processed.csv
│     └─ speed_data_chunks/
├─ preprocessing_scripts/
│  ├─ process_link_gps.py
│  ├─ process_rnsd.py
│  └─ merge_speed_data.py
└─ README.md
```

## 预处理脚本说明

### `preprocessing_scripts/process_link_gps.py`

功能：

- 读取 `data/raw/link_gps.v2`
- 去重并删除关键字段缺失值
- 转换字段类型
- 输出 `data/processed/link_gps_processed.csv`

### `preprocessing_scripts/process_rnsd.py`

功能：

- 读取 `data/raw/road_network_sub-dataset.v2`
- 将英文列名重命名为中文字段
- 清洗缺失值与重复路段
- 转换数值字段类型
- 根据方向和长度推导起点/终点经纬度
- 输出 `data/processed/rnsd_processed.csv`

### `preprocessing_scripts/merge_speed_data.py`

功能：

- 读取已处理的 `link_gps_processed.csv` 与 `rnsd_processed.csv`
- 生成用于关联交通速度数据的路网信息
- 使用 `Polars` 按块读取超大规模速度数据
- 将速度数据和路网信息合并后输出到 `data/processed/speed_data_chunks/`

## 分析脚本说明

### `analysis_scripts/summarize_speed_stats.py`

功能：

- 读取 `data/processed/speed_data_chunks/` 下的所有速度分片
- 基于 `Polars` 流式引擎统计各速度等级的总体速度分布
- 按日期、时段、速度等级生成分组统计结果
- 生成可供后续可视化和校验使用的聚合统计结果
- 输出 `data/analysis/speed_class_overall_stats.csv`
- 输出 `data/analysis/speed_class_daily_period_stats.csv`

### `analysis_scripts/visualize_speed_hist_by_period.py`

功能：

- 对所有速度分片按速度等级和时段进行聚合
- 生成各速度等级在 6 个时段下的平均速度频率直方图
- 输出 `data/analysis/speed_histogram_counts_by_period_by_class.csv`
- 输出到 `data/analysis/speed_histograms_by_period_by_class/`

### `analysis_scripts/add_p995_to_speed_histogram.py`

功能：

- 统计各速度等级的平均速度分布百分比直方图
- 基于细粒度分箱估计各速度等级的 `P99.5`
- 输出 `data/analysis/speed_histograms_by_class_p995.csv`
- 输出 `data/analysis/speed_histograms_by_class_with_p995_percent.png`

### `analysis_scripts/compute_greenshields_density.py`

功能：

- 按 `100 / 80 / 60 km/h` 三档为速度等级分配标准化自由流速度
- 为不同速度等级分配每车道临界密度参数
- 对超过自由流速度的观测做截断，避免出现负密度
- 输出路段密度、车辆数估计和 15 分钟流量等衍生指标
- 读取 `data/params/speed_class_density_params.csv`
- 按 `时间段`、`路段ID`、`速度等级` 升序输出密度分块结果
- 输出 `data/analysis/density_metrics_chunks/`

### `analysis_scripts/compare_node_flow_fourier_orders.py`

功能：

- 复用 `fit_node_flow_daily_curve.py` 中的日内平均曲线构造逻辑
- 对多个傅里叶阶数进行批量最小二乘拟合比较
- 统计各阶数在全体完整节点上的 `RMSE`、`MAE`、`R2`、`WMAPE` 和全局 `R2`
- 输出比较结果到 `data/analysis/node_flow_curve_fit/node_flow_fourier_order_comparison.json`

### `analysis_scripts/compare_date_type_curve_methods.py`

功能：

- 基于节点流量分片构造 `workday / weekend / holiday` 三类日期类型
- 对 `M0 / M1 / M2 / M3` 四种日内曲线构造方法进行统一拟合与聚类比较
- 输出各方法的拟合结果、聚类标签、聚类汇总与类中心结果到 `data/analysis/date_type_curve_method_comparison/`
- 输出方法级比较表和对比图片到 `data/analysis/date_type_curve_method_comparison/comparison/`

### `analysis_scripts/visualize_fitted_function_clusters.py`

功能：

- 读取 `compare_date_type_curve_methods.py` 已生成的各方法 parquet 结果
- 不重新聚类，只在“已拟合函数 + 已有 cluster_labels”基础上做曲线层面的可视化和解释
- 输出抽样函数云图、技术检查 overlay 图、分位带图、代表节点图、全量诊断图、原始/归一化残差图
- 输出目录为 `data/analysis/date_type_curve_method_comparison/function_cluster_visualization/`

### `analysis_scripts/compute_node_intersection_flow_optimized.py`

功能：

- 读取 `data/analysis/density_metrics_chunks/` 中 61 个路段流量分片
- 结合 `data/processed/rnsd_processed.csv` 中的路段起终点节点映射
- 计算每个节点、每个时间段的 `路口进入流量`、`路口离开流量` 和 `路口车流量`
- 保持按天分文件，每个文件覆盖 96 个连续的全局时间段
- 在脚本内部直接按 `时间段`、`节点ID` 升序输出节点流量分块结果
- 输出 `data/analysis/node_intersection_flow_parquet/`

### `analysis_scripts/check_spatial_node_completeness.py`

功能：

- 读取 `data/analysis/node_intersection_flow_parquet/` 下所有 `node_flow_chunk_*.parquet`
- 只检查节点时空数据的完整性，不执行空间均值填补，不写 `daily_parquet` 副本
- 检查输入分片数量、必需字段、时间段连续性、节点集合一致性、每日缺失/重复、非法流量值
- 输出完整性报告到 `data/analysis/node_intersection_flow_check_reports/`
- 可选使用 `--write-missing-detail` 仅在发现缺失时写出逐日缺失明细

### `analysis_scripts/fit_node_flow_daily_curve.py`

功能：

- 读取 `data/analysis/node_intersection_flow_parquet/` 下所有 `node_flow_chunk_*.parquet`
- 仅使用 `节点ID`、`时间段`、`路口车流量` 三列参与计算
- 将全局 `时间段` 转换为 `日内时间段 = 时间段 % 96`
- 对每个节点的 96 点日内平均 `路口车流量` 曲线做傅里叶最小二乘拟合
- 输出节点级拟合曲线结果和傅里叶系数结果到 `data/analysis/node_flow_curve_fit/`

### `analysis_scripts/visualize_node_flow_daily_curve_fit.py`

功能：

- 读取 `node_flow_fitted_daily_curves.parquet` 和 `node_flow_curve_coefficients.parquet`
- 绘制全体节点的 `RMSE`、`MAE`、`R2` 分布以及 `平均流量 vs R2` 散点图
- 默认选择平均流量最高的一批节点，绘制日内平均曲线与拟合曲线对比图
- 输出图片到 `data/analysis/node_flow_curve_fit/plots/`

## 查询脚本说明

### `dataset_inspection_scripts/inspect_speed_data_chunks.py`

功能：

- 查看 `data/processed/speed_data_chunks/` 下分块文件概览
- 输出样例分块的形状、列名、数据类型、空值统计和样例值
- 输出前 50 行与后 20 行数据预览

### `dataset_inspection_scripts/inspect_density_metrics_chunks.py`

功能：

- 查看 `data/analysis/density_metrics_chunks/` 下分块文件概览
- 输出样例分块的形状、列名、数据类型、空值统计和样例值
- 输出前 50 行与后 20 行数据预览

### `dataset_inspection_scripts/check_density_time_order.py`

功能：

- 逐文件读取 `density_metrics_chunks` 中的 `时间段` 列
- 按文件中实际出现顺序压缩连续重复值并输出时间段序列
- 校验每个文件的 `时间段` 是否连续升序
- 输出异常说明及全量汇总结果，便于确认分块顺序是否正确

### `dataset_inspection_scripts/inspect_node_intersection_flow.py`

功能：

- 查看 `data/analysis/node_intersection_flow_parquet/` 下节点流量分块文件概览
- 输出样例分块的形状、列名、数据类型、空值统计和样例值
- 输出前 50 行与后 20 行数据预览

## 密度建模文档

### `docs/project_pipeline.md`

文档内容包括：

- 从原始数据到拟合结果的完整处理链路
- 各脚本输入输出之间的依赖关系
- 主流程与检查脚本的分工说明
- 推荐执行顺序的流程化解释

### `docs/environment_setup.md`

文档内容包括：

- 推荐 Python 版本与核心依赖列表
- `venv` 和 `conda` 两种安装方式
- 数据准备要求、执行顺序和常见运行问题
- 大文件、磁盘空间和性能方面的注意事项

### `docs/greenshields_speed_density_scheme.md`

文档内容包括：

- 速度等级到自由流速度和临界密度的标准化映射
- 为什么 `P99.5` 只用于校准档位而不直接作为 `v_f`
- Greenshields 密度计算公式与超速截断规则
- 车辆数、流量与 15 分钟流量的派生计算方法

### `docs/function_cluster_visualization.md`

文档内容包括：

- `visualize_fitted_function_clusters.py` 的使用前提和输入 parquet 说明
- 各图的定位区分：主图、技术检查图、分位带图、代表节点图、诊断图
- 关键参数、默认值与推荐使用方式
- `function_cluster_visualization/` 输出目录结构说明

### `docs/parameter_files.md`

文档内容包括：

- `data/params/` 目录下参数文件的角色划分
- `speed_class_density_params.csv` 的字段定义与脚本使用位置
- `beijing_capacity_params.csv` 的当前定位与后续扩展建议
- 参数表与统计产物之间的关系说明

### `docs/node_intersection_flow_inspection.md`

文档内容包括：

- 节点流量分块的字段结构与分片规则
- 61 个日文件对应的全局时间段编号规则
- 输出排序规则与检查结论
- 相关生成脚本与检查脚本入口

### `docs/date_type_curve_method_comparison.md`

文档内容包括：

- `compare_date_type_curve_methods.py` 的实验目的和输入数据
- `M0 / M1 / M2 / M3` 四种日期类型处理方法的定义
- 拟合、特征提取、聚类和 `best_k` 选择流程
- `data/analysis/date_type_curve_method_comparison/` 输出结构说明

### `docs/node_flow_daily_curve_fit.md`

文档内容包括：

- 节点日内平均流量曲线的构造方式
- 傅里叶基函数拟合模型与最小二乘求解方法
- 不同傅里叶阶数的比较脚本与默认推荐阶数
- 两个 parquet 输出文件的字段定义
- 可视化脚本生成的拟合质量图和样本节点曲线图说明

## 当前产出文件

目前已生成以下可直接使用的处理结果：

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`

交通速度数据由于体量非常大，分块结果保存在：

- `data/processed/speed_data_chunks/`

当前新增的分析结果包括：

- `data/params/beijing_capacity_params.csv`
- `data/params/speed_class_density_params.csv`
- `data/analysis/speed_histogram_counts_by_period_by_class.csv`
- `data/analysis/speed_histograms_by_class_p995.csv`
- `data/analysis/speed_histograms_by_class_with_p995_percent.png`
- `data/analysis/speed_histograms_by_period_by_class/`
- `data/analysis/density_metrics_chunks/`
- `data/analysis/node_intersection_flow_parquet/`
- `data/analysis/node_intersection_flow_check_reports/`
- `data/analysis/node_flow_curve_fit/`
- `data/analysis/date_type_curve_method_comparison/`
- `data/analysis/date_type_curve_method_comparison/function_cluster_visualization/`

用于数据核查的脚本位于：

- `dataset_inspection_scripts/inspect_speed_data_chunks.py`
- `dataset_inspection_scripts/inspect_density_metrics_chunks.py`
- `dataset_inspection_scripts/check_density_time_order.py`
- `dataset_inspection_scripts/inspect_node_intersection_flow.py`
- `analysis_scripts/check_spatial_node_completeness.py`
- `dataset_inspection_scripts/inspect_road_directionality.py`
- `analysis_scripts/visualize_fitted_function_clusters.py`

## 大文件说明

仓库中的交通速度原始数据与分块合并结果体量较大：

- `data/raw/traffic_speed_sub-dataset.v2` 约 `7.9 GB`
- `data/processed/speed_data_chunks/` 总体约 `28.5 GB`

这两部分内容不适合直接纳入普通 GitHub 仓库版本管理，因此当前已在 `.gitignore` 中忽略。仓库将保留：

- 预处理脚本
- 轻量级处理结果
- 项目说明文档

如需复现完整数据处理流程，可在本地准备原始数据后执行脚本重新生成。

## 使用方式

建议按以下顺序执行：

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
python dataset_inspection_scripts/inspect_speed_data_chunks.py
python dataset_inspection_scripts/inspect_density_metrics_chunks.py
python dataset_inspection_scripts/check_density_time_order.py
python dataset_inspection_scripts/inspect_node_intersection_flow.py
python dataset_inspection_scripts/inspect_road_directionality.py
```

当前节点流量完整性检查结果显示：

- `246133536` 条节点-时间段观测完整覆盖 `42031` 个节点、`61` 天、每日 `96` 个时段
- 缺失记录数为 `0`，因此后续 `M0 / M1 / M2 / M3` 曲线拟合与聚类直接使用原始 `node_intersection_flow_parquet`
- 当前流程不再推荐执行空间均值填补，也不需要生成额外的 `daily_parquet` 副本

推荐检查命令：

```bash
python analysis_scripts/check_spatial_node_completeness.py
```

仅当发现异常且需要导出具体缺失点时，再使用：

```bash
python analysis_scripts/check_spatial_node_completeness.py --write-missing-detail
```

建议环境依赖至少包括：

- `polars`
- `matplotlib`
- `seaborn`

## 下一步建议

- 补充 `requirements.txt` 或环境安装说明，固定分析与预处理依赖版本
- 增加分析图表样例说明，帮助快速理解各速度等级的分布特征
- 补充联邦学习任务定义、客户端划分方案与评估指标
- 增加训练入口脚本与实验配置说明
