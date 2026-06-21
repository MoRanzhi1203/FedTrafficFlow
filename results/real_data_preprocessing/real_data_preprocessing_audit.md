# Real Data Preprocessing Audit / 真实数据预处理审计

## 1. Audit Scope / 审计范围

- 中文说明：本报告聚焦真实交通数据预处理链路本身，不涉及重训练或仿真实验重跑；主要目的是盘点真实数据主线当前已具备的处理能力与仍缺失的预测准备环节。

- Task focus: real traffic data preprocessing only; no retraining and no simulation rerun.
- Existing scan directories: data, data/raw, data/processed, data/analysis
- Missing scan directories: data/real, datasets

## 2. File Inventory Summary / 文件盘点摘要

- 中文说明：下列统计给出本轮扫描中候选文件、可读表格文件以及包含时间、交通指标、节点字段文件的总体数量。

- Candidate data files: 252
- Readable tabular files: 21
- Files with read errors: 231
- Files with time columns: 5
- Files with traffic metric columns: 15
- Files with node or sensor columns: 12

## 3. Identified Pipeline / 已识别处理链路

- 中文说明：本节逐项审查真实数据链路中是否已经具备原始读取、时间处理、缺失值处理、异常处理、指标构造、滑动窗口、归一化、数据集划分、客户端划分与图结构构造等能力。

### Raw Data Reading / 原始数据读取

- Found: Yes
- Evidence files: preprocessing_scripts/process_link_gps.py, preprocessing_scripts/process_rnsd.py, preprocessing_scripts/merge_speed_data.py, docs/project_pipeline.md
- Note: 项目当前将 `link_gps.v2`、`road_network_sub-dataset.v2` 与 `traffic_speed_sub-dataset.v2` 作为真实数据链路输入，并通过预处理脚本按表结构读取。

### Timestamp Processing / 时间戳处理

- Found: Yes
- Evidence files: preprocessing_scripts/merge_speed_data.py, analysis_scripts/compute_node_intersection_flow_optimized.py, analysis_scripts/check_spatial_node_completeness.py
- Note: 已发现按 `时间段` 获取最小/最大范围、分块处理、按 `[时间段, 路段ID]` 或 `[时间段, 节点ID]` 排序，以及按每日 96 个时段检查连续性的实现。

### Missing Value Handling / 缺失值处理

- Found: Yes
- Evidence files: preprocessing_scripts/process_link_gps.py, preprocessing_scripts/process_rnsd.py, analysis_scripts/compute_node_intersection_flow_optimized.py, analysis_scripts/check_spatial_node_completeness.py
- Note: 已发现去除关键字段空值、节点流量聚合后的 `fillna(0)`、以及完整性检查脚本对缺失记录的统计与报告。

### Anomaly Handling / 异常值处理

- Found: Yes
- Evidence files: analysis_scripts/compute_node_intersection_flow_optimized.py, analysis_scripts/check_spatial_node_completeness.py
- Note: 已发现将负车流量视为非法并删除的逻辑，以及对 `null / NaN / 负值` 的完整性检查；但未发现更细化的异常值裁剪或鲁棒插值策略。

### Traffic Metric Selection / 交通指标选择

- Found: Yes
- Evidence files: preprocessing_scripts/merge_speed_data.py, analysis_scripts/compute_greenshields_density.py, analysis_scripts/compute_node_intersection_flow_optimized.py
- Note: 当前真实数据链路以速度观测为起点，后续构造 `flow_q_hour`、`路口进入流量`、`路口离开流量` 与 `路口车流量` 等指标。

### Time Window Construction / 滑动窗口构造

- Found: No
- Note: 当前项目中暂未发现面向真实数据预测任务的明确滑动窗口、输入长度和预测步长实现，需要后续确认。

### Normalization / 归一化

- Found: No
- Note: 当前项目中未发现针对真实数据预测样本的 `MinMaxScaler` / `StandardScaler` 归一化与保存逻辑。现有归一化主要出现在曲线形态比较与聚类分析阶段，不等同于预测数据预处理。

### Train Val Test Split / 训练验证测试划分

- Found: No
- Note: 当前项目中暂未发现真实数据训练/验证/测试划分脚本或划分统计文件，需要后续确认。

### Client Partition / 客户端划分

- Found: No
- Note: 当前扫描未发现真实数据实验中明确的联邦客户端划分规则或客户端样本文件。

### Graph Construction / 图结构构造

- Found: Yes
- Evidence files: preprocessing_scripts/process_rnsd.py, analysis_scripts/compute_node_intersection_flow_optimized.py, analysis_scripts/check_spatial_node_completeness.py
- Note: 已发现基于 `起始节点ID/结束节点ID` 的路网拓扑映射与节点集合检查，但未发现面向真实数据预测模型的独立邻接矩阵输出文件。

### Preprocessed Outputs / 已产出预处理结果

- Found: Yes
- Evidence files: data/processed/link_gps_processed.csv, data/processed/rnsd_processed.csv, data/analysis/node_intersection_flow_check_reports/completeness_summary.csv
- Note: 已识别到坐标清洗结果、路网属性清洗结果，以及节点完整性统计结果等可直接引用的输出产物。

### Preprocessing Result Statistics / 预处理统计结果

- Found: Yes
- Evidence files: data/analysis/node_intersection_flow_check_reports/completeness_summary.csv
- Note: 已扫描到节点流量完整性检查汇总表，可用于引用节点数量、观测覆盖、缺失/重复/非法记录等统计。

## 4. Code File Inventory / 代码文件盘点

- 中文说明：下表从脚本职责与功能覆盖角度总结当前真实数据主线各文件在缺失处理、归一化、滑动窗口、图结构、客户端划分和数据划分方面的覆盖情况。

| File | Role | Missing | Normalization | Window | Graph | Client | Split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_scripts/add_p995_to_speed_histogram.py | add_p995_to_speed_histogram.py | Y | N | Y | N | N | Y |
| analysis_scripts/check_spatial_node_completeness.py | 检查道路交通节点时空数据的完整性。 | Y | N | Y | Y | N | Y |
| analysis_scripts/compare_date_type_curve_methods.py | 比较不同日期类型曲线构造方法的傅里叶拟合与聚类效果。 | N | Y | Y | Y | N | Y |
| analysis_scripts/compare_node_flow_fourier_orders.py | 比较不同傅里叶阶数对路口节点日内平均车流量曲线的拟合效果。 | N | N | N | Y | N | Y |
| analysis_scripts/compute_greenshields_density.py | compute_greenshields_density.py | Y | N | Y | N | N | Y |
| analysis_scripts/compute_node_intersection_flow_optimized.py | 将路段车流量聚合为路口节点车流量 | Y | N | N | Y | N | Y |
| analysis_scripts/fit_node_flow_daily_curve.py | 对每个路口节点的日内平均车流量曲线进行傅里叶拟合。 | Y | N | N | Y | N | Y |
| analysis_scripts/summarize_speed_stats.py | summarize_speed_stats.py | N | N | N | N | N | N |
| analysis_scripts/visualize_fitted_function_clusters.py | 基于已输出的拟合曲线与聚类标签，直接展示函数曲线层面的聚类结果。 | Y | Y | N | Y | N | Y |
| analysis_scripts/visualize_node_flow_daily_curve_fit.py | 可视化路口节点日内平均车流量曲线及其傅里叶拟合结果。 | Y | N | Y | Y | N | Y |
| analysis_scripts/visualize_speed_hist_by_period.py | visualize_speed_hist_by_period.py | N | N | Y | N | N | Y |
| dataset_inspection_scripts/check_density_time_order.py | 检查密度指标分片文件中的时间段字段是否按预期顺序排列。 | N | N | Y | N | N | Y |
| dataset_inspection_scripts/inspect_density_metrics_chunks.py | 检查密度指标 Parquet 分片的结构、样例值和基础统计信息。 | Y | N | N | N | N | Y |
| dataset_inspection_scripts/inspect_node_intersection_flow.py | 检查路口流量 Parquet 分片的结构、全局时间段覆盖和排序情况。 | Y | N | N | Y | N | Y |
| dataset_inspection_scripts/inspect_road_directionality.py | 分析当前路网是否为有向路段，以及是否区分进入路口和离开路口。 | Y | N | N | Y | N | Y |
| dataset_inspection_scripts/inspect_speed_data_chunks.py | 检查速度数据 Parquet 分片的结构、样例值和基础统计信息。 | Y | N | N | N | N | Y |
| docs/environment_setup.md | 环境安装与运行说明 | N | Y | Y | Y | Y | Y |
| docs/node_flow_daily_curve_fit.md | 路口节点日内车流量傅里叶拟合说明 | Y | N | Y | Y | N | N |
| docs/node_intersection_flow_inspection.md | 路口流量分片检查与使用说明 | Y | N | N | Y | N | N |
| docs/project_documentation.md | FedTrafficFlow 项目文档 | N | Y | Y | Y | Y | Y |
| docs/project_pipeline.md | 项目处理流程总览 | Y | Y | Y | Y | Y | Y |
| preprocessing_scripts/merge_speed_data.py | 将原始速度记录与路段静态属性合并，并按分片写出处理后的结果。 | N | N | N | N | N | Y |
| preprocessing_scripts/process_link_gps.py | process_link_gps.py | Y | Y | N | N | N | N |
| preprocessing_scripts/process_rnsd.py | process_rnsd.py | Y | N | N | Y | N | N |

## 5. Paper-Usable Evidence / 可直接用于论文的证据

- 中文说明：本节总结哪些证据已经可以直接支撑论文叙述，哪些部分仍需要后续补充实现或审计证明。

- 共识别到 252 个候选真实数据/分析文件，其中原始文件 3 个、处理后文件 63 个。
- 包含时间列的文件数量为 5，包含交通指标字段的文件数量为 15。
- 包含节点/传感器字段的文件数量为 12。
- 已检测到节点完整性汇总表，可直接支撑论文中关于覆盖完整性、缺失记录和非法值检查的描述。
- 当前未发现真实数据训练/验证/测试划分证据，正式论文中需要补充。
- 当前未发现真实数据联邦客户端划分证据，正式论文中需要补充。
- 当前未发现真实数据预测样本的归一化与 scaler 保存证据，正式论文中需要补充。

## 6. Open Questions / 仍待补充的问题

- 中文说明：以下问题代表当前真实数据预测主线尚未完全闭环的关键缺口，需要在后续联邦训练规范或实验实现中继续补齐。

- 真实数据来源、采集区域和正式引用信息仍需在论文中补充明确。
- 当前未发现真实数据预测任务的滑动窗口、输入长度和预测步长设置。
- 当前未发现真实数据训练/验证/测试划分比例或样本量统计。
- 当前未发现真实数据联邦客户端划分规则与客户端样本量统计。
- 当前未发现真实数据预测样本的归一化/scaler 保存文件。
- 当前仅发现基于路网起止节点的拓扑映射，尚未发现独立保存的真实数据邻接矩阵文件。
