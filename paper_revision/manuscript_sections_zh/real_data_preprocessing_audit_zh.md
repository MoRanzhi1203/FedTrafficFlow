# 真实数据预处理扫描审查报告

## 1. 扫描目标

本次扫描聚焦真实交通数据预处理链路，目标是梳理当前项目中与真实数据来源、清洗、缺失处理、异常处理、时间组织、空间节点处理、归一化、数据划分、联邦客户端划分以及预处理结果统计有关的可复核证据，为后续论文写作提供依据。扫描过程仅审查现有代码、数据与文档，不涉及模型重训练，也不涉及仿真实验重跑。

## 2. 扫描范围

本次实际扫描到的目录包括：

- `data/`
- `data/raw/`
- `data/processed/`
- `data/analysis/`
- `preprocessing_scripts/`
- `analysis_scripts/`
- `dataset_inspection_scripts/`
- `docs/`
- `paper_revision/manuscript_sections_zh/`

本次未发现但已按要求跳过的目录包括：

- `data/real/`
- `datasets/`
- `notebooks/`
- `scripts/`
- `preprocessing/`
- `src/`
- `models/`

扫描涉及的文件类型主要包括 `py`、`md`、`csv`、`json`、`txt`、`xlsx`、`npy`、`npz`、`pkl`，以及项目原始真实数据使用的 `v2` 文件。

## 3. 真实数据文件清单

根据 `results/real_data_preprocessing/real_data_file_inventory.csv`，当前项目共识别到 252 个候选真实数据与分析结果文件，其中原始数据文件 3 个。与真实数据预处理直接相关的核心文件如下。

| 文件路径 | 文件类型 | 文件大小 | 行数/列数 | 主要字段 |
| --- | --- | --- | --- | --- |
| `data/raw/link_gps.v2` | `.v2` | 1,433,051 B | 45,148 / 3 | `路段ID`、`经度`、`纬度` |
| `data/raw/road_network_sub-dataset.v2` | `.v2` | 2,170,603 B | 45,148 / 8 | `link_id`、`width`、`direction`、`snodeid`、`enodeid`、`length`、`speedclass`、`lanenum` |
| `data/raw/traffic_speed_sub-dataset.v2` | `.v2` | 8,293,882,527 B | 全量行数本次未统计 / 3 | 由现有读取脚本赋名为 `路段ID`、`时间段`、`平均速度` |
| `data/processed/link_gps_processed.csv` | `.csv` | 1,478,223 B | 45,148 / 3 | `路段ID`、`经度`、`纬度` |
| `data/processed/rnsd_processed.csv` | `.csv` | 3,386,375 B | 45,148 / 12 | `路段ID`、`起始节点ID`、`结束节点ID`、`长度`、`速度等级`、`车道数`、`start_lat`、`start_lon`、`end_lat`、`end_lon` 等 |
| `data/analysis/node_intersection_flow_check_reports/completeness_summary.csv` | `.csv` | 393 B | 1 / 15 | `file_count`、`observed_node_count`、`topology_node_count`、`total_missing_records`、`total_duplicate_records`、`total_negative_flow_count` 等 |
| `data/analysis/node_intersection_flow_check_reports/daily_completeness_summary.csv` | `.csv` | 9,781 B | 61 / 24 | `day_index`、`expected_start_time`、`actual_min_time`、`unique_time_count`、`missing_records`、`zero_flow_count`、`mean_flow` 等 |

需要说明的是，`traffic_speed_sub-dataset.v2` 体积较大，本次审查脚本未对其执行全量行数统计，而是基于前 5000 行完成结构与基础质量抽样；因此正式论文中若需报告其总记录数，仍需结合原始文件或额外离线统计结果补充。

## 4. 数据质量统计

根据 `results/real_data_preprocessing/real_data_quality_summary.csv` 以及现有完整性检查汇总文件，可以得到以下结果。

- `link_gps.v2` 与 `link_gps_processed.csv` 均为 45,148 行、3 列，关键字段未见缺失。
- `road_network_sub-dataset.v2` 与 `rnsd_processed.csv` 均为 45,148 行，其中清洗后表扩展为 12 列，保留了路段、宽度、方向、起止节点、长度、速度等级、车道数以及推导的起止经纬度。
- 对 `traffic_speed_sub-dataset.v2` 的 5000 行抽样显示，`时间段` 范围为 `0-4999`，`平均速度` 范围约为 `9.74-62.45`，均值约为 `36.77`，抽样中未见缺失值与负值。
- `completeness_summary.csv` 显示，节点流量分片文件共 61 个，与期望天数一致；观测节点数与拓扑节点数均为 42,031，`observed_not_in_topology_count` 与 `topology_not_in_observed_count` 均为 0。
- 同一汇总表显示，总原始记录数、总期望记录数与总唯一观测对数均为 246,133,536，`total_missing_records`、`total_duplicate_records`、`total_null_flow_count`、`total_nan_flow_count` 与 `total_negative_flow_count` 均为 0。
- `daily_completeness_summary.csv` 显示，每个日分片均包含 96 个时间段，每日记录数固定为 4,034,976，未发现缺失时间段、重复记录或非法流量。
- 按日统计的 `mean_flow` 约在 `2376.59-2459.67` 之间，`max_flow` 约在 `10260.16-10477.75` 之间；每日 `zero_flow_count` 约在 `18406-35994` 之间，说明零流量记录存在，但并未导致缺失、重复或非法值问题。

综合现有统计结果，可确认项目已经对节点级真实流量分片完成较充分的完整性检查，但尚未形成关于原始速度观测全量缺失率与全量异常率的独立报告。

## 5. 预处理代码清单

与真实数据预处理直接相关的核心代码文件如下。

| 文件路径 | 主要作用 | 缺失处理 | 归一化 | 窗口构造 | 图结构/拓扑 | 客户端划分 |
| --- | --- | --- | --- | --- | --- | --- |
| `preprocessing_scripts/process_link_gps.py` | 读取 `link_gps.v2`，去重、去空、类型统一，输出标准化路段坐标 | 是 | 否 | 否 | 否 | 否 |
| `preprocessing_scripts/process_rnsd.py` | 清洗路网属性表，重命名字段，删除重复/缺失，推导起止节点坐标 | 是 | 否 | 否 | 是 | 否 |
| `preprocessing_scripts/merge_speed_data.py` | 读取原始速度观测，关联路段静态属性，按时间段区间输出分块结果 | 否 | 否 | 否 | 否 | 否 |
| `analysis_scripts/compute_greenshields_density.py` | 基于速度与参数表推导密度、小时流率与交通量等派生指标 | 未见明确缺失处理主逻辑 | 否 | 否 | 否 | 否 |
| `analysis_scripts/compute_node_intersection_flow_optimized.py` | 将路段流量聚合为节点进入/离开/综合流量，删除负值并对聚合空值填 0 | 是 | 否 | 否 | 是 | 否 |
| `analysis_scripts/check_spatial_node_completeness.py` | 检查节点-时间网格的缺失、重复、非法值及拓扑一致性 | 是 | 否 | 否 | 是 | 否 |
| `dataset_inspection_scripts/check_density_time_order.py` | 检查密度分片时间顺序 | 否 | 否 | 否 | 否 | 否 |
| `dataset_inspection_scripts/inspect_node_intersection_flow.py` | 检查节点流量分片结构、时间覆盖与排序 | 是 | 否 | 否 | 是 | 否 |
| `docs/project_pipeline.md` | 汇总真实数据主流程的输入、输出和执行顺序说明 | 是 | 仅提到分析阶段归一化曲线，不对应预测预处理 | 否 | 是 | 否 |

当前扫描未发现与真实数据预测任务直接对应的窗口构造、训练/验证/测试划分、客户端划分和 scaler 保存代码文件。

## 6. 已识别的预处理流程

### 1. 原始数据读取

已发现明确实现。项目当前将 `link_gps.v2`、`road_network_sub-dataset.v2` 与 `traffic_speed_sub-dataset.v2` 作为真实数据输入文件，分别由 `process_link_gps.py`、`process_rnsd.py` 和 `merge_speed_data.py` 读取。

### 2. 时间戳处理

已发现明确实现。当前项目主要使用整数型 `时间段` 索引组织时序数据，`merge_speed_data.py` 会先获取最小/最大时间段，再按时间段区间分块输出，并在输出前按 `[时间段, 路段ID]` 排序；`compute_node_intersection_flow_optimized.py` 则按 `[时间段, 节点ID]` 排序；`check_spatial_node_completeness.py` 进一步以每日 96 个时段检查完整性。

### 3. 缺失值处理

已发现明确实现。`process_link_gps.py` 与 `process_rnsd.py` 会删除关键字段缺失记录；`compute_node_intersection_flow_optimized.py` 在聚合后对进入流量和离开流量缺失值执行 `fillna(0)`；`check_spatial_node_completeness.py` 会统计缺失记录并输出检查报告。

### 4. 异常值处理

已发现部分实现。`compute_node_intersection_flow_optimized.py` 明确删除 `flow_q_hour < 0` 的记录，并将其视为物理上不可能的异常值；`check_spatial_node_completeness.py` 也会检查 `null / NaN / 负值`。但当前项目中暂未发现更细化的异常值裁剪、平滑或鲁棒插值逻辑，需要后续确认。

### 5. 交通指标选择

已发现明确实现。当前链路以速度观测为起点，经 `compute_greenshields_density.py` 生成密度与流量相关指标，再由 `compute_node_intersection_flow_optimized.py` 构造 `路口进入流量`、`路口离开流量` 与 `路口车流量`。

### 6. 时间窗口构造

当前项目中暂未发现明确实现，需要后续确认。现有真实数据链路主要形成节点级时序分片、日内平均曲线与傅里叶拟合结果，但未扫描到面向预测任务的滑动窗口、输入长度或预测步长设置。

### 7. 归一化 / 标准化

当前项目中暂未发现明确实现，需要后续确认。现有代码中的归一化主要用于日期类型曲线比较与聚类分析阶段，不能直接视为真实数据预测输入的归一化预处理，也未发现 scaler 参数保存文件。

### 8. 训练/验证/测试划分

当前项目中暂未发现明确实现，需要后续确认。未扫描到真实数据预测任务的训练集、验证集、测试集划分比例或样本量统计。

### 9. 客户端划分

当前项目中暂未发现明确实现，需要后续确认。未发现真实数据实验中明确的联邦客户端划分规则、客户端数量说明或客户端级样本文件。

### 10. 图结构构造

已发现部分实现。`rnsd_processed.csv` 保存了 `起始节点ID` 与 `结束节点ID`，相关脚本也基于这些字段完成拓扑映射和节点集合一致性检查，因此可以确认项目已形成真实路网拓扑基础。当前项目中未发现与真实数据预测模型直接对应的独立邻接矩阵输出文件，后续需确认真实数据实验是否显式使用空间邻接矩阵。

### 11. 预处理后文件输出

已发现明确实现。当前可确认的核心输出包括：

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`
- `data/processed/speed_data_chunks/`
- `data/analysis/density_metrics_chunks/`
- `data/analysis/node_intersection_flow_parquet/`
- `data/analysis/node_intersection_flow_check_reports/`

## 7. 可用于论文正文的证据

当前可直接进入论文正文或支撑正文写作的证据主要包括：

- 原始输入由 3 个真实数据文件组成，分别对应路段坐标、路网属性和速度观测。
- `link_gps_processed.csv` 与 `rnsd_processed.csv` 均包含 45,148 条路段记录，说明坐标表与路网表在清洗后保持一致的路段规模。
- 节点拓扑检查结果显示，观测节点数与拓扑节点数均为 42,031，且两者完全一致。
- 节点流量完整性检查覆盖 61 个日分片、每日 96 个时段，总唯一观测对数为 246,133,536。
- 当前完整性汇总表中缺失记录、重复记录、空值、NaN 和负值均为 0，可直接支撑“完整性检查通过”的表述。
- 现有代码已明确包含去重、关键字段缺失删除、负值过滤、聚合后空值补零和时序排序等处理步骤。

## 8. 仍需补充或确认的问题

- 数据来源是否明确：当前项目文件中未发现明确的数据集名称、采集区域和正式引用信息。
- 是否缺少数据集引用：是，正式论文仍需补充数据来源说明与引用文献。
- 是否缺少缺失值处理说明：完整性检查存在，但原始速度观测层面的全量缺失率统计尚不充分。
- 是否缺少异常值处理说明：仅发现负值过滤，未发现更细致的异常值裁剪或鲁棒插值说明。
- 是否缺少真实数据划分统计：是，当前未发现训练/验证/测试划分。
- 是否缺少归一化参数保存说明：是，当前未发现 scaler 文件或保存逻辑。
- 是否缺少图结构构造说明：当前已发现拓扑基础，但未发现独立邻接矩阵文件与构造参数说明。
- 是否缺少客户端划分说明：是，当前未发现真实数据联邦客户端划分证据。
- 是否缺少时间窗口构造说明：是，当前未发现明确的输入窗口长度与预测步长设置。

## 9. 结论

综合本次扫描结果，当前项目已经形成较完整的真实交通数据基础预处理链路，能够明确支撑“原始数据读取、路段与路网清洗、速度与路网属性合并、流量派生、节点级聚合及完整性检查”等内容；尤其是 61 天、42,031 个节点、246,133,536 条节点-时间观测记录的完整性统计，为论文中的真实数据质量控制提供了较强证据。

但从“真实数据预测实验可复现性”角度看，当前项目中仍缺少若干关键证据，包括真实数据来源与引用信息、面向预测任务的时间窗口构造、训练/验证/测试划分、联邦客户端划分、预测输入归一化与 scaler 保存、以及显式邻接矩阵输出。因此，真实数据预处理部分已经能够支撑论文中关于数据清洗与质量保障的叙述，但若要完整支撑真实数据联邦预测实验写作，仍需在最终论文提交前补充上述缺失信息。
