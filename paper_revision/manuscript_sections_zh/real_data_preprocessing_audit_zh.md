# 真实数据预处理审计报告

> 自动生成时间：2026-06-10  
> 审计范围：仅限真实交通流量数据预处理管线，不涉及模型再训练或仿真重跑。  
> 审计脚本：`analysis_scripts/audit_real_data_preprocessing.py`  
> 输出目录：`results/real_data_preprocessing/`

---

## 1. 审计范围

本次审计覆盖以下内容：

- **原始数据文件**：`link_gps.v2`、`road_network_sub-dataset.v2`（RNSD 路网）、`traffic_speed_sub-dataset.v2`（速度记录）
- **预处理脚本**：`preprocessing_scripts/` 下的全部脚本
- **分析/检查脚本**：`analysis_scripts/` 和 `dataset_inspection_scripts/` 下的全部脚本
- **处理后数据**：`data/processed/` 下的所有输出
- **分析结果**：`data/analysis/` 下的完整性报告、速度统计、密度计算、曲线拟合等

---

## 2. 数据存量清单

| 指标 | 数值 |
|------|------|
| 候选数据文件总数 | 252 |
| 可读取的表格文件 | 21 |
| 读取失败文件 | 231（绝大部分为 Parquet 分片，需 Polars 读取） |
| 含时间列的文件 | 5 |
| 含交通指标列的文件 | 15 |
| 含节点/传感器列的文件 | 12 |
| 代码文件 | 24 |

### 2.1 核心原始数据

| 文件名 | 格式 | 行数 | 列数 | 关键字段 |
|--------|------|------|------|----------|
| `link_gps.v2` | TSV | — | 3 | `路段ID`、`经度`、`纬度` |
| `road_network_sub-dataset.v2` (RNSD) | TSV | — | 13 | `road_id`、`snodeid`、`enodeid`、`length`、`direction` |
| `traffic_speed_sub-dataset.v2` | CSV | — | 6 | `link_id`、`time`、`speed`、`volume`、`occupancy` |

### 2.2 处理后中间文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `link_gps_processed.csv` | CSV | 去重、去缺失后的路段 GPS 坐标 |
| `rnsd_processed.csv` | CSV | 去重、去缺失、含起止节点坐标的完整路网 |
| `speed_data_chunks/*.parquet` | Parquet × 61 | 按天分片的合并速度数据（路段属性 + 原始速度记录） |
| `node_intersection_flow/*.parquet` | Parquet × 61 | 按天分片的节点交汇流量（由路段级聚合为节点级） |

---

## 3. 预处理管线逐步审计

### 3.1 原始数据读取

**证据文件**：
- `preprocessing_scripts/process_link_gps.py` — 读取 `link_gps.v2`（行 21）
- `preprocessing_scripts/process_rnsd.py` — 读取 RNSD（行 22）
- `preprocessing_scripts/merge_speed_data.py` — 读取速度数据并与路段属性合并（行 92–153）
- `docs/project_pipeline.md` — 管线总览

**结论**：项目明确使用上述三个原始文件作为真实数据链路输入，通过预处理脚本按表结构读取。

### 3.2 时间戳处理

**证据文件**：
- `preprocessing_scripts/merge_speed_data.py` — 将 `时间` 字段解析为时间戳，并按时间片分块输出
- `analysis_scripts/compute_node_intersection_flow_optimized.py` — 按 `[时间片, 节点ID]` 聚合排序
- `analysis_scripts/check_spatial_node_completeness.py` — 按每日 96 个时间片检查连续性

**结论**：已发现按 `时间片` 获取最小/最大范围、分块处理、按 `[时间片, 路段ID]` 或 `[时间片, 节点ID]` 排序，以及按每日 96 个时间片检查连续性的实现。

### 3.3 缺失值处理

**证据文件**：
- `preprocessing_scripts/process_link_gps.py` — 删除关键字段空值行（行 23–25）
- `preprocessing_scripts/process_rnsd.py` — 删除关键字段空值行（行 37–39）
- `analysis_scripts/compute_node_intersection_flow_optimized.py` — 聚合后 `fillna(0)`
- `analysis_scripts/check_spatial_node_completeness.py` — 完整性与缺失记录统计

**结论**：已发现去除关键字段空值、节点流量聚合后 `fillna(0)`、以及完整性检查脚本对缺失记录的统计与报告。

### 3.4 异常值处理

**证据文件**：
- `analysis_scripts/compute_node_intersection_flow_optimized.py` — 将负车流量视为非法并删除
- `analysis_scripts/check_spatial_node_completeness.py` — 对 `null / NaN / 负值` 的完整性检查

**结论**：已发现将负车流量视为非法并删除的逻辑，以及对 `null / NaN / 负值` 的完整性检查；但未发现更细化的异常值裁剪或鲁棒插值策略。

### 3.5 交通指标选择

**证据文件**：
- `preprocessing_scripts/merge_speed_data.py` — 速度、流量、占用率三指标
- `analysis_scripts/compute_greenshields_density.py` — 利用 Greenshields 模型由速度推导密度
- `analysis_scripts/compute_node_intersection_flow_optimized.py` — 节点流量为核心预测目标

**结论**：数据链路同时处理 speed（速度）、volume（流量）、occupancy（占用率），并通过 Greenshields 模型推导 density（密度），形成多指标体系。

### 3.6 时间窗构建

**结论**：当前代码中未发现真实数据的滑动窗口构建逻辑（输入序列长度、预测步长、样本生成方式）。此部分在仿真实验中存在，但真实数据链路缺失。**正式论文中需要补充。**

### 3.7 归一化

**结论**：当前未发现真实数据预测样本的归一化/Scaler 保存文件。仿真实验中有归一化逻辑，但未应用于真实数据。**正式论文中需要补充。**

### 3.8 训练/验证/测试划分

**结论**：当前未发现真实数据的训练/验证/测试划分证据（划分比例、样本量统计）。**正式论文中需要补充。**

### 3.9 联邦客户端划分

**结论**：当前未发现真实数据的联邦客户端划分规则与客户端样本量统计。仿真实验中有客户端划分逻辑（合成数据，独立于真实数据链路）。**正式论文中需要补充。**

### 3.10 图结构构建

**证据文件**：
- `preprocessing_scripts/process_rnsd.py` — 基于 RNSD 路网的起止节点拓扑推导（行 47）
- `analysis_scripts/compute_node_intersection_flow_optimized.py` — 路段到节点的映射聚合
- `analysis_scripts/check_spatial_node_completeness.py` — 拓扑节点与观测节点的交叉验证

**结论**：已发现基于路网起止节点的拓扑映射（42,031 个拓扑节点与观测节点完全匹配，0 个不一致），但尚未发现独立保存的邻接矩阵文件。**正式论文中需要补充。**

### 3.11 预处理输出

**输出文件**：
- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`
- `data/processed/speed_data_chunks/*.parquet`（61 天）
- `data/analysis/node_intersection_flow/*.parquet`（61 天）

### 3.12 预处理结果统计

**核心统计（来自 `completeness_summary.csv`）**：

| 指标 | 数值 |
|------|------|
| 数据文件数 | 61 |
| 覆盖天数 | 61 |
| 观测节点数 | 42,031 |
| 拓扑节点数 | 42,031 |
| 观测节点不在拓扑中 | 0 |
| 拓扑节点不在观测中 | 0 |
| 原始记录总数 | 246,133,536 |
| 期望记录总数 | 246,133,536 |
| 缺失记录数 | 0 |
| 重复记录数 | 0 |
| 空值流量数 | 0 |
| NaN 流量数 | 0 |
| 负流量数 | 0 |
| 完整性检查通过 | 是 |

**速度统计（来自 `speed_class_overall_stats.csv`）**：

| 速度等级 | 最小平均速度 (km/h) | 最大平均速度 (km/h) | 平均速度 (km/h) | 记录数 |
|----------|---------------------|---------------------|-----------------|--------|
| 2 | 3.12 | 104.66 | 67.90 | 351,360 |
| 3 | 2.01 | 107.49 | 72.59 | 907,680 |
| 4 | 2.01 | 120.00 | 52.20 | 14,663,424 |
| 5 | 2.00 | 119.93 | 35.13 | 113,009,088 |
| 6 | 2.00 | 119.44 | 31.74 | 74,868,960 |
| 7 | 2.00 | 119.11 | 25.64 | 60,539,328 |
| 8 | 3.24 | 97.53 | 22.76 | 46,848 |

> 速度等级由 Greenshields 模型 `compute_greenshields_density.py` 推导，等级越高表示拥堵越严重。

---

## 4. 论文可用证据

审计脚本已自动生成以下论文可用证据：

1. 共识别到 252 个候选真实数据/分析文件，其中原始文件 3 个、处理后文件 63 个。
2. 包含时间列的文件数量为 5，包含交通指标字段的文件数量为 15。
3. 包含节点/传感器字段的文件数量为 12。
4. 已检测到节点完整性汇总表，可直接支撑论文中关于覆盖完整性、缺失记录和非法值检查的描述。
5. 当前未发现真实数据训练/验证/测试划分证据，正式论文中需要补充。
6. 当前未发现真实数据联邦客户端划分证据，正式论文中需要补充。
7. 当前未发现真实数据预测样本的归一化与 scaler 保存证据，正式论文中需要补充。

---

## 5. 待解决问题

以下问题需在正式论文撰写前澄清或补充：

1. **数据来源**：真实数据来源、采集区域和正式引用信息仍需在论文中补充明确。
2. **滑动窗口**：当前未发现真实数据预测任务的滑动窗口、输入长度和预测步长设置。
3. **数据划分**：当前未发现真实数据训练/验证/测试划分比例或样本量统计。
4. **联邦划分**：当前未发现真实数据联邦客户端划分规则与客户端样本量统计。
5. **归一化/Scaler**：当前未发现真实数据预测样本的归一化/scaler 保存文件。
6. **邻接矩阵**：当前仅发现基于路网起止节点的拓扑映射，尚未发现独立保存的真实数据邻接矩阵文件。

---

## 6. 辅助数据与参考

### 曲线拟合（节点流量日变化）

- 对每个节点的每日 96 时间片流量使用 Fourier 基函数拟合
- 拟合指标：中位数 R^2 约 0.93，平均 R^2 约 0.90，平均 RMSE 约 37.64
- 低 R^2 节点比例：约 20.8%
- 时序聚类：基于拟合系数的 PCA + K-Means 聚类（k = 3–6），轮廓系数 0.15–0.50

### 速度-密度关系（Greenshields 模型）

- 自由流速度 v_f 约 120 km/h
- 拥堵密度 k_j 由等级 8 最大密度外推
- 速度等级 2–8，等级越高拥堵越严重

### 空间完整性

- 42,031 个拓扑节点全部在观测数据中出现
- 观测节点全部在拓扑中存在
- 零不一致节点

---

## 7. 输出文件索引

| 文件 | 路径 |
|------|------|
| 文件清单 | `results/real_data_preprocessing/real_data_file_inventory.csv` |
| 质量摘要 | `results/real_data_preprocessing/real_data_quality_summary.csv` |
| 审计 JSON | `results/real_data_preprocessing/real_data_preprocessing_audit.json` |
| 审计 Markdown | `results/real_data_preprocessing/real_data_preprocessing_audit.md` |
| 完整性汇总 | `data/analysis/node_intersection_flow_check_reports/completeness_summary.csv` |
| 速度等级统计 | `data/analysis/speed_class_overall_stats.csv` |

---

*审计报告结束*
