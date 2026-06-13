# 当前真实数据缺失值设置与补全实验工作流说明

## 1. 本次审计目的

本次工作是对项目内真实数据缺失值设置、缺失数据生成、补全方法、输出结果和图件的资产级审计，只做读取、扫描、清点、解析、汇总和文档生成，不继续运行正式缺失实验。

## 2. 数据来源

真实数据主来源为 `data/analysis/node_intersection_flow_parquet`。

- 文件数量：61
- 每个 chunk 是否一天：根据 `node_flow_chunk_*.parquet` 命名、metadata 行数一致以及 `historical_test/full_intersection_missingness_audit.json` 中的 `day_index` 记录，可作为按天分片的证据。
- 每个 chunk 行数：metadata 一致，首个 chunk 为 chunk_file_range=node_flow_chunk_000.parquet..node_flow_chunk_060.parquet；检测到 61 个 node_flow_chunk_*.parquet；所有 chunk metadata 行数一致: 4034976；首个 chunk 节点数=42031, 时间片数=96；metadata 行数与 节点数*时间片数 一致，可作为按天完整网格的证据；sample_chunk_na_count=0；sample_chunk_duplicate_node_time=0；sample_chunk_negative_target=0
- 节点数量：42031
- 时间片数量：96
- 目标字段：路口车流量
- 节点字段：节点ID
- 时间字段：时间段

## 3. 缺失值设置方式

### 3.1 mcar_point

代码与结果均表明 `mcar_point` 已实际运行。已找到的运行目录包括：

- `results/real_data_missingness_full_intersection_causal_history`：missing_rates=["0.05"]，methods=["forward_fill"]
- `results/real_data_missingness_full_intersection_causal_history/historical_test`：missing_rates=["0.05"]，methods=["forward_fill", "historical_linear_extrapolation", "function_curve_fit", "geo_neighbor_fill", "geo_func_hybrid", "zero_fill"]
- `results/real_data_missingness_full_intersection_causal_history/smoke_test`：missing_rates=["0.05"]，methods=["zero_fill", "forward_fill", "historical_linear_extrapolation", "geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history_hybridtest`：missing_rates=["0.05"]，methods=["geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history_hybridtest_small`：missing_rates=["0.05"]，methods=["geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history_smoketest`：missing_rates=["0.05"]，methods=["forward_fill", "historical_linear_extrapolation", "geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid"]

### 3.2 node_temporal_block

`full_intersection_missingness_pipeline.py` 代码支持 `node_temporal_block`，但本次扫描未在运行配置和结果目录中找到其正式输出证据，因此当前判断为“代码支持但未找到正式运行证据”。

## 4. 补全方法

- `zero_fill`：baseline；不使用拓扑；不使用函数曲线；不使用未来数据；已运行。
- `forward_fill`：baseline；历史因果流水线版本不使用未来数据；已运行；旧样本脚本版本存在 bfill 风险。
- `historical_linear_extrapolation`：非 baseline；历史因果；不使用拓扑；不使用函数曲线；不使用未来数据；已运行。
- `geo_neighbor_fill`：非 baseline；历史因果；使用拓扑；不使用函数曲线；历史因果流水线中不使用未来数据；已运行。
- `function_curve_fit`：非 baseline；历史因果；不使用拓扑；使用函数曲线；历史因果流水线中不使用未来数据；已运行。
- `geo_func_hybrid`：非 baseline；历史因果；使用拓扑；使用函数曲线；历史因果流水线中不使用未来数据；已运行。
- `linear_interpolation`：早期样本实验中出现；使用双向插值，存在未来信息泄露风险；不属于当前历史因果主流程。

## 5. 当前已完成的实验

### 5.1 早期样本缺失实验

已发现 `results/real_data_missingness_experiments_sample` 与 `results/real_data_missingness_experiments`。其运行配置显示：
- 数据范围为 `node_intersection_flow_parquet` 的少量文件和少量行；
- 缺失率覆盖 `0, 0.05, 0.10, 0.20, 0.30`；
- 方法为 `zero_fill`、`forward_fill`、`linear_interpolation`；
- 结果文件包括 `missingness_design_summary.csv`、`missingness_mask_summary.csv`、`imputation_quality_summary.csv` 与 RMSE 图。

### 5.2 geo/function 子实验

已发现 `results/real_data_missingness_full_intersection_causal_history_hybridtest_small`，其配置显示只跑了 `geo_func_hybrid`、`max_chunks=1`、`max_rows=200`，属于小规模混合方法调试，不是正式主实验。

### 5.3 full intersection causal historical test

已发现 `results/real_data_missingness_full_intersection_causal_history/historical_test`，其 `run_config.json`、`audit.json`、`validation.json`、`batch_processing_report.json` 提供了完整证据：
- `max_chunks=8`
- `5% MCAR`
- `seed=42`
- `7 天 warmup`，第 8 天为主评估
- `6 种方法`
- `causal_history_only=true`
- `context_days_after=0`，不使用未来数据
- 已生成 `summary / audit / validation / figures / masks / missing_datasets / imputed_datasets`

### 5.4 61 chunk 主实验目录

已发现 `results/real_data_missingness_full_intersection_causal_history` 根目录级主实验资产，但是否已完成“61 天全量 generate_missing + 全部方法 impute + summarize + validate”不能直接按目录名下结论。当前证据显示：
- 存在根目录级 `run_config.json` 与 `run_commands.txt`；
- 存在 `imputed_datasets`、`manifests` 等资产；
- 根目录运行配置只直接指向部分 `impute` 配置，且当前未找到根目录级完整 `summaries/imputation_quality_summary_exclude_warmup.csv` 与 `validation` 完成证据；
- 因此不能写成“61 chunk 全量实验已完整完成”，需要以 inventory 结果和人工复核进一步确认。

## 6. 当前可视化图件

- overall RMSE 图：11
- zoom 图：6
- delta 图：6
- 当前多张 `missing_rate_vs_*` 图来自单缺失率 `5%` 结果，存在横轴重复 5% 的表达风险，不宜直接作为最终多缺失率曲线图。

## 7. 当前结果结论

- 现有 `8 chunk、5% MCAR historical_test` 证据中，`forward_fill` 的 RMSE 最低。
- `zero_fill` 在该测试中明显最差。
- `geo_func_hybrid` 在空间/函数类方法中优于 `geo_neighbor_fill` 与 `function_curve_fit`，但仍落后于 `forward_fill` 与 `historical_linear_extrapolation`。
- 以上结论只适用于 `8 chunk、5% MCAR historical_test`，不代表完整 61 天、多缺失率、多 seed 的最终结论。

## 8. 尚未完成的内容

- 多缺失率全量实验是否完成：未找到明确完整证据，需要人工确认。
- 多 seed 是否完成：早期样本实验存在多 seed，历史因果主流程未找到多 seed 正式完成证据。
- `node_temporal_block` 是否完成：代码支持但未找到正式运行证据。
- 61 chunk 全量 impute 是否完成：未找到明确完整证据，需要人工确认。
- summarize 是否完成：`historical_test` 已完成，根目录级 61 chunk 主目录未找到明确完整证据。
- error bar 图是否完成：未找到明确证据。
- FedAvg / Independent 真实预测是否完成：本次缺失实验资产中未找到真实预测训练输出证据，不能写已完成。

## 9. 风险与修正建议

- 单缺失率图横轴重复 5%：当前已发现该风险。
- `historical_test` 不能写成完整 61 天全量结果。
- 插补误差不能写成预测误差。
- 人工缺失不能写成真实数据天然缺失。
- `node_temporal_block` 需要单独验证。
- 多缺失率曲线需要后续正式补跑。
- 文档中已检出部分风险关键词，需要人工逐条复核表述口径。

## 10. 下一步建议

Phase 1：补全当前主目录审计，逐方法核对 61 chunk 主目录中的 masks、missing_datasets、imputed_datasets 与 summaries 一致性。
Phase 2：在严格历史因果口径下补跑多缺失率 MCAR 正式实验。
Phase 3：单独运行并验证 `node_temporal_block`。
Phase 4：补齐跨 seed mean±std。
Phase 5：根据正式 inventory 更新论文文档与图件。

附注：`data/processed/rnsd_processed.csv` 已存在，且字段证据支持地理邻近性补全。

