# 当前真实数据实验清单报告

## 1. 报告目的

本报告按新的真实数据实验 1-6 编号重新梳理目录、实验含义、旧新编号关系与结果路径引用。

## 2. 当前 Git 状态

- `git status --short --untracked-files=all`：

```text
M real_data_experiments/current_real_data_experiments_inventory_zh.md
 M real_data_experiments/formal_experiments_cleanup_report_zh.md
 M real_data_experiments/real_data_experiment_inventory.py
 M real_data_experiments/region_ablation/README_zh.md
 M real_data_experiments/region_ablation/historical_notes_zh.md
 M real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md
 M real_data_experiments/region_client/README_zh.md
 M real_data_experiments/region_client/historical_notes_zh.md
 M real_data_experiments/region_client/region_notebook_migration_zh.md
 M real_data_experiments/region_client_full_cells/README_zh.md
 M real_data_experiments/region_client_full_cells/full_cell_inventory_zh.md
 M real_data_experiments/single_intersection_ablation/README_zh.md
 M real_data_experiments/single_intersection_client/README_zh.md
 M real_data_experiments/single_intersection_client/debug-experiment1-constant-prediction.md
 M real_data_experiments/single_intersection_client/experiment1_client_allocation_requirement_audit_zh.md
 M real_data_experiments/single_intersection_client/experiment1_client_heterogeneity_diagnosis_zh.md
 M real_data_experiments/single_intersection_client/experiment1_constant_prediction_diagnosis_zh.md
 M real_data_experiments/single_intersection_client/experiment1_constant_prediction_fix_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_cuda_device_fix_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_cuda_environment_recheck_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_gap_diagnosis_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_metric_optimization_smoke_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_r60_smoke_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_smoke_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v2_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v2_sanity_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v3_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v3_sanity_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v4_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v4_sanity_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_optimization_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_review_meeting_alignment_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_selected_clients_rationale_zh.md
 M real_data_experiments/single_intersection_client/experiment1_selected_clients_spatial_distribution_zh.md
 M real_data_experiments/single_intersection_client/sic_client_allocation_requirement_audit.py
 M real_data_experiments/single_intersection_client/sic_client_heterogeneity_diagnosis.py
 M real_data_experiments/single_intersection_client/sic_fedavg_gap_diagnosis.py
 M real_data_experiments/single_intersection_client/sic_fedavg_metric_optimization_summary.py
 M real_data_experiments/single_intersection_client/sic_selected_clients_rationale_audit.py
 M real_data_experiments/single_intersection_client/sic_selected_clients_spatial_distribution_audit.py
?? .dbg/rfc-smoke-stall.env
?? .dbg/similarity_fedavg_probe.py
?? .dbg/trae-debug-log-rfc-smoke-stall.ndjson
?? debug-rfc-smoke-stall.md
```

- `git status -sb`：

```text
## main...origin/main
 M real_data_experiments/current_real_data_experiments_inventory_zh.md
 M real_data_experiments/formal_experiments_cleanup_report_zh.md
 M real_data_experiments/real_data_experiment_inventory.py
 M real_data_experiments/region_ablation/README_zh.md
 M real_data_experiments/region_ablation/historical_notes_zh.md
 M real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md
 M real_data_experiments/region_client/README_zh.md
 M real_data_experiments/region_client/historical_notes_zh.md
 M real_data_experiments/region_client/region_notebook_migration_zh.md
 M real_data_experiments/region_client_full_cells/README_zh.md
 M real_data_experiments/region_client_full_cells/full_cell_inventory_zh.md
 M real_data_experiments/single_intersection_ablation/README_zh.md
 M real_data_experiments/single_intersection_client/README_zh.md
 M real_data_experiments/single_intersection_client/debug-experiment1-constant-prediction.md
 M real_data_experiments/single_intersection_client/experiment1_client_allocation_requirement_audit_zh.md
 M real_data_experiments/single_intersection_client/experiment1_client_heterogeneity_diagnosis_zh.md
 M real_data_experiments/single_intersection_client/experiment1_constant_prediction_diagnosis_zh.md
 M real_data_experiments/single_intersection_client/experiment1_constant_prediction_fix_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_cuda_device_fix_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_cuda_environment_recheck_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_gap_diagnosis_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_metric_optimization_smoke_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_r60_smoke_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_smoke_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v2_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v2_sanity_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v3_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v3_sanity_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v4_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_formal_v4_sanity_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_optimization_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_review_meeting_alignment_report_zh.md
 M real_data_experiments/single_intersection_client/experiment1_selected_clients_rationale_zh.md
 M real_data_experiments/single_intersection_client/experiment1_selected_clients_spatial_distribution_zh.md
 M real_data_experiments/single_intersection_client/sic_client_allocation_requirement_audit.py
 M real_data_experiments/single_intersection_client/sic_client_heterogeneity_diagnosis.py
 M real_data_experiments/single_intersection_client/sic_fedavg_gap_diagnosis.py
 M real_data_experiments/single_intersection_client/sic_fedavg_metric_optimization_summary.py
 M real_data_experiments/single_intersection_client/sic_selected_clients_rationale_audit.py
 M real_data_experiments/single_intersection_client/sic_selected_clients_spatial_distribution_audit.py
?? .dbg/rfc-smoke-stall.env
?? .dbg/similarity_fedavg_probe.py
?? .dbg/trae-debug-log-rfc-smoke-stall.ndjson
?? debug-rfc-smoke-stall.md
```

- 当前未提交/未跟踪文件：
- `real_data_experiments/current_real_data_experiments_inventory_zh.md`
- `real_data_experiments/formal_experiments_cleanup_report_zh.md`
- `real_data_experiments/real_data_experiment_inventory.py`
- `real_data_experiments/region_ablation/README_zh.md`
- `real_data_experiments/region_ablation/historical_notes_zh.md`
- `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`
- `real_data_experiments/region_client/README_zh.md`
- `real_data_experiments/region_client/historical_notes_zh.md`
- `real_data_experiments/region_client/region_notebook_migration_zh.md`
- `real_data_experiments/region_client_full_cells/README_zh.md`
- `real_data_experiments/region_client_full_cells/full_cell_inventory_zh.md`
- `real_data_experiments/single_intersection_ablation/README_zh.md`
- `real_data_experiments/single_intersection_client/README_zh.md`
- `real_data_experiments/single_intersection_client/debug-experiment1-constant-prediction.md`
- `real_data_experiments/single_intersection_client/experiment1_client_allocation_requirement_audit_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_client_heterogeneity_diagnosis_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_constant_prediction_diagnosis_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_constant_prediction_fix_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_cuda_device_fix_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_cuda_environment_recheck_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_gap_diagnosis_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_metric_optimization_smoke_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_r60_smoke_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_smoke_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v2_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v2_sanity_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v3_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v3_sanity_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v4_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v4_sanity_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_optimization_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_review_meeting_alignment_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_selected_clients_rationale_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_selected_clients_spatial_distribution_zh.md`
- `real_data_experiments/single_intersection_client/sic_client_allocation_requirement_audit.py`
- `real_data_experiments/single_intersection_client/sic_client_heterogeneity_diagnosis.py`
- `real_data_experiments/single_intersection_client/sic_fedavg_gap_diagnosis.py`
- `real_data_experiments/single_intersection_client/sic_fedavg_metric_optimization_summary.py`
- `real_data_experiments/single_intersection_client/sic_selected_clients_rationale_audit.py`
- `real_data_experiments/single_intersection_client/sic_selected_clients_spatial_distribution_audit.py`
- `.dbg/rfc-smoke-stall.env`
- `.dbg/similarity_fedavg_probe.py`
- `.dbg/trae-debug-log-rfc-smoke-stall.ndjson`
- `debug-rfc-smoke-stall.md`

## 3. 新实验 1-6 编号总览

| 新编号 | 英文名称 | 实验含义 | 对应目录 | 旧新映射 |
| --- | --- | --- | --- | --- |
| 实验 1 | single grid client comparison | 单个网格作为单个客户端的对比实验 | `real_data_experiments/single_intersection_client/` | 原实验 1 -> 新实验 1 |
| 实验 2 | single grid client ablation | 单个网格作为单个客户端的消融实验 | `real_data_experiments/single_intersection_ablation/` | 原实验 2 -> 新实验 2 |
| 实验 3 | similar grid group client comparison | 多个相似网格合并为一个客户端的对比实验 | `real_data_experiments/region_client_full_cells/` | 原实验 5 -> 新实验 3 |
| 实验 4 | similar grid group client ablation | 多个相似网格合并为一个客户端的消融实验 | `real_data_experiments/region_client_full_cells/` | 原实验 5 的消融补齐 -> 新实验 4 |
| 实验 5 | global similarity partition comparison | 全局所有网格按相似度划分为客户端的对比实验 | `real_data_experiments/region_client/` | 原实验 3 -> 新实验 5 |
| 实验 6 | global similarity partition ablation | 全局所有网格按相似度划分为客户端的消融实验 | `real_data_experiments/region_ablation/` | 原实验 4 -> 新实验 6 |

## 4. 旧编号到新编号转换表

| 旧编号 | 新编号 | 含义 |
| --- | --- | --- |
| 原实验 1 | 新实验 1 | 单个网格作为单个客户端的对比实验 |
| 原实验 2 | 新实验 2 | 单个网格作为单个客户端的消融实验 |
| 原实验 5 | 新实验 3 | 多个相似网格合并为一个客户端的对比实验 |
| 原实验 5 的消融补齐 | 新实验 4 | 多个相似网格合并为一个客户端的消融实验 |
| 原实验 3 | 新实验 5 | 全局所有网格按相似度划分为客户端的对比实验 |
| 原实验 4 | 新实验 6 | 全局所有网格按相似度划分为客户端的消融实验 |

## 5. 三类客户端组织方式边界

### 第一类：单个网格作为单个客户端

- 定义：`client_i = grid_cell_i`。
- 对应：新实验 1、新实验 2。
- 边界：不能把多个 grid cells 合并进同一个 client。

### 第二类：多个相似网格合并为一个客户端

- 定义：`client_k = {grid_cell_a, grid_cell_b, grid_cell_c, ...}`。
- 对应：新实验 3、新实验 4。
- 边界：重点是把若干相似 grid cells 合并成一个 client，不把它写成全局覆盖式完整划分。

### 第三类：全局所有网格按相似度划分为客户端

- 定义：`All grid cells are partitioned into K non-overlapping clients.`
- 对应：新实验 5、新实验 6。
- 边界：强调全局覆盖式划分，且 `client_i and client_j are non-overlapping when i != j.`

## 6. 目录对应关系

| 目录 | 当前归属 | 重构后含义 | 是否存在 | 是否有 README | 文件数 |
| --- | --- | --- | --- | --- | --- |
| `real_data_experiments/common/` | 公共模块 | 公共张量、划分、FedAvg、指标与结果写出工具 | 是 | 是 | 38 |
| `real_data_experiments/single_intersection_client/` | 新实验 1 | 新实验 1：single grid client comparison | 是 | 是 | 56 |
| `real_data_experiments/single_intersection_ablation/` | 新实验 2 | 新实验 2：single grid client ablation | 是 | 是 | 12 |
| `real_data_experiments/region_client_full_cells/` | 新实验 3 / 4 | 新实验 3 / 4：similar grid group client comparison / ablation | 是 | 是 | 27 |
| `real_data_experiments/region_client/` | 新实验 5 | 新实验 5：global similarity partition comparison | 是 | 是 | 14 |
| `real_data_experiments/region_ablation/` | 新实验 6 | 新实验 6：global similarity partition ablation | 是 | 是 | 14 |

## 6. 新实验 1-6 当前目录与状态

### 新实验 1：single grid client comparison

- 对应目录：`real_data_experiments/single_intersection_client/`。
- 旧新映射：原实验 1 -> 新实验 1。
- 实验含义：单个网格作为单个客户端的对比实验。
- 客户端边界：`client_i = grid_cell_i`。
- 正式结果：`results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/`。
- formal v4 指标：FedAvg `RMSE=20815.803975`，NaiveLastValue `RMSE=N/A`。
- 诊断链：r40 FedAvg `RMSE=19177.795050`；r60 FedAvg `RMSE=18762.383340`；最佳 diagnostics 方案 `r80/e2/lr5e-4` 的 FedAvg `RMSE=18242.610962`。

### 新实验 2：single grid client ablation

- 对应目录：`real_data_experiments/single_intersection_ablation/`。
- 旧新映射：原实验 2 -> 新实验 2。
- 实验含义：单个网格作为单个客户端的消融实验。
- 结果归属：`results/real_data_experiments/single_intersection_ablation/` 与 `results/real_data_experiments/single_intersection_ablation_tensor/`。
- 当前状态：目录、README、core/config 和历史结果存在，但未新增 formal 结果链。

### 新实验 3：similar grid group client comparison

- 对应目录：`real_data_experiments/region_client_full_cells/`。
- 旧新映射：原实验 5 -> 新实验 3。
- 实验含义：多个相似网格合并为一个客户端的对比实验。
- 现有 similarity smoke：FedAvg `RMSE=279893.696446`。
- 辅助 spatial 对照：FedAvg `RMSE=174389.562767`；保留旧路径，只作为 grouped-client 线路的辅助诊断。

### 新实验 4：similar grid group client ablation

- 对应目录：仍暂挂在 `real_data_experiments/region_client_full_cells/`。
- 旧新映射：原实验 5 的消融补齐 -> 新实验 4。
- 当前状态：本阶段先补编号、README、inventory 与结果归属说明；尚未新增独立 ablation 训练入口。

### 新实验 5：global similarity partition comparison

- 对应目录：`real_data_experiments/region_client/`。
- 旧新映射：原实验 3 -> 新实验 5。
- 实验含义：全局所有网格按相似度划分为客户端的对比实验。
- smoke 结果：`results/real_data_experiments/region_client_tensor_smoke/`，FedAvg `RMSE=949738.336016`。

### 新实验 6：global similarity partition ablation

- 对应目录：`real_data_experiments/region_ablation/`。
- 旧新映射：原实验 4 -> 新实验 6。
- 实验含义：全局所有网格按相似度划分为客户端的消融实验。
- smoke 文件 `ablation_metrics.csv` 是否存在：是。

## 7. 结果目录引用与新编号归属

| 结果路径 | 归属新实验 | 类型 | 说明 |
| --- | --- | --- | --- |
| `results/real_data_experiments/formal/grid_cell_main_full_cuda_v4` | 新实验 1 | formal | 归入新实验 1 的正式结果。 |
| `results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r40_cuda` | 新实验 1 | diagnostics | 归入新实验 1 的 diagnostics/smoke 结果。 |
| `results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r60_cuda` | 新实验 1 | diagnostics | 归入新实验 1 的 diagnostics/smoke 结果。 |
| `results/real_data_experiments/diagnostics/experiment1_metric_opt_k5_r80_e2_lr5e4_cuda` | 新实验 1 | diagnostics | 归入新实验 1 的 diagnostics/smoke 结果。 |
| `results/real_data_experiments/diagnostics/full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda` | 新实验 3 | diagnostics | 归入新实验 3 的 similarity diagnostic/smoke 结果。 |
| `results/real_data_experiments/diagnostics/full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda` | 新实验 3 / 4 所在线的辅助诊断 | diagnostics | 保留为 grouped-client 目录下的 spatial 辅助对照，旧路径不移动。 |
| `results/real_data_experiments/region_ablation_tensor_smoke` | 新实验 6 | smoke | 若其客户端逻辑为全局覆盖式消融，则归入新实验 6 的 smoke。 |
| `results/real_data_experiments/region_client_tensor_smoke` | 新实验 5 | smoke | 若其客户端逻辑为全局覆盖式划分，则归入新实验 5 的 smoke。 |
| `results/real_data_experiments/single_intersection_ablation` | 新实验 2 | other | 归入新实验 2 的历史/轻量消融结果。 |
| `results/real_data_experiments/single_intersection_ablation_tensor` | 新实验 2 | other | 归入新实验 2 的历史/轻量消融结果。 |

## 8. 本次重构结论

- `single_intersection_client/` 固定对应新实验 1。
- `single_intersection_ablation/` 固定对应新实验 2。
- `region_client_full_cells/` 在最小改动方案下承接新实验 3，并为新实验 4 预留同目录补齐位。
- `region_client/` 固定对应新实验 5。
- `region_ablation/` 固定对应新实验 6。
- 旧 results 不删除、不移动；只在文档中新增新编号对应关系。

## 9. 边界声明

- 未运行训练。
- 未修改 FedAvg 聚合公式。
- 未修改模型结构。
- 未修改正式数据入口。
- 未生成 `6.池化网格张量.pt`。
- 未删除 `NaiveLastValue`。
- 未删除或替换 `289`。
- 未改动 `results/` 中已有结果文件，只更新文档中的引用与归属说明。

_本报告由 `real_data_experiments/current_real_data_experiments_inventory_zh.md` 生成；扫描根目录为 `real_data_experiments`，results 根目录为 `results/real_data_experiments`。_
