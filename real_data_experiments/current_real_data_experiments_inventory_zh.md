# 当前真实数据实验清单报告

## 1. 报告目的

本报告用于梳理当前项目中真实数据实验有哪些、每个实验当前状态如何、已有结果和缺口是什么。

## 2. 当前 Git 状态

- `git status --short`：

```text
?? .dbg/rfc-smoke-stall.env
?? .dbg/similarity_fedavg_probe.py
?? .dbg/trae-debug-log-rfc-smoke-stall.ndjson
?? debug-rfc-smoke-stall.md
?? real_data_experiments/current_real_data_experiments_inventory_zh.md
?? real_data_experiments/real_data_experiment_inventory.py
```

- 当前未提交文件：
- `.dbg/rfc-smoke-stall.env`
- `.dbg/similarity_fedavg_probe.py`
- `.dbg/trae-debug-log-rfc-smoke-stall.ndjson`
- `debug-rfc-smoke-stall.md`
- `real_data_experiments/current_real_data_experiments_inventory_zh.md`
- `real_data_experiments/real_data_experiment_inventory.py`
- 是否存在 results/data 越界改动：否
- 是否存在核心代码未提交改动：否
- `git status -sb`：

```text
## main...origin/main
?? .dbg/rfc-smoke-stall.env
?? .dbg/similarity_fedavg_probe.py
?? .dbg/trae-debug-log-rfc-smoke-stall.ndjson
?? debug-rfc-smoke-stall.md
?? real_data_experiments/current_real_data_experiments_inventory_zh.md
?? real_data_experiments/real_data_experiment_inventory.py
```

- 最近 10 条提交：

```text
7cf9948 feat: add full-cells region-client diagnostic experiment
bf7162b docs: record fedavg metric optimization smoke
408d34f docs: record selected clients spatial distribution
62fbd6e docs: explain selected clients rationale
6e58ee8 docs: audit real data client allocation requirements
0d68dde docs: record heterogeneity and review alignment analysis
cd3ec4a docs: record fedavg r60 rounds smoke
e96a5f7 docs: record fedavg rounds smoke diagnosis
a89885c chore: ignore real data diagnostic outputs
5a82ca5 docs: diagnose fedavg gap in grid cell experiment
```

## 3. 真实数据实验目录总览

| 目录 | 类型 | 是否独立实验 | 当前定位 | 状态 |
| --- | --- | --- | --- | --- |
| common/ | 公共工具目录 | 否 | 公共 tensor/划分/FedAvg/指标/结果写出工具 | 已存在 |
| single_intersection_client/ | 正式实验 | 是 | 实验 1：grid-cell-level client 主实验 | 最完整 |
| single_intersection_ablation/ | 计划实验 | 是 | 实验 2：grid-cell-level client 消融 | 部分完成 |
| region_client/ | 正式实验 | 是 | 实验 3：cluster-level / multi-grid client 主实验 | 部分完成 |
| region_ablation/ | 计划实验 | 是 | 实验 4：cluster-level / multi-grid client 消融 | 部分完成 |
| region_client_full_cells/ | 新增实验 | 是 | 新增 full-cells 多客户端组织实验 | 部分完成 |

## 4. 实验 1：single_intersection_client

- 实验定位：实验 1，当前最完整的 `grid-cell-level client` 正式主实验。
- client 组织方式：每个 client = 一个 active pooled region，即 one pooled grid cell / one pooled grid region。
- 当前 selected_clients：`290, 284, 318, 288, 289`。
- 数据入口：正式主入口为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`，辅以 `node_flow_grid_regions.csv`；`parquet` 仅保留 legacy fallback。
- 是否已有训练入口：是，`sic_core.py` 为正式训练入口，`sic_config.py` 提供 CLI 配置；当前正式 v4 `rounds=20`，`local_epochs=3`。
- 是否已有正式结果：是，`results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/` 已存在；FedAvg `RMSE=20815.803975`，NaiveLastValue 对比已在 formal v4 报告与后续诊断链中补充。
- 是否已有 diagnostics/smoke：是，至少包括 `experiment1_fedavg_rounds_smoke_r40_cuda`、`experiment1_fedavg_rounds_smoke_r60_cuda`、`experiment1_metric_opt_k5_r80_e1_lr5e4_cuda`、`experiment1_metric_opt_k5_r80_e2_lr5e4_cuda`、`experiment1_metric_opt_k5_r100_e1_lr5e4_cuda`。
- 是否已有报告：是，已有 formal v2/v3/v4 报告、client 异质性诊断、selected_clients rationale、空间覆盖与分布统计、FedAvg gap 诊断、FedAvg 多指标优化 smoke 报告。
- 已有正式 v4：是，且为当前实验 1 正式 CUDA 审计结果；FedAvg `R2=0.873045`。
- 已有 r40/r60 rounds 诊断：是，r40 FedAvg `RMSE=19177.795050`，r60 FedAvg `RMSE=18762.383340`。
- 已有 client 异质性诊断：是，报告明确指出 `289` 是主要拖累 client。
- 已有 selected_clients rationale：是，已说明当前 K=5 设置是相对原稿 K=3 的增强且已形成完整证据链。
- 已有空间覆盖与分布统计：是，已恢复 pooled row/col 与近似经纬度边界，确认当前覆盖主要是局部子区域。
- 已有 FedAvg 多指标优化 smoke：是；其中当前最佳 diagnostics 方案是 `experiment1_metric_opt_k5_r80_e2_lr5e4_cuda`。
- 当前结论：K=5 细粒度异质设置已完成正式链与诊断链，但即便 `r80/e2/lr5e-4` 将 FedAvg `RMSE` 优化到 `18242.610962`，仍未全面超过 NaiveLastValue；`289` 继续是关键异质点。
- 当前缺口：缺少能缓解强 non-IID 的新 client 组织方式正式对比；继续直接调 K=5 的边际收益有限。

## 5. 实验 2：single_intersection_ablation

- 是否已有目录：是。
- 是否已有 core：是，当前训练入口为 `sia_core.py`。
- 是否已有正式训练：未见 `results/real_data_experiments/formal/` 下对应 formal 结果；当前更像已实现 + 历史/轻量结果，而非正式结果链。
- 是否已有结果：有，存在 `results/real_data_experiments/single_intersection_ablation/` 与 `results/real_data_experiments/single_intersection_ablation_tensor/`，但不属于 formal/diagnostics 主线目录。
- 是否尚未启动：不是完全未启动；目录、README、core/config、结果样例均已存在，但尚未形成当前阶段正式消融结果链。
- 与一审消融要求的关系：该目录承载模型结构消融，但在当前推进节奏中优先级低于 client 组织方式调整，因此不宜先扩展实验 2。

## 6. 实验 3：region_client

- 是否已有目录：是。
- 是否已有 core：是，当前训练入口为 `rc_core.py`。
- 是否已有 region/cluster client 逻辑：是；默认 `partition_method=spatial_block`，可选 `flow_kmeans`，每个 client = 一组 pooled grid regions。
- 是否已有正式训练：未见 formal 结果。
- 是否已有结果：有，`results/real_data_experiments/region_client_tensor_smoke/` 表明 tensor smoke 已跑通。
- 是否可作为“多个 grid cells = 一个 client”的方向：是；这是 cluster-level / multi-grid client 的主线候选，但当前只完成 smoke，尚无正式对比结果。
- 当前缺口：缺 formal 训练结果、缺与实验 1 / full-cells 的统一对比、缺结论性报告。

## 7. 实验 4：region_ablation

- 是否已有目录：是。
- 是否已有 core：是，当前训练入口为 `ra_core.py`。
- 是否已有正式训练：未见 formal 结果。
- 是否已有结果：有，`results/real_data_experiments/region_ablation_tensor_smoke/` 表明 smoke 已跑通。
- 是否尚未启动：不是完全未启动；已完成 Python 化迁移与 smoke，但尚未进入正式阶段。
- 与 region/cluster client 消融的关系：该目录用于 cluster-level client 的结构消融，应在 region/full-cells 主线结果稳定后再推进。

## 8. 新增或计划实验：region_client_full_cells

- 是否存在目录：是。
- 它是新增 full-cells 多客户端组织实验：是，目标是使用全部有效 grid cells，将多个 cells 组织成一个 client。
- 是否已有 inventory / partition / dataset / core：是 / 是 / 是 / 是。
- 是否使用全部有效 grid cells：是，`full_cell_inventory_zh.md` 记录 `630` 个 pooled grid cells 中 `223` 个为 valid cells。
- 是否实现 spatial partition：是，已存在 `spatial_k5.json`、`spatial_k8.json`、`spatial_k10.json`。
- 是否实现 similarity partition：是，已存在 `similarity_k5.json`、`similarity_k8.json`、`similarity_k10.json`。
- 是否已有 smoke：是，已有 `full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda` 与 `full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda`；spatial FedAvg `RMSE=174389.562767`，similarity FedAvg `RMSE=279893.696446`。
- 当前状态：目录、inventory、partition、dataset、训练入口和 K=5 smoke 已完成；尚缺统一的 smoke 对比报告产物与是否扩展 K=8/K=10 的结论判断。

## 9. 已有 results 清单

| result_dir | 类型 | 对应实验 | 是否 formal | 是否 diagnostics | 核心文件是否齐全 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| results/real_data_experiments/formal/grid_cell_main_full_cuda | formal | single_intersection_client | 是 | 否 | 是 | - |
| results/real_data_experiments/formal/grid_cell_main_full_cuda_v2 | formal | single_intersection_client | 是 | 否 | 是 | - |
| results/real_data_experiments/formal/grid_cell_main_full_cuda_v3 | formal | single_intersection_client | 是 | 否 | 是 | - |
| results/real_data_experiments/formal/grid_cell_main_full_cuda_v4 | formal | single_intersection_client | 是 | 否 | 是 | 当前实验 1 正式 v4 CUDA |
| results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r40_cuda | diagnostics | single_intersection_client | 否 | 是 | 是 | - |
| results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r60_cuda | diagnostics | single_intersection_client | 否 | 是 | 是 | - |
| results/real_data_experiments/diagnostics/experiment1_metric_opt_k5_r100_e1_lr5e4_cuda | diagnostics | single_intersection_client | 否 | 是 | 是 | - |
| results/real_data_experiments/diagnostics/experiment1_metric_opt_k5_r80_e1_lr5e4_cuda | diagnostics | single_intersection_client | 否 | 是 | 是 | - |
| results/real_data_experiments/diagnostics/experiment1_metric_opt_k5_r80_e2_lr5e4_cuda | diagnostics | single_intersection_client | 否 | 是 | 是 | FedAvg_RMSE=18242.610962 |
| results/real_data_experiments/diagnostics/full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda | diagnostics | region_client_full_cells | 否 | 是 | 是 | full-cells similarity K=5 smoke |
| results/real_data_experiments/diagnostics/full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda | diagnostics | region_client_full_cells | 否 | 是 | 是 | full-cells spatial K=5 smoke |
| results/real_data_experiments/region_ablation_tensor_smoke | smoke | region_ablation | 否 | 否 | 是 | - |
| results/real_data_experiments/region_client_tensor_smoke | smoke | region_client | 否 | 否 | 是 | - |
| results/real_data_experiments/single_intersection_ablation | other | single_intersection_ablation | 否 | 否 | 是 | - |
| results/real_data_experiments/single_intersection_ablation_tensor | other | single_intersection_ablation | 否 | 否 | 是 | - |
| results/real_data_experiments/single_intersection_client | other | single_intersection_client | 否 | 否 | 是 | - |
| results/real_data_experiments/single_intersection_client_tensor | other | single_intersection_client | 否 | 否 | 是 | - |

## 10. 当前真实数据实验完成度判断

| 实验 | 完成度 | 已完成 | 未完成 | 下一步 |
| --- | --- | --- | --- | --- |
| single_intersection_client | 已完成 | 正式 v4 + r40/r60 + 异质性/选择依据/空间覆盖/metric optimization smoke | 未全面超过 NaiveLastValue | 转向新的 client 组织方式 |
| single_intersection_ablation | 部分完成 | 目录、README、core/config、已有历史与轻量结果 | 未形成 formal 结果链 | 暂不启动新跑，保留与一审消融要求对齐 |
| region_client | 部分完成 | 目录、README、core/config、spatial_block/flow_kmeans、tensor smoke | 无 formal 结果 | 继续作为多个 grid cells = 一个 client 主线候选 |
| region_ablation | 部分完成 | 目录、README、core/config、tensor smoke | 无 formal 结果 | 等待 region client 主线稳定后再做消融 |
| region_client_full_cells | 可进入下一阶段 | inventory、partition、dataset、core、K=5 spatial/similarity smoke | 缺统一对比报告，尚无 formal | 先生成对比报告，再决定是否扩展 K=8/K=10 |

## 11. 当前最重要结论

- 当前只有 `single_intersection_client` 最完整。
- 当前 K=5 grid-cell client 已进入边际收益阶段。
- 继续调 K=5 不如换 client 组织方式。
- `region/cluster/full-cells client` 是下一步。
- 不能删除 `NaiveLastValue`。
- 不能删除或替换 `289`，只能作为诊断或新实验对照。

## 12. 下一步建议

- 已经存在 `full-cells smoke`，建议先生成对比报告并判断是否扩展 `K=8/K=10`。

## 13. 边界声明

本阶段只读梳理当前真实数据实验；未运行训练，未修改 FedAvg，未修改模型结构，未修改数据划分，未提交 results，未执行 git add、git commit 或 git push。

_本报告由 `real_data_experiments/current_real_data_experiments_inventory_zh.md` 生成；扫描根目录为 `real_data_experiments`，results 根目录为 `results/real_data_experiments`。_
