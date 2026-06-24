# Real Data Experiment Report / 真实数据实验报告

## 1. 当前阶段结论

- 当前真实数据实验已完成以下四条 tensor-only Python 化主线：
  `single_intersection_client`、`single_intersection_ablation`、`region_client`、`region_ablation`。
- 当前正式训练入口为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前正式 sidecar 为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`。
- 当前正式 `tensor shape = (2, 630, 5856)`。
- 当前正式 `pool_mode = sum_mean`。
- 当前正式 `layout = standard`，即 `row = lat`、`col = lon`。
- 当前真实数据实验统一描述为两类客户端组织设置，而不是新的聚合机制。
- 当前正式 `FedAvg = standard sample-size weighted FedAvg`。
- `parquet-direct` 仅保留为 legacy fallback，不再作为正式主实验入口。
- `region_client` / `region_ablation` 已通过 smoke test。
- 当前 smoke test 结果仅用于验证代码链路、输出文件和可视化流程，不作为论文正式结果。

## 2. 命名对照

| 代码目录 | 正式中文名称 | 正式英文名称 | 客户端定义 |
|---|---|---|---|
| `single_intersection_client` / `single_intersection_ablation` | 网格单元级客户端联邦学习设置 | Grid-cell-level Client Federated Learning Setting | 每个 client = 1 个 pooled grid cell |
| `region_client` / `region_ablation` | 簇级客户端联邦学习设置 | Cluster-level Client Federated Learning Setting | 每个 client = 一组 temporal-similarity-clustered grid cells |

## 3. 命名说明

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示“网格单元级客户端设置”，而不是原始路口节点客户端。
- `single_intersection_client` 和 `single_intersection_ablation` 这两个代码目录在文档中统一解释为：
  网格单元级客户端联邦学习设置 / Grid-cell-level Client Federated Learning Setting。
- `region_client` 和 `region_ablation` 在文档中统一解释为：
  簇级客户端联邦学习设置 / Cluster-level Client Federated Learning Setting。
- 为避免把客户端组织方式误解为新的联邦聚合机制，本文统一使用“setting”而非“mechanism”描述两类真实数据实验。
- 两类设置共享相同的 tensor-only 数据入口、模型结构、时间顺序切分和标准样本量加权 FedAvg 聚合流程，区别仅在于客户端组织粒度不同。
- 在网格单元级客户端设置中，每个客户端对应一个池化后的空间网格单元；在簇级客户端设置中，多个具有相似时间变化模式的网格单元被划分到同一客户端中。
- 英文说明：
  To avoid implying a new federated aggregation algorithm, the two real-data experiments are described as client organization settings rather than mechanisms. Both settings share the same tensor-only input, temporal split protocol, local model architecture, and standard sample-size weighted FedAvg aggregation. The difference lies only in the granularity of client organization: in the grid-cell-level client setting, each client corresponds to a single pooled spatial grid cell, whereas in the cluster-level client setting, multiple grid cells with similar temporal patterns are grouped into the same client.

## 4. 客户端层级定义

- `single_intersection_client` / `single_intersection_ablation`：
  网格单元级客户端联邦学习设置；每个 client = 1 个 pooled grid region。
- `region_client` / `region_ablation`：
  簇级客户端联邦学习设置；每个 client = 一组 temporal-similarity-clustered pooled grid regions。
- 两类实验当前都基于同一正式 tensor-only 输入，但客户端构造粒度不同。

## 5. 正式数据与客户端语义

- `channel 0 = pooled total flow`
- `channel 1 = pooled mean flow`
- `R = 630` 表示 pooled grid regions
- `T = 5856` 表示 time steps
- 当前 `active_region_count = 223`
- 默认客户端从 active pooled regions 中选取。
- 默认 region 选择按 `channel 0` 的 `mean_total_flow` 从高到低排序。

## 6. 固定 region 推荐

- 固定 region 计划文件为 `real_data_experiments/selected_regions_fixed_plan.csv`。
- 当前 smoke test 已选 region 为 `290`、`284`。
- 推荐正式 top-3 regions：
  `290, 284, 318`
- 推荐正式 top-5 regions：
  `290, 284, 318, 288, 289`

| rank | region_id | pooled_row | pooled_col | source_node_count | mean_total_flow |
|---|---:|---:|---:|---:|---:|
| 1 | 290 | 9 | 20 | 667 | 1953917.875 |
| 2 | 284 | 9 | 14 | 698 | 1914353.125 |
| 3 | 318 | 10 | 18 | 711 | 1857832.250 |
| 4 | 288 | 9 | 18 | 698 | 1702061.250 |
| 5 | 289 | 9 | 19 | 663 | 1659179.750 |

## 7. 正式实验配置建议

| 方案 | num_clients | rounds | local_epochs | batch_size | sequence_length | learning_rate | seed | 用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 3 | 5 | 3 | 32 | 12 | 0.001 | 42 | 快速正式实验 |
| B | 5 | 20 | 3 | 32 | 12 | 0.001 | 42 | 主论文正式实验 |
| C | 5 | 20 | 3 | 32 | 12 | 0.001 | 15 / 42 / 48 | 稳健性 / 多 seed 实验 |

## 8. 区域实验迁移状态

- `test/区域客户端计算_3×2_最终版.ipynb` 已完成迁移到 `real_data_experiments/region_client/`。
- `test/区域客户端消融实验_2×2_最终版.ipynb` 已完成迁移到 `real_data_experiments/region_ablation/`。
- 两条区域实验线均已切换为 tensor-only 输入，并通过 smoke test。
- 簇级客户端设置的正式主流程保持：
  tensor-only 输入、多网格单元客户端划分、时间顺序切分、标准样本量加权 `FedAvg`、Independent baseline。
- notebook 中的 `FedProx`、`server damping`、`personalization` 仅作为历史探索记录，不进入默认主流程。

## 9. Smoke Test 结果说明

- 当前 `region_client_tensor_smoke` 与 `region_ablation_tensor_smoke` 仅用于验证：
  tensor-only 数据入口、active region 划分、簇级客户端设置、时间顺序切分、标准样本量加权 `FedAvg`、metrics 输出、可视化输出。
- 当前 smoke test 使用小轮次、小样本限制或 CPU 友好配置，不作为论文正式结果。
- 后续论文正式结果应来自独立输出目录中的正式长训练。
- smoke test 结果不作为论文正式结果。

## 10. 输出目录保护

- 为避免覆盖现有 smoke test 结果，正式运行建议使用独立 `--output-dir`。
- 推荐目录：
  `results/real_data_experiments/single_region_client_tensor_quick/`
  `results/real_data_experiments/single_region_client_tensor_main/`
  `results/real_data_experiments/single_region_client_tensor_seed15/`
  `results/real_data_experiments/single_region_client_tensor_seed42/`
  `results/real_data_experiments/single_region_client_tensor_seed48/`
  `results/real_data_experiments/single_region_ablation_tensor_main/`
  `results/real_data_experiments/region_client_tensor_smoke/`
  `results/real_data_experiments/region_ablation_tensor_smoke/`

## 11. 当前范围控制

- 本阶段只冻结 tensor-only 网格单元级客户端设置的实验配置并生成运行计划。
- 本阶段未直接运行正式长训练。
- 簇级客户端主实验和簇级客户端消融实验已完成 tensor-only Python 化迁移；当前已完成的是 smoke test 级别链路验证，尚未运行论文正式长训练。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未改变标准 `FedAvg` 主线。

## 12. 参考文档

- 配置冻结方案：`real_data_experiments/tensor_only_experiment_plan_zh.md`
- 正式运行计划：`real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- 数据入口审计：`real_data_experiments/data_entry_audit_zh.md`
 - 区域主实验迁移映射：`real_data_experiments/region_client/region_notebook_migration_zh.md`
 - 区域消融迁移映射：`real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`

## 13. 仍需作者确认

- 论文正文、图注和表头是否同步采用“网格单元级客户端联邦学习设置 / 簇级客户端联邦学习设置”作为正式命名。
- 主文是否只纳入方案 B，还是同时在附录中加入方案 C 的多 seed 汇总。
- 正式结果表是否同时展示固定 regions 的空间元数据和 `mean_total_flow`。
- 簇级客户端正式主实验最终是否固定 `num_clients = 3` 作为默认簇粒度。
