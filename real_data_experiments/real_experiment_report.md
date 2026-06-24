# Real Data Experiment Report / 真实数据实验报告

## 1. 当前阶段结论

- 当前真实数据主线已冻结为 tensor-only 单池化网格区域客户端实验配置。
- 当前正式训练入口为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前正式 sidecar 为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`。
- 当前正式 `tensor shape = (2, 630, 5856)`。
- 当前正式 `pool_mode = sum_mean`。
- 当前正式 `layout = standard`，即 `row = lat`、`col = lon`。
- 当前正式 `client = pooled-grid-region client`。
- 当前正式 `FedAvg = standard sample-size weighted FedAvg`。
- `parquet-direct` 仅保留为 legacy fallback，不再作为正式主实验入口。

## 2. 命名修正

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示 pooled grid region client，而不是原始路口节点客户端。
- 后续论文、报告、图表和运行说明中，建议统一使用“单池化网格区域客户端”。
- 英文建议统一使用 `single pooled-grid-region client`。
- 代码目录 `single_intersection_client` 和 `single_intersection_ablation` 暂时保留，但文档解释已统一修正。

## 3. 客户端层级定义

- `single_intersection_client` / `single_intersection_ablation`：
  单池化网格区域客户端；每个 client = 1 个 pooled grid region。
- `region_client` / `region_ablation`：
  区域网格客户端；每个 client = 一组 pooled grid regions。
- 两类实验当前都基于同一正式 tensor-only 输入，但客户端构造粒度不同。

## 4. 正式数据与客户端语义

- `channel 0 = pooled total flow`
- `channel 1 = pooled mean flow`
- `R = 630` 表示 pooled grid regions
- `T = 5856` 表示 time steps
- 当前 `active_region_count = 223`
- 默认客户端从 active pooled regions 中选取。
- 默认 region 选择按 `channel 0` 的 `mean_total_flow` 从高到低排序。

## 5. 固定 region 推荐

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

## 6. 正式实验配置建议

| 方案 | num_clients | rounds | local_epochs | batch_size | sequence_length | learning_rate | seed | 用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 3 | 5 | 3 | 32 | 12 | 0.001 | 42 | 快速正式实验 |
| B | 5 | 20 | 3 | 32 | 12 | 0.001 | 42 | 主论文正式实验 |
| C | 5 | 20 | 3 | 32 | 12 | 0.001 | 15 / 42 / 48 | 稳健性 / 多 seed 实验 |

## 7. 区域实验迁移状态

- `test/区域客户端计算_3×2_最终版.ipynb` 正在迁移到：
  `real_data_experiments/region_client/`
- `test/区域客户端消融实验_2×2_最终版.ipynb` 正在迁移到：
  `real_data_experiments/region_ablation/`
- 区域客户端正式主流程保持：
  tensor-only 输入、区域客户端多 region 划分、时间顺序切分、标准样本量加权 `FedAvg`、Independent baseline。
- notebook 中的 `FedProx`、`server damping`、`personalization` 仅作为历史探索记录，不进入默认主流程。

## 8. 输出目录保护

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

## 9. 当前范围控制

- 本阶段只冻结 tensor-only 单池化网格区域客户端实验配置并生成运行计划。
- 本阶段未直接运行正式长训练。
- 本阶段未迁移区域实验。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未改变标准 `FedAvg` 主线。

## 10. 参考文档

- 配置冻结方案：`real_data_experiments/tensor_only_experiment_plan_zh.md`
- 正式运行计划：`real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- 数据入口审计：`real_data_experiments/data_entry_audit_zh.md`
 - 区域主实验迁移映射：`real_data_experiments/region_client/region_notebook_migration_zh.md`
 - 区域消融迁移映射：`real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`

## 11. 仍需作者确认

- 论文正文是否统一采用“单池化网格区域客户端”作为中文正式命名。
- 主文是否只纳入方案 B，还是同时在附录中加入方案 C 的多 seed 汇总。
- 正式结果表是否同时展示固定 regions 的空间元数据和 `mean_total_flow`。
 - 区域客户端正式主实验最终是否固定 `num_clients = 3` 作为默认区域粒度。
