# Tensor-Only 单池化网格区域客户端实验配置冻结方案

## 1. 本阶段范围

- 本阶段只冻结 tensor-only 单池化网格区域客户端实验配置，并生成正式运行计划。
- 本阶段不直接运行正式长训练。
- 本阶段不迁移 `region_client` / `region_ablation`。
- 本阶段不修改 LaTeX。
- 本阶段不修改 `simulation_experiments/`。
- 本阶段不改变标准样本量加权 `FedAvg` 主线。
- 本阶段不生成历史命名 `6.池化网格张量.pt`。
- 本阶段不把 smoke test 指标写成论文正式结果。

## 2. 正式数据入口冻结

- 正式 `tensor_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式 `regions_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 正式 `pool_mode = sum_mean`
- 正式 `layout = standard`
- 正式 `tensor shape = (2, 630, 5856)`
- 其中 `channel 0 = pooled total flow`
- 其中 `channel 1 = pooled mean flow`
- `R = 630` 表示 pooled grid regions
- `T = 5856` 表示 time steps
- 当前 `active_region_count = 223`
- `parquet-direct = legacy fallback only`

## 3. 命名修正冻结

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示 `pooled-grid-region client`，不再表示原始路口节点客户端。
- 论文、报告和后续运行说明中，建议统一称为“单池化网格区域客户端”。
- 英文建议统一称为 `single pooled-grid-region client`。
- 代码目录名 `single_intersection_client` 与 `single_intersection_ablation` 暂不改动，但文档解释必须按上述语义修正。

## 4. 联邦主线冻结

- 当前正式联邦聚合仍为标准样本量加权 `FedAvg`。
- 服务器聚合只依赖客户端训练样本量，不引入 `FedProx`、loss-weighted aggregation、personalization 或其他自适应聚合。
- 本阶段冻结的是 tensor-only 数据入口和单池化网格区域客户端配置，不改变联邦训练主算法。

## 5. Region 选择策略冻结

### 5.1 选择原则

- 默认从 `active regions` 中按 `channel 0` 的 `mean_total_flow` 从高到低选择 top-K。
- 正式实验应固定 `region_id`，不建议每次根据新运行结果动态变化。
- 固定 region 推荐清单已写入：
  `real_data_experiments/selected_regions_fixed_plan.csv`
- 当前 smoke test 已使用的 region 为：
  `290`、`284`

### 5.2 固定推荐结果

| rank | region_id | pooled_row | pooled_col | source_node_count | mean_total_flow | top-3 | top-5 |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 290 | 9 | 20 | 667 | 1953917.875 | yes | yes |
| 2 | 284 | 9 | 14 | 698 | 1914353.125 | yes | yes |
| 3 | 318 | 10 | 18 | 711 | 1857832.250 | yes | yes |
| 4 | 288 | 9 | 18 | 698 | 1702061.250 | no | yes |
| 5 | 289 | 9 | 19 | 663 | 1659179.750 | no | yes |

### 5.3 冻结推荐

- 推荐正式 top-3 regions：
  `290, 284, 318`
- 推荐正式 top-5 regions：
  `290, 284, 318, 288, 289`
- 若后续论文正文需要固定客户端集合，建议直接引用上述固定 ID，而不是重新排序。

## 6. 正式实验候选配置

| 方案 | num_clients | rounds | local_epochs | batch_size | sequence_length | learning_rate | seed | selected_regions | 用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A | 3 | 5 | 3 | 32 | 12 | 0.001 | 42 | top-3 active regions by mean total flow | 快速正式实验 |
| B | 5 | 20 | 3 | 32 | 12 | 0.001 | 42 | top-5 active regions by mean total flow | 主论文正式实验 |
| C | 5 | 20 | 3 | 32 | 12 | 0.001 | 15, 42, 48 | fixed top-5 active regions | 稳健性 / 多 seed 实验 |

## 7. 推荐正式方案

- 默认推荐先执行方案 A，检查输出目录、图表链路和 CPU 运行时长。
- 主论文正式实验推荐采用方案 B。
- 若需要更稳健的表格和误差区间，补充执行方案 C，并对 `seed in [15, 42, 48]` 汇总均值与标准差。
- 单区域消融正式实验建议与方案 B 对齐，即固定 top-5 regions、`rounds = 20`、`local_epochs = 3`、`batch_size = 32`、`sequence_length = 12`、`learning_rate = 0.001`。

## 8. 输出目录保护

- 为避免覆盖现有 smoke test 结果，不建议继续写入：
  `results/real_data_experiments/single_intersection_client_tensor/`
- 为避免覆盖现有 smoke test 结果，不建议继续写入：
  `results/real_data_experiments/single_intersection_ablation_tensor/`
- 当前代码已支持 `--output-dir`，正式实验建议使用以下独立目录：
  `results/real_data_experiments/single_region_client_tensor_quick/`
  `results/real_data_experiments/single_region_client_tensor_main/`
  `results/real_data_experiments/single_region_client_tensor_seed15/`
  `results/real_data_experiments/single_region_client_tensor_seed42/`
  `results/real_data_experiments/single_region_client_tensor_seed48/`
  `results/real_data_experiments/single_region_ablation_tensor_main/`

## 9. 运行顺序建议

1. 先运行方案 A，验证固定 top-3 regions 与独立输出目录。
2. 再运行方案 B，作为主论文正式实验主结果来源。
3. 然后运行单区域消融正式实验，与方案 B 使用相同 top-5 regions。
4. 如需稳健性统计，再执行方案 C 的 3 个 seeds，并在后处理阶段汇总。

## 10. 本阶段冻结结论

- 正式 `pool_mode = sum_mean`
- 正式 `layout = standard`
- 正式 `tensor_path = data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式 `client = pooled-grid-region client`
- 正式 `FedAvg = standard sample-size weighted FedAvg`
- 正式 `selected_regions` 默认固定为 top-K active regions by `mean_total_flow`
- 正式单池化网格区域客户端推荐命名为“单池化网格区域客户端” / `single pooled-grid-region client`

## 11. 作者仍需确认

- 论文正文和图表标题中，是否统一采用“单池化网格区域客户端”作为中文正式名称。
- 主论文是否只报告方案 B，还是同时在附录中报告方案 C 的多 seed 汇总。
- 正式运行是否固定仅使用 CPU，还是后续允许在相同配置下切换 GPU 并单独说明设备差异。
- 正式结果表中，是否需要同时报告 `selected_regions_fixed_plan.csv` 的空间元数据字段。

## 12. 说明

- 本阶段只冻结 tensor-only 单池化网格区域客户端实验配置并生成运行计划。
- 本阶段未直接运行正式长训练。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未迁移区域实验。
- 本阶段未改变标准 `FedAvg` 主线。
