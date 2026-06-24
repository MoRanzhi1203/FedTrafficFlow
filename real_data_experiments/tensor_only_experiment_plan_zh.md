# Tensor-Only 网格单元级客户端设置实验配置冻结方案

## 当前状态更新

本文件最初用于冻结网格单元级客户端设置的 tensor-only 实验配置。后续阶段已经进一步完成 `region_client` 与 `region_ablation` 的 tensor-only Python 化迁移，并通过 smoke test。因此，本文中的历史阶段范围说明仅表示当时阶段边界，不代表当前项目最新状态。

当前最新状态为：

- `single_intersection_client` / `single_intersection_ablation` 已完成 tensor-only。
- `region_client` / `region_ablation` 已完成 tensor-only。
- `region_client` / `region_ablation` 已通过 smoke test。
- smoke test 结果只用于链路验证，不作为论文正式结果。
- smoke test 结果不作为论文正式结果。

## 1. 命名对照

| 代码目录 | 正式中文名称 | 正式英文名称 | 客户端定义 |
|---|---|---|---|
| `single_intersection_client` / `single_intersection_ablation` | 网格单元级客户端联邦学习设置 | Grid-cell-level Client Federated Learning Setting | 每个 client = 1 个 pooled grid cell |
| `region_client` / `region_ablation` | 簇级客户端联邦学习设置 | Cluster-level Client Federated Learning Setting | 每个 client = 一组 temporal-similarity-clustered grid cells |

## 2. 本阶段范围

- 本文件对应的历史阶段只冻结 tensor-only 网格单元级客户端设置实验配置，并生成正式运行计划。
- 本文件对应的历史阶段不直接运行正式长训练。
- 本文件对应的历史阶段尚未扩展到 `region_client` / `region_ablation` 的迁移实现。
- 本阶段不修改 LaTeX。
- 本阶段不修改 `simulation_experiments/`。
- 本阶段不改变标准样本量加权 `FedAvg` 主线。
- 本阶段不生成历史命名 `6.池化网格张量.pt`。
- 本阶段不把 smoke test 指标写成论文正式结果。

## 3. 正式数据入口冻结

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

## 4. 命名修正冻结

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示网格单元级客户端设置，而不再表示原始路口节点客户端。
- `single_intersection_client` / `single_intersection_ablation` 在文档中统一解释为：
  网格单元级客户端联邦学习设置 / Grid-cell-level Client Federated Learning Setting。
- `region_client` / `region_ablation` 在文档中统一解释为：
  簇级客户端联邦学习设置 / Cluster-level Client Federated Learning Setting。
- 在解释代码内部语义时，仍可补充说明：
  网格单元级客户端设置中每个 client = 1 个 pooled grid region；
  簇级客户端设置中每个 client = 一组 pooled grid regions。
- 为避免将客户端组织方式误解为新的联邦聚合机制，本文统一使用“setting”而非“mechanism”描述两类真实数据实验。
- 两类设置共享相同的 tensor-only 数据入口、模型结构、时间顺序切分和标准样本量加权 FedAvg 聚合流程，区别仅在于客户端组织粒度不同。
- 英文说明：
  To avoid implying a new federated aggregation algorithm, the two real-data experiments are described as client organization settings rather than mechanisms. Both settings share the same tensor-only input, temporal split protocol, local model architecture, and standard sample-size weighted FedAvg aggregation. The difference lies only in the granularity of client organization: in the grid-cell-level client setting, each client corresponds to a single pooled spatial grid cell, whereas in the cluster-level client setting, multiple grid cells with similar temporal patterns are grouped into the same client.

## 5. 联邦主线冻结

- 当前正式联邦聚合仍为标准样本量加权 `FedAvg`。
- 服务器聚合只依赖客户端训练样本量，不引入 `FedProx`、loss-weighted aggregation、personalization 或其他自适应聚合。
- 本阶段冻结的是 tensor-only 数据入口和网格单元级客户端设置配置，不改变联邦训练主算法。

## 6. Region 选择策略冻结

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

## 7. 正式实验候选配置

| 方案 | num_clients | rounds | local_epochs | batch_size | sequence_length | learning_rate | seed | selected_regions | 用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A | 3 | 5 | 3 | 32 | 12 | 0.001 | 42 | top-3 active regions by mean total flow | 快速正式实验 |
| B | 5 | 20 | 3 | 32 | 12 | 0.001 | 42 | top-5 active regions by mean total flow | 主论文正式实验 |
| C | 5 | 20 | 3 | 32 | 12 | 0.001 | 15, 42, 48 | fixed top-5 active regions | 稳健性 / 多 seed 实验 |

## 8. 推荐正式方案

- 默认推荐先执行方案 A，检查输出目录、图表链路和 CPU 运行时长。
- 主论文正式实验推荐采用方案 B。
- 若需要更稳健的表格和误差区间，补充执行方案 C，并对 `seed in [15, 42, 48]` 汇总均值与标准差。
- 单区域消融正式实验建议与方案 B 对齐，即固定 top-5 regions、`rounds = 20`、`local_epochs = 3`、`batch_size = 32`、`sequence_length = 12`、`learning_rate = 0.001`。

## 9. 输出目录保护

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

## 10. 运行顺序建议

1. 先运行方案 A，验证固定 top-3 regions 与独立输出目录。
2. 再运行方案 B，作为主论文正式实验主结果来源。
3. 然后运行单区域消融正式实验，与方案 B 使用相同 top-5 regions。
4. 如需稳健性统计，再执行方案 C 的 3 个 seeds，并在后处理阶段汇总。

## 11. 本阶段冻结结论

- 正式 `pool_mode = sum_mean`
- 正式 `layout = standard`
- 正式 `tensor_path = data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式采用两类客户端组织设置命名，而不是 `pooled-grid-region client` 这类旧称
- 正式 `FedAvg = standard sample-size weighted FedAvg`
- 正式 `selected_regions` 默认固定为 top-K active regions by `mean_total_flow`
- 正式命名统一为“网格单元级客户端联邦学习设置”与“簇级客户端联邦学习设置”
- 网格单元级客户端设置：每个 client = 1 个 pooled grid region
- 簇级客户端设置：每个 client = 一组 temporal-similarity-clustered pooled grid regions

## 12. 作者仍需确认

- 论文正文和图表标题中，是否统一采用“网格单元级客户端联邦学习设置 / 簇级客户端联邦学习设置”作为正式名称。
- 主论文是否只报告方案 B，还是同时在附录中报告方案 C 的多 seed 汇总。
- 正式运行是否固定仅使用 CPU，还是后续允许在相同配置下切换 GPU 并单独说明设备差异。
- 正式结果表中，是否需要同时报告 `selected_regions_fixed_plan.csv` 的空间元数据字段。

## 13. 说明

- 本文件记录的是网格单元级客户端设置配置冻结阶段。
- 当前项目状态已经推进到：网格单元级客户端设置与簇级客户端设置两条真实数据实验线均完成 tensor-only Python 化，并已通过 smoke test。
- 当前 smoke test 结果仅用于验证代码链路、输出文件和可视化流程，不作为论文正式结果。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未改变标准 `FedAvg` 主线。

## 14. 与簇级客户端实验的关系

- 网格单元级客户端设置：每个 client = 1 个 pooled grid region。
- 簇级客户端设置：每个 client = 一组 pooled grid regions。
- 两者都属于真实数据实验，但客户端粒度不同。
