# 真实数据训练入口审计

## 当前状态更新

- `single_intersection_client` / `single_intersection_ablation` 已完成 tensor-only Python 化。
- `region_client` / `region_ablation` 已完成 tensor-only Python 化，并已通过 smoke test。
- 当前 smoke test 结果仅用于验证代码链路、输出文件和可视化流程，不作为论文正式结果。

## 1. 当前正式入口结论

- 当前 `single_intersection_client` 与 `single_intersection_ablation` 的正式默认训练入口都已切换为 tensor-only。
- 当前正式 `tensor_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 当前正式 `regions_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 当前正式 `tensor shape = (2, 630, 5856)`
- 当前正式 `pool_mode = sum_mean`
- 当前正式 `layout = standard`
- 当前真实数据实验统一描述为两类客户端组织设置。
- 当前正式 `FedAvg = standard sample-size weighted FedAvg`
- `parquet-direct = legacy fallback only`

## 2. 命名修正结论

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示网格单元级客户端设置，而不是原始路口节点客户端。
- `single_intersection_client` / `single_intersection_ablation` 在文档中统一解释为网格单元级客户端联邦学习设置。
- `region_client` / `region_ablation` 在文档中统一解释为簇级客户端联邦学习设置。
- 代码目录名暂不调整，但正式文档中的设置名称统一为 grid-cell-level / cluster-level。

## 3. 上游链路与正式入口的关系

### 3.1 上游中间结果

- `data/analysis/node_intersection_flow_parquet/` 仍然是有效的上游节点流量中间结果。
- 它继续服务于节点流量审计、拟合、统计和必要的 legacy fallback。
- 它不再是当前网格单元级客户端设置实验的正式默认训练入口。

### 3.2 正式网格化与张量化产物

- `preprocessing_scripts/process_node_flow_grids.py`
- `preprocessing_scripts/process_node_flow_tensor.py`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_2ch.npy`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_pooled.npy`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor_metadata.json`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_metadata.json`

### 3.3 历史 notebook 命名处理

- `6.池化网格张量.pt` 属于历史 notebook 临时命名。
- 当前工程化代码不再生成该文件。
- 当前正式输出统一为 `node_flow_grid_tensor.pt`。

## 4. 当前冻结的训练语义

- `channel 0 = pooled total flow`
- `channel 1 = pooled mean flow`
- `R = 630` 表示 pooled grid regions
- `T = 5856` 表示 time steps
- 当前 `active_region_count = 223`
- 默认客户端语义为 pooled grid region，而不是单个原始路口节点。

## 5. 固定 region 选择结论

- 当前固定 region 推荐文件为 `real_data_experiments/selected_regions_fixed_plan.csv`。
- 默认规则：在 `active regions` 中按 `mean_total_flow` 降序选择 top-K。
- 正式实验不建议在每次运行时动态变更 region 集合。
- 推荐正式 top-3 regions：
  `290, 284, 318`
- 推荐正式 top-5 regions：
  `290, 284, 318, 288, 289`

## 6. 当前审计后的明确判断

### Q1. 当前正式单路口主实验与单路口消融实验是否已经使用 tensor-only 输入？

答：是，当前默认入口已经是 tensor-only。

### Q2. `node_intersection_flow_parquet` 现在是否仍是正式默认训练入口？

答：不是。它现在只作为上游中间结果和 legacy fallback。

### Q3. 当前是否还需要生成 `6.池化网格张量.pt`？

答：不需要。当前正式输出统一为 `node_flow_grid_tensor.pt`。

### Q4. 当前正式客户端是否等于原始路口节点？

答：不是。当前正式实验使用的是网格单元级客户端设置或簇级客户端设置，而不是原始路口节点客户端。

### Q5. 当前正式联邦主线是否改变？

答：没有。当前仍然是标准样本量加权 `FedAvg`。

## 7. 当前阶段范围控制

- 本文对应的是训练入口审计阶段文档，主要记录正式 tensor 数据入口与训练语义。
- 当前项目状态已经扩展到：网格单元级客户端设置与簇级客户端设置实验均完成 tensor-only Python 化。
- 当前已完成 smoke test 级别链路验证，尚未在本文中记录论文正式长训练结果。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未改变标准 `FedAvg` 主线。
