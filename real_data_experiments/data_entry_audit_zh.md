# 真实数据训练入口审计

## 1. 当前正式入口结论

- 当前 `single_intersection_client` 与 `single_intersection_ablation` 的正式默认训练入口都已切换为 tensor-only。
- 当前正式 `tensor_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 当前正式 `regions_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 当前正式 `tensor shape = (2, 630, 5856)`
- 当前正式 `pool_mode = sum_mean`
- 当前正式 `layout = standard`
- 当前正式 `client = pooled-grid-region client`
- 当前正式 `FedAvg = standard sample-size weighted FedAvg`
- `parquet-direct = legacy fallback only`

## 2. 命名修正结论

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示 pooled grid region client，而不是原始路口节点客户端。
- 建议后续论文、报告和图表统一使用“单池化网格区域客户端”。
- 英文建议统一使用 `single pooled-grid-region client`。
- 代码目录名暂不调整，但文档解释必须按上述语义理解。

## 3. 上游链路与正式入口的关系

### 3.1 上游中间结果

- `data/analysis/node_intersection_flow_parquet/` 仍然是有效的上游节点流量中间结果。
- 它继续服务于节点流量审计、拟合、统计和必要的 legacy fallback。
- 它不再是当前单池化网格区域客户端实验的正式默认训练入口。

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

答：不是。当前正式客户端是 pooled-grid-region client。

### Q5. 当前正式联邦主线是否改变？

答：没有。当前仍然是标准样本量加权 `FedAvg`。

## 7. 当前阶段范围控制

- 本阶段只冻结 tensor-only 单池化网格区域客户端实验配置并生成运行计划。
- 本阶段未直接运行正式长训练。
- 本阶段未迁移区域实验。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未改变标准 `FedAvg` 主线。
