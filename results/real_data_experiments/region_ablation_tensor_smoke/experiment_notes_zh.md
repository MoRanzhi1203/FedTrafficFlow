# 区域消融实验说明

- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前 `region client` 表示一组 pooled grid regions。
- 当前消融实验仅比较模型结构变体，不改变标准样本量加权 FedAvg。
- 当前默认划分方法为 `spatial_block`。
- 数据划分按 target time 的时间顺序执行，不使用随机切分。