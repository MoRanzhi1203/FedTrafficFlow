# 区域客户端实验说明

- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前 `region client` 表示一组 pooled grid regions，而不是单个原始路口节点。
- 当前默认划分方法为 `spatial_block`；`flow_kmeans` 仅为可选区域划分方法。
- 当前主线方法始终为标准样本量加权 FedAvg。
- 当前保留 Independent baseline 作为对比方法。
- 数据划分按 target time 连续执行，不进行随机切分。