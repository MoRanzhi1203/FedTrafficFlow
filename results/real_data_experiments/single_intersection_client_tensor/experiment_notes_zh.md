# 单路口客户端实验说明

- 当前主线方法始终为标准样本量加权 FedAvg。
- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前客户端表示 pooled-grid-region client，每个客户端对应一个 active pooled region。
- 数据划分按 target time 的时间顺序执行，不使用随机切分。
- `parquet-direct` 仅保留为 legacy fallback，不作为正式默认结果入口。