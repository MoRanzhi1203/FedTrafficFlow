# 单路口消融实验说明

- 本实验仅比较模型结构变体，不改变标准样本量加权 FedAvg 聚合。
- 当前正式默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前客户端表示 pooled-grid-region client，并默认仅使用 active regions。
- 数据划分为按 target time 的时间顺序 train/val/test，不复用训练集、验证集与测试集。