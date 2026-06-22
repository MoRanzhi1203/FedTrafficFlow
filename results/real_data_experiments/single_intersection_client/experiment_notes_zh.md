# 单路口客户端实验说明

- 本次迁移默认主方法为标准样本量加权 FedAvg。
- 数据入口为 data/analysis/node_intersection_flow_parquet，而非 notebook 中缺失的 6.池化网格张量.pt。
- 数据划分采用时间顺序 train/val/test，不使用随机打乱。
- 当前最小交付版本先完成单路口主实验主线与指标导出。