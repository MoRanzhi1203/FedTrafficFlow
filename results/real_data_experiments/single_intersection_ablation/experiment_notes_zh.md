# 单路口消融实验说明

- 本实验仅比较模型结构变体，不改变标准样本量加权 FedAvg 聚合。
- 数据入口与单路口主实验一致，均使用 data/analysis/node_intersection_flow_parquet。
- 数据划分为时间顺序 train/val/test，不复用训练集、验证集与测试集。