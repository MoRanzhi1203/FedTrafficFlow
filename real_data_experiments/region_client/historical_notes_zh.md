# 区域主实验历史探索记录

## 来源说明

- 本文记录 `test/区域客户端计算_3×2_最终版.ipynb` 中存在、但不进入当前默认主流程的历史探索逻辑。
- 这些内容来自真实实验 notebook，但属于历史探索或非主线设置。
- 它们不进入默认主流程，不作为论文主方法。
- 当前正式默认实验仍采用标准样本量加权 `FedAvg`。

## 历史探索内容

| notebook 中的内容 | 处理方式 | 是否进入默认主流程 |
|---|---|---|
| `FedProx` proximal term | 仅文档记录，正式代码默认不调用 | 否 |
| `server damping` / `server_lr` | 仅文档记录，正式代码默认不调用 | 否 |
| `personalization` 微调 | 仅文档记录，正式代码默认不调用 | 否 |
| mixed raw-scale loss | 仅文档记录，正式代码默认不调用 | 否 |
| cluster-balanced split 历史实现 | 迁移为可选划分思路说明 | 否，默认用 `spatial_block` |
| 随机 `split_indices()` | 仅记录问题，不迁移默认流程 | 否 |

## 备注

- 如后续确需保留历史代码，只能放入 `legacy` / `exploratory` 风格函数，并默认不调用。
- 当前正式区域主实验只保留：
  tensor-only 输入、区域客户端、多 region 数据集、时间顺序划分、标准 `FedAvg`、Independent baseline。
