# 簇级客户端设置消融实验历史说明

## 来源说明

- 本文记录 `test/区域客户端消融实验_2×2_最终版.ipynb` 的历史信息。
- 这些内容来自真实实验 notebook，但不改变当前正式主流程。
- 当前正式默认实验仍采用标准样本量加权 `FedAvg`。

## 历史内容处理

| notebook 中的内容 | 处理方式 | 是否进入默认主流程 |
|---|---|---|
| cluster-balanced split 历史实现 | 迁移为可选 `flow_kmeans` 思路说明 | 否，默认用 `spatial_block` |
| 原 notebook 的等权 `fedavg()` | 仅作为历史实现记录 | 否，正式版改为样本量加权 `FedAvg` |
| 原 notebook 的消融命名 `w/o Attn / w/o CNN / w/o LSTM` | 在 README 和代码中规范映射 | 是，作为正式消融变体 |

## 备注

- 当前簇级客户端设置消融主流程只保留：
  tensor-only 输入、簇级客户端设置、多 grid-cell 时间窗口数据集、时间顺序切分、标准 `FedAvg`、四种结构消融变体。
