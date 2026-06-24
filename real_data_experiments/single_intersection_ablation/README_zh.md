# 网格单元级客户端设置消融实验说明

## 当前状态

- 本目录用于承载网格单元级客户端设置的消融实验实现。
- 当前实现仅比较模型结构变体，不改变联邦聚合方式。
- 默认联邦聚合仍为标准样本量加权 `FedAvg`。
- 当前正式默认输入为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前正式名称为：网格单元级客户端联邦学习设置 / Grid-cell-level Client Federated Learning Setting。
- 当前 `client` 的含义为：每个客户端对应一个 active pooled region，也即 one pooled grid cell / one pooled grid region。

## 变体对应关系

- `Full`：`CNN + LSTM + Attention`
- `Without Attention`：对应原 notebook 的 `CNN + LSTM`
- `Without CNN / Spatial Encoder`：对应原 notebook 的 `LSTM + Attention`
- `Without LSTM`：对应原 notebook 的 `CNN + Attention`

## 文件说明

- `sia_config.py`：消融实验配置与 CLI。
- `sia_core.py`：正式 tensor 读取、active region 选择、时间顺序划分、四种结构变体的联邦训练与结果导出。
- `sia_visualization.py`：读取已有 CSV，生成消融对比图。

## 数据与划分

- 正式数据入口与单路口主实验一致：`final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 默认仅使用 active regions，并按 `channel 0` 平均总流量选择 top-K region 作为客户端。
- 训练、验证、测试采用 target time 的时间顺序划分。
- 不允许 `train/val/test` 复用。

## Legacy Fallback

- `data_mode = parquet` 仍保留，但仅作为历史 smoke test fallback。
- `parquet-direct` 不作为后续正式消融结果入口。

## 禁止进入主流程的内容

- `FedProx`
- `Proposed aggregation`
- `Loss-weighted aggregation`
- `Data-loss weighted aggregation`
- `server damping`
- `personalization`
- `adaptive aggregation`
- `similarity-aware aggregation`
- `quality-weighted aggregation`

## 运行示例

```bash
python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all
```
