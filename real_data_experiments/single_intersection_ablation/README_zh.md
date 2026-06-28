# 新实验 2：单个网格作为单个客户端的消融实验

## 当前定位

- 本目录固定对应新实验 2：`single grid client ablation`。
- 旧新映射：原实验 2 -> 新实验 2。
- 客户端定义与新实验 1 完全一致：`client_i = grid_cell_i`。
- 本目录只比较模型结构变体，不改变联邦聚合方式。
- 默认联邦聚合仍为标准样本量加权 `FedAvg`。

## 目录边界

- 该目录仍然属于单个网格作为单个客户端实验线。
- 不能把多个 grid cells 合并进同一个 client。
- grouped-client 与 global-partition 的实验语义不写入本目录。

## 变体对应关系

- `Full`：`CNN + LSTM + Attention`
- `Without Attention`：对应原 notebook 的 `CNN + LSTM`
- `Without CNN / Spatial Encoder`：对应原 notebook 的 `LSTM + Attention`
- `Without LSTM`：对应原 notebook 的 `CNN + Attention`

## 文件说明

- `sia_config.py`：消融实验配置与 CLI。
- `sia_core.py`：正式 tensor 读取、active region 选择、时间顺序划分、四种结构变体的联邦训练与结果导出。
- `sia_visualization.py`：读取已有 CSV，生成消融对比图。

## 数据与结果归属

- 正式数据入口与新实验 1 一致：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- `results/real_data_experiments/single_intersection_ablation/` 与 `results/real_data_experiments/single_intersection_ablation_tensor/` 统一归入新实验 2。
- 旧 results 路径不移动，只新增新编号说明。

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
