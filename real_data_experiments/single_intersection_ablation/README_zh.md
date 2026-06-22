# 单路口客户端消融实验说明

## 当前状态

- 本目录用于迁移 `test/单路口客户端消融实验_2×2.ipynb`。
- 当前实现仅比较模型结构变体，不改变联邦聚合方式。
- 默认联邦聚合仍为标准样本量加权 `FedAvg`。

## 变体对应关系

- `Full`：`CNN + LSTM + Attention`
- `Without Attention`：对应原 notebook 的 `CNN + LSTM`
- `Without CNN / Spatial Encoder`：对应原 notebook 的 `LSTM + Attention`
- `Without LSTM`：对应原 notebook 的 `CNN + Attention`

## 文件说明

- `sia_config.py`：消融实验配置与 CLI。
- `sia_core.py`：真实数据读取、时间顺序划分、四种结构变体的联邦训练与结果导出。
- `sia_visualization.py`：读取已有 CSV，生成消融对比图。

## 数据与划分

- 数据入口与单路口主实验一致：`data/analysis/node_intersection_flow_parquet/`。
- 训练、验证、测试采用时间顺序划分。
- 不允许 `train/val/test` 复用。

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
