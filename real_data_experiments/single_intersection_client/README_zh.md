# 网格单元级客户端设置主实验说明

## 当前状态

- 本目录已提供网格单元级客户端设置主实验的 tensor-only 版本。
- 当前正式默认输入为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前正式名称为：网格单元级客户端联邦学习设置 / Grid-cell-level Client Federated Learning Setting。
- 当前 `client` 的含义为：每个客户端对应一个 active pooled region，也即 one pooled grid cell / one pooled grid region。
- 主实验仅保留标准样本量加权 `FedAvg` 与 `Independent` 对比。

## 主要文件

- `sic_config.py`：实验配置与 CLI 参数解析。
- `sic_core.py`：正式 tensor 读取、active region 选择、时间顺序划分、FedAvg/Independent 训练与结果导出。
- `sic_visualization.py`：只读取已有 CSV 结果并生成图表，不重新训练。

## 当前默认行为

- 默认 `data_mode = tensor`，并从正式 tensor 中选择 active pooled regions。
- 默认按 `channel 0` 的平均总流量从高到低选择 top-K region 作为客户端。
- 按 target time 的时间顺序划分 `train/val/test`，不允许窗口跨 split 泄漏。
- 输出 `selected_regions.csv`、`run_config.json`、`split_summary.json`、`main_metrics.csv`、`client_metrics.csv`、`convergence_history.csv`、`prediction_samples.csv` 等结果文件。

## Legacy Fallback

- `data_mode = parquet` 仍保留，但仅作为历史 smoke test fallback。
- `parquet-direct` 不作为后续正式训练入口，也不应作为论文正式结果。

## 运行示例

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

## 说明

- 当前实现已切换到正式 tensor-only 输入，重点是修复可复现性、数据划分与标准 `FedAvg` 主线。
- 为避免把客户端组织方式误解为新的联邦聚合机制，文档统一使用“setting”描述该实验线。
