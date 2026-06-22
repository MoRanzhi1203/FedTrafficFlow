# 单路口客户端实验说明

## 当前状态

- 本目录已提供单路口客户端主实验的最小可运行版本。
- 数据入口已从 notebook 中缺失的 `6.池化网格张量.pt` 切换为 `data/analysis/node_intersection_flow_parquet/`。
- 主实验仅保留标准样本量加权 `FedAvg` 与 `Independent` 对比。

## 主要文件

- `sic_config.py`：实验配置与 CLI 参数解析。
- `sic_core.py`：真实数据读取、窗口构造、时间顺序划分、FedAvg/Independent 训练与结果导出。
- `sic_visualization.py`：只读取已有 CSV 结果并生成图表，不重新训练。

## 当前默认行为

- 自动选择活动度最高的若干节点作为客户端。
- 按时间顺序划分 `train/val/test`。
- 输出 `run_config.json`、`split_summary.json`、`main_metrics.csv`、`client_metrics.csv`、`convergence_history.csv`、`prediction_samples.csv` 等结果文件。

## 运行示例

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

## 说明

- 当前实现属于首次迁移交付版本，重点是修复可复现性、数据划分与标准 `FedAvg` 主线。
- 单路口消融、区域主实验、区域消融仍待继续迁移。
