# 新实验 1：单个网格作为单个客户端的对比实验

## 当前定位

- 本目录固定对应新实验 1：`single grid client comparison`。
- 旧新映射：原实验 1 -> 新实验 1。
- 客户端定义固定为：`client_i = grid_cell_i`。
- 该实验线只允许一个 grid cell 对应一个 client，不能把多个 grid cells 合并进同一个 client。
- 主实验仅保留标准样本量加权 `FedAvg`、`Independent` 与 `NaiveLastValue` 对比。

## 目录边界

- 该目录只承载单个网格作为单个客户端的对比实验。
- 它不是多个相似网格合并为一个客户端的实验线。
- 它也不是全局所有网格按相似度划分为客户端的实验线。
- 后续 grouped-client 与 global-partition 实验统一由新实验 3-6 承接。

## 主要文件

- `sic_config.py`：实验配置与 CLI 参数解析。
- `sic_core.py`：正式 tensor 读取、active region 选择、时间顺序划分、FedAvg/Independent/NaiveLastValue 训练与结果导出。
- `sic_visualization.py`：只读取已有 CSV 结果并生成图表，不重新训练。

## 默认行为

- 默认输入：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 默认从 active pooled regions 中选择 `top-K` 单网格客户端。
- 默认按 target time 做连续 `train/val/test` 划分，不允许窗口跨 split 泄漏。
- 输出 `selected_regions.csv`、`run_config.json`、`split_summary.json`、`main_metrics.csv`、`client_metrics.csv`、`convergence_history.csv`、`prediction_samples.csv` 等结果文件。

## 结果路径归属

- `results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/` 归入新实验 1 的正式结果。
- `results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r40_cuda/`、`experiment1_fedavg_rounds_smoke_r60_cuda/`、`experiment1_metric_opt_k5_r80_e2_lr5e4_cuda/` 等归入新实验 1 的诊断或 smoke 结果。
- 旧路径命名保持不动，只在文档中新增新编号对应关系。

## Legacy Fallback

- `data_mode = parquet` 仍保留，但仅作为历史 smoke fallback。
- `parquet-direct` 不作为正式实验结果入口。

## 运行示例

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

