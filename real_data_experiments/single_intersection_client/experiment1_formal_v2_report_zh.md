# 实验 1：grid_cell main full 正式重跑 v2 报告

## 1. 运行状态

- 正式实验 1 v2 已完成。
- 本次仅检查并审计实验 1：`grid_cell main full`
- 未运行新实验 2-6。

## 2. 输出目录

`results/real_data_experiments/formal/grid_cell_main_full_cuda_v2`

## 3. 核心文件检查

- `run_config.json`：存在
- `split_summary.json`：存在
- `main_metrics.csv`：存在
- `main_summary.csv`：存在
- `client_metrics.csv`：存在
- `convergence_history.csv`：存在
- `prediction_samples.csv`：存在
- `figure_index.csv`：不存在
- `figure_notes_zh.md`：不存在

说明：

- 正式评估所需核心结果文件已经齐全。
- 当前目录中不存在 `figure_index.csv` 与 `figure_notes_zh.md`，但不影响本次指标审计。

## 4. 主要指标

代码版本：

- commit hash：`dd648b9`
- commit message：`fix: normalize targets for grid cell main experiment`

主结果：

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 2.010767e+10 | 140086.892233 | 126900.491553 | 7.061248 | 6.992418 | -4.503150 |
| Independent | 7.003246e+09 | 80517.351878 | 67909.567491 | 3.774991 | 3.728347 | -0.002733 |

补充观察：

- 所有已读取结果文件均未出现 `NaN`
- 数值列均为有限值，未出现 `Inf`
- `R2` 已摆脱旧正式实验的 `-700` 级崩溃
- 但 `FedAvg` 的 `R2=-4.503150` 仍明显不理想

## 5. prediction_samples 检查

检查结论：

- `prediction_samples.csv` 同时包含 `FedAvg` 与 `Independent`
- `y_true` 与 `y_pred` 已处于同一原始尺度，均为百万级
- 不再出现旧正式实验中 `y_pred≈121` 的跨尺度错误

但仍存在新的明显问题：

- `FedAvg` 的 `y_pred` 在样本中恒为 `1824719.875`
- `Independent` 的 `y_pred` 在样本中恒为 `1958132.375`
- 即两种方法都仍表现为“按方法各自常数预测”

因此：

- 旧问题是“错误尺度的常数预测”，本次已修复
- 新结果变为“同尺度但仍接近常数预测”，说明训练有效性仍不足

## 6. sanity check 与 NaiveLastValue

`sic_sanity_check.py` 输出的对照结果：

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 2.010767e+10 | 140086.892233 | 126900.491553 | 7.061248 | 6.992418 | -4.503150 |
| Independent | 7.003246e+09 | 80517.351878 | 67909.567491 | 3.774991 | 3.728347 | -0.002733 |
| NaiveLastValue | 3.979079e+08 | 19419.217079 | 13619.880887 | 0.758148 | 0.758963 | 0.938585 |

结论：

- `NaiveLastValue` 明显优于 `FedAvg`
- `NaiveLastValue` 明显优于 `Independent`
- 当前 v2 虽然修复了 target normalization 缺失导致的尺度崩溃，但模型效果仍显著不如简单时间延续基线

## 7. 结论

- 实验 1 v2 已完成，并通过了“同尺度输出、无 NaN/Inf、摆脱 -700 级异常”的基础 sanity check
- 但实验 1 v2 未通过“优于合理 naive baseline”的有效性 sanity check
- 当前结果暂不建议作为论文候选正式结果
- 当前不建议进入实验 2

原因：

- 两种方法仍呈现明显常数预测
- `Independent` 虽优于 `FedAvg`，但仍远弱于 `NaiveLastValue`
- `FedAvg` 的 `R2` 仍明显为负，说明对真实变化的拟合能力不足

## 8. Git 边界

- `results/` 未进入 Git
- 未运行新实验 2-6
- 未修改 FedAvg
- 未修改模型结构
- 未修改数据划分
- 未修改 LaTeX
- 未修改 `simulation_experiments`

