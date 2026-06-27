# 实验 1：grid_cell main full 正式重跑 v3 报告

## 1. 本次运行范围

- 本次只运行实验 1：`grid_cell main full`
- 未运行实验 2 / 3 / 4

## 2. 使用代码版本

- commit hash：`c5ab36d`
- commit message：`fix: normalize inputs for grid cell main experiment`

## 3. 修复背景

- v1 问题：目标 `y` 未归一化，`y_true` 为百万级，`y_pred` 约为 `121`，发生尺度崩溃
- v2 问题：加入 target normalization 后尺度恢复，但 `FedAvg` 与 `Independent` 仍退化为常数预测
- v3 修复：在保留 target normalization 的基础上，为实验 1 增加 train-split 输入归一化，并加入 `tqdm` 进度条、实时 `[INFO]` 日志和只读 monitor 脚本

## 4. 运行命令

```bash
python -m real_data_experiments.single_intersection_client.sic_core \
  --workflow all \
  --data-mode tensor \
  --device cuda \
  --num-clients 5 \
  --rounds 20 \
  --local-epochs 3 \
  --batch-size 32 \
  --sequence-length 12 \
  --learning-rate 0.001 \
  --selected-clients 290,284,318,288,289 \
  --seed 42 \
  --show-progress \
  --output-dir results/real_data_experiments/formal/grid_cell_main_full_cuda_v3
```

补充说明：

- `run_config.json` 记录的配置设备为 `cuda`
- 但正式运行日志显示 `[INFO] Experiment started with device=cpu`
- 因此本次 v3 实际执行设备为 CPU fallback，而不是预期的 CUDA

## 5. 输出目录

- `results/real_data_experiments/formal/grid_cell_main_full_cuda_v3`

## 6. 核心配置

- `num_clients=5`
- `selected_clients=290,284,318,288,289`
- `rounds=20`
- `local_epochs=3`
- `batch_size=32`
- `sequence_length=12`
- `learning_rate=0.001`
- `seed=42`
- 请求设备：`cuda`
- 实际运行设备：`cpu`
- `target normalization=True`
- `input normalization=True`
- `show_progress=True`

## 7. 主要指标

主结果：

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 4.820293e+08 | 21669.184427 | 17419.790785 | 0.976708 | 0.974260 | 0.864100 |
| Independent | 2.251983e+08 | 14709.816236 | 10507.914932 | 0.587672 | 0.586940 | 0.962633 |
| NaiveLastValue | 3.979079e+08 | 19419.217079 | 13619.880887 | 0.758148 | 0.758963 | 0.938585 |

对比结论：

- `FedAvg` 相比 v2 明显改善，`R2` 从负值恢复到 `0.864100`
- `Independent` 相比 v2 明显改善，`R2` 达到 `0.962633`
- `Independent` 优于 `NaiveLastValue`
- `FedAvg` 仍落后于 `NaiveLastValue`

## 8. prediction_samples 检查

检查结果：

- `y_true / y_pred` 同尺度，均为原始百万级尺度
- `prediction_samples.csv` 中不再出现常数预测
- `unique y_pred count`：
  - `FedAvg = 100`
  - `Independent = 100`
- `FedAvg y_pred` 范围：
  - `min=1753243.875`
  - `max=2012818.125`
- `Independent y_pred` 范围：
  - `min=1739478.875`
  - `max=2035755.500`

结论：

- v3 已修复 v2 的常数预测问题
- 当前 `prediction_samples.csv` 反映的是真实非常数预测

## 9. convergence 检查

检查结果：

- `convergence_history.csv` 已生成
- `train_loss` 从首轮 `0.131268` 下降到末轮 `0.011947`
- `val_rmse` 从首轮 `100268.356295` 下降到最终约 `23575.289954`
- 所有数值均为有限值，未出现 `NaN / Inf`

结论：

- 训练过程存在明显收敛迹象
- 未再出现 v2 中训练输出退化为常数的失效现象

## 10. 进度条与运行可视化

- 已使用 `tqdm`
- 终端已出现 `FedAvg rounds`
- 终端已出现 `Independent clients`
- 终端已出现实时 `[INFO]` 日志
- 未启动只读 monitor 窗口
- 已确认没有重复训练进程

## 11. 结论

- 实验 1 v3 通过了“尺度正确、无常数预测、R2 为正、无 NaN/Inf”的 sanity check
- 但当前不建议直接作为论文候选正式结果，原因有两点：
  - 本次命令请求 `cuda`，实际却以 `cpu` fallback 完成，和目标正式实验设置不一致
  - `FedAvg` 虽显著优于 v2，但仍落后于 `NaiveLastValue` baseline
- 当前不建议进入实验 2
- 更稳妥的下一步是继续只处理实验 1，先查明本机为何没有按预期使用 CUDA，并在确认 CUDA 正常后重跑正式实验 1 v3 或 v4

## 12. Git 边界

- `results/` 未提交
- 未运行实验 2 / 3 / 4
- 未修改 FedAvg
- 未修改模型结构
- 未修改数据划分
- 未修改 LaTeX
- 未修改 `simulation_experiments`
