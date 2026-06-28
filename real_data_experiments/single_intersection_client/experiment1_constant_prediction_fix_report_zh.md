# 实验 1：常数预测问题排查与修复报告

## 1. 本阶段范围

- 本阶段只处理实验 1：`grid_cell main full`
- 代码范围限定在 `real_data_experiments/single_intersection_client/` 及实验 1 直接调用的必要公共逻辑
- 未处理实验 2/3/4

## 2. v2 遗留问题

实验 1 v2 已经修复了 target 未归一化导致的尺度崩溃，但仍存在明显常数预测：

- `FedAvg y_pred` 恒为 `1824719.875`
- `Independent y_pred` 恒为 `1958132.375`
- `NaiveLastValue` 显著优于 `FedAvg` 和 `Independent`
- 因此 v2 暂不适合作为论文正式结果

## 3. 诊断过程

本次按“先证据、后修复”执行：

- 检查 `prediction_samples.csv` 导出链路，确认其直接来自 `collect_predictions()` 拼接后的整段 `y_pred`，不是 `item()`、`mean()` 或单 batch 误写
- 检查 v2 输出，确认 `prediction_samples.csv` 中两个方法的 `unique y_pred count` 都为 `1`
- 检查 `convergence_history.csv`，发现旧实现下 FedAvg 的 `train_loss` 只有小幅波动，整体停留在近常数状态
- 检查 `X/y` shape，确认训练 batch 中 `x=(32,2,12)`、`y=(32,1)`、`pred=(32,1)`，没有直接广播错配证据
- 新增 `sic_constant_prediction_diagnosis.py` 做运行时诊断：
  - 输出每个 client 的 `X / y` 统计
  - 输出 `y_train / y_val / y_test` 与 `y_norm` 统计
  - 输出单 batch 的 `pred` 方差、loss
  - 输出 5 个 mini-batch 的 `grad_norm`、`update_norm`、`pred_std`
- 运行证据表明：
  - `grad_norm` 与 `update_norm` 持续大于 0，说明训练循环和参数更新本身没有失效
  - 但 `batch_x` 量级约为 `9.4e5`，而 `y_norm` 只在 `[-3, 2]` 左右
  - 模型输出的 `pred_denorm_std` 只有约 `436`，远小于真实目标波动

## 4. 定位原因

常数预测的直接原因是：

- 实验 1 v2 只对目标 `y` 做了 train-split target normalization
- 输入 `X` 仍保持百万级原始量纲
- 在“输入巨大、目标已标准化”的尺度失衡下，当前 CNN+LSTM+Attention 回归器虽然参数会更新，但前向输出退化为近常数低方差预测
- 因此常数预测不是导出错误，也不是训练循环完全未执行，而是训练输入尺度问题导致的模型退化

## 5. 实际修改

修改文件：

- `real_data_experiments/single_intersection_client/sic_core.py`
- `real_data_experiments/single_intersection_client/sic_config.py`
- 新增 `real_data_experiments/single_intersection_client/sic_constant_prediction_diagnosis.py`
- 新增 `real_data_experiments/single_intersection_client/sic_run_monitor.py`

具体修改点：

- 为实验 1 增加基于 train split 拟合的输入归一化 `InputScaler`
- 对 train/val/test 的输入特征统一应用相同的 train-split z-score normalization
- 保留目标 `y` 的 train-split target normalization，并保持评估和导出时反归一化到原始尺度
- 在 `split_summary.json` 和输出目录中记录 `input_normalization` 与 `input_scaler.json`
- 保持 `prediction_samples.csv` 仍直接来源于真实模型预测
- 增加 `tqdm` 进度条和实时 `flush=True` 日志：
  - FedAvg round 级进度
  - FedAvg client / epoch / batch 级进度
  - Independent client / epoch / batch 级进度
  - 实验启动、配置、训练开始/结束、结果写出路径日志
- 新增只读 `sic_run_monitor.py`，用于查看 output_dir、结果文件、python 进程与 GPU 状态，不启动第二个训练进程

## 6. 不改变的内容

- 未修改 FedAvg 聚合公式
- 未修改模型结构
- 未修改数据划分
- 未修改实验 2/3/4
- 未修改 LaTeX
- 未修改 `simulation_experiments`

## 7. smoke 结果

smoke 命令：

```bash
python -m real_data_experiments.single_intersection_client.sic_core \
  --workflow all \
  --data-mode tensor \
  --device cpu \
  --num-clients 2 \
  --rounds 3 \
  --local-epochs 2 \
  --batch-size 32 \
  --sequence-length 12 \
  --learning-rate 0.001 \
  --selected-clients 290,284 \
  --seed 42 \
  --output-dir results/real_data_experiments/formal/experiment1_constant_fix_smoke
```

输出目录：

- `results/real_data_experiments/formal/experiment1_constant_fix_smoke`

主要结果：

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 6.458804e+08 | 25338.983826 | 19555.904579 | 1.029203 | 1.028686 | 0.929059 |
| Independent | 4.919655e+08 | 22158.339880 | 16579.946459 | 0.875379 | 0.874532 | 0.945308 |

关键验证结果：

- `y_true / y_pred` 同尺度，均为百万级原始尺度
- `prediction_samples.csv` 中：
  - `FedAvg` 的 `unique y_pred count = 100`
  - `Independent` 的 `unique y_pred count = 100`
- 即修复后 `y_pred` 不再是每个 method 只有 1 个唯一值
- `convergence_history.csv` 中 `train_loss` 从 `0.339532` 下降到 `0.049025`
- 指标未出现 `NaN / Inf`

## 8. 是否建议重跑正式实验 1 v3

- 可以重跑正式实验 1 v3

原因：

- 当前 smoke 已经证明常数预测问题被修复
- 输出尺度正确
- 训练损失有明显下降
- `prediction_samples.csv` 反映的是真实非常数预测，而不是导出伪像

## 9. 进度条与运行可视化

- 已使用 `tqdm`
- 已增加 round / client / epoch / batch 级进度显示
- 已增加实时日志并使用 `flush=True`
- 已新增只读 monitor：`sic_run_monitor.py`
- 未弹出独立 PowerShell/CMD 窗口，本次在当前终端内验证进度显示
- 已确认没有重复启动训练进程
- 已确认进度显示不影响结果文件生成

