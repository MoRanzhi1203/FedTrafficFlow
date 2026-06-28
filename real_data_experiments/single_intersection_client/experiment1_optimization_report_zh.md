# 实验 1：grid_cell main full 指标异常排查与优化报告

## 1. 本阶段范围

- 本阶段只处理实验 1：`grid_cell main full`
- 实验目录：`real_data_experiments/single_intersection_client/`
- 不处理实验 2 `single_intersection_ablation`
- 不处理实验 3 `region_client`
- 不处理实验 4 `region_ablation`

## 2. 原始问题

旧正式实验 1 输出目录：`results/real_data_experiments/formal/grid_cell_main_full_cuda`

原始异常现象：

- `main_metrics.csv` 中 `FedAvg` / `Independent` 都异常
- `MAPE` 接近 `100`
- `SMAPE` 接近 `200`
- `R2` 约为 `-700`
- `client_metrics.csv` 显示 5 个 selected clients 全部异常，不是单个 client 偏差
- `convergence_history.csv` 显示 `train_loss` / `val_rmse` / `test_rmse` 20 轮几乎不动
- `prediction_samples.csv` 中 `y_true` 为约 `1.73e6 ~ 2.05e6`，而 `FedAvg y_pred` 几乎是常数 `121.161995`

原始正式配置确认：

- selected clients：`290, 284, 318, 288, 289`
- `sequence_length=12`
- `rounds=20`
- `local_epochs=3`
- `batch_size=32`
- `device=cuda`

## 3. 数据与预测 sanity check

使用 `sic_sanity_check.py` 对旧正式实验和原始 tensor 做审计后得到：

- selected clients 目标序列均为百万级，没有高比例零值
- 5 个 client 的 `y_train / y_val / y_test` 都处于约 `1.3e6 ~ 2.1e6` 范围
- `zero_ratio=0`
- `near_zero_ratio=0`
- 旧 `prediction_samples.csv` 仅保留了 `FedAvg` 的 200 条样本，且 `y_pred` 为小常数

旧正式实验的 naive baseline 结果：

| method | mse | rmse | mae | mape | smape | r2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 3.32035e+12 | 1.81856e+06 | 1.81668e+06 | 99.9933 | 199.973 | -700.764 |
| Independent | 3.32045e+12 | 1.81858e+06 | 1.81670e+06 | 99.9947 | 199.979 | -700.785 |
| NaiveLastValue | 3.97908e+08 | 1.94192e+04 | 1.36199e+04 | 0.758148 | 0.758963 | 0.938585 |

结论：

- 旧实验异常不是因为标签本身不可预测
- naive baseline 明显优于 FedAvg / Independent
- 主要问题出在训练目标尺度与训练动态，而不是 FedAvg 聚合公式

## 4. 问题判断

本次确认的主要异常原因属于：

- 目标尺度过大
- 训练阶段未做 target normalization
- 模型在百万级目标上几乎退化为常数预测
- `prediction_samples.csv` 旧实现按全表 `head(200)` 截断，导致只保留了 `FedAvg` 样本，不利于方法对比

本次未发现以下问题：

- 未发现 FedAvg 聚合公式异常，仍是标准样本量加权 FedAvg
- 未发现 `X/y` 时间错位
- 未发现 train/val/test split 口径错误
- 未发现评估时 raw `y_true` 与 normalized `y_pred` 直接混算的尺度错配 bug；旧问题更早发生在模型训练没有学到正确输出尺度
- 当前数据中 `y_true` 不接近 0，因此旧结果的 `MAPE/SMAPE` 爆炸不是零值放大，而是预测值严重失真

## 5. 实际修改

修改文件：

- `real_data_experiments/single_intersection_client/sic_core.py`
- `real_data_experiments/single_intersection_client/sic_config.py`
- 新增 `real_data_experiments/single_intersection_client/sic_sanity_check.py`

具体修改点：

- 在实验 1 中新增 `TargetScaler`
- 使用所有 selected clients 的 train split 目标值拟合全局 `mean/std`
- 训练阶段仅对 train targets 做 z-score normalization
- 验证、测试与导出预测时对模型输出做反归一化，保证 `y_true / y_pred` 始终处于原始尺度
- 在 `split_summary.json` / `run_config.json` 中记录 target normalization 开关与统计量
- 导出 `prediction_samples.csv` 时改为按方法均衡抽样，避免再次只保留 `FedAvg` 样本
- 新增 `sic_sanity_check.py`，用于输出目标分布、旧预测统计和 naive baseline 对比

## 6. 不改变的内容

- 未修改 FedAvg
- 未修改模型结构
- 未修改数据划分
- 未修改指标计算主口径
- 本次没有修改 `MAPE/SMAPE` epsilon，因为当前实验 1 数据不存在零值敏感主导问题
- 未修改实验 2/3/4

## 7. smoke 结果

smoke 参数：

- `num_clients=2`
- `selected_clients=290,284`
- `rounds=2`
- `local_epochs=1`
- `batch_size=32`
- `sequence_length=12`
- `device=cpu`
- output：`results/real_data_experiments/formal/experiment1_optimization_smoke`

smoke 关键结果：

| method | mse | rmse | mae | mape | smape | r2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 9.707584e+09 | 98344.546070 | 86209.987984 | 4.537760 | 4.510299 | -0.072648 |
| Independent | 9.168772e+09 | 95354.056314 | 80470.643700 | 4.272533 | 4.214005 | -0.001651 |
| NaiveLastValue | 5.231129e+08 | 22777.883330 | 15971.359073 | 0.846404 | 0.847219 | 0.942835 |

smoke sanity 结论：

- `main_metrics.csv` 存在
- `prediction_samples.csv` 存在
- `prediction_samples.csv` 中同时包含 `FedAvg` 和 `Independent`
- `y_true / y_pred` 已处于同一原始尺度
- 未出现 `NaN/Inf`
- 与旧正式实验相比，最致命的尺度崩溃已经消失
- 但在这个极小 smoke 下，FedAvg / Independent 仍明显弱于 naive baseline，且预测仍偏近常数

## 8. 是否建议重跑正式实验 1

- 可以重跑正式实验 1

说明：

- 当前已确认并修复导致旧正式结果崩溃的主要问题：target normalization 缺失
- 小规模 smoke 已验证输出尺度恢复正常，且不再出现 `R2=-700` 级别的异常
- 由于 smoke 仅使用 `2 clients / 2 rounds / 1 local epoch`，它只用于确认修复有效，不足以代表正式 full 性能上限
- 正式重跑后仍应把 naive baseline 一并作为 sanity reference；如果 full 结果仍显著落后于 naive baseline，再继续做只读排查

