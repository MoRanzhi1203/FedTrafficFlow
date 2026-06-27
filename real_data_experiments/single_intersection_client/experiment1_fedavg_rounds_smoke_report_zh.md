# 实验 1：FedAvg rounds=40 诊断 smoke 报告

## 1. 诊断目的

- 本轮只验证增加 communication rounds 是否改善 FedAvg。
- 本轮不是正式 full。
- 本轮不作为论文正式结果。

## 2. 运行边界

- 只处理实验 1。
- 未运行实验 2/3/4。
- 未修改 FedAvg 聚合公式。
- 未修改模型结构。
- 未修改数据划分。
- 未修改 LaTeX。
- 未修改 simulation_experiments。
- 未提交 results。

## 3. CLI 情况

- `sic_core` 不支持 FedAvg-only workflow。
- 本轮使用 `--workflow all`。
- `Independent` 是流程附带生成，不作为本轮重点。

## 4. 参数设置

- v4：`rounds=20`
- r40 smoke：`rounds=40`
- 其他参数保持一致：
  - `num_clients=5`
  - `local_epochs=3`
  - `batch_size=32`
  - `sequence_length=12`
  - `learning_rate=0.001`
  - `selected_clients=290,284,318,288,289`
  - `seed=42`
  - `device=cuda`

## 5. 指标对比

| 配置 | method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 | FedAvg | 445713657.805219 | 20815.803975 | 16604.522810 | 0.932170 | 0.929825 | 0.873045 |
| r40 smoke | FedAvg | 372583193.968334 | 19177.795050 | 14638.467264 | 0.819097 | 0.817484 | 0.913491 |
| baseline | NaiveLastValue | 397907880.378356 | 19419.217079 | 13619.880887 | 0.758148 | 0.758963 | 0.938585 |
| v4 参考 | Independent | 224696815.731175 | 14693.368581 | 10501.618629 | 0.587359 | 0.586664 | 0.962666 |

重点结论：

- r40 相比 v4，FedAvg `RMSE` 从 `20815.803975` 降到 `19177.795050`，改善约 `7.87%`。
- r40 相比 v4，FedAvg `MAE` 从 `16604.522810` 降到 `14638.467264`，改善约 `11.84%`。
- r40 相比 v4，FedAvg `MAPE` 从 `0.932170` 降到 `0.819097`，改善约 `12.13%`。
- r40 相比 v4，FedAvg `R2` 从 `0.873045` 提升到 `0.913491`，提升 `0.040446`。
- r40 的 FedAvg 已明显接近 `NaiveLastValue`，并且 `RMSE` 已优于 naive：`19177.795050 < 19419.217079`。
- 但 r40 的 FedAvg 仍未全面超过 naive：`MAE`、`MAPE`、`SMAPE`、`R2` 仍落后于 naive。

## 6. 收敛性分析

- r40 最后 10 轮 `train_loss` 从 `0.011557` 继续下降到 `0.011313`。
- r40 最后 10 轮 `val_rmse` 从 `22331.745713` 整体下降到 `21463.945901`。
- 中间存在轻微波动，例如第 33-34 轮和第 36-37 轮，但总体趋势仍然向下。
- 这说明 `rounds=20` 确实偏少，增加 rounds 对 FedAvg 有实际帮助。
- 但到第 40 轮时下降速度已经放缓，说明已经接近平台期，后续即使继续加 rounds，收益也可能小于 `20 -> 40` 这一阶段。

## 7. prediction_samples 检查

- `y_true / y_pred` 同尺度，均为原始百万级流量尺度。
- r40 FedAvg 的 `y_pred std = 72769.858071`，已比 v4 的 `70090.001181` 更接近 `y_true std = 75667.319159`，说明预测方差压缩有所缓解。
- `y_pred` 不是常数：
  - r40 FedAvg `unique y_pred count = 100`
  - r40 Independent `unique y_pred count = 100`
- `main_metrics.csv`、`convergence_history.csv`、`prediction_samples.csv` 中数值均为有限值，没有 `NaN / Inf`。

## 8. 结论

- 增加 `rounds=40` 是有效的，FedAvg 相比 v4 `rounds=20` 有明显改善。
- r40 已经显著缩小了 FedAvg 与 `NaiveLastValue` 的差距，并在 `RMSE` 上略优于 naive。
- 但 r40 还没有在 `MAE`、`MAPE`、`SMAPE`、`R2` 上全面超过 naive，因此还不能据此认定“只靠增加 rounds 就足以解决 FedAvg gap”。
- 结合收敛尾部仍在缓慢下降，值得继续做一次 `rounds=60` 的小规模 smoke。
- 当前仍不建议进入实验 2。

## 9. 下一步建议

- 下一步优先设计 `rounds=60` smoke，继续只验证“增加 communication rounds 是否还能进一步缩小 FedAvg gap”。
- 如果 `r60` 继续改善但仍无法全面超过 naive，再转向 client 异质性 / 分组审计。
- 在没有完成下一轮 smoke 前，不建议直接跑新的正式确认实验。
