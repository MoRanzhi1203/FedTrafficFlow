# 实验 1：FedAvg 多指标优化 smoke 报告

## 1. 目的

本报告用于判断 FedAvg 是否能在不改变聚合公式和模型结构的前提下，通过 rounds、local_epochs、learning_rate 和 client 组织诊断，改善相对 NaiveLastValue 的 MAE、MAPE、SMAPE 和 R2。

## 2. 边界

- 不删除 NaiveLastValue；
- 不删除 289；
- 不修改 FedAvg；
- 不修改模型结构；
- 不修改数据划分；
- 不进入实验 2/3/4；
- 所有结果均为 diagnostics/smoke。

## 3. 候选方案

| name | selected_clients | K | rounds | local_epochs | learning_rate | output_dir | 是否完成 | device |
|---|---|---|---|---|---|---|---|---|
| experiment1_metric_opt_k5_r80_e1_lr5e4_cuda | 290,284,318,288,289 | 5 | 80 | 1 | 0.0005 | E:\Jupter_Notebook\FedTrafficFlow\results\real_data_experiments\diagnostics\experiment1_metric_opt_k5_r80_e1_lr5e4_cuda | 是 | cuda |
| experiment1_metric_opt_k5_r80_e2_lr5e4_cuda | 290,284,318,288,289 | 5 | 80 | 2 | 0.0005 | E:\Jupter_Notebook\FedTrafficFlow\results\real_data_experiments\diagnostics\experiment1_metric_opt_k5_r80_e2_lr5e4_cuda | 是 | cuda |
| experiment1_metric_opt_k5_r100_e1_lr5e4_cuda | 290,284,318,288,289 | 5 | 100 | 1 | 0.0005 | E:\Jupter_Notebook\FedTrafficFlow\results\real_data_experiments\diagnostics\experiment1_metric_opt_k5_r100_e1_lr5e4_cuda | 是 | cuda |
| experiment1_metric_opt_k4_exclude289_r80_e1_lr5e4_cuda | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | E:\Jupter_Notebook\FedTrafficFlow\results\real_data_experiments\diagnostics\experiment1_metric_opt_k4_exclude289_r80_e1_lr5e4_cuda | 未运行 / 目录不存在 | 未运行 / 目录不存在 |

## 4. 主指标对比

| name | FedAvg_MSE | FedAvg_RMSE | FedAvg_MAE | FedAvg_MAPE | FedAvg_SMAPE | FedAvg_R2 | Naive_RMSE | Naive_MAE | Naive_MAPE | Naive_SMAPE | Naive_R2 | RMSE优于Naive | MAE优于Naive | MAPE优于Naive | SMAPE优于Naive | R2优于Naive | 全面超过Naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| experiment1_metric_opt_k5_r80_e1_lr5e4_cuda | 419685162.269529 | 20311.639101 | 16146.880063 | 0.907099 | 0.904595 | 0.894615 | 19947.628440 | 13619.880887 | 0.758148 | 0.758963 | 0.980148 | 否 | 否 | 否 | 否 | 否 | 否 |
| experiment1_metric_opt_k5_r80_e2_lr5e4_cuda | 338627118.026792 | 18242.610962 | 14214.557395 | 0.800096 | 0.798514 | 0.913003 | 19947.628440 | 13619.880887 | 0.758148 | 0.758963 | 0.980148 | 是 | 否 | 否 | 否 | 否 | 否 |
| experiment1_metric_opt_k5_r100_e1_lr5e4_cuda | 382218605.015230 | 19405.991664 | 15265.058703 | 0.856461 | 0.854482 | 0.907676 | 19947.628440 | 13619.880887 | 0.758148 | 0.758963 | 0.980148 | 是 | 否 | 否 | 否 | 否 | 否 |
| experiment1_metric_opt_k4_exclude289_r80_e1_lr5e4_cuda | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 否 |

## 5. 相对 r60 的改善

| name | RMSE改善量 | MAE改善量 | MAPE改善量 | SMAPE改善量 | R2改善量 |
|---|---|---|---|---|---|
| experiment1_metric_opt_k5_r80_e1_lr5e4_cuda | -1549.255761 | -1902.563709 | -0.110266 | -0.109345 | -0.024679 |
| experiment1_metric_opt_k5_r80_e2_lr5e4_cuda | 519.772379 | 29.758959 | -0.003263 | -0.003264 | -0.006291 |
| experiment1_metric_opt_k5_r100_e1_lr5e4_cuda | -643.608324 | -1020.742349 | -0.059628 | -0.059232 | -0.011618 |
| experiment1_metric_opt_k4_exclude289_r80_e1_lr5e4_cuda | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 | 未运行 / 目录不存在 |

## 6. per-client 诊断

### experiment1_metric_opt_k5_r80_e1_lr5e4_cuda
| region_id | FedAvg_RMSE | Naive_RMSE | FedAvg_minus_Naive_RMSE | FedAvg_RMSE优于Naive |
|---|---|---|---|---|
| 284 | 21224.291579 | 24846.927365 | -3622.635786 | 是 |
| 288 | 18670.550687 | 19297.015101 | -626.464415 | 是 |
| 289 | 22452.669232 | 11065.790314 | 11386.878918 | 否 |
| 290 | 23245.710775 | 20708.839296 | 2536.871479 | 否 |
| 318 | 15964.973232 | 21177.513318 | -5212.540086 | 是 |
- 该方案中相对 Naive 拖累最大的 client 是 `region 289`，`FedAvg_minus_Naive_RMSE=11386.878918`。
- FedAvg 在 `RMSE` 上优于 Naive 的 client 数为 `3/5`。

### experiment1_metric_opt_k5_r80_e2_lr5e4_cuda
| region_id | FedAvg_RMSE | Naive_RMSE | FedAvg_minus_Naive_RMSE | FedAvg_RMSE优于Naive |
|---|---|---|---|---|
| 284 | 21331.339088 | 24846.927365 | -3515.588277 | 是 |
| 288 | 16425.011721 | 19297.015101 | -2872.003380 | 是 |
| 289 | 20886.827608 | 11065.790314 | 9821.037294 | 否 |
| 290 | 17198.875074 | 20708.839296 | -3509.964222 | 是 |
| 318 | 15371.001316 | 21177.513318 | -5806.512002 | 是 |
- 该方案中相对 Naive 拖累最大的 client 是 `region 289`，`FedAvg_minus_Naive_RMSE=9821.037294`。
- FedAvg 在 `RMSE` 上优于 Naive 的 client 数为 `4/5`。

### experiment1_metric_opt_k5_r100_e1_lr5e4_cuda
| region_id | FedAvg_RMSE | Naive_RMSE | FedAvg_minus_Naive_RMSE | FedAvg_RMSE优于Naive |
|---|---|---|---|---|
| 284 | 20989.454917 | 24846.927365 | -3857.472448 | 是 |
| 288 | 17666.752033 | 19297.015101 | -1630.263069 | 是 |
| 289 | 20660.720054 | 11065.790314 | 9594.929740 | 否 |
| 290 | 22051.821691 | 20708.839296 | 1342.982395 | 否 |
| 318 | 15661.209626 | 21177.513318 | -5516.303692 | 是 |
- 该方案中相对 Naive 拖累最大的 client 是 `region 289`，`FedAvg_minus_Naive_RMSE=9594.929740`。
- FedAvg 在 `RMSE` 上优于 Naive 的 client 数为 `3/5`。

### experiment1_metric_opt_k4_exclude289_r80_e1_lr5e4_cuda
- 未运行 / 目录不存在。

## 7. 最优方案判断

- 当前最佳方案：`experiment1_metric_opt_k5_r80_e2_lr5e4_cuda`。
- 判断依据：没有方案全面超过 NaiveLastValue，按 MAE/MAPE/SMAPE/R2 改善且 RMSE 不退化的优先级选取。
- 不只按 RMSE 判断，而是优先看是否全面超过 Naive；若不能，则看 MAE/MAPE/SMAPE/R2 改善且 RMSE 不退化。
- 若 `K=4 exclude-289` 更好，也只能作为异质性诊断，不能直接替代当前 K=5 正式结果。

## 8. 结论

- 是否有方案全面超过 NaiveLastValue：否。
- 如果没有，当前最佳方案主要改善的方向：`experiment1_metric_opt_k5_r80_e2_lr5e4_cuda`。
- local_epochs 降低是否有效：需结合 K=5 e1/e2 对比判断；e1 更偏向抑制 non-IID 本地漂移。
- lr 降低是否有效：本阶段候选固定为 `0.0005`，相对先前正式设置属于更保守学习率。
- 继续增加 rounds 是否仍有效：需结合 `r80` 与 `r100` 对比判断；若只带来 RMSE 小幅改善而其他指标停滞，则继续增 rounds 的边际收益有限。
- 289 是否仍是关键问题：若 K=5 方案中仍由 `289` 贡献最大 FedAvg-vs-Naive gap，则答案为是。
- 是否建议进入 cluster/region client：若 K=5 与 K=4 都不能全面改善多指标，应建议进入 cluster/region client。
- 是否建议继续直接调 K=5：仅当 K=5 出现明显多指标改善且未退化 RMSE 时才继续，否则应降低优先级。

## 9. 推荐下一步

- 若 K=5/K=4 都未明显改善：建议停止继续调参，转向 cluster/region client。

## 10. 边界声明

本阶段只做实验 1 范围内 FedAvg 多指标优化 smoke；未修改 FedAvg 聚合公式，未修改模型结构，未修改数据划分，未运行实验 2/3/4，未提交 results。
