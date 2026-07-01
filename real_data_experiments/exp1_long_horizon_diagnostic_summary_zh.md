# Exp1 长历史窗口与长预测跨度诊断汇总

> 本报告汇总 Exp1 在 80/10/10 时间连续划分下的 long-horizon diagnostic。
> 本轮为 diagnostic，不是 formal 结果。

## 1. 实验设置

| run_name | sequence_length | 历史窗口 | prediction_horizon | 预测跨度 |
|---|---:|---:|---:|---:|
| seq96_h4 | 96 | 24 小时 | 4 | 未来 1 小时 |
| seq96_h12 | 96 | 24 小时 | 12 | 未来 3 小时 |
| seq96_h24 | 96 | 24 小时 | 24 | 未来 6 小时 |

统一设置：rounds=5, local_epochs=1, split=chronological_80_10_10, CalendarFeatureFedAvg v2 residual_gate。

## 2. RMSE 汇总

| method                            |   seq96_h12 |   seq96_h24 |   seq96_h4 |
|:----------------------------------|------------:|------------:|-----------:|
| CalendarFeatureFedAvg-Full        |     74976.5 |     83645.7 |    75521.2 |
| CalendarFeatureFedAvg-HolidayOnly |     83754   |    104040   |    91714.6 |
| CalendarProfileNaive              |     31970.1 |     31970.1 |    31970.1 |
| DailySeasonalNaive                |     46788.6 |     46788.6 |    46788.6 |
| FedAvg                            |     84811.5 |     84887.1 |    79984.2 |
| Independent                       |     61199   |     79193.5 |    57581.4 |
| NaiveLastValue                    |     94259   |    123563   |    50706.1 |
| WeeklySeasonalNaive               |     52474.1 |     52474.1 |    52474.1 |

## 3. R² 汇总

| method                            |   seq96_h12 |   seq96_h24 |   seq96_h4 |
|:----------------------------------|------------:|------------:|-----------:|
| CalendarFeatureFedAvg-Full        |   0.0122352 | -0.0762419  |  0.040401  |
| CalendarFeatureFedAvg-HolidayOnly |  -0.0648861 | -1.68137    | -0.511767  |
| CalendarProfileNaive              |   0.826245  |  0.826245   |  0.826245  |
| DailySeasonalNaive                |   0.583164  |  0.583164   |  0.583164  |
| FedAvg                            |  -0.62333   | -0.11279    | -0.0302548 |
| Independent                       |   0.311464  |  0.00733281 |  0.405933  |
| NaiveLastValue                    |  -0.373127  | -1.33588    |  0.585293  |
| WeeklySeasonalNaive               |   0.389378  |  0.389378   |  0.389378  |

## 4. 读取建议

1. 首先比较 NaiveLastValue 在 h4/h12/h24 下是否随预测跨度变长而明显变差。
2. 再比较 FedAvg 与 NaiveLastValue 的差距是否缩小。
3. 再观察 CalendarFeatureFedAvg-HolidayOnly / Full 是否在长 horizon 下接近或超过 FedAvg。
4. 即使总体 RMSE 未提升，也要结合 grouped_metrics_by_calendar.csv 检查节假日、周末、调休日分组表现。

## 5. 当前结论占位

本报告只汇总 diagnostic 数据。是否进入 r10/r20，需要基于 h4/h12/h24 的 RMSE/MAE/MAPE 和 grouped calendar metrics 决定。

## 6. 解释风险与待核查项

1. CalendarProfileNaive、DailySeasonalNaive、WeeklySeasonalNaive 在 seq96_h4 / seq96_h12 / seq96_h24 下 RMSE 完全相同。该现象不能直接解释为"周期性 baseline 对 horizon 稳定"，需要先检查 evaluator 是否真正按 prediction_horizon 改变了目标点，或这些 baseline 是否固定使用同一 target_time profile lookup。
2. NaiveLastValue、FedAvg、Independent、CalendarFeatureFedAvg 会随 horizon 变化，说明神经网络方法和 last-value baseline 至少受到 horizon 设置影响。
3. seq96_h12 long-horizon diagnostic 与 federated mechanism diagnostic 中同名方法的 RMSE 存在小幅差异，后续需要核对 seed、输出目录、method set、evaluator、local fine-tuning 设置是否完全一致。
4. 本报告仅用于决定是否进入 r10/r20 formal candidate，不能直接作为论文 formal 结果。

## 7. 后续检查任务

- 检查 CalendarProfileNaive / DailySeasonalNaive / WeeklySeasonalNaive 是否按 h4/h12/h24 正确取对应 horizon 的 target；
- 检查 prediction 输出中是否保留 target_time、date、slot_of_day、is_effective_workday、is_holiday、is_weekend、holiday_name；
- 如果 seasonal baselines 当前没有按 horizon 改变，应修正 evaluator 后重跑 long-horizon diagnostic；
- 如果 evaluator 正确，应在文档中解释为什么 seasonal baselines 在不同 horizon 下指标相同。
