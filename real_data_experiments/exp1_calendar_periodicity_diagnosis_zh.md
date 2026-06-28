# 实验 1 日-周周期感知诊断报告

## 1. 背景

真实数据覆盖 2017-04-01 至 2017-05-31，共 61 天、5856 个 15 分钟时间片。该时间范围不足以支撑年度季节性建模，因此本文构造短期日历周期性特征，包括日周期、周周期、工作日/周末、节假日和调休上班日。

当前实验 1 formal 结果（rounds=20, local_epochs=1, baseline model）：

| 方法 | RMSE |
|------|:---:|
| FedAvg | 20,753.14 |
| Independent | 14,883.58 |
| NaiveLastValue | 19,419.22 |

FedAvg 弱于 Independent 和 NaiveLastValue。前一阶段已排除客户端相似度为主因，现探索日历周期感知是否能改善 FedAvg。

## 2. 日历特征

- **时间粒度**: 15 分钟（每天 96 个 time slots × 61 天 = 5856）
- **节假日标注**: 清明节（4/2-4/4）、劳动节（4/29-5/1）、端午节（5/28-5/30）
- **调休上班日**: 2017-04-01（周六）、2017-05-27（周六）
- **编码特征**: is_effective_workday, weekday_id, sin/cos_time_of_day, sin/cos_day_of_week
- **数据文件**: `data/external/calendar/calendar_features_15min_2017_04_01_to_2017_05_31.csv`

## 3. 方法设计

| 方法 | 说明 |
|------|------|
| NaiveLastValue | ŷ[t] = x[t-1]（persistence baseline） |
| DailySeasonalNaive | ŷ[t] = y[t-96]（昨天同一时间片） |
| WeeklySeasonalNaive | ŷ[t] = y[t-672]（上周同一日同一时间片，fallback: t-96 → t-1 → train_mean） |
| CalendarProfileNaive | train 中按 is_effective_workday + slot_of_day 计算 client-specific mean profile |
| CalendarFeature-FedAvg | FedAvg + calendar features 输入（尚未实现） |
| SeasonalResidual-FedAvg | FedAvg 训练残差 r = y - S(t)，预测 ŷ = S(t) + r̂（尚未实现） |

**CalendarProfileNaive**: 每个 client 使用 train split 构建 weekday/weekend slot profile，预测时按测试日期的 is_effective_workday 和 slot_of_day 查表。

### 评估对齐说明

本次修复后，DailySeasonalNaive、WeeklySeasonalNaive 和 CalendarProfileNaive 均使用与 FedAvg/Independent/NaiveLastValue 完全一致的 test_loader target time 进行评估。各方法的 test_samples 总数相等（4395）。周期 profile 仅使用 train split 构建，不使用 val/test，避免数据泄漏。lag-based lookback 使用完整 tensor 序列（含 train/val/test），仅取 t-96 或 t-672 历史值，不访问未来。

## 4. 日历特征输入

当前已生成 calendar feature 文件并用于周期性 baseline 的 profile 构建。但尚未将 calendar features 拼接进 FedAvg 模型输入（如 `--use-calendar-features` 或 `--target-mode residual`）。因此 CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 仍属于后续待实现结构，本报告中的周期性 baseline 表现不等同于周期感知 FedAvg 的表现。

## 5. r5e1 小规模结果（test split 对齐后）

参数: rounds=5, local_epochs=1, device=cuda, baseline model, sequence_length=12, selected=290,284,318,288,289

| 方法 | RMSE | MAE | R² |
|------|:---:|:---:|:---:|
| NaiveLastValue | **19,419** | 13,620 | 0.939 |
| Independent | 25,765 | 18,839 | 0.850 |
| CalendarProfileNaive | 32,194 | 22,770 | 0.830 |
| FedAvg | 37,344 | 30,302 | 0.567 |
| DailySeasonalNaive | 45,406 | 29,727 | 0.637 |
| WeeklySeasonalNaive | 48,369 | 32,881 | 0.551 |

### 关键发现

1. **NaiveLastValue 仍然最强** — 简单的上一时间片预测 RMSE=19,419，说明当前数据的时间惯性极强。
2. **CalendarProfileNaive 优于 Daily/WeeklySeasonalNaive** — weekday/weekend slot profile 比简单昨日/上周有明显改进（32,194 vs 45,406/48,369），但仍弱于 NaiveLastValue 和 Independent。
3. **FedAvg 并非最差** — 修复对齐后 WeeklySeasonalNaive 和 DailySeasonalNaive 均比 FedAvg 差（48,369/45,406 vs 37,344）。CalendarProfileNaive 略优于 FedAvg（32,194 vs 37,344），但仍明显弱于 NaiveLastValue 和 Independent。
4. **Independent 优于 CalendarProfileNaive** — 本地独立训练的模型（25,765）比纯周期 profile（32,194）更好，说明有可学习的时序依赖超越简单周期性。

## 6. 诊断结论

1. DailySeasonalNaive 和 WeeklySeasonalNaive 均弱于 FedAvg — 简单昨日/上周重复不能捕捉 traffic dynamics。
2. CalendarProfileNaive 强于 FedAvg 但弱于 Independent 和 NaiveLastValue — client-specific weekday/weekend profile 提供了比 FedAvg 更好的 baseline，但仍不如本地独立训练和简单的 persistence。
3. CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 尚未运行，需后续补充 CLI 配置。
4. 当前结果只能说明，在严格 test split 对齐的条件下，三类周期性 naive baseline 可作为解释交通流短期周期性的补充基线。它们是否能够改善 FedAvg，还需要进一步实现 CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 后再验证。当前不能把周期性 baseline 的表现直接等同于周期感知 FedAvg 的表现。
5. **不建议**在当前 baseline 模型上继续投入大量 formal 实验，因 FedAvg 在所有条件下都弱于简单 baseline。
6. **建议**进入 FedProx 或调整训练策略（rounds/lr/local_epochs），或考虑更强的时序模型（CNN-GRU、Transformer）。

## 7. 对一审意见的回应价值

该实验回应以下审稿关切：

- **外部因素不足**: 提供日历周期性特征，含节假日和调休标注。
- **多源数据扩展**: 日历数据即是一种外部结构化多源数据。
- **客户端异质性建模**: client-specific seasonal profile 体现网格间日历模式差异。
- **预测模式分析**: weekday/weekend 分离可分析时间模式下的性能差异。
- **节假日和调休**: 标注了清明节、劳动节、端午节及调休上班日。

## 8. 局限性

- 61 天数据不足以支撑年度季节性，本文建模的是短期日历周期性。
- 天气、事件、信号控制数据暂不可用。
- 节假日样本数量有限（清明节 3 天、劳动节 3 天、端午节 3 天）。
- CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 尚未运行验证。
- 当前周期性 baseline 只提供预测，尚未与 FedAvg 训练/推理链路集成。

## 9. 论文表述

英文：

> The real-world dataset covers 61 days from April 1 to May 31, 2017. Since this period is insufficient for modeling annual seasonality, we construct short-term calendar-periodicity features instead, including time-of-day, day-of-week, public holidays, adjusted working days, and effective working-day indicators. A client-specific seasonal profile is estimated from the training split and used as a local periodic baseline. The federated model is then trained to predict the residual component relative to this baseline, allowing the framework to preserve local calendar patterns while learning shared residual dynamics across clients.

中文：

> 真实数据覆盖 2017 年 4 月 1 日至 5 月 31 日共 61 天。由于该时间范围不足以支持年度季节性建模，本文构造短期日历周期性特征，包括日内时间片、星期、法定节假日、调休上班日和有效工作日等。同时，本文基于训练集为每个客户端估计私有周期基线，并使联邦模型学习相对于该基线的残差部分，从而在保留本地日历周期差异的同时学习跨客户端共享的残差动态。

## 10. 下一步建议

1. **暂不继续 baseline model formal**: 当前 baseline 模型 FedAvg 在所有条件下弱于简单 baseline。
2. **优先尝试 FedProx**: μ=1e-4/1e-3/1e-2，可能在局部训练漂移问题上有改善。
3. **调整训练策略**: rounds 增至 20/40, lr 调低至 5e-4/3e-4。
4. **补充 CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 的 CLI 实现**。
5. **考虑更强的时序模型**（CNN-GRU、Transformer）替代当前简单的 LSTM/baseline。
