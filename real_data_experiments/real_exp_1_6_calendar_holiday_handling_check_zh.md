# 真实数据实验 1–6 时间与节假日处理检查报告

> 生成日期：2026-06-30
> 最后更新：2026-06-30（CalendarFeatureFedAvg v2 + Exp1 long-horizon diagnostic seq96_h4/h12/h24）
> 文档状态：已与当前源码状态同步（v2 + long-horizon diagnostic 完成）

## 1. Git 状态

- **分支**：`feature/real-exp4-rfc-ablation`
- **HEAD**：待本轮提交后更新
- **本轮是否运行 formal**：否
- **本轮是否运行 smoke**：是，Exp5 final align smoke 用于验证 CalendarProfileNaive 样本对齐
- **本轮是否修改源码**：是，前序提交已修改 `rc_core.py` 与 `calendar_utils.py`
- **本轮是否修改文档**：是，本文件已在多轮中更新
- **staged 区是否包含 results/logs/data**：否

## 2. 检查范围

本轮检查了以下文件和目录：

| 类别 | 检查内容 |
|------|----------|
| Exp1 源码 | `single_intersection_client/sic_core.py`、`sic_config.py` |
| Exp2 源码 | `single_intersection_ablation/sia_core.py`、`sia_config.py` |
| Exp3 源码 | `region_client_full_cells/rfc_core.py`、`rfc_config.py`、`rfc_dataset.py` |
| Exp4 源码 | `region_client_full_cells/rfc_ablation_core.py`、`rfc_ablation_config.py` |
| Exp5 源码 | `region_client/rc_core.py`、`rc_config.py` + `common/region_tensor_dataset.py` |
| Exp6 源码 | `region_ablation/ra_core.py`、`ra_config.py` |
| 日历脚本 | `common/calendar_baselines.py`、`tools/build_calendar_features_2017.py` |
| 日历数据 | `data/external/calendar/` 下所有 CSV |
| 运行配置 | `results/real_data_experiments/` 下 `run_config.json`、`split_summary.json` |
| 文档引用 | `real_exp_diagnostics_archive_zh.md`、`reviewer_response_experiment_mapping_zh.md`、`real_exp_1_6_result_table_plan_zh.md` |

## 3. 日历特征文件与节假日标注

### 3.1 文件清单

| 文件 | 是否存在 | 覆盖时间 | 行数 | 包含字段 |
|------|:---:|------|------|------|
| `data/external/calendar/calendar_2017_04_01_to_2017_05_31.csv` | 是 | 2017-04-01 至 2017-05-31 | 61 行（日级） | 16 列 |
| `data/external/calendar/calendar_features_15min_2017_04_01_to_2017_05_31.csv` | 是 | 2017-04-01 至 2017-05-31 | 5856 行（15 分钟级） | 23 列 |
| `tools/build_calendar_features_2017.py` | 是 | — | — | 构建脚本 |

### 3.2 日级表字段（16 列）

```
date, day_index, slot_start, slot_end, weekday, weekday_id,
is_weekend, is_holiday, is_adjusted_workday, is_effective_workday,
is_festival_day, holiday_name, holiday_group, is_pre_holiday,
is_post_holiday, days_to_nearest_holiday
```

### 3.3 15 分钟级表额外字段（+7 列）

```
time_index, slot_of_day, hour, minute,
sin_time_of_day, cos_time_of_day, sin_day_of_week, cos_day_of_week
```

### 3.4 关键字段对照检查

| 用户关注的字段 | CSV 中是否存在 | 说明 |
|---|---|---|
| `timestamp` | 无独立列 | 由 `date` + `time_index` / `slot_of_day` 联合表达 |
| `date` | 是 | `"2017-04-01"` 格式 |
| `day_of_week` | 无此列名 | 等价字段为 `weekday` / `weekday_id`（0=Mon, 6=Sun） |
| `is_weekend` | 是 | 0/1 |
| `is_holiday` | 是 | 0/1 |
| `is_adjusted_workday` | 是 | 0/1，标记调休上班日 |
| `is_effective_workday` | 是 | 0/1，综合工作日判定 |
| `slot_of_day` | 是 | 0–95（仅 15min 表） |
| `holiday_name` | 是 | `"清明节"` / `"劳动节"` / `"端午节"` 等 |

### 3.5 2017 年 4–5 月关键节假日标注

**三个节假日及调休日均已完整标注：**

| 节假日 | 日期 | `holiday_name` | `holiday_group` | `is_holiday` | `is_festival_day` |
|---|---|---|---|---|---|
| 清明节 | 2017-04-04 | 清明节 | qingming | 1 | 1 |
| 清明节假期 | 2017-04-02, 2017-04-03 | 清明节假期 | qingming | 1 | 0 |
| 劳动节 | 2017-05-01 | 劳动节 | labor | 1 | 1 |
| 劳动节连休 | 2017-04-29, 2017-04-30 | 劳动节连休 | labor | 1 | 0 |
| 端午节 | 2017-05-30 | 端午节 | dragon_boat | 1 | 1 |
| 端午节假期 | 2017-05-28, 2017-05-29 | 端午节假期 | dragon_boat | 1 | 0 |

**调休上班日（`is_adjusted_workday=1`）：**

| 日期 | 原因 |
|---|---|
| 2017-04-01（周六） | 清明节调休上班 |
| 2017-05-27（周六） | 端午节调休上班 |

### 3.6 构建脚本 `build_calendar_features_2017.py` 功能

1. `build_day_level_calendar()`：基于 `pd.date_range("2017-04-01", periods=61, freq="D")` 生成日级日历表
2. `expand_to_15min()`：扩展为 15 分钟级，新增 time_index / slot_of_day / sin/cos 编码
3. `validate()`：验证行数、关键日期标注正确性

## 4. 日历处理等级定义

| Level | 含义 |
|:---:|------|
| Level 0 | 数据时间范围包含节假日，但模型不知道节假日（无任何日历特征使用） |
| Level 1 | 有 calendar baseline 或周期性诊断（日历特征作为独立 baseline 评估） |
| Level 2 | calendar/holiday 特征进入模型输入（神经网络训练链路） |
| Level 3 | 专门做了节假日分组评估或消融实验 |

## 5. 实验 1–6 日历处理总表

| 实验 | 新版含义 | 时间索引 | weekday/weekend | 节假日/调休 | 日内 slot | calendar baseline | calendar 作为模型输入 | 当前等级 |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Exp1 | 单 grid cell 主实验 | 真实时间索引 (80/10/10 划分) | CalendarProfileNaive 使用 | 日历 CSV 标注 | slot_of_day 0-95 | CalendarProfileNaive + Daily/WeeklySeasonalNaive + CalendarFeatureFedAvg | 是（CalendarFeatureFedAvg diagnostic，Level 2） | **Level 2 diagnostic** |
| Exp2 | 单 grid cell 消融 | 纯整数 tensor index | 无 | 无 | 无 | 无（未继承 Exp1） | 否 | **Level 0** |
| Exp3 | 多相似 cell 主实验 | 纯整数 tensor index | 无 | 无 | 无 | CalendarProfileNaive (新增) | 否 | **Level 1** |
| Exp4 | 多相似 cell 消融 | 纯整数 tensor index | 无 | 无 | 无 | 无 | 否 | **Level 0** |
| Exp5 | 全部 cells 划分 client 主实验 | 纯整数 tensor index | 无 | 无 | 无 | CalendarProfileNaive (新增) + NaiveLastValue | 否 | **Level 1** |
| Exp6 | 全部 cells 划分 client 消融 | 纯整数 tensor index | 无 | 无 | 无 | 无 | 否 | **Level 0** |

## 6. 分实验结论

### 6.1 Exp1：单 grid cell 主实验

**文件**：`single_intersection_client/sic_core.py`（1247 行）、`sic_config.py`（127 行）

**日历处理详情：**

- **CalendarProfileNaive**：有。`sic_core.py` 第 909 行定义 `evaluate_calendar_profile_naive()`，调用 `calendar_baselines.build_client_seasonal_profile()` + `calendar_profile_naive_predict()`，按 `is_effective_workday` + `slot_of_day` 构建 client-specific 工作日/非工作日 slot profile 作为 baseline 预测。

- **DailySeasonalNaive**：有。`sic_core.py` 第 866 行定义 `evaluate_daily_seasonal_naive()`，预测 `y[t] = y[t-96]`（昨日同一时刻）。

- **WeeklySeasonalNaive**：有。`sic_core.py` 第 889 行定义 `evaluate_weekly_seasonal_naive()`，预测 `y[t] = y[t-672]`（上周同一日期同一时刻）。

- **NaiveLastValue**：有。`sic_core.py` 第 796 行定义 `evaluate_naive_last_value()`，预测 `y[t] = x[t-1]`（上一时刻观测值）。

- **calendar CSV 加载**：有。`sic_core.py` 第 1099 行 `_load_calendar_features()` 从 `data/external/calendar/calendar_features_15min_2017_04_01_to_2017_05_31.csv` 加载日历特征。

- **日历特征进入模型输入**：否。神经网络输入仅来自 grid tensor 的通道维度（`use_channels=[0, 1]`），calendar DataFrame 仅用于三个 baseline 评估。

- **时间切分**：`split_strategy = "temporal_contiguous_by_target_time"`，按真实时间顺序切分 train/val/test（当前比例 0.8/0.1/0.1；历史 formal 使用 0.7/0.15/0.15）。

- **已有运行结果**：`results/real_data_experiments/diagnostics/exp1_calendar_periodicity/calendar_baselines_r5e1_cuda/run_config.json` 记录了 Exp1 的 calendar baselines 运行配置。

**结论**：Exp1 已从 Level 1 升级至 Level 2 diagnostic。CalendarFeatureFedAvg 已将 calendar/holiday 特征作为辅助输入分支接入神经网络训练链路（diagnostic r5e1），但该 diagnostic 当前表现弱于 FedAvg，不可写成性能提升。CalendarProfileNaive 仍是独立 baseline。Exp3 和 Exp5 也已接入 CalendarProfileNaive baseline（见下文）。

> **历史划分说明**：Exp1 formal r20e1 使用 70%/15%/15% 时序划分。修订后的划分方案为 80%/10%/10%，以适应 61 天观测窗口。70%/15%/15% 结果保留作为 sensitivity check 参考。

### 6.2 Exp2：单 grid cell 消融

**文件**：`single_intersection_ablation/sia_core.py`（410 行）、`sia_config.py`（135 行）

**日历处理详情：**

- 从 `sic_core.py` 仅导入了模型构建、数据处理和归一化基础设施（`ClientData`、`InputScaler`、`TargetScaler`、`apply_dataset_normalization`、`build_client_data` 等），**未导入** `evaluate_calendar_profile_naive`、`evaluate_daily_seasonal_naive`、`evaluate_weekly_seasonal_naive`、`evaluate_naive_last_value` 中的任何一个。
- 不调用 `_load_calendar_features()`。
- 不导入 `calendar_baselines` 模块。
- 仅运行模型结构消融（Full / Without Attention / Without CNN / Without LSTM），无 baseline 评估。
- 数据划分方式与 Exp1 相同（`"temporal_contiguous_by_target_time"`）。

**结论**：Exp2 是纯粹的模型结构消融（Level 0），未使用任何日历特征。不能写"Exp2 处理了节假日"。最多写"与 Exp1 共享同一时间范围的数据"。

### 6.3 Exp3：多相似 cell 主实验

**文件**：`region_client_full_cells/rfc_core.py`、`rfc_config.py`、`rfc_dataset.py`

**日历处理详情：**

- **分区方式**：`rfc_config.py` 默认使用 `spatial_k5`，正式运行时推荐 `similarity_k5`（`README_zh.md` 推荐）。
- **时间索引**：纯整数 tensor index。`build_time_split_bounds(time_count=<int>)` 按 0.7/0.15/0.15 比例计算整数边界，无任何 datetime/timestamp 映射。
- **NaiveLastValue**：有。`rfc_core.py` 第 294–323 行实现，作为 baseline 之一（与 FedAvg、Independent 对比）。
- **CalendarProfileNaive**：已新增。该 baseline 基于 `is_effective_workday + slot_of_day` 构建 client-specific profile。
- **模型输入**：仅 tensor 通道（`use_channels=[0, 1]`），`rfc_ablation_core.py` 第 242/306 行明确记录 `"data_mode": "tensor"`。
- **时间切分**：`split_strategy = "temporal_contiguous_by_target_time"`，但 target_time 是整数索引，不在 prediction 输出中保留。

**结论**：Exp3 已接入 CalendarProfileNaive baseline（Level 1）。该 baseline 基于 `is_effective_workday + slot_of_day` 构建 client-specific profile，并与 NaiveLastValue 一起作为非神经网络 baseline。Calendar/holiday 特征仍未进入 FedAvg 或 Independent 的神经网络输入。

### 6.4 Exp4：多相似 cell 消融

**文件**：`region_client_full_cells/rfc_ablation_core.py`、`rfc_ablation_config.py`

**日历处理详情：**

- 复用 Exp3 的数据构建管线（`build_full_cells_client_data`、`fit_rfc_input_scaler`、`fit_rfc_target_scaler` 等从 `rfc_core.py` / `rfc_dataset.py` 导入）。
- 使用 `similarity_k5` 分区（配置文件明确写为 `similarity_k5.json`）。
- 仅运行模型结构消融（Full / Without Attention / Without CNN / Without LSTM）。
- 无任何 calendar/holiday/weekday/weekend 特征。
- 明确记录：`"data_mode": "tensor"`（第 306 行）。

**结论**：Exp4 是纯消融实验（Level 0），未使用任何日历特征。不能写"Exp4 具有 calendar/holiday feature"。数据时间段包含节假日，但模型完全不知道。

### 6.5 Exp5：全部 grid cells 划分 client 主实验

**文件**：`region_client/rc_core.py`、`rc_config.py` + `common/region_tensor_dataset.py`

**日历处理详情：**

- **NaiveLastValue**：有。`rc_core.py` 第 457–490 行实现，作为 baseline 之一。
- **calendar baseline**：无。`calendar_baselines.py` 未被 `rc_core.py` 或相关文件导入。
- **weekday/weekend 特征**：无。全实验链路未出现 `weekday`、`weekend`、`is_weekend`、`is_holiday`、`is_effective_workday` 等模式。
- **时间索引**：纯整数。`RegionClientWindowDataset` 的 `first_target_time` / `last_target_time` 是整数运算结果，`target_time` 是 tensor 第二维的索引。prediction 输出中不保留 target_time 列，无法反向映射到具体时间点。
- **分区方式**：`spatial_block` / `flow_kmeans`，均基于空间/流量特征，不使用日历信息。
- **时间切分**：`"temporal_contiguous_by_target_time"`。

**结论**：Exp5 已接入 CalendarProfileNaive baseline（Level 1）。本轮已修复 raw train/val/test dataset 与 capped dataloader 的样本对齐问题，确保 CalendarProfileNaive 与 NaiveLastValue/FedAvg/Independent 使用同一 capped split。Calendar/holiday 特征仍未进入神经网络输入。

### 6.6 Exp6：全部 grid cells 划分 client 消融

**文件**：`region_ablation/ra_core.py`、`ra_config.py`

**日历处理详情：**

- 从 `rc_core.py` 继承数据构建管线（`build_region_client_data`、`fit_rc_input_scaler`、`fit_rc_target_scaler` 等）。
- 仅运行模型结构消融（Full / Without Attention / Without CNN / Without LSTM）。
- 无任何 calendar/holiday/weekday/weekend 特征。
- 不使用 `calendar_baselines` 模块。

**结论**：Exp6 是纯消融实验（Level 0），未使用任何日历特征。数据时间段包含节假日但模型完全不知。

## 7. 已有 run_config / split_summary 中的 calendar 记录

- 唯一包含 `calendar` / `holiday` / `weekday` / `weekend` 的记录位于：
  `results/real_data_experiments/diagnostics/exp1_calendar_periodicity/calendar_baselines_r5e1_cuda/run_config.json`
- 这是 Exp1 的 calendar periodicity 诊断运行，使用了 `model_variant = "baseline"`，对应 calendar baselines（CalendarProfileNaive / DailySeasonalNaive / WeeklySeasonalNaive）的评估。
- Exp2–Exp6 的运行配置中 **均未** 包含任何 calendar/holiday 相关参数设置。

## 8. 对论文写作的影响

### 8.1 可以写

- 数据时间范围包含 2017 年 4–5 月清明节、劳动节、端午节及调休日；
- 已构建完整的 calendar feature 文件（日级 + 15 分钟级，含节假日/调休/工作日标注）；
- Exp1（单 cell 实验）已使用 CalendarProfileNaive / DailySeasonalNaive / WeeklySeasonalNaive 作为 calendar baselines，按 `is_effective_workday` + `slot_of_day` 检查周期性；
- CalendarProfileNaive 是独立 baseline，验证了简单工作日/非工作日 profile 不足以捕捉交通流动态（RMSE=32,194 vs FedAvg 更低）；
- 节假日变量当前 **已作为** 独立的 baseline 评估参考，但 **尚未** 作为 FedAvg 神经网络模型的输入特征。

### 8.2 不能写

- 不能写"Exp1–Exp6 全部都显式建模了节假日"；
- 不能写"CalendarFeatureFedAvg 已经提升 FedAvg"；
- 不能写"CalendarFeatureFedAvg 已经完成 formal"；
- 不能写"Exp2/4/6 已有 without_calendar 消融"；
- 不能写"Exp2/Exp3/Exp4/Exp5/Exp6 已经使用了 holiday feature 或 calendar baseline"（Exp3/Exp5 已有 CalendarProfileNaive baseline，可以写，但不能写成模型输入）；
- 不能写"calendar 特征已被所有实验共享使用"。

## 9. 缺口与下一步建议

按优先级排列：

### P0（建议在论文中声明）

- 如果审稿人要求节假日处理，应在论文中明确：
  - "当前 calendar/holiday 特征已作为 CalendarProfileNaive / DailySeasonalNaive / WeeklySeasonalNaive 三个独立 baseline 评估（Exp1），用于验证周期性效应；"
  - "但 calendar 特征尚未接入 FedAvg 等联邦模型的训练输入通道。"

### P1（已完成）

- Exp3 / Exp5 已新增 CalendarProfileNaive baseline。




### P2（建议开发 calendar 增强模型）

- 开发 CalendarFeature-FedAvg：将 `day_of_week` / `slot_of_day` / `is_holiday` / `is_effective_workday` 的 sin/cos 编码（日历 CSV 中已有）拼接为额外输入通道或辅助特征，进入 FedAvg 训练链路（Level 2）。

### P3（建议做节假日分组分析）

- 基于已有 calendar CSV 做 weekday vs weekend vs holiday 分组误差分析：
  - 将 test 集按 `is_holiday` / `is_weekend` / `is_effective_workday` 分组；
  - 报告各组 RMSE/MAE，揭示模型在不同日期类型下的性能差异（Level 3）。

### CalendarProfileNaive 样本对齐修复

#### 修复前问题

Exp5 的 `raw_test_dataset` 曾在 `_maybe_cap_dataset()` 之前保存，导致 `CalendarProfileNaive` 和 `NaiveLastValue` 可能使用 full test dataset，而 FedAvg/Independent 使用 capped test loader。当 `--max-samples-per-client-split 1000` 时，CalendarProfileNaive 会输出全量样本（如 196,017 行），而 FedAvg 仅 3,000 行。raw_train/raw_val 也存在同样的不对齐风险。

#### 修复方式

- [`rc_core.py`](file:///E:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/region_client/rc_core.py#L204-L209)：调整数据构造顺序，先对 train/val/test dataset 执行 `_maybe_cap_dataset()`，再保存 `raw_train_dataset`、`raw_val_dataset`、`raw_test_dataset`；
- [`calendar_utils.py`](file:///E:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/common/calendar_utils.py#L244-L252)：增加 `raw_test_dataset` 与 `test_loader.dataset` 的长度一致性检查，不匹配时 `raise ValueError`；
- [`calendar_utils.py`](file:///E:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/common/calendar_utils.py#L294-L303)：prediction 输出包含 `target_time`、`date`、`slot_of_day`、`is_effective_workday`、`is_holiday`、`is_weekend`、`holiday_name`。

#### 对齐验证

| 实验 | 验证方式 | CalendarProfileNaive rows | NaiveLastValue rows | FedAvg rows | 是否一致 | 说明 |
|---|:---:|---:|---:|---:|:---:|---|
| Exp3 | full test 对齐检查 | 196,017 | 196,017 | 196,017 | 是 | Exp3 无 capping 逻辑，所有方法使用 full test |
| Exp5 | 1k capped smoke | 3,000 | 3,000 | 3,000 | 是 | 3 clients × 1000，raw dataset 与 dataloader 完全对齐 |

## 10. 最终结论

| 项目 | 结论 |
|------|------|
| **当前显式使用日历/节假日 baseline 的实验** | Exp1、Exp3、Exp5。Exp1 含 CalendarProfileNaive + Daily/WeeklySeasonalNaive；Exp3/Exp5 新增 CalendarProfileNaive |
| **当前完全无 calendar baseline 的消融实验** | Exp2、Exp4、Exp6 |
| **calendar/holiday 是否进入神经网络输入** | 否。所有实验的神经网络输入仍为 tensor 通道，calendar 仅作为 baseline/诊断使用 |
| **论文中如何表述** | 数据覆盖清明节、劳动节、端午节及调休日；主实验 Exp1/3/5 均已有 CalendarProfileNaive baseline；节假日特征尚未进入 FedAvg 模型输入 |
| **是否误提交 results/logs/data** | 否（staged 区为空） |

### 分级总结

| 实验 | 等级 | 一句话结论 |
|------|:---:|------|
| Exp1 | **Level 2 diagnostic** | 有 CalendarProfileNaive + Daily/WeeklySeasonalNaive baseline + CalendarFeatureFedAvg diagnostic (calendar 已进入模型输入，但仅 diagnostic 阶段) |
| Exp2 | **Level 0** | 纯结构消融，无 calendar baseline |
| Exp3 | **Level 1** | 新增 CalendarProfileNaive baseline，未进入模型输入 |
| Exp4 | **Level 0** | 纯结构消融，无 calendar baseline |
| Exp5 | **Level 1** | 新增 CalendarProfileNaive baseline，且 capped split 对齐已修复 |
| Exp6 | **Level 0** | 纯结构消融，无 calendar baseline |
