# 真实数据实验 1–6 时间与节假日处理检查报告

> 生成日期：2026-06-30
> 本轮只做静态检查，不运行实验，不修改源码。

## 1. Git 状态

- **分支**：`feature/real-exp4-rfc-ablation`
- **HEAD**：`0cf5e69 docs(real-data): synchronize Exp4 status across reports`
- **本轮是否运行实验**：否
- **本轮是否修改源码**：否
- **staged 区是否包含 results/logs/data**：否（未暂存任何文件）

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
| Exp1 | 单 grid cell 主实验 | 真实时间索引 | CalendarProfileNaive 使用 | 日历 CSV 标注 | slot_of_day 0-95 | CalendarProfileNaive + Daily/WeeklySeasonalNaive | 否（仅 baseline） | **Level 1** |
| Exp2 | 单 grid cell 消融 | 纯整数 tensor index | 无 | 无 | 无 | 无（未继承 Exp1） | 否 | **Level 0** |
| Exp3 | 多相似 cell 主实验 | 纯整数 tensor index | 无 | 无 | 无 | 无（仅有 NaiveLastValue） | 否 | **Level 0** |
| Exp4 | 多相似 cell 消融 | 纯整数 tensor index | 无 | 无 | 无 | 无 | 否 | **Level 0** |
| Exp5 | 全部 cells 划分 client 主实验 | 纯整数 tensor index | 无 | 无 | 无 | 无（仅有 NaiveLastValue） | 否 | **Level 0** |
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

- **时间切分**：`split_strategy = "temporal_contiguous_by_target_time"`，按真实时间顺序切分 train/val/test（比例 0.7/0.15/0.15）。

- **已有运行结果**：`results/real_data_experiments/diagnostics/exp1_calendar_periodicity/calendar_baselines_r5e1_cuda/run_config.json` 记录了 Exp1 的 calendar baselines 运行配置。

**结论**：Exp1 是唯一显式使用日历特征的实验，但仅限于 baseline 评估层面（Level 1）。Calendar/holiday 特征未进入 FedAvg 神经网络训练链路。可写为"已构建 calendar baselines 作为周期性诊断基准"。

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
- **calendar baseline**：无。`CalendarProfileNaive`、`DailySeasonalNaive`、`WeeklySeasonalNaive` 均未被导入或调用。`calendar_baselines.py` 虽然存在于公共模块中，但 `rfc` 目录下任何文件都未引用。
- **模型输入**：仅 tensor 通道（`use_channels=[0, 1]`），`rfc_ablation_core.py` 第 242/306 行明确记录 `"data_mode": "tensor"`。
- **时间切分**：`split_strategy = "temporal_contiguous_by_target_time"`，但 target_time 是整数索引，不在 prediction 输出中保留。

**结论**：Exp3 仅有 NaiveLastValue 作为最简单的 persistence baseline，没有任何 calendar baseline 或 holiday feature（Level 0）。不能写"Exp3 处理了节假日"。最多写"数据时间范围包含节假日，但模型未显式建模节假日"。

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

**结论**：Exp5 仅有 NaiveLastValue baseline，无任何 calendar baseline 或 holiday feature（Level 0）。不能写"Exp5 处理了节假日"。

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
- 不能写"Exp2/Exp3/Exp4/Exp5/Exp6 已经使用了 holiday feature 或 calendar baseline"；
- 不能写"calendar feature 已进入 FedAvg 神经网络输入"（Exp1 中仅作为 baseline，Exp2–Exp6 完全未接入）；
- 不能写"节假日处理提升了 FedAvg 性能"，除非有专门的对照实验结果；
- 不能写"calendar 特征已被所有实验共享使用"。

## 9. 缺口与下一步建议

按优先级排列：

### P0（建议在论文中声明）

- 如果审稿人要求节假日处理，应在论文中明确：
  - "当前 calendar/holiday 特征已作为 CalendarProfileNaive / DailySeasonalNaive / WeeklySeasonalNaive 三个独立 baseline 评估（Exp1），用于验证周期性效应；"
  - "但 calendar 特征尚未接入 FedAvg 等联邦模型的训练输入通道。"

### P1（建议补充 baseline）

- 为 Exp3/Exp4（rfc）和 Exp5/Exp6（rc/ra）增加 CalendarProfileNaive baseline：
  - 在 `rfc_core.py` 和 `rc_core.py` 中导入 `calendar_baselines`，调用 `build_client_seasonal_profile()` 和 `calendar_profile_naive_predict()`；
  - 输出 CalendarProfileNaive 指标，纳入 `client_metrics_df` 与 FedAvg/Independent/NaiveLastValue 对比；
  - 便于跨 client setting（单 cell / 多 cell / 全局 region）统一对比节假日效应。

### P2（建议开发 calendar 增强模型）

- 开发 CalendarFeature-FedAvg：将 `day_of_week` / `slot_of_day` / `is_holiday` / `is_effective_workday` 的 sin/cos 编码（日历 CSV 中已有）拼接为额外输入通道或辅助特征，进入 FedAvg 训练链路（Level 2）。

### P3（建议做节假日分组分析）

- 基于已有 calendar CSV 做 weekday vs weekend vs holiday 分组误差分析：
  - 将 test 集按 `is_holiday` / `is_weekend` / `is_effective_workday` 分组；
  - 报告各组 RMSE/MAE，揭示模型在不同日期类型下的性能差异（Level 3）。

## 10. 最终结论

| 项目 | 结论 |
|------|------|
| **当前真正显式使用日历/节假日 baseline 的实验** | 仅 **Exp1**（CalendarProfileNaive + DailySeasonalNaive + WeeklySeasonalNaive），且仅作为独立 baseline，未进入模型输入 |
| **当前无任何 calendar 处理（仅 NaiveLastValue）的实验** | Exp3、Exp5（有 NaiveLastValue baseline，无 calendar baseline） |
| **当前完全无 baseline 评估的消融实验** | Exp2、Exp4、Exp6（纯模型结构消融，无任何 baseline） |
| **论文中如何表述** | 数据时间范围覆盖 2017 年 4–5 月三个节假日 + 调休日；Exp1 已通过三个 calendar baselines 验证周期性效应；Exp2–Exp6 作为模型/分区的拓展实验，未重复接入 calendar baseline，可作为未来补充工作 |
| **是否误提交 results/logs/data** | 否（staged 区为空） |

### 分级总结

| 实验 | 等级 | 一句话结论 |
|------|:---:|------|
| Exp1 | **Level 1** | 有 calendar baselines（CalendarProfileNaive + Daily/WeeklySeasonalNaive），但未进入模型输入 |
| Exp2 | **Level 0** | 纯消融实验，无任何 calendar 特征 |
| Exp3 | **Level 0** | 仅有 NaiveLastValue baseline，无 calendar baseline |
| Exp4 | **Level 0** | 纯消融实验，无任何 calendar 特征 |
| Exp5 | **Level 0** | 仅有 NaiveLastValue baseline，无 calendar baseline |
| Exp6 | **Level 0** | 纯消融实验，无任何 calendar 特征 |
