> 文档状态：历史记录，仅用于追溯；不代表当前最终实验状态。
> 当前状态以 `REAL_DATA_EXPERIMENTS_CURRENT_DOCS_zh.md` 为准。

# 真实数据实验诊断报告归档

> 归档日期：2026-06-30
> 这些诊断报告包含重要的根因分析和结论，但不作为当前实验状态依据。
> 当前实验状态请参考：[real_exp_1_6_current_status_and_revision_plan_zh.md](real_exp_1_6_current_status_and_revision_plan_zh.md)

---

## 1. 归档说明

本文档合并了以下诊断报告：

| 原文件 | 诊断主题 | 归档原因 |
|--------|---------|---------|
| `exp1_client_similarity_diagnosis_zh.md` | 客户端异质性与 FedAvg 性能关系 | 核心诊断结论保留 |
| `exp1_calendar_periodicity_diagnosis_zh.md` | 日历周期性 baseline 设计 | 基线方法设计保留 |
| `exp1_legacy_ipynb_model_diagnosis_zh.md` | 旧 notebook 模型结构迁移 | 模型结构对比保留 |
| `exp1_fedavg_vs_exp2_full_alignment_diagnosis_zh.md` | Exp1 FedAvg 与 Exp2 Full 对齐验证 | 修复验证记录 |
| `formal_cuda_exp1_exp2_anomaly_diagnosis_zh.md` | Exp1/2 指标异常诊断 | 异常根因分析 |
| `formal_cuda_exp1_exp2_metric_analysis_zh.md` | Exp1/2 formal CUDA 指标分析 | 详细指标分析 |

---

## 2. Exp1 客户端/网格异质性诊断

> 来源：[exp1_client_similarity_diagnosis_zh.md](exp1_client_similarity_diagnosis_zh.md)

**核心问题**：FedAvg (RMSE=20,753) 明显弱于 Independent (14,884) 和 NaiveLastValue (19,419)，是否由 selected clients 异质性造成？

### 三组 clients 对比 (r5e1)

| 组别 | selected clients | mean_pairwise_corr | FedAvg RMSE | Independent RMSE | NaiveLastValue RMSE |
|------|-----|:---:|:---:|:---:|:---:|
| formal_current | 290,284,318,288,289 | 0.669 | 37,344 | 25,765 | 19,419 |
| most_similar_5 | 281,279,341,404,311 | 0.957 | **53,324** | 18,358 | 15,297 |
| least_similar_5 | 287,395,136,322,284 | 0.077 | 101,701 | 25,765 | 7,853 |

### 关键发现
- **相似度越高不一定越好**：most_similar_5 相关性最高（0.957）但 FedAvg 反而更差（53,324 vs 37,344）
- least_similar_5 FedAvg 明显崩坏（101,701），NaiveLastValue 却极强（7,853）
- NaiveLastValue 在三组中都强于 FedAvg
- **相似度不是 FedAvg 弱的主因**

---

## 3. Exp1 日历周期性诊断

> 来源：[exp1_calendar_periodicity_diagnosis_zh.md](exp1_calendar_periodicity_diagnosis_zh.md)

**核心问题**：能否通过日-周周期感知改善 FedAvg？

### 日历特征设计
- 时间范围：2017-04-01 至 2017-05-31（61 天，5856 个 15 分钟时间片）
- 节假日：清明节、劳动节、端午节
- 编码：is_effective_workday, weekday_id, sin/cos_time_of_day, sin/cos_day_of_week

### 周期性 Baseline 方法

| 方法 | 说明 |
|------|------|
| NaiveLastValue | ŷ[t] = x[t-1] |
| DailySeasonalNaive | ŷ[t] = y[t-96] |
| WeeklySeasonalNaive | ŷ[t] = y[t-672] (fallback: t-96 → t-1 → train_mean) |
| CalendarProfileNaive | train 中按 is_effective_workday + slot_of_day 构建 client-specific mean profile |
| CalendarFeature-FedAvg | FedAvg + calendar features 输入（尚未实现） |
| SeasonalResidual-FedAvg | FedAvg 训练残差（尚未实现） |

### 关键结论
- CalendarProfileNaive 使用 train split 构建 profile，无数据泄漏
- CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 属于待实现结构
- 周期性 baseline 表现不等同于周期感知 FedAvg 的表现

---

## 4. Exp1 旧 ipynb 模型结构迁移诊断

> 来源：[exp1_legacy_ipynb_model_diagnosis_zh.md](exp1_legacy_ipynb_model_diagnosis_zh.md)

**核心问题**：旧 notebook 中的模型结构（更大容量、MultiheadAttention）是否优于当前 baseline？

### 模型结构对比

| 特性 | baseline | legacy_ipynb |
|------|----------|-------------|
| hidden_dim | 32 | **64** |
| Conv1d 层 | 2层 (in→16→32) | 2层 (in→64→64) |
| BatchNorm1d | 无 | **第1层Conv后** |
| 激活函数 | ReLU | **AdaptiveSwish** |
| Attention | Linear+Softmax | **nn.MultiheadAttention (4 heads)** |
| 参数总量 | 10,194 | **62,915** |

### 诊断结果

| 配置 | FedAvg RMSE | Independent RMSE | 结论 |
|------|:---:|:---:|------|
| baseline r1e1 | 107,579 | 59,784 | baseline 弱 |
| legacy r1e1 | **89,313** | **35,677** | legacy **-17%** |
| baseline r5e1 | **37,344** | 25,765 | baseline 收敛明显 |
| legacy r5e1 | 102,139 | 23,187 | **legacy 被反超** |

### 关键发现
- r1e1 阶段 legacy 的优势是虚假的（大容量偶然更有效）
- 随着 rounds 增加 (r5e1)，baseline 迅速收敛，legacy 停滞
- **legacy 大模型 (62,915 params) 在 local_epochs=1 下训练不足**
- **不推荐直接使用 legacy，建议回到 baseline 继续**

---

## 5. Exp1 FedAvg 与 Exp2 Full 对齐诊断

> 来源：[exp1_fedavg_vs_exp2_full_alignment_diagnosis_zh.md](exp1_fedavg_vs_exp2_full_alignment_diagnosis_zh.md)

**核心问题**：修复后 exp2 Full 是否与 exp1 FedAvg 完全对齐？

### 修复后对比（Commit: `d2b87f4`）

| 项目 | Exp1 FedAvg | Exp2 Full | 一致? |
|------|:---:|:---:|:---:|
| RMSE | 20,753.14 | **20,753.14** | ✓ |
| MAE | 16,847.81 | **16,847.81** | ✓ |
| MAPE | 0.942 | **0.942** | ✓ |
| R2 | 0.8827 | **0.8827** | ✓ |

**结论：修复后 exp1 FedAvg 与 exp2 Full 完全一致。** 之前怀疑 exp2 Full RMSE=17,345 优于 exp1，经确认来自修复前旧 formal 崩坏数据。

---

## 6. Exp1/2 指标异常诊断

> 来源：[formal_cuda_exp1_exp2_anomaly_diagnosis_zh.md](formal_cuda_exp1_exp2_anomaly_diagnosis_zh.md)

**诊断范围**：Commit `e02fe1a`，formal CUDA 结果只读分析。

### Exp1 结论
- FedAvg 在 0/5 client 上优于 Independent
- 存在跨 client 平均化欠拟合（region 289 最明显）
- FedAvg 已收敛但不足以追平 Independent

### Exp2 异常
- Full/Without Attention/Without CNN 三组 R2≈-700，MAPE≈100，属于明显异常区间
- 原因：**实验 2 缺少归一化/反归一化链路**（`sia_core.py` 未调用 `fit_input_scaler()` 等）
- Without LSTM 虽然最好但 R2 仍为负值

### 可能原因排序
1. LSTM 分支训练不稳定
2. 归一化/反归一化链路缺失（exp2 与 exp1 不一致）
3. variant 开关实现（不是主因）
4. 输出维度或时间步错位（概率低）
5. rounds/learning rate 不适合含 LSTM 模型（放大因素）
6. FedAvg 跨 client 平均化欠拟合

---

## 7. Exp1/2 Formal CUDA 指标分析

> 来源：[formal_cuda_exp1_exp2_metric_analysis_zh.md](formal_cuda_exp1_exp2_metric_analysis_zh.md)

**分析对象**：Commit `9323369` 的 formal CUDA 结果。

### Exp1 核心数据

| 方法 | RMSE | R2 |
|------|------|-----|
| Independent | 14,883.58 | 0.9621 |
| FedAvg | 20,753.14 | 0.8827 |

此版本缺少 NaiveLastValue baseline（`sic_core.py` 未纳入）。

### Exp2 消融（异常版本）

| Variant | RMSE | R2 |
|---------|------|-----|
| Without LSTM | 87,016.61 | -0.1637 |
| Full | 1,818,610.27 | -700.81 |
| Without Attention | 1,818,614.79 | -700.81 |
| Without CNN | 1,818,656.99 | -700.85 |

Full/Without Attention/Without CNN 三组三者差异极其微弱，属于训练失效。

### 收敛分析
- FedAvg train_loss: 0.1976 → 0.0127 (收敛)
- test_rmse: 94,706 → 20,753 (前10轮下降快，后期缓慢)
- 15-17轮有轻微震荡但18-20轮恢复

---

## 8. 当前仍影响后续工作的结论

1. **单 grid cell client 对 FedAvg 不友好**：跨 client 平均化削弱了对异质 client 的拟合能力
2. **client 相似度不是 FedAvg 弱的主因**：相似度最高组反而不如中等组
3. **legacy ipynb 大模型不推荐**：local_epochs=1 下训练不足，被 baseline 反超
4. **Exp1 FedAvg ≡ Exp2 Full**：修复后完全一致，之前差异来自旧 formal 崩坏数据
5. **Exp2 归一化链路必须与 Exp1 对齐**：缺失归一化是 exp2 异常的主要原因
6. **建议继续转向 region/full-cells client 组织**：单网格 client 组织对 FedAvg 不利

---

## 9. 文件来源索引

| 原文件路径 | 归档章节 |
|-----------|---------|
| `real_data_experiments/exp1_client_similarity_diagnosis_zh.md` | §2 |
| `real_data_experiments/exp1_calendar_periodicity_diagnosis_zh.md` | §3 |
| `real_data_experiments/exp1_legacy_ipynb_model_diagnosis_zh.md` | §4 |
| `real_data_experiments/exp1_fedavg_vs_exp2_full_alignment_diagnosis_zh.md` | §5 |
| `real_data_experiments/formal_cuda_exp1_exp2_anomaly_diagnosis_zh.md` | §6 |
| `real_data_experiments/formal_cuda_exp1_exp2_metric_analysis_zh.md` | §7 |
