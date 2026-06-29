# 实验 1–6 论文结果表重构建议

> 生成日期：2026-06-29
> 基于当前实验完成状态，给出论文结果表的最优组织方案

---

## 1. 主结果表建议

### 可纳入的实验

| 实验 | 状态 | 是否可用于论文主表 | 条件 |
|------|------|:---:|------|
| 实验 1 | formal r20e1 ✅ | ✅ 可直接使用 | FedAvg 弱于 Independent 和 NaiveLastValue，如实报告 |
| 实验 3 | 仅 smoke r1e1 | ❌ 暂不可 | 需 r20 formal 后才能纳入 |
| 实验 5 | formal pipeline ✅ 但指标异常 | ❌ 暂不可 | 修复 scaler 后需重跑 |

### 建议主结果表结构

仅纳入实验 1 的 6 个方法（按 RMSE 降序排列，含相对提升率）：

```
表 X: Grid-Cell-Level Client 预测性能对比 (Real Data, 5 grid cell clients, r20e1)

| Method | RMSE | MAE | MAPE(%) | R² | vs NaiveLastValue RMSE Δ% | vs Independent RMSE Δ% |
|--------|------|-----|---------|-----|--------------------------|------------------------|
| Independent | 15,921 | 11,615 | 0.650 | 0.953 | -18.0% | — |
| NaiveLastValue | 19,419 | 13,620 | 0.758 | 0.939 | — | +22.0% |
| FedAvg | 24,158 | 19,773 | 1.115 | 0.847 | +24.4% | +51.7% |
| CalendarProfileNaive | 32,194 | 22,770 | 1.269 | 0.830 | +65.8% | +102.2% |
| DailySeasonalNaive | 45,406 | 29,727 | 1.641 | 0.637 | +133.8% | +185.2% |
| WeeklySeasonalNaive | 48,369 | 32,881 | 1.831 | 0.551 | +149.1% | +203.8% |
```

**注意**: FedAvg RMSE 相比 Independent 高出 **+51.7%**，相比 NaiveLastValue 高出 **+24.4%**。论文不能宣称 FedAvg 优于 baselines，应如实报告并分析原因。

---

## 2. 消融结果表建议

### 可纳入的消融

| 实验 | 状态 | 是否可用 | 条件 |
|------|------|:---:|------|
| 实验 2 | 代码存在，结果已删除 | ⚠️ 需恢复 | 恢复历史结果或重跑 r20e3 formal |
| 实验 6 | 仅 full variant + 指标异常 | ❌ 不可用 | 修复 scaler + 补全 4 variant + 重跑 |
| 实验 4 | 未开发 | ❌ 不可用 | 需开发后再跑 |

### 建议消融表结构（如实验 2 恢复）

实验 2 修复后的历史结果（来自 report `formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md`）：

```
表 Y: Single Grid Cell 模型结构消融 (Real Data, 5 grid cell clients, r20e3)

| Variant | RMSE | MAE | MAPE(%) | R² | vs Full RMSE Δ% |
|---------|------|-----|---------|-----|-----------------|
| Full | 20,753 | 16,848 | 0.942 | 0.883 | — |
| Without Attention | 18,976 | 14,856 | 0.838 | 0.901 | -8.6% |
| Without LSTM | 19,332 | 15,119 | 0.853 | 0.897 | -6.8% |
| Without CNN | 23,371 | 19,837 | 1.101 | 0.858 | +12.6% |
```

---

## 3. 指标建议

### 主表指标（必须）

| 指标 | 优先级 | 说明 |
|------|:---:|------|
| MSE | P0 | 基础回归指标 |
| RMSE | P0 | 与原始数据同单位，直观 |
| MAE | P0 | 对异常值不敏感 |
| MAPE | P1 | 百分比误差，有量纲感知 |

### R²

- **不作为论文主指标**
- 仅作为附录或诊断指标
- 当前实验 5/6 的 R² 为负，若出现在论文中需额外解释

### 必须包含的比较

| 对比 | 说明 |
|------|------|
| FedAvg vs Independent | 联邦 vs 本地训练的差异 |
| FedAvg vs NaiveLastValue | 联邦模型是否超过简单 persistence |
| 消融 variants vs Full | 各组件贡献 |
| Client-level variance | 跨 client 的 RMSE/MAE 标准差（体现异质性） |

---

## 4. 相对提升率公式

论文中应给出相对提升率，便于审稿人快速理解：

```
Δ%_RMSE = (FedAvg_RMSE - Baseline_RMSE) / Baseline_RMSE × 100%
Δ%_MAE  = (FedAvg_MAE - Baseline_MAE) / Baseline_MAE × 100%
```

**注意**: 若提升率为正（即 FedAvg 更差），标记为不适合论文主结论，如实写入"FedAvg did not outperform the local baselines in this setting."

---

## 5. Client-Level Variability 表

建议补充 per-client 指标表，回应对 client 异质性的关切：

```
表 Z: Per-Client RMSE 对比 (Experiment 1, r20e1)

| Client ID (Region) | FedAvg RMSE | Independent RMSE | NaiveLastValue RMSE | FedAvg vs Independent Δ% |
|-------------------|:-----------:|:----------------:|:-------------------:|:------------------------:|
| 290 | 20,877 | 15,974 | — | +30.7% |
| 284 | 23,630 | 19,120 | — | +23.6% |
| 318 | 16,494 | 15,105 | — | +9.2% |
| 288 | 18,161 | 14,640 | — | +24.0% |
| 289 | 24,604 | 9,578 | — | +156.9% |
```

Client 289 的 FedAvg 相比 Independent 恶化 156.9%，是跨 client 平均化问题的集中体现。

---

## 6. 收敛曲线建议

实验 1 的 convergence_history 显示 clear 收敛趋势（RMSE 从 94,706 → 20,753），可放入论文作为 FedAvg 训练收敛性的证据。

---

## 7. 当前论文可写的结论

基于当前可用结果，论文可安全写出以下结论：

1. **Positive**: FedAvg 在 grid-cell-level client 设置下可以稳定收敛（r1→r20 RMSE 下降 78%）
2. **Negative (如实报告)**: FedAvg 在 5/5 clients 上均弱于 Independent，整体 RMSE 比 Independent 高 51.7%
3. **Analysis**: Client 289 是最不友好的 client，FedAvg 相比 Independent 恶化 156.9%，反映跨 client 平均化对异质性 client 的负面影响
4. **Baseline**: NaiveLastValue (RMSE=19,419) 强于 FedAvg (RMSE=24,158)，说明数据具有强时间惯性
5. **Calendar**: CalendarProfileNaive (RMSE=32,194) 弱于 FedAvg，说明简单周期 profile 不足以捕捉交通流动态
6. **Limitations**: 当前仅在 5 个 grid cells 上验证，更全面的 cluster-level 和 global partition 设置因训练链路问题尚未形成可信结论
