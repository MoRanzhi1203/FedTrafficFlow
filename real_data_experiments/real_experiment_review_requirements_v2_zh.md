# 一审意见驱动的真实数据实验修复要求 v2

> 生成日期：2026-07-01
> 基准 commit：d69833e
> 分支：feature/real-exp4-rfc-ablation

## 1. 来自审稿人与导师会议的真实实验要求

| 要求 | 具体含义 | 当前实验缺口 | 对应修复任务 |
|---|---|---|---|
| 多区域数据定量支撑 | 需要区域分布、client 样本量、异质性指标 | Exp3/5 有 non_iid_summary.csv 但不够系统 | 生成统一 client heterogeneity 分析脚本 |
| 收敛性和稳健性 | 每轮 loss/RMSE 曲线、client-level 曲线 | Exp1 有 convergence_history.json；Exp3/5/6 仅 r1e1/r3e1 | 统一所有实验的 round-level convergence 输出 |
| client 数量 | 当前真实实验 K 过少 | Exp5 K=3, Exp3 K=5 | 增加 k=8/10 对比 |
| 对比基线 | 不能只比 Independent | Exp3/5 缺 FedProx、LocalFT、CentralizedUpperBound | 将 Exp1 mechanism eval 能力扩展到 Exp3/5 |
| 个性化机制 | 显式建模 client 分布差异 | 仅 Exp1 有 LocalFT；无 client-specific head | 在 common/ 中实现通用 LocalFT 和 FedProx |
| 聚合策略 | 不只 sample-count FedAvg | 论文旧稿写了 λ/β/ρ 但代码未实现 | 修改论文为纯 FedAvg，或实现缺失策略 |
| 空间依赖 | 固定 CNN/Grid 不足 | 无真实数据 GCN 实验 | 生成 GCN feasibility 文档或轻量 diagnostic |
| 外部因素 | 天气/事件/信号缺失 | 无法完全获取 | calendar/holiday 分组分析；其余写 limitation |
| 节假日 | 2017 Apr-May 含清明节等 | 之前口径有误 | 做 holiday/weekend 分组指标 |
| 超参数与可复现性 | 训练设置、硬件、split、runtime | 已有 hyperparameter_tables_zh.md | 更新至 v2 完整版本 |
| 通信开销/掉线/DP | 需说明 FL 安全性 | 当前无实验 | 通信开销估计脚本 + limitation 文档 |
| 结果解释 | 不能只报 RMSE | Exp1 有 client_metrics；Exp3/5/6 缺 | 新增统一汇总脚本 |

## 2. 实验修复优先级

### P0：必须完成

1. 聚合策略代码审计 [aggregation_strategy_code_audit_zh.md](./aggregation_strategy_code_audit_zh.md) — ✅ 已完成
2. 将 Exp1 的 FedProx/LocalFT 能力提取到 `common/`，扩展到 Exp3/5
3. 补齐 Exp6 消融 variants (without_attention/without_cnn/without_lstm)
4. 恢复或重跑 Exp2 消融
5. 生成 k=8/k=10 partition 文件，Exp3/5 k 扩展 smoke
6. 生成 client heterogeneity 统一分析脚本
7. 生成通信开销估计脚本

### P1：高优先级

1. CalendarFeatureFedAvg 扩展到 Exp3/5
2. Calendar weekday/weekend/holiday 分组评估
3. GCN feasibility 或轻量 diagnostic
4. 多 seed 稳健性 (seed=42, 123, 2026)
5. 更新所有文档

### P2：中优先级

1. 实现 loss-aware / smoothed aggregation（如果论文保留 λ/β/ρ）
2. client-specific head 个性化
3. dropout simulation
4. DP privacy 分析文档

## 3. 当前代码能力矩阵

| 能力 | Exp1 | Exp2 | Exp3 | Exp4 | Exp5 | Exp6 |
|---|---|---|---|---|---|---|
| FedAvg | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FedProx | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| LocalFT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| FedAvg+LocalFT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| FedProx+LocalFT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CentralizedUB | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Independent | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NaiveLastValue | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| CalendarProfileNaive | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| DailySeasonalNaive | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| WeeklySeasonalNaive | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| CalendarFeatureFedAvg | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 结构消融 (4 variants) | ❌ | ✅ | ❌ | ✅ | ❌ | ⚠️ |
| grouped_metrics_by_cal | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| convergence_history | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| client_metrics | ✅ | ❌ | ✅ | ⚠️ | ✅ | ⚠️ |
| non_iid_summary | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
