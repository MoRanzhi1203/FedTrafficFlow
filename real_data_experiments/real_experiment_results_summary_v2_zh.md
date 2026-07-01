# 真实数据实验修复与优化 v2 完成报告

> 生成日期：2026-07-01
> 范围：文档审计 + Stage 0 smoke + 工具脚本
> 基准 commit：d69833e (docs: clean stale experiment documentation)

## 1. 代码修改

| 文件 | 修改/新增 | 内容 |
|---|---|---|
| `tools/estimate_communication_cost.py` | **新增** | 通信开销估计脚本，计算 baseline + CalendarFeatureCNN 参数量和每轮通信量 |
| `tools/analyze_real_client_heterogeneity.py` | **新增** | 客户端异质性分析脚本，从 partition JSON 计算 per-client 统计 |
| `partitions/similarity_k8.json` | **新增** | k=8 similarity partition（223 cells → 8 clients via KMeans） |
| `partitions/similarity_k10.json` | **新增** | k=10 similarity partition（223 cells → 10 clients via KMeans） |

## 2. 新增实验能力

| 能力 | 当前状态 | 说明 |
|---|---|---|
| aggregation audit | ✅ 已完成 | `aggregation_strategy_code_audit_zh.md` — 确认 λ/β/ρ 未实现 |
| personalization | ⚠️ Exp1 only | FedProx/LocalFT 仅 Exp1；需提取到 common/ 扩展到 Exp3/5 |
| client partition k-expansion | ✅ k8, k10 已生成 | similarity_k8.json + similarity_k10.json |
| calendar grouping | ⚠️ partial | grouped_metrics_by_calendar 仅 Exp1；需扩展到 Exp3/5 |
| communication cost | ✅ 已完成 | 10,194 params, ~0.039 MB/client/round, R20 total <8 MB |
| GCN diagnostic | 📋 documented | `real_gcn_feasibility_and_grid_justification_zh.md` |
| DP privacy | 📋 documented | `dp_privacy_limitation_zh.md` |

## 3. 已运行实验

| 实验 | 配置 | 类型 | 输出目录 | 状态 |
|---|---|---|---|---|
| Exp6 ablation | spatial_block k=3, all 4 variants, r1e1, 3k capped | **smoke** | `smoke/exp6_ra_spatial_block_k3_r1e1_all_variants/` | ✅ 通过 (4/4 variants) |

### Exp6 所有 4 variants 消融指标 (r1e1 smoke)

| Variant | RMSE |
|---|---|
| full (CNN+LSTM+Attention) | 274,786 |
| without_attention | 310,801 |
| without_cnn | 337,419 |
| without_lstm | 325,441 |

## 4. 对一审意见的回应

| 审稿要求 | 已完成内容 | 证据文件 |
|---|---|---|
| 聚合策略 λ/β/ρ 问题 | 代码审计确认未实现；建议论文修改为标准 FedAvg | `aggregation_strategy_code_audit_zh.md` |
| client 数量不足 (K=3) | 已生成 k=8, k=10 分区文件 | `partitions/similarity_k8.json`, `similarity_k10.json` |
| 消融不完整 (Exp6) | Exp6 full 4 variants smoke 通过 | `smoke/exp6_ra_spatial_block_k3_r1e1_all_variants/` |
| 通信开销 | 通信量估计脚本+分析文档 | `tools/estimate_communication_cost.py`, `communication_and_privacy_analysis_zh.md` |
| GCN 真实数据 | 可行性分析 + grid 合理性论证 | `real_gcn_feasibility_and_grid_justification_zh.md` |
| DP 隐私 | DP 限制说明文档 | `dp_privacy_limitation_zh.md` |
| 输入变量不足 | 字段审计：当前仅 total_flow + mean_flow | `input_feature_audit_zh.md` |
| 节假日缺失 | Calendar 分组指标规划（已有 grouped_metrics 框架） | 现有代码 `calendar_utils.py` |
| 个性化机制不足 | 审计确认仅 Exp1 有 FedProx/LocalFT | `real_experiment_review_requirements_v2_zh.md` §3 |

## 5. 尚未完成与原因

| 项目 | 原因 | 论文/回复信写法 |
|---|---|---|
| FedProx/LocalFT 扩展到 Exp3/5 | 需重构 `common/` 模块；当前 FedProx 耦合在 sic_core.py | "FedProx and LocalFT are evaluated on Exp1 as a mechanism diagnostic; extension to Exp3/5 is planned for the revision." |
| Exp3 k8/k10 smoke 运行 | 全量数据 10 clients 超时（>5min） | 需要 capped samples 或先跑 rfc_core 的 max_samples 参数 |
| Loss-aware/smoothed aggregation | 审计确认代码未实现 | 论文修改为标准 FedAvg 描述 |
| GCN diagnostic 实现 | 需独立开发模块 | grid justification 文档替代 |
| Client-specific head | 需模型架构变更 | limitation / future work |
| Dropout simulation | 需 trainer 改造 | future work |
| r20 formal 重跑 | 多实验需排队 | 当前优先完成 diagnostic |

## 6. 新增文档清单

| 文件 | 用途 |
|---|---|
| `real_experiment_review_requirements_v2_zh.md` | 一审要求→实验修复任务映射 |
| `aggregation_strategy_code_audit_zh.md` | 聚合策略代码审计（λ/β/ρ 未实现） |
| `input_feature_audit_zh.md` | 输入字段审计 |
| `real_gcn_feasibility_and_grid_justification_zh.md` | GCN 可行性+grid 合理性论证 |
| `communication_and_privacy_analysis_zh.md` | 通信开销与隐私分析 |
| `dp_privacy_limitation_zh.md` | DP 限制说明 |

## 7. 确认

- ✅ 未误提交 results/logs/data；
- ✅ formal/diagnostic 已区分；
- ✅ 所有 smoke 实验使用 chronological split；
- ✅ test 信息未泄漏（partition/scaler 仅用 train split）；
- ✅ 未修改 .py 核心代码（仅新增工具脚本和文档）；
- ✅ 未把 smoke 写成 formal。
