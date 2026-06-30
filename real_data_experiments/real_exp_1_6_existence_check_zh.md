# 真实数据实验 1–6 存在性检查报告

> 生成日期：2026-06-30
> 本轮只做静态检查，不运行实验，不修改源码。

---

## 1. Git 状态

- **分支**: `feature/real-exp4-rfc-ablation`
- **HEAD**: `1ceb287` — `docs(real-data): update hyperparameter tables, result plan, and reviewer mapping for Exp4`
- **staged 区是否为空**: 是
- **是否存在未跟踪 results**: 是（`results/real_data_experiments/smoke/` 等多处未跟踪）
- **本轮是否运行实验**: 否
- **本轮是否修改源码**: 否

---

## 2. 实验 1–6 代码存在性总表

| 实验 | 新版含义 | 代码入口 | config | CLI/variants | scaler | baseline | 代码层面是否存在 | 说明 |
|---|---|---|---|---|---|---|---|---|
| **Exp1** | 单 grid cell 主实验 | ✅ `sic_core.py` | ✅ `sic_config.py` | ✅ FedAvg/Independent/NaiveLastValue/DailySeasonal/WeeklySeasonal/CalendarProfile | ✅ input+target | ✅ 6 种方法 | **存在** | 功能完整 |
| **Exp2** | 单 grid cell 消融 | ✅ `sia_core.py` | ✅ `sia_config.py` | ✅ full/without_attention/without_cnn/without_lstm | ✅ input+target | N/A (消融) | **存在** | 独立消融入口，DEFAULT_VARIANTS 完整 |
| **Exp3** | 多相似 cell 主实验 | ✅ `rfc_core.py` | ✅ `rfc_config.py` | ✅ FedAvg/Independent/NaiveLastValue + `--partition-file` | ✅ input+target | ✅ 3 种方法 | **存在** | similarity_k5.json 可用 |
| **Exp4** | 多相似 cell 消融 | ✅ `rfc_ablation_core.py` | ✅ `rfc_ablation_config.py` | ✅ full/without_attention/without_cnn/without_lstm | ✅ input+target | N/A (消融) | **存在** | 独立消融入口已补全，复用 Exp3 similarity_k5 client 构造 |
| **Exp5** | 全部 grid cells 主实验 | ✅ `rc_core.py` | ✅ `rc_config.py` | ✅ spatial_block / flow_kmeans | ✅ input+target | ✅ FedAvg/Independent/NaiveLastValue | **存在** | 功能完整 |
| **Exp6** | 全部 grid cells 消融 | ✅ `ra_core.py` | ✅ `ra_config.py` | ✅ full/without_attention/without_cnn/without_lstm | ✅ input+target | N/A (消融) | **存在** | 输出 ablation_metrics.csv + ablation_summary.csv |

---

## 3. 实验 1–6 结果存在性总表

| 实验 | smoke 目录 | formal 目录 | diagnostic 目录 | metrics 文件 | 结果层面是否存在 | 说明 |
|---|---|---|---|---|---|---|
| **Exp1** | 无独立 smoke | ✅ `formal/exp1_single_grid_baseline_r20_e1_cuda/` | 无 | ✅ `main_metrics.csv` (R² 正常 0.55-0.95) | **存在，可用** | 6 种方法指标正常 |
| **Exp2** | 无 | ❌ 历史目录已删除 | 无 | 无 | **不存在** | 历史 formal 结果目录 (`exp2_single_grid_ablation_formal_cuda*`) 已删除 |
| **Exp3** | ✅ `smoke/exp3_rfc_similarity_k5_r1e1/` | ❌ 不存在 | 无 | ✅ smoke `main_metrics.csv` | **仅 smoke** | formal 尚未运行 |
| **Exp4** | ✅ diagnostic smoke: `exp4_rfc_ablation_similarity_k5_all_variants_r1_e1_cuda_1k/` | ❌ formal 未运行 | ✅ diagnostic 存在 | ✅ `ablation_metrics.csv` + `convergence_history.csv` | **diagnostic 存在，formal 不存在** | all variants smoke 通过，仅验证 pipeline |
| **Exp5** | ✅ `smoke/exp5_rc_spatial_block_k3_r1e1/` + `exp5_rc_flow_kmeans_k3_r1e1/` | ⚠️ formal 存在但**指标异常** (R² < 0) | ✅ `exp5_rc_spatial_block_k3_r3_e1_cuda_scaler_fix*` (2 个) | ✅ formal 文件存在但内容损坏 | **旧结果不可用** | 旧 formal 为 scaler 修复前运行，MAPE≈100%, R²<0；diagnostic 有 scaler 修复版 |
| **Exp6** | ✅ `smoke/exp6_ra_spatial_block_k3_r1e1/` | ⚠️ formal 仅 `full` variant，且**指标异常** (R²<0) | 无 | ✅ `ablation_metrics.csv` (仅 1 行 full) | **旧结果不可用** | 与 Exp5 同源损坏；仅 full variant，缺少 without_attention/without_cnn/without_lstm |

### 3.1 结果质量详情

**Exp1** (正常):
```
method         RMSE      MAE      R²
FedAvg         24,158    19,773   0.847
Independent    15,921    11,615   0.953
NaiveLastValue 19,419    13,620   0.939
CalendarNaive  32,194    22,770   0.830
DailyNaive     45,406    29,727   0.637
WeeklyNaive    48,369    32,881   0.551
```

**Exp5 / Exp6** (异常 — scaler 修复前):
```
method         RMSE        MAPE    R²
FedAvg         ~627,000    ~100%   -1.13
Independent    ~627,000    ~98%    -1.13
```
> 此为旧结果。代码中 scaler (input_normalization + target_normalization) 已接入（代码检查确认），但当前 formal 结果是修复前运行的，不可用。

---

## 4. 文档入口检查

| 文档 | 存在 | 说明 |
|------|:---:|------|
| `real_data_experiments/README.md` | ✅ | 文档入口索引 |
| `real_exp_1_6_current_status_and_revision_plan_zh.md` | ✅ | 当前权威总控 |
| `real_exp_1_6_hyperparameter_tables_zh.md` | ✅ | 超参数表 |
| `real_exp_1_6_result_table_plan_zh.md` | ✅ | 结果表计划 |
| `real_exp_5_6_training_failure_diagnosis_zh.md` | ✅ | 训练失效诊断 |
| `reviewer_response_experiment_mapping_zh.md` | ✅ | 审稿映射表 |
| `real_exp_1_6_legacy_reports_archive_zh.md` | ✅ | 历史报告归档 |
| `real_exp_diagnostics_archive_zh.md` | ✅ | 诊断报告归档 |

---

## 5. 结论

- **Exp1**：代码存在 ✅ / 结果存在且正常 ✅ / **可用**
- **Exp2**：代码存在 ✅ / 当前结果缺失 ❌ / **需恢复或重跑**
- **Exp3**：代码存在 ✅ / smoke 存在 ✅ / formal 缺失 ❌ / **可直接进入 formal**
- **Exp4**：代码存在 ✅ / diagnostic smoke 存在 ✅ / formal 缺失 ❌ / **已通过 all variants 1k r1e1 smoke，但不能作为论文正式结果。**
- **Exp5**：代码存在 ✅ / 旧 formal 存在但结果异常 ❌ (scaler 修复前) / diagnostic 有 scaler 修复版 ⚠️ / **需用修复后代码重跑 formal**
- **Exp6**：代码存在 ✅ / 旧 formal 仅 full variant 且异常 ❌ / **需补完整消融 (4 variants) 并重跑**

---

## 6. 下一步建议

1. **Exp4 已补全代码入口**：`rfc_ablation_core.py` + `rfc_ablation_config.py`，4 variants smoke 通过。下一步先跑 r5 diagnostic 验证多轮收敛，再决定是否进入 r20 formal。
2. **Exp2 结果不存在**：历史 formal 目录已删除，需恢复（若有备份）或重跑 `sia_core`。
3. **Exp3 仅 smoke**：pipeline 已验证，代码完整，可直接进入 formal (rounds=20)。
4. **Exp5/6 代码存在但结果不可用**：
   - 原因：当前 formal 结果为 scaler 修复前运行，指标异常 (MAPE≈100%, R²<0)
   - 代码中 scaler 已接入（`input_normalization=True`, `target_normalization=True`），修复代码已就绪
   - 建议先跑 exp5 medium diagnostic (r3e1) 验证 scaler 修复效果后，再跑完整 formal (r20e1)
5. **不建议直接 full formal** — 优先 diagnostic 验证 scaler 修复效果。
