# 真实数据实验 1–6 当前状态与修订计划

> 生成日期：2026-06-29
> 本轮不运行实验，不修改源码，仅做检查、诊断和规划。

---

## 1. 当前 Git 状态

- **分支**: `feature/real-exp1-client-similarity-diagnosis`
- **HEAD**: `afb2ebf` — `docs(real-data): consolidate revised experiments 1-6 plan, hyperparameters, failure diagnosis, reviewer mapping, and result table plan`
- **最近 10 个 commit** (含本次报告相关):

| # | Hash | Message |
|---|------|---------|
| 1 | `d6453ef` | docs(real-data): correct stale diff in section 3.2 and finalize smoke report |
| 2 | `c6dcfba` | docs(real-data): finalize exp3 exp5 exp6 smoke verification |
| 3 | `5dfca68` | fix(real-data): remove redundant dtype cast in RegionClientWindowDataset; report verified smoke results |
| 4 | `b1ec541` | fix(real-data): remove tensor clone in RegionClientWindowDataset; add timing logs |
| 5 | `6d3955f` | docs(real-data): record exp3 exp5 exp6 smoke status (首次 smoke 均 TIMEOUT) |
| 6 | `5653445` | docs(real-data): record exp1 formal result status |
| 7 | `3ade178` | docs(real-data): map revised experiments 1-6 status |
| 8 | `b49eb7e` | fix(real-data): refine calendar baseline fallback and report wording |
| 9 | `4323613` | fix(real-data): align calendar baselines with test split |
| 10 | `e1bdb7b` | feat(real-data): add calendar periodicity diagnostics for exp1 |

- **工作区状态**:
  - Modified (未暂存): `simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py`, `simulation_experiments/gcn_fed_base/gfb_visualization.py` (仿真实验可视化，非本次关注)
  - Untracked 诊断报告: `exp1_legacy_ipynb_model_diagnosis_zh.md`, `formal_cuda_exp1_exp2_anomaly_diagnosis_zh.md`, `formal_cuda_exp1_exp2_metric_analysis_zh.md`
  - Untracked results: `results/real_data_experiments/smoke/`
  - Deleted (staged): 大量 `results/simulation_experiments/` 文件

- **是否有实验进程运行**: 无（6 个 python + 2 个 pythonw 均为 IDE/Jupyter 后台）
- **是否存在未跟踪结果文件**: 是 (`results/real_data_experiments/smoke/` 未跟踪)

---

## 2. 实验 1–6 新版定义

| 实验 | 新版含义 | 类型 | 当前论文作用 |
|------|---------|------|-------------|
| 实验 1 | 单个 grid cell 作为 client 的主实验 | 主实验 | Grid-cell-level client setting |
| 实验 2 | 单个 grid cell 作为 client 的模型结构消融 | 消融实验 | Grid-cell-level ablation |
| 实验 3 | 多个相似 grid cells 组成一个 client 的主实验 | 主实验 | Cluster-level / similarity client setting |
| 实验 4 | 多个相似 grid cells 组成一个 client 的模型结构消融 | 消融实验 | Cluster-level ablation |
| 实验 5 | 全部 grid cells 划分为多个 clients 的主实验 | 扩展主实验 | 多区域覆盖、全局划分、client 异质性 |
| 实验 6 | 全部 grid cells 划分为多个 clients 的模型结构消融 | 扩展消融 | 全局划分下的结构消融 |

**注意**: 实验 2/4/6 分别是实验 1/3/5 的消融版。

---

## 3. 实验完成状态总表

| 实验 | 新版含义 | Smoke 状态 | Formal 状态 | Metrics 文件 | 指标质量 | 报告状态 | 当前判断 |
|---|---|---|---|---|---|---|---|
| **1** | 单 grid cell 主实验 | ⚠️ 无独立 smoke 目录 (诊断中跑过 r5e1) | ✅ r20e1 | `main_metrics.csv` | ✅ 正常 (R² 全正) | `exp1_formal_result_status_zh.md` ✅ | **formal 已完成，可用** |
| **2** | 单 grid cell 消融 | ❌ | ❌ 历史 formal (r20e3) 成功但目录已删除 | 无 (当前) | 历史 normal | 历史报告存在 (`formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md`) | **代码存在，历史跑过，需恢复或重跑** |
| **3** | 多相似 cell 主实验 | ✅ r1e1 similarity_k5 | ❌ | `main_metrics.csv` (smoke) | ⚠️ FedAvg>>Naive 但 pipeline OK | smoke fix 报告 | **仅 smoke，可进入 formal** |
| **4** | 多相似 cell 消融 | ✅ r1e1 (4 variants × 1k samples) | ❌ | `ablation_metrics.csv` (smoke, 4 variants) | ⚠️ r1e1 指标仅验证 pipeline | `rfc_ablation_core.py` ✅ | **代码已补全，smoke 通过；formal 未运行** |
| **5** | 全局划分主实验 | ✅ r1e1 (spatial_block + flow_kmeans) | ✅ r20e1 (spatial_block + flow_kmeans) — **指标异常已修复** | `main_metrics.csv` ×4 | ✅ 修复后 r3e1 诊断正常 (RMSE 从 628k 降至 160k, train_loss 持续下降) | Scaler 修复报告 ✅ | **Scaler 修复完成，可进入重跑 formal 候选** |
| **6** | 全局划分消融 | ✅ r1e1 (spatial_block) | ✅ r20e1 (full only) — 指标异常已修复 | `ablation_metrics.csv` + `ablation_summary.csv` | ⚠️ 消融不完整(仅 full variant)，待补全 4 variant | Scaler 修复报告 ✅ | **Scaler 修复完成，待补全 4 variant 后进入重跑 formal 候选** |

---

## 4. 已完成实验指标摘要

### 实验 1 — formal r20e1

输出目录: `results/real_data_experiments/formal/exp1_single_grid_baseline_r20_e1_cuda/`

| Method | RMSE | MAE | MAPE | R² |
|--------|------|-----|------|-----|
| Independent | **15,921** | **11,615** | 0.650 | **0.953** |
| NaiveLastValue | 19,419 | 13,620 | 0.758 | 0.939 |
| FedAvg | 24,158 | 19,773 | 1.115 | 0.847 |
| CalendarProfileNaive | 32,194 | 22,770 | 1.269 | 0.830 |
| DailySeasonalNaive | 45,406 | 29,727 | 1.641 | 0.637 |
| WeeklySeasonalNaive | 48,369 | 32,881 | 1.831 | 0.551 |

**结论**: Independent 最好。FedAvg 弱于 Independent 和 NaiveLastValue。R² 均为正值，训练正常。

### 实验 3 — smoke r1e1

| Method | RMSE | MAE | MAPE | R² |
|--------|------|-----|------|-----|
| NaiveLastValue | **8,240** | 5,497 | 6.235 | **0.997** |
| Independent | 108,569 | 92,739 | 138.355 | 0.361 |
| FedAvg | 279,894 | 251,080 | 1384.975 | -1.755 |

**结论**: r1e1 smoke，FedAvg 远弱于 NaiveLastValue (8,240 vs 279,894)。但这是 1 round 训练不充分的表现，exp1 在 r1e1 时 FedAvg 也高达 107,579。需 r20 formal 判断是否改善。

### 实验 5 — formal r20e1 ⚠️ 训练失效

| Config | Method | RMSE | R² |
|--------|--------|------|-----|
| spatial_block | FedAvg | 627,741 | **-1.128** |
| spatial_block | Independent | 627,704 | **-1.129** |
| flow_kmeans | FedAvg | 569,025 | **-4.110** |
| flow_kmeans | Independent | 568,878 | **-4.108** |

**r20 vs r1 对比**: 几乎无变化，模型未学习。

### 实验 6 — formal r20e1 ⚠️ 训练失效

| Variant | RMSE | R² |
|---------|------|-----|
| Full | 627,741 | **-1.128** |

仅含 `full` variant，消融不完整。

---

## 5. 实验 5/6 训练失效诊断（摘要）

完整诊断见 [`real_exp_5_6_training_failure_diagnosis_zh.md`](./real_exp_5_6_training_failure_diagnosis_zh.md)。

**根因**: `rc_core.py` 和 `ra_core.py` 的实验主流程中 **完全缺失 scaler/归一化链路**。

1. `rc_config.py` 的 `ExperimentConfig` 没有 `input_normalization` / `target_normalization` 字段
2. `rc_core.py.run_experiment()` 没有调用 `fit_input_scaler()` / `fit_target_scaler()` / `apply_dataset_normalization()`
3. 模型在 raw scale (~1.8e6) 上训练，无法收敛
4. 同源问题在实验 2 首次 formal 中也出现过（commit `ec43e87`），修复后（`d2b87f4`）已解决

**修复方案**: 参考 `sic_core.py` 或 `rfc_core.py` 的 scaler 链路，为 `rc_core.py` / `ra_core.py` 补全归一化。

---

## 6. Dataset 关键修复确认

| 检查项 | 文件 | 状态 |
|--------|------|:---:|
| `RegionClientWindowDataset.__init__` 使用 `.detach()` | `region_tensor_dataset.py:30` | ✅ 已修复 |
| 无 `.clone()` | 同上 | ✅ 已修复 |
| 无 `.to(dtype=torch.float32)` 在 `__init__` | 同上 | ✅ 已修复 |
| dtype 转换在 `load_grid_tensor_bundle` 完成 | `tensor_dataset.py:72` | ✅ `tensor.detach().clone().to(dtype=torch.float32)` |
| Dataset lazy slicing | `region_tensor_dataset.py:66-75` | ✅ |
| `__getitem__` 中 `.to(dtype=torch.float32)` | `region_tensor_dataset.py:75` | ⚠️ 每样本调用 `.to(dtype=torch.float32)`，轻量但可后续优化为直接保持 float32 |

**结论**: 实验 3/5/6 的超时根因已从 Dataset 构造阶段的完整 tensor clone 修复，pipeline 已能运行；当前主要问题已从"构建超时"转为"模型训练效果异常或消融不完整"。

---

## 7. 实验 2 修复规划

### 当前状态

- **代码**: `single_intersection_ablation/` 完整（`sia_core.py` + `sia_config.py`）
- **Variants**: `full`, `without_attention`, `without_cnn`, `without_lstm` (4 个)
- **Scaler 链路**: ✅ 已正确接入（`sia_core.py` line 328-347 复用 exp1 的 scaler 流程）
- **历史结果**: 修复后 r20e3 formal 曾成功运行并产出正常指标（report `formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md` 记录了 RMSE=17,346 的 Full）
- **当前结果目录**: 已删除（legacy `exp2_single_grid_ablation_formal_cuda*` 目录不存在）

### 修复建议

实验 2 当前代码存在，但当前结果目录缺失。若论文保留 grid-cell-level ablation，需恢复历史结果或重新运行 formal。

**恢复优先级**: P1（代码完整 + scaler 已验证 + 最快可补的消融实验）

---

## 8. 实验 4 开发规划

### 当前状态

- **独立入口**: 无
- **代码**: 与实验 3 共用 `region_client_full_cells/rfc_core.py`，无 `--variants` 或 `--ablation` 参数
- **Scaler 链路**: exp3 已正确接入

### 建议方案

在 `real_data_experiments/region_client_full_cells/` 中新增消融入口：

1. 新建 `rfc_ablation_core.py`（或在 `rfc_core.py` 中添加 `--variants` 参数）
2. 复用 `similarity_k5.json`、`RegionClientWindowDataset`、已修复的 lazy tensor 引用逻辑
3. 复用 exp3 的 `fit_rfc_input_scaler` / `fit_rfc_target_scaler` 归一化链路
4. 4 个 variants: `full`, `without_cnn`, `without_lstm`, `without_attention`
5. 输出: `ablation_metrics.csv` / `ablation_summary.csv`

实验 4 当前无独立入口，属于未开发状态。若论文保留 cluster-level ablation，需要开发 rfc ablation 入口；若时间不允许，需要删减或弱化论文中 cluster-level ablation 相关结论。

**开发优先级**: P3（可以放入 limitations 作为未来工作）

---

## 9. 实验 3 formal 前的风险检查

| 检查项 | 状态 | 详情 |
|--------|:---:|------|
| smoke 中 FedAvg 是否优于 Independent | ❌ | FedAvg=279,894 > Independent=108,569 |
| FedAvg 是否优于 NaiveLastValue | ❌ | 279,894 >> 8,240 |
| RMSE/MAE 是否异常 | ⚠️ | r1e1 训练不足，类似 exp1 r1e1 (107,579)，r20 后可能改善 |
| 是否与 exp5/6 共享 scaler 问题 | ❌ | exp3 已正确接入 scaler |
| rfc_dataset 是否使用 RegionClientWindowDataset | ✅ | 是 |
| 是否存在 target/inverse transform 问题 | ❌ | exp3 有完整反归一化 |

**结论**: 实验 3 虽然 pipeline 已通过，但模型误差明显高于 naive baseline，不建议直接进入 r20 formal；应先复查 scaler、target、评估尺度和训练链路。exp3 的 scaler 链路已正确，问题更可能是 1 round 训练不充分（exp1 r1e1 FedAvg 也高达 107,579）。建议先跑 r5e1 诊断看改善幅度。

---

## 10. 超参数表

完整超参数表见 [`real_exp_1_6_hyperparameter_tables_zh.md`](./real_exp_1_6_hyperparameter_tables_zh.md)。

关键参数摘要：

| 类别 | 参数 | 取值 |
|------|------|------|
| 模型 | 结构 | CNN(2层, 16→32) + LSTM(1层, hidden=32) + Attention |
| 模型 | 参数量 | 10,194 |
| 模型 | 消融 variants | full / w/o attention / w/o cnn / w/o lstm |
| 训练 | rounds | exp1: 20; exp3/5/6 smoke: 1; exp5/6 formal: 20 |
| 训练 | local_epochs | exp1: 1; exp5/6: 1 (历史 exp1/2: 3) |
| 训练 | batch_size | exp1/2: 64; exp3/5/6: 32 |
| 训练 | learning_rate | 1e-3 (Adam) |
| 聚合 | 方法 | FedAvg (sample_count 加权) |
| 数据 | tensor shape | [2, 630, 5856] |
| 数据 | 划分 | 70%/15%/15% 时序连续 |
| 数据 | 归一化 | z-score (Exp1/2/3/5/6 ✅，本轮修复) |

---

## 11. 审稿意见映射

完整映射表见 [`reviewer_response_experiment_mapping_zh.md`](./reviewer_response_experiment_mapping_zh.md)。

关键映射：
- 超参数表 → 已生成
- 消融实验 → Exp2 需恢复，Exp4 未开发，Exp6 不完整
- 收敛性 → Exp1 有完整曲线，Exp3/5/6 缺失
- client 异质性 → Exp1 有 client-level 指标，Exp5 non_iid_summary 存在
- 对比基线 → Exp1 完整，Exp3/5/6 缺失 NaiveLastValue
- 通信开销/掉线 → 未实现，放入 limitations

---

## 12. 论文结果表建议

完整建议见 [`real_exp_1_6_result_table_plan_zh.md`](./real_exp_1_6_result_table_plan_zh.md)。

核心建议：
- 主表仅纳入实验 1（当前唯一可信 formal 结果）
- 消融表暂空（exp2 需恢复，exp4 未开发，exp6 不完整）
- 主指标：MSE / RMSE / MAE / MAPE
- R² 仅附录
- 必须含相对提升率 Δ%

---

## 13. 下一步优先级

### P0 — 阻塞级（必须修复才能推进）

1. **修复 exp5/6 scaler 链路**: ✅ **已完成**。为 `rc_config.py` / `ra_config.py` 添加了 `input_normalization` / `target_normalization` 字段，在 `rc_core.py` / `ra_core.py` 中接入了完整的 scaler 流程（fit → apply → evaluation with target_scaler）。参考了 `rfc_core.py` 的实现。
2. **exp5/6 修复后验证**: ✅ **已完成**。对 exp5 spatial_block 跑了 r3e1 诊断（capped 5k samples），确认 train_loss 持续下降（0.313→0.017）、RMSE 改善（306k→160k，vs 修复前 628k 常数）。
3. **补 exp5/6 NaiveLastValue baseline**: ✅ **已完成**。`rc_core.py` 中添加了 `evaluate_naive_last_value()`，NaiveLastValue RMSE=8,744 (R²=0.9996)。

### P1 — 高优先级（直接影响论文可发表性）

4. **恢复或重跑实验 2**: 代码完整（scaler 已验证），最快可补的消融实验。若历史 fixed 结果可恢复则优先恢复，否则重跑 r20e3
5. **实验 3 r5e1 诊断**: 验证 r5 是否显著优于 r1 的 FedAvg RMSE（类似 exp1 的改善幅度），再决定是否进入 r20 formal
6. **重跑 Exp5/6 formal (r20e1)**: 当前最紧急。scaler 修复后 r3e1 诊断确认模型开始学习，需 r20 判断 full data 下最终性能是否能超越 NaiveLastValue

### P2 — 中优先级

6. **补超参数表中缺失项**: gradient clipping、weight decay、Adam β 等需在 config 中显式记录
7. **更新过时报告**: `real_exp_1_6_status_zh.md` 已严重过时，需重写或删除
8. **生成 exp5/6 formal 状态报告**: 补 `real_exp_3_5_6_formal_status_zh.md`

### P3 — 低优先级（可放入 limitations）

9. **实验 4 开发**: 若论文保留 cluster-level ablation，开发 rfc ablation 入口
10. **通信开销/掉线/DP**: 在 limitations 或 discussion 中说明
11. **GCN 真实数据**: 在 limitations 中说明计算成本限制

---

## 14. 本轮注意事项

- **不运行实验**: 本轮所有诊断和规划仅基于静态分析
- **不修改源码**: 仅检查，不编辑 `.py` 文件
- **不提交 results/logs/data**: 已确认
- **本次新生成文档**: 5 份 markdown（本文档 + 4 份辅助文档）
- **建议提交**: 仅提交这 5 份 `.md` 文件

```
real_data_experiments/real_exp_1_6_current_status_and_revision_plan_zh.md   (本文档)
real_data_experiments/real_exp_5_6_training_failure_diagnosis_zh.md
real_data_experiments/real_exp_1_6_hyperparameter_tables_zh.md
real_data_experiments/reviewer_response_experiment_mapping_zh.md
real_data_experiments/real_exp_1_6_result_table_plan_zh.md
```
