# 真实数据实验 1–6 当前状态与修订计划

> 生成日期：2026-06-30
> 最后更新：2026-07-01（HEAD 更新为 e91e2cd，补充 Exp1 federated mechanism diagnostic 完成状态与结果，删除过时声明）
> **数据划分更新**：本轮将真实数据实验的时序划分从 70%/15%/15% 修订为 80%/10%/10%，以在 61 天观测窗口内最大化训练数据并保留验证/测试集。Exp1 已完成的历史 formal 使用 70%/15%/15%，保留作为 sensitivity check。
> Exp1 当前为 Level 2 diagnostic。包含 CalendarFeatureFedAvg v2 residual-gated diagnostic、long-horizon seq96_h4/h12/h24 diagnostic、以及 federated mechanism seq96_h12 r5 diagnostic。

---

## 1. 当前 Git 状态

- **分支**: `feature/real-exp4-rfc-ablation`
- **HEAD**：`e91e2cd` — feat(real-data): add Exp1 federated mechanism diagnostic
- **最近 10 个 commit** (含本次报告相关):

| # | Hash | Message |
|---|------|---------|
| 1 | `e91e2cd` | feat(real-data): add Exp1 federated mechanism diagnostic |
| 2 | `d6453ef` | docs(real-data): correct stale diff in section 3.2 and finalize smoke report |
| 3 | `c6dcfba` | docs(real-data): finalize exp3 exp5 exp6 smoke verification |
| 4 | `5dfca68` | fix(real-data): remove redundant dtype cast in RegionClientWindowDataset; report verified smoke results |
| 5 | `b1ec541` | fix(real-data): remove tensor clone in RegionClientWindowDataset; add timing logs |
| 6 | `6d3955f` | docs(real-data): record exp3 exp5 exp6 smoke status (首次 smoke 均 TIMEOUT) |
| 7 | `5653445` | docs(real-data): record exp1 formal result status |
| 8 | `3ade178` | docs(real-data): map revised experiments 1-6 status |
| 9 | `b49eb7e` | fix(real-data): refine calendar baseline fallback and report wording |
| 10 | `4323613` | fix(real-data): align calendar baselines with test split |

- **工作区备注**: 当前仍可能存在未跟踪 results 目录；本轮文档同步不提交 results/logs/data。

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
| **5** | 全局划分主实验 | ✅ r1e1 (spatial_block + flow_kmeans) | ✅ r20e1 (spatial_block + flow_kmeans) — **scaler 已修复** | `main_metrics.csv` ×4 | ✅ 修复后 r3e1 诊断正常 (RMSE 从 628k 降至 160k, train_loss 持续下降) | Scaler 修复报告 ✅ | **Scaler 修复完成，可进入重跑 formal 候选** |
| **6** | 全局划分消融 | ✅ r1e1 (spatial_block) | ✅ r20e1 (full only) — scaler 已修复 | `ablation_metrics.csv` + `ablation_summary.csv` | ⚠️ 消融不完整(仅 full variant)，待补全 4 variant | Scaler 修复报告 ✅ | **Scaler 修复完成，待补全 4 variant 后进入重跑 formal 候选** |

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

> **注意**：Exp1 formal r20e1 使用 70%/15%/15% 时序划分。修订后的划分方案为 80%/10%/10%，该结果保留作为 sensitivity check 的参考基线。

### 实验 1 — Federated Mechanism Diagnostic (seq96_h12, r5)

| Method | RMSE |
|--------|------|
| CentralizedUpperBound | 44,819 |
| CalendarFeatureFedAvg-Full+LocalFT | 53,756 |
| FedAvg+LocalFT | 55,773 |
| Independent | 61,881 |
| FedAvg | 82,524 |

**机制结论**:
- **FedAvg+LocalFT > RandomInit+LocalFT** (55,773 < 61,881)：shared initialization 有效，FedAvg 全局模型为 local fine-tuning 提供了更好的起点
- **FedAvg+LocalFT > FedAvg** (55,773 < 82,524)：personalization 有效，local fine-tuning 显著改善了对本地数据的拟合
- **FedProx > FedAvg**：non-IID constraint 有效，FedProx 的 proximal term 约束了 client drift
- **CentralizedUpperBound** (44,819)：作为 oracle upper bound，说明在当前模型容量和数据量下可达到的最佳性能

**当前状态**: diagnostic stage，尚未 formal。

### 实验 3 — smoke r1e1

| Method | RMSE | MAE | MAPE | R² |
|--------|------|-----|------|-----|
| NaiveLastValue | **8,240** | 5,497 | 6.235 | **0.997** |
| Independent | 108,569 | 92,739 | 138.355 | 0.361 |
| FedAvg | 279,894 | 251,080 | 1384.975 | -1.755 |

**结论**: r1e1 smoke，FedAvg 远弱于 NaiveLastValue (8,240 vs 279,894)。但这是 1 round 训练不充分的表现，exp1 在 r1e1 时 FedAvg 也高达 107,579。需 r20 formal 判断是否改善。

### 实验 5 — formal r20e1（scaler 修复前，历史记录）

| Config | Method | RMSE | R² |
|--------|--------|------|-----|
| spatial_block | FedAvg | 627,741 | **-1.128** |
| spatial_block | Independent | 627,704 | **-1.129** |
| flow_kmeans | FedAvg | 569,025 | **-4.110** |
| flow_kmeans | Independent | 568,878 | **-4.108** |

> **注意**：以上为 scaler 缺失时的异常指标，已通过前序提交修复。修复后 r3e1 诊断确认模型开始学习（RMSE 从 628k 降至 160k）。需重跑 r20 formal 获取正常指标。

### 实验 6 — formal r20e1（scaler 修复前，历史记录）

| Variant | RMSE | R² |
|---------|------|-----|
| Full | 627,741 | **-1.128** |

仅含 `full` variant，消融不完整。scaler 已修复，需重跑。

---

## 5. 实验 5/6 Scaler 修复摘要

**根因**: `rc_core.py` 和 `ra_core.py` 的实验主流程中完全缺失 scaler/归一化链路。

**修复方案**: 参考 `sic_core.py` / `rfc_core.py` 的 scaler 链路，已为 `rc_core.py` / `ra_core.py` 补全归一化。

**修复后验证 (Exp5 spatial_block r3e1)**:
- FedAvg RMSE：627,741 → 159,527（改善 75%）
- train_loss：0.313 → 0.017（持续下降）
- 模型明确开始学习，不再停滞

完整诊断历史见：[`archive_legacy_docs/real_exp_5_6_training_failure_diagnosis_zh.md`](./archive_legacy_docs/real_exp_5_6_training_failure_diagnosis_zh.md)。

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

**数据划分修订说明**: 考虑到 61 天观测窗口较短，修订后的真实数据实验采用 80%/10%/10% 时序划分为训练/验证/测试集。70%/15%/15% 时序划分仅作为 sensitivity check 保留（见 Exp1 formal）。数据划分全程不进行随机打乱，以避免时序信息泄漏。

> Considering the relatively short 61-day observation window, the revised real-data experiments use a chronological 80%/10%/10% split for training, validation, and testing. A chronological 70%/15%/15% split is retained only as a sensitivity check. No random shuffling is applied during data splitting to avoid temporal information leakage.

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

## 8. 实验 4 状态更新

### 当前状态

- **独立入口**: ✅ 已新增 `real_data_experiments/region_client_full_cells/rfc_ablation_core.py`
- **配置文件**: ✅ 已新增 `real_data_experiments/region_client_full_cells/rfc_ablation_config.py`
- **client 构造**: ✅ 复用 Exp3 的 `similarity_k5.json` 和 `build_full_cells_client_data`
- **Scaler 链路**: ✅ 复用 Exp3 的 `fit_rfc_input_scaler` / `fit_rfc_target_scaler`
- **Variants**: ✅ `full`, `without_attention`, `without_cnn`, `without_lstm`
- **Smoke**: ✅ all variants r1e1 + 1k samples + CUDA 通过
- **Formal**: ❌ 尚未运行

### Smoke 指标（仅用于 pipeline 验证）

| Variant | RMSE |
|---|---:|
| full | 294,334 |
| without_attention | 311,891 |
| without_cnn | 130,883 |
| without_lstm | 201,132 |

**注意**: 以上结果只说明四个 variants 能跑通，不能作为正式消融结论。r1e1 下 `without_cnn` 表现最好属于单轮随机现象，不能据此认为 CNN 无效。

### 输出目录

```
results/real_data_experiments/diagnostic/exp4_rfc_ablation_similarity_k5_all_variants_r1_e1_cuda_1k/
```

（该目录不提交，仅用于 diagnostic 验证）

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
| 数据 | 划分 | 80%/10%/10% 时序连续 (Exp1 历史 formal 使用 70%/15%/15%，作为 sensitivity check) |
| 数据 | 归一化 | z-score (Exp1/2/3/5/6 ✅，本轮修复) |

---

## 11. 审稿意见映射

完整映射表见 [`reviewer_response_experiment_mapping_zh.md`](./reviewer_response_experiment_mapping_zh.md)。

关键映射：
- 超参数表 → 已生成
- 消融实验 → Exp2 需恢复，Exp4 代码已补全但 formal 待跑，Exp6 不完整
- 收敛性 → Exp1 有完整曲线，Exp3/5/6 缺失
- client 异质性 → Exp1 有 client-level 指标，Exp5 non_iid_summary 存在
- 对比基线 → Exp1 完整，Exp3/5/6 缺失 NaiveLastValue
- 通信开销/掉线 → 未实现，放入 limitations

---

## 12. 论文结果表建议

完整建议见 [`real_exp_1_6_result_table_plan_zh.md`](./real_exp_1_6_result_table_plan_zh.md)。

核心建议：
- 主表仅纳入实验 1（当前唯一可信 formal 结果）
- 消融表暂空（exp2 需恢复，exp4 代码已补全但 formal 未运行，exp6 不完整）
- 主指标：MSE / RMSE / MAE / MAPE
- R² 仅附录
- 必须含相对提升率 Δ%

---

## 13. 下一步优先级

### P0 — 阻塞级（必须修复才能推进）

1. **修复 exp5/6 scaler 链路**: ✅ **已完成**。为 `rc_config.py` / `ra_config.py` 添加了 `input_normalization` / `target_normalization` 字段，在 `rc_core.py` / `ra_core.py` 中接入了完整的 scaler 流程（fit → apply → evaluation with target_scaler）。参考了 `rfc_core.py` 的实现。
2. **exp5/6 修复后验证**: ✅ **已完成**。对 exp5 spatial_block 跑了 r3e1 诊断（capped 5k samples），确认 train_loss 持续下降（0.313→0.017）、RMSE 改善（306k→160k，vs 修复前 628k 常数）。
3. **补 exp5/6 NaiveLastValue baseline**: ✅ **已完成**。`rc_core.py` 中添加了 `evaluate_naive_last_value()`，NaiveLastValue RMSE=8,744 (R²=0.9996)。
4. **Exp1 federated mechanism diagnostic**: ✅ **已完成** (e91e2cd)。seq96_h12 r5 diagnostic 完整运行 6 种机制配置。

### P1 — 高优先级（直接影响论文可发表性）

5. **恢复或重跑实验 2**: 代码完整（scaler 已验证），最快可补的消融实验。若历史 fixed 结果可恢复则优先恢复，否则重跑 r20e3
6. **实验 3 r5e1 诊断**: 验证 r5 是否显著优于 r1 的 FedAvg RMSE（类似 exp1 的改善幅度），再决定是否进入 r20 formal
7. **重跑 Exp5/6 formal (r20e1)**: 当前最紧急。scaler 修复后 r3e1 诊断确认模型开始学习，需 r20 判断 full data 下最终性能是否能超越 NaiveLastValue

### P2 — 中优先级

8. **补超参数表中缺失项**: gradient clipping、weight decay、Adam β 等需在 config 中显式记录
9. **更新过时报告**: `real_exp_1_6_status_zh.md` 已严重过时，需重写或删除
10. **补充 FedProx 结果分析**: federated mechanism diagnostic 已有 FedProx 结果，可据此补充 non-IID 分析
11. **生成 exp5/6 formal 状态报告**: 补 `real_exp_3_5_6_formal_status_zh.md`

### P3 — 低优先级（可放入 limitations）

12. **实验 4 formal / r5 diagnostic**: 代码已补全并通过 1k smoke。下一步先运行 r5 diagnostic 验证多轮收敛，不建议直接 r20 formal。
13. **通信开销/掉线/DP**: 在 limitations 或 discussion 中说明
14. **GCN 真实数据**: 在 limitations 中说明计算成本限制

---

## 14. 本轮注意事项

- **实验运行状态**: Exp1 federated mechanism diagnostic 已在本轮运行完成。其他实验未运行新实验。
- **不修改源码**: 仅检查，不编辑 `.py` 文件
- **不提交 results/logs/data**: 已确认
- **当前有效文档**: 见 [`REAL_DATA_EXPERIMENTS_CURRENT_DOCS_zh.md`](./REAL_DATA_EXPERIMENTS_CURRENT_DOCS_zh.md)
