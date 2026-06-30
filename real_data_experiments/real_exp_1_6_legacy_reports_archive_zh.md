# 真实数据实验 1–6 历史报告归档

> 归档日期：2026-06-30
> 这些内容已不作为当前状态依据，仅用于追踪历史过程。
> 当前权威文档：[real_exp_1_6_current_status_and_revision_plan_zh.md](real_exp_1_6_current_status_and_revision_plan_zh.md)

---

## 1. 归档说明

本文档合并了以下已被新版总控文档覆盖或已过时的历史报告：

| 原文件 | 归档原因 |
|--------|---------|
| `real_exp_1_6_status_zh.md` | 已被 `real_exp_1_6_current_status_and_revision_plan_zh.md` 完整覆盖 |
| `real_exp_3_5_6_smoke_status_zh.md` | Smoke 超时状态已修复，记录保留在此 |
| `real_exp_3_5_6_smoke_fix_status_zh.md` | 修复过程已完成，核心信息保留在此 |
| `exp1_formal_result_status_zh.md` | 实验 1 r20e1 结果，当前正式指标引用自此 |
| `formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md` | 历史 CUDA 修复重跑记录 |
| `formal_cuda_exp1_exp2_run_report_zh.md` | 更早版本的 formal CUDA 运行记录 |
| `formal_experiments_cleanup_report_zh.md` | 较早阶段的目录清理历史 |
| `current_real_data_experiments_inventory_zh.md` | 被新版总控文档替代的实验清单 |

---

## 2. 历史实验 1–6 状态映射（来自 `real_exp_1_6_status_zh.md`）

> 生成日期：2026-06-29，当前已被新版总控文档覆盖。

### 早期实验编号定义

| 新编号 | 说明 | 旧对应 |
|--------|------|--------|
| 实验 1 | 单个 grid cell 作为 client | `single_intersection_client/` |
| 实验 2 | 实验 1 的扩展/消融 | 尚无独立目录 |
| 实验 3 | 多个相似 grid cells 组成一个 client | `region_client_full_cells/` |
| 实验 4 | 实验 3 的扩展/消融 | 与实验 3 共用代码 |
| 实验 5 | 全部 grid cells 按相似度划分多 clients | `region_client/` |
| 实验 6 | 实验 5 的扩展/消融 | `region_ablation/` |

### 早期各实验状态（已过时）
- 实验 1：Formal 结果 "尚未运行"（现已完成 r20e1）
- 实验 2：缺少独立消融模块（现已开发 `sia_core.py`）
- 实验 3/5/6：均标注 Smoke ❌（后已全部通过）

---

## 3. 历史 Smoke 超时记录（来自 `real_exp_3_5_6_smoke_status_zh.md`）

> 生成日期：2026-06-29，Commit: `5653445`

### 首次 Smoke（r1e1）结果：全部 TIMEOUT

| 实验 | 配置 | 结果 |
|------|------|:---:|
| Exp3 | similarity_k5, r1e1 | ❌ TIMEOUT (>300s) |
| Exp5 | flow_kmeans, k=3, r1e1 | ❌ TIMEOUT |
| Exp5 | spatial_block, k=3, r1e1 | ❌ TIMEOUT |
| Exp6 | spatial_block, full, k=3, r1e1 | ❌ TIMEOUT |

超时根因初步判断：数据加载/分区阶段耗时过长（三个模块都涉及全局 grid cell 分区逻辑）。

---

## 4. Smoke 修复记录（来自 `real_exp_3_5_6_smoke_fix_status_zh.md`）

> 生成日期：2026-06-29，Fix commits: `b1ec541`, `5dfca68`

### 根因

位于 `common/region_tensor_dataset.py` 的 `RegionClientWindowDataset.__init__`：
```python
# 原代码
self.tensor = tensor.detach().clone().to(dtype=torch.float32)
```
每次创建 Dataset 实例时对完整 `[2, 630, 5856]` tensor 执行 `.clone()`（~30 MB），导致 O(K × 3) 次全量内存复制。

### 修复

```python
# 修复后
self.tensor = tensor.detach()  # 共享引用
```
另去掉冗余 `.to(dtype=torch.float32)`，dtype 转换由 `load_grid_tensor_bundle` 统一完成。

### 修复后 Smoke 结果

| 实验 | 配置 | 数据构建 | 训练 | 状态 |
|------|------|:---:|:---:|:---:|
| Exp5 | spatial_block, k=3 | 0.2s | ~10min | ✅ |
| Exp5 | flow_kmeans, k=3 | 14.3s | ~12min | ✅ |
| Exp6 | spatial_block, full | <1s | ~10min | ✅ |
| Exp3 | similarity_k5 | <1s | ~15min | ✅ |

---

## 5. 实验 1 首次 Formal 结果（来自 `exp1_formal_result_status_zh.md`）

> 生成日期：2026-06-29，Commit: `3ade178`

### 运行参数
- rounds=20, local_epochs=1, device=cuda, model_variant=baseline, sequence_length=12
- selected_clients: [290, 284, 318, 288, 289]

### 核心指标

| 方法 | RMSE | MAE | R² |
|------|:---:|:---:|:---:|
| NaiveLastValue | 19,419 | 13,620 | 0.939 |
| Independent | 15,921 | 11,615 | 0.953 |
| CalendarProfileNaive | 32,194 | 22,770 | 0.830 |
| FedAvg | 24,158 | 19,773 | 0.847 |
| DailySeasonalNaive | 45,406 | 29,727 | 0.637 |
| WeeklySeasonalNaive | 48,369 | 32,881 | 0.551 |

### 关键发现
- FedAvg 从 r5e1 的 37,344 降至 24,158（-35%），rounds=20 显著改善联邦训练
- Independent (15,921) 优于 NaiveLastValue (19,419)
- FedAvg 排名第 4/6，仅优于 Daily/WeeklySeasonalNaive

---

## 6. 历史 Formal CUDA 修复重跑（来自 `formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md`）

> Commit: `d2b87f4`，参数: rounds=20, local_epochs=3, device=cuda

### 实验 1 指标

| Method | RMSE | MAE | R2 |
|--------|------|-----|-----|
| FedAvg | 20,753.14 | 16,847.81 | 0.8827 |
| Independent | 14,883.58 | 10,843.73 | 0.9621 |
| NaiveLastValue | 19,419.22 | 13,619.88 | 0.9386 |

### 实验 2 消融指标

| Variant | RMSE | R2 |
|---------|------|-----|
| Full | 17,345.62 | 0.9117 |
| Without Attention | 18,975.80 | 0.9014 |
| Without CNN | 23,371.29 | 0.8577 |
| Without LSTM | 19,331.81 | 0.8972 |

关键：修复后实验 2 normalization 生效，y_pred 从 ~66 恢复到 ~1,925,561（正常尺度）。

---

## 7. 更早的 Formal CUDA 运行（来自 `formal_cuda_exp1_exp2_run_report_zh.md`）

> Commit: `9323369`，环境：torch 2.8.0+cu126，RTX 3060 Laptop GPU

实验 1 和实验 2 均以 CUDA 完成（actual_device=cuda），输出目录：
- `results/real_data_experiments/formal/exp1_single_grid_client_formal_cuda`
- `results/real_data_experiments/formal/exp2_single_grid_ablation_formal_cuda`

此版本中实验 1 缺少 NaiveLastValue baseline，实验 2 Full 的预测存在尺度崩坏（constant ~66），后在 `d2b87f4` 修复。

---

## 8. 早期目录清理历史（来自 `formal_experiments_cleanup_report_zh.md`）

仅覆盖旧实验 1-4 的清理语境。在新编号体系下应以 `current_real_data_experiments_inventory_zh.md`（也已过时）和当前总控文档为准。

---

## 9. 早期实验清单（来自 `current_real_data_experiments_inventory_zh.md`）

按实验 1-6 新编号梳理了目录、实验含义和结果路径。已被 `real_exp_1_6_current_status_and_revision_plan_zh.md` 完整覆盖。

---

## 10. 文件来源索引

| 原文件路径 | 归档章节 |
|-----------|---------|
| `real_data_experiments/real_exp_1_6_status_zh.md` | §2 |
| `real_data_experiments/real_exp_3_5_6_smoke_status_zh.md` | §3 |
| `real_data_experiments/real_exp_3_5_6_smoke_fix_status_zh.md` | §4 |
| `real_data_experiments/exp1_formal_result_status_zh.md` | §5 |
| `real_data_experiments/formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md` | §6 |
| `real_data_experiments/formal_cuda_exp1_exp2_run_report_zh.md` | §7 |
| `real_data_experiments/formal_experiments_cleanup_report_zh.md` | §8 |
| `real_data_experiments/current_real_data_experiments_inventory_zh.md` | §9 |
