# 缺失值实验 Phase 运行手册

> 最后更新: 2026-06-13 | 基于 Phase 0 审计

---

## Phase 0 审计结论

### 关键发现

`real_data_missingness_full_intersection_causal_history` 目录：
- ✅ `generate_missing` 阶段：已完成，61天全量，mask + missing_datasets 完整
- ❌ `impute` 阶段：**仅完成 `zero_fill`**（56/61 chunks），其余5种方法未跑
- ❌ `summarize` 阶段：从未执行
- 可用 `--resume --skip_existing` 补跑缺失的 impute 方法

其他实验目录（sample, medium, geo_func, hybridtest 等）：均为早期测试或子集，不纳入论文主表。

---

## Phase 1 操作手册

### Step 1: 补跑 impute（5种方法）

对已有 main 实验补跑 forward_fill, historical_linear_extrapolation, geo_neighbor_fill, function_curve_fit, geo_func_hybrid。

关键参数：
- `--resume True --skip_existing True`（跳过已完成的 zero_fill chunks）
- `--write_imputed_datasets True`
- 按方法分开跑（function_curve_fit 和 geo_func_hybrid 最耗时）

### Step 2: 执行 summarize

汇总所有方法的插补结果，生成 summary CSV + RMSE 图 + 中英文 Markdown 表。

---

## Phase 2 操作手册

### 整体策略

对每个缺失率 (0.03, 0.05, 0.10, 0.20, 0.30) × 每个 seed (42, 2024, 3407)：
1. `generate_missing` → 生成该 rate+seed 的 mask 和 missing_datasets
2. `impute` → 用6种方法分别插补
3. `summarize` → 汇总

建议策略：
- 先跑 seed=42 的 5 个 rate（最优先：0.05已有，只需补 0.03/0.10/0.20/0.30）
- 每条 rate 用独立的 output_dir（如 `...causal_history_rate0p10`）
- 最后跨目录聚合

---

## Phase 3 操作手册

`node_temporal_block` 机制参数：
- `--mechanism node_temporal_block`
- `--block_lengths 4,8,12`
- 建议先跑 rates 0.05, 0.10, 0.20（3个rate × 1个seed × 3个block_length）
- 独立 output_dir（如 `...causal_history_block`）

---

## Python 环境

- Python: `E:\anaconda3\envs\analysis\python.exe`
- 工作目录: `E:\Jupter_Notebook\FedTrafficFlow`
- 脚本: `analysis_scripts/full_intersection_missingness_pipeline.py`
