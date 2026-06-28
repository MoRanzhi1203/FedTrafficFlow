# 真实数据实验 1–6 当前覆盖情况报告

> 生成日期：2026-06-29
> 不包含仿真实验 1–6（仿真实验已在上一个 feature 分支完成 resume 机制审计和修复）

---

## 1. 实验编号定义

| 新编号 | 说明 | 旧对应 |
|--------|------|--------|
| 实验 1 | 单个 grid cell 作为 client，FedAvg/Independent/NaiveLastValue 对比 | `single_intersection_client/` |
| 实验 2 | 实验 1 的扩展/消融（模型变体、训练策略、日历特征等） | 尚无独立目录 |
| 实验 3 | 多个相似 grid cells 组成一个 client，基础联邦实验 | `region_client_full_cells/` |
| 实验 4 | 实验 3 的扩展/消融（分区方式、聚合方式等） | 与实验 3 共用代码，消融未独立 |
| 实验 5 | 全部 grid cells 按相似度划分为多个 clients，基础联邦实验 | `region_client/` |
| 实验 6 | 实验 5 的扩展/消融（模型结构消融） | `region_ablation/` |

---

## 2. 实验目录与入口脚本清单

| 目录 | 核心入口 | CLI 是否可用 |
|------|---------|:---:|
| `single_intersection_client/` | `sic_core.py` + `sic_config.py` | ✅ |
| `region_client_full_cells/` | `rfc_core.py` + `rfc_config.py` | ✅ |
| `region_client/` | `rc_core.py` + `rc_config.py` | ✅ |
| `region_ablation/` | `ra_core.py` + `ra_config.py` | ✅ |

---

## 3. 各实验当前状态

### 实验 1：单个 grid cell client

- **入口**: `single_intersection_client/sic_core.py`
- **CLI**: `python -m real_data_experiments.single_intersection_client.sic_core --workflow train --rounds 20 ...`
- **Client 构造**: 每个 client = 一个 grid cell（5 个 selected clients: 290, 284, 318, 288, 289）
- **已支持方法**: FedAvg / Independent / NaiveLastValue / DailySeasonalNaive / WeeklySeasonalNaive / CalendarProfileNaive
- **诊断报告**: `exp1_calendar_periodicity_diagnosis_zh.md`、`exp1_client_similarity_diagnosis_zh.md`、`exp1_legacy_ipynb_model_diagnosis_zh.md` 等
- **Formal 结果**: ❌ 尚未运行（`results/real_data_experiments/formal/` 不存在）
- **Smoke r5e1**: ✅ 已完成（日历周期性基线已对齐 test split）

### 实验 2：单 grid cell 扩展/消融

- **入口**: 无独立目录，复用 `single_intersection_client/sic_core.py`
- **状态**: ⚠️ 缺少独立消融模块，无 `--variants` 参数进行模型结构/训练策略对比
- **需要**: 新增 `exp2_single_grid_ablation/` 或为 `sic_core.py` 增加 `--variant` 参数
- **Formal 结果**: ❌

### 实验 3：多个相似 grid cells 组成一个 client

- **入口**: `region_client_full_cells/rfc_core.py`
- **CLI**: `python -m real_data_experiments.region_client_full_cells.rfc_core --tensor-path ... --partition-file ... --output-dir ...`
- **Client 构造**: 按 spatial 或 similarity 将多个 grid cells 合并为一个 client
- **预生成分区**: `partitions/spatial_k{5,8,10}.json`、`partitions/similarity_k{5,8,10}.json`
- **已支持方法**: FedAvg / Independent / NaiveLastValue
- **相似度依据**: traffic pattern correlation（具体需查看 `rfc_partition.py`）
- **Formal 结果**: ❌
- **Smoke**: ❌

### 实验 4：多 grid cell client 消融

- **入口**: 与实验 3 共用 `region_client_full_cells/rfc_core.py`
- **状态**: ⚠️ 消融实验未独立，缺少模型变体、聚合方式、分区方式对比
- **需要**: 新增 `--variant` 或 `--ablation` 参数
- **Formal 结果**: ❌

### 实验 5：全部 grid cells 按相似度划分为多个 clients

- **入口**: `region_client/rc_core.py`
- **CLI**: `python -m real_data_experiments.region_client.rc_core --workflow train --partition-method spatial_block --num-clients 3 ...`
- **Client 构造**: 全局分区（spatial_block 或 flow_kmeans），所有 active grid cells 非重叠分配到 K 个 clients
- **相似度依据**: spatial_block（空间蛇形分块）或 flow_kmeans（KMeans on 空间+流量特征）
- **已支持方法**: FedAvg（可扩展）
- **Formal 结果**: ❌
- **Smoke**: ❌

### 实验 6：全 grid cell partition 的模型消融

- **入口**: `region_ablation/ra_core.py`
- **CLI**: `python -m real_data_experiments.region_ablation.ra_core --workflow train ... --variants full,without_attention,without_cnn,without_lstm`
- **Client 构造**: 与实验 5 相同（spatial_block 或 flow_kmeans）
- **已支持变体**: Full / Without Attention / Without CNN / Without LSTM
- **Formal 结果**: ❌
- **Smoke**: ❌

---

## 4. 覆盖情况汇总

| 实验 | CLI | Client 构造 | 诊断报告 | Smoke | Formal |
|------|:---:|:---:|:---:|:---:|:---:|
| 实验 1 | ✅ | ✅ single grid | ✅ 多份 | ✅ r5e1 | ❌ |
| 实验 2 | ❌ | ✅ 复用 Exp1 | ❌ | ❌ | ❌ |
| 实验 3 | ✅ | ✅ grouped cells | ❌ | ❌ | ❌ |
| 实验 4 | ❌ | ✅ 复用 Exp3 | ❌ | ❌ | ❌ |
| 实验 5 | ✅ | ✅ global partition | ❌ | ❌ | ❌ |
| 实验 6 | ✅ | ✅ 复用 Exp5 | ❌ | ❌ | ❌ |

---

## 5. 缺口清单

### 高优先级（阻止实验运行）

1. **实验 2 缺少独立入口和 CLI 变体参数** — 需要 `--variants` 或独立的 `exp2_single_grid_ablation/` 目录
2. **实验 4 缺少独立消融入口** — 需要 `--ablation` 参数或独立配置
3. **实验 1 缺少 formal 运行** — 只需执行 `sic_core.py` 即可
4. **实验 3/5/6 缺少 smoke 验证** — 需要至少跑一次 r1e1 确认 pipeline 可运行

### 中优先级（影响实验完整性）

5. **CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 尚未实现** — 实验 1 已生成日历特征但未接入模型
6. **实验 5/6 缺少 NaiveLastValue / Independent baseline** — 目前只有 FedAvg
7. **实验 3/4 中 grid cell 聚合方式未文档化** — 需要确认是 sum/mean/multi-channel/flatten
8. **相似度计算的 train split 隔离未审计** — 需要确认实验 3/5 的分区只用 train split

### 低优先级（完善类）

9. **断点续跑机制** — 实验 1–6 的 resume 机制需与仿真实验保持一致
10. **结果汇总脚本** — 各实验缺少统一的结果汇总和对比脚本

---

## 6. 旧路径 → 新实验编号映射表

| 旧路径 | 新实验 | 说明 |
|--------|:---:|------|
| `single_intersection_client/` | 1+2 | 单 grid cell client 主实验 + 扩展 |
| `region_client_full_cells/` | 3+4 | 多 grid cells 合并 client + 消融 |
| `region_client/` | 5 | 全局 partition 主实验 |
| `region_ablation/` | 6 | 模型结构消融 |
| `common/region_partition.py` | 5+6 | 共享的 partition 生成逻辑 |
| `common/calendar_baselines.py` | 1+2 | 日历周期性 baseline（当前仅用于 Exp1 诊断） |

---

## 7. 下一步最小修改方案

1. **优先运行实验 1 formal** — `sic_core.py --rounds 20 --device cuda`，产出正式 baseline 结果
2. **为实验 2 新增 `sic_config.py` 的 `--variant` 参数** — 支持 `baseline`/`baseline+calendar`/`baseline+residual`
3. **为实验 3 运行一次 r1e1 smoke** — 验证 `rfc_core.py` pipeline 完整可执行
4. **为实验 5/6 各运行一次 r1e1 smoke** — 验证 `rc_core.py` / `ra_core.py` pipeline
5. **暂不扩大实验 1 的日历 baseline** — 当前 CalendarProfileNaive 已对齐，CalendarFeature-FedAvg 作为后续扩展
6. **不重建 client 构造逻辑** — 现有 `region_partition.py` 和 `rfc_partition.py` 结构完整，优先验证而非重写

---

## 8. 附加说明

- **exp1_calendar_periodicity_diagnosis_zh.md** 属于实验 1 的日历周期性诊断补充，不代表实验 2–6 已完成。
- **当前 `feature/real-exp1-client-similarity-diagnosis` 分支** 的 commit 历史仅涉及实验 1 的客户端相似度诊断和日历周期性诊断。
- **所有实验均使用真实数据 tensor**：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- **不涉及仿真数据**：仿真实验 1–6 已在 `feature/simulation-resume-exp1-exp6` 分支完成并合并 main。
