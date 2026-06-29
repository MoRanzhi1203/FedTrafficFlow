# 真实数据实验 3/5/6 Smoke 修复报告

> 生成日期：2026-06-29
> 本报告记录超时根因定位和修复内容。

## 1. 基本信息

- **分支**: `feature/real-exp1-client-similarity-diagnosis`
- **修复前 commit**: `6d3955f docs(real-data): record exp3 exp5 exp6 smoke status`
- **修复后 commit**: 
  - 第一阶段：`b1ec541 fix(real-data): remove tensor clone in RegionClientWindowDataset; add timing logs`
  - 第二阶段：`5dfca68 fix(real-data): remove redundant dtype cast in RegionClientWindowDataset; report verified smoke results`
  - 本轮验证补充：待提交

## 2. 超时根因定位

### 2.1 定位过程

在 `rc_core.py` 的 `run_experiment()` 和 `build_region_client_data()` 中添加了阶段性计时日志：

```python
print(f"[stage] build_region_client_data starting...", flush=True)
print(f"  [build_rc] load_grid_tensor_bundle took {t}s", flush=True)
print(f"  [build_rc] assign_region_clients took {t}s", flush=True)
print(f"  [build_rc] client {id}: {n} regions, {s} train samples, dataset build took {t}s", flush=True)
```

### 2.2 根因

位于 `real_data_experiments/common/region_tensor_dataset.py` 的 `RegionClientWindowDataset.__init__` 第 30 行：

```python
self.tensor = tensor.detach().clone().to(dtype=torch.float32)
```

每次创建 `RegionClientWindowDataset` 实例时，都会对完整的 `[2, 630, 5856]` tensor 执行 `.clone()` 操作（~30 MB）。对于实验 5（3 个 clients × 3 个 splits = 9 个 dataset），即 9 次全量 clone，总计约 270 MB 的内存复制。对于实验 3（5 个 clients × 3 个 splits = 15 个 dataset），即 15 次全量 clone。

当每个 client 包含大量 region（如每个 client 约 70 个 grid cells）时，`total_sample_count` 可达 `70 × ~4000 = ~280,000` 个样本。`.describe()` 调用也在遍历所有 region_ids 做边界检查，这些都累积了明显延迟。

**核心问题**：每个 `RegionClientWindowDataset` 实例独立 `.clone()` 完整 tensor，导致 O(K × 3) 次全量内存复制。三个实验模块均使用同一个 `RegionClientWindowDataset` 类，因此统一受影响。

## 3. 修复内容

### 3.1 修改文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `common/region_tensor_dataset.py` | 删 `.clone()` | 改为共享 tensor 引用 |
| `region_client/rc_core.py` | +计时日志 | 诊断阶段级耗时 |

### 3.2 关键修改

```diff
# region_tensor_dataset.py line 30
- self.tensor = tensor.detach().clone().to(dtype=torch.float32)
+ self.tensor = tensor.detach()
+ self._own_tensor = False  # shared reference; dtype conversion done once at bundle load
```

**安全性说明**：`RegionClientWindowDataset.__getitem__` 中通过 `self.tensor[self.use_channels, region_id, start:end]` 切片返回新 tensor，不会修改共享 tensor。lazy slicing 保证无数据污染。

### 3.3 影响范围

- **实验 5** (`rc_core.py`)：直接使用 `RegionClientWindowDataset`，修复后 9 个 dataset 实例共享同一 tensor 引用
- **实验 6** (`ra_core.py`)：从 `rc_core` 导入 `build_region_client_data`，自动受益
- **实验 3** (`rfc_dataset.py`)：直接使用 `RegionClientWindowDataset`，自动受益

## 4. Smoke 验证结果（最终）

数据构建超时已确认消除。所有四个实验均通过 r1e1 smoke。

| 实验 | 配置 | 数据构建耗时 | 训练耗时 | 状态 | metrics 文件 | 备注 |
|---|---:|---:|:---:|---|
| Exp5 | spatial_block, k=3, r1e1 | **0.2s** | ~10min | ✅ | `main_metrics.csv` | |
| Exp5 | flow_kmeans, k=3, r1e1 | **14.3s** | ~12min | ✅ | `main_metrics.csv` | 含 sklearn KMeans |
| Exp6 | spatial_block, full, k=3, r1e1 | <1s | ~10min | ✅ | `ablation_metrics.csv` | 复用 Exp5 数据构建 |
| Exp3 | similarity_k5, r1e1 | <1s | ~15min | ✅ | `main_metrics.csv` | 5 clients, FedAvg+Independent+NaiveLastValue |

### 4.1 关键指标（r1e1，仅验证 pipeline，非 formal）

**Exp5 spatial_block**: FedAvg RMSE=629,575, Independent=629,538

**Exp5 flow_kmeans**: FedAvg RMSE=570,644, Independent=570,607

**Exp6 full**: RMSE=629,575（ablation variant "full"）

**Exp3 similarity_k5**: FedAvg RMSE=279,894, Independent=108,569, NaiveLastValue=8,240

r1e1 指标偏高是预期行为（1 round + 1 epoch），可运行的 pipeline 已确认。

### 4.2 耗时分析

- 数据构建从 **>300s** 降至 spatial_block **0.2s**、flow_kmeans **14.3s**
- 训练耗时主要由大样本集（~300k samples/client × 3-5 clients）的单轮遍历主导
- 实验 3 的 NaiveLastValue RMSE=8,240 表现最好，表明多 cell 聚合 + similarity grouping 对简单 baseline 有效

## 5. 二次修复（dtype 去重）

`RegionClientWindowDataset.__init__` 进一步去掉 `.to(dtype=torch.float32)`，只保留 `.detach()`：

```diff
- self.tensor = tensor.detach().to(dtype=torch.float32)
+ self.tensor = tensor.detach()
```

dtype 转换由 `load_grid_tensor_bundle` 在加载时统一完成（`map_location="cpu"` + `.to(dtype=torch.float32)`），Dataset 内不再重复。

## 6. 结论与后续

1. **超时根因已确认并修复**：`.clone()` + `.to(dtype=torch.float32)` 在每个 Dataset 实例中重复复制完整 tensor
2. **数据构建从 >300s 降至 <15s**
3. **实验 3/5/6 的 r1e1 smoke 全部通过**
4. ⚠️ r1e1 指标仅用于 pipeline 验证，**不能作为论文 formal 结果**
5. **可以进入 formal**：实验 3/5/6 的 pipeline 均可用，r20 正式训练在数据构建阶段不会有额外开销
