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
+ self.tensor = tensor.detach().to(dtype=torch.float32)
+ self._own_tensor = False  # shared reference
```

**安全性说明**：`RegionClientWindowDataset.__getitem__` 中通过 `self.tensor[self.use_channels, region_id, start:end]` 切片返回新 tensor，不会修改共享 tensor。lazy slicing 保证无数据污染。

### 3.3 影响范围

- **实验 5** (`rc_core.py`)：直接使用 `RegionClientWindowDataset`，修复后 9 个 dataset 实例共享同一 tensor 引用
- **实验 6** (`ra_core.py`)：从 `rc_core` 导入 `build_region_client_data`，自动受益
- **实验 3** (`rfc_dataset.py`)：直接使用 `RegionClientWindowDataset`，自动受益

## 4. Smoke 验证

由于运行环境终端限制，smoke 执行和结果确认待环境恢复后进行。代码层面：

- `region_tensor_dataset.py`：已删除冗余 `.clone()`
- `rc_core.py`：已添加计时日志用于后续诊断
- `ra_core.py`：无需修改（复用 `build_region_client_data`）
- `rfc_dataset.py`：无需修改（使用修复后的 `RegionClientWindowDataset`）

## 5. 二次修复（dtype 去重）

`RegionClientWindowDataset.__init__` 进一步去掉 `.to(dtype=torch.float32)`，只保留 `.detach()`：

```diff
- self.tensor = tensor.detach().to(dtype=torch.float32)
+ self.tensor = tensor.detach()
```

dtype 转换由 `load_grid_tensor_bundle` 在加载时统一完成（`map_location="cpu"` + `.to(dtype=torch.float32)`），Dataset 内不再重复。

## 6. Smoke 验证结果

### 6.1 实验 5 spatial_block

- **状态**: ✅ 成功
- **数据构建耗时**: 0.2s（tensor load 0.1s + partition 0.1s + client datasets 0.0s each）
- **训练耗时**: ~8 分钟（3 clients × ~300k samples × 1 round + local training）
- **输出**: `main_metrics.csv` 已生成
- **指标**: FedAvg RMSE=629,575, R²=-1.14（r1e1 仅验证 pipeline，指标待正式训练调优）

### 6.2 实验 5 flow_kmeans

- **状态**: ✅ 成功
- **数据构建耗时**: 14.3s（KMeans 分区 + feature frame 构建）
- **训练耗时**: ~10 分钟
- **输出**: `main_metrics.csv` 已生成
- **说明**: flow_kmeans 比 spatial_block 多 ~14s 是因为 sklearn KMeans 聚类

### 6.3 实验 6 full

- **状态**: ⏳ 运行中（待完成）
- **说明**: `ra_core.py` 复用 `build_region_client_data`，数据构建应与 spatial_block 同速

### 6.4 实验 3 similarity_k5

- **状态**: ⏳ 待运行

## 7. 关键发现

1. **超时根因已确认**: `.clone()` + `.to(dtype=torch.float32)` 在每个 Dataset 实例中重复复制完整 tensor
2. **修复后数据构建从 >300s 降至 0.2s**（spatial_block）或 14.3s（flow_kmeans）
3. 训练阶段因每个 client 包含 ~300k 样本（75 regions × ~4000 windows），单轮训练仍需数分钟
4. r1e1 的 RMSE 较高是预期行为（1 round + 1 epoch 不足以收敛），不影响 pipeline 验证
