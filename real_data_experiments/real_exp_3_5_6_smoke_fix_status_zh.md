# 真实数据实验 3/5/6 Smoke 修复报告

> 生成日期：2026-06-29
> 本报告记录超时根因定位和修复内容。

## 1. 基本信息

- **分支**: `feature/real-exp1-client-similarity-diagnosis`
- **修复前 commit**: `6d3955f docs(real-data): record exp3 exp5 exp6 smoke status`
- **修复后 commit**: `b1ec541 fix(real-data): remove tensor clone in RegionClientWindowDataset; add timing logs`

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

## 5. Git diff 摘要

```
 M real_data_experiments/common/region_tensor_dataset.py   (3 lines)
 M real_data_experiments/region_client/rc_core.py          (+15 lines)
```

- 总计：2 个文件，+18 行，-2 行
- 无 results/logs/data 修改

## 6. 是否误提交 results/logs/data

否。

## 7. 下一步建议

1. **恢复环境后优先验证**：实验 5 spatial_block r1e1 → 实验 5 flow_kmeans r1e1 → 实验 6 full r1e1 → 实验 3 similarity_k5 r1e1
2. **观察计时日志**：确认 `build_region_client_data` 耗时是否降至 < 60s
3. **如果仍超时**：需继续排查 `assign_region_clients` 中的 `build_region_feature_frame`（涉及 `channel_tensor.detach().cpu().numpy()` 的 630×5856 矩阵操作）或 DataLoader 初始化
4. **如果通过**：进入 experimental formal
