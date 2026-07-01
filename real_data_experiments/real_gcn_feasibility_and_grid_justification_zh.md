# GCN 真实数据可行性分析及 Grid 合理性论证

> 生成日期：2026-07-01

## 1. 真实 Road Graph 构建成本

### 1.1 拓扑数据需求

| 要素 | 需求 | 当前状态 |
|---|---|---|
| 路段节点坐标 | Q-Traffic link/segment 坐标表 | 未提取到实验侧 |
| 路段邻接关系 | 路口-路段拓扑表 | 未构建 |
| Grid-to-node 映射 | 每个 grid cell 内的道路节点索引 | grid regions.csv 有 source_node_count 但无具体 node id |

### 1.2 计算成本估计

- Pooled grid tensor: [2, 630, 5856]，223 active cells
- 若构建 223-node 图：邻接矩阵 ~223×223，但需要从原始 topology 推导
- Grid adjacency 简化方案：基于 pooled_row/pooled_col 构建 8-neighbor graph（O(1) 成本）

## 2. Grid Tensor 的优势论证

### 2.1 统一空间模板
- 630 个 pooled grid cells 覆盖统一空间范围
- 避免原始 topology 中缺失/不完整节点的对齐问题
- 所有 active regions 共享相同的时间轴

### 2.2 降低拓扑缺失影响
- Q-Traffic 原始数据中并非所有路段都有完整流量记录
- Grid pooling 通过聚合多节点流量缓解了单节点缺失
- CNN 的空间卷积等价于在整齐网格上学习局部空间模式

### 2.3 Synthetic GCN 已验证 Graph Pathway
- 仿真实验中 GCN 已在 `simulation_experiments/gcn_fed_base/` 和 `gcn_fed_enhanced_experiments/` 中验证
- 说明 GCN pathway 可行，真实数据上是工程适配问题

## 3. Real GCN 轻量 Diagnostic 方案

### 3.1 最小可行方案

```text
- Grid adjacency: 基于 pooled_row/pooled_col 的 8-neighbor 图
- Node features: 每个 cell 的 total_flow + mean_flow 时间序列
- 模型: GCN(2 layers) + LSTM + Attention (替换 CNN 为 GCN)
- 配置: Exp3 similarity_k5, r5e1, 1k capped samples
```

### 3.2 实现估算

改动量：新增 `real_gcn_diagnostic/` 目录，约 200 行代码，复用 Exp3 的 client partition 和数据管线。

## 4. 结论

- 当前 real-data grid CNN 作为主线是合理的：统一空间模板、简化拓扑依赖、已有完整实验结果
- 轻量 GCN diagnostic 可行（grid adjacency），但需独立开发
- 建议先完成 Exp1-6 主修复，GCN 放入 P2
