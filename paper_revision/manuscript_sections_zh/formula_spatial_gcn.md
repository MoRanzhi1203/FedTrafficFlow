# Formula: GCN Spatial Module

## 图卷积网络（GCN）空间特征提取

道路网络表示为无向图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$，通过图卷积提取空间依赖性。

### 图卷积层定义

$$
H_t
=
\sigma\left(
\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}
X_t
W
\right)
$$

其中各个符号的含义：

- $\tilde{A} = A + I$ 表示加自环后的邻接矩阵；
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ 表示度矩阵（对角矩阵）；
- $\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$ 表示对称归一化拉普拉斯算子；
- $X_t \in \mathbb{R}^{N \times F}$ 表示时刻 $t$ 所有节点的特征矩阵；
- $W$ 表示可学习权重矩阵；
- $\sigma(\cdot)$ 表示非线性激活函数（ReLU/ELU）；
- $H_t$ 表示 GCN 层在时刻 $t$ 的输出。

### 多层 GCN

$$
H_t^{(l+1)}
=
\sigma\left(
\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}
H_t^{(l)}
W^{(l)}
\right)
$$

其中 $H_t^{(0)} = X_t$，$l = 0, 1, \dots, L-1$。

## 代码实现

GCN 层实现在 GCN 仿真实验 core 文件中：

- `gcn_fed_base/gfb_core.py` 包含基础 GCN 模型类；
- `gcn_fed_enhanced_experiments/gfe_core.py` 包含增强 GCN 模型（动态图）。

## 与 CNN 空间模块对比

| 维度 | CNN | GCN |
|------|-----|-----|
| 数据结构 | 规则网格/局部邻域矩阵 | 图结构/邻接矩阵 |
| 空间建模 | 局部邻域卷积 | 图卷积（谱域/空域）|
| 拓扑编码 | 隐式（通过邻域窗口）| 显式（通过邻接矩阵）|
| 适用场景 | 网格化路网 | 非结构化路网 |
| 计算复杂度 | 较低 | 较高（需要图运算）|

## 注意力邻域聚合（GCN 变体）

对每个节点邻居进行加权聚合：

$$
z_{i,t}
=
\sum_{v_j \in \mathcal{N}(i)}
\alpha_{i,j,t}
h_{j,t}
$$

注意力权重：

$$
\alpha_{i,j,t}
=
\frac{\exp((q_i^\top k_j)/\sqrt{d})}
{\sum_{v_{j'} \in \mathcal{N}(i)} \exp((q_i^\top k_{j'})/\sqrt{d})}
$$

## 公式引用标记

- 图卷积层：Eq.(gcn-layer)
- 多层 GCN：Eq.(gcn-multi-layer)
- 注意力邻域聚合：Eq.(gcn-attn-agg)
- 注意力权重：Eq.(gcn-attn-weight)
