# Formula: Attention Fusion Module

## 注意力机制融合时空特征

### 空间注意力（CNN 内部）

对多通道卷积特征进行注意力加权：

$$
\alpha_{i,t}^{(m)}
=
\frac{\exp((q^\top k^{(m)})/\sqrt{d})}
{\sum_{m'=1}^M \exp((q^\top k^{(m')})/\sqrt{d})}
$$

加权聚合：

$$
z_{i,t}
=
\sum_{m=1}^M
\alpha_{i,t}^{(m)}
H_{i,t}^{(m)}
$$

其中各个符号的含义：

- $q$ 表示查询向量（可学习或由输入特征生成）；
- $k^{(m)}$ 表示第 $m$ 个通道的键向量；
- $d$ 表示缩放因子（向量维度）；
- $M$ 表示卷积通道数；
- $\alpha_{i,t}^{(m)}$ 表示节点 $i$ 在时刻 $t$ 对第 $m$ 个通道的注意力权重；
- $H_{i,t}^{(m)}$ 表示第 $m$ 个卷积通道提取的特征；
- $z_{i,t}$ 表示注意力加权聚合后的特征。

### GCN 节点注意力（空间）

对邻居节点特征进行注意力加权：

$$
\alpha_{i,j,t}
=
\frac{\exp((q_i^\top k_j)/\sqrt{d})}
{\sum_{v_{j'} \in \mathcal{N}(i)} \exp((q_i^\top k_{j'})/\sqrt{d})}
$$

$$
z_{i,t}
=
\sum_{v_j \in \mathcal{N}(i)}
\alpha_{i,j,t}
h_{j,t}
$$

### 时空融合注意力

将 CNN/GCN 空间特征与 BiLSTM 时间特征进行融合：

$$
h_t^{\text{fused}}
=
\text{Attention}(h_t^{\text{spatial}}, h_t^{\text{temporal}})
$$

可能的融合方式：

1. **拼接 + 全连接**：$h^{\text{fused}} = W \cdot [h^{\text{spatial}}; h^{\text{temporal}}] + b$
2. **加性注意力**：$\alpha = \text{softmax}(v^\top \tanh(W \cdot [h^{\text{spatial}}; h^{\text{temporal}}]))$
3. **门控融合**：$g = \sigma(W_g \cdot [h^{\text{spatial}}; h^{\text{temporal}}])$，$h^{\text{fused}} = g \odot h^{\text{spatial}} + (1-g) \odot h^{\text{temporal}}$

其中 $g$ 表示门控融合权重。

### 最终预测输出

融合特征通过全连接层得到预测值：

$$
\hat{Y}_{i,t+\tau}
=
W_o \cdot h_t^{\text{fused}}
+
b_o
$$

## 代码实现

Attention 模块集成在仿真实验核心文件中的模型定义部分。

## 公式引用标记

- 空间注意力权重：Eq.(attn-spatial-weight)
- 空间注意力聚合：Eq.(attn-spatial-agg)
- GCN 节点注意力：Eq.(attn-gcn-node)
- 时空融合注意力：Eq.(attn-spatial-temporal)
- 最终预测输出：Eq.(pred-output)
