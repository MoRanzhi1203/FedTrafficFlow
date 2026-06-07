# Formula: CNN Spatial Module

## 基于网格卷积的空间特征提取

### 局部邻域矩阵构造

对节点 $v_i$，构造局部邻域矩阵 $M_{i,t} \in \mathbb{R}^{p \times q}$：

$$
M_{i,t}
=
\begin{bmatrix}
X_{i-p/2, j-q/2, t} & \cdots & X_{i-p/2, j+q/2, t} \\
\vdots & \ddots & \vdots \\
X_{i+p/2, j-q/2, t} & \cdots & X_{i+p/2, j+q/2, t}
\end{bmatrix}
$$

其中 $p$ 和 $q$ 为邻域窗口大小。

### 单通道卷积特征提取

$$
H_{i,t}^{(m)}
=
\sigma(W^{(m)} * M_{i,t} + b^{(m)})
$$

其中各个符号的含义：

- $W^{(m)}$ 表示第 $m$ 个卷积核权重；
- $b^{(m)}$ 表示第 $m$ 个卷积核偏置；
- $*$ 表示卷积操作；
- $\sigma(\cdot)$ 表示非线性激活函数（ReLU/ELU）；
- $M$ 表示卷积通道总数；
- $H_{i,t}^{(m)}$ 表示第 $m$ 个卷积通道提取的特征。

### 多通道特征输出

每个节点输出 $M$ 个特征图：

$$
\{H_{i,t}^{(1)}, H_{i,t}^{(2)}, \dots, H_{i,t}^{(M)}\}
$$

## 代码实现

CNN 空间模块在仿真实验 core 文件中实现：

- `cnn_fed_base/cfb_core.py` 包含 CNN 模型类定义；
- `cnn_fed_enhanced_experiments/cfe_core.py` 包含 CNN 模型定义（含增强实验变体）。

### 关键参数（代码中）

- `NUM_NODES = 8`（仿真）或真实路网节点数
- `HIDDEN_DIM = 64`（隐藏层维度）
- 使用 `nn.Conv1d` 与 `nn.Conv2d`

## 公式引用标记

- 局部邻域矩阵：Eq.(cnn-local-matrix)
- 卷积特征提取：Eq.(cnn-conv-feature)
- 多通道输出：Eq.(cnn-multi-channel)
