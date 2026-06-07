# Formula: Data Preprocessing

## 缺失值填补方法

### 1. 地理邻近性填补

当节点 $i$ 在时刻 $t$ 的观测值缺失时，通过空间邻近节点观测值进行均值填补：

$$
X_{i,t}
\approx
\frac{1}{|\mathcal{N}(i)|}
\sum_{j \in \mathcal{N}(i)}
X_{j,t}
$$

加权均值版本（提高稳健性）：

$$
X_{i,t}
\approx
\frac{\sum_{j \in \mathcal{N}(i)} w_{i,j} X_{j,t}}
{\sum_{j \in \mathcal{N}(i)} w_{i,j}}
$$

其中权重 $w_{i,j} = \frac{1}{d_{i,j}}$，$d_{i,j}$ 为节点间物理距离或路段长度。

### 2. 函数曲线拟合填补

对节点 $i$ 完整时序数据 $\{X_{i,t}\}_{t=1}^T$ 进行基函数展开拟合：

$$
X_i(t)
\approx
\sum_{k=1}^{K_b}
\alpha_{i,k}
\phi_k(t)
$$

其中 $\{\phi_k(t)\}_{k=1}^{K_b}$ 为时间基函数（如 Fourier 基、B-spline 基），$\alpha_{i,k}$ 为系数，$K_b$ 为基函数数量。

最小二乘求解系数：

$$
\{\hat{\alpha}_{i,1}, \dots, \hat{\alpha}_{i,K_b}\}
=
\arg\min_{\alpha_{i,1}, \dots, \alpha_{i,K_b}}
\sum_{t=1}^T
\Big(
X_{i,t}
-
\sum_{k=1}^{K_b}
\alpha_{i,k}
\phi_k(t)
\Big)^2
$$

重构缺失值：

$$
\hat{X}_{i,t}
=
\sum_{k=1}^{K_b}
\hat{\alpha}_{i,k}
\phi_k(t)
$$

### 3. 混合填补

结合地理邻近性与函数曲线拟合：

$$
X_{i,t}^{\text{filled}}
=
\lambda \cdot \hat{X}_{i,t}^{\text{geo}}
+
(1-\lambda) \cdot \hat{X}_{i,t}^{\text{func}},
\quad \lambda \in [0,1]
$$

通过调整 $\lambda$ 在空间邻近性与时间连续性之间取得平衡。

其中：

- $\hat{X}_{i,t}^{\text{geo}}$ 表示地理邻近性填补值；
- $\hat{X}_{i,t}^{\text{func}}$ 表示函数曲线拟合填补值。

## 代码实现参考

- 地理邻近性：`analysis_scripts/check_spatial_node_completeness.py`
- 函数曲线拟合：`analysis_scripts/fit_node_flow_daily_curve.py`
- 速度统计：`analysis_scripts/summarize_speed_stats.py`
- 密度计算：`analysis_scripts/compute_greenshields_density.py`

## 公式引用标记（用于论文交叉引用）

- 地理邻近性填补（均值）：Eq.(pre-geo-mean)
- 地理邻近性填补（加权）：Eq.(pre-geo-weighted)
- 基函数展开：Eq.(pre-func-expand)
- 最小二乘求解：Eq.(pre-func-lsq)
- 重构填补：Eq.(pre-func-reconstruct)
- 混合填补：Eq.(pre-mixed)
