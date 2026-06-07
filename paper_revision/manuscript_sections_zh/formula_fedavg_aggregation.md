# Formula: FedAvg Aggregation

## 联邦平均算法（Federated Averaging / FedAvg）

### 标准 FedAvg 公式

服务器聚合客户端模型参数：

$$
\mathbf{w}^{t+1}
=
\sum_{k=1}^{K}
\frac{n_k}{\sum_{j=1}^{K} n_j}
\mathbf{w}_{k}^{t+1}
$$

其中：

- $\mathbf{w}^{t+1}$ 表示第 $t+1$ 轮全局模型参数；
- $\mathbf{w}_{k}^{t+1}$ 表示第 $k$ 个客户端在第 $t+1$ 轮本地训练后的模型参数；
- $K$ 表示客户端总数；
- $n_k$ 表示第 $k$ 个客户端的本地样本量；
- $\sum_{j=1}^{K} n_j$ 表示所有客户端的训练样本总量。

### 本地训练（客户端侧）

客户端 $k$ 的本地损失函数：

$$
\mathcal{L}_k(\mathbf{w})
=
\frac{1}{n_k}
\sum_{i=1}^{n_k}
\ell(\mathbf{w}; x_i^k, y_i^k)
$$

本地 SGD 更新（$E$ 个 epoch）：

$$
\mathbf{w}_k \leftarrow \mathbf{w}_k - \eta \nabla \mathcal{L}_k(\mathbf{w}_k)
$$

### 完整 FedAvg 流程

**Algorithm: FedAvg**

1. Server 初始化全局模型参数 $\mathbf{w}^0$
2. **for** each round $t = 0, 1, \dots, R-1$ **do**
3. $\qquad$ Server 广播 $\mathbf{w}^t$ 至所有 $K$ 个客户端
4. $\qquad$ **for** each client $k = 1, \dots, K$ **in parallel do**
5. $\qquad\qquad \mathbf{w}_k^{t} \leftarrow \mathbf{w}^t$（接收全局参数）
6. $\qquad\qquad$ **for** $e = 1, \dots, E$ **do**
7. $\qquad\qquad\qquad$ **for** each mini-batch $b \subset \mathcal{D}_k$ **do**
8. $\qquad\qquad\qquad\qquad \mathbf{w}_k^{t} \leftarrow \mathbf{w}_k^{t} - \eta \nabla \mathcal{L}_k(\mathbf{w}_k^{t}; b)$
9. $\qquad\qquad$ 上传 $\mathbf{w}_k^{t+1}$ 至 Server
10. $\qquad$ Server 聚合：$\mathbf{w}^{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} \mathbf{w}_{k}^{t+1}$
11. **return** $\mathbf{w}^{R}$

### 策略分类（根据 PRECHECK 约束）

| 策略 | 代码标识 | 论文处理 |
|------|----------|----------|
| FedAvg（标准加权平均）| FedAvg | **主方法** — 写入 Methodology |
| Independent（独立训练）| Independent | **对比基线** — 写入 Experiments |
| Loss-weighted | Loss-weighted | 历史探索 — 仅报告中，**不写入论文** |
| Data-loss weighted | Data-loss weighted | 历史探索 — 仅报告中，**不写入论文** |
| Proposed | Proposed | 历史探索 — 仅报告中，**不写入论文** |

### 论文中正确表述

- "We adopt the standard FedAvg algorithm for model aggregation"
- "The server aggregates client parameters via weighted averaging according to their sample sizes"
- "Following the FedAvg framework proposed by McMahan et al."
- "Model aggregation is performed using federated averaging"

## 公式引用标记

- FedAvg 聚合公式：Eq.(fedavg-aggregation)
- 客户端损失函数：Eq.(fedavg-client-loss)
- 本地 SGD 更新：Eq.(fedavg-local-sgd)
- 完整 FedAvg 算法：Algorithm.(fedavg)
