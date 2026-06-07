# Formula Notation Glossary

## 图网与拓扑符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $N$ | $N$ | 道路网络节点总数 |
| $\mathcal{V}$ | $\mathcal{V}$ | 节点集合，$\mathcal{V} = \{1,2,\dots,N\}$ |
| $\mathcal{E}$ | $\mathcal{E}$ | 边集合，表示道路连通关系 |
| $A$ | $A \in \mathbb{R}^{N \times N}$ | 邻接矩阵 |
| $A_{ij}$ | $A_{ij}$ | 邻接矩阵元素；$A_{ij} > 0$ 表示节点 $i$ 与节点 $j$ 连通 |
| $\tilde{A}$ | $\tilde{A} = A + I$ | 加自环后的邻接矩阵 |
| $\tilde{D}$ | $\tilde{D}$ | 度矩阵（对角矩阵），$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ |
| $\mathcal{N}(i)$ | $\mathcal{N}(i)$ | 节点 $i$ 的邻居节点集合 |
| $|\mathcal{N}(i)|$ | $|\mathcal{N}(i)|$ | 节点 $i$ 的邻居数量 |
| $w_{i,j}$ | $w_{i,j}$ | 节点 $i$ 与节点 $j$ 之间的空间权重 |
| $d_{i,j}$ | $d_{i,j}$ | 节点 $i$ 与节点 $j$ 之间的物理距离或路段长度 |
| $\mathcal{G}$ | $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$ | 道路网络图 |

## 时间序列与交通状态符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $t$ | $t$ | 时间步索引 |
| $\tau$ | $\tau$ | 预测步长（预测未来 $\tau$ 步） |
| $T$ | $T$ | 总时间步数 |
| $X_{i,t}$ | $X_{i,t} \in \mathbb{R}$ | 节点 $i$ 在时刻 $t$ 的交通状态值 |
| $X_t$ | $X_t \in \mathbb{R}^{N \times F}$ | 时刻 $t$ 所有节点的特征矩阵 |
| $F$ | $F$ | 特征维度 |
| $\hat{Y}_{i,t+\tau}$ | $\hat{Y}_{i,t+\tau}$ | 节点 $i$ 在未来时刻 $t+\tau$ 的预测值 |

## CNN 空间模块符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $v_i$ | $v_i$ | 第 $i$ 个道路网络节点 |
| $M_{i,t}$ | $M_{i,t} \in \mathbb{R}^{p \times q}$ | 节点 $v_i$ 在时刻 $t$ 的局部邻域矩阵 |
| $W^{(m)}$ | $W^{(m)}$ | 第 $m$ 个卷积核的权重矩阵 |
| $b^{(m)}$ | $b^{(m)}$ | 第 $m$ 个卷积核的偏置 |
| $*$ | $*$ | 卷积操作 |
| $\sigma(\cdot)$ | $\sigma(\cdot)$ | 非线性激活函数（如 ReLU 或 ELU） |
| $H_{i,t}^{(m)}$ | $H_{i,t}^{(m)}$ | 节点 $i$ 在时刻 $t$ 的第 $m$ 个卷积通道输出特征 |
| $M$ | $M$ | 卷积通道总数 |
| $p, q$ | $p, q$ | 局部邻域矩阵的行数和列数 |

## GCN 空间模块符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $H_t$ | $H_t$ | GCN 层在时刻 $t$ 的输出特征矩阵 |
| $H_t^{(l)}$ | $H_t^{(l)}$ | 第 $l$ 层 GCN 在时刻 $t$ 的输出 |
| $W$ | $W$ | GCN 可学习权重矩阵 |
| $W^{(l)}$ | $W^{(l)}$ | 第 $l$ 层 GCN 的可学习权重矩阵 |
| $L$ | $L$ | GCN 总层数 |

## BiLSTM 时间模块符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $h_t$ | $h_t$ | LSTM 在时刻 $t$ 的隐藏状态 |
| $C_t$ | $C_t$ | LSTM 在时刻 $t$ 的细胞状态 |
| $f_t$ | $f_t$ | 遗忘门在时刻 $t$ 的输出 |
| $i_t$ | $i_t$ | 输入门在时刻 $t$ 的输出 |
| $o_t$ | $o_t$ | 输出门在时刻 $t$ 的输出 |
| $\tilde{C}_t$ | $\tilde{C}_t$ | 候选细胞状态 |
| $\overrightarrow{h_t}$ | $\overrightarrow{h_t}$ | 前向 LSTM 在时刻 $t$ 的隐藏状态 |
| $\overleftarrow{h_t}$ | $\overleftarrow{h_t}$ | 后向 LSTM 在时刻 $t$ 的隐藏状态 |
| $h_t^{\text{BiLSTM}}$ | $h_t^{\text{BiLSTM}}$ | BiLSTM 在时刻 $t$ 的拼接隐藏状态 |
| $W_f, W_i, W_C, W_o$ | $W_f, W_i, W_C, W_o$ | 遗忘门、输入门、候选状态、输出门的权重矩阵 |
| $b_f, b_i, b_C, b_o$ | $b_f, b_i, b_C, b_o$ | 遗忘门、输入门、候选状态、输出门的偏置向量 |

## Attention 融合模块符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\alpha_{i,t}^{(m)}$ | $\alpha_{i,t}^{(m)}$ | 节点 $i$ 在时刻 $t$ 对第 $m$ 个通道的注意力权重 |
| $\alpha_{i,j,t}$ | $\alpha_{i,j,t}$ | 节点 $i$ 与节点 $j$ 在时刻 $t$ 之间的注意力权重 |
| $q$ | $q$ | 查询向量 |
| $q_i$ | $q_i$ | 节点 $i$ 的查询向量 |
| $k^{(m)}$ | $k^{(m)}$ | 第 $m$ 个通道的键向量 |
| $k_j$ | $k_j$ | 节点 $j$ 的键向量 |
| $d$ | $d$ | 缩放因子（键向量的维度） |
| $z_{i,t}$ | $z_{i,t}$ | 节点 $i$ 在时刻 $t$ 经过注意力加权聚合后的特征 |
| $h_t^{\text{spatial}}$ | $h_t^{\text{spatial}}$ | 时刻 $t$ 的空间特征表示 |
| $h_t^{\text{temporal}}$ | $h_t^{\text{temporal}}$ | 时刻 $t$ 的时间特征表示 |
| $h_t^{\text{fused}}$ | $h_t^{\text{fused}}$ | 时刻 $t$ 的时空融合特征 |

## 联邦学习符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $K_c$ | $K_c$ | 客户端总数（联邦学习上下文专用） |
| $n_k$ | $n_k$ | 客户端 $k$ 的本地训练样本数量 |
| $N$ | $N$ | 所有客户端训练样本总量，$N = \sum_{k=1}^{K_c} n_k$ |
| $\mathbf{w}^{t}$ | $\mathbf{w}^{t}$ | 第 $t$ 轮通信的全局模型参数 |
| $\mathbf{w}_k^{t}$ | $\mathbf{w}_k^{t}$ | 客户端 $k$ 在第 $t$ 轮通信结束时的本地模型参数 |
| $\mathcal{L}_k(\mathbf{w})$ | $\mathcal{L}_k(\mathbf{w})$ | 客户端 $k$ 在参数 $\mathbf{w}$ 下的本地损失函数 |
| $\ell(\cdot, \cdot)$ | $\ell(\cdot, \cdot)$ | 单个样本的损失函数（MSE、MAE 等） |
| $\eta$ | $\eta$ | 学习率 |
| $E$ | $E$ | 每轮通信中本地训练的 epoch 次数 |
| $R$ | $R$ | 联邦学习通信总轮次 |
| $\mathcal{D}_k$ | $\mathcal{D}_k$ | 客户端 $k$ 的本地数据集 |
| $b$ | $b$ | 一个 mini-batch，$b \subset \mathcal{D}_k$ |

## 数据预处理符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $\phi_k(t)$ | $\phi_k(t)$ | 第 $k$ 个时间基函数 |
| $\alpha_{i,k}$ | $\alpha_{i,k}$ | 节点 $i$ 在第 $k$ 个基函数上的拟合系数 |
| $K_b$ | $K_b$ | 基函数总数量（数据预处理上下文专用，区别于联邦学习 $K_c$） |
| $\lambda$ | $\lambda \in [0,1]$ | 缺失值混合填补权重系数 |
| $X_{i,t}^{\text{filled}}$ | $X_{i,t}^{\text{filled}}$ | 节点 $i$ 在时刻 $t$ 的混合填补结果 |
| $\hat{X}_{i,t}^{\text{geo}}$ | $\hat{X}_{i,t}^{\text{geo}}$ | 节点 $i$ 在时刻 $t$ 的地理邻近性填补结果 |
| $\hat{X}_{i,t}^{\text{func}}$ | $\hat{X}_{i,t}^{\text{func}}$ | 节点 $i$ 在时刻 $t$ 的函数曲线拟合填补结果 |

## 预测输出符号

| 符号 | LaTeX | 含义 |
|------|-------|------|
| $W_o$ | $W_o$ | 输出层权重矩阵 |
| $b_o$ | $b_o$ | 输出层偏置向量 |
| $f_{\theta}$ | $f_{\theta}$ | 参数为 $\theta$ 的交通流预测函数 |
