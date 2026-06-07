# Formula: BiLSTM Temporal Module

## 双向长短期记忆网络（BiLSTM）时间特征提取

### LSTM 单元（单向）

标准 LSTM 单元在时刻 $t$ 的计算：

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

其中各个符号的含义：

- $f_t$ 表示遗忘门（Forget Gate）输出；
- $i_t$ 表示输入门（Input Gate）输出；
- $o_t$ 表示输出门（Output Gate）输出；
- $C_t$ 表示细胞状态（Cell State）；
- $h_t$ 表示隐藏状态（Hidden State）；
- $\tilde{C}_t$ 表示候选细胞状态；
- $x_t$ 表示时刻 $t$ 的输入；
- $h_{t-1}$ 和 $C_{t-1}$ 分别表示上一时刻的隐藏状态和细胞状态；
- $W_f, W_i, W_C, W_o$ 和 $b_f, b_i, b_C, b_o$ 为对应的权重矩阵和偏置向量；
- $\sigma$ 表示 sigmoid 激活函数；
- $\odot$ 表示逐元素乘积。

### 双向 LSTM（BiLSTM）

$$
\begin{aligned}
\overrightarrow{h_t} &= \text{LSTM}(x_t, \overrightarrow{h}_{t-1}) \\
\overleftarrow{h_t} &= \text{LSTM}(x_t, \overleftarrow{h}_{t+1}) \\
h_t^{\text{BiLSTM}} &= [\overrightarrow{h_t}; \overleftarrow{h_t}]
\end{aligned}
$$

其中 $[\cdot; \cdot]$ 表示拼接操作，$\overrightarrow{h_t}$ 和 $\overleftarrow{h_t}$ 分别表示前向和后向 LSTM 的隐藏状态。

### 与 CNN/GCN 的集成

BiLSTM 接收空间模块的特征输出：

$$
\begin{aligned}
h_t^{\text{spatial}} &= \text{CNN}(X_t) \quad \text{or} \quad \text{GCN}(X_t, A) \\
h_t^{\text{temporal}} &= \text{BiLSTM}(h_1^{\text{spatial}}, h_2^{\text{spatial}}, \dots, h_t^{\text{spatial}})
\end{aligned}
$$

## 代码实现

BiLSTM 集成在仿真实验核心文件中：

- `cnn_fed_base/cfb_core.py`
- `gcn_fed_base/gfb_core.py`

使用 PyTorch 的 `nn.LSTM` 并将参数 `bidirectional` 设为 `True`。

## 公式引用标记

- LSTM 单元：Eq.(lstm-unit)
- BiLSTM 双向：Eq.(bilstm-bidirectional)
- 时空集成：Eq.(spatial-temporal-integration)
