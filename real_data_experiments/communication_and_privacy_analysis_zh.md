# 通信开销与隐私分析

> 生成日期：2026-07-01
> 工具脚本：`real_data_experiments/tools/estimate_communication_cost.py`

## 1. 通信开销估计

当前模型参数量约 **10,194** (baseline CNN-LSTM-Attention)。FP32 下每 client 每轮上传/下载约 **0.039 MB**。

### 1.1 各实验配置通信量

| 实验 | Clients | Rounds | 每轮总上传 (MB) | R20 总上传 (MB) |
|---|---|---|---|---|
| Exp1 (K=5) | 5 | 20 | 0.194 | 3.88 |
| Exp3 (K=5) | 5 | 20 | 0.194 | 3.88 |
| Exp5 (K=3) | 3 | 20 | 0.116 | 2.33 |
| Exp5 (K=10) | 10 | 20 | 0.389 | 7.77 |

### 1.2 CalendarFeature 模型额外开销

CalendarFeatureFedAvg 模型增加了 calendar encoder 和 gate 层，参数量略增，但仍在 ~11k 参数级别。

## 2. 掉线鲁棒性

### 2.1 当前状态

所有实验均在 **全量 client 参与** 条件下运行。无 client dropout 模拟。

### 2.2 论文处理建议

在 discussion/limitations 中说明：
- 当前实验假设所有 client 在每轮均参与训练
- 真实部署中 client 可能因网络/设备原因掉线
- FedProx 和 momentum-based aggregation 可增强掉线鲁棒性
- 通信开销极低（<10 MB per R20），不影响实际可行性

## 3. 差分隐私 (DP) 分析

### 3.1 当前状态

所有实验 **未添加 DP noise**。梯度/参数以明文传输和聚合。

### 3.2 已有隐私保护

当前系统的隐私保护仅基于：
- **数据本地化**：raw traffic data 不出 client 本地
- **模型聚合**：仅交换模型参数（非原始数据）

### 3.3 风险

- 参数可能通过梯度反演泄露局部交通流模式
- 无 formal DP guarantee

### 3.4 论文建议

在 limitations 中明确：
- 当前未实现 DP-SGD
- 隐私保护是 raw data localization
- DP-SGD 会引入 accuracy-privacy tradeoff
- 作为 future work
