# 差分隐私 (DP) 限制说明

> 生成日期：2026-07-01

## 1. 当前隐私保护现状

当前真实数据联邦实验的隐私保护仅基于：
- **数据不离开本地**：每个 client 只在本地使用自己的 grid cell 数据进行训练
- **仅交换模型参数**：通信内容为模型权重/梯度，不包含原始交通流量数据

## 2. 未实现的保护

| 机制 | 状态 | 原因 |
|---|---|---|
| DP-SGD (梯度裁剪+加噪) | 未实现 | 会增加 accuracy-privacy tradeoff；当前优先完成 baseline 实验 |
| Secure Aggregation | 未实现 | 需要多方安全计算基础设施 |
| 梯度反演防御 | 未实现 | 参数仍可能通过梯度反演泄露局部模式 |

## 3. 风险说明

- 模型参数可能通过 gradient inversion 攻击反推局部交通流特征
- 在跨机构 FL 场景下需要更严格隐私保护
- 当前实验设定为**单机多 client 仿真**，实际部署需额外防护

## 4. 论文写法

```
Privacy: The current experiments assume a cross-silo FL setting
where raw traffic data remains local. No differential privacy
noise is applied. While model parameter exchange could theoretically
leak local traffic patterns via gradient inversion, DP-SGD and
secure aggregation can be integrated as future extensions at the
cost of a known accuracy-privacy trade-off.
```
