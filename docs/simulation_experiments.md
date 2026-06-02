# 基础仿真实验文档

## 1. 概述

本目录包含两个自包含的联邦学习基础仿真实验脚本，基于合成交通数据验证 CNN/GCN + BiLSTM + Attention 模型在联邦场景下的性能。

## 2. 文件结构

```text
simulation_experiments/
    cnn_fed_base.py          # CNN + BiLSTM + Attention 基础联邦仿真
    gcn_fed_base.py          # GCN + BiLSTM + Attention 基础联邦仿真 (图结构)
    requirements_ccn.txt     # CCN/C 仿真依赖
    requirements_gcn.txt     # GCN 仿真依赖
```

## 3. 运行命令

```powershell
conda activate analysis
cd simulation_experiments
python cnn_fed_base.py --workflow all
python gcn_fed_base.py --workflow all
```

workflow 选项：`all`（默认）、`overview`（仅总览）、`ablation`（仅消融）。

## 4. 核心架构

### 4.1 CNN 模型 (`cnn_fed_base.py`)

**数据生成**：合成 sin/cos/周期项 + 噪声交通序列，按 `client_id` 控制振幅 `amp = 0.5 + 0.3*(cid+1)` 和噪声强度体现客户端异质性。按时间顺序 70%/10%/20% 划分 train/val/test。

**模型结构** `CNNBiLSTMAttention`：

```text
Conv1d(kernel=3)×2 + BN + AdaptiveSwish + AdaptiveAvgPool1d
    +
BiLSTM(bidirectional) → mean pooling → Linear projection
    →
MultiheadAttention(4 heads) + LayerNorm + residual
    →
Linear → LayerNorm → AdaptiveSwish → Dropout → Linear(1)
```

**输出**：`(prediction, attention_weights)` 元组。

### 4.2 GCN 模型 (`gcn_fed_base.py`)

**图结构**：链式邻接矩阵，经 `A_hat = D^(-1/2)(A+I)D^(-1/2)` 对称归一化。

**模型结构** `GCNBiLSTMAttention`：

```text
GCNEncoder:
    node_proj (Linear + LayerNorm + AdaptiveSwish)
    → GCNLayer × 2 + LayerNorm × 2 + AdaptiveSwish
    → mean pooling over nodes
    +
BiLSTM(bidirectional) → mean pooling → Linear projection
    →
MultiheadAttention(4 heads) + LayerNorm + residual
    →
Linear → LayerNorm → AdaptiveSwish → Dropout → Linear(1)
```

**注**：`USE_LEARNABLE_ADJ = False`（默认使用固定链式图，可切换为可学习邻接矩阵）。

### 4.3 联邦聚合 (`FedAvgServer`)

**聚合公式**（标准样本量加权 FedAvg）：

```python
global_model = sum(n_i / total_n * local_model_i)
```

其中 `n_i` 为客户端 i 的训练样本量，`total_n` 为所有客户端训练样本量总和。

> **历史变更**：从 `WeightedFederatedServer`（包含 `loss_weights`、`exp(-loss)`、0.9/0.1 server damping）重构为纯样本量加权 `FedAvgServer`。

### 4.4 三种训练方式

| 方式 | 类/函数 | 说明 |
|---|---|---|
| Independent | `run_independent_training` | 每客户端本地独立训练，不做全局同步 |
| FedAvg | `FedAvgServer` / `run_fedavg_training` | 标准样本量加权 FedAvg |
| Ablation | `run_fedavg_ablation` | 模型变体消融（CCN-LSTM-Attention 等 4 变体） |

## 5. 超参数配置

| 参数 | 值 |
|---|---|
| `NUM_CLIENTS` | 3 |
| `NUM_NODES` | 8 |
| `TIME_STEPS` | 400 |
| `SEQ_LEN` | 12 |
| `PRED_LEN` | 1 |
| `COMM_ROUNDS` | 20 |
| `LOCAL_EPOCHS` | 3 |
| `BATCH_SIZE` | 32 |
| `LEARNING_RATE` | 0.001 |
| `HIDDEN_DIM` | 32 |
| `DROPOUT` | 0.1 |
| `SEED` | 42 (统一) |
| train/val/test 比例 | 70%/10%/20% (时间顺序，不打乱) |

## 6. 输出文件

### CCN/C 输出 (`results/simulation_experiments/cnn/`)

| 文件 | 内容 |
|---|---|
| `cnn_overview_figure.png` | overview 总览对比图 (6 子图) |
| `cnn_overview_metrics.csv` | overview 指标表 |
| `cnn_ablation_figure.png` | ablation 消融对比图 (4 子图) |
| `cnn_ablation_metrics.csv` | ablation 消融汇总表 |
| `cnn_run_log.txt` | 运行日志 |

### GCN 输出 (`results/simulation_experiments/gcn/`)

| 文件 | 内容 |
|---|---|
| `gcn_overview_figure.png` | overview 总览对比图 |
| `gcn_overview_metrics.csv` | overview 指标表 |
| `gcn_ablation_figure.png` | ablation 消融对比图 |
| `gcn_ablation_metrics.csv` | ablation 消融汇总表 |
| `gcn_run_log.txt` | 运行日志 |

## 7. 消融实验模型变体

| CCN/C 变体 | GCN 变体 | 说明 |
|---|---|---|
| CCN-LSTM-Attention | GCN-LSTM-Attention | 完整模型 |
| CCN-LSTM | GCN-LSTM | 移除注意力 (用 concat+FC 替代) |
| LSTM-Attention | LSTM-Attention | 移除空间编码器 |
| CCN-Attention | GCN-Attention | 移除 LSTM 时序分支 |

## 8. 历史变更记录

| 日期 | 变更 |
|---|---|
| 2026-06 | **初始创建**：cnn_fed_base.py、gcn_fed_base.py |
| 2026-06 | **激活函数优化**：ReLU → AdaptiveSwish |
| 2026-06 | **聚合重构**：WeightedFederatedServer → FedAvgServer (纯样本量加权) |
| 2026-06 | **命名统一**：ccn_simulation.py → cnn_fed_base.py，Federated → CCN-FedAvg/GCN-FedAvg |
| 2026-06 | **种子统一**：所有随机种子 → 42 |
| 2026-06 | **输出目录**：ccn/ → cnn/ (文件名同步更新) |
| 2026-06 | **动量混合**：FedAvg 聚合增加 0.9/0.1 动量混合 |
| 2026-06 | **StepLR**：增加学习率调度器 (step_size=3, gamma=0.9) |
| 2026-06 | **seaborn 可视化**：dpi 150→300, whitegrid 主题 |
