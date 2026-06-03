# 仿真实验文档

## 1. 概述

本目录包含 5 个联邦学习仿真实验脚本，覆盖基础 CNN/GCN、增强 CNN/GCN 以及鲁棒性实验。所有实验均基于合成交通数据，统一采用论文风格可视化输出，便于直接用于结果汇报和论文制图。

## 2. 文件结构

```text
simulation_experiments/
    cnn_fed_base.py          # CNN + BiLSTM + Attention 基础联邦仿真
    gcn_fed_base.py          # GCN + BiLSTM + Attention 基础联邦仿真 (图结构)
    cnn_fed_enhanced_experiments.py   # CNN 增强联邦仿真
    gcn_fed_enhanced_experiments.py   # GCN 增强联邦仿真
    fed_robustness_experiments.py     # 鲁棒性实验
    requirements_ccn.txt     # CNN 仿真依赖
    requirements_gcn.txt     # GCN 仿真依赖
```

## 3. 运行命令

```powershell
E:\anaconda3\envs\analysis\python.exe simulation_experiments/cnn_fed_base.py --workflow all
E:\anaconda3\envs\analysis\python.exe simulation_experiments/gcn_fed_base.py --workflow all
E:\anaconda3\envs\analysis\python.exe simulation_experiments/cnn_fed_enhanced_experiments.py --workflow all
E:\anaconda3\envs\analysis\python.exe simulation_experiments/gcn_fed_enhanced_experiments.py --workflow all
E:\anaconda3\envs\analysis\python.exe simulation_experiments/fed_robustness_experiments.py --workflow all
```

常用 workflow 示例：

```powershell
E:\anaconda3\envs\analysis\python.exe simulation_experiments/cnn_fed_base.py --workflow data_viz
E:\anaconda3\envs\analysis\python.exe simulation_experiments/gcn_fed_base.py --workflow convergence
E:\anaconda3\envs\analysis\python.exe simulation_experiments/cnn_fed_enhanced_experiments.py --workflow client_metrics
E:\anaconda3\envs\analysis\python.exe simulation_experiments/gcn_fed_enhanced_experiments.py --workflow fixed_vs_dynamic
E:\anaconda3\envs\analysis\python.exe simulation_experiments/fed_robustness_experiments.py --workflow communication_delay
```

说明：

- 基础实验支持 `all`、`data_viz`、`main`、`convergence`
- CNN 增强实验支持 `all`、`data_viz`、`main`、`aggregation`、`lambda`、`client_scale`、`noniid`、`convergence`、`client_metrics`、`peak`、`feature_ablation`
- GCN 增强实验支持 `all`、`data_viz`、`fixed_vs_dynamic`、`main`、`aggregation`、`lambda`、`client_scale`、`noniid`、`convergence`、`client_metrics`、`peak`、`congestion_delay`
- 鲁棒性实验支持 `all`、`communication_cost`、`client_dropout`、`communication_delay`、`dp_noise`

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

## 6. 可视化规范

所有实验脚本内部统一配置 `configure_academic_plot_style()`，核心规范如下：

- 使用 `seaborn` 的 `whitegrid` + `paper` 主题
- 默认输出分辨率为 `300 dpi`
- 统一方法颜色映射，确保 `Independent`、`FedAvg`、`Proposed` 及其 CNN/GCN 变体跨图一致
- 所有图片通过 `save_figure(fig, output_dir, file_name)` 保存，统一 `bbox_inches="tight"` 并自动关闭图对象
- 图题、坐标轴和图例尽量使用英文，避免论文排版中的中文字体乱码
- 每个结果目录额外生成 `figure_index.csv`，用于图表检索和论文选图

## 7. 输出文件

### 基础 CNN 输出 (`results/simulation_experiments/cnn_fed_base/`)

| 文件 | 内容 |
|---|---|
| `base_dataset_*.png` | 基础数据集时间序列、热力图、箱线图与样本统计图 |
| `cnn_base_main_comparison.png` | 基础 CNN 主结果对比图 |
| `cnn_base_convergence.png` | 收敛与训练损失图 |
| `cnn_base_metrics*.csv` | 指标结果与汇总 |
| `figure_index.csv` | 图表索引 |

### 基础 GCN 输出 (`results/simulation_experiments/gcn_fed_base/`)

| 文件 | 内容 |
|---|---|
| `base_dataset_*.png` | 基础数据集可视化 |
| `base_gcn_*.png` | 固定邻接矩阵与节点度分布 |
| `gcn_base_main_comparison.png` | 基础 GCN 主结果对比图 |
| `gcn_base_convergence.png` | 收敛与训练损失图 |
| `gcn_base_metrics*.csv` | 指标结果与汇总 |
| `figure_index.csv` | 图表索引 |

### 增强 CNN 输出 (`results/simulation_experiments/cnn_fed_enhanced/`)

| 文件 | 内容 |
|---|---|
| `enhanced_dataset_*.png` | 增强数据集分布、峰值模式、相关性与事件示例 |
| `cnn_enhanced_main_rmse_comparison.png` | 主结果对比图 |
| `cnn_enhanced_global_validation_rmse.png` | 全局验证 RMSE 收敛图 |
| `cnn_enhanced_client_training_loss.png` | 客户端训练损失图 |
| `cnn_enhanced_client_rmse_comparison.png` | 按客户端 RMSE 对比图 |
| `cnn_enhanced_peak_offpeak_comparison.png` | 峰值/平峰/事件时段对比图 |
| `cnn_enhanced_feature_ablation.png` | 特征消融图 |
| `figure_index.csv` | 图表索引 |

### 增强 GCN 输出 (`results/simulation_experiments/gcn_fed_enhanced/`)

| 文件 | 内容 |
|---|---|
| `enhanced_dataset_*.png` | 增强数据集可视化 |
| `enhanced_gcn_*.png` | 固定/动态图结构、功能相似、拥堵延迟相关图 |
| `gcn_enhanced_main_rmse_comparison.png` | 主结果对比图 |
| `gcn_enhanced_fixed_vs_dynamic_comparison.png` | 固定图与动态图对比 |
| `gcn_enhanced_global_validation_rmse.png` | 全局验证 RMSE 收敛图 |
| `gcn_enhanced_client_rmse_comparison.png` | 按客户端 RMSE 对比图 |
| `gcn_enhanced_peak_offpeak_comparison.png` | 峰值/平峰/事件时段对比图 |
| `gcn_enhanced_congestion_delay_comparison.png` | 拥堵延迟图结构对比 |
| `figure_index.csv` | 图表索引 |

### 鲁棒性实验输出 (`results/simulation_experiments/fed_robustness/`)

| 文件 | 内容 |
|---|---|
| `fed_robustness_communication_cost.png` | 通信开销图 |
| `fed_robustness_client_dropout.png` | 客户端掉线鲁棒性图 |
| `fed_robustness_communication_delay.png` | 通信延迟鲁棒性图 |
| `fed_robustness_dp_noise.png` | 隐私噪声敏感性图 |
| `figure_index.csv` | 图表索引 |

## 8. 消融实验模型变体

| CCN/C 变体 | GCN 变体 | 说明 |
|---|---|---|
| CCN-LSTM-Attention | GCN-LSTM-Attention | 完整模型 |
| CCN-LSTM | GCN-LSTM | 移除注意力 (用 concat+FC 替代) |
| LSTM-Attention | LSTM-Attention | 移除空间编码器 |
| CCN-Attention | GCN-Attention | 移除 LSTM 时序分支 |

## 9. 历史变更记录

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
| 2026-06 | **文档同步**：补充基础/增强/鲁棒性实验脚本、结果目录、workflow 与 figure_index 说明 |
| 2026-06 | **图表规范**：统一论文风格 PNG 命名、300 dpi 输出与学术风格保存函数 |
