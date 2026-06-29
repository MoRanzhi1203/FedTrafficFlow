# 真实数据实验 1–6 超参数与可复现设置表

> 生成日期：2026-06-29
> 参数来源：源码（*_config.py、*_core.py）、run_config.json、metrics 文件

---

## 1. 参数来源说明

| 来源层级 | 文件 | 说明 |
|----------|------|------|
| 配置默认值 | `sic_config.py`, `rc_config.py`, `ra_config.py`, `rfc_config.py`, `sia_config.py` | dataclass 默认值 |
| CLI 覆盖 | `--rounds`, `--local-epochs`, `--batch-size`, `--learning-rate` 等 | 命令行覆盖 |
| 实际运行记录 | `run_config.json` (各输出目录) | 最终生效值 |
| 模型硬编码 | `sic_core.py` (CNNLSTMAttentionRegressor), `sia_core.py` (SingleIntersectionAblationModel), `ra_core.py` (RegionAblationModel) | 未暴露为参数 |
| 训练硬编码 | `trainer.py`, `client.py`, `fedavg.py` | 聚合逻辑 |
| 标记 `—` | — | 代码未显式记录，需人工确认 |

---

## 2. 模型结构超参数表

| 模块 | 参数 | 实验适用范围 | 取值 | 来源文件 | 作用 | 选择依据 |
|---|---|---|---|---|---|---|
| Input | input_channels | Exp1/2/3: 2; Exp5/6: 2 | `use_channels=[0,1]` | `*_config.py` | 双通道输入（流量+速度） | 下游信道提供辅助信息 |
| Input | input_length (sequence_length) | 全部 | 12 | `*_config.py` 默认; exp1 formal run_config 确认 | 12步历史窗口（3小时@15min） | 覆盖短时交通流模式 |
| Output | prediction_horizon | 全部 | 1 | `*_config.py` 默认 | 单步预测（15min ahead） | 交通流短时预测标准设置 |
| CNN | Conv1d layer 1 | 全部 | in→16, kernel=3, padding=1, ReLU | `sic_core.py` line 80-81 | 空间特征提取 | — |
| CNN | Conv1d layer 2 | 全部 | 16→32, kernel=3, padding=1, ReLU | `sic_core.py` line 82-83 | 空间特征提取 | — |
| CNN | hidden_dim (=CNN output) | 全部 | 32 | `sic_core.py` line 77 默认; `ra_core.py` line 61 硬编码 | LSTM 输入维度 | — |
| LSTM | type | 全部 | 单向 LSTM (num_layers=1) | `sic_core.py` line 85; `ra_core.py` line 81 | 时序建模 | 非 BiLSTM（代码使用单向 LSTM） |
| LSTM | hidden_size | 全部 | 32 | `sic_core.py` line 77 默认; `ra_core.py` line 61 硬编码 | 隐状态维度 | — |
| Attention | type (Exp1/2) | Exp1/2 | Linear+Softmax (scaled dot-product) | `sic_core.py` line 62-71 (Attention class) | 时间注意力 | — |
| Attention | type (Exp5/6) | Exp5/6 | 同上 (复用 Attention) | `ra_core.py` line 87 复用 sic_core.Attention | 时间注意力 | — |
| Output | head | 全部 | Linear(hidden_dim→prediction_horizon) | `sic_core.py` line 87; `ra_core.py` line 88 | 回归输出 | — |
| Model | variant (Exp1) | Exp1 | `baseline` (CNNLSTMAttentionRegressor) | `sic_core.py`; run_config 确认 | 主模型 | — |
| Model | variants (Exp2) | Exp2 | `full`, `without_attention`, `without_cnn`, `without_lstm` | `sia_core.py` line 40-45; `sia_config.py` DEFAULT_VARIANTS | 消融变体 | CNN/LSTM/Attention 各组件消融 |
| Model | variants (Exp6) | Exp6 | `full`, `without_attention`, `without_cnn`, `without_lstm` | `ra_core.py` line 34-38; `ra_config.py` DEFAULT_VARIANTS | 消融变体 | 同 Exp2 的 4 变体结构 |
| Model | 参数量 (baseline) | Exp1/2/3/5 | 10,194 | `exp1_legacy_ipynb_model_diagnosis_zh.md` 记录 | 模型规模 | 轻量级 |
| Model | 参数量 (legacy_ipynb) | Exp1 optional | 62,915 | `exp1_legacy_ipynb_model_diagnosis_zh.md` 记录 | 模型规模 | 仅诊断用，当前不推荐 |
| Regularization | dropout | 全部 | 无 | 代码中无 nn.Dropout | — | 代码未显式记录，需人工确认 |

---

## 3. 联邦训练与聚合超参数表

| 参数 | 符号 | 实验适用范围 | 取值 | 来源文件 | 作用 | 选择依据 |
|---|---|---|---|---|---|---|
| 客户端数量 | K | Exp1: 5; Exp3: 5; Exp5/6: 3 | `num_clients` | `*_config.py` + run_config | 联邦参与方数量 | Exp1=5单网格, Exp5=3区域分区 |
| 通信轮数 | R | Exp1 formal: 20; Exp3/5/6 smoke: 1; Exp5/6 formal: 20 | `communication_rounds` | run_config.json | 联邦聚合迭代次数 | r20 在 exp1 上收敛 |
| 本地 epoch | E | Exp1 formal: 1; Exp5/6 formal: 1; (历史 exp1/2: 3) | `local_epochs` | run_config.json | 每轮本地训练迭代 | — |
| Batch size | B | Exp1/2: 64; Exp3: 32; Exp5/6: 32 | `batch_size` | `*_config.py` | 每批样本数 | — |
| 学习率 | η | 全部: 1e-3 | `learning_rate` | `*_config.py` 默认 1e-3 | Adam 初始学习率 | — |
| 优化器 | — | Adam | `sic_core.py` line 558; `rc_core.py` line 341 | torch.optim.Adam | 自适应优化 | PyTorch 默认 β=(0.9, 0.999) |
| 损失函数 | — | MSELoss | `sic_core.py` line 559; `rc_core.py` line 300; `trainer.py` 中 client 内部 | nn.MSELoss() | 回归损失 | 标准回归损失 |
| 聚合方式 | — | FedAvg (sample_count 加权) | `trainer.py` line 27-29; `fedavg.py` | 样本量加权平均 | 联邦聚合 | 标准 FedAvg |
| 客户端加权 | — | 按 train split 样本数加权 | `rc_core.py` 使用 FedClient.sample_count | `client.py` 中 `len(train_loader.dataset)` | 聚合权重 | — |
| 全局种子 | — | Exp1: 42; Exp3 smoke: 2026; Exp5/6: 配置默认 42 | `seed` | run_config.json / `*_config.py` | 可复现性 | — |
| Weight decay | — | 无 | 代码中无 weight_decay | — | — | 代码未显式记录，需人工确认 |
| Gradient clipping | — | 无 | 代码中无 clip_grad_norm_ | — | — | 代码未显式记录，需人工确认 |
| λ (聚合系数) | — | 无 | 仅标准 FedAvg 加权 | — | — | 标准 FedAvg 无 λ |
| β (动量) | — | 无 | 无 momentum/memory 机制 | — | — | 标准 FedAvg 无动量 |
| ρ (平滑因子) | — | 无 | 无 smoothing | — | — | 代码未显式记录，需人工确认 |

---

## 4. 数据划分、client 构造与运行设置表

| 实验 | 数据入口 | client 组织方式 | partition 方法 | train/val/test | rounds | local_epochs | batch_size | device | 输出目录 |
|---|---|---|---|---|---|---|---|---|---|
| Exp1 | `node_flow_grid_tensor.pt` | 5 selected grid cells (290,284,318,288,289) | 手动指定 selected_clients | 70%/15%/15% 时序连续 | 20 | 1 | 64 | cuda (RTX 3060) | `formal/exp1_single_grid_baseline_r20_e1_cuda/` |
| Exp2 | 同上 | 同上 | 同上 | 同上 | 历史 20 (已删) | 历史 3 (已删) | 64 | cuda | 历史目录已删除 |
| Exp3 smoke | 同上 + `similarity_k5.json` | 5 clients, 每 client 含 ~44 个相似 grid cells | similarity clustering (Pearson corr + 特征距离) | 70%/15%/15% | 1 | 1 | 32 | cuda | `smoke/exp3_rfc_similarity_k5_r1e1/` |
| Exp4 | — | — | — | — | — | — | — | — | 未开发 |
| Exp5 smoke | 同上 | 3 clients, spatial 蛇形分块 | spatial_block | 70%/15%/15% | 1 | 1 | 32 | cuda | `smoke/exp5_rc_spatial_block_k3_r1e1/` |
| Exp5 smoke | 同上 | 3 clients, KMeans 聚类 | flow_kmeans | 70%/15%/15% | 1 | 1 | 32 | cuda | `smoke/exp5_rc_flow_kmeans_k3_r1e1/` |
| Exp5 formal | 同上 | 同上 | spatial_block | 70%/15%/15% | 20 | 1 | 32 | cuda | `formal/exp5_rc_spatial_block_k3_r20_e1_cuda/` |
| Exp5 formal | 同上 | 同上 | flow_kmeans | 70%/15%/15% | 20 | 1 | 32 | cuda | `formal/exp5_rc_flow_kmeans_k3_r20_e1_cuda/` |
| Exp6 smoke | 同上 | 3 clients, 复用 Exp5 spatial_block | spatial_block | 70%/15%/15% | 1 | 1 | 32 | cuda | `smoke/exp6_ra_spatial_block_k3_r1e1/` |
| Exp6 formal | 同上 | 同上 | spatial_block | 70%/15%/15% | 20 | 1 | 32 | cuda | `formal/exp6_ra_spatial_block_k3_full_r20_e1_cuda/` |

### 补充信息

| 项目 | 取值 |
|------|------|
| Tensor shape | `[2, 630, 5856]` (2 channels × 630 regions × 5856 timesteps) |
| 时间范围 | 2017-04-01 ~ 2017-05-31 (61天, 15min粒度) |
| Active regions | 223 个 `is_active_region=True` |
| 输入归一化 | z-score (per-channel, train-split statistics) |
| 目标归一化 | z-score (train-split statistics) — Exp1/2/3 有，Exp5/6 ❌缺失 |
| 数据划分策略 | 时序连续 (temporal contiguous), 非随机 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Python | E:\anaconda3\envs\FedTrafficFlow\python.exe |
| PyTorch | 2.8.0+cu126 |

---

## 5. 缺失参数清单

以下参数在源码或 run_config 中未显式记录，但论文可能需要报告：

| 参数 | 状态 | 说明 |
|------|------|------|
| dropout rate | 缺失 | 代码中未使用 nn.Dropout |
| gradient clipping | 缺失 | 未使用 clip_grad_norm_ |
| weight_decay | 缺失 | Adam 默认 weight_decay=0 |
| Adam β₁, β₂ | 隐式 | PyTorch 默认 (0.9, 0.999)，未在 config 中记录 |
| λ / β / ρ (聚合调优) | 缺失 | 仅标准 FedAvg，无超参数 |
| local_epochs (Exp5/6 formal) | 1 | 极低，可能导致训练不足（但对当前失效问题不是主因） |
| learning rate schedule | 缺失 | 无 lr scheduler |
| early stopping | 缺失 | 无 |
| dataset shuffle | 缺失 | DataLoader 均为 shuffle=False |

---

## 6. 论文可写入内容建议

### 6.1 参数选择理由

- **sequence_length=12**：覆盖 3 小时（15min×12）的短期交通流动态，与城市交通流短时预测文献一致
- **prediction_horizon=1**：单步 15min 预测，是交通流预测的标准设置
- **local_epochs=1**：在 sample-count 加权 FedAvg 中，每轮仅 1 个 local epoch 可减少 client drift（但可能导致训练不充分，需配合足够 rounds）
- **batch_size=32/64**：适应 GPU 内存（RTX 3060 6GB），multi-region client 场景下每个 client 样本量很大（~300k samples），32 可加速 DataLoader 遍历
- **learning_rate=1e-3**：Adam 默认学习率，在归一化训练空间下表现稳定

### 6.2 关于 λ (聚合权重)

本工作使用标准 FedAvg，按各 client 的 train split 样本数加权聚合，未引入额外的 λ 超参数：

```
global_weights = Σ_k (n_k / n_total) × local_weights_k
```

其中 n_k 为 client k 的训练样本数。该策略确保大样本 client 对全局模型贡献更大，是 FedAvg 的标准做法。

### 6.3 关于 ρ (平滑因子)

本工作未使用 FedAvg 动量或平滑机制，每轮独立聚合：

```
w_{t+1} = Σ_k (n_k / n_total) × w_k^t
```

若审稿人要求讨论收敛稳定性，可在 revision 中补充 FedAvgM (momentum) 或 FedProx 作为对照，但当前主要实验未引入。

### 6.4 local_epochs 与 client drift

local_epochs=1 的选择基于以下考虑：
- 在样本量不均匀的 multi-region clients 下，过多 local epochs 会导致 client drift（局部模型偏离全局最优）
- exp1 实验中 local_epochs=1 + rounds=20 已观察到收敛（FedAvg test_rmse 从 94,706 降至 20,753）
- 历史 exp1/2 使用 local_epochs=3 也对齐了相同的收敛趋势

### 6.5 关于 rounds 与收敛

exp1 的 round-level 收敛曲线显示：
- Round 1: test_rmse=94,706
- Round 5: test_rmse=27,396
- Round 10: test_rmse=22,030
- Round 20: test_rmse=20,753

前 10 轮快速下降，后 10 轮边际改善。rounds=20 已达到基本收敛，但在 exp1 中 FedAvg 最终仍弱于 Independent 和 NaiveLastValue，说明不是 rounds 不足，而是单网格 client 组织对 FedAvg 不友好。
