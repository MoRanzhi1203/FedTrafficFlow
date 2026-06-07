# 仿真实验模块正式稿（中文版）

## 1. 仿真实验设计与目的

为与正式论文 `Data analysis` 章节的整体论证保持一致，仿真实验部分定位为受控验证模块，其目的不是提出新的联邦聚合算法，而是在可控数据生成条件下系统检验标准 FedAvg 联邦学习框架在交通流预测任务中的有效性、收敛性、异质性适应能力、鲁棒性以及图结构建模价值。基于这一定位，本节统一采用标准 FedAvg 作为联邦聚合机制，并以 `Independent` 作为无联邦协同的对比基线。CNN 与 GCN 在本节中均被视为空间建模路径，而非聚合算法本身；BiLSTM 负责时间依赖建模，Attention 用于空间表征与时间表征的融合。

与原论文中偏重基础结果展示的 `Synthetic Experiments` 相比，本正式稿进一步补足了联邦训练流程、公式定义、指标说明、非 IID 与客户端异质性分析、鲁棒性分析以及图结构解释，使仿真实验能够更完整地支撑后续 `Real-World Data Analysis` 的实证结论。该写法同时对应了一审意见中关于实验充分性、图表解释、方法验证与流程说明不足等问题，但正文保持论文自然叙述，不采用回复信口吻。

## 2. 仿真数据构造与客户端划分

基础仿真实验采用统一的数据生成管线。根据 `cfb_core.py` 与 `gfb_core.py` 的设定，基础场景包含 5 个客户端、8 个交通节点、长度为 24 的时间窗口和 1 步预测步长；每个客户端包含 200 个样本，观测噪声标准差为 0.05，训练集、验证集和测试集按照 70%:10%:20% 的比例划分，随机种子固定为 42。该设置提供了一个结构均衡、便于比较联邦协同收益的基准场景。

增强异质性实验在 `cfe_core.py` 中构造了更复杂的客户端分布。5 个客户端分别采用 `normal`、`student-t`、`chi-square`、`gaussian_mixture` 与 `log_normal` 等不同扰动机制，样本规模分别为 600、500、700、550 和 450，噪声水平和基础流量幅值亦不相同；其中最后一个客户端额外引入事件冲击，从而形成更强的样本量异质性、噪声异质性与分布异质性。这一设定更接近联邦交通流预测在跨区域部署中的非均匀数据条件。

GCN 路径在基础场景中显式构造了路网拓扑。`gcn_fed_base/base_graph_summary.csv` 给出的固定图包含 8 个节点、10 条边，图密度为 0.3571，平均度为 2.5。进一步地，`enhanced_gcn_graph_summary.csv` 在增强实验中给出了固定图与动态图的统计特征：固定图平均权重为 0.2188，动态图在早高峰、晚高峰和平峰条件下的图密度均达到 0.875，平均边权约为 0.786 至 0.831。因而，GCN 路径不仅能够验证 FedAvg 在图结构输入上的适用性，还为后续讨论动态图结构提供了实验基础。

## 3. 联邦训练流程与 FedAvg 聚合公式

本节采用标准 FedAvg 联邦训练协议。服务器在每一轮通信开始时向各客户端广播全局模型参数，各客户端基于本地数据完成若干轮 mini-batch SGD 更新后上传本地模型，服务器再按照客户端样本量加权平均得到下一轮全局模型。其聚合公式写为：

$$
\mathbf{w}^{t+1}
=
\sum_{k=1}^{K}
\frac{n_k}{\sum_{j=1}^{K} n_j}
\mathbf{w}_{k}^{t+1}
$$

其中，$\mathbf{w}^{t+1}$ 表示第 $t+1$ 轮全局模型参数，$\mathbf{w}_{k}^{t+1}$ 表示第 $k$ 个客户端完成本地训练后的模型参数，$K$ 表示客户端数量，$n_k$ 表示第 $k$ 个客户端的样本量。该公式明确表明，本文联邦聚合遵循标准 FedAvg，而非基于损失、相似度或其他启发式规则构造的新聚合机制。

客户端侧的本地经验风险写为：

$$
\mathcal{L}_k(\mathbf{w})
=
\frac{1}{n_k}
\sum_{i=1}^{n_k}
\ell(\mathbf{w}; x_i^k, y_i^k)
$$

其本地 SGD 更新写为：

$$
\mathbf{w}_k
\leftarrow
\mathbf{w}_k
-
\eta \nabla \mathcal{L}_k(\mathbf{w}_k)
$$

上述三式分别对应全局聚合、客户端局部目标和局部优化步骤，构成本文仿真实验的统一训练主线。后文中的 CNN-BiLSTM-Attention 与 GCN-BiLSTM-Attention 仅在局部模型结构上不同，而联邦训练流程保持一致。

**算法 1：FedAvg 联邦交通流预测训练流程**

```text
Input: 客户端集合 {1,...,K}，本地数据集 D_k，通信轮数 R，本地训练轮数 E，学习率 η
Output: 全局模型参数 w^R

1. 服务器初始化全局模型参数 w^0
2. for t = 0,1,...,R-1 do
3.     服务器将 w^t 下发至所有参与客户端
4.     for each client k in parallel do
5.         w_k^t ← w^t
6.         for e = 1,2,...,E do
7.             使用 D_k 对本地模型进行 mini-batch SGD 更新
8.         end for
9.         客户端上传 w_k^{t+1}
10.    end for
11.    服务器执行 FedAvg 聚合：
          w^{t+1} = Σ_k (n_k / Σ_j n_j) w_k^{t+1}
12. end for
13. return w^R
```

## 4. 模型结构与空间建模路径

### 4.1 CNN-BiLSTM-Attention 路径

CNN-BiLSTM-Attention 路径适用于规则节点排列或局部邻域结构较清晰的仿真交通流场景。其空间建模思想是在每个时间窗口内利用卷积算子提取局部节点模式，再借助 BiLSTM 建模双向时间依赖，并通过 Attention 实现空间特征与时间特征的自适应融合。若以局部空间输入矩阵 $M_{i,t}$ 表示客户端 $i$ 在时刻 $t$ 的局部邻域观测，则卷积型空间编码可表述为：

$$
H_{i,t}^{(m)}
=
\sigma \left( W^{(m)} * M_{i,t} + b^{(m)} \right)
$$

该路径的意义不在于改变联邦聚合，而在于为 FedAvg 提供一种局部空间特征提取方式。当交通节点可以近似为规则网格或邻域关系主要体现为局部空间相关时，卷积型编码具有较强的归纳偏置。

### 4.2 GCN-BiLSTM-Attention 路径

GCN-BiLSTM-Attention 路径将交通网络表示为图结构，并通过图卷积显式建模拓扑依赖。若 $X_t$ 表示时刻 $t$ 的节点特征矩阵，$\tilde{A}=A+I$ 表示加自环后的邻接矩阵，$\tilde{D}$ 表示其度矩阵，则图卷积可写为：

$$
H_t
=
\sigma
\left(
\tilde{D}^{-\frac{1}{2}}
\tilde{A}
\tilde{D}^{-\frac{1}{2}}
X_t
W
\right)
$$

该公式与现有公式说明文件及 `gfb_core.py`、`gfe_core.py` 中的图传播实现相一致，可用于后续 LaTeX 论文的正式转写。与 CNN 路径相比，GCN 路径并不假设规则网格，而是通过显式邻接结构传播空间信息，因此更适合存在明确路网连通关系的交通场景。BiLSTM 和 Attention 在该路径中分别负责时间依赖建模和融合图结构表示与时序表示。

## 5. 对比方法、实验设置与评价指标

本文仿真实验只保留两类与主文结论直接相关的方法：标准 `FedAvg` 和对比基线 `Independent`。代码与结果目录中出现的 `Proposed`、`Loss-weighted` 与 `Data-loss weighted` 均属于历史探索性聚合策略，不纳入主文方法论，也不作为本节核心结论依据。

**表 1 仿真实验设置表**

| 项目 | 基础联邦实验 | 增强异质性实验 | 鲁棒性实验 | 来源 |
|---|---|---|---|---|
| 客户端数量 | 5 | 5；另设计有 3/5/8 客户端规模实验 | 5 | `cfb_core.py`，`gfb_core.py`，`cfe_core.py`，`fr_core.py` |
| 节点数量 | 8 | 8 | 8 | 同上 |
| 每客户端样本量 | 200, 200, 200, 200, 200 | 600, 500, 700, 550, 450 | 继承增强异质性数据构造 | `cfb_core.py`，`cfe_core.py` |
| 时间窗口 | 24 | 12 | 12 | `cfb_core.py`，`cfe_core.py` |
| 预测步长 | 1 | 1 | 1 | 同上 |
| 通信轮次 | CNN/GCN 基础实验均为 10 | CNN 增强为 5；GCN 增强为 4 | 3 | `cfb_core.py`，`gfb_core.py`，`cfe_core.py`，`gfe_core.py`，`fr_core.py` |
| 本地 epoch | 3 | CNN 增强为 2；GCN 增强为 1 | 1 | 同上 |
| batch size | 16 | 32 | 32 | `cfb_core.py`，`cfe_core.py` |
| hidden dim | 64 | 64 | 64 | `cfb_core.py`，`gfb_core.py`，`cfe_core.py`，`gfe_core.py` |
| 训练/验证/测试比例 | 70% / 10% / 20% | 70% / 10% / 20% | 继承增强异质性划分 | `cfb_core.py`，`cfe_core.py` |
| 噪声强度 | 观测噪声标准差 0.05 | 客户端噪声配置为 2.0, 5.0, 8.0, 4.0, 6.0，且含事件冲击 | 扰动噪声标准差 0.00, 0.02, 0.05 | `cfb_core.py`，`cfe_core.py`，`fr_core.py` |
| 随机种子 | 42 | CNN 增强为 42, 2024, 2025；GCN 增强为 42 | 42, 2024, 2025 | `cfb_core.py`，`cfe_core.py`，`gfe_core.py`，`fr_core.py` |

评价指标采用 MSE、RMSE、MAE 与 MAPE，其定义分别为：

$$
\mathrm{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
$$

$$
\mathrm{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
$$

$$
\mathrm{MAPE}
=
\frac{100\%}{N}
\sum_{i=1}^{N}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
$$

这四项指标共同用于评价平均误差水平、平方惩罚下的误差幅度、绝对偏差以及相对误差表现，从而形成对联邦交通流预测性能的多维度衡量。

## 6. 基础联邦实验结果

### 6.1 CNN-FedAvg 与 Independent 对比

基础 CNN 实验表明，在规则化仿真场景下，标准 FedAvg 能够稳定降低预测误差。`cnn_fed_base/main_summary.csv` 显示，FedAvg 的平均 MSE 为 $1.8285\times 10^{-4}$，平均 RMSE 为 $1.3503\times 10^{-2}$，平均 MAE 为 $1.0789\times 10^{-2}$，平均 MAPE 为 1.0789%；Independent 的对应结果分别为 $2.4686\times 10^{-4}$、$1.5326\times 10^{-2}$、$1.2746\times 10^{-2}$ 和 1.2746%。这说明在相同模型结构下，跨客户端参数共享能够有效降低平均误差。

**表 2 CNN-FedAvg 与 Independent 指标表**

| 方法 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 | 逐客户端观察 |
|---|---:|---:|---:|---:|---|
| FedAvg | 0.00018285 | 0.01350306 | 0.01078908 | 1.0789% | 在 5 个客户端中的 4 个上取得更低 MSE |
| Independent | 0.00024686 | 0.01532609 | 0.01274630 | 1.2746% | 客户端间波动更大，客户端 3 误差显著偏高 |

从 `cnn_fed_base/main_metrics.csv` 的逐客户端结果可见，FedAvg 在客户端 0、1、3 和 4 上均优于 Independent，其中客户端 3 的 MSE 从 0.00048777 降至 0.00019087，下降最为明显。该结果说明，即使在基础仿真环境下，不同客户端之间仍存在足以影响局部优化方向的差异，而 FedAvg 的参数共享有助于抑制局部过拟合和客户端性能失衡。

### 6.2 GCN-FedAvg 与 Independent 对比

在 GCN 空间建模路径下，FedAvg 同样优于 Independent。`gcn_fed_base/main_summary.csv` 显示，FedAvg 的平均 MSE、RMSE、MAE 和 MAPE 分别为 0.00019809、0.01402799、0.01120720 和 1.1207%；Independent 的对应结果分别为 0.00020828、0.01426744、0.01211595 和 1.2116%。虽然绝对误差改善幅度小于 CNN 路径，但联邦协同训练的收益仍然一致存在。

**表 3 GCN-FedAvg 与 Independent 指标表**

| 方法 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 |
|---|---:|---:|---:|---:|
| FedAvg | 0.00019809 | 0.01402799 | 0.01120720 | 1.1207% |
| Independent | 0.00020828 | 0.01426744 | 0.01211595 | 1.2116% |

这一结果说明，FedAvg 的有效性并不依赖于特定的空间特征提取器。当局部模型从卷积路径切换到图卷积路径后，联邦参数聚合仍然能够带来误差降低，表明本文主线应被理解为“标准联邦训练框架适配不同的时空预测模型”，而不是“某一空间模块本身取代了联邦聚合”。

### 6.3 CNN 与 GCN 空间路径对比

将两条空间建模路径置于统一 FedAvg 框架下比较可以发现：在基础合成数据的规则节点布局中，CNN-FedAvg 的平均 MSE 略低于 GCN-FedAvg，而 GCN-FedAvg 仍然保持对 Independent 的稳定优势。这一现象表明，在规则化仿真条件下，卷积型局部空间编码具有较强适配性；然而 GCN 路径的价值在于其能够显式利用路网拓扑，为后续动态图结构分析和复杂网络场景扩展提供建模基础。因此，本节对 CNN 与 GCN 的比较不应被理解为“二者谁替代谁”，而应被理解为联邦交通流预测框架对两类空间表征机制的兼容性验证。

## 7. Non-IID 与客户端异质性实验

增强异质性数据集通过样本量不平衡、噪声水平差异、分布族差异及事件冲击共同构造更接近实际联邦场景的非 IID 条件。`cnn_enhanced_main_summary.csv` 显示，在该更复杂的数据环境下，FedAvg 的平均 MSE 为 69.6236，平均 RMSE 为 7.1846，平均 MAE 为 5.7256；Independent 的对应结果为 73.2112、7.6420 和 6.0888。尽管误差量级明显高于基础实验，但 FedAvg 仍然优于 Independent，说明标准联邦平均在中等异质性场景下仍具有一定适应能力。

需要指出的是，用户要求主文的 Non-IID 分层结果只保留 FedAvg 行。然而当前 `cnn_enhanced_noniid_summary.csv` 与 `cnn_enhanced_client_scale_summary.csv` 在现有结果目录中仅保留 `Proposed` 行，缺少可以直接写入主文的 FedAvg 分层统计。因此，在不重跑实验且不编造数值的前提下，本节只保留异质性设计说明和 `cnn_enhanced_main_summary.csv` 中可核验的 FedAvg 结果，不将探索性聚合结果写入主文结论。

**表 4 Non-IID 实验结果表（FedAvg 主文口径核验）**

| 结果文件 | 预期内容 | 当前是否存在 FedAvg 行 | 正文处理 |
|---|---|---|---|
| `cnn_enhanced_noniid_summary.csv` | 低/中/高非 IID 分层比较 | 否 | 不直接写入定量主结论 |
| `cnn_enhanced_client_scale_summary.csv` | 3/5/8 客户端规模比较 | 否 | 仅保留实验设计，不给出主文统计表 |
| `cnn_enhanced_main_summary.csv` | 默认增强异质性场景总体对比 | 是 | 可作为异质性适应能力的定量证据 |

因此，就正式论文正文而言，可以稳妥得出的结论是：当客户端间样本规模、噪声强度与分布模式发生变化时，标准 FedAvg 仍优于 Independent，但关于异质性强度分层和客户端规模变化的更细粒度量化结论，尚需补充与 FedAvg 主线一致的结果文件后方可纳入主文。

## 8. 收敛性分析

`cnn_fed_base/convergence_history.csv` 与 `gcn_fed_base/convergence_history.csv` 共同表明，FedAvg 在仿真交通流预测任务中具有清晰的收敛轨迹。对于 CNN 路径，平均验证 RMSE 从第 1 轮的 0.09474 快速下降至第 10 轮的 0.01234，并在第 12 至 15 轮附近稳定在约 0.0117 至 0.0124 区间；训练损失则从第 1 轮的 0.00113 下降到第 15 轮的 0.000070。对于 GCN 路径，平均验证 RMSE 从第 1 轮的 0.02676 下降至第 10 轮的 0.01385，训练损失同步从 0.000721 降至 0.000195。

这两条收敛曲线具有共同特征：前几轮通信内误差迅速下降，随后进入平稳区间，未观察到明显震荡或发散。这说明在本文的合成交通流条件下，FedAvg 能够在较少通信轮次内形成稳定的全局模型，为真实数据实验中的联邦训练设置提供经验支持。进一步地，CNN 与 GCN 路径在收敛形态上的一致性，也再次说明联邦训练稳定性主要来自统一的 FedAvg 协议，而非某一特定空间模块的偶然行为。

## 9. 鲁棒性实验

鲁棒性实验采用 `fr_core.py` 中定义的三类系统扰动，分别为客户端掉线、通信延迟和梯度噪声扰动。需要特别说明的是，所谓梯度噪声仅指 simulated gradient perturbation，用于考察聚合过程对随机扰动的敏感性，不能被表述为正式差分隐私机制。

**表 5 鲁棒性实验结果表（仅保留 FedAvg 行）**

| 场景 | 扰动水平 | RMSE | MAE | MAPE |
|---|---:|---:|---:|---:|
| 客户端掉线 | 0.0 | 7.9281 | 6.3987 | 66.0344% |
| 客户端掉线 | 0.2 | 7.8411 | 6.3299 | 71.2527% |
| 客户端掉线 | 0.4 | 7.9762 | 6.5025 | 73.6387% |
| 通信延迟 | 0 | 7.9281 | 6.3987 | 66.0344% |
| 通信延迟 | 1 | 8.2686 | 6.7544 | 70.6013% |
| 通信延迟 | 2 | 8.0433 | 6.5395 | 70.5930% |
| 梯度噪声扰动 | 0.00 | 7.9281 | 6.3987 | 66.0344% |
| 梯度噪声扰动 | 0.02 | 7.9154 | 6.4166 | 64.0647% |
| 梯度噪声扰动 | 0.05 | 8.1910 | 6.6506 | 67.9804% |

### 9.1 客户端掉线

当掉线率从 0 提高到 0.4 时，FedAvg 的 RMSE 仅由 7.9281 变化至 7.9762，说明在部分客户端缺席的情况下，样本量加权聚合仍能维持整体性能稳定。这一结果表明，联邦交通流预测对适度参与波动具有一定容错能力。

### 9.2 通信延迟

通信延迟对性能的影响更为明显。1 轮延迟时 RMSE 上升到 8.2686，MAE 上升到 6.7544，说明陈旧参数会削弱服务器端聚合对当前局部优化方向的跟踪能力。尽管 2 轮延迟下的 RMSE 略低于 1 轮延迟，但总体上延迟仍带来误差劣化，因此更合理的解释是延迟环境下模型表现波动增大，而不是延迟本身具有收益。

### 9.3 梯度噪声扰动

在噪声标准差为 0.02 的轻度扰动下，FedAvg 的 RMSE 基本保持稳定；当噪声标准差提高到 0.05 时，RMSE 增加到 8.1910。该结果表明，加权平均聚合对小幅随机扰动具有一定平滑作用，但较强扰动仍会削弱预测精度。这里的噪声机制仅用于鲁棒性仿真，不应被写成正式隐私保护方案。

## 10. GCN 图结构与动态邻接分析

基础 GCN 实验已验证显式图结构输入在联邦场景中的可行性，增强 GCN 实验则进一步比较固定邻接矩阵与动态图邻接矩阵。`gcn_enhanced_fixed_vs_dynamic_summary.csv` 中，FedAvg 在固定图上的平均 MSE 为 49.6998，而在早高峰、晚高峰和平峰动态图上的平均 MSE 分别为 49.1631、49.1752 和 49.1648。三种动态图结果均略优于固定图，说明时段相关的邻接关系有助于改善图结构表示与交通状态之间的匹配程度。

与此同时，`enhanced_gcn_graph_summary.csv` 显示动态图的图密度和平均边权均显著高于基础固定图，这意味着动态图不仅改变了连接强度，也改变了信息传播范围。就论文论证而言，这一部分的价值主要在于说明图结构建模能够从“固定拓扑编码”进一步拓展到“时变拓扑编码”，从而为交通网络的动态关联建模提供可能。

但必须强调，`gfe_core.py` 中增强 GCN 实验当前只采用单一随机种子 `SEEDS = [42]`。因此，固定图与动态图之间的差异只能作为趋势性证据，用于说明图结构建模方向的可行性，而不宜写成强统计结论。

## 11. 特征消融实验

特征消融实验原本用于比较不同输入特征组合对预测精度的影响。然而按照当前用户要求，主文只能保留 FedAvg 行；而现有 `cnn_enhanced_feature_ablation_summary.csv` 仅包含 `Proposed` 行，不包含可直接引用的 FedAvg 消融结果。因此，在不重跑实验的前提下，本节不写入任何 FedAvg 数值结论，而只保留证据状态说明和后续资产安排。

**表 6 特征消融结果表（FedAvg 主文口径核验）**

| 结果文件 | 预期特征组合 | 当前是否存在 FedAvg 行 | 正文处理 |
|---|---|---|---|
| `cnn_enhanced_feature_ablation_summary.csv` | `flow_only`、`flow_region`、`flow_event`、`flow_time`、`full` | 否 | 不直接写入定量主结论 |

换言之，特征消融实验的实验设计与可视化素材已经存在，但就正式论文主文而言，仍需补充与 FedAvg 主线一致的消融统计后，方可将其写成正式定量结论。当前版本中，本节仅将其作为图表与素材规划的一部分。

## 12. 图表、表格与伪代码安排

为便于后续转写为 LaTeX 论文正文，本节建议将图表与算法安排为“基础联邦结果图 + 收敛曲线 + 鲁棒性图 + 图结构图”的组合形式。其中，FedAvg 与 Independent 的基础对比图可直接采用现有结果目录中的图像；含有 `Proposed` 或其他探索性聚合策略的增强图应重绘为 FedAvg-only 的 paper-ready 版本后再进入主文。

建议直接引用或占位如下：

![图 1 CNN-FedAvg 与 Independent 指标对比](../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png)

![图 2 CNN-FedAvg 收敛曲线](../../results/simulation_experiments/cnn_fed_base/convergence_curve.png)

![图 3 GCN-FedAvg 收敛曲线](../../results/simulation_experiments/gcn_fed_base/convergence_curve.png)

![图 4 基础 GCN 邻接矩阵示意](../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png)

![图 5 GCN 固定邻接矩阵示意](../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png)

【图 6 待制作：基于 `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv` 生成 paper-ready FedAvg-only 非 IID 等级对比图；当前现有 `cnn_enhanced_noniid.pdf` 含探索性聚合结果，不宜直接用于主文。】

【图 7 待制作：基于 `results/simulation_experiments/fed_robustness/fed_client_dropout_summary.csv`、`fed_communication_delay_summary.csv` 与 `fed_gradient_noise_summary.csv` 生成 paper-ready FedAvg-only 鲁棒性对比图。】

【图 8 待制作：基于 `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv` 重绘 paper-ready FedAvg-only 特征消融图；当前结果文件缺少 FedAvg 行，需作者确认是否补充证据后再入正文。】

【图 9 待制作：基于 `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_summary.csv` 与 `enhanced_gcn_graph_summary.csv` 绘制固定图与动态图对比图；若保留现有图件，需检查是否包含 `Proposed` 曲线并剔除。】

表格方面，表 1 至表 6 可分别对应实验设置、基础 CNN 结果、基础 GCN 结果、Non-IID 证据核验、鲁棒性结果和特征消融证据核验。算法 1 建议放置在实验设置与模型结构说明之后，以便图表与数值分析均建立在统一的联邦训练定义之上。

## 13. 仿真实验小结

综合上述结果，可以得到以下结论。第一，在标准 FedAvg 主线下，CNN-BiLSTM-Attention 与 GCN-BiLSTM-Attention 两类时空预测模型均能通过跨客户端参数共享降低预测误差，说明联邦交通流预测框架在不同空间建模路径上均具有可行性。第二，基础实验中的收敛曲线表明，FedAvg 在仿真交通流场景下能够在有限通信轮次内稳定收敛。第三，增强异质性数据集的总体结果显示，FedAvg 在样本量不平衡、噪声差异和分布差异共存的条件下仍优于 Independent，体现出一定的异质性适应能力。第四，鲁棒性实验表明客户端掉线、通信延迟和梯度噪声扰动均会影响预测性能，其中通信延迟更值得在实际部署中重点关注。第五，GCN 动态图实验说明显式图结构及其时变扩展具有方法学价值，但由于增强 GCN 当前为单种子结果，其结论应保持趋势性表述。

需要同时指出，当前 Non-IID 分层表与特征消融表在结果文件层面尚不满足 FedAvg-only 主文证据口径。因此，本正式稿已经将其纳入章节结构和图表规划，但没有将探索性聚合数值写成论文主结论。后续若作者补齐与 FedAvg 主线一致的结果文件，本模块即可进一步扩展为完整的 LaTeX 正文。
