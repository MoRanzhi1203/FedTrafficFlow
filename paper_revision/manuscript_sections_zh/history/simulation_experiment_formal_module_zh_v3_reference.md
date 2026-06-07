# 仿真实验模块正式稿 v3（中文版）

## 1. 仿真实验定位与设计目标

在正式论文结构中，`Synthetic Experiments` 位于 `Data analysis` 章节之下，并先于 `Real-World Data Analysis` 展开。因而，仿真实验的职责应限定为受控条件下的机制性验证：其一，检验标准 FedAvg 联邦训练框架在交通流预测任务中的有效性、收敛性与鲁棒性；其二，为后续真实交通数据分析提供方法学铺垫，而不是提出新的联邦聚合算法。基于这一定位，本文在仿真实验模块中继续坚持“标准 FedAvg 为唯一联邦聚合主线，Independent 为非协同基线”的写作口径。

需要特别说明的是，代码与历史结果目录中仍可见 `Proposed`、`Loss-weighted` 与 `Data-loss weighted` 等探索性策略，但这些内容只保留为历史探索材料，不作为本文主方法，也不进入主文核心结论。v3 的核心更新在于：前期缺失的 CNN 增强实验 FedAvg-only 结果已经补齐，新增 paper-ready 图表也已完成风格同步，因此 Non-IID 分层、客户端数量扩展、特征消融、鲁棒性图表以及 GCN 固定图/动态图对比图均可以在严格的 FedAvg 证据口径下纳入主文讨论。

## 2. 仿真数据构造、客户端划分与 Non-IID 设置

基础 CNN 与基础 GCN 实验采用一致的数据生成框架。代码设定显示，基础场景包含 5 个客户端、8 个交通节点、长度为 24 的输入窗口和 1 步预测步长；每个客户端包含 200 个样本，观测噪声标准差为 0.05，随机种子固定为 42。需要继续强调的是，训练集、验证集和测试集的实际划分比例均为 70%/10%/20%，而非早期材料中曾出现的 70%/15%/15%。

增强 CNN 实验在 `cfe_core.py` 中构造了更复杂的客户端异构环境。五个客户端分别对应 `normal`、`student-t`、`chi-square`、`gaussian_mixture` 与 `log_normal` 扰动机制，样本规模依次为 600、500、700、550 和 450，且噪声水平、基础流量、高峰参数与事件扰动存在差异。因此，增强实验中的 Non-IID 不仅体现为样本量不平衡，也体现为分布族、噪声强度和场景事件的联合异构。这一设定更贴近跨区域交通监测中常见的客户端差异。

基础 GCN 实验显式给出了合成路网结构。`base_graph_summary.csv` 表明，基础图包含 8 个节点、10 条边，图密度为 0.3571，平均度为 2.5。增强 GCN 实验进一步提供了固定图与动态图的统计属性，其中固定图平均权重为 0.2188，而动态图在早高峰、晚高峰和平峰条件下的图密度均提升至 0.875，平均边权约在 0.786 至 0.831 之间。这些设定使 GCN 路径不仅承担空间结构建模的职责，也为后续动态图分析提供了独立证据。

与早期版本不同，v3 已能够对增强 CNN 实验中的低、中、高 Non-IID 分层、不同客户端数量以及特征消融给出 FedAvg-only 定量结果。这意味着增强实验不再只是设计说明，而可以作为标准 FedAvg 在更复杂异质性场景下的补充证据来源。

## 3. FedAvg 联邦训练流程与评价指标

本文仿真实验严格采用标准 FedAvg 聚合。设第 $t$ 轮服务器下发的全局模型参数为 $\mathbf{w}^{t}$，第 $k$ 个客户端本地训练后返回的参数为 $\mathbf{w}_{k}^{t+1}$，其样本量为 $n_k$，则服务器端聚合更新写为：

$$
\mathbf{w}^{t+1}
=
\sum_{k=1}^{K}
\frac{n_k}{\sum_{j=1}^{K} n_j}
\mathbf{w}_{k}^{t+1}
$$

客户端 $k$ 的本地目标函数可写为：

$$
\mathcal{L}_k(\mathbf{w})
=
\frac{1}{n_k}
\sum_{i=1}^{n_k}
\ell(\mathbf{w}; x_i^k, y_i^k)
$$

其本地 SGD 更新形式为：

$$
\mathbf{w}_k
\leftarrow
\mathbf{w}_k
- \eta \nabla \mathcal{L}_k(\mathbf{w}_k)
$$

上述三式定义了全文统一的联邦训练主线。这里必须继续与历史探索材料作出切割：任何基于损失加权、数据损失加权或自定义质量权重的聚合方式都不作为本文方法论的组成部分。论文的贡献在于将标准 FedAvg 框架用于多区域交通流预测，并分析其在 CNN 与 GCN 两类时空模型下的表现，而不在于提出新的联邦聚合算法。

为便于后续转写为 LaTeX 的 `algorithm` 环境，本节保留“算法 1：FedAvg 联邦交通流预测训练流程”如下：

```text
算法 1：FedAvg 联邦交通流预测训练流程

输入：客户端集合 {1,2,...,K}，本地数据集 D_k，通信轮数 R，本地训练轮数 E，学习率 η
输出：通信结束后的全局模型参数 w^R

1. 服务器初始化全局模型参数 w^0
2. for t = 0,1,...,R-1 do
3.     服务器将 w^t 广播至所有参与客户端
4.     for each client k in parallel do
5.         客户端以 w^t 作为本地模型初值
6.         for e = 1,2,...,E do
7.             使用本地数据集 D_k 进行 mini-batch SGD 更新
8.         end for
9.         上传本地更新后的参数 w_k^{t+1}
10.    end for
11.    服务器按样本量执行 FedAvg 聚合
12.    w^{t+1} = Σ_k (n_k / Σ_j n_j) w_k^{t+1}
13. end for
14. 返回 w^R
```

评价指标统一采用 MSE、RMSE、MAE 与 MAPE，用于衡量平方误差、均方根误差、绝对误差以及相对误差水平。v3 中新增的增强实验表格仍然全部沿用这四项指标，从而保证基础实验、增强实验、鲁棒性实验与 GCN 动态图分析之间具有统一的比较标尺。

## 4. 时空预测模型结构

### 4.1 CNN-BiLSTM-Attention

CNN 路径将局部空间依赖表示为规则邻域上的卷积编码，再通过 BiLSTM 建模时间演化，并以 Attention 融合空间与时间表征。若以 $M_{i,t}$ 表示客户端局部空间邻域输入，则卷积式空间编码可写为：

$$
H_{i,t}^{(m)}
=
\sigma \left( W^{(m)} * M_{i,t} + b^{(m)} \right)
$$

这一表述与现有公式说明文档和 CNN 路径实现保持一致，说明在规则节点布局较清晰的合成场景中，卷积编码能够稳定提取局部空间模式。

### 4.2 GCN-BiLSTM-Attention

GCN 路径将交通系统表示为图结构，通过图卷积传播节点间的拓扑依赖。设 $X_t$ 为时刻 $t$ 的节点特征矩阵，$\tilde{A}=A+I$ 为加自环后的邻接矩阵，$\tilde{D}$ 为对应度矩阵，则图卷积可写为：

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

GCN 的价值在于将显式道路连通关系纳入局部模型内部，而不是改变联邦聚合规则。因此，CNN 与 GCN 两条路径在本文中承担的是不同空间建模思路下对标准 FedAvg 的互补验证。

## 5. 基础联邦实验结果

### 5.1 CNN-FedAvg 与 Independent 对比

基础 CNN 实验结果表明，FedAvg 的平均 MSE、RMSE、MAE 和 MAPE 分别为 0.00018285、0.013503、0.010789 和 1.0789%，而 Independent 的对应结果为 0.00024686、0.015326、0.012746 和 1.2746%。这说明在规则化合成交通流场景中，跨客户端参数共享能够稳定降低预测误差。

**表 1 仿真实验设置表**

| 项目 | CNN 基础实验 | GCN 基础实验 | CNN 增强实验 | GCN 增强实验 | 鲁棒性实验 |
|---|---|---|---|---|---|
| 客户端数量 | 5 | 5 | 默认 5，另含 3/5/8 客户端设计 | 5 | 5 |
| 节点数量 | 8 | 8 | 8 | 8 | 继承增强数据 |
| 时间窗口 | 24 | 24 | 12 | 12 | 12 |
| 预测步长 | 1 | 1 | 1 | 1 | 1 |
| 训练/验证/测试 | 70% / 10% / 20% | 70% / 10% / 20% | 70% / 10% / 20% | 继承增强数据划分 | 继承增强数据划分 |
| 每客户端样本 | 200 | 200 | 600/500/700/550/450 | 与增强数据一致 | 与增强数据一致 |
| 通信轮次 | 主训练 10；收敛记录 15 | 10 | 5 | 4 | 3 |
| 本地 epoch | 3 | 3 | 2 | 1 | 1 |
| batch size | 16 | 16 | 32 | 继承增强实现 | 继承增强实现 |
| 随机种子 | 42 | 42 | 42、2024、2025 | 42 | 42、2024、2025 |

**表 2 CNN-FedAvg 与 Independent 指标表**

| 方法 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 | 结果说明 |
|---|---:|---:|---:|---:|---|
| FedAvg | 0.00018285 | 0.013503 | 0.010789 | 1.0789% | 在 5 个客户端中的 4 个上取得更低 MSE |
| Independent | 0.00024686 | 0.015326 | 0.012746 | 1.2746% | 客户端间误差波动更大 |

![图 1 CNN-FedAvg 与 Independent 指标对比](../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf`，后续 LaTeX 排版应优先引用 PDF。

### 5.2 GCN-FedAvg 与 Independent 对比

基础 GCN 实验同样显示出联邦协同训练的稳定收益。FedAvg 的平均 MSE、RMSE、MAE 和 MAPE 分别为 0.00019809、0.014028、0.011207 和 1.1207%，Independent 的对应值分别为 0.00020828、0.014267、0.012116 和 1.2116%。虽然改善幅度略小于 CNN 路径，但整体方向一致，说明联邦协同收益并不依赖某一类局部空间模块。

**表 3 GCN-FedAvg 与 Independent 指标表**

| 方法 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 | 结果说明 |
|---|---:|---:|---:|---:|---|
| FedAvg | 0.00019809 | 0.014028 | 0.011207 | 1.1207% | 四项汇总指标均优于 Independent |
| Independent | 0.00020828 | 0.014267 | 0.012116 | 1.2116% | 跨客户端稳定性较弱 |

![图 2 CNN-FedAvg 收敛曲线](../../results/simulation_experiments/cnn_fed_base/convergence_curve.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_base/convergence_curve.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 3 GCN-FedAvg 收敛曲线](../../results/simulation_experiments/gcn_fed_base/convergence_curve.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_base/convergence_curve.pdf`，后续 LaTeX 排版应优先引用 PDF。

### 5.3 CNN 与 GCN 路径对比

将两条路径置于统一的 FedAvg 框架下可以发现：在基础合成数据的规则节点布局中，CNN-FedAvg 的平均 MSE 略低于 GCN-FedAvg，而 GCN-FedAvg 仍保持对 Independent 的稳定优势。这说明卷积路径在规则场景下具有较强的归纳偏置，而图卷积路径则为后续动态图结构分析提供了更自然的空间建模基础。

## 6. 异质性与客户端扩展实验

### 6.1 增强异质性默认场景

增强异质性数据上的默认总体结果表明，FedAvg 的平均 MSE、RMSE 与 MAE 分别为 69.6236、7.1846 和 5.7256，Independent 的对应结果为 73.2112、7.6420 和 6.0888。这说明在样本量不平衡、噪声差异和分布差异共同存在的条件下，标准 FedAvg 相对独立训练仍保持一定优势。与前一版不同的是，v3 已补齐细粒度 FedAvg-only 结果，因此可以进一步分析 Non-IID 强度、客户端数量与特征配置对性能的影响。

### 6.2 不同 Non-IID 程度下的 FedAvg 表现

新增的 FedAvg-only 结果显示，低、中、高三个 Non-IID 层级下的性能差异较为明显。随着异质性增强，MSE、RMSE、MAE 与 MAPE 整体呈上升趋势，说明客户端分布差异会明显增加标准 FedAvg 的优化难度。

**表 4 不同 Non-IID 程度下 FedAvg 的预测性能**

| Non-IID 程度 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 |
|---|---:|---:|---:|---:|
| Low | 9.9756 | 3.1368 | 2.5167 | 3.3066% |
| Medium | 69.6236 | 7.1846 | 5.7256 | 51.5859% |
| High | 232.0964 | 14.6169 | 11.2763 | 144.8906% |

从趋势上看，低异质性条件下 FedAvg 仍可维持较好的误差控制，而中等异质性条件下 RMSE 已显著上升至 7 以上；当异质性进一步增强至高 Non-IID 时，误差出现更大幅度恶化。这一结果说明，在分布族差异、噪声差异和事件扰动叠加的场景中，标准 FedAvg 仍可工作，但其性能对客户端异构程度较为敏感，高异质性是需要重点面对的挑战。

![图 6 不同 Non-IID 程度下的 FedAvg 结果](../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。

### 6.3 不同客户端数量下的 FedAvg 表现

客户端数量扩展实验进一步考察了 3、5 与 8 个客户端配置下的性能变化。该实验并不用于给出“客户端越多越好”或“客户端越少越好”的绝对规律，而是用于观察数据多样性增加与联邦优化复杂度提升之间可能存在的权衡。

**表 5 不同客户端数量下 FedAvg 的预测性能**

| 客户端数量 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 |
|---|---:|---:|---:|---:|
| 3 | 46.1328 | 6.1716 | 5.0023 | 6.3529% |
| 5 | 69.6236 | 7.1846 | 5.7256 | 51.5859% |
| 8 | 64.3814 | 6.8067 | 5.3965 | 37.4893% |

结果显示，3 个客户端时误差最低，8 个客户端时略优于 5 个客户端，但并未形成单调关系。一个合理的解释是：当客户端数量变化时，数据多样性、局部样本规模与优化协调难度会同步变化；某些设置下，多样性提升带来的收益可能会被额外的异构性和优化不稳定性部分抵消。因此，这一实验更适合被理解为“客户端组织方式会影响 FedAvg 表现”，而不是简单的规模规律。

![图 7 不同客户端数量下的 FedAvg 结果](../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。

### 6.4 FedAvg 框架下的特征消融结果

特征消融实验用于考察不同输入特征组合在同一 FedAvg 框架下的稳定性。需要强调的是，这里比较的是输入特征配置，而不是不同聚合策略。所有数值均来自 FedAvg-only 结果文件，不涉及 `Proposed` 或其他探索性聚合方法。

**表 6 FedAvg 框架下的特征消融结果**

| 特征组合 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 |
|---|---:|---:|---:|---:|
| Flow Only | 69.6236 | 7.1846 | 5.7256 | 51.5859% |
| Flow + Time | 107.3548 | 8.9726 | 7.4027 | 80.1533% |
| Flow + Event | 72.1918 | 7.2675 | 5.8626 | 57.3154% |
| Flow + Region | 72.2081 | 7.3037 | 5.8882 | 59.3951% |
| Full | 104.2930 | 8.8561 | 7.2729 | 79.4819% |

从当前结果看，`Flow Only` 配置取得了最低的整体误差，而 `Flow + Event` 与 `Flow + Region` 的表现与其较为接近；相较之下，`Flow + Time` 和 `Full` 配置的误差更高。这一现象说明，在当前增强异质性设置下，附加特征并不必然转化为更稳定的收益，部分特征组合可能引入了更复杂的跨客户端分布差异。由于不同配置之间仍伴随较大的方差，相关结论应理解为趋势性结果，而不宜被夸大为对所有场景均成立的强规律。

![图 8 FedAvg 框架下的特征消融结果](../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。

## 7. 收敛性分析

基础实验的收敛结果需要区分 CNN 与 GCN 两条路径。CNN 基础实验的主训练轮次为 10，但 `convergence_history.csv` 记录了 15 轮收敛轨迹；GCN 基础实验的训练与收敛记录均为 10 轮。CNN 的平均验证 RMSE 从第 1 轮的 0.09474 快速下降至第 10 轮的 0.01234，并在第 12 至 15 轮维持在约 0.0117 至 0.0124 区间；GCN 的平均验证 RMSE 则从第 1 轮的 0.02676 下降到第 10 轮的 0.01385。

这说明标准 FedAvg 在合成交通流场景中能够在有限通信轮次内形成稳定的全局模型，而且这种收敛稳定性在 CNN 与 GCN 两类空间模块下均可观察到。因此，收敛收益主要来自统一的联邦训练协议，而不是某一特定局部模型的偶然表现。

## 8. 鲁棒性实验

鲁棒性实验涵盖客户端掉线、通信延迟和梯度噪声三类扰动。此处的梯度噪声必须继续理解为模拟梯度扰动，用于分析联邦训练对随机参数扰动的敏感性，而不构成正式差分隐私机制。v3 的关键变化在于，三张鲁棒性 paper-ready 图均已完成 FedAvg-only 重绘和风格同步，因此可以作为主文图件引用。

**表 7 鲁棒性实验结果表（FedAvg 行）**

| 扰动类型 | 扰动水平 | RMSE | MAE | MAPE | 结果说明 |
|---|---:|---:|---:|---:|---|
| 客户端掉线 | 0.0 | 7.9281 | 6.3987 | 66.0344% | 作为无掉线基准 |
| 客户端掉线 | 0.2 | 7.8411 | 6.3299 | 71.2527% | RMSE 变化较小 |
| 客户端掉线 | 0.4 | 7.9762 | 6.5025 | 73.6387% | 整体仍保持稳定 |
| 通信延迟 | 0 | 7.9281 | 6.3987 | 66.0344% | 作为无延迟基准 |
| 通信延迟 | 1 | 8.2686 | 6.7544 | 70.6013% | 性能劣化最明显 |
| 通信延迟 | 2 | 8.0433 | 6.5395 | 70.5930% | 仍高于无延迟条件 |
| 梯度噪声扰动 | 0.00 | 7.9281 | 6.3987 | 66.0344% | 作为无噪声基准 |
| 梯度噪声扰动 | 0.02 | 7.9154 | 6.4166 | 64.0647% | 轻度扰动影响有限 |
| 梯度噪声扰动 | 0.05 | 8.1910 | 6.6506 | 67.9804% | 强扰动造成可见劣化 |

从结果趋势看，客户端掉线对整体误差的影响相对有限，通信延迟带来的性能劣化更为明显，而梯度噪声在低强度下影响较小、在更高强度下才表现出更清晰的误差上升。这说明标准 FedAvg 对参与波动和轻度随机扰动具有一定容错能力，但对时延问题更为敏感。

![图 9 FedAvg 鲁棒性：客户端掉线](../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 10 FedAvg 鲁棒性：通信延迟](../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 11 FedAvg 鲁棒性：模拟梯度扰动](../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。

## 9. GCN 图结构与动态图分析

基础 GCN 实验已经表明，在显式图结构输入下，FedAvg 同样能够获得稳定收益。增强 GCN 实验进一步比较了固定图与动态图在标准 FedAvg 框架下的差异。根据结果文件，固定图条件下平均 MSE 为 49.6998，而早高峰、晚高峰和平峰动态图条件下的平均 MSE 分别为 49.1631、49.1752 和 49.1648，均略优于固定图。

**表 8 GCN 固定图与动态图结果表（FedAvg 行）**

| 图类型 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 | 说明 |
|---|---:|---:|---:|---:|---|
| Fixed | 49.6998 | 6.2163 | 4.8295 | 23.8204% | 固定邻接基准 |
| Dynamic-Morning | 49.1631 | 6.1907 | 4.8069 | 23.7915% | 略优于固定图 |
| Dynamic-Evening | 49.1752 | 6.1912 | 4.8071 | 23.7922% | 略优于固定图 |
| Dynamic-Offpeak | 49.1648 | 6.1906 | 4.8067 | 23.7922% | 略优于固定图 |

不过，这一部分仍需保持审慎。增强 GCN 实验当前仍基于单一随机种子，因此固定图与动态图之间的差异只能被理解为单种子趋势性证据，而不能被写成强统计结论。其意义主要在于表明：时段相关图结构具有潜在价值，值得在未来通过多种子或更大规模实验继续检验。

![图 4 基础 GCN 邻接矩阵示意](../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 5 增强 GCN 固定邻接矩阵示意](../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 12 FedAvg 下固定图与动态图结构对比](../../results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf`，后续 LaTeX 排版应优先引用 PDF。该结果仅作为单种子趋势性证据，不宜扩展为强统计结论。

## 10. 图表、表格与伪代码安排

为便于后续转写为 LaTeX，v3 建议保留“5 张基础与结构图 + 7 张新增 FedAvg-only 图 + 8 张结果表 + 1 个 FedAvg 伪代码”的组合形式。与前一版相比，v3 的变化主要体现在两点：第一，先前存在证据缺口的 Non-IID 分层、客户端数量扩展与特征消融结果已经补齐 FedAvg-only CSV，因此可以转入正文；第二，先前需重绘的鲁棒性图与 GCN 固定图/动态图图件已具备风格统一的 paper-ready PNG/PDF 成对资产，可进入主文。

与此同时，旧版混入 `Proposed`、`Loss-weighted` 或 `Data-loss weighted` 的图件仍不应直接进入正文。Markdown 正文继续统一引用 PNG，且每张图后都应显式说明同名 PDF 路径，以保证后续 LaTeX 排版阶段的可追踪性和资产一致性。

## 11. 小结

综上，v3 已在不改变 FedAvg 主线、不修改 LaTeX、不重跑实验的前提下，将仿真实验模块推进到更完整的主文证据状态。其核心结论可概括为以下五点。第一，基础 CNN 与 GCN 实验继续证明，标准 FedAvg 在两类时空预测路径下均优于 Independent，说明联邦协同训练能够稳定提升基础合成交通流预测性能。第二，增强异质性默认场景下，FedAvg 相对 Independent 仍保持一定优势；更重要的是，Non-IID 分层、客户端数量扩展和特征消融三类细粒度结果现已补齐 FedAvg-only 证据，可以作为主文的补充定量支持。第三，鲁棒性 FedAvg-only 图表已补齐并可进入主文，结果表明客户端掉线影响相对较小、通信延迟更为敏感，而梯度噪声只能被理解为模拟梯度扰动，不构成正式差分隐私机制。第四，GCN 固定图与动态图的 FedAvg-only 图表也已补齐，显示动态图条件下存在轻微误差改善趋势，但由于仍为单种子结果，只能写成趋势性证据。第五，v3 已形成较完整的 PNG/PDF 成对图表资产，可作为后续“中文 v3 转写为英文 LaTeX 草稿”的直接基础。
