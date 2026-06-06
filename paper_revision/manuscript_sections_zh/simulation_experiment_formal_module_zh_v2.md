# 仿真实验模块正式稿 v2（中文版）

## 1. 仿真实验定位与设计目标

在正式论文结构中，`Synthetic Experiments` 位于 `Data analysis` 章节之下，并先于 `Real-World Data Analysis` 展开。因此，仿真实验的功能应被限定为受控条件下的机制性验证：一方面检验标准 FedAvg 联邦训练框架在交通流预测任务中的有效性、收敛性与鲁棒性，另一方面为后续真实交通数据实验提供方法学铺垫，而不是引入新的联邦聚合算法。结合当前正式论文 `main.tex` 的章节安排以及现有代码与结果文件，本模块在 v2 中继续保留“仿真先验证、真实数据再外推”的逻辑顺序，并主动与 `Real-World Data Analysis` 的叙述风格保持一致。

基于这一定位，v2 全文只保留两类与主文直接相关的方法角色：标准 `FedAvg` 作为联邦训练主线，`Independent` 作为无联邦协同的对比基线。代码和结果目录中仍可见 `Proposed`、`Loss-weighted` 与 `Data-loss weighted` 等历史探索策略，但它们在本版中不再承担主结果证据功能，只在证据核验和资产计划中被识别、隔离并标注为不进入主文的材料。

## 2. 仿真数据构造、客户端划分与 Non-IID 设置

基础仿真实验在 `cfb_core.py` 与 `gfb_core.py` 中采用一致的数据生成框架。代码显示，基础场景包含 5 个客户端、8 个交通节点、长度为 24 的输入时间窗口以及 1 步预测步长；每个客户端包含 200 个样本，观测噪声标准差为 0.05，随机种子固定为 42。需要特别修正的是，训练集、验证集和测试集的实际划分比例均为 70%/10%/20%，而不是旧稿中曾出现的 70%/15%/15%。这一比例在 `cfb_core.py` 的 `BASE_TRAIN_RATIO = 0.70`、`BASE_VAL_RATIO = 0.10`、`BASE_TEST_RATIO = 0.20` 以及 `gfb_core.py` 中的相同设定均可直接核验。

增强异质性实验在 `cfe_core.py` 中构造了更复杂的客户端分布。五个客户端分别对应 `normal`、`student-t`、`chi-square`、`gaussian_mixture` 与 `log_normal` 扰动机制，样本规模分别为 600、500、700、550 和 450，噪声水平、基础流量和高峰参数亦不相同，且最后一个客户端引入了突发事件冲击。由此，增强实验中的异质性并非只体现为样本量不平衡，还同时包含分布族差异、噪声差异和事件扰动差异。该设计与真实交通场景中跨区域部署时常见的客户端异构现象具有较高对应性。

对于 GCN 路径，基础实验显式给出了合成路网结构。`gcn_fed_base/base_graph_summary.csv` 表明，基础图包含 8 个节点、10 条边，图密度为 0.3571，平均度为 2.5。增强 GCN 实验则进一步在 `enhanced_gcn_graph_summary.csv` 中给出了固定图与动态图的统计属性，其中固定图平均权重为 0.2188，而按时段构造的动态图在早高峰、晚高峰和平峰条件下的密度均为 0.875，平均边权约在 0.786 至 0.831 之间。这些结果说明，GCN 路径不仅承担“显式图结构建模”的对比职责，也为后续动态图分析提供了独立证据基础。

需要说明的是，增强实验虽然设计了低、中、高 Non-IID 分层、不同客户端数量和特征消融等子实验，但这些子实验是否满足 FedAvg 主文证据口径，必须以实际 CSV 为准。v2 的处理原则是：保留实验设计与问题设置，但只有当结果文件中能够明确核验到 FedAvg 行时，相关数值才进入主文结论。

## 3. FedAvg 联邦训练流程与评价指标

本模块的联邦训练协议严格采用标准 FedAvg。设第 $t$ 轮服务器下发的全局模型参数为 $\mathbf{w}^{t}$，第 $k$ 个客户端本地训练后返回的参数为 $\mathbf{w}_{k}^{t+1}$，其样本量为 $n_k$，则服务器端聚合更新写为：

$$
\mathbf{w}^{t+1}
=
\sum_{k=1}^{K}
\frac{n_k}{\sum_{j=1}^{K} n_j}
\mathbf{w}_{k}^{t+1}
$$

与之对应，客户端 $k$ 的本地目标函数写为：

$$
\mathcal{L}_k(\mathbf{w})
=
\frac{1}{n_k}
\sum_{i=1}^{n_k}
\ell(\mathbf{w}; x_i^k, y_i^k)
$$

本地参数更新采用 SGD 形式表示为：

$$
\mathbf{w}_k
\leftarrow
\mathbf{w}_k
-
\eta \nabla \mathcal{L}_k(\mathbf{w}_k)
$$

这三式共同定义了本文仿真实验中的统一训练主线。需要与正式论文现有英文稿作出的关键切割是：当前 `main.tex` 的仿真实验段落仍保留了基于损失的混合加权与平滑更新等历史表述，但这些内容与本轮 v2 的证据口径不一致，也与“本文采用标准 FedAvg”这一最高优先级约束冲突，因此在中文版 v2 中不再沿用。

为便于后续转写为 LaTeX 的 `algorithm` 环境，本节保留并修订“算法 1：FedAvg 联邦交通流预测训练流程”如下：

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

评价指标统一采用 MSE、RMSE、MAE 与 MAPE。其定义分别为：

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

这些指标共同用于衡量平均误差、平方惩罚下的偏差幅度、绝对偏差以及相对误差水平，从而为后续 CNN 与 GCN 两条路径的比较提供统一标尺。

## 4. 时空预测模型结构

### 4.1 CNN-BiLSTM-Attention

CNN 路径将局部空间依赖视为规则邻域中的卷积表征问题，并在卷积空间编码之后使用 BiLSTM 建模时间演化，再以 Attention 融合空间与时间表示。若以 $M_{i,t}$ 表示客户端局部空间邻域输入，则卷积式空间编码可写为：

$$
H_{i,t}^{(m)}
=
\sigma \left( W^{(m)} * M_{i,t} + b^{(m)} \right)
$$

这一表达与现有公式说明文档 `formula_spatial_cnn.md` 以及 `cfb_core.py`、`cfe_core.py` 中的 CNN 分支实现相一致。其作用在于说明：在规则节点排列或局部邻域结构较明确的合成流量场景中，卷积操作能够有效提取局部空间模式，从而为后续时序建模提供稳定输入。

### 4.2 GCN-BiLSTM-Attention

GCN 路径将交通系统表示为图结构，并通过图卷积显式传播节点间的拓扑依赖。设 $X_t$ 为时刻 $t$ 的节点特征矩阵，$\tilde{A}=A+I$ 为加自环后的邻接矩阵，$\tilde{D}$ 为对应度矩阵，则图卷积写为：

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

这一公式与 `formula_spatial_gcn.md` 以及 `gfb_core.py`、`gfe_core.py` 中的实现保持一致。相较于 CNN 路径，GCN 的优势并不体现在改变聚合规则，而在于它可以将显式道路连通关系直接纳入局部模型内部，从而使“联邦训练框架”和“图结构建模路径”在逻辑上保持清晰分层。

## 5. 基础联邦实验结果

### 5.1 CNN-FedAvg 与 Independent 对比

基础 CNN 实验来自 `results/simulation_experiments/cnn_fed_base/`。`main_summary.csv` 显示，FedAvg 的平均 MSE、RMSE、MAE 和 MAPE 分别为 0.00018285、0.013503、0.010789 和 1.0789%，而 Independent 的对应结果为 0.00024686、0.015326、0.012746 和 1.2746%。这表明，在规则化合成交通流场景下，跨客户端的参数共享能够稳定降低平均预测误差。

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

从 `main_metrics.csv` 的逐客户端结果看，FedAvg 在客户端 0、1、3 和 4 上均优于 Independent，其中客户端 3 的 MSE 从 0.00048777 降至 0.00019087，改善最为显著。这说明即使在基础仿真环境中，不同客户端之间仍存在足以影响局部优化方向的差异，而 FedAvg 的样本量加权平均能够缓解局部过拟合和跨客户端性能失衡。

![图 1 CNN-FedAvg 与 Independent 指标对比](../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf`，后续 LaTeX 排版应优先引用 PDF。

### 5.2 GCN-FedAvg 与 Independent 对比

基础 GCN 实验同样显示出联邦协同训练的稳定收益。`gcn_fed_base/main_summary.csv` 表明，FedAvg 的平均 MSE、RMSE、MAE 和 MAPE 分别为 0.00019809、0.014028、0.011207 和 1.1207%，Independent 的对应值分别为 0.00020828、0.014267、0.012116 和 1.2116%。虽然误差改善幅度小于 CNN 路径，但整体方向保持一致。

**表 3 GCN-FedAvg 与 Independent 指标表**

| 方法 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 | 结果说明 |
|---|---:|---:|---:|---:|---|
| FedAvg | 0.00019809 | 0.014028 | 0.011207 | 1.1207% | 四项汇总指标均优于 Independent |
| Independent | 0.00020828 | 0.014267 | 0.012116 | 1.2116% | 跨客户端稳定性较弱 |

这说明联邦协同收益并不依赖某一种特定空间模块。当局部模型从卷积路径切换为图卷积路径后，FedAvg 仍能带来误差降低，因此本文主线应被理解为“标准联邦训练框架适配不同的时空预测模型”，而不是“空间模块本身替代联邦聚合”。

![图 2 CNN-FedAvg 收敛曲线](../../results/simulation_experiments/cnn_fed_base/convergence_curve.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/cnn_fed_base/convergence_curve.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 3 GCN-FedAvg 收敛曲线](../../results/simulation_experiments/gcn_fed_base/convergence_curve.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_base/convergence_curve.pdf`，后续 LaTeX 排版应优先引用 PDF。

### 5.3 CNN 与 GCN 路径对比

将两条路径放在统一的 FedAvg 框架下比较可以发现：在基础合成数据的规则节点布局中，CNN-FedAvg 的平均 MSE 略低于 GCN-FedAvg，而 GCN-FedAvg 仍保持对 Independent 的稳定优势。这表明在规则化仿真环境下，卷积型局部空间编码具有较强归纳偏置；与此同时，GCN 的核心价值在于能够显式纳入图拓扑，从而为后续动态图分析和路网结构扩展提供建模基础。因此，两条路径在本节中承担的是互补验证而非相互替代的角色。

## 6. 异质性与客户端扩展实验

增强异质性实验的默认总体结果来自 `cnn_enhanced_main_summary.csv`。其中，FedAvg 的平均 MSE、RMSE、MAE 分别为 69.6236、7.1846 和 5.7256，Independent 的对应结果为 73.2112、7.6420 和 6.0888。这说明在样本量不平衡、噪声差异与分布差异共同存在的条件下，标准 FedAvg 仍能相对 Independent 保持一定优势。

然而，当前主文最需要修正的并不是“是否存在异质性收益”，而是“哪些细粒度结果真正满足 FedAvg-only 证据口径”。核验结果显示，`cnn_enhanced_noniid_summary.csv`、`cnn_enhanced_client_scale_summary.csv` 和 `cnn_enhanced_feature_ablation_summary.csv` 当前均只有 `Proposed` 行，不包含可直接写入主文的 FedAvg 汇总结果。因此，这三类实验在 v2 中只能作为实验设计与证据缺口说明，不能再被写成正式的 FedAvg 主结论。

**表 4 FedAvg 主文证据核验表**

| 实验主题 | 结果文件 | 当前是否存在 FedAvg 行 | v2 处理方式 | 说明 |
|---|---|---|---|---|
| Non-IID 分层 | `cnn_enhanced_noniid_summary.csv` | 否 | 不写入定量主结论 | 仅保留低/中/高异质性设计说明 |
| 客户端数量扩展 | `cnn_enhanced_client_scale_summary.csv` | 否 | 不写入定量主结论 | 3/5/8 客户端设计保留，但不引述数值 |
| 特征消融 | `cnn_enhanced_feature_ablation_summary.csv` | 否 | 不写入定量主结论 | 仅说明已有消融设计，FedAvg 证据不足 |

因此，本节在正式论文口径下能够稳妥得出的结论是：增强异质性数据上的默认总体比较支持“FedAvg 相对 Independent 仍保持一定优势”，但对于异质性强度分层、客户端规模变化和特征消融等更细粒度结论，当前尚无满足 FedAvg-only 主文要求的汇总证据，后续若作者补充相应结果文件，方可再转入正文。

## 7. 收敛性分析

基础实验的收敛结果必须区分 CNN 与 GCN 两条路径，而不能再笼统写成“基础实验均为 10 轮”。代码和结果联合表明，CNN 基础实验的主训练轮次为 10，但 `convergence_history.csv` 记录了 15 轮收敛轨迹；GCN 基础实验的训练与收敛记录均为 10 轮。具体而言，CNN 的平均验证 RMSE 从第 1 轮的 0.09474 快速下降至第 10 轮的 0.01234，并在第 12 至 15 轮维持在约 0.0117 至 0.0124 区间；GCN 的平均验证 RMSE 则从第 1 轮的 0.02676 下降到第 10 轮的 0.01385。

这一结果具有两层含义。第一，FedAvg 在合成交通流场景中能够在有限通信轮次内形成稳定的全局模型，前期误差快速下降而后逐步进入平稳区间。第二，CNN 与 GCN 路径在收敛形态上保持一致，说明收敛稳定性主要来自统一的联邦训练协议，而不是某一特定空间模块的偶然表现。

## 8. 鲁棒性实验

鲁棒性实验来自 `fed_robustness` 目录，涵盖客户端掉线、通信延迟和梯度噪声三类扰动。需要特别强调的是，这里的梯度噪声只是 simulated gradient perturbation，用于考察联邦训练对随机参数扰动的敏感性，不能被表述为正式差分隐私机制。

**表 5 鲁棒性实验结果表（仅保留 FedAvg 行）**

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

从趋势上看，客户端掉线对整体误差的影响最小，通信延迟带来的波动最为明显，而梯度噪声在低强度下影响有限、在较高强度下才出现较清晰的误差上升。这说明标准 FedAvg 对参与波动和小幅随机扰动具有一定容错能力，但对参数时延更为敏感。

## 9. GCN 图结构与动态图分析

基础 GCN 实验已经证明，在显式图结构输入下，FedAvg 同样能够获得稳定收益。增强 GCN 实验则进一步比较固定图与动态图的差异。`gcn_enhanced_fixed_vs_dynamic_summary.csv` 的 FedAvg 行显示，固定图条件下平均 MSE 为 49.6998，而早高峰、晚高峰和平峰动态图条件下的平均 MSE 分别为 49.1631、49.1752 和 49.1648，均略优于固定图。结合 `enhanced_gcn_graph_summary.csv` 可知，动态图的平均边权与图密度均明显高于基础固定图，表明时段相关的连通关系确实改变了空间信息传播方式。

不过，这一部分结论必须保持克制表达。`gfe_core.py` 明确显示增强 GCN 实验当前只有 `SEEDS = [42]`，即仅基于单一随机种子。因而，固定图与动态图之间的差异只能被写为趋势性证据，用于说明动态图结构具有潜在价值，而不能上升为强统计结论。

**表 6 GCN 固定图与动态图结果表（FedAvg 行）**

| 图类型 | MSE 均值 | RMSE 均值 | MAE 均值 | MAPE 均值 | 说明 |
|---|---:|---:|---:|---:|---|
| Fixed | 49.6998 | 6.2163 | 4.8295 | 23.8204% | 固定邻接基准 |
| Dynamic-Morning | 49.1631 | 6.1907 | 4.8069 | 23.7915% | 略优于固定图 |
| Dynamic-Evening | 49.1752 | 6.1912 | 4.8071 | 23.7922% | 略优于固定图 |
| Dynamic-Offpeak | 49.1648 | 6.1906 | 4.8067 | 23.7922% | 略优于固定图 |

![图 4 基础 GCN 邻接矩阵示意](../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf`，后续 LaTeX 排版应优先引用 PDF。

![图 5 增强 GCN 固定邻接矩阵示意](../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png)

注：该图的同名 PDF 版本位于 `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf`，后续 LaTeX 排版应优先引用 PDF。

## 10. 图表、表格与伪代码安排

为便于后续转写为 LaTeX，v2 建议保留“3 张基础结果图 + 2 张图结构图 + 6 张辅助表 + 1 个 FedAvg 伪代码”的组合形式。与 v1 相比，这一安排的核心变化在于：所有可直接预览的图均在 Markdown 中使用 PNG 路径，但必须同时核验同名 PDF 是否存在；凡是图中混入 `Proposed`、`Loss-weighted` 或 `Data-loss weighted` 曲线的现有图件，均不得直接进入主文，而只作为待重绘或待确认资产。

基于这一原则，当前不宜直接写成 Markdown 图片引用的图件包括以下几类：

【图 6 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.png`，且存在同名 PDF `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.pdf`，但其来源结果与当前主文证据口径不一致，暂不进入主文。】

【图 7 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.png`，且存在同名 PDF `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.pdf`，但汇总文件缺少 FedAvg 行，暂不进入主文。】

【图 8 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.png`，且存在同名 PDF `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.pdf`，但汇总文件缺少 FedAvg 行，暂不进入主文。】

【图 9 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.png`，且存在同名 PDF `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.pdf`，但现有图件同时绘制了 `FedAvg` 与 `Proposed`，需重绘 FedAvg-only 版本后再进入 LaTeX 正文。】

【图 10 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.png`，且存在同名 PDF `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.pdf`，但现有图件同时绘制了 `FedAvg` 与 `Proposed`，需重绘 FedAvg-only 版本后再进入 LaTeX 正文。】

【图 11 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.png`，且存在同名 PDF `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.pdf`，但现有图件同时绘制了 `FedAvg` 与 `Proposed`，需重绘 FedAvg-only 版本后再进入 LaTeX 正文。】

【图 12 待处理：当前存在 PNG 预览图 `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.png`，且存在同名 PDF `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.pdf`，但现有图件混入 `Proposed` 曲线，需重绘 FedAvg-only 版本后再进入 LaTeX 正文。】

## 11. 小结

综上，v2 可以在不改变 FedAvg 主线、不修改实验代码、不重跑实验的前提下，形成与正式论文 `Data analysis` 章节风格一致的仿真实验中文正式稿。其核心结论是：第一，标准 FedAvg 在 CNN 与 GCN 两条时空预测路径下均优于 Independent，说明联邦协同训练能够稳定提升基础合成交通流预测性能；第二，FedAvg 在基础场景中表现出明确的收敛性，且 CNN 收敛记录为 15 轮、GCN 收敛记录为 10 轮，二者需明确区分；第三，增强异质性默认场景支持“FedAvg 相对 Independent 仍保持一定优势”，但 Non-IID 分层、客户端数量扩展和特征消融三类细粒度结果当前不满足 FedAvg-only 主文证据口径；第四，鲁棒性实验表明 FedAvg 对掉线和轻度扰动具有一定容错能力，但对通信延迟更为敏感；第五，增强 GCN 的动态图结果仅能作为趋势性证据，因为当前实验仍是单种子设置。

从与正式论文结构的衔接看，本模块已经具备进入“中文 v2 审阅完成后，再转写为英文 LaTeX 草稿”的条件，但前提是后续转写时继续坚持两条原则：一是删除或改写 `main.tex` 中与标准 FedAvg 不一致的旧聚合表述；二是对含探索性聚合策略的图件进行 FedAvg-only 重绘或暂缓纳入正文。
