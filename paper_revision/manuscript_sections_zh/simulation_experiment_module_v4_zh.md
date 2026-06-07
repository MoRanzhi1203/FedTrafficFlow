# 仿真实验模块正式稿 v4（中文版）

## 4 仿真实验与结果分析

### 4.1 实验设置

本文采用 FedAvg 作为联邦交通流量预测的核心研究方法。该方法通过服务器端对多客户端局部模型参数进行聚合，实现多区域交通数据的协同建模，同时保持原始交通数据存储于本地，从而降低集中式训练对原始数据汇聚的依赖。对于交通流量预测任务而言，客户端可对应不同交通区域、检测点或数据子集，因此 FedAvg 能够较好适配多区域、多检测点和多客户端并存的交通预测场景。

在模型结构方面，本文分别考察 CNN 与 GCN 两类空间建模路径。其中，CNN 结构用于提取局部规则空间邻域中的时空特征，GCN 结构则利用图拓扑显式刻画交通节点之间的空间关联。两类结构均在统一的 FedAvg 框架下开展联邦训练，以比较不同空间表示方式对联邦交通流量预测性能与稳定性的影响。与此同时，本文引入 Independent 作为非联邦对照实验，即各客户端仅基于本地数据独立训练而不进行联邦聚合，用以检验 FedAvg 是否能够通过跨客户端参数共享带来更稳定的整体预测表现。

实验评价指标采用 RMSE、MAE、MAPE 和 R²。RMSE 与 MAE 分别衡量预测误差的平方根尺度和绝对偏差水平，MAPE 反映相对误差变化，R² 则用于刻画模型对目标序列方差的解释能力。所有核心仿真实验均在五个随机种子下重复执行，随机种子统一设定为 42、2024、3407、1234 和 5678，并以 mean ± std 的形式报告结果，以降低单次运行偶然性对结论的影响。

需要说明的是，基础结构实验与增强/鲁棒性实验可能采用了不同的数据尺度或反归一化评价口径，因此不同实验组之间的绝对误差数值不宜直接横向比较。本文主要在同一实验设置内部比较不同方法或不同场景下的相对表现，并结合多随机种子的均值和标准差分析结果稳定性。

### 4.2 多随机种子实验设计

在联邦交通流量预测任务中，随机初始化、本地 mini-batch 顺序、客户端采样过程以及扰动机制设定都可能影响模型的优化轨迹与最终指标表现。因此，仅依赖单一随机种子往往难以全面反映方法的稳定性。基于这一考虑，本文对基础实验、增强实验、收敛性实验以及鲁棒性实验均采用五随机种子重复执行，以降低偶然波动对实验判断的干扰。

从结果汇总情况来看，五个随机种子 42、2024、3407、1234 和 5678 已覆盖 `cnn_fed_base`、`gcn_fed_base`、`cnn_fed_enhanced_experiments`、`gcn_fed_enhanced_experiments`、`fed_robustness_experiments` 以及相关收敛统计结果。对各实验组的多随机种子原始结果与汇总结果进行核对后，当前未发现空表、缺 seed 或所有 seed 完全一致等异常现象。因此，后续分析均以五随机种子下的均值和标准差作为主要统计依据。

此外，多随机种子设计还有助于从“均值表现”和“波动幅度”两个层面理解联邦训练行为。若某一设置虽然均值较优但标准差较大，则其稳定性仍需谨慎判断；反之，若某一结构在多个种子下均表现出较小波动，则说明其在当前交通预测场景下具有更稳健的训练特征。基于这一原则，本文在后续小节中均结合 mean 与 std 同时展开讨论，而不以单一最优值作为主要结论依据。

### 4.3 FedAvg 在 CNN 与 GCN 结构下的主结果分析

表 4-1 给出了 FedAvg 在 CNN 与 GCN 两类结构下的五随机种子主结果。对应图件建议为：图 4-1 使用 `../results/simulation_experiments/cnn_fed_base/multi_seed_mean_std_metrics.png`，图 4-2 使用 `../results/simulation_experiments/gcn_fed_base/multi_seed_mean_std_metrics.png`。

**表 4-1 FedAvg 在 CNN 与 GCN 结构下的五随机种子主结果**

| Setting | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| FedAvg-CNN | 0.0144 ± 0.0025 | 0.0118 ± 0.0021 | 1.1790 ± 0.2104 | -0.3849 ± 0.4127 |
| FedAvg-GCN | 0.0132 ± 0.0006 | 0.0105 ± 0.0005 | 1.0467 ± 0.0496 | -0.1114 ± 0.1220 |

由表 4-1 可知，在当前实验设置下，GCN-FedAvg 的 RMSE、MAE 和 MAPE 均低于 CNN-FedAvg，说明显式图结构对交通节点间空间关联的刻画有助于降低整体预测误差。从五个随机种子的标准差可以进一步看出，GCN-FedAvg 在 RMSE、MAE、MAPE 和 R² 上的波动幅度均小于 CNN-FedAvg，表明其跨随机种子的稳定性更好。

需要注意的是，基础 CNN 与 GCN 实验中 FedAvg 的 R² 指标仍为负值，说明在该归一化评价口径下，模型对目标序列方差的解释能力仍有进一步提升空间。因此，本文不将单一 R² 指标作为评价 FedAvg 有效性的唯一依据，而是结合 RMSE、MAE、MAPE、多随机种子波动和收敛曲线进行综合分析。尽管如此，GCN-FedAvg 的 R² 更接近 0，说明图结构对交通流空间关系建模具有一定帮助。综合来看，FedAvg 在 CNN 和 GCN 两种结构下均能够完成联邦交通流量预测，而在当前实验设置下，GCN-FedAvg 表现出更稳定的统计特征。

### 4.4 FedAvg 与独立训练的对比分析

Independent 在本文中表示“各客户端独立训练，不进行联邦聚合”。该对照实验的引入，主要用于验证 FedAvg 是否能够通过跨客户端参数聚合降低整体误差并提升模型稳定性。表 4-2 给出了基础结构与增强设置下 FedAvg 和 Independent 的五随机种子对比结果。

**表 4-2 FedAvg 与独立训练方式的五随机种子对比结果**

| Setting | FedAvg RMSE | Independent RMSE | FedAvg MAE | Independent MAE | FedAvg MAPE | Independent MAPE | FedAvg R² | Independent R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN base | 0.0144 ± 0.0025 | 0.0174 ± 0.0016 | 0.0118 ± 0.0021 | 0.0147 ± 0.0016 | 1.1790 ± 0.2104 | 1.4746 ± 0.1585 | -0.3849 ± 0.4127 | -1.1746 ± 0.3980 |
| GCN base | 0.0132 ± 0.0006 | 0.0150 ± 0.0007 | 0.0105 ± 0.0005 | 0.0124 ± 0.0007 | 1.0467 ± 0.0496 | 1.2388 ± 0.0687 | -0.1114 ± 0.1220 | -0.4900 ± 0.1660 |
| CNN enhanced | 7.1055 ± 0.4556 | 7.5618 ± 0.4190 | 5.5194 ± 0.5239 | 6.0109 ± 0.3545 | 39.1712 ± 23.7928 | 44.6723 ± 42.7436 | 0.3302 ± 0.0604 | 0.1894 ± 0.0825 |
| GCN enhanced | 6.5973 ± 0.2835 | 7.5507 ± 0.2503 | 4.9860 ± 0.3413 | 5.8670 ± 0.2695 | 37.8112 ± 22.9801 | 30.2367 ± 17.2021 | 0.3864 ± 0.0366 | 0.1025 ± 0.1139 |

从基础实验结果看，无论是在 CNN base 还是 GCN base 场景下，FedAvg 的 RMSE、MAE 和 MAPE 均低于 Independent，且 R² 也相对更优。这说明在基础交通预测场景中，跨客户端参数聚合能够有效缓解单客户端独立训练的信息局限，使全局模型获得更稳定的整体误差表现。

从增强设置结果看，FedAvg 在 CNN enhanced 和 GCN enhanced 两组实验中的 RMSE、MAE 和 R² 也整体优于 Independent，表明当数据异质性增强后，联邦协同建模仍具有实际必要性。需要特别指出的是，在 GCN enhanced 设置中，Independent 的 MAPE 低于 FedAvg，但其 RMSE、MAE 和 R² 明显不如 FedAvg，因此不能仅凭单一 MAPE 指标判断 Independent 更优。考虑到 MAPE 对局部低流量样本更为敏感，本文仍以多指标综合判断方法表现。总体而言，FedAvg 通过跨客户端参数聚合，在多数设置和多数指标上优于独立训练，说明联邦协同建模对于交通流量预测是必要的。

### 4.5 FedAvg 增强设置的补充实验分析

在增强实验结果文件中，历史字段中曾出现增强变体标记。为保持全文方法口径一致，本文将该部分统一表述为“FedAvg 增强变体”或“FedAvg 框架下的增强设置”。需要强调的是，该部分实验用于分析在 FedAvg 框架下加入特定增强机制后，模型预测性能和收敛行为是否发生变化，因此其定位是对 FedAvg 框架的补充分析，而不是将 FedAvg 作为被替代的普通基线。

**表 4-3 FedAvg 增强设置下的五随机种子主结果**

| Setting | Method | RMSE | MAE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| CNN enhanced | FedAvg | 7.1055 ± 0.4556 | 5.5194 ± 0.5239 | 39.1712 ± 23.7928 | 0.3302 ± 0.0604 |
| CNN enhanced | Independent | 7.5618 ± 0.4190 | 6.0109 ± 0.3545 | 44.6723 ± 42.7436 | 0.1894 ± 0.0825 |
| CNN enhanced | FedAvg 增强变体 | 7.1025 ± 0.4225 | 5.4930 ± 0.5214 | 38.2997 ± 22.5504 | 0.3318 ± 0.0654 |
| GCN enhanced | FedAvg | 6.5973 ± 0.2835 | 4.9860 ± 0.3413 | 37.8112 ± 22.9801 | 0.3864 ± 0.0366 |
| GCN enhanced | Independent | 7.5507 ± 0.2503 | 5.8670 ± 0.2695 | 30.2367 ± 17.2021 | 0.1025 ± 0.1139 |
| GCN enhanced | FedAvg 增强变体 | 6.5168 ± 0.2630 | 4.9008 ± 0.3231 | 36.0653 ± 21.5643 | 0.4021 ± 0.0301 |

由表 4-3 可知，在 CNN enhanced 设置中，FedAvg 增强变体与标准 FedAvg 的差异较小，RMSE、MAE、MAPE 和 R² 的变化幅度均较为有限，说明增强机制在卷积结构场景中的收益并不显著。换言之，CNN enhanced 中的主要结论并非“增强设置替代了 FedAvg”，而是该增强机制在当前卷积结构与当前任务设定下只带来了较小的补充改进。

相较之下，在 GCN enhanced 设置中，FedAvg 增强变体在 RMSE、MAE、MAPE 和 R² 上相对标准 FedAvg 均表现出一定改善。这说明在图结构建模场景下，增强设置更容易体现其作用，可能与图拓扑能够更充分地利用增强机制所表达的空间关联信息有关。不过，这一结果仍应被理解为 FedAvg 框架内的变体分析，而不是“FedAvg 被替代”。综合来看，增强机制在图结构建模条件下更有可能带来补充收益，但其作用强度仍需结合任务设定与收敛行为审慎解读。

增强实验相关图件建议作为补充材料引用，包括 `../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_mean_std.png`、`../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_convergence_curve.png`、`../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_mean_std.png` 和 `../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_convergence_curve.png`。

### 4.6 多随机种子收敛性分析

收敛性分析不仅关注最终轮指标，还关注多轮训练过程中验证误差的下降趋势与波动范围。表 4-5 给出了多随机种子收敛曲线在最终轮的统计结果。正文建议引用图 4-3 `../results/simulation_experiments/cnn_fed_base/convergence_curve.png` 和图 4-4 `../results/simulation_experiments/gcn_fed_base/convergence_curve.png`，以辅助说明基础 FedAvg 训练的收敛过程。

**表 4-5 多随机种子收敛曲线最终轮统计结果**

| Setting | Method | Final-round Val RMSE | Final-round Val MAE | Final-round Val MAPE |
|---|---|---:|---:|---:|
| CNN FedAvg base | FedAvg | 0.0116 ± 0.0009 | 0.0093 ± 0.0009 | 0.9328 ± 0.0911 |
| GCN FedAvg base | FedAvg | 0.0128 ± 0.0007 | 0.0102 ± 0.0006 | 1.0221 ± 0.0566 |
| CNN enhanced | FedAvg | 7.3178 ± 0.6780 | 5.4015 ± 0.4899 | 21.2546 ± 30.6412 |
| CNN enhanced | FedAvg 增强变体 | 7.3146 ± 0.6804 | 5.4396 ± 0.5363 | 21.6161 ± 31.3967 |
| GCN enhanced | FedAvg | 7.5583 ± 0.5910 | 5.6894 ± 0.3534 | 23.3897 ± 35.4072 |
| GCN enhanced | FedAvg 增强变体 | 7.4103 ± 0.6005 | 5.5662 ± 0.3840 | 22.1986 ± 33.1006 |

从表 4-5 可以看出，CNN base 和 GCN base 在多随机种子下均形成了较为稳定的收敛结果。基础 CNN 的最终轮验证 RMSE 为 0.0116 ± 0.0009，基础 GCN 的最终轮验证 RMSE 为 0.0128 ± 0.0007，两者均处于较低波动水平，说明 FedAvg 在基础交通流预测任务中能够在有限通信轮次内形成相对稳定的全局模型。

对于增强设置而言，CNN enhanced 中标准 FedAvg 与 FedAvg 增强变体的最终轮指标较为接近，进一步印证了前文关于“卷积结构下增强收益有限”的判断。相比之下，GCN enhanced 中 FedAvg 增强变体的最终轮 Val RMSE、Val MAE 和 Val MAPE 均低于标准 FedAvg，说明图结构增强设置下的收敛结果更具优势。与此同时，MAPE 的标准差在增强实验中明显偏大，说明该指标对局部低流量样本或极端样本较为敏感。因此，收敛性结论不能仅依赖最后一轮指标，还应结合收敛曲线图的整体变化趋势，以及 RMSE 和 MAE 的共同表现进行综合判断。

作为补充材料，建议附录中进一步给出 `../results/simulation_experiments/cnn_fed_base/multi_seed_rmse_boxplot.png`、`../results/simulation_experiments/gcn_fed_base/multi_seed_rmse_boxplot.png`、`../results/simulation_experiments/cnn_fed_base/multi_seed_rmse_seed_pairing.png` 和 `../results/simulation_experiments/gcn_fed_base/multi_seed_rmse_seed_pairing.png`，以呈现不同随机种子之间的配对差异与误差分布。

### 4.7 联邦扰动场景下的鲁棒性分析

为进一步分析 FedAvg 在联邦环境不稳定条件下的适应能力，本文考察客户端掉线、通信延迟和梯度噪声三类扰动场景。对应正文图件建议为图 4-5 `../results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_mean_std_metrics.png`。

**表 4-4 FedAvg 在联邦扰动场景下的鲁棒性结果**

| Scenario | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| client_dropout@0.0 | 7.7210 ± 0.4694 | 6.0880 ± 0.6071 | 50.4387 ± 33.8852 | 0.1867 ± 0.0923 |
| client_dropout@0.2 | 7.5768 ± 0.5060 | 5.9475 ± 0.5709 | 48.3667 ± 31.5024 | 0.2212 ± 0.0756 |
| client_dropout@0.4 | 7.5882 ± 0.5381 | 5.9534 ± 0.5764 | 46.4964 ± 31.6569 | 0.2289 ± 0.0721 |
| communication_delay@0 | 7.7210 ± 0.4694 | 6.0880 ± 0.6071 | 50.4387 ± 33.8852 | 0.1867 ± 0.0923 |
| communication_delay@1 | 7.9783 ± 0.5793 | 6.3644 ± 0.7326 | 54.3768 ± 39.2385 | 0.1315 ± 0.1059 |
| communication_delay@2 | 7.8319 ± 0.5282 | 6.2225 ± 0.6427 | 53.1185 ± 38.4140 | 0.1647 ± 0.0612 |
| gradient_noise@0.0 | 7.7210 ± 0.4694 | 6.0880 ± 0.6071 | 50.4387 ± 33.8852 | 0.1867 ± 0.0923 |
| gradient_noise@0.02 | 7.7203 ± 0.3707 | 6.0837 ± 0.4399 | 46.5238 ± 27.8278 | 0.2097 ± 0.1132 |
| gradient_noise@0.05 | 8.0273 ± 0.3277 | 6.3916 ± 0.4338 | 50.9977 ± 33.5397 | 0.1222 ± 0.1043 |

由表 4-4 可知，在客户端掉线场景下，当 dropout 从 0.0 提升至 0.4 时，RMSE 并未出现明显恶化，MAE 与 R² 甚至略有改善，说明 FedAvg 对一定比例客户端缺失具有一定容忍度。这一结果表明，在部分客户端临时掉线的条件下，联邦聚合仍能维持相对稳定的整体预测性能。

在通信延迟场景中，`delay@1` 和 `delay@2` 相比 `delay@0` 出现了一定程度的误差上升，其中 `delay@1` 的 RMSE 与 MAE 退化更为明显，R² 也下降至 0.1315 ± 0.1059。这说明通信延迟会削弱参数聚合的及时性和一致性，从而对联邦优化效率带来不利影响。换言之，FedAvg 对通信延迟并非完全不敏感，时延仍是需要关注的重要系统因素。

在梯度噪声场景中，`noise@0.02` 的 RMSE 和 MAE 基本保持稳定，说明弱梯度噪声下模型仍具有较好的适应能力；当噪声强度上升至 `noise@0.05` 时，RMSE、MAE 和 R² 均出现一定退化，但整体并未失稳。综合来看，FedAvg 在联邦扰动场景下具有一定鲁棒性，但通信延迟和较强噪声仍会带来性能下降。作为补充材料，建议在附录中引用 `../results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_rmse_boxplot.png`、`../results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_seed_pairing.png` 和 `../results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_improvement_heatmap.png`，以呈现不同扰动类型下的误差分布差异与相对变化情况。

### 4.8 特征消融补充分析

除上述主实验外，`cnn_enhanced_feature_ablation` 的五随机种子结果也已完成并纳入当前结果体系。该部分实验主要用于分析不同特征组合对 FedAvg 增强设置的影响，从而考察输入特征构成与联邦预测性能之间的关系。

需要强调的是，特征消融补充分析的定位在于进一步解释增强设置中性能差异的可能来源，而非作为本文主结论的核心依据。考虑到当前任务重点在于重构 FedAvg 作为核心研究方法的仿真实验主线，因此该部分更适合放入附录或补充材料中，与增强实验的补充图表一并呈现，而不必在正文中展开过长篇幅。相关补充结果可结合 `../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_metrics_fedavg.csv` 及其对应图表进行整理。

### 4.9 本章小结

本章围绕 FedAvg 在联邦交通流量预测中的表现，从基础结构、训练模式、增强设置、收敛特性和扰动鲁棒性等多个角度进行了系统分析。结果表明，FedAvg 在 CNN 和 GCN 两类结构下均能够完成稳定的联邦交通流量预测任务，其中 GCN-FedAvg 在当前实验设置下表现出更低的误差和更小的跨随机种子波动，说明图结构建模对交通流空间关联的表达具有积极作用。

与 Independent 对照相比，FedAvg 在多数设置和多数指标上均表现更优，说明跨客户端参数聚合能够带来联邦协同优势。在增强设置中，FedAvg 增强变体在 CNN 场景下改进较小，而在 GCN 场景下表现出一定补充提升，说明增强机制更容易在图结构建模条件下发挥作用。同时，收敛性分析表明，FedAvg 在基础结构中能够形成较稳定的训练过程，增强设置中的收敛结果则需要结合最终轮指标与完整收敛曲线共同解读。

鲁棒性实验进一步表明，FedAvg 对客户端掉线具有一定容忍度，在弱梯度噪声下仍能保持较稳定表现，但通信延迟和较强噪声会带来一定性能下降。因此，FedAvg 在当前交通流量预测任务中已经展现出较好的稳定性与适用性，但基础实验中的负 R²、增强实验中的高 MAPE 波动以及高扰动条件下的性能退化也说明，其在模型表达能力、指标稳定性和复杂联邦环境适应性方面仍存在进一步优化空间。

---

自检结果：
1. 已统一 FedAvg 为核心研究方法；
2. 未使用“增强方法替代 FedAvg”的错误口径；
3. 已纳入 5-seed 主结果、对比结果、增强结果、收敛结果和鲁棒性结果；
4. 已说明不同实验尺度不可直接横向比较；
5. 已说明 R² 负值和 MAPE 高波动问题；
6. 已给出正文图和附录图建议；
7. 生成文件：paper_revision/simulation_experiment_module_v4_zh.md。
