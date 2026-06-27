# 实验 1：一审意见与会议要求对齐梳理报告

## 1. 梳理目的

本报告用于梳理：

- 一审修改意见中与真实数据联邦学习实验相关的要求；
- 老师会议记录或导师意见引用中与 CCN/CNN 客户端、真实数据、FedAvg、client 划分相关的要求；
- 当前实验 1 已完成工作的对应关系；
- 仍未完成或需要调整的部分。

## 2. 检索范围

本次实际检索的目录主要包括：

- `paper_revision/`
- `docs/`
- `real_data_experiments/`
- 项目根目录下的项目说明文档

检索文件类型主要包括：

- `.md`
- `.txt`
- `.rst`
- `.tex`
- `.csv`
- `.json`
- `.py`
- `.ipynb`

本次明确排除：

- `results/`
- `data/`
- `.git/`
- `__pycache__/`
- `.venv/` / `venv/`
- `node_modules/`
- 大型二进制文件（如 `.pt`、`.parquet`、`.png`、`.pdf`）

补充说明：

- 已新增只读检索脚本 `real_data_experiments/single_intersection_client/sic_review_meeting_alignment_audit.py` 用于批量扫描文本类文件。
- 该脚本全项目扫描后命中文件较多（236 个文件，9321 条行级命中），因此本报告只抽取与“真实数据联邦学习客户端实验”直接相关的核心证据，而不使用泛命中结果直接下结论。
- 本次自动检索未直接覆盖原始一审 `docx` 与原稿 `pdf` 原文；当前结论基于项目内可读取的文本化材料与实验报告整理，最终对外答复前仍需人工核对原始 `docx/pdf`。

## 3. 关键词说明

本次使用的关键词分组如下：

- 一审修改意见：`一审`、`审稿`、`修改意见`、`外审`、`review`、`revision`、`response`
- 真实数据实验：`真实数据`、`real data`、`实际数据`、`实证`、`主实验`、`消融`、`baseline`、`NaiveLastValue`
- 联邦学习客户端：`联邦学习`、`FedAvg`、`客户端`、`client`、`non-IID`、`非IID`、`异质性`、`heterogeneity`、`聚合`
- CCN/CNN 模型：`CCN`、`CNN`、`CNN-LSTM`、`LSTM`、`Attention`、`CNN-LSTM-Attention`
- 实验对象：`网格`、`路口`、`单路口`、`区域`、`cluster`、`region`、`grid cell`、`intersection`
- baseline / 对比：`Independent`、`baseline`、`NaiveLastValue`

关于 `CCN` 的说明：

- 本次检索**检索到了 `CCN` 原文**，但主要集中在仿真分支与项目说明文档中，例如：
  - `simulation_experiments/cnn_fed_base/cfb_core.py`
  - `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py`
  - `docs/simulation_experiments.md`
  - `docs/project_pipeline.md`
- 当前真实数据实验 1 的实际模型命名为 `CNNLSTMAttentionRegressor`，即真实数据代码侧使用的是 `CNN + LSTM + Attention` 命名，而不是 `CCN`。
- 因此，在真实数据实验 1 语境下，应优先使用 `CNN / CNN-LSTM / CNN-LSTM-Attention / client / 客户端` 相关表述；项目中检索到的 `CCN` 更适合暂时视为历史命名或并行命名线索，而不应直接断定其一定是笔误，仍需结合历史文档继续核对。

## 4. 一审修改意见中相关内容清单

本次检索到包含一审 / revision 关键词的文件共 `19` 个，但与“真实数据联邦学习客户端实验”**直接相关**且可作为当前对齐依据的核心要求，可归纳为以下 `5` 类 / `5` 条：

| 编号 | 来源文件 | 行号/位置 | 原文摘要 | 归类 | 是否与真实数据联邦学习客户端相关 | 当前回应状态 |
|---|---|---|---|---|---|---|
| 1 | `paper_revision/00_NATURE_SKILLS_PRECHECK.md` | `95-110` | 一审正确方向包括：补充 FedAvg 流程、公式、客户端定义、交通流数据划分、non-IID 影响、CNN-FedAvg vs GCN-FedAvg、FedAvg vs Independent | FedAvg / 联邦学习实验；client 划分；非 IID；模型结构 | 是 | 部分回应 |
| 2 | `paper_revision/01_prerequisite_constraints.md` | `10-34` | 明确标准 FedAvg 是主方法，Independent 是对比基线，不提出新聚合算法 | FedAvg 主线；baseline 对比 | 是 | 已回应 |
| 3 | `paper_revision/02_revision_strategy.md` | `19-21` | 创新模板要求：建立包含 N 个节点、K 个客户端的联邦框架；采用 CNN/GCN + BiLSTM + Attention；以真实大规模路网验证 non-IID、缺失数据、模型异构场景 | 真实数据验证；模型结构说明；联邦客户端 | 是 | 部分回应 |
| 4 | `paper_revision/02_revision_strategy.md` | `27-28` | 引言需加入定量结果，并补充“完整 CNN+BiLSTM+Attention 相比消融模型更优”的关键发现 | 结果解释；消融实验；模型结构说明 | 是 | 部分回应 |
| 5 | `paper_revision/02_revision_strategy.md` | `59-73` | 实验增强需包含客户端分布差异、异构量化、IID vs non-IID 差异、通信轮次 vs 精度、以及移除 CNN/BiLSTM/Attention 的消融实验 | 非 IID / 异质性；通信轮次；消融实验 | 是 | 部分回应 |

对这 5 条的总体判断：

- 已充分回应：`1` 条
- 部分回应：`4` 条
- 尚未完全回应：主要集中在“正式论文候选结果尚未稳定”“实验 2 消融尚未启动”“client 划分与论文文字说明尚未固化”

## 5. 老师会议记录中相关内容清单

本次**没有检索到独立命名的“会议记录/纪要”文件**。与导师或会议相关的记录，主要以“导师意见/与导师确认”的方式散落在策略文件中。

本次检索到包含老师/导师/会议相关关键词的文件共 `2` 个：

| 编号 | 会议文件 | 时间/位置 | 原文摘要 | 归类 | 是否与真实数据联邦学习客户端相关 | 当前回应状态 |
|---|---|---|---|---|---|---|
| 1 | `paper_revision/00_NATURE_SKILLS_PRECHECK.md` | `6, 18, 133, 408, 458, 509` | 要求整理导师意见、会议记录，与项目文档冲突时要对齐；并明确“导师意见如何落实” | 会议要求整理；回应策略 | 间接相关 | 部分回应 |
| 2 | `paper_revision/manuscript_sections_zh/history/simulation_experiment_insertion_plan_zh.md` | `44-54` | 需要与导师确认篇幅和图表责任；并明确指出仿真实验的 non-IID/鲁棒性分析应与真实数据实验保持逻辑衔接，真实数据实验也应增加异质性讨论段落 | 会议建议；真实数据异质性解释 | 是 | 部分回应 |

重点统计：

- 检索到会议相关文件数量：`2`
- 检索到独立会议纪要文件数量：`0`
- 其中与真实数据联邦学习客户端**直接相关**的会议要求条数：`1`
- 其中已经被当前实验 1 在实验侧回应的条数：`1`
- 其中尚未在论文正文侧落地的条数：`1`

说明：

- 当前实验 1 已经通过 `client` 异质性诊断补出了“真实数据实验也应增加异质性讨论”的证据链。
- 但这种回应仍停留在实验报告与诊断报告层面，尚未转写为论文正文中的正式表述。

## 6. 与当前实验 1 进展的对比

本次对比时已纳入以下当前材料：

- `real_data_experiments/single_intersection_client/experiment1_formal_v4_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_formal_v4_sanity_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_gap_diagnosis_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_smoke_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_fedavg_rounds_r60_smoke_report_zh.md`
- `real_data_experiments/single_intersection_client/experiment1_client_heterogeneity_diagnosis_zh.md`

当前实验 1 已完成内容包括：

- `grid_cell main full`
- 使用真实 tensor 数据
- `selected_clients = 290,284,318,288,289`
- `FedAvg vs Independent`
- `NaiveLastValue baseline`
- target normalization 修复
- input normalization 修复
- CUDA 环境修复
- 常数预测修复
- `r20 / r40 / r60` rounds 诊断
- client 异质性诊断
- 发现 `289` 是主要拖累 client
- 当前仍未进入实验 2 / 3 / 4

补充的 client 组织口径：

- 原稿与当前项目口径都区分 `grid-cell-level client` 与 `cluster-level client` 两类真实数据 client 组织方式。
- 当前实验 1 属于 `grid-cell-level client`：每个 client 对应一个 active pooled grid region。
- 当前实验 1 使用 `selected_clients = 290,284,318,288,289`，即 `K=5` 的 grid-cell-level client 设置。
- 结合当前项目内可读取文本版本，原稿的 cluster / region client 口径可对应为 `K=3`；因此当前 `K=5` 可视为对原稿 `K=3` 的增强，而不是更弱设置。
- 但当前仍缺少“为什么固定选择这 5 个 grid cells”的论文级依据说明。
- `cluster-level client` 当前尚未完成，后续更适合由实验 3 承接。

| 原始要求 | 当前是否已回应 | 对应文件/结果 | 证据 | 仍需补充 |
|---|---|---|---|---|
| 真实数据联邦主实验要闭环 | 是 | `experiment1_formal_v4_report_zh.md` | 已用真实 tensor 数据完成 `grid_cell main full` 正式 v4 CUDA | 仍需形成论文候选正式结果 |
| FedAvg 与 Independent 对比 | 是 | `experiment1_formal_v4_report_zh.md`、`experiment1_formal_v4_sanity_report_zh.md` | 已给出主指标对比，Independent 明显更强 | 需转写为论文正文 |
| baseline 对比 | 是 | `experiment1_formal_v4_report_zh.md`、`experiment1_formal_v4_sanity_report_zh.md` | 已保留 `Independent` 作为主学习型 baseline，并补充 `NaiveLastValue` 作为真实交通流短时惯性的 sanity baseline | 需解释为何 naive 很强，且不应因当前 FedAvg 未全面超过 naive 就直接删除该 baseline |
| client 划分要明确 | 部分回应 | `experiment1_formal_v4_report_zh.md` | 已明确 selected clients 为 `290,284,318,288,289` | 仍需补充为何选这 5 个 grid cell、其论文叙事逻辑是什么 |
| non-IID / 异质性分析 | 是 | `experiment1_fedavg_gap_diagnosis_zh.md`、`experiment1_client_heterogeneity_diagnosis_zh.md` | 已确认 non-IID 明显，且 `289` 为主要拖累 client | 若写论文，最好再配图或表格化说明 |
| 通信轮次 vs 精度权衡 | 是 | `experiment1_fedavg_rounds_smoke_report_zh.md`、`experiment1_fedavg_rounds_r60_smoke_report_zh.md` | `20 -> 40 -> 60` 明显改善，但边际收益变小 | 需形成论文中的“收益递减”解释 |
| 模型结构为 CNN/GCN + BiLSTM + Attention | 部分回应 | `sic_core.py`、实验 1诊断报告 | 真实数据实验 1 实际模型为 `CNNLSTMAttentionRegressor` | 需统一 `CCN/CNN` 命名，并与论文叙事一致 |
| grid-cell-level 与 cluster-level client 组织 | 部分回应 | `single_intersection_client/README_zh.md`、`region_client/README_zh.md` | 当前已落地 `grid-cell-level client`，并在项目内保留 `cluster-level client` 定义与默认 `num_clients=3` 口径 | 当前实验 1 尚未给出 cluster-level 结果，需由实验 3 承接 |
| 消融实验 | 否 | 当前仅有计划，无正式结果 | 一审策略明确要求移除 CNN/BiLSTM/Attention 的消融 | 需在实验 1 主结果稳定后进入实验 2 |
| 证明 FedAvg 在真实数据上有效 | 部分回应 | v4/r40/r60 报告 | 已证明联邦链路可运行、修复有效、加 rounds 有帮助 | 但 FedAvg 仍未全面超过 `NaiveLastValue` |
| 结果解释框架 | 是 | `experiment1_fedavg_gap_diagnosis_zh.md`、`experiment1_client_heterogeneity_diagnosis_zh.md` | 已将问题归因为“异质性 + 强 naive baseline”，并排除了实现错误 | 需转化为论文级文字表述 |

## 7. 当前工作与一审/会议要求的匹配程度

已充分回应：

- 标准 FedAvg 主线定位
- `FedAvg vs Independent` 对比
- 真实数据联邦训练链路已经跑通
- `NaiveLastValue` baseline 已补齐
- non-IID / 异质性问题已被实验侧识别并定位到 `289`

部分回应：

- 真实数据联邦主实验虽然已跑通，但 `FedAvg` 尚未成为论文候选正式结果
- client 划分在实现层面已存在，但缺少论文中的设计逻辑说明
- `CCN/CNN` 命名在历史文档、仿真分支与真实数据代码之间仍不完全统一
- 通信轮次 vs 精度权衡已有 smoke 证据，但尚未固化为论文表述
- 老师会议里提出的“真实数据也要补异质性讨论”已在实验侧回应，但未写入正文
- 当前实验 1 属于 `K=5` 的 `grid-cell-level client`，可视为对原稿 `K=3` 的增强，但这一增强关系与限制尚未明确写出

尚未回应：

- 实验 2：单路口消融（去 CNN / 去 LSTM / 去 Attention）尚未启动
- “FedAvg 在真实数据上优于强基线”的最终候选结果尚未拿到
- client 分组 / cluster-level client 解释路径尚未正式展开，且应由实验 3 承接

需要重新表述：

- 论文或汇报中的 `CCN` 应核实是否确为历史命名；真实数据实验 1 当前更准确的技术表述应是 `CNN-LSTM-Attention`
- `Independent` 应明确保留为主学习型 baseline，`NaiveLastValue` 应明确表述为真实交通流短时惯性的 sanity baseline
- 若要回应一审与导师意见，应避免把当前实验 1 写成“FedAvg 已经充分优于真实数据基线”，因为证据并不支持这一点

## 8. 当前实验 1 的问题解释框架

基于当前结果，较稳妥的解释框架是：

1. 实验 1 已经证明真实数据联邦学习链路可运行；
2. 在修复 target normalization、input normalization、CUDA 环境与常数预测问题后，FedAvg 已摆脱实现错误；
3. 增加 rounds 从 `20 -> 40 -> 60` 可以持续改善 FedAvg；
4. 但在 `5-client` 设置下，FedAvg 仍未全面超过 `NaiveLastValue`；
5. 当前主因不是实现错误，而是 `client` 异质性，尤其是 `289` 与其他 client 差异显著；
6. `Independent` 仍应被视为主学习型 baseline，而 `NaiveLastValue` 是用于刻画短时惯性的 sanity baseline，不应因当前 FedAvg 未全面超过它就直接删除；
7. `NaiveLastValue` 本身是强基线，平均 lag-1 相关性达到 `0.969279`；
8. 因此，当前更适合把问题解释为“真实数据联邦链路已闭环，但在强非 IID 条件下，标准 FedAvg 仍受跨 client 平均欠拟合影响”；
9. 当前实验 1 更准确的定位是 `grid-cell-level` 强异质性场景：`K=5` 相比原稿 `K=3` 有增强，但仍缺少 selected clients 选择依据和 cluster-level 对照；
10. 后续更合适的回应方向是 client 分组、区域 / cluster client 设计，或在实验 3 中通过更适合的 client 组织方式解释这一现象，而不是继续简单增加 rounds。

## 9. 后续修改思路

只整理思路，不改代码。

- 如果一审要求加强真实数据验证：当前实验 1 已完成基础闭环，但还需补充最终候选结果。
- 如果一审要求说明 client 划分：需要补充原稿中的 `grid-cell-level client` 与 `cluster-level client` 组织逻辑，以及为什么当前选取 `290,284,318,288,289` 作为 `K=5` 的 grid-cell-level client。
- 如果一审要求证明 FedAvg 有效：当前真实数据 5-client 结果只能说明“FedAvg 可运行且可改进”，不能直接说明“已全面优于强基线”；更适合在更同质的 client 组织或 cluster client 设定下补证据。
- 如果会议要求做消融：应继续遵守当前推进边界，在实验 1 主结果口径稳定后再进入实验 2。
- 如果会议要求强调真实数据：应把正式 tensor 数据入口、selected clients、可复现命令、v4 CUDA 环境和报告路径整理成论文可引用材料。
- 如果会议关注 non-IID：当前 `289` 异质性诊断正好可以作为解释材料，并可支撑“为什么 FedAvg 仍弱于 NaiveLastValue”；但应把 `289` 写成诊断结果，而不是随意删除某个 client 的依据。

## 10. 结论

- 本次检索到的一审 / revision 相关文件共 `19` 个，其中与“真实数据联邦学习客户端实验”直接相关的核心要求可归纳为 `5` 条。
- 本次检索到会议 / 导师意见相关文件共 `2` 个，但**未发现独立会议纪要文件**；其中与真实数据联邦学习客户端直接相关的核心要求可归纳为 `1` 条，即“真实数据实验也需要补异质性讨论”。
- 由于本次自动检索未直接覆盖原始 `docx/pdf`，上述结论在最终对外使用前仍需人工核对原始一审意见与原稿 PDF。
- 当前实验 1 已经回应的内容主要包括：
  - 真实数据联邦链路可运行；
  - `FedAvg vs Independent` 已对比；
  - `Independent` 作为主学习型 baseline 已保留，`NaiveLastValue` 已作为 sanity baseline 纳入；
  - normalization / CUDA / 常数预测问题已修复；
  - `r20 / r40 / r60` 收敛与轮次诊断已完成；
  - `client` 异质性诊断已完成，并锁定 `289` 为主要拖累 client。
- 当前实验 1 尚未完全回应的内容主要包括：
  - `FedAvg` 尚未全面超过 `NaiveLastValue`，还不能作为论文候选正式结果；
  - client 划分逻辑尚未形成论文级说明，尤其缺少为什么固定选择 `290,284,318,288,289`；
  - 当前实验 1 只是 `K=5` 的 `grid-cell-level client`，`cluster-level client` 尚未完成；
  - 消融实验（实验 2）尚未开始；
  - 真实数据侧的 `CCN/CNN` 命名与论文叙述仍需统一。
- 下一步最小动作：**先补充 client 分配检测报告，并修正对齐报告中的 client 组织表述**。

## 11. 边界声明

- 本阶段未运行训练。
- 未运行实验 2 / 3 / 4。
- 未修改 FedAvg。
- 未修改模型结构。
- 未修改数据划分。
- 未提交 `results/`。
- 只新增整理报告和一个只读检索脚本。
