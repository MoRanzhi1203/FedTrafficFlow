# 实验 1：真实数据 client 分配要求检测判断报告

## 1. 检测目的

本报告用于判断当前真实数据实验 1 的 client 分配是否符合一审意见和导师会议要求。

## 2. 一审与导师会议中的 client 分配要求

补充说明：当前自动检索基于项目内可读取的文本化材料；未直接解析原始 `docx/pdf` 原文，因此在最终对外答复前仍需人工核对原始一审 `docx/pdf`。

| 编号 | 来源 | 原始要求摘要 | 对 client 分配的含义 | 当前是否回应 |
|---|---|---|---|---|
| 1 | 一审 / revision | 真实数据 client 需要体现多区域数据分布，而不是单一局部样本。 | client 组织方式需要能说明跨区域联合建模与真实数据覆盖。 | 部分回应 |
| 2 | 一审 / 导师会议 | 需要分析区域异质性 / non-IID 对联邦训练稳定性和结果差异的影响。 | 必须补 client 分布差异、异质 client 证据与 FedAvg 受影响机制。 | 已回应 |
| 3 | 一审 / revision | 需要补充 client-level variability，而不是只给总体平均指标。 | 必须有 per-client 指标、client 级差异或主要拖累 client 诊断。 | 已回应 |
| 4 | 一审 / 原稿口径 | 需要说明 client 数量设置是否合理，并与原稿设定保持可解释的一致性。 | 要解释当前 K 的选择、增强点与局限。 | 部分回应 |
| 5 | 一审 / 原稿口径 | 需要说明 clustering procedure 或 cluster client 的组织方式。 | 不仅要有 grid-cell client，还要说明 cluster-level client 的组织逻辑。 | 未回应 |
| 6 | 一审 / revision | 需要说明 train/validation/test split 的构造方式，避免实验口径不清。 | 要明确时间连续切分与样本数边界。 | 已回应 |
| 7 | 一审 / revision | 需要解释 non-IID 下 client weights 与标准 FedAvg 的稳定性限制。 | 要说明标准样本量加权 FedAvg 在强异质条件下的表现边界。 | 部分回应 |
| 8 | 原稿真实数据设定 | 原稿需要同时区分 grid-cell-level 与 cluster-level 两类 client 组织。 | 实验 1 只能覆盖其中一类，另一类需在后续实验补齐。 | 部分回应 |

## 3. 原稿真实数据实验中的 client 组织方式

- 原稿包含 `grid-cell-level client`。
- 原稿包含 `cluster-level client`。
- `grid-cell-level`：一个 grid cell 对应一个 client。
- `cluster-level`：多个 grid cells 按时间模式相似性聚成一个 client。
- 原稿真实数据设定按当前项目口径整理为 `K=3`、`R=5`；其中 `K=3` 与现有 `region_client` 默认 `num_clients=3` 一致，但最终仍需人工核对原始稿件。
- `grid-cell-level` 强调 `local heterogeneity`。
- `cluster-level` 强调 `intra-client homogeneity`。

## 4. 当前实验 1 的 client 分配事实

- 当前 `selected_clients = 290,284,318,288,289`。
- 当前 client 数量 = `5`。
- 当前 workflow = `all`，rounds = `20`，device = `cuda`。
- 当前属于 grid-cell-level client：`是`。
- 当前不是 cluster-level client：`是`。
- 当前已完成 `FedAvg vs Independent`。
- 当前已补充 `NaiveLastValue`。
- 当前已完成 `r20/r40/r60` rounds 诊断。
- 当前已完成 client 异质性诊断：`是`。
- 当前已发现 `289` 是主要拖累 client：`是`。

当前 5 个 selected clients 的空间与数据统计如下：

| region_id | pooled_row | pooled_col | centroid_lon | centroid_lat | source_node_count | mean_total_flow | train_samples | val_samples | test_samples |
|---|---|---|---|---|---|---|---|---|---|
| 290 | 9 | 20 | 116.465194 | 39.91456970750988 | 667 | 1953917.875 | 4087 | 878 | 879 |
| 284 | 9 | 14 | 116.357194 | 39.91456970750988 | 698 | 1914353.125 | 4087 | 878 | 879 |
| 318 | 10 | 18 | 116.429194 | 39.93256970750988 | 711 | 1857832.25 | 4087 | 878 | 879 |
| 288 | 9 | 18 | 116.429194 | 39.91456970750988 | 698 | 1702061.25 | 4087 | 878 | 879 |
| 289 | 9 | 19 | 116.447194 | 39.91456970750988 | 663 | 1659179.75 | 4087 | 878 | 879 |

补充判断：

- 当前 `selected_clients` 是否与审计输入一致：`是`。
- 当前是否已有 train/val/test split 证据：`是`。
- 当前是否已有 client 选择依据：`否`。

## 5. 当前设置与一审/导师要求的符合程度

| 要求 | 当前状态 | 判断 | 缺口 | 修复方式 |
|---|---|---|---|---|
| 当前 5-client 设置是否比原稿 K=3 有增强 | 当前 `num_clients=5`，原稿口径按 K=3 组织。 | 已满足 | 仍需人工核对原始 PDF / docx 中 K=3 表述。 | 在文档中明确把 K=5 写成对 K=3 的增强，而非与原稿冲突。 |
| 当前 client 数量是否仍可能被认为偏少 | 5 个真实数据 client 已高于 K=3，但总体仍属小规模真实数据 FL。 | 需要说明限制 | 可能仍被审稿人认为覆盖范围有限。 | 明确写出“增强但仍是小规模真实数据 FL”这一限制。 |
| 当前是否已充分说明为什么选这 5 个 client | 已明确固定 ID，但缺少论文级选择依据。 | 未满足 | 缺少 selected_clients 选择逻辑、筛选准则和与原稿叙事的映射。 | 补充选择依据与固定 5 个 grid cells 的设计说明。 |
| 当前是否已充分说明空间覆盖 | 已有 pooled_row / pooled_col / centroid / source_node_count 事实。 | 部分满足 | 已有事实表，但尚未转化为空间覆盖解释。 | 在报告中补 5 个 grid cells 的空间覆盖与流量分布说明。 |
| 当前是否已充分说明 cluster procedure | 项目内已有后续多网格客户端迁移口径，但实验 1 未展开。 | 未满足 | 当前实验 1 没有 grouped-client / global-partition 结果或正式说明。 | 明确 grouped-client / global-partition 客户端实验线由新实验 3-6 分别承接，当前仅说明原稿组织方式。 |
| 当前是否已经完成 cluster-level client 实验 | 当前 result_dir 为 `grid_cell_main_full_cuda_v4`，且对齐报告已说明尚未完成。 | 未满足 | 缺少 cluster-level 实验结果与报告。 | 不要在实验 1 中冒充已完成；保留为后续新实验 3-6 对比/消融线补齐。 |
| 当前是否已经回应 non-IID | 已有 gap diagnosis 与 heterogeneity diagnosis，并锁定 289。 | 已满足 | 需要把结论转写为论文级表述。 | 把 289 异质性写成诊断结果和 FedAvg 局限解释。 |
| 当前是否已经回应 client-level variability | 已有 per-client 指标、pairwise correlation、leave-one-client-out 统计。 | 已满足 | 正文尚缺精炼版表格。 | 补一张 client-level variability 汇总表。 |
| 当前实验 1 是否满足一审 / 导师要求 | 已有真实数据闭环、split 证据、non-IID 证据，但缺选择依据与 cluster-level 补充。 | 部分满足 | selected_clients 依据不足，cluster-level 未完成。 | 先补文档和审计报告，再决定后续实验方向。 |

## 6. 需要修复优化的点

- 补充 `selected_clients` 选择依据，不只给出固定 ID。
- 补充 5 个 grid cells 的空间覆盖和流量分布统计。
- 补充 `client-level variability` 表，保留 per-client 指标与差异摘要。
- 把 `289` 异质性写成“诊断结果”，不要写成随意删除依据。
- 说明 5-client 是对原稿 `K=3` 的增强，但仍属于小规模真实数据 FL。
- 说明 grouped-client / global-partition 客户端实验线还未完成，后续由新实验 3-6 分别承接。
- 统一 `CCN/CNN` 表述，真实数据实验 1 以 `CNN-LSTM-Attention` 为准。
- 明确 `NaiveLastValue` 是 sanity baseline，不是原始主对比 baseline。
- 明确 `Independent` 仍是主学习型 baseline。
- 不要声称 `FedAvg` 已全面超过所有 baseline。

## 7. 修复后的推荐实验叙事

- 当前实验 1 应定位为 `grid-cell-level fine-grained heterogeneous client setting`。
- 该设置用于验证真实数据联邦链路，并暴露强 non-IID 下标准 FedAvg 的局限。
- `selected_clients=290,284,318,288,289` 将原稿 `K=3` 扩展为 `K=5`，增强了 `client-level variability` 分析。
- 其中 `289` 显示出明显异质性，是解释 FedAvg 未全面超过 `NaiveLastValue` 的关键证据。
- 后续新实验 3/4 的 grouped-client 组织与新实验 5/6 的全局覆盖式划分，将用于验证更同质或更系统的 client 组织是否能缓解 FedAvg 的跨 client 平均欠拟合问题。

## 8. 是否需要立即修改实验

- 不建议现在直接删除 `NaiveLastValue`。
- 不建议现在直接删除 `289`。
- 不建议现在直接进入实验 2。
- 不建议现在改 `FedAvg`。
- 建议先补齐 client 分配依据和诊断报告。
- 下一步可做 `leave-one-client-out / 4-client smoke`，或转入新实验 3-6 的 grouped-client / global-partition 审计。

## 9. 结论

- 一审 / 导师没有要求固定 client ID。
- 一审 / 导师真正要求的是 client 分配逻辑、异质性证据、client 数量说明、clustering procedure、client-level variability。
- 当前实验 1 已部分满足。
- 当前最大缺口是 `selected_clients` 选择依据，以及新实验 3-6 的 grouped-client / global-partition 客户端实验线尚未完成。
- 当前最小修复动作是补充 client 分配检测报告，并修正对齐报告中的相关表述。

## 10. 边界声明

- 本阶段未运行训练。
- 未运行新实验 2-6。
- 未修改 FedAvg。
- 未修改模型结构。
- 未修改数据划分。
- 未提交 `results/`。

