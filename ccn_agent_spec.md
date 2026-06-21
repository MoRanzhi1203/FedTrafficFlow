# CCN Federated Learning Agent Specification / CCN 联邦学习 Agent 规范说明

## 1. Data Pipeline / 数据链路规范

### 1.1 Current Verified Pipeline / 当前已核实链路

**中文：**基于当前代码与已执行的预处理结果，CCN 真实联邦训练的数据链路已经统一为从节点流量 parquet 中间数据出发，经网格化与张量化，最终落到唯一训练输入 `node_flow_grid_tensor.pt`。

**English:** Based on the current codebase and verified preprocessing runs, the CCN federated workflow is now unified from node-flow parquet intermediates to grid features and finally to the single training tensor `node_flow_grid_tensor.pt`.

```text
data/analysis/node_intersection_flow_parquet/*.parquet
-> preprocessing_scripts/process_node_flow_grids.py
-> data/processed/node_flow_grid/node_flow_grid_2ch.npy
-> data/processed/node_flow_grid/node_flow_grid_pooled.npy
-> preprocessing_scripts/process_node_flow_tensor.py
-> data/processed/node_flow_grid/node_flow_grid_tensor.pt
-> analysis_scripts/federated_learning/ccn_region_client_train.py
-> analysis_scripts/federated_learning/ccn_region_client_ablation.py
```

**中文：**当前最终训练张量已核实为 `torch.Tensor`，形状为 `(2, 928, 5856)`。其中通道 0 为池化后流量和特征，通道 1 为池化后流量均值特征；`928 = 29 x 32` 表示池化后的空间区域数，`5856 = 61 x 96` 表示 61 天、每天 96 个 15 分钟时段。

**English:** The current final training tensor is a `torch.Tensor` with shape `(2, 928, 5856)`. Channel 0 stores pooled flow-sum features, channel 1 stores pooled flow-mean features, `928 = 29 x 32` corresponds to flattened pooled spatial regions, and `5856 = 61 x 96` corresponds to 61 days with 96 slots per day.

### 1.2 Data Layer Definitions / 数据层级定义

**中文：**CCN 联邦子系统必须采用三层数据定义，且各层职责不能混淆。

**English:** The CCN federated subsystem must use a strict three-layer data hierarchy with clearly separated responsibilities.

- Raw / 原始层: `data/analysis/node_intersection_flow_parquet`
- Intermediate / 中间层: `data/processed/node_flow_grid/node_flow_grid_2ch.npy`
- Final training dataset / 最终训练集: `data/processed/node_flow_grid/node_flow_grid_tensor.pt`

**中文：**虽然 parquet 已经是上游处理过的中间结果，但对 CCN 联邦训练而言，它仍然只能被视为“原始输入层”，不能直接进入训练阶段。

**English:** Although the parquet files are already derived from upstream preprocessing, they must still be treated as the raw source for the CCN federated subsystem and must not be read directly by the training stage.

### 1.3 Mandatory Data Rules / 强制数据规则

**中文：**

- 联邦训练只能读取最终数据集 `node_flow_grid_tensor.pt`。
- 训练脚本与消融脚本禁止直接读取 parquet。
- 禁止绕过 `process_node_flow_tensor.py` 手动构造训练张量。
- 任何重新训练前，必须先核验 `node_flow_grid_tensor.pt` 是否存在且为最新版本。
- 若最终张量缺失或过旧，必须重新执行：
  - `preprocessing_scripts/process_node_flow_grids.py`
  - `preprocessing_scripts/process_node_flow_tensor.py`

**English:**

- Federated training must only read the final dataset `node_flow_grid_tensor.pt`.
- Direct parquet loading inside training or ablation code is prohibited.
- Bypassing `process_node_flow_tensor.py` to manually reconstruct training tensors is prohibited.
- Any retraining request must first verify the existence and freshness of `node_flow_grid_tensor.pt`.
- If the final tensor is missing or stale, the system must rerun:
  - `preprocessing_scripts/process_node_flow_grids.py`
  - `preprocessing_scripts/process_node_flow_tensor.py`

### 1.4 Data Validation Requirements / 训练前数据校验

**中文：**训练开始前，Agent 必须完成以下最小校验。

**English:** Before training starts, the Agent must perform the following minimum checks.

- dataset path exists / 数据路径存在
- file type is `torch.Tensor` / 文件类型为 `torch.Tensor`
- tensor rank is exactly 3 / 张量维度必须为 3
- tensor dtype is floating-point / 张量类型必须为浮点型
- tensor contains no `NaN` or `Inf` / 张量中不得含 `NaN` 或 `Inf`
- total time steps are larger than `t_in + t_out` / 总时间步必须大于 `t_in + t_out`
- region count is not smaller than requested client count / 区域数不得小于客户端数

### 1.5 Metadata Gap / 当前元数据缺口

**中文：**当前流程没有把池化区域的空间元数据与 `node_flow_grid_tensor.pt` 一起持久化。这是当前客户端划分难以做到“真正空间可解释”的主要缺口。后续若要重构客户端划分，应优先补充轻量 sidecar 元数据文件，而不是直接引入更复杂模型。

**English:** The current pipeline does not persist pooled-region spatial metadata together with `node_flow_grid_tensor.pt`. This is the main gap that prevents a fully spatially grounded client-partition strategy. A lightweight sidecar metadata file should be added before introducing more complex client-construction schemes.

Recommended future sidecar / 推荐未来补充文件:

- `data/processed/node_flow_grid/node_flow_grid_regions.csv`

Recommended fields / 推荐字段:

- `region_id`
- `grid_row`
- `grid_col`
- `centroid_lon`
- `centroid_lat`
- `source_node_count`

## 2. Client Construction Strategy / 客户端构造策略

### 2.1 Current Real Status / 当前真实实现状态

**中文：**当前客户端并不是基于原始节点、原始 parquet 分片或固定地理区域构造的，而是基于池化后的 grid tensor 区域进行时间统计后聚类。

**English:** The current clients are not constructed from raw nodes, raw parquet partitions, or fixed geographic regions. Instead, they are built from pooled grid-tensor regions using temporal/statistical clustering.

Current implementation / 当前实现:

- training reads `node_flow_grid_tensor.pt`
- each flattened pooled grid cell is treated as one region
- temporal/statistical features are extracted for each region
- KMeans-like clustering is applied to those features
- cluster sizes are balanced by estimated sample counts

Therefore, current clients are based on / 因此当前客户端本质上基于:

- pooled grid tensor regions / 池化后网格区域
- temporal/statistical clustering / 时间统计特征聚类
- not direct node clustering / 不是节点级聚类
- not explicit lat/lon spatial clustering / 不是显式经纬度空间聚类

### 2.2 Current Risks / 当前主要风险

#### A. Client Distribution Imbalance / 客户端分布不均衡

**中文：**当前代码虽然通过样本量平衡对极端不均衡做了部分抑制，但这种平衡主要作用于估计样本负载，而不是空间覆盖、交通模式多样性、峰谷结构或拓扑连续性。因此“样本数平衡”不等于“客户端分布真正均衡”。

**English:** The current code partially mitigates extreme imbalance through size balancing, but that balancing mainly operates on estimated sample load rather than spatial coverage, traffic-regime diversity, peak/off-peak structure, or topology continuity. Sample-count balancing is therefore not equivalent to true client-distribution balance.

Conclusion / 结论:

- sample-count imbalance is partially controlled / 样本量不均衡得到部分控制
- distributional balance is not fully guaranteed / 分布层面的均衡仍未真正保证

#### B. Spatial Leakage / 空间信息泄漏风险

**中文：**当前客户端划分的聚类目标是时间相似性，而不是空间连续性，因此相邻区域可能被分配到不同客户端，一个客户端内部也可能由多个地理上不连续的碎片区域组成。这会削弱“区域联邦学习”的空间解释性。

**English:** Because the clustering objective is temporal similarity rather than spatial continuity, nearby pooled cells may be assigned to different clients and a single client may contain multiple geographically disconnected fragments. This weakens the spatial interpretability of “regional federated learning.”

**中文：**这不是传统意义上的 train-test leakage，而是客户端构造阶段的结构性空间泄漏。

**English:** This is not classic train-test leakage, but a structural spatial-leakage issue during client construction.

#### C. Grid Homogenization / 网格粒度同质化风险

**中文：**网格池化将原始节点空间压缩成 `928` 个区域，虽然有利于训练，但也会带来局部拓扑细节被削弱、多节点异质性被平均化的问题。如果客户端划分完全在池化之后执行，就可能低估客户端异构性。

**English:** Grid pooling compresses the original node space into `928` pooled regions. This is efficient for training, but it also weakens local topology detail and averages out part of the original heterogeneity. If client partitioning is performed only after pooling, the degree of client heterogeneity may be underestimated.

### 2.3 Critical Decision / 关键判断

**中文：**当前 CCN 联邦学习是否需要重新设计客户端划分？

**English:** Does the current CCN federated learning system need a redesigned client-partition strategy?

**Answer / 结论: YES**

### 2.4 Recommended Strategy / 推荐策略

Priority order / 推荐优先级:

1. `spatial clustering (lat/lon)` as the primary production strategy  
   `spatial clustering (lat/lon)` 作为首选正式策略
2. `KMeans on node embeddings` only after region metadata or node embeddings are persisted  
   `KMeans on node embeddings` 仅在节点或区域嵌入与元数据可持久化后再采用
3. `grid-based fixed partition` as a reproducible baseline  
   `grid-based fixed partition` 作为强可复现基线
4. `temporal clustering` as an optional auxiliary analysis, not the default partition rule  
   `temporal clustering` 只作为辅助分析，不应再作为默认划分规则

### 2.5 Recommended Production Rule / 推荐正式规则

**中文：**下一版稳定 CCN 系统中，客户端构造应优先基于空间坐标或池化区域质心，再叠加轻量负载平衡，避免继续把“时间相似性”作为主划分依据。

**English:** In the next stable CCN revision, client construction should first use spatial coordinates or pooled-region centroids, and then apply lightweight load balancing. Temporal similarity should no longer be the primary partition rule.

Recommended production client strategy / 推荐正式客户端策略:

```text
Spatial clustering (lat/lon) + load balancing
```

### 2.6 Recommended Experimental Baselines / 推荐对比基线

**中文：**为满足论文与审稿回复需要，至少保留以下三类客户端划分对比。

**English:** To support reviewer-facing analysis, at least the following three partition baselines should be compared.

- current temporal-feature clustering baseline / 当前时间特征聚类基线
- spatial clustering baseline / 空间聚类基线
- fixed grid partition baseline / 固定网格划分基线

Optional additional comparison / 可选补充:

- spatial clustering + temporal feature refinement / 空间聚类 + 时间特征微调

## 3. Federated Training Protocol / 联邦训练流程规范

### 3.1 Current Verified Training Pattern / 当前已验证训练模式

**中文：**当前训练系统已经形成可运行基线，包括固定最终输入张量、时间顺序滑动窗口、区域聚类客户端划分、本地训练、FedProx 正则、加权聚合、个性化微调以及与独立训练的对比。

**English:** The current training system already forms a usable baseline, including a fixed final tensor input, chronological sliding windows, region clustering for client assignment, local training, FedProx regularization, weighted aggregation, personalization, and comparison against independent local models.

### 3.2 Standard Training Input / 标准训练输入

**中文：**当前标准样本构造为：

- 输入 `x = data[:, region_id, t : t + t_in]`
- 目标 `y = data[0, region_id, t + t_in]`

当前预测目标只使用通道 0，除非后续明确实现多目标预测协议，否则该规则保持不变。

**English:** The standard sample definition is:

- input `x = data[:, region_id, t : t + t_in]`
- target `y = data[0, region_id, t + t_in]`

The current target uses channel 0 only. This rule should remain fixed unless a documented multi-target forecasting protocol is added later.

### 3.3 Time Split Constraints / 时间划分约束

**中文：**训练、验证和测试必须严格保持时间顺序。

**English:** Training, validation, and testing must strictly preserve chronological order.

Allowed / 允许:

- train -> val -> test in chronological order

Not allowed / 禁止:

- random shuffle before temporal split
- mixing future slots into earlier training windows

### 3.4 Default Training Protocol / 标准执行顺序

**中文：**

1. 从 `node_flow_grid_tensor.pt` 读取最终张量
2. 校验张量结构与数值合法性
3. 构造客户端划分
4. 构造按时间顺序的滑动窗口
5. 在各客户端内完成 train/val/test 划分
6. 初始化共享全局模型
7. 进行多轮联邦训练
8. 分别评估全局模型与个性化模型
9. 保存配置、历史、客户端指标与图表

**English:**

1. load the final tensor from `node_flow_grid_tensor.pt`
2. validate the tensor structure and values
3. build client partitions
4. construct chronological sliding windows
5. split train/val/test within each client
6. initialize one shared global model
7. run federated rounds
8. evaluate global and personalized models separately
9. save configs, histories, client metrics, and figures

### 3.5 Aggregation Rule / 聚合规则

**中文：**在未验证更复杂替代方案之前，正式系统仍保持当前聚合规则：本地目标为 FedProx 风格目标，服务端聚合为样本量加权 FedAvg，并通过 `server_lr` 做服务器侧插值更新。

**English:** Until a fully validated alternative is implemented, the production aggregation rule remains the current one: a FedProx-style local objective and sample-count-weighted FedAvg server aggregation with server-side interpolation via `server_lr`.

The Agent must not silently replace this with / Agent 不得私自替换为:

- similarity-aware aggregation
- representation alignment
- meta-learning aggregation
- adaptive graph aggregation

unless these are separately implemented, tested, and documented / 除非这些方法被单独实现、验证并文档化。

### 3.6 Personalization Rule / 个性化规则

**中文：**个性化阶段是当前 CCN 系统的有效组成部分，应继续保留，但必须和全局模型、独立本地模型的结果显式区分开。

**English:** Personalization is a valid part of the current CCN system and should be preserved, but it must be reported separately from the global federated model and independent local models.

Required reporting separation / 必须区分报告:

- global federated model performance / 全局联邦模型性能
- personalized model performance / 个性化微调模型性能
- independent local model performance / 独立本地模型性能

## 4. Evaluation Metrics / 指标体系设计

### 4.1 Current Status / 当前状态

**中文：**当前代码已输出 `MSE`、`RMSE`、`MAE`，但对于联邦学习论文或审稿回复来说，这一指标集合仍然不够完整。

**English:** The current code reports `MSE`, `RMSE`, and `MAE`, which is a useful starting point but still insufficient for a reviewer-facing federated evaluation framework.

### 4.2 Mandatory Global Metrics / 全局指标

Mandatory metrics / 强制指标:

- MAE
- RMSE
- MAPE

Definitions / 定义:

- `MAE`: mean absolute error over all test samples / 全部测试样本上的平均绝对误差
- `RMSE`: square root of mean squared error / 均方误差平方根
- `MAPE`: masked mean absolute percentage error / 带掩码的平均绝对百分比误差

**中文：**对于 `MAPE`，零流量目标必须做掩码处理，仅在 `target > epsilon` 时参与计算，推荐 `epsilon = 1e-6`。

**English:** For `MAPE`, zero-flow targets must be masked out so that only targets greater than `epsilon` participate in the calculation. A recommended choice is `epsilon = 1e-6`.

### 4.3 Mandatory Client-Level Metrics / 客户端指标

For each client / 每个客户端必须输出:

- client MAE
- client RMSE
- client MAPE
- client sample count / 客户端样本数
- client region count / 客户端区域数

Additional federated fairness indicators / 额外联邦公平性指标:

- `client variance`
- `performance skew`

Definitions / 定义:

- `client variance`: variance of client RMSE across clients / 各客户端 RMSE 方差
- `performance skew`: `max(client_rmse) / min(client_rmse + epsilon)` or P90-P10 gap / 最大最小误差偏斜或 P90-P10 差距

### 4.4 Federated-Specific Metrics / 联邦特有指标

Mandatory federated metrics / 强制联邦指标:

- convergence speed / 收敛速度
- communication efficiency / 通信效率
- personalization gap / 个性化收益差

Definitions / 定义:

- `convergence speed`: earliest round where validation RMSE reaches within 5% of the best final RMSE / 验证 RMSE 首次进入最终最优值 5% 范围内的轮次
- `communication efficiency`: performance gain per transmitted model size / 单位通信量带来的性能提升
- `personalization gap`: `global_rmse - personalized_rmse` / 全局 RMSE 与个性化 RMSE 的差值

Recommended communication estimator / 推荐通信估算:

```text
total_comm_mb = rounds * active_clients * model_parameter_bytes * 2 / 1024^2
efficiency = (independent_rmse - personalized_rmse) / total_comm_mb
```

### 4.5 Stability Reporting / 稳定性报告要求

**中文：**正式实验表格至少要报告均值、标准差和随机种子列表，推荐最终结论至少基于 3 个随机种子。单种子结果只能用于 smoke test 或趋势性参考，不足以支撑正式结论。

**English:** Formal result tables should report mean, standard deviation, and the seed list. At least 3 random seeds are recommended for final conclusions. Single-seed results are acceptable for smoke tests or trend checks only.

## 5. Ablation Study Design / 消融实验规范

### 5.1 Current Real Status / 当前真实状态

**中文：**当前消融代码已经覆盖四种结构变体：

- Full (`CNN + LSTM + Attention`)
- w/o Attention (`CNN + LSTM`)
- w/o CNN (`LSTM + Attention`)
- w/o LSTM (`CNN + Attention`)

该结构消融基础有效，应继续保留。

**English:** The current ablation code already compares four architecture variants:

- Full (`CNN + LSTM + Attention`)
- w/o Attention (`CNN + LSTM`)
- w/o CNN (`LSTM + Attention`)
- w/o LSTM (`CNN + Attention`)

This is a sound ablation basis and should be retained.

### 5.2 Mandatory Ablation Constraints / 强制消融约束

All ablations must share the same / 所有消融必须共享相同的:

- final dataset input / 最终训练数据入口
- client partition / 客户端划分
- train/val/test split / 训练验证测试划分
- random seed / 随机种子
- evaluation metrics / 评估指标
- output format / 输出格式

**中文：**消融只能改变模型结构或被声明的实验轴，不能偷偷更换数据路径或划分方式。

**English:** Ablation is allowed to change model structure or the explicitly declared study axis only; it must not silently alter the data path or split protocol.

### 5.3 Optimization Budget Constraint / 训练预算一致性

**中文：**为保证公平性，消融实验必须使用统一预算：相同通信轮数、相同本地 epoch、相同 batch size、相同学习率策略。若故意修改其中一项，必须另列为独立预算实验，而不能混入主表。

**English:** To ensure fairness, ablation runs must use a standardized optimization budget: the same communication rounds, local epochs, batch size, and learning-rate policy. If any of these are intentionally changed, the run must be documented as a separate budget condition rather than mixed into the main ablation table.

### 5.4 Required Ablation Axes / 建议保留的消融轴

Architecture ablation / 结构消融:

- Full
- w/o Attention
- w/o CNN
- w/o LSTM

Federated mechanism ablation / 联邦机制消融:

- personalization on vs off / 个性化开关
- FedProx on vs off / FedProx 开关
- temporal clustering vs spatial clustering / 时间聚类 vs 空间聚类

Sensitivity ablation / 敏感性消融:

- `num_clients`
- `server_lr`
- `mu`
- `t_in`
- `stride`

## 6. System Constraints / 系统约束

### 6.1 Reproducibility Constraints / 可复现性约束

The Agent must always / Agent 必须始终:

- save the full config as JSON / 保存完整 JSON 配置
- save the exact dataset path / 保存精确数据路径
- record the final tensor shape / 记录最终张量形状
- record the random seed / 记录随机种子
- keep chronological split / 保持时间顺序划分
- keep client partition reproducible / 保证客户端划分可复现
- record runtime device and software versions when possible / 条件允许时记录设备与软件版本

### 6.2 Data Access Constraints / 数据访问约束

The following are prohibited / 以下行为禁止:

- parquet direct read inside federated training
- manual reconstruction of training tensors in training code
- mixed use of parquet and tensor in one experiment
- swapping to an undocumented alternative `.pt` file

### 6.3 Client Partition Constraints / 客户端划分约束

Required reporting / 必须记录:

- partition method / 划分方法
- number of clients / 客户端数量
- region counts and sample counts per client / 各客户端区域数与样本数
- centroid source if spatial clustering is used / 使用空间聚类时须说明质心来源

### 6.4 Reporting Constraints / 输出报告约束

Every formal experiment must save / 每个正式实验至少保存:

- config JSON
- convergence history CSV
- client metric CSV
- summary JSON or CSV
- at least one convergence figure / 至少一张收敛图

Paper-ready experiments should additionally save / 论文级实验建议额外保存:

- per-client metric table / 客户端明细表
- multi-seed summary table / 多种子汇总表
- communication estimate table / 通信估算表
- global vs personalized comparison table / 全局与个性化对比表

### 6.5 Scope Constraints / 范围约束

**中文：**当前正式 CCN 路径中，不应引入未经验证的复杂新框架。以下内容仅能作为未来工作或单独实验，不属于当前标准系统：

- adaptive dynamic graphs
- meta-learning personalization
- domain adaptation modules
- representation alignment aggregators
- differential privacy training
- client-dropout simulation as production default

**English:** Unverified complex frameworks should not be introduced into the standard CCN production path. The following items remain future-work options or separate experiments only:

- adaptive dynamic graphs
- meta-learning personalization
- domain adaptation modules
- representation alignment aggregators
- differential privacy training
- client-dropout simulation as a production default

## 7. Recommendations (Critical Decisions) / 关键建议与决策

### 7.1 Decision Summary / 决策摘要

- Re-cluster clients: **YES** / 是否重构客户端划分：**YES**
- Primary strategy: **spatial clustering (lat/lon)** / 首选策略：**spatial clustering (lat/lon)**
- Future option: **KMeans on node or region embeddings** / 后续扩展：**KMeans on node or region embeddings**
- Reproducible baseline: **grid-based fixed partition** / 可复现基线：**grid-based fixed partition**
- Optional diagnostic mode: **temporal clustering** / 可选诊断模式：**temporal clustering**

### 7.2 Why This Decision Is Necessary / 为什么必须这样决策

**中文：**之所以需要重构客户端划分，是因为当前训练输入已经是池化网格区域张量，而现有客户端又主要按时间统计特征聚类构造，导致空间语义较弱、异质性表达不足、客户端级分析说服力不够。这正是当前代码状态与审稿要求之间的关键缺口。

**English:** A client-partition redesign is necessary because the current training input is already a pooled-region tensor, while client assignment is still driven mainly by temporal-statistical clustering. This weakens spatial semantics, under-represents heterogeneity, and reduces the explanatory power of client-level analysis. That is the central gap between the current implementation and reviewer expectations.

### 7.3 Immediate Production Recommendations / 立即执行建议

1. Use `node_flow_grid_tensor.pt` as the only training input.  
   仅使用 `node_flow_grid_tensor.pt` 作为训练输入。
2. Treat current temporal-feature KMeans partition as a baseline, not the long-term default.  
   将当前时间特征 KMeans 视为基线，而非长期默认方案。
3. Add client-partition metadata outputs to every federated run.  
   每次联邦运行都输出客户端划分元数据。
4. Extend evaluation from `{MSE, RMSE, MAE}` to `{MAE, RMSE, MAPE}` plus client variance, performance skew, convergence speed, communication efficiency, and personalization gap.  
   将评估从 `{MSE, RMSE, MAE}` 扩展到 `{MAE, RMSE, MAPE}`，并补充客户端方差、性能偏斜、收敛速度、通信效率和个性化收益差。
5. Keep architecture ablations fixed to the same data split and client split.  
   保证结构消融使用同一数据划分和同一客户端划分。
6. Report at least one multi-seed result set for formal claims.  
   正式结论至少提供一组多随机种子结果。

### 7.4 Near-Term Engineering Priorities / 近期工程优先级

- persist pooled-region centroid metadata / 持久化池化区域质心元数据
- implement spatial client partition as an alternative mode / 新增空间聚类客户端划分模式
- add global pre-personalization evaluation / 增加个性化前全局模型评估
- add MAPE and federated-specific metrics to outputs / 在输出中增加 MAPE 与联邦特有指标
- standardize experiment tables for reviewer response / 统一审稿回复与论文图表用表

### 7.5 Explicit Non-Recommendations / 明确不建议事项

The Agent should not currently / 当前 Agent 不应:

- switch back to parquet-based training input / 回退到 parquet 直接训练
- introduce dynamic graph models without verified preprocessing support / 在预处理未打通前引入动态图主线
- replace aggregation with a complex new algorithm without ablation evidence / 无消融证据就更换复杂聚合算法
- claim node-level federated semantics when the real input is pooled-region tensor data / 在实际输入为池化区域张量时声称“节点级联邦语义”

### 7.6 Final Position / 最终立场

**中文：**当前 CCN 联邦系统已经具备可运行的 tensor-only 基线训练链路，但尚未成为完全审稿就绪的标准系统。核心问题已经不再是“能不能训练”，而是“客户端语义是否合理、指标体系是否足够联邦化、报告是否足够规范”。

**English:** The current CCN federated system is now a working tensor-only baseline, but it is not yet a fully reviewer-ready standard system. The central issue is no longer whether the system can train, but whether client semantics are meaningful, evaluation is federated-aware, and reporting is standardized.

Official position / 官方规范立场:

```text
Keep the tensor-only training path.
Redesign client partition toward spatially meaningful regions.
Upgrade evaluation from basic prediction error to federated-aware analysis.
Use the current implementation as the controlled baseline.
```
