# 实验 1：selected_clients 选择依据说明报告

## 1. 报告目的

本报告回答：
- 为什么当前实验 1 使用 `selected_clients=290,284,318,288,289`；
- 为什么不是更少；
- 为什么不是更多；
- 为什么不是其他 client；
- 该组合在一审/导师要求下的定位是什么。

## 2. 当前实验 1 的 client 设置事实

- `selected_clients`：`290,284,318,288,289`
- client 数量：`5`
- `result_dir`：`E:\Jupter_Notebook\FedTrafficFlow\results\real_data_experiments\formal\grid_cell_main_full_cuda_v4`
- `workflow`：`all`
- `rounds`：`20`
- `device`：`cuda`
- `split_summary`：`temporal_contiguous_by_target_time`，`train=[0,4099)`，`val=[4099,4977)`，`test=[4977,5856)`

当前 `client_metrics`、`split_summary` 与 tensor 统计合并后的 per-client 信息如下：

| region_id | train_samples | val_samples | test_samples | train_split | val_split | test_split | source_node_count | mean_total_flow | series_mean | series_std | series_min | series_max | cv | lag1_autocorr | fedavg_rmse | independent_rmse | naive_rmse | fedavg_r2 | independent_r2 | naive_r2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 284 | 4087 | 878 | 879 | [0,4099) | [4099,4977) | [4977,5856) | 698 | 1914353.125 | 1914353.015 | 107014.851 | 1572611.125 | 2056486.500 | 0.055901 | 0.971499 | 22531.650 | 18648.343 | 24846.927 | 0.953147 | 0.967905 | 0.943023 |
| 288 | 4087 | 878 | 879 | [0,4099) | [4099,4977) | [4977,5856) | 698 | 1702061.250 | 1702061.233 | 78704.939 | 1446559.375 | 1819865.750 | 0.046241 | 0.966661 | 18842.817 | 14672.743 | 19297.015 | 0.936442 | 0.961461 | 0.933341 |
| 289 | 4087 | 878 | 879 | [0,4099) | [4099,4977) | [4977,5856) | 663 | 1659179.750 | 1659179.471 | 38846.188 | 1315587.625 | 1716224.125 | 0.023413 | 0.960662 | 26184.123 | 9597.820 | 11065.790 | 0.559750 | 0.940848 | 0.921370 |
| 290 | 4087 | 878 | 879 | [0,4099) | [4099,4977) | [4977,5856) | 667 | 1953917.875 | 1953917.774 | 86680.388 | 1612853.375 | 2059067.250 | 0.044362 | 0.971308 | 20844.986 | 16203.385 | 20708.839 | 0.941890 | 0.964888 | 0.942647 |
| 318 | 4087 | 878 | 879 | [0,4099) | [4099,4977) | [4977,5856) | 711 | 1857832.250 | 1857832.287 | 94299.980 | 1557965.875 | 1981110.125 | 0.050758 | 0.976262 | 15675.444 | 14344.552 | 21177.513 | 0.973999 | 0.978226 | 0.952542 |

当前 5 个 client 的 Pearson 相关性矩阵如下：

| region_id | 290 | 284 | 318 | 288 | 289 |
|---|---|---|---|---|---|
| 290 | 1.000000 | 0.877496 | 0.902350 | 0.885950 | 0.203391 |
| 284 | 0.877496 | 1.000000 | 0.914440 | 0.920636 | 0.034876 |
| 318 | 0.902350 | 0.914440 | 1.000000 | 0.938005 | 0.060115 |
| 288 | 0.885950 | 0.920636 | 0.938005 | 1.000000 | 0.092672 |
| 289 | 0.203391 | 0.034876 | 0.060115 | 0.092672 | 1.000000 |

## 3. 一审/导师对 client 数量和分配逻辑的约束

- 一审/导师没有要求固定 client ID。
- 要求的是多区域数据分布、异质性、client-level variability、client 数量说明、clustering procedure。
- 原稿 `K=3` 被认为偏小，当前 `K=5` 是增强，但仍属于小规模真实数据 FL。
- 当前自动检索基于项目内可读取材料；若需对外正式表述，仍应人工核对原始 `docx/pdf`。

## 4. 为什么是这 5 个 client

- 这 5 个 client 是当前实验 1 正式 v4 CUDA 链路中已经完整跑通的 `selected_clients`。
- 它们已有完整的 `FedAvg vs Independent`、`NaiveLastValue`、`r20/r40/r60`、client 异质性诊断证据。
- 它们形成了一个 `K=5 grid-cell-level heterogeneous setting`。
- `K=5` 相比原稿 `K=3` 增强了 `client-level variability`。
- 其中 `289` 提供了强异质 client 证据，可以解释标准 `FedAvg` 在 non-IID 下的局限。
- 因此这 5 个适合作为当前阶段“细粒度异质 client 设置”的审计对象。

当前可直接支撑上述判断的证据包括：
- 当前 5 个 client 已在 v4 CUDA 正式链路中完整跑通，并在 `run_config.json` 与 `split_summary.json` 中可复现。
- 当前 5 个 client 已具备 `FedAvg vs Independent`、`NaiveLastValue`、`r20/r40/r60` 和 client 异质性诊断链路。
- 当前 client 数量为 `5`，相对原稿口径中的 `K=3` 已形成增强。
- 5 个 client 的最小 Pearson 相关性为 `0.034876`，说明该组合具有明显 non-IID。
- region `289` 被现有异质性诊断识别为主要拖累 client。
- test split 平均 lag-1 自相关为 `0.969279`，说明短时惯性 baseline 较强。
- 当前文档链已明确：K=5 是增强后的 grid-cell-level 异质设置，但仍属于小规模真实数据 FL。
- 5 个 client 都是 active pooled-grid regions，且均有非零 `source_node_count`。

## 5. 为什么不是更少

- 原稿 `K=3` 已被认为偏小。
- 更少 client 会削弱多区域联邦学习属性。
- 更少 client 不利于分析 `client-level variability`。
- 更少 client 可能掩盖 `289` 这类异质 client 对 `FedAvg` 的影响。
- 因此当前不建议回退到 `K=3` 或更少。

## 6. 为什么不是更多

- 更多 client 需要新实验。
- 当前阶段只解释已有实验。
- 更多 client 会改变实验边界。
- 当前已有完整诊断链路的是这 5 个 client。
- 更多 client 可作为后续 `region/cluster` 或扩展实验方向，而不是当前阶段立即替换实验 1。

## 7. 为什么不是其他 client

- 其他 client 缺少同等完整结果链。
- 替换 client 需要重新训练和重新审计。
- 当前没有证据说明其他 client 更适合作为当前实验 1 的解释对象。
- 因为当前只读阶段只覆盖已有 v4/r40/r60/异质性证据链，所以不能伪造“其他 client 一定更优”的结论。
- 若要更换，应作为独立 client selection 实验。

## 8. 当前 5-client 设置的局限

- 不代表全部真实路网。
- 不代表最终最优 client 组合。
- 仍需补充空间覆盖解释。
- 仍需补充 `cluster-level client`。
- 仍需补充更多 client 或 `region-client` 结果。

## 9. 推荐论文表述

当前实验 1 采用 K=5 的 grid-cell-level client 设置。与原稿 K=3 的设置相比，该设置增加了参与客户端数量，使实验能够更直接地观察 client-level variability 和强 non-IID 对标准 FedAvg 的影响。五个客户端均来自当前真实数据正式实验链路，并已完成 FedAvg、Independent、NaiveLastValue、通信轮次和异质性诊断。需要强调的是，该设置不是对真实路网全部客户端的穷尽覆盖，而是用于构造一个细粒度异质客户端场景，以检验标准 FedAvg 在真实交通流数据中的可运行性、收敛性和局限性。后续 cluster-level 或 region-level client 实验将进一步用于验证更同质的客户端组织方式是否能够缓解跨客户端分布差异带来的性能下降。

## 10. 结论

- 选择这 5 个 client 是当前阶段合理的。
- 不能说它们是唯一最优组合。
- 不建议减少。
- 不建议当前直接增加。
- 不建议当前直接替换。
- 下一步应补充空间覆盖与分布统计，或进入 `cluster/region client` 审计。

## 11. 边界声明

- 本阶段未运行训练。
- 未运行实验 2/3/4。
- 未修改 FedAvg。
- 未修改模型结构。
- 未修改数据划分。
- 未提交 `results/`。
