# 实验 1 客户端/网格异质性诊断报告

## 1. 诊断目的

当前真实实验 1 的 formal 结果中，FedAvg 明显弱于 Independent 和 NaiveLastValue：

| 方法 | RMSE |
|------|:---:|
| FedAvg | 20,753.14 |
| Independent | 14,883.58 |
| NaiveLastValue | 19,419.22 |

需要判断 FedAvg 弱是否主要由 selected clients 的异质性（网格选择不够相似）造成。

## 2. 方法说明

- **数据入口**: `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`（shape [2, 630, 5856]）
- **Regions 元数据**: `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- **相似度计算范围**: 仅使用 train split（前 70%，4099 步），未使用 val/test 以防数据泄漏
- **Target channel**: 0（路口车流量）
- **Group size**: 5
- **相似度公式**: `score = Pearson_correlation − 0.25 × normalized_feature_euclidean_distance`
- **特征维度**: mean_flow, std_flow, cv, p10, p50, p90, peak_time_index, autocorr_lag1, autocorr_lag12
- **selected clients 传入**: `sic_core` 的 `--selected-clients` 参数（逗号分隔）
- **Active grids**: 223 个 `is_active_region=True` 的网格

## 3. 三组 clients

| 组别 | selected clients | mean_pairwise_corr | 说明 |
|------|-----|:---:|---|
| formal_current | 290, 284, 318, 288, 289 | 0.669 | 原 formal 组 |
| most_similar_5 | 281, 279, 341, 404, 311 | 0.957 | train split 内部综合相似度最高 |
| least_similar_5 | 287, 395, 136, 322, 284 | 0.077 | train split 内部综合相似度最低 |

## 4. r5e1 诊断结果

参数：`rounds=5, local_epochs=1, device=cuda, model_variant=baseline, sequence_length=12`

| 组别 | FedAvg RMSE | Independent RMSE | NaiveLastValue RMSE |
|------|:---:|:---:|:---:|
| formal_current | 37,344 | 25,765 | 19,419 |
| most_similar_5 | **53,324** | 18,358 | 15,297 |
| least_similar_5 | 101,701 | 25,765 | 7,853 |

关键发现：

- **most_similar_5 的相关性最高（0.957），但 FedAvg 反而比 formal_current 更差**（53,324 vs 37,344），上升了 43%
- least_similar_5 的 FedAvg 明显崩坏（101,701），但 NaiveLastValue 极强（7,853），说明该组不是不可预测，而是 FedAvg 没学好
- NaiveLastValue 在三组中都强于 FedAvg
- Independent 在 most_similar_5 上表现最好（18,358）

## 5. Independent RMSE 相同的复核

formal_current 和 least_similar_5 的 Independent RMSE 都约为 25,765，但 **client-level 指标完全不同**：

| Client | formal_current RMSE | least_similar_5 RMSE |
|:---:|:---:|:---:|
| c0 | 28,238 | 29,953 |
| c1 | 30,901 | **4,006** |
| c2 | 21,939 | 10,278 |
| c3 | 22,323 | **816** |
| c4 | 25,420 | **83,771** |

**结论**: 两组 client-level RMSE 分布差异极大（formal_current 方差小，least_similar_5 方差极大），均值接近纯属巧合。这不是 selected clients 没生效，也不是汇总脚本复用旧结果。

## 6. Prediction scale 复核

- 无 NaN
- 无 Inf
- y_true / y_pred 尺度正常（各组 mean 在 0.5M–1.9M 范围，量级一致）
- 无明显 y_pred 尺度崩坏

## 7. 诊断结论

在当前 r5e1 小规模诊断下，单纯选择曲线相关性更高的 5 个网格并未改善 FedAvg。most_similar_5 的 mean_pairwise_corr 达到 0.957，但 FedAvg RMSE 反而上升到 53,324，明显差于 formal_current 的 37,344。

该结果说明，当前实验 1 中 FedAvg 弱于 Independent 和 NaiveLastValue 的问题，**不太可能仅由 formal selected clients 不够相似造成**。后续应优先检查 FedAvg 训练策略、局部训练漂移、学习率、local epochs 和强时间惯性 baseline，而不是继续单纯更换 selected grids。

## 8. 下一步建议

优先进入 **FedAvg 训练策略优化**，暂不继续 legacy 大模型和 selected grids 调整：

1. **local_epochs 对照**: 1 / 2 / 3
2. **learning rate 对照**: 1e-3 / 5e-4 / 3e-4 / 1e-4
3. **FedProx 对照**: μ=1e-4 / 1e-3 / 1e-2
4. **rounds 增加**: 20 / 40
5. 暂不把"更相似网格"作为主要优化路线
6. 真实实验 3/5/6 后续仍可作为论文结构实验，但不应被当作解决 FedAvg 当前弱表现的唯一手段
