# 结构化缺失分布分析报告

## 1. 总体结论

- 本报告仅基于已完整生成的结构化缺失场景。
- 原始字段列表：`节点ID, 时间段, 路口进入流量, 路口离开流量, 路口车流量`
- 目标缺失字段：`路口车流量`
- 缺失模式判断：当前场景均为结构化非随机缺失，不属于 MCAR。

## 2. 场景摘要

### mechanism_node_temporal_block__rate_0p05__mixed_short_mid_long__seed_42

- observed_missing_rate: `0.0500`
- field_missing_rate_target_col: `0.0500`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0328`
- short_missing_ratio: `0.1282`
- mid_missing_ratio: `0.4270`
- long_missing_ratio: `0.4449`

### mechanism_node_temporal_block__rate_0p10__mixed_short_mid_long__seed_42

- observed_missing_rate: `0.1000`
- field_missing_rate_target_col: `0.1000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0232`
- short_missing_ratio: `0.1257`
- mid_missing_ratio: `0.4230`
- long_missing_ratio: `0.4513`

## 3. 空间分布 Top Nodes

### mechanism_node_temporal_block__rate_0p05__mixed_short_mid_long__seed_42

- 节点 `1520443330`: `368`
- 节点 `1520314318`: `364`
- 节点 `1520420156`: `361`
- 节点 `1520460174`: `361`
- 节点 `1526675432`: `361`
- 节点 `1520480209`: `359`
- 节点 `1520488431`: `359`
- 节点 `1520460897`: `355`
- 节点 `1520461972`: `355`
- 节点 `1520498325`: `354`

### mechanism_node_temporal_block__rate_0p10__mixed_short_mid_long__seed_42

- 节点 `1520465622`: `699`
- 节点 `1549752873`: `679`
- 节点 `1520461350`: `677`
- 节点 `1530637497`: `677`
- 节点 `1549742558`: `677`
- 节点 `1520290604`: `676`
- 节点 `1520426081`: `676`
- 节点 `1532872325`: `676`
- 节点 `1520287735`: `675`
- 节点 `1520492202`: `675`
