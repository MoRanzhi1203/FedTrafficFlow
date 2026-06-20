# 结构化缺失分布分析报告

## 1. 总体结论

- 本报告仅基于已完整生成的结构化缺失场景。
- 原始字段列表：`节点ID, 时间段, 路口进入流量, 路口离开流量, 路口车流量`
- 目标缺失字段：`路口车流量`
- 缺失模式判断：当前场景均为结构化非随机缺失，不属于 MCAR。

## 2. 场景摘要

### ntb_r05_mix_s42

- observed_missing_rate: `0.0500`
- field_missing_rate_target_col: `0.0500`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0328`
- short_missing_ratio: `0.1282`
- mid_missing_ratio: `0.4270`
- long_missing_ratio: `0.4449`

### ntb_r10_mix_s42

- observed_missing_rate: `0.1000`
- field_missing_rate_target_col: `0.1000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0232`
- short_missing_ratio: `0.1257`
- mid_missing_ratio: `0.4230`
- long_missing_ratio: `0.4513`

### ntb_r20_mix_s42

- observed_missing_rate: `0.2000`
- field_missing_rate_target_col: `0.2000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0166`
- short_missing_ratio: `0.1926`
- mid_missing_ratio: `0.5251`
- long_missing_ratio: `0.2823`

### ntb_r30_mix_s42

- observed_missing_rate: `0.3000`
- field_missing_rate_target_col: `0.3000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0135`
- short_missing_ratio: `0.1707`
- mid_missing_ratio: `0.4876`
- long_missing_ratio: `0.3417`

## 3. 空间分布 Top Nodes

### ntb_r05_mix_s42

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

### ntb_r10_mix_s42

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

### ntb_r20_mix_s42

- 节点 `1520408926`: `1326`
- 节点 `1530767220`: `1317`
- 节点 `1520484172`: `1311`
- 节点 `1553398929`: `1310`
- 节点 `1520338345`: `1300`
- 节点 `1554146490`: `1300`
- 节点 `1520299082`: `1298`
- 节点 `1520463953`: `1298`
- 节点 `1520460188`: `1296`
- 节点 `1549333496`: `1296`

### ntb_r30_mix_s42

- 节点 `1549277365`: `1922`
- 节点 `1520437410`: `1921`
- 节点 `1520486981`: `1914`
- 节点 `1553942046`: `1913`
- 节点 `1531207745`: `1912`
- 节点 `1549277213`: `1912`
- 节点 `1553480354`: `1912`
- 节点 `1554155233`: `1909`
- 节点 `1520430253`: `1908`
- 节点 `1530809448`: `1907`

## 4. Manifest 一致性校验

- validated: `True`
- all_consistent: `True`
- validation_rows: `4`
- validation_csv: `structured_missingness_consistency_validation.csv`
