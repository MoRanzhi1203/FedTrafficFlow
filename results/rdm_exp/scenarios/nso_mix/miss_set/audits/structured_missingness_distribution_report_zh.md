# 结构化缺失分布分析报告

## 1. 总体结论

- 本报告仅基于已完整生成的结构化缺失场景。
- 原始字段列表：`节点ID, 时间段, 路口进入流量, 路口离开流量, 路口车流量`
- 目标缺失字段：`路口车流量`
- 缺失模式判断：当前场景均为结构化非随机缺失，不属于 MCAR。

## 2. 场景摘要

### nso_r05_mix_s42

- observed_missing_rate: `0.0500`
- field_missing_rate_target_col: `0.0500`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0598`
- short_missing_ratio: `0.7956`
- mid_missing_ratio: `0.1117`
- long_missing_ratio: `0.0927`

### nso_r10_mix_s42

- observed_missing_rate: `0.1000`
- field_missing_rate_target_col: `0.1000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0319`
- short_missing_ratio: `0.9040`
- mid_missing_ratio: `0.0586`
- long_missing_ratio: `0.0374`

### nso_r20_mix_s42

- observed_missing_rate: `0.2000`
- field_missing_rate_target_col: `0.2000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0181`
- short_missing_ratio: `0.9507`
- mid_missing_ratio: `0.0282`
- long_missing_ratio: `0.0212`

### nso_r30_mix_s42

- observed_missing_rate: `0.3000`
- field_missing_rate_target_col: `0.3000`
- field_missing_rate_non_target_cols: `0.0000`
- missing_pattern: `non_random_structured_missingness`
- node_missing_gini: `0.0128`
- short_missing_ratio: `0.9669`
- mid_missing_ratio: `0.0191`
- long_missing_ratio: `0.0140`

## 3. 空间分布 Top Nodes

### nso_r05_mix_s42

- 节点 `1520469545`: `440`
- 节点 `1552898162`: `437`
- 节点 `1520421958`: `436`
- 节点 `1520413040`: `435`
- 节点 `1520336062`: `433`
- 节点 `1526676974`: `429`
- 节点 `1554153003`: `429`
- 节点 `1520459348`: `427`
- 节点 `1520457229`: `425`
- 节点 `1530829936`: `421`

### nso_r10_mix_s42

- 节点 `1531067909`: `764`
- 节点 `1520319277`: `749`
- 节点 `1520440364`: `729`
- 节点 `1520500479`: `726`
- 节点 `1520403049`: `724`
- 节点 `1520468150`: `723`
- 节点 `1520321825`: `719`
- 节点 `1520466422`: `718`
- 节点 `1529960419`: `716`
- 节点 `1553819492`: `716`

### nso_r20_mix_s42

- 节点 `1520496835`: `1340`
- 节点 `1549752378`: `1339`
- 节点 `1520435265`: `1333`
- 节点 `1549300201`: `1321`
- 节点 `1520483508`: `1317`
- 节点 `1520495845`: `1317`
- 节点 `1554151940`: `1316`
- 节点 `1554157575`: `1316`
- 节点 `1530632849`: `1315`
- 节点 `1520430691`: `1313`

### nso_r30_mix_s42

- 节点 `1553471206`: `1937`
- 节点 `1549751605`: `1920`
- 节点 `1520422639`: `1916`
- 节点 `1549300036`: `1915`
- 节点 `1520460778`: `1911`
- 节点 `1553396637`: `1908`
- 节点 `1520412192`: `1905`
- 节点 `1520468485`: `1905`
- 节点 `1520495014`: `1905`
- 节点 `1549428209`: `1905`

## 4. Manifest 一致性校验

- validated: `True`
- all_consistent: `True`
- validation_rows: `4`
- validation_csv: `structured_missingness_consistency_validation.csv`
