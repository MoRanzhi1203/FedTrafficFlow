# snh constraint relaxation audit

- 当高缺失率下空间约束导致缺失位置无法放满时，本实验允许逐级降低空间约束。
- 其中 none 等级只用于满足完整数据集 global missing rate，不作为严格空间邻居保留样本解释。
- 正式分析空间结构效果时，应优先查看 strict_anchor、relaxed_anchor 和 weak_neighbor_available 子集。

## 0.05

- global_missing_count: `12306677`
- sum_observed_missing_count: `12306677`
- strict_anchor_missing_count: `12306677`
- relaxed_anchor_missing_count: `0`
- weak_neighbor_available_missing_count: `0`
- none_missing_count: `0`
- spatially_constrained_ratio: `1.000000`
- warning_if_spatially_constrained_ratio_below_threshold: `False`

## 0.10

- global_missing_count: `24613354`
- sum_observed_missing_count: `24613354`
- strict_anchor_missing_count: `24613354`
- relaxed_anchor_missing_count: `0`
- weak_neighbor_available_missing_count: `0`
- none_missing_count: `0`
- spatially_constrained_ratio: `1.000000`
- warning_if_spatially_constrained_ratio_below_threshold: `False`

## 0.20

- global_missing_count: `49226707`
- sum_observed_missing_count: `49226707`
- strict_anchor_missing_count: `49226707`
- relaxed_anchor_missing_count: `0`
- weak_neighbor_available_missing_count: `0`
- none_missing_count: `0`
- spatially_constrained_ratio: `1.000000`
- warning_if_spatially_constrained_ratio_below_threshold: `False`

## 0.30

- global_missing_count: `73840061`
- sum_observed_missing_count: `73840061`
- strict_anchor_missing_count: `71461922`
- relaxed_anchor_missing_count: `1676582`
- weak_neighbor_available_missing_count: `480572`
- none_missing_count: `220985`
- spatially_constrained_ratio: `0.997007`
- warning_if_spatially_constrained_ratio_below_threshold: `False`
