# snh_mix fast missingness audit

- mechanism: `spatial_neighbor_holdout`
- evaluation_protocol: `online_spatial_interpolation`
- 当前时刻允许使用邻居观测。
- 不允许使用目标节点当前真实值。
- 不允许使用未来信息。
- fast 版本默认采用 anchor 邻居保护，而不是旧版 all-neighbor 保护。

## 0.05

- mask 文件数: `61`
- miss_data 文件数: `61`
- observed_missing_rate: `0.050000`
- neighbor_observed_ratio: `1.000000`
- min_available_neighbor_count: `1`
- skipped_missing_slots: `0`

## 0.10

- mask 文件数: `61`
- miss_data 文件数: `61`
- observed_missing_rate: `0.100000`
- neighbor_observed_ratio: `1.000000`
- min_available_neighbor_count: `1`
- skipped_missing_slots: `0`

## 0.20

- mask 文件数: `61`
- miss_data 文件数: `61`
- observed_missing_rate: `0.200000`
- neighbor_observed_ratio: `1.000000`
- min_available_neighbor_count: `1`
- skipped_missing_slots: `0`

## 0.30

- mask 文件数: `61`
- miss_data 文件数: `61`
- observed_missing_rate: `0.300000`
- neighbor_observed_ratio: `0.996708`
- min_available_neighbor_count: `0`
- skipped_missing_slots: `0`
