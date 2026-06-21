# snh global allocation audit / `snh_mix` 全局缺失分配审计

## Summary / 摘要

- 中文说明：本文件记录 `snh_mix` 场景在全局掩码分配设置下的缺失分配一致性审计结果，重点核验各缺失率下理论缺失数量、分配数量与实际观测数量是否完全一致。
- English: This file records the allocation-consistency audit for the `snh_mix` scenario under the global missingness allocation setting, focusing on whether the theoretical missing count, allocated count, and observed count match at each missing rate.

- mask_scope: `global`
- allocation_method: `sequential_hypergeometric_global_without_replacement`
- day_stratified_generation_used: `False`
- per_chunk_round_rate_used: `False`

## 0.05 / 缺失率 0.05

- global_eligible_count: `246133536`
- global_missing_count: `12306677`
- allocation_sum: `12306677`
- observed_sum: `12306677`
- allocation_sum_matches_global_missing_count: `True`
- observed_sum_matches_global_missing_count: `True`

## 0.10 / 缺失率 0.10

- global_eligible_count: `246133536`
- global_missing_count: `24613354`
- allocation_sum: `24613354`
- observed_sum: `24613354`
- allocation_sum_matches_global_missing_count: `True`
- observed_sum_matches_global_missing_count: `True`

## 0.20 / 缺失率 0.20

- global_eligible_count: `246133536`
- global_missing_count: `49226707`
- allocation_sum: `49226707`
- observed_sum: `49226707`
- allocation_sum_matches_global_missing_count: `True`
- observed_sum_matches_global_missing_count: `True`

## 0.30 / 缺失率 0.30

- global_eligible_count: `246133536`
- global_missing_count: `73840061`
- allocation_sum: `73840061`
- observed_sum: `73840061`
- allocation_sum_matches_global_missing_count: `True`
- observed_sum_matches_global_missing_count: `True`
