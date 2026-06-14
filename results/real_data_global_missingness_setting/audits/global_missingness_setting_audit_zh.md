# 完整数据全局缺失值设置审计报告

## 1. 任务范围

- 本次仅生成完整 61 天真实路口流量数据的 global MCAR 缺失值设置。
- 未执行任何补全方法。
- 未生成 imputed_datasets。
- 未计算任何插补误差。

## 2. 数据范围

- input_dir: `E:\Jupter_Notebook\FedTrafficFlow\data\analysis\node_intersection_flow_parquet`
- output_dir: `E:\Jupter_Notebook\FedTrafficFlow\results\real_data_global_missingness_setting`
- chunk_count: `61`
- global_eligible_count: `246133536`
- target_col: `路口车流量`
- node_col: `节点ID`
- time_col: `时间段`

## 3. 缺失机制

- mechanism: `mcar_point`
- mask_scope: `global`
- seed: `42`
- allocation_method: `sequential_hypergeometric_global_without_replacement`
- missing_unit: `node_time_observation`
- mask_uses_row_index: `True`
- drops_entire_time_slot: `False`
- drops_all_nodes_at_same_time: `False`
- creates_temporal_blocks: `False`
- forces_non_contiguous_missing: `False`
- missingness_type: `global_mcar_point`

说明：缺失单位是单个 `(day_index, 节点ID, 时间段)` 行级观测，mask 记录具体 `row_index`，不会因为某个时间段被抽中而把该时间段下所有路口整体置缺失，也不会人为构造连续 block 或强制缺失点彼此不相邻。

## 4. 0% 对照组

- 英文说明：0% is treated as no-missing control. No mask positions are generated. No missing dataset copy is written. No imputation metrics are applicable.
- 中文说明：0% 作为无缺失对照组，不生成大体积缺失数据副本，不参与缺失位置补全误差计算。

## 5. 各缺失率审计结果

### 缺失率 0%

- global_missing_count: `0`
- observed_global_missing_rate: `0.0`
- per_day_missing_rate_min: `0.0`
- per_day_missing_rate_max: `0.0`
- per_day_missing_rate_mean: `0.0`
- per_day_missing_rate_std: `0.0`
- sum_allocated_missing_count: `0`
- is_global_count_exact: `True`
- is_day_stratified_like: `True`

### 缺失率 5%

- global_missing_count: `12306677`
- observed_global_missing_rate: `0.050000000812567044`
- per_day_missing_rate_min: `0.0497903828919924`
- per_day_missing_rate_max: `0.0502664699864385`
- per_day_missing_rate_mean: `0.050000000812567`
- per_day_missing_rate_std: `0.00012266055907151052`
- sum_allocated_missing_count: `12306677`
- is_global_count_exact: `True`
- is_day_stratified_like: `False`

### 缺失率 10%

- global_missing_count: `24613354`
- observed_global_missing_rate: `0.10000000162513409`
- per_day_missing_rate_min: `0.0997069127548714`
- per_day_missing_rate_max: `0.1003113773167424`
- per_day_missing_rate_mean: `0.10000000162513403`
- per_day_missing_rate_std: `0.00013000668168094517`
- sum_allocated_missing_count: `24613354`
- is_global_count_exact: `True`
- is_day_stratified_like: `False`

### 缺失率 20%

- global_missing_count: `49226707`
- observed_global_missing_rate: `0.19999999918743294`
- per_day_missing_rate_min: `0.1995635166107555`
- per_day_missing_rate_max: `0.2004351450913215`
- per_day_missing_rate_mean: `0.1999999991874329`
- per_day_missing_rate_std: `0.00018972111217604692`
- sum_allocated_missing_count: `49226707`
- is_global_count_exact: `True`
- is_day_stratified_like: `False`

### 缺失率 30%

- global_missing_count: `73840061`
- observed_global_missing_rate: `0.30000000081256706`
- per_day_missing_rate_min: `0.2994458950933041`
- per_day_missing_rate_max: `0.300744787577423`
- per_day_missing_rate_mean: `0.300000000812567`
- per_day_missing_rate_std: `0.00025037082912518645`
- sum_allocated_missing_count: `73840061`
- is_global_count_exact: `True`
- is_day_stratified_like: `False`
