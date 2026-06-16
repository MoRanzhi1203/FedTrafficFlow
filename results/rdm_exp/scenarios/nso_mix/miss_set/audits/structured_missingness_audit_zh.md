# 结构化缺失设置审计报告

## 1. 基本信息

- input_dir: `E:\Jupter_Notebook\FedTrafficFlow\data\analysis\node_intersection_flow_parquet`
- output_dir: `E:\Jupter_Notebook\FedTrafficFlow\results\rdm_exp`
- mechanisms: `node_temporal_block, node_subset_temporal_outage`
- missing_rates: `[0.05, 0.1, 0.2, 0.3]`
- length_mode: `mixed_short_mid_long`
- short_length_range: `[1, 4]`
- mid_length_range: `[5, 12]`
- long_length_range: `[13, 24]`
- length_group_probs: `[0.4, 0.4, 0.2]`
- seed: `42`
- period: `96`
- chunk_count: `61`

## 2. 机制说明

- 现有 global MCAR point 目录保留不变，本轮输出全部写入全新目录。
- node_temporal_block 用于模拟单节点连续离线。
- node_subset_temporal_outage 用于模拟部分节点在连续窗口内统一离线。
- 连续缺失长度为事件级随机变量，而不是实验等级。

## 3. 场景审计

### mechanism_node_subset_temporal_outage__rate_0p05__mixed_short_mid_long__seed_42

- mechanism: `node_subset_temporal_outage`
- missing_rate_target: `0.05`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.050000000812567`
- absolute_error: `8.125670414305121e-10`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `6.583333333333333`
- length_std: `5.391164234160096`
- short_event_ratio: `0.4894366197183099`
- mid_event_ratio: `0.3791079812206572`
- long_event_ratio: `0.1314553990610328`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_subset_temporal_outage__rate_0p10__mixed_short_mid_long__seed_42

- mechanism: `node_subset_temporal_outage`
- missing_rate_target: `0.1`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.100000001625134`
- absolute_error: `1.6251340828610241e-09`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `6.209320695102686`
- length_std: `5.302353498128315`
- short_event_ratio: `0.5355450236966824`
- mid_event_ratio: `0.3601895734597156`
- long_event_ratio: `0.1042654028436018`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_subset_temporal_outage__rate_0p20__mixed_short_mid_long__seed_42

- mechanism: `node_subset_temporal_outage`
- missing_rate_target: `0.2`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.1999999991874329`
- absolute_error: `8.125670691860876e-10`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `6.476230899830221`
- length_std: `5.544375273112032`
- short_event_ratio: `0.5203735144312394`
- mid_event_ratio: `0.3548387096774194`
- long_event_ratio: `0.1247877758913412`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_subset_temporal_outage__rate_0p30__mixed_short_mid_long__seed_42

- mechanism: `node_subset_temporal_outage`
- missing_rate_target: `0.3`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.300000000812567`
- absolute_error: `8.125670691860876e-10`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `6.3357783211083945`
- length_std: `5.362009753776584`
- short_event_ratio: `0.5118174409127955`
- mid_event_ratio: `0.3691931540342298`
- long_event_ratio: `0.1189894050529747`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_temporal_block__rate_0p05__mixed_short_mid_long__seed_42

- mechanism: `node_temporal_block`
- missing_rate_target: `0.05`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.050000000812567`
- absolute_error: `8.125670414305121e-10`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `12.739203685934068`
- length_std: `6.555948495468565`
- short_event_ratio: `0.1281651415731476`
- mid_event_ratio: `0.4269751290295503`
- long_event_ratio: `0.4448597293973019`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_temporal_block__rate_0p10__mixed_short_mid_long__seed_42

- mechanism: `node_temporal_block`
- missing_rate_target: `0.1`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.100000001625134`
- absolute_error: `1.6251340828610241e-09`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `12.844919225555364`
- length_std: `6.569104098032152`
- short_event_ratio: `0.1256670261192359`
- mid_event_ratio: `0.4230308880293194`
- long_event_ratio: `0.4513020858514447`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_temporal_block__rate_0p20__mixed_short_mid_long__seed_42

- mechanism: `node_temporal_block`
- missing_rate_target: `0.2`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.1999999991874329`
- absolute_error: `8.125670691860876e-10`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `6.191099875756734`
- length_std: `4.802517825753396`
- short_event_ratio: `0.4927297167465781`
- mid_event_ratio: `0.4007096583060849`
- long_event_ratio: `0.1065606249473369`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`

### mechanism_node_temporal_block__rate_0p30__mixed_short_mid_long__seed_42

- mechanism: `node_temporal_block`
- missing_rate_target: `0.3`
- parameter_setting: `mixed_short_mid_long: short=1-4@0.4, mid=5-12@0.4, long=13-24@0.2`
- observed_missing_rate: `0.300000000812567`
- absolute_error: `8.125670691860876e-10`
- is_within_tolerance: `True`
- fixed_lengths_only: `False`
- length_is_event_level_random_variable: `True`
- length_min: `1`
- length_max: `24`
- length_mean: `6.708209045936114`
- length_std: `5.357999853222377`
- short_event_ratio: `0.4696649114260947`
- mid_event_ratio: `0.3991573682402176`
- long_event_ratio: `0.1311777203336877`
- mask_file_count: `61`
- missing_dataset_file_count: `61`
- drops_entire_day: `False`
- drops_all_nodes_at_same_time: `False`
- uses_row_index_mask: `True`
- modifies_only_target_col: `True`
