# 结构化缺失设计说明

1. 已有 `results\rdm_exp\scenarios\g_mcar_pt` 继续保留为 global MCAR point 随机点缺失基准。
2. 本轮新增 `node_temporal_block` 和 `node_subset_temporal_outage` 两类结构化缺失机制。
3. 每个机制、每个缺失率只生成一套结构化缺失数据集，不再按固定长度拆分为多个 block 目录。
4. 连续缺失长度采用事件级随机变量 `mixed_short_mid_long`。
5. short_block 范围为 `1-4` 个时间片，采样概率 `0.4`。
6. mid_block 范围为 `5-12` 个时间片，采样概率 `0.4`。
7. long_block 范围为 `13-24` 个时间片，采样概率 `0.2`。
8. `mask` 与 `miss_data` 仍精确到 `row_index`，且只修改目标列。
9. 两类机制都不会把同一时间片下全部路口整体置缺失，也不会覆盖现有 global MCAR point 结果。
