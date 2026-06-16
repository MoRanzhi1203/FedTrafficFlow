# 缺失实验结果目录迁移报告

## 迁移目的

- 统一整理三类缺失机制的结果目录，便于论文写作、可视化检索和 Git 管理。

## 旧目录

- global_source: `results\real_data_global_missingness_setting`
- structured_source: `results\real_data_structured_missingness_setting`
- comparison_source: `results\real_data_missingness_visual_comparison`

## 新目录

- `results\real_data_missingness_experiments`

## 三类机制的新 scenario_id

- `global_mcar_point`
- `node_temporal_block_mixed_short_mid_long`
- `node_subset_temporal_outage_mixed_short_mid_long`

## 迁移结果

- 已移动或已解析条目数：`317`
- 大体积 parquet 相关目录条目数：`96`
- 缺失或跳过条目数：`0`

## 说明

- 大体积 parquet 目录只在文件系统层面移动，不应提交到 Git。
- shared structured 审计与 manifest 采用复制方式放入 block/outage 两个 scenario。
- 旧目录保留 `MIGRATED_TO_README.md` 指向新根目录。
- 是否生成 experiment_registry：`是`
- 是否生成 path_aliases：`是`
- 是否验证通过：`是`

## 建议

- 后续脚本、可视化和论文表格优先使用新 root 或通过 `experiment_registry.json` / `path_aliases.json` 查找路径。
