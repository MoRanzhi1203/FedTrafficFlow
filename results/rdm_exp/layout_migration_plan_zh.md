# 缺失实验结果目录迁移计划

- 计划状态：`migrated`
- 计划条目数：`53`
- move 条目数：`9`
- copy 条目数：`44`
- source_missing 条目数：`0`

## 说明

- `is_large_data = true` 代表 masks、miss_data、imp_data 或 parquet 大文件目录。
- dry_run 阶段只生成计划，不实际移动文件。
- shared structured 审计与 manifest 会复制到 block/outage 两个 scenario 中。
