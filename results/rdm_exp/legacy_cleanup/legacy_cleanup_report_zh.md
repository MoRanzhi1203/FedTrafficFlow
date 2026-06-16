# 旧目录清理报告

- 本轮只清除迁移前旧结果路径。
- 清理目标：
  - `results\rdm_exp\scenarios\g_mcar_pt`
  - `results\rdm_exp`
  - `results\rdm_exp\comparison`
- 新统一根目录保留为：`results\rdm_exp`
- `layout_validation.json` 中 `all_complete = true`
- 删除前清单已生成在 `results\rdm_exp\legacy_cleanup`
- 三个旧目录已删除，删除后 `Test-Path` 全部为 `False`
- 未重跑缺失生成
- 未重跑补全
- 未修改算法脚本
- 未删除新统一根目录
- 未删除原始数据目录
