# 当前真实数据缺失实验开放问题与下一步建议

## 1. 已完成事项

- 已确认真实数据主来源为 `data/analysis/node_intersection_flow_parquet`。
- 已确认 `historical_test` 存在完整 `run_config / run_commands / audit / validation / summaries / figures / masks / missing_datasets / imputed_datasets` 证据。
- 已确认历史因果主流水线代码支持 `generate_missing / impute / validate / summarize`。

## 2. 部分完成事项

- 61 chunk 主目录存在阶段性输出，但未完成全量闭环证据整理。
- 真实数据图件已生成多张，但部分仍是单缺失率 5% 表达方式。

## 3. 未完成事项

- 多缺失率全量 MCAR 正式实验。
- `node_temporal_block` 正式实验。
- 多 seed mean±std 汇总。
- error bar 图。
- FedAvg / Independent 真实预测正式输出证据。

## 4. 证据不足事项

- 当前不要直接写“完整全量实验完成”，除非 inventory 证明 61 个 chunk 的 `generate_missing`、`impute`、`summarize`、`validate` 均完成。
- 当前不能把 `historical_test` 写成 61 chunk 全量结果。

## 5. 后续优先级 P0/P1/P2

- P0：核对 61 chunk 主目录中各方法的实际 chunk 覆盖数与阶段完成度。
- P1：补齐多缺失率 MCAR 与 `node_temporal_block`。
- P2：补齐多 seed 统计并更新论文图文。

## 6. 下一步建议命令

- `E:\anaconda3\envs\analysis\python.exe analysis_scripts\inventory_real_missingness_assets.py --project_root E:\Jupter_Notebook\FedTrafficFlow --output_dir results\real_data_missingness_inventory`
- 仅在人工确认 inventory 结论后，再决定是否运行后续正式实验命令。

## 7. 不建议现在做的事情

- 不建议现在直接写“完整全量实验完成”。
- 不建议现在直接把插补误差写成预测误差。
- 不建议现在把人工缺失写成天然缺失。
- 不建议现在继续跑 `generate_missing / impute / summarize / validate / FedAvg / Independent`。

附注：本次风险关键词命中 74 条，建议先做文档口径复核。

