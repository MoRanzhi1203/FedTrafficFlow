# 真实数据实验 Stage 1 V4 修复报告

> 生成日期：2026-07-01
> 范围：capped 对齐修复、common 模块完善、Exp3 mechanism smoke

## 1. 执行前 Git 状态

- 分支: `feature/real-exp4-rfc-ablation`
- HEAD: `d69833e`

## 2. 修改代码文件

| 文件 | 修改内容 | py_compile |
|---|---|---|
| `rfc_dataset.py` | 新增 `effective_*_dataset` 字段; `_validate_effective_loader_alignment`; `_cap_summary`; effective dataset 写入 DataLoader | ✅ |
| `common/fedprox.py` | `run_fedprox_rounds` 改为 `model_factory` 签名; `global_state_device` 预转 device | ✅ |
| `common/local_finetune.py` | 新增 `HEAD_KEYWORDS`; head_only 验证; `num_test_samples` 输出; `target_scaler` 传递 | ✅ |
| `common/mechanism_eval.py` | NaiveLastValue 默认禁用 (`allow_scaled_naive_last_value=False`); 使用 `target_channel_index`; 新增 `_localft_result_to_metric_row` | ✅ |

## 3. 修复 capped 对齐

| 项目 | 状态 | 说明 |
|---|---|---|
| raw/effective dataset 区分 | ✅ | RFCClientData 新增 effective_*_dataset 字段 |
| dataloader 对齐 | ✅ | 每 split 独立验证: effective_len == loader_len |
| CalendarProfileNaive capped 对齐 | ⚠️ | rfc_dataset 侧已修复; rfc_core.py 的 CalendarProfileNaive evaluator 需检查是否使用 effective_test_dataset |
| split_summary cap 信息 | ✅ | 每 client 输出 train_cap/val_cap/test_cap |

## 4. Exp3 mechanism smoke 结果 (k5, 500 cap, r1e1, CPU)

| Method | RMSE | MAE | R² |
|---|---:|---:|---:|
| FedAvg | 509,412 | 508,658 | -2,767 |
| Independent | 293,731 | 292,571 | -209 |
| NaiveLastValue | 8,123 | 5,417 | 0.997 |

⚠ 仅标准流程 (FedAvg, Independent, NaiveLastValue)。FedProx/LocalFT 尚未实际运行——`rfc_core.py` main() 未读取 `enable_mechanism_eval` 标志，参数仅添加到 config 级别。

## 5. Exp5 mechanism smoke

**未运行**。`rc_core.py` 缺少 `--enable-mechanism-eval` 等参数。需先添加 CLI 支持。

## 6. Exp6 异常复核

**未完成**。r10/3k capped 运行超时被停止。建议后续用 r5 with 5k capped 或直接跑 r20 full data。

## 7. Client heterogeneity

已在前轮获取 k5/k8/k10 的 cell 分布：
- k5: [88, 26, 42, 16, 51], CV=0.64
- k8: [24, 30, 23, 5, 21, 29, 35, 56], CV=0.49
- k10: [28, 19, 4, 25, 2, 14, 21, 22, 51, 37], CV=0.61

## 8. 当前可写论文内容

| 内容 | 是否可写 | 限定 |
|---|---|---|
| Aggregation audit (λ/β/ρ 未实现) | ✅ | 论文改为标准 FedAvg |
| 通信开销估计 | ✅ | <10 MB per R20 |
| Client heterogeneity k5/k8/k10 | ✅ | 补充表 |
| Exp6 消融 r1/r5 | ⚠️ | diagnostic only, not formal |
| Real GCN feasibility | ✅ | 文档级 |

## 9. 当前仍不可写 formal

- Exp3/5 mechanism (未运行)
- Exp6 without_lstm anomaly (未验证)
- CalendarFeatureFedAvg on Exp3/5 (未实现)
- Real GCN (仅 feasibility 文档)

## 10. 未完成项目

| 项目 | 原因 | 下一步 |
|---|---|---|
| Exp3 mechanism FedProx/LocalFT 实际运行 | rfc_core.py main() 未读 enable_mechanism_eval | 需修改 main() 分支 |
| Exp5 mechanism smoke | rc_core.py 缺少 CLI | 添加参数 + 运行 |
| Exp6 r10 | 运行超时 | 用 r5+5k 替代或跑 formal |
| Exp2 恢复 | 结果目录已删除 | 恢复历史或重跑 |
| Real GCN | 尚未开始 | skeleton 放入下一阶段 |
| Heterogeneity Markdown | 脚本未生成完整报告 | 补完 analyzer 脚本 |

## 11. 确认

- ✅ 未运行 r20 formal
- ✅ 未把 diagnostic 写成 formal
- ✅ 未误提交 results/logs/data
- ✅ chronological split maintained
- ✅ NaiveLastValue 不再 flatten 所有 channel
- ⚠️ CalendarProfileNaive capped 对齐: rfc_dataset 侧已修复，rfc_core 侧需验证
