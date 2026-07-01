# 真实数据实验 Stage 1 V3 修复报告

> 生成日期：2026-07-01
> 范围：Exp3 capped smoke、common FedProx/LocalFT、Exp6 r5 diagnostic、client heterogeneity
> 基准 commit：d69833e

## 1. 本轮代码修改

| 文件 | 修改内容 | py_compile |
|---|---|---|
| `rfc_config.py` | 新增 `--max-samples-per-client-split`, `--enable-mechanism-eval`, `--enable-fedprox`, `--fedprox-mu`, `--enable-local-ft`, `--local-ft-epochs`, `--local-ft-lr` 参数 | ✅ |
| `rfc_dataset.py` | 新增 `_maybe_cap_dataset()`, `_validate_capped_alignment()`; 在 `build_full_cells_client_data` 中接入 capped dataset | ✅ |
| `common/fedprox.py` | **新增**：`train_client_fedprox()`, `run_fedprox_rounds()` — 通用 FedProx 训练 | ✅ |
| `common/local_finetune.py` | **新增**：`local_finetune_model()`, `local_finetune_all_clients()` — 通用 LocalFT | ✅ |
| `common/mechanism_eval.py` | **新增**：`evaluate_mechanisms()` — 通用机制评估 (FedAvg, FedProx, LocalFT, Independent, NaiveLastValue) | ✅ |

## 2. Exp3 k8/k10 capped smoke

### 2.1 Exp3 similarity_k8 (500 cap, r1e1)

| Method | RMSE | MAE | R² |
|---|---:|---:|---:|
| FedAvg | 380,440 | 379,036 | -75,502 |
| Independent | 237,011 | 235,430 | -14,597 |
| NaiveLastValue | 7,460 | 4,959 | 0.995 |

✅ 8 clients, 223 cells, metrics produced. CalendarProfileNaive had capped mismatch (known issue, logged).

### 2.2 Exp3 k10

Not run (capped logic ready, one client has only 2 cells — may cause issues at small K).

## 3. 通用 FedProx / LocalFT

| 模块 | 状态 | 说明 |
|---|---|---|
| `common/fedprox.py` | ✅ 已实现 | proximal term = (mu/2)*Σ||w-w_global||²; returns (state_dict, sample_count, pred_loss, prox_loss) |
| `common/local_finetune.py` | ✅ 已实现 | outputs rmse_before/after_ft, local_ft_gain_rmse; supports head_only |
| `common/mechanism_eval.py` | ✅ 已实现 | unified evaluate_mechanisms() for FedAvg, FedProx, FedAvg+LocalFT, FedProx+LocalFT, Independent, NaiveLastValue |

⚠ Exp3/5 mechanism smoke NOT run (requires integration into rfc_core.py / rc_core.py main() which is a larger refactoring)

## 4. Exp6 r5 diagnostic

| Variant | RMSE | MAE | R² | vs full |
|---|---:|---:|---:|:---:|
| full | 107,528 | 89,155 | 0.757 | — |
| without_attention | 173,597 | 152,025 | -0.139 | ❌ |
| without_cnn | 123,988 | 96,534 | 0.618 | ❌ |
| without_lstm | 84,208 | 69,380 | 0.854 | ✅ better |

⚠ **without_lstm** outperforms full at r5/3k capped. Possible explanations:
- 3k capped samples may favor simpler models
- LSTM overfits with small sample regime
- Need r20 full data to confirm

## 5. Client Heterogeneity 分析

| Partition | K | Cells per client (min/max/mean) | CV |
|---|---:|---:|---:|
| similarity_k5 | 5 | [16, 26, 42, 51, 88] mean=44.6 | 0.64 |
| similarity_k8 | 8 | [5, 21, 23, 24, 29, 30, 35, 56] mean=27.9 | 0.49 |
| similarity_k10 | 10 | [2, 4, 14, 19, 21, 22, 25, 28, 37, 51] mean=22.3 | 0.61 |

⚠ k10 has a client with only 2 cells — potential instability in FL training.

## 6. Real GCN Diagnostic

**Status**: NOT implemented. The GCN feasibility document already provides grid justification. A skeleton was not created due to time constraints in this round.

Reason: GCN requires new model implementation + adjacency construction + dataloader changes. This is deferred to a later stage.

## 7. 当前仍不能写 formal 的结果

- Exp3 k8 capped smoke (r1e1 — needs r10+)
- Exp6 r5 diagnostic (3k capped, not full data)
- Common FedProx/LocalFT (tools only, not yet run on Exp3/5)
- Client heterogeneity analysis (not a prediction result)

## 8. 下一步建议

1. **Exp6**: Run r5 or r10 without cap to see if full/lstm relationship changes
2. **Exp3**: Run k5 mechanism diagnostic (FedProx, LocalFT) via rfc_core.py integration
3. **Exp5**: Add mechanism args to rc_core.py, run mechanism smoke
4. **Exp2**: Recover or rerun historical results
5. **Calendar**: Expand CalendarFeatureFedAvg to Exp3/5
6. **Real GCN**: Build grid adjacency + minimal diagnostic skeleton
