# 聚合策略代码审计报告

> 生成日期：2026-07-01
> 审计范围：`real_data_experiments/common/`、`real_data_experiments/*/` 下所有 `.py` 文件
> 审计目标：确认论文中描述的 λ/β/ρ/loss-aware/smoothing 聚合策略是否已实现

## 1. 审计方法

```powershell
Select-String -Path real_data_experiments\**\*.py -Pattern "lambda","beta","rho","loss-aware","smoothing","FedAvg","sample_count","fedprox"
```

## 2. 当前实际实现的聚合策略

### 2.1 核心聚合模块 (common/)

| 文件 | 实现内容 | 状态 |
|---|---|---|
| [fedavg.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/common/fedavg.py) | **标准 FedAvg**：sample_count 加权聚合 `w_global = Σ_k (n_k / n_total) * w_k` | 已实现 |
| [trainer.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/common/trainer.py) | `run_federated_rounds()`：每轮调用 `fedavg_aggregate` | 已实现 |
| [client.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/common/client.py) | `FedClient`：标准 SGD/Adam 本地训练，无 proximal term，仅返回 `(state_dict, sample_count, train_loss)` | 已实现 |

### 2.2 实验级别聚合扩展

| 实验/文件 | 额外实现 | 状态 |
|---|---|---|
| Exp1 [sic_core.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/single_intersection_client/sic_core.py) | `run_fedprox_experiment()` — FedProx 带 proximal mu 约束 | ✅ 已实现（仅 Exp1） |
| Exp1 [sic_core.py] | `run_local_finetune_experiment()` — LocalFT 本地微调 | ✅ 已实现（仅 Exp1） |
| Exp1 [sic_core.py] | `run_centralized_upper_bound()` — 集中式上界 | ✅ 已实现（仅 Exp1） |
| Exp1 [sic_core.py] | `run_calendar_feature_fedavg()` — CalendarFeatureFedAvg | ✅ 已实现（仅 Exp1） |
| Exp3 [rfc_core.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/region_client_full_cells/rfc_core.py) | 仅标准 FedAvg + Independent + NaiveLastValue | ❌ 无扩展 |
| Exp4 [rfc_ablation_core.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/region_client_full_cells/rfc_ablation_core.py) | 仅标准 FedAvg ablation | ❌ 无扩展 |
| Exp5 [rc_core.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/region_client/rc_core.py) | 仅标准 FedAvg + Independent + NaiveLastValue + CalendarProfileNaive | ❌ 无扩展 |
| Exp6 [ra_core.py](file:///e:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/region_ablation/ra_core.py) | 仅标准 FedAvg ablation | ❌ 无扩展 |

## 3. 论文中的 λ/β/ρ 参数审计

| 参数 | 论文描述 | 代码实现 | 结论 |
|---|---|---|---|
| **λ (data-loss balance)** | 数据量与 loss 的混合权重 | **未实现**。`common/` 中无任何 λ 参数（lambda 关键字仅用于 pandas `.map()`） | ❌ 论文与代码不一致 |
| **β (loss scaling)** | loss-aware 权重的指数缩放因子 | **未实现**。代码中无 loss-aware 权重计算 | ❌ 论文与代码不一致 |
| **ρ (smoothing)** | 服务器端动量平滑因子 | **未实现**。无 momentum/memory/smoothing 机制 | ❌ 论文与代码不一致 |
| **loss-aware weights** | 按 `exp(-βL_i)` 计算聚合权重 | **未实现** | ❌ 论文与代码不一致 |
| **smoothed aggregation** | 带 ρ 平滑的混合聚合 | **未实现** | ❌ 论文与代码不一致 |

## 4. 已完成与未完成的聚合对照

### 4.1 已实现

```text
FedAvg (sample_count 加权)：common/fedavg.py、common/trainer.py
FedProx (proximal mu)：仅 Exp1 sic_core.py
FedAvg+LocalFT：仅 Exp1 sic_core.py
FedProx+LocalFT：仅 Exp1 sic_core.py
RandomInit+LocalFT：仅 Exp1 sic_core.py
CentralizedUpperBound：仅 Exp1 sic_core.py
CalendarFeatureFedAvg-Full+LocalFT：仅 Exp1 sic_core.py
Independent (每个 client 独立训练)：所有实验
NaiveLastValue：所有主实验
CalendarProfileNaive：Exp1/3/5
```

### 4.2 未实现（但论文可能已描述）

```text
loss_aware aggregation (exp(-βL_i) 权重)：全未实现
mixed_loss_data (λ * data + (1-λ) * loss)：全未实现
smoothed_mixed (mixed + ρ smoothing)：全未实现
FedProx for Exp3/4/5/6：仅 Exp1 有
LocalFT for Exp3/4/5/6：仅 Exp1 有
client-specific head / FedPer：全未实现
dropout simulation：全未实现
DP noise：全未实现
```

## 5. 结论与建议

### 5.1 推荐方案：修改论文描述

由于所有真实实验均已基于标准 FedAvg (sample_count) 完成，建议：
1. **论文中统一写为标准 FedAvg**，删除 λ/β/ρ/loss-aware/smoothing 描述；
2. 在 discussion/future work 中说明 loss-aware aggregation 和 momentum 可作为扩展；
3. 将 FedProx + LocalFT 从 Exp1 扩展到 Exp3/5。

### 5.2 备选方案：实现论文算法

如果必须保留 λ/β/ρ：
1. 需在 `common/` 中实现 `loss_aware_aggregate()`、`smoothed_aggregate()`；
2. 需在 `common/client.py` 中实现 FedProx-aware client；
3. 需对所有实验重跑 formal。
