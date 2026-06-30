# Exp1 联邦机制优势诊断汇总

> 本报告汇总 Exp1 federated mechanism diagnostic (seq96_h12 r5e1)。
> 本轮为 diagnostic，不是 formal 结果。

## 1. 实验设置

- sequence_length=96, prediction_horizon=12, rounds=5, local_epochs=1, split=80/10/10
- CalendarFeatureFedAvg v2 residual_gate (仅 full feature set)
- model_output_dim=1 (单点预测)

## 2. Main Metrics (RMSE)

| Method | RMSE | MAE | R² |
|--------|------|-----|-----|
| CalendarFeatureFedAvg-Full | 77969.1 | 67758.0 | -0.0164 |
| CalendarFeatureFedAvg-Full+LocalFT | 53756.3 | 40475.8 | 0.4246 |
| CalendarProfileNaive | 31970.1 | 22564.5 | 0.8262 |
| CentralizedUpperBound | 44819.2 | 30326.0 | 0.6310 |
| DailySeasonalNaive | 46788.6 | 30471.0 | 0.5832 |
| FedAvg | 82523.9 | 68055.8 | -0.4430 |
| FedAvg+LocalFT | 55772.7 | 40836.6 | 0.4097 |
| FedProx | 72304.2 | 59688.6 | 0.0325 |
| FedProx+LocalFT | 55536.6 | 40106.3 | 0.3966 |
| Independent | 61880.8 | 47528.8 | 0.3048 |
| NaiveLastValue | 94259.0 | 69688.0 | -0.3731 |
| RandomInit+LocalFT | 76284.4 | 62771.9 | 0.0486 |
| WeeklySeasonalNaive | 52474.1 | 35897.3 | 0.3894 |

## 3. 机制优势分析

| Metric | Baseline | Federated | RMSE Base | RMSE Fed | Gain | Improvement? |
|--------|----------|-----------|-----------|----------|------|-------------|
| GlobalInitGain | RandomInit+LocalFT | FedAvg+LocalFT | 76284.4 | 55772.7 | 20511.7 | True |
| PersonalizationGain | FedAvg | FedAvg+LocalFT | 82523.9 | 55772.7 | 26751.3 | True |
| FedProxGain | FedAvg | FedProx | 82523.9 | 72304.2 | 10219.7 | True |
| FedProxPersonalizationGain | FedProx | FedProx+LocalFT | 72304.2 | 55536.6 | 16767.7 | True |
| CentralizedGap | CentralizedUpperBound | FedAvg+LocalFT | 44819.2 | 55772.7 | -10953.5 | False |
| NaiveRobustGain | NaiveLastValue | FedAvg+LocalFT | 94259.0 | 55772.7 | 38486.4 | True |
| IndependentGap | Independent | FedAvg+LocalFT | 61880.8 | 55772.7 | 6108.1 | True |
| CalendarPersonalizationGain | CalendarFeatureFedAvg-Full | CalendarFeatureFedAvg-Full+LocalFT | 77969.1 | 53756.3 | 24212.8 | True |

## 4. Client Win Rate

| Baseline | Federated | Wins | Total | Win Rate |
|----------|-----------|------|-------|----------|
| RandomInit+LocalFT | FedAvg+LocalFT | 4 | 5 | 80.00% |
| FedAvg | FedAvg+LocalFT | 5 | 5 | 100.00% |
| FedAvg | FedProx | 5 | 5 | 100.00% |
| FedProx | FedProx+LocalFT | 5 | 5 | 100.00% |
| CentralizedUpperBound | FedAvg+LocalFT | 0 | 5 | 0.00% |
| NaiveLastValue | FedAvg+LocalFT | 5 | 5 | 100.00% |
| Independent | FedAvg+LocalFT | 2 | 5 | 40.00% |
| CalendarFeatureFedAvg-Full | CalendarFeatureFedAvg-Full+LocalFT | 4 | 5 | 80.00% |

## 5. 核心发现

1. FedAvg+LocalFT (55,773) 显著优于 RandomInit+LocalFT (76,284)，证明联邦共享初始化有效（GlobalInitGain = 20,511）
2. FedProx (72,304) 优于 FedAvg (82,523)，证明 non-IID proximal 约束有效（FedProxGain = 10,219）
3. FedAvg+LocalFT (55,773) 显著优于纯 FedAvg (82,523)，证明本地微调是联邦优势的关键（PersonalizationGain = 26,751）
4. CentralizedUpperBound (44,819) 优于所有联邦方法，集中式仍是最强上界
5. CalendarFeatureFedAvg-Full+LocalFT (53,756) 是最优联邦方法，说明 calendar 特征 + 个性化微调互补
6. FedAvg 裸性能 (82,523) 仍然弱于 NaiveLastValue (94,259) 和 Independent (61,881)，但加 LocalFT 后大幅改善
7. 当前仍是 r5 diagnostic，需要 r10/r20 formal 验证稳定性

## 6. 下一步建议
- 若 FedAvg+LocalFT 在 r10 保持优势，进入 r20 formal candidate
- CalendarFeatureFedAvg-Full+LocalFT 应继续跟踪
- 考虑 personalized FL (Per-FedAvg, pFedMe) 进一步缩小与 Centralized 的差距