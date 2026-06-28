# 实验 1 Formal 结果状态记录

> 生成日期：2026-06-29
> 该结果代表真实数据实验 1 首次正式运行（非 smoke）。

## 1. 运行信息

- **分支**: `feature/real-exp1-client-similarity-diagnosis`
- **Commit**: `3ade178 docs(real-data): map revised experiments 1-6 status`
- **运行命令**:
  ```
  python -m real_data_experiments.single_intersection_client.sic_core \
    --workflow train \
    --rounds 20 \
    --local-epochs 1 \
    --device cuda \
    --model-variant baseline \
    --sequence-length 12 \
    --selected-clients "290,284,318,288,289" \
    --output-dir results/real_data_experiments/formal/exp1_single_grid_baseline_r20_e1_cuda
  ```
- **输出目录**: `results/real_data_experiments/formal/exp1_single_grid_baseline_r20_e1_cuda`
- **运行状态**: ✅ 成功完成
- **输入数据**: `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU
- **运行时间**: 约 2 分 30 秒

## 2. Main Metrics

| 方法 | RMSE | MAE | R² |
|------|:---:|:---:|:---:|
| NaiveLastValue | **19,419** | 13,620 | 0.939 |
| Independent | 15,921 | 11,615 | 0.953 |
| CalendarProfileNaive | 32,194 | 22,770 | 0.830 |
| FedAvg | 24,158 | 19,773 | 0.847 |
| DailySeasonalNaive | 45,406 | 29,727 | 0.637 |
| WeeklySeasonalNaive | 48,369 | 32,881 | 0.551 |

## 3. 与 r5e1 诊断对比

| 方法 | r5e1 RMSE | r20e1 RMSE | 变化 |
|------|:---:|:---:|:---:|
| FedAvg | 37,344 | **24,158** | -35% 🟢 |
| Independent | 25,765 | **15,921** | -38% 🟢 |
| NaiveLastValue | 19,419 | 19,419 | 不变 (non-learning) |
| CalendarProfileNaive | 32,194 | 32,194 | 不变 (profile-based) |
| DailySeasonalNaive | 45,406 | 45,406 | 不变 (lag-based) |
| WeeklySeasonalNaive | 48,369 | 48,369 | 不变 (lag-based) |

**关键发现**:
- FedAvg 从 r5e1 的 37,344 降至 24,158（-35%），说明 rounds=20 对联邦训练有显著改善
- Independent 从 25,765 降至 15,921，本地训练也受益于更多 epoch
- NaiveLastValue 仍然强于 FedAvg（19,419 vs 24,158），说明数据具有强时间惯性
- Independent (15,921) 优于 NaiveLastValue (19,419)，说明有可学习的时序依赖超过简单 persistence
- FedAvg 排名第 4/6，仅优于 Daily/WeeklySeasonalNaive

## 4. 输出文件清单

- `main_metrics.csv` / `.json` — 汇总指标
- `client_metrics.csv` / `.json` — 各 client 指标
- `convergence_history.csv` / `.json` — FedAvg 收敛曲线
- `prediction_samples.csv` / `.json` — 预测样本
- `run_config.json` — 运行配置
- `split_summary.json` — 数据划分摘要
- `environment_summary.json` — 运行环境
- `input_scaler.json` / `target_scaler.json` — 归一化参数

## 5. 说明

- 这是 **formal 结果**（rounds=20），不是 smoke test（rounds=5）
- 不提交 `results/` 目录，只提交本文档
- 该结果为实验 1 的首次正式运行基准
