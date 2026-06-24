# 图表说明

- `region_client_distribution`: 展示每个区域客户端包含的 active pooled regions 数量与估计样本量。
- `region_non_iid_summary`: 展示跨客户端的 region_count、sample_count、mean_total_flow 等统计离散程度。
- `region_main_metrics_comparison`: 展示 FedAvg 与 Independent 的总体指标对比。
- `region_client_rmse`: 展示各区域客户端的测试集 RMSE。
- `region_convergence_curve`: 展示 FedAvg 通信轮次上的 train loss、val RMSE 与 test RMSE。
- `region_prediction_vs_truth`: 展示部分样本的预测值与真实值对比。