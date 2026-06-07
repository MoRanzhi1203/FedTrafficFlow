# 仿真实验输出资产审查 v4.1

## 1. 审查范围
- 遍历目录：`results/simulation_experiments/cnn_fed_base`、`results/simulation_experiments/gcn_fed_base`、`results/simulation_experiments/cnn_fed_enhanced_experiments`、`results/simulation_experiments/gcn_fed_enhanced_experiments`、`results/simulation_experiments/fed_robustness_experiments`、`results/simulation_experiments/fed_robustness`。
- 文件类型：`CSV`、`PNG`、`PDF`、`MD`。
- 审查目标：识别适合正文、适合附录、不建议使用的资产，并核对 5-seed 覆盖、PNG/PDF 配对与路径口径。

## 2. 总体结论
- 基础 `CNN` 与 `GCN` 主结果目录均已具备完整 5-seed 强证据，正文应优先使用 `multi_seed_mean_std_metrics.png/.pdf` 与对应 `multi_seed_summary.csv`。
- 基础收敛性正文仍可使用 `convergence_curve.png/.pdf`；其统计补充应来自 `multi_seed_convergence_summary.csv`。
- `CNN` 增强目录中的 `noniid/client_scale/feature_ablation` 已存在完整 5-seed 原始 CSV，但现有 `paper_ready` 的 `fedavg_only` 图更接近旧 3-seed 资产，因此更适合做趋势性配图，强统计结论应以表格和完整 5-seed CSV 为准。
- `GCN` 增强默认主结果已具备 5-seed 强证据，但固定图/动态图比较图仍属于单种子趋势证据。
- 鲁棒性正文主图宜从旧目录 `fed_robustness/paper_ready/*.png` 切换为新目录 `fed_robustness_experiments/multi_seed_robustness_mean_std_metrics.png/.pdf`。旧目录三张 `FedAvg-only` 图更适合作为附录。
- 六个目录中的 `PNG` 与 `PDF` 图件均已形成同名配对，未发现缺 PDF 配对的图。
- 当前 `V4` 中相对路径整体有效，但部分正文图仍偏向旧单次/旧口径资产，宜替换为 5-seed 汇总图。

## 3. 5-seed 强证据与趋势性证据
### 3.1 可作为 5-seed 强证据的核心结果
- `results/simulation_experiments/cnn_fed_base/multi_seed_raw_results.csv`
- `results/simulation_experiments/cnn_fed_base/multi_seed_summary.csv`
- `results/simulation_experiments/cnn_fed_base/multi_seed_convergence_raw.csv`
- `results/simulation_experiments/cnn_fed_base/multi_seed_convergence_summary.csv`
- `results/simulation_experiments/cnn_fed_base/multi_seed_mean_std_metrics.png/.pdf`
- `results/simulation_experiments/gcn_fed_base/multi_seed_raw_results.csv`
- `results/simulation_experiments/gcn_fed_base/multi_seed_summary.csv`
- `results/simulation_experiments/gcn_fed_base/multi_seed_convergence_raw.csv`
- `results/simulation_experiments/gcn_fed_base/multi_seed_convergence_summary.csv`
- `results/simulation_experiments/gcn_fed_base/multi_seed_mean_std_metrics.png/.pdf`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_raw_results.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_summary.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_convergence_raw.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_convergence_summary.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_mean_std.png/.pdf`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_raw_results.csv`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_summary.csv`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_convergence_raw.csv`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_convergence_summary.csv`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_mean_std.png/.pdf`
- `results/simulation_experiments/fed_robustness_experiments/multi_seed_raw_results.csv`
- `results/simulation_experiments/fed_robustness_experiments/multi_seed_summary.csv`
- `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_mean_std_metrics.png/.pdf`
### 3.2 只能作为趋势性证据的结果
- results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png/.pdf：图件来源更接近旧 `42/2024/2025` FedAvg-only 资产，适合展示趋势，不宜替代 5-seed 表格。
- results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png/.pdf：同上。
- results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png/.pdf：同上。
- results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png/.pdf：当前仍为单种子趋势证据。
- results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_metrics.csv：仅含 seed=42。
- results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_scale_metrics.csv：仅含 seed=42。
- results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_noniid_metrics.csv：仅含 seed=42。
- results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png/.pdf、`...communication_delay...`、`...gradient_noise...`：更适合作为附录图。

## 4. 各目录资产清单与 CSV 审查
### `results/simulation_experiments/cnn_fed_base`
- 统计：CSV 15 个，PNG 11 个，PDF 11 个，MD 0 个。
- CSV 清单：
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_distribution.csv`
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_sample_size.csv`
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_timeseries.csv`
  - `results/simulation_experiments/cnn_fed_base/base_dataset_node_heatmap.csv`
  - `results/simulation_experiments/cnn_fed_base/base_dataset_split_overview.csv`
  - `results/simulation_experiments/cnn_fed_base/base_dataset_summary.csv`
  - `results/simulation_experiments/cnn_fed_base/convergence_history.csv`
  - `results/simulation_experiments/cnn_fed_base/main_metrics.csv`
  - `results/simulation_experiments/cnn_fed_base/main_predictions.csv`
  - `results/simulation_experiments/cnn_fed_base/main_summary.csv`
  - `results/simulation_experiments/cnn_fed_base/multi_seed_convergence_raw.csv`
  - `results/simulation_experiments/cnn_fed_base/multi_seed_convergence_summary.csv`
  - `results/simulation_experiments/cnn_fed_base/multi_seed_improvement_summary.csv`
  - `results/simulation_experiments/cnn_fed_base/multi_seed_raw_results.csv`
  - `results/simulation_experiments/cnn_fed_base/multi_seed_summary.csv`
- PNG 清单：
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_sample_size.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_timeseries.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_node_heatmap.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_split_overview.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/convergence_curve.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/main_predictions_comparison.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/multi_seed_mean_std_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/multi_seed_rmse_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_base/multi_seed_rmse_seed_pairing.png`，同名 PDF：是
- PDF 清单：
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_sample_size.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_client_timeseries.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_node_heatmap.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/base_dataset_split_overview.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/convergence_curve.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/main_predictions_comparison.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/multi_seed_mean_std_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/multi_seed_rmse_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_base/multi_seed_rmse_seed_pairing.pdf`，同名 PNG：是
- MD 清单：
  - 无

| CSV | 行数 | 列名 | 含 seed | 含 method | 含 scenario | multi_seed 是否覆盖 42/2024/3407/1234/5678 |
|---|---:|---|---|---|---|---|
| `base_dataset_client_distribution.csv` | 24000 | client_id<br>traffic_flow | 否 | 否 | 否 | - |
| `base_dataset_client_sample_size.csv` | 5 | client_id<br>num_samples | 否 | 否 | 否 | - |
| `base_dataset_client_timeseries.csv` | 120 | client_id<br>time_step<br>traffic_flow | 否 | 否 | 否 | - |
| `base_dataset_node_heatmap.csv` | 960 | client_id<br>node_id<br>time_step<br>traffic_flow | 否 | 否 | 否 | - |
| `base_dataset_split_overview.csv` | 3 | split<br>num_samples<br>ratio | 否 | 否 | 否 | - |
| `base_dataset_summary.csv` | 5 | client_id<br>num_samples<br>num_nodes<br>seq_len<br>pred_len<br>train_size<br>val_size<br>test_size<br>mean_flow<br>std_flow<br>min_flow<br>max_flow | 否 | 否 | 否 | - |
| `convergence_history.csv` | 75 | seed<br>round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape | 是 | 是 | 否 | - |
| `main_metrics.csv` | 50 | seed<br>method<br>client_id<br>mse<br>rmse<br>mape<br>mae<br>r2 | 是 | 是 | 否 | - |
| `main_predictions.csv` | 2000 | seed<br>method<br>client_id<br>sample_id<br>y_true<br>y_pred | 是 | 是 | 否 | - |
| `main_summary.csv` | 2 | method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std<br>r2_mean<br>r2_std | 否 | 是 | 否 | - |
| `multi_seed_convergence_raw.csv` | 75 | seed<br>round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_convergence_summary.csv` | 15 | method<br>round<br>n<br>avg_train_loss_mean<br>avg_train_loss_std<br>avg_train_loss_ci95_lower<br>avg_train_loss_ci95_upper<br>avg_val_loss_mean<br>avg_val_loss_std<br>avg_val_loss_ci95_lower<br>avg_val_loss_ci95_upper<br>avg_val_rmse_mean<br>avg_val_rmse_std<br>avg_val_rmse_ci95_lower<br>avg_val_rmse_ci95_upper | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_improvement_summary.csv` | 4 | experiment<br>baseline_method<br>enhanced_method<br>metric<br>mean_improvement_percent<br>std_improvement_percent<br>improved_seed_count<br>total_seed_count<br>improved_seed_ratio<br>per_seed_improved | 否 | 否 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_raw_results.csv` | 10 | experiment<br>method<br>seed<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_summary.csv` | 16 | experiment<br>method<br>metric<br>mean<br>std<br>ci95_lower<br>ci95_upper<br>best<br>worst<br>n | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |

- 正文建议：
  - `multi_seed_mean_std_metrics.png/.pdf`：正文优先，五随机种子均值与标准差，适合主结果小节。
  - `convergence_curve.png/.pdf`：正文优先，用于收敛性分析。
- 不建议或需谨慎使用：
  - `main_metrics_comparison.png/.pdf` 不再适合作为正文主图，原因是已有更强的 `multi_seed_mean_std_metrics`。

### `results/simulation_experiments/gcn_fed_base`
- 统计：CSV 17 个，PNG 12 个，PDF 12 个，MD 0 个。
- CSV 清单：
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_distribution.csv`
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_sample_size.csv`
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_timeseries.csv`
  - `results/simulation_experiments/gcn_fed_base/base_dataset_node_heatmap.csv`
  - `results/simulation_experiments/gcn_fed_base/base_dataset_split_overview.csv`
  - `results/simulation_experiments/gcn_fed_base/base_dataset_summary.csv`
  - `results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.csv`
  - `results/simulation_experiments/gcn_fed_base/base_graph_summary.csv`
  - `results/simulation_experiments/gcn_fed_base/convergence_history.csv`
  - `results/simulation_experiments/gcn_fed_base/main_metrics.csv`
  - `results/simulation_experiments/gcn_fed_base/main_predictions.csv`
  - `results/simulation_experiments/gcn_fed_base/main_summary.csv`
  - `results/simulation_experiments/gcn_fed_base/multi_seed_convergence_raw.csv`
  - `results/simulation_experiments/gcn_fed_base/multi_seed_convergence_summary.csv`
  - `results/simulation_experiments/gcn_fed_base/multi_seed_improvement_summary.csv`
  - `results/simulation_experiments/gcn_fed_base/multi_seed_raw_results.csv`
  - `results/simulation_experiments/gcn_fed_base/multi_seed_summary.csv`
- PNG 清单：
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_sample_size.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_timeseries.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_node_heatmap.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_split_overview.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/convergence_curve.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/main_metrics_comparison.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/main_predictions_comparison.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/multi_seed_mean_std_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/multi_seed_rmse_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_base/multi_seed_rmse_seed_pairing.png`，同名 PDF：是
- PDF 清单：
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_sample_size.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_client_timeseries.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_node_heatmap.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/base_dataset_split_overview.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/convergence_curve.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/main_metrics_comparison.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/main_predictions_comparison.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/multi_seed_mean_std_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/multi_seed_rmse_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_base/multi_seed_rmse_seed_pairing.pdf`，同名 PNG：是
- MD 清单：
  - 无

| CSV | 行数 | 列名 | 含 seed | 含 method | 含 scenario | multi_seed 是否覆盖 42/2024/3407/1234/5678 |
|---|---:|---|---|---|---|---|
| `base_dataset_client_distribution.csv` | 24000 | client_id<br>traffic_flow | 否 | 否 | 否 | - |
| `base_dataset_client_sample_size.csv` | 5 | client_id<br>num_samples | 否 | 否 | 否 | - |
| `base_dataset_client_timeseries.csv` | 120 | client_id<br>time_step<br>traffic_flow | 否 | 否 | 否 | - |
| `base_dataset_node_heatmap.csv` | 960 | client_id<br>node_id<br>time_step<br>traffic_flow | 否 | 否 | 否 | - |
| `base_dataset_split_overview.csv` | 3 | split<br>num_samples<br>ratio | 否 | 否 | 否 | - |
| `base_dataset_summary.csv` | 5 | client_id<br>num_samples<br>mean_flow<br>std_flow<br>min_flow<br>max_flow | 否 | 否 | 否 | - |
| `base_graph_adjacency_matrix.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `base_graph_summary.csv` | 1 | num_nodes<br>num_edges<br>density<br>avg_degree | 否 | 否 | 否 | - |
| `convergence_history.csv` | 50 | round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape<br>seed | 是 | 是 | 否 | - |
| `main_metrics.csv` | 50 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>r2 | 是 | 是 | 否 | - |
| `main_predictions.csv` | 2000 | seed<br>method<br>client_id<br>sample_id<br>y_true<br>y_pred | 是 | 是 | 否 | - |
| `main_summary.csv` | 2 | method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `multi_seed_convergence_raw.csv` | 50 | round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape<br>seed | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_convergence_summary.csv` | 10 | method<br>round<br>n<br>avg_train_loss_mean<br>avg_train_loss_std<br>avg_train_loss_ci95_lower<br>avg_train_loss_ci95_upper<br>avg_val_loss_mean<br>avg_val_loss_std<br>avg_val_loss_ci95_lower<br>avg_val_loss_ci95_upper<br>avg_val_rmse_mean<br>avg_val_rmse_std<br>avg_val_rmse_ci95_lower<br>avg_val_rmse_ci95_upper | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_improvement_summary.csv` | 4 | experiment<br>baseline_method<br>enhanced_method<br>metric<br>mean_improvement_percent<br>std_improvement_percent<br>improved_seed_count<br>total_seed_count<br>improved_seed_ratio<br>per_seed_improved | 否 | 否 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_raw_results.csv` | 10 | experiment<br>method<br>seed<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_summary.csv` | 12 | experiment<br>method<br>metric<br>mean<br>std<br>ci95_lower<br>ci95_upper<br>best<br>worst<br>n | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |

- 正文建议：
  - `multi_seed_mean_std_metrics.png/.pdf`：正文优先，五随机种子均值与标准差，适合主结果小节。
  - `convergence_curve.png/.pdf`：正文优先，用于收敛性分析。
  - `base_graph_adjacency_matrix.png/.pdf`：正文可用，属于结构示意图而非统计强证据。
- 不建议或需谨慎使用：
  - `main_metrics_comparison.png/.pdf` 不再适合作为正文主图，原因是已有更强的 `multi_seed_mean_std_metrics`。

### `results/simulation_experiments/cnn_fed_enhanced_experiments`
- 统计：CSV 39 个，PNG 25 个，PDF 25 个，MD 0 个。
- CSV 清单：
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_metrics_fedavg.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary_fedavg.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_convergence_history.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_metrics_fedavg.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary_fedavg.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_lambda_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_lambda_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_predictions.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_metrics_fedavg.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary_fedavg.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_peak_metrics.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_peak_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_config.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_correlation_matrix.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_timeseries.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_incident_example.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_node_correlation_matrix.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_peak_pattern.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_convergence_raw.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_convergence_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_improvement_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_raw_results.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/multi_seed_summary.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.csv`
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.csv`
- PNG 清单：
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_convergence.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_lambda.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_comparison.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_predictions.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_convergence_curve.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_mean_std.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_rmse_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_seed_pairing.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_peak_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/convergence_curve.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_config.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_correlation_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_timeseries.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_incident_example.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_node_correlation_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_peak_pattern.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png`，同名 PDF：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png`，同名 PDF：是
- PDF 清单：
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_convergence.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_lambda.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_comparison.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_predictions.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_convergence_curve.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_mean_std.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_rmse_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_seed_pairing.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_peak_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/convergence_curve.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_config.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_correlation_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_client_timeseries.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_incident_example.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_node_correlation_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_peak_pattern.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf`，同名 PNG：是
  - `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf`，同名 PNG：是
- MD 清单：
  - 无

| CSV | 行数 | 列名 | 含 seed | 含 method | 含 scenario | multi_seed 是否覆盖 42/2024/3407/1234/5678 |
|---|---:|---|---|---|---|---|
| `cnn_enhanced_aggregation_metrics.csv` | 100 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_aggregation_summary.csv` | 4 | method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_client_metrics.csv` | 75 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_client_scale_metrics.csv` | 80 | seed<br>method<br>num_clients<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_client_scale_metrics_fedavg.csv` | 48 | num_clients<br>seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_client_scale_summary.csv` | 3 | num_clients<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_client_scale_summary_fedavg.csv` | 3 | num_clients<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_convergence_history.csv` | 100 | round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape<br>seed | 是 | 是 | 否 | - |
| `cnn_enhanced_feature_ablation_metrics.csv` | 125 | seed<br>method<br>feature_set<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_feature_ablation_metrics_fedavg.csv` | 75 | feature_set<br>seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_feature_ablation_summary.csv` | 5 | feature_set<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_feature_ablation_summary_fedavg.csv` | 5 | feature_set<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_lambda_metrics.csv` | 125 | seed<br>method<br>lambda_value<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_lambda_summary.csv` | 5 | lambda_value<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_main_metrics.csv` | 75 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>r2 | 是 | 是 | 否 | - |
| `cnn_enhanced_main_predictions.csv` | 8625 | workflow<br>method<br>seed<br>client_id<br>sample_id<br>y_true<br>y_pred<br>period | 是 | 是 | 否 | - |
| `cnn_enhanced_main_summary.csv` | 3 | method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_noniid_metrics.csv` | 75 | seed<br>method<br>noniid_level<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_noniid_metrics_fedavg.csv` | 45 | noniid_level<br>seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_noniid_summary.csv` | 3 | noniid_level<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_noniid_summary_fedavg.csv` | 3 | noniid_level<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `cnn_enhanced_peak_metrics.csv` | 90 | seed<br>method<br>client_id<br>period<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `cnn_enhanced_peak_summary.csv` | 6 | period<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `enhanced_dataset_client_config.csv` | 5 | client_id<br>distribution_type<br>traffic_pattern<br>sample_size<br>noise_level<br>base_flow<br>morning_mu<br>evening_mu<br>morning_amp<br>evening_amp<br>incident_prob | 否 | 否 | 否 | - |
| `enhanced_dataset_client_correlation_matrix.csv` | 25 | source_client<br>target_client<br>correlation | 否 | 否 | 否 | - |
| `enhanced_dataset_client_timeseries.csv` | 1200 | client_id<br>time_step<br>traffic_flow | 否 | 否 | 否 | - |
| `enhanced_dataset_distribution.csv` | 2560 | client_id<br>traffic_flow | 否 | 否 | 否 | - |
| `enhanced_dataset_incident_example.csv` | 240 | time_step<br>traffic_flow<br>incident_flag<br>client_id | 否 | 否 | 否 | - |
| `enhanced_dataset_node_correlation_matrix.csv` | 320 | client_id<br>source_node<br>target_node<br>correlation | 否 | 否 | 否 | - |
| `enhanced_dataset_peak_pattern.csv` | 120 | client_id<br>hour<br>traffic_flow | 否 | 否 | 否 | - |
| `enhanced_dataset_summary.csv` | 5 | client_id<br>num_samples<br>mean_flow<br>std_flow<br>min_flow<br>max_flow<br>incident_ratio | 否 | 否 | 否 | - |
| `multi_seed_convergence_raw.csv` | 100 | round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape<br>seed | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_convergence_summary.csv` | 20 | method<br>round<br>n<br>avg_train_loss_mean<br>avg_train_loss_std<br>avg_train_loss_ci95_lower<br>avg_train_loss_ci95_upper<br>avg_val_loss_mean<br>avg_val_loss_std<br>avg_val_loss_ci95_lower<br>avg_val_loss_ci95_upper<br>avg_val_rmse_mean<br>avg_val_rmse_std<br>avg_val_rmse_ci95_lower<br>avg_val_rmse_ci95_upper<br>avg_val_mae_mean<br>avg_val_mae_std<br>avg_val_mae_ci95_lower<br>avg_val_mae_ci95_upper<br>avg_val_mape_mean<br>avg_val_mape_std<br>avg_val_mape_ci95_lower<br>avg_val_mape_ci95_upper | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_improvement_summary.csv` | 4 | experiment<br>baseline_method<br>enhanced_method<br>metric<br>mean_improvement_percent<br>std_improvement_percent<br>improved_seed_count<br>total_seed_count<br>improved_seed_ratio<br>per_seed_improved | 否 | 否 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_raw_results.csv` | 15 | experiment<br>method<br>seed<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_summary.csv` | 18 | experiment<br>method<br>metric<br>mean<br>std<br>ci95_lower<br>ci95_upper<br>best<br>worst<br>n | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |
| `cnn_enhanced_client_scale_fedavg_only.csv` | 3 | num_clients<br>experiment_name<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std<br>source_seeds | 否 | 是 | 否 | - |
| `cnn_enhanced_feature_ablation_fedavg_only.csv` | 5 | feature_set<br>experiment_name<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std<br>source_seeds | 否 | 是 | 否 | - |
| `cnn_enhanced_noniid_fedavg_only.csv` | 3 | noniid_level<br>experiment_name<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std<br>source_seeds | 否 | 是 | 否 | - |

- 正文建议：
  - `cnn_enhanced_noniid_fedavg_only.png/.pdf`、`cnn_enhanced_client_scale_fedavg_only.png/.pdf`、`cnn_enhanced_feature_ablation_fedavg_only.png/.pdf`：正文可用，但仅宜作为趋势性配图。
  - `cnn_enhanced_multi_seed_mean_std.png/.pdf` 与 `cnn_enhanced_multi_seed_convergence_curve.png/.pdf`：更适合附录，用于补充 5-seed 稳定性。
- 不建议或需谨慎使用：
  - `cnn_enhanced_client_scale_metrics_fedavg.csv`、`cnn_enhanced_feature_ablation_metrics_fedavg.csv`、`cnn_enhanced_noniid_metrics_fedavg.csv` 仅覆盖 42/2024/2025，不建议作为 5-seed 强证据。

### `results/simulation_experiments/gcn_fed_enhanced_experiments`
- 统计：CSV 38 个，PNG 33 个，PDF 33 个，MD 0 个。
- CSV 清单：
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_config.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_correlation_matrix.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_timeseries.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_distribution.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_incident_example.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_node_correlation_matrix.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_peak_pattern.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_interaction.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_matrix.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_strength_matrix.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_adjacency_evening_peak.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_adjacency_morning_peak.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_adjacency_offpeak.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency_matrix.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_functional_similarity_matrix.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_graph_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_aggregation_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_aggregation_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_scale_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_congestion_delay_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_congestion_delay_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_convergence_history.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_lambda_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_lambda_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_noniid_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_peak_metrics.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_convergence_raw.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_convergence_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_improvement_summary.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_raw_results.csv`
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/multi_seed_summary.csv`
- PNG 清单：
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/convergence_curve.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_config.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_correlation_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_timeseries.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_distribution.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_incident_example.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_node_correlation_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_peak_pattern.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_distribution.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_interaction.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_strength_matrix.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_offpeak.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_dynamic_comparison.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_functional_similarity.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_peak_graph_change.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_aggregation.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_scale.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_congestion_delay_comp.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_convergence.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_lambda.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_results.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_convergence_curve.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_mean_std.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_rmse_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_seed_pairing.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_noniid.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_peak_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png`，同名 PDF：是
- PDF 清单：
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/convergence_curve.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_config.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_correlation_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_client_timeseries.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_distribution.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_incident_example.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_node_correlation_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_dataset_peak_pattern.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_distribution.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_interaction.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_delay_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_congestion_strength_matrix.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_offpeak.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_dynamic_comparison.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_functional_similarity.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_peak_graph_change.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_aggregation.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_scale.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_congestion_delay_comp.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_convergence.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_lambda.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_results.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_convergence_curve.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_mean_std.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_rmse_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_seed_pairing.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_noniid.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_peak_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf`，同名 PNG：是
- MD 清单：
  - 无

| CSV | 行数 | 列名 | 含 seed | 含 method | 含 scenario | multi_seed 是否覆盖 42/2024/3407/1234/5678 |
|---|---:|---|---|---|---|---|
| `enhanced_dataset_client_config.csv` | 5 | client_id<br>distribution_type<br>traffic_pattern<br>sample_size<br>noise_level<br>base_flow<br>morning_mu<br>evening_mu<br>morning_amp<br>evening_amp<br>incident_prob | 否 | 否 | 否 | - |
| `enhanced_dataset_client_correlation_matrix.csv` | 25 | source_client<br>target_client<br>correlation | 否 | 否 | 否 | - |
| `enhanced_dataset_client_timeseries.csv` | 1200 | client_id<br>time_step<br>traffic_flow | 否 | 否 | 否 | - |
| `enhanced_dataset_distribution.csv` | 2560 | client_id<br>traffic_flow | 否 | 否 | 否 | - |
| `enhanced_dataset_incident_example.csv` | 240 | time_step<br>traffic_flow<br>incident_flag<br>client_id | 否 | 否 | 否 | - |
| `enhanced_dataset_node_correlation_matrix.csv` | 320 | client_id<br>source_node<br>target_node<br>correlation | 否 | 否 | 否 | - |
| `enhanced_dataset_peak_pattern.csv` | 120 | client_id<br>hour<br>traffic_flow | 否 | 否 | 否 | - |
| `enhanced_dataset_summary.csv` | 5 | client_id<br>num_samples<br>mean_flow<br>std_flow<br>min_flow<br>max_flow<br>incident_ratio | 否 | 否 | 否 | - |
| `enhanced_gcn_congestion_delay.csv` | 64 | source_node<br>target_node<br>delay_rounds<br>strength | 否 | 否 | 否 | - |
| `enhanced_gcn_congestion_delay_interaction.csv` | 56 | source_node<br>target_node<br>delay_rounds<br>strength | 否 | 否 | 否 | - |
| `enhanced_gcn_congestion_delay_matrix.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_congestion_strength_matrix.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_dynamic_adjacency_evening_peak.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_dynamic_adjacency_morning_peak.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_dynamic_adjacency_offpeak.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_fixed_adjacency_matrix.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_functional_similarity_matrix.csv` | 8 | 0<br>1<br>2<br>3<br>4<br>5<br>6<br>7 | 否 | 否 | 否 | - |
| `enhanced_gcn_graph_summary.csv` | 7 | graph_type<br>num_nodes<br>mean_weight<br>max_weight<br>density | 否 | 否 | 否 | - |
| `gcn_enhanced_aggregation_metrics.csv` | 100 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `gcn_enhanced_aggregation_summary.csv` | 4 | method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `gcn_enhanced_client_metrics.csv` | 15 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `gcn_enhanced_client_scale_metrics.csv` | 16 | seed<br>method<br>num_clients<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `gcn_enhanced_congestion_delay_metrics.csv` | 20 | graph_type<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 否 | 是 | 否 | - |
| `gcn_enhanced_congestion_delay_summary.csv` | 4 | graph_type<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `gcn_enhanced_convergence_history.csv` | 40 | round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape<br>seed | 是 | 是 | 否 | - |
| `gcn_enhanced_fixed_vs_dynamic_metrics.csv` | 40 | graph_type<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape | 否 | 是 | 否 | - |
| `gcn_enhanced_fixed_vs_dynamic_summary.csv` | 8 | graph_type<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `gcn_enhanced_lambda_metrics.csv` | 125 | seed<br>method<br>lambda_value<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `gcn_enhanced_lambda_summary.csv` | 5 | lambda_value<br>method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `gcn_enhanced_main_metrics.csv` | 75 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>r2 | 是 | 是 | 否 | - |
| `gcn_enhanced_main_summary.csv` | 3 | method<br>mse_mean<br>mse_std<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `gcn_enhanced_noniid_metrics.csv` | 15 | seed<br>method<br>noniid_level<br>client_id<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `gcn_enhanced_peak_metrics.csv` | 18 | seed<br>method<br>client_id<br>period<br>mse<br>rmse<br>mae<br>mape | 是 | 是 | 否 | - |
| `multi_seed_convergence_raw.csv` | 40 | round<br>method<br>avg_train_loss<br>avg_val_loss<br>avg_val_rmse<br>avg_val_mae<br>avg_val_mape<br>seed | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_convergence_summary.csv` | 8 | method<br>round<br>n<br>avg_train_loss_mean<br>avg_train_loss_std<br>avg_train_loss_ci95_lower<br>avg_train_loss_ci95_upper<br>avg_val_loss_mean<br>avg_val_loss_std<br>avg_val_loss_ci95_lower<br>avg_val_loss_ci95_upper<br>avg_val_rmse_mean<br>avg_val_rmse_std<br>avg_val_rmse_ci95_lower<br>avg_val_rmse_ci95_upper<br>avg_val_mae_mean<br>avg_val_mae_std<br>avg_val_mae_ci95_lower<br>avg_val_mae_ci95_upper<br>avg_val_mape_mean<br>avg_val_mape_std<br>avg_val_mape_ci95_lower<br>avg_val_mape_ci95_upper | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_improvement_summary.csv` | 4 | experiment<br>baseline_method<br>enhanced_method<br>metric<br>mean_improvement_percent<br>std_improvement_percent<br>improved_seed_count<br>total_seed_count<br>improved_seed_ratio<br>per_seed_improved | 否 | 否 | 否 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_raw_results.csv` | 15 | experiment<br>method<br>seed<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round | 是 | 是 | 否 | 是：42/2024/3407/1234/5678 |
| `multi_seed_summary.csv` | 18 | experiment<br>method<br>metric<br>mean<br>std<br>ci95_lower<br>ci95_upper<br>best<br>worst<br>n | 否 | 是 | 否 | 无 seed 列，不能直接由该文件判定 |

- 正文建议：
  - `gcn_fixed_vs_dynamic_fedavg_only.png/.pdf`：正文可用，但应明确为单种子趋势性证据。
  - `enhanced_gcn_fixed_adjacency.png/.pdf`：正文可用，属于结构示意图。
  - `gcn_enhanced_multi_seed_mean_std.png/.pdf` 与 `gcn_enhanced_multi_seed_convergence_curve.png/.pdf`：附录优先，用于补充 5-seed 稳定性。
- 不建议或需谨慎使用：
  - `gcn_enhanced_client_metrics.csv`、`gcn_enhanced_client_scale_metrics.csv`、`gcn_enhanced_noniid_metrics.csv` 仅含 seed=42，不建议作为强统计结论依据。

### `results/simulation_experiments/fed_robustness_experiments`
- 统计：CSV 10 个，PNG 8 个，PDF 8 个，MD 0 个。
- CSV 清单：
  - `results/simulation_experiments/fed_robustness_experiments/fed_client_dropout_metrics.csv`
  - `results/simulation_experiments/fed_robustness_experiments/fed_client_dropout_summary.csv`
  - `results/simulation_experiments/fed_robustness_experiments/fed_communication_cost.csv`
  - `results/simulation_experiments/fed_robustness_experiments/fed_communication_delay_metrics.csv`
  - `results/simulation_experiments/fed_robustness_experiments/fed_communication_delay_summary.csv`
  - `results/simulation_experiments/fed_robustness_experiments/fed_gradient_noise_metrics.csv`
  - `results/simulation_experiments/fed_robustness_experiments/fed_gradient_noise_summary.csv`
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_improvement_summary.csv`
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_raw_results.csv`
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_summary.csv`
- PNG 清单：
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_client_dropout.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_communication_cost.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_communication_delay.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_gradient_noise.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_improvement_heatmap.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_mean_std_metrics.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_rmse_boxplot.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_seed_pairing.png`，同名 PDF：是
- PDF 清单：
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_client_dropout.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_communication_cost.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_communication_delay.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/fed_robustness_gradient_noise.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_improvement_heatmap.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_mean_std_metrics.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_rmse_boxplot.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_seed_pairing.pdf`，同名 PNG：是
- MD 清单：
  - 无

| CSV | 行数 | 列名 | 含 seed | 含 method | 含 scenario | multi_seed 是否覆盖 42/2024/3407/1234/5678 |
|---|---:|---|---|---|---|---|
| `fed_client_dropout_metrics.csv` | 150 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round<br>dropout_rate<br>experiment<br>scenario_type<br>scenario<br>delay_rounds<br>noise_std | 是 | 是 | 是 | - |
| `fed_client_dropout_summary.csv` | 6 | dropout_rate<br>method<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `fed_communication_cost.csv` | 24 | model_type<br>num_clients<br>rounds<br>num_parameters<br>parameter_size_mb<br>total_communication_mb | 否 | 否 | 否 | - |
| `fed_communication_delay_metrics.csv` | 150 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round<br>delay_rounds<br>experiment<br>scenario_type<br>scenario<br>dropout_rate<br>noise_std | 是 | 是 | 是 | - |
| `fed_communication_delay_summary.csv` | 6 | delay_rounds<br>method<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `fed_gradient_noise_metrics.csv` | 150 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round<br>noise_std<br>experiment<br>scenario_type<br>scenario<br>dropout_rate<br>delay_rounds | 是 | 是 | 是 | - |
| `fed_gradient_noise_summary.csv` | 6 | noise_std<br>method<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `multi_seed_improvement_summary.csv` | 36 | experiment<br>scenario<br>baseline_method<br>enhanced_method<br>metric<br>mean_improvement_percent<br>std_improvement_percent<br>improved_seed_count<br>total_seed_count<br>per_seed_improved | 否 | 否 | 是 | 无 seed 列，不能直接由该文件判定 |
| `multi_seed_raw_results.csv` | 90 | experiment<br>scenario_type<br>scenario<br>method<br>seed<br>dropout_rate<br>delay_rounds<br>noise_std<br>mse<br>rmse<br>mae<br>mape<br>r2<br>final_loss<br>best_loss<br>communication_rounds<br>convergence_round | 是 | 是 | 是 | 是：42/2024/3407/1234/5678 |
| `multi_seed_summary.csv` | 108 | experiment<br>scenario<br>method<br>metric<br>mean<br>std<br>ci95_lower<br>ci95_upper<br>best<br>worst<br>n | 否 | 是 | 是 | 无 seed 列，不能直接由该文件判定 |

- 正文建议：
  - `multi_seed_robustness_mean_std_metrics.png/.pdf`：正文优先，覆盖完整 5-seed，优于旧单场景图。
  - `multi_seed_robustness_rmse_boxplot.png/.pdf`、`multi_seed_robustness_seed_pairing.png/.pdf`、`multi_seed_robustness_improvement_heatmap.png/.pdf`：附录优先。
- 不建议或需谨慎使用：
  - 无新增禁用项。

### `results/simulation_experiments/fed_robustness`
- 统计：CSV 7 个，PNG 7 个，PDF 7 个，MD 0 个。
- CSV 清单：
  - `results/simulation_experiments/fed_robustness/fed_client_dropout_metrics.csv`
  - `results/simulation_experiments/fed_robustness/fed_client_dropout_summary.csv`
  - `results/simulation_experiments/fed_robustness/fed_communication_cost.csv`
  - `results/simulation_experiments/fed_robustness/fed_communication_delay_metrics.csv`
  - `results/simulation_experiments/fed_robustness/fed_communication_delay_summary.csv`
  - `results/simulation_experiments/fed_robustness/fed_gradient_noise_metrics.csv`
  - `results/simulation_experiments/fed_robustness/fed_gradient_noise_summary.csv`
- PNG 清单：
  - `results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness/fed_robustness_communication_cost.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png`，同名 PDF：是
  - `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png`，同名 PDF：是
- PDF 清单：
  - `results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness/fed_robustness_communication_cost.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf`，同名 PNG：是
  - `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf`，同名 PNG：是
- MD 清单：
  - 无

| CSV | 行数 | 列名 | 含 seed | 含 method | 含 scenario | multi_seed 是否覆盖 42/2024/3407/1234/5678 |
|---|---:|---|---|---|---|---|
| `fed_client_dropout_metrics.csv` | 90 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>dropout_rate | 是 | 是 | 否 | - |
| `fed_client_dropout_summary.csv` | 6 | dropout_rate<br>method<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `fed_communication_cost.csv` | 24 | model_type<br>num_clients<br>rounds<br>num_parameters<br>parameter_size_mb<br>total_communication_mb | 否 | 否 | 否 | - |
| `fed_communication_delay_metrics.csv` | 90 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>delay_rounds | 是 | 是 | 否 | - |
| `fed_communication_delay_summary.csv` | 6 | delay_rounds<br>method<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |
| `fed_gradient_noise_metrics.csv` | 90 | seed<br>method<br>client_id<br>mse<br>rmse<br>mae<br>mape<br>noise_std | 是 | 是 | 否 | - |
| `fed_gradient_noise_summary.csv` | 6 | noise_std<br>method<br>rmse_mean<br>rmse_std<br>mae_mean<br>mae_std<br>mape_mean<br>mape_std | 否 | 是 | 否 | - |

- 正文建议：
  - `paper_ready` 子目录下三张 `*_fedavg_only.png/.pdf`：建议移至附录，适合作为单类扰动的补充可视化。
- 不建议或需谨慎使用：
  - 旧目录下 `fed_client_dropout_summary.csv`、`fed_communication_delay_summary.csv`、`fed_gradient_noise_summary.csv` 与新目录同名文件口径不同，不建议继续作为正文主表来源。

## 5. 旧目录 `fed_robustness` 与新目录 `fed_robustness_experiments` 的口径差异
- 旧目录与新目录存在同名 CSV，但不完全同口径。新目录在原有指标基础上新增了 `r2`、`final_loss`、`best_loss`、`communication_rounds`、`convergence_round`、`experiment`、`scenario_type`、`scenario` 等字段。
- 新目录三类 `*_metrics.csv` 的行数均从旧目录的 90 增长到 150，表明其已扩展为 5-seed 全覆盖结果；旧目录仅覆盖 `42/2024/2025`。
- 同名 `*_summary.csv` 在列名上保持一致，但数值口径已更新。例如 `fed_client_dropout_summary.csv` 中 `dropout_rate=0.0, method=FedAvg` 的 `rmse_mean` 从旧目录的 `7.9281` 更新为新目录的 `7.7210`，`mape_mean` 也从 `66.0344` 更新为 `50.4387`。因此，正文和补充表应统一以新目录 `fed_robustness_experiments` 为准。
- 旧目录的价值主要在于保留了 `paper_ready` 子目录下三张 `FedAvg-only` 视觉资产；这些图件适合附录，不宜继续作为正文主图。

## 6. 正文与附录建议
### 6.1 正文优先
- 基础 CNN：`results/simulation_experiments/cnn_fed_base/multi_seed_mean_std_metrics.png/.pdf`
- 基础 GCN：`results/simulation_experiments/gcn_fed_base/multi_seed_mean_std_metrics.png/.pdf`
- 基础收敛：`results/simulation_experiments/cnn_fed_base/convergence_curve.png/.pdf` 与 `results/simulation_experiments/gcn_fed_base/convergence_curve.png/.pdf`
- 鲁棒性汇总：`results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_mean_std_metrics.png/.pdf`
- 结构示意：`results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png/.pdf`、`results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png/.pdf`
- 动态图趋势图：`results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png/.pdf`，但需明确为趋势性证据。
### 6.2 附录优先
- `results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png/.pdf` 与 `results/simulation_experiments/gcn_fed_base/main_metrics_comparison.png/.pdf`：已被更强的 multi-seed 主图替代。
- `results/simulation_experiments/cnn_fed_base/multi_seed_rmse_boxplot.png/.pdf`、`multi_seed_rmse_seed_pairing.png/.pdf`：适合展示分布与 seed 配对。
- `results/simulation_experiments/gcn_fed_base/multi_seed_rmse_boxplot.png/.pdf`、`multi_seed_rmse_seed_pairing.png/.pdf`：同上。
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_multi_seed_mean_std.png/.pdf`、`cnn_enhanced_multi_seed_convergence_curve.png/.pdf`：适合补充增强默认场景的 5-seed 稳定性。
- `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_multi_seed_mean_std.png/.pdf`、`gcn_enhanced_multi_seed_convergence_curve.png/.pdf`：同上。
- `results/simulation_experiments/fed_robustness_experiments/multi_seed_robustness_rmse_boxplot.png/.pdf`、`multi_seed_robustness_seed_pairing.png/.pdf`、`multi_seed_robustness_improvement_heatmap.png/.pdf`。
- 旧目录 `results/simulation_experiments/fed_robustness/paper_ready/*.png/.pdf`：作为单类扰动的附录可视化。
### 6.3 不建议作为正文主证据
- 任何包含 `Proposed`、`Loss-weighted`、`Data-loss weighted` 作为主比较对象的图表。
- 仅覆盖 `42/2024/2025` 或仅含 seed=42 的旧 `fedavg_only`/单种子 CSV。
- 旧目录 `fed_robustness/*.csv` 的 summary 值，因其已被新目录更新。

## 7. 路径引用与编号检查建议
- 当前 `V4` 中引用的相对路径未发现失效文件。
- 但从资产强度看，基础主结果图建议替换为 `multi_seed_mean_std_metrics`，鲁棒性主图建议替换为 `multi_seed_robustness_mean_std_metrics`。
- Markdown 正文仍应继续使用中文图/表编号，并在每张 PNG 后标注同名 PDF 路径，避免后续 PDF/LaTeX 版本出现 `Figure/Table` 混排与编号跳跃。
- 若后续生成 PDF，建议避免在正文中并列放入过多单类扰动图，以降低表格断裂和跨页编号重复的概率。

