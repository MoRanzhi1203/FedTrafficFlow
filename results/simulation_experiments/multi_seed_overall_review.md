# Multi-seed Overall Review

## Review Scope
- This round uses five seeds: `42, 2024, 3407, 1234, 5678`.
- The paper narrative is centered on `FedAvg` as the core method for federated traffic flow prediction.
- Therefore, the review focuses on `FedAvg` stability, convergence, and robustness rather than framing `FedAvg` as a disposable baseline.
- In enhanced experiment directories, `Proposed` is treated only as a `FedAvg-based enhanced setting` or `FedAvg variant`.

## Directory Status

### `cnn_fed_base`
- This directory completed the target `5-seed` main experiment and `5-seed` convergence experiment.
- Core CSV files exist and are non-empty: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, `multi_seed_improvement_summary.csv`, `multi_seed_stability_report.txt`, `multi_seed_convergence_raw.csv`, `multi_seed_convergence_summary.csv`.
- Main figures exist: `main_metrics_comparison.png`, `main_predictions_comparison.png`, `multi_seed_mean_std_metrics.png`, `multi_seed_rmse_boxplot.png`, `multi_seed_rmse_seed_pairing.png`, `convergence_curve.png`.
- CSV audit: `FedAvg` and `Independent` both cover all `5` seeds in main results; `FedAvg` covers all `5` seeds in convergence results.
- Base data update: the original nearly homogeneous setting has been replaced by a `balanced but moderately weak heterogeneous` setting. All `5` clients still keep `200` samples, `8` nodes, and the same `24 -> 1` input-output setup, while controlled differences in flow level, peak amplitude, phase shift, noise level, and local trend are introduced.
- Weak-heterogeneity audit: `target_mean CV=0.094936`, `target_std CV=0.082151`, `balanced weak heterogeneity=YES`.
- FedAvg main metrics: `RMSE=0.070911 +- 0.005389`, `MAE=0.056805 +- 0.004936`, `MAPE=5.680205 +- 0.493599`, `R2=0.844617 +- 0.024160`.
- FedAvg convergence at final round: `Val RMSE=0.074594 +- 0.007916`, `Val Loss=0.006354 +- 0.001324`.
- Audit issues: no empty table, no missing seed, no all-seed-identical anomaly.

### `gcn_fed_base`
- This directory completed the target `5-seed` main experiment and `5-seed` convergence experiment.
- Core CSV files exist and are non-empty: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, `multi_seed_improvement_summary.csv`, `multi_seed_stability_report.txt`, `multi_seed_convergence_raw.csv`, `multi_seed_convergence_summary.csv`.
- Main figures exist: `main_metrics_comparison.png`, `main_predictions_comparison.png`, `multi_seed_mean_std_metrics.png`, `multi_seed_rmse_boxplot.png`, `multi_seed_rmse_seed_pairing.png`, `convergence_curve.png`.
- CSV audit: `FedAvg` and `Independent` both cover all `5` seeds in main results; `FedAvg` covers all `5` seeds in convergence results.
- Base data update: the GCN base workflow now reuses the same `balanced but moderately weak heterogeneous` synthetic dataset as the CNN base workflow, preserving equal sample size and graph/input-output settings while avoiding fully identical client data.
- Weak-heterogeneity audit: `target_mean CV=0.094936`, `target_std CV=0.082151`, `balanced weak heterogeneity=YES`.
- FedAvg main metrics: `RMSE=0.058872 +- 0.003001`, `MAE=0.047355 +- 0.002702`, `MAPE=4.735208 +- 0.269947`, `R2=0.890190 +- 0.019319`.
- FedAvg convergence at final round: `Val RMSE=0.058899 +- 0.003191`, `Val Loss=0.004557 +- 0.000948`.
- Audit issues: no empty table, no missing seed, no all-seed-identical anomaly.

### `cnn_fed_enhanced_experiments`
- This directory has now completed both the target `5-seed` main experiment and the target `5-seed` convergence experiment for the enhanced FedAvg setting.
- Core CSV files exist and are non-empty: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, `multi_seed_improvement_summary.csv`, `multi_seed_stability_report.txt`, `multi_seed_convergence_raw.csv`, `multi_seed_convergence_summary.csv`.
- Main and convergence figures exist: `cnn_enhanced_multi_seed_mean_std.png`, `cnn_enhanced_multi_seed_rmse_boxplot.png`, `cnn_enhanced_multi_seed_seed_pairing.png`, `convergence_curve.png`, `cnn_enhanced_multi_seed_convergence_curve.png`.
- CSV audit: `FedAvg`, `Independent`, and `Proposed` each cover all `5` seeds in main results; `FedAvg` and `Proposed` each cover all `5` seeds in convergence results.
- FedAvg-based enhanced setting main metrics: `RMSE=7.105499 +- 0.455617`, `MAE=5.519366 +- 0.523906`, `MAPE=39.171213 +- 23.792808`, `R2=0.330167 +- 0.060420`.
- FedAvg-variant convergence at final round:
- `FedAvg`: `Val RMSE=7.317845 +- 0.677953`, `Val MAE=5.401517 +- 0.489881`, `Val MAPE=21.254626 +- 30.641247`.
- `Proposed` field is retained in code/results and should be described only as a `FedAvg-based enhanced setting`: `Val RMSE=7.314624 +- 0.680359`, `Val MAE=5.439574 +- 0.536262`, `Val MAPE=21.616120 +- 31.396728`.
- Status update: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, and `multi_seed_improvement_summary.csv` have been refreshed from the earlier `2-seed` state to full `5-seed` outputs.
- Audit issues: no empty table, no missing seed, no all-seed-identical anomaly.

### `gcn_fed_enhanced_experiments`
- This directory has now completed both the target `5-seed` main experiment and the target `5-seed` convergence experiment for the enhanced FedAvg setting.
- Core CSV files exist and are non-empty: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, `multi_seed_improvement_summary.csv`, `multi_seed_stability_report.txt`, `multi_seed_convergence_raw.csv`, `multi_seed_convergence_summary.csv`.
- Main and convergence figures exist: `gcn_enhanced_multi_seed_mean_std.png`, `gcn_enhanced_multi_seed_rmse_boxplot.png`, `gcn_enhanced_multi_seed_seed_pairing.png`, `convergence_curve.png`, `gcn_enhanced_multi_seed_convergence_curve.png`.
- CSV audit: `FedAvg`, `Independent`, and `Proposed` each cover all `5` seeds in main results; `FedAvg` and `Proposed` each cover all `5` seeds in convergence results.
- FedAvg variant main metrics: `RMSE=6.597295 +- 0.283484`, `MAE=4.986027 +- 0.341326`, `MAPE=37.811195 +- 22.980082`, `R2=0.386398 +- 0.036648`.
- FedAvg-variant convergence at final round:
- `FedAvg`: `Val RMSE=7.558264 +- 0.591032`, `Val MAE=5.689360 +- 0.353361`, `Val MAPE=23.389697 +- 35.407232`.
- `Proposed` field is retained in code/results and should be described only as a `FedAvg variant`: `Val RMSE=7.410310 +- 0.600537`, `Val MAE=5.566195 +- 0.383980`, `Val MAPE=22.198595 +- 33.100592`.
- Status update: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, and `multi_seed_improvement_summary.csv` have been refreshed from the earlier `2-seed` state to full `5-seed` outputs.
- Audit issues: no empty table, no missing seed, no all-seed-identical anomaly.

### `fed_robustness_experiments`
- This directory completed the target `5-seed` robustness experiment.
- Core CSV files exist and are non-empty: `multi_seed_raw_results.csv`, `multi_seed_summary.csv`, `multi_seed_improvement_summary.csv`, `multi_seed_stability_report.txt`.
- Robustness figures exist: `multi_seed_robustness_mean_std_metrics.png`, `multi_seed_robustness_rmse_boxplot.png`, `multi_seed_robustness_seed_pairing.png`, `multi_seed_robustness_improvement_heatmap.png`.
- Convergence CSV files are not applicable to this workflow.
- CSV audit: every `scenario + method` combination covers all `5` seeds.
- FedAvg scenario-level results:
- `client_dropout@0.0`: `RMSE=7.721048 +- 0.524764`, `MAE=6.087983 +- 0.678765`, `MAPE=50.438664 +- 37.884784`, `R2=0.186652 +- 0.103238`.
- `client_dropout@0.2`: `RMSE=7.576755 +- 0.565713`, `MAE=5.947481 +- 0.638244`, `MAPE=48.366720 +- 35.220718`, `R2=0.221249 +- 0.084544`.
- `client_dropout@0.4`: `RMSE=7.588212 +- 0.601661`, `MAE=5.953370 +- 0.644414`, `MAPE=46.496383 +- 35.393438`, `R2=0.228854 +- 0.080573`.
- `communication_delay@0`: `RMSE=7.721048 +- 0.524764`, `MAE=6.087983 +- 0.678765`, `MAPE=50.438664 +- 37.884784`, `R2=0.186652 +- 0.103238`.
- `communication_delay@1`: `RMSE=7.978335 +- 0.647706`, `MAE=6.364362 +- 0.819022`, `MAPE=54.376834 +- 43.869958`, `R2=0.131491 +- 0.118395`.
- `communication_delay@2`: `RMSE=7.831860 +- 0.590530`, `MAE=6.222507 +- 0.718590`, `MAPE=53.118548 +- 42.948119`, `R2=0.164749 +- 0.068380`.
- `gradient_noise@0.0`: `RMSE=7.721048 +- 0.524764`, `MAE=6.087983 +- 0.678765`, `MAPE=50.438664 +- 37.884784`, `R2=0.186652 +- 0.103238`.
- `gradient_noise@0.02`: `RMSE=7.720291 +- 0.414481`, `MAE=6.083675 +- 0.491804`, `MAPE=46.523824 +- 31.112444`, `R2=0.209732 +- 0.126520`.
- `gradient_noise@0.05`: `RMSE=8.027318 +- 0.366351`, `MAE=6.391560 +- 0.485016`, `MAPE=50.997738 +- 37.498544`, `R2=0.122224 +- 0.116608`.
- Overall interpretation: `FedAvg` remains relatively stable under client dropout and mild noise, while larger communication delay and stronger noise introduce a measurable but bounded degradation.
- Audit issues: no empty table, no missing seed, no duplicate scenario-method seed coverage issue, no all-seed-identical anomaly.

## Cross-directory Findings
- `cnn_fed_base`, `gcn_fed_base`, `cnn_fed_enhanced_experiments`, `gcn_fed_enhanced_experiments`, and `fed_robustness_experiments` completed their target `5-seed` outputs for this round.
- The base CNN and base GCN data-generation logic has been updated from a fully homogeneous or near-homogeneous setup to a `balanced but moderately weak heterogeneous` scenario. All clients still share the same sample size, node scale, and input-output window, while controlled differences in flow level, peak amplitude, phase shift, and noise are introduced.
- `cnn_fed_enhanced_experiments` and `gcn_fed_enhanced_experiments` now contain complete `5-seed` main results and `5-seed` convergence results.
- Across all targeted `5-seed` outputs, the seed set is exactly `42, 2024, 3407, 1234, 5678`.
- No targeted CSV is empty.
- No targeted `method` or `scenario + method` group is missing seeds.
- No targeted group shows an abnormal all-seed-identical metric pattern.

## Paper-oriented Notes
- The core conclusion of this round is that `FedAvg` demonstrates reproducible behavior across multiple random seeds in federated traffic flow prediction.
- On both CNN and GCN structures, `FedAvg` shows stable main metrics and a clear convergence trend when results are reported as `mean +- std`.
- Under federated perturbation scenarios including client dropout, communication delay, and gradient noise, `FedAvg` preserves workable predictive performance with moderate variation across seeds.
- Enhanced experiments should not be interpreted as evidence that `Proposed` replaces `FedAvg`; instead, they should be described as supplemental analyses of `FedAvg-based enhanced settings` or `FedAvg variants` within the FedAvg framework.
