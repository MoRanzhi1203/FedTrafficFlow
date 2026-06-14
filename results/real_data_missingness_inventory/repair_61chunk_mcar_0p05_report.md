# Repair Report: 61 Chunk 5% MCAR Historical Causal Main Experiment

## 1. Repair Scope

- Project root: `E:\Jupter_Notebook\FedTrafficFlow`
- Main result directory: `results/real_data_missingness_full_intersection_causal_history`
- Target setting: `5% MCAR`, `seed=42`, `61 chunks`, `historical causal only`
- This round only executes post-processing closure and does not rerun any `--stage impute` command.

## 2. Status Before Closure

- The six imputation methods had already reached chunk-level completion at `61/61`.
- Root-level lightweight manifests had drifted during method-by-method resumable repair and needed read-only rebuild.
- Root-level `run_config.json` still reflected the last single-method impute resume, so inventory could not yet prove the full six-method closure.
- Single-rate plotting still needed formal single-rate figures instead of a missing-rate line-chart presentation.

## 3. Post-Processing Actions

1. Confirmed there was no running `full_intersection_missingness_pipeline.py` impute process.
2. Re-ran `check_full_missingness_completion.py` and confirmed all six methods remained `61/61`.
3. Added `analysis_scripts/rebuild_full_missingness_manifests.py` as a read-only rebuild utility.
4. Rebuilt:
   - `summaries/imputation_quality_detail.csv`
   - `manifests/imputation_runs.csv`
   - `manifests/impute_chunk_status.csv`
   - `manifests/impute_stage_summary.csv`
5. Re-ran `summarize` using existing detail CSV only.
6. Re-ran `validate` for structure and causal-history constraints only.
7. Replaced single-rate formal figures with dedicated single-rate plots.
8. Re-ran `inventory_real_missingness_assets.py`.
9. Updated inventory inference so the main directory can be recognized from rebuilt manifests instead of only the stale last impute `run_config.json`.

## 4. Final Method Completion

- `zero_fill`: `61/61`
- `forward_fill`: `61/61`
- `historical_linear_extrapolation`: `61/61`
- `geo_neighbor_fill`: `61/61`
- `function_curve_fit`: `61/61`
- `geo_func_hybrid`: `61/61`
- All six methods are fully closed at `61/61`.

## 5. Rebuilt Lightweight Manifests

- `results/real_data_missingness_full_intersection_causal_history\summaries\imputation_quality_detail.csv`: rebuilt, `1464` rows
- `results/real_data_missingness_full_intersection_causal_history\manifests\imputation_runs.csv`: rebuilt, `366` rows
- `results/real_data_missingness_full_intersection_causal_history\manifests\impute_chunk_status.csv`: rebuilt, `366` rows
- `results/real_data_missingness_full_intersection_causal_history\manifests\impute_stage_summary.csv`: rebuilt, `6` rows

## 6. Summarize and Validate

- `summarize`: completed
- `validate`: completed
- `causal_history_only = true`
- `context_days_after = 0`
- `uses_future_days = false`
- `uses_same_day_future_slots = false`
- `uses_bfill = false`
- `uses_bidirectional_interpolation = false`

## 7. Formal Single-Rate Figures

- `figures\single_rate_0p05_rmse_by_method.png`
- `figures\single_rate_0p05_rmse_by_method.pdf`
- `figures\single_rate_0p05_delta_rmse_relative_to_forward_fill.png`
- `figures\single_rate_0p05_delta_rmse_relative_to_forward_fill.pdf`
- `figures\single_rate_0p05_flow_group_rmse.png`
- `figures\single_rate_0p05_flow_group_rmse.pdf`

Note:

- In the single-rate setting, the formal figures are now bar-style single-rate comparisons rather than missing-rate line charts.
- Any old missing-rate line chart produced under a single-rate-only context should be treated as a temporary figure and not as a formal paper figure.

## 8. Inventory Conclusion

- `results/real_data_missingness_inventory\missingness_experiment_run_matrix.csv` now marks the main directory as:
  - `method_count = 6`
  - `chunk_count_detected = 61`
  - `has_generate_missing = true`
  - `has_impute = true`
  - `has_summarize = true`
  - `has_validate = true`
  - `current_status = full_61_chunk_6_method_closed`
- The inventory now explicitly supports the statement:
  - `5% MCAR, seed=42, 61 chunks, 6 methods, generate_missing/impute/summarize/validate closed`

## 9. Result Boundary

- The current result can be used as the historical-causal main experiment result for `61 chunks`, `5% MCAR`, `seed=42`.
- The current result is not yet a multi-missing-rate final comparison.
- The current result is not yet a multi-seed mean±std result.
- The current result is not a `node_temporal_block` result.
- The current result is not a FedAvg or Independent prediction result.
