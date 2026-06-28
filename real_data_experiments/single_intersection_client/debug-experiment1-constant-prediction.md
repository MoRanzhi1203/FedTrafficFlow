# [OPEN] Debug Session: experiment1-constant-prediction

## Summary

- Scope: `real_data_experiments/single_intersection_client/`
- Symptom: experiment 1 v2 no longer has scale collapse, but `FedAvg` and `Independent` still output method-wise constants in `prediction_samples.csv`.
- Constraint: only debug experiment 1; do not touch experiment 2/3/4, FedAvg formula, model architecture, split policy, LaTeX, simulation, conda, or formal full reruns.

## Falsifiable Hypotheses

1. `prediction_samples.csv` export is wrong and repeats one prediction value, while metrics are computed from non-constant predictions.
2. `pred` / `target` shape mismatch causes silent broadcasting in training or evaluation, so the model effectively optimizes toward a batch/global mean.
3. Model parameters are not being updated as intended in local training, despite the loop running.
4. Evaluation or inference is using the wrong model state, such as an untrained/global-init model or overwritten scaler path.
5. Dataset / loader / normalization path collapses targets or predictions into a near-constant stream before export.

## Plan

1. Verify Git boundary and current v2 artifacts.
2. Audit export path vs metric path to test whether the constant appears only in exported samples.
3. Add experiment-1-only runtime diagnosis tooling and minimal instrumentation.
4. Reproduce on a small smoke run and collect evidence.
5. Apply the smallest fix supported by evidence, then re-verify.

## Evidence Log

- Git boundary is clean except debug artifact and two v2 reports; no forbidden changes under `results/`, `data/`, experiment 2/3/4, LaTeX, simulation, or conda files.
- `tqdm` is available in the current environment (`python -c "import tqdm"` succeeded), so progress display can use `from tqdm.auto import tqdm`.
- `prediction_samples.csv` is produced from `collect_predictions()` and `evaluate_client_model()` by concatenating full-batch outputs, not by repeating `item()` or taking a batch mean during export.
- v2 `prediction_samples.csv` shows method-wise constants (`FedAvg` unique `y_pred` count = 1, `Independent` unique `y_pred` count = 1), and `main_metrics.csv` / `client_metrics.csv` are consistent with poor predictions, so this is not only an export artifact.
- Runtime diagnosis on one client / one batch shows:
  - `x` shape = `(32, 2, 12)`, `y` shape = `(32, 1)`, `pred` shape = `(32, 1)`; no direct shape mismatch evidence.
  - `batch_x` scale is still huge (`mean≈9.4e5`, `std≈9.4e5`) while normalized target `batch_y_norm` is around `[-0.52, 1.54]`.
  - Before training, `batch_pred_denorm_before_train.std≈430`; after 5 optimizer steps, `grad_norm≈4.12~4.30` and `update_norm≈0.00545~0.00548` remain non-zero, but `pred_denorm_std` stays only `≈436`, far below target variability.
- Current best-supported explanation: local training is running and parameters are changing, but the model is effectively collapsing to near-constant outputs under very large raw input scale relative to normalized targets.
- Implemented minimal fix for experiment 1 only:
  - added train-split input normalization for all splits used by experiment 1;
  - kept target normalization and original model / split / FedAvg formula unchanged;
  - added progress logs and `tqdm`-based run visibility.
- Post-fix smoke (`num_clients=2`, `rounds=3`, `local_epochs=2`) evidence:
  - `train_loss` drops from `0.3395` to `0.0490`;
  - `FedAvg` `R2=0.9291`, `Independent` `R2=0.9453`;
  - `prediction_samples.csv` now has `unique y_pred count = 100` for both methods;
  - outputs stay on the correct original scale and no `NaN/Inf` appears.

## Hypothesis Status

1. `prediction_samples.csv` export is wrong and repeats one prediction value, while metrics are computed from non-constant predictions.
   - Rejected by export-path audit and matching constant behavior in exported samples plus poor aggregate metrics.
2. `pred` / `target` shape mismatch causes silent broadcasting in training or evaluation, so the model effectively optimizes toward a batch/global mean.
   - Not supported by current evidence; observed batch shapes are `(32, 1)` vs `(32, 1)`.
3. Model parameters are not being updated as intended in local training, despite the loop running.
   - Rejected by non-zero `grad_norm` and `update_norm` in repeated mini-batch steps.
4. Evaluation or inference is using the wrong model state, such as an untrained/global-init model or overwritten scaler path.
   - Not yet fully ruled out, but current evidence points earlier in the pipeline because even on-training-batch outputs remain near-constant.
5. Dataset / loader / normalization path collapses targets or predictions into a near-constant stream before export.
   - Confirmed in refined form: targets are not collapsed, but raw million-scale inputs versus normalized targets cause the model to collapse toward near-constant outputs. Post-fix smoke supports this explanation.

## Status

- OPEN

