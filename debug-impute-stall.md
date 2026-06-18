# [OPEN] Debug Session: impute-stall

## Context
- Task: continue `snh_mix` imputation with `--stage impute --resume true`
- Symptom: process stays running, but `imp_data` and `progress` counts stop changing
- Scope: `results\rdm_exp\scenarios\snh_mix`

## Current Evidence
- `5%` is complete for all 8 methods: `61/61`
- `10%` is partial for all 8 methods: `17/61`
- `20%` and `30%` do not have method directories yet
- Background command stays alive without new terminal output
- A 60-second observation window shows no new `.parquet` or `.done.json`

## Hypotheses
- H1: the process is blocked on a specific chunk because one method enters an extremely slow branch
- H2: resume scanning or file existence checks are hanging on a corrupted or partially written output
- H3: the process is waiting on I/O for a parquet read/write operation instead of CPU-side computation
- H4: the process is stuck in a high-cost correlation/topology computation for `10%` chunk `017+`
- H5: the background process is alive but not making forward progress because of an unhandled runtime wait state

## Plan
- Add instrumentation only
- Reproduce with targeted logs around chunk/method boundaries
- Confirm the exact stall point from runtime evidence
- Apply a minimal fix only after evidence is collected
