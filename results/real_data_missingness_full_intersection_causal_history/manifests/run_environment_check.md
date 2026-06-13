# Run Environment Check

**Timestamp**: 2026-06-12
**Python**: E:\\anaconda3\\python.exe
**Working Directory**: E:\\Jupter_Notebook\\FedTrafficFlow

## Input Data

- Input dir: data/analysis/node_intersection_flow_parquet
- Files: 61 node_flow_chunk_*.parquet
- Total rows: 246,133,536
- Target column: 路口车流量 (float64)
- Node column: 节点ID (int64)
- Time column: 时间段 (int64, 0-95)
- Data integrity: zero NaN, zero negative, zero inf, zero duplicates

## Topology

- Topology file: data/processed/rnsd_processed.csv (exists: True)

## Disk Space

- E: drive free: 466.74 GB
- Input data: 3.52 GB
- Estimated full run (5 rates x 5 methods): 105.58 GB -- sufficient: True

## Pipeline

- Script: analysis_scripts/full_intersection_missingness_pipeline.py (1874 lines)
- Supports --causal_history_only, --history_days, --warmup_days
- Disables bfill, bidirectional interpolation, future data in causal mode
- Replaces linear_interpolation with historical_linear_extrapolation

## No speed_chunk Usage

- Confirmed: using only node_flow_chunk_*.parquet, not speed_chunk_*.parquet.