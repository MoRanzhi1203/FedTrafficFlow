# 缺失率设置范围核查报告

## 1. 核查目的

本报告用于判断当前 5% MCAR 缺失率是在每日 chunk 内设置，还是在完整 61 天全局数据上统一设置。

## 2. 核查对象

- 代码文件：`analysis_scripts/full_intersection_missingness_pipeline.py`
- 代码文件：`analysis_scripts/real_data_missingness_experiment.py`
- 代码文件：`analysis_scripts/check_full_missingness_completion.py`
- 配置文件：`results/real_data_missingness_full_intersection_causal_history/run_config.json`
- 配置文件：`results/real_data_missingness_full_intersection_causal_history/run_commands.txt`
- manifest：`results/real_data_missingness_full_intersection_causal_history/manifests/generate_missing_chunk_status.csv`
- manifest：`results/real_data_missingness_full_intersection_causal_history/manifests/missing_runs.csv`
- manifest：`results/real_data_missingness_full_intersection_causal_history/manifests/chunk_index_summary.csv`
- mask 目录：`results/real_data_missingness_full_intersection_causal_history/masks/rate_0p05__mechanism_mcar_point__seed_42`

## 3. 代码证据

### make_mcar_mask

- 文件：`analysis_scripts/full_intersection_missingness_pipeline.py`
- 代码范围：`L433-L445`
- 判断：在单个 DataFrame 内，用 eligible 行数乘 missing_rate 计算缺失点数量，并在该 DataFrame 内随机抽样。

```text
433: def make_mcar_mask(df: pd.DataFrame, target_col: str, missing_rate: float, seed: int) -> np.ndarray:
434:     values = pd.to_numeric(df[target_col], errors="coerce")
435:     eligible = np.flatnonzero(values.notna().to_numpy())
436:     mask = np.zeros(len(df), dtype=bool)
437:     if len(eligible) == 0 or missing_rate <= 0:
438:         return mask
439:     count = int(round(len(eligible) * float(missing_rate)))
440:     if count <= 0:
441:         return mask
442:     rng = np.random.RandomState(seed)
443:     selected = rng.choice(eligible, size=count, replace=False)
444:     mask[selected] = True
445:     return mask
```

### run_generate_missing

- 文件：`analysis_scripts/full_intersection_missingness_pipeline.py`
- 代码范围：`L963-L986`
- 判断：mask 在 day_index/file_path 循环内部生成，每个 chunk 单独执行一次。

```text
963:     for day_index, file_path in enumerate(files):
964:         df, meta = read_chunk_frame(
965:             file_path=file_path,
966:             day_index=day_index,
967:             target_col=target_col,
968:             time_col=time_col,
969:             node_col=node_col,
970:             period=args.period,
971:             warmup_days=args.warmup_days,
972:             max_rows=args.max_rows,
973:         )
974:         chunk_records.append(meta)
975:         for missing_rate in missing_rates:
976:             for seed in seeds:
977:                 run_seed = stable_seed(seed, file_path.name, mechanism, missing_rate)
978:                 if mechanism == "mcar_point":
979:                     mask = make_mcar_mask(df, target_col, missing_rate, run_seed)
980:                 elif mechanism == "node_temporal_block":
981:                     mask = make_temporal_block_mask(df, target_col, node_col, missing_rate, run_seed, block_lengths)
982:                 else:
983:                     raise ValueError(f"不支持的缺失机制: {mechanism}")
984:                 actual_missing_count = int(mask.sum())
985:                 eligible_count = int(pd.to_numeric(df[target_col], errors="coerce").notna().sum())
986:                 actual_missing_rate = float(actual_missing_count / eligible_count) if eligible_count else 0.0
```

### read_chunk_frame

- 文件：`analysis_scripts/full_intersection_missingness_pipeline.py`
- 代码范围：`L337-L339`
- 判断：global_time_index 被构造出来，但仅作为时间索引字段，不是全局 MCAR 抽样入口。

```text
337:     df["day_index"] = int(day_index)
338:     df["global_time_index"] = df["day_index"] * int(period) + df["time_slot"]
339:     df["source_chunk_name"] = file_path.name
```

### make_mask

- 文件：`analysis_scripts/real_data_missingness_experiment.py`
- 代码范围：`L671-L698`
- 判断：旧实验脚本同样是在单文件 DataFrame 内用 missing_count=round(len(eligible_indices)*missing_rate) 随机抽样。

```text
671: def make_mask(
672:     df: pd.DataFrame,
673:     target_col: Optional[str],
674:     missing_rate: float,
675:     mechanism: str,
676:     seed: int,
677:     time_col: Optional[str],
678:     node_col: Optional[str],
679: ) -> np.ndarray:
680:     if target_col is None:
681:         raise ValueError("target_col 不能为空。")
682:     if mechanism != "mcar_point":
683:         raise ValueError(f"当前仅支持 `mcar_point`，收到: {mechanism}")
684: 
685:     values = df[target_col]
686:     eligible_indices = np.flatnonzero(values.notna().to_numpy())
687:     mask = np.zeros(len(df), dtype=bool)
688:     if len(eligible_indices) == 0 or missing_rate <= 0:
689:         return mask
690: 
691:     missing_count = int(round(len(eligible_indices) * float(missing_rate)))
692:     if missing_count <= 0:
693:         return mask
694: 
695:     rng = np.random.RandomState(seed)
696:     selected = rng.choice(eligible_indices, size=missing_count, replace=False)
697:     mask[selected] = True
698:     return mask
```

### experiment main loop

- 文件：`analysis_scripts/real_data_missingness_experiment.py`
- 代码范围：`L1144-L1186`
- 判断：旧实验脚本按 selected_files 循环，逐文件生成 mask，没有先构建 61 天全局索引统一抽样。

```text
1144:     for file_path in selected_files:
1145:         file_columns = get_parquet_columns(file_path)
1146:         file_target_col, file_time_col, file_node_col = resolve_columns(
1147:             file_columns,
1148:             target_col,
1149:             time_col,
1150:             node_col,
1151:         )
1152:         df = read_input_frame(file_path, file_target_col, file_time_col, file_node_col, max_rows=max_rows)
1153:         if df.empty:
1154:             continue
1155: 
1156:         eligible_count = int(df[file_target_col].notna().sum())
1157:         rel_file_path = get_relative_path(file_path)
1158:         file_name = file_path.name
1159: 
1160:         for mechanism in mechanisms:
1161:             for missing_rate in missing_rates:
1162:                 for seed in seeds:
1163:                     run_seed = stable_seed(seed, file_name, mechanism, missing_rate)
1164:                     mask = make_mask(
1165:                         df=df,
1166:                         target_col=file_target_col,
1167:                         missing_rate=missing_rate,
1168:                         mechanism=mechanism,
1169:                         seed=run_seed,
1170:                         time_col=file_time_col,
1171:                         node_col=file_node_col,
1172:                     )
1173:                     corrupted = apply_mask(df, file_target_col, mask)
1174:                     actual_missing_count = int(mask.sum())
1175:                     actual_missing_rate = float(actual_missing_count / eligible_count) if eligible_count else 0.0
1176:                     requested_missing_count = int(round(eligible_count * missing_rate))
1177:                     file_stub = f"{file_path.stem}_seed{seed}_{mechanism}_{str(missing_rate).replace('.', 'p')}"
1178: 
1179:                     mask_path: Optional[str] = None
1180:                     corrupted_path: Optional[str] = None
1181:                     if args.save_masks:
1182:                         mask_path = get_relative_path(maybe_write_mask(mask, args.output_dir, file_stub))
1183:                     if args.write_corrupted:
1184:                         corrupted_path = get_relative_path(
1185:                             maybe_write_dataframe(corrupted, args.output_dir, "corrupted", f"{file_stub}_corrupted")
1186:                         )
```

### mask/missing chunk checks

- 文件：`analysis_scripts/check_full_missingness_completion.py`
- 代码范围：`L203-L204`
- 判断：完成度检查本身也是按 chunk 文件粒度检查 masks/missing_datasets，而不是检查单个全局 mask。

```text
203:         count_chunk_dir("masks", mask_dir, "node_flow_chunk_*_mask.parquet", "_mask.parquet", expected),
204:         count_chunk_dir("missing_datasets", missing_dir, "node_flow_chunk_*_missing.parquet", "_missing.parquet", expected),
```

## 4. 配置与 manifest 证据

- `run_config.json` 当前保留的 `stage`：`impute`
- `run_commands.txt` 是否包含 `generate_missing` 命令：`true`
- `missing_runs.csv` 行数：`61`
- `missing_runs.csv` 是否一行一个 day chunk：`true`
- `generate_missing_chunk_status.csv` 行数：`61`
- `chunk_index_summary.csv` 行数：`61`
- `missing_runs.csv` 中 `actual_missing_count` 唯一值：`[201749]`
- `missing_runs.csv` 中 `actual_missing_rate` 最小值：`0.050000049567`
- `missing_runs.csv` 中 `actual_missing_rate` 最大值：`0.050000049567`
- `missing_runs.csv` 中 `actual_missing_rate` 均值：`0.050000049567`
- 未发现单个全局总 mask 记录。
- 未发现先构造完整 61 天全局索引再一次性统一抽样的 manifest 或代码路径。

## 5. mask 统计证据

- 每日缺失率最小值：`0.050000049567`
- 每日缺失率最大值：`0.050000049567`
- 每日缺失率均值：`0.050000049567`
- 每日缺失率标准差：`0.000000000000`
- 全局缺失率：`0.050000049567`

## 6. 最终判断

### 结论 A：每日内设置

当前 5% MCAR 缺失是在每个日级 chunk 内分别按节点—时间片位置随机抽取约 5% 观测点，因此属于按日分层 MCAR 点级缺失，而不是严格的完整 61 天全局统一抽样。

当前机制属于 day-stratified MCAR point missingness，即按日分层的 MCAR 点级缺失。

## 7. 对论文表述的影响

- 如果每日内：在每个日级节点流量矩阵中分别随机选择 5% 的节点—时间片观测作为人工缺失点。
- 如果完整全局：不适用。

## 8. 后续建议

- 当前已完成的 5% 结果应标记为 day-stratified MCAR。 如需 global MCAR，应另起实验目录重新生成全局 mask。
