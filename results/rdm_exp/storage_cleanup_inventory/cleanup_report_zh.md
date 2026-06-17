# 前三个机制补全数据集清理报告

- stage: `validate`
- scenarios: `g_mcar_pt,ntb_mix,nso_mix`
- cleanup_started_at: `2026-06-17T14:30:55`
- cleanup_finished_at: `2026-06-17T14:30:57`
- free_space_before: `{"C": 128.457, "D": 419.803, "E": 275.095}`
- free_space_after: `{"C": 128.457, "D": 419.803, "E": 275.095}`
- deleted_size_gb: `236.936642`
- deleted_path_count: `3`

## 处理对象

- `g_mcar_pt`: mechanism=`mcar_point`, mask_file_count=`244`, miss_data_file_count=`244`, imputed_data_file_count_before_cleanup=`1464`
- `ntb_mix`: mechanism=`node_temporal_block`, mask_file_count=`244`, miss_data_file_count=`244`, imputed_data_file_count_before_cleanup=`1464`
- `nso_mix`: mechanism=`node_subset_temporal_outage`, mask_file_count=`244`, miss_data_file_count=`244`, imputed_data_file_count_before_cleanup=`1464`

## 待删或已删目录

- `E:\Jupter_Notebook\FedTrafficFlow\results\rdm_exp\scenarios\g_mcar_pt\imp\imp_data`: exists_before_cleanup=`True`, total_size_gb=`78.981182`, delete_mode=`deleted_reconstructed_from_manifests`
- `E:\Jupter_Notebook\FedTrafficFlow\results\rdm_exp\scenarios\ntb_mix\imp\imp_data`: exists_before_cleanup=`True`, total_size_gb=`78.97495`, delete_mode=`deleted_reconstructed_from_manifests`
- `E:\Jupter_Notebook\FedTrafficFlow\results\rdm_exp\scenarios\nso_mix\imp\imp_data`: exists_before_cleanup=`True`, total_size_gb=`78.98051`, delete_mode=`deleted_reconstructed_from_manifests`

## 验证结果

- missing_datasets_preserved: `True`
- masks_preserved: `True`
- imputed_datasets_deleted: `True`
- summaries_preserved: `True`
- audits_preserved: `True`
- figures_preserved: `True`
- comparison_preserved: `True`
- no_parquet_staged: `True`
- all_complete: `True`
