# ==========================
# merge_speed_data.py
# ==========================
# -*- coding: utf-8 -*-
"""
批量分块合并交通速度数据
1. 读取处理后的 link_gps_processed.csv 和 rnsd_processed.csv
2. 合并生成 section 路网
3. 使用 Polars LazyFrame 分块处理交通速度数据
"""
import polars as pl
import os

os.makedirs('./data/processed/speed_data_chunks', exist_ok=True)

# 读取处理好的 CSV
link_gps = pl.read_csv('./data/processed/link_gps_processed.csv')
rnsd = pl.read_csv('./data/processed/rnsd_processed.csv')
section = link_gps.join(rnsd, on='路段ID', how='inner')

chunk_size = 4_334_208
lf = pl.scan_csv('./data/raw/traffic_speed_sub-dataset.v2', has_header=False,
                 new_columns=['路段ID','时间段','平均速度'])
lf = lf.with_columns(pl.col('路段ID').cast(pl.Int64))

total_rows = lf.collect().height
print(f'交通速度数据总行数: {total_rows}')

speed_output_folder = './data/processed/speed_data_chunks'
os.makedirs(speed_output_folder, exist_ok=True)

for start in range(0, total_rows, chunk_size):
    chunk = lf.slice(start, chunk_size).collect()
    merged_chunk = chunk.join(section, on='路段ID', how='left')
    output_file = os.path.join(speed_output_folder, f'speed_chunk_{start//chunk_size:03d}.csv')
    merged_chunk.write_csv(output_file)
    print(f'已保存: {output_file}')

print('全部交通速度数据分块合并完成')
