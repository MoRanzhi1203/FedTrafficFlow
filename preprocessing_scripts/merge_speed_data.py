# ==========================
# merge_speed_data_fast_memory_safe.py
# ==========================
# -*- coding: utf-8 -*-

import os
import shutil

# 必须放在 import polars 之前
os.environ["POLARS_MAX_THREADS"] = "8"

import polars as pl


# ==========================
# 配置
# ==========================
input_speed_file = './data/raw/traffic_speed_sub-dataset.v2'
link_gps_file = './data/processed/link_gps_processed.csv'
rnsd_file = './data/processed/rnsd_processed.csv'

output_folder = './data/processed/speed_data_chunks'

# 原始设定
chunk_size = 4_334_208

# Parquet 压缩方式
parquet_compression = 'lz4'

# 是否清空旧输出
clear_old_output = True

# 路段ID 是否按数值排序
sort_link_id_as_int = True

# 如果内存还是吃紧，把这个值改小，例如 48、24、12
# 默认会自动计算为 96
manual_time_periods_per_chunk = None


# ==========================
# 清洗函数
# ==========================
def clean_str_col(col_name: str):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
    )


def clean_int_col(col_name: str):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Int64)
    )


def clean_float_col(col_name: str):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .cast(pl.Float64)
    )


# ==========================
# 初始化输出目录
# ==========================
if clear_old_output and os.path.exists(output_folder):
    print(f'正在清空旧输出目录: {os.path.abspath(output_folder)}')
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

print(f'输出目录: {os.path.abspath(output_folder)}')


# ==========================
# 读取并清洗 link_gps、rnsd
# ==========================
print('正在读取 link_gps 数据...')

link_gps = (
    pl.read_csv(link_gps_file)
    .with_columns([
        clean_str_col('路段ID')
    ])
)

print('正在读取 rnsd 数据...')

rnsd = (
    pl.read_csv(rnsd_file)
    .with_columns([
        clean_str_col('路段ID')
    ])
)


# ==========================
# 合并路段静态信息
# ==========================
print('正在合并 link_gps 与 rnsd...')

section = link_gps.join(
    rnsd,
    on='路段ID',
    how='inner'
)

road_count = section.height

print(f'section 行数，也就是路段数: {road_count}')
print(f'section 字段数: {section.width}')

if road_count <= 0:
    raise ValueError('section 为空，请检查 link_gps_processed.csv 和 rnsd_processed.csv 是否能按 路段ID 正确合并。')

section_lf = section.lazy()


# ==========================
# 计算每个输出文件包含多少个时间段
# ==========================
if manual_time_periods_per_chunk is None:
    time_periods_per_chunk = chunk_size // road_count
else:
    time_periods_per_chunk = manual_time_periods_per_chunk

if time_periods_per_chunk <= 0:
    raise ValueError('time_periods_per_chunk <= 0，请减小 road_count 或增大 chunk_size。')

actual_chunk_rows = road_count * time_periods_per_chunk

print(f'每个输出文件包含时间段数: {time_periods_per_chunk}')
print(f'每个完整输出文件预计行数: {actual_chunk_rows}')


# ==========================
# 扫描速度数据
# ==========================
print('正在扫描交通速度数据...')

speed_lf = (
    pl.scan_csv(
        input_speed_file,
        has_header=False,
        new_columns=['路段ID', '时间段', '平均速度']
    )
    .with_columns([
        clean_str_col('路段ID'),
        clean_int_col('时间段'),
        clean_float_col('平均速度')
    ])
)


# ==========================
# 获取时间段范围
# 只收集 min/max，不收集全量数据
# ==========================
print('正在获取时间段范围...')

time_range_df = (
    speed_lf
    .select([
        pl.col('时间段').min().alias('min_time'),
        pl.col('时间段').max().alias('max_time')
    ])
    .collect()
)

min_time = int(time_range_df[0, 'min_time'])
max_time = int(time_range_df[0, 'max_time'])

print(f'时间段范围: {min_time} - {max_time}')


# ==========================
# 按时间段区间分块处理
#
# 核心优化：
# 原来一次只处理 1 个时间段，需要扫描 5856 次。
# 现在一次处理 96 个时间段，只需要大约 61 次。
# ==========================
print('\n开始按时间段区间分块处理...')

chunk_index = 0
total_written_rows = 0

for start_time in range(min_time, max_time + 1, time_periods_per_chunk):
    end_time = min(start_time + time_periods_per_chunk, max_time + 1)

    print(
        f'\n正在处理第 {chunk_index:03d} 块: '
        f'时间段 {start_time} - {end_time - 1}'
    )

    current_lf = (
        speed_lf
        .filter(
            (pl.col('时间段') >= start_time) &
            (pl.col('时间段') < end_time)
        )
        .join(section_lf, on='路段ID', how='left')
    )

    # 最终排序：
    # 先按 时间段 升序
    # 再按 路段ID 升序
    if sort_link_id_as_int:
        current_lf = (
            current_lf
            .with_columns([
                pl.col('路段ID')
                .cast(pl.Int64, strict=False)
                .alias('_路段ID排序')
            ])
            .sort(
                ['时间段', '_路段ID排序'],
                descending=[False, False]
            )
            .drop('_路段ID排序')
        )
    else:
        current_lf = (
            current_lf
            .sort(
                ['时间段', '路段ID'],
                descending=[False, False]
            )
        )

    output_file = os.path.join(
        output_folder,
        f'speed_chunk_{chunk_index:03d}.parquet'
    )

    output_file_abs = os.path.abspath(output_file)

    # 直接从 LazyFrame 写到 Parquet
    # 避免 current_df = current_lf.collect() 后长时间占用内存
    current_lf.sink_parquet(
        output_file_abs,
        compression=parquet_compression,
        statistics=False
    )

    # 为了打印行数，只读取 parquet 元信息不太方便；
    # 这里用理论值估算，最后一个 chunk 可能小一点
    expected_rows = road_count * (end_time - start_time)

    total_written_rows += expected_rows

    print(f'已保存: {output_file_abs}')
    print(f'预计行数: {expected_rows}')

    chunk_index += 1


print('\n全部交通速度数据分块合并完成')
print(f'总输出文件数: {chunk_index}')
print(f'预计总写出行数: {total_written_rows}')
print(f'输出目录: {os.path.abspath(output_folder)}')