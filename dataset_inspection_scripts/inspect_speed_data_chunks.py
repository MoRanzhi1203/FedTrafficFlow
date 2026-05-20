"""检查速度数据 Parquet 分片的结构、样例值和基础统计信息。"""

from pathlib import Path
from typing import Optional

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT_DIR / 'data' / 'processed' / 'speed_data_chunks'

HEAD_ROWS = 50
TAIL_ROWS = 20
SAMPLE_VALUES_PER_COLUMN = 10
CALCULATE_N_UNIQUE = True
LOW_CARDINALITY_THRESHOLD = 20
FOLDER_PREVIEW_LIMIT = 10

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def configure_display() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_fmt_str_lengths(100)
    pl.Config.set_tbl_width_chars(240)


def print_title(title: str) -> None:
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)


def list_parquet_files(folder_path: Path) -> list[Path]:
    return sorted(folder_path.glob('*.parquet'))


def build_dtype_df(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        '列名': df.columns,
        '数据类型': [str(dtype) for dtype in df.dtypes],
    })


def build_column_stats_df(df: pl.DataFrame) -> pl.DataFrame:
    column_stats = []

    for col in df.columns:
        null_count = df[col].null_count()
        non_null_count = df.height - null_count
        unique_count = df[col].n_unique() if CALCULATE_N_UNIQUE else None

        column_stats.append({
            '列名': col,
            '数据类型': str(df[col].dtype),
            '空值数': null_count,
            '非空数': non_null_count,
            '空值比例': null_count / df.height if df.height > 0 else 0,
            '唯一值数': unique_count,
        })

    return pl.DataFrame(column_stats)


def build_numeric_stats_df(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    numeric_columns = [col for col in df.columns if df[col].dtype in NUMERIC_DTYPES]

    if not numeric_columns:
        return None

    numeric_stats = []
    for col in numeric_columns:
        numeric_stats.append({
            '列名': col,
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
        })

    return pl.DataFrame(numeric_stats)


def build_sample_df(df: pl.DataFrame) -> pl.DataFrame:
    sample_rows = []

    for col in df.columns:
        sample_values = (
            df.select(pl.col(col))
            .filter(pl.col(col).is_not_null())
            .unique()
            .head(SAMPLE_VALUES_PER_COLUMN)
            .get_column(col)
            .to_list()
        )
        sample_rows.append({
            '列名': col,
            '样例值': sample_values,
        })

    return pl.DataFrame(sample_rows)


def print_low_cardinality_stats(df: pl.DataFrame) -> None:
    print_title(f'低唯一值列的频数统计（唯一值数 <= {LOW_CARDINALITY_THRESHOLD}）')

    if not CALCULATE_N_UNIQUE:
        print('已关闭唯一值统计，跳过低唯一值列频数统计。')
        return

    found_low_cardinality = False

    for col in df.columns:
        unique_count = df[col].n_unique()
        if unique_count <= LOW_CARDINALITY_THRESHOLD:
            found_low_cardinality = True
            print(f'\n字段: {col}')
            print(
                df.group_by(col)
                .agg(pl.len().alias('记录数'))
                .sort('记录数', descending=True)
            )

    if not found_low_cardinality:
        print('没有低唯一值列。')


def inspect_parquet_file(file_path: Path) -> None:
    configure_display()
    df = pl.read_parquet(file_path)

    print_title('基础信息')
    print('文件路径:', file_path)
    print('数据形状:', df.shape)
    print('总行数:', df.height)
    print('总列数:', df.width)

    print_title('字段名')
    for i, col in enumerate(df.columns, start=1):
        print(f'{i:02d}. {col}')

    print_title('数据类型')
    print(build_dtype_df(df))

    print_title('每列空值、非空值、唯一值统计')
    print(build_column_stats_df(df))

    print_title('数值列描述统计')
    numeric_stats_df = build_numeric_stats_df(df)
    if numeric_stats_df is None:
        print('没有数值列。')
    else:
        print(numeric_stats_df)

    print_title(f'每列前 {SAMPLE_VALUES_PER_COLUMN} 个非空唯一样例值')
    print(build_sample_df(df))

    print_low_cardinality_stats(df)

    print_title(f'前 {HEAD_ROWS} 行：全部字段')
    print(df.head(HEAD_ROWS))

    print_title(f'后 {TAIL_ROWS} 行：全部字段')
    print(df.tail(TAIL_ROWS))

    print_title(f'全部字段预览：前 {HEAD_ROWS} 行')
    print(df.select(df.columns).head(HEAD_ROWS))

    print_title('查看完成')
    print('文件读取和字段检查完成。')


def inspect_parquet_directory(folder_path: Path, sample_file_count: int = 1) -> None:
    parquet_files = list_parquet_files(folder_path)

    print_title('文件夹概览')
    print('文件夹路径:', folder_path)
    print('Parquet 文件数量:', len(parquet_files))

    if not parquet_files:
        print('该文件夹下没有找到 parquet 文件。')
        return

    print_title(f'前 {min(FOLDER_PREVIEW_LIMIT, len(parquet_files))} 个文件')
    for index, parquet_file in enumerate(parquet_files[:FOLDER_PREVIEW_LIMIT], start=1):
        print(f'{index:02d}. {parquet_file.name}')

    for index, parquet_file in enumerate(parquet_files[:sample_file_count], start=1):
        print_title(f'示例文件 {index}/{sample_file_count}')
        inspect_parquet_file(parquet_file)


def main() -> None:
    inspect_parquet_directory(TARGET_DIR, sample_file_count=1)


if __name__ == '__main__':
    main()
