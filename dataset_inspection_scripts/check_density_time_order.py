"""检查密度指标分片文件中的时间段字段是否按预期顺序排列。"""

from pathlib import Path

import polars as pl


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT_DIR / 'data' / 'analysis' / 'density_metrics_chunks'
TARGET_COLUMN = '时间段'
MAX_ISSUE_PREVIEW = 10


def list_parquet_files(folder_path: Path) -> list[Path]:
    return sorted(folder_path.glob('*.parquet'))


def load_time_sequence(file_path: Path) -> list[int]:
    df = (
        pl.scan_parquet(file_path.as_posix())
        .select(pl.col(TARGET_COLUMN).cast(pl.Int64, strict=False))
        .collect()
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f'文件缺少字段 `{TARGET_COLUMN}`: {file_path}')

    return df.get_column(TARGET_COLUMN).drop_nulls().to_list()


def compress_consecutive_values(values: list[int]) -> list[int]:
    if not values:
        return []

    compressed = [values[0]]
    for value in values[1:]:
        if value != compressed[-1]:
            compressed.append(value)

    return compressed


def build_order_issues(values: list[int]) -> list[str]:
    if not values:
        return ['时间段序列为空']

    issues = []

    for index in range(1, len(values)):
        prev_value = values[index - 1]
        current_value = values[index]

        if current_value < prev_value:
            issues.append(
                f'回退: 位置 {index - 1}->{index}，{prev_value}->{current_value}'
            )
            continue

        if current_value > prev_value + 1:
            issues.append(
                f'跳段: 位置 {index - 1}->{index}，{prev_value}->{current_value}'
            )

    return issues


def format_issue_summary(issues: list[str]) -> str:
    if not issues:
        return '无'

    if len(issues) <= MAX_ISSUE_PREVIEW:
        return '；'.join(issues)

    preview = '；'.join(issues[:MAX_ISSUE_PREVIEW])
    return f'{preview}；其余 {len(issues) - MAX_ISSUE_PREVIEW} 处异常已省略'


def print_file_report(file_path: Path, ordered_time_values: list[int]) -> bool:
    issues = build_order_issues(ordered_time_values)
    is_ok = len(issues) == 0

    print('=' * 120)
    print(f'文件：{file_path.name}')
    print(f'顺序检查：{"正常" if is_ok else "异常"}')
    # print(f'时间段序列：{ordered_time_values}')

    if ordered_time_values:
        print(f'起始时间段：{ordered_time_values[0]}')
        print(f'结束时间段：{ordered_time_values[-1]}')
        print(f'去重后序列长度：{len(ordered_time_values)}')
    else:
        print('起始时间段：无')
        print('结束时间段：无')
        print('去重后序列长度：0')

    print(f'异常说明：{format_issue_summary(issues)}')
    return is_ok


def print_summary(total_count: int, ok_files: list[str], bad_files: list[str]) -> None:
    print('\n' + '=' * 120)
    print('汇总结果')
    print('=' * 120)
    print(f'总文件数：{total_count}')
    print(f'顺序正常文件数：{len(ok_files)}')
    print(f'顺序异常文件数：{len(bad_files)}')
    print(f'顺序正常文件：{ok_files if ok_files else "无"}')
    print(f'顺序异常文件：{bad_files if bad_files else "无"}')


def main() -> None:
    parquet_files = list_parquet_files(TARGET_DIR)

    if not parquet_files:
        print(f'未在目录中找到 parquet 文件：{TARGET_DIR}')
        return

    ok_files = []
    bad_files = []

    for parquet_file in parquet_files:
        time_values = load_time_sequence(parquet_file)
        ordered_time_values = compress_consecutive_values(time_values)
        is_ok = print_file_report(parquet_file, ordered_time_values)

        if is_ok:
            ok_files.append(parquet_file.name)
        else:
            bad_files.append(parquet_file.name)

    print_summary(len(parquet_files), ok_files, bad_files)


if __name__ == '__main__':
    main()
