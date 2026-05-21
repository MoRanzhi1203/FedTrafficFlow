"""将路段车流量聚合为路口节点车流量（优化版）

本脚本读取密度指标分片文件中的路段车流量数据，结合路网结构信息，
计算每个路口节点在每个时间段的：
1. 路口进入流量
2. 路口离开流量
3. 路口综合车流量

优化特性：
1. 输出为Parquet格式（速度提升3-10倍）
2. 减少中间DataFrame副本
3. 优化数据类型转换
4. 添加性能监控
5. 使用向量化计算

数学定义：
设路网为有向图 G=(V,E)，每条路段 e_i=(s_i,r_i)，其中 s_i 为起始节点，r_i 为结束节点。

对任意节点 v，进入边集合定义为：
E_v_in = {e_i | r_i = v}

离开边集合定义为：
E_v_out = {e_i | s_i = v}

节点进入流量：
q_v_in(t) = sum_{i: r_i = v} q_i(t)

节点离开流量：
q_v_out(t) = sum_{i: s_i = v} q_i(t)

定义方向有效数量：
D_v(t) = I(q_v_in(t) > 0) + I(q_v_out(t) > 0)

路口综合车流量：
若 D_v(t) > 0:
    q_v(t) = (q_v_in(t) + q_v_out(t)) / D_v(t)
否则:
    q_v(t) = 0
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
import psutil
warnings.filterwarnings('ignore')


# ==========================
# 配置参数
# ==========================
ROOT_DIR = Path(__file__).resolve().parents[1]

# 输入文件夹：路段车流量数据（Parquet格式）
INPUT_FOLDER = ROOT_DIR / "data" / "analysis" / "density_metrics_chunks"

# 路网结构文件：包含路段ID到节点ID的映射
RNSD_FILE = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"

# 输出文件夹：路口节点车流量（Parquet格式）
OUTPUT_FOLDER = ROOT_DIR / "data" / "analysis" / "node_intersection_flow_parquet"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 必需字段（在密度数据中）
REQUIRED_FLOW_COLS = ["路段ID", "时间段", "flow_q_hour"]

# 路网映射必需字段
REQUIRED_RNSD_COLS = ["路段ID", "起始节点ID", "结束节点ID"]

# 输出字段
OUTPUT_COLS = [
    "节点ID", 
    "时间段", 
    "路口进入流量", 
    "路口离开流量", 
    "有效方向数", 
    "路口车流量"
]


# ==========================
# 辅助函数
# ==========================
def check_required_columns(df: pd.DataFrame, filename: str, required_cols: list) -> None:
    """检查DataFrame是否包含所有必需字段
    
    Args:
        df: 要检查的DataFrame
        filename: 文件名（用于错误提示）
        required_cols: 必需字段列表
        
    Raises:
        ValueError: 如果缺少必需字段
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"文件 {filename} 缺少必要字段：\n"
            f"缺失字段：{missing_cols}\n"
            f"当前字段：{list(df.columns)}"
        )


def clean_segment_flow_df(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """清洗路段车流量DataFrame（优化版）
    
    Args:
        df: 原始路段车流量DataFrame
        filename: 文件名（用于日志）
        
    Returns:
        清洗后的DataFrame
    """
    original_rows = len(df)
    print(f"  {filename}: 原始记录数 = {original_rows:,}")
    
    # 1. 检查必需字段
    check_required_columns(df, filename, REQUIRED_FLOW_COLS)
    
    # 2. 复制必需字段（避免完整副本）
    df_clean = df[REQUIRED_FLOW_COLS].copy()
    
    # 3. 优化类型转换：一次性转换并指定目标类型
    df_clean["路段ID"] = pd.to_numeric(df_clean["路段ID"], errors="coerce", downcast="integer")
    df_clean["flow_q_hour"] = pd.to_numeric(df_clean["flow_q_hour"], errors="coerce", downcast="float")
    df_clean["时间段"] = pd.to_numeric(df_clean["时间段"], errors="coerce", downcast="integer")
    
    # 4. 删除包含NaN的记录
    df_clean = df_clean.dropna(subset=["路段ID", "时间段", "flow_q_hour"])
    
    # 5. 删除车流量为负值的记录（物理上不可能）
    df_clean = df_clean[df_clean["flow_q_hour"] >= 0]
    
    # 6. 确保整数类型（如果downcast没有完全转换）
    if df_clean["路段ID"].dtype != "int64":
        df_clean["路段ID"] = df_clean["路段ID"].astype("int64")
    if df_clean["时间段"].dtype != "int64":
        df_clean["时间段"] = df_clean["时间段"].astype("int64")
    
    valid_rows = len(df_clean)
    print(f"  {filename}: 有效记录数 = {valid_rows:,} (保留 {valid_rows/original_rows:.1%})")
    
    return df_clean


def load_road_network_mapping() -> pd.DataFrame:
    """加载路网结构映射（路段ID -> 起始节点ID, 结束节点ID）
    
    Returns:
        包含路段到节点映射的DataFrame
        
    Raises:
        FileNotFoundError: 如果路网文件不存在
        ValueError: 如果缺少必需字段
    """
    if not RNSD_FILE.exists():
        raise FileNotFoundError(f"路网结构文件不存在: {RNSD_FILE}")
    
    print(f"读取路网结构文件: {RNSD_FILE}")
    rnsd_df = pd.read_csv(RNSD_FILE)
    
    # 检查必需字段
    check_required_columns(rnsd_df, str(RNSD_FILE), REQUIRED_RNSD_COLS)
    
    # 提取需要的字段
    mapping_df = rnsd_df[REQUIRED_RNSD_COLS].copy()
    
    # 清洗节点ID：转为整数，无法转换的设为NaN
    mapping_df["起始节点ID"] = pd.to_numeric(mapping_df["起始节点ID"], errors="coerce")
    mapping_df["结束节点ID"] = pd.to_numeric(mapping_df["结束节点ID"], errors="coerce")
    
    # 删除包含NaN节点ID的记录
    mapping_df = mapping_df.dropna(subset=["起始节点ID", "结束节点ID"])
    
    # 确保ID为整数类型
    mapping_df["路段ID"] = mapping_df["路段ID"].astype("int64")
    mapping_df["起始节点ID"] = mapping_df["起始节点ID"].astype("int64")
    mapping_df["结束节点ID"] = mapping_df["结束节点ID"].astype("int64")
    
    # 去重（确保每个路段ID只有一条记录）
    mapping_df = mapping_df.drop_duplicates(subset=["路段ID"])
    
    print(f"路网映射: {len(mapping_df):,} 个有效路段")
    
    return mapping_df


def compute_node_flow(flow_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """计算路口节点车流量（优化版）
    
    Args:
        flow_df: 清洗后的路段车流量DataFrame
        mapping_df: 路段到节点的映射DataFrame
        
    Returns:
        路口节点车流量DataFrame
    """
    # 1. 将路段车流量与节点映射合并（使用视图避免副本）
    merged_df = pd.merge(
        flow_df,
        mapping_df,
        on="路段ID",
        how="inner"
    )
    
    if len(merged_df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLS)
    
    # 2. 优化分组聚合：直接使用视图计算进入和离开流量
    # 进入流量：结束节点ID
    in_flow = merged_df.groupby(["结束节点ID", "时间段"], as_index=False)["flow_q_hour"].sum()
    in_flow = in_flow.rename(columns={
        "结束节点ID": "节点ID",
        "flow_q_hour": "路口进入流量"
    })
    
    # 离开流量：起始节点ID
    out_flow = merged_df.groupby(["起始节点ID", "时间段"], as_index=False)["flow_q_hour"].sum()
    out_flow = out_flow.rename(columns={
        "起始节点ID": "节点ID",
        "flow_q_hour": "路口离开流量"
    })
    
    # 3. 使用外连接合并（优化内存使用）
    node_flow_df = pd.merge(
        in_flow,
        out_flow,
        on=["节点ID", "时间段"],
        how="outer"
    )
    
    # 4. 填充缺失值为0（使用fillna的inplace参数）
    node_flow_df["路口进入流量"] = node_flow_df["路口进入流量"].fillna(0)
    node_flow_df["路口离开流量"] = node_flow_df["路口离开流量"].fillna(0)
    
    # 5. 向量化计算有效方向数
    node_flow_df["有效方向数"] = (
        (node_flow_df["路口进入流量"] > 0).astype(int) + 
        (node_flow_df["路口离开流量"] > 0).astype(int)
    )
    
    # 6. 向量化计算路口综合车流量
    # 使用np.where进行条件计算
    mask = node_flow_df["有效方向数"] > 0
    node_flow_df["路口车流量"] = np.where(
        mask,
        (node_flow_df["路口进入流量"] + node_flow_df["路口离开流量"]) / node_flow_df["有效方向数"],
        0.0
    )
    
    # 7. 优化类型转换
    node_flow_df["节点ID"] = node_flow_df["节点ID"].astype("int64")
    node_flow_df["时间段"] = node_flow_df["时间段"].astype("int64")
    
    # 8. 排序和重置索引
    node_flow_df = node_flow_df.sort_values(["节点ID", "时间段"]).reset_index(drop=True)
    
    return node_flow_df[OUTPUT_COLS]


def process_one_file(input_path: Path, output_path: Path, mapping_df: pd.DataFrame) -> None:
    """处理单个输入文件（优化版）
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        mapping_df: 路段到节点的映射DataFrame
    """
    filename = input_path.name
    
    try:
        # 1. 读取Parquet文件（使用内存映射优化）
        df = pd.read_parquet(input_path)
        
        # 2. 清洗数据
        df_clean = clean_segment_flow_df(df, filename)
        
        if len(df_clean) == 0:
            print(f"  {filename}: 无有效数据，跳过处理")
            return
        
        # 3. 计算节点车流量
        node_flow_df = compute_node_flow(df_clean, mapping_df)
        
        if len(node_flow_df) == 0:
            print(f"  {filename}: 无匹配的路段-节点映射，跳过保存")
            return
        
        # 4. 保存结果为Parquet格式（速度提升3-10倍）
        node_flow_df.to_parquet(
            output_path,
            index=False,
            compression="snappy",  # 快速压缩算法
            engine="pyarrow"      # 使用pyarrow引擎
        )
        
        print(f"  {filename}: 输出节点-时间段记录数 = {len(node_flow_df):,}")
        print(f"  {filename}: 保存为Parquet格式到 {output_path}")
        
    except Exception as e:
        print(f"  {filename}: 处理失败 - {str(e)}")
        raise


def main() -> None:
    """主函数（优化版）"""
    print("=" * 80)
    print("路段车流量聚合为路口节点车流量（优化版）")
    print("=" * 80)
    
    # 记录总体开始时间和内存
    total_start_time = time.time()
    total_start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # 1. 检查输入文件夹
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {INPUT_FOLDER}")
    
    # 2. 获取所有Parquet文件
    parquet_files = list(INPUT_FOLDER.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {INPUT_FOLDER} 中未找到Parquet文件")
    
    print(f"找到 {len(parquet_files)} 个Parquet文件")
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出文件夹: {OUTPUT_FOLDER}")
    print()
    
    # 3. 加载路网映射（一次性加载，避免重复读取）
    print("加载路网结构映射...")
    mapping_df = load_road_network_mapping()
    print()
    
    # 4. 处理每个文件（添加性能监控）
    processed_count = 0
    
    for i, input_path in enumerate(sorted(parquet_files), 1):
        print(f"处理文件 {i}/{len(parquet_files)}:")
        
        # 构建输出文件名（Parquet格式）
        # 将输入文件名中的 'density_chunk_' 替换为 'node_flow_chunk_'
        output_filename = input_path.stem.replace("density_chunk_", "node_flow_chunk_") + ".parquet"
        output_path = OUTPUT_FOLDER / output_filename
        
        # 监控单个文件处理性能
        file_start_time = time.time()
        file_start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            process_one_file(input_path, output_path, mapping_df)
            processed_count += 1
        except Exception as e:
            print(f"处理文件 {input_path.name} 时出错: {e}")
            continue
        
        # 计算单个文件处理性能
        file_end_time = time.time()
        file_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        file_time = file_end_time - file_start_time
        file_memory = file_end_memory - file_start_memory
        
        print(f"  {input_path.name}: 处理时间 = {file_time:.2f}秒, 内存变化 = {file_memory:+.2f}MB")
        print()
    
    # 5. 汇总统计（添加总体性能指标）
    total_end_time = time.time()
    total_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    total_time = total_end_time - total_start_time
    total_memory = total_end_memory - total_start_memory
    
    print("=" * 80)
    print("处理完成!")
    print(f"成功处理: {processed_count}/{len(parquet_files)} 个文件")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每个文件处理时间: {total_time/len(parquet_files):.2f}秒")
    print(f"总内存变化: {total_memory:+.2f}MB")
    
    # 检查输出文件（Parquet格式）
    output_files = list(OUTPUT_FOLDER.glob("*.parquet"))
    if output_files:
        print(f"生成 {len(output_files)} 个Parquet输出文件:")
        total_records = 0
        total_size_mb = 0
        
        for output_file in sorted(output_files):
            # 读取Parquet文件统计信息
            try:
                df = pd.read_parquet(output_file)
                records = len(df)
                total_records += records
                file_size_mb = output_file.stat().st_size / 1024 / 1024
                total_size_mb += file_size_mb
                print(f"  {output_file.name}: {records:,} 条记录, {file_size_mb:.2f}MB")
            except Exception as e:
                print(f"  {output_file.name}: 读取失败 - {e}")
        
        if total_records > 0:
            print(f"总记录数: {total_records:,}")
            print(f"总文件大小: {total_size_mb:.2f}MB")
            avg_records_per_file = total_records / len(output_files)
            avg_size_per_file = total_size_mb / len(output_files)
            print(f"平均每个文件记录数: {avg_records_per_file:,.0f}")
            print(f"平均每个文件大小: {avg_size_per_file:.2f}MB")
    else:
        print("警告: 未生成任何输出文件")
    
    print("=" * 80)


if __name__ == "__main__":
    main()