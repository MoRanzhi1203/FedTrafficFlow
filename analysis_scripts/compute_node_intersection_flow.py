"""将路段车流量聚合为路口节点车流量

本脚本读取密度指标分片文件中的路段车流量数据，结合路网结构信息，
计算每个路口节点在每个时间段的：
1. 路口进入流量
2. 路口离开流量
3. 路口综合车流量

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
warnings.filterwarnings('ignore')


# ==========================
# 配置参数
# ==========================
ROOT_DIR = Path(__file__).resolve().parents[1]

# 输入文件夹：路段车流量数据（Parquet格式）
INPUT_FOLDER = ROOT_DIR / "data" / "analysis" / "density_metrics_chunks"

# 路网结构文件：包含路段ID到节点ID的映射
RNSD_FILE = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"

# 输出文件夹：路口节点车流量
OUTPUT_FOLDER = ROOT_DIR / "data" / "analysis" / "node_intersection_flow"
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
    """清洗路段车流量DataFrame
    
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
    
    # 2. 复制数据避免修改原始数据
    df_clean = df[REQUIRED_FLOW_COLS].copy()
    
    # 3. 将路段ID转为整数（无法转换的设为NaN）
    df_clean["路段ID"] = pd.to_numeric(df_clean["路段ID"], errors="coerce")
    
    # 4. 将车流量转为数值型（无法转换的设为NaN）
    df_clean["flow_q_hour"] = pd.to_numeric(df_clean["flow_q_hour"], errors="coerce")
    
    # 5. 删除包含NaN的记录
    df_clean = df_clean.dropna(subset=["路段ID", "时间段", "flow_q_hour"])
    
    # 6. 确保路段ID为整数类型
    df_clean["路段ID"] = df_clean["路段ID"].astype("int64")
    
    # 7. 确保时间段为整数类型
    df_clean["时间段"] = df_clean["时间段"].astype("int64")
    
    # 8. 删除车流量为负值的记录（物理上不可能）
    df_clean = df_clean[df_clean["flow_q_hour"] >= 0]
    
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
    """计算路口节点车流量
    
    Args:
        flow_df: 清洗后的路段车流量DataFrame
        mapping_df: 路段到节点的映射DataFrame
        
    Returns:
        路口节点车流量DataFrame
    """
    # 1. 将路段车流量与节点映射合并
    merged_df = pd.merge(
        flow_df,
        mapping_df,
        on="路段ID",
        how="inner"
    )
    
    if len(merged_df) == 0:
        # 如果没有匹配的记录，返回空DataFrame
        return pd.DataFrame(columns=OUTPUT_COLS)
    
    # 2. 构造离开流量DataFrame（起始节点 -> 离开流量）
    out_df = merged_df[["起始节点ID", "时间段", "flow_q_hour"]].copy()
    out_df = out_df.rename(columns={
        "起始节点ID": "节点ID",
        "flow_q_hour": "路口离开流量"
    })
    
    # 按节点ID和时间段分组求和
    out_df = out_df.groupby(["节点ID", "时间段"], as_index=False)["路口离开流量"].sum()
    
    # 3. 构造进入流量DataFrame（结束节点 -> 进入流量）
    in_df = merged_df[["结束节点ID", "时间段", "flow_q_hour"]].copy()
    in_df = in_df.rename(columns={
        "结束节点ID": "节点ID",
        "flow_q_hour": "路口进入流量"
    })
    
    # 按节点ID和时间段分组求和
    in_df = in_df.groupby(["节点ID", "时间段"], as_index=False)["路口进入流量"].sum()
    
    # 4. 使用外连接合并进入和离开流量
    node_flow_df = pd.merge(
        in_df,
        out_df,
        on=["节点ID", "时间段"],
        how="outer"
    )
    
    # 5. 填充缺失值为0（表示该方向没有流量）
    node_flow_df["路口进入流量"] = node_flow_df["路口进入流量"].fillna(0)
    node_flow_df["路口离开流量"] = node_flow_df["路口离开流量"].fillna(0)
    
    # 6. 计算有效方向数
    # 指示函数：流量大于0为1，否则为0
    has_in = (node_flow_df["路口进入流量"] > 0).astype(int)
    has_out = (node_flow_df["路口离开流量"] > 0).astype(int)
    node_flow_df["有效方向数"] = has_in + has_out
    
    # 7. 计算路口综合车流量
    # 初始化所有值为0
    node_flow_df["路口车流量"] = 0.0
    
    # 对于有效方向数>0的行，计算综合车流量
    mask = node_flow_df["有效方向数"] > 0
    if mask.any():
        node_flow_df.loc[mask, "路口车流量"] = (
            node_flow_df.loc[mask, "路口进入流量"] + 
            node_flow_df.loc[mask, "路口离开流量"]
        ) / node_flow_df.loc[mask, "有效方向数"]
    
    # 8. 确保数值类型正确
    node_flow_df["节点ID"] = node_flow_df["节点ID"].astype("int64")
    node_flow_df["时间段"] = node_flow_df["时间段"].astype("int64")
    
    # 9. 按节点ID和时间段排序
    node_flow_df = node_flow_df.sort_values(["节点ID", "时间段"])
    
    # 10. 重置索引
    node_flow_df = node_flow_df.reset_index(drop=True)
    
    return node_flow_df[OUTPUT_COLS]


def process_one_file(input_path: Path, output_path: Path, mapping_df: pd.DataFrame) -> None:
    """处理单个输入文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        mapping_df: 路段到节点的映射DataFrame
    """
    filename = input_path.name
    
    try:
        # 1. 读取Parquet文件
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
        
        # 4. 保存结果
        node_flow_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        print(f"  {filename}: 输出节点-时间段记录数 = {len(node_flow_df):,}")
        print(f"  {filename}: 保存到 {output_path}")
        
    except Exception as e:
        print(f"  {filename}: 处理失败 - {str(e)}")
        raise


def main() -> None:
    """主函数"""
    print("=" * 80)
    print("路段车流量聚合为路口节点车流量")
    print("=" * 80)
    
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
    
    # 4. 处理每个文件
    processed_count = 0
    for i, input_path in enumerate(sorted(parquet_files), 1):
        print(f"处理文件 {i}/{len(parquet_files)}:")
        
        # 构建输出文件名
        # 将输入文件名中的 'density_chunk_' 替换为 'node_flow_chunk_'
        output_filename = input_path.stem.replace("density_chunk_", "node_flow_chunk_") + ".csv"
        output_path = OUTPUT_FOLDER / output_filename
        
        try:
            process_one_file(input_path, output_path, mapping_df)
            processed_count += 1
        except Exception as e:
            print(f"处理文件 {input_path.name} 时出错: {e}")
            # 可以选择继续处理其他文件或停止
            # 这里选择继续处理
            continue
        
        print()
    
    # 5. 汇总统计
    print("=" * 80)
    print("处理完成!")
    print(f"成功处理: {processed_count}/{len(parquet_files)} 个文件")
    
    # 检查输出文件
    output_files = list(OUTPUT_FOLDER.glob("*.csv"))
    if output_files:
        print(f"生成 {len(output_files)} 个输出文件:")
        for output_file in sorted(output_files):
            # 读取文件行数（不包括标题）
            try:
                df = pd.read_csv(output_file)
                print(f"  {output_file.name}: {len(df):,} 条记录")
            except:
                print(f"  {output_file.name}: 读取失败")
    else:
        print("警告: 未生成任何输出文件")
    
    print("=" * 80)


if __name__ == "__main__":
    main()