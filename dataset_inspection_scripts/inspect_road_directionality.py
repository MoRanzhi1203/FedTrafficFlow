"""分析当前路网是否为有向路段，以及是否区分进入路口和离开路口。"""

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RNSD_PATH = ROOT_DIR / "data" / "processed" / "rnsd_processed.csv"


def load_rnsd() -> pd.DataFrame:
    """读取处理后的路网数据，并检查核心字段是否存在。"""
    df = pd.read_csv(RNSD_PATH)

    required_columns = ["路段ID", "方向", "起始节点ID", "结束节点ID"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"文件缺少必要字段：{RNSD_PATH}\n"
            f"缺失字段：{missing_columns}\n"
            f"当前字段：{list(df.columns)}"
        )

    return df


def build_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    """提取分析路段方向关系所需的节点对。"""
    pair_df = (
        df[["路段ID", "方向", "起始节点ID", "结束节点ID"]]
        .dropna(subset=["起始节点ID", "结束节点ID"])
        .copy()
    )
    pair_df["起始节点ID"] = pd.to_numeric(pair_df["起始节点ID"], errors="coerce")
    pair_df["结束节点ID"] = pd.to_numeric(pair_df["结束节点ID"], errors="coerce")
    pair_df = pair_df.dropna(subset=["起始节点ID", "结束节点ID"])
    pair_df["起始节点ID"] = pair_df["起始节点ID"].astype("int64")
    pair_df["结束节点ID"] = pair_df["结束节点ID"].astype("int64")
    return pair_df


def analyze_directionality(pair_df: pd.DataFrame) -> dict:
    """统计有向性、反向配对和入度/出度情况。"""
    forward_pairs = set(
        zip(pair_df["起始节点ID"], pair_df["结束节点ID"])
    )
    reverse_pair_count = sum(
        1 for start_node, end_node in forward_pairs if (end_node, start_node) in forward_pairs
    )

    start_node_set = set(pair_df["起始节点ID"].unique())
    end_node_set = set(pair_df["结束节点ID"].unique())

    both_direction_nodes = start_node_set & end_node_set
    only_out_nodes = start_node_set - end_node_set
    only_in_nodes = end_node_set - start_node_set

    self_loop_count = int((pair_df["起始节点ID"] == pair_df["结束节点ID"]).sum())

    return {
        "total_links": int(pair_df["路段ID"].nunique()),
        "total_pair_rows": int(len(pair_df)),
        "unique_directed_pairs": int(len(forward_pairs)),
        "reverse_pair_count": int(reverse_pair_count),
        "self_loop_count": self_loop_count,
        "start_node_count": int(len(start_node_set)),
        "end_node_count": int(len(end_node_set)),
        "both_direction_nodes": int(len(both_direction_nodes)),
        "only_out_nodes": int(len(only_out_nodes)),
        "only_in_nodes": int(len(only_in_nodes)),
    }


def print_direction_distribution(df: pd.DataFrame) -> None:
    """输出方向字段分布。"""
    print("=" * 100)
    print("方向字段分布")
    print("=" * 100)
    print(df["方向"].value_counts(dropna=False).sort_index())


def print_summary(stats: dict) -> None:
    """输出关于有向路段和路口出入边区分的结论。"""
    print("\n" + "=" * 100)
    print("节点与边方向分析")
    print("=" * 100)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 100)
    print("结论判断")
    print("=" * 100)

    if stats["reverse_pair_count"] == 0:
        print("1. 当前数据更符合“有向路段”存储方式：每条记录都有明确的起始节点和结束节点，且未发现成对反向路段。")
    else:
        print("1. 当前数据包含部分成对反向路段，说明至少有一部分道路以双向分别建边。")

    if stats["both_direction_nodes"] > 0:
        print("2. 数据区分“进入路口”和“离开路口”：大量节点同时具有入边和出边，可以按起始节点/结束节点区分出入方向。")
    else:
        print("2. 数据中没有明显的入边/出边区分特征。")

    if stats["only_out_nodes"] > 0 or stats["only_in_nodes"] > 0:
        print("3. 部分节点只出不进或只进不出，说明网络边界、断头路或截断区域是存在的。")
    else:
        print("3. 所有节点都同时有入边和出边，网络更接近完整闭合结构。")


def main() -> None:
    df = load_rnsd()
    pair_df = build_pair_frame(df)
    stats = analyze_directionality(pair_df)

    print(f"读取文件：{RNSD_PATH}")
    print(f"原始记录数：{len(df)}")
    print(f"可用于方向分析的记录数：{len(pair_df)}")

    print_direction_distribution(df)
    print_summary(stats)


if __name__ == "__main__":
    main()
