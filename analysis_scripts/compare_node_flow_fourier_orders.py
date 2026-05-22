"""比较不同傅里叶阶数对路口节点日内平均车流量曲线的拟合效果。"""

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis_scripts.fit_node_flow_daily_curve import (
    AVG_FLOW_COL,
    DAY_SLOT_COL,
    INPUT_DIR,
    NODE_COL,
    SLOTS_PER_DAY,
    build_fourier_design_matrix,
    load_and_build_daily_profile,
)


DEFAULT_OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "analysis"
    / "node_flow_curve_fit"
    / "node_flow_fourier_order_comparison.json"
)
DEFAULT_HARMONICS_LIST = [4, 6, 8, 10]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="比较不同傅里叶阶数下的节点日内车流量拟合效果。"
    )
    parser.add_argument(
        "--harmonics",
        nargs="+",
        type=int,
        default=DEFAULT_HARMONICS_LIST,
        help="需要比较的傅里叶阶数列表，默认: 4 6 8 10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="比较结果输出 JSON 路径",
    )
    return parser.parse_args()


def iter_complete_node_frames(profile_df) -> Iterable:
    """仅返回拥有完整 96 个日内时间段的节点曲线。"""
    expected_slots = np.arange(SLOTS_PER_DAY, dtype=np.int64)

    for node_df in profile_df.partition_by(NODE_COL, maintain_order=True):
        day_slots = node_df.get_column(DAY_SLOT_COL).to_numpy()
        if len(day_slots) != SLOTS_PER_DAY:
            continue
        if not np.array_equal(day_slots, expected_slots):
            continue
        yield node_df


def evaluate_harmonics(profile_df, harmonics: int) -> dict:
    """评估单个傅里叶阶数在全部节点上的拟合指标。"""
    rmse_list: List[float] = []
    mae_list: List[float] = []
    r2_list: List[float] = []
    wmape_num = 0.0
    wmape_den = 0.0
    sse_total = 0.0
    sst_total = 0.0
    fitted_nodes = 0

    for node_df in iter_complete_node_frames(profile_df):
        day_slots = node_df.get_column(DAY_SLOT_COL).to_numpy()
        observed = node_df.get_column(AVG_FLOW_COL).to_numpy().astype(np.float64)

        design_matrix = build_fourier_design_matrix(day_slots, harmonics)
        coeffs, _, _, _ = np.linalg.lstsq(design_matrix, observed, rcond=None)
        fitted = np.clip(design_matrix @ coeffs, a_min=0.0, a_max=None)
        residual = observed - fitted

        abs_residual = np.abs(residual)
        sq_residual = np.square(residual)
        ss_res = float(np.sum(sq_residual))
        ss_tot = float(np.sum(np.square(observed - observed.mean())))
        if ss_tot == 0.0:
            r2 = 1.0 if ss_res == 0.0 else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        rmse_list.append(float(np.sqrt(np.mean(sq_residual))))
        mae_list.append(float(np.mean(abs_residual)))
        r2_list.append(float(r2))
        wmape_num += float(np.sum(abs_residual))
        wmape_den += float(np.sum(np.abs(observed)))
        sse_total += ss_res
        sst_total += ss_tot
        fitted_nodes += 1

    if not rmse_list:
        raise ValueError("没有任何节点满足完整 96 点曲线条件，无法比较傅里叶阶数")

    rmse_arr = np.array(rmse_list, dtype=np.float64)
    mae_arr = np.array(mae_list, dtype=np.float64)
    r2_arr = np.array(r2_list, dtype=np.float64)

    return {
        "harmonics": harmonics,
        "parameter_count": int(1 + 2 * harmonics),
        "fitted_nodes": fitted_nodes,
        "mean_rmse": float(rmse_arr.mean()),
        "median_rmse": float(np.median(rmse_arr)),
        "p90_rmse": float(np.quantile(rmse_arr, 0.9)),
        "mean_mae": float(mae_arr.mean()),
        "median_mae": float(np.median(mae_arr)),
        "p90_mae": float(np.quantile(mae_arr, 0.9)),
        "mean_r2": float(r2_arr.mean()),
        "median_r2": float(np.median(r2_arr)),
        "p10_r2": float(np.quantile(r2_arr, 0.1)),
        "global_r2": float(1.0 - sse_total / sst_total) if sst_total != 0.0 else 1.0,
        "wmape": float(wmape_num / wmape_den) if wmape_den != 0.0 else 0.0,
    }


def compare_harmonics_orders(harmonics_list: List[int]) -> List[dict]:
    """批量比较多个傅里叶阶数。"""
    cleaned_orders = sorted({order for order in harmonics_list if order > 0})
    if not cleaned_orders:
        raise ValueError("至少需要提供一个正整数傅里叶阶数")

    profile_df, input_file_count = load_and_build_daily_profile(INPUT_DIR)
    comparison = []

    print("=" * 80)
    print("路口节点日内车流量傅里叶阶数比较")
    print("=" * 80)
    print(f"输入文件数量: {input_file_count}")
    print(f"日内平均曲线记录数: {profile_df.height}")

    complete_node_count = sum(1 for _ in iter_complete_node_frames(profile_df))
    print(f"参与比较的完整节点数: {complete_node_count}")

    for harmonics in cleaned_orders:
        result = evaluate_harmonics(profile_df, harmonics)
        comparison.append(result)
        print(
            "阶数 {harmonics}: mean_RMSE={mean_rmse:.4f}, mean_MAE={mean_mae:.4f}, "
            "mean_R2={mean_r2:.4f}, global_R2={global_r2:.4f}, WMAPE={wmape:.6f}".format(
                **result
            )
        )

    print("=" * 80)
    return comparison


def save_comparison_results(results: List[dict], output_path: Path) -> None:
    """保存傅里叶阶数比较结果。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)


def main() -> None:
    """主函数。"""
    args = parse_args()
    results = compare_harmonics_orders(args.harmonics)
    save_comparison_results(results, args.output)
    print(f"比较结果输出: {args.output}")


if __name__ == "__main__":
    main()
