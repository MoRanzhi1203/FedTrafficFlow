"""分析实验 1 客户端/网格异质性：基于 train split 计算 pairwise similarity。"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TENSOR = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt"
DEFAULT_REGIONS = "data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv"
DEFAULT_OUTPUT = "results/real_data_experiments/diagnostics/exp1_client_similarity_diagnosis"
FORMAL_SELECTED = [290, 284, 318, 288, 289]


def compute_grid_features(series: np.ndarray) -> dict:
    """对一个 grid 的目标通道 train 序列计算统计特征。"""
    s = series.astype(np.float64)
    mean_v = float(np.mean(s))
    std_v = float(np.std(s))
    cv_v = std_v / mean_v if mean_v > 0 else 0.0
    return {
        "mean_flow": mean_v, "std_flow": std_v, "min_flow": float(np.min(s)),
        "max_flow": float(np.max(s)), "cv": cv_v,
        "zero_ratio": float(np.mean(s == 0)),
        "p10": float(np.percentile(s, 10)), "p50": float(np.percentile(s, 50)),
        "p90": float(np.percentile(s, 90)),
        "peak_time_index": int(np.argmax(s)),
        "autocorr_lag1": _safe_autocorr(s, 1),
        "autocorr_lag12": _safe_autocorr(s, 12),
        "autocorr_lag24": _safe_autocorr(s, 24) if len(s) > 24 else 0.0,
    }


def _safe_autocorr(s, lag):
    if len(s) <= lag:
        return 0.0
    return float(np.corrcoef(s[:-lag], s[lag:])[0, 1]) if np.std(s) > 1e-8 else 0.0


def compute_all_features(tensor: np.ndarray, train_len: int, channel: int):
    """对所有 active grid 计算 features。tensor shape: [C, N, T]。"""
    n_grids = tensor.shape[1]
    features, curves = [], []
    for g in range(n_grids):
        ts = tensor[channel, g, :train_len]
        features.append(compute_grid_features(ts))
        curves.append(ts)
    return pd.DataFrame(features), np.array(curves)


def correlation_matrix(curves: np.ndarray) -> np.ndarray:
    """N×N Pearson correlation matrix (train 段)。"""
    n = curves.shape[0]
    corr = np.corrcoef(curves)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(corr, -1, 1)


def feature_distance_matrix(features_df: pd.DataFrame) -> np.ndarray:
    """标准化统计特征 Euclidean distance。"""
    cols = ["mean_flow", "std_flow", "cv", "p10", "p50", "p90",
            "peak_time_index", "autocorr_lag1", "autocorr_lag12"]
    data = features_df[cols].values.astype(np.float64)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    n = data.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        diff = data - data[i]
        dist[i] = np.sqrt((diff ** 2).sum(axis=1))
    return dist / (dist.max() + 1e-8)


def combined_similarity(corr_mat: np.ndarray, feat_dist: np.ndarray) -> np.ndarray:
    """综合相似度: score = corr - 0.25 * feat_dist。"""
    return corr_mat - 0.25 * feat_dist


def select_most_similar(candidates: list[int], sim_mat: np.ndarray, id_to_idx: dict, k: int = 5) -> list[int]:
    """从候选 indices 中选择内部平均相似度最高的 k 个。"""
    cand_indices = [id_to_idx[c] for c in candidates]
    best_group, best_score = None, -1e9
    for seed_idx in cand_indices:
        neighbors = sorted(cand_indices, key=lambda j: -sim_mat[seed_idx, j])
        group_idx = neighbors[:k]
        score = sim_mat[np.ix_(group_idx, group_idx)].mean()
        if score > best_score:
            best_score = score
            best_group = [list(id_to_idx.keys())[list(id_to_idx.values()).index(i)] for i in group_idx]
    return best_group


def select_least_similar(candidates: list[int], sim_mat: np.ndarray, id_to_idx: dict, k: int = 5) -> list[int]:
    """从候选 indices 中选择内部平均相似度最低的 k 个。"""
    cand_indices = set(id_to_idx[c] for c in candidates)
    best_group, best_score = None, 1e9
    for seed_idx in sorted(cand_indices):
        group = {seed_idx}
        remaining = cand_indices - {seed_idx}
        while len(group) < k and remaining:
            worst = min(remaining, key=lambda j: sim_mat[j][list(group)].mean())
            group.add(worst)
            remaining.discard(worst)
        if len(group) == k:
            glist = list(group)
            score = sim_mat[np.ix_(glist, glist)].mean()
            if score < best_score:
                best_score = score
                best_group = [list(id_to_idx.keys())[list(id_to_idx.values()).index(i)] for i in glist]
    return best_group


def group_stats(curves, corr_mat, feat_dist, sim_mat, group_ids, active_ids):
    indices = [active_ids.index(g) for g in group_ids]
    return {
        "mean_pairwise_corr": float(corr_mat[np.ix_(indices, indices)].mean()),
        "mean_pairwise_feat_dist": float(feat_dist[np.ix_(indices, indices)].mean()),
        "mean_pairwise_sim": float(sim_mat[np.ix_(indices, indices)].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-path", default=DEFAULT_TENSOR)
    parser.add_argument("--regions-path", default=DEFAULT_REGIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--target-channel", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    tensor = torch.load(args.tensor_path, map_location="cpu", weights_only=False)
    regions = pd.read_csv(args.regions_path)
    active = regions[regions["is_active_region"] == True].copy()
    active_ids = active["region_id"].tolist()
    n_total = tensor.shape[1]
    train_len = int(tensor.shape[2] * args.train_ratio)

    print(f"Tensor: {list(tensor.shape)}, active grids: {len(active_ids)}, train_len: {train_len}")

    # Slice active grids only
    active_tensor = tensor[:, active_ids, :].numpy()

    # Features
    feat_df, curves = compute_all_features(active_tensor, train_len, args.target_channel)
    feat_df.insert(0, "region_id", active_ids)
    corr = correlation_matrix(curves)
    fdist = feature_distance_matrix(feat_df)
    sim = combined_similarity(corr, fdist)

    print(f"Features: {feat_df.shape}, corr range: [{corr.min():.3f}, {corr.max():.3f}]")

    # Candidate groups
    id_to_idx = {rid: i for i, rid in enumerate(active_ids)}
    formal_current = [g for g in FORMAL_SELECTED if g in active_ids]
    most_sim = select_most_similar(active_ids, sim, id_to_idx, args.group_size)
    least_sim = select_least_similar(active_ids, sim, id_to_idx, args.group_size)

    print(f"formal_current: {formal_current}")
    print(f"most_similar: {most_sim}")
    print(f"least_similar: {least_sim}")

    # Save artifacts
    feat_df.to_csv(out / "client_feature_table.csv", index=False)

    def _save_corr(name, ids):
        indices = [active_ids.index(g) for g in ids if g in active_ids]
        df = pd.DataFrame(corr[np.ix_(indices, indices)],
                          index=ids, columns=ids)
        df.to_csv(out / f"{name}_similarity_matrix.csv")

    _save_corr("formal_current", formal_current)
    _save_corr("most_similar_5", most_sim)
    _save_corr("least_similar_5", least_sim)

    # Candidate groups JSON
    groups = {
        "formal_current": formal_current,
        "most_similar_5": most_sim,
        "least_similar_5": least_sim,
        "selection_basis": "train_split_only",
        "train_ratio": args.train_ratio,
        "similarity_formula": "corr - 0.25 * normalized_feature_euclidean_distance",
    }
    json.dump(groups, open(out / "candidate_groups.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # Group summary
    summary_rows = []
    for name, ids in [("formal_current", formal_current), ("most_similar_5", most_sim), ("least_similar_5", least_sim)]:
        stats = group_stats(curves, corr, fdist, sim, ids, active_ids)
        g_feat = feat_df[feat_df["region_id"].isin(ids)]
        summary_rows.append({
            "group_name": name,
            "selected_ids": ",".join(map(str, ids)),
            "mean_pairwise_corr": stats["mean_pairwise_corr"],
            "mean_pairwise_feat_dist": stats["mean_pairwise_feat_dist"],
            "mean_pairwise_sim": stats["mean_pairwise_sim"],
            "mean_flow_mean": float(g_feat["mean_flow"].mean()),
            "std_flow_mean": float(g_feat["std_flow"].mean()),
            "cv_mean": float(g_feat["cv"].mean()),
            "naive_train_rmse_mean": "N/A",
        })
    pd.DataFrame(summary_rows).to_csv(out / "candidate_groups_summary.csv", index=False)

    # Markdown diagnosis draft
    md = f"""# 实验 1 客户端/网格异质性诊断报告

## 诊断目的
当前实验 1 FedAvg RMSE=20753 弱于 Independent=14883 和 NaiveLastValue=19419，需判断是否由客户端异质性/网格选择造成。

## 数据
- Tensor: `{args.tensor_path}` shape={list(tensor.shape)}
- Regions: `{args.regions_path}`
- Active grids: {len(active_ids)}
- Train split: first {train_len}/{tensor.shape[2]} ({args.train_ratio*100:.0f}%)
- Target channel: {args.target_channel}

## 三组 clients

| 组别 | selected_ids | 选择依据 |
|---|---|---|
| formal_current | {formal_current} | 原 formal |
| most_similar_5 | {most_sim} | train split 内部综合相似度最高 |
| least_similar_5 | {least_sim} | train split 内部综合相似度最低 |

## 组内相似度

| 组别 | mean_pairwise_corr | mean_pairwise_sim |
|---|---|---|
"""
    for name, ids in [("formal_current", formal_current), ("most_similar_5", most_sim), ("least_similar_5", least_sim)]:
        stats = group_stats(curves, corr, fdist, sim, ids, active_ids)
        md += f"| {name} | {stats['mean_pairwise_corr']:.4f} | {stats['mean_pairwise_sim']:.4f} |\n"

    md += f"""
## 相似度公式
`score = Pearson_correlation - 0.25 × normalized_feature_euclidean_distance`

特征维度: mean_flow, std_flow, cv, p10, p50, p90, peak_time_index, autocorr_lag1, autocorr_lag12
"""
    (out / "exp1_client_similarity_diagnosis_zh.md").write_text(md, encoding="utf-8")
    print(f"\nAll artifacts saved to {out}")


if __name__ == "__main__":
    main()
