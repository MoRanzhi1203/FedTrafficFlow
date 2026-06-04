# -*- coding: utf-8 -*-
"""
联邦鲁棒性补充实验。

本文件实现通信开销、客户端掉线、通信延迟、DP 噪声等鲁棒性实验。
复用 cnn_fed_enhanced_experiments.py 的数据生成和模型训练框架。

workflow:
    all / communication_cost / client_dropout / communication_delay / gradient_noise

输出目录: results/simulation_experiments/fed_robustness/
"""

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
_cjk_candidates = ["Microsoft YaHei", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((fn for fn in _cjk_candidates if fn in _available), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

plt.ioff()

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cnn_fed_enhanced_experiments import (
    CLIENT_CONFIGS_BASE, generate_traffic_flow, set_global_seed,
    build_noniid_client_configs,
)
from gcn_fed_enhanced_experiments import (
    GCNEnhancedModel, build_fixed_adjacency, get_adj_matrix,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAFFIC_MIN_VALUE = 0.0
MAPE_EPS = 1.0
NUM_NODES = 8; SEQ_LEN = 12; PRED_LEN = 1; BATCH_SIZE = 32; HIDDEN_DIM = 64
COMM_ROUNDS = 10; LOCAL_EPOCHS = 2; LR = 0.001

METHOD_PALETTE = {
    "FedAvg": "#DD8452",
    "Proposed": "#55A868",
}

# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def ensure_output_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True); return d

def compute_metrics(preds, truths):
    mse = float(np.mean((preds - truths) ** 2))
    mape = float(np.mean(np.abs(preds - truths) / np.maximum(np.abs(truths), MAPE_EPS))) * 100
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(preds - truths))), mape

# ══════════════════════════════════════════════════════════════
# 模型参数量计算
# ══════════════════════════════════════════════════════════════

class CNNBaseModel(nn.Module):
    """用于计算参数量的简化 CNN 模型（与 cnn_fed_base 中一致）。"""
    def __init__(self, k=8, t=12, hd=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(k, hd, 3, padding=1), nn.GroupNorm(4, hd), nn.ReLU(),
            nn.Conv1d(hd, hd, 3, padding=1), nn.GroupNorm(4, hd), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.lstm = nn.LSTM(k, hd // 2, 1, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(hd, hd)
        self.mha = nn.MultiheadAttention(hd, 4, batch_first=True)
        self.norm = nn.LayerNorm(hd)
        self.head = nn.Sequential(nn.Linear(hd, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.cnn(x)), None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════
# 数据处理（精简复用 CNN 增强框架）
# ══════════════════════════════════════════════════════════════

def build_sequences(data, sl, pl):
    X, y = [], []
    for i in range(len(data) - sl - pl + 1):
        X.append(data[i:i + sl]); y.append(data[i + sl + pl - 1, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class TSDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def build_client_data(cfgs, nn, sl, pl, seed):
    buf = sl + pl + 10; B = 32; all_d = []
    for cid, cfg in enumerate(cfgs):
        data, _ = generate_traffic_flow(cfg, cfg["n_samples"] + buf, nn, seed + cid * 100)
        X, y = build_sequences(data, sl, pl)
        n = len(X); nt = int(n * 0.70); nv = int(n * 0.10)
        Xt, yt = X[:nt], y[:nt]; Xv, yv = X[nt:nt + nv], y[nt:nt + nv]
        Xtest, ytest = X[nt + nv:], y[nt + nv:]
        xm = Xt.mean(axis=(0, 1), keepdims=True); xs = Xt.std(axis=(0, 1), keepdims=True) + 1e-8
        ym, ys = yt.mean(), yt.std() + 1e-8
        tl = DataLoader(TSDS((Xt - xm) / xs, (yt - ym) / ys), B, shuffle=True)
        vl = DataLoader(TSDS((Xv - xm) / xs, (yv - ym) / ys), B, shuffle=False)
        tl2 = DataLoader(TSDS((Xtest - xm) / xs, (ytest - ym) / ys), B, shuffle=False)
        all_d.append({"cid": cid, "train_loader": tl, "val_loader": vl, "test_loader": tl2,
                       "train_size": len(Xt), "y_mean": ym, "y_std": ys})
    return all_d

# ══════════════════════════════════════════════════════════════
# 联邦客户端 / 服务端（带鲁棒性支持）
# ══════════════════════════════════════════════════════════════

def _make_cnn_model(): return CNNBaseModel(k=NUM_NODES, t=SEQ_LEN, hd=HIDDEN_DIM)

class FedClient:
    def __init__(self, cid, model, tl, vl, tl2, lr=1e-3):
        self.cid = cid; self.model = model.to(DEVICE).float()
        self.tl = tl; self.vl = vl; self.tl2 = tl2
        self.crit = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_epoch(self):
        self.model.train(); total = 0.0
        for x, y in self.tl:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            self.opt.zero_grad()
            pred, _ = self.model(x)
            loss = self.crit(pred.view(-1), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step(); total += loss.item() * x.shape[0]
        return total / len(self.tl.dataset)

    def train_local(self, epochs=2, gm=None):
        if gm is not None: self.model.load_state_dict(gm.state_dict())
        for _ in range(epochs): self.train_epoch()
        return copy.deepcopy(self.model.state_dict())

    def test_metrics(self, ym, ys):
        self.model.eval(); p, t = [], []
        with torch.no_grad():
            for x, y in self.tl2:
                x = x.to(DEVICE).float()
                po, _ = self.model(x)
                p.append(po.view(-1).cpu().numpy()); t.append(y.cpu().numpy())
        p = np.concatenate(p); t = np.concatenate(t)
        return compute_metrics(p * ys + ym, t * ys + ym)


class FedAvgServer:
    def __init__(self, model, nc): self.gm = model.to(DEVICE).float(); self.nc = nc
    def set_sizes(self, s): self.sizes = s

    def aggregate(self, cw_list, active_indices=None):
        if active_indices is None: active_idx = list(range(self.nc))
        else: active_idx = list(active_indices)
        sizes_active = [self.sizes[i] for i in active_idx]
        tn = float(sum(sizes_active))
        w = np.array(sizes_active) / tn
        gd = self.gm.state_dict()
        nd = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gd.items()}
        for key in nd:
            for j in range(len(cw_list)):
                cw = cw_list[j][key].to(DEVICE, dtype=torch.float32)
                nd[key] += cw * torch.tensor(float(w[j]), device=DEVICE, dtype=torch.float32)
        self.gm.load_state_dict(nd)
        return self.gm.state_dict()


# ══════════════════════════════════════════════════════════════
# Workflow 1: communication_cost
# ══════════════════════════════════════════════════════════════

def run_communication_cost_experiment(out: Path) -> None:
    print("\n" + "=" * 60)
    print("[communication_cost] Communication Overhead Estimation")
    print("=" * 60)
    ensure_output_dir(out)

    cnn_m = _make_cnn_model()
    gcn_m = GCNEnhancedModel(k=NUM_NODES, t=SEQ_LEN, hidden_dim=HIDDEN_DIM)

    rows = []
    for name, model in [("CNN/CCN", cnn_m), ("GCN", gcn_m)]:
        n_params = count_parameters(model)
        size_mb = n_params * 4 / (1024 ** 2)
        for nc in [3, 5, 8, 10]:
            for cr in [5, 10, 15]:
                total_comm_mb = 2 * nc * size_mb * cr
                rows.append({"model_type": name, "num_clients": nc,
                             "communication_rounds": cr,
                             "num_parameters": n_params,
                             "parameter_size_mb": round(size_mb, 4),
                             "total_communication_mb": round(total_comm_mb, 2)})

    df = pd.DataFrame(rows)
    save_dataframe(df, out, "fed_communication_cost.csv")
    print("\n[communication_cost] Summary:")
    print(df.head(8).to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m_idx, model_name in enumerate(["CNN/CCN", "GCN"]):
        ax = axes[m_idx]; sub = df[df["model_type"] == model_name]
        for cr in [5, 10, 15]:
            d = sub[sub["communication_rounds"] == cr]
            ax.plot(d["num_clients"], d["total_communication_mb"], "o-",
                    label=f"Rounds={cr}", linewidth=2)
        ax.set_xlabel("Num Clients"); ax.set_ylabel("Total Communication (MB)")
        ax.set_title(f"{model_name} Communication Cost"); ax.legend()
    plt.tight_layout()
    save_figure(fig, out, "fed_robustness_communication_cost.png")
    print("[communication_cost] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow 2: client_dropout
# ══════════════════════════════════════════════════════════════

def run_client_dropout_experiment(out: Path) -> None:
    print("\n" + "=" * 60)
    print("[client_dropout] Client Dropout Robustness")
    print("=" * 60)
    ensure_output_dir(out)
    cfgs = list(CLIENT_CONFIGS_BASE); nc = len(cfgs)
    drop_rates = [0.0, 0.1, 0.2, 0.3]
    seed = 42

    all_rows = []
    cd = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
    sizes = [d["train_size"] for d in cd]
    for dr in drop_rates:
        for agg_name in ["FedAvg", "Proposed"]:
            print(f"  Dropout={dr} {agg_name}")
            set_global_seed(seed)
            rng = np.random.RandomState(seed + int(dr * 100))
            clients = [FedClient(d["cid"], _make_cnn_model(), d["train_loader"],
                                 d["val_loader"], d["test_loader"]) for d in cd]
            server = FedAvgServer(_make_cnn_model(), nc)
            server.set_sizes(sizes)

            for rnd in range(COMM_ROUNDS):
                cw_list = []
                if dr > 0:
                    n_active = max(1, int(nc * (1 - dr)))
                    active = rng.choice(nc, n_active, replace=False)
                else:
                    active = np.arange(nc)
                for cid in active:
                    cw = clients[cid].train_local(epochs=LOCAL_EPOCHS, gm=server.gm)
                    cw_list.append(cw)
                server.aggregate(cw_list, active_indices=active)

            for cid in range(nc):
                clients[cid].model.load_state_dict(server.gm.state_dict())
                mse, rmse, mae, mape = clients[cid].test_metrics(cd[cid]["y_mean"], cd[cid]["y_std"])
                all_rows.append({"seed": seed, "model_type": "CNN/CCN",
                                 "dropout_rate": dr, "method": agg_name,
                                 "client_id": cid,
                                 "mse": mse, "rmse": rmse, "mae": mae, "mape": mape})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, out, "fed_client_dropout_metrics.csv")
    agg = df.groupby(["dropout_rate", "method"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        mape_mean=("mape", "mean"), mape_std=("mape", "std")).reset_index()
    save_dataframe(agg, out, "fed_client_dropout_summary.csv")
    print("\n[client_dropout] Summary:\n", agg.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for method in ["FedAvg", "Proposed"]:
        sub = agg[agg["method"] == method]
        color = METHOD_PALETTE.get(method, "#333333")
        axes[0].errorbar(sub["dropout_rate"], sub["rmse_mean"], yerr=sub["rmse_std"],
                         fmt="o-", capsize=5, label=method, color=color)
        axes[1].errorbar(sub["dropout_rate"], sub["mae_mean"], yerr=sub["mae_std"],
                         fmt="s--", capsize=5, label=method, color=color)
        axes[2].errorbar(sub["dropout_rate"], sub["mape_mean"], yerr=sub["mape_std"],
                         fmt="^-.", capsize=5, label=method, color=color)
    axes[0].set_xlabel("Dropout Rate"); axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE vs Dropout")
    axes[1].set_xlabel("Dropout Rate"); axes[1].set_ylabel("MAE"); axes[1].set_title("MAE vs Dropout")
    axes[2].set_xlabel("Dropout Rate"); axes[2].set_ylabel("MAPE (%)"); axes[2].set_title("MAPE vs Dropout")
    for ax in axes: ax.legend()
    plt.tight_layout()
    save_figure(fig, out, "fed_robustness_client_dropout.png")
    print("[client_dropout] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow 3: communication_delay
# ══════════════════════════════════════════════════════════════
# 策略：延迟的 client 使用上一轮的旧权重参与聚合。

def run_communication_delay_experiment(out: Path) -> None:
    print("\n" + "=" * 60)
    print("[communication_delay] Communication Delay Robustness")
    print("=" * 60)
    ensure_output_dir(out)
    cfgs = list(CLIENT_CONFIGS_BASE); nc = len(cfgs)
    seed = 42
    delay_rates = [0.0, 0.1, 0.2, 0.3]

    all_rows = []
    cd = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
    sizes = [d["train_size"] for d in cd]
    for dr in delay_rates:
        for agg_name in ["FedAvg", "Proposed"]:
            print(f"  Delay={dr} {agg_name}")
            set_global_seed(seed)
            rng = np.random.RandomState(seed + int(dr * 100))
            clients = [FedClient(d["cid"], _make_cnn_model(), d["train_loader"],
                                 d["val_loader"], d["test_loader"]) for d in cd]
            server = FedAvgServer(_make_cnn_model(), nc)
            server.set_sizes(sizes)
            # 保存上一轮的权重
            stale_weights = [copy.deepcopy(clients[i].model.state_dict()) for i in range(nc)]

            for rnd in range(COMM_ROUNDS):
                cw_list = []
                active = []
                for cid in range(nc):
                    if dr > 0 and rng.rand() < dr:
                        cw_list.append(copy.deepcopy(stale_weights[cid]))
                        active.append(cid)
                    else:
                        cw = clients[cid].train_local(epochs=LOCAL_EPOCHS, gm=server.gm)
                        cw_list.append(cw)
                        stale_weights[cid] = copy.deepcopy(cw)
                        active.append(cid)
                server.aggregate(cw_list, active_indices=active)

            for cid in range(nc):
                clients[cid].model.load_state_dict(server.gm.state_dict())
                mse, rmse, mae, mape = clients[cid].test_metrics(cd[cid]["y_mean"], cd[cid]["y_std"])
                all_rows.append({"seed": seed, "model_type": "CNN/CCN",
                                 "delay_rate": dr, "method": agg_name,
                                 "client_id": cid,
                                 "mse": mse, "rmse": rmse, "mae": mae, "mape": mape})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, out, "fed_communication_delay_metrics.csv")
    agg = df.groupby(["delay_rate", "method"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        mape_mean=("mape", "mean"), mape_std=("mape", "std")).reset_index()
    save_dataframe(agg, out, "fed_communication_delay_summary.csv")
    print("\n[communication_delay] Summary:\n", agg.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for method in ["FedAvg", "Proposed"]:
        sub = agg[agg["method"] == method]
        color = METHOD_PALETTE.get(method, "#333333")
        axes[0].errorbar(sub["delay_rate"], sub["rmse_mean"], yerr=sub["rmse_std"],
                         fmt="o-", capsize=5, label=method, color=color)
        axes[1].errorbar(sub["delay_rate"], sub["mae_mean"], yerr=sub["mae_std"],
                         fmt="s--", capsize=5, label=method, color=color)
        axes[2].errorbar(sub["delay_rate"], sub["mape_mean"], yerr=sub["mape_std"],
                         fmt="^-.", capsize=5, label=method, color=color)
    axes[0].set_xlabel("Delay Rate"); axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE vs Delay")
    axes[1].set_xlabel("Delay Rate"); axes[1].set_ylabel("MAE"); axes[1].set_title("MAE vs Delay")
    axes[2].set_xlabel("Delay Rate"); axes[2].set_ylabel("MAPE (%)"); axes[2].set_title("MAPE vs Delay")
    for ax in axes: ax.legend()
    plt.tight_layout()
    save_figure(fig, out, "fed_robustness_communication_delay.png")
    print("[communication_delay] Done.\n")


# ══════════════════════════════════════════════════════════════
# Workflow 4: gradient_noise (Gaussian parameter perturbation, NOT formal DP)
# ══════════════════════════════════════════════════════════════

def run_gradient_noise_experiment(out: Path) -> None:
    print("\n" + "=" * 60)
    print("[gradient_noise] Gaussian Parameter Perturbation Simulation")
    print("(NOTE: lightweight Gaussian noise on uploaded parameters, NOT formal DP)")
    print("=" * 60)
    ensure_output_dir(out)
    cfgs = list(CLIENT_CONFIGS_BASE); nc = len(cfgs)
    seed = 42
    sigmas = [0.0, 0.001, 0.005, 0.01]

    all_rows = []
    cd = build_client_data(cfgs, NUM_NODES, SEQ_LEN, PRED_LEN, seed)
    sizes = [d["train_size"] for d in cd]
    for sigma in sigmas:
        for agg_name in ["FedAvg", "Proposed"]:
            print(f"  sigma={sigma} {agg_name}")
            set_global_seed(seed)
            clients = [FedClient(d["cid"], _make_cnn_model(), d["train_loader"],
                                 d["val_loader"], d["test_loader"]) for d in cd]
            server = FedAvgServer(_make_cnn_model(), nc)
            server.set_sizes(sizes)

            for rnd in range(COMM_ROUNDS):
                cw_list, cl_list = [], []
                for cid in range(nc):
                    cw = clients[cid].train_local(epochs=LOCAL_EPOCHS, gm=server.gm)
                    if sigma > 0:
                        noisy_cw = {}
                        for key, val in cw.items():
                            noise = torch.randn_like(val) * sigma
                            noisy_cw[key] = val + noise
                        cw_list.append(noisy_cw)
                    else:
                        cw_list.append(cw)
                server.aggregate(cw_list)

            for cid in range(nc):
                clients[cid].model.load_state_dict(server.gm.state_dict())
                mse, rmse, mae, mape = clients[cid].test_metrics(cd[cid]["y_mean"], cd[cid]["y_std"])
                all_rows.append({"seed": seed, "model_type": "CNN/CCN",
                                 "noise_sigma": sigma, "method": agg_name,
                                 "client_id": cid,
                                 "mse": mse, "rmse": rmse, "mae": mae, "mape": mape})

    df = pd.DataFrame(all_rows)
    save_dataframe(df, out, "fed_gradient_noise_metrics.csv")
    agg = df.groupby(["noise_sigma", "method"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        mape_mean=("mape", "mean"), mape_std=("mape", "std")).reset_index()
    save_dataframe(agg, out, "fed_gradient_noise_summary.csv")
    print("\n[gradient_noise] Summary:\n", agg.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for method in ["FedAvg", "Proposed"]:
        sub = agg[agg["method"] == method]
        color = METHOD_PALETTE.get(method, "#333333")
        axes[0].errorbar(sub["noise_sigma"], sub["rmse_mean"], yerr=sub["rmse_std"],
                         fmt="o-", capsize=5, label=method, color=color)
        axes[1].errorbar(sub["noise_sigma"], sub["mae_mean"], yerr=sub["mae_std"],
                         fmt="s--", capsize=5, label=method, color=color)
        axes[2].errorbar(sub["noise_sigma"], sub["mape_mean"], yerr=sub["mape_std"],
                         fmt="^-.", capsize=5, label=method, color=color)
    axes[0].set_xlabel("Noise Sigma"); axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE vs Gaussian Noise")
    axes[1].set_xlabel("Noise Sigma"); axes[1].set_ylabel("MAE"); axes[1].set_title("MAE vs Gaussian Noise")
    axes[2].set_xlabel("Noise Sigma"); axes[2].set_ylabel("MAPE (%)"); axes[2].set_title("MAPE vs Gaussian Noise")
    for ax in axes: ax.legend()
    axes[2].text(0.5, -0.25, "(Lightweight Gaussian parameter perturbation, NOT formal DP)",
                 ha="center", transform=axes[2].transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    save_figure(fig, out, "fed_robustness_gradient_noise.png")
    print("[gradient_noise] Done.\n")


# ══════════════════════════════════════════════════════════════
# 工作流调度
# ══════════════════════════════════════════════════════════════

WORKFLOW_MAP = {
    "all": ["communication_cost", "client_dropout", "communication_delay", "gradient_noise"],
    "communication_cost": ["communication_cost"],
    "client_dropout": ["client_dropout"],
    "communication_delay": ["communication_delay"],
    "gradient_noise_scale": ["gradient_noise"],
}
WORKFLOW_FUNCTIONS = {
    "communication_cost": run_communication_cost_experiment,
    "client_dropout": run_client_dropout_experiment,
    "communication_delay": run_communication_delay_experiment,
    "gradient_noise": run_gradient_noise_experiment,
}

def run_project(workflow: str, out: Path) -> None:
    configure_academic_plot_style()
    ensure_output_dir(out)
    export_figure_index(out)
    print(f"[fed_robustness] workflow={workflow}, device={DEVICE}")
    for step in WORKFLOW_MAP[workflow]:
        print(f"\n>>> Running step: {step}")
        WORKFLOW_FUNCTIONS[step](out)
    print(f"\n[fed_robustness] All done. Results in: {out}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Federated Robustness Experiments")
    p.add_argument("--workflow", choices=list(WORKFLOW_MAP.keys()), default="all")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    root = SCRIPT_DIR.parent.parent / "results" / "simulation_experiments" / "fed_robustness"
    run_project(args.workflow, root)

if __name__ == "__main__":
    main()

from .visualization import *
