# -*- coding: utf-8 -*-
"""
仿真实验统一断点续跑工具。

提供：
- 原子写文件 (CSV / JSON / checkpoint)
- TaskContext：管理单个 task 的完整生命周期
- checkpoint 保存/加载
- completed.json 管理
- --resume / --skip-completed / --force 行为逻辑
"""

import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch


# ══════════════════════════════════════════════════════════════
# 原子写工具
# ══════════════════════════════════════════════════════════════

def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """将临时文件 atomically rename 到目标路径（跨平台兼容）。"""
    if final_path.exists():
        final_path.unlink()
    try:
        os.replace(tmp_path, final_path)
    except OSError:
        # Windows fallback
        if final_path.exists():
            final_path.unlink()
        os.rename(tmp_path, final_path)


def atomic_write_json(path: Path, data: dict) -> Path:
    """atomically 写入 JSON 文件，避免中断写坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".json.tmp", prefix="atomic_", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    _atomic_replace(Path(tmp_name), path)
    return path


def atomic_write_csv(path: Path, df: pd.DataFrame) -> Path:
    """atomically 写入 CSV 文件，避免中断写坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".csv.tmp", prefix="atomic_", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            df.to_csv(f, index=False, encoding="utf-8")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    _atomic_replace(Path(tmp_name), path)
    return path


def save_checkpoint_atomic(path: Path, checkpoint_data: dict) -> Path:
    """atomically 写入 PyTorch checkpoint 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".pt.tmp", prefix="ckpt_", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            torch.save(checkpoint_data, f)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    _atomic_replace(Path(tmp_name), path)
    return path


def load_checkpoint(path: Path) -> dict:
    """加载 checkpoint 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")


# ══════════════════════════════════════════════════════════════
# TaskContext — 管理单个 task 的完整生命周期
# ══════════════════════════════════════════════════════════════

class TaskContext:
    """
    管理一个实验 task 的目录结构、checkpoint、completed 标记和防覆盖逻辑。

    目录结构：
        <task_dir>/
            run_config.json
            checkpoints/
                latest.pt
            completed.json

    用法：
        ctx = TaskContext(
            task_dir=Path("results/exp1_main/seed_42"),
            task_id="exp1_main/seed_42",
            config={"seed": 42, "workflow": "main"},
            resume=False,
            skip_completed=False,
            force=False,
        )
        ctx.prepare()  # 检查冲突 / 恢复 / 跳过
        # ... 运行训练 ...
        ctx.mark_completed()
    """

    def __init__(
        self,
        task_dir: Path,
        task_id: str,
        config: Optional[dict] = None,
        resume: bool = False,
        skip_completed: bool = False,
        force: bool = False,
    ):
        self.task_dir = Path(task_dir)
        self.task_id = task_id
        self.config = config or {}
        self.resume = resume
        self.skip_completed = skip_completed
        self.force = force

        # 内部路径
        self.checkpoints_dir = self.task_dir / "checkpoints"
        self.latest_checkpoint_path = self.checkpoints_dir / "latest.pt"
        self.completed_path = self.task_dir / "completed.json"
        self.config_path = self.task_dir / "run_config.json"

        # 恢复状态
        self._start_round = 0
        self._loaded_checkpoint = None
        self._was_resumed = False
        self._was_skipped = False

    # ── task 生命周期 ────────────────────────────────────

    def prepare(self) -> "TaskContext":
        """准备 task 运行环境。检查冲突 / 恢复 / 跳过。返回 self 以支持链式调用。"""
        # 1. 如果 --force：允许清空当前 task 目录
        if self.force:
            if self.task_dir.exists():
                import shutil
                shutil.rmtree(self.task_dir)
                print(f"[force] Cleared existing task directory: {self.task_dir}")
            self._ensure_dirs()
            self._save_run_config()
            return self

        # 2. 如果 --skip-completed：检查是否已完成
        if self.skip_completed and self.completed_path.exists():
            self._was_skipped = True
            print(f"[skip] Task already completed: {self.task_id}")
            return self

        # 3. 如果 --resume：尝试恢复
        if self.resume:
            if self.latest_checkpoint_path.exists():
                self._load_and_resume()
                return self
            else:
                raise FileNotFoundError(
                    f"[resume] No checkpoint found at {self.latest_checkpoint_path}. "
                    f"Run without --resume to start fresh, or use --force to overwrite."
                )

        # 4. 默认：检查是否已有部分结果，防止静默覆盖
        if self.task_dir.exists():
            if self.completed_path.exists():
                raise FileExistsError(
                    f"[error] Output exists: {self.task_dir}\n"
                    f"  Task {self.task_id} already completed.\n"
                    f"  Use --resume, --skip-completed, or --force."
                )
            if self.latest_checkpoint_path.exists():
                raise FileExistsError(
                    f"[error] Output exists: {self.task_dir}\n"
                    f"  Incomplete run found (checkpoint exists).\n"
                    f"  Use --resume, --skip-completed, or --force."
                )
            # 目录存在但无 checkpoint 也无 completed — 视为新目录，允许
            print(f"[info] Output directory exists but appears fresh: {self.task_dir}")

        self._ensure_dirs()
        self._save_run_config()
        return self

    def mark_completed(self, extra: Optional[dict] = None) -> Path:
        """标记 task 完成，写入 completed.json。"""
        data = {
            "status": "success",
            "task_id": self.task_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.task_dir),
            **(extra or {}),
        }
        atomic_write_json(self.completed_path, data)
        print(f"[completed] Task {self.task_id} — {self.completed_path}")
        return self.completed_path

    # ── checkpoint 操作 ─────────────────────────────────

    def save_checkpoint(
        self,
        round_idx: int,
        total_rounds: int,
        model_state_dict: dict,
        metrics_history: Optional[list] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        """保存 checkpoint 到 checkpoints/latest.pt（原子写）。"""
        data = {
            "task_id": self.task_id,
            "round": round_idx,
            "total_rounds": total_rounds,
            "model_state_dict": model_state_dict,
            "metrics_history": metrics_history or [],
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
        }
        if extra:
            data.update(extra)
        path = save_checkpoint_atomic(self.latest_checkpoint_path, data)
        print(f"[checkpoint] Round {round_idx}/{total_rounds} saved to {path}")
        return path

    def load_latest_checkpoint(self) -> dict:
        """加载最新 checkpoint。"""
        return load_checkpoint(self.latest_checkpoint_path)

    def has_checkpoint(self) -> bool:
        return self.latest_checkpoint_path.exists()

    def is_completed(self) -> bool:
        return self.completed_path.exists()

    # ── 属性 ────────────────────────────────────────────

    @property
    def start_round(self) -> int:
        return self._start_round

    @property
    def was_resumed(self) -> bool:
        return self._was_resumed

    @property
    def was_skipped(self) -> bool:
        return self._was_skipped

    @property
    def loaded_checkpoint(self) -> Optional[dict]:
        return self._loaded_checkpoint

    # ── 内部方法 ────────────────────────────────────────

    def _ensure_dirs(self):
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _save_run_config(self):
        data = {
            "task_id": self.task_id,
            "config": self.config,
            "resume": self.resume,
            "skip_completed": self.skip_completed,
            "force": self.force,
            "checkpoint_every": self.config.get("checkpoint_every", 1),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._was_resumed and self._loaded_checkpoint:
            data["resumed_from_checkpoint"] = str(self.latest_checkpoint_path)
            data["resume_start_round"] = self._start_round
        atomic_write_json(self.config_path, data)

    def _load_and_resume(self):
        ckpt = load_checkpoint(self.latest_checkpoint_path)
        self._loaded_checkpoint = ckpt
        self._start_round = ckpt.get("round", 0)
        self._was_resumed = True

        # 恢复随机数状态
        rng = ckpt.get("rng_state", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])

        print(
            f"[resume] Task {self.task_id} resuming from round {self._start_round}"
        )
        # Save updated run_config with resume info
        self._save_run_config()


# ══════════════════════════════════════════════════════════════
# CLI 参数构造辅助
# ══════════════════════════════════════════════════════════════

def add_resume_args(parser) -> None:
    """为 argparse.ArgumentParser 添加统一的 resume/skip/force 参数。"""
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from latest checkpoint if exists.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=False,
        help="Skip tasks that already have a completed.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-run even if output directory exists.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N rounds (default: 1).",
    )


def validate_resume_force_conflict(args) -> None:
    """检查 --resume 与 --force 不能同时使用。"""
    if args.resume and args.force:
        raise ValueError(
            "--resume and --force cannot be used together. "
            "Choose one: resume from checkpoint or force fresh start."
        )
