"""日-周周期感知 baseline 和 client-specific seasonal profile。

提供：
- DailySeasonalNaive: y_hat[t] = y[t - 96]
- WeeklySeasonalNaive: y_hat[t] = y[t - 672] (7*96)
- CalendarProfileNaive: client-specific weekday/weekend slot profile
- build_client_seasonal_profile: 计算 client-specific season profile
"""
import numpy as np
import pandas as pd
from typing import Optional


def daily_seasonal_naive(
    y_train: np.ndarray,
    y_test_raw: np.ndarray,
    test_time_indices: np.ndarray,
) -> np.ndarray:
    """DailySeasonalNaive: predict y[t] = y[t - 96] (yesterday same slot)."""
    preds = np.full(len(y_test_raw), np.nan)
    full = np.concatenate([y_train, y_test_raw])
    for i, t in enumerate(test_time_indices):
        ref = t - 96
        if ref >= 0:
            preds[i] = full[ref]
        else:
            preds[i] = full[t - 1] if t > 0 else float(np.mean(y_train))
    return preds


def weekly_seasonal_naive(
    y_train: np.ndarray,
    y_test_raw: np.ndarray,
    test_time_indices: np.ndarray,
) -> np.ndarray:
    """WeeklySeasonalNaive: predict y[t] = y[t - 672] (last week same day+slot)."""
    preds = np.full(len(y_test_raw), np.nan)
    full = np.concatenate([y_train, y_test_raw])
    for i, t in enumerate(test_time_indices):
        ref = t - 672
        if ref >= 0:
            preds[i] = full[ref]
        else:
            # Fallback to daily seasonal
            ref_daily = t - 96
            if ref_daily >= 0:
                preds[i] = full[ref_daily]
            else:
                preds[i] = full[t - 1] if t > 0 else float(np.mean(y_train))
    return preds


def build_client_seasonal_profile(
    y_train: np.ndarray,
    calendar_features_train: pd.DataFrame,
    profile_type: str = "weekday_weekend",
) -> dict:
    """构建 client-specific seasonal profile。

    返回 dict 形式，用于 CalendarProfileNaive 预测。
    """
    slots = calendar_features_train["slot_of_day"].values.astype(int)
    if profile_type == "weekday_weekend" and "is_effective_workday" in calendar_features_train.columns:
        day_type = calendar_features_train["is_effective_workday"].values.astype(int)
        profile = {}
        for dt in [0, 1]:
            profile[dt] = {}
            for s in range(96):
                mask = (day_type == dt) & (slots == s)
                if mask.sum() > 0:
                    profile[dt][s] = float(np.mean(y_train[mask]))
                else:
                    profile[dt][s] = None
        # Fill missing with daily average
        daily_profile = {}
        for s in range(96):
            mask = slots == s
            daily_profile[s] = float(np.mean(y_train[mask])) if mask.sum() > 0 else float(np.mean(y_train))
        for dt in [0, 1]:
            for s in range(96):
                if profile[dt][s] is None:
                    profile[dt][s] = daily_profile[s]
        return {"type": "weekday_weekend", "profile": profile, "daily_fallback": daily_profile,
                "train_mean": float(np.mean(y_train))}
    # Fallback: daily slot profile
    daily = {}
    for s in range(96):
        mask = slots == s
        daily[s] = float(np.mean(y_train[mask])) if mask.sum() > 0 else float(np.mean(y_train))
    return {"type": "daily", "profile": daily, "train_mean": float(np.mean(y_train))}


def calendar_profile_naive_predict(
    y_train: np.ndarray,
    y_test_raw: np.ndarray,
    test_time_indices: np.ndarray,
    calendar_features_test: pd.DataFrame,
    profile: dict,
) -> np.ndarray:
    """CalendarProfileNaive: 使用 client-specific profile 预测。"""
    if profile["type"] == "weekday_weekend":
        workday_flag = calendar_features_test["is_effective_workday"].values.astype(int)
        slot_of_day = calendar_features_test["slot_of_day"].values.astype(int)
        prof = profile["profile"]
        preds = np.array([prof[workday_flag[i]][slot_of_day[i]] for i in range(len(slot_of_day))])
    else:
        slot_of_day = calendar_features_test["slot_of_day"].values.astype(int)
        prof = profile["daily_fallback"] if "daily_fallback" in profile else profile["profile"]
        preds = np.array([prof.get(s, profile["train_mean"]) for s in slot_of_day])
    return preds


def daily_seasonal_naive_from_full_sequence(
    full_seq: np.ndarray,
    test_time_indices: np.ndarray,
) -> np.ndarray:
    """DailySeasonalNaive: predict y[t] = full_seq[t - 96] for each test target time.

    Uses the complete time series (including train/val/test) for lag-based lookback.
    No future data leakage: only t-96 is accessed (which may be in train, val, or earlier test).
    """
    preds = np.empty(len(test_time_indices), dtype=np.float64)
    train_mean = float(np.mean(full_seq))
    for i, t in enumerate(test_time_indices):
        ref = int(t) - 96
        if 0 <= ref < len(full_seq):
            preds[i] = full_seq[ref]
        elif t - 1 >= 0:
            preds[i] = full_seq[int(t) - 1] if int(t) - 1 < len(full_seq) else train_mean
        else:
            preds[i] = train_mean
    return preds


def weekly_seasonal_naive_from_full_sequence(
    full_seq: np.ndarray,
    test_time_indices: np.ndarray,
) -> np.ndarray:
    """WeeklySeasonalNaive: predict y[t] = full_seq[t - 672] for each test target time.

    Fallback chain: t-672 -> t-96 -> t-1 -> train_mean.
    Uses the complete time series for lag-based lookback.
    """
    preds = np.empty(len(test_time_indices), dtype=np.float64)
    train_mean = float(np.mean(full_seq))
    for i, t in enumerate(test_time_indices):
        ti = int(t)
        ref_weekly = ti - 672
        ref_daily = ti - 96
        ref_last = ti - 1
        if 0 <= ref_weekly < len(full_seq):
            preds[i] = full_seq[ref_weekly]
        elif 0 <= ref_daily < len(full_seq):
            preds[i] = full_seq[ref_daily]
        elif 0 <= ref_last < len(full_seq):
            preds[i] = full_seq[ref_last]
        else:
            preds[i] = train_mean
    return preds
