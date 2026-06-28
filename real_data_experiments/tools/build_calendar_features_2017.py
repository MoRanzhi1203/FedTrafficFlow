"""生成 2017-04-01 至 2017-05-31 日历特征表（day-level + 15min-level）。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "calendar"

# 61 天, 每天 96 个 15 分钟 time slots
N_DAYS = 61
SLOTS_PER_DAY = 96
TOTAL_SLOTS = N_DAYS * SLOTS_PER_DAY  # 5856

HOLIDAYS = {
    "2017-04-02": ("qingming", "清明节假期"),
    "2017-04-03": ("qingming", "清明节假期"),
    "2017-04-04": ("qingming", "清明节"),
    "2017-04-29": ("labor", "劳动节连休"),
    "2017-04-30": ("labor", "劳动节连休"),
    "2017-05-01": ("labor", "劳动节"),
    "2017-05-28": ("dragon_boat", "端午节假期"),
    "2017-05-29": ("dragon_boat", "端午节假期"),
    "2017-05-30": ("dragon_boat", "端午节"),
}

FESTIVAL_DAYS = {"2017-04-04", "2017-05-01", "2017-05-30"}

ADJUSTED_WORKDAYS = {"2017-04-01", "2017-05-27"}


def build_day_level_calendar() -> pd.DataFrame:
    dates = pd.date_range("2017-04-01", periods=N_DAYS, freq="D")
    rows = []
    for day_idx, dt in enumerate(dates, start=1):
        date_str = dt.strftime("%Y-%m-%d")
        weekday = dt.weekday()  # 0=Mon ... 6=Sun
        is_weekend = 1 if weekday >= 5 else 0
        is_holiday = 1 if date_str in HOLIDAYS else 0
        is_adjusted_workday = 1 if date_str in ADJUSTED_WORKDAYS else 0
        is_festival = 1 if date_str in FESTIVAL_DAYS else 0

        holiday_group = "none"
        holiday_name = ""
        pre_holiday = 0
        post_holiday = 0
        if date_str in HOLIDAYS:
            holiday_group, holiday_name = HOLIDAYS[date_str]
        # pre/post holiday: neighbouring days
        prev_day = (dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        next_day = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prev_day in HOLIDAYS and date_str not in HOLIDAYS:
            post_holiday = 1
        if next_day in HOLIDAYS and date_str not in HOLIDAYS:
            pre_holiday = 1

        # effective workday
        if is_adjusted_workday:
            effective_workday = 1
        elif is_holiday:
            effective_workday = 0
        else:
            effective_workday = 0 if is_weekend else 1

        # days to nearest holiday
        all_holiday_dates = [pd.Timestamp(d) for d in HOLIDAYS]
        distances = [abs((dt - hd).days) for hd in all_holiday_dates]
        days_to_nearest = min(distances) if distances else 99

        rows.append({
            "date": date_str,
            "day_index": day_idx,
            "slot_start": (day_idx - 1) * SLOTS_PER_DAY,
            "slot_end": day_idx * SLOTS_PER_DAY - 1,
            "weekday": weekday,
            "weekday_id": weekday,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "is_adjusted_workday": is_adjusted_workday,
            "is_effective_workday": effective_workday,
            "is_festival_day": is_festival,
            "holiday_name": holiday_name,
            "holiday_group": holiday_group,
            "is_pre_holiday": pre_holiday,
            "is_post_holiday": post_holiday,
            "days_to_nearest_holiday": days_to_nearest,
        })
    return pd.DataFrame(rows)


def expand_to_15min(day_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, day_row in day_df.iterrows():
        for slot in range(SLOTS_PER_DAY):
            time_index = int(day_row["slot_start"] + slot)
            hour = slot // 4
            minute = (slot % 4) * 15
            sin_tod = np.sin(2 * np.pi * slot / SLOTS_PER_DAY)
            cos_tod = np.cos(2 * np.pi * slot / SLOTS_PER_DAY)
            sin_dow = np.sin(2 * np.pi * day_row["weekday_id"] / 7)
            cos_dow = np.cos(2 * np.pi * day_row["weekday_id"] / 7)

            r = dict(day_row)
            r.update({
                "time_index": time_index,
                "slot_of_day": slot,
                "hour": hour,
                "minute": minute,
                "sin_time_of_day": round(sin_tod, 8),
                "cos_time_of_day": round(cos_tod, 8),
                "sin_day_of_week": round(sin_dow, 8),
                "cos_day_of_week": round(cos_dow, 8),
            })
            rows.append(r)
    return pd.DataFrame(rows)


def validate(day_df, min15_df):
    assert len(min15_df) == TOTAL_SLOTS, f"Expected {TOTAL_SLOTS} rows, got {len(min15_df)}"
    assert min15_df["time_index"].min() == 0
    assert min15_df["time_index"].max() == TOTAL_SLOTS - 1
    assert (min15_df["slot_of_day"] == np.tile(np.arange(96), N_DAYS)).all()
    assert (min15_df["day_index"] == np.repeat(np.arange(1, N_DAYS + 1), 96)).all()

    # Check specific dates
    d0401 = day_df[day_df["date"] == "2017-04-01"].iloc[0]
    assert d0401["is_adjusted_workday"] == 1 and d0401["is_effective_workday"] == 1
    for d in ["2017-04-02", "2017-04-03", "2017-04-04"]:
        assert day_df[day_df["date"] == d].iloc[0]["is_holiday"] == 1
    may01 = day_df[day_df["date"] == "2017-05-01"].iloc[0]
    assert may01["is_holiday"] == 1 and may01["is_festival_day"] == 1
    d0527 = day_df[day_df["date"] == "2017-05-27"].iloc[0]
    assert d0527["is_adjusted_workday"] == 1 and d0527["is_effective_workday"] == 1
    for d in ["2017-05-28", "2017-05-29", "2017-05-30"]:
        assert day_df[day_df["date"] == d].iloc[0]["is_holiday"] == 1
    print("All validations passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    day_df = build_day_level_calendar()
    day_df.to_csv(out / "calendar_2017_04_01_to_2017_05_31.csv", index=False)

    min15_df = expand_to_15min(day_df)
    min15_df.to_csv(out / "calendar_features_15min_2017_04_01_to_2017_05_31.csv", index=False)

    validate(day_df, min15_df)
    print(f"Day calendar: {len(day_df)} days -> {out / 'calendar_2017_04_01_to_2017_05_31.csv'}")
    print(f"15min features: {len(min15_df)} rows -> {out / 'calendar_features_15min_2017_04_01_to_2017_05_31.csv'}")


if __name__ == "__main__":
    main()
