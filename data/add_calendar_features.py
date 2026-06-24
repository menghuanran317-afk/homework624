import numpy as np
import pandas as pd
from pathlib import Path

try:
    import holidays
except ImportError:
    raise ImportError(
        "请先安装 holidays：pip install holidays"
    )


DATA_DIR = Path("data/processed")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUT_TRAIN_PATH = DATA_DIR / "train_with_calendar.csv"
OUT_TEST_PATH = DATA_DIR / "test_with_calendar.csv"


def get_season(month):
    if month in [12, 1, 2]:
        return 0  # winter
    elif month in [3, 4, 5]:
        return 1  # spring
    elif month in [6, 7, 8]:
        return 2  # summer
    else:
        return 3  # autumn


def add_calendar_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    years = range(
        df["date"].dt.year.min(),
        df["date"].dt.year.max() + 1
    )

    fr_holidays = holidays.France(years=years)

    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_holiday"] = df["date"].dt.date.isin(fr_holidays).astype(int)

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["season"] = df["month"].apply(get_season)

    # 周期编码：比直接 month/day_of_week 更适合神经网络
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["sin_dayofweek"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["sin_dayofyear"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_new = add_calendar_features(train)
    test_new = add_calendar_features(test)

    train_new.to_csv(OUT_TRAIN_PATH, index=False)
    test_new.to_csv(OUT_TEST_PATH, index=False)

    print("完成！已生成：")
    print(OUT_TRAIN_PATH)
    print(OUT_TEST_PATH)

    print("\n新增列：")
    new_cols = [
        col for col in train_new.columns
        if col not in train.columns
    ]
    print(new_cols)

    print("\ntrain 缺失值统计：")
    print(train_new.isna().sum())

    print("\ntest 缺失值统计：")
    print(test_new.isna().sum())


if __name__ == "__main__":
    main()