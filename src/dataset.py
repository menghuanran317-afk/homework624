import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "voltage",
    "global_intensity",
    "sub_metering_remainder",

    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",

    "is_weekend",
    "is_holiday",
    "month",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "season",

    "sin_month",
    "cos_month",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
]

TARGET_COL = "global_active_power"


def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3


def add_calendar_features(df):
    df = df.copy()

    if "date" not in df.columns:
        return df

    df["date"] = pd.to_datetime(df["date"])

    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    if "is_holiday" not in df.columns:
        try:
            import holidays

            years = range(
                df["date"].dt.year.min(),
                df["date"].dt.year.max() + 1
            )
            fr_holidays = holidays.France(years=years)
            df["is_holiday"] = df["date"].dt.date.isin(fr_holidays).astype(int)
        except Exception:
            warnings.warn("holidays 未安装，is_holiday 全部设为 0")
            df["is_holiday"] = 0

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["season"] = df["month"].apply(get_season)

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["sin_dayofweek"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["sin_dayofyear"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def create_windows(data, target_index, input_len=90, output_len=90):
    X, y = [], []
    total_len = input_len + output_len

    for i in range(len(data) - total_len + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + total_len, target_index])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataframe(df):
    df = add_calendar_features(df)

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少这些列：{missing_cols}")

    df = df[FEATURE_COLS].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate().bfill().ffill()

    if df.isna().any().any():
        raise ValueError("数据中仍然存在 NaN。")

    return df


def load_data(
    train_path=None,
    test_path=None,
    full_data_path=None,
    input_len=90,
    output_len=90,
    train_ratio=0.8
):
    target_index = FEATURE_COLS.index(TARGET_COL)

    # 推荐用于 365 天预测：完整数据先滑窗，再按窗口时间顺序划分
    if full_data_path is not None and full_data_path != "":
        df = pd.read_csv(full_data_path)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        df = prepare_dataframe(df)

        total_len = input_len + output_len
        num_windows = len(df) - total_len + 1

        if num_windows <= 0:
            raise ValueError(
                f"完整数据长度不足：len={len(df)}, "
                f"input_len={input_len}, output_len={output_len}"
            )

        split_idx = int(num_windows * train_ratio)

        # 用训练窗口覆盖到的时间范围拟合 scaler
        fit_end = min(split_idx + total_len - 1, len(df))
        scaler = StandardScaler()
        scaler.fit(df.iloc[:fit_end])

        data_scaled = scaler.transform(df)

        X, y = create_windows(
            data_scaled,
            target_index,
            input_len,
            output_len
        )

        X_train = X[:split_idx]
        y_train = y[:split_idx]

        X_test = X[split_idx:]
        y_test = y[split_idx:]

        print("使用完整数据滑窗划分")
        print(f"总天数: {len(df)}")
        print(f"总窗口数: {len(X)}")
        print(f"训练窗口数: {len(X_train)}")
        print(f"测试窗口数: {len(X_test)}")

        return X_train, y_train, X_test, y_test, scaler

    # 旧方式：短期 90 天预测可继续使用 train/test
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = prepare_dataframe(train_df)
    test_df = prepare_dataframe(test_df)

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = create_windows(
        train_scaled,
        target_index,
        input_len,
        output_len
    )

    X_test, y_test = create_windows(
        test_scaled,
        target_index,
        input_len,
        output_len
    )

    if len(X_train) == 0:
        raise ValueError("训练集长度不足，无法构造窗口。")

    if len(X_test) == 0:
        raise ValueError("测试集长度不足，无法构造窗口。")

    return X_train, y_train, X_test, y_test, scaler