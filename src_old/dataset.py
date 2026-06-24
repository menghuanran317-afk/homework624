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


def load_data(
    train_path,
    test_path,
    input_len=90,
    output_len=90
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    missing_train = [c for c in FEATURE_COLS if c not in train_df.columns]
    missing_test = [c for c in FEATURE_COLS if c not in test_df.columns]

    if missing_train:
        raise ValueError(f"train.csv 缺少这些列：{missing_train}")

    if missing_test:
        raise ValueError(f"test.csv 缺少这些列：{missing_test}")

    train_df = train_df[FEATURE_COLS].copy()
    test_df = test_df[FEATURE_COLS].copy()

    train_df = train_df.apply(pd.to_numeric, errors="coerce")
    test_df = test_df.apply(pd.to_numeric, errors="coerce")

    train_df = train_df.interpolate().bfill().ffill()
    test_df = test_df.interpolate().bfill().ffill()

    if train_df.isna().any().any():
        raise ValueError("train 数据中仍然存在 NaN，请检查数据。")

    if test_df.isna().any().any():
        raise ValueError("test 数据中仍然存在 NaN，请检查数据。")

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    target_index = FEATURE_COLS.index(TARGET_COL)

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
        raise ValueError(
            f"训练集长度不足，无法构造窗口：input_len={input_len}, output_len={output_len}"
        )

    if len(X_test) == 0:
        raise ValueError(
            f"测试集长度不足，无法构造窗口：input_len={input_len}, output_len={output_len}"
        )

    return X_train, y_train, X_test, y_test, scaler