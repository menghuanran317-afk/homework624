import zipfile
import gzip
import shutil
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

WEATHER_DATASET_API = (
    "https://www.data.gouv.fr/api/1/datasets/"
    "donnees-climatologiques-de-base-mensuelles/"
)

POWER_FILE = RAW_DIR / "household_power_consumption.txt"


def download_uci_power_data():
    if POWER_FILE.exists():
        print("电力数据已存在，跳过下载")
        return

    print("正在下载 UCI 家庭电力数据...")
    r = requests.get(UCI_URL, timeout=180)
    r.raise_for_status()

    with zipfile.ZipFile(BytesIO(r.content)) as z:
        z.extractall(RAW_DIR)

    print("UCI 数据下载完成")


def load_power_data():
    print("正在读取电力数据...")

    df = pd.read_csv(
        POWER_FILE,
        sep=";",
        na_values="?",
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].interpolate()
    df[numeric_cols] = df[numeric_cols].bfill().ffill()

    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg({
        "Global_active_power": "sum",
        "Global_reactive_power": "sum",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
        "Voltage": "mean",
        "Global_intensity": "mean",
    }).reset_index()

    daily = daily.rename(columns={
        "Global_active_power": "global_active_power",
        "Global_reactive_power": "global_reactive_power",
        "Voltage": "voltage",
        "Global_intensity": "global_intensity",
        "Sub_metering_1": "sub_metering_1",
        "Sub_metering_2": "sub_metering_2",
        "Sub_metering_3": "sub_metering_3",
    })

    daily["date"] = pd.to_datetime(daily["date"])

    daily["sub_metering_remainder"] = (
        daily["global_active_power"] * 1000 / 60
        - (
            daily["sub_metering_1"]
            + daily["sub_metering_2"]
            + daily["sub_metering_3"]
        )
    )

    print("电力数据按天聚合完成")
    print(daily.head())

    return daily


def get_weather_resource_urls():
    print("正在获取 data.gouv.fr 天气数据资源列表...")

    r = requests.get(WEATHER_DATASET_API, timeout=60)
    r.raise_for_status()

    data = r.json()
    urls = []

    for res in data.get("resources", []):
        url = res.get("url", "")
        title = res.get("title", "")
        fmt = res.get("format", "")

        text = f"{url} {title} {fmt}".lower()

        if not url:
            continue

        if "liste-change" in text:
            continue

        if "csv" not in text:
            continue

        if "mens" not in text and "recmens" not in text:
            continue

        urls.append(url)

    print(f"找到候选天气资源数量：{len(urls)}")

    return urls


def read_weather_from_url(url):
    print(f"尝试读取天气资源：{url}")

    tmp_file = RAW_DIR / "weather_tmp"

    try:
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        tmp_file.write_bytes(r.content)

        if r.content[:2] == b"\x1f\x8b" or url.endswith(".gz"):
            with gzip.open(tmp_file, "rb") as f:
                df = pd.read_csv(f, sep=";", low_memory=False)

        elif r.content[:2] == b"PK" or url.endswith(".zip"):
            with zipfile.ZipFile(tmp_file) as z:
                csv_files = [
                    name for name in z.namelist()
                    if name.lower().endswith(".csv")
                ]

                if not csv_files:
                    return None

                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f, sep=";", low_memory=False)

        else:
            try:
                df = pd.read_csv(tmp_file, sep=";", low_memory=False)
            except Exception:
                df = pd.read_csv(tmp_file, sep=None, engine="python")

        df.columns = [c.strip() for c in df.columns]

        return df

    except Exception as e:
        print(f"读取失败：{e}")
        return None

    finally:
        tmp_file.unlink(missing_ok=True)


def load_weather_data():
    required_cols = [
        "RR",
        "NBJRR1",
        "NBJRR5",
        "NBJRR10",
        "NBJBROU"
    ]

    urls = get_weather_resource_urls()

    all_valid = []

    for url in urls:
        df = read_weather_from_url(url)

        if df is None:
            continue

        if "AAAAMM" not in df.columns:
            continue

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            continue

        keep_cols = ["AAAAMM"] + required_cols

        if "NUM_POSTE" in df.columns:
            keep_cols = ["NUM_POSTE"] + keep_cols

        df = df[keep_cols].copy()

        df["month"] = pd.to_datetime(
            df["AAAAMM"].astype(str),
            format="%Y%m",
            errors="coerce"
        )

        df = df.dropna(subset=["month"])

        df = df[
            (df["month"] >= "2006-12-01")
            & (df["month"] <= "2010-11-01")
        ]

        if len(df) == 0:
            continue

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required_cols, how="all")

        if len(df) == 0:
            continue

        all_valid.append(df)

    if not all_valid:
        raise RuntimeError(
            "没有找到包含有效 RR、NBJRR1、NBJRR5、NBJRR10、NBJBROU 数值的天气数据。"
        )

    weather = pd.concat(all_valid, ignore_index=True)

    print("有效天气数据总行数：", len(weather))

    nearby = weather.copy()

    if "NUM_POSTE" in nearby.columns:
        nearby["NUM_POSTE"] = nearby["NUM_POSTE"].astype(str)

        nearby = nearby[
            nearby["NUM_POSTE"].str.startswith(
                ("92", "75", "94", "91", "78")
            )
        ]

    if len(nearby) > 0:
        print("使用巴黎附近气象站数据")
        used_weather = nearby
    else:
        print("巴黎附近气象站数据为空，使用全法国站点月平均")
        used_weather = weather

    weather_monthly = (
        used_weather
        .groupby("month")[required_cols]
        .mean()
        .reset_index()
    )

    weather_monthly[required_cols] = (
        weather_monthly[required_cols]
        .interpolate()
        .bfill()
        .ffill()
    )

    weather_monthly["RR"] = weather_monthly["RR"] / 10.0

    if weather_monthly[required_cols].isna().any().any():
        print(weather_monthly)
        raise RuntimeError("天气变量仍然存在空值，请检查天气数据。")

    weather_monthly.to_csv(
        OUT_DIR / "weather_monthly_used.csv",
        index=False
    )

    print("天气数据整理完成")
    print(weather_monthly.head())
    print(weather_monthly.isna().sum())

    return weather_monthly


def merge_power_weather(power_daily, weather_monthly):
    print("正在合并电力数据和天气数据...")

    power_daily = power_daily.copy()

    power_daily["month"] = (
        power_daily["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df = power_daily.merge(
        weather_monthly,
        on="month",
        how="left"
    )

    weather_cols = [
        "RR",
        "NBJRR1",
        "NBJRR5",
        "NBJRR10",
        "NBJBROU"
    ]

    df[weather_cols] = (
        df[weather_cols]
        .interpolate()
        .bfill()
        .ffill()
    )

    if df[weather_cols].isna().any().any():
        print(df[weather_cols].isna().sum())
        raise RuntimeError("合并后天气变量仍然存在空值。")

    df = df.drop(columns=["month"])

    print("合并完成")
    print(df.head())
    print(df.isna().sum())

    return df


def split_train_test(df):
    print("正在划分 train.csv 和 test.csv...")

    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"train.csv 已保存：{OUT_DIR / 'train.csv'}")
    print(f"test.csv 已保存：{OUT_DIR / 'test.csv'}")
    print(f"训练集天数：{len(train)}")
    print(f"测试集天数：{len(test)}")


def main():
    download_uci_power_data()

    power_daily = load_power_data()
    weather_monthly = load_weather_data()

    final_df = merge_power_weather(
        power_daily,
        weather_monthly
    )

    final_df.to_csv(
        OUT_DIR / "daily_power_weather.csv",
        index=False
    )

    split_train_test(final_df)

    print("全部完成！")


if __name__ == "__main__":
    main()