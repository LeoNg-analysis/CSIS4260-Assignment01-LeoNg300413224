
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "part2_analysis"

df = pd.read_csv(DATA_DIR / "all_stocks_5yr.csv")

#Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

#Sort by company and date
df = df.sort_values(["name", "date"]).reset_index(drop=True)

#Calculate SMA 20 and SMA 50 for each company
df["SMA_20"] = df.groupby("name")["close"].transform(
    lambda x: x.rolling(window=20, min_periods=20).mean()
)

df["SMA_50"] = df.groupby("name")["close"].transform(
    lambda x: x.rolling(window=50, min_periods=50).mean()
)

#Calculate daily price changes
df["price_change"] = df.groupby("name")["close"].diff()

#Separate gains and losses
df["gain"] = df["price_change"].clip(lower=0)
df["loss"] = -df["price_change"].clip(upper=0)

#Calculate rolling average gains and losses
df["avg_gain"] = df.groupby("name")["gain"].transform(
    lambda x: x.rolling(window=14, min_periods=14).mean()
)

df["avg_loss"] = df.groupby("name")["loss"].transform(
    lambda x: x.rolling(window=14, min_periods=14).mean()
)

#Calculate RSI
rs = df["avg_gain"] / df["avg_loss"].replace(0, np.nan)
df["RSI_14"] = 100 - (100 / (1 + rs))

#Handle zero-loss periods
df.loc[
    (df["avg_loss"] == 0) & (df["avg_gain"] > 0),
    "RSI_14"
] = 100

df.loc[
    (df["avg_loss"] == 0) & (df["avg_gain"] == 0),
    "RSI_14"
] = 50


df = df.drop(
    columns=["price_change", "gain", "loss", "avg_gain", "avg_loss"]
)


output_path = OUTPUT_DIR / "stocks_with_indicators.parquet"
df.to_parquet(
    output_path,
    engine="pyarrow",
    compression="zstd",
    index=False
)

print("\n=== TECHNICAL INDICATORS ===")
print(
    df[df["name"] == "AAPL"][
        ["date", "name", "close", "SMA_20", "SMA_50", "RSI_14"]
    ].tail(10).to_string(index=False)
)

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df[["SMA_20", "SMA_50", "RSI_14"]].isna().sum())

print("\nSaved to:", output_path)
