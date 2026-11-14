
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "part2_analysis" / "stocks_with_indicators.parquet"
OUTPUT_PATH = PROJECT_DIR / "part2_analysis" / "ml_dataset.parquet"

df = pd.read_parquet(INPUT_PATH)

# Sort by stock and date
df = df.sort_values(["name", "date"]).reset_index(drop=True)

# Calculate daily return for each stock
df["daily_return"] = df.groupby("name")["close"].pct_change(
    fill_method=None
)

df["next_day_close"] = df.groupby("name")["close"].shift(-1)

features = [
    "close",
    "volume",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "daily_return"
]

columns = ["date", "name"] + features + ["next_day_close"]

ml_df = df[columns].copy()

print("Before cleaning:", ml_df.shape)

ml_df = ml_df.dropna(subset=features + ["next_day_close"])

print("After cleaning:", ml_df.shape)

ml_df.to_parquet(
    OUTPUT_PATH,
    engine="pyarrow",
    compression="zstd",
    index=False
)

print("\n=== ML DATASET SAMPLE ===")
print(ml_df.head(10).to_string(index=False))

print("\nMissing values:")
print(ml_df.isna().sum())

print("\nSaved to:", OUTPUT_PATH)
