
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError(
        f"Expected exactly 1 CSV file, found {len(csv_files)}"
    )

csv_path = csv_files[0]

df = pd.read_csv(csv_path)

print("Dataset:", csv_path.name)
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nMemory usage (MB):")
print(round(df.memory_usage(deep=True).sum() / 1024**2, 2))
