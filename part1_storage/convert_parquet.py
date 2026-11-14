
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = DATA_DIR / "all_stocks_5yr.csv"

df = pd.read_csv(csv_path)

print("Original dataset shape:", df.shape)

# Convert to Parquet
compression_options = {
    "uncompressed": None,
    "snappy": "snappy",
    "zstd": "zstd"
}

for name, compression in compression_options.items():

    output_path = OUTPUT_DIR / f"stocks_{name}.parquet"

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression=compression,
        index=False
    )

    file_size = output_path.stat().st_size / (1024 ** 2)

    print(f"{name}: {file_size:.2f} MB")

print("Parquet conversion completed.")
