
from pathlib import Path
import pandas as pd
import time
import gc
import statistics

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"

files = {
    "CSV": DATA_DIR / "all_stocks_5yr.csv",
    "Parquet Uncompressed": PARQUET_DIR / "stocks_uncompressed.parquet",
    "Parquet Snappy": PARQUET_DIR / "stocks_snappy.parquet",
    "Parquet ZSTD": PARQUET_DIR / "stocks_zstd.parquet"
}

results = []
iterations = 5

for format_name, file_path in files.items():

    print(f"\nBenchmarking {format_name}...")

    times = []

    for i in range(iterations):

        gc.collect()

        start = time.perf_counter()

        if format_name == "CSV":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(
                file_path,
                engine="pyarrow"
            )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"Run {i + 1}: {elapsed:.4f} seconds")

        del df

    avg_time = statistics.mean(times)

    results.append({
        "Format": format_name,
        "Average Read Time (s)": round(avg_time, 4),
        "Min Read Time (s)": round(min(times), 4),
        "Max Read Time (s)": round(max(times), 4)
    })

#Display results
results_df = pd.DataFrame(results)

print("\n=== READ BENCHMARK RESULTS ===")
print(results_df.to_string(index=False))

#Save results
output_path = PROJECT_DIR / "part1_storage" / "read_results_1x.csv"
results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
