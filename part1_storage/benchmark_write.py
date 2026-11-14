
from pathlib import Path
import pandas as pd
import time
import gc
import statistics

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "part1_storage" / "benchmark_output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


csv_path = DATA_DIR / "all_stocks_5yr.csv"
df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)


formats = {
    "CSV": None,
    "Parquet Uncompressed": None,
    "Parquet Snappy": "snappy",
    "Parquet ZSTD": "zstd"
}

iterations = 5
results = []

for format_name, compression in formats.items():

    print(f"\nBenchmarking {format_name}...")

    times = []
    file_sizes = []

    for i in range(iterations):

        if format_name == "CSV":
            output_path = OUTPUT_DIR / "benchmark.csv"
        else:
            output_path = OUTPUT_DIR / "benchmark.parquet"

        gc.collect()

        start = time.perf_counter()

        if format_name == "CSV":
            df.to_csv(output_path, index=False)
        else:
            df.to_parquet(
                output_path,
                engine="pyarrow",
                compression=compression,
                index=False
            )

        elapsed = time.perf_counter() - start

        times.append(elapsed)
        file_sizes.append(
            output_path.stat().st_size / (1024 ** 2)
        )

        print(f"Run {i + 1}: {elapsed:.4f} seconds")

    results.append({
        "Format": format_name,
        "Average Write Time (s)": round(
            statistics.mean(times), 4
        ),
        "Min Write Time (s)": round(min(times), 4),
        "Max Write Time (s)": round(max(times), 4),
        "File Size (MB)": round(
            statistics.mean(file_sizes), 2
        )
    })

results_df = pd.DataFrame(results)

print("\n=== WRITE BENCHMARK RESULTS ===")
print(results_df.to_string(index=False))

results_path = PROJECT_DIR / "part1_storage" / "write_results_1x.csv"
results_df.to_csv(results_path, index=False)

print("\nResults saved to:", results_path)
