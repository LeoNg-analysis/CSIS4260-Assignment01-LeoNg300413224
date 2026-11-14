
from pathlib import Path
import time
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TEMP_DIR = PROJECT_DIR / "part1_storage" / "scalability_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "all_stocks_5yr.csv")

#Configurations
SCALES = [10, 100]
FORMATS = {
    "CSV": None,
    "Parquet Uncompressed": "NONE",
    "Parquet Snappy": "snappy",
    "Parquet ZSTD": "zstd"
}

results = []

def write_scaled_data(scale, format_name, compression, path):
    start = time.perf_counter()

    if format_name == "CSV":
        for i in range(scale):
            df.to_csv(
                path,
                mode="w" if i == 0 else "a",
                header=(i == 0),
                index=False
            )
    else:
        schema = pa.Schema.from_pandas(df, preserve_index=False)

        with pq.ParquetWriter(
            path,
            schema=schema,
            compression=compression
        ) as writer:
            for i in range(scale):
                table = pa.Table.from_pandas(
                    df,
                    schema=schema,
                    preserve_index=False
                )
                writer.write_table(table)

    return time.perf_counter() - start


def read_scaled_data(format_name, path):
    start = time.perf_counter()
    row_count = 0

    if format_name == "CSV":
        for chunk in pd.read_csv(path, chunksize=100_000):
            row_count += len(chunk)
    else:
        parquet_file = pq.ParquetFile(path)

        for batch in parquet_file.iter_batches(batch_size=100_000):
            row_count += batch.num_rows

    elapsed = time.perf_counter() - start
    return elapsed, row_count


for scale in SCALES:
    print(f"\n========== SCALE {scale}x ==========", flush=True)

    for format_name, compression in FORMATS.items():

        extension = "csv" if format_name == "CSV" else "parquet"
        safe_name = format_name.lower().replace(" ", "_")
        path = TEMP_DIR / f"{safe_name}_{scale}x.{extension}"

        print(f"\nBenchmarking {format_name}...", flush=True)

        gc.collect()

        try:
            write_time = write_scaled_data(
                scale, format_name, compression, path
            )

            size_mb = path.stat().st_size / (1024 ** 2)

            read_time, row_count = read_scaled_data(
                format_name, path
            )

            expected_rows = len(df) * scale
            assert row_count == expected_rows, (
                f"Expected {expected_rows}, got {row_count}"
            )

            result = {
                "Scale": f"{scale}x",
                "Format": format_name,
                "Rows": row_count,
                "Write Time (s)": round(write_time, 4),
                "Read Time (s)": round(read_time, 4),
                "File Size (MB)": round(size_mb, 2)
            }

            results.append(result)
            print(result, flush=True)

        finally:
            if path.exists():
                path.unlink()
                print("Temporary file deleted.", flush=True)

   
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        PROJECT_DIR / "part1_storage" / "scalability_results.csv",
        index=False
    )

print("\n=== SCALABILITY BENCHMARK RESULTS ===")
print(pd.DataFrame(results).to_string(index=False))
