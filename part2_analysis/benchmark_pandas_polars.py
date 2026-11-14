
from pathlib import Path
import pandas as pd
import polars as pl
import time
import statistics
import gc

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "data" / "all_stocks_5yr.csv"
OUTPUT_DIR = PROJECT_DIR / "part2_analysis"

ITERATIONS = 5
results = []


def benchmark(task_name, library, function):
    times = []

    for i in range(ITERATIONS):
        gc.collect()

        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        del result

    results.append({
        "Task": task_name,
        "Library": library,
        "Average Time (s)": round(statistics.mean(times), 5),
        "Min Time (s)": round(min(times), 5),
        "Max Time (s)": round(max(times), 5)
    })


benchmark(
    "Read CSV",
    "Pandas",
    lambda: pd.read_csv(CSV_PATH)
)

benchmark(
    "Read CSV",
    "Polars",
    lambda: pl.read_csv(CSV_PATH)
)

pdf = pd.read_csv(CSV_PATH)
pldf = pl.read_csv(CSV_PATH)


benchmark(
    "Filter AAPL",
    "Pandas",
    lambda: pdf[pdf["name"] == "AAPL"]
)

benchmark(
    "Filter AAPL",
    "Polars",
    lambda: pldf.filter(pl.col("name") == "AAPL")
)


benchmark(
    "GroupBy Mean Close",
    "Pandas",
    lambda: pdf.groupby("name")["close"].mean()
)

benchmark(
    "GroupBy Mean Close",
    "Polars",
    lambda: pldf.group_by("name").agg(
        pl.col("close").mean()
    )
)

benchmark(
    "Sort",
    "Pandas",
    lambda: pdf.sort_values(["name", "date"])
)

benchmark(
    "Sort",
    "Polars",
    lambda: pldf.sort(["name", "date"])
)


results_df = pd.DataFrame(results)

print("\n=== PANDAS VS POLARS BENCHMARK ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "pandas_polars_results.csv",
    index=False
)

print("\nBenchmark completed!")
