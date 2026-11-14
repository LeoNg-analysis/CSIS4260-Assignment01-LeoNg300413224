
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PART2_DIR = PROJECT_DIR / "part2_analysis"

df = pd.read_parquet(PART2_DIR / "predictions.parquet")

#Naive baseline: tomorrow's close = today's close
df["pred_naive"] = df["close"]

models = {
    "Naive Baseline": "pred_naive",
    "Linear Regression": "pred_linear",
    "Random Forest": "pred_random_forest"
}

results = []

for model_name, column in models.items():

    actual = df["next_day_close"]
    predicted = df[column]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })

results_df = pd.DataFrame(results)

print("\n=== BACKTESTING RESULTS ===")
print(results_df.to_string(index=False))

#Compare MAE improvement
baseline_mae = results_df.loc[
    results_df["Model"] == "Naive Baseline", "MAE"
].iloc[0]

print("\n=== IMPROVEMENT VS NAIVE BASELINE ===")

for _, row in results_df.iterrows():
    improvement = (
        (baseline_mae - row["MAE"]) / baseline_mae
    ) * 100

    print(f"{row['Model']}: {improvement:.2f}%")


results_df.to_csv(
    PART2_DIR / "backtesting_results.csv",
    index=False
)

df.to_parquet(
    PART2_DIR / "predictions.parquet",
    index=False
)

print("\nBacktesting completed!")
