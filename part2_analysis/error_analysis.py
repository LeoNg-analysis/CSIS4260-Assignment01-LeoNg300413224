
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
PART2_DIR = PROJECT_DIR / "part2_analysis"

df = pd.read_parquet(PART2_DIR / "predictions.parquet")

models = {
    "Naive": "pred_naive",
    "Linear": "pred_linear",
    "RandomForest": "pred_random_forest"
}

#Calculate absolute errors
for model_name, column in models.items():
    df[f"{model_name}_error"] = (
        df["next_day_close"] - df[column]
    ).abs()

#Top 10 largest Random Forest errors
print("\n=== TOP 10 RANDOM FOREST ERRORS ===")

top_errors = df.nlargest(
    10, "RandomForest_error"
)[
    [
        "date",
        "name",
        "close",
        "next_day_close",
        "pred_random_forest",
        "RandomForest_error"
    ]
]

print(top_errors.to_string(index=False))

#Average error by company
company_errors = df.groupby("name")[
    [
        "Naive_error",
        "Linear_error",
        "RandomForest_error"
    ]
].mean()

company_errors = company_errors.sort_values(
    "RandomForest_error",
    ascending=False
)

print("\n=== TOP 10 COMPANIES BY RANDOM FOREST MAE ===")
print(company_errors.head(10).to_string())

#Percentage errors, excluding zero actual prices
valid = df["next_day_close"] != 0

print("\n=== OVERALL MAPE ===")

for model_name, column in models.items():
    actual = df.loc[valid, "next_day_close"]
    predicted = df.loc[valid, column]

    mape = (
        np.abs((actual - predicted) / actual).mean()
        * 100
    )

    print(f"{model_name}: {mape:.2f}%")

company_errors.to_csv(
    PART2_DIR / "company_errors.csv"
)

top_errors.to_csv(
    PART2_DIR / "top_prediction_errors.csv",
    index=False
)

print("\nError analysis completed!")
