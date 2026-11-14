
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
PART2_DIR = PROJECT_DIR / "part2_analysis"

df = pd.read_parquet(PART2_DIR / "ml_dataset.parquet")
df["date"] = pd.to_datetime(df["date"])

features = [
    "close",
    "volume",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "daily_return"
]

#Split by unique trading dates (80/20)
dates = np.sort(df["date"].unique())
split_idx = int(len(dates) * 0.8)
split_date = dates[split_idx]

#The target is the NEXT trading day's close.
#Exclude the final training date to prevent
#training labels from crossing into test dates.
train = df[df["date"] < dates[split_idx - 1]].copy()
test = df[df["date"] >= split_date].copy()

X_train = train[features]
y_train = train["next_day_close"]

X_test = test[features]
y_test = test["next_day_close"]

print("Train rows:", len(train))
print("Test rows:", len(test))
print("Train date range:", train["date"].min(), "to", train["date"].max())
print("Test date range:", test["date"].min(), "to", test["date"].max())

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

results = []
predictions = test[["date", "name", "close", "next_day_close"]].copy()

for model_name, model in models.items():
    print(f"\nTraining {model_name}...", flush=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })

    column = (
        "pred_linear"
        if model_name == "Linear Regression"
        else "pred_random_forest"
    )
    predictions[column] = y_pred

    filename = (
        "linear_regression.joblib"
        if model_name == "Linear Regression"
        else "random_forest.joblib"
    )
    joblib.dump(model, PART2_DIR / filename)

results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    PART2_DIR / "model_results.csv",
    index=False
)

predictions.to_parquet(
    PART2_DIR / "predictions.parquet",
    index=False
)

print("\nModel training completed!")
