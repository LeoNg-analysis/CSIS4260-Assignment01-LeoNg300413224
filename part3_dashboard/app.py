
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

PROJECT_DIR = Path(__file__).resolve().parent.parent
PART2_DIR = PROJECT_DIR / "part2_analysis"

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    predictions = pd.read_parquet(
        PART2_DIR / "predictions.parquet"
    )
    indicators = pd.read_parquet(
        PART2_DIR / "stocks_with_indicators.parquet"
    )
    return predictions, indicators

predictions, indicators = load_data()

st.title("📈 Stock Price Prediction Dashboard")
st.caption(
    "S&P 500 Historical Stock Analysis | CSIS 4260 Assignment 1"
)

#Stock selection
tickers = sorted(predictions["name"].unique())

default_index = tickers.index("AAPL") if "AAPL" in tickers else 0

ticker = st.selectbox(
    "Search or select a stock ticker",
    tickers,
    index=default_index
)

stock_pred = predictions[
    predictions["name"] == ticker
].sort_values("date").copy()

stock_ind = indicators[
    indicators["name"] == ticker
].sort_values("date").copy()

#Date range filter
st.subheader("Date Range")

min_date = stock_pred["date"].min().date()
max_date = stock_pred["date"].max().date()

start_date, end_date = st.date_input(
    "Select backtesting period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

#Filter predictions by selected date range
stock_pred = stock_pred[
    (stock_pred["date"].dt.date >= start_date) &
    (stock_pred["date"].dt.date <= end_date)
].copy()

st.subheader(f"{ticker} — Closing Price Predictions")

#Predictions refer to the next trading day.
#Move each prediction to its target date.
#Find the actual next trading date for each stock
stock_ind["target_date"] = stock_ind["date"].shift(-1)

#Match predictions with the next trading date
stock_pred = stock_pred.merge(
    stock_ind[["date", "target_date"]],
    on="date",
    how="left",
    validate="one_to_one"
)

chart_df = stock_pred.dropna(subset=["target_date"])


fig = go.Figure()

for column, label, dash in [
    ("next_day_close", "Actual Close", "solid"),
    ("pred_linear", "Linear Regression", "dash"),
    ("pred_random_forest", "Random Forest", "dot"),
    ("pred_naive", "Naive Baseline", "dashdot")
]:
    fig.add_trace(go.Scatter(
        x=chart_df["target_date"],
        y=chart_df[column],
        name=label,
        mode="lines",
        line=dict(dash=dash)
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Closing Price (USD)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

#Technical indicators
st.subheader("Technical Indicators")

fig_sma = go.Figure()

for column, label in [
    ("close", "Closing Price"),
    ("SMA_20", "SMA 20"),
    ("SMA_50", "SMA 50")
]:
    fig_sma.add_trace(go.Scatter(
        x=stock_ind["date"],
        y=stock_ind[column],
        name=label,
        mode="lines"
    ))

fig_sma.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=400
)

st.plotly_chart(fig_sma, use_container_width=True)

#RSI
fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
    x=stock_ind["date"],
    y=stock_ind["RSI_14"],
    name="RSI 14",
    mode="lines"
))

fig_rsi.add_hline(y=70, line_dash="dash")
fig_rsi.add_hline(y=30, line_dash="dash")

fig_rsi.update_layout(
    title="Relative Strength Index (RSI 14)",
    xaxis_title="Date",
    yaxis_title="RSI",
    yaxis_range=[0, 100],
    height=300
)

st.plotly_chart(fig_rsi, use_container_width=True)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.subheader(f"{ticker} — Model Performance")

model_columns = {
    "Naive Baseline": "pred_naive",
    "Linear Regression": "pred_linear",
    "Random Forest": "pred_random_forest"
}

ticker_results = []

for model_name, column in model_columns.items():
    actual = stock_pred["next_day_close"]
    predicted = stock_pred[column]

    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    ticker_results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })

ticker_results_df = pd.DataFrame(ticker_results)

st.dataframe(
    ticker_results_df,
    use_container_width=True,
    hide_index=True
)

#model comparison
st.subheader("Overall Model Performance")

results = pd.read_csv(
    PART2_DIR / "backtesting_results.csv"
)

st.dataframe(results, use_container_width=True)

st.caption(
    "Historical backtesting only. "
    "Predictions are not live forecasts or investment advice."
)
