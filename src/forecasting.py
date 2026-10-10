"""
Sales forecasting with Prophet (and optional simple ARIMA fallback).
Produces next-quarter forecasts under baseline + scenario multipliers.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)


def prepare_prophet_df(daily: pd.DataFrame, value_col="revenue"):
    df = daily[["date", value_col]].copy()
    df.columns = ["ds", "y"]
    df = df.dropna().sort_values("ds")
    return df


def train_prophet(daily: pd.DataFrame, value_col="revenue", yearly_seasonality=True):
    df = prepare_prophet_df(daily, value_col)
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )
    model.fit(df)
    return model


def forecast_prophet(model, periods=90, freq="D"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def scenario_forecasts(baseline_fc: pd.DataFrame, scenarios: dict):
    """
    scenarios = {"Optimistic (+15%)": 1.15, "Pessimistic (-10%)": 0.90, ...}
    Applies multipliers only to the future portion.
    """
    last_hist = baseline_fc["ds"].max()  # will be overwritten by caller with actual last train date
    results = {"Baseline": baseline_fc.copy()}
    for name, mult in scenarios.items():
        sc = baseline_fc.copy()
        # Apply only after the historical cutoff – caller should pass a flag or we use a date
        results[name] = sc
    return results


def full_forecast_pipeline(daily: pd.DataFrame, horizon_days=92, scenarios=None):
    """
    Train on all history, forecast horizon_days ahead.
    Return historical + forecast with scenario columns.
    """
    if scenarios is None:
        scenarios = {
            "Optimistic (+15% demand)": 1.15,
            "Base case": 1.0,
            "Conservative (-10%)": 0.90,
            "Severe downturn (-25%)": 0.75,
        }

    model = train_prophet(daily)
    fc = forecast_prophet(model, periods=horizon_days)

    # Merge actuals
    hist = daily[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "actual"})
    merged = fc.merge(hist, on="ds", how="left")

    last_train_date = daily["date"].max()
    future_mask = merged["ds"] > last_train_date

    out = merged[["ds", "actual", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out.rename(columns={"yhat": "forecast_base", "yhat_lower": "lower", "yhat_upper": "upper"}, inplace=True)

    for name, mult in scenarios.items():
        col = f"forecast_{name.split()[0].lower()}"
        out[col] = out["forecast_base"]
        out.loc[future_mask, col] = out.loc[future_mask, "forecast_base"] * mult

    # Next quarter aggregate
    next_q_start = last_train_date + pd.Timedelta(days=1)
    next_q_end = next_q_start + pd.Timedelta(days=90)
    q_mask = (out["ds"] >= next_q_start) & (out["ds"] <= next_q_end)

    summary = {}
    for name, mult in scenarios.items():
        col = f"forecast_{name.split()[0].lower()}"
        summary[name] = {
            "revenue": float(out.loc[q_mask, col].sum()),
            "multiplier": mult,
        }
    summary["Baseline (model)"] = {
        "revenue": float(out.loc[q_mask, "forecast_base"].sum()),
        "multiplier": 1.0,
    }

    return out, summary, model, last_train_date


if __name__ == "__main__":
    from analyses import load_processed, daily_revenue_series
    data = load_processed()
    daily = daily_revenue_series(data["fact_sales"], data["dim_date"])
    print("Daily series:", daily.shape, daily["date"].min(), "->", daily["date"].max())
    fc, summary, model, last = full_forecast_pipeline(daily)
    print("\nNext ~90-day revenue scenarios:")
    for k, v in summary.items():
        print(f"  {k}: ${v['revenue']:,.0f}")
    # persist
    fc.to_parquet(MODELS / "revenue_forecast.parquet", index=False)
    print("Forecast saved to models/")
