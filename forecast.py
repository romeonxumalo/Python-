"""Forecasting daily consumption with Linear Regression and Random Forest."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["t", "doy_sin", "doy_cos", "dow", "is_weekend", "temp_c"]


def add_features(df, t0):
    d = df.copy()
    doy = d.date.dt.dayofyear
    d["t"] = (d.date - t0).dt.days
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["dow"] = d.date.dt.dayofweek
    d["is_weekend"] = (d.dow >= 5).astype(int)
    return d


def evaluate(y, p):
    mae = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    mape = np.mean(np.abs((y - p) / y)) * 100
    return mae, rmse, mape


def run_models(df, test_days=90):
    t0 = df.date.min()
    d = add_features(df, t0)
    train, test = d.iloc[:-test_days], d.iloc[-test_days:]
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)}
    results, preds = {}, {}
    for name, m in models.items():
        m.fit(train[FEATURES], train.kwh)
        preds[name] = m.predict(test[FEATURES])
        results[name] = evaluate(test.kwh.values, preds[name])
    best = min(results, key=lambda k: results[k][1])   # lowest RMSE
    return d, test, models, results, preds, best


def future_forecast(df, d, model, days=30):
    """Forecast next `days`. Future temperature is unknown, so use the
    average temperature seen on the same day-of-year in past years."""
    t0 = df.date.min()
    start = df.date.max() + pd.Timedelta(days=1)
    fut = pd.DataFrame({"date": pd.date_range(start, periods=days)})
    clim = df.groupby(df.date.dt.dayofyear).temp_c.mean()
    fut["temp_c"] = fut.date.dt.dayofyear.map(clim).values
    fut = add_features(fut, t0)
    fut["forecast_kwh"] = model.predict(fut[FEATURES])
    return fut


def make_charts(df, test, preds, best, fut, folder="charts"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test.date, test.kwh, color="black", lw=1.5, label="Actual")
    ax.plot(test.date, preds["Linear Regression"], color="#fd8d3c", lw=1.2, label="Linear Regression")
    ax.plot(test.date, preds["Random Forest"], color="#3182bd", lw=1.2, label="Random Forest")
    ax.set(title="Model Test: Actual vs Predicted (last 90 days)", ylabel="kWh")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{folder}/5_model_comparison.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    hist = df.tail(90)
    ax.plot(hist.date, hist.kwh, color="#08519c", label="Recent actual")
    ax.plot(fut.date, fut.forecast_kwh, color="#e6550d", lw=2, label=f"Forecast ({best})")
    ax.axvline(fut.date.min(), color="gray", ls="--")
    ax.set(title="30-Day Electricity Consumption Forecast", ylabel="kWh")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{folder}/6_future_forecast.png", dpi=130); plt.close(fig)
