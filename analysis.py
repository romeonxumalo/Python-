"""Exploratory analysis of electricity consumption."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARIFF_PER_KWH = 6.5   # Rs per unit (change as per your DISCOM slab)


def load(path="data/electricity.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def summarize(df):
    out = {}
    out["days"] = len(df)
    out["total_kwh"] = df.kwh.sum()
    out["avg_daily"] = df.kwh.mean()
    out["peak_day"] = df.loc[df.kwh.idxmax(), ["date", "kwh"]]
    out["low_day"] = df.loc[df.kwh.idxmin(), ["date", "kwh"]]
    out["monthly"] = df.set_index("date").kwh.resample("MS").sum()
    out["yearly"] = df.groupby(df.date.dt.year).kwh.sum()
    out["weekday_avg"] = df[df.date.dt.dayofweek < 5].kwh.mean()
    out["weekend_avg"] = df[df.date.dt.dayofweek >= 5].kwh.mean()
    out["corr_temp"] = df.kwh.corr(df.temp_c)
    out["est_bill"] = out["total_kwh"] * TARIFF_PER_KWH
    return out


def make_charts(df, s, folder="charts"):
    # 1. daily consumption + 30-day rolling average
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.date, df.kwh, color="#9ecae1", lw=0.7, label="Daily")
    ax.plot(df.date, df.kwh.rolling(30).mean(), color="#08519c", lw=2, label="30-day average")
    ax.set(title="Daily Electricity Consumption", ylabel="kWh")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{folder}/1_daily_consumption.png", dpi=130); plt.close(fig)

    # 2. monthly totals
    fig, ax = plt.subplots(figsize=(11, 4))
    m = s["monthly"]
    ax.bar(m.index, m.values, width=25, color="#3182bd")
    ax.set(title="Monthly Consumption", ylabel="kWh")
    fig.tight_layout(); fig.savefig(f"{folder}/2_monthly_consumption.png", dpi=130); plt.close(fig)

    # 3. average by month-of-year & weekday
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.groupby(df.date.dt.month).kwh.mean().plot(kind="bar", ax=axes[0], color="#fd8d3c")
    axes[0].set(title="Average Daily kWh by Month", xlabel="Month", ylabel="kWh")
    wd = df.groupby(df.date.dt.dayofweek).kwh.mean()
    wd.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd.plot(kind="bar", ax=axes[1], color="#74c476")
    axes[1].set(title="Average Daily kWh by Weekday", xlabel="")
    fig.tight_layout(); fig.savefig(f"{folder}/3_seasonal_patterns.png", dpi=130); plt.close(fig)

    # 4. temperature vs consumption
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df.temp_c, df.kwh, s=8, alpha=0.5, color="#de2d26")
    ax.set(title=f"Temperature vs Consumption (r = {s['corr_temp']:.2f})",
           xlabel="Temperature (°C)", ylabel="kWh")
    fig.tight_layout(); fig.savefig(f"{folder}/4_temp_vs_kwh.png", dpi=130); plt.close(fig)
