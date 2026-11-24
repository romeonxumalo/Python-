"""Electricity Consumption Analysis & Forecasting - run: python main.py"""
import os
import generate_data
import analysis
import forecast

os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)
if not os.path.exists("data/electricity.csv"):
    generate_data.generate().to_csv("data/electricity.csv", index=False)

df = analysis.load()
s = analysis.summarize(df)

print("=" * 55)
print(" PART 1: CONSUMPTION ANALYSIS")
print("=" * 55)
print(f"Period            : {df.date.min():%d %b %Y} to {df.date.max():%d %b %Y} ({s['days']} days)")
print(f"Total consumption : {s['total_kwh']:,.0f} kWh")
print(f"Average per day   : {s['avg_daily']:.2f} kWh")
print(f"Peak day          : {s['peak_day'].date:%d %b %Y} ({s['peak_day'].kwh:.1f} kWh)")
print(f"Lowest day        : {s['low_day'].date:%d %b %Y} ({s['low_day'].kwh:.1f} kWh)")
print(f"Weekday vs weekend: {s['weekday_avg']:.2f} vs {s['weekend_avg']:.2f} kWh/day")
print(f"Temp correlation  : {s['corr_temp']:.2f}")
print(f"Estimated bill    : Rs {s['est_bill']:,.0f} (at Rs {analysis.TARIFF_PER_KWH}/unit)")
print("\nYearly consumption (kWh):")
for y, v in s["yearly"].items():
    print(f"  {y}: {v:,.0f}")
top = s["monthly"].sort_values(ascending=False).head(3)
print("\nTop 3 highest months:")
for d, v in top.items():
    print(f"  {d:%b %Y}: {v:,.0f} kWh")
analysis.make_charts(df, s)

print("\n" + "=" * 55)
print(" PART 2: FORECASTING")
print("=" * 55)
d, test, models, results, preds, best = forecast.run_models(df)
print(f"{'Model':<20}{'MAE':>8}{'RMSE':>8}{'MAPE %':>9}")
for name, (mae, rmse, mape) in results.items():
    print(f"{name:<20}{mae:>8.2f}{rmse:>8.2f}{mape:>9.2f}")
print(f"\nBest model: {best}")

fut = forecast.future_forecast(df, d, models[best], days=30)
total = fut.forecast_kwh.sum()
print(f"\nNext 30 days ({fut.date.min():%d %b} to {fut.date.max():%d %b %Y}):")
print(f"  Forecast total  : {total:,.0f} kWh")
print(f"  Forecast daily  : {fut.forecast_kwh.mean():.2f} kWh")
print(f"  Estimated bill  : Rs {total * analysis.TARIFF_PER_KWH:,.0f}")
print("\nFirst 7 forecast days:")
for _, r in fut.head(7).iterrows():
    print(f"  {r.date:%a %d %b %Y}: {r.forecast_kwh:.2f} kWh")
fut[["date", "forecast_kwh"]].assign(forecast_kwh=fut.forecast_kwh.round(2)).to_csv("data/forecast_next_30_days.csv", index=False)

forecast.make_charts(df, test, preds, best, fut)
print("\nCharts saved in charts/ | Forecast saved to data/forecast_next_30_days.csv")
