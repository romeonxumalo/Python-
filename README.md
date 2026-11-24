# Electricity Consumption Analysis & Forecasting (Python)

## 1. Project Summary
A data-analysis mini project that studies 3 years (2023-2025) of daily household electricity use,
finds consumption patterns, and forecasts the next 30 days and their estimated bill.

- **Analysis:** totals, yearly/monthly trends, peak and lowest days, weekday vs weekend, temperature correlation, bill estimate.
- **Forecasting:** Linear Regression vs Random Forest using trend, season (sin/cos of day-of-year), weekday and temperature; the better model (lowest RMSE on the last 90 days) is used for the 30-day forecast.
- **Libraries:** pandas, numpy, matplotlib, scikit-learn.
- **Data:** `data/electricity.csv` is a *synthetic* dataset created by `generate_data.py` (summer AC load, winter heating, weekend effect, slow growth). To use real data, replace it with a CSV having columns `date, temp_c, kwh`.

| File | Purpose |
|---|---|
| `generate_data.py` | Creates the sample dataset |
| `analysis.py` | Summary statistics and charts 1-4 |
| `forecast.py` | Models, evaluation, forecast and charts 5-6 |
| `main.py` | Runs everything |

**Run:** `pip install -r requirements.txt` then `python main.py`

## 2. Project Source Code
See `generate_data.py`, `analysis.py`, `forecast.py`, `main.py`.

## 3. Project Output
Console output: `output.txt`. Charts in `charts/`:

![Daily](charts/1_daily_consumption.png)
![Monthly](charts/2_monthly_consumption.png)
![Patterns](charts/3_seasonal_patterns.png)
![Temp](charts/4_temp_vs_kwh.png)
![Models](charts/5_model_comparison.png)
![Forecast](charts/6_future_forecast.png)

## 4. GitHub Repository Link
https://github.com/<your-username>/electricity-analysis-forecast
