"""Creates a realistic sample dataset: 3 years of daily household electricity use.
Replace data/electricity.csv with your own file (columns: date, temp_c, kwh) to analyse real data."""
import os
import numpy as np
import pandas as pd


def generate(start="2023-01-01", end="2025-12-31", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.values
    # North-India style temperature: hot summer peak around day 160
    temp = 25 + 11 * np.sin(2 * np.pi * (doy - 105) / 365) + rng.normal(0, 1.8, len(dates))
    # AC use rises above ~28C, heater/geyser use rises below ~15C
    cooling = np.clip(temp - 28, 0, None) * 2.4
    heating = np.clip(15 - temp, 0, None) * 0.9
    weekend = (dates.dayofweek >= 5).astype(int) * 2.2
    trend = np.linspace(0, 1.5, len(dates))          # slow growth in usage
    kwh = 9 + cooling + heating + weekend + trend + rng.normal(0, 1.1, len(dates))
    return pd.DataFrame({"date": dates, "temp_c": temp.round(1), "kwh": kwh.round(2)})


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv("data/electricity.csv", index=False)
    print(f"Saved data/electricity.csv with {len(df)} rows")
