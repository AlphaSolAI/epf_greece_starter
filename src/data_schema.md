# Raw Data Schema (expected)

File: `data/raw/henex_prices.csv`

- `timestamp`: ISO datetime string with timezone or naive UTC (e.g., `2024-01-05 00:00:00`)
- `price`: float — day-ahead hourly clearing price **€/MWh**
- `load` (optional): float — system load for that hour (MW)
- Other optional exogenous: `temp`, `res_wind`, `res_solar`, `gas_price`, etc.

> If your file is already daily, keep `timestamp` as date and one price per day.
