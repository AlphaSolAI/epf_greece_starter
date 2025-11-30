# EPF Greece Starter (Day-Ahead, Daily)

This is a minimal, **runnable** starter for your diploma project on electricity price forecasting (day-ahead) for the Greek market (HEnEx).

### Assumptions
- **Target**: Daily average of day-ahead hourly clearing prices (can switch to any daily aggregation).
- **Features**: Calendar (dow, month, holidays placeholder), lagged prices, rolling statistics; optional load/weather exogenous.
- **Baselines**: Naive (yesterday), Seasonal Naive (same weekday last week), ARIMA.
- **ML**: XGBoost regressor.
- **DL** (placeholders to fill next): LSTM/GRU, Transformer/TFT.

> Replace `data/raw/henex_prices.csv` with your file (see format in `src/data_schema.md`).

### Quickstart
```bash
# 1) Create a venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Put your raw CSV at data/raw/henex_prices.csv
#    Required columns: timestamp (ISO), price (float), [optional] load, res_wind, res_solar, temp

# 3) Preprocess -> creates data/processed/daily.parquet
python -m src.data

# 4) Run baselines (Naive, Seasonal Naive, ARIMA)
python -m src.baselines

# 5) Train XGBoost model with rolling CV
python -m src.train_xgb

# 6) Launch simple UI
streamlit run app/streamlit_app.py
```
