import pandas as pd
from pathlib import Path

processed = Path('data/processed')

df_price = pd.read_parquet(processed / 'hourly.parquet', engine='fastparquet')
df_price.index = pd.to_datetime(df_price.index)
print('=== PRICE DATASET ===')
print(f'Shape: {df_price.shape}')
print(f'Date range: {df_price.index.min()} -> {df_price.index.max()}')
print(f'Columns ({len(df_price.columns)}): {list(df_price.columns)}')
print()

df_load = pd.read_parquet(processed / 'hourly_load.parquet', engine='fastparquet')
df_load.index = pd.to_datetime(df_load.index)
print('=== LOAD DATASET ===')
print(f'Shape: {df_load.shape}')
print(f'Date range: {df_load.index.min()} -> {df_load.index.max()}')
print(f'Columns ({len(df_load.columns)}): {list(df_load.columns)}')
print()

df_w = pd.read_parquet(processed / 'weather_gr_hourly.parquet', engine='fastparquet')
df_w.index = pd.to_datetime(df_w.index)
print('=== WEATHER ===')
print(f'Shape: {df_w.shape}')
print(f'Date range: {df_w.index.min()} -> {df_w.index.max()}')
print(f'Columns: {list(df_w.columns)}')
print()

df_e = pd.read_parquet(processed / 'entsoe_extra_hourly.parquet', engine='fastparquet')
df_e.index = pd.to_datetime(df_e.index)
print('=== ENTSOE EXTRA ===')
print(f'Shape: {df_e.shape}')
print(f'Date range: {df_e.index.min()} -> {df_e.index.max()}')
print(f'Columns: {list(df_e.columns)}')
print()

df_lf = pd.read_parquet(processed / 'load_forecast_hourly.parquet', engine='fastparquet')
df_lf.index = pd.to_datetime(df_lf.index)
print('=== LOAD FORECAST ===')
print(f'Shape: {df_lf.shape}')
print(f'Date range: {df_lf.index.min()} -> {df_lf.index.max()}')
print(f'Columns: {list(df_lf.columns)}')
