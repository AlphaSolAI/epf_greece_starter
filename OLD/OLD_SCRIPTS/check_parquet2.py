import pandas as pd
# Check entsoe_extra_hourly
extra = pd.read_parquet(r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\data\processed\entsoe_extra_hourly.parquet')
print('=== entsoe_extra_hourly.parquet ===')
print('Shape:', extra.shape)
print('Columns:', list(extra.columns))
print('Index range:', extra.index.min(), 'to', extra.index.max())
print('First 3 rows:')
print(extra.head(3))
print()
# Check what columns are in hourly.parquet relating to load_fc
hourly = pd.read_parquet(r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\data\processed\hourly.parquet')
load_cols = [c for c in hourly.columns if 'load' in c.lower()]
print('load-related columns in hourly.parquet:', load_cols)
print()
print('Dec 1-7 load_fc in hourly.parquet:')
print(hourly.loc['2025-12-01':'2025-12-01 05:00', load_cols[:4]])
