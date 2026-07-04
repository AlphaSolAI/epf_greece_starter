import json
# Check which load models are in the OL load monthly JSON
with open('dashboard_data_hourly_load_openloop_h24_monthly.json') as f:
    d = json.load(f)
print('Load OL 24h models:')
for name in d['series']:
    if name not in ['Naive-1','Naive-24','Naive-168','Seasonal Profile']:
        vals = [v for v in d['series'][name] if v is not None]
        print(f'  {name}: {len(vals)} values')

from src.split_utils import load_processed, make_xy
df = load_processed('hourly', task='price')
X, _ = make_xy(df.iloc[:100])
load_cols = [c for c in X.columns if 'load' in c.lower()]
print('\nLoad-related features in price model:', load_cols)
print('Is load_fc in price features?', 'load_fc' in X.columns)
