import pandas as pd
import joblib, os

base = r"C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter"

df_price = pd.read_parquet(os.path.join(base, 'data/processed/hourly.parquet'))
print('Price dataset cols:', df_price.shape[1], '| rows:', len(df_price))
print('Has load_fc:', 'load_fc' in df_price.columns)
print('Load-related cols:', [c for c in df_price.columns if 'load' in c.lower()])

df_load = pd.read_parquet(os.path.join(base, 'data/processed/hourly_load.parquet'))
print('\nLoad dataset cols:', df_load.shape[1], '| rows:', len(df_load))

df_lfc = pd.read_parquet(os.path.join(base, 'data/processed/load_forecast_hourly.parquet'))
print('\nLoad forecast parquet cols:', df_lfc.columns.tolist())
print('Load forecast rows:', len(df_lfc), '| index range:', df_lfc.index.min(), '->', df_lfc.index.max())

m = joblib.load(os.path.join(base, 'models/lgbm_hourly_price_openloop.pkl'))
if hasattr(m, 'booster_'):
    fn = m.booster_.feature_name()
    print('\nLGBM price openloop n_features:', len(fn))
    print('Has load_fc in model:', 'load_fc' in fn)
    print('Load-related features:', [f for f in fn if 'load' in f.lower()])
elif hasattr(m, 'feature_names_in_'):
    print('\nModel n_features:', len(m.feature_names_in_))
