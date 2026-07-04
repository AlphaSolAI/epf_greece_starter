import sys, os
sys.path.insert(0, r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter')
os.chdir(r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter')
from src.data import load_data
df = load_data('hourly')
print('Data range:', df.index.min(), 'to', df.index.max())
print('Total rows:', len(df))
if 'load_fc' in df.columns:
    print('load_fc Dec 1 first 5h:', df.loc['2025-12-01':'2025-12-01 04:00', 'load_fc'].values)
    print('load_fc nulls total:', df['load_fc'].isna().sum())
    print('Values per day Dec-1:', df.loc['2025-12-01', 'load_fc'].count())
