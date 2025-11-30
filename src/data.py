import pandas as pd
import numpy as np
from pathlib import Path
import holidays
import sys
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "henex_prices.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"

def main():
    print("🛠️ Advanced Feature Engineering (Price + Load)...")
    
    # 1. Φόρτωση Τιμών (Αν δεν υπάρχει, τρέχουμε process_zips)
    if not RAW_PATH.exists():
        print("❌ Δεν βρέθηκε το henex_prices.csv")
        return
        
    df_hourly = pd.read_csv(RAW_PATH)
    df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'])
    
    # 2. Daily Price
    df_hourly['date'] = df_hourly['timestamp'].dt.date
    daily = df_hourly.groupby('date')['price'].mean().to_frame(name='y')
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = 'ds'
    
    # 3. Φόρτωση Load (Αν υπάρχει ήδη στο daily.parquet από το add_load)
    # Πρέπει να διαβάσουμε το ΠΑΛΙΟ daily.parquet για να πάρουμε το Load που προσθέσαμε με κόπο!
    if PROCESSED_PATH.exists():
        old_df = pd.read_parquet(PROCESSED_PATH)
        if 'load' in old_df.columns:
            # Κάνουμε merge το load στο νέο dataframe
            daily = daily.join(old_df['load'], how='left')
            print("   ✅ Load column preserved.")
    
    # 4. Feature Engineering
    
    # Calendar
    gr_holidays = holidays.Greece()
    daily['dow'] = daily.index.dayofweek
    daily['month'] = daily.index.month
    daily['dayofyear'] = daily.index.dayofyear
    daily['is_weekend'] = (daily['dow'] >= 5).astype(int)
    daily['is_holiday'] = daily.index.map(lambda x: int(x in gr_holidays))

    # PRICE LAGS
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        daily[f'y_lag{lag}'] = daily['y'].shift(lag)

    # PRICE ROLLING
    for win in [7, 30]:
        daily[f'y_rollmean_{win}'] = daily['y'].rolling(win).mean()
        daily[f'y_rollstd_{win}'] = daily['y'].rolling(win).std()
        
    # LOAD FEATURES (Αν υπάρχει load)
    if 'load' in daily.columns:
        print("   🔹 Generating Load Features...")
        # Load Lags (Χθες, προχθές, 1 βδομάδα πριν)
        for lag in [1, 2, 7]:
            daily[f'load_lag{lag}'] = daily['load'].shift(lag)
            
        # Load Rolling
        daily['load_rollmean_7'] = daily['load'].rolling(7).mean()
        
        # Relative Load (Πόσο % πάνω/κάτω από το μέσο όρο είναι σήμερα)
        daily['load_ratio'] = daily['load'] / (daily['load_rollmean_7'] + 1e-9)
        
        # Load Diff (Τάση φορτίου)
        daily['load_diff'] = daily['load'].diff()

    # Drop NaNs
    df_final = daily.dropna()
    
    # Save
    df_final.to_parquet(PROCESSED_PATH)
    print(f"✅ Dataset updated: {len(df_final)} rows, {len(df_final.columns)} features.")
    print(f"   Features: {df_final.columns.tolist()}")

if __name__ == "__main__":
    main()