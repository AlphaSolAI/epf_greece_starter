import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

LOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "load"
DAILY_PARQUET = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"

def parse_binary_xls(filepath):
    """
    Διαβάζει Binary .xls (OLE2) αρχεία του ΑΔΜΗΕ.
    Ψάχνει τη γραμμή που περιέχει τα headers 'Date' και τις ώρες '1', '2'...
    """
    try:
        # 1. Διαβάζουμε τις πρώτες 20 γραμμές χωρίς header για να βρούμε πού είναι τα data
        # engine='xlrd' είναι ΥΠΟΧΡΕΩΤΙΚΟ για αυτά τα αρχεία
        df_scan = pd.read_excel(filepath, header=None, nrows=20, engine='xlrd')
        
        header_idx = -1
        for i, row in df_scan.iterrows():
            # Μετατροπή σε string για αναζήτηση
            row_vals = [str(v).strip().lower() for v in row.values]
            
            # Το κλειδί: Η γραμμή header έχει "date" και τον αριθμό "1" ή "1.0"
            if any('date' in x or 'ημερ' in x for x in row_vals) and \
               any(x == '1' or x == '1.0' for x in row_vals):
                header_idx = i
                break
        
        if header_idx == -1:
            return None

        # 2. Ξαναδιαβάζουμε με το σωστό header
        df = pd.read_excel(filepath, header=header_idx, engine='xlrd')
        
        # Καθαρισμός ονομάτων
        df.columns = [str(c).strip() for c in df.columns]
        
        # 3. Εντοπισμός στήλης Date
        date_col = next((c for c in df.columns if 'Date' in c or 'Ημερ' in c), None)
        if not date_col: return None
        
        # 4. Εντοπισμός στηλών Ώρας (1 έως 24)
        # Ψάχνουμε στήλες που είναι αριθμοί 1..24
        hour_cols = []
        for c in df.columns:
            try:
                # Μερικές φορές είναι float 1.0, μερικές int 1
                val = float(c)
                if 1 <= val <= 24:
                    hour_cols.append(c)
            except: continue
            
        if not hour_cols: return None

        # 5. Υπολογισμός
        # Κρατάμε μόνο γραμμές που έχουν valid date
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col])
        
        # Μετατροπή φορτίων σε αριθμούς
        for c in hour_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Μέσος όρος γραμμής (axis=1) για τις 24 ώρες
        df['daily_load'] = df[hour_cols].mean(axis=1)
        
        return df.set_index(date_col)['daily_load']

    except Exception as e:
        # print(f"Error in {filepath.name}: {e}")
        return None

def main():
    print("🔌 Processing Binary XLS Files (xlrd mode)...")
    
    if not DAILY_PARQUET.exists():
        print("❌ Missing daily.parquet")
        return

    files = list(LOAD_DIR.glob("*"))
    # Φιλτράρουμε μόνο τα .xls
    files = [f for f in files if f.name.endswith('xls') or f.name.endswith('xlsx')]
    
    print(f"   Found {len(files)} excel files. Scanning...")
    
    all_daily_loads = []
    
    for i, f in enumerate(files):
        daily = parse_binary_xls(f)
        if daily is not None:
            all_daily_loads.append(daily)
            
        if (i+1) % 200 == 0:
            print(f"   ... processed {i+1} files ...")

    if not all_daily_loads:
        print("❌ FAILED. Δεν μπόρεσα να βρω τη δομή (Date ... 1 ... 24).")
        return

    # Merge
    full_load = pd.concat(all_daily_loads)
    full_load = full_load.groupby(level=0).mean() # Handle duplicates
    
    print(f"   ✅ Success! Extracted load for {len(full_load)} days.")

    # Save
    df_prices = pd.read_parquet(DAILY_PARQUET)
    df_prices.index = pd.to_datetime(df_prices.index)
    
    load_df = full_load.to_frame(name='load')
    load_df.index = pd.to_datetime(load_df.index)
    
    if 'load' in df_prices.columns:
        df_prices = df_prices.drop(columns=['load'])
        
    merged = df_prices.join(load_df, how='left')
    
    # Fill gaps
    before = merged['load'].isna().sum()
    merged['load'] = merged['load'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    merged.to_parquet(DAILY_PARQUET)
    print(f"✅ SAVED daily.parquet WITH LOAD. (Filled {before} gaps)")
    print(merged[['y', 'load']].tail())

if __name__ == "__main__":
    main()