import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error
import sys; import io; import warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'); warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid'); sns.set_context("paper", font_scale=1.4)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

def main():
    print("📊 Ανάλυση Μεταβλητότητας (Volatility Analysis)...")
    if not DATA_PATH.exists(): return
    
    # 1. Φόρτωση Δεδομένων & Υπολογισμός Volatility
    df = pd.read_parquet(DATA_PATH)
    # Volatility = Τυπική απόκλιση των τελευταίων 7 ημερών
    df['Volatility_7d'] = df['y'].rolling(window=7).std()
    
    # 2. Φόρτωση του Νικητή (SVR)
    model_path = MODELS_DIR / "svr_day.pkl"
    if not model_path.exists():
        print("⚠️ Δεν βρέθηκε το SVR μοντέλο. Τρέξε πρώτα το train_svr.")
        return
    model = joblib.load(model_path)
    
    # 3. Υπολογισμός Σφάλματος (σε όλο το ιστορικό)
    # Αφαιρούμε τα NaN που δημιουργεί το rolling volatility
    df_clean = df.dropna()
    df_clean['Prediction'] = model.predict(df_clean.drop(columns=['y', 'Volatility_7d']))
    df_clean['AbsError'] = np.abs(df_clean['y'] - df_clean['Prediction'])
    
    # 4. Γραφήματα
    
    # Plot 1: Scatter (Συσχέτιση)
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_clean, x='Volatility_7d', y='AbsError', 
                scatter_kws={'alpha':0.3, 'color':'#457B9D'}, line_kws={'color':'#E63946'})
    plt.title("Correlation: Market Volatility vs SVR Forecast Error", fontsize=16)
    plt.xlabel("Market Volatility (7-day Std Dev)", fontsize=12)
    plt.ylabel("Absolute Forecast Error (€/MWh)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(REPORT_DIR / "volatility_scatter_svr.png", dpi=300)
    
    # Plot 2: Χρονοσειρά (Οπτική Επιβεβαίωση)
    # Χρησιμοποιούμε κινητό μέσο όρο 30 ημερών για να φαίνεται η τάση καθαρά
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color_err = '#E63946' # Κόκκινο για το λάθος
    color_vol = '#1D3557' # Μπλε για το volatility
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (30d Avg)', color=color_err, fontsize=12)
    line1 = ax1.plot(df_clean.index, df_clean['AbsError'].rolling(30).mean(), color=color_err, linewidth=2, label='SVR Error')
    ax1.tick_params(axis='y', labelcolor=color_err)
    
    ax2 = ax1.twinx() # Δεύτερος άξονας Υ
    ax2.set_ylabel('Market Volatility (7d Std Dev, 30d Avg)', color=color_vol, fontsize=12)
    line2 = ax2.plot(df_clean.index, df_clean['Volatility_7d'].rolling(30).mean(), color=color_vol, linewidth=2, linestyle='--', label='Volatility')
    ax2.tick_params(axis='y', labelcolor=color_vol)
    
    # Θρύλος (Legend)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    
    plt.title("Timeline: Impact of High Volatility Periods on SVR Error", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "volatility_timeline_svr.png", dpi=300)
    
    print(f"✅ Ολοκληρώθηκε. Γραφήματα στο: {REPORT_DIR}")

if __name__ == "__main__":
    main()