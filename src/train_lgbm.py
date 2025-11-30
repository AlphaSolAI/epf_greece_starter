import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import joblib
import sys
import io
import warnings

# Ρυθμίσεις
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lgbm_day.pkl"

def main():
    # 1. Φόρτωση
    if not DATA_PATH.exists():
        print("❌ Δεν βρέθηκε το daily.parquet")
        return
    
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    print(f"Dataset: {len(df)} ημέρες. Features: {X.shape[1]}")
    
    # 2. Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    
    print("-" * 50)
    print("Εκπαίδευση LightGBM (Cross Validation)...")
    
    # Θα κρατήσουμε το μοντέλο που θα εκπαιδευτεί σε ΟΛΑ τα δεδομένα στο τέλος
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        print(f"Fold {fold}: MAE = {mae:.3f}")

    print(f"\nΜέσος όρος CV MAE: {np.mean(maes):.3f}")
    
    # 3. Τελική Εκπαίδευση (σε όλο το dataset εκτός από τις τελευταίες 60 μέρες που είναι το 'κρυφό' test set)
    # Σημαντικό: Το 'eval_ml_models' θα το δοκιμάσει στις τελευταίες 60.
    # Άρα εμείς πρέπει να το εκπαιδεύσουμε σε ΟΛΑ τα προηγούμενα.
    
    test_size = 60
    X_final_train = X.iloc[:-test_size]
    y_final_train = y.iloc[:-test_size]
    
    print("..." * 10)
    print("Εκπαίδευση Τελικού Μοντέλου...")
    model.fit(X_final_train, y_final_train)
    
    # 4. Αποθήκευση
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Το μοντέλο αποθηκεύτηκε: {MODEL_PATH}")

if __name__ == "__main__":
    main()