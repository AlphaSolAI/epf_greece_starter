import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import sys
import io
import warnings
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_day.pkl"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

def main():
    if not DATA_PATH.exists(): return

    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    print(f"Dataset Features: {X.columns.tolist()}") # Εδώ θα δούμε αν υπάρχει το 'load'

    param_dist = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
    }

    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)

    print(f"🔍 Training XGBoost (Checking Feature Importance)...")
    
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)
    best_model = search.best_estimator_
    
    # Τελική εκπαίδευση (εκτός 60 ημερών test)
    test_size = 60
    best_model.fit(X.iloc[:-test_size], y.iloc[:-test_size])
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    
    # --- FEATURE IMPORTANCE CHECK ---
    importance = best_model.feature_importances_
    feats = X.columns
    
    # Δημιουργία DataFrame
    imp_df = pd.DataFrame({'Feature': feats, 'Importance': importance})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    print("\n📊 FEATURE IMPORTANCE (Top 5):")
    print(imp_df.head(5))
    
    if 'load' in imp_df['Feature'].values:
        load_rank = imp_df[imp_df['Feature']=='load'].index[0]
        load_imp = imp_df[imp_df['Feature']=='load']['Importance'].values[0]
        print(f"   -> Το 'load' είναι στη θέση {load_rank} με βαρύτητα {load_imp:.4f}")
    else:
        print("❌ ΤΟ ΜΟΝΤΕΛΟ ΑΓΝΟΗΣΕ ΤΟ LOAD!")

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['Feature'][:10], imp_df['Importance'][:10], color='#1D3557')
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "xgb_importance.png")

if __name__ == "__main__":
    main()