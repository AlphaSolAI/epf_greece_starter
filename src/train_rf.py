import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
import io
import warnings

# Ρυθμίσεις
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rf_day.pkl"

def main():
    if not DATA_PATH.exists():
        print("❌ Δεν βρέθηκε το daily.parquet")
        return

    # 1. Φόρτωση
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    # 2. Ορισμός του Grid για αναζήτηση (Πιο επιθετικό)
    param_dist = {
        'n_estimators': [200, 500, 800],
        'max_depth': [None, 10, 20, 30],        # None σημαίνει να μεγαλώσει όσο θέλει
        'min_samples_split': [2, 5, 10],        # Μικρότερο = πιο πολύπλοκο δέντρο
        'min_samples_leaf': [1, 2, 4],          # Μικρότερο = πιο ευαίσθητο
        'max_features': ['sqrt', 'log2', None]  # None = βλέπει όλα τα features
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)

    print(f"🔍 Ξεκινάει το Optimization του Random Forest...")
    print("   (Αυτό είναι βαρύ μοντέλο, θα πάρει ώρα...)")

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=15,              # 15 Δοκιμές
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Εκπαίδευση (Search)
    search.fit(X, y)

    print("\n✅ Βρέθηκαν οι καλύτερες παράμετροι RF:")
    print(search.best_params_)
    
    # 3. Τελική Εκπαίδευση και Αποθήκευση
    best_model = search.best_estimator_
    
    # Εκπαίδευση σε όλο το train set (μείον τις 60 μέρες test)
    test_size = 60
    best_model.fit(X.iloc[:-test_size], y.iloc[:-test_size])
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Το βέλτιστο RF αποθηκεύτηκε στο {MODEL_PATH}")

if __name__ == "__main__":
    main()