import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import sys
import io
import warnings

# Ρυθμίσεις
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svr_day.pkl"

def main():
    if not DATA_PATH.exists():
        print("❌ Δεν βρέθηκε το daily.parquet")
        return

    # 1. Φόρτωση
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    print(f"Dataset SVR: {len(df)} ημέρες. Features: {X.shape[1]}")

    # 2. Pipeline (Scaling + Model)
    # Το SVR είναι ευαίσθητο στις κλίμακες, θέλει οπωσδήποτε Scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # 3. Hyperparameter Grid
    # C: Πόσο αυστηρό είναι το μοντέλο (Μεγάλο C = κίνδυνος overfitting, Μικρό C = underfitting)
    # epsilon: Το πλάτος του "σωλήνα" ανοχής λάθους
    # gamma: Πόσο μακριά φτάνει η επιρροή κάθε δείγματος
    param_dist = {
        'svr__kernel': ['rbf'], # Το rbf είναι το στάνταρ για χρονοσειρές
        'svr__C': [10, 50, 100, 500, 1000], 
        'svr__epsilon': [0.1, 1, 2, 5],
        'svr__gamma': ['scale', 'auto', 0.01, 0.1]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    print(f"🔍 Ξεκινάει το Optimization του SVR...")
    print("   (Δοκιμάζουμε 15 συνδυασμούς - θα πάρει λίγο χρόνο...)")
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=15,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    print("\n✅ Βέλτιστες παράμετροι SVR:")
    print(search.best_params_)
    print(f"   Best CV Score (MAE): {-search.best_score_:.3f}")
    
    # 4. Τελική Εκπαίδευση
    best_model = search.best_estimator_
    test_size = 60
    best_model.fit(X.iloc[:-test_size], y.iloc[:-test_size])
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Το μοντέλο SVR αποθηκεύτηκε στο {MODEL_PATH}")

if __name__ == "__main__":
    main()