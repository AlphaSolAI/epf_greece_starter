import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
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
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mlp_day.pkl"

def main():
    if not DATA_PATH.exists():
        print("❌ Δεν βρέθηκε το daily.parquet")
        return

    # 1. Φόρτωση
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=["y"])
    y = df["y"]
    
    print(f"🧠 Dataset MLP (Neural Net): {len(df)} ημέρες.")

    # 2. Pipeline (Scaling + Neural Network)
    # Τα νευρωνικά "σπάνε" αν δεν κάνεις scaling (StandardScaler)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(max_iter=1000, early_stopping=True, random_state=42))
    ])

    # 3. Παράμετροι για Optimization
    param_dist = {
        # (50,) = 1 κρυφό επίπεδο
        # (100, 50) = 2 κρυφά επίπεδα (Deep Learning)
        'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50)], 
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001, 0.01],        # Regularization
        'mlp__learning_rate_init': [0.001, 0.01]    # Πόσο γρήγορα μαθαίνει
    }

    tscv = TimeSeriesSplit(n_splits=3)

    print(f"🔍 Ξεκινάει το Optimization του MLP Neural Network...")
    print("   (Δοκιμάζει αρχιτεκτονικές... υπομονή)")
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Εκπαίδευση
    search.fit(X, y)

    print("\n✅ Βέλτιστες παράμετροι MLP:")
    print(search.best_params_)
    print(f"   Best CV Score (MAE): {-search.best_score_:.3f}")
    
    # 4. Τελική Εκπαίδευση (στο Train set)
    best_model = search.best_estimator_
    test_size = 60
    best_model.fit(X.iloc[:-test_size], y.iloc[:-test_size])
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Το μοντέλο MLP αποθηκεύτηκε στο {MODEL_PATH}")

if __name__ == "__main__":
    main()