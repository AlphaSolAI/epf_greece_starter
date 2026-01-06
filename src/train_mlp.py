import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def train(mode: str) -> None:
    data_path = DATA_DIR / f"{mode}.parquet"
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["y"])
    y = df["y"].values

    print(f"🧠 MLP Training ({mode}) on {len(df)} samples and {X.shape[1]} features...")

    if mode == "daily":
        hidden = (64, 64)
        max_iter = 400
    else:  # hourly
        hidden = (128, 64)
        max_iter = 200

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=20,
        verbose=False,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", mlp),
        ]
    )

    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"mlp_{mode}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()

