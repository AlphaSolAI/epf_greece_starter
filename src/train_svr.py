import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

    print(f"📉 SVR Training ({mode}) on {len(df)} samples and {X.shape[1]} features...")

    if mode == "daily":
        params = dict(C=100, gamma=0.01, epsilon=2, kernel="rbf")
    else:  # hourly
        params = dict(C=50, gamma=0.01, epsilon=1, kernel="rbf")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(**params)),
        ]
    )

    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"svr_{mode}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()
