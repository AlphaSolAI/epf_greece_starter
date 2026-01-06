import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

    print(f"🌲 TRAINING Random Forest ({mode.upper()})...")
    print(f"   -> {len(df)} samples, {X.shape[1]} features")

    if mode == "daily":
        model = RandomForestRegressor(
            n_estimators=1200,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    else:  # hourly
        model = RandomForestRegressor(
            n_estimators=900,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"rf_{mode}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved Random Forest to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()
