import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

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

    print(f"💡 TRAINING LightGBM ({mode.upper()})...")
    print(f"   -> {len(df)} samples, {X.shape[1]} features")

    params = dict(
        objective="regression",
        metric="mae",
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=20,
        verbose=-1,
        seed=42,
    )

    if mode == "hourly":
        params["num_leaves"] = 96
        params["min_data_in_leaf"] = 40

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=800 if mode == "daily" else 600,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"lgbm_{mode}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved LightGBM to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()
