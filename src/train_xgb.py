import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

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

    print(f"🚀 TRAINING XGBoost ({mode.upper()})...")
    print(f"   -> {len(df)} samples, {X.shape[1]} features")

    params = dict(
        objective="reg:squarederror",
        eval_metric="mae",
        learning_rate=0.05,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
    )

    if mode == "hourly":
        params["max_depth"] = 6
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.9

    model = xgb.XGBRegressor(
        n_estimators=1200 if mode == "daily" else 900,
        **params,
    )

    model.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"xgb_{mode}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved XGBoost to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    args = parser.parse_args()
    train(args.mode)


if __name__ == "__main__":
    main()
