import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from .split_utils import load_processed, make_xy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def select_train_window(
    df,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
):
    if train_start:
        df = df.loc[pd.to_datetime(train_start) :]
    if train_end:
        df = df.loc[: pd.to_datetime(train_end)]
    return df


def train(
    mode: str,
    task: str = "price",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> None:
    df = load_processed(mode, task=task)
    df_train = select_train_window(df, train_start=train_start, train_end=train_end)

    X_train, y_train = make_xy(df_train)

    print(f"💡 TRAINING LightGBM OPENLOOP ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> train={len(df_train)} | features={X_train.shape[1]}")

    model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=128,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"lgbm_{mode}_{task}_openloop.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved LightGBM OPENLOOP to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    args = parser.parse_args()

    train(
        args.mode,
        task=args.task,
        train_start=args.train_start,
        train_end=args.train_end,
    )


if __name__ == "__main__":
    main()
