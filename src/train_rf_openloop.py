import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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

    print(f"🌲 TRAINING RandomForest OPENLOOP ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> train={len(df_train)} | features={X_train.shape[1]}")

    model = RandomForestRegressor(
        n_estimators=900 if mode == "hourly" else 1200,
        max_depth=16 if mode == "hourly" else 18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"rf_{mode}_{task}_openloop.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved RandomForest OPENLOOP to: {model_path}")


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
