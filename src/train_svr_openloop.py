import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .split_utils import load_processed, make_xy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

DEFAULT_SVR_TRAIN_START = "2022-01-01 00:00:00"


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
    # SVR rule: default train_start from 2022-01-01 unless user overrides
    if train_start is None:
        train_start = DEFAULT_SVR_TRAIN_START

    df = load_processed(mode, task=task)
    df_train = select_train_window(df, train_start=train_start, train_end=train_end)

    X_train, y_train = make_xy(df_train)

    print(f"🧷 TRAINING SVR OPENLOOP ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> train={len(df_train)} | features={X_train.shape[1]}")
    print(f"   -> train_start(default_if_none)={train_start}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svr", SVR(C=20.0, epsilon=0.1, gamma="scale", kernel="rbf")),
        ]
    )

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"svr_{mode}_{task}_openloop.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved SVR OPENLOOP to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--train_start", type=str, default=None, help=f"Default: {DEFAULT_SVR_TRAIN_START}")
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
