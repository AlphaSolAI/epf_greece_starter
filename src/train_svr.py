import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

DEFAULT_SVR_TRAIN_START = "2022-01-01 00:00:00"


def train(
    mode: str,
    task: str = "price",
    test_size: Optional[int] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> None:
    # SVR rule: default train_start from 2022-01-01 unless user overrides
    if train_start is None:
        train_start = DEFAULT_SVR_TRAIN_START

    df = load_processed(mode, task=task)
    df_train, df_test = split_time_series(
        df,
        mode=mode,
        test_size=test_size,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    X_train, y_train = make_xy(df_train)

    print(f"📉 TRAINING SVR ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> train={len(df_train)} | test={len(df_test)} | features={X_train.shape[1]}")
    print(f"   -> train_start(default_if_none)={train_start}")

    # HOURLY only (daily removed)
    params = dict(C=50, gamma=0.01, epsilon=1, kernel="rbf")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(**params)),
        ]
    )

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"svr_{mode}_{task}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved SVR to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--test_size", type=int, default=None, help="Rows in test (hours)")
    parser.add_argument("--train_start", type=str, default=None, help=f"Default: {DEFAULT_SVR_TRAIN_START}")
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)
    args = parser.parse_args()

    train(
        args.mode,
        task=args.task,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
