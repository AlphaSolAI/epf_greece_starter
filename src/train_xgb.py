import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from xgboost import XGBRegressor

from .split_utils import load_processed, make_xy, split_time_series

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def _fit_xgb_gpu_first(model_kwargs: dict):
    """
    Try CUDA first (device='cuda' + tree_method='hist').
    If CUDA not available in xgboost build, fallback to CPU.
    Also handles older xgboost versions where 'device' is not accepted.
    """
    # 1) New-style (xgboost >= 2.0): device='cuda', tree_method='hist'
    try:
        m = XGBRegressor(**model_kwargs, tree_method="hist", device="cuda")
        return m, "cuda"
    except TypeError:
        # 'device' not supported -> try older gpu_hist style
        pass
    except Exception:
        # CUDA build missing or runtime error -> fallback later
        pass

    # 2) Older-style GPU: tree_method='gpu_hist', predictor='gpu_predictor'
    try:
        m = XGBRegressor(**model_kwargs, tree_method="gpu_hist", predictor="gpu_predictor")
        return m, "gpu_hist"
    except Exception:
        pass

    # 3) CPU fallback
    m = XGBRegressor(**model_kwargs, tree_method="hist")
    return m, "cpu"


def train(
    mode: str,
    task: str = "price",
    test_size: Optional[int] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> None:
    # daily dataset removed (align with your openloop script convention)
    if mode != "hourly":
        raise ValueError("Only mode='hourly' is supported (daily dataset removed).")

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

    model_kwargs = dict(
        objective="reg:squarederror",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        n_jobs=-1,
        random_state=42,
        eval_metric="mae",
    )

    model, device_used = _fit_xgb_gpu_first(model_kwargs)

    print(f"🚀 TRAINING XGBoost ({mode.upper()} | task={task.upper()}) [NO-LEAK] [{device_used.upper()}]")
    print(f"   -> train={len(df_train)} | test={len(df_test)} | features={X_train.shape[1]}")
    if isinstance(df_train.index, pd.DatetimeIndex):
        print(f"   -> train_range={df_train.index.min()} -> {df_train.index.max()}")

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"xgb_{mode}_{task}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved XGBoost to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="hourly", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--test_size", type=int, default=None, help="Rows in test (hourly=hours)")
    parser.add_argument("--train_start", type=str, default=None)
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
