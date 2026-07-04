"""
generate_load_forecast.py

Generate day-ahead load forecasts for a given period and save to
data/processed/load_forecast_hourly.parquet (used as load_fc feature in price models).

Two modes:
  1. Train a fresh LGBM (default):
       --train_end  "2025-09-30 23:00"  --test_start ... --test_end ...

  2. Use a pre-trained model (best accuracy, recommended):
       --model_path models/lgbm_hourly_load_openloop_daily_ss_optuna.pkl
       --test_start "2025-10-01 00:00"  --test_end "2025-12-31 23:00"
       (no --train_end needed)

Usage examples:
    # Fresh LGBM with Sep30 cutoff:
    python -m src.generate_load_forecast hourly \
        --train_end  "2025-09-30 23:00" \
        --test_start "2025-10-01 00:00" \
        --test_end   "2025-12-31 23:00"

    # Best model (LGBM-Daily-SS-Optuna, Nov30 cutoff):
    python -m src.generate_load_forecast hourly \
        --model_path models/lgbm_hourly_load_openloop_daily_ss_optuna.pkl \
        --test_start "2025-10-01 00:00" \
        --test_end   "2025-12-31 23:00"
"""

from __future__ import annotations

import argparse
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data" / "processed"
OUTPUT_PARQUET = DATA_DIR / "load_forecast_hourly.parquet"

# ── alias for unpickling (just in case) ───────────────────────────────────────
class ResidualAddBaselineWrapper:
    def __init__(self, model, baseline_col, feature_names):
        self.model = model; self.baseline_col = baseline_col
        self.feature_names = list(feature_names)
        self.baseline_idx = self.feature_names.index(baseline_col)
    def predict(self, X):
        r = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        if isinstance(X, pd.DataFrame):
            base = X[self.baseline_col].to_numpy(dtype=float).reshape(-1)
        else:
            base = np.asarray(X)[:, self.baseline_idx].astype(float).reshape(-1)
        return base + r

def _register_aliases():
    for name in ["src.model_wrappers", "src.train_xgb_openloop", "src.eval_openloop"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], "ResidualAddBaselineWrapper", ResidualAddBaselineWrapper)


import joblib

from .split_utils import load_processed, make_xy, split_time_series
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop


def _fnp(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def mae(y_true, y_pred) -> float:
    yt, yp = _fnp(y_true), _fnp(y_pred)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yt[m] - yp[m]))) if m.sum() > 0 else float("nan")


def predict_chunked_24h(model, df_full, test_index, feature_cols, cfg):
    """Run day-by-day (24h) recursive predictions, resetting at each day boundary."""
    preds = np.zeros(len(test_index), dtype=float)
    ptr   = 0
    n     = len(test_index)
    while ptr < n:
        chunk_idx   = test_index[ptr: ptr + 24]
        chunk_preds = _fnp(recursive_predict_openloop(
            model=model, df_full=df_full,
            test_index=chunk_idx, feature_cols=feature_cols, config=cfg,
        ))
        k = len(chunk_preds)
        preds[ptr: ptr + k] = chunk_preds
        ptr += k
    return preds


def generate_load_forecast(
    mode: str,
    test_start: str,
    test_end: str,
    train_end: str = None,
    n_estimators: int = 800,
    num_leaves: int = 63,
    model_path: str = None,
):
    """
    Generate load forecasts for test_start..test_end.

    If model_path is provided: load the pre-trained model (no training).
    Otherwise: train a fresh LightGBM with train_end cutoff.
    """
    _register_aliases()

    print("=" * 70)
    print(f"  GENERATE LOAD FORECAST | mode={mode}")
    if model_path:
        print(f"  model_path = {model_path}")
        print(f"  test: {test_start} → {test_end}")
    else:
        print(f"  train → {train_end}   test: {test_start} → {test_end}")
        print(f"  LightGBM: n_est={n_estimators}, num_leaves={num_leaves}")
    print("=" * 70)

    # ── load features ─────────────────────────────────────────────────────────
    df = load_processed(mode, task="load")
    df_test = df.loc[test_start:test_end]
    _, y_test     = make_xy(df_test)
    test_index    = df_test.index

    n_days = len(test_index) // 24
    print(f"[INFO] test  rows={len(test_index):,} ({n_days} days)")

    # ── get/train model ───────────────────────────────────────────────────────
    if model_path:
        print(f"\n[LOAD] Pre-trained model: {model_path} ...")
        model = joblib.load(model_path)

        # Get feature cols from model (LightGBM stores feature_name_)
        if hasattr(model, "feature_name_"):
            feature_cols = list(model.feature_name_)
        else:
            # Fallback: derive from training data slice up to test_start
            df_any_train = df.loc[:test_start].iloc[:-1]
            X_any, _ = make_xy(df_any_train)
            feature_cols = list(X_any.columns)
        print(f"[INFO] features={len(feature_cols)}")

    else:
        if train_end is None:
            raise ValueError("Provide either --model_path or --train_end")

        import lightgbm as lgb

        df_train, _ = split_time_series(
            df, mode=mode, test_size=None, train_start=None,
            train_end=train_end, test_start=test_start, test_end=test_end,
        )
        X_train, y_train = make_xy(df_train)
        feature_cols = list(X_train.columns)

        print(f"[INFO] train rows={len(X_train):,} | features={len(feature_cols)}")

        print("\n[TRAIN] LightGBM load model ...")
        params = dict(
            objective="regression", metric="mae",
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=0.05, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, verbosity=-1, n_jobs=-1,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        tr_mae = mae(y_train, model.predict(X_train))
        print(f"   train MAE = {tr_mae:.2f} MW")

    # ── recursive day-by-day predictions ─────────────────────────────────────
    print(f"\n[PREDICT] Running {n_days} daily (24h) recursive chunks ...")
    cfg   = OpenLoopConfig(y_floor=None)
    preds = predict_chunked_24h(model, df, test_index, feature_cols, cfg)

    # evaluate against actual
    y_true = _fnp(y_test)
    test_mae = mae(y_true, preds)
    print(f"   test  MAE = {test_mae:.2f} MW  ({n_days} days OOS)")

    # ── save to parquet ───────────────────────────────────────────────────────
    fc_df = pd.DataFrame({"load_fc": preds}, index=test_index)
    fc_df.index.name = "datetime"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fc_df.to_parquet(OUTPUT_PARQUET)
    print(f"\n✅ Saved {len(fc_df)} rows → {OUTPUT_PARQUET}")
    print(f"   Period : {fc_df.index.min()}  →  {fc_df.index.max()}")
    print(f"   Test MAE: {test_mae:.2f} MW")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("mode", default="hourly", nargs="?")
    p.add_argument("--train_end",    default=None,
                   help="Train cutoff (required if --model_path not given)")
    p.add_argument("--test_start",   default="2025-10-01 00:00")
    p.add_argument("--test_end",     default="2025-12-31 23:00")
    p.add_argument("--n_estimators", type=int, default=800)
    p.add_argument("--num_leaves",   type=int, default=63)
    p.add_argument("--model_path",   default=None,
                   help="Path to pre-trained load model pkl (skips training)")
    return p.parse_args()


def main():
    args = _parse()
    generate_load_forecast(
        mode         = args.mode,
        train_end    = args.train_end,
        test_start   = args.test_start,
        test_end     = args.test_end,
        n_estimators = args.n_estimators,
        num_leaves   = args.num_leaves,
        model_path   = args.model_path,
    )


if __name__ == "__main__":
    main()
