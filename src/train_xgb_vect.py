# -*- coding: utf-8 -*-
"""
Train XGBoost VECTOR (single model outputs the whole horizon).

Important:
- XGBoost vector-output / multi-target is NOT supported on GPU (as of xgboost 3.x).
  If you request --device cuda/auto, we will TRAIN the vector model on CPU automatically.

Saved as a single .pkl payload with meta + feature columns + estimator.
"""
from __future__ import annotations

import argparse
import inspect
import subprocess
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

MODELS_DIR = Path("models")


def _has_nvidia() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return r.returncode == 0
    except Exception:
        return False


def _call_load_processed(mode: str, task: str) -> pd.DataFrame:
    from . import split_utils
    fn = split_utils.load_processed
    sig = inspect.signature(fn)
    if "task" in sig.parameters:
        return fn(mode, task=task)
    return fn(mode)


def _infer_target_col(df: pd.DataFrame, task: str) -> str:
    if "y" in df.columns:
        return "y"
    task = task.lower()
    candidates = [c for c in df.columns if task in c.lower()]
    if candidates:
        return candidates[0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RuntimeError("Could not infer target column.")
    return num_cols[0]


def _prepare_features(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, List[str]]:
    X = df.drop(columns=[y_col], errors="ignore").copy()
    X = X.select_dtypes(include=["number", "bool"]).copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X, list(X.columns)


def _build_vect_dataset(df: pd.DataFrame, y_col: str, horizon: int):
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ds" in df.columns:
            df = df.set_index("ds")
        else:
            raise RuntimeError("Expected DatetimeIndex (or a 'ds' datetime column).")

    X, feature_cols = _prepare_features(df, y_col)
    y = df[y_col].astype(np.float32)
    Y = np.column_stack([y.shift(-k).to_numpy() for k in range(1, horizon + 1)])
    valid = ~np.isnan(Y).any(axis=1)
    return X.iloc[valid], Y[valid], df.index[valid], feature_cols


def train_xgb_vect(
    mode: str,
    task: str,
    horizon: int,
    train_end: str,
    n_estimators: int,
    device: str,
    seed: int = 42,
) -> Path:
    df = _call_load_processed(mode, task)
    y_col = _infer_target_col(df, task)

    train_end_ts = pd.Timestamp(train_end)
    df_train = df.loc[:train_end_ts].copy()

    X_train, Y_train, idx_train, feature_cols = _build_vect_dataset(df_train, y_col, horizon)

    if device == "auto":
        device = "cuda" if _has_nvidia() else "cpu"

    # FORCE CPU for vector-output
    if device == "cuda":
        print("⚠️  XGBoost vector-output (multi-target) is NOT supported on GPU.")
        print("   -> Switching to CPU for the VECTOR model.")
    xgb_device = "cpu"

    base = XGBRegressor()
    supports_multi = "multi_strategy" in base.get_params()

    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        device=xgb_device,
        verbosity=0,
    )
    if supports_multi:
        params["multi_strategy"] = "multi_output_tree"

    print(f"🚀 TRAINING XGBoost VECT ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> horizon={horizon} | device={xgb_device} | n_estimators={n_estimators}")
    print(f"   -> train={len(X_train)} | features={len(feature_cols)}")
    print(f"   -> train_window={idx_train.min()} -> {idx_train.max()}")
    if not supports_multi:
        print("⚠️  multi_strategy not available -> fallback to MultiOutputRegressor (H separate XGB models).")

    if supports_multi:
        model = XGBRegressor(**params)
        model.fit(X_train, Y_train)
        strategy = "vect"
    else:
        from sklearn.multioutput import MultiOutputRegressor
        base_est = XGBRegressor(**params)
        base_est.set_params(**{k: v for k, v in base_est.get_params().items() if k != "multi_strategy"})
        model = MultiOutputRegressor(base_est, n_jobs=-1)
        model.fit(X_train, Y_train)
        strategy = "multioutput_fallback"

    payload = {
        "meta": {
            "model": "xgb",
            "strategy": strategy,
            "mode": mode,
            "task": task,
            "horizon": horizon,
            "train_end": str(train_end_ts),
            "xgb_device": xgb_device,
            "supports_multi_strategy": bool(supports_multi),
        },
        "feature_cols": feature_cols,
        "model": model,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"xgb_vect_{mode}_{task}_h{horizon}.pkl"
    joblib.dump(payload, out, compress=3)

    print(f"✅ Saved: {out.resolve()}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", required=True, choices=["price", "load"])
    p.add_argument("--horizon", type=int, default=168)
    p.add_argument("--train_end", required=True)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_xgb_vect(
        mode=args.mode,
        task=args.task,
        horizon=args.horizon,
        train_end=args.train_end,
        n_estimators=args.n_estimators,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
