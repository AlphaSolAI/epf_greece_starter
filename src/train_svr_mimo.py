# src/train_rf_mimo.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def _find_processed_parquet(mode: str, task: str) -> Path:
    cand = [
        DATA_DIR / "processed" / f"{mode}_{task}.parquet",
        DATA_DIR / "processed" / f"{mode}-{task}.parquet",
        DATA_DIR / "processed" / f"{task}_{mode}.parquet",
        DATA_DIR / "processed" / f"{task}-{mode}.parquet",
        DATA_DIR / "processed" / f"{mode}.parquet",
        DATA_DIR / "processed" / f"{task}.parquet",
        DATA_DIR / "processed" / f"{mode}_{task}_processed.parquet",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Δεν βρήκα processed parquet. Περίμενα κάτι σαν:\n"
        + "\n".join(str(x) for x in cand)
    )


def _infer_target_col(df: pd.DataFrame, task: str) -> str:
    task = task.lower()
    if "y" in df.columns:
        return "y"
    if task == "load":
        for c in ["load", "load_mw", "load_mwh", "demand", "demand_mw", "target"]:
            if c in df.columns:
                return c
    if task == "price":
        for c in ["price", "dam_price", "price_eur_mwh", "target"]:
            if c in df.columns:
                return c
    raise KeyError(f"Δεν βρήκα target col για task={task}. Columns: {list(df.columns)[:50]}...")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ["timestamp", "datetime", "ds", "time", "date"]:
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c])
            out = out.set_index(c).sort_index()
            return out
    raise ValueError("Το dataframe δεν έχει DatetimeIndex ούτε γνωστή datetime στήλη (timestamp/datetime/ds/...).")


def _build_mimo_xy(df: pd.DataFrame, target_col: str, horizon: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    # features = όλα τα numeric εκτός target
    feat_df = df.drop(columns=[target_col], errors="ignore")
    feat_df = feat_df.select_dtypes(include=[np.number, "bool"]).copy()
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = df[target_col].astype(np.float32).to_numpy()
    n = len(df)
    usable = n - horizon
    if usable <= 0:
        raise ValueError(f"Λίγα rows ({n}) για horizon={horizon}")

    X = feat_df.iloc[:usable].copy()
    # Y[t] = [y[t], y[t+1], ..., y[t+h-1]]
    Y = np.stack([y[i : i + usable] for i in range(horizon)], axis=1).astype(np.float32)

    feature_cols = list(X.columns)
    return X, Y, feature_cols


def train_rf_mimo(
    mode: str,
    task: str,
    horizon: int,
    train_end: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_samples: float | None,
    n_jobs: int,
    random_state: int,
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    path = _find_processed_parquet(mode, task)
    df = pd.read_parquet(path)
    df = _ensure_datetime_index(df)

    target_col = _infer_target_col(df, task)

    train_end_ts = pd.to_datetime(train_end)
    df_tr = df.loc[:train_end_ts].copy()

    X, Y, feature_cols = _build_mimo_xy(df_tr, target_col, horizon)

    # Memory-safe defaults:
    # - float32 X
    X_np = X.to_numpy(dtype=np.float32, copy=False)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        n_jobs=n_jobs,
        random_state=random_state,
        bootstrap=True,
        max_features="sqrt",
        oob_score=False,
        verbose=0,
    )

    print(f"🚀 TRAINING RF MIMO SINGLE ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> horizon={horizon} | n_estimators={n_estimators} | n_jobs={n_jobs}")
    print(f"   -> train={len(df_tr)} -> usable={len(X)} | features={X_np.shape[1]}")
    print(f"   -> train_window={df_tr.index.min()} -> {df_tr.index.max()}")

    t0 = time.time()
    model.fit(X_np, Y)
    dt = time.time() - t0

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "mode": mode,
        "task": task,
        "horizon": horizon,
        "strategy": "mimo_single_rf",
        "fit_seconds": dt,
    }

    out = MODELS_DIR / f"rf_mimo_{mode}_{task}_single_h{horizon}.pkl"
    joblib.dump(bundle, out, compress=3)
    print(f"✅ Saved RF MIMO (SINGLE) to: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hourly", "daily"])
    ap.add_argument("--task", required=True, choices=["price", "load"])
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--n_estimators", type=int, default=900)
    ap.add_argument("--max_depth", type=int, default=16)  # για να ΜΗΝ σκάει μνήμη
    ap.add_argument("--min_samples_leaf", type=int, default=5)
    ap.add_argument("--max_samples", type=float, default=0.7)
    ap.add_argument("--n_jobs", type=int, default=1)      # κρατάει RAM χαμηλά
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    train_rf_mimo(
        mode=args.mode,
        task=args.task,
        horizon=args.horizon,
        train_end=args.train_end,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
