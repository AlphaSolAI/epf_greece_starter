from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor

from src.split_utils import load_processed, make_xy, split_time_series

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def _ensure_numeric_X(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.select_dtypes(include=[np.number, bool]).copy()
    bool_cols = Xn.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        Xn[bool_cols] = Xn[bool_cols].astype(np.int8)
    Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return Xn


def _build_mimo_single_xy(df_train: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    X0, _ = make_xy(df_train)
    X0 = _ensure_numeric_X(X0)

    y = df_train["y"].astype(float)
    Y = np.column_stack([y.shift(-k).to_numpy() for k in range(1, horizon + 1)]).astype(np.float32)

    valid = np.isfinite(Y).all(axis=1)
    X = X0.loc[valid]
    Y = Y[valid]

    X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
    Y_np = np.ascontiguousarray(Y)
    return X_np, Y_np, list(X.columns)


def _parse_max_features(s: str):
    ss = str(s).strip().lower()
    if ss in {"sqrt", "log2"}:
        return ss
    if ss in {"none", "null"}:
        return None
    try:
        return int(ss)
    except Exception:
        pass
    try:
        return float(ss)
    except Exception:
        raise ValueError(f"Invalid --max_features={s}. Use float/int/'sqrt'/'log2'/None.")


def _effective_n_jobs(user_n_jobs: int, cap: int) -> int:
    cpu = os.cpu_count() or 1
    if user_n_jobs < 0:
        return max(1, min(cpu, int(cap)))
    return max(1, int(user_n_jobs))


def train_rf_mimo_single(
    *,
    task: str,
    horizon: int,
    train_start: Optional[str],
    train_end: str,
    n_estimators: int,
    max_depth: Optional[int],
    max_features,
    min_samples_leaf: int,
    min_samples_split: int,
    bootstrap: bool,
    n_jobs: int,
    n_jobs_cap: int,
    random_state: int,
    max_rows: Optional[int],
    backend: str,
    chunk_size: int,
) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)

    try:
        df = load_processed("hourly", task=task)  # type: ignore
    except TypeError:
        df = load_processed("hourly")

    df_train, _ = split_time_series(
        df,
        mode="hourly",
        train_start=train_start,
        train_end=train_end,
        test_size=168,   # dummy — only df_train is used
    )

    if max_rows is not None:
        mr = int(max_rows)
        if mr > 0 and len(df_train) > mr:
            df_train = df_train.iloc[-mr:].copy()

    X_train_np, Y_train_np, feature_cols = _build_mimo_single_xy(df_train, horizon=horizon)

    n_jobs_eff = _effective_n_jobs(int(n_jobs), int(n_jobs_cap))
    chunk = max(1, int(chunk_size))
    chunk = min(chunk, int(n_estimators))

    print(f"🚀 TRAINING RF MIMO (HOURLY | task={task.upper()}) [NO-LEAK]", flush=True)
    print(f"   -> horizon={horizon} | n_estimators={n_estimators} | backend={backend}", flush=True)
    print(f"   -> requested n_jobs={n_jobs} | n_jobs_cap={n_jobs_cap} | n_jobs_effective={n_jobs_eff}", flush=True)
    print(f"   -> warm_start chunk_size={chunk}", flush=True)
    print(f"   -> max_depth={max_depth} | max_features={max_features} | bootstrap={bootstrap}", flush=True)
    print(f"   -> min_samples_leaf={min_samples_leaf} | min_samples_split={min_samples_split}", flush=True)
    print(f"   -> train_rows={len(df_train)} -> usable={X_train_np.shape[0]} | features={X_train_np.shape[1]}", flush=True)
    if isinstance(df_train.index, pd.DatetimeIndex):
        print(f"   -> train_window={df_train.index.min()} -> {df_train.index.max()}", flush=True)

    rf = RandomForestRegressor(
        n_estimators=chunk,          # will grow with warm_start
        warm_start=True,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=int(min_samples_leaf),
        min_samples_split=int(min_samples_split),
        bootstrap=bool(bootstrap),
        n_jobs=int(n_jobs_eff),
        random_state=int(random_state),
        verbose=0,
    )

    def fit_loop(n_jobs_use: int, chunk_use: int):
        rf.set_params(n_jobs=int(n_jobs_use))
        # build gradually to reduce peak RAM
        with parallel_backend(backend):
            built = 0
            while built < int(n_estimators):
                built = min(int(n_estimators), built + int(chunk_use))
                rf.set_params(n_estimators=built)
                rf.fit(X_train_np, Y_train_np)
                print(f"   -> built_trees={built}/{n_estimators}", flush=True)

    try:
        fit_loop(n_jobs_eff, chunk)
    except MemoryError:
        # fallback that does NOT change RF hyperparams affecting accuracy, only parallelism/scheduling
        print("\n[WARN] MemoryError: retrying with n_jobs_effective=1 and smaller chunk (same model, slower fit).", flush=True)
        fit_loop(1, max(5, min(25, chunk)))

    bundle = {
        "model": rf,
        "feature_cols": feature_cols,
        "horizon": int(horizon),
        "mode": "hourly",
        "task": task,
        "train_start": str(pd.Timestamp(train_start)) if train_start else None,
        "train_end": str(pd.Timestamp(train_end)),
        "strategy": "mimo_single",
        "lib": "sklearn_rf",
        "n_jobs_effective": int(rf.n_jobs),
        "chunk_size": int(chunk),
    }

    out_path = MODELS_DIR / f"rf_mimo_single_hourly_{task}_h{horizon}.pkl"
    joblib.dump(bundle, out_path)
    print(f"✅ Saved RF MIMO to: {out_path}", flush=True)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"], help="Only hourly is supported")
    p.add_argument("--task", required=True, choices=["price", "load"])
    p.add_argument("--horizon", type=int, default=168)
    p.add_argument("--train_end", required=True)
    p.add_argument("--train_start", type=str, default=None)

    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--max_features", type=str, default="1.0")
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--bootstrap", action="store_true")

    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--n_jobs_cap", type=int, default=1, help="Caps parallel tree building when n_jobs=-1 (saves RAM).")
    p.add_argument("--chunk_size", type=int, default=50, help="Warm-start chunk size (reduces peak RAM).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--backend", type=str, default="threading", choices=["threading", "loky"])

    args = p.parse_args()

    train_rf_mimo_single(
        task=args.task,
        horizon=int(args.horizon),
        train_start=args.train_start,
        train_end=str(args.train_end),
        n_estimators=int(args.n_estimators),
        max_depth=args.max_depth,
        max_features=_parse_max_features(args.max_features),
        min_samples_leaf=int(args.min_samples_leaf),
        min_samples_split=int(args.min_samples_split),
        bootstrap=bool(args.bootstrap),
        n_jobs=int(args.n_jobs),
        n_jobs_cap=int(args.n_jobs_cap),
        random_state=int(args.seed),
        max_rows=args.max_rows,
        backend=str(args.backend),
        chunk_size=int(args.chunk_size),
    )


if __name__ == "__main__":
    main()
