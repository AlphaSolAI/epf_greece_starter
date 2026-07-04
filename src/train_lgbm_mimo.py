from __future__ import annotations

import argparse
import os
import pickle
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import joblib

from sklearn.multioutput import MultiOutputRegressor


@contextmanager
def _suppress_output(enabled: bool = True):
    """Suppress stdout/stderr (useful to silence repetitive LightGBM logs)."""
    if not enabled:
        yield
        return
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


def _load_processed(mode: str, task: str) -> pd.DataFrame:
    # Your repo's split_utils likely supports task; keep a safe fallback.
    from src.split_utils import load_processed

    try:
        return load_processed(mode, task=task)  # type: ignore
    except TypeError:
        # Older signature
        df = load_processed(mode)  # type: ignore
        return df


def _make_supervised_xy(df_train: pd.DataFrame, horizon: int,
                        dense_lags: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Create supervised (X, Y) for multi-step.

    Y is [y(t+1), ..., y(t+H)] for each row t.

    When dense_lags=True, adds y_lag4..y_lag23 (fills the intraday-lag gap)
    so each Direct sub-model can access the same-hour-yesterday value directly.
    """
    from src.split_utils import make_xy

    X, _ = make_xy(df_train)
    if "y" not in df_train.columns:
        raise ValueError("Expected target column 'y' in processed dataframe")

    y = df_train["y"].astype(float)

    # Add dense intraday lags if requested
    if dense_lags:
        # Find which y_lagN columns already exist
        existing_nums: set = set()
        for c in X.columns:
            if c.startswith("y_lag"):
                try:
                    existing_nums.add(int(c[len("y_lag"):]))
                except ValueError:
                    pass
        # Add missing lags 4-23 (covers "same-hour-yesterday" for h=2..11,14..23).
        # IMPORTANT: shift on the FULL y series first, then reindex to X.
        # This correctly handles y[t - k*1h] even for rows near the start of X.
        for lag in range(1, 24):
            if lag not in existing_nums:
                col_name = f"y_lag{lag}"
                X[col_name] = y.shift(lag).reindex(X.index)

    Y_df = pd.concat([y.shift(-k).rename(f"y_t+{k}") for k in range(1, horizon + 1)], axis=1)

    joined = X.join(Y_df, how="inner")

    # Keep only rows where ALL future targets are present
    joined = joined.dropna(subset=list(Y_df.columns))

    X_aligned = joined[X.columns]

    # Hard rule: use only numeric features (prevents Timestamp/object leakage into models)
    X_aligned = X_aligned.select_dtypes(include=[np.number]).copy()

    # Convert bool -> int (if any)
    for c in X_aligned.columns:
        if X_aligned[c].dtype == bool:
            X_aligned[c] = X_aligned[c].astype(np.int8)

    Y = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)

    return X_aligned, Y


def train_lgbm_direct(
    *,
    mode: str,
    task: str,
    horizon: int,
    train_end: str,
    device: str,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    subsample: float,
    colsample_bytree: float,
    random_state: int,
    quiet: bool,
    parallel: bool = True,
    dense_lags: bool = False,
):
    import lightgbm as lgb
    from src.split_utils import split_time_series

    df_all = _load_processed(mode, task)
    df_train, _ = split_time_series(df_all, mode=mode, train_end=train_end, test_size=168)

    X_train, Y_train = _make_supervised_xy(df_train, horizon=horizon, dense_lags=dense_lags)

    if parallel:
        # PARALLEL MODE (default, fast):
        # Each of the 168 horizon-models uses 1 CPU thread.
        # MultiOutputRegressor trains all 168 in parallel via joblib threading.
        # LightGBM releases the GIL -> true parallelism, no subprocess overhead.
        # Speed: ~8x faster than sequential on 8-core machine.
        device_type = "cpu"  # parallel joblib + GPU = context conflicts
        n_jobs_base = 1       # 1 thread per model; parallelism from MultiOutputRegressor
        n_jobs_outer = -1     # use all available cores
    else:
        # SEQUENTIAL GPU MODE: each model uses all CPU threads + GPU.
        device_type = "cpu"
        if device.lower() in {"gpu", "cuda"}:
            device_type = "gpu"
        elif device.lower() == "auto":
            device_type = "gpu"
        n_jobs_base = -1   # all threads per model
        n_jobs_outer = 1   # sequential

    base = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs_base,
        device_type=device_type,
        verbosity=-1,
    )

    model = MultiOutputRegressor(base, n_jobs=n_jobs_outer)

    dense_tag = " DENSE-LAGS" if dense_lags else ""
    mode_str = f"PARALLEL-CPU(n_jobs={n_jobs_outer})" if parallel else f"SEQUENTIAL(device={device_type})"
    print(f"🚀 TRAINING LightGBM DIRECT{dense_tag} (HOURLY | task={task.upper()}) [NO-LEAK]")
    print(f"   -> horizon={horizon} | n_estimators={n_estimators} | mode={mode_str}")
    print(f"   -> train={len(df_train)} -> usable={len(X_train)} | features={X_train.shape[1]}")
    print(f"   -> train_window={X_train.index.min()} -> {X_train.index.max()}")

    with _suppress_output(enabled=quiet):
        if parallel:
            # Threading backend: LightGBM releases GIL -> true parallel execution
            with joblib.parallel_backend("threading", n_jobs=n_jobs_outer):
                model.fit(X_train.to_numpy(dtype=np.float32), Y_train)
        else:
            model.fit(X_train.to_numpy(dtype=np.float32), Y_train)

    bundle = {
        "model": model,
        "feature_cols": list(X_train.columns),
        "horizon": int(horizon),
        "mode": mode,
        "task": task,
        "train_end": train_end,
        "strategy": "direct",
        "lib": "lightgbm",
        "dense_lags": dense_lags,
    }

    os.makedirs("models", exist_ok=True)
    dense_suffix = "_dense" if dense_lags else ""
    out_path = os.path.join("models", f"lgbm_direct_{mode}_{task}_h{horizon}{dense_suffix}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Saved LightGBM DIRECT{dense_tag} to: {os.path.abspath(out_path)}")


def train_xgb_direct(
    *,
    mode: str,
    task: str,
    horizon: int,
    train_end: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    random_state: int,
    dense_lags: bool = False,
):
    from xgboost import XGBRegressor
    from src.split_utils import split_time_series

    df_all = _load_processed(mode, task)
    df_train, _ = split_time_series(df_all, mode=mode, train_end=train_end, test_size=168)

    X_train, Y_train = _make_supervised_xy(df_train, horizon=horizon, dense_lags=dense_lags)

    base = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=1,       # 1 thread per model; outer parallelism from MultiOutputRegressor
        tree_method="hist",
        verbosity=0,
    )

    model = MultiOutputRegressor(base, n_jobs=-1)  # parallel across horizon steps

    dense_tag = " DENSE-LAGS" if dense_lags else ""
    print(f"🚀 TRAINING XGBoost DIRECT{dense_tag} (HOURLY | task={task.upper()}) [NO-LEAK]")
    print(f"   -> horizon={horizon} | n_estimators={n_estimators} | max_depth={max_depth}")
    print(f"   -> train={len(df_train)} -> usable={len(X_train)} | features={X_train.shape[1]}")
    print(f"   -> train_window={X_train.index.min()} -> {X_train.index.max()}")

    with joblib.parallel_backend("threading", n_jobs=-1):
        model.fit(X_train.to_numpy(dtype=np.float32), Y_train)

    bundle = {
        "model": model,
        "feature_cols": list(X_train.columns),
        "horizon": int(horizon),
        "mode": mode,
        "task": task,
        "train_end": train_end,
        "strategy": "direct",
        "lib": "xgboost",
        "dense_lags": dense_lags,
    }

    os.makedirs("models", exist_ok=True)
    dense_suffix = "_dense" if dense_lags else ""
    out_path = os.path.join("models", f"xgb_direct_{mode}_{task}_h{horizon}{dense_suffix}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Saved XGBoost DIRECT{dense_tag} to: {os.path.abspath(out_path)}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"], help="Only hourly is supported")
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb"],
                   help="Base learner: lgbm (default) or xgb")
    p.add_argument("--task", required=True, choices=["price", "load"], help="Target task")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--train_end", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu", "cuda"])
    p.add_argument("--n_estimators", type=int, default=1500)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--max_depth", type=int, default=6, help="XGB only: max tree depth")
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true", help="Silence repetitive LightGBM output")
    p.add_argument("--no-parallel", action="store_true", dest="no_parallel",
                   help="Disable parallel mode (use sequential GPU mode instead)")
    p.add_argument("--dense_lags", action="store_true",
                   help="Add dense intraday lags 1-23 (fills same-hour-yesterday gap for each Direct horizon)")
    return p


def main() -> int:
    args = _build_argparser().parse_args()

    if args.model == "xgb":
        train_xgb_direct(
            mode=args.mode,
            task=args.task,
            horizon=args.horizon,
            train_end=args.train_end,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.seed,
            dense_lags=args.dense_lags,
        )
    else:
        train_lgbm_direct(
            mode=args.mode,
            task=args.task,
            horizon=args.horizon,
            train_end=args.train_end,
            device=args.device,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            random_state=args.seed,
            quiet=args.quiet,
            parallel=not args.no_parallel,
            dense_lags=args.dense_lags,
        )

    return 0


if __name__ == "__main__":
    main()
