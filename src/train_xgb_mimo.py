import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def _ensure_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost must see pure numeric arrays.
    - datetime64 / Timestamp columns -> int64 (ns)
    - bool -> int
    - object -> try datetime else numeric coercion
    - keep only numeric, fill NaN/inf
    """
    X = X.copy()

    for c in list(X.columns):
        s = X[c]

        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(np.int8)
            continue

        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            X[c] = s.view("int64")
            continue

        if s.dtype == object:
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() > 0 and dt.isna().sum() < len(dt):
                X[c] = dt.view("int64")
            else:
                X[c] = pd.to_numeric(s, errors="coerce")

    X = (
        X.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    return X


def _build_mimo_xy(df_train: pd.DataFrame, horizon: int,
                   dense_lags: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    # features at time t
    X0, _ = make_xy(df_train)

    y = df_train["y"].astype(float)

    # Add dense intraday lags if requested (BEFORE _ensure_numeric_features so
    # NaN values from shift are handled consistently by fillna(0.0) below)
    if dense_lags:
        existing_nums: set = set()
        for c in X0.columns:
            if c.startswith("y_lag"):
                try:
                    existing_nums.add(int(c[len("y_lag"):]))
                except ValueError:
                    pass
        for lag in range(1, 24):
            if lag not in existing_nums:
                col_name = f"y_lag{lag}"
                X0[col_name] = y.shift(lag).reindex(X0.index)

    X0 = _ensure_numeric_features(X0)

    # targets: [y(t+1), ..., y(t+h)]
    Y = np.column_stack([y.shift(-k).to_numpy() for k in range(1, horizon + 1)]).astype(np.float32)

    valid = np.isfinite(Y).all(axis=1)
    return X0.loc[valid], Y[valid]


def _resolve_device(device: str) -> str:
    device = (device or "auto").lower()
    if device in {"cuda", "gpu"}:
        return "cuda"
    if device == "cpu":
        return "cpu"
    return "auto"


def _make_xgb_regressor(n_estimators: int, use_cuda: bool, n_jobs: int = 0) -> XGBRegressor:
    """
    XGBoost new API: GPU works with device='cuda' and tree_method='hist'.
    n_jobs=0 means all threads (XGBoost convention, unlike sklearn's -1).
    """
    params = dict(
        n_estimators=int(n_estimators),
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=n_jobs,
    )
    if use_cuda:
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"
    return XGBRegressor(**params)


def train_xgb_mimo(
    mode: str,
    task: str,
    horizon: int,
    train_start: Optional[str],
    train_end: Optional[str],
    n_estimators: int,
    device: str,
    parallel: bool = True,
    dense_lags: bool = False,
) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(
        df,
        mode=mode,
        test_size=168,          # dummy — we only use df_train
        train_start=train_start,
        train_end=train_end,
        test_start=None,
        test_end=None,
    )

    X_train, Y_train = _build_mimo_xy(df_train, horizon=horizon, dense_lags=dense_lags)

    resolved = _resolve_device(device)

    # Strategy: if CUDA available and not in parallel mode, use CUDA sequential.
    # If parallel mode, use CPU threading (CUDA + parallel joblib = GPU conflicts).
    use_cuda = (resolved in {"cuda", "auto"}) and not parallel

    if parallel:
        mode_str = "PARALLEL-CPU(threading)"
    else:
        mode_str = f"SEQUENTIAL-{'CUDA' if use_cuda else 'CPU'}"

    dense_tag = " DENSE-LAGS" if dense_lags else ""
    print(f"🚀 TRAINING XGBoost MIMO{dense_tag} (HOURLY | task={task.upper()}) [NO-LEAK]")
    print(f"   -> horizon={horizon} | device={resolved} | n_estimators={n_estimators} | mode={mode_str}")
    print(f"   -> train={len(df_train)} -> usable={len(X_train)} | features={X_train.shape[1]}")
    print(f"   -> train_window={df_train.index.min()} -> {df_train.index.max()}")

    model = None
    chosen = None

    if parallel:
        # Parallel threading: n_jobs=1 per model, all models run simultaneously
        # XGBoost releases the GIL -> threading gives true parallelism
        base = _make_xgb_regressor(n_estimators=n_estimators, use_cuda=False, n_jobs=1)
        model = MultiOutputRegressor(base, n_jobs=-1)
        with joblib.parallel_backend("threading", n_jobs=-1):
            model.fit(X_train.to_numpy(dtype=np.float32), Y_train)
        chosen = "cpu-parallel"
    else:
        # Sequential: try CUDA first, fallback to CPU
        def fit_with(use_c: bool):
            base = _make_xgb_regressor(n_estimators=n_estimators, use_cuda=use_c, n_jobs=0)
            m = MultiOutputRegressor(base, n_jobs=1)
            m.fit(X_train.to_numpy(dtype=np.float32), Y_train)
            return m, ("cuda" if use_c else "cpu")

        if resolved == "cpu":
            model, chosen = fit_with(False)
        elif resolved == "cuda":
            model, chosen = fit_with(True)
        else:  # auto: try cuda, fallback cpu
            try:
                model, chosen = fit_with(True)
            except Exception as e:
                print(f"[WARN] CUDA failed -> CPU. ({type(e).__name__}: {e})")
                model, chosen = fit_with(False)

    dense_suffix = "_dense" if dense_lags else ""
    out_path = MODELS_DIR / f"xgb_{mode}_{task}_mimo_h{horizon}{dense_suffix}.pkl"
    joblib.dump(
        {
            "kind": "xgb_mimo",
            "mode": mode,
            "task": task,
            "horizon": int(horizon),
            "feature_cols": list(X_train.columns),
            "device_used": chosen,
            "model": model,
            "dense_lags": dense_lags,
        },
        out_path,
    )
    print(f"Saved XGBoost MIMO{dense_tag} to: {out_path} (device_used={chosen})")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")

    p.add_argument("--horizon", type=int, default=168)
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)

    p.add_argument("--n_estimators", type=int, default=1500)

    p.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "auto"])
    p.add_argument("--no-parallel", action="store_true", dest="no_parallel",
                   help="Disable parallel mode (use sequential CUDA/CPU mode instead)")
    p.add_argument("--dense_lags", action="store_true",
                   help="Add dense intraday lags 1-23 (fills same-hour-yesterday gap)")

    args = p.parse_args()

    train_xgb_mimo(
        mode=args.mode,
        task=args.task,
        horizon=args.horizon,
        train_start=args.train_start,
        train_end=args.train_end,
        n_estimators=args.n_estimators,
        device=args.device,
        parallel=not args.no_parallel,
        dense_lags=args.dense_lags,
    )


if __name__ == "__main__":
    main()
