"""
train_mimo_with_load.py  (v2 — all algorithms + load task)

Two-stage MIMO price forecasting (H=24):
  Stage 1  — Load MIMO H=24 (must be pre-trained):
               X_load[t] → load_model → load_preds[t+1..t+24]
  Stage 2  — Price MIMO H=24 with augmented features:
               X_price[t] ++ load_preds[t+1..t+24] → price_model → preds

Direct MIMO for load task (no two-stage):
  X_load[t] → model → load_preds[t+1..t+24]

Supported algorithms: lgbm, xgb, rf, svr, mlp

For rf/svr/mlp price: always uses XGB load model (fallback: LGBM).
SVR uses subsampled data (--max_svr_rows, default 20000) for tractability.
MLP (sklearn) uses StandardScaler; SVR also uses StandardScaler.

Output files (price / two-stage):
  lgbm_direct_hourly_price_h24_with_load.pkl
  xgb_hourly_price_h24_with_load.pkl
  rf_hourly_price_mimo_h24_with_load.pkl
  svr_hourly_price_mimo_h24_with_load.pkl
  mlp_hourly_price_mimo_h24_with_load.pkl

Output files (load / direct):
  rf_hourly_load_mimo_h24.pkl
  svr_hourly_load_mimo_h24.pkl
  mlp_hourly_load_mimo_h24.pkl

Usage:
  # --- PRICE (two-stage) ---
  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo lgbm --task price --train_end "2025-11-30 23:00" --n_estimators 1000 --quiet

  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo rf --task price --train_end "2025-11-30 23:00"

  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo svr --task price --train_end "2025-11-30 23:00"

  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo mlp --task price --train_end "2025-11-30 23:00"

  # --- LOAD (direct MIMO) ---
  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo rf --task load --train_end "2025-11-30 23:00"

  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo svr --task load --train_end "2025-11-30 23:00"

  conda run -n epf --no-capture-output python -m src.train_mimo_with_load hourly \\
      --algo mlp --task load --train_end "2025-11-30 23:00"
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HORIZON = 24  # fixed to day-ahead (24h)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _suppress_output(enabled: bool = True):
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


def _make_mimo_xy(df_train: pd.DataFrame, horizon: int,
                  dense_lags: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create supervised (X, Y) for MIMO prediction.
    Y: [y(t+1), ..., y(t+H)] for each row t.
    When dense_lags=True, adds y_lag4..y_lag23 (fills intraday-lag gap).
    """
    from src.split_utils import make_xy

    X, _ = make_xy(df_train)
    if "y" not in df_train.columns:
        raise ValueError("Expected 'y' column in processed dataframe.")

    y = df_train["y"].astype(float)

    # Add dense intraday lags if requested
    if dense_lags:
        existing_nums: set = set()
        for c in X.columns:
            if c.startswith("y_lag"):
                try:
                    existing_nums.add(int(c[len("y_lag"):]))
                except ValueError:
                    pass
        for lag in range(1, 24):
            if lag not in existing_nums:
                X[f"y_lag{lag}"] = y.shift(lag).reindex(X.index)

    Y_df = pd.concat(
        [y.shift(-k).rename(f"y_t+{k}") for k in range(1, horizon + 1)], axis=1
    )
    joined = X.join(Y_df, how="inner").dropna(subset=list(Y_df.columns))

    X_aligned = joined[X.columns].select_dtypes(include=[np.number]).copy()
    for c in X_aligned.columns:
        if X_aligned[c].dtype == bool:
            X_aligned[c] = X_aligned[c].astype(np.int8)
    # Fill any residual NaN (rare missing features at series start) so MLP/SVR work.
    X_aligned = X_aligned.fillna(0.0)

    Y = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_aligned, Y


def _load_load_model(algo: str, mode: str, H: int) -> Tuple[object, List[str]]:
    """Load the pre-trained load MIMO H=24 model.
    For rf/svr/mlp: prefer XGB load model (user requested), fallback LGBM.
    Returns (load_model_object, load_feature_cols).
    """
    candidates_xgb = [
        f"xgb_{mode}_load_mimo_h{H}.pkl",
        f"xgb_vect_{mode}_load_h{H}_optuna.pkl",
        f"xgb_vect_{mode}_load_h{H}.pkl",
    ]
    candidates_lgbm = [
        f"lgbm_direct_{mode}_load_h{H}.pkl",
        f"lgbm_direct_{mode}_load_h{H}_optuna.pkl",
    ]

    # For rf/svr/mlp: use XGB load model (as requested), fallback LGBM
    if algo in {"lgbm"}:
        ordered = candidates_lgbm + candidates_xgb
    elif algo in {"xgb"}:
        ordered = candidates_xgb + candidates_lgbm
    else:
        # rf, svr, mlp → prefer XGB
        ordered = candidates_xgb + candidates_lgbm

    for fname in ordered:
        p = MODELS_DIR / fname
        if p.exists():
            print(f"   [load model] Using: {fname}", flush=True)
            bundle = joblib.load(p)
            if not isinstance(bundle, dict):
                raise ValueError(f"Expected dict bundle in {p}")
            model = bundle.get("model")
            fcols = bundle.get("feature_cols")
            if model is None or fcols is None:
                raise ValueError(f"Bundle in {p} missing 'model' or 'feature_cols'")
            return model, list(fcols)

    tried = ", ".join(ordered)
    raise FileNotFoundError(
        f"No load MIMO H={H} model found for algo={algo}. Tried: {tried}\n"
        f"Train first: python -m src.train_lgbm_mimo hourly --task load --horizon {H}"
    )


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _build_lgbm_model(n_estimators: int, learning_rate: float, num_leaves: int) -> MultiOutputRegressor:
    import lightgbm as lgb
    base = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def _build_xgb_model(n_estimators: int, learning_rate: float) -> MultiOutputRegressor:
    from xgboost import XGBRegressor
    base = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def _build_rf_model(n_estimators: int, max_depth: Optional[int],
                    min_samples_leaf: int) -> object:
    """RandomForest natively supports multi-output — no wrapper needed."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=0.8,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def _build_svr_model() -> MultiOutputRegressor:
    """SVR needs MultiOutputRegressor (no native multi-output)."""
    from sklearn.svm import SVR
    base = SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1)
    return MultiOutputRegressor(base, n_jobs=-1)


def _build_mlp_model(
    hidden_layer_sizes: Tuple[int, ...],
    max_iter: int,
    lr_init: float = 5e-4,
    alpha: float = 1e-3,
    n_iter_no_change: int = 30,
) -> object:
    """sklearn MLPRegressor natively supports multi-output.

    Improvements over the original v1:
    - lr_init lowered 1e-3 → 5e-4  (more stable adam steps)
    - alpha increased 1e-4 → 1e-3  (moderate L2, larger nets overfit)
    - n_iter_no_change 15 → 30     (more patience before early-stop)
    - tol tightened 1e-4 → 1e-5   (finer convergence criterion)
    """
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=n_iter_no_change,
        tol=1e-5,
        random_state=42,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# PRICE task: two-stage MIMO
# ---------------------------------------------------------------------------

def train_mimo_price(
    *,
    mode: str,
    algo: str,
    train_end: str,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
    mlp_hidden: Tuple[int, ...],
    mlp_max_iter: int,
    mlp_lr_init: float = 5e-4,
    mlp_alpha: float = 1e-3,
    mlp_n_iter_no_change: int = 30,
    max_svr_rows: int,
    quiet: bool,
    dense_lags: bool = False,
) -> Path:
    from src.split_utils import load_processed, make_xy, split_time_series

    H = HORIZON
    dense_tag = " DENSE-LAGS" if dense_lags else ""

    print(f"\n{'='*65}", flush=True)
    print(f"  Two-Stage MIMO Price H={H} | algo={algo.upper()}{dense_tag}", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"\n[1/4] Loading pre-trained load MIMO H={H} model ...", flush=True)

    load_model, load_feature_cols = _load_load_model(algo, mode, H)

    print(f"[2/4] Loading load dataset for feature extraction ...", flush=True)
    df_load = load_processed(mode, task="load")
    df_load_train, _ = split_time_series(
        df_load, mode=mode, train_end=train_end, test_size=168
    )
    X_load_full, _ = make_xy(df_load_train)
    print(f"   -> load features: {X_load_full.shape[1]} | rows: {len(X_load_full)}", flush=True)

    print(f"[3/4] Loading price dataset and building MIMO XY (H={H}) ...", flush=True)
    df_price = load_processed(mode, task="price")
    df_price_train, _ = split_time_series(
        df_price, mode=mode, train_end=train_end, test_size=168
    )
    X_price_train, Y_price_train = _make_mimo_xy(df_price_train, horizon=H, dense_lags=dense_lags)
    price_feature_cols_base = list(X_price_train.columns)
    print(f"   -> price features: {X_price_train.shape[1]} | usable rows: {len(X_price_train)}", flush=True)

    print(f"[4/4] Generating load predictions for price training features ...", flush=True)
    X_load_at_price = X_load_full.reindex(X_price_train.index)
    n_price = len(X_price_train)
    n_lf = len(load_feature_cols)
    X_for_load_pred = np.zeros((n_price, n_lf), dtype=np.float32)
    for i, col in enumerate(load_feature_cols):
        if col in X_load_at_price.columns:
            vals = X_load_at_price[col].fillna(0).to_numpy(dtype=np.float32)
            X_for_load_pred[:, i] = vals

    with _suppress_output(enabled=quiet):
        load_preds_train = load_model.predict(X_for_load_pred)  # (n_price, H)
    print(f"   -> load predictions shape: {load_preds_train.shape}", flush=True)

    load_pred_cols = [f"load_pred_h{i+1:02d}" for i in range(H)]
    load_pred_df = pd.DataFrame(load_preds_train, index=X_price_train.index, columns=load_pred_cols)
    X_price_aug = pd.concat([X_price_train, load_pred_df], axis=1)
    all_feature_cols = list(X_price_aug.columns)

    print(
        f"   -> augmented features: {len(price_feature_cols_base)} base + {H} load_pred"
        f" = {len(all_feature_cols)} total", flush=True,
    )

    X_aug_np = X_price_aug.to_numpy(dtype=np.float32)
    scaler = None

    # ── Subsample for SVR
    X_fit, Y_fit = X_aug_np, Y_price_train
    if algo == "svr" and max_svr_rows > 0 and len(X_fit) > max_svr_rows:
        print(f"   [SVR] Subsampling to last {max_svr_rows} rows (from {len(X_fit)})", flush=True)
        X_fit = X_fit[-max_svr_rows:]
        Y_fit = Y_fit[-max_svr_rows:]

    # ── Scale for SVR and MLP
    if algo in {"svr", "mlp"}:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit).astype(np.float32)
        print(f"   [scale] StandardScaler fitted on {len(X_fit)} rows", flush=True)

    # ── Train
    print(f"\n🚀 Training {algo.upper()} PRICE MIMO with load features (H={H}) ...", flush=True)

    if algo == "lgbm":
        model = _build_lgbm_model(n_estimators, learning_rate, num_leaves)
        with _suppress_output(enabled=quiet):
            with joblib.parallel_backend("threading", n_jobs=-1):
                model.fit(X_fit, Y_fit)

    elif algo == "xgb":
        model = _build_xgb_model(n_estimators, learning_rate)
        with _suppress_output(enabled=quiet):
            with joblib.parallel_backend("threading", n_jobs=-1):
                model.fit(X_fit, Y_fit)

    elif algo == "rf":
        model = _build_rf_model(n_estimators, max_depth, min_samples_leaf)
        with _suppress_output(enabled=quiet):
            model.fit(X_fit, Y_fit)
        print(f"   -> RF trained: {model.n_estimators} trees, {len(X_fit)} rows", flush=True)

    elif algo == "svr":
        model = _build_svr_model()
        with _suppress_output(enabled=quiet):
            model.fit(X_fit, Y_fit)
        print(f"   -> SVR MultiOutputRegressor trained", flush=True)

    elif algo == "mlp":
        model = _build_mlp_model(mlp_hidden, mlp_max_iter, lr_init=mlp_lr_init, alpha=mlp_alpha, n_iter_no_change=mlp_n_iter_no_change)
        with _suppress_output(enabled=quiet):
            model.fit(X_fit, Y_fit)
        iters = getattr(model, "n_iter_", "?")
        print(f"   -> MLP trained: {iters} iterations, {len(X_fit)} rows", flush=True)

    else:
        raise ValueError(f"Unsupported algo: {algo}")

    # ── Save
    MODELS_DIR.mkdir(exist_ok=True)
    ds = "_dense" if dense_lags else ""
    out_name = {
        "lgbm": f"lgbm_direct_{mode}_price_h{H}_with_load{ds}.pkl",
        "xgb":  f"xgb_{mode}_price_h{H}_with_load{ds}.pkl",
        "rf":   f"rf_{mode}_price_mimo_h{H}_with_load{ds}.pkl",
        "svr":  f"svr_{mode}_price_mimo_h{H}_with_load{ds}.pkl",
        "mlp":  f"mlp_{mode}_price_mimo_h{H}_with_load{ds}.pkl",
    }[algo]

    bundle = {
        "model":                model,
        "load_model":           load_model,
        "feature_cols":         all_feature_cols,
        "price_feature_cols":   price_feature_cols_base,
        "load_feature_cols":    load_feature_cols,
        "has_load_preds":       True,
        "load_pred_cols":       load_pred_cols,
        "scaler":               scaler,           # None for lgbm/xgb/rf; set for svr/mlp
        "horizon":              H,
        "mode":                 mode,
        "task":                 "price",
        "algo":                 algo,
        "strategy":             "mimo_with_load",
        "train_end":            train_end,
        "dense_lags":           dense_lags,
    }

    out_path = MODELS_DIR / out_name
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ Saved two-stage MIMO bundle: {out_path}", flush=True)
    print(f"   price model: {algo.upper()} H={H}, {len(all_feature_cols)} feats", flush=True)
    print(f"   load  model: embedded, {len(load_feature_cols)} feats", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# LOAD task: direct MIMO (no two-stage)
# ---------------------------------------------------------------------------

def train_mimo_load(
    *,
    mode: str,
    algo: str,
    train_end: str,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: Optional[int],
    min_samples_leaf: int,
    mlp_hidden: Tuple[int, ...],
    mlp_max_iter: int,
    mlp_lr_init: float = 5e-4,
    mlp_alpha: float = 1e-3,
    mlp_n_iter_no_change: int = 30,
    max_svr_rows: int,
    quiet: bool,
) -> Path:
    from src.split_utils import load_processed, split_time_series

    H = HORIZON

    print(f"\n{'='*65}", flush=True)
    print(f"  Direct MIMO Load H={H} | algo={algo.upper()}", flush=True)
    print(f"{'='*65}", flush=True)

    df_load = load_processed(mode, task="load")
    df_load_train, _ = split_time_series(
        df_load, mode=mode, train_end=train_end, test_size=168
    )
    X_train, Y_train = _make_mimo_xy(df_load_train, horizon=H)
    feature_cols = list(X_train.columns)
    print(f"   -> load features: {X_train.shape[1]} | usable rows: {len(X_train)}", flush=True)

    X_np = X_train.to_numpy(dtype=np.float32)
    scaler = None

    # ── Subsample for SVR
    X_fit, Y_fit = X_np, Y_train
    if algo == "svr" and max_svr_rows > 0 and len(X_fit) > max_svr_rows:
        print(f"   [SVR] Subsampling to last {max_svr_rows} rows (from {len(X_fit)})", flush=True)
        X_fit = X_fit[-max_svr_rows:]
        Y_fit = Y_fit[-max_svr_rows:]

    # ── Scale for SVR and MLP
    if algo in {"svr", "mlp"}:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit).astype(np.float32)
        print(f"   [scale] StandardScaler fitted on {len(X_fit)} rows", flush=True)

    print(f"\n🚀 Training {algo.upper()} LOAD MIMO H={H} ...", flush=True)

    if algo == "rf":
        model = _build_rf_model(n_estimators, max_depth, min_samples_leaf)
        with _suppress_output(enabled=quiet):
            model.fit(X_fit, Y_fit)
        print(f"   -> RF trained: {model.n_estimators} trees, {len(X_fit)} rows", flush=True)

    elif algo == "svr":
        model = _build_svr_model()
        with _suppress_output(enabled=quiet):
            model.fit(X_fit, Y_fit)
        print(f"   -> SVR MultiOutputRegressor trained", flush=True)

    elif algo == "mlp":
        model = _build_mlp_model(mlp_hidden, mlp_max_iter, lr_init=mlp_lr_init, alpha=mlp_alpha, n_iter_no_change=mlp_n_iter_no_change)
        with _suppress_output(enabled=quiet):
            model.fit(X_fit, Y_fit)
        iters = getattr(model, "n_iter_", "?")
        print(f"   -> MLP trained: {iters} iterations, {len(X_fit)} rows", flush=True)

    else:
        raise ValueError(f"For task=load, use algo in {{rf, svr, mlp}} with this script. "
                         f"For lgbm/xgb, use train_lgbm_mimo or train_xgb_mimo instead.")

    # ── Save
    MODELS_DIR.mkdir(exist_ok=True)
    out_name = {
        "rf":  f"rf_{mode}_load_mimo_h{H}.pkl",
        "svr": f"svr_{mode}_load_mimo_h{H}.pkl",
        "mlp": f"mlp_{mode}_load_mimo_h{H}.pkl",
    }[algo]

    bundle = {
        "model":        model,
        "feature_cols": feature_cols,
        "scaler":       scaler,        # None for rf; set for svr/mlp
        "has_load_preds": False,
        "horizon":      H,
        "mode":         mode,
        "task":         "load",
        "algo":         algo,
        "strategy":     "mimo_direct",
        "train_end":    train_end,
    }

    out_path = MODELS_DIR / out_name
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ Saved direct MIMO bundle: {out_path}", flush=True)
    print(f"   {algo.upper()} H={H}, {len(feature_cols)} feats", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Train MIMO H=24 model: two-stage (price) or direct (load)."
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--algo", choices=["lgbm", "xgb", "rf", "svr", "mlp"], default="lgbm")
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_end", required=True,
                   help="Training cutoff, e.g. '2025-11-30 23:00'")

    # LGBM / XGB
    p.add_argument("--n_estimators",   type=int,   default=1000)
    p.add_argument("--learning_rate",  type=float, default=0.05)
    p.add_argument("--num_leaves",     type=int,   default=63, help="LightGBM only")

    # RF
    p.add_argument("--max_depth",         type=int,   default=None, help="RF only")
    p.add_argument("--min_samples_leaf",  type=int,   default=5,    help="RF only")
    p.add_argument("--rf_n_estimators",   type=int,   default=600,  help="RF trees (overrides --n_estimators for RF)")

    # MLP
    p.add_argument("--mlp_hidden",    type=str,   default="512,256,128",
                   help="Hidden layer sizes for MLP, comma-separated (default: 512,256,128)")
    p.add_argument("--mlp_max_iter",  type=int,   default=1000,
                   help="Max adam iterations (default: 1000)")
    p.add_argument("--mlp_lr_init",   type=float, default=5e-4,
                   help="Adam initial learning rate (default: 5e-4)")
    p.add_argument("--mlp_alpha",     type=float, default=1e-3,
                   help="L2 regularisation α (default: 1e-3)")
    p.add_argument("--mlp_no_change", type=int,   default=30,
                   help="n_iter_no_change for early stopping (default: 30)")

    # SVR
    p.add_argument("--max_svr_rows",  type=int, default=20000,
                   help="Subsample to last N rows for SVR (0=no limit)")

    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dense_lags", action="store_true",
                   help="Add dense intraday lags 1-23 (fills same-hour-yesterday gap)")
    args = p.parse_args()

    # Parse MLP hidden
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())

    # For RF: use rf_n_estimators unless not set
    n_est = args.rf_n_estimators if args.algo == "rf" else args.n_estimators

    if args.task == "price":
        if args.algo == "load":
            raise ValueError("algo=load is not valid; did you mean --task load?")
        train_mimo_price(
            mode=args.mode,
            algo=args.algo,
            train_end=args.train_end,
            n_estimators=n_est,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            mlp_hidden=mlp_hidden,
            mlp_max_iter=args.mlp_max_iter,
            mlp_lr_init=args.mlp_lr_init,
            mlp_alpha=args.mlp_alpha,
            mlp_n_iter_no_change=args.mlp_no_change,
            max_svr_rows=args.max_svr_rows,
            quiet=args.quiet,
            dense_lags=args.dense_lags,
        )
    else:  # task == "load"
        train_mimo_load(
            mode=args.mode,
            algo=args.algo,
            train_end=args.train_end,
            n_estimators=n_est,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            mlp_hidden=mlp_hidden,
            mlp_max_iter=args.mlp_max_iter,
            mlp_lr_init=args.mlp_lr_init,
            mlp_alpha=args.mlp_alpha,
            mlp_n_iter_no_change=args.mlp_no_change,
            max_svr_rows=args.max_svr_rows,
            quiet=args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
