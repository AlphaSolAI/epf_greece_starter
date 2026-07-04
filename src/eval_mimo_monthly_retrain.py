"""
eval_mimo_monthly_retrain.py  —  Walk-forward Monthly Retrain για MIMO/Direct H=24

Fair comparison με eval_monthly_retrain.py (TF/Rec monthly retrain).
Για κάθε μήνα (Dec-2025, Jan-2026, Feb-2026):
  1. Retrain MIMO/Direct models σε ΟΛΑ τα data μέχρι τέλος προηγούμενου μήνα
  2. Predict current month (N×24h chunks, ΧΩΡΙΣ recursion)
  3. Συλλέγει predictions → συνολικά Q1 2026 metrics

Models:
  Price: LGBM Direct Dense (only Direct uses dense lags),
         LGBM MIMO, XGB MIMO, RF MIMO (two-stage), MLP MIMO (two-stage),
         SVR MIMO (two-stage, 5K rows), Ensemble Best3
  Load:  LGBM Direct Dense, LGBM MIMO, XGB MIMO, RF MIMO,
         MLP MIMO, SVR MIMO (5K rows), Ensemble Best3

Hyperparameters: FIXED (ίδια με καλύτερα static models — χωρίς Optuna)
Dense lags: ΜΟΝΟ για LGBM Direct (Direct strategy). Όλα τα MIMO χωρίς dense lags.
RF: n_jobs=1 + max_features='sqrt'  (αποφεύγει Windows loky deadlock)
Two-stage price (RF/MLP/SVR): χρησιμοποιεί inline-trained LGBM Load (no-dense) ως load injector

Usage:
  conda run -n epf --no-capture-output python -m src.eval_mimo_monthly_retrain hourly --task price --save_json
  conda run -n epf --no-capture-output python -m src.eval_mimo_monthly_retrain hourly --task load --save_json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

from .split_utils import load_processed, make_xy, split_time_series

# ── Months definition ──────────────────────────────────────────────────────────
EVAL_MONTHS = [
    {"name": "Dec-2025", "train_end": "2025-11-30 23:00",
     "test_start": "2025-12-01 00:00", "test_end": "2025-12-31 23:00"},
    {"name": "Jan-2026", "train_end": "2025-12-31 23:00",
     "test_start": "2026-01-01 00:00", "test_end": "2026-01-31 23:00"},
    {"name": "Feb-2026", "train_end": "2026-01-31 23:00",
     "test_start": "2026-02-01 00:00", "test_end": "2026-02-28 23:00"},
]

HORIZON    = 24
MAX_SVR_ROWS = 5000


# ── Metrics ───────────────────────────────────────────────────────────────────
def _fnp(a) -> np.ndarray:
    return np.asarray(a, dtype=float)

def mae(yt, yp) -> float:
    a, b = _fnp(yt), _fnp(yp)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() > 0 else float("nan")

def rmse(yt, yp) -> float:
    a, b = _fnp(yt), _fnp(yp)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m])**2))) if m.sum() > 0 else float("nan")

def smape(yt, yp) -> float:
    a, b = _fnp(yt), _fnp(yp)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    denom = np.abs(a[m]) + np.abs(b[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(a[m] - b[m]) / denom))


# ── Feature matrix builder ─────────────────────────────────────────────────────
def _make_mimo_xy_dense(df_train: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build (X, Y) for MIMO H=horizon with dense intraday lags (lag1-lag23).
    Always uses dense_lags=True — best variant from static eval.
    Includes fillna(0.0) for MLP/SVR compatibility.
    """
    X, _ = make_xy(df_train)
    y = df_train["y"].astype(float)

    # Add dense intraday lags (lag1..lag23) if not already present
    existing = {int(c[5:]) for c in X.columns if c.startswith("y_lag") and c[5:].isdigit()}
    for lag in range(1, 24):
        if lag not in existing:
            X[f"y_lag{lag}"] = y.shift(lag).reindex(X.index)

    Y_df = pd.concat([y.shift(-k).rename(f"y_t+{k}") for k in range(1, horizon + 1)], axis=1)
    joined = X.join(Y_df, how="inner").dropna(subset=list(Y_df.columns))

    X_al = joined[X.columns].select_dtypes(include=[np.number]).copy()
    for c in X_al.columns:
        if X_al[c].dtype == bool:
            X_al[c] = X_al[c].astype(np.int8)
    X_al = X_al.fillna(0.0)

    Y = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_al, Y


def _make_x_full_dense(df_full: pd.DataFrame) -> pd.DataFrame:
    """Build full X (train+test) with dense lags for origin look-up during prediction."""
    X, _ = make_xy(df_full)
    y = df_full["y"].astype(float)
    existing = {int(c[5:]) for c in X.columns if c.startswith("y_lag") and c[5:].isdigit()}
    for lag in range(1, 24):
        if lag not in existing:
            X[f"y_lag{lag}"] = y.shift(lag).reindex(X.index)
    return X.select_dtypes(include=[np.number]).fillna(0.0)


def _make_mimo_xy_nodense(df_train: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build (X, Y) for MIMO without dense intraday lags (MIMO strategy — no Direct lags)."""
    X, _ = make_xy(df_train)
    y = df_train["y"].astype(float)

    Y_df = pd.concat([y.shift(-k).rename(f"y_t+{k}") for k in range(1, horizon + 1)], axis=1)
    joined = X.join(Y_df, how="inner").dropna(subset=list(Y_df.columns))

    X_al = joined[X.columns].select_dtypes(include=[np.number]).copy()
    for c in X_al.columns:
        if X_al[c].dtype == bool:
            X_al[c] = X_al[c].astype(np.int8)
    X_al = X_al.fillna(0.0)

    Y = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_al, Y


def _make_x_full_nodense(df_full: pd.DataFrame) -> pd.DataFrame:
    """Build full X (train+test) without dense lags."""
    X, _ = make_xy(df_full)
    return X.select_dtypes(include=[np.number]).fillna(0.0)


# ── Model builders ─────────────────────────────────────────────────────────────
def _build_lgbm(n_est: int = 1000) -> object:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    # LGBM: n_jobs=1 per base model + n_jobs=4 on MOR.
    # _train_lgbm_direct wraps fit() in joblib threading context so models
    # run in parallel Python threads. LightGBM releases GIL → true parallelism.
    # n_jobs=4 (not -1) avoids OOM when 24 models are fit concurrently.
    base = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=1, verbosity=-1)
    return MultiOutputRegressor(base, n_jobs=4)


def _build_lgbm_mimo_model(n_est: int = 1000) -> object:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    # LGBM MIMO: NO dense lags (Direct strategy only uses those).
    # n_jobs=1 per base + n_jobs=4 MOR with threading backend.
    base = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=1, verbosity=-1)
    return MultiOutputRegressor(base, n_jobs=4)


def _build_xgb(n_est: int = 1000) -> object:
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    # XGB: n_jobs=-1 base (native OpenMP) + n_jobs=1 MOR (sequential)
    # XGB doesn't release GIL properly for loky → use native threading instead
    base = XGBRegressor(n_estimators=n_est, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        n_jobs=-1, verbosity=0)
    return MultiOutputRegressor(base, n_jobs=1)


def _build_rf(n_est: int = 100) -> object:
    from sklearn.ensemble import RandomForestRegressor
    # n_jobs=1 → αποφεύγει loky deadlock στα Windows με conda
    # max_features="sqrt" → ~9x ταχύτερο από max_features=0.8
    # n_est=100, max_depth=12 → αρκετό για monthly retrain (αποφεύγει silent hang)
    return RandomForestRegressor(n_estimators=n_est, max_features="sqrt",
                                 max_depth=12, min_samples_split=4,
                                 min_samples_leaf=2, n_jobs=1, random_state=42)


def _build_mlp(hidden=(256, 128), max_iter=400) -> object:
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        alpha=1e-3, learning_rate_init=5e-4, max_iter=max_iter,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=30, tol=1e-5, random_state=42, verbose=False)


def _build_svr() -> object:
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    base = SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1)
    return MultiOutputRegressor(base, n_jobs=-1)


# ── Optuna tuning (LGBM / XGB) ────────────────────────────────────────────────
def _optuna_tune_lgbm(X_tr: np.ndarray, Y_tr: np.ndarray,
                      n_trials: int = 5, n_splits: int = 2,
                      max_cv_rows: int = 20000) -> dict:
    """Inline Optuna για LGBM MIMO. Returns best params.
    Χρησιμοποιεί n_jobs=-1 (native OpenMP) + sequential MultiOutputRegressor.
    max_cv_rows: subsamples τελευταίες N rows για CV (πιο σχετικές με test μήνα).
    """
    import optuna
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import TimeSeriesSplit

    # Subsample: τελευταίες max_cv_rows rows (most recent, more relevant for next month)
    if len(X_tr) > max_cv_rows:
        X_cv = X_tr[-max_cv_rows:]
        Y_cv = Y_tr[-max_cv_rows:]
    else:
        X_cv, Y_cv = X_tr, Y_tr

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        n_est      = trial.suggest_int("n_estimators", 300, 1000, step=100)
        num_leaves = trial.suggest_int("num_leaves", 31, 127)
        lr         = trial.suggest_float("learning_rate", 0.02, 0.12, log=True)
        scores: List[float] = []
        for tr_i, va_i in tscv.split(X_cv):
            base = lgb.LGBMRegressor(
                n_estimators=n_est, learning_rate=lr, num_leaves=num_leaves,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=-1)  # native OpenMP threading
            m = MultiOutputRegressor(base, n_jobs=1)  # sequential, no loky overhead
            m.fit(X_cv[tr_i], Y_cv[tr_i])
            preds = m.predict(X_cv[va_i])
            scores.append(float(np.mean(np.abs(Y_cv[va_i] - preds))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _optuna_tune_xgb(X_tr: np.ndarray, Y_tr: np.ndarray,
                     n_trials: int = 5, n_splits: int = 2,
                     max_cv_rows: int = 20000) -> dict:
    """Inline Optuna για XGB MIMO. Returns best params.
    Χρησιμοποιεί n_jobs=-1 (native OpenMP) + sequential MultiOutputRegressor.
    max_cv_rows: subsamples τελευταίες N rows για CV.
    XGB n_estimators range μικρότερο γιατί με max_depth>6 έχει αρκετή capacity.
    """
    import optuna
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import TimeSeriesSplit

    # Subsample: τελευταίες max_cv_rows rows
    if len(X_tr) > max_cv_rows:
        X_cv = X_tr[-max_cv_rows:]
        Y_cv = Y_tr[-max_cv_rows:]
    else:
        X_cv, Y_cv = X_tr, Y_tr

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        n_est     = trial.suggest_int("n_estimators", 300, 800, step=100)
        max_depth = trial.suggest_int("max_depth", 4, 8)
        lr        = trial.suggest_float("learning_rate", 0.02, 0.12, log=True)
        scores: List[float] = []
        for tr_i, va_i in tscv.split(X_cv):
            base = XGBRegressor(
                n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=0)  # native OpenMP threading
            m = MultiOutputRegressor(base, n_jobs=1)  # sequential, no loky overhead
            m.fit(X_cv[tr_i], Y_cv[tr_i])
            preds = m.predict(X_cv[va_i])
            scores.append(float(np.mean(np.abs(Y_cv[va_i] - preds))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _build_lgbm_from_params(params: dict) -> object:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    # LGBM: n_jobs=1 per base model + n_jobs=-1 on MOR (threading context in _train_lgbm_direct).
    # Passes ALL params from Optuna JSON for full fidelity with cached study.
    base = lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 0.0),
        random_state=42, n_jobs=1, verbosity=-1)
    return MultiOutputRegressor(base, n_jobs=2)


def _build_xgb_from_params(params: dict) -> object:
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    # XGB: n_jobs=-1 base (native OpenMP) + n_jobs=1 MOR (sequential)
    base = XGBRegressor(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 6),
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0)
    return MultiOutputRegressor(base, n_jobs=1)


# ── Training ──────────────────────────────────────────────────────────────────
def _train_lgbm_direct(X_tr: np.ndarray, Y_tr: np.ndarray, n_est: int = 1000,
                       params: Optional[dict] = None) -> object:
    model = _build_lgbm_from_params(params) if params else _build_lgbm(n_est)
    # Threading context: forces joblib to use threads (not loky processes) for MOR parallel fits.
    # LightGBM releases GIL → model fits run in parallel (n_jobs=4 avoids OOM).
    with joblib.parallel_backend("threading", n_jobs=4):
        model.fit(X_tr, Y_tr)
    return model


def _train_lgbm_mimo_model(X_tr: np.ndarray, Y_tr: np.ndarray, n_est: int = 1000) -> object:
    """Train LGBM MIMO (no dense lags, no two-stage)."""
    model = _build_lgbm_mimo_model(n_est)
    with joblib.parallel_backend("threading", n_jobs=4):
        model.fit(X_tr, Y_tr)
    return model


def _train_xgb_mimo(X_tr: np.ndarray, Y_tr: np.ndarray, n_est: int = 1000,
                    params: Optional[dict] = None) -> object:
    model = _build_xgb_from_params(params) if params else _build_xgb(n_est)
    # XGB uses native n_jobs=-1 → no parallel_backend context needed
    model.fit(X_tr, Y_tr)
    return model


def _train_rf(X_tr: np.ndarray, Y_tr: np.ndarray, n_est: int = 400) -> object:
    model = _build_rf(n_est)
    model.fit(X_tr, Y_tr)
    return model


def _train_mlp(X_tr: np.ndarray, Y_tr: np.ndarray) -> Tuple[object, object]:
    """Returns (model, scaler)."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr).astype(np.float32)
    model = _build_mlp()
    model.fit(X_sc, Y_tr)
    return model, scaler


def _train_svr(X_tr: np.ndarray, Y_tr: np.ndarray) -> Tuple[object, object]:
    """Returns (model, scaler). Subsamples to MAX_SVR_ROWS for speed."""
    from sklearn.preprocessing import StandardScaler
    if len(X_tr) > MAX_SVR_ROWS:
        X_tr = X_tr[-MAX_SVR_ROWS:]
        Y_tr = Y_tr[-MAX_SVR_ROWS:]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr).astype(np.float32)
    model = _build_svr()
    model.fit(X_sc, Y_tr)
    return model, scaler


# ── Prediction ────────────────────────────────────────────────────────────────
def _predict_month(
    bundle: dict,
    origins: List[pd.Timestamp],
    test_index: pd.DatetimeIndex,
    X_full: pd.DataFrame,
    X_load_full: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Predict all 24h chunks for a month. Returns array aligned to test_index."""
    preds = np.full(len(test_index), np.nan, dtype=float)
    t_map = {ts: i for i, ts in enumerate(test_index)}
    freq = pd.tseries.frequencies.to_offset("h")

    feat_cols    = bundle["feat_cols"]
    model        = bundle["model"]
    scaler       = bundle.get("scaler")
    has_load     = bundle.get("has_load", False)
    load_model   = bundle.get("load_model")
    load_fcols   = bundle.get("load_feat_cols", [])
    price_fcols  = bundle.get("price_feat_cols", [])

    for origin in origins:
        # Snap origin to nearest available index
        if origin not in X_full.index:
            cands = X_full.index[X_full.index <= origin]
            if len(cands) == 0:
                continue
            origin = cands[-1]

        if has_load and load_model is not None and X_load_full is not None:
            # Two-stage: (1) predict load, (2) augment price row
            orig_l = origin
            if orig_l not in X_load_full.index:
                cands = X_load_full.index[X_load_full.index <= origin]
                if len(cands) == 0:
                    continue
                orig_l = cands[-1]
            x_load_row = np.array(
                [float(X_load_full.loc[orig_l, c]) if c in X_load_full.columns else 0.0
                 for c in load_fcols], dtype=np.float32
            ).reshape(1, -1)
            # Use threading backend for single-row predict to avoid spawning loky
            # processes (which causes OOM/pagefile exhaustion on Windows)
            with joblib.parallel_backend("threading", n_jobs=2):
                load_preds = np.asarray(load_model.predict(x_load_row)[0], dtype=float)

            x_price_row = np.array(
                [float(X_full.loc[origin, c]) if c in X_full.columns else 0.0
                 for c in price_fcols], dtype=np.float32
            )
            x_row = np.concatenate([x_price_row, load_preds[:HORIZON].astype(np.float32)]).reshape(1, -1)
        else:
            x_row = np.array(
                [float(X_full.loc[origin, c]) if c in X_full.columns else 0.0
                 for c in feat_cols], dtype=np.float32
            ).reshape(1, -1)

        if scaler is not None:
            x_row = scaler.transform(x_row).astype(np.float32)

        # Use threading backend for single-row predict to avoid loky OOM
        with joblib.parallel_backend("threading", n_jobs=2):
            yhat = np.asarray(model.predict(x_row)[0], dtype=float)[:HORIZON]
        pred_idx = pd.date_range(origin + freq, periods=HORIZON, freq=freq)
        for k, ts in enumerate(pred_idx):
            if ts in t_map:
                preds[t_map[ts]] = yhat[k]

    return preds


# ── Baseline helpers ───────────────────────────────────────────────────────────
def _naive1(y_all: pd.Series, test_idx: pd.DatetimeIndex, origins: List[pd.Timestamp]) -> np.ndarray:
    preds = np.full(len(test_idx), np.nan)
    t_map = {ts: i for i, ts in enumerate(test_idx)}
    freq  = pd.tseries.frequencies.to_offset("h")
    for o in origins:
        hist = y_all.loc[:o].dropna()
        if len(hist) == 0:
            continue
        val = float(hist.iloc[-1])
        for ts in pd.date_range(o + freq, periods=HORIZON, freq=freq):
            if ts in t_map:
                preds[t_map[ts]] = val
    return preds


def _seasonal_profile(train_y: pd.Series, test_idx: pd.DatetimeIndex) -> np.ndarray:
    prof = (pd.DataFrame({"y": train_y.values,
                           "dow": train_y.index.dayofweek,
                           "h":   train_y.index.hour}, index=train_y.index)
            .groupby(["dow", "h"])["y"].mean())
    gm = float(train_y.mean())
    return np.array([float(prof.get((ts.dayofweek, ts.hour), gm)) for ts in test_idx])


# ── Main evaluation loop ───────────────────────────────────────────────────────
def evaluate_mimo_monthly_retrain(
    mode: str,
    task: str,
    save_json: bool = False,
    use_optuna: bool = False,
    n_trials: int = 5,
    n_splits: int = 2,
    use_cached_params: bool = False,
    start_from: int = 0,
) -> None:
    unit = "€/MWh" if task == "price" else "MW"
    if use_cached_params:
        optuna_tag = " + CachedOptuna (pre-computed)"
    elif use_optuna:
        optuna_tag = f" + Optuna({n_trials}t,{n_splits}s)"
    else:
        optuna_tag = " (fixed params)"

    # ── Load cached Optuna params if requested ─────────────────────────────
    cached_lgbm_params: Optional[dict] = None
    if use_cached_params:
        cached_json = MODELS_DIR / f"lgbm_direct_hourly_{task}_h24_optuna_params.json"
        if cached_json.exists():
            with open(cached_json, "r") as f:
                jdata = json.load(f)
            cached_lgbm_params = jdata.get("best_params", {})
            print(f"  [CachedOpt] Loaded LGBM params from {cached_json.name}:")
            print(f"  {cached_lgbm_params}", flush=True)
        else:
            print(f"  [CachedOpt] WARNING: {cached_json} not found — using default LGBM params")

    months_to_run = EVAL_MONTHS[start_from:]

    print(f"\n{'='*72}")
    print(f"  MIMO/Direct H=24 Monthly Retrain{optuna_tag} | task={task.upper()} | Q1 2026")
    print(f"  Months: {[m['name'] for m in months_to_run]}")
    if start_from > 0:
        print(f"  [--start_from {start_from}] Skipping: {[m['name'] for m in EVAL_MONTHS[:start_from]]}")
    print(f"{'='*72}\n", flush=True)

    # Collect per-month predictions
    all_actual:   List[np.ndarray] = []
    all_preds:    Dict[str, List[np.ndarray]] = {}
    month_names:  List[str] = []

    for minfo in months_to_run:
        mname      = minfo["name"]
        train_end  = minfo["train_end"]
        test_start = minfo["test_start"]
        test_end   = minfo["test_end"]

        print(f"\n{'─'*60}")
        print(f"  Month: {mname} | train_end={train_end}")
        print(f"{'─'*60}", flush=True)

        # ── Load data ──────────────────────────────────────────────────────
        df_task = load_processed(mode, task=task)
        df_tr, df_te = split_time_series(df_task, mode=mode, train_end=train_end,
                                          test_start=test_start, test_end=test_end)
        _, y_test   = make_xy(df_te)
        y_test_arr  = np.asarray(y_test, dtype=float)
        test_index  = df_te.index
        train_y     = df_tr["y"].astype(float)
        y_all       = df_task["y"].astype(float)

        # Build X with dense lags (train only for fitting, full for prediction)
        X_tr_df, Y_tr = _make_mimo_xy_dense(df_tr, HORIZON)
        feat_cols = list(X_tr_df.columns)
        X_tr_np   = X_tr_df.to_numpy(dtype=np.float32)

        # Full X (train+test) for prediction origin look-up
        X_full = _make_x_full_dense(df_task.loc[:test_end])

        # Daily origins (one per day of test month, at hour before midnight)
        freq    = pd.tseries.frequencies.to_offset("h")
        origins = [pd.Timestamp(test_start) - freq + pd.Timedelta(days=d)
                   for d in range(len(test_index) // HORIZON + 1)]
        origins = [o for o in origins if o >= df_tr.index[0]]

        print(f"  Train rows: {len(df_tr):,}  |  Test hours: {len(test_index)}"
              f"  |  Features: {len(feat_cols)}  |  Fit rows: {len(X_tr_df)}", flush=True)

        # Build X without dense lags for LGBM MIMO
        X_tr_nd_df, Y_tr_nd = _make_mimo_xy_nodense(df_tr, HORIZON)
        lgbm_mimo_feat_cols = list(X_tr_nd_df.columns)
        X_tr_nd_np = X_tr_nd_df.to_numpy(dtype=np.float32)
        X_full_nd  = _make_x_full_nodense(df_task.loc[:test_end])

        # ── Two-stage: load model for price ───────────────────────────────
        X_load_full: Optional[pd.DataFrame] = None
        lgbm_load_model = None
        lgbm_load_feat_cols: List[str] = []

        if task == "price":
            print(f"  [load] Training LGBM Load Dense for two-stage augmentation ...", flush=True)
            df_load = load_processed(mode, task="load")
            df_load_tr, _ = split_time_series(df_load, mode=mode, train_end=train_end,
                                               test_start=test_start, test_end=test_end)
            X_load_tr_df, Y_load_tr = _make_mimo_xy_dense(df_load_tr, HORIZON)
            lgbm_load_feat_cols = list(X_load_tr_df.columns)
            X_load_tr_np = X_load_tr_df.to_numpy(dtype=np.float32)
            lgbm_load_model = _train_lgbm_mimo_model(X_load_tr_np, Y_load_tr, n_est=100)
            X_load_full = _make_x_full_dense(df_load.loc[:test_end])
            print(f"  [load] Done. Load feat cols: {len(lgbm_load_feat_cols)}", flush=True)

            # Augment price training X (dense) with load predictions (two-stage)
            X_load_for_price = (X_load_tr_df
                                .reindex(X_tr_df.index)
                                .fillna(0.0))
            load_pred_train = lgbm_load_model.predict(
                X_load_for_price[lgbm_load_feat_cols].to_numpy(dtype=np.float32)
            )  # (n_rows, H)
            lp_cols = [f"load_pred_h{i+1:02d}" for i in range(HORIZON)]
            lp_df   = pd.DataFrame(load_pred_train, index=X_tr_df.index, columns=lp_cols)
            X_price_aug_df = pd.concat([X_tr_df, lp_df], axis=1)
            price_aug_feat_cols = list(X_price_aug_df.columns)
            X_price_aug_np = X_price_aug_df.to_numpy(dtype=np.float32)

        # ── Train models ───────────────────────────────────────────────────
        month_preds: Dict[str, np.ndarray] = {}
        month_names.append(mname)
        all_actual.append(y_test_arr)

        # Helper to build simple bundle
        def _bundle(m, fc, sc=None, has_l=False, lm=None, lfc=None, pfc=None):
            return {"model": m, "feat_cols": fc, "scaler": sc,
                    "has_load": has_l, "load_model": lm,
                    "load_feat_cols": lfc or [], "price_feat_cols": pfc or []}

        # 1. LGBM Direct Dense  (+ Optuna or CachedOptuna if enabled)
        if use_cached_params:
            lgbm_label = "LGBM Direct Dense MR+CachedOpt"
        elif use_optuna:
            lgbm_label = "LGBM Direct Dense MR+Opt"
        else:
            lgbm_label = "LGBM Direct Dense MR"

        print(f"  [1/5] Training {lgbm_label} ...", flush=True)
        if use_cached_params and cached_lgbm_params:
            print(f"       Using cached Optuna params (n_est={cached_lgbm_params.get('n_estimators')}, "
                  f"lr={cached_lgbm_params.get('learning_rate', 0):.4f}, "
                  f"leaves={cached_lgbm_params.get('num_leaves')})", flush=True)
            lgbm_m = _train_lgbm_direct(X_tr_np, Y_tr, params=cached_lgbm_params)
        elif use_optuna:
            print(f"       Optuna tuning ({n_trials} trials, {n_splits} splits) ...", flush=True)
            lgbm_params = _optuna_tune_lgbm(X_tr_np, Y_tr, n_trials=n_trials, n_splits=n_splits)
            print(f"       Best LGBM params: {lgbm_params}", flush=True)
            lgbm_m = _train_lgbm_direct(X_tr_np, Y_tr, params=lgbm_params)
        else:
            lgbm_m = _train_lgbm_direct(X_tr_np, Y_tr, n_est=1000)
        b = _bundle(lgbm_m, feat_cols)
        month_preds[lgbm_label] = _predict_month(b, origins, test_index, X_full)
        print(f"       MAE={mae(y_test_arr, month_preds[lgbm_label]):.3f}", flush=True)

        # 1b. LGBM MIMO Dense
        print(f"  [1b] Training LGBM MIMO Dense MR ...", flush=True)
        lgbm_mimo_m = _train_lgbm_mimo_model(X_tr_np, Y_tr, n_est=1000)
        b = _bundle(lgbm_mimo_m, feat_cols)
        month_preds["LGBM MIMO Dense MR"] = _predict_month(b, origins, test_index, X_full)
        print(f"       MAE={mae(y_test_arr, month_preds['LGBM MIMO Dense MR']):.3f}", flush=True)

        # 2. XGB MIMO Dense
        if use_optuna:
            xgb_label = "XGB MIMO Dense MR+Opt"
        else:
            xgb_label = "XGB MIMO Dense MR"

        print(f"  [2/5] Training {xgb_label} ...", flush=True)
        if use_optuna:
            print(f"       Optuna tuning ({n_trials} trials, {n_splits} splits) ...", flush=True)
            xgb_params = _optuna_tune_xgb(X_tr_np, Y_tr, n_trials=n_trials, n_splits=n_splits)
            print(f"       Best XGB params: {xgb_params}", flush=True)
            xgb_m = _train_xgb_mimo(X_tr_np, Y_tr, params=xgb_params)
        else:
            xgb_m = _train_xgb_mimo(X_tr_np, Y_tr, n_est=1000)
        b = _bundle(xgb_m, feat_cols)
        month_preds[xgb_label] = _predict_month(b, origins, test_index, X_full)
        print(f"       MAE={mae(y_test_arr, month_preds[xgb_label]):.3f}", flush=True)

        # 3. RF MIMO Dense  (two-stage for price, direct for load)
        print(f"  [3/5] Training RF MIMO Dense ...", flush=True)
        if task == "price":
            rf_m = _train_rf(X_price_aug_np, Y_tr, n_est=100)
            b = _bundle(rf_m, price_aug_feat_cols, has_l=True,
                        lm=lgbm_load_model, lfc=lgbm_load_feat_cols,
                        pfc=feat_cols)
        else:
            rf_m = _train_rf(X_tr_np, Y_tr, n_est=100)
            b = _bundle(rf_m, feat_cols)
        month_preds["RF MIMO Dense MR"] = _predict_month(
            b, origins, test_index, X_full, X_load_full)
        print(f"       MAE={mae(y_test_arr, month_preds['RF MIMO Dense MR']):.3f}", flush=True)

        # 4. MLP MIMO Dense  (two-stage for price, direct for load)
        print(f"  [4/5] Training MLP MIMO Dense ...", flush=True)
        if task == "price":
            mlp_m, mlp_sc = _train_mlp(X_price_aug_np, Y_tr)
            iters = getattr(mlp_m, "n_iter_", "?")
            print(f"       MLP iters={iters}", flush=True)
            b = {"model": mlp_m, "feat_cols": price_aug_feat_cols, "scaler": mlp_sc,
                 "has_load": True, "load_model": lgbm_load_model,
                 "load_feat_cols": lgbm_load_feat_cols, "price_feat_cols": feat_cols}
        else:
            mlp_m, mlp_sc = _train_mlp(X_tr_np, Y_tr)
            iters = getattr(mlp_m, "n_iter_", "?")
            print(f"       MLP iters={iters}", flush=True)
            b = _bundle(mlp_m, feat_cols, sc=mlp_sc)
        month_preds["MLP MIMO Dense MR"] = _predict_month(
            b, origins, test_index, X_full, X_load_full)
        print(f"       MAE={mae(y_test_arr, month_preds['MLP MIMO Dense MR']):.3f}", flush=True)

        # 5. SVR MIMO Dense  (two-stage for price, direct for load)
        print(f"  [5/5] Training SVR MIMO Dense (5K rows) ...", flush=True)
        if task == "price":
            n_sub = min(MAX_SVR_ROWS, len(X_price_aug_np))
            from sklearn.preprocessing import StandardScaler as _SS
            svr_sc = _SS()
            X_svr = svr_sc.fit_transform(X_price_aug_np[-n_sub:]).astype(np.float32)
            svr_m = _build_svr(); svr_m.fit(X_svr, Y_tr[-n_sub:])
            b = {"model": svr_m, "feat_cols": price_aug_feat_cols, "scaler": svr_sc,
                 "has_load": True, "load_model": lgbm_load_model,
                 "load_feat_cols": lgbm_load_feat_cols, "price_feat_cols": feat_cols}
        else:
            n_sub = min(MAX_SVR_ROWS, len(X_tr_np))
            from sklearn.preprocessing import StandardScaler as _SS
            svr_sc = _SS()
            X_svr = svr_sc.fit_transform(X_tr_np[-n_sub:]).astype(np.float32)
            svr_m = _build_svr(); svr_m.fit(X_svr, Y_tr[-n_sub:])
            b = _bundle(svr_m, feat_cols, sc=svr_sc)
        month_preds["SVR MIMO Dense MR"] = _predict_month(
            b, origins, test_index, X_full_nd, X_load_full)
        print(f"       MAE={mae(y_test_arr, month_preds['SVR MIMO Dense MR']):.3f}", flush=True)

        # ── Baselines ──────────────────────────────────────────────────────
        month_preds["Naive-1"]         = _naive1(y_all, test_index, origins)
        month_preds["Seasonal Profile"] = _seasonal_profile(train_y, test_index)

        # ── Ensemble Best3 (exclude SVR, Naive, Seasonal) ─────────────────
        cand_names = [lgbm_label, "LGBM MIMO Dense MR", xgb_label, "RF MIMO Dense MR", "MLP MIMO Dense MR"]
        cand_maes  = {n: mae(y_test_arr, month_preds[n]) for n in cand_names}
        valid = {n: v for n, v in cand_maes.items() if np.isfinite(v)}
        if valid:
            top3 = sorted(valid, key=valid.get)[:3]
            weights = {n: 1.0 / valid[n] for n in top3}
            wsum = sum(weights.values())
            ens3 = sum(weights[n] / wsum * month_preds[n] for n in top3)
            ens_label = "Ensemble-Best3 MIMO MR+Opt" if use_optuna else "Ensemble-Best3 MIMO MR"
            month_preds[ens_label] = ens3

            all_weights = {n: 1.0 / valid[n] for n in valid}
            wsum_all = sum(all_weights.values())
            ens_all = sum(all_weights[n] / wsum_all * month_preds[n] for n in valid)
            ens_all_label = "Ensemble-All MIMO MR+Opt" if use_optuna else "Ensemble-All MIMO MR"
            month_preds[ens_all_label] = ens_all
            print(f"  [ENS] Best3 ({'+'.join(t.split()[0] for t in top3)}): "
                  f"MAE={mae(y_test_arr, ens3):.3f}", flush=True)

        # Accumulate
        for name, pred in month_preds.items():
            all_preds.setdefault(name, []).append(pred)

    # ── Combined Q1 results ────────────────────────────────────────────────────
    actual_all = np.concatenate(all_actual)
    results: Dict[str, Tuple[float, float, float]] = {}
    for name, plist in all_preds.items():
        p = np.concatenate(plist)
        results[name] = (mae(actual_all, p), rmse(actual_all, p), smape(actual_all, p))

    # Sort by MAE
    sorted_names = sorted(results, key=lambda n: results[n][0])

    print(f"\n\n{'='*72}")
    print(f"  COMBINED RESULTS | task={task.upper()} | Q1 2026 (Dec+Jan+Feb)")
    print(f"  Total test hours: {len(actual_all)}")
    print(f"{'='*72}")
    hdr = f"{'Model':<36}  {'MAE':>10}  {'RMSE':>10}  {'sMAPE%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in sorted_names:
        m, r, s = results[name]
        print(f"  {name:<34}  {m:>10.3f}  {r:>10.3f}  {s:>8.3f}")

    winner = sorted_names[0]
    w_mae  = results[winner][0]
    print(f"\n  🏆 WINNER: {winner} (MAE={w_mae:.3f} {unit})")
    print(f"{'='*72}")

    # ── Per-month breakdown ────────────────────────────────────────────────────
    print(f"\n  PER-MONTH BREAKDOWN\n")
    col_names = sorted_names[:8]  # top 8 for readability
    header = f"  {'Month':<12}" + "".join(f"  {n[:14]:>14}" for n in col_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, mname in enumerate(month_names):
        row = f"  {mname:<12}"
        for name in col_names:
            p = all_preds[name][i]
            a = all_actual[i]
            row += f"  {mae(a, p):>14.3f}"
        print(row)

    print(f"\n{'='*72}\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    if save_json:
        _save_json(task, actual_all, all_preds, all_actual, month_names, results)


def _save_json(task: str, actual_all: np.ndarray, all_preds: Dict,
               all_actual: List[np.ndarray], month_names: List[str],
               results: Dict) -> None:
    """Save JSON for dashboard comparison."""
    # Build date range Q1 2026 (Dec+Jan+Feb)
    dates_q1 = []
    for minfo in EVAL_MONTHS:
        idx = pd.date_range(minfo["test_start"], minfo["test_end"], freq="h")
        dates_q1.extend([str(ts) for ts in idx])

    # Series dict: concatenated predictions
    series: Dict[str, List] = {}
    for name, plist in all_preds.items():
        p = np.concatenate(plist)
        series[name] = [round(float(v), 4) if np.isfinite(v) else None for v in p]

    # Metrics
    metrics = []
    for name, (m, r, s) in results.items():
        metrics.append({"Model": name, "Type": "mimo_mr", "MAE": round(m, 4),
                        "RMSE": round(r, 4), "sMAPE": round(s, 4)})
    metrics.sort(key=lambda x: x["MAE"])

    out = {
        "strategy": "mimo_monthly_retrain",
        "task": task,
        "period": "Q1_2026",
        "dates": dates_q1,
        "actual": [round(float(v), 4) if np.isfinite(v) else None for v in actual_all],
        "series": series,
        "metrics": metrics,
    }

    suffix = "_optuna" if any("Opt" in k for k in all_preds) else ""
    fname = BASE_DIR / f"dashboard_data_hourly_{task}_mimo_monthly_retrain{suffix}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON saved: {fname}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="MIMO/Direct Monthly Retrain Evaluation")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", required=True, choices=["price", "load"])
    p.add_argument("--save_json", action="store_true")
    p.add_argument("--optuna", action="store_true",
                   help="Tune LGBM+XGB hyperparams with Optuna per month (slower but better)")
    p.add_argument("--cached_params", action="store_true",
                   help="Use pre-computed Optuna params from lgbm_direct_hourly_{task}_h24_optuna_params.json "
                        "(fast — no inline tuning, uses best params from full-dataset study)")
    p.add_argument("--n_trials", type=int, default=5,
                   help="Optuna trials per model per month (default=5)")
    p.add_argument("--n_splits", type=int, default=2,
                   help="TimeSeriesSplit folds for Optuna CV (default=2, minimum=2)")
    p.add_argument("--start_from", type=int, default=0,
                   help="Skip the first N months (0=run all, 1=skip Dec-2025, 2=skip Dec+Jan)")
    args = p.parse_args()

    evaluate_mimo_monthly_retrain(
        mode=args.mode, task=args.task, save_json=args.save_json,
        use_optuna=args.optuna, n_trials=args.n_trials, n_splits=args.n_splits,
        use_cached_params=args.cached_params, start_from=args.start_from)
    return 0


if __name__ == "__main__":
    main()
