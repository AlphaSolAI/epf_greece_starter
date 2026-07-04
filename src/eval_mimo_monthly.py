"""
eval_mimo_monthly.py  (v2 — full monthly comparison)

1-month MIMO/Direct evaluation comparing:
  - H=168 (weekly MIMO)  : 5 predictions × up to 168h = 744h total
  - H=24  (daily  MIMO)  : 31 predictions × 24h = 744h total

For price task: also evaluates two-stage models that use load MIMO predictions
as additional features (lgbm/xgb_*_price_h24_with_load.pkl).

No recursion is involved — all H hours are predicted simultaneously from
the feature vector at the origin timestamp.

Saves:
  dashboard_data_hourly_{task}_mimo_monthly.json   (if --save_json)

Usage:
  conda run -n epf --no-capture-output python -m src.eval_mimo_monthly hourly \\
      --task price --train_end "2025-11-30 23:00" \\
      --test_start "2025-12-01 00:00" --test_end "2025-12-31 23:00" --save_json

  conda run -n epf --no-capture-output python -m src.eval_mimo_monthly hourly \\
      --task load --train_end "2025-11-30 23:00" \\
      --test_start "2025-12-01 00:00" --test_end "2025-12-31 23:00" --save_json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Model registry  (label, filename, chunk_hours)
# ---------------------------------------------------------------------------

MODELS_PRICE: List[Tuple[str, str, int]] = [
    # ── Direct tab (LGBM-based + XGB-based) ────────────────────────────
    ("LGBM DIRECT 168h",          "lgbm_direct_hourly_price_h168.pkl",             168),
    ("LGBM DIRECT 24h",           "lgbm_direct_hourly_price_h24.pkl",               24),
    ("LGBM DIRECT 24h Optuna",    "lgbm_direct_hourly_price_h24_optuna.pkl",        24),
    ("LGBM DIRECT 24h+Load",      "lgbm_direct_hourly_price_h24_with_load.pkl",     24),
    ("LGBM DIRECT 24h Dense",     "lgbm_direct_hourly_price_h24_dense.pkl",         24),
    ("XGB DIRECT 24h Dense",      "xgb_direct_hourly_price_h24_dense.pkl",          24),
    # ── MIMO tab (LGBM / XGB / RF / SVR / MLP) ─────────────────────────
    ("XGB MIMO 168h",             "xgb_hourly_price_mimo_h168.pkl",                168),
    ("XGB MIMO 24h",              "xgb_hourly_price_mimo_h24.pkl",                  24),
    ("XGB MIMO 24h Dense",        "xgb_hourly_price_mimo_h24_dense.pkl",            24),
    ("XGB MIMO 24h Optuna",       "xgb_vect_hourly_price_h24_optuna.pkl",           24),
    ("XGB MIMO 24h+Load",         "xgb_hourly_price_h24_with_load.pkl",             24),
    ("LGBM MIMO 24h",             "lgbm_hourly_price_mimo_h24.pkl",                 24),
    ("RF MIMO 24h+Load",          "rf_hourly_price_mimo_h24_with_load.pkl",         24),
    ("RF MIMO 24h+Load Dense",    "rf_hourly_price_mimo_h24_with_load_dense.pkl",   24),
    ("RF MIMO 24h Optuna",        "rf_mimo_hourly_price_h24_optuna.pkl",            24),
    ("SVR MIMO 24h+Load",         "svr_hourly_price_mimo_h24_with_load.pkl",        24),
    ("SVR MIMO 24h+Load Dense",   "svr_hourly_price_mimo_h24_with_load_dense.pkl",  24),
    ("MLP MIMO 24h+Load",         "mlp_hourly_price_mimo_h24_with_load.pkl",        24),
    ("MLP MIMO 24h+Load Dense",   "mlp_hourly_price_mimo_h24_with_load_dense.pkl",  24),
]

MODELS_LOAD: List[Tuple[str, str, int]] = [
    # ── Direct tab (LGBM-based + XGB-based) ────────────────────────────
    ("LGBM DIRECT 168h",          "lgbm_direct_hourly_load_h168.pkl",              168),
    ("LGBM DIRECT 24h",           "lgbm_direct_hourly_load_h24.pkl",                24),
    ("LGBM DIRECT 24h Optuna",    "lgbm_direct_hourly_load_h24_optuna.pkl",         24),
    ("LGBM DIRECT 24h Dense",     "lgbm_direct_hourly_load_h24_dense.pkl",          24),
    ("XGB DIRECT 24h Dense",      "xgb_direct_hourly_load_h24_dense.pkl",           24),
    # ── MIMO tab (LGBM / XGB / RF / SVR / MLP) ─────────────────────────
    ("XGB MIMO 168h",             "xgb_hourly_load_mimo_h168.pkl",                 168),
    ("XGB MIMO 24h",              "xgb_hourly_load_mimo_h24.pkl",                   24),
    ("XGB MIMO 24h Dense",        "xgb_hourly_load_mimo_h24_dense.pkl",             24),
    ("XGB MIMO 24h Optuna",       "xgb_vect_hourly_load_h24_optuna.pkl",            24),
    ("LGBM MIMO 24h",             "lgbm_hourly_load_mimo_h24.pkl",                  24),
    ("RF MIMO 24h",               "rf_hourly_load_mimo_h24.pkl",                    24),
    ("RF MIMO 24h Optuna",        "rf_mimo_hourly_load_h24_optuna.pkl",             24),
    ("SVR MIMO 24h",              "svr_hourly_load_mimo_h24.pkl",                   24),
    ("MLP MIMO 24h",              "mlp_hourly_load_mimo_h24.pkl",                   24),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _mae(yt, yp) -> float:
    a, b = _arr(yt), _arr(yp)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() > 0 else float("nan")

def _rmse(yt, yp) -> float:
    a, b = _arr(yt), _arr(yp)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() > 0 else float("nan")

def _smape(yt, yp) -> float:
    a, b = _arr(yt), _arr(yp)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    denom = np.abs(a[m]) + np.abs(b[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(a[m] - b[m]) / denom))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _naive1_monthly(df_y: pd.Series, test_index: pd.DatetimeIndex,
                    daily_origins: List[pd.Timestamp], freq) -> np.ndarray:
    preds = np.full(len(test_index), np.nan, dtype=float)
    t_map = {ts: i for i, ts in enumerate(test_index)}
    for origin in daily_origins:
        y_hist = df_y.loc[:origin].dropna()
        if len(y_hist) == 0:
            continue
        last_val = float(y_hist.iloc[-1])
        pred_idx = pd.date_range(origin + freq, periods=24, freq=freq)
        for ts in pred_idx:
            if ts in t_map:
                preds[t_map[ts]] = last_val
    return preds


def _seasonal_naive_monthly(df_y: pd.Series, test_index: pd.DatetimeIndex,
                             daily_origins: List[pd.Timestamp], freq,
                             season: int) -> np.ndarray:
    preds = np.full(len(test_index), np.nan, dtype=float)
    t_map = {ts: i for i, ts in enumerate(test_index)}
    for origin in daily_origins:
        y_hist = df_y.loc[:origin].dropna()
        if len(y_hist) == 0:
            continue
        pred_idx = pd.date_range(origin + freq, periods=24, freq=freq)
        last_val = float(y_hist.iloc[-1])
        for k, ts in enumerate(pred_idx):
            if ts not in t_map:
                continue
            src = ts - season * freq
            val = float(y_hist.loc[src]) if src in y_hist.index else last_val
            preds[t_map[ts]] = val
    return preds


def _seasonal_profile_monthly(df_y: pd.Series, test_index: pd.DatetimeIndex,
                               train_y: pd.Series) -> np.ndarray:
    prof = (
        pd.DataFrame({"y": train_y.values, "dow": train_y.index.dayofweek,
                       "hour": train_y.index.hour}, index=train_y.index)
        .groupby(["dow", "hour"])["y"].mean()
    )
    global_mean = float(train_y.mean()) if len(train_y) > 0 else 0.0
    out = []
    for ts in test_index:
        key = (int(ts.dayofweek), int(ts.hour))
        out.append(float(prof.get(key, global_mean)))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _get_feat_cols(bundle: dict, X_ref: pd.DataFrame) -> Optional[List[str]]:
    """Infer usable feature columns from bundle metadata."""
    fcols = bundle.get("feature_cols")
    if fcols:
        missing = [c for c in fcols if c not in X_ref.columns]
        if missing:
            return None
        return list(fcols)
    model = bundle.get("model")
    if model is None:
        return None
    fni = getattr(model, "feature_names_in_", None)
    if fni is not None:
        cols = list(fni)
        return cols if all(c in X_ref.columns for c in cols) else None
    nfi = getattr(model, "n_features_in_", None)
    if nfi and int(nfi) == X_ref.shape[1]:
        return list(X_ref.columns)
    return None


def _single_mimo_predict(
    bundle: dict,
    origin: pd.Timestamp,
    H: int,
    X_task: pd.DataFrame,
    feat_cols: Optional[List[str]],
    X_load: Optional[pd.DataFrame] = None,
) -> Optional[np.ndarray]:
    """Predict H steps from a single origin. Returns ndarray(H,) or None."""
    has_load = bundle.get("has_load_preds", False)
    freq = pd.tseries.frequencies.to_offset("h")

    def _snap(origin_ts, X_df):
        if origin_ts in X_df.index:
            return origin_ts
        cand = X_df.index[X_df.index <= origin_ts]
        return cand[-1] if len(cand) > 0 else None

    if has_load:
        load_model    = bundle.get("load_model")
        lf_cols: List[str] = bundle.get("load_feature_cols", [])
        pf_cols: List[str] = bundle.get("price_feature_cols", [])
        lp_cols: List[str] = bundle.get("load_pred_cols",
                                         [f"load_pred_h{i+1:02d}" for i in range(24)])
        if load_model is None or X_load is None:
            return None

        orig_l = _snap(origin, X_load)
        if orig_l is None:
            return None
        X_load_row = np.array(
            [float(X_load.loc[orig_l, c]) if c in X_load.columns else 0.0
             for c in lf_cols], dtype=np.float32
        )
        import joblib as _jl
        with _jl.parallel_backend("threading"):
            load_preds = np.asarray(
                load_model.predict(X_load_row.reshape(1, -1))[0], dtype=float
            )

        orig_p = _snap(origin, X_task)
        if orig_p is None:
            return None
        X_price_row = np.array(
            [float(X_task.loc[orig_p, c]) if c in X_task.columns else 0.0
             for c in pf_cols], dtype=np.float32
        )
        X_aug = np.concatenate([X_price_row,
                                 load_preds[:len(lp_cols)].astype(np.float32)
                                 ]).reshape(1, -1)
        # Apply scaler if present (SVR / MLP two-stage bundles)
        scaler = bundle.get("scaler")
        if scaler is not None:
            X_aug = scaler.transform(X_aug).astype(np.float32)
        with _jl.parallel_backend("threading"):
            raw = bundle["model"].predict(X_aug)
        return np.asarray(raw, dtype=float).reshape(-1)[:H]

    else:
        if feat_cols is None:
            return None
        orig = _snap(origin, X_task)
        if orig is None:
            return None
        x_row = np.array(
            [float(X_task.loc[orig, c]) if c in X_task.columns else 0.0
             for c in feat_cols], dtype=np.float32
        ).reshape(1, -1)
        # Apply scaler if present (SVR / MLP bundles store StandardScaler)
        scaler = bundle.get("scaler")
        if scaler is not None:
            x_row = scaler.transform(x_row).astype(np.float32)
        # Force n_jobs=1 on MOR to avoid thread exhaustion when many jobs run concurrently
        m = bundle["model"]
        if hasattr(m, "n_jobs") and m.n_jobs != 1:
            m.n_jobs = 1
        raw = m.predict(x_row)
        return np.asarray(raw, dtype=float).reshape(-1)[:H]


def _predict_all_origins(
    bundle: dict,
    origins: List[pd.Timestamp],
    chunk_hours: int,
    test_index: pd.DatetimeIndex,
    X_task: pd.DataFrame,
    feat_cols: Optional[List[str]],
    X_load: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Run MIMO prediction for all origins; fill result array aligned to test_index."""
    preds = np.full(len(test_index), np.nan, dtype=float)
    t_map = {ts: i for i, ts in enumerate(test_index)}
    freq  = pd.tseries.frequencies.to_offset("h")

    for origin in origins:
        yhat = _single_mimo_predict(bundle, origin, chunk_hours, X_task, feat_cols, X_load)
        if yhat is None:
            continue
        pred_idx = pd.date_range(origin + freq, periods=chunk_hours, freq=freq)
        for k, ts in enumerate(pred_idx):
            if ts in t_map:
                preds[t_map[ts]] = yhat[k]
    return preds


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _hdr(cols, widths):
    parts = [c.ljust(w) if i == 0 else c.rjust(w)
             for i, (c, w) in enumerate(zip(cols, widths))]
    line = "  ".join(parts)
    print(line)
    print("-" * len(line))

def _row(vals, widths):
    parts = ([str(vals[0]).ljust(widths[0])] +
             [str(v).rjust(w) for v, w in zip(vals[1:], widths[1:])])
    print("  ".join(parts))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_mimo_monthly(
    mode: str,
    task: str,
    train_end: str,
    test_start: str,
    test_end: str,
    save_json: bool = False,
    out_suffix_append: str = "",
) -> None:
    unit = "€/MWh" if task == "price" else "MW"
    freq = pd.tseries.frequencies.to_offset("h")

    # ── Load data ─────────────────────────────────────────────────────────────
    df_task = load_processed(mode, task=task)
    df_train, df_test = split_time_series(
        df_task, mode=mode, train_end=train_end,
        test_start=test_start, test_end=test_end,
    )
    X_task_full, _ = make_xy(df_task)   # full (train+test) for feature look-up
    _, y_test      = make_xy(df_test)
    y_test         = np.asarray(y_test, dtype=float)
    test_index     : pd.DatetimeIndex = df_test.index
    df_y           = df_task["y"].astype(float)
    train_y        = df_train["y"].astype(float)

    # Augment X_task_full with dense intraday lags (y_lag4..y_lag23)
    # needed by LGBM Direct Dense models; safe to add (no-op if already present)
    _existing_lag_nums: set = set()
    for _c in X_task_full.columns:
        if _c.startswith("y_lag"):
            try:
                _existing_lag_nums.add(int(_c[len("y_lag"):]))
            except ValueError:
                pass
    for _lag in range(1, 24):
        if _lag not in _existing_lag_nums:
            X_task_full[f"y_lag{_lag}"] = df_y.shift(_lag).reindex(X_task_full.index)

    # Load dataset for two-stage models (price task only)
    X_load_full: Optional[pd.DataFrame] = None
    if task == "price":
        try:
            df_load = load_processed(mode, task="load")
            X_load_full, _ = make_xy(df_load)
        except Exception as exc:
            print(f"[WARN] Cannot load load dataset for two-stage: {exc}", flush=True)

    n_hours = len(test_index)
    te      = pd.Timestamp(train_end)

    print(f"\n{'='*70}", flush=True)
    print(f"  MIMO Monthly Eval | task={task.upper()} | "
          f"{test_start[:10]} → {test_end[:10]}", flush=True)
    print(f"  {n_hours}h = {n_hours//24} days | {n_hours//168} full weeks", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Build origin lists ────────────────────────────────────────────────────
    # Daily origins: te, then each day boundary - 1h
    daily_origins: List[pd.Timestamp] = []
    ptr = 0
    while ptr < n_hours:
        if ptr == 0:
            daily_origins.append(te)
        else:
            daily_origins.append(test_index[ptr - 1])
        ptr += 24

    # Weekly origins
    weekly_origins: List[pd.Timestamp] = []
    ptr = 0
    while ptr < n_hours:
        if ptr == 0:
            weekly_origins.append(te)
        else:
            weekly_origins.append(test_index[ptr - 1])
        ptr += 168

    print(f"  daily origins: {len(daily_origins)} | "
          f"weekly origins: {len(weekly_origins)}", flush=True)

    # ── Load and run models ───────────────────────────────────────────────────
    registry = MODELS_PRICE if task == "price" else MODELS_LOAD
    results: List[Tuple[str, np.ndarray, int]] = []   # (label, preds, chunk_h)

    for label, fname, chunk_hours in registry:
        mp = MODELS_DIR / fname
        if not mp.exists():
            print(f"[SKIP] {label}", flush=True)
            continue
        try:
            bundle = joblib.load(mp)
            if not isinstance(bundle, dict):
                bundle = {"model": bundle, "feature_cols": None}
        except Exception as exc:
            print(f"[SKIP] {label}: {exc}", flush=True)
            continue

        has_load = bundle.get("has_load_preds", False)
        if has_load:
            feat_cols = None
        else:
            feat_cols = _get_feat_cols(bundle, X_task_full)
            if feat_cols is None:
                print(f"[SKIP] {label}: feature mismatch", flush=True)
                continue

        origins = daily_origins if chunk_hours == 24 else weekly_origins
        print(f"[INFO] {label}: {len(origins)} chunks ...", flush=True)
        preds = _predict_all_origins(
            bundle, origins, chunk_hours, test_index,
            X_task_full, feat_cols, X_load_full,
        )
        mae_v = _mae(y_test, preds)
        print(f"   MAE={mae_v:.3f} {unit}", flush=True)
        results.append((label, preds, chunk_hours))

    if not results:
        print("[ERROR] No models found. Train H=24 MIMO models first.", flush=True)
        return

    # ── Ensemble (1/MAE) of 24h ML models ────────────────────────────────────
    ml24 = [(l, p) for l, p, ch in results if ch == 24]
    if len(ml24) >= 2:
        maes_ = [_mae(y_test, p) for _, p in ml24]
        valid = [(l, p, m) for (l, p), m in zip(ml24, maes_)
                 if np.isfinite(m) and m > 0]
        if len(valid) >= 2:
            ws = np.array([1.0 / m for _, _, m in valid])
            ws /= ws.sum()
            ens_pred = sum(w * p for w, (_, p, _) in zip(ws, valid))
            results.append(("Ensemble 24h (1/MAE)", ens_pred, 24))
            print(f"[ENS] Ensemble-24h {len(valid)} models, "
                  f"weights={[round(float(w),3) for w in ws]}", flush=True)

    # ── Baselines ─────────────────────────────────────────────────────────────
    baselines = [
        ("NAIVE-1",          _naive1_monthly(df_y, test_index, daily_origins, freq)),
        ("NAIVE-24",         _seasonal_naive_monthly(df_y, test_index, daily_origins, freq, 24)),
        ("NAIVE-168",        _seasonal_naive_monthly(df_y, test_index, daily_origins, freq, 168)),
        ("SEASONAL PROFILE", _seasonal_profile_monthly(df_y, test_index, train_y)),
    ]
    for bl_label, bl_preds in baselines:
        results.append((bl_label, bl_preds, 24))

    all_preds_sorted = sorted(results, key=lambda x: _mae(y_test, x[1]))

    # ── Overall table ─────────────────────────────────────────────────────────
    W = [38, 12, 12, 10, 7]
    COLS = ["Strategy", f"MAE ({unit})", f"RMSE ({unit})", "sMAPE (%)", "Chunks"]
    print(f"\n{'='*84}", flush=True)
    print(f"  MONTHLY RESULTS | task={task.upper()} | "
          f"{test_start[:10]} → {test_end[:10]}", flush=True)
    print(f"{'='*84}", flush=True)
    _hdr(COLS, W)
    for lab, preds, ch in all_preds_sorted:
        _row([lab, f"{_mae(y_test,preds):.3f}",
              f"{_rmse(y_test,preds):.3f}", f"{_smape(y_test,preds):.3f}",
              f"{ch}h"], W)

    # ── Per-week breakdown ─────────────────────────────────────────────────────
    labels_all = [l for l, _, _ in results]
    Ww = [13, 5] + [10] * len(results)
    col_w = ["Week", "Days"] + [l[:9] for l in labels_all]
    print(f"\n  PER-WEEK BREAKDOWN", flush=True)
    _hdr(col_w, Ww)
    ptr, wk = 0, 1
    while ptr < n_hours:
        chunk = test_index[ptr : ptr + 168]
        mask  = np.zeros(n_hours, dtype=bool)
        mask[ptr : ptr + len(chunk)] = True
        yt_w  = y_test[mask]
        wlabel = f"W{wk} {chunk[0].strftime('%b %d')}"
        dstr   = f"{len(chunk)//24}d"
        maes_ = [f"{_mae(yt_w, p[mask]):.3f}" for _, p, _ in results]
        _row([wlabel, dstr] + maes_, Ww)
        ptr += 168
        wk += 1

    # ── Per-day breakdown ──────────────────────────────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days = sorted(set(test_index.normalize()))
    Wd = [13, 5] + [10] * len(results)
    col_d = ["Date", "DoW"] + [l[:9] for l in labels_all]
    print(f"\n  PER-DAY BREAKDOWN (* = best ML)", flush=True)
    _hdr(col_d, Wd)
    bl_labels = {l for l, _ in baselines}
    for day in days:
        mask   = test_index.normalize() == day
        yt_d   = y_test[mask]
        day_m  = [_mae(yt_d, p[mask]) for _, p, _ in results]
        ml_idx = [i for i, (l, _, _) in enumerate(results) if l not in bl_labels]
        best_i = min(ml_idx, key=lambda i: day_m[i]) if ml_idx else -1
        strs   = [f"{v:.3f}{'*' if i==best_i else ' '}"
                  for i, v in enumerate(day_m)]
        _row([day.strftime("%Y-%m-%d"), dow_names[day.dayofweek]] + strs, Wd)

    # ── Summary ────────────────────────────────────────────────────────────────
    ml_only = [(l, _mae(y_test, p)) for l, p, _ in results if l not in bl_labels]
    if ml_only:
        winner = min(ml_only, key=lambda x: x[1])
        print(f"\n  Winner: {winner[0]}  MAE={winner[1]:.3f} {unit}", flush=True)
    print(f"{'='*84}\n", flush=True)

    # ── JSON ───────────────────────────────────────────────────────────────────
    if save_json:
        _save_json(task, test_index, y_test, results, out_suffix_append=out_suffix_append)


# ---------------------------------------------------------------------------
# JSON saving
# ---------------------------------------------------------------------------

def _fix_nan(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _fix_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_nan(x) for x in obj]
    return obj


BL_LABELS = {"NAIVE-1", "NAIVE-24", "NAIVE-168", "SEASONAL PROFILE"}


def _model_type_of(label: str) -> str:
    """Classify a model label as 'lgbm' (→ Direct tab) or algo family (→ MIMO tab)."""
    u = label.upper()
    if "LGBM" in u:
        return "lgbm"
    if "XGB" in u:
        return "xgb"
    if "RF" in u or "RANDOM" in u:
        return "rf"
    if "SVR" in u:
        return "svr"
    if "MLP" in u or "NEURAL" in u:
        return "mlp"
    return "other"


def _is_mimo_model(label: str) -> bool:
    """Returns True only for genuine MIMO models (not Direct architecture).
    DIRECT models belong only to the Direct panel, not the MIMO panel.
    """
    return "DIRECT" not in label.upper()


def _best_per_mimo_family(
    entries: List[Tuple[str, np.ndarray, int]],
    y_test: np.ndarray,
) -> List[Tuple[str, np.ndarray, int]]:
    """Keep only the best (lowest MAE) model per algorithm family for the MIMO tab.
    This ensures exactly 1 XGB, 1 RF, 1 SVR, 1 MLP curve in the dashboard.
    """
    family_best: Dict[str, Tuple[str, np.ndarray, int, float]] = {}
    for lab, preds, ch in entries:
        fam = _model_type_of(lab)
        mae_v = _mae(y_test, preds)
        if not np.isfinite(mae_v):
            continue
        if fam not in family_best or mae_v < family_best[fam][3]:
            family_best[fam] = (lab, preds, ch, mae_v)
    return [(lab, preds, ch) for lab, preds, ch, _ in family_best.values()]


def _save_json_filtered(
    task: str,
    test_index: pd.DatetimeIndex,
    y_test: np.ndarray,
    results: List[Tuple[str, np.ndarray, int]],
    horizon_filter: int,        # 24 or 168
    model_type_filter: str,     # "lgbm" → Direct tab, "mimo" → MIMO tab
    strategy_name: str,         # "direct" or "mimo"
    out_suffix: str,            # e.g. "direct_h24_monthly"
) -> None:
    """Save filtered subset.
    For MIMO tab: all non-LGBM models, best-per-family applied.
    For Direct tab: LGBM models only.
    """
    dates  = [str(ts) for ts in test_index]
    actual = [float(v) for v in y_test]

    baselines = [(l, p, ch) for l, p, ch in results if l in BL_LABELS]
    ml_all    = [(l, p, ch) for l, p, ch in results if l not in BL_LABELS]

    # Filter by horizon + model type
    if model_type_filter == "mimo":
        # MIMO tab: ALL model families at given horizon → best-per-family
        ml_filt = [(l, p, ch) for l, p, ch in ml_all
                   if ch == horizon_filter and _is_mimo_model(l)]
        # Apply best-per-family: 1 LGBM + 1 XGB + 1 RF + 1 SVR + 1 MLP
        ml_filt = _best_per_mimo_family(ml_filt, y_test)
    else:
        # Direct tab: LGBM + XGB DIRECT models at given horizon → best-per-family
        ml_filt = [(l, p, ch) for l, p, ch in ml_all
                   if ch == horizon_filter and "DIRECT" in l.upper()]
        # Apply best-per-family: best LGBM variant + best XGB DIRECT variant
        ml_filt = _best_per_mimo_family(ml_filt, y_test)

    if not ml_filt:
        print(f"[SKIP/{out_suffix}] no models (type={model_type_filter}, h={horizon_filter})",
              flush=True)
        return

    # Ensemble All (1/MAE weighted, ≥2 models)
    ensemble_entries: List[Tuple[str, np.ndarray, int]] = []
    maes_all = [_mae(y_test, p) for _, p, _ in ml_filt]
    valid_all = [(l, p, ch, m) for (l, p, ch), m in zip(ml_filt, maes_all)
                 if np.isfinite(m) and m > 0]
    if len(valid_all) >= 2:
        ws = np.array([1.0 / m for _, _, _, m in valid_all])
        ws /= ws.sum()
        ens_all = sum(w * p for w, (_, p, _, _) in zip(ws, valid_all))
        ensemble_entries.append((f"Ensemble All {horizon_filter}h (1/MAE)", ens_all, horizon_filter))
        # Ensemble Best3
        if len(valid_all) >= 3:
            top3 = sorted(valid_all, key=lambda x: x[3])[:3]
            ws3 = np.array([1.0 / m for _, _, _, m in top3]); ws3 /= ws3.sum()
            ens_top3 = sum(w * p for w, (_, p, _, _) in zip(ws3, top3))
            ensemble_entries.append((f"Ensemble Best3 {horizon_filter}h (1/MAE)", ens_top3, horizon_filter))

    kept = ml_filt + ensemble_entries + baselines
    kept_sorted = sorted(kept, key=lambda x: _mae(y_test, x[1]))

    print(f"\n[JSON/{out_suffix.upper()}] {len(kept_sorted)} series "
          f"(type={model_type_filter}, h={horizon_filter}):")
    for lab, preds, ch in kept_sorted:
        if lab not in BL_LABELS:
            print(f"   {lab:55s}  MAE={_mae(y_test,preds):.3f}")

    series: Dict[str, List] = {}
    for lab, preds, _ in kept_sorted:
        series[lab] = [float(v) if np.isfinite(v) else None for v in preds]

    metrics_out = []
    for lab, preds, ch in kept_sorted:
        m = _mae(y_test, preds)
        if np.isfinite(m):
            mtype = ("baseline" if lab in BL_LABELS
                     else ("ensemble" if "ensemble" in lab.lower() else "ml"))
            metrics_out.append({
                "Model":       lab,
                "Type":        mtype,
                "chunk_hours": ch,
                "MAE":         round(m, 4),
                "RMSE":        round(_rmse(y_test, preds), 4),
                "sMAPE":       round(_smape(y_test, preds), 4),
            })

    out = {
        "strategy": strategy_name,
        "task":     task,
        "dates":    dates,
        "actual":   actual,
        "series":   series,
        "metrics":  metrics_out,
    }
    out_path = ROOT / f"dashboard_data_hourly_{task}_{out_suffix}.json"
    out_path.write_text(json.dumps(_fix_nan(out), indent=2), encoding="utf-8")
    print(f"✅ JSON saved: {out_path.name}", flush=True)


def _save_json(task: str, test_index: pd.DatetimeIndex,
               y_test: np.ndarray,
               results: List[Tuple[str, np.ndarray, int]],
               out_suffix_append: str = "") -> None:
    """Save TWO JSONs per task (H=168 removed from dashboard per user request):
      direct_h24_monthly  — LGBM Direct H=24, best-per-family (1 LGBM variant)
      mimo_h24_monthly    — MIMO H=24: ALL families (LGBM+XGB+RF+SVR+MLP),
                            best-per-family + ensembles + baselines
    out_suffix_append: extra string appended to JSON filenames (e.g. "_q1_2026")
    """
    sfx = out_suffix_append  # e.g. "" or "_q1_2026"
    # ── Direct tab (LGBM H=24) ────────────────────────────────────────────────
    _save_json_filtered(task, test_index, y_test, results,
                        horizon_filter=24,  model_type_filter="lgbm",
                        strategy_name="direct", out_suffix=f"direct_h24_monthly{sfx}")
    # ── MIMO tab (XGB + RF + SVR + MLP, H=24, best-per-family) ───────────────
    _save_json_filtered(task, test_index, y_test, results,
                        horizon_filter=24,  model_type_filter="mimo",
                        strategy_name="mimo", out_suffix=f"mimo_h24_monthly{sfx}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="1-month MIMO evaluation: H=24 (daily) vs H=168 (weekly)"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",       choices=["price", "load"], default="price")
    p.add_argument("--train_end",  default="2025-11-30 23:00")
    p.add_argument("--test_start", default="2025-12-01 00:00")
    p.add_argument("--test_end",   default="2025-12-31 23:00")
    p.add_argument("--save_json",  action="store_true")
    p.add_argument("--out_suffix", default="",
                   help="Appended to JSON filenames, e.g. '_q1_2026' -> mimo_h24_monthly_q1_2026.json")
    args = p.parse_args()

    evaluate_mimo_monthly(
        mode=args.mode,
        task=args.task,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        save_json=args.save_json,
        out_suffix_append=args.out_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
