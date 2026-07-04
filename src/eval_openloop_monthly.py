"""
eval_openloop_monthly.py

1-month evaluation comparing two prediction strategies on Dec 2025 (or any period):

  FULL-WEEK  model (LGBM-OL-Optuna):
    → split into N×7-day (168h) chunks, each chunk is a fresh recursive run
    → models the real-world scenario: "retrain / re-forecast every Monday"

  DAY-BY-DAY model (LGBM-OL-Daily-Optuna):
    → split into M×1-day (24h) chunks, each chunk is a fresh recursive run
    → models the real-world scenario: "day-ahead market re-forecast every day"

Both use the SAME trained models (no re-training during the evaluation period).
The models are fixed; only the recursive reset boundary changes.

Usage:
  conda run -n epf --no-capture-output python -m src.eval_openloop_monthly hourly \\
      --task price \\
      --train_end "2025-11-30 23:00" \\
      --test_start "2025-12-01 00:00" \\
      --test_end "2025-12-31 23:00"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
import warnings
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop
from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ── Unpickle aliases ──────────────────────────────────────────────────────────
class ResidualAddBaselineWrapper:
    def __init__(self, model, baseline_col: str, feature_names: list):
        self.model = model
        self.baseline_col = baseline_col
        self.feature_names = list(feature_names)
        self.baseline_idx = self.feature_names.index(baseline_col)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        residual_hat = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        if isinstance(X, pd.DataFrame):
            base = X[self.baseline_col].to_numpy(dtype=float).reshape(-1)
        else:
            base = np.asarray(X)[:, self.baseline_idx].astype(float).reshape(-1)
        return base + residual_hat


def _register_aliases():
    for name in ["src.model_wrappers", "src.train_xgb_openloop", "src.eval_openloop"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], "ResidualAddBaselineWrapper", ResidualAddBaselineWrapper)


# ── Metrics ───────────────────────────────────────────────────────────────────
def _fnp(a) -> np.ndarray:
    return np.asarray(a, dtype=float)

def mae(y_true, y_pred) -> float:
    yt, yp = _fnp(y_true), _fnp(y_pred)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yt[m] - yp[m]))) if m.sum() > 0 else float("nan")

def rmse(y_true, y_pred) -> float:
    yt, yp = _fnp(y_true), _fnp(y_pred)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2))) if m.sum() > 0 else float("nan")

def smape(y_true, y_pred) -> float:
    yt, yp = _fnp(y_true), _fnp(y_pred)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return float("nan")
    denom = np.abs(yt[m]) + np.abs(yp[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(yt[m] - yp[m]) / denom))


# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_CONFIG = [
    # (key, label, chunk_hours, filename_pattern, is_dense)
    # ── Weekly baseline (original OL-Optuna, run as 168h chunks) ──
    ("lgbm_fw",    "LGBM-OL-Optuna (weekly)",          168, "lgbm_{mode}_{task}_openloop_optuna.pkl",                     False),
    ("xgb_fw",     "XGB-OL-Optuna (weekly)",           168, "xgb_{mode}_{task}_openloop_optuna.pkl",                      False),
    # ── Daily-Optuna only (no SS) ──
    ("lgbm_do",    "LGBM-Daily-Optuna (24h)",           24, "lgbm_{mode}_{task}_openloop_daily_optuna.pkl",               False),
    ("lgbm_do_d",  "LGBM-Daily-Optuna Dense (24h)",     24, "lgbm_{mode}_{task}_openloop_daily_optuna_dense.pkl",         True),
    ("xgb_do",     "XGB-Daily-Optuna (24h)",            24, "xgb_{mode}_{task}_openloop_daily_optuna.pkl",                False),
    ("xgb_do_d",   "XGB-Daily-Optuna Dense (24h)",      24, "xgb_{mode}_{task}_openloop_daily_optuna_dense.pkl",          True),
    ("rf_do",      "RF-Daily-Optuna (24h)",             24, "rf_{mode}_{task}_openloop_daily_optuna.pkl",                 False),
    # ── Combined daily-SS-Optuna (joint optimisation) ──
    ("lgbm_joint", "LGBM-Daily-SS-Optuna (24h)",        24, "lgbm_{mode}_{task}_openloop_daily_ss_optuna.pkl",            False),
    ("lgbm_joint_d","LGBM-Daily-SS-Optuna Dense (24h)", 24, "lgbm_{mode}_{task}_openloop_daily_ss_optuna_dense.pkl",      True),
    ("xgb_joint",  "XGB-Daily-SS-Optuna (24h)",         24, "xgb_{mode}_{task}_openloop_daily_ss_optuna.pkl",             False),
    ("xgb_joint_d","XGB-Daily-SS-Optuna Dense (24h)",   24, "xgb_{mode}_{task}_openloop_daily_ss_optuna_dense.pkl",       True),
    ("rf_ss_d",    "RF-Daily-SS (24h)",                 24, "rf_{mode}_{task}_openloop_daily_optuna_ss.pkl",              False),
    # ── MLP & SVR best versions in 24h chunks ──
    ("mlp_ss",     "MLP-SS (daily 24h)",                24, "mlp_{mode}_{task}_scheduled_openloop.pkl",                  False),
    ("svr_ol",     "SVR-OL (daily 24h)",                24, "svr_{mode}_{task}_openloop.pkl",                            False),
]

def _get_model_path(pattern: str, mode: str, task: str) -> Path:
    return MODELS_DIR / pattern.format(mode=mode, task=task)


# ── Chunked recursive prediction ──────────────────────────────────────────────
def predict_chunked(
    model,
    df_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    feature_cols: List[str],
    chunk_hours: int,
    cfg: OpenLoopConfig,
) -> np.ndarray:
    """
    Split test_index into chunks of chunk_hours, run a fresh recursive
    prediction for each chunk (y_run resets to actual at each chunk start).
    Remaining hours (<chunk_hours) are run as a shorter chunk.
    """
    preds = np.zeros(len(test_index), dtype=float)
    ptr = 0
    n = len(test_index)

    while ptr < n:
        chunk_idx = test_index[ptr : ptr + chunk_hours]
        chunk_preds = _fnp(
            recursive_predict_openloop(
                model=model,
                df_full=df_full,       # always has actual y → fresh start each chunk
                test_index=chunk_idx,
                feature_cols=feature_cols,
                config=cfg,
            )
        )
        k = len(chunk_preds)
        preds[ptr : ptr + k] = chunk_preds
        ptr += k

    return preds


# ── Printing ──────────────────────────────────────────────────────────────────
def _hdr(cols, widths):
    parts = [c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cols, widths))]
    line = "  ".join(parts)
    print(line)
    print("-" * len(line))

def _row(vals, widths):
    parts = [str(vals[0]).ljust(widths[0])] + [str(v).rjust(w) for v, w in zip(vals[1:], widths[1:])]
    print("  ".join(parts))


# ── Best-per-family filtering ─────────────────────────────────────────────────
BASELINE_NAMES = {"Naive-1", "Naive-24", "Naive-168", "Seasonal Profile"}
FAMILY_PREFIXES = ["LGBM", "XGB", "RF", "MLP", "SVR"]


def _family_of(label: str):
    """Return algorithm family prefix or None."""
    up = label.upper()
    for p in FAMILY_PREFIXES:
        if up.startswith(p):
            return p
    return None


def _chunk_type(label: str) -> str:
    """Return 'daily' for 24h-chunk models, 'weekly' for 168h-chunk models."""
    ll = label.lower()
    if "weekly" in ll or "168h" in ll:
        return "weekly"
    return "daily"


def _best_per_family(results, y_test):
    """
    Keep best (lowest MAE) model per (algorithm family, chunk_type) pair.
    This preserves BOTH the best 24h and the best 168h model for each family,
    enabling direct comparison of daily vs weekly chunking strategies.
    Ensembles and baselines are always kept unchanged.
    Returns filtered list preserving original (key, label, chunk_hours, preds) tuples.
    """
    family_chunk_best: dict = {}   # (family, chunk_type) → (mae_val, entry)
    always_keep = []

    for entry in results:
        _, label, _, preds = entry
        m = mae(y_test, preds)
        if label in BASELINE_NAMES or "ensemble" in label.lower():
            always_keep.append(entry)
            continue
        fam = _family_of(label)
        if fam is None:
            always_keep.append(entry)
            continue
        chunk = _chunk_type(label)
        key = (fam, chunk)
        if key not in family_chunk_best or (math.isfinite(m) and m < family_chunk_best[key][0]):
            family_chunk_best[key] = (m, entry)

    filtered = [v[1] for v in family_chunk_best.values()] + always_keep
    # Sort: ML by MAE (daily first within same MAE), then ensembles, then baselines
    def _sort_key(e):
        _, label, _, preds = e
        if label in BASELINE_NAMES:
            return (2, mae(y_test, preds))
        if "ensemble" in label.lower():
            return (1, mae(y_test, preds))
        return (0, mae(y_test, preds))
    return sorted(filtered, key=_sort_key)


# ── JSON saving ───────────────────────────────────────────────────────────────
def _fix_nan(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _fix_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_nan(v) for v in obj]
    return obj


def _save_json_ol_filtered(
    task: str,
    test_index,
    y_test: np.ndarray,
    results,
    chunk_filter: int,   # 24 or 168
    out_suffix: str,     # e.g. "openloop_h24_monthly"
) -> None:
    """Save OL monthly results filtered to one chunk size (24h daily or 168h weekly)."""
    BASE_DIR2 = Path(__file__).resolve().parents[1]
    out_path = BASE_DIR2 / f"dashboard_data_hourly_{task}_{out_suffix}.json"
    unit = "€/MWh" if task == "price" else "MW"

    bl_entries  = [(k, l, ch, p) for k, l, ch, p in results if l in BASELINE_NAMES]
    ens_entries = [(k, l, ch, p) for k, l, ch, p in results if "ensemble" in l.lower()]
    ml_entries  = [(k, l, ch, p) for k, l, ch, p in results
                   if l not in BASELINE_NAMES and "ensemble" not in l.lower()]

    # Keep only ML models with matching chunk size
    ml_filt = [(k, l, ch, p) for k, l, ch, p in ml_entries if ch == chunk_filter]
    if not ml_filt:
        print(f"[SKIP/{out_suffix}] no models with chunk_hours={chunk_filter}", flush=True)
        return

    # ── Apply best-per-family: keep only 1 model per algorithm family ──────────
    # This removes duplicates like LGBM-Daily-Optuna + LGBM-Daily-SS-Optuna,
    # keeping only the one with the lower MAE.
    family_chunk_best: dict = {}
    for k, l, ch, p in ml_filt:
        fam = _family_of(l)
        if fam is None:
            fam = l  # keep as-is if unrecognised
        m = mae(y_test, p)
        chunk = _chunk_type(l)
        fc_key = (fam, chunk)
        if fc_key not in family_chunk_best or (math.isfinite(m) and m < family_chunk_best[fc_key][4]):
            family_chunk_best[fc_key] = (k, l, ch, p, m)
    ml_filt = [(k, l, ch, p) for k, l, ch, p, _ in family_chunk_best.values()]

    if chunk_filter == 24:
        # Use the pre-computed ensembles from ENSEMBLE_GROUPS (consistent with main table).
        # These use fixed model compositions (no SVR, both gradient-boost variants) and
        # match the values printed in the summary table.
        ens_use = [(k, l, ch, p) for k, l, ch, p in ens_entries if ch == chunk_filter]
        if not ens_use:
            # Fallback: build fresh from best-per-family if no pre-computed ensembles
            valid = [(k, l, p, mae(y_test, p)) for k, l, ch, p in ml_filt]
            valid = [(k, l, p, m) for k, l, p, m in valid if math.isfinite(m) and m > 0]
            ens_use = []
            if len(valid) >= 2:
                ws = np.array([1.0 / m for _, _, _, m in valid]); ws /= ws.sum()
                ens_p = sum(w * p for w, (_, _, p, _) in zip(ws, valid))
                ens_use.append(("ens_all24", "Ensemble-All-24h (1/MAE)", 24, ens_p))
            if len(valid) >= 3:
                top3 = sorted(valid, key=lambda x: x[3])[:3]
                ws3 = np.array([1.0 / m for _, _, _, m in top3]); ws3 /= ws3.sum()
                ens_top3 = sum(w * p for w, (_, _, p, _) in zip(ws3, top3))
                ens_use.append(("ens_top3_24", "Ensemble-Top3 (1/MAE)", 24, ens_top3))
    else:
        # Build a weekly ensemble from the best-per-family 168h models
        valid = [(l, p, mae(y_test, p)) for _, l, _, p in ml_filt]
        valid = [(l, p, m) for l, p, m in valid if math.isfinite(m) and m > 0]
        ens_use = []
        if len(valid) >= 2:
            ws = np.array([1.0 / m for _, _, m in valid]); ws /= ws.sum()
            ens_p = sum(w * p for w, (_, p, _) in zip(ws, valid))
            ens_use = [("ens_weekly", "Ensemble Weekly 168h (1/MAE)", 168, ens_p)]

    kept = ml_filt + ens_use + bl_entries
    kept_sorted = sorted(kept, key=lambda x: mae(y_test, x[3]))

    print(f"[JSON/{out_suffix.upper()}] {len(kept_sorted)} series (chunk={chunk_filter}h):")
    for _, l, _, p in kept_sorted:
        if l not in BASELINE_NAMES:
            print(f"   {l:50s}  MAE={mae(y_test,p):.3f}")

    dates  = [str(ts) for ts in test_index]
    actual = [float(v) if math.isfinite(float(v)) else None for v in y_test]
    series = {}
    for _, l, _, p in kept_sorted:
        series[l] = [float(v) if math.isfinite(float(v)) else None for v in p]

    metrics_out = []
    for _, l, ch, p in kept_sorted:
        m_val = mae(y_test, p)
        r_val = rmse(y_test, p)
        s_val = smape(y_test, p)
        mtype = ("baseline" if l in BASELINE_NAMES
                 else ("ensemble" if "ensemble" in l.lower() else "ml"))
        metrics_out.append({
            "Model": l, "Type": mtype, "chunk_hours": ch,
            "MAE":   round(m_val, 4) if math.isfinite(m_val) else None,
            "RMSE":  round(r_val, 4) if math.isfinite(r_val) else None,
            "sMAPE": round(s_val, 4) if math.isfinite(s_val) else None,
        })

    payload = {
        "strategy": "ol",
        "task": task,
        "period": "month",
        "chunk_hours": chunk_filter,
        "dates": dates,
        "actual": actual,
        "series": series,
        "metrics": metrics_out,
        "unit": unit,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_fix_nan(payload), f, ensure_ascii=False, indent=None)
    print(f"✅ JSON saved: {out_path.name}", flush=True)


def _save_json_ol(task: str, test_index, y_test, results, outer_suffix: str = ""):
    """Save TWO OL monthly JSONs: one for 24h daily chunks, one for 168h weekly chunks.
    outer_suffix: appended at the end, e.g. '_q4' → openloop_h24_monthly_q4.json
    """
    _save_json_ol_filtered(task, test_index, y_test, results, 24,  f"openloop_h24_monthly{outer_suffix}")
    _save_json_ol_filtered(task, test_index, y_test, results, 168, f"openloop_h168_monthly{outer_suffix}")


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate_monthly(
    mode: str,
    task: str,
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    save_json: bool = False,
    out_suffix: str = "",
) -> None:
    _register_aliases()

    df = load_processed(mode, task=task)
    df_train, df_test = split_time_series(
        df, mode=mode, test_size=None, train_start=None,
        train_end=train_end, test_start=test_start, test_end=test_end,
    )
    X_train, _ = make_xy(df_train)
    _, y_test  = make_xy(df_test)
    y_test       = _fnp(y_test)
    test_index   = df_test.index
    feature_cols = list(X_train.columns)
    cfg          = OpenLoopConfig(y_floor=None)
    unit         = "€/MWh" if task == "price" else "MW"

    # ── Pre-compute dense-lag feature set (for _dense models) ──────────────────
    # Add y_lag4..y_lag23 to df (values will be overridden by recursive engine anyway,
    # but df_full must have the columns so recursive_predict_openloop doesn't KeyError).
    existing_lag_nums = {int(c[5:]) for c in feature_cols if c.startswith("y_lag") and c[5:].isdigit()}
    df_dense = df.copy()
    y_full_series = df_dense["y"].astype(float)
    for lag in range(1, 24):
        col = f"y_lag{lag}"
        if lag not in existing_lag_nums:
            df_dense[col] = y_full_series.shift(lag)
    feature_cols_dense = list(feature_cols)
    for lag in range(1, 24):
        col = f"y_lag{lag}"
        if lag not in existing_lag_nums:
            feature_cols_dense.append(col)

    n_days  = len(test_index) // 24
    n_weeks = len(test_index) // 168
    rem_h   = len(test_index) % 168

    print(f"\n[INFO] task={task.upper()} | features={len(feature_cols)} (dense: {len(feature_cols_dense)})")
    print(f"[INFO] test = {test_index.min()} → {test_index.max()}")
    print(f"[INFO] {len(test_index)}h = {n_days} days = {n_weeks} full weeks + {rem_h}h remainder\n")

    # ── Load models and run predictions ──
    results = []   # list of (key, label, chunk_hours, preds)
    for key, label, chunk_hours, fname_pat, is_dense in MODELS_CONFIG:
        mp = _get_model_path(fname_pat, mode, task)
        if not mp.exists():
            print(f"[SKIP] {label} — model not found: {mp.name}")
            continue
        print(f"[INFO] Loading {label} from {mp.name}")
        model = joblib.load(mp)
        chunk_desc = f"{n_weeks} weekly" if chunk_hours == 168 else f"{n_days} daily"
        print(f"   → Running {chunk_desc} chunks ({chunk_hours}h each) ...")
        # Dense models need augmented df + feature_cols
        _df  = df_dense       if is_dense else df
        _fc  = feature_cols_dense if is_dense else feature_cols
        preds = predict_chunked(model, _df, test_index, _fc, chunk_hours=chunk_hours, cfg=cfg)
        results.append((key, label, chunk_hours, preds))

    if not results:
        raise RuntimeError("No models found. Train them first.")

    # ── Compute ensembles (oracle 1/MAE weights on full test set) ──────────────
    loaded_keys = {key for key, _, _, _ in results}
    preds_by_key = {key: preds for key, _, _, preds in results}

    ENSEMBLE_GROUPS = [
        # (ensemble_label, [keys_to_include])
        ("Ensemble-Top3 1/MAE (LGBM+RF+XGB)",
         ["lgbm_do", "rf_ss_d", "xgb_joint"]),
        ("Ensemble-All-24h 1/MAE",
         ["lgbm_do", "xgb_do", "lgbm_joint", "xgb_joint", "rf_ss_d", "mlp_ss"]),
    ]

    for ens_label, ens_keys in ENSEMBLE_GROUPS:
        avail = [k for k in ens_keys if k in loaded_keys]
        if len(avail) < 2:
            print(f"[SKIP ENSEMBLE] {ens_label} — need ≥2 models, got {len(avail)}")
            continue
        ens_preds_list = [preds_by_key[k] for k in avail]
        ens_maes       = [mae(y_test, p) for p in ens_preds_list]
        weights        = np.array([1.0 / m for m in ens_maes])
        weights       /= weights.sum()
        ens_preds      = sum(w * p for w, p in zip(weights, ens_preds_list))
        results.append((f"ens_{ens_label[:6]}", ens_label, 24, ens_preds))
        print(f"[ENS] {ens_label} — n={len(avail)} models, weights={[f'{w:.3f}' for w in weights]}")

    print()

    # ── Overall summary ──
    W = [46, 12, 12, 10]
    COLS = ["Strategy", f"MAE ({unit})", f"RMSE ({unit})", "sMAPE (%)"]
    print(f"\n{'='*84}")
    print(f"  MONTHLY COMPARISON | task={task.upper()} | {test_start[:10]} → {test_end[:10]}")
    print(f"{'='*84}")
    _hdr(COLS, W)
    for _, label, _, preds in results:
        _row([label, f"{mae(y_test,preds):.3f}", f"{rmse(y_test,preds):.3f}", f"{smape(y_test,preds):.3f}"], W)
    print()

    # ── Per-week breakdown ──
    col_headers = ["Week", "Days"] + [f"MAE-{l[:8]}" for _, l, _, _ in results]
    W3 = [22, 5] + [10] * len(results)
    print("  PER-WEEK BREAKDOWN")
    _hdr(col_headers, W3)
    ptr = 0
    week_num = 1
    while ptr < len(test_index):
        chunk = test_index[ptr : ptr + 168]
        mask  = np.zeros(len(test_index), dtype=bool)
        mask[ptr : ptr + len(chunk)] = True
        yt_w  = y_test[mask]
        week_label = (f"Week {week_num} ({chunk[0].strftime('%b %d')}–{chunk[-1].strftime('%b %d')})")
        days_str   = f"{len(chunk)//24}d" if len(chunk) % 24 == 0 else f"{len(chunk)}h"
        maes = [f"{mae(yt_w, p[mask]):.3f}" for _, _, _, p in results]
        _row([week_label, days_str] + maes, W3)
        ptr += len(chunk)
        week_num += 1
    print()

    # ── Per-day breakdown ──
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    days = sorted(set(test_index.normalize()))
    col_headers2 = ["Date", "DoW"] + [f"MAE-{l[:8]}" for _, l, _, _ in results]
    W4 = [13, 5] + [10] * len(results)
    print("  PER-DAY BREAKDOWN  (* = best model for that day)")
    _hdr(col_headers2, W4)
    for day in days:
        mask = test_index.normalize() == day
        yt_d = y_test[mask]
        dow  = dow_names[day.dayofweek]
        day_maes  = [mae(yt_d, p[mask]) for _, _, _, p in results]
        best_idx  = int(np.argmin(day_maes))
        mae_strs  = [f"{v:.3f}{'*' if i==best_idx else ' '}" for i, v in enumerate(day_maes)]
        _row([day.strftime("%Y-%m-%d"), dow] + mae_strs, W4)
    print()

    # ── Summary ──
    all_maes  = [(label, mae(y_test, p)) for _, label, _, p in results]
    winner    = min(all_maes, key=lambda x: x[1])
    print(f"  Overall winner: {winner[0]}  (MAE={winner[1]:.3f} {unit})")
    if len(results) >= 2:
        ref_label, ref_mae_val = all_maes[0]
        for label, m in all_maes[1:]:
            delta = m - ref_mae_val
            pct   = 100.0 * delta / ref_mae_val
            print(f"  Δ vs {ref_label[:30]}: {delta:+.3f} {unit} ({pct:+.1f}%)")
    print(f"{'='*84}\n")

    # ── Baselines (for JSON output) ────────────────────────────────────────────
    y_full = df["y"].astype(float)
    baseline_results = []

    for lag, bname in [(1, "Naive-1"), (24, "Naive-24"), (168, "Naive-168")]:
        bp = y_full.shift(lag).reindex(test_index).to_numpy(dtype=float)
        baseline_results.append((f"baseline_{lag}", bname, 0, bp))

    # Seasonal Profile (hour-of-week mean from train)
    train_y = df_train["y"].astype(float)
    prof = (
        pd.DataFrame({"y": train_y.values,
                      "dow": train_y.index.dayofweek,
                      "hour": train_y.index.hour}, index=train_y.index)
        .groupby(["dow", "hour"])["y"].mean()
    )
    global_mean = float(train_y.mean()) if len(train_y) > 0 else 0.0
    sp_preds = np.array([
        float(prof.get((int(ts.dayofweek), int(ts.hour)), global_mean))
        for ts in test_index
    ], dtype=float)
    baseline_results.append(("baseline_sp", "Seasonal Profile", 0, sp_preds))

    # ── JSON output ──
    if save_json:
        all_results = results + baseline_results
        _save_json_ol(task, test_index, y_test, all_results, outer_suffix=out_suffix)


def main():
    p = argparse.ArgumentParser(
        description="1-month comparison: full-week chunks vs daily chunks"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_end",   type=str, default="2025-11-30 23:00")
    p.add_argument("--test_start",  type=str, default="2025-12-01 00:00")
    p.add_argument("--test_end",    type=str, default="2025-12-31 23:00")
    p.add_argument("--save_json",   action="store_true",
                   help="Save results as dashboard JSON")
    p.add_argument("--out_suffix",  type=str, default="",
                   help="Suffix appended to JSON filenames, e.g. '_q4' → openloop_h24_monthly_q4.json")
    args = p.parse_args()

    evaluate_monthly(
        mode=args.mode,
        task=args.task,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        save_json=args.save_json,
        out_suffix=args.out_suffix,
    )


if __name__ == "__main__":
    main()
