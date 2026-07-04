"""
eval_openloop_daily.py

Compares two open-loop recursive strategies on the test week (Dec 1-7 2025):

  1. FULL-WEEK   : single 168-hour recursive run (current standard approach)
  2. DAY-BY-DAY  : 7 × 24-hour recursive runs; each day resets to actual history
                   → lag-1 at day-start = actual (not a prediction from prev day)
                   → within each day, short lags are still predicted values

Usage:
  conda run -n epf --no-capture-output python -m src.eval_openloop_daily hourly \
      --task price \
      --model lgbm_openloop_optuna \
      --train_end "2025-11-30 23:00" \
      --test_start "2025-12-01 00:00" \
      --test_end "2025-12-07 23:00"
"""

from __future__ import annotations

import argparse
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ── Unpickle alias (same as eval_openloop.py) ────────────────────────────────
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


# ── Model path (mirrors eval_openloop.py logic) ───────────────────────────────
def _model_path(model_key: str, mode: str, task: str) -> Path:
    if model_key in ("lgbm_openloop_optuna", "xgb_openloop_optuna", "rf_openloop_optuna"):
        base = model_key.replace("_openloop_optuna", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_openloop_optuna.pkl"
    if model_key in ("lgbm_openloop_daily_optuna", "xgb_openloop_daily_optuna", "rf_openloop_daily_optuna"):
        base = model_key.replace("_openloop_daily_optuna", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_openloop_daily_optuna.pkl"
    if model_key in ("lgbm_openloop_optuna_scheduled", "xgb_openloop_optuna_scheduled"):
        base = model_key.replace("_openloop_optuna_scheduled", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_openloop_optuna_scheduled_openloop.pkl"
    if model_key in ("lgbm_scheduled", "xgb_scheduled", "rf_scheduled", "mlp_scheduled"):
        base = model_key.replace("_scheduled", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_scheduled_openloop.pkl"
    return MODELS_DIR / f"{model_key}_{mode}_{task}_openloop.pkl"


MODEL_DISPLAY = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "RandomForest",
    "lgbm_openloop_optuna": "LGBM-OL-Optuna",
    "xgb_openloop_optuna": "XGB-OL-Optuna",
    "lgbm_openloop_daily_optuna": "LGBM-OL-Daily-Optuna",
    "xgb_openloop_daily_optuna": "XGB-OL-Daily-Optuna",
    "lgbm_openloop_optuna_scheduled": "LGBM-OL-Optuna-SS",
    "xgb_openloop_optuna_scheduled": "XGB-OL-Optuna-SS",
    "lgbm_scheduled": "LGBM-SS",
    "xgb_scheduled": "XGB-SS",
}


# ── Core ──────────────────────────────────────────────────────────────────────
def run_fullweek(model, df_full, test_index, feature_cols, cfg) -> np.ndarray:
    """Single 168-hour recursive run (standard open-loop)."""
    return _fnp(
        recursive_predict_openloop(
            model=model,
            df_full=df_full,
            test_index=test_index,
            feature_cols=feature_cols,
            config=cfg,
        )
    )


def run_daily(model, df_full, test_index, feature_cols, cfg) -> np.ndarray:
    """
    7 × 24-hour recursive runs.

    Each day resets y_run to actual history (recursive_predict_openloop always
    starts from df_full["y"] which contains actual values). So at the start of
    day d, lag-1 = actual[end of day d-1], NOT the prediction from day d-1.
    Within each day the recursion still compounds normally (lag-1 at t+k is
    the prediction from t+k-1).
    """
    preds_all = np.zeros(len(test_index), dtype=float)
    days = sorted(set(test_index.normalize()))  # unique calendar dates

    ptr = 0
    for day in days:
        day_idx = test_index[test_index.normalize() == day]
        day_preds = _fnp(
            recursive_predict_openloop(
                model=model,
                df_full=df_full,
                test_index=day_idx,
                feature_cols=feature_cols,
                config=cfg,
            )
        )
        n = len(day_preds)
        preds_all[ptr : ptr + n] = day_preds
        ptr += n

    return preds_all


# ── Printing ──────────────────────────────────────────────────────────────────
def _hdr(cols, widths):
    parts = [c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cols, widths))]
    line = "  ".join(parts)
    print(line)
    print("-" * len(line))


def _row(vals, widths):
    parts = [str(vals[0]).ljust(widths[0])] + [str(v).rjust(w) for v, w in zip(vals[1:], widths[1:])]
    print("  ".join(parts))


# ── Main ──────────────────────────────────────────────────────────────────────
def _load_model(model_key: str, mode: str, task: str):
    mp = _model_path(model_key, mode, task)
    if not mp.exists():
        return None, mp
    print(f"[INFO] Loading {MODEL_DISPLAY.get(model_key, model_key)} from {mp.name}")
    return joblib.load(mp), mp


def evaluate(
    mode: str,
    task: str,
    model_key: str,
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
) -> None:
    _register_aliases()

    # Derive the matching daily-trained model key automatically
    # e.g. lgbm_openloop_optuna → lgbm_openloop_daily_optuna
    daily_key = model_key.replace("_openloop_optuna", "_openloop_daily_optuna") \
                if "_openloop_optuna" in model_key else None

    model_fw, mp = _load_model(model_key, mode, task)
    if model_fw is None:
        raise FileNotFoundError(f"Base model not found: {mp}")

    model_daily, mp_daily = (None, None)
    if daily_key:
        model_daily, mp_daily = _load_model(daily_key, mode, task)
        if model_daily is None:
            print(f"[INFO] Daily-trained model not found ({mp_daily.name}) — will skip that column")

    df = load_processed(mode, task=task)

    df_train, df_test = split_time_series(
        df, mode=mode, test_size=None, train_start=None,
        train_end=train_end, test_start=test_start, test_end=test_end,
    )

    X_train, _ = make_xy(df_train)
    _, y_test = make_xy(df_test)
    y_test = _fnp(y_test)

    test_index = df_test.index
    feature_cols = list(X_train.columns)
    cfg = OpenLoopConfig(y_floor=None)
    label_fw = MODEL_DISPLAY.get(model_key, model_key)
    label_db = MODEL_DISPLAY.get(daily_key, daily_key) if daily_key else None
    unit = "€/MWh" if task == "price" else "MW"

    print(f"\n[INFO] task={task.upper()} | features={len(feature_cols)}")
    print(f"[INFO] test = {test_index.min()} → {test_index.max()} ({len(test_index)}h)\n")

    # ── Run predictions ──
    print(f"[{label_fw}] Full-week (168h) ...")
    preds_fw = run_fullweek(model_fw, df, test_index, feature_cols, cfg)

    print(f"[{label_fw}] Day-by-day (7×24h) ...")
    preds_db_fw = run_daily(model_fw, df, test_index, feature_cols, cfg)

    preds_db_daily = None
    if model_daily is not None:
        print(f"[{label_db}] Day-by-day (7×24h) ...")
        preds_db_daily = run_daily(model_daily, df, test_index, feature_cols, cfg)

    # ── Overall comparison table ──
    days = sorted(set(test_index.normalize()))
    W = [38, 12, 12, 10]
    COLS = ["Strategy", f"MAE ({unit})", f"RMSE ({unit})", "sMAPE (%)"]

    print(f"\n{'='*76}")
    print(f"  COMPARISON | task={task.upper()} | test Dec 1-7 2025")
    print(f"{'='*76}")
    _hdr(COLS, W)
    _row([f"{label_fw}  →  Full-week (168h)",
          f"{mae(y_test,preds_fw):.3f}", f"{rmse(y_test,preds_fw):.3f}", f"{smape(y_test,preds_fw):.3f}"], W)
    _row([f"{label_fw}  →  Day-by-day (7×24h)",
          f"{mae(y_test,preds_db_fw):.3f}", f"{rmse(y_test,preds_db_fw):.3f}", f"{smape(y_test,preds_db_fw):.3f}"], W)
    if preds_db_daily is not None:
        _row([f"{label_db}  →  Day-by-day (7×24h)",
              f"{mae(y_test,preds_db_daily):.3f}", f"{rmse(y_test,preds_db_daily):.3f}", f"{smape(y_test,preds_db_daily):.3f}"], W)
    print()

    # ── Per-day breakdown ──
    has_daily = preds_db_daily is not None
    if has_daily:
        W2 = [12, 5, 10, 10, 10, 10, 10, 10]
        COLS2 = ["Date", "DoW", "MAE-FW", "MAE-DB", "MAE-DB-D", "RMSE-FW", "RMSE-DB", "RMSE-DB-D"]
    else:
        W2 = [12, 5, 10, 10, 10, 10, 10, 10]
        COLS2 = ["Date", "DoW", "MAE-FW", "MAE-DB", "RMSE-FW", "RMSE-DB", "sMAPE-FW", "sMAPE-DB"]

    print(f"  PER-DAY BREAKDOWN  (FW={label_fw} full-week, DB={label_fw} day-by-day"
          + (f", DB-D={label_db} day-by-day)" if has_daily else ")"))
    _hdr(COLS2, W2)

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in days:
        mask = test_index.normalize() == day
        yt_d  = y_test[mask]
        fw_d  = preds_fw[mask]
        db_d  = preds_db_fw[mask]
        dow   = dow_names[day.dayofweek]
        if has_daily:
            dbd_d = preds_db_daily[mask]
            _row([
                day.strftime("%Y-%m-%d"), dow,
                f"{mae(yt_d,fw_d):.3f}", f"{mae(yt_d,db_d):.3f}", f"{mae(yt_d,dbd_d):.3f}",
                f"{rmse(yt_d,fw_d):.3f}", f"{rmse(yt_d,db_d):.3f}", f"{rmse(yt_d,dbd_d):.3f}",
            ], W2)
        else:
            _row([
                day.strftime("%Y-%m-%d"), dow,
                f"{mae(yt_d,fw_d):.3f}", f"{mae(yt_d,db_d):.3f}",
                f"{rmse(yt_d,fw_d):.3f}", f"{rmse(yt_d,db_d):.3f}",
                f"{smape(yt_d,fw_d):.3f}", f"{smape(yt_d,db_d):.3f}",
            ], W2)

    print()

    # ── Summary ──
    mae_fw  = mae(y_test, preds_fw)
    mae_db  = mae(y_test, preds_db_fw)
    delta1  = mae_db - mae_fw
    pct1    = 100.0 * delta1 / mae_fw if mae_fw > 0 else float("nan")
    print(f"  Δ MAE ({label_fw} DB − FW)  = {delta1:+.3f} {unit}  ({pct1:+.1f}%)")
    if preds_db_daily is not None:
        mae_dbd = mae(y_test, preds_db_daily)
        delta2  = mae_dbd - mae_fw
        pct2    = 100.0 * delta2 / mae_fw if mae_fw > 0 else float("nan")
        delta3  = mae_dbd - mae_db
        pct3    = 100.0 * delta3 / mae_db if mae_db > 0 else float("nan")
        print(f"  Δ MAE ({label_db} DB − FW model FW) = {delta2:+.3f} {unit}  ({pct2:+.1f}%)")
        print(f"  Δ MAE ({label_db} DB − {label_fw} DB) = {delta3:+.3f} {unit}  ({pct3:+.1f}%)")
        scores = {
            f"{label_fw} Full-week": mae_fw,
            f"{label_fw} Day-by-day": mae_db,
            f"{label_db} Day-by-day": mae_dbd,
        }
        winner = min(scores, key=scores.get)
    else:
        winner = f"{label_fw} Full-week" if mae_fw <= mae_db else f"{label_fw} Day-by-day"
    print(f"  Winner: {winner}")
    print(f"{'='*76}\n")


def main():
    p = argparse.ArgumentParser(description="Compare full-week vs day-by-day open-loop recursive prediction")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--model", type=str, default="lgbm_openloop_optuna",
                   help="Model key (e.g. lgbm_openloop_optuna, xgb_openloop_optuna, lgbm, ...)")
    p.add_argument("--train_end", type=str, default="2025-11-30 23:00")
    p.add_argument("--test_start", type=str, default="2025-12-01 00:00")
    p.add_argument("--test_end", type=str, default="2025-12-07 23:00")
    args = p.parse_args()

    evaluate(
        mode=args.mode,
        task=args.task,
        model_key=args.model,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
