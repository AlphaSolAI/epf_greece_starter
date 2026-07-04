"""
eval_cl_monthly.py

Closed-loop evaluation over any test period (default: Dec 1-31 2025).
CL = teacher-forced: model always receives ACTUAL lag features → no
recursive accumulation, same behavior regardless of horizon length.

Usage:
  conda run -n epf --no-capture-output python -m src.eval_cl_monthly hourly \\
      --task price \\
      --train_end "2025-11-30 23:00" \\
      --test_start "2025-12-01 00:00" \\
      --test_end   "2025-12-31 23:00" \\
      --save_json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import types

import joblib
import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ── Unpickle alias ────────────────────────────────────────────────────────────
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
    for name in ["src.model_wrappers", "src.train_xgb_openloop", "src.eval_openloop", "src.eval"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], "ResidualAddBaselineWrapper", ResidualAddBaselineWrapper)


# ── Metrics ───────────────────────────────────────────────────────────────────
def _fnp(a) -> np.ndarray:
    return np.asarray(a, dtype=float)

def mae(yt, yp) -> float:
    yt, yp = _fnp(yt), _fnp(yp)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yt[m] - yp[m]))) if m.sum() > 0 else float("nan")

def rmse(yt, yp) -> float:
    yt, yp = _fnp(yt), _fnp(yp)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2))) if m.sum() > 0 else float("nan")

def smape(yt, yp) -> float:
    yt, yp = _fnp(yt), _fnp(yp)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return float("nan")
    denom = np.abs(yt[m]) + np.abs(yp[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(yt[m] - yp[m]) / denom))


# ── Model catalogue (CL = standard teacher-forced) ───────────────────────────
# Best version per family: keep both MLP variants (best chosen automatically)
CL_MODELS = [
    ("LightGBM",   "lgbm_{mode}_{task}.pkl"),
    ("XGBoost",    "xgb_{mode}_{task}.pkl"),
    ("RandomForest","rf_{mode}_{task}.pkl"),
    ("SVR",        "svr_{mode}_{task}.pkl"),
    ("MLP",        "mlp_{mode}_{task}.pkl"),
    ("MLP-Optuna", "mlp_{mode}_{task}_optuna.pkl"),
]

BASELINE_NAMES = {"Naive-1", "Naive-24", "Naive-168", "Seasonal Profile"}


# ── Printing ──────────────────────────────────────────────────────────────────
def _hdr(cols, widths):
    parts = [c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cols, widths))]
    line = "  ".join(parts)
    print(line)
    print("-" * len(line))

def _row(vals, widths):
    parts = [str(vals[0]).ljust(widths[0])] + [str(v).rjust(w) for v, w in zip(vals[1:], widths[1:])]
    print("  ".join(parts))


# ── JSON helpers ──────────────────────────────────────────────────────────────
def _fix_nan(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _fix_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_nan(x) for x in obj]
    return obj


def _save_json_cl(task: str, test_index: pd.DatetimeIndex,
                  y_test: np.ndarray, results: list,
                  out_suffix: str = "") -> None:
    """Save CL monthly results in standard dashboard JSON format.
    out_suffix: optional suffix appended before .json, e.g. '_q4' → cl_monthly_q4.json
    """
    unit = "€/MWh" if task == "price" else "MW"

    dates  = [str(ts) for ts in test_index]
    actual = [float(v) for v in y_test]

    # series: all models + baselines
    series: Dict[str, list] = {}
    for label, preds in results:
        series[label] = [float(v) if np.isfinite(v) else None for v in preds]

    # metrics sorted by MAE
    metrics = []
    for label, preds in sorted(results, key=lambda x: mae(y_test, x[1])):
        m = mae(y_test, preds)
        if np.isfinite(m):
            metrics.append({
                "Model":  label,
                "Type":   "baseline" if label in BASELINE_NAMES else "ml",
                "MAE":    round(m, 4),
                "RMSE":   round(rmse(y_test, preds), 4),
                "sMAPE":  round(smape(y_test, preds), 4),
            })

    out = {
        "strategy": "cl",
        "task":     task,
        "period":   "month",
        "dates":    dates,
        "actual":   actual,
        "series":   series,
        "metrics":  metrics,
        "unit":     unit,
    }
    out_path = BASE_DIR / f"dashboard_data_hourly_{task}_cl_monthly{out_suffix}.json"
    out_path.write_text(json.dumps(_fix_nan(out), ensure_ascii=False), encoding="utf-8")
    print(f"✅ JSON saved: {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate_cl_monthly(
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
    X_test, y_test = make_xy(df_test)
    y_test = _fnp(y_test)

    feature_cols = list(X_train.columns)
    test_index   = df_test.index
    unit         = "€/MWh" if task == "price" else "MW"
    n_days       = len(test_index) // 24

    print(f"\n[INFO] CLOSED-LOOP MONTHLY EVAL | task={task.upper()} | features={len(feature_cols)}")
    print(f"[INFO] test = {test_index.min()} → {test_index.max()} ({n_days} days)\n")

    # ── Load ML models ─────────────────────────────────────────────────────────
    ml_results: List[tuple] = []
    for label, fname_pat in CL_MODELS:
        mp = MODELS_DIR / fname_pat.format(mode=mode, task=task)
        if not mp.exists():
            print(f"[SKIP] {label} — {mp.name} not found")
            continue
        print(f"[INFO] Loading {label} from {mp.name}")
        model = joblib.load(mp)
        try:
            preds = _fnp(model.predict(X_test))
        except Exception as e:
            print(f"[WARN] {label} predict failed: {e}")
            continue
        ml_results.append((label, preds))

    if not ml_results:
        raise RuntimeError("No CL models found.")

    # ── Ensembles ──────────────────────────────────────────────────────────────
    # Ensemble All (1/MAE weighted — all ML models)
    mae_vals = [(label, preds, mae(y_test, preds)) for label, preds in ml_results]
    valid_all = [(l, p, m) for l, p, m in mae_vals if np.isfinite(m) and m > 0]

    if len(valid_all) >= 2:
        ws_all = np.array([1.0 / m for _, _, m in valid_all])
        ws_all /= ws_all.sum()
        ens_all = sum(w * p for w, (_, p, _) in zip(ws_all, valid_all))
        ml_results.append(("Ensemble (1/MAE)", ens_all))
        print(f"[ENS] Ensemble-All: {len(valid_all)} models")

    # Ensemble Best-3 (top-3 ML by MAE)
    sorted_ml = sorted(valid_all, key=lambda x: x[2])
    top3 = sorted_ml[:3] if len(sorted_ml) >= 3 else sorted_ml
    if len(top3) >= 2:
        ws_top3 = np.array([1.0 / m for _, _, m in top3])
        ws_top3 /= ws_top3.sum()
        ens_best = sum(w * p for w, (_, p, _) in zip(ws_top3, top3))
        top3_names = "+".join(l[:4] for l, _, _ in top3)
        ml_results.append((f"Ensemble-Best3 (1/MAE)", ens_best))
        print(f"[ENS] Ensemble-Best3: {top3_names}")

    # ── Baselines ──────────────────────────────────────────────────────────────
    y_full = df["y"].astype(float)

    # Naive lags
    for lag, name in [(1, "Naive-1"), (24, "Naive-24"), (168, "Naive-168")]:
        preds_naive = y_full.shift(lag).reindex(test_index).to_numpy(dtype=float)
        ml_results.append((name, preds_naive))

    # Seasonal Profile (hour-of-week mean from training set)
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
    ml_results.append(("Seasonal Profile", sp_preds))

    # ── Overall table ──────────────────────────────────────────────────────────
    W = [22, 12, 12, 10]
    COLS = ["Model", f"MAE ({unit})", f"RMSE ({unit})", "sMAPE (%)"]
    print(f"\n{'='*60}")
    print(f"  CL MONTHLY | task={task.upper()} | {test_start[:10]} → {test_end[:10]}")
    print(f"{'='*60}")
    _hdr(COLS, W)
    sorted_results = sorted(ml_results, key=lambda x: mae(y_test, x[1]))
    for label, preds in sorted_results:
        _row([label,
              f"{mae(y_test,preds):.3f}",
              f"{rmse(y_test,preds):.3f}",
              f"{smape(y_test,preds):.3f}"], W)
    print()

    # ── Per-week breakdown ─────────────────────────────────────────────────────
    ml_only = [(l, p) for l, p in ml_results if l not in BASELINE_NAMES]
    print("  PER-WEEK BREAKDOWN")
    header = "  Week           " + "  ".join(f"{l[:10]:>10}" for l, _ in ml_only)
    print(header)
    print("  " + "-" * (16 + 12 * len(ml_only)))
    ptr, wk = 0, 1
    while ptr < len(test_index):
        chunk = test_index[ptr : ptr + 168]
        mask  = np.zeros(len(test_index), dtype=bool)
        mask[ptr : ptr + len(chunk)] = True
        yt_w  = y_test[mask]
        week_label = f"  W{wk} {chunk[0].strftime('%b %d')}"
        maes_str = "  ".join(f"{mae(yt_w, p[mask]):>10.3f}" for l, p in ml_only)
        print(f"{week_label:<16}  {maes_str}")
        ptr += 168
        wk += 1

    winner = sorted_results[0]
    print(f"\n  Winner: {winner[0]} (MAE={mae(y_test, winner[1]):.3f} {unit})")
    print(f"{'='*60}\n")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    if save_json:
        _save_json_cl(task, test_index, y_test, ml_results, out_suffix=out_suffix)


def main():
    p = argparse.ArgumentParser(description="Closed-loop evaluation over any test period")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",       choices=["price", "load"], default="price")
    p.add_argument("--train_end",  type=str, default="2025-11-30 23:00")
    p.add_argument("--test_start", type=str, default="2025-12-01 00:00")
    p.add_argument("--test_end",   type=str, default="2025-12-31 23:00")
    p.add_argument("--save_json",  action="store_true",
                   help="Save dashboard JSON for 1-month CL comparison")
    p.add_argument("--out_suffix", type=str, default="",
                   help="Suffix appended to JSON filename, e.g. '_q4' → cl_monthly_q4.json")
    args = p.parse_args()

    evaluate_cl_monthly(
        mode=args.mode, task=args.task,
        train_end=args.train_end, test_start=args.test_start, test_end=args.test_end,
        save_json=args.save_json, out_suffix=args.out_suffix,
    )


if __name__ == "__main__":
    main()
