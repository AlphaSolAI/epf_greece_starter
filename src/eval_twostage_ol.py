"""
eval_twostage_ol.py
====================
Two-stage Open-Loop pipeline:
  Stage 1 — best OL Load forecast (LGBM-Daily-SS-Optuna, from JSON or re-run)
  Stage 2 — best OL Price models, but with  load_fc = ML_load_pred
             instead of the official ENTSO-E day-ahead load forecast

This answers the question:
  "If we replace the official ENTSO-E load_fc with our own ML load forecast,
   how does price-prediction accuracy change?"

Usage:
  conda run -n epf --no-capture-output python -m src.eval_twostage_ol hourly \\
      --task price --train_end "2025-11-30 23:00" \\
      --test_start "2025-12-01 00:00" --test_end "2025-12-31 23:00"

Optional flags:
  --load_model  name of load OL series in load JSON (default: "LGBM-Daily-SS-Optuna (24h)")
  --save_json   save results to dashboard JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
import warnings
from pathlib import Path
from typing import List, Optional

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

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ── Unpickle aliases ──────────────────────────────────────────────────────────
class ResidualAddBaselineWrapper:
    def __init__(self, model, baseline_col: str, feature_names: list):
        self.model = model
        self.baseline_col = baseline_col
        self.feature_names = list(feature_names)
        self.baseline_idx = self.feature_names.index(baseline_col)

    def predict(self, X):
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


# ── Price models to evaluate in two-stage mode ────────────────────────────────
# (key, label, chunk_hours, filename)
PRICE_MODELS = [
    ("lgbm_do",    "LGBM-Daily-Optuna",       24,  "lgbm_{mode}_price_openloop_daily_optuna.pkl"),
    ("rf_ss_d",    "RF-Daily-SS",             24,  "rf_{mode}_price_openloop_daily_optuna_ss.pkl"),
    ("xgb_joint",  "XGB-Daily-SS-Optuna",     24,  "xgb_{mode}_price_openloop_daily_ss_optuna.pkl"),
    ("mlp_ss",     "MLP-SS",                  24,  "mlp_{mode}_price_scheduled_openloop.pkl"),
    ("lgbm_fw",    "LGBM-OL-Optuna (weekly)", 168, "lgbm_{mode}_price_openloop_optuna.pkl"),
]


def _predict_chunked(model, df_full, test_index, feature_cols, chunk_hours, cfg):
    preds = np.zeros(len(test_index), dtype=float)
    ptr = 0
    n = len(test_index)
    while ptr < n:
        chunk_idx = test_index[ptr : ptr + chunk_hours]
        chunk_preds = _fnp(
            recursive_predict_openloop(
                model=model,
                df_full=df_full,
                test_index=chunk_idx,
                feature_cols=feature_cols,
                config=cfg,
            )
        )
        k = len(chunk_preds)
        preds[ptr : ptr + k] = chunk_preds
        ptr += k
    return preds


def _load_load_preds_from_json(
    load_json_path: Path,
    load_model_name: str,
    test_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Read ML load predictions from the OL load monthly JSON.
    Returns array aligned with test_index (744 values for Dec 2025).
    """
    with open(load_json_path, encoding="utf-8") as f:
        jdata = json.load(f)

    if load_model_name not in jdata["series"]:
        available = [k for k in jdata["series"] if "Naive" not in k and "Seasonal" not in k]
        raise KeyError(
            f"Load model '{load_model_name}' not found in JSON.\n"
            f"Available: {available}"
        )

    raw_dates = pd.to_datetime(jdata["dates"])
    raw_vals  = np.array(
        [v if v is not None else np.nan for v in jdata["series"][load_model_name]],
        dtype=float,
    )
    series = pd.Series(raw_vals, index=raw_dates)

    # Align to test_index (reindex — fills NaN if any gap)
    aligned = series.reindex(test_index)
    n_missing = int(np.isnan(aligned.values).sum())
    if n_missing > 0:
        print(f"  [WARN] {n_missing} missing values in load preds — filling with NaN")
    return aligned.values


def evaluate_twostage(
    mode: str,
    task: str,
    train_end: str,
    test_start: str,
    test_end: str,
    load_model_name: str = "LGBM-Daily-SS-Optuna (24h)",
    save_json: bool = False,
) -> None:
    _register_aliases()

    # ── Load price data ─────────────────────────────────────────────────────
    df = load_processed(mode, task="price")
    df_load = load_processed(mode, task="load")   # for ENTSO-E load_fc comparison

    _, df_price_test = split_time_series(
        df, mode=mode, test_size=None, train_start=None,
        train_end=train_end, test_start=test_start, test_end=test_end,
    )
    _, df_load_test = split_time_series(
        df_load, mode=mode, test_size=None, train_start=None,
        train_end=train_end, test_start=test_start, test_end=test_end,
    )

    X_train, _ = make_xy(
        df[df.index <= pd.Timestamp(train_end)]
    )
    feature_cols = list(X_train.columns)
    _, y_price_test = make_xy(df_price_test)
    y_price_test = _fnp(y_price_test)
    test_index   = df_price_test.index
    unit = "€/MWh"
    cfg = OpenLoopConfig(y_floor=None)

    print(f"\n{'='*72}")
    print(f"  TWO-STAGE OL EVAL | {test_start[:10]} → {test_end[:10]}")
    print(f"  Load model: {load_model_name}")
    print(f"  n_test = {len(test_index)}h = {len(test_index)//24} days")
    print(f"{'='*72}\n")

    # ── ENTSO-E load_fc accuracy (reference) ───────────────────────────────
    if "load_fc" in df.columns and "y" in df_load.columns:
        entso_fc  = df.loc[test_index, "load_fc"].to_numpy(dtype=float)
        actual_ld = df_load.loc[test_index, "y"].to_numpy(dtype=float) \
                    if test_index[0] in df_load.index else None
        if actual_ld is not None:
            entso_mae = mae(actual_ld, entso_fc)
            print(f"[ENTSO-E load_fc quality]  MAE = {entso_mae:.1f} MW  (how good the official forecast is)")

    # ── Read ML load predictions from JSON ─────────────────────────────────
    load_json_path = BASE_DIR / f"dashboard_data_hourly_load_openloop_h24_monthly.json"
    if not load_json_path.exists():
        raise FileNotFoundError(
            f"Load OL JSON not found: {load_json_path.name}\n"
            "Run: python -m src.eval_openloop_monthly hourly --task load --save_json"
        )

    print(f"[LOAD] Reading ML load predictions from {load_json_path.name} ...")
    load_preds_ml = _load_load_preds_from_json(load_json_path, load_model_name, test_index)
    load_mae_val  = float("nan")
    if actual_ld is not None:
        load_mae_val = mae(actual_ld, load_preds_ml)
        print(f"[LOAD] ML load MAE = {load_mae_val:.1f} MW  (cf. ENTSO-E {entso_mae:.1f} MW)")

    # ── Build augmented price df (load_fc = ML load predictions) ───────────
    df_aug = df.copy()
    df_aug.loc[test_index, "load_fc"] = load_preds_ml
    print(f"[PRICE] Replaced load_fc[test] with ML load predictions.\n")

    # ── Run each price model on BOTH standard and augmented df ────────────
    results_std = []   # (key, label, chunk_hours, preds) using actual ENTSO-E load_fc
    results_aug = []   # same but with ML load_fc

    n_days = len(test_index) // 24

    for key, label, chunk_hours, fname_pat in PRICE_MODELS:
        mp = MODELS_DIR / fname_pat.format(mode=mode)
        if not mp.exists():
            print(f"[SKIP] {label} — model not found: {mp.name}")
            continue
        model = joblib.load(mp)
        chunk_desc = f"{n_days} daily" if chunk_hours == 24 else f"{len(test_index)//168} weekly"
        print(f"[INFO] {label} ({chunk_hours}h chunks) ...")

        preds_std = _predict_chunked(model, df,     test_index, feature_cols, chunk_hours, cfg)
        preds_aug = _predict_chunked(model, df_aug, test_index, feature_cols, chunk_hours, cfg)

        results_std.append((key, label, chunk_hours, preds_std))
        results_aug.append((key, label, chunk_hours, preds_aug))

        mae_std = mae(y_price_test, preds_std)
        mae_aug = mae(y_price_test, preds_aug)
        delta   = mae_aug - mae_std
        pct     = 100 * delta / mae_std
        sign    = "🟢" if delta < 0 else ("🔴" if delta > 0.05 else "➡️")
        print(f"   Standard  (ENTSO-E load_fc):  MAE = {mae_std:.3f} {unit}")
        print(f"   Two-stage (ML load_fc):        MAE = {mae_aug:.3f} {unit}  "
              f"Δ={delta:+.3f} ({pct:+.1f}%)  {sign}")
        print()

    if not results_std:
        print("No price models found.")
        return

    # ── Ensemble (Top-3 best standard models by family) ────────────────────
    # Use same 3 keys as ENSEMBLE_GROUPS: lgbm_do + rf_ss_d + xgb_joint
    ens_keys = ["lgbm_do", "rf_ss_d", "xgb_joint"]
    std_by_key = {k: p for k, _, _, p in results_std}
    aug_by_key = {k: p for k, _, _, p in results_aug}

    avail_ens = [k for k in ens_keys if k in std_by_key]
    if len(avail_ens) >= 2:
        # Standard ensemble
        std_ens_maes = [mae(y_price_test, std_by_key[k]) for k in avail_ens]
        w_std = np.array([1.0 / m for m in std_ens_maes]); w_std /= w_std.sum()
        ens_std = sum(w * std_by_key[k] for w, k in zip(w_std, avail_ens))
        # Two-stage ensemble (same weights — oracle on standard, then applied to aug)
        ens_aug = sum(w * aug_by_key[k] for w, k in zip(w_std, avail_ens))

        mae_ens_std = mae(y_price_test, ens_std)
        mae_ens_aug = mae(y_price_test, ens_aug)
        delta_ens   = mae_ens_aug - mae_ens_std
        pct_ens     = 100 * delta_ens / mae_ens_std
        sign_ens    = "🟢" if delta_ens < 0 else ("🔴" if delta_ens > 0.05 else "➡️")
        print(f"[ENSEMBLE Top-3 (LGBM+RF+XGB)]")
        print(f"   Standard  (ENTSO-E load_fc):  MAE = {mae_ens_std:.3f} {unit}")
        print(f"   Two-stage (ML load_fc):        MAE = {mae_ens_aug:.3f} {unit}  "
              f"Δ={delta_ens:+.3f} ({pct_ens:+.1f}%)  {sign_ens}")
        print()

        results_std.append(("ens_top3", "Ensemble-Top3 (1/MAE)", 24, ens_std))
        results_aug.append(("ens_top3", "Ensemble-Top3 (1/MAE)", 24, ens_aug))

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"  SUMMARY | load forecast: ENTSO-E vs ML ({load_model_name})")
    print(f"  ML load MAE = {load_mae_val:.1f} MW")
    print(f"{'='*72}")
    H = ["Price Model", "MAE(ENTSO-E)", "MAE(ML-load)", "Δ MAE", "Δ%"]
    print(f"  {H[0]:<35} {H[1]:>12} {H[2]:>12} {H[3]:>8} {H[4]:>7}")
    print(f"  {'-'*78}")
    for (k, label, ch, pstd), (_, _, _, paug) in zip(results_std, results_aug):
        m_std = mae(y_price_test, pstd)
        m_aug = mae(y_price_test, paug)
        d = m_aug - m_std
        p = 100 * d / m_std
        print(f"  {label:<35} {m_std:>12.3f} {m_aug:>12.3f} {d:>+8.3f} {p:>+6.1f}%")
    print(f"{'='*72}\n")

    # ── JSON output ────────────────────────────────────────────────────────
    if save_json:
        out_path = BASE_DIR / f"dashboard_data_hourly_price_twostage_ol_monthly.json"
        dates  = [str(ts) for ts in test_index]
        actual = [float(v) if math.isfinite(float(v)) else None for v in y_price_test]

        series_out = {}
        metrics_out = []

        # Standard results (Actual ENTSO-E load_fc)
        for k, label, ch, p in results_std:
            series_out[f"{label} [ENTSO-E load]"] = [
                float(v) if math.isfinite(float(v)) else None for v in p
            ]
            m_val = mae(y_price_test, p)
            r_val = rmse(y_price_test, p)
            s_val = smape(y_price_test, p)
            metrics_out.append({
                "Model": f"{label} [ENTSO-E load]", "Type": "ml",
                "load_src": "entso_e",
                "MAE": round(m_val, 4), "RMSE": round(r_val, 4),
                "sMAPE": round(s_val, 4),
            })

        # Two-stage results (ML load_fc)
        for k, label, ch, p in results_aug:
            series_out[f"{label} [ML load]"] = [
                float(v) if math.isfinite(float(v)) else None for v in p
            ]
            m_val = mae(y_price_test, p)
            r_val = rmse(y_price_test, p)
            s_val = smape(y_price_test, p)
            metrics_out.append({
                "Model": f"{label} [ML load]", "Type": "ml",
                "load_src": "ml",
                "MAE": round(m_val, 4), "RMSE": round(r_val, 4),
                "sMAPE": round(s_val, 4),
            })

        payload = {
            "strategy": "ol_twostage",
            "task": "price",
            "period": "month",
            "load_model": load_model_name,
            "load_mae_mw": round(load_mae_val, 3) if math.isfinite(load_mae_val) else None,
            "entso_e_load_mae_mw": round(entso_mae, 3) if "entso_mae" in dir() else None,
            "dates": dates,
            "actual": actual,
            "series": series_out,
            "metrics": metrics_out,
            "unit": "€/MWh",
        }

        def _fix_nan(obj):
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            if isinstance(obj, dict):
                return {k2: _fix_nan(v2) for k2, v2 in obj.items()}
            if isinstance(obj, list):
                return [_fix_nan(v2) for v2 in obj]
            return obj

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_fix_nan(payload), f, ensure_ascii=False, indent=None)
        print(f"✅ JSON saved: {out_path.name}")


def main():
    p = argparse.ArgumentParser(description="Two-stage OL: ML load → price")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",        type=str, default="price")
    p.add_argument("--train_end",   type=str, default="2025-11-30 23:00")
    p.add_argument("--test_start",  type=str, default="2025-12-01 00:00")
    p.add_argument("--test_end",    type=str, default="2025-12-31 23:00")
    p.add_argument("--load_model",  type=str,
                   default="LGBM-Daily-SS-Optuna (24h)",
                   help="Name of load model series in load OL JSON")
    p.add_argument("--save_json",   action="store_true")
    args = p.parse_args()

    evaluate_twostage(
        mode=args.mode,
        task=args.task,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        load_model_name=args.load_model,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
