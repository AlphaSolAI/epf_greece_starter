"""
retrain_ol_from_params.py

Fast retraining of an OL model using EXISTING Optuna-tuned hyperparameters.
Useful for evaluating the same model architecture on a different time period
(e.g. Q4 Oct-Dec 2025) without re-running expensive Optuna search.

Workflow:
  1. Load dataset with --train_end (e.g. 2025-09-30)
  2. Load hyperparams from --params_json (produced by tune_openloop_daily_optuna.py etc.)
  3. Teacher-forced training (+ optional daily-SS iterations)
  4. Save pkl to models/ using --out_name (or auto-derived from params_json name)

Usage examples:
  # LGBM-Daily-Optuna price, Q4 cutoff:
  conda run -n epf --no-capture-output python -m src.retrain_ol_from_params hourly \\
      --algo lgbm --task price \\
      --params_json models/lgbm_hourly_price_openloop_daily_optuna_params.json \\
      --train_end "2025-09-30 23:00"

  # LGBM-Daily-SS-Optuna price, Q4 cutoff (with 3 SS iterations):
  conda run -n epf --no-capture-output python -m src.retrain_ol_from_params hourly \\
      --algo lgbm --task price \\
      --params_json models/lgbm_hourly_price_openloop_daily_ss_optuna_params.json \\
      --train_end "2025-09-30 23:00" --with_ss --ss_n_iter 3 \\
      --ss_eps_start 0.10 --ss_eps_end 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

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
from .recursive_openloop import OpenLoopConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_y_lag_cols(X: pd.DataFrame):
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _daily_ss_round(model, X_tr: pd.DataFrame, y_tr: pd.Series,
                    epsilon: float, rng: np.random.Generator) -> pd.DataFrame:
    """One round of daily Scheduled Sampling (same logic as train_scheduled_daily.py)."""
    y_hat = pd.Series(model.predict(X_tr), index=X_tr.index, dtype=float)
    X_aug = X_tr.copy()
    lag_cols    = _get_y_lag_cols(X_tr)
    replace_mask = rng.random(len(X_tr)) < epsilon
    hour_of_day  = pd.Series(X_tr.index.hour, index=X_tr.index)
    for col in lag_cols:
        lag_k      = int(col.replace("y_lag", ""))
        within_day = hour_of_day >= lag_k
        pseudo_lag = y_hat.shift(lag_k).reindex(X_tr.index)
        valid      = replace_mask & within_day.values & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values
    return X_aug


def _build_model(algo: str, params: dict, n_est: int):
    """Build a model without early stopping (for SS refits)."""
    if algo == "lgbm":
        from lightgbm import LGBMRegressor
        p = {k: v for k, v in params.items() if k != "n_estimators"}
        return LGBMRegressor(n_estimators=n_est, verbose=-1, n_jobs=-1, **p)
    elif algo == "xgb":
        import xgboost as xgb
        p = {k: v for k, v in params.items() if k != "n_estimators"}
        return xgb.XGBRegressor(n_estimators=n_est, tree_method="hist", device="cuda",
                                 n_jobs=-1, verbosity=0, **p)
    elif algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        p = {k: v for k, v in params.items()
             if k not in ("n_estimators", "random_state")}
        return RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=42, **p)
    raise ValueError(f"Unknown algo: {algo}")


def _fit_with_early_stopping(algo: str, model, X_tr, y_tr, X_va, y_va, es: int):
    """Teacher-forced fit with early stopping (for initial fit before SS)."""
    if algo == "lgbm":
        import lightgbm as lgb
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(es, verbose=False),
                             lgb.log_evaluation(-1)])
    elif algo == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:  # rf — no early stopping
        model.fit(X_tr, y_tr)
    return model


def _add_dense_lags(X: pd.DataFrame, y_series: pd.Series) -> pd.DataFrame:
    existing = set()
    for c in X.columns:
        if c.startswith("y_lag") and c[5:].isdigit():
            existing.add(int(c[5:]))
    X = X.copy()
    for lag in range(1, 24):
        if lag not in existing:
            X[f"y_lag{lag}"] = y_series.shift(lag).reindex(X.index).fillna(0.0)
    return X


# ── Main function ──────────────────────────────────────────────────────────────

def retrain_from_params(
    mode: str,
    algo: str,
    task: str,
    params_json: Path,
    train_end: Optional[str],
    out_name: Optional[str],
    with_ss: bool,
    ss_n_iter: int,
    ss_eps_start: float,
    ss_eps_end: float,
    dense_lags: bool,
    es_rounds: int,
    seed: int,
) -> None:
    # ── Load params ──
    params_json = Path(params_json)
    with open(params_json, encoding="utf-8") as f:
        raw_params = json.load(f)

    # n_estimators may be stored in params JSON
    n_est = int(raw_params.pop("n_estimators", 1000))
    params = raw_params

    # ── Output filename ──
    if out_name:
        out_path = MODELS_DIR / out_name
    else:
        # derive from params_json name: strip _params.json → .pkl
        stem = params_json.stem  # e.g. lgbm_hourly_price_openloop_daily_ss_optuna_params
        if stem.endswith("_params"):
            stem = stem[:-7]     # → lgbm_hourly_price_openloop_daily_ss_optuna
        out_path = MODELS_DIR / (stem + ".pkl")

    print(f"\n{'='*72}")
    print(f"  RETRAIN FROM PARAMS | algo={algo.upper()} | task={task.upper()} | mode={mode}")
    print(f"  params_json : {params_json.name}")
    print(f"  n_estimators: {n_est}")
    print(f"  train_end   : {train_end}")
    print(f"  with_ss     : {with_ss} ({ss_n_iter} iter, ε:{ss_eps_start:.2f}→{ss_eps_end:.2f})")
    print(f"  dense_lags  : {dense_lags}")
    print(f"  output      : {out_path.name}")
    print(f"{'='*72}\n")

    # ── Load data ──
    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(df, mode=mode, train_end=train_end, test_size=168)

    n_val = 168
    df_tr = df_train.iloc[:len(df_train) - n_val]
    df_va = df_train.iloc[len(df_train) - n_val:]

    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    if dense_lags:
        y_full = df["y"].astype(float)
        X_tr = _add_dense_lags(X_tr, y_full)
        X_va = _add_dense_lags(X_va, y_full)
        print(f"[INFO] Dense lags added: {len(X_tr.columns)} features")
    else:
        print(f"[INFO] Features: {len(X_tr.columns)}")

    # ── Teacher-forced training ──
    model = _build_model(algo, params, n_est)
    if algo in ("lgbm", "xgb"):
        model_es = _build_model(algo, params, n_est)
        # Add early stopping for lgbm/xgb to find best iter
        if algo == "lgbm":
            model_es.set_params(n_estimators=n_est)
        else:
            model_es.set_params(n_estimators=n_est, early_stopping_rounds=es_rounds)
        model_es = _fit_with_early_stopping(algo, model_es, X_tr, y_tr, X_va, y_va, es_rounds)
        best_iter = int(
            getattr(model_es, "best_iteration_", None)
            or getattr(model_es, "best_iteration", None)
            or n_est
        )
        print(f"[INFO] Teacher-forced best iter (with ES): {best_iter}")
        # Retrain without ES at best_iter on full data
        model = _build_model(algo, params, best_iter)
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
        best_iter = n_est

    # ── SS iterations (optional) ──
    if with_ss and ss_n_iter > 0:
        rng = np.random.default_rng(seed)
        epsilons = np.linspace(ss_eps_start, ss_eps_end, ss_n_iter)
        for i, eps in enumerate(epsilons, 1):
            X_aug = _daily_ss_round(model, X_tr, y_tr, eps, rng)
            m_new = _build_model(algo, params, best_iter)
            m_new.fit(X_aug, y_tr)
            model = m_new
            print(f"   SS iter {i}/{ss_n_iter} (ε={eps:.2f}) done")

    # ── Save ──
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\n✅ Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Retrain an OL model from existing params JSON (no Optuna re-run)"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--algo",  choices=["lgbm", "xgb", "rf"], required=True)
    p.add_argument("--task",  choices=["price", "load"], required=True)
    p.add_argument("--params_json", type=str, required=True,
                   help="Path to JSON file with hyperparameters (e.g. from tune_openloop_daily_optuna)")
    p.add_argument("--train_end", type=str, default=None,
                   help="Training cutoff (e.g. '2025-09-30 23:00')")
    p.add_argument("--out_name", type=str, default=None,
                   help="Output filename (in models/). If not set, derived from params_json name.")
    p.add_argument("--with_ss", action="store_true",
                   help="Run daily-SS iterations after teacher-forced training")
    p.add_argument("--ss_n_iter",   type=int,   default=3)
    p.add_argument("--ss_eps_start", type=float, default=0.10)
    p.add_argument("--ss_eps_end",   type=float, default=0.40)
    p.add_argument("--dense_lags",  action="store_true",
                   help="Add y_lag1..y_lag23 dense intraday lags")
    p.add_argument("--es_rounds",   type=int, default=100,
                   help="Early stopping rounds for lgbm/xgb (default: 100)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    retrain_from_params(
        mode=args.mode, algo=args.algo, task=args.task,
        params_json=args.params_json, train_end=args.train_end,
        out_name=args.out_name, with_ss=args.with_ss,
        ss_n_iter=args.ss_n_iter, ss_eps_start=args.ss_eps_start,
        ss_eps_end=args.ss_eps_end, dense_lags=args.dense_lags,
        es_rounds=args.es_rounds, seed=args.seed,
    )


if __name__ == "__main__":
    main()
