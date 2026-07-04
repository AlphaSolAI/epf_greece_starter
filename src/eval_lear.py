"""
eval_lear.py — LEAR (LASSO Estimated AutoRegressive) statistical baseline for EPF/ELF.

LEAR (Lago et al., 2021) is the de-facto statistical benchmark in the electricity
price forecasting literature: a high-dimensional linear model with L1 (LASSO)
regularisation over an autoregressive + exogenous feature set, with standardised
inputs and the penalty selected automatically (here via cross-validation).

This script reuses the project's own processed feature matrix and split logic, so
the LEAR numbers are directly comparable to the ML models evaluated in eval.py
(same features, same train/test windows, same metric definitions). LEAR is a
Teacher-Forcing-style model (it sees the actual lagged target), i.e. the linear
analogue of the TF tree models.

Usage (from project root):
  conda run -n epf --no-capture-output python -m src.eval_lear hourly --task price \
      --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2025-12-07 23:00"

  # 1-month December:
  ... --test_start "2025-12-01 00:00" --test_end "2025-12-31 23:00"
  # Q1 2026 (static):
  ... --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split_utils import load_processed, make_xy, split_time_series


def mae(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(yt, float) - np.asarray(yp, float))))


def rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def smape(yt: np.ndarray, yp: np.ndarray) -> float:
    # same definition as src/eval.py (symmetric, range [0,200]%)
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    denom = (np.abs(yt) + np.abs(yp)) + 1e-9
    return float(100.0 * np.mean(2.0 * np.abs(yp - yt) / denom))


def main() -> None:
    ap = argparse.ArgumentParser(description="LEAR (LASSO-AR) statistical baseline.")
    ap.add_argument("mode", nargs="?", default="hourly")
    ap.add_argument("--task", choices=["price", "load"], default="price")
    ap.add_argument("--train_end", type=str, default="2025-11-30 23:00")
    ap.add_argument("--test_start", type=str, default="2025-12-01 00:00")
    ap.add_argument("--test_end", type=str, default="2025-12-07 23:00")
    ap.add_argument("--n_alphas", type=int, default=100)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    df = load_processed(args.mode, task=args.task)
    df_train, df_test = split_time_series(
        df, mode=args.mode,
        train_end=args.train_end, test_start=args.test_start, test_end=args.test_end,
    )
    X_train, y_train = make_xy(df_train)
    X_test, y_test = make_xy(df_test)
    # keep identical feature columns
    cols = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[cols], X_test[cols]

    print(f"\n=== LEAR ({args.task}) ===")
    print(f"train={len(X_train)} rows  test={len(X_test)} rows  features={len(cols)}")
    print(f"test window: {args.test_start} .. {args.test_end}")

    # LEAR: standardised features + LASSO with CV-selected penalty.
    model = make_pipeline(
        StandardScaler(),
        LassoCV(n_alphas=args.n_alphas, cv=args.cv, max_iter=20000,
                n_jobs=-1, random_state=42),
    )
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    unit = "€/MWh" if args.task == "price" else "MW"
    lasso = model.named_steps["lassocv"]
    nz = int(np.sum(np.abs(lasso.coef_) > 1e-8))
    print(f"selected alpha={lasso.alpha_:.5g}  non-zero coefs={nz}/{len(cols)}")
    print("-" * 44)
    print(f"  MAE   = {mae(y_test, yhat):8.3f} {unit}")
    print(f"  RMSE  = {rmse(y_test, yhat):8.3f} {unit}")
    print(f"  sMAPE = {smape(y_test, yhat):8.3f} %")
    print("-" * 44)
    print("Insert this MAE as the 'LEAR' statistical-baseline row in the results "
          "tables (Chapter: Αποτελέσματα) and reference it in Methods §Baselines.")


if __name__ == "__main__":
    main()
