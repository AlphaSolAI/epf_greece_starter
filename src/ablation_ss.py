"""
Scheduled Sampling Ablation Study.

Trains LGBM + XGB with different (n_iter, epsilon) configs in-process,
evaluates each on the test split using recursive open-loop prediction,
and prints a comparison table.

Usage
-----
python -m src.ablation_ss hourly --task price \\
    --train_end "2025-11-30 23:00" \\
    --test_start "2025-12-01 00:00" \\
    --test_end "2025-12-07 23:00"
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy, split_time_series
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop
from .train_scheduled_openloop import (
    _get_y_lag_cols,
    _make_model,
    _scheduled_sampling_round,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    denom = np.abs(y_true[m]) + np.abs(y_pred[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(y_true[m] - y_pred[m]) / denom))


# -----------------------------------------------------------------------
# One experiment
# -----------------------------------------------------------------------

def _run_experiment(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    y_test: np.ndarray,
    feature_cols: List[str],
    n_iter: int,
    epsilon_start: float,
    epsilon_final: float,
    seed: int,
    mode: str,
    device: str,
) -> Tuple[float, float, float]:
    """Train with SS config and return (MAE, RMSE, sMAPE)."""
    rng = np.random.default_rng(seed)

    # ---- Phase 0: base model
    model = _make_model(model_type, mode=mode, device=device)
    model.fit(X_train, y_train)

    # ---- SS rounds
    epsilons = np.linspace(epsilon_start, epsilon_final, n_iter) if n_iter > 1 else [epsilon_final]
    for eps in epsilons:
        X_aug = _scheduled_sampling_round(model, X_train, y_train, eps, rng)
        model_new = _make_model(model_type, mode=mode, device=device)
        model_new.fit(X_aug, y_train)
        model = model_new

    # ---- Eval: recursive open-loop on test split
    cfg = OpenLoopConfig(y_floor=None)
    yhat = np.asarray(
        recursive_predict_openloop(
            model=model,
            df_full=df_full,
            test_index=test_index,
            feature_cols=feature_cols,
            config=cfg,
        ),
        dtype=float,
    )

    return _mae(y_test, yhat), _rmse(y_test, yhat), _smape(y_test, yhat)


# -----------------------------------------------------------------------
# Ablation grid
# -----------------------------------------------------------------------

CONFIGS = [
    # label,         n_iter, ε_start, ε_final
    ("Standard",     0,      0.0,     0.0),    # no SS (teacher-forced baseline)
    ("SS ε→0.60",    3,      0.20,    0.60),   # current best
    ("SS ε→0.75",    3,      0.20,    0.75),
    ("SS ε→0.90",    3,      0.30,    0.90),
    ("SS ε→1.00",    3,      0.30,    1.00),   # full self-play
    ("SS 5iter→0.80",5,      0.10,    0.80),   # more iterations
]


def run_ablation(
    mode: str,
    task: str,
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    models: List[str],
    device: str,
    seed: int,
) -> None:
    # Load data once
    df = load_processed(mode, task=task)
    df_train, df_test = split_time_series(
        df,
        mode=mode,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    X_train, y_train = make_xy(df_train)
    _, y_test_s = make_xy(df_test)
    y_test = np.asarray(y_test_s, dtype=float)
    test_index = df_test.index
    feature_cols = list(X_train.columns)

    print(f"\n{'='*65}")
    print(f" SS ABLATION STUDY | mode={mode} | task={task.upper()}")
    print(f" train={len(df_train)} rows | test={len(df_test)} rows | features={X_train.shape[1]}")
    print(f" test: {test_index.min()} → {test_index.max()}")
    print(f"{'='*65}\n")

    results = []

    for model_type in models:
        print(f"\n{'─'*65}")
        print(f" Model: {model_type.upper()}")
        print(f"{'─'*65}")

        for label, n_iter, eps_start, eps_final in CONFIGS:
            print(f"\n  [{label}]  n_iter={n_iter}, ε={eps_start:.2f}→{eps_final:.2f} ...", end="", flush=True)

            mae_v, rmse_v, smape_v = _run_experiment(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                df_full=df,
                test_index=test_index,
                y_test=y_test,
                feature_cols=feature_cols,
                n_iter=n_iter,
                epsilon_start=eps_start,
                epsilon_final=eps_final,
                seed=seed,
                mode=mode,
                device=device,
            )
            print(f"  MAE={mae_v:.3f}  RMSE={rmse_v:.3f}  sMAPE={smape_v:.2f}%")
            results.append({
                "Model": model_type.upper(),
                "Config": label,
                "n_iter": n_iter,
                "ε_final": eps_final,
                "MAE": round(mae_v, 3),
                "RMSE": round(rmse_v, 3),
                "sMAPE": round(smape_v, 3),
            })

    # Summary table
    print(f"\n{'='*65}")
    print(f" SUMMARY TABLE (sorted by MAE within each model)")
    print(f"{'='*65}")

    df_res = pd.DataFrame(results)
    for model_type in models:
        sub = df_res[df_res["Model"] == model_type.upper()].sort_values("MAE")
        print(f"\n  {model_type.upper()}:")
        print(f"  {'Config':<20} {'n_iter':>6} {'ε_final':>8} {'MAE':>8} {'RMSE':>8} {'sMAPE':>8}")
        print(f"  {'-'*60}")
        for _, r in sub.iterrows():
            marker = " ← BEST" if r["MAE"] == sub["MAE"].min() else ""
            print(
                f"  {r['Config']:<20} {r['n_iter']:>6} {r['ε_final']:>8.2f} "
                f"{r['MAE']:>8.3f} {r['RMSE']:>8.3f} {r['sMAPE']:>8.3f}%{marker}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled Sampling Ablation Study")
    parser.add_argument("mode", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="lgbm,xgb",
        help="Comma-separated model types to test (lgbm, xgb, rf)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    run_ablation(
        mode=args.mode,
        task=args.task,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        models=models,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
