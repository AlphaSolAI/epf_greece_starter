"""
Optuna Hyperparameter Optimization για MLP (TorchMLPRegressor) — Closed-Loop CV.

Τυνάρει:
  - αρχιτεκτονική (n_layers x units)
  - dropout, lr, weight_decay, batch_size

CV metric = mean MAE across rolling-origin folds (teacher-forced / closed-loop).
Μετά το tuning, επανεκπαιδεύει στο FULL train set με τα best params.

Saved:
  models/mlp_{mode}_{task}_optuna.pkl
  models/mlp_{mode}_{task}_optuna_params.json
  tuning/optuna_mlp_{mode}_{task}.json

Usage
-----
python -m src.tune_mlp_optuna hourly --task price \\
    --train_end "2025-11-30 23:00" --n_trials 25 --n_splits 3

python -m src.tune_mlp_optuna hourly --task load \\
    --train_end "2025-11-30 23:00" --n_trials 25 --n_splits 3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

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
TUNING_DIR = BASE_DIR / "tuning"

from .split_utils import load_processed, make_xy, split_time_series
from .train_mlp   import TorchMLPRegressor


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def _mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() > 0 else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Rolling-origin CV folds
# ──────────────────────────────────────────────────────────────────────────────
def _rolling_origin_folds(
    df_train: pd.DataFrame,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Chronological rolling-origin folds — oldest first."""
    n = len(df_train)
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        val_end   = n - i * step
        val_start = val_end - val_size
        train_end = val_start
        if val_start <= 0 or train_end < min_train_size:
            break
        tr = df_train.iloc[:train_end].copy()
        va = df_train.iloc[val_start:val_end].copy()
        folds.append((tr, va))
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────────────
def _build_mlp(trial, mode: str) -> TorchMLPRegressor:
    """Δημιουργεί TorchMLPRegressor με params από Optuna trial."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units    = [
        trial.suggest_categorical(f"units_l{i}", [64, 128, 256, 512, 1024])
        for i in range(n_layers)
    ]
    hidden   = tuple(units)

    dropout      = trial.suggest_float("dropout",      0.0, 0.4)
    lr           = trial.suggest_float("lr",           1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    return TorchMLPRegressor(
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=150,
        patience=15,
        seed=42,
        device="auto",
    )


def _cv_mae(params_model: TorchMLPRegressor,
            folds: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> float:
    """Αξιολογεί παρόμοιο model σε rolling-origin folds. Returns mean MAE."""
    import copy, dataclasses

    fold_maes = []
    for tr, va in folds:
        X_tr, y_tr = make_xy(tr)
        X_va, y_va = make_xy(va)

        # fresh copy για κάθε fold
        model = copy.copy(params_model)
        # reset learned attributes
        model.feature_names_ = None
        model.x_scaler_ = None
        model.y_scaler_ = None
        model.model_    = None

        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, verbose=False)
        yhat = model.predict(X_va)
        fold_maes.append(_mae(np.asarray(y_va, dtype=float), yhat))

    return float(np.mean(fold_maes)) if fold_maes else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Main tuning
# ──────────────────────────────────────────────────────────────────────────────
def tune_mlp(
    mode: str,
    task: str,
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    n_trials: int,
    n_splits: int,
    val_size: int,
) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(
        df, mode=mode, test_size=None, train_end=train_end,
        test_start=test_start, test_end=test_end, train_start=None,
    )

    X_full, _ = make_xy(df_train)
    n_features = X_full.shape[1]
    print(f"🧠 MLP Optuna | mode={mode} | task={task} | train={len(df_train)} | features={n_features}")
    print(f"   n_trials={n_trials} | n_splits={n_splits} | val_size={val_size}")

    step = val_size  # non-overlapping folds
    min_train = val_size * 4
    folds = _rolling_origin_folds(df_train, n_splits, val_size, step, min_train)
    if not folds:
        raise ValueError("Δεν υπάρχουν αρκετά δεδομένα για CV folds.")
    print(f"   folds={len(folds)} (requested={n_splits})")

    trial_results = []

    def objective(trial) -> float:
        model = _build_mlp(trial, mode)
        cv = _cv_mae(model, folds)
        trial_results.append({
            "trial": trial.number,
            "value": cv,
            "params": trial.params,
        })
        print(f"  Trial {trial.number:3d} | CV MAE={cv:.3f} | hidden={trial.params.get('units_l0')} layers={trial.params.get('n_layers')} lr={trial.params.get('lr'):.2e}")
        return cv

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n🏆 Best trial #{best.number}: CV MAE={best.value:.4f}")
    print(f"   Params: {best.params}")

    # ── Retrain on full train set ──────────────────────────────────────────
    n_layers_best = best.params["n_layers"]
    hidden_best   = tuple(best.params[f"units_l{i}"] for i in range(n_layers_best))

    final_model = TorchMLPRegressor(
        hidden=hidden_best,
        dropout=best.params["dropout"],
        lr=best.params["lr"],
        weight_decay=best.params["weight_decay"],
        batch_size=best.params["batch_size"],
        max_epochs=200,
        patience=20,
        seed=42,
        device="auto",
    )

    # Χρησιμοποιεί το τελευταίο fold ως validation για early stopping
    last_tr, last_va = folds[0]   # πιο πρόσφατο fold (index 0 = closest to test)
    X_ltr, y_ltr = make_xy(last_tr)
    X_lva, y_lva = make_xy(last_va)

    print(f"\n🔁 Retraining on full train (n={len(df_train)}) with best params...")
    X_full_tr, y_full_tr = make_xy(df_train)
    final_model.fit(X_full_tr, y_full_tr, X_val=X_lva, y_val=y_lva, verbose=True)
    final_model.to_cpu()

    # ── Save ──────────────────────────────────────────────────────────────
    out_model = MODELS_DIR / f"mlp_{mode}_{task}_optuna.pkl"
    joblib.dump(final_model, out_model)
    print(f"✅ Saved: {out_model}")

    params_to_save = {
        "hidden":       list(hidden_best),
        "dropout":      best.params["dropout"],
        "lr":           best.params["lr"],
        "weight_decay": best.params["weight_decay"],
        "batch_size":   best.params["batch_size"],
        "cv_mae":       round(best.value, 4),
    }
    out_params = MODELS_DIR / f"mlp_{mode}_{task}_optuna_params.json"
    out_params.write_text(json.dumps(params_to_save, indent=2), encoding="utf-8")
    print(f"✅ Saved params: {out_params}")

    # ── Tuning log ────────────────────────────────────────────────────────
    log = {
        "task": task, "mode": mode, "n_trials": n_trials, "n_splits": n_splits,
        "val_size": val_size, "best_trial": best.number,
        "best_cv_mae": round(best.value, 4), "best_params": best.params,
        "trials": trial_results,
    }
    out_log = TUNING_DIR / f"optuna_mlp_{mode}_{task}.json"
    out_log.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"✅ Tuning log: {out_log}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Optuna MLP tuning (Closed-Loop CV)")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",       choices=["price", "load"], default="price")
    p.add_argument("--train_end",  type=str, default=None)
    p.add_argument("--test_start", type=str, default=None)
    p.add_argument("--test_end",   type=str, default=None)
    p.add_argument("--n_trials",   type=int, default=25)
    p.add_argument("--n_splits",   type=int, default=3)
    p.add_argument("--val_size",   type=int, default=168,
                   help="Validation window size per fold (default=168 = 1 εβδομάδα)")
    args = p.parse_args()

    tune_mlp(
        mode=args.mode, task=args.task,
        train_end=args.train_end, test_start=args.test_start, test_end=args.test_end,
        n_trials=args.n_trials, n_splits=args.n_splits, val_size=args.val_size,
    )


if __name__ == "__main__":
    main()
