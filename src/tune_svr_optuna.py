"""
Optuna-based Hyperparameter Optimization for SVR (Closed-Loop).

Βελτιώσεις έναντι του default SVR:
  1. Feature reduction: αφαίρεση _missing indicator columns (binary noise για SVR)
  2. Optuna tuning: C, gamma, epsilon με rolling-origin time-series CV
  3. Χρήση ΟΛΩΝ των διαθέσιμων δεδομένων (χωρίς train_start restriction)

CV metric: teacher-forced (closed-loop) MAE — κατάλληλο για SVR που
           δεν τρέχει recursive open-loop prediction.

Saved:
  models/svr_{mode}_{task}_optuna.pkl
  models/svr_{mode}_{task}_optuna_params.json
  tuning/optuna_svr_{mode}_{task}.json

Usage
-----
python -m src.tune_svr_optuna hourly --task price \\
    --train_end "2025-11-30 23:00" --n_trials 40 --n_splits 3

python -m src.tune_svr_optuna hourly --task load \\
    --train_end "2025-11-30 23:00" --n_trials 40 --n_splits 3
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TUNING_DIR = BASE_DIR / "tuning"

from .split_utils import load_processed, make_xy, split_time_series


# -----------------------------------------------------------------------
# Feature reduction: αφαίρεση _missing indicator columns
# Υλοποιείται ως sklearn Transformer ώστε να αποθηκεύεται μέσα στο Pipeline
# -----------------------------------------------------------------------

class DropMissingIndicators(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer που αφαιρεί columns που τελειώνουν σε '_missing'.
    Fit: μαθαίνει ποιες columns να κρατήσει.
    Transform: κρατά μόνο τις selected columns.
    Λειτουργεί και με DataFrame και με numpy array (αν έχει ήδη γίνει fit).
    """
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            self.keep_cols_ = [c for c in X.columns
                                if not c.endswith("_missing")]
            self.keep_idx_  = [list(X.columns).index(c)
                                for c in self.keep_cols_]
        else:
            # numpy array — δεν γνωρίζουμε names, κρατάμε όλα
            self.keep_cols_ = None
            self.keep_idx_  = list(range(X.shape[1]))
        return self

    def transform(self, X, y=None):
        if hasattr(X, "columns"):
            # DataFrame — χρησιμοποιούμε τα ονόματα
            keep = [c for c in self.keep_cols_ if c in X.columns]
            return X[keep].to_numpy()
        else:
            return X[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_ or [])


def _drop_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Helper για απευθείας χρήση εκτός Pipeline (π.χ. για μέτρηση n_features)."""
    missing_cols = [c for c in X.columns if c.endswith("_missing")]
    if missing_cols:
        X = X.drop(columns=missing_cols)
    return X


# -----------------------------------------------------------------------
# Rolling-origin CV (teacher-forced / closed-loop)
# -----------------------------------------------------------------------

def _rolling_origin_folds(
    df_train: pd.DataFrame,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Chronological rolling-origin folds. Oldest first."""
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
        if len(va) != val_size:
            continue
        folds.append((tr, va))
    return list(reversed(folds))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else float("nan")


# -----------------------------------------------------------------------
# Fold evaluation (closed-loop — teacher-forced)
# -----------------------------------------------------------------------

def _eval_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    C: float,
    gamma: float,
    epsilon: float,
) -> float:
    """Train SVR on df_tr, evaluate on df_va (closed-loop)."""
    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    model = Pipeline([
        ("dropper", DropMissingIndicators()),
        ("scaler",  StandardScaler()),
        ("svr",     SVR(C=C, gamma=gamma, epsilon=epsilon, kernel="rbf")),
    ])
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_va)
    return _mae(y_va, y_hat)


def _eval_params_cv(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    C: float,
    gamma: float,
    epsilon: float,
    trial=None,
) -> float:
    """Mean closed-loop MAE across CV folds."""
    import optuna
    maes = []
    for step_idx, (df_tr, df_va) in enumerate(folds):
        fold_mae = _eval_fold(df_tr, df_va, C, gamma, epsilon)
        maes.append(fold_mae)
        if trial is not None:
            trial.report(float(np.mean(maes)), step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(maes))


# -----------------------------------------------------------------------
# Optuna search space
# -----------------------------------------------------------------------

def _suggest_svr(trial, task: str) -> Tuple[float, float, float]:
    C       = trial.suggest_float("C",       1e0,  1e4,  log=True)
    gamma   = trial.suggest_float("gamma",   1e-5, 1e-1, log=True)
    # epsilon: task-dependent scale
    if task == "price":
        epsilon = trial.suggest_float("epsilon", 0.01, 20.0, log=True)
    else:  # load — MW scale
        epsilon = trial.suggest_float("epsilon", 1.0, 500.0, log=True)
    return C, gamma, epsilon


# -----------------------------------------------------------------------
# Main tuning function
# -----------------------------------------------------------------------

DEFAULT_SVR_TRAIN_START = "2022-01-01"  # SVR O(n²) — limitάρουμε σε ~3 χρόνια


def run_optuna(
    mode: str,
    task: str,
    train_start: Optional[str],
    train_end: Optional[str],
    n_trials: int,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
    seed: int,
) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # SVR: O(n²) complexity → χρησιμοποιούμε train_start για να κρατάμε
    # το dataset tractable. Default: 2022-01-01 (~34k hourly rows).
    effective_start = train_start or DEFAULT_SVR_TRAIN_START

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(
        df, mode=mode,
        train_start=effective_start,
        train_end=train_end,
        test_size=168,
    )

    X_sample, _ = make_xy(df_train)
    X_sample     = _drop_missing_indicators(X_sample)
    n_features   = X_sample.shape[1]

    print(f"\n{'='*68}")
    print(f" SVR OPTUNA | mode={mode} | task={task.upper()}")
    print(f" train rows : {len(df_train)}  (from {effective_start})")
    print(f" features   : {n_features}  (after dropping _missing indicators)")
    print(f" n_trials   : {n_trials} | n_splits={n_splits} | val_size={val_size}h")
    print(f" CV metric  : closed-loop (teacher-forced) MAE")
    print(f"{'='*68}\n")

    folds = _rolling_origin_folds(
        df_train=df_train,
        n_splits=n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
    )
    print(f"[CV] {len(folds)} folds created (val={val_size}h each, step={step}h)")

    if not folds:
        print("ERROR: No folds created — check min_train_size / val_size.")
        return

    # ---- Baseline (παλιά hardcoded params)
    baseline_C, baseline_gamma, baseline_epsilon = 50.0, 0.01, 1.0
    baseline_mae = _eval_params_cv(
        folds, baseline_C, baseline_gamma, baseline_epsilon
    )
    print(f"[Baseline] C={baseline_C}, gamma={baseline_gamma}, "
          f"epsilon={baseline_epsilon} → MAE={baseline_mae:.4f}")

    # ---- Optuna objective
    def objective(trial):
        C, gamma, epsilon = _suggest_svr(trial, task)
        return _eval_params_cv(folds, C, gamma, epsilon, trial=trial)

    def callback(study, trial):
        tag = " <- BEST" if trial.value == study.best_value else ""
        status = ("PRUNED" if trial.state.name == "PRUNED"
                  else f"{trial.value:.4f}")
        print(f"[Trial {trial.number:03d}] MAE={status}{tag}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner
    )
    study.optimize(
        objective, n_trials=n_trials, callbacks=[callback],
        show_progress_bar=False,
    )

    best_C       = study.best_params["C"]
    best_gamma   = study.best_params["gamma"]
    best_epsilon = study.best_params["epsilon"]
    best_mae     = study.best_value

    print(f"\nBest MAE   : {best_mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"Improvement: {100*(baseline_mae-best_mae)/baseline_mae:+.2f}%")
    print(f"Best params: C={best_C:.4f}, gamma={best_gamma:.6f}, "
          f"epsilon={best_epsilon:.4f}")

    # ---- Fit final model on FULL training data
    print("\n[Final] Fitting best SVR on full training data ...")
    X_full, y_full = make_xy(df_train)

    final_model = Pipeline([
        ("dropper", DropMissingIndicators()),
        ("scaler",  StandardScaler()),
        ("svr",     SVR(C=best_C, gamma=best_gamma, epsilon=best_epsilon,
                        kernel="rbf")),
    ])
    final_model.fit(X_full, y_full)

    # ---- Save
    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / f"svr_{mode}_{task}_optuna.pkl"
    joblib.dump(final_model, model_path)

    params_out = {
        "C": best_C,
        "gamma": best_gamma,
        "epsilon": best_epsilon,
        "kernel": "rbf",
        "features_excluded": "_missing indicator columns",
        "n_features": n_features,
    }
    params_path = MODELS_DIR / f"svr_{mode}_{task}_optuna_params.json"
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    tuning_log = {
        "mode": mode,
        "task": task,
        "cv_metric": "closed_loop_MAE",
        "n_trials": n_trials,
        "n_splits": len(folds),
        "val_size": val_size,
        "train_rows": len(df_train),
        "n_features": n_features,
        "baseline_mae": round(baseline_mae, 4),
        "best_mae": round(best_mae, 4),
        "improvement_pct": round(100 * (baseline_mae - best_mae) / baseline_mae, 2),
        "best_params": params_out,
        "all_trials": [
            {
                "trial": t.number,
                "value": round(t.value, 4) if t.value is not None else None,
                "state": t.state.name,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    log_path = TUNING_DIR / f"optuna_svr_{mode}_{task}.json"
    log_path.write_text(json.dumps(tuning_log, indent=2), encoding="utf-8")

    print(f"\nSaved model  : {model_path}")
    print(f"Saved params : {params_path}")
    print(f"Saved log    : {log_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for SVR (closed-loop CV)"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",        choices=["price", "load"], default="price")
    p.add_argument("--train_start", type=str, default=None,
                   help=f"Default: {DEFAULT_SVR_TRAIN_START} (SVR O(n²) — keep tractable)")
    p.add_argument("--train_end",   type=str, default=None)
    p.add_argument("--n_trials",    type=int, default=40)
    p.add_argument("--n_splits",  type=int, default=3)
    p.add_argument("--val_size",  type=int, default=None,
                   help="Validation window per fold in hours (default: 168)")
    p.add_argument("--step",      type=int, default=None,
                   help="Step between folds in hours (default: val_size)")
    p.add_argument("--min_train_size", type=int, default=None,
                   help="Min training rows per fold (default: 4*val_size)")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    val_size       = args.val_size or 168
    step           = args.step or val_size
    min_train_size = args.min_train_size or (4 * val_size)

    run_optuna(
        mode=args.mode,
        task=args.task,
        train_start=args.train_start,
        train_end=args.train_end,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
