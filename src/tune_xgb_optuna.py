"""
Optuna-based Bayesian Hyperparameter Optimization for XGBoost (Open-Loop Price/Load).

Uses rolling-origin cross-validation inside the training window to evaluate each
trial without leaking test data. After optimization, the best model is saved
as xgb_{mode}_{task}_optuna.pkl.

Usage
-----
python -m src.tune_xgb_optuna hourly --task price \\
    --train_end "2025-11-30 23:00" \\
    --n_trials 60 --n_splits 4

python -m src.tune_xgb_optuna hourly --task load \\
    --train_end "2025-11-30 23:00" \\
    --n_trials 60 --n_splits 4
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

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TUNING_DIR = BASE_DIR / "tuning"

from .split_utils import load_processed, make_xy, split_time_series


# -----------------------------------------------------------------------
# Rolling-origin CV
# -----------------------------------------------------------------------

def _rolling_origin_folds(
    df_train: pd.DataFrame,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Chronological rolling-origin folds. Newest fold first, then reversed."""
    n = len(df_train)
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        val_end = n - i * step
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


def _eval_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    params: dict,
    max_estimators: int,
    early_stopping_rounds: int,
) -> float:
    """Train XGB with early stopping on one fold, return MAE on val."""
    import xgboost as xgb_mod

    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    model = xgb_mod.XGBRegressor(
        n_estimators=max_estimators,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="mae",
        verbosity=0,
        **params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    y_hat = model.predict(X_va)
    return _mae(np.asarray(y_va, dtype=float), np.asarray(y_hat, dtype=float))


def _eval_params_cv(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    params: dict,
    max_estimators: int,
    early_stopping_rounds: int,
    trial=None,
) -> float:
    """Mean MAE across CV folds. Reports intermediate pruning if trial given."""
    import optuna

    maes = []
    for step_idx, (df_tr, df_va) in enumerate(folds):
        fold_mae = _eval_fold(df_tr, df_va, params, max_estimators, early_stopping_rounds)
        maes.append(fold_mae)
        if trial is not None:
            trial.report(float(np.mean(maes)), step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(maes))


# -----------------------------------------------------------------------
# Optuna study
# -----------------------------------------------------------------------

def _suggest_params(trial) -> dict:
    """Bayesian search space for XGBoost hyperparameters."""
    return dict(
        objective="reg:squarederror",
        random_state=42,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 50),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
    )


def run_optuna(
    mode: str,
    task: str,
    train_end: Optional[str],
    n_trials: int,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
    max_estimators: int,
    early_stopping_rounds: int,
    seed: int,
) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- Load and split data
    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(
        df, mode=mode, train_end=train_end, test_size=168
    )

    X_train, _ = make_xy(df_train)

    print(f"\n{'='*65}")
    print(f" OPTUNA XGB TUNING | mode={mode} | task={task.upper()}")
    print(f" train={len(df_train)} rows | features={X_train.shape[1]}")
    print(f" n_trials={n_trials} | n_splits={n_splits} | val_size={val_size}h")
    print(f"{'='*65}\n")

    folds = _rolling_origin_folds(
        df_train=df_train,
        n_splits=n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
    )
    print(f"[CV] {len(folds)} folds created (val={val_size}h each, step={step}h)")

    # ---- Baseline (default XGB params)
    baseline_params = dict(
        objective="reg:squarederror",
        random_state=42,
        learning_rate=0.01,
        max_depth=10,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=1.0,
        reg_alpha=0.1,
        reg_lambda=0.2,
        gamma=0.0,
    )
    baseline_mae = _eval_params_cv(
        folds=folds,
        params=baseline_params,
        max_estimators=max_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    print(f"[Baseline] mean_MAE={baseline_mae:.4f}")

    # ---- Define Optuna objective
    def objective(trial):
        params = _suggest_params(trial)
        return _eval_params_cv(
            folds=folds,
            params=params,
            max_estimators=max_estimators,
            early_stopping_rounds=early_stopping_rounds,
            trial=trial,
        )

    def callback(study, trial):
        tag = " <- BEST" if trial.value == study.best_value else ""
        status = "PRUNED" if trial.state.name == "PRUNED" else f"{trial.value:.4f}"
        print(f"[Trial {trial.number:03d}] MAE={status}{tag}")

    # ---- Run study
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best_params = study.best_params
    best_mae = study.best_value

    print(f"\nBest MAE: {best_mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"   Improvement: {100*(baseline_mae - best_mae)/baseline_mae:+.2f}%")
    print(f"   Best params: {best_params}")

    # ---- Fit final model on full training data with best params
    print("\n[Final] Fitting best model on full training data ...")
    import xgboost as xgb_mod

    X_full, y_full = make_xy(df_train)

    # Use last 168h of training as val for early stopping
    n = len(df_train)
    n_val = 168
    df_tr_fin = df_train.iloc[: n - n_val]
    df_va_fin = df_train.iloc[n - n_val :]
    X_tf, y_tf = make_xy(df_tr_fin)
    X_vf, y_vf = make_xy(df_va_fin)

    lgbm_params = dict(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="mae",
        verbosity=0,
        **best_params,
    )

    final_model = xgb_mod.XGBRegressor(
        n_estimators=max_estimators,
        **lgbm_params,
    )
    final_model.fit(
        X_tf, y_tf,
        eval_set=[(X_vf, y_vf)],
        verbose=False,
    )
    best_iter = int(getattr(final_model, "best_iteration", max_estimators))
    print(f"   Best iteration: {best_iter}")

    # ---- Save
    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / f"xgb_{mode}_{task}_optuna.pkl"
    joblib.dump(final_model, model_path)

    params_out = {**best_params, "n_estimators": best_iter,
                  "objective": "reg:squarederror", "random_state": 42}
    params_path = MODELS_DIR / f"xgb_{mode}_{task}_optuna_params.json"
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    tuning_log = {
        "mode": mode,
        "task": task,
        "n_trials": n_trials,
        "n_splits": len(folds),
        "val_size": val_size,
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
    log_path = TUNING_DIR / f"optuna_xgb_{mode}_{task}.json"
    log_path.write_text(json.dumps(tuning_log, indent=2), encoding="utf-8")

    print(f"\nSaved tuned model  : {model_path}")
    print(f"   Saved tuned params : {params_path}")
    print(f"   Saved tuning log   : {log_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna XGB hyperparameter search")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--n_trials", type=int, default=60)
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--val_size", type=int, default=None,
                   help="Validation window per fold in hours (default: 168 = 1 week)")
    p.add_argument("--step", type=int, default=None,
                   help="Step between folds in hours (default: val_size)")
    p.add_argument("--min_train_size", type=int, default=None,
                   help="Minimum training rows per fold (default: 4*val_size)")
    p.add_argument("--max_estimators", type=int, default=3000)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    val_size = args.val_size or 168
    step = args.step or val_size
    min_train_size = args.min_train_size or (4 * val_size)

    run_optuna(
        mode=args.mode,
        task=args.task,
        train_end=args.train_end,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
