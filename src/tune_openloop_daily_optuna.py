"""
tune_openloop_daily_optuna.py

Like tune_openloop_optuna.py, but the CV metric is evaluated with
DAY-BY-DAY recursive prediction (7 × 24h resets per fold) instead of
a single 168h recursive run.

The idea: the model learns hyperparams that minimise compounding error
within each 24h window, rather than across a full week.

Saved files (distinct suffix so existing models are NOT overwritten):
  models/{model}_{mode}_{task}_openloop_daily_optuna.pkl
  models/{model}_{mode}_{task}_openloop_daily_optuna_params.json
  tuning/optuna_openloop_daily_{model}_{mode}_{task}.json

Usage:
  conda run -n epf --no-capture-output python -m src.tune_openloop_daily_optuna hourly \\
      --model lgbm --task price \\
      --train_end "2025-11-30 23:00" \\
      --n_trials 25 --n_splits 3
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
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop


# -----------------------------------------------------------------------
# Rolling-origin CV  (identical to tune_openloop_optuna.py)
# -----------------------------------------------------------------------

def _rolling_origin_folds(
    df_train: pd.DataFrame,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
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


def _add_dense_lags(X: pd.DataFrame, y_series: pd.Series) -> pd.DataFrame:
    """Add y_lag4..y_lag23 to X, filling the intraday-lag gap.
    y_series must be the FULL series (train+val history) aligned to X's timestamps.
    Only adds lags that don't already exist in X.columns.
    """
    existing = set()
    for c in X.columns:
        if c.startswith("y_lag") and c[5:].isdigit():
            existing.add(int(c[5:]))
    X = X.copy()
    for lag in range(1, 24):
        if lag not in existing:
            X[f"y_lag{lag}"] = y_series.shift(lag).reindex(X.index).fillna(0.0)
    return X


# -----------------------------------------------------------------------
# Model builders / fitters / search spaces (identical)
# -----------------------------------------------------------------------

def _build_lgbm(params, n_estimators, early_stopping_rounds):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=n_estimators, verbose=-1, n_jobs=-1, **params)


def _fit_lgbm(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds):
    import lightgbm as lgb
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    return model


def _build_xgb(params, n_estimators, early_stopping_rounds):
    import xgboost as xgb_mod
    return xgb_mod.XGBRegressor(
        n_estimators=n_estimators, tree_method="hist", device="cuda",
        n_jobs=-1, early_stopping_rounds=early_stopping_rounds,
        eval_metric="mae", verbosity=0, **params,
    )


def _fit_xgb(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def _build_rf(params, n_estimators, early_stopping_rounds=0):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42, **params)


def _fit_rf(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds=0):
    model.fit(X_tr, y_tr)
    return model


def _suggest_lgbm(trial) -> dict:
    return dict(
        objective="regression", random_state=42,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
    )


def _suggest_xgb(trial) -> dict:
    return dict(
        objective="reg:squarederror", random_state=42,
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


def _suggest_rf(trial) -> dict:
    return dict(
        max_depth=trial.suggest_int("max_depth", 8, 24),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        max_samples=trial.suggest_float("max_samples", 0.6, 1.0),
    )


MODEL_BUILDERS   = {"lgbm": _build_lgbm,   "xgb": _build_xgb,   "rf": _build_rf}
MODEL_FITTERS    = {"lgbm": _fit_lgbm,     "xgb": _fit_xgb,     "rf": _fit_rf}
MODEL_SUGGESTERS = {"lgbm": _suggest_lgbm, "xgb": _suggest_xgb, "rf": _suggest_rf}


# -----------------------------------------------------------------------
# KEY DIFFERENCE: day-by-day fold evaluation
# -----------------------------------------------------------------------

def _eval_fold_daily(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    params: dict,
    model_type: str,
    feature_cols: List[str],
    n_estimators: int,
    early_stopping_rounds: int,
    dense_lags: bool = False,
) -> float:
    """
    Train on df_tr (teacher-forced).
    Evaluate on df_va using DAY-BY-DAY recursive prediction:
      - split df_va into calendar days
      - for each day, run a fresh 24h recursive prediction
        (y_run resets to actual at the start of each day)
    CV metric = mean MAE over all hours of df_va.
    """
    build_fn = MODEL_BUILDERS[model_type]
    fit_fn   = MODEL_FITTERS[model_type]

    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    # Context = train + val (actual y throughout)
    df_context = pd.concat([df_tr, df_va], axis=0)

    if dense_lags:
        y_context = df_context["y"].astype(float)
        X_tr = _add_dense_lags(X_tr, y_context)
        X_va = _add_dense_lags(X_va, y_context)
        # df_context needs the dense-lag columns so recursive engine can look them up
        for lag in range(1, 24):
            col = f"y_lag{lag}"
            if col not in df_context.columns:
                df_context = df_context.copy()
                df_context[col] = y_context.shift(lag)

    # Always use the actual X_tr columns as feature_cols for recursive prediction
    fc = list(X_tr.columns)

    model = build_fn(params, n_estimators, early_stopping_rounds)
    model = fit_fn(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds)

    val_index  = df_va.index
    cfg = OpenLoopConfig(y_floor=None)

    # Day-by-day: split into calendar days, fresh recursive run each day
    days = sorted(set(val_index.normalize()))
    y_hat_parts: List[np.ndarray] = []
    for day in days:
        day_idx = val_index[val_index.normalize() == day]
        day_preds = recursive_predict_openloop(
            model=model,
            df_full=df_context,   # always actual y in df_context["y"]
            test_index=day_idx,
            feature_cols=fc,
            config=cfg,
        )
        y_hat_parts.append(np.asarray(day_preds, dtype=float))

    y_hat = np.concatenate(y_hat_parts)
    return _mae(np.asarray(y_va, dtype=float), y_hat)


def _eval_params_cv(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    params: dict,
    model_type: str,
    feature_cols: List[str],
    n_estimators: int,
    early_stopping_rounds: int,
    trial=None,
    dense_lags: bool = False,
) -> float:
    import optuna
    maes = []
    for step_idx, (df_tr, df_va) in enumerate(folds):
        fold_mae = _eval_fold_daily(
            df_tr, df_va, params, model_type, feature_cols,
            n_estimators, early_stopping_rounds,
            dense_lags=dense_lags,
        )
        maes.append(fold_mae)
        if trial is not None:
            trial.report(float(np.mean(maes)), step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(maes))


# -----------------------------------------------------------------------
# Optuna study
# -----------------------------------------------------------------------

def run_optuna(
    mode: str,
    task: str,
    model_type: str,
    train_end: Optional[str],
    n_trials: int,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
    n_estimators: int,
    early_stopping_rounds: int,
    seed: int,
    dense_lags: bool = False,
) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    suggest_fn = MODEL_SUGGESTERS[model_type]
    dense_tag  = " | DENSE-LAGS" if dense_lags else ""
    dense_sfx  = "_dense" if dense_lags else ""

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(df, mode=mode, train_end=train_end, test_size=168)
    X_train, _ = make_xy(df_train)
    if dense_lags:
        X_train = _add_dense_lags(X_train, df_train["y"].astype(float))
    feature_cols = list(X_train.columns)

    print(f"\n{'='*68}")
    print(f" DAILY-OL-AWARE OPTUNA | model={model_type.upper()} | mode={mode} | task={task.upper()}{dense_tag}")
    print(f" train={len(df_train)} rows | features={X_train.shape[1]}")
    print(f" n_trials={n_trials} | n_splits={n_splits} | val_size={val_size}h ({val_size//24} days)")
    print(f" CV metric: DAY-BY-DAY RECURSIVE MAE (7×24h resets per fold)")
    print(f"{'='*68}\n")

    folds = _rolling_origin_folds(
        df_train=df_train,
        n_splits=n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
    )
    print(f"[CV] {len(folds)} folds created (val={val_size}h = {val_size//24} days each, step={step}h)")

    baseline_params_map = {
        "lgbm": dict(objective="regression", random_state=42,
                     learning_rate=0.01, num_leaves=128, max_depth=-1,
                     min_child_samples=40, subsample=0.85, subsample_freq=1,
                     colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.2,
                     min_split_gain=0.0),
        "xgb":  dict(objective="reg:squarederror", random_state=42,
                     learning_rate=0.01, max_depth=10, min_child_weight=5,
                     subsample=0.85, colsample_bytree=0.85, colsample_bylevel=1.0,
                     reg_alpha=0.1, reg_lambda=0.2, gamma=0.0),
        "rf":   dict(max_depth=16, min_samples_split=4, min_samples_leaf=2,
                     max_features="sqrt", max_samples=None),
    }
    baseline_params = baseline_params_map[model_type]
    baseline_mae = _eval_params_cv(
        folds, baseline_params, model_type, feature_cols,
        n_estimators, early_stopping_rounds,
        dense_lags=dense_lags,
    )
    print(f"[Baseline] daily_ol_MAE={baseline_mae:.4f}")

    def objective(trial):
        params = suggest_fn(trial)
        return _eval_params_cv(
            folds, params, model_type, feature_cols,
            n_estimators, early_stopping_rounds, trial=trial,
            dense_lags=dense_lags,
        )

    def callback(study, trial):
        tag = " <- BEST" if trial.value == study.best_value else ""
        status = "PRUNED" if trial.state.name == "PRUNED" else f"{trial.value:.4f}"
        print(f"[Trial {trial.number:03d}] daily_ol_MAE={status}{tag}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best_params = study.best_params
    best_mae    = study.best_value

    print(f"\nBest daily-OL MAE: {best_mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"   Improvement: {100*(baseline_mae - best_mae)/baseline_mae:+.2f}%")
    print(f"   Best params: {best_params}")

    # ---- Final model: fit on full training data
    print("\n[Final] Fitting best model on full training data ...")
    build_fn = MODEL_BUILDERS[model_type]
    fit_fn   = MODEL_FITTERS[model_type]

    n = len(df_train)
    n_val_fin = 168
    df_tr_fin = df_train.iloc[: n - n_val_fin]
    df_va_fin = df_train.iloc[n - n_val_fin :]
    X_tf, y_tf = make_xy(df_tr_fin)
    X_vf, y_vf = make_xy(df_va_fin)
    if dense_lags:
        y_full_fin = df_train["y"].astype(float)
        X_tf = _add_dense_lags(X_tf, y_full_fin)
        X_vf = _add_dense_lags(X_vf, y_full_fin)

    final_model = build_fn(best_params, n_estimators, early_stopping_rounds)
    final_model = fit_fn(final_model, X_tf, y_tf, X_vf, y_vf, early_stopping_rounds)

    best_iter = int(
        getattr(final_model, "best_iteration_", None)
        or getattr(final_model, "best_iteration", None)
        or n_estimators
    )
    print(f"   Best iteration / n_estimators: {best_iter}")

    # ---- Save  (different suffix: _daily_optuna or _daily_optuna_dense)
    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / f"{model_type}_{mode}_{task}_openloop_daily_optuna{dense_sfx}.pkl"
    joblib.dump(final_model, model_path)

    if model_type in ("lgbm", "xgb"):
        params_out = {**best_params, "n_estimators": best_iter}
        if model_type == "lgbm":
            params_out.setdefault("objective", "regression")
            params_out.setdefault("subsample_freq", 1)
        elif model_type == "xgb":
            params_out.setdefault("objective", "reg:squarederror")
            params_out["random_state"] = 42
    else:
        params_out = {**best_params, "n_estimators": n_estimators, "random_state": 42}

    params_path = MODELS_DIR / f"{model_type}_{mode}_{task}_openloop_daily_optuna{dense_sfx}_params.json"
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    tuning_log = {
        "mode": mode,
        "task": task,
        "model_type": model_type,
        "dense_lags": dense_lags,
        "cv_metric": "daily_ol_MAE",
        "n_trials": n_trials,
        "n_splits": len(folds),
        "val_size": val_size,
        "n_estimators": n_estimators,
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
    log_path = TUNING_DIR / f"optuna_openloop_daily_{model_type}_{mode}_{task}{dense_sfx}.json"
    log_path.write_text(json.dumps(tuning_log, indent=2), encoding="utf-8")

    print(f"\n✅ Saved model  : {model_path}")
    print(f"   Saved params : {params_path}")
    print(f"   Saved log    : {log_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search with DAY-BY-DAY OPEN-LOOP CV"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--model", choices=["lgbm", "xgb", "rf"], default="lgbm")
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--n_trials", type=int, default=25)
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--val_size", type=int, default=None,
                   help="Validation window per fold in hours (default: 168 = 7 days)")
    p.add_argument("--step", type=int, default=None,
                   help="Step between folds in hours (default: val_size)")
    p.add_argument("--min_train_size", type=int, default=None)
    p.add_argument("--n_estimators", type=int, default=None,
                   help="Max trees: LGBM default=5000, XGB default=3000, RF default=400")
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense_lags", action="store_true",
                   help="Add y_lag4..y_lag23 (fills intraday-lag gap). Saves with _dense suffix.")
    args = p.parse_args()

    val_size       = args.val_size or 168
    step           = args.step or val_size
    min_train_size = args.min_train_size or (4 * val_size)
    default_n_est  = {"lgbm": 5000, "xgb": 3000, "rf": 400}
    n_estimators   = args.n_estimators or default_n_est[args.model]

    run_optuna(
        mode=args.mode,
        task=args.task,
        model_type=args.model,
        train_end=args.train_end,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
        n_estimators=n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        dense_lags=args.dense_lags,
    )


if __name__ == "__main__":
    main()
