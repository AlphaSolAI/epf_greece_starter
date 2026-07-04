"""
tune_openloop_daily_ss_optuna.py

COMBINED: Day-by-day SS training + Day-by-day CV evaluation in ONE Optuna study.

Each trial:
  1. Teacher-forced train on df_tr  (with early stopping on df_va)
  2. N quick daily-SS iterations   (ε fixed at eps_quick, using best_iter from step 1)
  3. Evaluate on df_va with day-by-day recursive prediction (7×24h fresh runs)

→ Hyperparameters are optimised jointly for how the model is TRAINED (daily SS)
  and how it is DEPLOYED (day-by-day recursive inference).

Final model (saved at end):
  Full 3-iteration daily SS with best params on full training data.

Saved files:
  models/{algo}_{mode}_{task}_openloop_daily_ss_optuna.pkl
  models/{algo}_{mode}_{task}_openloop_daily_ss_optuna_params.json
  tuning/optuna_daily_ss_{algo}_{mode}_{task}.json

Usage:
  conda run -n epf --no-capture-output python -m src.tune_openloop_daily_ss_optuna hourly \\
      --algo lgbm --task price \\
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

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TUNING_DIR = BASE_DIR / "tuning"

from .split_utils import load_processed, make_xy, split_time_series
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop


# ── Rolling-origin CV folds ───────────────────────────────────────────────────
def _rolling_origin_folds(df_train, n_splits, val_size, step, min_train_size):
    n = len(df_train)
    folds = []
    for i in range(n_splits):
        val_end   = n - i * step
        val_start = val_end - val_size
        if val_start <= 0 or val_start < min_train_size:
            break
        tr = df_train.iloc[:val_start].copy()
        va = df_train.iloc[val_start:val_end].copy()
        if len(va) != val_size:
            continue
        folds.append((tr, va))
    return list(reversed(folds))


def _mae(yt, yp) -> float:
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yt[m] - yp[m]))) if m.any() else float("nan")


# ── Lag column helpers ────────────────────────────────────────────────────────
def _get_y_lag_cols(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _add_dense_lags(X: pd.DataFrame, y_series: pd.Series) -> pd.DataFrame:
    """Add y_lag1..y_lag23 to X, filling the intraday-lag gap.
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


# ── Daily SS round (same logic as train_scheduled_daily.py) ───────────────────
def _daily_ss_round(model, X_tr: pd.DataFrame, y_tr: pd.Series,
                    epsilon: float, rng: np.random.Generator) -> pd.DataFrame:
    y_hat = pd.Series(model.predict(X_tr), index=X_tr.index, dtype=float)
    X_aug = X_tr.copy()
    lag_cols = _get_y_lag_cols(X_tr)
    replace_mask = rng.random(len(X_tr)) < epsilon
    hour_of_day  = pd.Series(X_tr.index.hour, index=X_tr.index)
    for col in lag_cols:
        lag_k      = int(col.replace("y_lag", ""))
        within_day = hour_of_day >= lag_k
        pseudo_lag = y_hat.shift(lag_k).reindex(X_tr.index)
        valid      = replace_mask & within_day.values & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values
    return X_aug


# ── Model builders & fitters ──────────────────────────────────────────────────
def _build_lgbm(params, n_est, es):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=n_est, verbose=-1, n_jobs=-1, **params)

def _fit_lgbm(m, Xtr, ytr, Xva, yva, es):
    import lightgbm as lgb
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(-1)])
    return m

def _build_xgb(params, n_est, es):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=n_est, tree_method="hist", device="cuda",
                             n_jobs=-1, early_stopping_rounds=es,
                             eval_metric="mae", verbosity=0, **params)

def _fit_xgb(m, Xtr, ytr, Xva, yva, es):
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m

def _build_rf(params, n_est, es=0):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=42, **params)

def _fit_rf(m, Xtr, ytr, Xva, yva, es=0):
    m.fit(Xtr, ytr)
    return m

def _build_no_es(algo, params, n_est, device="cuda"):
    """Build without early stopping (for SS re-fits)."""
    if algo == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=n_est, verbose=-1, n_jobs=-1, **params)
    elif algo == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=n_est, tree_method="hist", device=device,
                                 n_jobs=-1, verbosity=0, **params)
    elif algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=n_est, n_jobs=-1, random_state=42, **params)

BUILDERS = {"lgbm": _build_lgbm, "xgb": _build_xgb, "rf": _build_rf}
FITTERS  = {"lgbm": _fit_lgbm,   "xgb": _fit_xgb,   "rf": _fit_rf}


# ── Search spaces ─────────────────────────────────────────────────────────────
def _suggest_lgbm(trial):
    return dict(
        objective="regression", random_state=42,
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
    )

def _suggest_xgb(trial):
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

def _suggest_rf(trial):
    return dict(
        max_depth=trial.suggest_int("max_depth", 8, 24),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        max_samples=trial.suggest_float("max_samples", 0.6, 1.0),
    )

SUGGESTERS = {"lgbm": _suggest_lgbm, "xgb": _suggest_xgb, "rf": _suggest_rf}


# ── Core: combined SS-train + day-by-day CV eval ─────────────────────────────
def _eval_fold_daily_ss(
    df_tr, df_va, params, algo, feature_cols,
    n_estimators, early_stopping_rounds,
    ss_n_iter, ss_epsilon,  # quick SS for Optuna trials
    rng,
    dense_lags: bool = False,
    y_full: Optional[pd.Series] = None,
) -> float:
    """
    Train on df_tr:
      1. Teacher-forced + early stopping
      2. ss_n_iter rounds of daily SS at epsilon=ss_epsilon (using best_iter from step 1)
    Evaluate on df_va: day-by-day recursive (7×24h fresh runs).

    If dense_lags=True, add y_lag1..y_lag23 to X_tr/X_va before training.
    The SS mechanism will then explicitly train with noisy dense lags (within-day
    lags replaced by pseudo-predictions), making the model more robust to them.
    """
    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    # Optionally augment with dense intraday lags
    if dense_lags and y_full is not None:
        X_tr = _add_dense_lags(X_tr, y_full)
        X_va = _add_dense_lags(X_va, y_full)

    feat_cols = list(X_tr.columns)  # actual feature cols (may include dense lags)

    # Step 1: teacher-forced
    model = BUILDERS[algo](params, n_estimators, early_stopping_rounds)
    model = FITTERS[algo](model, X_tr, y_tr, X_va, y_va, early_stopping_rounds)

    best_iter = int(
        getattr(model, "best_iteration_", None)
        or getattr(model, "best_iteration", None)
        or n_estimators
    )

    # Step 2: daily SS quick pass
    for _ in range(ss_n_iter):
        X_aug = _daily_ss_round(model, X_tr, y_tr, ss_epsilon, rng)
        m_new = _build_no_es(algo, params, best_iter)
        m_new.fit(X_aug, y_tr)
        model = m_new

    # Step 3: day-by-day recursive CV eval
    df_context = pd.concat([df_tr, df_va])

    # If dense_lags: add y_lag4..23 to df_context so recursive engine can find them
    if dense_lags and y_full is not None:
        missing = [f"y_lag{k}" for k in range(1, 24) if f"y_lag{k}" not in df_context.columns]
        if missing:
            df_context = df_context.copy()
            for col_name in missing:
                lag = int(col_name[5:])
                df_context[col_name] = y_full.shift(lag).reindex(df_context.index).fillna(0.0)

    val_index  = df_va.index
    cfg        = OpenLoopConfig(y_floor=None)
    days       = sorted(set(val_index.normalize()))

    preds = []
    for day in days:
        day_idx   = val_index[val_index.normalize() == day]
        day_preds = recursive_predict_openloop(
            model=model, df_full=df_context,
            test_index=day_idx, feature_cols=feat_cols, config=cfg,
        )
        preds.extend(np.asarray(day_preds, float))

    return _mae(np.asarray(y_va, float), np.array(preds))


def _eval_params_cv(folds, params, algo, feature_cols, n_est, es,
                    ss_n_iter, ss_epsilon, rng, trial=None,
                    dense_lags: bool = False,
                    y_full: Optional[pd.Series] = None) -> float:
    import optuna
    maes = []
    for i, (df_tr, df_va) in enumerate(folds):
        fold_mae = _eval_fold_daily_ss(
            df_tr, df_va, params, algo, feature_cols,
            n_est, es, ss_n_iter, ss_epsilon, rng,
            dense_lags=dense_lags, y_full=y_full,
        )
        maes.append(fold_mae)
        if trial is not None:
            trial.report(float(np.mean(maes)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(maes))


# ── Optuna study ──────────────────────────────────────────────────────────────
def run_optuna(
    mode, task, algo, train_end,
    n_trials, n_splits, val_size, step, min_train_size,
    n_estimators, early_stopping_rounds,
    ss_n_iter_quick, ss_epsilon_quick,       # fast SS for Optuna trials
    ss_n_iter_final, ss_eps_start, ss_eps_end,  # full SS for final model
    seed,
    dense_lags: bool = False,
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng        = np.random.default_rng(seed)
    suggest_fn = SUGGESTERS[algo]
    dense_sfx  = "_dense" if dense_lags else ""

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(df, mode=mode, train_end=train_end, test_size=168)
    X_train, _  = make_xy(df_train)
    feature_cols = list(X_train.columns)

    # Augment feature_cols with dense intraday lags if requested
    y_full: Optional[pd.Series] = None
    if dense_lags:
        y_full = df["y"].astype(float)
        X_train_aug = _add_dense_lags(X_train, y_full)
        feature_cols = list(X_train_aug.columns)

    dense_tag = " | DENSE-LAGS" if dense_lags else ""
    print(f"\n{'='*72}")
    print(f"  DAILY-SS OPTUNA | algo={algo.upper()} | task={task.upper()} | mode={mode}{dense_tag}")
    print(f"  train={len(df_train)} | features={len(feature_cols)}")
    print(f"  n_trials={n_trials} | n_splits={n_splits} | val_size={val_size}h")
    print(f"  CV metric : day-by-day recursive MAE")
    print(f"  Per trial : {ss_n_iter_quick} daily-SS iter (ε={ss_epsilon_quick:.2f}) + day-by-day eval")
    print(f"  Final     : {ss_n_iter_final} daily-SS iter (ε:{ss_eps_start:.2f}→{ss_eps_end:.2f})")
    print(f"{'='*72}\n")

    folds = _rolling_origin_folds(df_train, n_splits, val_size, step, min_train_size)
    print(f"[CV] {len(folds)} folds (val={val_size}h, step={step}h)")

    # baseline
    baseline_params = {
        "lgbm": dict(objective="regression", random_state=42, learning_rate=0.01,
                     num_leaves=128, max_depth=-1, min_child_samples=40,
                     subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
                     reg_alpha=0.1, reg_lambda=0.2, min_split_gain=0.0),
        "xgb":  dict(objective="reg:squarederror", random_state=42, learning_rate=0.01,
                     max_depth=10, min_child_weight=5, subsample=0.85,
                     colsample_bytree=0.85, colsample_bylevel=1.0,
                     reg_alpha=0.1, reg_lambda=0.2, gamma=0.0),
        "rf":   dict(max_depth=16, min_samples_split=4, min_samples_leaf=2,
                     max_features="sqrt", max_samples=None),
    }[algo]

    baseline_mae = _eval_params_cv(
        folds, baseline_params, algo, feature_cols, n_estimators, early_stopping_rounds,
        ss_n_iter_quick, ss_epsilon_quick, rng,
        dense_lags=dense_lags, y_full=y_full,
    )
    print(f"[Baseline] daily-SS CV MAE = {baseline_mae:.4f}\n")

    def objective(trial):
        params = suggest_fn(trial)
        return _eval_params_cv(
            folds, params, algo, feature_cols, n_estimators, early_stopping_rounds,
            ss_n_iter_quick, ss_epsilon_quick, rng, trial=trial,
            dense_lags=dense_lags, y_full=y_full,
        )

    def callback(study, trial):
        tag    = " <- BEST" if trial.value == study.best_value else ""
        status = "PRUNED" if trial.state.name == "PRUNED" else f"{trial.value:.4f}"
        print(f"  [Trial {trial.number:03d}] {status}{tag}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    best_params = study.best_params
    best_mae    = study.best_value
    print(f"\nBest daily-SS CV MAE: {best_mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"  Improvement: {100*(baseline_mae-best_mae)/baseline_mae:+.2f}%")
    print(f"  Best params: {best_params}")

    # ── Final model: full SS on full training data ──
    print(f"\n[Final] Full daily-SS training ({ss_n_iter_final} iter) on all training data ...")
    n_val_fin = 168
    df_tr_fin = df_train.iloc[: len(df_train) - n_val_fin]
    df_va_fin = df_train.iloc[len(df_train) - n_val_fin :]
    X_tf, y_tf = make_xy(df_tr_fin)
    X_vf, y_vf = make_xy(df_va_fin)

    if dense_lags and y_full is not None:
        y_full_fin = df["y"].astype(float)
        X_tf = _add_dense_lags(X_tf, y_full_fin)
        X_vf = _add_dense_lags(X_vf, y_full_fin)

    model = BUILDERS[algo](best_params, n_estimators, early_stopping_rounds)
    model = FITTERS[algo](model, X_tf, y_tf, X_vf, y_vf, early_stopping_rounds)
    best_iter = int(
        getattr(model, "best_iteration_", None)
        or getattr(model, "best_iteration", None)
        or n_estimators
    )
    print(f"   Teacher-forced best iter: {best_iter}")

    epsilons = np.linspace(ss_eps_start, ss_eps_end, ss_n_iter_final)
    X_ss = X_tf.copy()   # already augmented with dense lags if applicable
    y_ss = y_tf
    for i, eps in enumerate(epsilons, 1):
        X_aug = _daily_ss_round(model, X_ss, y_ss, eps, rng)
        m_new = _build_no_es(algo, best_params, best_iter)
        m_new.fit(X_aug, y_ss)
        model = m_new
        print(f"   SS iter {i}/{ss_n_iter_final} (ε={eps:.2f}) done")

    # ── Save ──
    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / f"{algo}_{mode}_{task}_openloop_daily_ss_optuna{dense_sfx}.pkl"
    joblib.dump(model, model_path)

    if algo in ("lgbm", "xgb"):
        params_out = {**best_params, "n_estimators": best_iter}
        if algo == "lgbm":
            params_out.setdefault("objective", "regression")
            params_out.setdefault("subsample_freq", 1)
        else:
            params_out.setdefault("objective", "reg:squarederror")
            params_out["random_state"] = 42
    else:
        params_out = {**best_params, "n_estimators": n_estimators, "random_state": 42}

    params_path = MODELS_DIR / f"{algo}_{mode}_{task}_openloop_daily_ss_optuna{dense_sfx}_params.json"
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    log = {
        "mode": mode, "task": task, "algo": algo,
        "cv_metric": "daily_ss_MAE",
        "dense_lags": dense_lags,
        "n_trials": n_trials, "n_splits": len(folds), "val_size": val_size,
        "ss_n_iter_quick": ss_n_iter_quick, "ss_epsilon_quick": ss_epsilon_quick,
        "ss_n_iter_final": ss_n_iter_final,
        "baseline_mae": round(baseline_mae, 4),
        "best_mae": round(best_mae, 4),
        "improvement_pct": round(100*(baseline_mae-best_mae)/baseline_mae, 2),
        "best_params": params_out,
        "all_trials": [
            {"trial": t.number,
             "value": round(t.value, 4) if t.value is not None else None,
             "state": t.state.name, "params": t.params}
            for t in study.trials
        ],
    }
    log_path = TUNING_DIR / f"optuna_daily_ss_{algo}_{mode}_{task}{dense_sfx}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n✅ Saved model  : {model_path}")
    print(f"   Saved params : {params_path}")
    print(f"   Saved log    : {log_path}")


def main():
    p = argparse.ArgumentParser(description="Combined daily-SS training + day-by-day CV Optuna")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--algo",  choices=["lgbm", "xgb", "rf"], default="lgbm")
    p.add_argument("--task",  choices=["price", "load"], default="price")
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--n_trials",  type=int, default=25)
    p.add_argument("--n_splits",  type=int, default=3)
    p.add_argument("--val_size",  type=int, default=None)
    p.add_argument("--step",      type=int, default=None)
    p.add_argument("--min_train_size", type=int, default=None)
    p.add_argument("--n_estimators",   type=int, default=None)
    p.add_argument("--early_stopping_rounds", type=int, default=100)
    # SS settings for Optuna trials (fast)
    p.add_argument("--ss_n_iter_quick",  type=int,   default=1,
                   help="SS iterations per Optuna trial (default: 1 for speed)")
    p.add_argument("--ss_epsilon_quick", type=float, default=0.25,
                   help="SS epsilon for Optuna trial (default: 0.25)")
    # SS settings for final model (full)
    p.add_argument("--ss_n_iter_final",  type=int,   default=3)
    p.add_argument("--ss_eps_start",     type=float, default=0.10)
    p.add_argument("--ss_eps_end",       type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense_lags", action="store_true",
                   help="Add y_lag1..y_lag23 dense intraday lags. SS explicitly trains "
                        "with noisy dense lags (within-day replaced by pseudo-predictions). "
                        "Saves with _dense suffix.")
    args = p.parse_args()

    val_size       = args.val_size or 168
    step           = args.step or val_size
    min_train_size = args.min_train_size or (4 * val_size)
    default_n_est  = {"lgbm": 5000, "xgb": 3000, "rf": 400}
    n_estimators   = args.n_estimators or default_n_est[args.algo]

    run_optuna(
        mode=args.mode, task=args.task, algo=args.algo, train_end=args.train_end,
        n_trials=args.n_trials, n_splits=args.n_splits, val_size=val_size,
        step=step, min_train_size=min_train_size,
        n_estimators=n_estimators, early_stopping_rounds=args.early_stopping_rounds,
        ss_n_iter_quick=args.ss_n_iter_quick, ss_epsilon_quick=args.ss_epsilon_quick,
        ss_n_iter_final=args.ss_n_iter_final, ss_eps_start=args.ss_eps_start,
        ss_eps_end=args.ss_eps_end, seed=args.seed,
        dense_lags=args.dense_lags,
    )


if __name__ == "__main__":
    main()
