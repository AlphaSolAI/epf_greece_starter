"""
tune_mimo_optuna.py — Optuna tuning για MIMO (MultiOutputRegressor).

Αρχιτεκτονική:
  MultiOutputRegressor(LGBM | XGB | RF) — ΕΝΑ μοντέλο για ΟΛΑ τα horizons (1..H).
  Δεν χρειάζεται OL-aware training: δεν υπάρχει recursive component.
  RF: RandomForestRegressor supports multi-output natively (no wrapper).

CV metric: mean MAE across all H horizons × rolling-origin folds.

Saved (παράδειγμα για lgbm, task=price):
  models/lgbm_direct_hourly_price_h168_optuna.pkl     ← drop-in για eval_mimo.py
  models/lgbm_direct_hourly_price_h168_optuna_params.json
  tuning/optuna_mimo_lgbm_price_h168.json

RF output (task=price, H=24):
  models/rf_mimo_hourly_price_h24_optuna.pkl
  models/rf_mimo_hourly_price_h24_optuna_params.json
  tuning/optuna_mimo_rf_price_h24.json

Usage
-----
python -m src.tune_mimo_optuna hourly --algo lgbm --task price \\
    --train_end "2025-11-30 23:00" --n_trials 50 --n_splits 3

python -m src.tune_mimo_optuna hourly --algo xgb --task load \\
    --train_end "2025-11-30 23:00" --n_trials 50 --n_splits 3

python -m src.tune_mimo_optuna hourly --algo rf --task price \\
    --horizon 24 --train_end "2025-11-30 23:00" --n_trials 20 --n_splits 3

Ονοματολογία αρχείων (ώστε eval_mimo.py να τα βρίσκει αυτόματα):
  lgbm → lgbm_direct_hourly_{task}_h{H}_optuna.pkl  (→ "LGBM DIRECT-Optuna")
  xgb  → xgb_vect_hourly_{task}_h{H}_optuna.pkl     (→ "XGB MIMO-Optuna")
  rf   → rf_mimo_hourly_{task}_h{H}_optuna.pkl       (→ "RF MIMO-Optuna")
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TUNING_DIR = BASE_DIR / "tuning"

from .split_utils import load_processed, make_xy, split_time_series  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Multi-output supervised data builder
# ──────────────────────────────────────────────────────────────────────────────

def _make_supervised_xy(df: pd.DataFrame, H: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Δημιουργεί (X, Y) για multi-step output.
    Y[i] = [y(i+1), y(i+2), ..., y(i+H)]  → shape (n_usable, H)
    Αφαιρεί τις τελευταίες H γραμμές που δεν έχουν πλήρη targets.
    """
    X, _ = make_xy(df)
    y = df["y"].astype(float)

    # H-step shift targets
    Y_cols = {f"y_t+{k}": y.shift(-k) for k in range(1, H + 1)}
    Y_df = pd.DataFrame(Y_cols, index=df.index)

    joined = X.join(Y_df, how="inner").dropna(subset=list(Y_df.columns))
    X_out = joined[X.columns].copy()

    # Numeric only
    X_out = X_out.select_dtypes(include=[np.number])
    bool_cols = X_out.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        X_out[bool_cols] = X_out[bool_cols].astype(np.int8)
    X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Y_out = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_out, Y_out


def _mae_multioutput(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Mean MAE across all horizons and all samples."""
    err = np.abs(Y_true - Y_pred)
    return float(np.mean(err))


# ──────────────────────────────────────────────────────────────────────────────
# Rolling-origin CV folds
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_origin_folds(
    df_train: pd.DataFrame,
    H: int,
    n_splits: int,
    val_size: int,   # επιθυμητές usable γραμμές validation
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Δημιουργεί chronological rolling-origin folds.
    val window = val_size + H ώστε _make_supervised_xy να αφήσει val_size usable rows.
    """
    n = len(df_train)
    val_window = val_size + H
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

    for i in range(n_splits):
        val_end = n - i * step
        val_start = val_end - val_window
        train_end_idx = val_start

        if val_start <= 0 or train_end_idx < min_train_size:
            break

        tr = df_train.iloc[:train_end_idx].copy()
        va = df_train.iloc[val_start:val_end].copy()
        folds.append((tr, va))

    return list(reversed(folds))


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(algo: str, params: Dict, n_jobs_inner: int = 1):
    """Φτιάχνει MultiOutputRegressor(LGBM|XGB) ή RandomForestRegressor με δοθέντα params."""
    if algo == "lgbm":
        import lightgbm as lgb
        base = lgb.LGBMRegressor(
            n_jobs=1, verbosity=-1,
            **params,
        )
        return MultiOutputRegressor(base, n_jobs=n_jobs_inner)
    elif algo == "xgb":
        import xgboost as xgb
        base = xgb.XGBRegressor(
            n_jobs=1, verbosity=0, device="cpu",
            **params,
        )
        return MultiOutputRegressor(base, n_jobs=n_jobs_inner)
    elif algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        # RF natively supports multi-output — no wrapper needed
        return RandomForestRegressor(
            n_jobs=n_jobs_inner if n_jobs_inner > 0 else -1,
            random_state=42,
            verbose=0,
            **params,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")


# ──────────────────────────────────────────────────────────────────────────────
# Fold evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _eval_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    algo: str,
    params: Dict,
    H: int,
) -> float:
    X_tr, Y_tr = _make_supervised_xy(df_tr, H)
    X_va, Y_va = _make_supervised_xy(df_va, H)

    if len(X_va) == 0:
        return float("nan")

    if algo == "rf":
        # RF is already parallel internally (n_jobs=-1 set in _build_model)
        model = _build_model(algo, params, n_jobs_inner=-1)
        model.fit(X_tr.to_numpy(dtype=np.float32), Y_tr)
    else:
        model = _build_model(algo, params, n_jobs_inner=-1)
        with joblib.parallel_backend("threading", n_jobs=-1):
            model.fit(X_tr.to_numpy(dtype=np.float32), Y_tr)

    Y_pred = model.predict(X_va.to_numpy(dtype=np.float32))
    return _mae_multioutput(Y_va, Y_pred)


def _eval_params_cv(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    algo: str,
    params: Dict,
    H: int,
    trial=None,
) -> float:
    import optuna
    maes = []
    for step_idx, (df_tr, df_va) in enumerate(folds):
        fold_mae = _eval_fold(df_tr, df_va, algo, params, H)
        if not np.isfinite(fold_mae):
            return float("nan")
        maes.append(fold_mae)
        if trial is not None:
            trial.report(float(np.mean(maes)), step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(maes))


# ──────────────────────────────────────────────────────────────────────────────
# Optuna search space
# ──────────────────────────────────────────────────────────────────────────────

def _suggest_lgbm(trial) -> Dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
    }


def _suggest_xgb(trial) -> Dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":            trial.suggest_float("gamma", 1e-5, 5.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
    }


def _suggest_rf(trial) -> Dict:
    return {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth":          trial.suggest_int("max_depth", 8, 40),
        "min_samples_leaf":   trial.suggest_int("min_samples_leaf", 1, 15),
        "min_samples_split":  trial.suggest_int("min_samples_split", 2, 20),
        "max_features":       trial.suggest_float("max_features", 0.3, 1.0),
        "bootstrap":          trial.suggest_categorical("bootstrap", [True, False]),
    }


_SUGGEST = {"lgbm": _suggest_lgbm, "xgb": _suggest_xgb, "rf": _suggest_rf}

# Default params (baseline για σύγκριση)
_DEFAULTS: Dict[str, Dict] = {
    "lgbm": {
        "n_estimators": 1500, "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.0, "reg_lambda": 0.0,
    },
    "xgb": {
        "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": 6, "min_child_weight": 1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
    },
    "rf": {
        "n_estimators": 600, "max_depth": 25,
        "min_samples_leaf": 5, "min_samples_split": 2,
        "max_features": 0.8, "bootstrap": True,
    },
}

# Filename convention ώστε eval_mimo.py να αναγνωρίζει σωστά τα μοντέλα
_MODEL_PREFIX = {
    "lgbm": "lgbm_direct",   # → "LGBM DIRECT"
    "xgb":  "xgb_vect",      # → "XGB MIMO"
    "rf":   "rf_mimo",        # → "RF MIMO"
}


# ──────────────────────────────────────────────────────────────────────────────
# Main tuning
# ──────────────────────────────────────────────────────────────────────────────

def run_optuna(
    mode: str,
    algo: str,
    task: str,
    H: int,
    train_end: Optional[str],
    n_trials: int,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
    seed: int,
) -> None:
    import optuna
    import joblib
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(
        df, mode=mode,
        train_end=train_end,
        test_size=H,
    )

    X_sample, Y_sample = _make_supervised_xy(df_train, H)
    n_usable = len(X_sample)
    n_features = X_sample.shape[1]

    print(f"\n{'='*70}")
    print(f" MIMO OPTUNA | algo={algo.upper()} | task={task.upper()} | H={H}")
    print(f" train rows : {len(df_train)}  →  usable (after shift): {n_usable}")
    print(f" features   : {n_features}")
    print(f" n_trials   : {n_trials} | n_splits={n_splits} | val_size={val_size}h")
    print(f" CV metric  : mean MAE across all {H} horizons")
    print(f"{'='*70}\n")

    folds = _rolling_origin_folds(
        df_train=df_train,
        H=H,
        n_splits=n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
    )
    print(f"[CV] {len(folds)} folds | val_size≈{val_size}h | step={step}h")
    for i, (tr, va) in enumerate(folds):
        _, Y_va = _make_supervised_xy(va, H)
        print(f"     fold {i}: train={len(tr)} rows | val_usable={len(Y_va)} rows")

    if not folds:
        print("ERROR: No folds — check min_train_size / val_size.")
        return

    # ── Baseline (default params)
    baseline_mae = _eval_params_cv(folds, algo, _DEFAULTS[algo], H)
    print(f"\n[Baseline] default params → CV MAE = {baseline_mae:.4f}")

    # ── Optuna objective
    suggest_fn = _SUGGEST[algo]

    def objective(trial):
        params = suggest_fn(trial)
        return _eval_params_cv(folds, algo, params, H, trial=trial)

    def callback(study, trial):
        tag = " ← BEST" if trial.value == study.best_value else ""
        status = (
            "PRUNED"
            if trial.state.name == "PRUNED"
            else f"{trial.value:.4f}" if trial.value is not None else "nan"
        )
        print(f"[Trial {trial.number:03d}] MAE={status}{tag}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best_params = study.best_params
    best_mae    = study.best_value
    print(f"\nBest CV MAE : {best_mae:.4f}  (baseline: {baseline_mae:.4f})")
    print(f"Improvement : {100*(baseline_mae - best_mae)/baseline_mae:+.2f}%")
    print(f"Best params : {best_params}")

    # ── Final model on full training data
    print(f"\n[Final] Fitting best {algo.upper()} on full data …")
    X_full, Y_full = _make_supervised_xy(df_train, H)
    final_model = _build_model(algo, best_params, n_jobs_inner=-1)
    if algo == "rf":
        final_model.fit(X_full.to_numpy(dtype=np.float32), Y_full)
    else:
        with joblib.parallel_backend("threading", n_jobs=-1):
            final_model.fit(X_full.to_numpy(dtype=np.float32), Y_full)

    # ── Save
    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    prefix = _MODEL_PREFIX[algo]
    model_path  = MODELS_DIR / f"{prefix}_hourly_{task}_h{H}_optuna.pkl"
    params_path = MODELS_DIR / f"{prefix}_hourly_{task}_h{H}_optuna_params.json"
    log_path    = TUNING_DIR / f"optuna_mimo_{algo}_{task}_h{H}.json"

    bundle = {
        "model":        final_model,
        "feature_cols": list(X_full.columns),
        "horizon":      H,
        "mode":         mode,
        "task":         task,
        "train_end":    train_end,
        "strategy":     "mimo_optuna",
        "lib":          algo,
        "best_params":  best_params,
    }
    joblib.dump(bundle, model_path)

    params_out = {
        "algo": algo, "task": task, "horizon": H,
        "cv_metric": "mean_MAE_all_horizons",
        "n_features": n_features,
        "baseline_mae": round(baseline_mae, 4),
        "best_mae": round(best_mae, 4),
        "improvement_pct": round(100 * (baseline_mae - best_mae) / baseline_mae, 2),
        "best_params": best_params,
    }
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    tuning_log = {
        **params_out,
        "n_trials": n_trials,
        "n_splits": len(folds),
        "val_size": val_size,
        "train_rows": len(df_train),
        "all_trials": [
            {
                "trial":  t.number,
                "value":  round(t.value, 4) if t.value is not None else None,
                "state":  t.state.name,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    log_path.write_text(json.dumps(tuning_log, indent=2), encoding="utf-8")

    print(f"\nSaved model  : {model_path}")
    print(f"Saved params : {params_path}")
    print(f"Saved log    : {log_path}")
    print(f"\n✅ Done! Run eval_mimo.py to see results in dashboard.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna tuning for MIMO MultiOutputRegressor (LGBM / XGB)"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--algo",     choices=["lgbm", "xgb", "rf"], default="lgbm")
    p.add_argument("--task",     choices=["price", "load"], default="price")
    p.add_argument("--horizon",  type=int, default=168, help="Forecast horizon (default: 168)")
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--val_size", type=int, default=None,
                   help="Usable validation rows per fold (default: horizon=168)")
    p.add_argument("--step",     type=int, default=None,
                   help="Step between folds in hours (default: val_size)")
    p.add_argument("--min_train_size", type=int, default=None,
                   help="Min training rows per fold (default: 6*val_size)")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    H              = args.horizon
    val_size       = args.val_size or H
    step           = args.step or val_size
    min_train_size = args.min_train_size or (6 * val_size)

    run_optuna(
        mode=args.mode,
        algo=args.algo,
        task=args.task,
        H=H,
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
