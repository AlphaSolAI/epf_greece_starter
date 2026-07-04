"""
tune_direct_optuna.py — Optuna tuning για Direct Multistep forecasting.

Στρατηγικές:
  --mode-direct shared  : 1 Optuna study για ΟΛΑ τα horizons (MultiOutputRegressor)
                          Γρήγορο, καλό starting point.
  --mode-direct grouped : 3 ξεχωριστές μελέτες για ομάδες horizons:
                          short  h=1..24   (day-ahead)
                          medium h=25..72  (2-3 days)
                          long   h=73..168 (4-7 days)
                          Ακριβότερο αλλά πιο εξειδικευμένο.

Αρχιτεκτονική:
  MultiOutputRegressor(LGBM | XGB) — ένα μοντέλο ανά ομάδα horizons.
  Δεν χρειάζεται OL-aware: δεν υπάρχει recursive component.

CV metric: mean MAE across horizons × rolling-origin folds.

Saved (παράδειγμα lgbm, task=price, shared):
  models/lgbm_hourly_price_mimo_h168_direct_optuna.pkl
  models/lgbm_hourly_price_mimo_h168_direct_optuna_params.json
  tuning/optuna_direct_lgbm_price_h168.json

Saved (παράδειγμα lgbm, task=price, grouped):
  models/lgbm_hourly_price_h24_direct_optuna_short.pkl   (h=1..24)
  models/lgbm_hourly_price_h48_direct_optuna_medium.pkl  (h=25..72)
  models/lgbm_hourly_price_h96_direct_optuna_long.pkl    (h=73..168)
  tuning/optuna_direct_lgbm_price_grouped.json

Usage
-----
# Shared study (1 study για όλα):
python -m src.tune_direct_optuna hourly --algo lgbm --task price \\
    --train_end "2025-11-30 23:00" --n_trials 50 --n_splits 3

# Grouped (3 studies):
python -m src.tune_direct_optuna hourly --algo lgbm --task price \\
    --train_end "2025-11-30 23:00" --n_trials 40 --n_splits 3 --mode-direct grouped
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

# Ομάδες horizons για grouped mode
HORIZON_GROUPS = {
    "short":  (1,  24),   # day-ahead
    "medium": (25, 72),   # 2-3 days
    "long":   (73, 168),  # 4-7 days
}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-output data builder (για συγκεκριμένο range horizons)
# ──────────────────────────────────────────────────────────────────────────────

def _make_supervised_xy(
    df: pd.DataFrame,
    h_start: int,
    h_end: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Δημιουργεί (X, Y) για horizons h_start..h_end (inclusive).
    Y[i] = [y(i+h_start), ..., y(i+h_end)]
    """
    X, _ = make_xy(df)
    y = df["y"].astype(float)

    Y_cols = {f"y_t+{k}": y.shift(-k) for k in range(h_start, h_end + 1)}
    Y_df = pd.DataFrame(Y_cols, index=df.index)

    joined = X.join(Y_df, how="inner").dropna(subset=list(Y_df.columns))
    X_out = joined[X.columns].copy()
    X_out = X_out.select_dtypes(include=[np.number])
    bool_cols = X_out.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        X_out[bool_cols] = X_out[bool_cols].astype(np.int8)
    X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    Y_out = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_out, Y_out


def _mae_multioutput(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(Y_true - Y_pred)))


# ──────────────────────────────────────────────────────────────────────────────
# Rolling-origin CV folds
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_origin_folds(
    df_train: pd.DataFrame,
    H_max: int,      # μέγιστο horizon (για correct window sizing)
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df_train)
    val_window = val_size + H_max
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

def _build_model(algo: str, params: Dict, n_jobs_inner: int = 1) -> MultiOutputRegressor:
    if algo == "lgbm":
        import lightgbm as lgb
        base = lgb.LGBMRegressor(n_jobs=1, verbosity=-1, **params)
    elif algo == "xgb":
        import xgboost as xgb
        base = xgb.XGBRegressor(n_jobs=1, verbosity=0, device="cpu", **params)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    return MultiOutputRegressor(base, n_jobs=n_jobs_inner)


# ──────────────────────────────────────────────────────────────────────────────
# Fold CV evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _eval_fold(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    algo: str,
    params: Dict,
    h_start: int,
    h_end: int,
) -> float:
    X_tr, Y_tr = _make_supervised_xy(df_tr, h_start, h_end)
    X_va, Y_va = _make_supervised_xy(df_va, h_start, h_end)
    if len(X_va) == 0:
        return float("nan")
    model = _build_model(algo, params)
    model.fit(X_tr.to_numpy(dtype=np.float32), Y_tr)
    Y_pred = model.predict(X_va.to_numpy(dtype=np.float32))
    return _mae_multioutput(Y_va, Y_pred)


def _eval_params_cv(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    algo: str,
    params: Dict,
    h_start: int,
    h_end: int,
    trial=None,
) -> float:
    import optuna
    maes = []
    for step_idx, (df_tr, df_va) in enumerate(folds):
        fold_mae = _eval_fold(df_tr, df_va, algo, params, h_start, h_end)
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


_SUGGEST = {"lgbm": _suggest_lgbm, "xgb": _suggest_xgb}

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
}


# ──────────────────────────────────────────────────────────────────────────────
# Run a single Optuna study για horizons h_start..h_end
# ──────────────────────────────────────────────────────────────────────────────

def _run_study(
    algo: str,
    task: str,
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    h_start: int,
    h_end: int,
    n_trials: int,
    seed: int,
    label: str,
) -> Tuple[Dict, float, float]:
    """
    Τρέχει Optuna study για horizons h_start..h_end.
    Επιστρέφει (best_params, best_mae, baseline_mae).
    """
    import optuna

    H_group = h_end - h_start + 1
    baseline_mae = _eval_params_cv(folds, algo, _DEFAULTS[algo], h_start, h_end)
    print(f"  [Baseline-{label}] h={h_start}..{h_end} ({H_group} horizons) → MAE={baseline_mae:.4f}")

    suggest_fn = _SUGGEST[algo]

    def objective(trial):
        params = suggest_fn(trial)
        return _eval_params_cv(folds, algo, params, h_start, h_end, trial=trial)

    def callback(study, trial):
        tag = " ← BEST" if trial.value == study.best_value else ""
        status = (
            "PRUNED" if trial.state.name == "PRUNED"
            else f"{trial.value:.4f}" if trial.value is not None else "nan"
        )
        print(f"  [{label} Trial {trial.number:03d}] MAE={status}{tag}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    best_params = study.best_params
    best_mae    = study.best_value
    impr = 100 * (baseline_mae - best_mae) / baseline_mae if baseline_mae > 0 else 0
    print(f"  [{label}] Best MAE={best_mae:.4f}  baseline={baseline_mae:.4f}  ({impr:+.2f}%)")
    return best_params, best_mae, baseline_mae


# ──────────────────────────────────────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────────────────────────────────────

def run_shared(
    mode: str, algo: str, task: str, H: int,
    train_end: Optional[str], n_trials: int, n_splits: int,
    val_size: int, step: int, min_train_size: int, seed: int,
) -> None:
    """1 study για όλα τα horizons (shared params)."""
    print(f"\n{'='*70}")
    print(f" DIRECT OPTUNA (SHARED) | algo={algo.upper()} | task={task.upper()} | H=1..{H}")
    print(f" 1 study για ΟΛΑ τα horizons — shared hyperparameters")
    print(f"{'='*70}\n")

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(df, mode=mode, train_end=train_end, test_size=H)

    X_s, _ = _make_supervised_xy(df_train, 1, H)
    print(f" train rows: {len(df_train)} | usable: {len(X_s)} | features: {X_s.shape[1]}")
    print(f" n_trials={n_trials} | n_splits={n_splits} | val_size={val_size}h\n")

    folds = _rolling_origin_folds(df_train, H, n_splits, val_size, step, min_train_size)
    print(f"[CV] {len(folds)} folds created")

    best_params, best_mae, baseline_mae = _run_study(
        algo, task, folds, h_start=1, h_end=H,
        n_trials=n_trials, seed=seed, label="ALL",
    )

    # Fit final model
    print(f"\n[Final] Fitting on full training data …")
    X_full, Y_full = _make_supervised_xy(df_train, 1, H)
    final_model = _build_model(algo, best_params, n_jobs_inner=-1)
    final_model.fit(X_full.to_numpy(dtype=np.float32), Y_full)

    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    model_path  = MODELS_DIR / f"{algo}_hourly_{task}_mimo_h{H}_direct_optuna.pkl"
    params_path = MODELS_DIR / f"{algo}_hourly_{task}_mimo_h{H}_direct_optuna_params.json"
    log_path    = TUNING_DIR / f"optuna_direct_{algo}_{task}_h{H}.json"

    bundle = {
        "model":        final_model,
        "feature_cols": list(X_full.columns),
        "horizon":      H,
        "h_start":      1,
        "h_end":        H,
        "mode":         mode,
        "task":         task,
        "train_end":    train_end,
        "strategy":     "direct_optuna_shared",
        "lib":          algo,
        "best_params":  best_params,
    }
    joblib.dump(bundle, model_path)

    params_out = {
        "algo": algo, "task": task, "horizon": H,
        "mode_direct": "shared",
        "cv_metric": f"mean_MAE_h1..{H}",
        "n_features": X_full.shape[1],
        "baseline_mae": round(baseline_mae, 4),
        "best_mae": round(best_mae, 4),
        "improvement_pct": round(100 * (baseline_mae - best_mae) / baseline_mae, 2),
        "best_params": best_params,
    }
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")

    print(f"\nSaved model  : {model_path}")
    print(f"Saved params : {params_path}")
    print(f"\n✅ Done! Re-run eval_mimo.py to update dashboard.")


def run_grouped(
    mode: str, algo: str, task: str, H: int,
    train_end: Optional[str], n_trials: int, n_splits: int,
    val_size: int, step: int, min_train_size: int, seed: int,
) -> None:
    """3 studies για ομάδες horizons: short/medium/long."""
    print(f"\n{'='*70}")
    print(f" DIRECT OPTUNA (GROUPED) | algo={algo.upper()} | task={task.upper()}")
    print(f" short=h1-24 | medium=h25-72 | long=h73-168")
    print(f"{'='*70}\n")

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(df, mode=mode, train_end=train_end, test_size=H)

    folds = _rolling_origin_folds(df_train, H, n_splits, val_size, step, min_train_size)
    print(f"[CV] {len(folds)} folds (shared across all groups)\n")

    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    all_results = {}
    group_seeds = {"short": seed, "medium": seed + 1, "long": seed + 2}

    for group_name, (h_start, h_end) in HORIZON_GROUPS.items():
        h_end_eff = min(h_end, H)
        if h_start > H:
            continue

        print(f"\n{'─'*50}")
        print(f" Group: {group_name.upper()}  (h={h_start}..{h_end_eff})")
        print(f"{'─'*50}")

        best_params, best_mae, baseline_mae = _run_study(
            algo, task, folds,
            h_start=h_start, h_end=h_end_eff,
            n_trials=n_trials, seed=group_seeds[group_name],
            label=group_name.upper(),
        )

        # Fit per-group model
        X_full, Y_full = _make_supervised_xy(df_train, h_start, h_end_eff)
        final_model = _build_model(algo, best_params, n_jobs_inner=-1)
        final_model.fit(X_full.to_numpy(dtype=np.float32), Y_full)

        model_path = MODELS_DIR / f"{algo}_hourly_{task}_h{h_end_eff}_direct_optuna_{group_name}.pkl"
        bundle = {
            "model":        final_model,
            "feature_cols": list(X_full.columns),
            "horizon":      h_end_eff,
            "h_start":      h_start,
            "h_end":        h_end_eff,
            "mode":         mode,
            "task":         task,
            "train_end":    train_end,
            "strategy":     f"direct_optuna_grouped_{group_name}",
            "lib":          algo,
            "best_params":  best_params,
        }
        joblib.dump(bundle, model_path)
        print(f"  Saved: {model_path.name}")

        all_results[group_name] = {
            "h_start": h_start, "h_end": h_end_eff,
            "baseline_mae": round(baseline_mae, 4),
            "best_mae": round(best_mae, 4),
            "improvement_pct": round(100 * (baseline_mae - best_mae) / baseline_mae, 2),
            "best_params": best_params,
        }

    log_path = TUNING_DIR / f"optuna_direct_{algo}_{task}_grouped.json"
    log_path.write_text(json.dumps({
        "algo": algo, "task": task, "mode_direct": "grouped",
        "n_trials": n_trials, "n_splits": n_splits,
        "groups": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved log: {log_path}")
    print(f"\n✅ Grouped Direct Optuna done! 3 models saved.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna tuning for Direct Multistep (shared or grouped horizons)"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--algo",         choices=["lgbm", "xgb"], default="lgbm")
    p.add_argument("--task",         choices=["price", "load"], default="price")
    p.add_argument("--horizon",      type=int, default=168)
    p.add_argument("--mode-direct",  choices=["shared", "grouped"], default="shared",
                   dest="mode_direct",
                   help="shared=1 study all horizons | grouped=3 studies (short/medium/long)")
    p.add_argument("--train_end",    type=str, default=None)
    p.add_argument("--n_trials",     type=int, default=50)
    p.add_argument("--n_splits",     type=int, default=3)
    p.add_argument("--val_size",     type=int, default=None,
                   help="Usable val rows per fold (default: horizon=168)")
    p.add_argument("--step",         type=int, default=None)
    p.add_argument("--min_train_size", type=int, default=None)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    H              = args.horizon
    val_size       = args.val_size or H
    step           = args.step or val_size
    min_train_size = args.min_train_size or (6 * val_size)

    kwargs = dict(
        mode=args.mode, algo=args.algo, task=args.task, H=H,
        train_end=args.train_end, n_trials=args.n_trials,
        n_splits=args.n_splits, val_size=val_size,
        step=step, min_train_size=min_train_size, seed=args.seed,
    )

    if args.mode_direct == "shared":
        run_shared(**kwargs)
    else:
        run_grouped(**kwargs)


if __name__ == "__main__":
    main()
