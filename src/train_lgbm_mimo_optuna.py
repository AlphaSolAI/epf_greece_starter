"""
Train a LightGBM MIMO model for price or load forecasting.

Hyperparameters are tuned with Optuna on a tail-validation split.
After tuning the final model is retrained on ALL training data.

Save path:  models/lgbm_{mode}_{task}_mimo_h{horizon}.pkl
Bundle:     {"kind": "lgbm_mimo", "mode", "task", "horizon",
             "feature_cols", "model", "dense_lags", "best_params"}

Usage:
    python -m src.train_lgbm_mimo_optuna hourly --task price --horizon 24
    python -m src.train_lgbm_mimo_optuna hourly --task load  --horizon 24
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers  (mirrors train_xgb_mimo.py)
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(np.int8)
            continue
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            X[c] = s.view("int64")
            continue
        if s.dtype == object:
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() > 0 and dt.isna().sum() < len(dt):
                X[c] = dt.view("int64")
            else:
                X[c] = pd.to_numeric(s, errors="coerce")
    X = (
        X.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    return X


def _build_mimo_xy(df_train: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Build (X, Y) for MIMO training.  No dense lags (those belong to Direct)."""
    X0, _ = make_xy(df_train)
    y = df_train["y"].astype(float)
    X0 = _ensure_numeric_features(X0)

    # targets: [y(t+1), ..., y(t+h)]
    Y = np.column_stack(
        [y.shift(-k).to_numpy() for k in range(1, horizon + 1)]
    ).astype(np.float32)

    valid = np.isfinite(Y).all(axis=1)
    return X0.loc[valid], Y[valid]


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgbm(params: dict, n_jobs_mor: int = 4) -> MultiOutputRegressor:
    base = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        verbose=-1,
        random_state=42,
        **params,
    )
    return MultiOutputRegressor(base, n_jobs=n_jobs_mor)


def _build_objective(X_tr, Y_tr, X_val, Y_val, n_jobs_mor: int):
    """Return an Optuna objective function closed over the val split."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 3000, step=100),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        }

        model = _make_lgbm(params, n_jobs_mor=n_jobs_mor)
        try:
            with joblib.parallel_backend("threading"):
                model.fit(X_tr, Y_tr)
            Y_pred = model.predict(X_val)
            mae = float(mean_absolute_error(Y_val, Y_pred))
        except Exception as e:
            print(f"  [trial {trial.number}] FAILED: {e}")
            return float("inf")

        return mae

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm_mimo(
    mode: str,
    task: str,
    horizon: int,
    train_start: Optional[str],
    train_end: Optional[str],
    n_trials: int,
    val_hours: int,
    n_jobs_mor: int,
) -> Path:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    MODELS_DIR.mkdir(exist_ok=True)

    df = load_processed(mode, task=task)
    df_train, _ = split_time_series(
        df,
        mode=mode,
        test_size=168,          # dummy – only df_train is used
        train_start=train_start,
        train_end=train_end,
        test_start=None,
        test_end=None,
    )

    X_all, Y_all = _build_mimo_xy(df_train, horizon=horizon)

    # ── Validation split (tail) ──────────────────────────────────────────────
    n_total = len(X_all)
    n_val   = min(val_hours, n_total // 5)   # cap at 20 % of data
    n_tr    = n_total - n_val

    X_tr = X_all.iloc[:n_tr].to_numpy(dtype=np.float32)
    Y_tr = Y_all[:n_tr]
    X_val = X_all.iloc[n_tr:].to_numpy(dtype=np.float32)
    Y_val = Y_all[n_tr:]

    print(f"🚀 TRAINING LightGBM MIMO (HOURLY | task={task.upper()}) with Optuna")
    print(f"   horizon={horizon} | n_trials={n_trials} | n_jobs_mor={n_jobs_mor}")
    print(f"   train total={n_total} | opt-train={n_tr} | val={n_val}")
    print(f"   features={X_all.shape[1]}")
    print(f"   train_window={df_train.index.min()} -> {df_train.index.max()}")

    # ── Optuna study ─────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    objective = _build_objective(X_tr, Y_tr, X_val, Y_val, n_jobs_mor=n_jobs_mor)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_val_mae = study.best_value
    print(f"\n🏆 Best trial: val_MAE={best_val_mae:.4f}")
    print(f"   params: {best_params}")

    # ── Refit on ALL training data ────────────────────────────────────────────
    print("\n🔁 Refitting on full training set …")
    final_model = _make_lgbm(best_params, n_jobs_mor=n_jobs_mor)
    X_full = X_all.to_numpy(dtype=np.float32)
    with joblib.parallel_backend("threading"):
        final_model.fit(X_full, Y_all)
    print("   Done.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = MODELS_DIR / f"lgbm_{mode}_{task}_mimo_h{horizon}.pkl"
    joblib.dump(
        {
            "kind":         "lgbm_mimo",
            "mode":         mode,
            "task":         task,
            "horizon":      int(horizon),
            "feature_cols": list(X_all.columns),
            "model":        final_model,
            "dense_lags":   False,
            "best_params":  best_params,
            "best_val_mae": float(best_val_mae),
        },
        out_path,
    )
    print(f"✅ Saved LightGBM MIMO to: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM MIMO with Optuna tuning.")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",        choices=["price", "load"], default="price")
    p.add_argument("--horizon",     type=int, default=24)
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end",   type=str, default=None)
    p.add_argument("--n_trials",    type=int, default=40,
                   help="Number of Optuna trials (default: 40)")
    p.add_argument("--val_hours",   type=int, default=720,
                   help="Hours for tail validation split (default: 720 = 30 days)")
    p.add_argument("--n_jobs_mor",  type=int, default=4,
                   help="n_jobs for MultiOutputRegressor (default: 4)")
    args = p.parse_args()

    train_lgbm_mimo(
        mode=args.mode,
        task=args.task,
        horizon=args.horizon,
        train_start=args.train_start,
        train_end=args.train_end,
        n_trials=args.n_trials,
        val_hours=args.val_hours,
        n_jobs_mor=args.n_jobs_mor,
    )


if __name__ == "__main__":
    main()
