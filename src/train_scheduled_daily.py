"""
train_scheduled_daily.py

Scheduled Sampling trainer for the DAY-BY-DAY open-loop scenario.

KEY DIFFERENCE from train_scheduled_openloop.py:
  During training, a y_lag_k feature at time t is eligible for replacement
  with a prediction ONLY if k <= t.hour  (i.e., the lag stays within the
  current calendar day).

  Lags that cross a day boundary (k > t.hour, or k >= 24 always) are
  kept as actual values — exactly as they would be at inference time when
  the recursive run resets to actual at each day start.

  Result: the model learns to handle prediction noise for intra-day lags
  while treating cross-day lags as reliable (actual) anchors.

Saved model (distinct suffix, does NOT overwrite other models):
  models/{model}_{mode}_{task}_openloop_daily_optuna_ss.pkl

Typical usage (after tune_openloop_daily_optuna.py has run):
  conda run -n epf --no-capture-output python -m src.train_scheduled_daily hourly \\
      --task price --model lgbm \\
      --train_end "2025-11-30 23:00" \\
      --params_json "models/lgbm_hourly_price_openloop_daily_optuna_params.json" \\
      --n_iter 3 --epsilon_start 0.10 --epsilon_final 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_y_lag_cols(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _make_model(model_type: str, device: str = "cuda", params_override: Optional[dict] = None):
    if model_type == "lgbm":
        from lightgbm import LGBMRegressor
        defaults = dict(
            n_estimators=5000, learning_rate=0.01, num_leaves=128,
            max_depth=-1, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.2, random_state=42, n_jobs=-1, verbose=-1,
        )
        if params_override:
            skip = {"objective", "subsample_freq"}
            for k, v in params_override.items():
                if k not in skip:
                    defaults[k] = v
        return LGBMRegressor(**defaults)

    elif model_type == "xgb":
        import xgboost as xgb_mod
        defaults = dict(
            n_estimators=3000, learning_rate=0.01, max_depth=10,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=0.2, random_state=42,
            tree_method="hist", device=device, n_jobs=-1,
            objective="reg:squarederror",
        )
        if params_override:
            skip = {"objective", "tree_method", "device", "n_jobs"}
            for k, v in params_override.items():
                if k not in skip:
                    defaults[k] = v
        return xgb_mod.XGBRegressor(**defaults)

    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        defaults = dict(
            n_estimators=400, max_depth=16, min_samples_split=4,
            min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1,
        )
        if params_override:
            skip = {"n_jobs", "random_state"}
            for k, v in params_override.items():
                if k not in skip:
                    defaults[k] = v
        return RandomForestRegressor(**defaults)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ── Day-by-day SS round ───────────────────────────────────────────────────────
def _daily_ss_round(
    model_prev,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build augmented training set for one day-by-day SS round.

    For each y_lag_k feature and training sample at time t:
      - If k > t.hour  → lag crosses day boundary → keep ACTUAL (never replace)
      - If k <= t.hour → lag is within same day   → replace with prob ε

    This matches the inference scenario: at hour h of a day, lags 1..h are
    predictions (from the current day's recursion), while lags h+1..23+ are
    actual (from previous days).
    """
    y_hat = pd.Series(
        model_prev.predict(X_train),
        index=X_train.index,
        dtype=float,
    )
    X_aug = X_train.copy()
    y_lag_cols = _get_y_lag_cols(X_train)

    # Per-sample replacement decision (same for all eligible lags → consistent trajectory)
    replace_mask = rng.random(len(X_train)) < epsilon

    # Hour of day for each training sample (0-23)
    hour_of_day = pd.Series(X_train.index.hour, index=X_train.index)

    n_replaced = 0
    n_eligible = 0

    for col in y_lag_cols:
        lag_k = int(col.replace("y_lag", ""))

        # Eligible only if lag is within the same day (lag_k <= hour_of_day)
        within_day = hour_of_day >= lag_k  # boolean Series, same index as X_train

        pseudo_lag = y_hat.shift(lag_k).reindex(X_train.index)

        valid = replace_mask & within_day.values & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values

        n_replaced += int(valid.sum())
        n_eligible += int((within_day & pseudo_lag.notna()).sum())

    total_cells = len(y_lag_cols) * len(X_train)
    print(
        f"   [SS-daily] ε={epsilon:.2f} | "
        f"eligible={n_eligible}/{total_cells} ({100*n_eligible/total_cells:.1f}%) | "
        f"replaced={n_replaced} ({100*n_replaced/max(n_eligible,1):.1f}% of eligible)"
    )
    return X_aug


# ── Main training ─────────────────────────────────────────────────────────────
def train(
    mode: str,
    task: str,
    model_type: str,
    train_end: Optional[str],
    n_iter: int,
    epsilon_start: float,
    epsilon_final: float,
    device: str,
    seed: int,
    params_json: Optional[str],
) -> None:
    rng = np.random.default_rng(seed)

    params_override = None
    if params_json:
        with open(params_json, "r", encoding="utf-8") as f:
            params_override = json.load(f)
        print(f"   -> Params loaded from: {params_json}")

    df = load_processed(mode, task=task)
    if train_end:
        df = df.loc[: pd.to_datetime(train_end)]

    X_train, y_train = make_xy(df)
    y_lag_cols = _get_y_lag_cols(X_train)

    print(f"\n🔄 TRAINING {model_type.upper()} DAY-BY-DAY SCHEDULED SAMPLING")
    print(f"   mode={mode} | task={task.upper()} | train={len(df)} rows | features={X_train.shape[1]}")
    print(f"   y_lag cols ({len(y_lag_cols)}): {y_lag_cols}")
    print(f"   n_iter={n_iter} | ε: {epsilon_start:.2f} → {epsilon_final:.2f}")
    print(f"   Rule: only replace lag-k if k <= hour_of_day (intra-day lags only)")

    # ── Iter 0: teacher-forced ──
    print(f"\n📚 [Iter 0] Teacher-forced base training …")
    model = _make_model(model_type, device=device, params_override=params_override)
    model.fit(X_train, y_train)
    print(f"   ✅ Base model trained.")

    # ── SS iterations ──
    epsilons = np.linspace(epsilon_start, epsilon_final, n_iter) if n_iter > 1 else [epsilon_final]
    for i, eps in enumerate(epsilons, start=1):
        print(f"\n🔁 [Iter {i}/{n_iter}] Day-by-day SS (ε={eps:.3f}) …")
        X_aug = _daily_ss_round(model, X_train, y_train, eps, rng)
        model_new = _make_model(model_type, device=device, params_override=params_override)
        model_new.fit(X_aug, y_train)
        model = model_new
        print(f"   ✅ Iteration {i} done.")

    # ── Save ──
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"{model_type}_{mode}_{task}_openloop_daily_optuna_ss.pkl"
    joblib.dump(model, out_path)
    print(f"\n✅ Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Day-by-day Scheduled Sampling trainer")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",    choices=["price", "load"], default="price")
    p.add_argument("--model",   choices=["lgbm", "xgb", "rf"], default="lgbm")
    p.add_argument("--train_end",      type=str,   default=None)
    p.add_argument("--n_iter",         type=int,   default=3)
    p.add_argument("--epsilon_start",  type=float, default=0.10)
    p.add_argument("--epsilon_final",  type=float, default=0.40)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument(
        "--params_json", type=str, default=None,
        help="Path to Optuna params JSON (e.g. models/lgbm_hourly_price_openloop_daily_optuna_params.json)"
    )
    args = p.parse_args()

    train(
        mode=args.mode,
        task=args.task,
        model_type=args.model,
        train_end=args.train_end,
        n_iter=args.n_iter,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        device=args.device,
        seed=args.seed,
        params_json=args.params_json,
    )


if __name__ == "__main__":
    main()
