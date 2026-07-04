"""
Recursive-Aware Training via Scheduled Sampling for LightGBM Open-loop.

Motivation
----------
In standard training, y_lag features always hold the true historical values
(teacher-forcing). At open-loop inference time however, the model receives
its *own* predictions for those lag positions — which carry accumulated
prediction errors. This train/test distribution mismatch degrades multi-step
accuracy.

Scheduled Sampling (Bengio et al., 2015 — adapted for regression/trees)
------------------------------------------------------------------------
1.  Train a base model M0 on the original data (teacher-forced).
2.  Use M0 to compute rolling in-sample predictions on the training set.
3.  Build an augmented training set: with probability ε replace each y_lag
    column with M0's prediction shifted by the corresponding lag, simulating
    the noise the model will see at inference time.
4.  Retrain model M1 on the augmented data.
5.  Optionally repeat for N_ITER rounds with increasing ε (curriculum).

The saved model (`lgbm_hourly_{task}_scheduled_openloop.pkl`) is a standard
LGBMRegressor and is fully compatible with the existing `eval_openloop.py`
evaluation pipeline.

Usage
-----
python -m src.train_lgbm_scheduled_openloop hourly --task price \\
    --train_end "2025-11-30 23:00" --n_iter 3 --epsilon_final 0.6
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .split_utils import load_processed, make_xy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _select_train_window(
    df: pd.DataFrame,
    train_start: Optional[str],
    train_end: Optional[str],
) -> pd.DataFrame:
    if train_start:
        df = df.loc[pd.to_datetime(train_start):]
    if train_end:
        df = df.loc[: pd.to_datetime(train_end)]
    return df


def _get_y_lag_cols(X: pd.DataFrame) -> List[str]:
    """Return only direct y_lagN columns (not rolling means, not load_lag*)."""
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _make_lgbm(**kwargs) -> LGBMRegressor:
    defaults = dict(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=128,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(kwargs)
    return LGBMRegressor(**defaults)


# -----------------------------------------------------------------------
# Core: single scheduled-sampling round
# -----------------------------------------------------------------------

def _scheduled_sampling_round(
    model_prev: LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build an augmented training set for one scheduled-sampling round.

    For each training sample t and each y_lag_h column:
      - With probability `epsilon`, replace y_lag_h(t) with the prediction
        that `model_prev` would make for time (t - h), i.e. shift the
        rolling in-sample prediction series by h steps.
      - Otherwise keep the original actual value.

    Parameters
    ----------
    model_prev : fitted LGBMRegressor from the previous iteration
    X_train    : original feature matrix (DataFrame with DatetimeIndex)
    y_train    : original target series
    epsilon    : probability of replacing a lag value with a pseudo-prediction
    rng        : numpy random generator for reproducibility

    Returns
    -------
    X_aug : augmented feature DataFrame (same shape as X_train)
    """
    # In-sample predictions on the ORIGINAL features (not augmented) so
    # we propagate the previous model's output, not a compound error.
    y_hat = pd.Series(
        model_prev.predict(X_train),
        index=X_train.index,
        dtype=float,
    )

    X_aug = X_train.copy()
    y_lag_cols = _get_y_lag_cols(X_train)

    # Sampling mask: shape (n_samples,) — same mask for all lag columns
    # (mimics the realistic scenario where either the whole autoregressive
    # trajectory is predicted or not, not a random per-column flip)
    mask = rng.random(len(X_train)) < epsilon

    n_replaced = 0
    for col in y_lag_cols:
        lag_h = int(col.replace("y_lag", ""))
        # Shift rolling predictions backwards to align with the lag:
        # y_hat.shift(lag_h)[t] = ŷ(t - lag_h)  ← what open-loop would use
        pseudo_lag = y_hat.shift(lag_h).reindex(X_train.index)
        # Only replace where mask is True AND the pseudo value is available
        valid = mask & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values
        n_replaced += int(valid.sum())

    print(
        f"   [SS] ε={epsilon:.2f} | lag-col replacements: "
        f"{n_replaced} / {len(y_lag_cols) * len(X_train)}"
        f" ({100*n_replaced/(len(y_lag_cols)*len(X_train)):.1f}%)"
    )
    return X_aug


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------

def train(
    mode: str,
    task: str = "price",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    n_iter: int = 3,
    epsilon_start: float = 0.2,
    epsilon_final: float = 0.6,
    seed: int = 42,
) -> None:
    """
    Train LightGBM with scheduled sampling (recursive-aware) for open-loop.

    Parameters
    ----------
    mode          : 'hourly' (or 'daily' if applicable)
    task          : 'price' or 'load'
    train_start   : optional ISO datetime string, lower bound of training
    train_end     : optional ISO datetime string, upper bound of training
    n_iter        : number of scheduled-sampling rounds (default: 3)
    epsilon_start : initial replacement probability (default: 0.2)
    epsilon_final : final replacement probability (default: 0.6)
    seed          : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    df = load_processed(mode, task=task)
    df_train = _select_train_window(df, train_start=train_start, train_end=train_end)
    X_train, y_train = make_xy(df_train)

    y_lag_cols = _get_y_lag_cols(X_train)

    print(
        f"🔄 TRAINING LightGBM SCHEDULED SAMPLING OPENLOOP "
        f"({mode.upper()} | task={task.upper()}) [RECURSIVE-AWARE]"
    )
    print(f"   -> train={len(df_train)} | features={X_train.shape[1]}")
    print(f"   -> y_lag cols: {y_lag_cols}")
    print(f"   -> n_iter={n_iter} | ε: {epsilon_start:.2f} → {epsilon_final:.2f}")

    # ---- Iteration 0: standard training (teacher-forced baseline)
    print("\n📚 [Iter 0] Standard training (teacher-forced) …")
    model = _make_lgbm()
    model.fit(X_train, y_train)
    print("   ✅ Base model trained.")

    # ---- Scheduled-sampling rounds
    # Linear schedule from epsilon_start to epsilon_final over n_iter steps
    if n_iter > 1:
        epsilons = np.linspace(epsilon_start, epsilon_final, n_iter)
    else:
        epsilons = [epsilon_final]

    for i, eps in enumerate(epsilons, start=1):
        print(f"\n🔁 [Iter {i}/{n_iter}] Scheduled sampling round (ε={eps:.3f}) …")
        X_aug = _scheduled_sampling_round(
            model_prev=model,
            X_train=X_train,
            y_train=y_train,
            epsilon=eps,
            rng=rng,
        )
        model_new = _make_lgbm()
        model_new.fit(X_aug, y_train)
        model = model_new
        print(f"   ✅ Iteration {i} model trained.")

    # ---- Save
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"lgbm_{mode}_{task}_scheduled_openloop.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Saved LightGBM SCHEDULED OPENLOOP to: {model_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM with Recursive-Aware Scheduled Sampling"
    )
    parser.add_argument("mode", choices=["daily", "hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="Number of scheduled-sampling rounds (default: 3)",
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=0.2,
        help="Initial replacement probability (default: 0.2)",
    )
    parser.add_argument(
        "--epsilon_final",
        type=float,
        default=0.6,
        help="Final replacement probability (default: 0.6)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        mode=args.mode,
        task=args.task,
        train_start=args.train_start,
        train_end=args.train_end,
        n_iter=args.n_iter,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
