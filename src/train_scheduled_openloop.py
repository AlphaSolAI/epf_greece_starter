"""
Generic Recursive-Aware Training (Scheduled Sampling) for tree-based models.

Supports: lgbm, xgb, rf
Saves: {model}_{mode}_{task}_scheduled_openloop.pkl

Usage
-----
python -m src.train_scheduled_openloop hourly --task price --model xgb \\
    --train_end "2025-11-30 23:00"
python -m src.train_scheduled_openloop hourly --task price --model rf  \\
    --train_end "2025-11-30 23:00"
python -m src.train_scheduled_openloop hourly --task price --model lgbm \\
    --train_end "2025-11-30 23:00"

# Train all three sequentially:
python -m src.train_scheduled_openloop hourly --task price --model all \\
    --train_end "2025-11-30 23:00"
"""

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

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

ALL_MODELS = ["lgbm", "xgb", "rf"]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _get_y_lag_cols(X: pd.DataFrame) -> List[str]:
    """Return only direct y_lagN columns (not rolling means, not load_lag*)."""
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _make_model(model_type: str, mode: str = "hourly", device: str = "cuda", params_override: Optional[dict] = None):
    """Return a fresh (unfitted) estimator with production-grade hyperparameters.

    If params_override is provided (e.g. from Optuna JSON), those values
    replace the defaults for the lgbm model.
    """
    if model_type == "lgbm":
        from lightgbm import LGBMRegressor
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
        if params_override:
            # Strip non-LGBMRegressor keys, then merge
            skip = {"objective", "subsample_freq"}
            for k, v in params_override.items():
                if k not in skip:
                    defaults[k] = v
        return LGBMRegressor(**defaults)

    elif model_type == "xgb":
        import xgboost as xgb_module
        defaults = dict(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42,
            tree_method="hist",
            device=device,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        if params_override:
            skip = {"objective", "tree_method", "device", "n_jobs", "verbosity", "eval_metric", "early_stopping_rounds"}
            for k, v in params_override.items():
                if k not in skip:
                    defaults[k] = v
        return xgb_module.XGBRegressor(**defaults)

    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        # Fewer trees than standard (900) to keep SS iterations tractable
        defaults = dict(
            n_estimators=400,
            max_depth=16 if mode == "hourly" else 18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        if params_override:
            skip = {"n_jobs", "random_state"}
            for k, v in params_override.items():
                if k not in skip:
                    defaults[k] = v
        return RandomForestRegressor(**defaults)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use: lgbm, xgb, rf.")


# -----------------------------------------------------------------------
# Core: one scheduled-sampling round
# -----------------------------------------------------------------------

def _scheduled_sampling_round(
    model_prev,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build augmented training set for one SS round.

    For each y_lag_h feature and training sample t:
    - With probability `epsilon` → replace y_lag_h(t) with ŷ(t-h)
                                    (model_prev's prediction shifted by h steps)
    - Otherwise            → keep original actual value y(t-h)
    """
    y_hat = pd.Series(
        model_prev.predict(X_train),
        index=X_train.index,
        dtype=float,
    )
    X_aug = X_train.copy()
    y_lag_cols = _get_y_lag_cols(X_train)

    # One mask per sample — same decision for all lag columns (realistic trajectory)
    mask = rng.random(len(X_train)) < epsilon

    n_replaced = 0
    for col in y_lag_cols:
        lag_h = int(col.replace("y_lag", ""))
        pseudo_lag = y_hat.shift(lag_h).reindex(X_train.index)
        valid = mask & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values
        n_replaced += int(valid.sum())

    total = len(y_lag_cols) * len(X_train)
    print(
        f"   [SS] ε={epsilon:.2f} | replacements: "
        f"{n_replaced}/{total} ({100 * n_replaced / total:.1f}%)"
    )
    return X_aug


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------

def train(
    mode: str,
    task: str = "price",
    model_type: str = "xgb",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    n_iter: int = 3,
    epsilon_start: float = 0.2,
    epsilon_final: float = 0.6,
    device: str = "cuda",
    seed: int = 42,
    params_json: Optional[str] = None,
    model_suffix: Optional[str] = None,
) -> None:
    rng = np.random.default_rng(seed)

    # Load optional params override (e.g. from Optuna)
    params_override = None
    if params_json:
        with open(params_json, "r", encoding="utf-8") as f:
            params_override = json.load(f)
        print(f"   -> Loaded params override from: {params_json}")

    df = load_processed(mode, task=task)
    if train_start:
        df = df.loc[pd.to_datetime(train_start):]
    if train_end:
        df = df.loc[: pd.to_datetime(train_end)]

    X_train, y_train = make_xy(df)
    y_lag_cols = _get_y_lag_cols(X_train)

    suffix_tag = f" [{model_suffix}]" if model_suffix else ""
    print(
        f"\n🔄 TRAINING {model_type.upper()} SCHEDULED SAMPLING OPENLOOP "
        f"({mode.upper()} | task={task.upper()}) [RECURSIVE-AWARE]{suffix_tag}"
    )
    print(f"   -> train={len(df)} | features={X_train.shape[1]}")
    print(f"   -> y_lag cols: {y_lag_cols}")
    print(f"   -> n_iter={n_iter} | ε: {epsilon_start:.2f} → {epsilon_final:.2f}")

    # --- Iter 0: standard teacher-forced training
    print(f"\n📚 [Iter 0] Standard training (teacher-forced) …")
    model = _make_model(model_type, mode=mode, device=device, params_override=params_override)
    model.fit(X_train, y_train)
    print(f"   ✅ Base model trained.")

    # --- SS rounds with linearly increasing epsilon (curriculum)
    epsilons = np.linspace(epsilon_start, epsilon_final, n_iter) if n_iter > 1 else [epsilon_final]

    for i, eps in enumerate(epsilons, start=1):
        print(f"\n🔁 [Iter {i}/{n_iter}] Scheduled sampling (ε={eps:.3f}) …")
        X_aug = _scheduled_sampling_round(model, X_train, y_train, eps, rng)
        model_new = _make_model(model_type, mode=mode, device=device, params_override=params_override)
        model_new.fit(X_aug, y_train)
        model = model_new
        print(f"   ✅ Iteration {i} model trained.")

    # --- Save
    MODELS_DIR.mkdir(exist_ok=True)
    fname = f"{model_type}_{mode}_{task}"
    if model_suffix:
        fname += f"_{model_suffix}"
    fname += "_scheduled_openloop.pkl"
    model_path = MODELS_DIR / fname
    joblib.dump(model, model_path)
    print(f"\n✅ Saved {model_type.upper()} SCHEDULED OPENLOOP to: {model_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic Scheduled Sampling (Recursive-Aware) trainer"
    )
    parser.add_argument("mode", choices=["daily", "hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument(
        "--model",
        choices=["lgbm", "xgb", "rf", "all"],
        default="xgb",
        help="Model type (or 'all' to train lgbm, xgb, rf sequentially)",
    )
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--epsilon_start", type=float, default=0.2)
    parser.add_argument("--epsilon_final", type=float, default=0.6)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--params_json",
        type=str,
        default=None,
        help="Path to JSON file with hyperparameter overrides (e.g. from Optuna). "
             "Only applied to lgbm model type.",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default=None,
        help="Optional suffix for saved model filename, e.g. 'optuna' → "
             "{model}_{mode}_{task}_optuna_scheduled_openloop.pkl",
    )
    args = parser.parse_args()

    models_to_train = ALL_MODELS if args.model == "all" else [args.model]

    for m in models_to_train:
        train(
            mode=args.mode,
            task=args.task,
            model_type=m,
            train_start=args.train_start,
            train_end=args.train_end,
            n_iter=args.n_iter,
            epsilon_start=args.epsilon_start,
            epsilon_final=args.epsilon_final,
            device=args.device,
            seed=args.seed,
            params_json=args.params_json,
            model_suffix=args.model_suffix,
        )


if __name__ == "__main__":
    main()
