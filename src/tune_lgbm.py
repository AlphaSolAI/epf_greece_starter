import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy, split_time_series

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TUNING_DIR = BASE_DIR / "tuning"


def _import_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        return lgb, LGBMRegressor
    except Exception as e:
        raise RuntimeError(
            "LightGBM is not available in your env. Install it in conda env 'epf' (e.g., pip/conda install lightgbm). "
            f"Original error: {type(e).__name__}: {e}"
        )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else float("nan")


def rolling_origin_folds(
    df_train: pd.DataFrame,
    n_splits: int,
    val_size: int,
    step: int,
    min_train_size: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Rolling-origin CV inside TRAIN (no leakage):
      train = [0 : val_start), val = [val_start : val_end)
    anchored at the END of df_train.

    Returns folds in chronological order (oldest -> newest).
    """
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

    folds = list(reversed(folds))
    if not folds:
        raise ValueError(
            f"Could not create rolling folds. "
            f"len(train)={n}, n_splits={n_splits}, val_size={val_size}, step={step}, min_train_size={min_train_size}"
        )
    return folds


@dataclass
class TuneResult:
    params: Dict[str, object]
    mean_mae: float
    std_mae: float
    mean_best_iter: float


def sample_params(rng: np.random.Generator, mode: str) -> Dict[str, object]:
    # mild, safe search space
    if mode == "hourly":
        num_leaves = int(rng.integers(64, 256))
        min_child_samples = int(rng.integers(20, 240))
    else:
        num_leaves = int(rng.integers(31, 160))
        min_child_samples = int(rng.integers(10, 120))

    learning_rate = float(np.exp(rng.uniform(np.log(0.01), np.log(0.08))))
    subsample = float(rng.uniform(0.6, 1.0))
    colsample_bytree = float(rng.uniform(0.6, 1.0))
    reg_alpha = float(rng.uniform(0.0, 2.0))
    reg_lambda = float(rng.uniform(0.0, 10.0))
    min_split_gain = float(rng.uniform(0.0, 1.0))
    max_depth = int(rng.choice([-1, 6, 8, 10, 12]))

    return dict(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_split_gain=min_split_gain,
    )


def evaluate_params_on_folds(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    params: Dict[str, object],
    max_estimators: int,
    early_stopping_rounds: int,
) -> TuneResult:
    lgb, LGBMRegressor = _import_lightgbm()

    fold_maes = []
    fold_best_iters = []

    for (tr_df, va_df) in folds:
        X_tr, y_tr = make_xy(tr_df)
        X_va, y_va = make_xy(va_df)

        model = LGBMRegressor(**params, n_estimators=max_estimators)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=callbacks,
        )

        best_it = int(model.best_iteration_ or max_estimators)
        yhat = model.predict(X_va, num_iteration=best_it)

        fold_maes.append(mae(y_va, yhat))
        fold_best_iters.append(best_it)

    return TuneResult(
        params=params,
        mean_mae=float(np.mean(fold_maes)),
        std_mae=float(np.std(fold_maes)),
        mean_best_iter=float(np.mean(fold_best_iters)),
    )


def fit_final_model(
    df_train: pd.DataFrame,
    params: Dict[str, object],
    val_size: int,
    max_estimators: int,
    early_stopping_rounds: int,
) -> Tuple[object, int]:
    lgb, LGBMRegressor = _import_lightgbm()

    if len(df_train) <= val_size + 50:
        raise ValueError(f"Train too small for internal val. train={len(df_train)} val_size={val_size}")

    df_tr = df_train.iloc[:-val_size].copy()
    df_va = df_train.iloc[-val_size:].copy()

    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    tmp = LGBMRegressor(**params, n_estimators=max_estimators)
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    tmp.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="mae", callbacks=callbacks)
    best_iter = int(tmp.best_iteration_ or max_estimators)

    X_full, y_full = make_xy(df_train)
    final_model = LGBMRegressor(**params, n_estimators=best_iter)
    final_model.fit(X_full, y_full)
    return final_model, best_iter


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m src.tune_lgbm",
        description="Rolling-origin CV tuning for LightGBM INSIDE TRAIN (no leakage).",
    )
    ap.add_argument("mode", choices=["daily", "hourly"])
    ap.add_argument("--task", choices=["price", "load"], default="price")

    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--n_splits", type=int, default=8)
    ap.add_argument("--val_size", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--min_train_size", type=int, default=None)

    ap.add_argument("--max_estimators", type=int, default=6000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--test_size", type=int, default=None)
    ap.add_argument("--train_start", type=str, default=None)
    ap.add_argument("--train_end", type=str, default=None)
    ap.add_argument("--test_start", type=str, default=None)
    ap.add_argument("--test_end", type=str, default=None)

    args = ap.parse_args(argv)

    # IMPORTANT: your pipeline writes the target into hourly.parquet / daily.parquet.
    # So we load_processed(mode) only (no task argument).
    df = load_processed(args.mode)

    df_train, _df_test = split_time_series(
        df,
        mode=args.mode,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    val_size = args.val_size or (168 if args.mode == "hourly" else 30)
    step = args.step or val_size
    min_train_size = args.min_train_size or ((24 * 365) if args.mode == "hourly" else 365)

    folds = rolling_origin_folds(
        df_train=df_train,
        n_splits=args.n_splits,
        val_size=val_size,
        step=step,
        min_train_size=min_train_size,
    )

    rng = np.random.default_rng(args.seed)

    baseline_params = dict(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        learning_rate=0.05,
        num_leaves=96 if args.mode == "hourly" else 64,
        max_depth=-1,
        min_child_samples=40 if args.mode == "hourly" else 20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_split_gain=0.0,
    )

    print(f"\n🔎 TUNING LightGBM ({args.mode.upper()} | declared task={args.task.upper()}) [rolling-origin CV inside TRAIN]")
    print(f"[INFO] train_rows={len(df_train)} folds={len(folds)} val_size={val_size} step={step} min_train_size={min_train_size}")
    print(f"[INFO] n_trials={args.n_trials} max_estimators={args.max_estimators} early_stop={args.early_stopping_rounds} seed={args.seed}")
    print("[NOTE] This assumes you already built the processed dataset for the correct task (price/load) in the parquet.")

    results: List[Dict[str, object]] = []

    best = evaluate_params_on_folds(
        folds=folds,
        params=baseline_params,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    results.append(
        dict(
            trial=0,
            mean_mae=best.mean_mae,
            std_mae=best.std_mae,
            mean_best_iter=best.mean_best_iter,
            params=baseline_params,
        )
    )
    print(f"[TRIAL 0] meanMAE={best.mean_mae:.4f} ±{best.std_mae:.4f} | meanBestIter={best.mean_best_iter:.1f} | (baseline params)")

    for t in range(1, args.n_trials + 1):
        params = sample_params(rng, args.mode)
        tr = evaluate_params_on_folds(
            folds=folds,
            params=params,
            max_estimators=args.max_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        results.append(
            dict(
                trial=t,
                mean_mae=tr.mean_mae,
                std_mae=tr.std_mae,
                mean_best_iter=tr.mean_best_iter,
                params=params,
            )
        )
        improved = tr.mean_mae < best.mean_mae
        tag = "✅ NEW BEST" if improved else ""
        if improved:
            best = tr
        print(f"[TRIAL {t:02d}] meanMAE={tr.mean_mae:.4f} ±{tr.std_mae:.4f} | meanBestIter={tr.mean_best_iter:.1f} {tag}")

    final_model, best_iter = fit_final_model(
        df_train=df_train,
        params=best.params,
        val_size=val_size,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    MODELS_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / f"lgbm_{args.mode}_{args.task}_tuned.pkl"
    joblib.dump(final_model, model_path)

    params_out = dict(best.params)
    params_out["n_estimators"] = int(best_iter)

    params_path = MODELS_DIR / f"lgbm_{args.mode}_{args.task}_tuned_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_out, f, ensure_ascii=False, indent=2)

    tuning_path = TUNING_DIR / f"tune_lgbm_{args.mode}_{args.task}.json"
    with open(tuning_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                mode=args.mode,
                task=args.task,
                n_trials=args.n_trials,
                n_splits=len(folds),
                val_size=val_size,
                step=step,
                results=sorted(results, key=lambda r: r["mean_mae"]),
                best=dict(mean_mae=best.mean_mae, std_mae=best.std_mae, params=params_out),
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n✅ DONE")
    print(f"   -> Saved tuned model : {model_path}")
    print(f"   -> Saved tuned params: {params_path}")
    print(f"   -> Saved tuning log  : {tuning_path}")


if __name__ == "__main__":
    main()
