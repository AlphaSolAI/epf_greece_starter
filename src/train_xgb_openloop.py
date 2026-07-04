import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
import xgboost as xgb
import shutil
import inspect

from .split_utils import load_processed, make_xy
from .model_wrappers import ResidualAddBaselineWrapper

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def select_train_window(
    df: pd.DataFrame,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> pd.DataFrame:
    if train_start:
        df = df.loc[pd.to_datetime(train_start):]
    if train_end:
        df = df.loc[:pd.to_datetime(train_end)]
    return df


def split_train_val_by_time(
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(test_start)
    te = pd.to_datetime(test_end)
    if te < ts:
        raise ValueError(f"test_end ({test_end}) must be >= test_start ({test_start}).")

    df_train = df.loc[: ts - pd.Timedelta(hours=1)].copy()
    df_val = df.loc[ts:te].copy()

    if len(df_train) == 0:
        raise ValueError("Train split is empty. Choose an earlier test_start or provide more history.")
    if len(df_val) == 0:
        raise ValueError("Validation split is empty. Check test_start/test_end exist in index.")

    return df_train, df_val


def make_residual_target_naive1(df: pd.DataFrame) -> pd.DataFrame:
    if "y" not in df.columns:
        raise ValueError("Target column 'y' not found.")
    out = df.copy()
    out["y"] = out["y"].astype(float) - out["y"].astype(float).shift(1)
    out = out.dropna(subset=["y"]).copy()
    return out


def fit_with_early_stopping_compat(
    model: xgb.XGBRegressor,
    X_tr,
    y_tr,
    X_val,
    y_val,
    early_stopping_rounds: int,
    verbose_period: int = 200,
):
    """
    Compatibility early stopping for different XGBoost sklearn versions.
    Tries:
      1) fit(..., early_stopping_rounds=...) if available
      2) fit(..., callbacks=[EarlyStopping]) if available
      3) else: warn and fit without early stopping
    """
    sig = inspect.signature(model.fit)
    params = sig.parameters
    common_kwargs = {"eval_set": [(X_val, y_val)]}

    if "early_stopping_rounds" in params:
        if "verbose" in params:
            common_kwargs["verbose"] = verbose_period
        common_kwargs["early_stopping_rounds"] = early_stopping_rounds
        model.fit(X_tr, y_tr, **common_kwargs)
        return model

    if "callbacks" in params:
        cbs = []
        if hasattr(xgb, "callback") and hasattr(xgb.callback, "EarlyStopping"):
            try:
                cbs.append(xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True))
            except TypeError:
                cbs.append(xgb.callback.EarlyStopping(rounds=early_stopping_rounds))
        if hasattr(xgb, "callback") and hasattr(xgb.callback, "EvaluationMonitor"):
            cbs.append(xgb.callback.EvaluationMonitor(period=verbose_period))
        common_kwargs["callbacks"] = cbs
        if "verbose" in params:
            common_kwargs["verbose"] = False
        model.fit(X_tr, y_tr, **common_kwargs)
        return model

    print("[WARN] This XGBoost version does not support early stopping via sklearn fit(). Training without early stopping.")
    model.fit(X_tr, y_tr)
    return model


def train(
    mode: str,
    task: str = "price",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    device: str = "cuda",
    eval_metric: str = "mae",
    early_stopping_rounds: int = 200,
    residual_mode: str = "none",   # "none" | "naive1"
    baseline_col: str = "y_lag1",
) -> None:
    df = load_processed(mode, task=task)
    df = select_train_window(df, train_start=train_start, train_end=train_end)

    use_val = (test_start is not None) and (test_end is not None)
    if (test_start is None) ^ (test_end is None):
        raise ValueError("Provide BOTH --test_start and --test_end, or neither.")

    # residual transform (target only)
    df_for_xy = df
    if residual_mode == "naive1":
        df_for_xy = make_residual_target_naive1(df_for_xy)

    if use_val:
        df_tr, df_val = split_train_val_by_time(df_for_xy, test_start=test_start, test_end=test_end)
        X_tr, y_tr = make_xy(df_tr)
        X_val, y_val = make_xy(df_val)
        print(f"🚀 TRAINING XGBoost OPENLOOP ({mode.upper()} | task={task.upper()}) [VAL + EARLY STOPPING]")
        print(f"   -> train={len(df_tr)} | val={len(df_val)} | features={X_tr.shape[1]} | device={device}")
        print(f"   -> val window: [{test_start} .. {test_end}] | metric={eval_metric}")
    else:
        X_tr, y_tr = make_xy(df_for_xy)
        X_val, y_val = None, None
        print(f"🚀 TRAINING XGBoost OPENLOOP ({mode.upper()} | task={task.upper()})")
        print(f"   -> train={len(df_for_xy)} | features={X_tr.shape[1]} | device={device}")

    feature_names = list(X_tr.columns)

    # sanity baseline_col for residual wrapper
    if residual_mode == "naive1":
        if baseline_col not in feature_names:
            alt = "y_lag_1"
            if alt in feature_names:
                baseline_col = alt
            else:
                cand = [c for c in feature_names if "lag1" in c or "lag_1" in c or c.startswith("y_lag")]
                raise ValueError(
                    f"Residual mode naive1 needs baseline feature '{baseline_col}' (or '{alt}') in X. "
                    f"Did not find it. Candidates: {cand[:20]}"
                )

    base_model = xgb.XGBRegressor(
        n_estimators=6000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        tree_method="hist",
        device=device,                 # "cuda" or "cpu" (your setup supports this)
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric=eval_metric,
    )

    if use_val:
        base_model = fit_with_early_stopping_compat(
            base_model, X_tr, y_tr, X_val, y_val,
            early_stopping_rounds=early_stopping_rounds,
            verbose_period=200,
        )
        print(f"✅ Early stopping info (if supported): best_iteration={getattr(base_model,'best_iteration',None)} | best_score={getattr(base_model,'best_score',None)}")
    else:
        base_model.fit(X_tr, y_tr)

    MODELS_DIR.mkdir(exist_ok=True)

    default_path = MODELS_DIR / f"xgb_{mode}_{task}_openloop.pkl"
    backup_path = MODELS_DIR / "backup.pkl"
    tagged_path = MODELS_DIR / f"xgb_{mode}_{task}_openloop_resid_naive1.pkl"

    if residual_mode == "none":
        # backup + overwrite default
        if default_path.exists():
            try:
                shutil.copy2(default_path, backup_path)
                print(f"📦 Backup previous DEFAULT model -> {backup_path}")
            except Exception as e:
                print(f"[WARN] Could not create backup at {backup_path}: {e}")

        joblib.dump(base_model, default_path)
        print(f"✅ Saved DEFAULT XGBoost OPENLOOP to: {default_path}")

    elif residual_mode == "naive1":
        wrapped = ResidualAddBaselineWrapper(
            model=base_model,
            baseline_col=baseline_col,
            feature_names=feature_names,
        )
        joblib.dump(wrapped, tagged_path)
        print(f"✅ Saved RESIDUAL model (naive1) to: {tagged_path}")
        print("ℹ️ Default model NOT overwritten (compare xgb vs xgb_test in eval).")

    else:
        raise ValueError(f"Unknown residual_mode={residual_mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)

    p.add_argument("--test_start", type=str, default=None)
    p.add_argument("--test_end", type=str, default=None)

    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--eval_metric", type=str, default="mae")
    p.add_argument("--early_stopping_rounds", type=int, default=200)

    p.add_argument("--residual_mode", choices=["none", "naive1"], default="none")
    p.add_argument("--baseline_col", type=str, default="y_lag1")

    args = p.parse_args()

    train(
        mode=args.mode,
        task=args.task,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        device=args.device,
        eval_metric=args.eval_metric,
        early_stopping_rounds=args.early_stopping_rounds,
        residual_mode=args.residual_mode,
        baseline_col=args.baseline_col,
    )


if __name__ == "__main__":
    main()
