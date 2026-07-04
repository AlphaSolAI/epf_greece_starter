import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from .split_utils import split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def _load_processed(mode: str, task: str) -> pd.DataFrame:
    if mode != "hourly":
        raise ValueError("This residual trainer currently supports mode='hourly' only.")
    fname = "hourly.parquet" if task == "price" else "hourly_load.parquet"
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing processed parquet: {path}")

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ["timestamp", "ds", "date", "datetime", "time"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], errors="coerce")
                df = df.dropna(subset=[cand]).set_index(cand)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Processed dataframe must have DatetimeIndex.")

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if "y" not in df.columns:
        raise ValueError("Column 'y' not found in processed parquet.")
    return df


def _fit_seasonal_profile(df_train: pd.DataFrame) -> Dict[str, object]:
    tmp = df_train.copy()

    # ensure keys exist
    if "hour" not in tmp.columns:
        tmp["hour"] = tmp.index.hour
    if "dow" not in tmp.columns:
        tmp["dow"] = tmp.index.dayofweek
    if "is_holiday" not in tmp.columns:
        tmp["is_holiday"] = 0

    prof_hdh = tmp.groupby(["hour", "dow", "is_holiday"])["y"].mean()
    prof_hd = tmp.groupby(["hour", "dow"])["y"].mean()
    prof_h = tmp.groupby(["hour"])["y"].mean()
    mu = float(tmp["y"].mean())

    return {"prof_hdh": prof_hdh, "prof_hd": prof_hd, "prof_h": prof_h, "mu": mu}


def _predict_seasonal_profile(keys: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    prof_hdh = keys["prof_hdh"]
    prof_hd = keys["prof_hd"]
    prof_h = keys["prof_h"]
    mu = float(keys["mu"])

    tmp = X.copy()
    if "hour" not in tmp.columns:
        tmp["hour"] = tmp.index.hour
    if "dow" not in tmp.columns:
        tmp["dow"] = tmp.index.dayofweek
    if "is_holiday" not in tmp.columns:
        tmp["is_holiday"] = 0

    out = pd.Series(np.nan, index=tmp.index, dtype=float)

    # 3-way
    idx3 = pd.MultiIndex.from_arrays([tmp["hour"], tmp["dow"], tmp["is_holiday"]])
    s3 = pd.Series(prof_hdh.reindex(idx3).to_numpy(), index=tmp.index)
    out = out.fillna(s3)

    # 2-way
    idx2 = pd.MultiIndex.from_arrays([tmp["hour"], tmp["dow"]])
    s2 = pd.Series(prof_hd.reindex(idx2).to_numpy(), index=tmp.index)
    out = out.fillna(s2)

    # 1-way
    s1 = tmp["hour"].map(prof_h).astype(float)
    out = out.fillna(s1)

    out = out.fillna(mu)
    return out.to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")

    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)
    parser.add_argument("--test_size", type=int, default=None)

    parser.add_argument(
        "--params_json",
        type=str,
        default=None,
        help="Optional path to tuned LightGBM params JSON (uses its params + num_boost_round if present).",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    mode = args.mode.lower()
    task = args.task.lower()

    df = _load_processed(mode, task)
    df_train, df_test = split_time_series(
        df,
        mode=mode,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    feature_cols = [c for c in df_train.columns if c != "y"]
    X_train = df_train[feature_cols].copy()
    y_train = df_train["y"].astype(float).to_numpy()

    # Fit baseline on TRAIN only
    prof = _fit_seasonal_profile(df_train)
    y_base_train = _predict_seasonal_profile(prof, df_train[feature_cols])
    resid = y_train - y_base_train

    # Params
    params = dict(
        objective="regression",
        metric="mae",
        learning_rate=0.03,
        num_leaves=96,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=40,
        verbose=-1,
        seed=int(args.seed),
    )
    num_boost_round = 1200

    if args.params_json:
        p = Path(args.params_json)
        if p.exists():
            j = json.loads(p.read_text(encoding="utf-8"))
            # accept either {"params": {...}} or direct dict
            tuned_params = j.get("params", j)
            if isinstance(tuned_params, dict):
                # keep only LGBM params
                for k, v in tuned_params.items():
                    if k in ["objective", "metric", "learning_rate", "num_leaves", "feature_fraction",
                             "bagging_fraction", "bagging_freq", "min_data_in_leaf", "lambda_l1", "lambda_l2",
                             "min_gain_to_split", "max_depth", "max_bin", "seed", "verbose",
                             "reg_alpha", "reg_lambda"]:
                        params[k] = v
            # rounds
            if "num_boost_round" in j:
                num_boost_round = int(j["num_boost_round"])
            elif "best_iter" in j:
                num_boost_round = int(j["best_iter"])
            elif "best_iteration" in j:
                num_boost_round = int(j["best_iteration"])
        else:
            print(f"[WARN] params_json not found: {p} (using defaults)")

    print(f"🧩 TRAINING LightGBM RESIDUAL ({mode.upper()} | task={task.upper()})")
    print(f"[INFO] train={len(df_train)} test={len(df_test)} features={len(feature_cols)}")
    print(f"[INFO] baseline=SeasonalProfile(hour×dow×holiday) -> training on residuals")
    print(f"[INFO] num_boost_round={num_boost_round}")

    dtrain = lgb.Dataset(X_train, label=resid)
    model = lgb.train(params, dtrain, num_boost_round=int(num_boost_round))

    artifact = {
        "kind": "seasonal_residual_lgbm",
        "mode": mode,
        "task": task,
        "feature_cols": feature_cols,
        "baseline": {
            "mu": float(prof["mu"]),
            "prof_hdh": prof["prof_hdh"],
            "prof_hd": prof["prof_hd"],
            "prof_h": prof["prof_h"],
        },
        "model_str": model.model_to_string(),
        "num_boost_round": int(num_boost_round),
        "params": params,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    out_pkl = MODELS_DIR / f"lgbm_residual_{mode}_{task}.pkl"
    joblib.dump(artifact, out_pkl)
    print(f"✅ Saved residual model: {out_pkl}")
    print(f"[INFO] untouched test window: {df_test.index.min()} .. {df_test.index.max()} (n={len(df_test)})")


if __name__ == "__main__":
    main()
