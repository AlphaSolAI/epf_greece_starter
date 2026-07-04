import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

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
        raise ValueError("This tuner currently supports mode='hourly' only.")

    fname = "hourly.parquet" if task == "price" else "hourly_load.parquet"
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing processed parquet: {path}")

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        # best-effort index detection
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


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _pick_device(device_pref: str, seed: int) -> str:
    """
    device_pref: cpu|cuda|auto
    Returns: "cpu" or "cuda"
    """
    device_pref = (device_pref or "auto").lower()
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        # user forced cuda -> we will try, but may still fail later if no GPU build
        return "cuda"

    # auto: try a tiny smoke-test
    try:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(256, 8))
        y = rng.normal(size=(256,))
        d = xgb.DMatrix(X, label=y)
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": 3,
            "eta": 0.1,
            "seed": seed,
        }
        _ = xgb.train(params, d, num_boost_round=5, verbose_eval=False)
        return "cuda"
    except Exception:
        return "cpu"


def _sample_params(rng: np.random.Generator, seed: int, device: str) -> Dict[str, object]:
    # log-uniform helper
    def logu(a: float, b: float) -> float:
        return float(np.exp(rng.uniform(np.log(a), np.log(b))))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": int(seed),
        "eta": logu(0.01, 0.25),
        "max_depth": int(rng.integers(3, 11)),
        "min_child_weight": logu(0.5, 20.0),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "gamma": logu(1e-8, 5.0),
        "reg_alpha": logu(1e-8, 10.0),
        "reg_lambda": logu(0.1, 10.0),
    }
    if device == "cuda":
        # XGBoost 2.x GPU selection
        params["device"] = "cuda"
    return params


def _rolling_origin_folds(
    n: int,
    n_splits: int,
    val_size: int,
    min_train_size: int,
) -> List[Tuple[slice, slice]]:
    """
    Build folds near the END of the training set (rolling-origin).
    Each fold validates on a contiguous block of size val_size.
    """
    if n <= (n_splits * val_size + min_train_size):
        raise ValueError(
            f"Not enough rows for CV. n={n}, n_splits={n_splits}, val_size={val_size}, min_train_size={min_train_size}"
        )

    folds = []
    # last validation block ends at n
    for i in range(n_splits):
        val_end = n - (n_splits - 1 - i) * val_size
        val_start = val_end - val_size
        tr_end = val_start
        if tr_end < min_train_size:
            continue
        folds.append((slice(0, tr_end), slice(val_start, val_end)))
    if len(folds) == 0:
        raise ValueError("No valid folds produced (min_train_size too large).")
    return folds


def _cv_score(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: List[Tuple[slice, slice]],
    params: Dict[str, object],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[float, float]:
    maes = []
    best_iters = []

    for tr_sl, va_sl in folds:
        Xtr, ytr = X.iloc[tr_sl], y[tr_sl]
        Xva, yva = X.iloc[va_sl], y[va_sl]

        dtr = xgb.DMatrix(Xtr, label=ytr)
        dva = xgb.DMatrix(Xva, label=yva)

        bst = xgb.train(
            dict(params),
            dtr,
            num_boost_round=num_boost_round,
            evals=[(dva, "val")],
            early_stopping_rounds=int(early_stopping_rounds),
            verbose_eval=False,
        )

        # best_iteration is 0-based
        bi = getattr(bst, "best_iteration", None)
        if bi is None:
            bi = num_boost_round - 1
        best_iters.append(float(bi + 1))

        # predict using best iteration range when available
        try:
            yhat = bst.predict(dva, iteration_range=(0, int(bi + 1)))
        except Exception:
            yhat = bst.predict(dva)

        maes.append(_mae(yva, yhat))

    mean_mae = float(np.mean(maes))
    std_mae = float(np.std(maes))
    mean_best_iter = float(np.mean(best_iters))
    # we return mean mae and mean best iteration; caller prints ±std separately
    return mean_mae + 0.0 * std_mae, mean_best_iter  # keep signature stable


def fit_final_model(
    X: pd.DataFrame,
    y: np.ndarray,
    params: Dict[str, object],
    num_boost_round: int,
) -> xgb.Booster:
    dtr = xgb.DMatrix(X, label=y)
    bst = xgb.train(dict(params), dtr, num_boost_round=int(num_boost_round), verbose_eval=False)
    return bst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")

    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)
    parser.add_argument("--test_size", type=int, default=None)

    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--n_splits", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=168)
    parser.add_argument("--min_train_size", type=int, default=8760)
    parser.add_argument("--early_stopping_rounds", type=int, default=200)
    parser.add_argument("--num_boost_round", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")

    args = parser.parse_args()

    task = args.task.lower()
    mode = args.mode.lower()

    print(f"🔎 TUNING XGBoost ({mode.upper()} | declared task={task.upper()}) [rolling-origin CV inside TRAIN]")

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
    Xtr = df_train[feature_cols].copy()
    ytr = df_train["y"].astype(float).to_numpy()

    folds = _rolling_origin_folds(
        n=len(df_train),
        n_splits=int(args.n_splits),
        val_size=int(args.val_size),
        min_train_size=int(args.min_train_size),
    )

    device = _pick_device(args.device, args.seed)
    step = int(args.val_size)

    print(f"[INFO] train_rows={len(df_train)} folds={len(folds)} val_size={args.val_size} step={step} min_train_size={args.min_train_size}")
    print(f"[INFO] n_trials={args.n_trials} num_boost_round={args.num_boost_round} early_stop={args.early_stopping_rounds} seed={args.seed} device={device}")
    print("[NOTE] This assumes you already built the processed dataset for the correct task (price/load) in the parquet.")

    rng = np.random.default_rng(int(args.seed))

    best_mae = float("inf")
    best_params: Optional[Dict[str, object]] = None
    best_mean_iter = 500.0

    # Trial 0: baseline params
    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": int(args.seed),
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    if device == "cuda":
        base_params["device"] = "cuda"

    m0, it0 = _cv_score(
        Xtr, ytr, folds,
        base_params,
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    print(f"[TRIAL 0] meanMAE={m0:.4f} | meanBestIter={it0:.1f} | (baseline params)")
    best_mae, best_params, best_mean_iter = m0, dict(base_params), it0

    for t in range(1, int(args.n_trials) + 1):
        params = _sample_params(rng, int(args.seed + t), device)
        m, it = _cv_score(
            Xtr, ytr, folds,
            params,
            num_boost_round=int(args.num_boost_round),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        tag = ""
        if m < best_mae:
            best_mae = m
            best_params = dict(params)
            best_mean_iter = it
            tag = " ✅ NEW BEST"
        print(f"[TRIAL {t:02d}] meanMAE={m:.4f} | meanBestIter={it:.1f}{tag}")

    assert best_params is not None

    final_rounds = int(max(50, round(best_mean_iter)))
    print(f"\n✅ BEST: meanMAE={best_mae:.4f} | using num_boost_round={final_rounds}")
    print(f"[BEST PARAMS] {best_params}")

    # Fit final on FULL TRAIN
    try:
        bst = fit_final_model(Xtr, ytr, best_params, num_boost_round=final_rounds)
    except Exception as e:
        # fallback to cpu if cuda build missing
        if "device" in best_params:
            print(f"[WARN] Final fit failed on CUDA ({e}). Falling back to CPU.")
            best_params = dict(best_params)
            best_params.pop("device", None)
            bst = fit_final_model(Xtr, ytr, best_params, num_boost_round=final_rounds)
        else:
            raise

    MODELS_DIR.mkdir(exist_ok=True)

    out_pkl = MODELS_DIR / f"xgb_{mode}_{task}_tuned.pkl"
    artifact = {
        "kind": "xgb_booster",
        "mode": mode,
        "task": task,
        "feature_cols": feature_cols,
        "params": best_params,
        "num_boost_round": final_rounds,
        "booster": bst,
    }
    joblib.dump(artifact, out_pkl)
    print(f"✅ Saved tuned XGBoost to: {out_pkl}")

    out_json = MODELS_DIR / f"xgb_{mode}_{task}_tuned_params.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "task": task,
                "best_mean_cv_mae": best_mae,
                "num_boost_round": final_rounds,
                "params": best_params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ Saved tuned params to: {out_json}")

    # Keep test untouched (we only print split info)
    print(f"[INFO] untouched test window: {df_test.index.min()} .. {df_test.index.max()} (n={len(df_test)})")


if __name__ == "__main__":
    main()
