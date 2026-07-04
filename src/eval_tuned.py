import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
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


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def print_table(rows: List[Dict[str, object]], sort_by: str = "MAE") -> None:
    if not rows:
        print("[WARN] No rows to print.")
        return

    cols = ["Model", "Type", "MAE", "RMSE", "sMAPE"]
    if sort_by in cols:
        rows = sorted(rows, key=lambda r: float(r.get(sort_by, 1e18)))

    formatted = []
    for r in rows:
        fr = dict(r)
        for k in ["MAE", "RMSE", "sMAPE"]:
            fr[k] = f"{float(fr[k]):.3f}"
        formatted.append(fr)

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in formatted)) for c in cols}
    widths["Model"] = min(widths["Model"], 55)

    def clip(s: str, w: int) -> str:
        return s if len(s) <= w else s[: w - 1] + "…"

    header = "  ".join(clip(c, widths[c]).ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in formatted:
        line = "  ".join(clip(str(r.get(c, "")), widths[c]).ljust(widths[c]) for c in cols)
        print(line)


def _load_processed(mode: str, task: str) -> pd.DataFrame:
    if mode != "hourly":
        raise ValueError("eval_tuned supports mode='hourly' only.")
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


def baseline_naive_shift(df_full: pd.DataFrame, test_index: pd.DatetimeIndex, k: int) -> np.ndarray:
    return df_full["y"].shift(k).reindex(test_index).to_numpy(dtype=float)


def baseline_seasonal_profile(df_train: pd.DataFrame, df_full: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
    hol_col = "is_holiday" if "is_holiday" in df_train.columns else None

    tmp = df_train.copy()
    tmp["__hour"] = tmp.index.hour
    tmp["__dow"] = tmp.index.dayofweek
    tmp["__hol"] = tmp[hol_col].astype(int) if hol_col else 0

    prof3 = tmp.groupby(["__hour", "__dow", "__hol"])["y"].mean()
    prof2 = tmp.groupby(["__hour", "__dow"])["y"].mean()
    prof1 = tmp.groupby(["__hour"])["y"].mean()
    mu = float(tmp["y"].mean())

    out = []
    for ts in test_index:
        h = ts.hour
        d = ts.dayofweek
        hol = int(df_full.loc[ts, hol_col]) if (hol_col and ts in df_full.index) else 0
        k3 = (h, d, hol)
        if k3 in prof3.index:
            out.append(float(prof3.loc[k3])); continue
        k2 = (h, d)
        if k2 in prof2.index:
            out.append(float(prof2.loc[k2])); continue
        if h in prof1.index:
            out.append(float(prof1.loc[h])); continue
        out.append(mu)

    return np.asarray(out, dtype=float)


def _predict_seasonal_from_artifact(baseline: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    mu = float(baseline["mu"])
    prof_hdh = baseline["prof_hdh"]
    prof_hd = baseline["prof_hd"]
    prof_h = baseline["prof_h"]

    tmp = X.copy()
    if "hour" not in tmp.columns:
        tmp["hour"] = tmp.index.hour
    if "dow" not in tmp.columns:
        tmp["dow"] = tmp.index.dayofweek
    if "is_holiday" not in tmp.columns:
        tmp["is_holiday"] = 0

    out = pd.Series(np.nan, index=tmp.index, dtype=float)

    idx3 = pd.MultiIndex.from_arrays([tmp["hour"], tmp["dow"], tmp["is_holiday"]])
    s3 = pd.Series(prof_hdh.reindex(idx3).to_numpy(), index=tmp.index)
    out = out.fillna(s3)

    idx2 = pd.MultiIndex.from_arrays([tmp["hour"], tmp["dow"]])
    s2 = pd.Series(prof_hd.reindex(idx2).to_numpy(), index=tmp.index)
    out = out.fillna(s2)

    s1 = tmp["hour"].map(prof_h).astype(float)
    out = out.fillna(s1)

    out = out.fillna(mu)
    return out.to_numpy(dtype=float)


def _load_if_exists(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def _predict_any(model_obj, X: pd.DataFrame) -> np.ndarray:
    # tuned XGB artifact
    if isinstance(model_obj, dict) and model_obj.get("kind") == "xgb_booster":
        cols = model_obj["feature_cols"]
        bst = model_obj["booster"]
        d = xgb.DMatrix(X.loc[:, cols])
        return bst.predict(d).astype(float)

    # residual LGBM artifact
    if isinstance(model_obj, dict) and model_obj.get("kind") == "seasonal_residual_lgbm":
        cols = model_obj["feature_cols"]
        base = _predict_seasonal_from_artifact(model_obj["baseline"], X.loc[:, cols])
        booster = lgb.Booster(model_str=model_obj["model_str"])
        resid = booster.predict(X.loc[:, cols])
        return (base + resid).astype(float)

    # raw lightgbm Booster
    if isinstance(model_obj, lgb.Booster):
        return model_obj.predict(X).astype(float)

    # sklearn-like
    if hasattr(model_obj, "predict"):
        yhat = model_obj.predict(X.values)
        return np.asarray(yhat, dtype=float).reshape(-1)

    raise TypeError(f"Unknown model type: {type(model_obj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")

    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)

    parser.add_argument("--sort_by", choices=["MAE", "RMSE", "sMAPE"], default="MAE")
    parser.add_argument("--include_default", action="store_true")

    args = parser.parse_args()
    mode = args.mode.lower()
    task = args.task.lower()

    print(f"📊 EVAL_TUNED ({mode.upper()}) [CLOSED LOOP] declared task={task.upper()}")

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

    feature_cols = [c for c in df.columns if c != "y"]
    X_test = df_test[feature_cols].copy()
    y_true = df_test["y"].to_numpy(dtype=float)

    print(f"[INFO] train={len(df_train)} test={len(df_test)} features={len(feature_cols)}")
    print(f"[INFO] test_start={df_test.index.min()} test_end={df_test.index.max()}")

    rows: List[Dict[str, object]] = []

    # ---- tuned models ----
    candidates = [
        ("LightGBM (tuned)", MODELS_DIR / f"lgbm_{mode}_{task}_tuned.pkl"),
        ("XGBoost (tuned)", MODELS_DIR / f"xgb_{mode}_{task}_tuned.pkl"),
        ("LightGBM (residual)", MODELS_DIR / f"lgbm_residual_{mode}_{task}.pkl"),
    ]

    for name, path in candidates:
        if not path.exists():
            print(f"[INFO] missing -> {path.name} (skip)")
            continue
        try:
            obj = joblib.load(path)
            print(f"[INFO] LOADED: {name} <- {path.name}")
            yhat = _predict_any(obj, X_test)
            rows.append(
                {"Model": name, "Type": "ml", "MAE": mae(y_true, yhat), "RMSE": rmse(y_true, yhat), "sMAPE": smape(y_true, yhat)}
            )
        except Exception as e:
            print(f"[WARN] FAILED: {name} from {path.name}: {type(e).__name__}: {e}")

    # ---- optionally include defaults ----
    if args.include_default:
        defaults = [
            ("LightGBM (default)", MODELS_DIR / f"lgbm_{mode}_{task}.pkl"),
            ("XGBoost (default)", MODELS_DIR / f"xgb_{mode}_{task}.pkl"),
        ]
        for name, path in defaults:
            if not path.exists():
                print(f"[INFO] missing -> {path.name} (skip)")
                continue
            try:
                obj = joblib.load(path)
                print(f"[INFO] LOADED: {name} <- {path.name}")
                yhat = _predict_any(obj, X_test)
                rows.append(
                    {"Model": name, "Type": "ml", "MAE": mae(y_true, yhat), "RMSE": rmse(y_true, yhat), "sMAPE": smape(y_true, yhat)}
                )
            except Exception as e:
                print(f"[WARN] FAILED: {name} from {path.name}: {type(e).__name__}: {e}")

    # ---- baselines ----
    df_full = pd.concat([df_train, df_test], axis=0)
    test_index = df_test.index

    yhat_prof = baseline_seasonal_profile(df_train, df_full, test_index)
    rows.append(
        {"Model": "Seasonal profile (hour×dow×holiday)", "Type": "baseline",
         "MAE": mae(y_true, yhat_prof), "RMSE": rmse(y_true, yhat_prof), "sMAPE": smape(y_true, yhat_prof)}
    )

    for k in [1, 24, 168]:
        yhat_k = baseline_naive_shift(df_full, test_index, k)
        rows.append(
            {"Model": f"Naive-{k}", "Type": "baseline",
             "MAE": mae(y_true, yhat_k), "RMSE": rmse(y_true, yhat_k), "sMAPE": smape(y_true, yhat_k)}
        )

    print(f"\n📌 METRICS sorted by {args.sort_by}")
    print_table(rows, sort_by=args.sort_by)

    out = {
        "mode": mode,
        "task": task,
        "loop": "closed",
        "train_start": str(df_train.index.min()),
        "train_end": str(df_train.index.max()),
        "test_start": str(df_test.index.min()),
        "test_end": str(df_test.index.max()),
        "test_size": int(len(df_test)),
        "metrics": rows,
    }

    out_path = BASE_DIR / f"dashboard_data_{mode}_{task}_tuned.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
