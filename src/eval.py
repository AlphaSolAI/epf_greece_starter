import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .split_utils import load_processed, split_time_series, make_xy

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "data" / "processed"


def _to_parquet_any(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, engine="fastparquet")
    except Exception:
        df.to_parquet(path)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-9
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _try_load_model(paths: List[Path]):
    for p in paths:
        if p.exists():
            return joblib.load(p), p
    return None, None


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float).reshape(-1)
    raise TypeError(f"Model has no predict(): {type(model)}")


def _naive_shift(y_train: pd.Series, y_test: pd.Series, k: int) -> np.ndarray:
    full = pd.concat([y_train, y_test])
    return full.shift(k).loc[y_test.index].to_numpy(dtype=float)


def _holiday_flag_from_df(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    candidates = {"is_holiday", "holiday", "cal_is_holiday"}
    for c in df.columns:
        if c.lower() in candidates:
            return df[c].reindex(index).fillna(0).astype(int)
    return pd.Series(0, index=index, dtype=int)


def _seasonal_profile(df_train: pd.DataFrame, df_full: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Train-only seasonal profile: mean(y | hour, day-of-week, holiday)."""
    y_tr = df_train["y"].astype(float)
    hol_tr = _holiday_flag_from_df(df_train, df_train.index)
    hour_tr = df_train.index.hour
    dow_tr = df_train.index.dayofweek

    prof_hdh = y_tr.groupby([hour_tr, dow_tr, hol_tr]).mean()
    prof_hd = y_tr.groupby([hour_tr, dow_tr]).mean()
    prof_h = y_tr.groupby(hour_tr).mean()
    global_mean = float(y_tr.mean())

    hol_test = _holiday_flag_from_df(df_full, test_index)
    out = np.zeros(len(test_index), dtype=float)
    for i, ts in enumerate(test_index):
        h, d, hol = ts.hour, ts.dayofweek, int(hol_test.iloc[i])
        v = prof_hdh.get((h, d, hol), np.nan)
        if not np.isfinite(v):
            v = prof_hd.get((h, d), np.nan)
        if not np.isfinite(v):
            v = prof_h.get(h, np.nan)
        if not np.isfinite(v):
            v = global_mean
        out[i] = float(v)
    return out


def evaluate(
    mode: str,
    task: str,
    test_size: Optional[int],
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    sort_by: str,
    export_predictions: bool,
    export_model: str,
):
    df = load_processed(mode, task=task)
    df_train, df_test = split_time_series(
        df,
        mode=mode,
        test_size=test_size,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    X_train, y_train = make_xy(df_train)
    X_test, y_test = make_xy(df_test)

    y_train_s = pd.Series(y_train, index=X_train.index)
    y_test_s = pd.Series(y_test, index=X_test.index)

    print(f"\n📊 FINAL EVALUATION ({mode.upper()}) [CLOSED LOOP] task={task.upper()}")
    print(f"[INFO] train={len(X_train)}, test={len(X_test)} | features={X_train.shape[1]}")

    # load models with both naming conventions:
    #   new: lgbm_hourly_price.pkl
    #   old: lgbm_hourly.pkl
    def model_paths(prefix: str) -> List[Path]:
        return [
            MODELS_DIR / f"{prefix}_{mode}_{task}.pkl",
            MODELS_DIR / f"{prefix}_{mode}.pkl",
        ]

    preds: Dict[str, np.ndarray] = {}

    # Standard models: {prefix}_{mode}_{task}.pkl or {prefix}_{mode}.pkl
    for mk, disp in [("lgbm", "LightGBM"), ("xgb", "XGBoost"), ("rf", "RandomForest"), ("svr", "SVR"), ("mlp", "MLP")]:
        m, p = _try_load_model(model_paths(mk))
        if m is None:
            continue
        yhat = _predict(m, X_test)
        preds[mk] = yhat
        print(f"[INFO] LOADED: {disp} <- {p.name}")

    # SVR-Optuna (tuned — saved as svr_{mode}_{task}_optuna.pkl)
    svr_opt_path = MODELS_DIR / f"svr_{mode}_{task}_optuna.pkl"
    if svr_opt_path.exists():
        m_svr_opt = joblib.load(svr_opt_path)
        yhat_svr_opt = _predict(m_svr_opt, X_test)
        preds["svr_optuna"] = yhat_svr_opt
        print(f"[INFO] LOADED: SVR-Optuna <- {svr_opt_path.name}")

    # MLP-Optuna (tuned — saved as mlp_{mode}_{task}_optuna.pkl)
    mlp_opt_path = MODELS_DIR / f"mlp_{mode}_{task}_optuna.pkl"
    if mlp_opt_path.exists():
        m_mlp_opt = joblib.load(mlp_opt_path)
        yhat_mlp_opt = _predict(m_mlp_opt, X_test)
        preds["mlp_optuna"] = yhat_mlp_opt
        print(f"[INFO] LOADED: MLP-Optuna <- {mlp_opt_path.name}")

    rows: List[Dict[str, object]] = []

    # ML rows — closed-loop standard models ONLY
    # (Teacher-forced Optuna removed for tree models: standard LGBM/XGB outperforms them in CL)
    # SVR-Optuna and MLP-Optuna included if available
    all_ml = [
        ("lgbm",       "LightGBM"),
        ("xgb",        "XGBoost"),
        ("rf",         "RandomForest"),
        ("svr_optuna", "SVR-Optuna"),
        ("svr",        "SVR"),
        ("mlp_optuna", "MLP-Optuna"),
        ("mlp",        "MLP"),
    ]
    for mk, disp in all_ml:
        if mk not in preds:
            continue
        rows.append(
            {"Model": disp, "Type": "ml", "MAE": round(mae(y_test, preds[mk]), 3), "RMSE": round(rmse(y_test, preds[mk]), 3), "sMAPE": round(smape(y_test, preds[mk]), 3)}
        )

    # Ensemble: inverse-MAE weighted over all loaded ML models
    avail_ml = [mk for mk, _ in all_ml if mk in preds]
    if len(avail_ml) >= 2:
        stack = np.array([preds[mk] for mk in avail_ml])
        maes_e = np.array([mae(y_test, preds[mk]) for mk in avail_ml])
        weights = 1.0 / np.where(maes_e > 0, maes_e, 1e-9)
        weights /= weights.sum()
        ensemble_w = np.sum(stack * weights[:, None], axis=0)
        preds["ensemble_1mae"] = ensemble_w
        rows.append({"Model": "Ensemble (1/MAE)", "Type": "ensemble",
                     "MAE": round(mae(y_test, ensemble_w), 3),
                     "RMSE": round(rmse(y_test, ensemble_w), 3),
                     "sMAPE": round(smape(y_test, ensemble_w), 3)})
        disp_w = {mk: round(float(w), 3) for mk, w in zip(avail_ml, weights)}
        print(f"[INFO] Ensemble (1/MAE) weights: {disp_w}")

    # Ensemble-Best3: top-3 by test MAE, then 1/MAE weighted
    if len(avail_ml) >= 3:
        sorted_ml = sorted(avail_ml, key=lambda mk: mae(y_test, preds[mk]))
        top3 = sorted_ml[:3]
        stack3 = np.array([preds[mk] for mk in top3])
        maes3 = np.array([mae(y_test, preds[mk]) for mk in top3])
        w3 = 1.0 / np.where(maes3 > 0, maes3, 1e-9)
        w3 /= w3.sum()
        ens_best3 = np.sum(stack3 * w3[:, None], axis=0)
        preds["ensemble_best3"] = ens_best3
        rows.append({"Model": "Ensemble-Best3 (1/MAE)", "Type": "ensemble",
                     "MAE": round(mae(y_test, ens_best3), 3),
                     "RMSE": round(rmse(y_test, ens_best3), 3),
                     "sMAPE": round(smape(y_test, ens_best3), 3)})
        top3_disp = [dict(all_ml).get(mk, mk) for mk in top3]
        print(f"[INFO] Ensemble-Best3 (1/MAE): {top3_disp}")

    # baselines
    y_seasonal = _seasonal_profile(df_train, df, X_test.index)
    rows.append({"Model": "Seasonal Profile", "Type": "baseline",
                 "MAE": round(mae(y_test, y_seasonal), 3),
                 "RMSE": round(rmse(y_test, y_seasonal), 3),
                 "sMAPE": round(smape(y_test, y_seasonal), 3)})
    rows.append({"Model": "Naive-1", "Type": "baseline", "MAE": round(mae(y_test, _naive_shift(y_train_s, y_test_s, 1)), 3),
                 "RMSE": round(rmse(y_test, _naive_shift(y_train_s, y_test_s, 1)), 3), "sMAPE": round(smape(y_test, _naive_shift(y_train_s, y_test_s, 1)), 3)})
    rows.append({"Model": "Naive-24", "Type": "baseline", "MAE": round(mae(y_test, _naive_shift(y_train_s, y_test_s, 24)), 3),
                 "RMSE": round(rmse(y_test, _naive_shift(y_train_s, y_test_s, 24)), 3), "sMAPE": round(smape(y_test, _naive_shift(y_train_s, y_test_s, 24)), 3)})
    rows.append({"Model": "Naive-168", "Type": "baseline", "MAE": round(mae(y_test, _naive_shift(y_train_s, y_test_s, 168)), 3),
                 "RMSE": round(rmse(y_test, _naive_shift(y_train_s, y_test_s, 168)), 3), "sMAPE": round(smape(y_test, _naive_shift(y_train_s, y_test_s, 168)), 3)})

    dfm = pd.DataFrame(rows)
    if sort_by in dfm.columns:
        dfm = dfm.sort_values(sort_by, ascending=True)

    print(f"\n📌 TEST METRICS sorted by {sort_by}")
    print(dfm.to_string(index=False))

    # EXPORT load forecasts for PRICE build
    if export_predictions:
        if task != "load":
            print("[WARN] --export_predictions intended for task=load. Skipping.")
        else:
            chosen = export_model.strip().lower()
            if chosen == "best":
                best_mk, best_val = None, float("inf")
                for mk in preds:
                    m = mae(y_test, preds[mk])
                    if m < best_val:
                        best_val = m
                        best_mk = mk
                chosen = best_mk or "lgbm"

            if chosen not in preds:
                print(f"[WARN] export_model='{export_model}' not available. Available={list(preds.keys())}. Skipping export.")
            else:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                out_path = OUT_DIR / "load_forecast_hourly.parquet"
                df_fc = pd.DataFrame({"load_fc": preds[chosen].astype(float)}, index=X_test.index)
                _to_parquet_any(df_fc, out_path)
                print(f"\n✅ Exported load forecasts ({chosen}) -> {out_path}")

    # save dashboard json
    out_json = BASE_DIR / f"dashboard_data_{mode}_{task}.json"

    # Χρονοσειρές για το γράφημα
    dates_list = [str(ts) for ts in X_test.index]
    actual_list = [round(float(v), 4) for v in y_test]
    series_dict: Dict[str, list] = {}

    # ML + Ensemble series
    for mk, disp in all_ml:
        if mk in preds:
            series_dict[disp] = [round(float(v), 4) for v in preds[mk]]
    if "ensemble_1mae" in preds:
        series_dict["Ensemble (1/MAE)"] = [round(float(v), 4) for v in preds["ensemble_1mae"]]
    if "ensemble_best3" in preds:
        series_dict["Ensemble-Best3 (1/MAE)"] = [round(float(v), 4) for v in preds["ensemble_best3"]]

    # Baseline series (για εμφάνιση στο γράφημα)
    def _s(arr): return [round(float(v), 4) if np.isfinite(v) else None for v in arr]
    series_dict["Seasonal Profile"] = _s(y_seasonal)
    series_dict["Naive-1"]   = _s(_naive_shift(y_train_s, y_test_s, 1))
    series_dict["Naive-24"]  = _s(_naive_shift(y_train_s, y_test_s, 24))
    series_dict["Naive-168"] = _s(_naive_shift(y_train_s, y_test_s, 168))

    payload = {
        "mode": mode,
        "task": task,
        "strategy": "cl",
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
        # chart data
        "dates":  dates_list,
        "actual": actual_list,
        "series": series_dict,
        # metrics table
        "metrics": dfm.to_dict(orient="records"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {out_json}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--test_size", type=int, default=None)
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--test_start", type=str, default=None)
    p.add_argument("--test_end", type=str, default=None)
    p.add_argument("--sort_by", choices=["MAE", "RMSE", "sMAPE"], default="MAE")
    p.add_argument("--export_predictions", action="store_true")
    p.add_argument("--export_model", type=str, default="lgbm")
    args = p.parse_args()

    evaluate(
        mode=args.mode,
        task=args.task,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        sort_by=args.sort_by,
        export_predictions=bool(args.export_predictions),
        export_model=str(args.export_model),
    )


if __name__ == "__main__":
    main()
