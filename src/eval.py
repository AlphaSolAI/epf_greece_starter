import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore", message="No frequency information was provided*")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def _align_and_filter(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = _align_and_filter(y_true, y_pred)
    if len(y_true) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "sMAPE": float("nan")}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    denom = (np.abs(y_true) + np.abs(y_pred))
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-9)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for cand in ["timestamp", "ds", "date", "datetime", "time"]:
        if cand in df.columns:
            out = df.copy()
            out[cand] = pd.to_datetime(out[cand], errors="coerce")
            out = out.dropna(subset=[cand]).set_index(cand)
            return out
    return df


def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[-target_len:]
    pad = np.full(target_len - len(arr), np.nan, dtype=float)
    return np.concatenate([pad, arr])


def _series_to_json_list(arr: np.ndarray) -> List[Optional[float]]:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return [float(v) if np.isfinite(v) else None for v in arr]


def _safe_fill_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().values.any():
        X = X.interpolate(limit_direction="both").ffill().bfill()
    return X


def _safe_fill_series(s: pd.Series) -> pd.Series:
    s = s.astype(float).copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.isna().any():
        s = s.interpolate(limit_direction="both").ffill().bfill()
    return s


def baseline_naive(y_train: pd.Series, y_test: pd.Series) -> np.ndarray:
    preds = np.empty(len(y_test), dtype=float)
    if len(y_test) == 0:
        return preds
    preds[0] = float(y_train.iloc[-1])
    if len(y_test) > 1:
        preds[1:] = y_test.iloc[:-1].to_numpy()
    return preds


def baseline_seasonal_naive(y_train: pd.Series, y_test: pd.Series, season_len: int) -> np.ndarray:
    full = pd.concat([y_train, y_test], axis=0)
    preds = full.shift(season_len).iloc[len(y_train):].to_numpy(dtype=float)
    if len(preds) > 0 and not np.isfinite(preds[0]):
        preds[: min(season_len, len(preds))] = float(y_train.iloc[-1])
    return preds


def _load_model(model_id: str, mode: str):
    path = MODELS_DIR / f"{model_id}_{mode}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def _predict_model(model, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X)
    return np.asarray(pred, dtype=float).reshape(-1)


def _sarima_get_params(mode: str, errors: List[str]):
    params_path = MODELS_DIR / f"sarima_params_{mode}.json"
    if not params_path.exists():
        errors.append(f"[SARIMA] Params file not found: {params_path}")
        return None, None
    params = json.loads(params_path.read_text(encoding="utf-8"))
    order = tuple(params.get("order", (1, 0, 0)))
    seasonal_order = tuple(params.get("seasonal_order", (0, 1, 1, 7)))
    return order, seasonal_order


def sarima_walk_forward_blocks(
    y_train: pd.Series,
    y_test: pd.Series,
    mode: str,
    errors: List[str],
    block_size: int,
    fit_window: Optional[int],
    maxiter: int = 60,
) -> np.ndarray:
    """
    Walk-forward / rolling-origin SARIMA:
    - split test into blocks (e.g., 168h)
    - for each block: fit from scratch on (train + observed test so far), optionally capped to fit_window
    - forecast next block_size steps
    This avoids 8760-step open-loop drift and keeps compute reasonable.
    Uses VALUES ONLY (no datetime freq issues).
    """
    order, seasonal_order = _sarima_get_params(mode, errors)
    n_test = len(y_test)
    if order is None:
        return np.full(n_test, np.nan, dtype=float)

    y_train = _safe_fill_series(y_train)
    y_test = _safe_fill_series(y_test)

    train_vals = y_train.to_numpy(dtype=float)
    test_vals = y_test.to_numpy(dtype=float)

    preds = np.full(n_test, np.nan, dtype=float)

    n_blocks = int(np.ceil(n_test / block_size))
    print(f"[SARIMA] walk-forward blocks: block_size={block_size}, blocks={n_blocks}, fit_window={fit_window}")
    print(f"[SARIMA] Using order={order}, seasonal_order={seasonal_order}")

    t_all = time.time()

    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n_test)
        H = end - start

        # history = train + observed test so far
        hist = np.concatenate([train_vals, test_vals[:start]], axis=0)

        if fit_window is not None and len(hist) > fit_window:
            hist = hist[-fit_window:]

        if len(hist) < 50:
            errors.append(f"[SARIMA] Too little history at block {b}: len(hist)={len(hist)}")
            preds[start:end] = np.nan
            continue

        try:
            t0 = time.time()
            res = SARIMAX(
                hist,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, method="lbfgs", maxiter=maxiter)

            block_pred = np.asarray(res.forecast(steps=H), dtype=float).reshape(-1)
            preds[start:end] = block_pred

            dt = time.time() - t0
            print(f"[SARIMA] block {b+1}/{n_blocks} fit+forecast in {dt:.1f}s | hist={len(hist)} | H={H}")

        except Exception as e:
            errors.append(f"[SARIMA] block {b+1}/{n_blocks} failed: {e}")
            preds[start:end] = np.nan

    print(f"[SARIMA] total walk-forward time: {time.time() - t_all:.1f}s")
    return preds


def evaluate(mode: str) -> None:
    errors: List[str] = []

    data_path = DATA_DIR / f"{mode}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed data: {data_path}")

    df = pd.read_parquet(data_path)
    df = _ensure_datetime_index(df)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = df.sort_index()

    if "y" not in df.columns:
        raise ValueError(f"'y' column not found in {data_path}")

    # Drop missing target (cannot evaluate)
    df = df.dropna(subset=["y"])

    y_full = df["y"].astype(float)
    X_full = _safe_fill_features(df.drop(columns=["y"]))

    # 1-year test split
    test_size = 365 if mode == "daily" else 24 * 365  # 8760
    if len(df) <= test_size + 10:
        raise ValueError(f"Not enough rows for 1-year test after cleaning. rows={len(df)} test={test_size}")

    split_idx = len(df) - test_size
    y_train, y_test = y_full.iloc[:split_idx], y_full.iloc[split_idx:]
    X_train, X_test = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    n_test = len(y_test)

    print(f"📊 FINAL EVALUATION ({mode.upper()})")
    print(f"[INFO] mode={mode} | train={len(y_train)}, test={len(y_test)}")
    print(f"[INFO] features={X_full.shape[1]}")

    season_len = 7 if mode == "daily" else 24

    # SARIMA block settings (your request):
    # hourly: 168h blocks refit each week
    # daily: 30-day blocks (optional, keeps behavior consistent)
    if mode == "hourly":
        sar_block = 168
        # keep SARIMA fast: fit only on last 30 days of history (720 points)
        sar_fit_window = 24 * 30  # 720
        sar_maxiter = 60
        print(f"[INFO] SARIMA hourly: weekly refit (block={sar_block}) | fit_window(last)={sar_fit_window} | maxiter={sar_maxiter}")
    else:
        sar_block = 30
        sar_fit_window = 365 * 3  # last 3 years daily (safe)
        sar_maxiter = 150
        print(f"[INFO] SARIMA daily: block={sar_block} | fit_window(last)={sar_fit_window} | maxiter={sar_maxiter}")

    print("→ Evaluating baselines (NO Holt-Winters).")

    series_test: Dict[str, np.ndarray] = {}
    series_train: Dict[str, np.ndarray] = {}

    # Naive
    naive_test = baseline_naive(y_train, y_test)
    series_test["Naive"] = _pad_to_length(naive_test, n_test)
    series_train["Naive"] = _pad_to_length(np.r_[np.nan, y_train.iloc[:-1].to_numpy()], len(y_train))

    # Seasonal Naive
    sn_test = baseline_seasonal_naive(y_train, y_test, season_len)
    series_test["Seasonal Naive"] = _pad_to_length(sn_test, n_test)
    series_train["Seasonal Naive"] = _pad_to_length(y_train.shift(season_len).bfill().to_numpy(), len(y_train))

    # SARIMA walk-forward blocks (full test coverage, no null gaps)
    sar_pred_test = sarima_walk_forward_blocks(
        y_train=y_train,
        y_test=y_test,
        mode=mode,
        errors=errors,
        block_size=sar_block,
        fit_window=sar_fit_window,
        maxiter=sar_maxiter,
    )
    series_test["SARIMA"] = _pad_to_length(sar_pred_test, n_test)
    series_train["SARIMA"] = np.full(len(y_train), np.nan, dtype=float)

    print("→ Evaluating ML models (SVR / MLP / XGBoost / RF / LGBM).")

    ml_specs = [
        ("SVR", "svr"),
        ("MLP", "mlp"),
        ("XGBoost", "xgb"),
        ("RandomForest", "rf"),
        ("LightGBM", "lgbm"),
    ]

    for disp_name, model_id in ml_specs:
        model = _load_model(model_id, mode)
        if model is None:
            errors.append(f"[ML] Missing model: {MODELS_DIR / f'{model_id}_{mode}.pkl'}")
            continue
        try:
            pred_test = _predict_model(model, X_test)
            pred_train = _predict_model(model, X_train)
            series_test[disp_name] = _pad_to_length(pred_test, n_test)
            series_train[disp_name] = _pad_to_length(pred_train, len(y_train))
        except Exception as e:
            errors.append(f"[ML] {disp_name} predict failed: {e}")

    # UI safety (lengths)
    for k, v in list(series_test.items()):
        series_test[k] = _pad_to_length(v, n_test)
    for k, v in list(series_train.items()):
        series_train[k] = _pad_to_length(v, len(y_train))

    def add_metrics(name: str, typ: str, y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> Dict[str, float]:
        m = compute_metrics(y_true_arr, y_pred_arr)
        out = {"Model": name, "Type": typ}
        out.update(m)
        return out

    metrics_test: List[Dict[str, float]] = []
    metrics_train: List[Dict[str, float]] = []

    baseline_names = ["Naive", "Seasonal Naive", "SARIMA"]
    ml_names = ["SVR", "MLP", "XGBoost", "RandomForest", "LightGBM"]

    for name in baseline_names:
        metrics_test.append(add_metrics(name, "baseline", y_test.to_numpy(), series_test[name]))
        metrics_train.append(add_metrics(name, "baseline", y_train.to_numpy(), series_train[name]))

    for name in ml_names:
        if name in series_test:
            metrics_test.append(add_metrics(name, "ml", y_test.to_numpy(), series_test[name]))
        if name in series_train:
            metrics_train.append(add_metrics(name, "ml", y_train.to_numpy(), series_train[name]))

    df_test = pd.DataFrame(metrics_test).sort_values("sMAPE", ascending=True).reset_index(drop=True)
    df_train = pd.DataFrame(metrics_train).sort_values("sMAPE", ascending=True).reset_index(drop=True)

    out = {
        "mode": mode,
        "train_window": int(len(y_train)),
        "test_window": int(len(y_test)),
        "feature_columns": list(X_full.columns),

        "dates": [d.isoformat() for d in y_test.index],
        "actual": [float(v) if np.isfinite(v) else None for v in np.asarray(y_test.to_numpy(), dtype=float)],

        "series": {k: _series_to_json_list(series_test[k]) for k in sorted(series_test.keys())},
        "metrics": json.loads(df_test.to_json(orient="records")),

        "train_dates": [d.isoformat() for d in y_train.index],
        "train_actual": [float(v) if np.isfinite(v) else None for v in np.asarray(y_train.to_numpy(), dtype=float)],

        "train_series": {k: _series_to_json_list(series_train[k]) for k in sorted(series_train.keys())},
        "metrics_train": json.loads(df_train.to_json(orient="records")),

        "errors": errors,
        "sarima_info": {
            "block_size": int(sar_block),
            "fit_window": int(sar_fit_window) if sar_fit_window is not None else None,
            "maxiter": int(sar_maxiter),
            "note": "Walk-forward rolling-origin SARIMA: refit each block, forecast next block.",
        },
    }

    out_path = BASE_DIR / f"dashboard_data_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] test lengths: dates={len(out['dates'])}, actual={len(out['actual'])}")
    for k in sorted(out["series"].keys()):
        print(f"[DEBUG] series[{k}] len={len(out['series'][k])}")

    print(f"✅ Saved: {out_path}")

    if errors:
        print("⚠️ Notes (first 20):")
        for e in errors[:20]:
            print(" -", e)
        if len(errors) > 20:
            print(f" - ... {len(errors) - 20} more")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    args = parser.parse_args()
    evaluate(args.mode)


if __name__ == "__main__":
    main()
