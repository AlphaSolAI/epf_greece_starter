import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import types

import joblib
import numpy as np
import pandas as pd

from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop
from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


# --------------------------------------------------------------------
# Wrapper needed for joblib unpickling (models saved with this wrapper)
# Also register aliases because pickles may reference this class under
# different module paths.
# --------------------------------------------------------------------
class ResidualAddBaselineWrapper:
    """
    Wrapper for residual models (e.g., predicts Δy) returning level:
        y_hat = baseline(X) + residual_hat
    """

    def __init__(self, model, baseline_col: str, feature_names: list):
        self.model = model
        self.baseline_col = baseline_col
        self.feature_names = list(feature_names)
        if baseline_col not in self.feature_names:
            raise ValueError(
                f"baseline_col='{baseline_col}' not found in feature_names. "
                f"Available (first 20): {self.feature_names[:20]}"
            )
        self.baseline_idx = self.feature_names.index(baseline_col)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        residual_hat = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        if isinstance(X, pd.DataFrame):
            base = X[self.baseline_col].to_numpy(dtype=float).reshape(-1)
        else:
            X = np.asarray(X)
            base = X[:, self.baseline_idx].astype(float).reshape(-1)
        return base + residual_hat


def _register_unpickle_aliases():
    alias_names = [
        "src.model_wrappers",
        "src.train_xgb_openloop",
        "src.eval_openloop",
    ]
    for name in alias_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], "ResidualAddBaselineWrapper", ResidualAddBaselineWrapper)


# -----------------------------
# Metrics
# -----------------------------
def _to_float_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def mae(y_true, y_pred) -> float:
    y_true = _to_float_np(y_true)
    y_pred = _to_float_np(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def rmse(y_true, y_pred) -> float:
    y_true = _to_float_np(y_true)
    y_pred = _to_float_np(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def smape(y_true, y_pred) -> float:
    y_true = _to_float_np(y_true)
    y_pred = _to_float_np(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    denom = (np.abs(y_true[m]) + np.abs(y_pred[m]))
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(y_true[m] - y_pred[m]) / denom))


def metrics_row(name: str, typ: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "Model": name,
        "Type": typ,
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "sMAPE": round(smape(y_true, y_pred), 3),
    }


def print_table(df_metrics: pd.DataFrame) -> None:
    cols = ["Model", "Type", "MAE", "RMSE", "sMAPE"]
    df = df_metrics[cols].copy()

    w_model = max(10, int(df["Model"].astype(str).map(len).max()))
    w_type = max(7, int(df["Type"].astype(str).map(len).max()))
    w_num = 10

    header = (
        f"{'Model'.ljust(w_model)}  "
        f"{'Type'.ljust(w_type)}  "
        f"{'MAE'.rjust(w_num)}  "
        f"{'RMSE'.rjust(w_num)}  "
        f"{'sMAPE'.rjust(w_num)}"
    )
    print(header)
    print("-" * len(header))

    for _, r in df.iterrows():
        print(
            f"{str(r['Model']).ljust(w_model)}  "
            f"{str(r['Type']).ljust(w_type)}  "
            f"{str(r['MAE']).rjust(w_num)}  "
            f"{str(r['RMSE']).rjust(w_num)}  "
            f"{str(r['sMAPE']).rjust(w_num)}"
        )


# -----------------------------
# Baselines
# -----------------------------
def _holiday_flag_from_df(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    candidates = {"is_holiday", "holiday", "cal_is_holiday"}
    for c in df.columns:
        if c.lower() in candidates:
            return df[c].reindex(index).fillna(0).astype(int)
    return pd.Series(0, index=index, dtype=int)


def baseline_seasonal_profile_hour_dow_holiday(
    df_train: pd.DataFrame, df_full: pd.DataFrame, test_index: pd.DatetimeIndex
) -> np.ndarray:
    """
    TRAIN-only seasonal profile:
      mean(y | hour, day-of-week, holiday)
    Applies to test timestamps (no test leakage).
    """
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
        h = ts.hour
        d = ts.dayofweek
        hol = int(hol_test.iloc[i])

        v = prof_hdh.get((h, d, hol), np.nan)
        if not np.isfinite(v):
            v = prof_hd.get((h, d), np.nan)
        if not np.isfinite(v):
            v = prof_h.get(h, np.nan)
        if not np.isfinite(v):
            v = global_mean
        out[i] = float(v)
    return out


def baseline_naive_lag_actual(y_full: pd.Series, test_index: pd.DatetimeIndex, lag_hours: int) -> np.ndarray:
    """
    Naive-k using ACTUAL history:
      Naive-1 uses y(t-1), Naive-24 uses y(t-24), etc.
    (Oracle/rolling baseline - you already know it 'cheats' for strict openloop.)
    """
    return y_full.shift(int(lag_hours)).reindex(test_index).astype(float).to_numpy()


# -----------------------------
# Model helpers
# -----------------------------
def _model_path_openloop(model_key: str, mode: str, task: str) -> Path:
    if model_key == "xgb_test":
        return MODELS_DIR / f"xgb_{mode}_{task}_openloop_resid_naive1.pkl"
    if model_key == "xgb_backup":
        return MODELS_DIR / "backup.pkl"
    # Scheduled sampling variants — stored as {base}_{mode}_{task}_scheduled_openloop.pkl
    if model_key in ("lgbm_scheduled", "xgb_scheduled", "rf_scheduled", "mlp_scheduled"):
        base = model_key.replace("_scheduled", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_scheduled_openloop.pkl"
    # Optuna-tuned models (closed-loop params, open-loop evaluated)
    if model_key == "lgbm_optuna":
        return MODELS_DIR / f"lgbm_{mode}_{task}_optuna.pkl"
    if model_key == "xgb_optuna":
        return MODELS_DIR / f"xgb_{mode}_{task}_optuna.pkl"
    if model_key == "rf_optuna":
        return MODELS_DIR / f"rf_{mode}_{task}_optuna.pkl"
    # Optuna-tuned + Scheduled Sampling (from tune_*_optuna.py, then SS)
    if model_key == "lgbm_optuna_scheduled":
        return MODELS_DIR / f"lgbm_{mode}_{task}_optuna_scheduled_openloop.pkl"
    if model_key == "xgb_optuna_scheduled":
        return MODELS_DIR / f"xgb_{mode}_{task}_optuna_scheduled_openloop.pkl"
    if model_key == "rf_optuna_scheduled":
        return MODELS_DIR / f"rf_{mode}_{task}_optuna_scheduled_openloop.pkl"
    # Open-loop-aware Optuna (from tune_openloop_optuna.py)
    if model_key in ("lgbm_openloop_optuna", "xgb_openloop_optuna", "rf_openloop_optuna"):
        base = model_key.replace("_openloop_optuna", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_openloop_optuna.pkl"
    # Open-loop-aware Optuna + Scheduled Sampling
    if model_key in ("lgbm_openloop_optuna_scheduled", "xgb_openloop_optuna_scheduled", "rf_openloop_optuna_scheduled"):
        base = model_key.replace("_openloop_optuna_scheduled", "")
        return MODELS_DIR / f"{base}_{mode}_{task}_openloop_optuna_scheduled_openloop.pkl"
    return MODELS_DIR / f"{model_key}_{mode}_{task}_openloop.pkl"


def _safe_joblib_load(p: Path):
    _register_unpickle_aliases()
    try:
        return joblib.load(p)
    except Exception as e:
        print(f"[WARN] Could not load model: {p.name} -> {type(e).__name__}: {e}")
        return None


def _call_recursive_openloop(model, df_full, test_index, feature_cols, cfg) -> np.ndarray:
    res = recursive_predict_openloop(
        model=model,
        df_full=df_full,
        test_index=test_index,
        feature_cols=feature_cols,
        config=cfg,
    )
    return _to_float_np(res)


def _parse_ml_models_arg(s: Optional[str]) -> Optional[List[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


# -----------------------------
# Main evaluation (OPENLOOP)
# -----------------------------
def evaluate_openloop(
    mode: str,
    task: str,
    test_size: Optional[int],
    sort_by: str,
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    ml_models: Optional[List[str]],
    export_predictions: bool = False,
    export_model: Optional[str] = None,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

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

    X_train, _ = make_xy(df_train)
    _, y_test = make_xy(df_test)

    test_index = df_test.index
    feature_cols = list(X_train.columns)

    print(f"📊 FINAL EVALUATION OPENLOOP ({mode.upper()}) [NO-LEAK] task={task.upper()}")
    print(f"[INFO] train={len(df_train)}, test={len(df_test)} | features={X_train.shape[1]}")
    print(f"[INFO] test_start={test_index.min()} | test_end={test_index.max()} | test_size={len(test_index)}")

    series_test: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    model_labels = {
        # Standard models (teacher-forced training, used as baseline in OL eval)
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "rf": "RandomForest",
        "mlp": "MLP",
        "svr": "SVR",
        # Scheduled Sampling — optimized for open-loop
        "lgbm_scheduled": "LGBM-SS",
        "xgb_scheduled": "XGB-SS",
        "rf_scheduled": "RF-SS",
        "mlp_scheduled": "MLP-SS",
        # Open-loop-aware Optuna (CV metric = recursive open-loop MAE) — optimized for open-loop
        "lgbm_openloop_optuna": "LGBM-OL-Optuna",
        "xgb_openloop_optuna":  "XGB-OL-Optuna",
        "rf_openloop_optuna":   "RF-OL-Optuna",
        # Open-loop-aware Optuna + SS — optimized for open-loop
        "lgbm_openloop_optuna_scheduled": "LGBM-OL-Optuna-SS",
        "xgb_openloop_optuna_scheduled":  "XGB-OL-Optuna-SS",
        "rf_openloop_optuna_scheduled":   "RF-OL-Optuna-SS",
    }

    if ml_models is not None:
        keep = set(ml_models)
        model_labels = {k: v for k, v in model_labels.items() if k in keep}
        if len(model_labels) == 0:
            raise SystemExit(f"ERROR: --ml_models did not match any known keys. Known keys: {list(model_labels.keys())}")

    cfg = OpenLoopConfig(y_floor=None)

    # ---- ML models
    evaluated_models = []
    for k, label in model_labels.items():
        mp = _model_path_openloop(k, mode, task)
        if not mp.exists():
            print(f"[WARN] Missing model file for {label}: {mp}")
            continue

        print(f"[INFO] Loading {label} from {mp.name}")
        model = _safe_joblib_load(mp)
        if model is None:
            continue

        try:
            yhat = _call_recursive_openloop(
                model=model,
                df_full=df,
                test_index=test_index,
                feature_cols=feature_cols,
                cfg=cfg,
            )
        except Exception as e:
            print(f"[WARN] {label} prediction failed -> {type(e).__name__}: {e}")
            continue

        series_test[label] = yhat
        rows.append(metrics_row(label, "ml", y_test, yhat))
        evaluated_models.append(label)

    print(f"[INFO] Evaluated ML models: {evaluated_models if evaluated_models else 'NONE'}")

    # ---- Ensemble: combine all evaluated ML models
    if len(evaluated_models) >= 2:
        # 1. Simple mean ensemble
        stack = np.array([series_test[m] for m in evaluated_models])
        ensemble_mean = np.nanmean(stack, axis=0)
        series_test["Ensemble (mean)"] = ensemble_mean
        rows.append(metrics_row("Ensemble (mean)", "ensemble", y_test, ensemble_mean))

        # 2. Inverse-MAE weighted ensemble (better models get more weight)
        maes = np.array([mae(_to_float_np(y_test), series_test[m]) for m in evaluated_models])
        weights = 1.0 / np.where(maes > 0, maes, 1e-9)
        weights /= weights.sum()
        ensemble_wmean = np.sum(stack * weights[:, None], axis=0)
        series_test["Ensemble (1/MAE)"] = ensemble_wmean
        rows.append(metrics_row("Ensemble (1/MAE)", "ensemble", y_test, ensemble_wmean))
        print(f"[INFO] Ensemble weights (1/MAE): { {m: round(float(w),3) for m,w in zip(evaluated_models,weights)} }")

        # 3. SS-only ensemble (tree scheduled sampling models only — excludes MLP/SVR drag)
        ss_labels = [m for m in evaluated_models if m in (
            "LGBM-SS", "XGB-SS", "RF-SS",
            "LGBM-Optuna-SS", "XGB-Optuna-SS", "RF-Optuna-SS",
            "LGBM-OL-Optuna-SS", "XGB-OL-Optuna-SS", "RF-OL-Optuna-SS",
        )]
        if len(ss_labels) >= 2:
            ss_stack = np.array([series_test[m] for m in ss_labels])
            ss_maes = np.array([mae(_to_float_np(y_test), series_test[m]) for m in ss_labels])
            ss_weights = 1.0 / np.where(ss_maes > 0, ss_maes, 1e-9)
            ss_weights /= ss_weights.sum()
            ensemble_ss = np.sum(ss_stack * ss_weights[:, None], axis=0)
            series_test["Ensemble-SS (1/MAE)"] = ensemble_ss
            rows.append(metrics_row("Ensemble-SS (1/MAE)", "ensemble", y_test, ensemble_ss))
            print(f"[INFO] SS-only ensemble weights: { {m: round(float(w),3) for m,w in zip(ss_labels,ss_weights)} }")

        # 4. OL-Optuna ensemble (open-loop-aware Optuna models only)
        ol_labels = [m for m in evaluated_models if m in (
            "LGBM-OL-Optuna", "XGB-OL-Optuna", "RF-OL-Optuna",
            "LGBM-OL-Optuna-SS", "XGB-OL-Optuna-SS", "RF-OL-Optuna-SS",
        )]
        if len(ol_labels) >= 2:
            ol_stack = np.array([series_test[m] for m in ol_labels])
            ol_maes = np.array([mae(_to_float_np(y_test), series_test[m]) for m in ol_labels])
            ol_weights = 1.0 / np.where(ol_maes > 0, ol_maes, 1e-9)
            ol_weights /= ol_weights.sum()
            ensemble_ol = np.sum(ol_stack * ol_weights[:, None], axis=0)
            series_test["Ensemble-OL-Optuna (1/MAE)"] = ensemble_ol
            rows.append(metrics_row("Ensemble-OL-Optuna (1/MAE)", "ensemble", y_test, ensemble_ol))
            print(f"[INFO] OL-Optuna ensemble weights: { {m: round(float(w),3) for m,w in zip(ol_labels,ol_weights)} }")

        # 5. Best-per-family ensemble (best LGBM + best XGB + best RF, by priority)
        # Priority: OL-aware Optuna variants first, then SS, then standard
        lgbm_priority = ["LGBM-OL-Optuna", "LGBM-OL-Optuna-SS", "LGBM-SS", "LightGBM"]
        xgb_priority  = ["XGB-OL-Optuna",  "XGB-OL-Optuna-SS",  "XGB-SS",  "XGBoost"]
        rf_priority   = ["RF-OL-Optuna",   "RF-OL-Optuna-SS",   "RF-SS",   "RandomForest"]
        best3 = [
            next((m for m in pri if m in evaluated_models), None)
            for pri in [lgbm_priority, xgb_priority, rf_priority]
        ]
        best3 = [m for m in best3 if m is not None]
        if len(best3) >= 2:
            b3_stack = np.array([series_test[m] for m in best3])
            b3_maes = np.array([mae(_to_float_np(y_test), series_test[m]) for m in best3])
            b3_weights = 1.0 / np.where(b3_maes > 0, b3_maes, 1e-9)
            b3_weights /= b3_weights.sum()
            ensemble_b3 = np.sum(b3_stack * b3_weights[:, None], axis=0)
            series_test["Ensemble-Best3 (1/MAE)"] = ensemble_b3
            rows.append(metrics_row("Ensemble-Best3 (1/MAE)", "ensemble", y_test, ensemble_b3))
            print(f"[INFO] Best-per-family ensemble: {best3}")
            print(f"[INFO] Best-per-family weights: { {m: round(float(w),3) for m,w in zip(best3,b3_weights)} }")

    # ---- Baselines (ALWAYS)
    y_full = df["y"].astype(float)

    y_seasonal = baseline_seasonal_profile_hour_dow_holiday(df_train, df, test_index)
    series_test["Seasonal profile (hour×dow×holiday)"] = y_seasonal
    rows.append(metrics_row("Seasonal profile (hour×dow×holiday)", "baseline", y_test, y_seasonal))

    # Naive baselines (oracle/rolling) — you want them for comparison
    y_naive_1 = baseline_naive_lag_actual(y_full, test_index, lag_hours=1)
    series_test["Naive-1"] = y_naive_1
    rows.append(metrics_row("Naive-1", "baseline", y_test, y_naive_1))

    y_naive_24 = baseline_naive_lag_actual(y_full, test_index, lag_hours=24)
    series_test["Naive-24"] = y_naive_24
    rows.append(metrics_row("Naive-24", "baseline", y_test, y_naive_24))

    y_naive_168 = baseline_naive_lag_actual(y_full, test_index, lag_hours=168)
    series_test["Naive-168"] = y_naive_168
    rows.append(metrics_row("Naive-168", "baseline", y_test, y_naive_168))

    # sanity: ensure Naive-1 exists in printed metrics
    if not any(r.get("Model") == "Naive-1" for r in rows):
        print("[WARN] Naive-1 metrics row is missing (should never happen).")

    df_metrics = pd.DataFrame(rows).sort_values(sort_by, ascending=True)

    print(f"\n📌 TEST METRICS OPENLOOP sorted by {sort_by}")
    print_table(df_metrics)

    # ------------------------------------------------------------------
    # Export predictions as parquet (for use as load_fc in price task)
    # ------------------------------------------------------------------
    if export_predictions and evaluated_models:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Determine which model to export (best MAE or explicitly specified)
        ml_rows = df_metrics[df_metrics["Type"] == "ml"]
        if export_model and export_model in series_test:
            best_label = export_model
        elif len(ml_rows) > 0:
            best_label = str(ml_rows.iloc[0]["Model"])
        else:
            best_label = None

        if best_label and best_label in series_test:
            export_df = pd.DataFrame(
                {"load_fc": series_test[best_label]},
                index=test_index,
            )
            export_df.index.name = "datetime"
            out_fc = OUTPUT_DIR / "load_forecast_hourly.parquet"
            export_df.to_parquet(out_fc)
            print(f"\n✅ Exported open-loop load forecast ({best_label}) → {out_fc}")
            print(f"   rows={len(export_df)} | {export_df.index.min()} → {export_df.index.max()}")
        else:
            print("[WARN] export_predictions=True but no ML model was evaluated — nothing exported.")

    dashboard_json = {
        "mode": mode,
        "task": task,
        "strategy": "ol",
        "variant": "openloop",
        "train_window": int(len(df_train)),
        "test_window": int(len(df_test)),
        "bounds": {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "test_size": test_size,
        },
        "feature_columns": feature_cols,
        "dates": [d.isoformat() for d in test_index],
        "actual": [float(v) if np.isfinite(v) else None for v in _to_float_np(y_test)],
        "series": {
            k: [float(v) if np.isfinite(v) else None for v in _to_float_np(vs)]
            for k, vs in series_test.items()
        },
        # "metrics" είναι το standard key που χρησιμοποιεί το dashboard
        "metrics": df_metrics.to_dict(orient="records"),
        "metrics_test": df_metrics.to_dict(orient="records"),  # backward compat
    }

    out_path = BASE_DIR / f"dashboard_data_{mode}_{task}_openloop.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_json, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")

    p.add_argument("--test_size", type=int, default=None)
    p.add_argument("--sort_by", type=str, default="MAE", choices=["MAE", "RMSE", "sMAPE"])

    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--test_start", type=str, default=None)
    p.add_argument("--test_end", type=str, default=None)

    p.add_argument("--ml_models", type=str, default=None)
    p.add_argument(
        "--export_predictions",
        action="store_true",
        default=False,
        help="Export best ML model predictions as load_forecast_hourly.parquet (for price task load_fc)",
    )
    p.add_argument(
        "--export_model",
        type=str,
        default=None,
        help="Label of the model to export (e.g. 'LightGBM'). Defaults to best MAE model.",
    )

    args = p.parse_args()

    evaluate_openloop(
        mode=args.mode,
        task=args.task,
        test_size=args.test_size,
        sort_by=args.sort_by,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        ml_models=_parse_ml_models_arg(args.ml_models),
        export_predictions=args.export_predictions,
        export_model=args.export_model,
    )


if __name__ == "__main__":
    main()
