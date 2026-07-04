# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import inspect
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Silence noisy-but-harmless warnings (clean terminal output)
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but RandomForestRegressor was fitted without feature names",
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Falling back to prediction using DMatrix due to mismatched devices.*",
)
warnings.filterwarnings(
    "ignore",
    message=r'mmap_mode "r" is not compatible with compressed file.*',
)

# -----------------------------
# MLP unpickle compatibility
# (old pickles may reference: src.eval_mimo.MLPConfig / TorchMLPConfig / TorchMLPMIMORegressor)
# -----------------------------
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class MLPConfig:
    hidden_layers: List[int]
    dropout: float = 0.0
    activation: str = "relu"
    device: str = "auto"


TorchMLPConfig = MLPConfig  # alias for older pickles


class _TorchMLPNet(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: List[int], dropout: float, activation: str):
        super().__init__()
        if torch is None or nn is None:
            raise RuntimeError("PyTorch not available")

        act = activation.lower()
        Act = nn.ReLU if act == "relu" else (nn.GELU if act == "gelu" else nn.ReLU)

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), Act()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _remap_state_dict_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Handles old saved keys like '0.weight' vs new 'net.0.weight'."""
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd
    keys = list(sd.keys())
    has_net = any(k.startswith("net.") for k in keys)
    has_plain = any(k.startswith("0.") or k.startswith("1.") for k in keys)
    if has_plain and not has_net:
        return {f"net.{k}": v for k, v in sd.items()}
    return sd


class TorchMLPMIMORegressor:
    """
    Minimal runtime to load old pickles and run predict().
    Expects (typical fields):
      - cfg, input_dim, output_dim
      - x_scaler (optional), y_scaler (optional)
      - _model_state (state_dict)
      - _is_fitted True
    """

    def __init__(self, cfg: Optional[MLPConfig] = None, input_dim: Optional[int] = None, output_dim: Optional[int] = None):
        self.cfg = cfg if cfg is not None else MLPConfig(hidden_layers=[256, 128])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.x_scaler = None
        self.y_scaler = None
        self._model_state = None
        self._is_fitted = False

    @staticmethod
    def _device(req: str):
        if torch is None:
            return None
        r = (req or "auto").lower()
        if r == "cpu":
            return torch.device("cpu")
        if r == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, X: Any) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch not available")
        if not getattr(self, "_is_fitted", False) or getattr(self, "_model_state", None) is None:
            raise RuntimeError("MLP bundle missing state / not fitted")

        if isinstance(X, pd.DataFrame):
            Xn = X.to_numpy(dtype=np.float32, copy=False)
        else:
            Xn = np.asarray(X, dtype=np.float32)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        if self.x_scaler is not None:
            Xn = self.x_scaler.transform(Xn).astype(np.float32, copy=False)

        dev = self._device(getattr(self.cfg, "device", "auto"))
        model = _TorchMLPNet(
            in_dim=int(self.input_dim),
            out_dim=int(self.output_dim),
            hidden_layers=list(self.cfg.hidden_layers),
            dropout=float(self.cfg.dropout),
            activation=str(self.cfg.activation),
        ).to(dev)

        sd = _remap_state_dict_keys(self._model_state)  # type: ignore[arg-type]
        model.load_state_dict(sd, strict=True)  # type: ignore[arg-type]
        model.eval()

        xb = torch.from_numpy(Xn).to(dev)
        with torch.no_grad():
            pred = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)

        if self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred).astype(np.float32, copy=False)

        return pred


# -----------------------------
# Project utils
# -----------------------------
from src.split_utils import load_processed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def _freq_offset():
    return pd.tseries.frequencies.to_offset("h")


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _safe_load_processed(task: str) -> pd.DataFrame:
    sig = inspect.signature(load_processed)
    kwargs: Dict[str, Any] = {}
    if "task" in sig.parameters:
        kwargs["task"] = task
    out = load_processed("hourly", **kwargs)  # type: ignore[arg-type]
    if isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"load_processed returned {type(out)} not DataFrame")
    return out


def _make_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "y" not in df.columns:
        raise KeyError("Processed df must contain column 'y'")
    y = df["y"].astype(float)

    X = df.select_dtypes(include=[np.number, bool]).copy()
    drop_cols = [c for c in ["y", "pm", "publish_time", "publishment_time", "pm_pub_ts"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(np.int8)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X, y


def _scan_model_files(task: str, horizon: int) -> List[Path]:
    htag = f"h{horizon}"
    out: List[Path] = []
    for p in sorted(MODELS_DIR.glob("*.pkl")):
        name = p.name.lower()
        if "hourly" not in name:
            continue
        if htag not in name:
            continue
        # if filename mentions load/price, enforce match
        if ("load" in name or "price" in name) and (task not in name):
            continue
        out.append(p)
    return out


def _joblib_load_mmap(path: Path):
    """
    Best-effort memory-map loading.
    If file is compressed, joblib warns that mmap is not compatible; we silently fallback.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r'mmap_mode "r" is not compatible with compressed file.*')
            return joblib.load(path, mmap_mode="r")
    except Warning:
        return joblib.load(path)
    except Exception:
        return joblib.load(path)


def _load_bundle(path: Path) -> Tuple[str, Any, Optional[List[str]]]:
    obj = _joblib_load_mmap(path)
    name = path.stem
    if isinstance(obj, dict):
        fcols = obj.get("feature_cols")
        if isinstance(fcols, list):
            return name, obj, [str(c) for c in fcols]
        meta = obj.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("feature_cols"), list):
            obj["feature_cols"] = meta["feature_cols"]
            return name, obj, [str(c) for c in meta["feature_cols"]]
        return name, obj, None
    return name, obj, None


def _is_lgbm_estimator(est: Any) -> bool:
    try:
        cls = est.__class__
        name = cls.__name__.lower()
        mod = (cls.__module__ or "").lower()
        if "lgbm" in name or "lightgbm" in mod:
            return True
    except Exception:
        pass
    # sklearn wrapper often has feature_names_in_ when trained with DF
    if hasattr(est, "feature_names_in_"):
        return True
    return False


def _is_rf_estimator(est: Any) -> bool:
    try:
        cls = est.__class__
        name = cls.__name__.lower()
        mod = (cls.__module__ or "").lower()
        if "randomforest" in name or ("sklearn.ensemble" in mod and "forest" in mod):
            return True
    except Exception:
        pass
    return False


def _predict_est(est: Any, X_df: pd.DataFrame, X_np: np.ndarray) -> np.ndarray:
    """
    Key point to silence BOTH sides:
      - LGBMRegressor trained with feature names => predict with DataFrame (keeps names)
      - RandomForest fitted without feature names => predict with numpy (no names)
    """
    try:
        if _is_lgbm_estimator(est):
            return np.asarray(est.predict(X_df))
        if _is_rf_estimator(est):
            return np.asarray(est.predict(X_np))
        # default: try DF then numpy
        try:
            return np.asarray(est.predict(X_df))
        except Exception:
            return np.asarray(est.predict(X_np))
    except Exception:
        # final fallback
        return np.asarray(est.predict(X_np))


def _infer_expected_features(bundle: Any) -> Tuple[Optional[int], Optional[List[str]]]:
    def infer_from_est(est: Any) -> Tuple[Optional[int], Optional[List[str]]]:
        nfi = getattr(est, "n_features_in_", None)
        expected_n = int(nfi) if isinstance(nfi, (int, np.integer)) else None

        fni = getattr(est, "feature_names_in_", None)
        if isinstance(fni, (list, np.ndarray)) and len(fni) > 0:
            return expected_n, [str(x) for x in list(fni)]

        booster = getattr(est, "booster_", None)
        if booster is not None:
            try:
                fn = booster.feature_name()
                if isinstance(fn, list) and len(fn) > 0:
                    return expected_n, [str(x) for x in fn]
            except Exception:
                pass

        gb = None
        try:
            gb = est.get_booster()
        except Exception:
            gb = None
        if gb is not None:
            try:
                fn = gb.feature_names
                if isinstance(fn, list) and len(fn) > 0:
                    return expected_n, [str(x) for x in fn]
            except Exception:
                pass

        return expected_n, None

    if isinstance(bundle, dict):
        if "feature_cols" in bundle and isinstance(bundle["feature_cols"], list):
            return None, [str(c) for c in bundle["feature_cols"]]
        if "model" in bundle:
            m = bundle["model"]
            xs = getattr(m, "x_scaler", None)
            if xs is not None:
                nfi = getattr(xs, "n_features_in_", None)
                if isinstance(nfi, (int, np.integer)):
                    return int(nfi), None
            return infer_from_est(m)

    return infer_from_est(bundle)


def _select_feature_cols(
    *,
    saved_feature_cols: Optional[List[str]],
    bundle: Any,
    X_all: pd.DataFrame,
    model_file: str,
) -> Optional[List[str]]:
    if saved_feature_cols is not None:
        missing = [c for c in saved_feature_cols if c not in X_all.columns]
        if missing:
            print(f"[WARN] skip {model_file}: missing {len(missing)} feature cols", flush=True)
            return None
        return saved_feature_cols

    expected_n, inferred_names = _infer_expected_features(bundle)
    if inferred_names is not None:
        missing = [c for c in inferred_names if c not in X_all.columns]
        if missing:
            print(f"[WARN] skip {model_file}: feature mismatch ({len(missing)} missing)", flush=True)
            return None
        return inferred_names

    if expected_n is None:
        return list(X_all.columns)

    if expected_n == int(X_all.shape[1]):
        return list(X_all.columns)

    print(
        f"[WARN] skip {model_file}: expects {expected_n} feats but current has {X_all.shape[1]} (no feature_cols stored)",
        flush=True,
    )
    return None


def _predict_vector(bundle: Any, x_row: pd.Series, feature_cols: List[str], horizon: int) -> np.ndarray:
    X_df = pd.DataFrame([x_row[feature_cols].values], columns=feature_cols)
    X_np = x_row[feature_cols].to_numpy(dtype=np.float32, copy=False).reshape(1, -1)

    if isinstance(bundle, dict) and "model" in bundle:
        p = _predict_est(bundle["model"], X_df, X_np).reshape(-1)
        if p.size == 1:
            return np.repeat(float(p[0]), horizon).astype(float)
        return p[:horizon].astype(float)

    p = _predict_est(bundle, X_df, X_np).reshape(-1)
    if p.size == 1:
        return np.repeat(float(p[0]), horizon).astype(float)
    return p[:horizon].astype(float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Returns MAE, RMSE, sMAPE, WAPE, N
    - sMAPE: 200*|e|/(|y|+|yhat|+eps)
    - WAPE: 100*sum(|e|)/sum(|y|+eps)
    """
    if y_true.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "sMAPE": float("nan"), "WAPE": float("nan"), "N": 0.0}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    eps = 1e-6
    denom_smape = np.abs(y_true) + np.abs(y_pred) + eps
    smape = float(np.mean(200.0 * np.abs(err) / denom_smape))

    denom_wape = float(np.sum(np.abs(y_true)) + eps)
    wape = float(100.0 * np.sum(np.abs(err)) / denom_wape)

    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "WAPE": wape, "N": float(y_true.size)}


# -----------------------------
# BASELINES
# -----------------------------
def _baseline_naive_1(y_hist: pd.Series, horizon: int) -> np.ndarray:
    """Persistence: repeat last observed for all H steps (no cheating)."""
    y_hist = y_hist.dropna()
    if len(y_hist) == 0:
        return np.full(horizon, np.nan, dtype=float)
    last = float(y_hist.iloc[-1])
    return np.full(horizon, last, dtype=float)


def _baseline_seasonal_naive_recursive(
    y_hist: pd.Series, origin: pd.Timestamp, horizon: int, season: int, freq
) -> np.ndarray:
    """
    Seasonal naive multi-step (no cheating):
      for h<=season: yhat(t+h)=y(t+h-season) (from history)
      for h>season:  yhat(t+h)=yhat(t+h-season) (repeat pattern)
    """
    y_hist = y_hist.dropna()
    if len(y_hist) == 0:
        return np.full(horizon, np.nan, dtype=float)

    pred_idx = pd.date_range(origin + freq, periods=horizon, freq=freq)
    yhat = np.full(horizon, np.nan, dtype=float)

    first = min(season, horizon)
    shifted = pred_idx[:first] - season * freq
    base = y_hist.reindex(shifted).to_numpy(dtype=float)
    yhat[:first] = base

    for i in range(season, horizon):
        yhat[i] = yhat[i - season]

    last = float(y_hist.iloc[-1])
    yhat = np.where(np.isfinite(yhat), yhat, last)
    return yhat


def _baseline_shift(
    y_all: pd.Series, origin: pd.Timestamp, horizon: int, shift_steps: int, freq
) -> np.ndarray:
    """
    SHIFT baseline (cheating baseline):
    For each target timestamp t in the horizon window, predict y(t - shift_steps).
    This uses values that may be inside the test window (so it's NOT a real forecast baseline).
    """
    pred_idx = pd.date_range(origin + freq, periods=horizon, freq=freq)
    src_idx = pred_idx - shift_steps * freq
    yhat = y_all.reindex(src_idx).to_numpy(dtype=float)

    # safe fallback if missing: use last available <= origin
    y_hist = y_all.loc[:origin].dropna()
    if len(y_hist) > 0:
        last = float(y_hist.iloc[-1])
        yhat = np.where(np.isfinite(yhat), yhat, last)

    return yhat.astype(float, copy=False)


def _seasonal_profile_predict(y_hist: pd.Series, pred_idx: pd.DatetimeIndex) -> np.ndarray:
    y_hist = y_hist.dropna()
    if len(y_hist) == 0:
        return np.full(len(pred_idx), np.nan, dtype=float)

    global_mean = float(y_hist.mean())
    dfp = pd.DataFrame({"y": y_hist.values}, index=y_hist.index)
    dfp["dow"] = dfp.index.dayofweek
    dfp["hour"] = dfp.index.hour
    prof = dfp.groupby(["dow", "hour"])["y"].mean()

    out = []
    for ts in pred_idx:
        key = (int(ts.dayofweek), int(ts.hour))
        out.append(float(prof.get(key, global_mean)))
    return np.asarray(out, dtype=float)


def _display_name(model_stem: str, model_file: str) -> str:
    s = model_stem.lower()
    fn = model_file.lower()

    if "lgbm" in s:
        base = "LGBM"
        if "direct" in fn:
            strat = "DIRECT"
        elif "mimo" in fn:
            strat = "MIMO"
        else:
            strat = "MIMO"
        return f"{base} {strat}"

    if "xgb" in s:
        base = "XGB"
        # Your convention:
        #  - xgb_vect_* => MIMO
        #  - xgb_*_mimo_* => DIRECT (vector multistep)
        if "vect" in fn:
            strat = "MIMO"
        elif "_mimo_" in fn:
            strat = "DIRECT"
        elif "direct" in fn:
            strat = "DIRECT"
        else:
            strat = "MIMO"
        return f"{base} {strat}"

    if "mlp" in s:
        return "MLP MIMO"
    if "rf" in s:
        return "RF MIMO"

    return "MODEL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hourly"])  # HOURLY ONLY
    ap.add_argument("--task", required=True, choices=["load", "price"])
    ap.add_argument("--horizon", required=True, type=int)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--test_start", required=True)
    ap.add_argument("--test_size", required=True, type=int)
    ap.add_argument("--eval_mode", choices=["single_origin", "multi_origin"], default="single_origin")
    ap.add_argument("--origin_stride", type=int, default=24)
    ap.add_argument("--sort_by", choices=["MAE", "RMSE", "sMAPE", "WAPE"], default="MAE")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated substrings to INCLUDE (e.g. 'lgbm,mlp'). If omitted, all models are loaded.",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated substrings to EXCLUDE from auto-scanned models (e.g. 'rf,segmented').",
    )
    args = ap.parse_args()

    task = args.task.lower()
    H = int(args.horizon)
    freq = _freq_offset()

    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = test_start + (args.test_size - 1) * freq

    print(f"🔎 EVAL START | mode=hourly task={task} H={H}", flush=True)
    print("   -> loading processed data ...", flush=True)

    df_all = _safe_load_processed(task)
    df_all = _ensure_dt_index(df_all)
    X_all, y_all = _make_Xy(df_all)

    print(
        f"   -> data rows={len(df_all)} | features={X_all.shape[1]} | index=[{X_all.index.min()} .. {X_all.index.max()}]",
        flush=True,
    )

    if train_end not in X_all.index:
        prev = X_all.index[X_all.index <= train_end]
        if len(prev) == 0:
            raise KeyError(f"train_end {train_end} not found and no earlier timestamps exist")
        train_end = prev[-1]

    last_origin = test_end - (H - 1) * freq

    if args.eval_mode == "single_origin":
        origins = [train_end]
    else:
        step = int(args.origin_stride) * freq
        if train_end > last_origin:
            origins = [train_end]
        else:
            origins = list(pd.date_range(train_end, last_origin, freq=step))
            if len(origins) == 0:
                origins = [train_end]

    print(f"   -> origins={len(origins)} | origin_stride={args.origin_stride} | test=[{test_start}..{test_end}]", flush=True)

    model_files = _scan_model_files(task, H)
    # Apply --models include filter
    if args.models:
        include_subs = [s.strip().lower() for s in args.models.split(",") if s.strip()]
        model_files = [mf for mf in model_files if any(s in mf.name.lower() for s in include_subs)]
    # Apply --exclude filter
    if args.exclude:
        exclude_subs = [s.strip().lower() for s in args.exclude.split(",") if s.strip()]
        model_files = [mf for mf in model_files if not any(s in mf.name.lower() for s in exclude_subs)]
    print(f"   -> found {len(model_files)} model files in {MODELS_DIR}", flush=True)
    for mf in model_files:
        print(f"      - {mf.name}", flush=True)

    rows: List[Dict[str, Any]] = []

    # Dashboard time-series store (populated only for single_origin)
    _dash_dates:  List[str]   = []
    _dash_actual: List[float] = []
    _dash_series: Dict[str, List[float]] = {}

    # -----------------------------
    # BASELINES
    # -----------------------------
    def _accumulate(pred_idx: pd.DatetimeIndex, yhat: np.ndarray) -> Tuple[List[float], List[float]]:
        mask = (pred_idx >= test_start) & (pred_idx <= test_end)
        idx2 = pred_idx[mask]
        yhat2 = np.asarray(yhat)[mask]
        if len(idx2) == 0:
            return [], []
        ytrue2 = y_all.reindex(idx2).to_numpy(dtype=float)
        ok = np.isfinite(ytrue2) & np.isfinite(yhat2)
        return ytrue2[ok].tolist(), np.asarray(yhat2)[ok].tolist()

    # NAIVE-1 (no cheating)
    yt, yp = [], []
    for origin in origins:
        y_hist = y_all.loc[:origin]
        pred_idx = pd.date_range(origin + freq, periods=H, freq=freq)
        yhat = _baseline_naive_1(y_hist, H)
        a, b = _accumulate(pred_idx, yhat)
        yt.extend(a)
        yp.extend(b)
    rows.append({"Model": "NAIVE-1", "Type": "baseline", **_metrics(np.asarray(yt), np.asarray(yp))})

    # NAIVE-24 (no cheating)
    yt, yp = [], []
    for origin in origins:
        y_hist = y_all.loc[:origin]
        pred_idx = pd.date_range(origin + freq, periods=H, freq=freq)
        yhat = _baseline_seasonal_naive_recursive(y_hist, origin, H, season=24, freq=freq)
        a, b = _accumulate(pred_idx, yhat)
        yt.extend(a)
        yp.extend(b)
    rows.append({"Model": "NAIVE-24", "Type": "baseline", **_metrics(np.asarray(yt), np.asarray(yp))})

    # NAIVE-168 (no cheating)
    yt, yp = [], []
    for origin in origins:
        y_hist = y_all.loc[:origin]
        pred_idx = pd.date_range(origin + freq, periods=H, freq=freq)
        yhat = _baseline_seasonal_naive_recursive(y_hist, origin, H, season=168, freq=freq)
        a, b = _accumulate(pred_idx, yhat)
        yt.extend(a)
        yp.extend(b)
    rows.append({"Model": "NAIVE-168", "Type": "baseline", **_metrics(np.asarray(yt), np.asarray(yp))})

    # NAIVE-1 SHIFT (cheating)
    yt, yp = [], []
    for origin in origins:
        pred_idx = pd.date_range(origin + freq, periods=H, freq=freq)
        yhat = _baseline_shift(y_all, origin, H, shift_steps=1, freq=freq)
        a, b = _accumulate(pred_idx, yhat)
        yt.extend(a)
        yp.extend(b)
    rows.append({"Model": "NAIVE-1 SHIFT", "Type": "baseline", **_metrics(np.asarray(yt), np.asarray(yp))})

    # NAIVE-24 SHIFT (cheating)
    yt, yp = [], []
    for origin in origins:
        pred_idx = pd.date_range(origin + freq, periods=H, freq=freq)
        yhat = _baseline_shift(y_all, origin, H, shift_steps=24, freq=freq)
        a, b = _accumulate(pred_idx, yhat)
        yt.extend(a)
        yp.extend(b)
    rows.append({"Model": "NAIVE-24 SHIFT", "Type": "baseline", **_metrics(np.asarray(yt), np.asarray(yp))})

    # SEASONAL PROFILE
    yt, yp = [], []
    for origin in origins:
        y_hist = y_all.loc[:origin]
        pred_idx = pd.date_range(origin + freq, periods=H, freq=freq)
        yhat = _seasonal_profile_predict(y_hist, pred_idx)
        a, b = _accumulate(pred_idx, yhat)
        yt.extend(a)
        yp.extend(b)
    rows.append({"Model": "SEASONAL PROFILE", "Type": "baseline", **_metrics(np.asarray(yt), np.asarray(yp))})

    # Init dashboard chart arrays for single_origin mode
    if args.eval_mode == "single_origin" and len(origins) > 0:
        _orig0 = origins[0]
        if _orig0 not in X_all.index:
            _prev0 = X_all.index[X_all.index <= _orig0]
            _orig0 = _prev0[-1] if len(_prev0) > 0 else _orig0
        _pidx0 = pd.date_range(_orig0 + freq, periods=H, freq=freq)
        _mask0 = (_pidx0 >= test_start) & (_pidx0 <= test_end)
        _dash_dates  = [str(ts) for ts in _pidx0[_mask0]]
        _dash_actual = y_all.reindex(_pidx0[_mask0]).to_numpy(dtype=float).tolist()
        _y0_hist = y_all.loc[:_orig0]
        # Baselines — keys must match metric Model names (uppercase)
        _yn1   = _baseline_naive_1(_y0_hist, H)
        _dash_series["NAIVE-1"]  = [float(v) for v in _yn1[:H][_mask0]]
        _yn24  = _baseline_seasonal_naive_recursive(_y0_hist, _orig0, H, season=24,  freq=freq)
        _dash_series["NAIVE-24"] = [float(v) for v in _yn24[:H][_mask0]]
        _yn168 = _baseline_seasonal_naive_recursive(_y0_hist, _orig0, H, season=168, freq=freq)
        _dash_series["NAIVE-168"] = [float(v) for v in _yn168[:H][_mask0]]
        _ysp   = _seasonal_profile_predict(_y0_hist, _pidx0[_mask0])
        _dash_series["SEASONAL PROFILE"] = [float(v) for v in _ysp]

    # -----------------------------
    # MODELS
    # -----------------------------
    for mf in model_files:
        print(f"   -> loading model: {mf.name}", flush=True)
        try:
            raw_name, bundle, fcols = _load_bundle(mf)
        except Exception as e:
            print(f"[WARN] could not load {mf.name}: {type(e).__name__}: {e}", flush=True)
            continue

        feature_cols = _select_feature_cols(saved_feature_cols=fcols, bundle=bundle, X_all=X_all, model_file=mf.name)
        if feature_cols is None:
            continue

        disp = _display_name(raw_name, mf.name)

        yt, yp = [], []
        failed = False
        for origin in origins:
            origin2 = origin
            if origin2 not in X_all.index:
                prev = X_all.index[X_all.index <= origin2]
                if len(prev) == 0:
                    continue
                origin2 = prev[-1]

            x_row = X_all.loc[origin2, feature_cols]
            if isinstance(x_row, pd.DataFrame):
                x_row = x_row.iloc[-1]

            try:
                yhat_vec = _predict_vector(bundle, x_row, feature_cols, H)
            except Exception as e:
                print(f"[WARN] eval failed for {mf.name}: {type(e).__name__}: {e}", flush=True)
                failed = True
                break

            pred_idx = pd.date_range(origin2 + freq, periods=H, freq=freq)

            # Store for dashboard chart (single_origin only — overwrite is fine)
            if args.eval_mode == "single_origin":
                _mask_d = (pred_idx >= test_start) & (pred_idx <= test_end)
                _vec_d  = np.full(H, np.nan, dtype=float)
                _vec_d[:min(len(yhat_vec), H)] = yhat_vec[:H]
                _dash_series[disp] = [float(v) for v in _vec_d[_mask_d]]

            a, b = _accumulate(pred_idx, yhat_vec)
            yt.extend(a)
            yp.extend(b)

        if failed:
            continue

        rows.append({"Model": disp, "Type": "ml", **_metrics(np.asarray(yt), np.asarray(yp))})

    # ------------------------------------------------------------------
    # ENSEMBLE (1/MAE weighted) across all successfully evaluated ML models
    # ------------------------------------------------------------------
    ml_rows = [r for r in rows if r.get("Type") == "ml" and math.isfinite(r["MAE"]) and r["MAE"] > 0]
    if len(ml_rows) >= 2:
        # Collect per-origin predictions for each ML model
        # Re-run predictions to gather arrays — store as {disp_name: predictions_array}
        ml_pred_store: Dict[str, np.ndarray] = {}

        for mf in model_files:
            try:
                raw_name2, bundle2, fcols2 = _load_bundle(mf)
            except Exception:
                continue
            feature_cols2 = _select_feature_cols(saved_feature_cols=fcols2, bundle=bundle2, X_all=X_all, model_file=mf.name)
            if feature_cols2 is None:
                continue
            disp2 = _display_name(raw_name2, mf.name)
            # only include models that made it through the first pass
            if not any(r["Model"] == disp2 for r in ml_rows):
                continue

            yt2: List[float] = []
            yp2: List[float] = []
            preds_list: List[float] = []
            trues_list: List[float] = []
            failed2 = False
            for origin in origins:
                origin2 = origin
                if origin2 not in X_all.index:
                    prev = X_all.index[X_all.index <= origin2]
                    if len(prev) == 0:
                        continue
                    origin2 = prev[-1]
                x_row2 = X_all.loc[origin2, feature_cols2]
                if isinstance(x_row2, pd.DataFrame):
                    x_row2 = x_row2.iloc[-1]
                try:
                    yhat2 = _predict_vector(bundle2, x_row2, feature_cols2, H)
                except Exception:
                    failed2 = True
                    break
                pred_idx2 = pd.date_range(origin2 + freq, periods=H, freq=freq)
                a2, b2 = _accumulate(pred_idx2, yhat2)
                trues_list.extend(a2)
                preds_list.extend(b2)

            if not failed2 and len(preds_list) > 0:
                ml_pred_store[disp2] = np.asarray(preds_list, dtype=float)
                # also keep trues (same for all models)
                _ens_true = np.asarray(trues_list, dtype=float)

        if len(ml_pred_store) >= 2:
            # 1/MAE weights
            maes_ens = {k: float(np.mean(np.abs(ml_pred_store[k] - _ens_true))) for k in ml_pred_store}
            weights_raw = {k: 1.0 / max(v, 1e-9) for k, v in maes_ens.items()}
            w_total = sum(weights_raw.values())
            weights = {k: v / w_total for k, v in weights_raw.items()}
            print(
                f"[INFO] Ensemble weights (1/MAE): { {k: round(v,3) for k,v in weights.items()} }",
                flush=True,
            )
            ens_pred = sum(weights[k] * ml_pred_store[k] for k in weights)
            rows.append({"Model": "Ensemble (1/MAE)", "Type": "ensemble", **_metrics(_ens_true, ens_pred)})

            # mean ensemble
            mean_pred = np.mean(np.stack(list(ml_pred_store.values()), axis=0), axis=0)
            rows.append({"Model": "Ensemble (mean)", "Type": "ensemble", **_metrics(_ens_true, mean_pred)})

    df = pd.DataFrame(rows).sort_values(by=args.sort_by, ascending=True)

    print(f"\n📊 FINAL EVALUATION (HOURLY) task={task.upper()} H={H}", flush=True)
    print(
        df[["Model", "Type", "MAE", "RMSE", "sMAPE", "WAPE", "N"]].to_string(
            index=False,
            formatters={
                "MAE": lambda x: f"{x:8.3f}" if math.isfinite(x) else "    nan",
                "RMSE": lambda x: f"{x:8.3f}" if math.isfinite(x) else "    nan",
                "sMAPE": lambda x: f"{x:8.3f}" if math.isfinite(x) else "    nan",
                "WAPE": lambda x: f"{x:8.3f}" if math.isfinite(x) else "    nan",
                "N": lambda x: f"{int(x):6d}",
            },
        ),
        flush=True,
    )

    if args.save_json:
        _metrics_list = df[["Model", "Type", "MAE", "RMSE", "sMAPE"]].to_dict(orient="records")
        out = {
            "mode": "hourly",
            "task": task,
            "strategy": "mimo",
            "horizon": H,
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
            "eval_mode": args.eval_mode,
            "origin_stride": args.origin_stride,
            "results":  df.to_dict(orient="records"),
            "metrics":  _metrics_list,
        }
        # Add time-series payload for dashboard chart
        if args.eval_mode == "single_origin" and _dash_dates:
            out["dates"]  = _dash_dates
            out["actual"] = _dash_actual
            out["series"] = _dash_series
        # NaN → null for valid JSON
        def _fix_nan(obj):
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            if isinstance(obj, dict):
                return {k: _fix_nan(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_fix_nan(x) for x in obj]
            return obj
        out_path = ROOT / f"dashboard_data_hourly_{task}_mimo_h{H}.json"
        out_path.write_text(json.dumps(_fix_nan(out), indent=2), encoding="utf-8")
        print(f"\n✅ Saved: {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
