from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .feature_availability import (
    apply_crosslag_freeze_row,
    build_frozen_lookup,
    detect_crosslag_cols,
)


@dataclass
class OpenLoopConfig:
    # keep ONLY what we actually want
    y_floor: Optional[float] = None


def _to_float_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _is_torch_bundle(obj) -> bool:
    return isinstance(obj, dict) and ("state_dict" in obj or "torch" in str(obj.get("type", "")).lower())


def _strip_prefix_from_state_dict(sd: Dict[str, "np.ndarray"], prefix: str) -> Dict[str, "np.ndarray"]:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _infer_layer_order_from_keys(keys: Iterable[str]) -> List[int]:
    # expects keys like "0.weight", "3.weight" OR "net.0.weight"
    idxs = []
    for k in keys:
        kk = k.replace("net.", "")
        parts = kk.split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1] == "weight":
            idxs.append(int(parts[0]))
    return sorted(set(idxs))


def _torch_predictor_from_bundle(bundle: dict) -> Tuple[Callable[[pd.Series, List[str]], float], List[str]]:
    """
    Returns:
      predict_row(row_series, feature_cols) -> float
      feature_cols_used
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError(f"Torch not available but torch bundle was loaded: {e}")

    state_dict = bundle.get("state_dict")
    if state_dict is None:
        raise RuntimeError("Torch bundle missing 'state_dict'.")

    # feature columns: prefer bundle's, else caller-provided at predict time
    bundle_feature_cols = bundle.get("feature_cols", None)

    x_scaler = bundle.get("x_scaler", None)
    y_scaler = bundle.get("y_scaler", None)

    # infer dims from weights
    sd_keys = list(state_dict.keys())
    layer_idxs = _infer_layer_order_from_keys(sd_keys)
    if not layer_idxs:
        raise RuntimeError(f"Could not infer layers from state_dict keys: {sd_keys[:10]}")

    # state_dict may be "net.0.weight" etc. Use "net." stripped for shape inference.
    sd_no_net = _strip_prefix_from_state_dict(state_dict, "net.")
    first_w_key = f"{layer_idxs[0]}.weight"
    w0 = sd_no_net.get(first_w_key, None)
    if w0 is None:
        # fallback: search any weight
        weight_keys = [k for k in sd_no_net.keys() if k.endswith(".weight")]
        if not weight_keys:
            raise RuntimeError("Could not locate any *.weight keys in state_dict.")
        w0 = sd_no_net[weight_keys[0]]
    in_dim = int(w0.shape[1])

    # infer hidden dims and output dim
    last_w_key = f"{layer_idxs[-1]}.weight"
    w_last = sd_no_net.get(last_w_key, None)
    if w_last is None:
        weight_keys = [k for k in sd_no_net.keys() if k.endswith(".weight")]
        w_last = sd_no_net[weight_keys[-1]]
    out_dim = int(w_last.shape[0])

    # collect dims from weights
    dims = []
    for li in layer_idxs:
        wk = f"{li}.weight"
        if wk in sd_no_net:
            dims.append(int(sd_no_net[wk].shape[0]))
    hidden_dims = dims[:-1]

    class SimpleMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    net = SimpleMLP(in_dim=in_dim, hidden=hidden_dims, out_dim=out_dim)

    try:
        net.load_state_dict(state_dict, strict=False)
    except Exception:
        net.load_state_dict(sd_no_net, strict=False)

    net.eval()

    def predict_row(row: pd.Series, cols: List[str]) -> float:
        x = row.reindex(cols).astype(float).to_numpy().reshape(1, -1)
        if x.shape[1] != in_dim:
            raise RuntimeError(f"Torch bundle expects {in_dim} features, got {x.shape[1]}.")
        if x_scaler is not None:
            x = x_scaler.transform(x)
        xt = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            y = net(xt).cpu().numpy().reshape(-1)
        if y_scaler is not None:
            y = y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        return float(y[0])

    return predict_row, (bundle_feature_cols if bundle_feature_cols else [])


def recursive_predict_openloop(
    model,
    df_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    feature_cols: List[str],
    config: Optional[OpenLoopConfig] = None,
    aux_models: Optional[Dict[str, object]] = None,
    gate: Optional[object] = None,
    crosslag_mode: str = "freeze",
    crosslag_cutoff: Optional[pd.Timestamp] = None,
):
    """
    STRICT open-loop recursive prediction:
    - builds each test row from df_full features
    - overwrites:
        (a) any y-lag columns found in feature_cols using a running series y_run
        (b) any y-roll columns y_rollW / y_roll_W as MEAN of y_run[t-1 ... t-W]
      so there is NO leakage from actual test y inside the horizon.
    - if `gate` (feature_availability.GateSpec) is given: freezes/NaNs any
      gen_solar/gen_wind/residual_load/load lag ("crosslag") columns that
      reference a timestamp AFTER that family's availability cutoff for this
      anchor — SYSTEM_DESIGN §4.8 (AEL). Χωρίς gate, συμπεριφορά αμετάβλητη
      (backward-compatible: παλιά callers/poisoning tests συνεχίζουν να δουλεύουν).
    - crosslag_cutoff: το ΣΤΑΘΕΡΟ cutoff του block anchor (gate.
      crosslag_cutoff_for_anchor(block_start)) — αυτό είναι το σωστό στο eval:
      σε multi-day blocks (forward 168h) ΟΛΕΣ οι ώρες μοιράζονται το ίδιο cutoff
      (τίποτα μετά το issue time δεν είναι γνωστό). Αν None, fallback στο per-row
      day-of-t σχήμα (crosslag_cutoff_index) — σωστό μόνο για single-day blocks.

    aux_models: προαιρετικό dict {name: model} — προβλέπουν στην ΙΔΙΑ γραμμή
    (ίδιο row) με το κύριο model, χωρίς να οδηγούν τη recursive ανατροφοδότηση
    (μόνο το κύριο model γράφει στο y_run). Χρήση: quantile-LGBM p10/p90 side
    predictions ενώ το p50 model οδηγεί το rollout (βλ. src/conformal.py).
    Αν δοθεί aux_models, επιστρέφεται tuple (preds, aux_preds); αλλιώς μόνο preds
    (παλιά συμπεριφορά, αμετάβλητη).
    """
    cfg = config or OpenLoopConfig()

    # running y series: start from actual history, then overwrite test timestamps with predictions
    y_run = df_full["y"].astype(float).copy()

    # AEL (§4.8): ανίχνευση crosslag στηλών (gen_*/residual_load/load lags) + frozen
    # lookup, ΜΙΑ φορά πριν το rollout. Χωρίς gate, crosslag_cols μένει κενό ⇒ no-op
    # (backward-compatible με παλιά callers/poisoning tests).
    crosslag_cols: Dict[str, Dict[str, int]] = {}
    frozen_lookup: Dict[str, pd.Series] = {}
    crosslag_cutoffs: Optional[pd.DatetimeIndex] = None
    crosslag_protect_cols: set = set()
    if gate is not None:
        crosslag_cols = detect_crosslag_cols(feature_cols)
        if crosslag_cols:
            frozen_lookup = build_frozen_lookup(df_full, crosslag_cols)
            if crosslag_cutoff is not None:
                # anchor-based (eval): ένα cutoff για όλο το rollout του block
                crosslag_cutoffs = pd.DatetimeIndex([crosslag_cutoff] * len(test_index))
            else:
                crosslag_cutoffs = gate.crosslag_cutoff_index(test_index)
            for colmap in crosslag_cols.values():
                crosslag_protect_cols.update(colmap.keys())

    import re

    # detect lag feature columns
    lag_map: Dict[str, int] = {}
    for c in feature_cols:
        cl = c.lower()
        # accept patterns like y_lag1, y_lag_24, lag_y_168, etc.
        m = re.search(r"(?:^|_)y_?lag_?(\d+)$", cl)
        if m:
            lag_map[c] = int(m.group(1))
            continue
        m = re.search(r"(?:^|_)lag_?y_?(\d+)$", cl)
        if m:
            lag_map[c] = int(m.group(1))
            continue

    # detect rolling mean columns y_roll24, y_roll_168, etc.
    roll_map: Dict[str, int] = {}
    for c in feature_cols:
        cl = c.lower()
        m = re.search(r"(?:^|_)y_?roll_?(\d+)$", cl)
        if m:
            roll_map[c] = int(m.group(1))

    # torch predictor if needed
    torch_predict = None
    torch_bundle_cols: List[str] = []
    if _is_torch_bundle(model):
        torch_predict, torch_bundle_cols = _torch_predictor_from_bundle(model)

    preds = np.zeros(len(test_index), dtype=float)
    aux_preds: Dict[str, np.ndarray] = {name: np.zeros(len(test_index), dtype=float)
                                        for name in (aux_models or {})}

    for i, t in enumerate(test_index):
        # base row from df_full at time t
        if t not in df_full.index:
            row = pd.Series(index=feature_cols, dtype=float)
        else:
            row = df_full.loc[t, feature_cols].copy()

        # AEL (§4.8): freeze/NaN crosslag cols (gen/load actuals) not yet
        # available at this anchor's cutoff — BEFORE y-lag overwrite (disjoint
        # column sets, order does not matter).
        if crosslag_cols:
            cutoff_t = crosslag_cutoffs[i]
            row = apply_crosslag_freeze_row(
                row, t, crosslag_cols, frozen_lookup, cutoff_t, mode=crosslag_mode,
            )

        # overwrite lag features from running y
        for col, lag in lag_map.items():
            t_lag = t - pd.Timedelta(hours=int(lag))
            v = y_run.reindex([t_lag]).iloc[0] if t_lag in y_run.index else np.nan
            row[col] = float(v) if np.isfinite(v) else np.nan

        # overwrite rolling mean features from running y
        # y_rollW at time t corresponds to mean of y(t-1),...,y(t-W)
        for col, win in roll_map.items():
            start = t - pd.Timedelta(hours=int(win))
            end = t - pd.Timedelta(hours=1)
            if end < start:
                row[col] = np.nan
            else:
                vals = y_run.loc[start:end]
                mv = float(np.nanmean(vals.to_numpy(dtype=float))) if len(vals) > 0 else np.nan
                row[col] = mv if np.isfinite(mv) else np.nan

        # fill NaNs (keep exogenous NaNs as 0 to avoid model crashes) — EXCEPT
        # crosslag cols under crosslag_mode='nan' (sensitivity variant): trees
        # must see a REAL NaN to exercise native missing-value handling, so we
        # don't launder it into 0.0 here (§4.8 NaN-variant).
        row = row.astype(float)
        if crosslag_mode == "nan" and crosslag_protect_cols:
            fillable = ~row.index.isin(crosslag_protect_cols)
            row[fillable] = row[fillable].where(np.isfinite(row[fillable]), 0.0)
        else:
            row = row.where(np.isfinite(row), 0.0)

        # predict
        if torch_predict is not None:
            yhat = torch_predict(row, feature_cols if not torch_bundle_cols else torch_bundle_cols)
        else:
            X = pd.DataFrame([row.values], columns=feature_cols)
            yhat = float(_to_float_np(model.predict(X)).reshape(-1)[0])
            for name, aux_model in (aux_models or {}).items():
                aux_preds[name][i] = float(_to_float_np(aux_model.predict(X)).reshape(-1)[0])

        if cfg.y_floor is not None:
            yhat = max(yhat, float(cfg.y_floor))

        preds[i] = yhat
        # overwrite running series at time t with prediction (no leakage forward)
        y_run.loc[t] = yhat

    if aux_models:
        return preds, aux_preds
    return preds
