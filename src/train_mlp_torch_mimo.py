# =========================
# FILE: src/train_mlp_torch_mimo.py
# =========================
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .split_utils import load_processed, split_time_series

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

DEFAULT_FUTURE_CAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday"]


def _set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_hidden_layers(s: str) -> Tuple[int, ...]:
    xs = [x.strip() for x in (s or "").split(",") if x.strip()]
    return tuple(int(x) for x in xs) if xs else (256, 128)


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_mimo_xy(
    df_train: pd.DataFrame,
    horizon: int,
    use_future_calendar: bool,
    future_calendar_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], List[str]]:
    if "y" not in df_train.columns:
        raise ValueError("Training dataframe must contain target column 'y'.")
    df_train = df_train.sort_index()

    past_cols = [c for c in df_train.columns if c != "y"]
    X_past = df_train[past_cols].to_numpy(dtype=float)
    y = df_train["y"].to_numpy(dtype=float)

    n = len(df_train)
    H = int(horizon)
    n_samples = n - H
    if n_samples <= 0:
        raise ValueError(f"Not enough training rows for horizon={H}. train_rows={n}.")

    Y = np.stack([y[1 + k : 1 + k + n_samples] for k in range(H)], axis=1)
    X0 = X_past[:n_samples, :]

    meta: Dict[str, object] = {
        "past_feature_cols": past_cols,
        "use_future_calendar": bool(use_future_calendar),
        "future_calendar_cols": [],
        "horizon": int(H),
    }

    if not use_future_calendar:
        return X0, Y, meta, past_cols

    cal_cols = [c for c in future_calendar_cols if c in df_train.columns]
    if not cal_cols:
        meta["use_future_calendar"] = False
        return X0, Y, meta, past_cols

    cal = df_train[cal_cols].to_numpy(dtype=float)
    X_cal = np.concatenate([cal[s : s + n_samples, :] for s in range(1, H + 1)], axis=1)
    X = np.concatenate([X0, X_cal], axis=1)

    meta["future_calendar_cols"] = cal_cols

    feature_cols = past_cols + [f"{c}_t+{k}" for k in range(1, H + 1) for c in cal_cols]
    return X, Y, meta, feature_cols


def _fit_y_transform(Y: np.ndarray, target_transform: str, asinh_scale: Optional[float]) -> Tuple[np.ndarray, Dict[str, object]]:
    tt = (target_transform or "none").lower()
    if tt == "none":
        return Y, {"name": "none", "scale": None}

    if tt != "asinh":
        raise ValueError("Unsupported --target_transform. Use 'none' or 'asinh'.")

    if asinh_scale is None or asinh_scale <= 0:
        med = float(np.median(np.abs(Y)))
        scale = med if med > 1e-6 else float(np.std(Y) + 1e-6)
        if scale <= 0:
            scale = 1.0
    else:
        scale = float(asinh_scale)

    Yt = np.arcsinh(Y / scale)
    return Yt, {"name": "asinh", "scale": scale}


class MLPHead(nn.Module):
    def __init__(self, n_in: int, hidden: Tuple[int, ...], n_out: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPMIMORegressor:
    def __init__(
        self,
        feature_cols: List[str],
        x_scaler: StandardScaler,
        hidden: Tuple[int, ...],
        horizon: int,
        target_transform: Dict[str, object],
        scale_y: bool,
        y_scaler: Optional[StandardScaler],
        seed: int = 42,
    ):
        self.feature_cols = list(feature_cols)
        self.x_scaler = x_scaler
        self.hidden = tuple(hidden)
        self.horizon = int(horizon)
        self.target_transform = dict(target_transform)
        self.scale_y = bool(scale_y)
        self.y_scaler = y_scaler
        self.seed = int(seed)
        self.state = None  # state_dict

    def _build(self, n_in: int) -> MLPHead:
        _set_seed(self.seed)
        return MLPHead(n_in=n_in, hidden=self.hidden, n_out=self.horizon)

    def fit(
        self,
        Xtr: np.ndarray,
        Ytr: np.ndarray,
        Xva: np.ndarray,
        Yva: np.ndarray,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        max_epochs: int = 200,
        patience: int = 20,
    ) -> None:
        dev = _device()
        model = self._build(Xtr.shape[1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
        Xva_t = torch.tensor(Xva, dtype=torch.float32).to(dev)
        Yva_t = torch.tensor(Yva, dtype=torch.float32).to(dev)

        n = Xtr_t.shape[0]
        best = float("inf")
        best_state = None
        bad = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            idx = torch.randperm(n)

            for s in range(0, n, batch_size):
                j = idx[s : s + batch_size]
                xb = Xtr_t[j].to(dev)
                yb = Ytr_t[j].to(dev)

                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vp = model(Xva_t)
                vl = loss_fn(vp, Yva_t).item()

            if vl < best - 1e-8:
                best = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.state = best_state

    def predict(self, X):
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            Xnp = X[self.feature_cols].to_numpy(dtype=float)
        else:
            Xnp = np.asarray(X, dtype=float)

        Xnp = self.x_scaler.transform(Xnp).astype(np.float32, copy=False)

        dev = _device()
        model = self._build(Xnp.shape[1]).to(dev)
        model.load_state_dict({k: v.to(torch.float32) for k, v in self.state.items()})
        model.eval()

        with torch.no_grad():
            xt = torch.tensor(Xnp, dtype=torch.float32).to(dev)
            Yp = model(xt).detach().cpu().numpy()

        # inverse y scaling
        if self.scale_y and self.y_scaler is not None:
            Yp = self.y_scaler.inverse_transform(Yp)

        # inverse target transform
        if self.target_transform.get("name") == "asinh":
            scale = float(self.target_transform.get("scale") or 1.0)
            Yp = np.sinh(Yp) * scale

        return Yp


def train_mlp_torch_mimo(
    mode: str,
    horizon: int,
    test_size: Optional[int],
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    hidden_layers: str,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    use_future_calendar: bool,
    target_transform: str,
    asinh_scale: Optional[float],
    scale_y: bool,
) -> None:
    if mode != "hourly":
        raise ValueError("Only hourly supported")

    df = load_processed(mode)
    df_train, df_test = split_time_series(
        df,
        mode=mode,
        test_size=test_size,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    print(f"🧠 TRAINING MLP-TORCH-MIMO ({mode.upper()}) [SINGLE] [NO-LEAK] [GPU if available]")
    print(
        f"[INFO] train={len(df_train)} | test={len(df_test)} | horizon={int(horizon)} | "
        f"future_calendar={use_future_calendar} | target_transform={target_transform} | scale_y={scale_y}"
    )
    print(f"[INFO] torch={torch.__version__} | cuda_available={torch.cuda.is_available()} | device={_device()}")

    X, Y, meta, feature_cols = _build_mimo_xy(df_train, horizon, use_future_calendar, DEFAULT_FUTURE_CAL_COLS)
    print(f"[INFO] X.shape={X.shape} | Y.shape={Y.shape}")

    Yt, tt_params = _fit_y_transform(Y, target_transform, asinh_scale)

    y_scaler = None
    if scale_y:
        y_scaler = StandardScaler()
        Yt = y_scaler.fit_transform(Yt)

    # scale X
    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    # chronological val split (last 10%)
    n = len(Xs)
    n_val = max(1, int(0.1 * n))
    n_tr = n - n_val
    Xtr, Ytr = Xs[:n_tr], Yt[:n_tr]
    Xva, Yva = Xs[n_tr:], Yt[n_tr:]

    hidden = _parse_hidden_layers(hidden_layers)

    reg = TorchMLPMIMORegressor(
        feature_cols=feature_cols,
        x_scaler=x_scaler,
        hidden=hidden,
        horizon=int(horizon),
        target_transform=tt_params,
        scale_y=bool(scale_y),
        y_scaler=y_scaler,
        seed=int(seed),
    )

    reg.fit(
        Xtr.astype(np.float32, copy=False),
        Ytr.astype(np.float32, copy=False),
        Xva.astype(np.float32, copy=False),
        Yva.astype(np.float32, copy=False),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        patience=20,
    )

    bundle: Dict[str, object] = {
        "kind": "mimo_bundle",
        "model_id": "mlp_torch",
        "mode": mode,
        "mimo_mode": "single",
        "horizon": int(horizon),
        "meta": meta,
        "model": reg,
        "target_transform": tt_params.get("name", "none"),
        "target_transform_params": {"scale": tt_params.get("scale", None)},
        "scale_y": bool(scale_y),
        "y_scaler": y_scaler,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"mlp_torch_mimo_{mode}_single_h{int(horizon)}.pkl"
    joblib.dump(bundle, out_path)
    print(f"✅ Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("mimo_mode", choices=["single"])
    p.add_argument("--horizon", type=int, required=True)

    p.add_argument("--test_size", type=int, default=None)
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--test_start", type=str, default=None)
    p.add_argument("--test_end", type=str, default=None)

    p.add_argument("--hidden_layers", type=str, default="256,128")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no_future_calendar", action="store_true")
    p.add_argument("--target_transform", choices=["none", "asinh"], default="asinh")
    p.add_argument("--asinh_scale", type=float, default=None)
    p.add_argument("--scale_y", action="store_true")

    args = p.parse_args()

    train_mlp_torch_mimo(
        mode=args.mode,
        horizon=int(args.horizon),
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        hidden_layers=args.hidden_layers,
        max_epochs=int(args.max_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        use_future_calendar=(not args.no_future_calendar),
        target_transform=str(args.target_transform),
        asinh_scale=args.asinh_scale,
        scale_y=bool(args.scale_y),
    )


if __name__ == "__main__":
    main()