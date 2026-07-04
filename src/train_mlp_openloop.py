import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from .split_utils import load_processed, make_xy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# -----------------------------
# Torch MLP
# -----------------------------
class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _pick_device(device: str) -> torch.device:
    d = (device or "auto").lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainCfg:
    # Hardcoded BIGGER model (4 hidden layers)
    hidden: Tuple[int, ...] = (1024, 512, 256, 128)
    dropout: float = 0.10

    # ~4x epochs vs old small setup
    epochs: int = 240
    batch_size: int = 1024

    lr: float = 1e-3
    weight_decay: float = 1e-4

    patience: int = 30
    min_delta: float = 1e-5


def select_train_window(df: pd.DataFrame, train_start: Optional[str], train_end: Optional[str]) -> pd.DataFrame:
    out = df
    if train_start:
        out = out.loc[pd.to_datetime(train_start) :]
    if train_end:
        out = out.loc[: pd.to_datetime(train_end)]
    return out


def train(
    mode: str,
    task: str = "price",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    device: str = "auto",
) -> None:
    if mode != "hourly":
        raise ValueError("Only mode='hourly' is supported πλέον (daily removed).")

    df = load_processed(mode, task=task)
    df_train = select_train_window(df, train_start=train_start, train_end=train_end)

    X_train, y_train = make_xy(df_train)

    # Safety: no NaNs for torch
    X_train = X_train.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_train = pd.Series(y_train).astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    feat_cols = list(X_train.columns)

    # Scale X and y for stable training
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xs = x_scaler.fit_transform(X_train.to_numpy(dtype=float))
    ys = y_scaler.fit_transform(y_train.to_numpy(dtype=float).reshape(-1, 1)).reshape(-1)

    # internal validation: chronological tail of train (10%)
    n = len(Xs)
    n_val = max(1, int(0.10 * n))
    n_tr = n - n_val

    X_tr, y_tr = Xs[:n_tr], ys[:n_tr]
    X_va, y_va = Xs[n_tr:], ys[n_tr:]

    cfg = TrainCfg()
    torch_device = _pick_device(device)

    print(f"🧠 TRAINING MLP OPENLOOP (TORCH) ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> train={len(df_train)} | features={X_train.shape[1]} | device={torch_device}")
    if train_start or train_end:
        print(f"   -> train_window={df_train.index.min()} -> {df_train.index.max()}")
    print(f"   -> internal_val={n_val} (chronological tail of train)")
    print(f"   -> cfg: hidden={cfg.hidden} dropout={cfg.dropout} epochs={cfg.epochs} bs={cfg.batch_size}")

    model = MLPNet(input_dim=X_train.shape[1], hidden=cfg.hidden, dropout=cfg.dropout).to(torch_device)
    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=torch_device)
    y_tr_t = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32, device=torch_device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=torch_device)
    y_va_t = torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32, device=torch_device)

    best_val = float("inf")
    best_state = None
    bad = 0

    # mini-batch indices
    rng = np.random.default_rng(42)

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        idx = np.arange(n_tr)
        rng.shuffle(idx)

        # batch loop
        tr_losses = []
        for s in range(0, n_tr, cfg.batch_size):
            b = idx[s : s + cfg.batch_size]
            xb = X_tr_t[b]
            yb = y_tr_t[b]

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            va_pred = model(X_va_t)
            va_loss = criterion(va_pred, y_va_t).item()

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        if epoch == 1 or epoch % 10 == 0:
            print(f"[MLP-OPENLOOP] epoch={epoch:03d} train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")

        if va_loss < best_val - cfg.min_delta:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[MLP-OPENLOOP] early stop @ epoch={epoch:03d} best_val_mse={best_val:.6f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    save_obj = {
        "type": "mlp_torch_openloop",
        "state_dict": best_state,
        "feature_cols": feat_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "hidden": cfg.hidden,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"mlp_{mode}_{task}_openloop.pkl"
    joblib.dump(save_obj, model_path)
    print(f"✅ Saved MLP OPENLOOP (TORCH) to: {model_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    train(
        mode=args.mode,
        task=args.task,
        train_start=args.train_start,
        train_end=args.train_end,
        device=args.device,
    )


if __name__ == "__main__":
    main()
