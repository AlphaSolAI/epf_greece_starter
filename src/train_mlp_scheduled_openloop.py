"""
Scheduled Sampling (Recursive-Aware) Training for MLP Open-Loop.

The existing MLP collapses in open-loop (MAE ≈ 29-30 vs LGBM-SS ≈ 17)
because it was trained with teacher-forced true lag values but evaluated
with its own predicted lags. This script fixes that via Scheduled Sampling.

Algorithm
---------
1. Load the pre-trained base MLP (teacher-forced) from mlp_{mode}_{task}_openloop.pkl
2. For each SS round i (epsilon increasing from eps_start to eps_final):
   a. Batch-predict on the full training set (in-sample predictions)
   b. Replace y_lag features with pseudo-predictions at probability epsilon
   c. Retrain MLP on augmented data from scratch (full training loop)
3. Save final model as mlp_{mode}_{task}_scheduled_openloop.pkl

Usage
-----
python -m src.train_mlp_scheduled_openloop hourly --task price \\
    --train_end "2025-11-30 23:00" --n_iter 2 --epsilon_final 0.5
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

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


# -----------------------------------------------------------------------
# MLP architecture (must match train_mlp_openloop.py)
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# Batch predict from MLP bundle dict
# -----------------------------------------------------------------------

def _mlp_batch_predict(bundle: dict, X_df: pd.DataFrame) -> np.ndarray:
    """
    Run batch inference on an MLP bundle dict.
    Returns array of predictions (original scale).
    """
    feature_cols = bundle["feature_cols"]
    x_scaler = bundle["x_scaler"]
    y_scaler = bundle["y_scaler"]
    state_dict = bundle["state_dict"]
    hidden = tuple(bundle["hidden"])
    dropout = float(bundle.get("dropout", 0.0))

    # Align features
    X = X_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    Xs = x_scaler.transform(X.to_numpy(dtype=float)).astype(np.float32)

    # Build model
    model = MLPNet(input_dim=len(feature_cols), hidden=hidden, dropout=dropout)
    # State dict may have "net." prefix
    sd = state_dict
    if not any(k.startswith("net.") for k in sd.keys()):
        sd = {f"net.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    with torch.no_grad():
        Xt = torch.from_numpy(Xs)
        pred_s = model(Xt).numpy().reshape(-1)

    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred.astype(float)


# -----------------------------------------------------------------------
# Helpers (shared with train_scheduled_openloop.py)
# -----------------------------------------------------------------------

def _get_y_lag_cols(X: pd.DataFrame) -> List[str]:
    """Return only direct y_lagN columns (not rolling means, not load_lag*)."""
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _scheduled_sampling_round_mlp(
    bundle: dict,
    X_train: pd.DataFrame,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build augmented X for one SS round using the MLP bundle.
    Replaces y_lag features with bundle predictions at probability epsilon.
    """
    y_hat = pd.Series(
        _mlp_batch_predict(bundle, X_train),
        index=X_train.index,
        dtype=float,
    )
    X_aug = X_train.copy()
    y_lag_cols = _get_y_lag_cols(X_train)

    mask = rng.random(len(X_train)) < epsilon
    n_replaced = 0
    for col in y_lag_cols:
        lag_h = int(col.replace("y_lag", ""))
        pseudo_lag = y_hat.shift(lag_h).reindex(X_train.index)
        valid = mask & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values
        n_replaced += int(valid.sum())

    total = len(y_lag_cols) * len(X_train)
    print(
        f"   [SS-MLP] ε={epsilon:.2f} | replacements: "
        f"{n_replaced}/{total} ({100 * n_replaced / total:.1f}%)"
    )
    return X_aug


# -----------------------------------------------------------------------
# MLP training loop (reusable)
# -----------------------------------------------------------------------

def _train_mlp_on_data(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    feature_cols: List[str],
    hidden: Tuple[int, ...],
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    torch_device: torch.device,
    x_scaler=None,
    y_scaler=None,
    verbose: bool = True,
    seed: int = 42,
) -> dict:
    """
    Train MLP on (X_df, y_series) and return a bundle dict.
    If x_scaler/y_scaler are provided (from base model), reuse them.
    Otherwise fit new scalers.
    """
    from sklearn.preprocessing import StandardScaler

    X = X_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = pd.Series(y_series).astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    if x_scaler is None:
        x_scaler = StandardScaler()
        Xs = x_scaler.fit_transform(X.to_numpy(dtype=float)).astype(np.float32)
    else:
        Xs = x_scaler.transform(X.to_numpy(dtype=float)).astype(np.float32)

    if y_scaler is None:
        y_scaler = StandardScaler()
        ys = y_scaler.fit_transform(y.to_numpy(dtype=float).reshape(-1, 1)).reshape(-1).astype(np.float32)
    else:
        ys = y_scaler.transform(y.to_numpy(dtype=float).reshape(-1, 1)).reshape(-1).astype(np.float32)

    # Chronological 90/10 split for early stopping
    n = len(Xs)
    n_val = max(168, int(0.08 * n))
    n_tr = n - n_val
    X_tr, y_tr = Xs[:n_tr], ys[:n_tr]
    X_va, y_va = Xs[n_tr:], ys[n_tr:]

    model = MLPNet(input_dim=len(feature_cols), hidden=hidden, dropout=dropout).to(torch_device)
    criterion = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.tensor(X_tr, device=torch_device)
    y_tr_t = torch.tensor(y_tr.reshape(-1, 1), device=torch_device)
    X_va_t = torch.tensor(X_va, device=torch_device)
    y_va_t = torch.tensor(y_va.reshape(-1, 1), device=torch_device)

    rng = np.random.default_rng(seed)
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        idx = np.arange(n_tr)
        rng.shuffle(idx)
        tr_losses = []
        for s in range(0, n_tr, batch_size):
            b = idx[s: s + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(X_tr_t[b])
            loss = criterion(pred, y_tr_t[b])
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            va_loss = criterion(model(X_va_t), y_va_t).item()

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"     epoch={epoch:03d} train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")

        if va_loss < best_val - min_delta:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"     [early stop] epoch={epoch} best_val={best_val:.6f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "type": "mlp_torch_openloop",
        "state_dict": best_state,
        "feature_cols": feature_cols,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "hidden": hidden,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
    }


# -----------------------------------------------------------------------
# Main SS training
# -----------------------------------------------------------------------

def train(
    mode: str,
    task: str = "price",
    train_end: Optional[str] = None,
    n_iter: int = 2,
    epsilon_start: float = 0.20,
    epsilon_final: float = 0.50,
    device: str = "auto",
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    torch_device = _pick_device(device)

    # Load data
    df = load_processed(mode, task=task)
    if train_end:
        df = df.loc[: pd.to_datetime(train_end)]
    X_train, y_train = make_xy(df)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_train = y_train.astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    feature_cols = list(X_train.columns)

    print(f"\n🔄 TRAINING MLP SCHEDULED SAMPLING OPENLOOP ({mode.upper()} | task={task.upper()})")
    print(f"   -> train={len(df)} rows | features={len(feature_cols)}")
    print(f"   -> n_iter={n_iter} | ε: {epsilon_start:.2f} → {epsilon_final:.2f} | device={torch_device}")

    # ---- Load base model (teacher-forced)
    base_path = MODELS_DIR / f"mlp_{mode}_{task}_openloop.pkl"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base MLP not found at {base_path}. "
            f"Train it first: python -m src.train_mlp_openloop {mode} --task {task} --train_end ..."
        )
    bundle = joblib.load(base_path)
    print(f"   -> Loaded base MLP from: {base_path.name}")

    # Reuse scalers from base model (ensures consistent feature scaling across SS rounds)
    x_scaler = bundle["x_scaler"]
    y_scaler = bundle["y_scaler"]
    hidden = tuple(bundle["hidden"])
    dropout = float(bundle.get("dropout", 0.10))
    epochs = int(bundle.get("epochs", 240))
    batch_size = int(bundle.get("batch_size", 1024))
    lr = float(bundle.get("lr", 1e-3))
    weight_decay = float(bundle.get("weight_decay", 1e-4))

    # ---- SS rounds
    epsilons = np.linspace(epsilon_start, epsilon_final, n_iter) if n_iter > 1 else [epsilon_final]

    current_bundle = bundle
    for i, eps in enumerate(epsilons, start=1):
        print(f"\n🔁 [Iter {i}/{n_iter}] Scheduled sampling (ε={eps:.3f}) …")
        X_aug = _scheduled_sampling_round_mlp(current_bundle, X_train, eps, rng)

        print(f"   Retraining MLP on augmented data …")
        current_bundle = _train_mlp_on_data(
            X_df=X_aug,
            y_series=y_train,
            feature_cols=feature_cols,
            hidden=hidden,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=30,
            min_delta=1e-5,
            torch_device=torch_device,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            verbose=True,
            seed=seed + i,
        )
        print(f"   ✅ Iteration {i} complete.")

    # ---- Save
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / f"mlp_{mode}_{task}_scheduled_openloop.pkl"
    joblib.dump(current_bundle, out_path)
    print(f"\n✅ Saved MLP-SS to: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Scheduled Sampling for MLP Open-Loop")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--n_iter", type=int, default=2)
    p.add_argument("--epsilon_start", type=float, default=0.20)
    p.add_argument("--epsilon_final", type=float, default=0.50)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(
        mode=args.mode,
        task=args.task,
        train_end=args.train_end,
        n_iter=args.n_iter,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
