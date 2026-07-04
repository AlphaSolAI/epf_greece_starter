import argparse
import copy
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .split_utils import load_processed, make_xy, split_time_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TorchMLPRegressor:
    """
    scikit-like wrapper γύρω από PyTorch MLP.
    Αποθηκεύεται με joblib και έχει .predict(X) για συμβατότητα με eval.py
    """
    hidden: Tuple[int, ...] = (256, 256)
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 200
    patience: int = 20
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # learned
    feature_names_: Optional[list] = None
    x_scaler_: Optional[StandardScaler] = None
    y_scaler_: Optional[StandardScaler] = None
    model_: Optional[nn.Module] = None
    device_: str = "cpu"

    def _pick_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y, X_val=None, y_val=None, verbose: bool = True, warmstart_state: Optional[dict] = None) -> "TorchMLPRegressor":
        _set_seed(self.seed)

        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.x_scaler_ = StandardScaler()
        Xs = self.x_scaler_.fit_transform(X_np).astype(np.float32)

        self.y_scaler_ = StandardScaler()
        ys = self.y_scaler_.fit_transform(y_np).astype(np.float32).reshape(-1)

        if X_val is not None and y_val is not None:
            Xv_np = np.asarray(X_val, dtype=np.float32)
            yv_np = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            Xv = self.x_scaler_.transform(Xv_np).astype(np.float32)
            yv = self.y_scaler_.transform(yv_np).astype(np.float32).reshape(-1)
        else:
            Xv, yv = None, None

        self.device_ = self._pick_device()
        self.model_ = _MLPNet(input_dim=Xs.shape[1], hidden=self.hidden, dropout=self.dropout).to(self.device_)
        if warmstart_state is not None:
            self.model_.load_state_dict(copy.deepcopy(warmstart_state))

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if Xv is not None:
            vds = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
            vdl = DataLoader(vds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        else:
            vdl = None

        best_val = float("inf")
        best_state = None
        bad = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            tr_losses = []

            for xb, yb in dl:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                opt.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            tr = float(np.mean(tr_losses)) if tr_losses else float("nan")

            if vdl is None:
                if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == self.max_epochs):
                    print(f"[MLP-TORCH] epoch={epoch:03d} train_mse={tr:.6f}")
                continue

            self.model_.eval()
            va_losses = []
            with torch.no_grad():
                for xb, yb in vdl:
                    xb = xb.to(self.device_)
                    yb = yb.to(self.device_)
                    pred = self.model_(xb)
                    loss = loss_fn(pred, yb)
                    va_losses.append(loss.item())
            va = float(np.mean(va_losses)) if va_losses else float("nan")

            if verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"[MLP-TORCH] epoch={epoch:03d} train_mse={tr:.6f} val_mse={va:.6f}")

            if va + 1e-10 < best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    if verbose:
                        print(f"[MLP-TORCH] early stop @ epoch={epoch:03d} best_val_mse={best_val:.6f}")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X) -> np.ndarray:
        if self.model_ is None or self.x_scaler_ is None or self.y_scaler_ is None:
            raise RuntimeError("Model is not fitted.")

        X_np = np.asarray(X, dtype=np.float32)
        Xs = self.x_scaler_.transform(X_np).astype(np.float32)

        self.model_.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xs).to(self.device_)
            pred_s = self.model_(xb).detach().cpu().numpy().reshape(-1, 1)

        pred = self.y_scaler_.inverse_transform(pred_s).reshape(-1)
        return pred.astype(float)

    def to_cpu(self) -> None:
        if self.model_ is not None:
            self.model_ = self.model_.cpu()
        self.device_ = "cpu"


def _make_chrono_val_split(df_train: "pd.DataFrame", mode: str) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Chronological tail validation split από το END του train (no leakage).
    Για hourly: τουλάχιστον 7 ημέρες (168).
    Για daily: τουλάχιστον 30 ημέρες.
    Αν δεν φτάνει, παίρνει 10% του train.
    """
    n = len(df_train)
    if mode == "hourly":
        min_val = 24 * 7
    else:
        min_val = 30

    val_n = max(min_val, int(0.10 * n))
    val_n = min(val_n, max(1, n // 2))  # μην “φάει” πάνω από το μισό train

    df_tr = df_train.iloc[:-val_n].copy()
    df_va = df_train.iloc[-val_n:].copy()
    return df_tr, df_va


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train(
    mode: str,
    task: str = "price",
    test_size: Optional[int] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> None:
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

    X_train_full, y_train_full = make_xy(df_train)
    X_test, y_test = make_xy(df_test)

    # ---- change requested: slightly larger hidden for hourly ----
    if mode == "daily":
        hidden = (192, 128)  # mild increase
    else:
        hidden = (1024, 512)  # mild increase (requested)

    df_tr, df_va = _make_chrono_val_split(df_train, mode=mode)
    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    print(f"🧠 TRAINING MLP (TORCH) ({mode.upper()} | task={task.upper()}) [NO-LEAK]")
    print(f"   -> train={len(df_train)} | test={len(df_test)} | features={X_train_full.shape[1]}")
    print(f"   -> internal_val={len(df_va)} (chronological tail of train)")
    print(f"   -> hidden={hidden}")

    model = TorchMLPRegressor(
        hidden=hidden,
        dropout=0.10,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=1024,
        max_epochs=200,
        patience=20,
        seed=42,
        device="auto",
    )

    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, verbose=True)

    # sanity on test
    yhat_test = model.predict(X_test)
    mae = _mae(np.asarray(y_test, dtype=float), np.asarray(yhat_test, dtype=float))
    rmse = _rmse(np.asarray(y_test, dtype=float), np.asarray(yhat_test, dtype=float))
    print(f"   -> sanity(test): MAE={mae:.4f} | RMSE={rmse:.4f}")

    # save
    MODELS_DIR.mkdir(exist_ok=True)
    model.to_cpu()
    model_path = MODELS_DIR / f"mlp_{mode}_{task}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved MLP(Torch) to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--test_size", type=int, default=None, help="Rows in test (daily=days, hourly=hours)")
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)
    args = parser.parse_args()

    train(
        args.mode,
        task=args.task,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
