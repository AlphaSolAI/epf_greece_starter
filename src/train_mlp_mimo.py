from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _load_processed(mode: str, task: str) -> pd.DataFrame:
    from src.split_utils import load_processed

    try:
        return load_processed(mode, task=task)  # type: ignore
    except TypeError:
        return load_processed(mode)  # type: ignore


def _make_supervised_xy(df_train: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    from src.split_utils import make_xy

    X, _ = make_xy(df_train)
    if "y" not in df_train.columns:
        raise ValueError("Expected target column 'y' in processed dataframe")

    y = df_train["y"].astype(float)
    Y_df = pd.concat([y.shift(-k).rename(f"y_t+{k}") for k in range(1, horizon + 1)], axis=1)
    joined = X.join(Y_df, how="inner")
    joined = joined.dropna(subset=list(Y_df.columns))

    X_aligned = joined[X.columns]
    X_aligned = X_aligned.select_dtypes(include=[np.number]).copy()
    for c in X_aligned.columns:
        if X_aligned[c].dtype == bool:
            X_aligned[c] = X_aligned[c].astype(np.int8)

    Y = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_aligned, Y


@dataclass
class TorchMLPConfig:
    hidden_layers: tuple[int, ...] = (1028, 512, 128)
    dropout: float = 0.0
    batch_size: int = 1024
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    val_frac: float = 0.10
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    amp: bool = True
    scale_y: bool = True


class TorchMLPMIMORegressor:
    """Small torch MLP regressor for multi-output."""

    def __init__(self, input_dim: int, output_dim: int, cfg: TorchMLPConfig):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.cfg = cfg

        self._is_fitted = False
        self._model_state = None
        self.x_scaler = None
        self.y_scaler = None

    @staticmethod
    def _as_float32(X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return np.asarray(X, dtype=np.float32)

    def _choose_device(self) -> str:
        d = str(self.cfg.device).lower()
        if d in {"cuda", "gpu"}:
            return "cuda"
        if d == "cpu":
            return "cpu"

        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _build_model(self):
        import torch.nn as nn

        layers = []
        prev = self.input_dim
        for h in self.cfg.hidden_layers:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            if self.cfg.dropout and self.cfg.dropout > 0:
                layers.append(nn.Dropout(float(self.cfg.dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, self.output_dim))
        return nn.Sequential(*layers)

    def fit(self, X, Y):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))

        X = self._as_float32(X)
        Y = np.asarray(Y, dtype=np.float32)

        # Standardize X
        self.x_scaler = StandardScaler()
        Xs = self.x_scaler.fit_transform(X).astype(np.float32, copy=False)

        # Standardize Y (optional)
        if self.cfg.scale_y:
            self.y_scaler = StandardScaler()
            Ys = self.y_scaler.fit_transform(Y).astype(np.float32, copy=False)
        else:
            self.y_scaler = None
            Ys = Y

        # Split train/val chronologically (no shuffle)
        n = Xs.shape[0]
        n_val = max(1, int(round(float(self.cfg.val_frac) * n)))
        n_train = max(1, n - n_val)

        Xtr, Ytr = Xs[:n_train], Ys[:n_train]
        Xva, Yva = Xs[n_train:], Ys[n_train:]

        device = self._choose_device()
        use_amp = bool(self.cfg.amp) and device == "cuda"

        model = self._build_model().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))
        loss_fn = nn.MSELoss()

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best = float("inf")
        best_state = None
        bad = 0

        def batches(Xa, Ya, bs):
            for i in range(0, Xa.shape[0], bs):
                yield Xa[i : i + bs], Ya[i : i + bs]

        for ep in range(int(self.cfg.epochs)):
            model.train()
            for xb, yb in batches(Xtr, Ytr, int(self.cfg.batch_size)):
                xb_t = torch.from_numpy(xb).to(device)
                yb_t = torch.from_numpy(yb).to(device)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(xb_t)
                    loss = loss_fn(pred, yb_t)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            # val
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(Xva).to(device)
                yv = torch.from_numpy(Yva).to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pv = model(xv)
                    vloss = loss_fn(pv, yv).item()

            if vloss < best:
                best = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(self.cfg.patience):
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        self._model_state = best_state
        self._is_fitted = True
        return self

    def predict(self, X):
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        X = self._as_float32(X)
        if self.x_scaler is not None:
            X = self.x_scaler.transform(X).astype(np.float32, copy=False)

        device = self._choose_device()

        model = self._build_model().to(device)
        model.load_state_dict(self._model_state)
        model.eval()

        with torch.no_grad():
            x_t = torch.from_numpy(X).to(device)
            pred = model(x_t).detach().cpu().numpy().astype(np.float32)

        if self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred).astype(np.float32, copy=False)

        return pred


def train_mlp_mimo_single(
    *,
    mode: str,
    task: str,
    horizon: int,
    train_end: str,
    cfg: TorchMLPConfig,
):
    from src.split_utils import split_time_series

    df_all = _load_processed(mode, task)
    df_train, _ = split_time_series(df_all, mode=mode, train_end=train_end)

    X_train, Y_train = _make_supervised_xy(df_train, horizon=horizon)

    print(f"🚀 TRAINING MLP MIMO-SINGLE (HOURLY | task={task.upper()}) [NO-LEAK]")
    print(f"   -> horizon={horizon} | device={cfg.device} | epochs={cfg.epochs}")
    print(f"   -> train={len(df_train)} -> usable={len(X_train)} | features={X_train.shape[1]}")
    print(f"   -> train_window={df_train.index.min()} -> {df_train.index.max()}")

    model = TorchMLPMIMORegressor(input_dim=X_train.shape[1], output_dim=horizon, cfg=cfg)
    model.fit(X_train, Y_train)

    bundle = {
        "model": model,
        "feature_cols": list(X_train.columns),
        "horizon": int(horizon),
        "mode": mode,
        "task": task,
        "train_end": train_end,
        "strategy": "mimo_single",
        "lib": "torch_mlp",
        "cfg": cfg,
    }

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", f"mlp_mimo_single_{mode}_{task}_h{horizon}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Saved MLP MIMO-SINGLE to: {os.path.abspath(out_path)}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"], help="Only hourly is supported")
    p.add_argument("--task", required=True, choices=["price", "load"])
    p.add_argument("--horizon", type=int, default=168)
    p.add_argument("--train_end", required=True)

    p.add_argument("--hidden_layers", type=str, default="1028,512,256")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_scale_y", action="store_true")
    return p


def main() -> int:
    args = _build_argparser().parse_args()

    hidden = tuple(int(x) for x in args.hidden_layers.split(",") if x.strip())

    cfg = TorchMLPConfig(
        hidden_layers=hidden,
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        device=str(args.device),
        amp=bool(args.amp),
        scale_y=not bool(args.no_scale_y),
    )

    train_mlp_mimo_single(
        mode=args.mode,
        task=args.task,
        horizon=int(args.horizon),
        train_end=str(args.train_end),
        cfg=cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
