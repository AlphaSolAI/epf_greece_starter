from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.split_utils import load_processed, split_time_series

TARGET_COL = "y"

DEFAULT_FORCED = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "y_lag1",
    "y_lag24",
    "y_lag168",
]


def _parse_csv_list(s: Optional[str], cast=str) -> List:
    if s is None:
        return []
    if isinstance(s, list):
        return [cast(x) for x in s]
    s = str(s).strip()
    if not s:
        return []
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(cast(chunk))
    return parts


def _parse_int_list(s: Optional[str]) -> List[int]:
    out: List[int] = []
    if s is None:
        return out
    s = str(s).strip()
    if not s:
        return out
    if "," in s:
        toks = [t.strip() for t in s.split(",") if t.strip()]
    else:
        toks = [t.strip() for t in s.split() if t.strip()]
    for t in toks:
        out.append(int(t))
    return out


def _parse_datetime(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / denom))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _available_features(df: pd.DataFrame, forced: Sequence[str]) -> List[str]:
    return [f for f in forced if f in df.columns]


def _build_origins(
    test_start: pd.Timestamp,
    test_size: int,
    horizon: int,
    origin_stride: int,
) -> List[pd.Timestamp]:
    offsets = []
    off = 0
    while off + horizon <= test_size:
        offsets.append(off)
        off += origin_stride
    return [test_start + pd.Timedelta(hours=int(o)) for o in offsets]


def _subsample(df: pd.DataFrame, max_rows: Optional[int], seed: int = 13) -> pd.DataFrame:
    if (max_rows is None) or (max_rows <= 0) or (len(df) <= max_rows):
        return df
    return df.sample(n=int(max_rows), random_state=seed).sort_index()


@dataclass
class TorchMLPConfig:
    hidden_layers: List[int]
    dropout: float = 0.0
    batch_size: int = 1024
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 25
    val_frac: float = 0.10
    seed: int = 13
    device: str = "auto"   # auto|cpu|cuda
    amp: bool = True       # mixed precision on CUDA
    scale_y: bool = True   # StandardScaler on targets


class TorchMLPMIMORegressor:
    def __init__(self, input_dim: int, output_dim: int, cfg: TorchMLPConfig):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.cfg = cfg
        self._is_fitted = False

        self.x_scaler = None
        self.y_scaler = None
        self._model_state = None
        self._best_val = None

    @staticmethod
    def _choose_device(req: str) -> str:
        req = (req or "auto").lower().strip()
        try:
            import torch
            if req == "cuda":
                return "cuda" if torch.cuda.is_available() else "cpu"
            if req == "cpu":
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _build_model(self):
        import torch.nn as nn

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for h in self.cfg.hidden_layers:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            if self.cfg.dropout and self.cfg.dropout > 0:
                layers.append(nn.Dropout(float(self.cfg.dropout)))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, self.output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _as_float32(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        return X.astype(np.float32, copy=False)

    def fit(self, X, Y):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        rng_seed = int(self.cfg.seed)
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)

        device = self._choose_device(self.cfg.device)

        X = self._as_float32(X)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y length mismatch: {X.shape[0]} vs {Y.shape[0]}")
        if Y.shape[1] != self.output_dim:
            raise ValueError(f"Y output_dim mismatch: got {Y.shape[1]} expected {self.output_dim}")

        self.x_scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = self.x_scaler.fit_transform(X).astype(np.float32, copy=False)

        if bool(self.cfg.scale_y):
            self.y_scaler = StandardScaler(with_mean=True, with_std=True)
            Ys = self.y_scaler.fit_transform(Y).astype(np.float32, copy=False)
        else:
            self.y_scaler = None
            Ys = Y.astype(np.float32, copy=False)

        n = Xs.shape[0]
        val_frac = float(self.cfg.val_frac) if self.cfg.val_frac is not None else 0.0
        n_val = int(max(0, round(val_frac * n)))
        n_val = min(n_val, max(0, n - 200))
        if n_val < 200:
            n_val = 0

        if n_val > 0:
            X_train, Y_train = Xs[:-n_val], Ys[:-n_val]
            X_val, Y_val = Xs[-n_val:], Ys[-n_val:]
        else:
            X_train, Y_train = Xs, Ys
            X_val, Y_val = None, None

        X_train_t = torch.from_numpy(X_train)
        Y_train_t = torch.from_numpy(Y_train)
        train_ds = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        if X_val is not None:
            X_val_t = torch.from_numpy(X_val)
            Y_val_t = torch.from_numpy(Y_val)
            val_ds = torch.utils.data.TensorDataset(X_val_t, Y_val_t)
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        else:
            val_loader = None

        model = self._build_model().to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        loss_fn = nn.MSELoss()

        use_amp = bool(self.cfg.amp) and (device == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_state = None
        best_val = float("inf")
        bad = 0
        patience = int(self.cfg.patience)

        for _epoch in range(int(self.cfg.epochs)):
            model.train()
            running = 0.0
            n_seen = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                bs = xb.shape[0]
                running += float(loss.detach().cpu().item()) * bs
                n_seen += bs

            train_loss = running / max(1, n_seen)

            if val_loader is None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val = train_loss
                break

            model.eval()
            v_running = 0.0
            v_seen = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                    bs = xb.shape[0]
                    v_running += float(loss.detach().cpu().item()) * bs
                    v_seen += bs

            val_loss = v_running / max(1, v_seen)

            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        self._model_state = best_state
        self._best_val = float(best_val)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted or self._model_state is None:
            raise RuntimeError("Model is not fitted.")

        import torch
        X = self._as_float32(X)
        Xs = self.x_scaler.transform(X).astype(np.float32, copy=False)

        device = self._choose_device(self.cfg.device)
        model = self._build_model().to(device)
        model.load_state_dict(self._model_state, strict=True)
        model.eval()

        xb = torch.from_numpy(Xs).to(device)
        with torch.no_grad():
            pred_s = model(xb).detach().cpu().numpy().astype(np.float32, copy=False)

        if self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred_s).astype(np.float32, copy=False)
        else:
            pred = pred_s
        return pred


def rank_features(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    y_col: str,
    rank_steps: Sequence[int],
    n_estimators: int,
    max_rows: Optional[int],
    seed: int = 13,
) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor

    df_train = _subsample(df_train, max_rows=max_rows, seed=seed)
    X_full = df_train.loc[:, feature_cols]
    y_full = df_train.loc[:, y_col].astype(float)

    n = len(df_train)
    imp_acc = np.zeros(len(feature_cols), dtype=float)
    used_steps = 0

    for s in rank_steps:
        s = int(s)
        if s <= 0:
            continue
        if n - s <= 200:
            continue

        n_samples = n - s
        X = X_full.iloc[:n_samples].copy()
        y = y_full.iloc[s:s + n_samples].copy()

        mask = np.isfinite(y.values)
        mask &= np.isfinite(X.values).all(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]
        if len(X) < 200:
            continue

        ranker = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=seed + s,
            n_jobs=-1,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
        )
        ranker.fit(X, y)
        imp = np.asarray(ranker.feature_importances_, dtype=float)
        if np.all(imp == 0):
            continue

        imp = imp / (imp.sum() + 1e-12)
        imp_acc += imp
        used_steps += 1

    if used_steps == 0:
        imp_acc = np.ones(len(feature_cols), dtype=float)
        used_steps = 1

    imp_mean = imp_acc / used_steps
    out = pd.DataFrame({"feature": list(feature_cols), "importance": imp_mean})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _select_topk(ranked: pd.DataFrame, k: int, forced: Sequence[str]) -> List[str]:
    forced_set = set(forced)
    feats = [f for f in ranked["feature"].tolist() if f not in forced_set]
    return feats[: int(k)]


def build_xy_mimo(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon: int,
    y_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, np.ndarray]:
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive")

    n = len(df)
    n_samples = n - h
    if n_samples <= 0:
        raise ValueError("Not enough rows for horizon")

    X = df.loc[:, feature_cols].iloc[:n_samples].copy()
    y_series = df.loc[:, y_col].astype(float).values

    Y = np.zeros((n_samples, h), dtype=float)
    for kk in range(h):
        Y[:, kk] = y_series[1 + kk: 1 + kk + n_samples]

    mask = np.isfinite(X.values).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X.loc[mask]
    Y = Y[mask]
    return X, Y


def eval_multi_origin_mimo(
    model,
    df_full: pd.DataFrame,
    feature_cols: Sequence[str],
    test_start: pd.Timestamp,
    test_size: int,
    horizon: int,
    origin_stride: int,
    y_col: str = TARGET_COL,
) -> Dict[str, float]:
    origins = _build_origins(test_start, int(test_size), int(horizon), int(origin_stride))
    maes, rmses, smapes = [], [], []

    for origin in origins:
        t0 = origin - pd.Timedelta(hours=1)
        if t0 not in df_full.index:
            continue

        X0 = df_full.loc[[t0], feature_cols]
        yhat = model.predict(X0)
        yhat = np.asarray(yhat).reshape(-1)
        if len(yhat) != int(horizon):
            yhat = np.asarray(yhat).reshape(int(horizon),)

        y_true = df_full.loc[origin: origin + pd.Timedelta(hours=int(horizon) - 1), y_col].astype(float).values
        if len(y_true) != int(horizon):
            continue

        mask = np.isfinite(y_true) & np.isfinite(yhat)
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = yhat[mask]
        maes.append(_mae(yt, yp))
        rmses.append(_rmse(yt, yp))
        smapes.append(_smape(yt, yp))

    if len(maes) == 0:
        return {
            "n_origins": 0,
            "MAE": np.nan, "RMSE": np.nan, "sMAPE": np.nan,
            "MAE_std": np.nan, "RMSE_std": np.nan, "sMAPE_std": np.nan,
        }

    return {
        "n_origins": int(len(maes)),
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "sMAPE": float(np.mean(smapes)),
        "MAE_std": float(np.std(maes, ddof=0)),
        "RMSE_std": float(np.std(rmses, ddof=0)),
        "sMAPE_std": float(np.std(smapes, ddof=0)),
    }


@dataclass
class WeekSpec:
    label: str
    test_start: pd.Timestamp


def run_week(
    df_full: pd.DataFrame,
    mode: str,
    week: WeekSpec,
    train_end: pd.Timestamp,
    test_size: int,
    horizon: int,
    eval_mode: str,
    origin_stride: int,
    strategies: Sequence[str],
    topk_list: Sequence[int],
    forced_features: Sequence[str],
    rank_steps: Sequence[int],
    rank_n_estimators: int,
    rank_max_rows: Optional[int],
    train_max_rows: Optional[int],
    print_features: bool,
    mlp_cfg: TorchMLPConfig,
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    df_train, _ = split_time_series(
        df_full, mode=mode,
        train_end=train_end,
        test_start=week.test_start,
        test_size=int(test_size),
    )

    feature_cols_all = [c for c in df_full.columns if c != TARGET_COL]
    forced_ok = _available_features(df_full, forced_features)

    ranked = rank_features(
        df_train=df_train,
        feature_cols=feature_cols_all,
        y_col=TARGET_COL,
        rank_steps=rank_steps,
        n_estimators=int(rank_n_estimators),
        max_rows=rank_max_rows,
    )

    feature_sets: Dict[Tuple[str, Optional[int]], List[str]] = {}
    if "all" in strategies:
        feature_sets[("all", None)] = feature_cols_all

    for k in topk_list:
        if "topk" in strategies:
            feature_sets[("topk", int(k))] = _select_topk(ranked, int(k), forced_ok)
        if "hybrid" in strategies:
            topk_feats = _select_topk(ranked, int(k), forced_ok)
            hybrid = list(dict.fromkeys(list(forced_ok) + list(topk_feats)))
            feature_sets[("hybrid", int(k))] = hybrid

    if print_features:
        print(f"\n[WEEK {week.label}] Forced features used ({len(forced_ok)}): {forced_ok}")
        print(f"[WEEK {week.label}] Ranked features (top 30):")
        for _, row in ranked.head(30).iterrows():
            print(f"  {int(row['rank']):>3d}. {row['feature']:<35s}  imp={row['importance']:.6f}")

    rows: List[Dict] = []
    rows_json: List[Dict] = []

    meta_week = {
        "label": week.label,
        "test_start": str(week.test_start),
        "train_end": str(train_end),
        "test_size": int(test_size),
        "horizon": int(horizon),
        "origin_stride": int(origin_stride),
        "rank_steps": [int(x) for x in rank_steps],
        "rank_n_estimators": int(rank_n_estimators),
        "rank_max_rows": int(rank_max_rows) if rank_max_rows else None,
        "train_max_rows": int(train_max_rows) if train_max_rows else None,
        "forced_used": forced_ok,
        "ranked_features_top50": ranked.head(50).to_dict(orient="records"),
        "feature_sets": {f"{s}" + (f"_{kk}" if kk is not None else ""): v for (s, kk), v in feature_sets.items()},
    }

    for (strategy, k), feats in feature_sets.items():
        X_train, Y_train = build_xy_mimo(df_train, feats, horizon=horizon, y_col=TARGET_COL)
        if train_max_rows and len(X_train) > train_max_rows:
            idx = np.linspace(0, len(X_train) - 1, int(train_max_rows)).astype(int)
            X_train = X_train.iloc[idx]
            Y_train = Y_train[idx]

        model = TorchMLPMIMORegressor(input_dim=len(feats), output_dim=int(horizon), cfg=mlp_cfg)
        model.fit(X_train, Y_train)

        if eval_mode == "multi_origin":
            metrics = eval_multi_origin_mimo(
                model=model,
                df_full=df_full,
                feature_cols=feats,
                test_start=week.test_start,
                test_size=int(test_size),
                horizon=int(horizon),
                origin_stride=int(origin_stride),
                y_col=TARGET_COL,
            )
        else:
            metrics = eval_multi_origin_mimo(
                model=model,
                df_full=df_full,
                feature_cols=feats,
                test_start=week.test_start,
                test_size=int(test_size),
                horizon=int(horizon),
                origin_stride=int(test_size),
                y_col=TARGET_COL,
            )

        row = {
            "week": week.label,
            "test_start": str(week.test_start),
            "model": "MLP",
            "strategy": strategy,
            "k": (int(k) if k is not None else None),
            "n_features": int(len(feats)),
            **metrics,
        }
        rows_json.append(row)
        rows.append(row)

    df_res = pd.DataFrame(rows).sort_values(["sMAPE", "RMSE"], ascending=True)
    return df_res, meta_week, rows_json


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="feature_strategy_compare1.py", add_help=True)
    sub = p.add_subparsers(dest="mode", required=True)

    ph = sub.add_parser("hourly", help="Hourly processed dataset")
    ph.add_argument("--horizon", type=int, required=True)
    ph.add_argument("--test_size", type=int, default=168)
    ph.add_argument("--train_end", type=str, required=True)

    ph.add_argument("--test_start", type=str, required=True)
    ph.add_argument("--week_label", type=str, default="week")

    ph.add_argument("--eval_mode", type=str, default="multi_origin", choices=["multi_origin", "single_origin"])
    ph.add_argument("--origin_stride", type=int, default=24)

    ph.add_argument("--strategies", type=str, default="all,topk,hybrid")
    ph.add_argument("--topk", type=str, default="10,20,30")

    ph.add_argument("--rank_steps", type=str, default="24,168")
    ph.add_argument("--rank_n_estimators", type=int, default=1000)
    ph.add_argument("--rank_max_rows", type=int, default=30000)
    ph.add_argument("--train_max_rows", type=int, default=30000)

    ph.add_argument("--forced", type=str, default=",".join(DEFAULT_FORCED))
    ph.add_argument("--print_features", action="store_true")

    ph.add_argument("--mlp_hidden", type=str, default="256,128")
    ph.add_argument("--mlp_dropout", type=float, default=0.0)
    ph.add_argument("--mlp_epochs", type=int, default=200)
    ph.add_argument("--mlp_batch_size", type=int, default=1024)
    ph.add_argument("--mlp_lr", type=float, default=1e-3)
    ph.add_argument("--mlp_weight_decay", type=float, default=1e-5)
    ph.add_argument("--mlp_patience", type=int, default=25)
    ph.add_argument("--mlp_val_frac", type=float, default=0.10)
    ph.add_argument("--mlp_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ph.add_argument("--mlp_no_amp", action="store_true")
    ph.add_argument("--mlp_no_scale_y", action="store_true")

    ph.add_argument("--save_json", type=str, default=None)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    train_end = _parse_datetime(args.train_end)
    test_start = _parse_datetime(args.test_start)

    strategies = [s.strip() for s in _parse_csv_list(args.strategies, cast=str)]
    topk_list = _parse_int_list(args.topk)
    rank_steps = _parse_int_list(args.rank_steps)
    forced_features = _parse_csv_list(args.forced, cast=str)

    hidden = _parse_int_list(args.mlp_hidden)
    if not hidden:
        raise ValueError("--mlp_hidden parsed empty. Example: --mlp_hidden \"512,512,256\"")

    mlp_cfg = TorchMLPConfig(
        hidden_layers=hidden,
        dropout=float(args.mlp_dropout),
        batch_size=int(args.mlp_batch_size),
        epochs=int(args.mlp_epochs),
        lr=float(args.mlp_lr),
        weight_decay=float(args.mlp_weight_decay),
        patience=int(args.mlp_patience),
        val_frac=float(args.mlp_val_frac),
        seed=13,
        device=str(args.mlp_device),
        amp=(not bool(args.mlp_no_amp)),
        scale_y=(not bool(args.mlp_no_scale_y)),
    )

    lines: List[str] = []

    def log(s: str = ""):
        print(s)
        lines.append(s)

    log("\n📌 FEATURE STRATEGY COMPARE — MLP (Torch) ONLY")
    log(f"[INFO] horizon={int(args.horizon)} | test_size={int(args.test_size)}")
    log(f"[INFO] train_end={train_end} | test_start={test_start} | week={args.week_label}")
    log(f"[INFO] strategies={strategies} | topk={topk_list} | rank_steps={rank_steps}")
    log(f"[INFO] mlp_hidden={mlp_cfg.hidden_layers} | dropout={mlp_cfg.dropout} | epochs={mlp_cfg.epochs} | batch={mlp_cfg.batch_size} | lr={mlp_cfg.lr} | wd={mlp_cfg.weight_decay} | device={mlp_cfg.device} | amp={mlp_cfg.amp}")

    df_full = load_processed(mode="hourly")

    df_res, meta_week, rows_json = run_week(
        df_full=df_full,
        mode="hourly",
        week=WeekSpec(label=str(args.week_label), test_start=test_start),
        train_end=train_end,
        test_size=int(args.test_size),
        horizon=int(args.horizon),
        eval_mode=str(args.eval_mode),
        origin_stride=int(args.origin_stride),
        strategies=strategies,
        topk_list=topk_list,
        forced_features=forced_features,
        rank_steps=rank_steps,
        rank_n_estimators=int(args.rank_n_estimators),
        rank_max_rows=int(args.rank_max_rows) if args.rank_max_rows else None,
        train_max_rows=int(args.train_max_rows) if args.train_max_rows else None,
        print_features=bool(args.print_features),
        mlp_cfg=mlp_cfg,
    )

    log(f"\n=== RESULTS: {args.week_label} | test_start={test_start} ===")
    show_cols = ["model", "strategy", "k", "n_features", "MAE", "RMSE", "sMAPE", "MAE_std", "RMSE_std", "sMAPE_std", "n_origins"]
    log(df_res.loc[:, show_cols].to_string(index=False))

    meta = {
        "mode": "hourly",
        "horizon": int(args.horizon),
        "test_size": int(args.test_size),
        "train_end": str(train_end),
        "eval_mode": str(args.eval_mode),
        "origin_stride": int(args.origin_stride),
        "strategies": strategies,
        "topk": topk_list,
        "rank_steps": rank_steps,
        "rank_n_estimators": int(args.rank_n_estimators),
        "rank_max_rows": int(args.rank_max_rows) if args.rank_max_rows else None,
        "train_max_rows": int(args.train_max_rows) if args.train_max_rows else None,
        "forced": forced_features,
        "mlp_cfg": {
            "hidden_layers": mlp_cfg.hidden_layers,
            "dropout": mlp_cfg.dropout,
            "epochs": mlp_cfg.epochs,
            "batch_size": mlp_cfg.batch_size,
            "lr": mlp_cfg.lr,
            "weight_decay": mlp_cfg.weight_decay,
            "patience": mlp_cfg.patience,
            "val_frac": mlp_cfg.val_frac,
            "device": mlp_cfg.device,
            "amp": mlp_cfg.amp,
            "scale_y": mlp_cfg.scale_y,
        },
        "weeks": {str(args.week_label): meta_week},
        "rows": rows_json,
    }

    if args.save_json:
        _ensure_dir(args.save_json)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        log(f"\n✅ Saved JSON: {args.save_json}")

        txt_path = os.path.splitext(args.save_json)[0] + ".txt"
        _ensure_dir(txt_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log(f"✅ Saved TXT:  {txt_path}")


if __name__ == "__main__":
    main()
