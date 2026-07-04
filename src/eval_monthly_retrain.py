"""
eval_monthly_retrain.py  —  v2 (comprehensive)

Walk-forward MONTHLY retraining evaluation (σαν Kanousis αλλά μηνιαία).

Για κάθε μήνα (Dec 2025, Jan 2026, Feb 2026):
  1. Retrain models σε ΟΛΑ τα data μέχρι το τέλος του προηγούμενου μήνα
  2. Predict τον επόμενο μήνα
  3. Συλλέγει predictions → ενιαία μετρικά Q1 2026

═══════════════ ΣΩΣΤΗ ΟΡΟΛΟΓΙΑ ═══════════════
  ┌──────────────┬──────────────────────────────────────────────┐
  │ Παλιό όνομα  │ Σωστό (control theory)                       │
  ├──────────────┼──────────────────────────────────────────────┤
  │ "CL"         │ TF = Teacher-Forcing / Open-Loop Oracle       │
  │              │ Χρησιμοποιεί ΠΡΑΓΜΑΤΙΚΕΣ προηγούμενες τιμές  │
  │              │ → ΔΕΝ υπάρχει ανατροφοδότηση (open-loop)     │
  ├──────────────┼──────────────────────────────────────────────┤
  │ "OL/Recursive"│ Rec = Recursive / Closed-Loop Autoregressive │
  │              │ Χρησιμοποιεί ΠΡΟΒΛΕΨΕΙΣ ως input             │
  │              │ → Ανατροφοδότηση (closed-loop)               │
  └──────────────┴──────────────────────────────────────────────┘

TF Models  (5 families): LGBM, XGB, RF, SVR, MLP
Rec Models (4 variants): LGBM-DO, LGBM-SS-DO, XGB-DO, XGB-SS-DO
  Per-month Optuna: Dec→existing params, Jan/Feb→inline Optuna (n_trials CLI)

Usage:
  conda run -n epf --no-capture-output python -m src.eval_monthly_retrain hourly \\
      --task price --strategies tf,rec --save_json
  conda run -n epf --no-capture-output python -m src.eval_monthly_retrain hourly \\
      --task load --strategies tf,rec --save_json --n_trials_rec 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

from .split_utils import load_processed, make_xy, split_time_series
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop


# ── Months definition ─────────────────────────────────────────────────────────
EVAL_MONTHS = [
    {
        "name":       "Dec-2025",
        "code":       "dec2025",
        "train_end":  "2025-11-30 23:00",
        "test_start": "2025-12-01 00:00",
        "test_end":   "2025-12-31 23:00",
    },
    {
        "name":       "Jan-2026",
        "code":       "jan2026",
        "train_end":  "2025-12-31 23:00",
        "test_start": "2026-01-01 00:00",
        "test_end":   "2026-01-31 23:00",
    },
    {
        "name":       "Feb-2026",
        "code":       "feb2026",
        "train_end":  "2026-01-31 23:00",
        "test_start": "2026-02-01 00:00",
        "test_end":   "2026-02-28 23:00",
    },
]

BASELINE_NAMES = {"Naive-1", "Naive-24", "Naive-168", "Seasonal Profile"}


# ── Metrics ───────────────────────────────────────────────────────────────────
def _fnp(a) -> np.ndarray:
    return np.asarray(a, dtype=float)

def mae(yt, yp) -> float:
    yt, yp = _fnp(yt), _fnp(yp)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.mean(np.abs(yt[m] - yp[m]))) if m.sum() > 0 else float("nan")

def rmse(yt, yp) -> float:
    yt, yp = _fnp(yt), _fnp(yp)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2))) if m.sum() > 0 else float("nan")

def smape(yt, yp) -> float:
    yt, yp = _fnp(yt), _fnp(yp)
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() == 0:
        return float("nan")
    denom = np.abs(yt[m]) + np.abs(yp[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(yt[m] - yp[m]) / denom))


# ── JSON helpers ──────────────────────────────────────────────────────────────
def _fix_nan(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _fix_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_nan(x) for x in obj]
    return obj


# ── TF model builders ─────────────────────────────────────────────────────────
def _build_lgbm_cl(task: str):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=1000, learning_rate=0.05,
                         num_leaves=63, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, verbose=-1)


def _build_xgb_cl(task: str):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=-1, verbosity=0)


def _build_rf_cl(task: str):
    from sklearn.ensemble import RandomForestRegressor
    # Παράμετροι ίδιοι με train_rf.py (αρχικό Dec-2025 training):
    #   max_features="sqrt" ≈ 12 features/split (αντί 0.7*152=107) → ~9x ταχύτερο
    #   max_depth=16, min_samples_split=4, min_samples_leaf=2
    # n_jobs=1: αποφεύγει το loky spawn deadlock στα Windows
    return RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                 max_depth=16, min_samples_split=4,
                                 min_samples_leaf=2,
                                 n_jobs=1, random_state=42)


def _build_mlp_cl(task: str):
    """Sklearn MLPRegressor με (256,128) hidden layers."""
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=300,
        learning_rate_init=5e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )


def _build_lstm_cl(task: str):
    """PyTorch LSTM wrapper. Χρησιμοποιεί y_lag1..y_lag24 ως 24-step sequence
    + όλα τα features ως context. Αντικαθιστά SVR.
    Μειωμένο hidden_size=64 & max_epochs=50 για CPU εκτέλεση."""
    return _LSTMWrapper(
        seq_len=24,
        hidden_size=64,
        n_layers=2,
        max_epochs=50,
        lr=1e-3,
        batch_size=512,
        patience=5,
    )


class _LSTMWrapper:
    """
    PyTorch LSTM για tabular EPF.

    Architecture:
      ┌─ y_lag24 → y_lag23 → … → y_lag1 (24 steps × 1 value) ─┐
      │            LSTM(hidden=128, layers=2)                     │
      │            → final hidden state h                         │
      └──────────────────────────────────────────────────────────┘
      concat(h, full_feature_vec) → Linear(128+n_feat, 64) → ReLU → Linear(64,1)

    Falls back to flat input (seq_len=1) if y_lag columns not found.
    Works as sklearn-style fit/predict — compatible with recursive_predict_openloop.
    """

    def __init__(self, seq_len=24, hidden_size=128, n_layers=2,
                 max_epochs=100, lr=1e-3, batch_size=512, patience=10):
        self.seq_len    = seq_len
        self.hidden_size = hidden_size
        self.n_layers   = n_layers
        self.max_epochs = max_epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.patience   = patience
        self._model     = None
        self._lag_idx   = None   # indices of y_lag1..y_lag24 in feature matrix
        self._scaler_X  = None
        self._scaler_y  = None
        self._n_features = None

    def _find_lag_indices(self, feature_names):
        """Return column indices for y_lag1..y_lag{seq_len}, oldest first."""
        lag_to_idx = {}
        for i, col in enumerate(feature_names):
            if col.startswith("y_lag"):
                try:
                    k = int(col[5:])
                    if 1 <= k <= self.seq_len:
                        lag_to_idx[k] = i
                except ValueError:
                    pass
        if len(lag_to_idx) < self.seq_len:
            return None  # not enough lag columns → fallback
        # Order: oldest (seq_len) → newest (1)
        return [lag_to_idx[k] for k in range(self.seq_len, 0, -1)]

    def _build_net(self, n_features):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, seq_len, hidden_size, n_layers, n_features):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1, hidden_size=hidden_size,
                    num_layers=n_layers, batch_first=True, dropout=0.2
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size + n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
            def forward(self, seq, ctx):
                # seq: (B, seq_len, 1)  ctx: (B, n_features)
                _, (h, _) = self.lstm(seq)   # h: (n_layers, B, hidden)
                h_last = h[-1]               # (B, hidden)
                x = torch.cat([h_last, ctx], dim=1)
                return self.head(x).squeeze(-1)

        return _Net(self.seq_len, self.hidden_size, self.n_layers, n_features)

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        # Limit PyTorch threads σε CPU training (αποφεύγει over-subscription)
        torch.set_num_threads(4)

        if hasattr(X, "columns"):
            feat_names = list(X.columns)
            X = X.values
        else:
            feat_names = [f"f{i}" for i in range(X.shape[1])]
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Find lag columns
        self._lag_idx = self._find_lag_indices(feat_names)
        self._n_features = X.shape[1]

        # Normalise
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_sc = self._scaler_X.fit_transform(X)
        y_sc = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Build sequences and contexts
        if self._lag_idx is not None:
            seq_np = X_sc[:, self._lag_idx].reshape(-1, self.seq_len, 1)  # (N,24,1)
        else:
            seq_np = X_sc[:, :1].reshape(-1, 1, 1)  # fallback: flat seq_len=1

        seq_t = torch.FloatTensor(seq_np)
        ctx_t = torch.FloatTensor(X_sc)
        y_t   = torch.FloatTensor(y_sc)

        # Train/val split (last 10%)
        n_val = max(168, int(0.08 * len(y_t)))
        seq_tr, seq_va = seq_t[:-n_val], seq_t[-n_val:]
        ctx_tr, ctx_va = ctx_t[:-n_val], ctx_t[-n_val:]
        y_tr,   y_va   = y_t[:-n_val],   y_t[-n_val:]

        net = self._build_net(self._n_features)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val, best_state, no_imp = float("inf"), None, 0
        ds_tr = torch.utils.data.TensorDataset(seq_tr, ctx_tr, y_tr)
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)

        net.train()
        for epoch in range(self.max_epochs):
            for sb, cb, yb in dl_tr:
                opt.zero_grad()
                loss_fn(net(sb, cb), yb).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(seq_va, ctx_va), y_va).item()
            net.train()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        return self

    def predict(self, X):
        import torch
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        X_sc = self._scaler_X.transform(X)

        if self._lag_idx is not None:
            seq_np = X_sc[:, self._lag_idx].reshape(-1, self.seq_len, 1)
        else:
            seq_np = X_sc[:, :1].reshape(-1, 1, 1)

        seq_t = torch.FloatTensor(seq_np)
        ctx_t = torch.FloatTensor(X_sc)
        self._model.eval()
        with torch.no_grad():
            y_sc = self._model(seq_t, ctx_t).numpy()
        return self._scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()


# TF (Teacher-Forcing) = Open-Loop Oracle (αυτό που παλιά λέγαμε "CL")
# LSTM αφαιρέθηκε — πολύπλοκο για CPU εκτέλεση, δεν προσφέρει σαφές πλεονέκτημα
# RF: LGBM RF-mode (γρηγορότερο από sklearn RF στα Windows)
TF_BUILDERS = [
    ("TF-LGBM-MR",  _build_lgbm_cl),
    ("TF-XGB-MR",   _build_xgb_cl),
    ("TF-RF-MR",    _build_rf_cl),
    ("TF-MLP-MR",   _build_mlp_cl),
]


# ── OL / Rec helpers ──────────────────────────────────────────────────────────
def _get_lag_cols(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _daily_ss_round(model, X_tr: pd.DataFrame, y_tr: pd.Series,
                    epsilon: float, rng: np.random.Generator) -> pd.DataFrame:
    """One round of daily Scheduled Sampling."""
    y_hat = pd.Series(model.predict(X_tr), index=X_tr.index, dtype=float)
    X_aug = X_tr.copy()
    lag_cols     = _get_lag_cols(X_tr)
    replace_mask = rng.random(len(X_tr)) < epsilon
    hour_of_day  = pd.Series(X_tr.index.hour, index=X_tr.index)
    for col in lag_cols:
        lag_k      = int(col.replace("y_lag", ""))
        within_day = hour_of_day >= lag_k
        pseudo_lag = y_hat.shift(lag_k).reindex(X_tr.index)
        valid      = replace_mask & within_day.values & pseudo_lag.notna()
        X_aug.loc[valid, col] = pseudo_lag[valid].values
    return X_aug


def _build_rec_model(algo: str, params: dict, n_estimators: int):
    """Build Rec model (LGBM or XGB) from params dict."""
    p = {k: v for k, v in params.items() if k not in ("n_estimators",)}
    if algo == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=n_estimators, verbose=-1, n_jobs=-1, **p)
    elif algo == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=n_estimators, tree_method="hist",
                                 n_jobs=-1, verbosity=0, **p)
    raise ValueError(f"Unsupported algo: {algo}")


def _retrain_rec_from_params(
    algo: str,
    params: dict,
    n_estimators: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    with_ss: bool = False,
    ss_n_iter: int = 3,
    ss_eps: Tuple[float, float] = (0.10, 0.40),
    es_rounds: int = 100,
) -> object:
    """Retrain a Rec model from params dict (in-memory, no disk save)."""
    # Validation split for early stopping
    n_val = min(168, len(X_tr) // 10)
    X_tr2, X_va = X_tr.iloc[:-n_val], X_tr.iloc[-n_val:]
    y_tr2, y_va = y_tr.iloc[:-n_val], y_tr.iloc[-n_val:]

    best_iter = n_estimators
    if algo == "lgbm":
        import lightgbm as lgb
        m_es = _build_rec_model(algo, params, n_estimators)
        m_es.fit(X_tr2, y_tr2, eval_set=[(X_va, y_va)],
                 callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                             lgb.log_evaluation(-1)])
        best_iter = int(getattr(m_es, "best_iteration_", None) or n_estimators)
        print(f"      ES best_iter={best_iter}", flush=True)
    elif algo == "xgb":
        m_es = _build_rec_model(algo, params, n_estimators)
        m_es.set_params(early_stopping_rounds=es_rounds)
        m_es.fit(X_tr2, y_tr2, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = int(getattr(m_es, "best_iteration", None) or n_estimators)
        print(f"      ES best_iter={best_iter}", flush=True)

    model = _build_rec_model(algo, params, best_iter)
    model.fit(X_tr2, y_tr2)

    if with_ss and ss_n_iter > 0:
        rng = np.random.default_rng(42)
        epsilons = np.linspace(ss_eps[0], ss_eps[1], ss_n_iter)
        for i, eps in enumerate(epsilons, 1):
            X_aug = _daily_ss_round(model, X_tr2, y_tr2, eps, rng)
            m_new = _build_rec_model(algo, params, best_iter)
            m_new.fit(X_aug, y_tr2)
            model = m_new
            print(f"      SS iter {i}/{ss_n_iter} (ε={eps:.2f})", flush=True)

    return model, best_iter


def _predict_ol_daily_chunks(
    model,
    df_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    feature_cols: List[str],
) -> np.ndarray:
    """Day-by-day (24h) closed-loop recursive prediction."""
    all_preds = np.full(len(test_index), np.nan)
    cfg = OpenLoopConfig()
    dates = sorted(set(t.date() for t in test_index))
    for d in dates:
        chunk_mask = pd.DatetimeIndex([t for t in test_index if t.date() == d])
        if len(chunk_mask) == 0:
            continue
        chunk_preds = recursive_predict_openloop(
            model, df_full, chunk_mask, feature_cols, cfg
        )
        for ti, t in enumerate(test_index):
            if t.date() == d:
                pos = list(chunk_mask).index(t)
                all_preds[ti] = chunk_preds[pos]
    return all_preds


# ── Per-month Optuna for Rec ──────────────────────────────────────────────────
def _suggest_algo_params(trial, algo: str) -> Tuple[dict, int]:
    """Suggest hyperparameters for LGBM or XGB."""
    if algo == "lgbm":
        n_est = trial.suggest_int("n_estimators", 300, 1500)
        params = {
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 32, 128),
            "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 50),
        }
    elif algo == "xgb":
        n_est = trial.suggest_int("n_estimators", 300, 1200)
        params = {
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    else:
        raise ValueError(f"Unknown algo: {algo}")
    return params, n_est


def _cv_score_daily_rec(
    algo: str,
    params: dict,
    n_estimators: int,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    df_full: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 2,
    fold_days: int = 28,
    with_ss: bool = False,
) -> float:
    """
    Rolling-origin CV for Rec daily 24h prediction.
    Uses last n_splits * fold_days as validation set.
    Returns mean MAE across folds.
    """
    scores = []
    total_val_h = n_splits * fold_days * 24
    if len(X_full) < total_val_h + 1000:
        return 1e9  # not enough data

    # Build folds from end of training data
    fold_ends = []
    for i in range(n_splits):
        fold_end_idx = len(X_full) - i * fold_days * 24
        fold_start_idx = fold_end_idx - fold_days * 24
        fold_ends.append((fold_start_idx, fold_end_idx))

    for fold_start, fold_end in fold_ends:
        if fold_start < 500:
            continue
        X_cv_train = X_full.iloc[:fold_start]
        y_cv_train = y_full.iloc[:fold_start]
        fold_index = X_full.index[fold_start:fold_end]
        y_cv_val   = y_full.iloc[fold_start:fold_end].to_numpy(float)

        if len(y_cv_val) == 0 or len(X_cv_train) < 500:
            continue

        try:
            model, _ = _retrain_rec_from_params(
                algo=algo, params=params, n_estimators=n_estimators,
                X_tr=X_cv_train, y_tr=y_cv_train,
                with_ss=with_ss, es_rounds=50,
            )
            preds = _predict_ol_daily_chunks(model, df_full, fold_index, feature_cols)
            fold_mae = mae(y_cv_val, preds)
            if np.isfinite(fold_mae):
                scores.append(fold_mae)
        except Exception:
            pass

    return float(np.mean(scores)) if scores else 1e9


def _load_or_run_optuna(
    algo: str,
    task: str,
    month_code: str,
    variant: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    df_full: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 15,
    with_ss: bool = False,
    fallback_params_path: Optional[Path] = None,
) -> Tuple[dict, int]:
    """
    Load cached per-month params or run inline Optuna.
    Cache location: models/{algo}_{task}_mr_{month_code}_{variant}_params.json
    Returns (params_dict, n_estimators).
    """
    cache_path = MODELS_DIR / f"{algo}_{task}_mr_{month_code}_{variant}_params.json"

    # 1. Try cached per-month params
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        n_est = int(raw.pop("n_estimators", 1000))
        print(f"      Loaded cached params: {cache_path.name}")
        return raw, n_est

    # 2. For Dec: use standard fallback params (already tuned)
    if fallback_params_path is not None and fallback_params_path.exists():
        with open(fallback_params_path, encoding="utf-8") as f:
            raw = json.load(f)
        n_est = int(raw.pop("n_estimators", 1000))
        print(f"      Using standard params: {fallback_params_path.name}")
        # Cache it with month-specific name too
        raw_save = dict(raw); raw_save["n_estimators"] = n_est
        cache_path.write_text(json.dumps(raw_save, ensure_ascii=False), encoding="utf-8")
        return raw, n_est

    # 3. Run inline Optuna
    print(f"      Running Optuna ({n_trials} trials, n_splits=2) ...", flush=True)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("      [WARN] optuna not installed — using default params")
        return _default_params(algo), 800

    def objective(trial):
        params, n_est = _suggest_algo_params(trial, algo)
        score = _cv_score_daily_rec(
            algo=algo, params=params, n_estimators=n_est,
            X_full=X_tr, y_full=y_tr, df_full=df_full,
            feature_cols=feature_cols,
            n_splits=2, fold_days=28, with_ss=with_ss,
        )
        return score

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best = study.best_params
    n_est = int(best.pop("n_estimators", 800))
    cv_score = study.best_value
    print(f"      Optuna done — CV MAE={cv_score:.3f}, n_est={n_est}", flush=True)

    # Cache
    save_dict = dict(best); save_dict["n_estimators"] = n_est
    cache_path.write_text(json.dumps(save_dict, ensure_ascii=False), encoding="utf-8")
    print(f"      Cached: {cache_path.name}", flush=True)
    return best, n_est


def _default_params(algo: str) -> dict:
    if algo == "lgbm":
        return {"learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_samples": 20}
    return {"learning_rate": 0.05, "max_depth": 5, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3}


# Rec variants config
# fallback_key: key to locate standard Dec params (algo_{mode}_{task}_openloop_daily_optuna_params.json)
# LSTM αφαιρέθηκε — μόνο LGBM/XGB variants
REC_VARIANTS = [
    {"name": "Rec-LGBM-DO-MR",    "algo": "lgbm", "with_ss": False, "variant_key": "do"},
    {"name": "Rec-LGBM-SS-DO-MR", "algo": "lgbm", "with_ss": True,  "variant_key": "ss_do"},
    {"name": "Rec-XGB-DO-MR",     "algo": "xgb",  "with_ss": False, "variant_key": "do"},
    {"name": "Rec-XGB-SS-DO-MR",  "algo": "xgb",  "with_ss": True,  "variant_key": "ss_do"},
]


# ── Seasonal profile baseline ─────────────────────────────────────────────────
def _seasonal_profile_preds(train_y: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    prof = (
        pd.DataFrame({"y": train_y.values,
                      "dow": train_y.index.dayofweek,
                      "hour": train_y.index.hour}, index=train_y.index)
        .groupby(["dow", "hour"])["y"].mean()
    )
    global_mean = float(train_y.mean())
    return np.array([
        float(prof.get((int(ts.dayofweek), int(ts.hour)), global_mean))
        for ts in test_index
    ], dtype=float)


# ── Printing helpers ──────────────────────────────────────────────────────────
def _hdr(cols, widths):
    parts = [c.ljust(w) if i == 0 else c.rjust(w)
             for i, (c, w) in enumerate(zip(cols, widths))]
    line = "  ".join(parts)
    print(line)
    print("-" * len(line))

def _row(vals, widths):
    parts = [str(vals[0]).ljust(widths[0])] + [
        str(v).rjust(w) for v, w in zip(vals[1:], widths[1:])
    ]
    print("  ".join(parts))


# ── Main evaluation function ──────────────────────────────────────────────────
def evaluate_monthly_retrain(
    mode: str,
    task: str,
    strategies: List[str],
    save_json: bool = False,
    out_suffix: str = "",
    n_trials_rec: int = 15,
    rec_variants: Optional[List[str]] = None,  # None = all 4
) -> None:
    unit = "€/MWh" if task == "price" else "MW"

    # Standard Dec fallback params (per algo)
    dec_params = {
        "lgbm_do":  MODELS_DIR / f"lgbm_{mode}_{task}_openloop_daily_optuna_params.json",
        "xgb_do":   MODELS_DIR / f"xgb_{mode}_{task}_openloop_daily_optuna_params.json",
        "lgbm_ss_do": MODELS_DIR / f"lgbm_{mode}_{task}_openloop_daily_ss_optuna_params.json",
        "xgb_ss_do":  MODELS_DIR / f"xgb_{mode}_{task}_openloop_daily_ss_optuna_params.json",
    }

    # Filter Rec variants
    active_rec = REC_VARIANTS
    if rec_variants:
        rv_set = set(rec_variants)
        active_rec = [v for v in REC_VARIANTS if v["name"] in rv_set or
                      v["algo"] in rv_set]

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD MONTHLY RETRAIN | task={task.upper()} | strategies={strategies}")
    print(f"  Months: {[m['name'] for m in EVAL_MONTHS]}")
    print(f"  TF models : {[b[0] for b in TF_BUILDERS]}")
    if "rec" in strategies or "ol" in strategies:
        print(f"  Rec models: {[v['name'] for v in active_rec]}")
        print(f"  Rec Optuna: Dec=cached/standard | Jan/Feb=inline ({n_trials_rec} trials each)")
    print(f"  Terminology: TF=Teacher-Forcing(open-loop) | Rec=Recursive(closed-loop)")
    print(f"{'='*72}\n")

    # Load full dataset once
    df = load_processed(mode, task=task)
    y_full_series = df["y"].astype(float)

    # Accumulate predictions over all months
    all_timestamps: List[pd.Timestamp] = []
    all_actuals:    List[float] = []
    month_preds:    Dict[str, List[float]] = {}
    month_results:  List[Dict] = []

    for mi, mdef in enumerate(EVAL_MONTHS):
        mname      = mdef["name"]
        mcode      = mdef["code"]
        train_end  = mdef["train_end"]
        test_start = mdef["test_start"]
        test_end   = mdef["test_end"]
        is_dec     = (mcode == "dec2025")

        print(f"\n{'─'*60}")
        print(f"  Month {mi+1}/3: {mname} | train_end={train_end}")
        print(f"{'─'*60}")

        df_train, df_test = split_time_series(
            df, mode=mode,
            train_end=train_end, test_start=test_start, test_end=test_end,
        )
        X_train, y_train_arr = make_xy(df_train)
        X_test,  y_test_arr  = make_xy(df_test)
        feature_cols = list(X_train.columns)
        test_index   = df_test.index
        y_train_s    = df_train["y"].astype(float)

        print(f"  Train rows: {len(X_train):,}  |  Test hours: {len(test_index)}")
        print(f"  Features:   {len(feature_cols)}")

        all_timestamps.extend(list(test_index))
        all_actuals.extend(list(y_test_arr))

        month_m: Dict[str, np.ndarray] = {}

        # ── TF (Teacher-Forcing) ─────────────────────────────────────────────
        if "tf" in strategies or "cl" in strategies:
            print(f"\n  [TF] Training {len(TF_BUILDERS)} Teacher-Forcing models...")
            for name, builder in TF_BUILDERS:
                print(f"    → {name} ...", end=" ", flush=True)
                try:
                    m = builder(task)
                    m.fit(X_train, y_train_arr)
                    preds = _fnp(m.predict(X_test))
                    month_m[name] = preds
                    print(f"MAE={mae(y_test_arr, preds):.3f}", flush=True)
                except Exception as e:
                    print(f"FAILED: {e}", flush=True)

        # ── Rec (Recursive/Closed-Loop) ─────────────────────────────────────
        if "rec" in strategies or "ol" in strategies:
            print(f"\n  [Rec] {mname} — training {len(active_rec)} Recursive variants...")
            for rv in active_rec:
                rname    = rv["name"]
                algo     = rv["algo"]
                with_ss  = rv["with_ss"]
                vkey     = f"{algo}_{rv['variant_key']}"
                dec_fallback = dec_params.get(vkey)

                print(f"    → {rname} ...", flush=True)

                # ── LSTM: train from scratch (no Optuna) ──
                if algo == "lstm":
                    try:
                        lstm_model = _LSTMWrapper(
                            seq_len=24, hidden_size=64, n_layers=2,
                            max_epochs=50, lr=1e-3, batch_size=512, patience=5,
                        )
                        lstm_model.fit(X_train, y_train_arr)
                        preds_rec = _predict_ol_daily_chunks(
                            lstm_model, df, test_index, feature_cols
                        )
                        month_m[rname] = preds_rec
                        print(f"      → {rname}  MAE={mae(y_test_arr, preds_rec):.3f}",
                              flush=True)
                    except Exception as e:
                        print(f"      [FAIL] {rname}: {e}", flush=True)
                    continue  # skip Optuna logic for LSTM

                # ── LGBM/XGB: Get params (cached / Dec fallback / inline Optuna) ──
                try:
                    params, n_est = _load_or_run_optuna(
                        algo=algo, task=task, month_code=mcode,
                        variant=rv["variant_key"],
                        X_tr=X_train, y_tr=y_train_s.reindex(X_train.index),
                        df_full=df, feature_cols=feature_cols,
                        n_trials=n_trials_rec,
                        with_ss=with_ss,
                        fallback_params_path=dec_fallback if is_dec else None,
                    )
                    model, best_iter = _retrain_rec_from_params(
                        algo=algo, params=params, n_estimators=n_est,
                        X_tr=X_train,
                        y_tr=y_train_s.reindex(X_train.index),
                        with_ss=with_ss, es_rounds=100,
                    )
                    preds_rec = _predict_ol_daily_chunks(
                        model, df, test_index, feature_cols
                    )
                    month_m[rname] = preds_rec
                    print(f"      → {rname}  MAE={mae(y_test_arr, preds_rec):.3f}",
                          flush=True)
                except Exception as e:
                    print(f"      [FAIL] {rname}: {e}", flush=True)

        # ── Baselines ────────────────────────────────────────────────────────
        for lag, bname in [(1, "Naive-1"), (24, "Naive-24")]:
            bpreds = y_full_series.shift(lag).reindex(test_index).to_numpy(dtype=float)
            month_m[bname] = bpreds
        sp = _seasonal_profile_preds(y_train_s, test_index)
        month_m["Seasonal Profile"] = sp

        # ── Ensembles ────────────────────────────────────────────────────────
        tf_names = {b[0] for b in TF_BUILDERS}
        tf_only  = [(n, p) for n, p in month_m.items() if n in tf_names]
        if len(tf_only) >= 2:
            mae_vals = [(n, p, mae(y_test_arr, p)) for n, p in tf_only
                        if np.isfinite(mae(y_test_arr, p))]
            if len(mae_vals) >= 2:
                sorted_tf = sorted(mae_vals, key=lambda x: x[2])
                ws = np.array([1.0/m for _, _, m in sorted_tf])
                ws /= ws.sum()
                ens_all = sum(w * p for w, (_, p, _) in zip(ws, sorted_tf))
                month_m["Ensemble-TF-All (1/MAE)"] = ens_all

                top3 = sorted_tf[:min(3, len(sorted_tf))]
                ws3 = np.array([1.0/m for _, _, m in top3])
                ws3 /= ws3.sum()
                ens_b3 = sum(w * p for w, (_, p, _) in zip(ws3, top3))
                month_m["Ensemble-TF-Best3 (1/MAE)"] = ens_b3
                names3 = "+".join(n.split("-")[1] for n, _, _ in top3)
                print(f"\n  [ENS] TF Ensemble-Best3 ({names3}): "
                      f"MAE={mae(y_test_arr, ens_b3):.3f}", flush=True)

        # Ensemble over ALL models (TF + Rec, excluding baselines)
        all_ml = [(n, p) for n, p in month_m.items()
                  if n not in BASELINE_NAMES
                  and not n.startswith("Ensemble")]
        if len(all_ml) >= 2:
            mae_all = [(n, p, mae(y_test_arr, p)) for n, p in all_ml
                       if np.isfinite(mae(y_test_arr, p))]
            if len(mae_all) >= 2:
                sorted_all = sorted(mae_all, key=lambda x: x[2])
                ws_a = np.array([1.0/m for _, _, m in sorted_all])
                ws_a /= ws_a.sum()
                ens_combo = sum(w * p for w, (_, p, _) in zip(ws_a, sorted_all))
                month_m["Ensemble-All (1/MAE)"] = ens_combo
                print(f"  [ENS] All-models Ensemble: "
                      f"MAE={mae(y_test_arr, ens_combo):.3f}", flush=True)

        # ── Store this month's predictions ────────────────────────────────
        for mn, preds in month_m.items():
            if mn not in month_preds:
                month_preds[mn] = []
            month_preds[mn].extend(list(preds))

        month_results.append({
            "name":   mname,
            "n_test": len(y_test_arr),
            "y_test": y_test_arr,
            "preds":  dict(month_m),
        })

    # ── Combined Q1 results ───────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  COMBINED RESULTS | task={task.upper()} | Q1 2026 (Dec+Jan+Feb)")
    print(f"  Total test hours: {len(all_actuals)}")
    print(f"{'='*72}")

    y_combined  = np.array(all_actuals, dtype=float)
    active_models = {n: np.array(v, dtype=float) for n, v in month_preds.items()
                     if len(v) == len(y_combined)}

    W = [30, 14, 14, 10]
    COLS = ["Model", f"MAE ({unit})", f"RMSE ({unit})", "sMAPE (%)"]
    _hdr(COLS, W)

    sorted_combined = sorted(active_models.items(),
                              key=lambda x: mae(y_combined, x[1]))
    for mname_s, preds in sorted_combined:
        m_val = mae(y_combined, preds)
        r_val = rmse(y_combined, preds)
        s_val = smape(y_combined, preds)
        _row([mname_s, f"{m_val:.3f}", f"{r_val:.3f}", f"{s_val:.3f}"], W)
    print()

    # ── Per-month breakdown ───────────────────────────────────────────────────
    print("  PER-MONTH BREAKDOWN")
    print()
    ml_models = [(n, p) for n, p in sorted_combined if n not in BASELINE_NAMES]
    header_parts = ["  Month          "] + [f"{n[:16]:>16}" for n, _ in ml_models]
    print("  ".join(header_parts))
    print("  " + "-" * (16 + 18 * len(ml_models)))

    for mres in month_results:
        yt = mres["y_test"]
        row_parts = [f"  {mres['name']:<14}"]
        for mname_m, _ in ml_models:
            if mname_m in mres["preds"]:
                row_parts.append(f"{mae(yt, mres['preds'][mname_m]):>16.3f}")
            else:
                row_parts.append(f"{'—':>16}")
        print("  ".join(row_parts))
    print()

    winner_name, winner_preds = sorted_combined[0]
    print(f"  🏆 WINNER: {winner_name} "
          f"(MAE={mae(y_combined, winner_preds):.3f} {unit})")
    print(f"{'='*72}\n")

    if save_json:
        _save_json_mr(task, all_timestamps, y_combined,
                      active_models, out_suffix, strategies)


def _save_json_mr(
    task: str,
    timestamps: List,
    y_combined: np.ndarray,
    active_models: Dict[str, np.ndarray],
    out_suffix: str,
    strategies: List[str],
) -> None:
    unit = "€/MWh" if task == "price" else "MW"
    dates  = [str(ts) for ts in timestamps]
    actual = [float(v) for v in y_combined]

    series: Dict[str, list] = {}
    for mname, preds in active_models.items():
        series[mname] = [float(v) if np.isfinite(v) else None for v in preds]

    metrics = []
    for mname, preds in sorted(active_models.items(),
                                key=lambda x: mae(y_combined, x[1])):
        m = mae(y_combined, preds)
        if np.isfinite(m):
            metrics.append({
                "Model": mname,
                "Type":  "baseline" if mname in BASELINE_NAMES else "ml",
                "MAE":   round(m, 4),
                "RMSE":  round(rmse(y_combined, preds), 4),
                "sMAPE": round(smape(y_combined, preds), 4),
            })

    strat_str = "-".join(
        ("TF" if s in ("tf", "cl") else "Rec" if s in ("rec", "ol") else s.upper())
        for s in strategies
    )
    out = {
        "strategy": f"monthly_retrain_{strat_str}",
        "task":     task,
        "period":   "Q1-2026-monthly-retrain",
        "dates":    dates,
        "actual":   actual,
        "series":   series,
        "metrics":  metrics,
        "unit":     unit,
    }

    fname = f"dashboard_data_hourly_{task}_monthly_retrain{out_suffix}.json"
    out_path = BASE_DIR / fname
    out_path.write_text(json.dumps(_fix_nan(out), ensure_ascii=False), encoding="utf-8")
    print(f"✅ JSON saved: {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Walk-forward monthly retraining evaluation (comprehensive v2)"
    )
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task",       choices=["price", "load"], default="price")
    p.add_argument("--strategies", type=str, default="tf,rec",
                   help="Comma-separated: tf,rec (or cl,ol for legacy). "
                        "tf=Teacher-Forcing, rec=Recursive Closed-Loop.")
    p.add_argument("--save_json",  action="store_true")
    p.add_argument("--out_suffix", type=str, default="",
                   help="Extra suffix appended to JSON filename")
    p.add_argument("--n_trials_rec", type=int, default=15,
                   help="Optuna trials per month for Rec variants (Jan/Feb only). Default=15")
    p.add_argument("--rec_only", type=str, default=None,
                   help="Comma-separated Rec variant names to include (default: all 4). "
                        "E.g. 'Rec-LGBM-DO-MR,Rec-LGBM-SS-DO-MR'")
    args = p.parse_args()

    strats = [s.strip().lower() for s in args.strategies.split(",")]
    rv = [s.strip() for s in args.rec_only.split(",")] if args.rec_only else None

    evaluate_monthly_retrain(
        mode=args.mode,
        task=args.task,
        strategies=strats,
        save_json=args.save_json,
        out_suffix=args.out_suffix,
        n_trials_rec=args.n_trials_rec,
        rec_variants=rv,
    )


if __name__ == "__main__":
    main()
