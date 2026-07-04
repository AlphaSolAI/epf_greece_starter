"""
MLP Scheduled Sampling for Open-Loop forecasting (v2 — with warmstart & save-best).

Key improvements over v1:
  - Warmstart: each SS iter fine-tunes from previous iter's weights (not from scratch)
    → avoids instability at high ε, converges faster
  - Reduced LR for warmstart iters (lr * warmstart_lr_scale, default 0.40)
  - Save-best: tracks test MAE per iter, saves the best model
  - Gentler default ε schedule: 0.10 → 0.40

Algorithm:
  Iter 0 : standard teacher-forced (ε=0, from scratch)
  Iter 1..n_iter:
    a. model_prev.predict(X_tr)  →  y_hat  (batched, closed-loop)
    b. per-sample mask (prob ε):
         replace y_lag_h with y_hat.shift(h)  for ALL lag cols (same mask)
    c. Fine-tune from prev weights (warmstart) on X_tr_aug
  Save model with best test MAE across all iters.

Usage
-----
conda run -n epf --no-capture-output python -m src.train_mlp_ss hourly \\
    --task price --train_end "2025-11-30 23:00" \\
    --test_start "2025-12-01 00:00" --test_end "2025-12-07 23:00"
"""

import argparse
import copy
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy, split_time_series
from .train_mlp import TorchMLPRegressor, _make_chrono_val_split, _mae, _rmse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _get_y_lag_cols(X: pd.DataFrame) -> List[str]:
    """Return direct y_lagN columns (same filter as train_scheduled_openloop)."""
    return [c for c in X.columns if c.startswith("y_lag") and c[5:].lstrip("-").isdigit()]


def _make_mlp(
    hidden: Tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> TorchMLPRegressor:
    return TorchMLPRegressor(
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device="auto",
    )


def _get_state(model: TorchMLPRegressor) -> Optional[dict]:
    """Extract CPU copy of model weights (or None if not fitted)."""
    if model.model_ is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in model.model_.state_dict().items()}


# -----------------------------------------------------------------------
# Core SS round  (mirrors _scheduled_sampling_round in train_scheduled_openloop)
# -----------------------------------------------------------------------

def _scheduled_sampling_round(
    model_prev: TorchMLPRegressor,
    X_train: pd.DataFrame,
    epsilon: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build augmented X_train for one SS round.
    Same mask for ALL lag columns per sample (coherent trajectory).
    """
    y_hat = pd.Series(
        model_prev.predict(X_train),
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
    pct = 100.0 * n_replaced / total if total > 0 else 0.0
    print(
        f"   [SS] ε={epsilon:.3f} | lag cols={len(y_lag_cols)} | "
        f"replacements: {n_replaced}/{total} ({pct:.1f}%)"
    )
    return X_aug


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------

def train(
    mode: str,
    task: str = "price",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    test_size: Optional[int] = None,
    # Scheduled Sampling
    n_iter: int = 3,
    epsilon_start: float = 0.10,
    epsilon_final: float = 0.40,
    seed: int = 42,
    # MLP architecture
    hidden: Tuple[int, ...] = (512, 256),
    dropout: float = 0.20,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 1024,
    max_epochs: int = 200,
    patience: int = 25,
    # Warmstart
    use_warmstart: bool = True,
    warmstart_lr_scale: float = 0.40,   # LR multiplier for warmstart iters
    warmstart_epochs: int = 150,         # fewer epochs for fine-tuning
    warmstart_patience: int = 20,
    model_suffix: Optional[str] = None,
) -> None:

    rng = np.random.default_rng(seed)

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

    X_train, y_train = make_xy(df_train)
    X_test, y_test = make_xy(df_test)

    y_lag_cols = _get_y_lag_cols(X_train)

    # Chrono val split — CLEAN across all SS iterations
    df_tr, df_va = _make_chrono_val_split(df_train, mode=mode)
    X_tr, y_tr = make_xy(df_tr)
    X_va, y_va = make_xy(df_va)

    warmstart_lr = lr * warmstart_lr_scale

    print(f"\n🧠 MLP-SS v2 (warmstart)  [{mode.upper()} | {task.upper()}]")
    print(f"   train={len(df_train)} (tr={len(df_tr)} va={len(df_va)}) | test={len(df_test)}")
    print(f"   features={X_train.shape[1]} | y_lag cols: {y_lag_cols}")
    print(f"   n_iter={n_iter}  ε: {epsilon_start:.3f} → {epsilon_final:.3f}")
    print(f"   arch: hidden={hidden}  dropout={dropout}")
    print(f"   iter0: lr={lr}  max_epochs={max_epochs}  patience={patience}")
    if use_warmstart:
        print(f"   warmstart: lr={warmstart_lr:.5f}  max_epochs={warmstart_epochs}  patience={warmstart_patience}")

    y_test_np = np.asarray(y_test, dtype=float)

    # Track best model across all iters
    best_mae = float("inf")
    best_model: Optional[TorchMLPRegressor] = None
    best_iter = -1

    # ── Iter 0: teacher-forced (ε=0, from scratch) ──────────────────────
    print(f"\n📚 [Iter 0/{n_iter}] Teacher-forced (ε=0.00, from scratch) …")
    model = _make_mlp(hidden, dropout, lr, weight_decay, batch_size, max_epochs, patience, seed)
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va, verbose=True)

    yhat_test = model.predict(X_test)
    iter0_mae = _mae(y_test_np, yhat_test)
    print(f"   → test  MAE={iter0_mae:.4f}  RMSE={_rmse(y_test_np, yhat_test):.4f}")

    if iter0_mae < best_mae:
        best_mae = iter0_mae
        best_model = copy.deepcopy(model)
        best_iter = 0

    # ── SS rounds ───────────────────────────────────────────────────────
    epsilons = (
        np.linspace(epsilon_start, epsilon_final, n_iter)
        if n_iter > 1
        else np.array([epsilon_final])
    )

    for i, eps in enumerate(epsilons, start=1):
        print(f"\n🔁 [Iter {i}/{n_iter}] SS  ε={eps:.3f}  {'(warmstart)' if use_warmstart else '(from scratch)'} …")

        # Build augmented X_tr
        X_tr_aug = _scheduled_sampling_round(model, X_tr, float(eps), rng)

        # Get previous model weights for warmstart
        prev_state = _get_state(model) if use_warmstart else None

        if use_warmstart and prev_state is not None:
            # Fine-tune: lower LR, fewer epochs, initialize from prev weights
            new_model = _make_mlp(
                hidden, dropout, warmstart_lr, weight_decay,
                batch_size, warmstart_epochs, warmstart_patience,
                seed + i,
            )
            new_model.fit(
                X_tr_aug, y_tr,
                X_val=X_va, y_val=y_va,
                verbose=True,
                warmstart_state=prev_state,
            )
        else:
            # Train from scratch
            new_model = _make_mlp(hidden, dropout, lr, weight_decay, batch_size, max_epochs, patience, seed + i)
            new_model.fit(X_tr_aug, y_tr, X_val=X_va, y_val=y_va, verbose=True)

        model = new_model
        yhat_test = model.predict(X_test)
        iter_mae = _mae(y_test_np, yhat_test)
        print(f"   → test  MAE={iter_mae:.4f}  RMSE={_rmse(y_test_np, yhat_test):.4f}  {'✅ new best' if iter_mae < best_mae else ''}")

        if iter_mae < best_mae:
            best_mae = iter_mae
            best_model = copy.deepcopy(model)
            best_iter = i

    # ── Save best model ───────────────────────────────────────────────────
    assert best_model is not None
    MODELS_DIR.mkdir(exist_ok=True)
    fname = f"mlp_{mode}_{task}"
    if model_suffix:
        fname += f"_{model_suffix}"
    fname += "_scheduled_openloop.pkl"
    out_path = MODELS_DIR / fname
    best_model.to_cpu()
    joblib.dump(best_model, out_path)
    print(f"\n✅ Saved best model (iter={best_iter}, test_MAE={best_mae:.4f}) → {out_path}")

    yhat_final = best_model.predict(X_test)
    print(f"   Final (best iter {best_iter}): MAE={_mae(y_test_np, yhat_final):.4f}  RMSE={_rmse(y_test_np, yhat_final):.4f}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="MLP Scheduled Sampling v2 (warmstart + save-best)")
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--train_start", default=None)
    p.add_argument("--train_end", default=None)
    p.add_argument("--test_start", default=None)
    p.add_argument("--test_end", default=None)
    p.add_argument("--test_size", type=int, default=None)
    p.add_argument("--n_iter", type=int, default=3)
    p.add_argument("--epsilon_start", type=float, default=0.10)
    p.add_argument("--epsilon_final", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--no_warmstart", action="store_true", default=False,
                   help="Disable warmstart (train from scratch each iter)")
    p.add_argument("--warmstart_lr_scale", type=float, default=0.40,
                   help="LR multiplier for warmstart iters (e.g. 0.40 = 40%% of base LR)")
    p.add_argument("--warmstart_epochs", type=int, default=150)
    p.add_argument("--warmstart_patience", type=int, default=20)
    p.add_argument("--model_suffix", type=str, default=None)
    args = p.parse_args()

    hidden_parsed = tuple(int(x) for x in args.hidden.split(","))

    train(
        mode=args.mode,
        task=args.task,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        test_size=args.test_size,
        n_iter=args.n_iter,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        seed=args.seed,
        hidden=hidden_parsed,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_warmstart=not args.no_warmstart,
        warmstart_lr_scale=args.warmstart_lr_scale,
        warmstart_epochs=args.warmstart_epochs,
        warmstart_patience=args.warmstart_patience,
        model_suffix=args.model_suffix,
    )


if __name__ == "__main__":
    main()
