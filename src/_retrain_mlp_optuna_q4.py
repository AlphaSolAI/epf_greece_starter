"""
Retrain MLP-Optuna models with Sep30 2025 cutoff for Q4 evaluation.
Uses existing Optuna best-params JSON files.
"""
import json
import sys
import joblib
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

from .split_utils import load_processed, make_xy, split_time_series
from .train_mlp import TorchMLPRegressor


def retrain_mlp_optuna(task: str, train_end: str = "2025-09-30 23:00"):
    params_path = MODELS_DIR / f"mlp_hourly_{task}_optuna_params.json"
    out_path    = MODELS_DIR / f"mlp_hourly_{task}_optuna.pkl"

    if not params_path.exists():
        print(f"[SKIP] {params_path.name} not found")
        return

    params = json.loads(params_path.read_text(encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"  RETRAIN MLP-Optuna | task={task.upper()} | train_end={train_end}")
    print(f"  hidden={params['hidden']}, dropout={params['dropout']:.3f}")
    print(f"  lr={params['lr']:.6f}, weight_decay={params['weight_decay']:.6f}")
    print(f"  batch_size={params['batch_size']}")
    print(f"{'='*60}\n")

    df = load_processed("hourly", task=task)
    df_train, _ = split_time_series(df, mode="hourly", train_end=train_end, test_size=168)
    X_train, y_train = make_xy(df_train)

    print(f"[INFO] train rows={len(X_train):,} | features={len(X_train.columns)}")

    mlp = TorchMLPRegressor(
        hidden         = tuple(params["hidden"]),
        dropout        = float(params["dropout"]),
        lr             = float(params["lr"]),
        weight_decay   = float(params["weight_decay"]),
        batch_size     = int(params["batch_size"]),
        max_epochs     = 500,
        patience       = 30,
        seed           = 42,
        device         = "auto",
    )
    mlp.fit(X_train.values, np.asarray(y_train, dtype=float))

    joblib.dump(mlp, out_path)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task",      choices=["price", "load", "both"], default="both")
    p.add_argument("--train_end", default="2025-09-30 23:00")
    args = p.parse_args()

    tasks = ["price", "load"] if args.task == "both" else [args.task]
    for t in tasks:
        retrain_mlp_optuna(t, train_end=args.train_end)
