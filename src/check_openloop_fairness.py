import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Union
import types
import re

import numpy as np
import pandas as pd
import joblib

from .split_utils import load_processed, make_xy, split_time_series
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# --- Wrapper for unpickling (models saved with this wrapper) ---
class ResidualAddBaselineWrapper:
    def __init__(self, model, baseline_col: str, feature_names: list):
        self.model = model
        self.baseline_col = baseline_col
        self.feature_names = list(feature_names)
        self.baseline_idx = self.feature_names.index(baseline_col)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        residual_hat = np.asarray(self.model.predict(X), dtype=float).reshape(-1)
        if isinstance(X, pd.DataFrame):
            base = X[self.baseline_col].to_numpy(dtype=float).reshape(-1)
        else:
            X = np.asarray(X)
            base = X[:, self.baseline_idx].astype(float).reshape(-1)
        return base + residual_hat


def _register_unpickle_aliases():
    alias_names = [
        "src.model_wrappers",
        "src.train_xgb_openloop",
        "src.eval_openloop",
        "src.check_openloop_fairness",
    ]
    for name in alias_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], "ResidualAddBaselineWrapper", ResidualAddBaselineWrapper)


def _call_recursive(model, df_full, test_index, feature_cols, cfg):
    res = recursive_predict_openloop(
        model=model,
        df_full=df_full,
        test_index=test_index,
        feature_cols=feature_cols,
        config=cfg,
    )
    return np.asarray(res, dtype=float)


def _detect_lag_cols(feature_cols: List[str]) -> List[str]:
    out = []
    for c in feature_cols:
        cl = c.lower()
        m1 = re.search(r"(?:^|_)y_?lag_?(\d+)$", cl)
        m2 = re.search(r"(?:^|_)lag_?y_?(\d+)$", cl)
        if m1 or m2:
            out.append(c)
    return sorted(out)


def _detect_roll_cols(feature_cols: List[str]) -> List[str]:
    out = []
    for c in feature_cols:
        cl = c.lower()
        m = re.search(r"(?:^|_)y_?roll_?(\d+)$", cl)
        if m:
            out.append(c)
    return sorted(out)


def main():
    print("=== check_openloop_fairness START ===", flush=True)

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hourly"])
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--model_key", choices=["xgb", "xgb_test", "xgb_backup"], default="xgb_test")
    p.add_argument("--test_start", type=str, required=True)
    p.add_argument("--test_end", type=str, required=True)
    p.add_argument("--poison_value", type=float, default=123456.0)
    args = p.parse_args()

    print(f"[INFO] mode={args.mode} task={args.task} model_key={args.model_key}", flush=True)

    df = load_processed(args.mode, task=args.task)
    df_train, df_test = split_time_series(
        df,
        mode=args.mode,
        test_size=None,
        train_start=None,
        train_end=None,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    X_train, _ = make_xy(df_train)
    feature_cols = list(X_train.columns)
    test_index = df_test.index

    print(f"[INFO] test window: {test_index.min()} .. {test_index.max()} (n={len(test_index)})", flush=True)
    print(f"[INFO] num features: {len(feature_cols)}", flush=True)

    if args.model_key == "xgb_test":
        mp = MODELS_DIR / f"xgb_{args.mode}_{args.task}_openloop_resid_naive1.pkl"
    elif args.model_key == "xgb_backup":
        mp = MODELS_DIR / "backup.pkl"
    else:
        mp = MODELS_DIR / f"xgb_{args.mode}_{args.task}_openloop.pkl"

    print(f"[INFO] model_file={mp}", flush=True)
    if not mp.exists():
        raise SystemExit(f"[ERROR] Missing model file: {mp}")

    _register_unpickle_aliases()
    model = joblib.load(mp)

    cfg = OpenLoopConfig(y_floor=None)

    print("[INFO] Running reference prediction...", flush=True)
    y_ref = _call_recursive(model, df, test_index, feature_cols, cfg)

    lag_cols = _detect_lag_cols(feature_cols)
    roll_cols = _detect_roll_cols(feature_cols)
    overwritten_expected = set(lag_cols + roll_cols)

    other_y_cols = sorted([c for c in feature_cols if c.lower().startswith("y_") and c not in overwritten_expected])

    print(f"[INFO] lag cols detected: {len(lag_cols)}", flush=True)
    print(f"[INFO] roll cols detected: {len(roll_cols)}", flush=True)
    if roll_cols:
        print(f"[INFO] roll cols: {roll_cols}", flush=True)
    print(f"[INFO] other y_* cols (not lag/roll): {len(other_y_cols)}", flush=True)
    if other_y_cols:
        print(f"[INFO] sample other y cols: {other_y_cols[:20]}", flush=True)

    poison_idx = test_index[1:] if len(test_index) > 1 else test_index

    def summarize(tag, y_alt):
        d = np.abs(y_ref - y_alt)
        print(f"\n--- {tag} ---", flush=True)
        print(f"mean_abs_diff={float(np.nanmean(d)):.6f}", flush=True)
        print(f"max_abs_diff ={float(np.nanmax(d)):.6f}", flush=True)

    # TEST A1: poison LAG cols (should be ~0 if lag overwrite works)
    print("[INFO] Test A1: poisoning LAG cols inside test (except first step)...", flush=True)
    df_a1 = df.copy()
    if lag_cols:
        df_a1.loc[poison_idx, lag_cols] = args.poison_value
    y_a1 = _call_recursive(model, df_a1, test_index, feature_cols, cfg)
    summarize("A1_POISON_LAGS", y_a1)

    # TEST A2: poison ROLL cols (should be ~0 if roll overwrite works)
    print("[INFO] Test A2: poisoning ROLL cols inside test (except first step)...", flush=True)
    df_a2 = df.copy()
    if roll_cols:
        df_a2.loc[poison_idx, roll_cols] = args.poison_value
    y_a2 = _call_recursive(model, df_a2, test_index, feature_cols, cfg)
    summarize("A2_POISON_ROLLS", y_a2)

    # TEST B: poison other y_* cols (if big => more y-derived leakage)
    print("[INFO] Test B: poisoning OTHER y_* cols inside test (except first step)...", flush=True)
    df_b = df.copy()
    if other_y_cols:
        df_b.loc[poison_idx, other_y_cols] = args.poison_value
    y_b = _call_recursive(model, df_b, test_index, feature_cols, cfg)
    summarize("B_POISON_OTHER_Y_COLS", y_b)

    # TEST C: poison TRUE y inside horizon (anchor concern)
    # If recursion uses df_full['y'] inside horizon (not allowed), this will change predictions.
    print("[INFO] Test C: poisoning df_full['y'] inside test (except first step)...", flush=True)
    df_c = df.copy()
    if "y" in df_c.columns:
        df_c.loc[poison_idx, "y"] = args.poison_value
    y_c = _call_recursive(model, df_c, test_index, feature_cols, cfg)
    summarize("C_POISON_TRUE_Y", y_c)

    print("\n=== INTERPRETATION (strict openloop) ===", flush=True)
    print("A1_POISON_LAGS:", flush=True)
    print("  ~0 => y_lag* are overwritten from running predictions (good).", flush=True)
    print("A2_POISON_ROLLS:", flush=True)
    print("  ~0 => y_roll* are overwritten from running predictions (good).", flush=True)
    print("  big => you are still using precomputed roll features from df_full inside horizon (openloop leakage).", flush=True)
    print("B_POISON_OTHER_Y_COLS:", flush=True)
    print("  big => you have additional y-derived columns leaking (need to overwrite or remove).", flush=True)
    print("C_POISON_TRUE_Y:", flush=True)
    print("  ~0 => no direct use of actual df_full['y'] inside horizon beyond the initial anchor (good).", flush=True)
    print("  big => direct/indirect dependency on actual y inside horizon (NOT allowed).", flush=True)

    print("=== check_openloop_fairness END ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(str(e), flush=True)
        raise
    except Exception:
        print("❌ Exception occurred:\n", flush=True)
        traceback.print_exc()
        raise
