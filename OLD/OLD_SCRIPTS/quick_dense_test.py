"""Quick test: LGBM MIMO Dense vs No-Dense for Dec-2025 price."""
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from src.split_utils import load_processed, make_xy, split_time_series

HORIZON = 24
TRAIN_END  = "2025-11-30 23:00"
TEST_START = "2025-12-01 00:00"
TEST_END   = "2025-12-31 23:00"

def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m]-b[m])))

def build_xy(df_train, dense=False):
    X, _ = make_xy(df_train)
    y = df_train["y"].astype(float)
    if dense:
        existing = {int(c[5:]) for c in X.columns if c.startswith("y_lag") and c[5:].isdigit()}
        for lag in range(1, 24):
            if lag not in existing:
                X[f"y_lag{lag}"] = y.shift(lag).reindex(X.index)
    Y_df = pd.concat([y.shift(-k).rename(f"y_t+{k}") for k in range(1, HORIZON+1)], axis=1)
    joined = X.join(Y_df, how="inner").dropna(subset=list(Y_df.columns))
    X_al = joined[X.columns].select_dtypes(include=[np.number]).fillna(0.0).astype(np.float32)
    Y = joined[list(Y_df.columns)].to_numpy(dtype=np.float32)
    return X_al, Y

def build_x_full(df_full, dense=False):
    X, _ = make_xy(df_full)
    y = df_full["y"].astype(float)
    if dense:
        existing = {int(c[5:]) for c in X.columns if c.startswith("y_lag") and c[5:].isdigit()}
        for lag in range(1, 24):
            if lag not in existing:
                X[f"y_lag{lag}"] = y.shift(lag).reindex(X.index)
    return X.select_dtypes(include=[np.number]).fillna(0.0)

def train_lgbm_mimo(X_tr, Y_tr):
    base = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=1, verbosity=-1)
    model = MultiOutputRegressor(base, n_jobs=4)
    with joblib.parallel_backend("threading", n_jobs=4):
        model.fit(X_tr, Y_tr)
    return model

def predict(model, feat_cols, X_full, test_index):
    freq = pd.tseries.frequencies.to_offset("h")
    origins = [pd.Timestamp(TEST_START) - freq + pd.Timedelta(days=d)
               for d in range(len(test_index)//HORIZON + 1)]
    preds = np.full(len(test_index), np.nan)
    t_map = {ts: i for i, ts in enumerate(test_index)}
    for origin in origins:
        if origin not in X_full.index:
            cands = X_full.index[X_full.index <= origin]
            if len(cands) == 0: continue
            origin = cands[-1]
        x_row = np.array([float(X_full.loc[origin, c]) if c in X_full.columns else 0.0
                          for c in feat_cols], dtype=np.float32).reshape(1, -1)
        with joblib.parallel_backend("threading", n_jobs=2):
            yhat = np.asarray(model.predict(x_row)[0], dtype=float)[:HORIZON]
        pred_idx = pd.date_range(origin + freq, periods=HORIZON, freq=freq)
        for k, ts in enumerate(pred_idx):
            if ts in t_map:
                preds[t_map[ts]] = yhat[k]
    return preds

print("Loading data...", flush=True)
df = load_processed("hourly", task="price")
df_tr, df_te = split_time_series(df, mode="hourly", train_end=TRAIN_END,
                                  test_start=TEST_START, test_end=TEST_END)
_, y_test = make_xy(df_te)
y_test_arr = np.asarray(y_test, dtype=float)
test_index  = df_te.index

print(f"Train: {len(df_tr):,} rows | Test: {len(test_index)} hours", flush=True)

# ── No-Dense ───────────────────────────────────────────────────────────────────
print("\n[NO-DENSE] Building features...", flush=True)
X_tr_nd, Y_tr_nd = build_xy(df_tr, dense=False)
X_full_nd = build_x_full(df.loc[:TEST_END], dense=False)
feat_nd = list(X_tr_nd.columns)
print(f"  Features: {len(feat_nd)}", flush=True)

print("[NO-DENSE] Training LGBM MIMO...", flush=True)
m_nd = train_lgbm_mimo(X_tr_nd.to_numpy(), Y_tr_nd)
preds_nd = predict(m_nd, feat_nd, X_full_nd, test_index)
mae_nd = mae(y_test_arr, preds_nd)
print(f"  LGBM MIMO No-Dense  MAE = {mae_nd:.3f} €/MWh", flush=True)

# ── Dense ──────────────────────────────────────────────────────────────────────
print("\n[DENSE] Building features...", flush=True)
X_tr_d, Y_tr_d = build_xy(df_tr, dense=True)
X_full_d = build_x_full(df.loc[:TEST_END], dense=True)
feat_d = list(X_tr_d.columns)
print(f"  Features: {len(feat_d)}", flush=True)

print("[DENSE] Training LGBM MIMO...", flush=True)
m_d = train_lgbm_mimo(X_tr_d.to_numpy(), Y_tr_d)
preds_d = predict(m_d, feat_d, X_full_d, test_index)
mae_d = mae(y_test_arr, preds_d)
print(f"  LGBM MIMO Dense     MAE = {mae_d:.3f} €/MWh", flush=True)

print(f"\n{'='*50}")
print(f"  LGBM MIMO No-Dense : MAE = {mae_nd:.3f} €/MWh")
print(f"  LGBM MIMO Dense    : MAE = {mae_d:.3f} €/MWh")
winner = "Dense" if mae_d < mae_nd else "No-Dense"
diff = abs(mae_d - mae_nd)
print(f"  Winner: {winner} (διαφορά: {diff:.3f})")
print(f"{'='*50}")
