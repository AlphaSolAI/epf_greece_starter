import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# UTF-8 για να μην βλέπεις ιερογλυφικά
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
HOURLY_PATH = ROOT / "data" / "processed" / "hourly.parquet"


# -----------------------------------------------------------
# 1. Features από το hourly.parquet
# -----------------------------------------------------------
def build_features():
    """
    Φορτώνει το hourly.parquet και φτιάχνει features για
    1-hour-ahead forecast: y(t) = price(t), features μέχρι t-1.
    """
    df = pd.read_parquet(HOURLY_PATH)

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    df = df.sort_index()

    # calendar
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # price lags / rolling
    df["price_lag1"] = df["price"].shift(1)
    df["price_lag2"] = df["price"].shift(2)
    df["price_lag3"] = df["price"].shift(3)
    df["price_lag24"] = df["price"].shift(24)
    df["price_lag48"] = df["price"].shift(48)
    df["price_lag168"] = df["price"].shift(168)

    df["price_rollmean_24"] = df["price"].rolling(24, min_periods=12).mean()
    df["price_rollstd_24"] = df["price"].rolling(24, min_periods=12).std()
    df["price_rollmean_168"] = df["price"].rolling(168, min_periods=84).mean()
    df["price_rollstd_168"] = df["price"].rolling(168, min_periods=84).std()

    # load lags / rolling
    df["load_lag1"] = df["load"].shift(1)
    df["load_lag2"] = df["load"].shift(2)
    df["load_lag3"] = df["load"].shift(3)
    df["load_lag24"] = df["load"].shift(24)
    df["load_lag48"] = df["load"].shift(48)
    df["load_lag168"] = df["load"].shift(168)

    df["load_rollmean_24"] = df["load"].rolling(24, min_periods=12).mean()
    df["load_rollstd_24"] = df["load"].rolling(24, min_periods=12).std()
    df["load_rollmean_168"] = df["load"].rolling(168, min_periods=84).mean()
    df["load_rollstd_168"] = df["load"].rolling(168, min_periods=84).std()

    df["load_to_price_ratio"] = df["load_lag1"] / (df["price_lag1"] + 1e-6)
    df["load_diff"] = df["load_lag1"] - df["load_lag24"]

    # drop NaNs από lags/rollings
    df = df.dropna()

    y = df["price"].copy()

    base_feats = [
        "hour",
        "dow",
        "month",
        "is_weekend",
        "price_lag1",
        "price_lag2",
        "price_lag3",
        "price_lag24",
        "price_lag48",
        "price_lag168",
        "price_rollmean_24",
        "price_rollstd_24",
        "price_rollmean_168",
        "price_rollstd_168",
    ]

    load_feats = [
        "load",
        "load_lag1",
        "load_lag2",
        "load_lag3",
        "load_lag24",
        "load_lag48",
        "load_lag168",
        "load_rollmean_24",
        "load_rollstd_24",
        "load_rollmean_168",
        "load_rollstd_168",
        "load_to_price_ratio",
        "load_diff",
    ]

    X_base = df[base_feats].copy()
    X_with_load = df[base_feats + load_feats].copy()

    return df.index, X_base, X_with_load, y


def time_split(index, test_days=60):
    """
    Time-based split: test = τελευταίες test_days μέρες.
    """
    last_time = index.max()
    cutoff = last_time - pd.Timedelta(days=test_days)
    train_mask = index <= cutoff
    test_mask = index > cutoff
    return train_mask, test_mask, cutoff


# -----------------------------------------------------------
# 2. Helper για training / evaluation
# -----------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    smape = (
        np.mean(2.0 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test) + 1e-6))
        * 100.0
    )

    print(f"\n----- {label} -----")
    print(f"MAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"sMAPE= {smape:.2f}%")

    return {"model": label, "mae": mae, "rmse": rmse, "smape": smape}


# -----------------------------------------------------------
# 3. Main: πολλά μοντέλα, με & χωρίς load
# -----------------------------------------------------------
def main():
    print("=== HOURLY Load Ablation – Multiple Models ===")
    print(f"Loading hourly dataset from: {HOURLY_PATH}")

    index, X_base, X_with_load, y = build_features()
    print(f"Total hourly samples (after lags/rollings): {len(y)}")
    print(f"Base features shape   : {X_base.shape}")
    print(f"With-load features    : {X_with_load.shape}")

    train_mask, test_mask, cutoff = time_split(index, test_days=60)
    print(
        f"Train end: {cutoff} | "
        f"Train size: {train_mask.sum()}, Test size: {test_mask.sum()}"
    )

    Xb_tr, Xb_te = X_base[train_mask], X_base[test_mask]
    Xl_tr, Xl_te = X_with_load[train_mask], X_with_load[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    results = []

    # --------- XGBoost ----------
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    results.append(evaluate_model(xgb, Xb_tr, y_tr, Xb_te, y_te, "XGB_NO_LOAD"))

    xgb_load = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    results.append(evaluate_model(xgb_load, Xl_tr, y_tr, Xl_te, y_te, "XGB_WITH_LOAD"))

    # --------- Random Forest ----------
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=1,
        n_jobs=4,
        random_state=42,
    )
    results.append(evaluate_model(rf, Xb_tr, y_tr, Xb_te, y_te, "RF_NO_LOAD"))

    rf_load = RandomForestRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=1,
        n_jobs=4,
        random_state=42,
    )
    results.append(evaluate_model(rf_load, Xl_tr, y_tr, Xl_te, y_te, "RF_WITH_LOAD"))

    # --------- SVR (με scaling) ----------
    svr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, epsilon=1.0, kernel="rbf")),
        ]
    )
    results.append(
        evaluate_model(svr_pipe, Xb_tr, y_tr, Xb_te, y_te, "SVR_NO_LOAD")
    )

    svr_pipe_load = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, epsilon=1.0, kernel="rbf")),
        ]
    )
    results.append(
        evaluate_model(svr_pipe_load, Xl_tr, y_tr, Xl_te, y_te, "SVR_WITH_LOAD")
    )

    # --------- MLP (με scaling) ----------
    mlp_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    results.append(
        evaluate_model(mlp_pipe, Xb_tr, y_tr, Xb_te, y_te, "MLP_NO_LOAD")
    )

    mlp_pipe_load = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    learning_rate_init=0.001,
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    results.append(
        evaluate_model(mlp_pipe_load, Xl_tr, y_tr, Xl_te, y_te, "MLP_WITH_LOAD")
    )

    # --------- Summary table ----------
    print("\n============================================================")
    print("FINAL COMPARISON (HOURLY – All Models)")
    print("============================================================")
    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'sMAPE':>8}")
    for r in results:
        print(
        f"{r['model']:<15} {r['mae']:8.3f} {r['rmse']:8.3f} {r['smape']:7.2f}%"
        )


if __name__ == "__main__":
    main()
