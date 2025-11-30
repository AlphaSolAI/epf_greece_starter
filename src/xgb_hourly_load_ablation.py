import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
import joblib


# Path to daily.parquet (με HEnEx + features + load)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE σε %."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)

    mask = denom != 0
    if not np.any(mask):
        return 0.0

    return float(np.mean(diff[mask] / denom[mask]) * 100.0)


def train_and_eval(use_load: bool):
    """Train & test XGBoost με ή χωρίς ΟΛΑ τα features που σχετίζονται με 'load'."""
    print("\n" + "=" * 60)
    print(f"XGBoost experiment | use_load = {use_load}")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"daily.parquet not found at: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    # Ensure sorted by date index
    df = df.sort_index()

    if "y" not in df.columns:
        raise ValueError("Column 'y' (price) not found in daily.parquet")

    X_all = df.drop(columns=["y"])
    y = df["y"].astype("float64").values

    if use_load:
        # Χρησιμοποιούμε ΟΛΑ τα features, συμπεριλαμβανομένων των load, load_lag*, load_* κ.λπ.
        X = X_all
    else:
        # Πετάμε ΚΑΘΕ στήλη που περιέχει "load" στο όνομα (load, load_lag*, load_ratio, load_diff, κ.λπ.)
        cols = [c for c in X_all.columns if "load" not in c.lower()]
        X = X_all[cols]

    print(f"Total samples: {len(df)}, features: {X.shape[1]}")
    print("Feature columns:", list(X.columns))

    # Train / test split: last 60 days as test
    test_size = 60
    if len(df) <= test_size + 50:
        raise ValueError("Not enough data for 60-day test split.")

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Base model
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    # Small randomized search around base hyperparameters
    param_dist = {
        "n_estimators": [300, 500, 700],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    print("Fitting XGBoost (RandomizedSearchCV)...")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    # Test evaluation
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    # RMSE χωρίς το 'squared' argument (συμβατό με παλιότερο sklearn)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    smape_val = smape(y_test, y_pred)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = "xgb_with_load.pkl" if use_load else "xgb_no_load.pkl"
    out_path = MODELS_DIR / model_name
    joblib.dump(best, out_path)
    print(f"Saved model to: {out_path}")

    return {
        "use_load": use_load,
        "mae": mae,
        "rmse": rmse,
        "smape": smape_val,
    }


def main():
    print("=== XGBoost Load Ablation Experiment ===")

    # 1) Χωρίς ΚΑΝΕΝΑ load-related feature
    res_no_load = train_and_eval(use_load=False)

    # 2) Με ΟΛΑ τα load-related features
    res_with_load = train_and_eval(use_load=True)

    # Summary table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON (XGBoost)")
    print("=" * 60)

    rows = []
    for res in [res_no_load, res_with_load]:
        label = "XGB_NO_LOAD" if not res["use_load"] else "XGB_WITH_LOAD"
        rows.append(
            [
                label,
                round(res["mae"], 3),
                round(res["rmse"], 3),
                round(res["smape"], 3),
            ]
        )

    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'sMAPE (%)':>11}")
    for row in rows:
        print(f"{row[0]:<15} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>11.3f}")


if __name__ == "__main__":
    main()
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# UTF-8 για να μην βλέπεις ιερογλυφικά
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
HOURLY_PATH = ROOT / "data" / "processed" / "hourly.parquet"


def build_features():
    """
    Φορτώνει το hourly.parquet και φτιάχνει features για
    1-hour-ahead forecast: y(t) = price(t), features μέχρι t-1.
    """
    df = pd.read_parquet(HOURLY_PATH)
    # index είναι ήδη timestamp
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    # ------------- calendar features ------------- #
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # ------------- price lags / rolling ------------- #
    df["price_lag1"] = df["price"].shift(1)
    df["price_lag2"] = df["price"].shift(2)
    df["price_lag3"] = df["price"].shift(3)
    df["price_lag24"] = df["price"].shift(24)
    df["price_lag48"] = df["price"].shift(48)
    df["price_lag168"] = df["price"].shift(168)  # 7 μέρες

    df["price_rollmean_24"] = df["price"].rolling(24, min_periods=12).mean()
    df["price_rollstd_24"] = df["price"].rolling(24, min_periods=12).std()
    df["price_rollmean_168"] = df["price"].rolling(168, min_periods=84).mean()
    df["price_rollstd_168"] = df["price"].rolling(168, min_periods=84).std()

    # ------------- load lags / rolling ------------- #
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

    # target: price(t)
    df = df.dropna()
    y = df["price"].copy()

    # βάση features (χωρίς load)
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
    Χωρίζει σε train / test με βάση το χρόνο (όχι shuffle).
    Test = τελευταίες test_days μέρες.
    """
    last_time = index.max()
    cutoff = last_time - pd.Timedelta(days=test_days)
    train_mask = index <= cutoff
    test_mask = index > cutoff
    return train_mask, test_mask, cutoff


def train_eval_xgb(X_train, y_train, X_test, y_test, label):
    """
    Εκπαίδευση ενός XGBRegressor με σταθερά, λογικά hyperparams
    (όχι RandomizedSearch για να μην περιμένεις αιώνες).
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

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

    return {"label": label, "mae": mae, "rmse": rmse, "smape": smape}


def main():
    print("=== XGBoost HOURLY Load Ablation ===")
    print(f"Loading hourly dataset from: {HOURLY_PATH}")

    index, X_base, X_with_load, y = build_features()
    print(f"Total hourly samples (after lags/rollings): {len(y)}")
    print(f"Base features shape   : {X_base.shape}")
    print(f"With-load features    : {X_with_load.shape}")

    # time-based split
    train_mask, test_mask, cutoff = time_split(index, test_days=60)
    print(f"Train end: {cutoff} | Train size: {train_mask.sum()}, Test size: {test_mask.sum()}")

    X_base_train, X_base_test = X_base[train_mask], X_base[test_mask]
    X_wl_train, X_wl_test = X_with_load[train_mask], X_with_load[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # 1) Χωρίς load features
    res_no = train_eval_xgb(X_base_train, y_train, X_base_test, y_test, "XGB_NO_LOAD")

    # 2) Με load features
    res_wl = train_eval_xgb(X_wl_train, y_train, X_wl_test, y_test, "XGB_WITH_LOAD")

    # Summary
    print("\n============================================================")
    print("FINAL COMPARISON (HOURLY XGBoost)")
    print("============================================================")
    print(
        "Model           MAE     RMSE    sMAPE\n"
        f"{res_no['label']:<14} {res_no['mae']:.3f}  {res_no['rmse']:.3f}  {res_no['smape']:.2f}%\n"
        f"{res_wl['label']:<14} {res_wl['mae']:.3f}  {res_wl['rmse']:.3f}  {res_wl['smape']:.2f}%"
    )


if __name__ == "__main__":
    main()
