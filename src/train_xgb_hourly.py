import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[1]
INP_PATH = BASE_DIR / "data" / "processed" / "hourly_features.parquet"


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask] / denom[mask]) * 100.0)


def main():
    if not INP_PATH.exists():
        raise FileNotFoundError(f"{INP_PATH} not found")

    df = pd.read_parquet(INP_PATH).sort_index()

    y = df["price"].values.astype("float64")
    X = df.drop(columns=["price"]).values

    n = len(df)
    test_horizon = 24 * 7  # 7 μέρες test
    if n <= test_horizon + 200:
        raise ValueError(f"Not enough hourly data (n={n}) for 7-day test split")

    X_train, X_test = X[:-test_horizon], X[-test_horizon:]
    y_train, y_test = y[:-test_horizon], y[-test_horizon:]

    print(f"Total hourly samples: {n}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    param_dist = {
        "n_estimators": [300, 500, 800],
        "max_depth": [4, 6, 8],
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
        verbose=1,
    )

    print("Fitting XGB hourly...")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # χωρίς squared arg
    s = smape(y_test, y_pred)

    print("\n=== Hourly XGBoost Results (last 7 days as test) ===")
    print(f"MAE = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    print(f"sMAPE = {s:.3f}%")

if __name__ == "__main__":
    main()
