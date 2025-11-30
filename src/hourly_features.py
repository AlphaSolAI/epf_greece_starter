import pandas as pd
import numpy as np
from pathlib import Path
import holidays


BASE_DIR = Path(__file__).resolve().parents[1]
INP_PATH = BASE_DIR / "data" / "processed" / "hourly_raw.parquet"
OUT_PATH = BASE_DIR / "data" / "processed" / "hourly_features.parquet"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    gr_holidays = holidays.country_holidays("GR")
    df["is_holiday"] = df.index.date.astype("O").map(lambda d: int(d in gr_holidays))

    # cyclical encoding example
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_lags_rolls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PRICE lags
    for h in [1, 24, 48, 168]:
        df[f"price_lag{h}"] = df["price"].shift(h)

    # LOAD lags
    for h in [1, 24, 48, 168]:
        df[f"load_lag{h}"] = df["load"].shift(h)

    # Rolling means / stds (π.χ. 24h, 168h)
    df["price_rollmean_24"] = df["price"].rolling(24).mean()
    df["price_rollstd_24"] = df["price"].rolling(24).std()
    df["load_rollmean_24"] = df["load"].rolling(24).mean()
    df["load_rollstd_24"] = df["load"].rolling(24).std()

    df["price_rollmean_168"] = df["price"].rolling(168).mean()
    df["load_rollmean_168"] = df["load"].rolling(168).mean()

    # Ratios
    df["load_ratio_24"] = df["load"] / (df["load_rollmean_24"] + 1e-6)
    df["price_ratio_24"] = df["price"] / (df["price_rollmean_24"] + 1e-6)

    return df


def main():
    if not INP_PATH.exists():
        raise FileNotFoundError(f"{INP_PATH} not found")

    df = pd.read_parquet(INP_PATH)
    # περιμένω df.index = ts, df.columns = ['price', 'load']
    df = df.sort_index()

    df = add_time_features(df)
    df = add_lags_rolls(df)

    # πετάμε τις πρώτες γραμμές όπου λείπουν lag/rolling
    df = df.dropna()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Saved hourly_features.parquet → {OUT_PATH}")
    print("Final columns:", list(df.columns))
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
