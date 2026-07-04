import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

PRICES_DIR = RAW_DIR / "dam_prices"
LOAD_DIR = RAW_DIR / "entsoe_load"
GEN_DIR = RAW_DIR / "entsoe_generation"
GAS_DIR = RAW_DIR / "gas"
CO2_DIR = RAW_DIR / "co2"
ENTSOE_EXTRA_DIR = RAW_DIR / "entsoe_extra"
WEATHER_DIR = RAW_DIR / "weather"
HENEX_PREMARKET_DIR = RAW_DIR / "henex_premarket"

LOAD_FC_PATH = OUTPUT_DIR / "load_forecast_hourly.parquet"


def _read_parquet_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        return pd.read_parquet(path)


def _to_parquet_any(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, engine="fastparquet")
    except Exception:
        df.to_parquet(path)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ["ds", "timestamp", "date", "datetime", "time"]:
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], errors="coerce")
            out = out.dropna(subset=[c]).set_index(c)
            return out.sort_index()
    raise ValueError("No datetime index/column found.")


def _read_csv_any(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["ds", "timestamp", "date", "datetime", "time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).set_index(c)
            break
    df = ensure_datetime_index(df)
    return df


def _concat_series(series_list):
    if not series_list:
        return None
    s = pd.concat(series_list).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def load_prices_hourly() -> pd.Series:
    files = sorted(glob.glob(str(PRICES_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No price CSVs in {PRICES_DIR}")
    out = []
    for f in files:
        df = _read_csv_any(Path(f))
        col = None
        for cand in ["price", "dam_price", "y", "value"]:
            if cand in df.columns:
                col = cand
                break
        if col is None:
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            if num:
                col = num[0]
        if col is None:
            continue
        out.append(pd.to_numeric(df[col], errors="coerce").rename("price"))
    s = _concat_series(out)
    if s is None:
        raise ValueError("Could not load price series.")
    return s


def load_load_hourly() -> pd.Series:
    files = sorted(glob.glob(str(LOAD_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No load CSVs in {LOAD_DIR}")
    out = []
    for f in files:
        df = _read_csv_any(Path(f))
        col = None
        for cand in ["load", "y", "value"]:
            if cand in df.columns:
                col = cand
                break
        if col is None:
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            if num:
                col = num[0]
        if col is None:
            continue
        out.append(pd.to_numeric(df[col], errors="coerce").rename("load"))
    s = _concat_series(out)
    if s is None:
        raise ValueError("Could not load load series.")
    return s


def load_gen_hourly() -> pd.DataFrame | None:
    files = sorted(glob.glob(str(GEN_DIR / "*.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        df = _read_csv_any(Path(f))
        num = df.select_dtypes(include=[np.number]).copy()
        if num.empty:
            continue
        num = num.rename(columns=lambda c: f"gen_{c}" if not str(c).startswith("gen_") else str(c))
        dfs.append(num)
    if not dfs:
        return None
    out = pd.concat(dfs, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_gas_hourly() -> pd.Series | None:
    files = sorted(glob.glob(str(GAS_DIR / "*.csv")))
    if not files:
        return None
    out = []
    for f in files:
        df = _read_csv_any(Path(f))
        col = None
        for cand in ["gas_price", "price", "y", "value"]:
            if cand in df.columns:
                col = cand
                break
        if col is None:
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            if num:
                col = num[0]
        if col is None:
            continue
        out.append(pd.to_numeric(df[col], errors="coerce").rename("gas_price"))
    s = _concat_series(out)
    if s is None:
        return None
    return s.resample("H").ffill()


def load_co2_hourly() -> pd.Series | None:
    files = sorted(glob.glob(str(CO2_DIR / "*.csv")))
    if not files:
        return None
    out = []
    for f in files:
        df = _read_csv_any(Path(f))
        col = None
        for cand in ["co2_price", "price", "y", "value"]:
            if cand in df.columns:
                col = cand
                break
        if col is None:
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            if num:
                col = num[0]
        if col is None:
            continue
        out.append(pd.to_numeric(df[col], errors="coerce").rename("co2_price"))
    s = _concat_series(out)
    if s is None:
        return None
    return s.resample("H").ffill()


def load_entsoe_extra_hourly() -> pd.DataFrame | None:
    files = sorted(glob.glob(str(ENTSOE_EXTRA_DIR / "*.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        df = _read_csv_any(Path(f))
        num = df.select_dtypes(include=[np.number]).copy()
        if not num.empty:
            dfs.append(num)
    if not dfs:
        return None
    out = pd.concat(dfs, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_weather_hourly() -> pd.DataFrame | None:
    # prefer your known parquet
    p = WEATHER_DIR / "weather_gr_hourly.parquet"
    if p.exists():
        df = _read_parquet_any(p)
        df = ensure_datetime_index(df)
        return df.select_dtypes(include=[np.number]).copy()

    pqs = sorted(glob.glob(str(WEATHER_DIR / "*.parquet")))
    if pqs:
        df = _read_parquet_any(Path(pqs[0]))
        df = ensure_datetime_index(df)
        return df.select_dtypes(include=[np.number]).copy()

    csvs = sorted(glob.glob(str(WEATHER_DIR / "*.csv")))
    if csvs:
        df = _read_csv_any(Path(csvs[0]))
        return df.select_dtypes(include=[np.number]).copy()

    return None


def load_premarket_hourly() -> pd.DataFrame | None:
    cache = OUTPUT_DIR / "henex_premarket_hourly.parquet"
    if cache.exists():
        df = _read_parquet_any(cache)
        df = ensure_datetime_index(df)
        return df.select_dtypes(include=[np.number]).copy()

    files = sorted(glob.glob(str(HENEX_PREMARKET_DIR / "*.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        df = _read_csv_any(Path(f))
        num = df.select_dtypes(include=[np.number]).copy()
        if not num.empty:
            dfs.append(num)
    if not dfs:
        return None
    out = pd.concat(dfs, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = ensure_datetime_index(out)
    # cache
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _to_parquet_any(out, cache)
    return out


def load_load_forecast_hourly() -> pd.Series | None:
    if not LOAD_FC_PATH.exists():
        return None
    df = _read_parquet_any(LOAD_FC_PATH)
    df = ensure_datetime_index(df)
    col = None
    for cand in ["load_fc", "yhat", "y_pred", "pred", "forecast"]:
        if cand in df.columns:
            col = cand
            break
    if col is None:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num) == 1:
            col = num[0]
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce").rename("load_fc")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df = df.copy()
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    if "holiday" not in df.columns:
        df["holiday"] = 0
    if "is_holiday" not in df.columns:
        df["is_holiday"] = df["holiday"]
    return df


def build_hourly(task: str, include_price_feature_for_load: bool) -> pd.DataFrame:
    task = task.lower().strip()
    if task not in ("price", "load"):
        raise ValueError("task must be price or load")

    price = load_prices_hourly()
    load = load_load_hourly()
    gen = load_gen_hourly()
    gas = load_gas_hourly()
    co2 = load_co2_hourly()
    extra = load_entsoe_extra_hourly()
    weather = load_weather_hourly()
    premarket = load_premarket_hourly()
    load_fc = load_load_forecast_hourly()  # may be None

    # target
    y = price.rename("y") if task == "price" else load.rename("y")
    df = y.to_frame()
    df = add_time_features(df)

    # common joins
    if extra is not None:
        df = df.join(extra, how="left")
    if weather is not None:
        df = df.join(weather, how="left")
    if premarket is not None:
        df = df.join(premarket, how="left")

    # price task: market covariates
    if task == "price":
        if gas is not None:
            df = df.join(gas.rename("gas_price"), how="left")
        if co2 is not None:
            df = df.join(co2.rename("co2_price"), how="left")

    # load as exogenous for price (BUT LAGGED ONLY) + include load forecast as load_fc
    df = df.join(load.rename("load"), how="left")

    # load_fc is a day-ahead forecast → legitimate only for PRICE task.
    # For LOAD task it would be leakage (predicting the target itself).
    if task == "price" and load_fc is not None:
        df = df.join(load_fc.rename("load_fc"), how="left")

    # gen (BUT LAGGED ONLY)
    if gen is not None and not gen.empty:
        df = df.join(gen, how="left")

    # if load task and you want price as feature -> shift by 1
    if task == "load" and include_price_feature_for_load:
        df["price_feat"] = price.ffill().shift(1)

    # --------- CREATE LAGS (NO SAME-HOUR load/gen kept) ----------
    lags = [1, 2, 24, 48, 168]

    # target lags (closed-loop)
    for L in lags:
        df[f"y_lag{L}"] = df["y"].shift(L)
    df["y_roll24_mean"] = df["y"].shift(1).rolling(24).mean()
    df["y_roll24_std"] = df["y"].shift(1).rolling(24).std()

    # load lags
    for L in lags:
        df[f"load_lag{L}"] = df["load"].shift(L)

    # residual_load lags (computed from same-hour load/gen, then lagged; then drop contemporaneous)
    gen_cols = [c for c in df.columns if str(c).startswith("gen_")]
    if gen_cols:
        df["residual_load"] = df["load"] - df[gen_cols].sum(axis=1)
        for L in lags:
            df[f"residual_load_lag{L}"] = df["residual_load"].shift(L)

    # gen lags
    for c in gen_cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)

    # DROP contemporaneous load/gen/residual_load (availability leakage fix)
    drop_cols = ["load", "residual_load"] + gen_cols
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # load_fc: day-ahead load forecast (price task only).
    # The forecast file covers only the test period (168 h).
    # For training rows where load_fc is NaN, use load_lag24 as proxy
    # (best available substitute for a D-1 load forecast).
    # This avoids the entire training set being dropped by dropna().
    if "load_fc" in df.columns:
        if "load_lag24" in df.columns:
            df["load_fc"] = df["load_fc"].fillna(df["load_lag24"])
        else:
            df["load_fc"] = df["load_fc"].ffill()

    # Past-only fill for all features (no bfill)
    df = df.replace([np.inf, -np.inf], np.nan).ffill()

    # Drop rows with missing target or key lag features
    df = df.dropna(subset=["y", "y_lag1"])

    # final drop rows with any NaN (keeps training stable)
    df = df.dropna()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hourly"])
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--price", action="store_true")
    g.add_argument("--load", action="store_true")
    parser.add_argument("--include_price_feature_for_load", action="store_true")
    args = parser.parse_args()

    task = "price"
    if args.load:
        task = "load"
    elif args.price:
        task = "price"

    print(f"🔨 BUILDING DATASET (HOURLY, task={task}) ...")
    df = build_hourly(task=task, include_price_feature_for_load=bool(args.include_price_feature_for_load))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / ("hourly.parquet" if task == "price" else "hourly_load.parquet")
    _to_parquet_any(df, out_path)

    print(f"✅ Saved: {out_path}")
    print(f"[INFO] shape={df.shape} | cols={df.shape[1]}")


if __name__ == "__main__":
    main()
