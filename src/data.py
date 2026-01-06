import argparse
import sys
import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

# Ensure utf-8 on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PRICES_DIR = RAW_DIR / "dam_prices"
LOAD_DIR = RAW_DIR / "load"
GEN_DIR = RAW_DIR / "generation"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------------
# Helper functions for ENTSO-E CSVs
# ---------------------------------------------------------------------
def _read_entsoe_csv(path: Path) -> pd.DataFrame:
    """Read ENTSO-E CSV trying ';' then ',' as separator."""
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",")
    except Exception:
        df = pd.read_csv(path)
    return df


def _find_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ("mtu", "time", "date", "datetime")):
            return col
    return df.columns[0]


def _find_price_column(df: pd.DataFrame) -> str:
    # Day-ahead price in EUR/MWh
    candidates = [
        c
        for c in df.columns
        if "price" in str(c).lower()
        and ("eur" in str(c).lower() or "€/mwh" in str(c).lower())
    ]
    if candidates:
        return candidates[0]

    candidates = [c for c in df.columns if "price" in str(c).lower()]
    if candidates:
        return candidates[0]

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]


def _find_actual_load_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "actual total load" in str(c).lower()]
    if candidates:
        return candidates[0]

    candidates = [c for c in df.columns if "load" in str(c).lower()]
    if candidates:
        return candidates[0]

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]


def _find_generation_value_column(df: pd.DataFrame) -> str:
    candidates = [
        c
        for c in df.columns
        if "generation" in str(c).lower() and "mw" in str(c).lower()
    ]
    if candidates:
        return candidates[0]

    candidates = [c for c in df.columns if "generation" in str(c).lower()]
    if candidates:
        return candidates[0]

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]


# ---------------------------------------------------------------------
# Loaders: price, load, generation
# ---------------------------------------------------------------------
def load_entsoe_prices() -> pd.Series:
    """Load day-ahead electricity prices (hourly) from ENTSO-E GUI_ENERGY_PRICES_*.csv."""
    if not PRICES_DIR.exists():
        raise FileNotFoundError(f"Prices directory not found: {PRICES_DIR}")
    files = sorted(PRICES_DIR.glob("GUI_ENERGY_PRICES_*.csv"))
    if not files:
        raise FileNotFoundError(f"No GUI_ENERGY_PRICES_*.csv files in {PRICES_DIR}")

    frames = []
    for f in files:
        df = _read_entsoe_csv(f)
        time_col = _find_time_column(df)
        price_col = _find_price_column(df)

        # Typical ENTSO-E time format: "01/01/2015 00:00 - 01/01/2015 01:00"
        ts_start = df[time_col].astype(str).str.split(" - ").str[0]
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce")

        price = pd.to_numeric(
            df[price_col].astype(str).str.replace(",", ""), errors="coerce"
        )
        tmp = pd.DataFrame({"timestamp": dt, "price": price})
        tmp = tmp.dropna(subset=["timestamp", "price"])
        frames.append(tmp)

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    full = full.set_index("timestamp")
    full = full[~full.index.duplicated(keep="first")]
    return full["price"]


def load_entsoe_load() -> pd.Series | None:
    """Load Actual Total Load (MW) hourly from ENTSO-E GUI_TOTAL_LOAD_*.csv."""
    if not LOAD_DIR.exists():
        print("[INFO] LOAD_DIR does not exist; no load features.")
        return None
    files = sorted(LOAD_DIR.glob("*.csv"))
    if not files:
        print("[INFO] No load CSVs in LOAD_DIR.")
        return None

    frames = []
    for f in files:
        df = _read_entsoe_csv(f)
        time_col = _find_time_column(df)
        load_col = _find_actual_load_column(df)

        ts_start = df[time_col].astype(str).str.split(" - ").str[0]
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce")

        load_vals = pd.to_numeric(
            df[load_col].astype(str).str.replace(",", ""), errors="coerce"
        )
        tmp = pd.DataFrame({"timestamp": dt, "load": load_vals})
        tmp = tmp.dropna(subset=["timestamp", "load"])
        frames.append(tmp)

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    full = full.set_index("timestamp")
    full = full[~full.index.duplicated(keep="first")]
    return full["load"]


def load_entsoe_generation() -> pd.DataFrame | None:
    """Load Actual Generation per Production Type and aggregate to key fuels (hourly)."""
    if not GEN_DIR.exists():
        print("[INFO] GEN_DIR does not exist; no generation features.")
        return None
    files = sorted(GEN_DIR.glob("*.csv"))
    if not files:
        print("[INFO] No generation CSVs in GEN_DIR.")
        return None

    frames = []
    for f in files:
        df = _read_entsoe_csv(f)
        time_col = _find_time_column(df)

        type_col = None
        for c in df.columns:
            if "production type" in str(c).lower():
                type_col = c
                break
        if type_col is None:
            continue

        val_col = _find_generation_value_column(df)

        ts_start = df[time_col].astype(str).str.split(" - ").str[0]
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce")
        gen_vals = (
            pd.to_numeric(df[val_col].astype(str).str.replace(",", ""), errors="coerce")
            .fillna(0.0)
        )

        tmp = pd.DataFrame(
            {
                "timestamp": dt,
                "type": df[type_col].astype(str),
                "gen": gen_vals,
            }
        )
        tmp = tmp.dropna(subset=["timestamp"])
        frames.append(tmp)

    if not frames:
        return None

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["timestamp"])

    pivot = full.pivot_table(index="timestamp", columns="type", values="gen", aggfunc="sum")
    pivot = pivot.sort_index()
    pivot = pivot[~pivot.index.duplicated(keep="first")]

    idx = pivot.index

    def _col(name: str) -> pd.Series:
        return pivot[name] if name in pivot.columns else pd.Series(0.0, index=idx)

    gas = _col("Fossil Gas")
    lignite = _col("Fossil Brown coal/Lignite")
    hydro = (
        _col("Hydro Pumped Storage")
        + _col("Hydro Run-of-river and poundage")
        + _col("Hydro Water Reservoir")
    )
    solar = _col("Solar")
    wind_on = _col("Wind Onshore")
    wind_off = _col("Wind Offshore")

    gen_df = pd.DataFrame(
        {
            "gen_solar": solar,
            "gen_wind": wind_on.add(wind_off, fill_value=0.0),
            "gen_gas": gas,
            "gen_lignite": lignite,
            "gen_hydro": hydro,
        },
        index=idx,
    )

    gen_df = gen_df.fillna(0.0)
    return gen_df


# ---------------------------------------------------------------------
# Gas & CO2 from Investing CSVs (daily)
# ---------------------------------------------------------------------
def load_gas_daily() -> pd.Series | None:
    """Load Dutch TTF Natural Gas Futures (daily) from CSV under data/raw."""
    candidates = sorted(RAW_DIR.glob("*Natural Gas Futures Historical Data*.csv"))
    if not candidates:
        print("[INFO] No gas CSV found in RAW_DIR.")
        return None

    df = pd.read_csv(candidates[0])
    date_col = [c for c in df.columns if "date" in str(c).lower()][0]
    price_col = [c for c in df.columns if "price" in str(c).lower()][0]

    dates = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    prices = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")

    ser = pd.Series(prices.values, index=dates)
    ser = ser.dropna()
    ser = ser.sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


def load_co2_daily() -> pd.Series | None:
    """Load Carbon Emissions Futures (daily) from CSV under data/raw."""
    candidates = sorted(RAW_DIR.glob("*Carbon Emissions Futures Historical Data*.csv"))
    if not candidates:
        print("[INFO] No CO2 CSV found in RAW_DIR.")
        return None

    df = pd.read_csv(candidates[0])
    date_col = [c for c in df.columns if "date" in str(c).lower()][0]
    price_col = [c for c in df.columns if "price" in str(c).lower()][0]

    dates = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    prices = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")

    ser = pd.Series(prices.values, index=dates)
    ser = ser.dropna()
    ser = ser.sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def process_daily(
    price_hourly: pd.Series,
    load_hourly: pd.Series | None,
    gen_hourly: pd.DataFrame | None,
    gas_daily: pd.Series | None,
    co2_daily: pd.Series | None,
) -> pd.DataFrame:
    """Build daily dataset with price + load + generation + gas + CO2."""
    print("   -> Features: DAILY...")

    df_price = price_hourly.resample("D").mean().to_frame(name="y")

    # Load
    if load_hourly is not None:
        load_daily = load_hourly.resample("D").mean()
        df = df_price.join(load_daily.rename("load"), how="left")
        df["load"] = df["load"].interpolate().fillna(method="bfill").fillna(method="ffill")
        df["load_lag1"] = df["load"].shift(1)
        df["load_roll7"] = df["load"].rolling(7).mean()
        df["load_ratio"] = df["load"] / (df["load_roll7"] + 1e-9)
    else:
        df = df_price.copy()

    # Generation
    if gen_hourly is not None:
        gen_daily = gen_hourly.resample("D").mean()
        df = df.join(gen_daily, how="left")
        for c in gen_daily.columns:
            df[c] = df[c].fillna(0.0)
        if "load" in df.columns:
            df["residual_load"] = df["load"] - (
                df.get("gen_solar", 0.0) + df.get("gen_wind", 0.0)
            )

    # Gas & CO2
    if gas_daily is not None:
        gas_daily = gas_daily.sort_index()
        df = df.join(gas_daily.rename("gas_price"), how="left")
        df["gas_price"] = (
            df["gas_price"].interpolate().fillna(method="bfill").fillna(method="ffill")
        )
        df["gas_lag1"] = df["gas_price"].shift(1)
        df["gas_roll7"] = df["gas_price"].rolling(7).mean()

    if co2_daily is not None:
        co2_daily = co2_daily.sort_index()
        df = df.join(co2_daily.rename("co2_price"), how="left")
        df["co2_price"] = (
            df["co2_price"].interpolate().fillna(method="bfill").fillna(method="ffill")
        )
        df["co2_lag1"] = df["co2_price"].shift(1)
        df["co2_roll7"] = df["co2_price"].rolling(7).mean()

    # Calendar features
    df.index.name = "ds"
    gr_holidays = holidays.Greece()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_holiday"] = df.index.map(lambda x: int(x in gr_holidays))
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Price lags / rolling
    for lag in [1, 2, 7, 14]:
        df[f"y_lag{lag}"] = df["y"].shift(lag)
    for win in [7, 30]:
        df[f"y_roll{win}"] = df["y"].rolling(win).mean()

    # Drop rows before all exogenous are available
    df = df.dropna()
    return df


def process_hourly(
    price_hourly: pd.Series,
    load_hourly: pd.Series | None,
    gen_hourly: pd.DataFrame | None,
    gas_daily: pd.Series | None,
    co2_daily: pd.Series | None,
) -> pd.DataFrame:
    """Build hourly dataset with price + load + generation + gas + CO2."""
    print("   -> Features: HOURLY...")

    df = price_hourly.to_frame(name="y").copy()

    # Load
    if load_hourly is not None:
        df = df.join(load_hourly.rename("load"), how="left")
        df["load"] = df["load"].interpolate().fillna(method="bfill").fillna(method="ffill")
        df["load_lag24"] = df["load"].shift(24)

    # Generation
    if gen_hourly is not None:
        df = df.join(gen_hourly, how="left")
        for c in gen_hourly.columns:
            df[c] = df[c].fillna(0.0)
        if "load" in df.columns:
            df["residual_load"] = df["load"] - (
                df.get("gen_solar", 0.0) + df.get("gen_wind", 0.0)
            )

    # Gas & CO2 - broadcast daily values to hourly
    if gas_daily is not None:
        gas_daily = gas_daily.sort_index()
        gas_map = gas_daily.to_dict()
        df["gas_price"] = df.index.normalize().map(lambda d: gas_map.get(d, np.nan))
        df["gas_lag24"] = df["gas_price"].shift(24)

    if co2_daily is not None:
        co2_daily = co2_daily.sort_index()
        co2_map = co2_daily.to_dict()
        df["co2_price"] = df.index.normalize().map(lambda d: co2_map.get(d, np.nan))
        df["co2_lag24"] = df["co2_price"].shift(24)

    # Calendar
    gr_holidays = holidays.Greece()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_holiday"] = df.index.map(lambda x: int(x.date() in gr_holidays))
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # Price lags
    for lag in [24, 48, 168]:
        df[f"y_lag{lag}"] = df["y"].shift(lag)

    df = df.dropna()
    return df


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly"], help="Dataset mode")
    args = parser.parse_args()

    print(f"🔨 BUILDING DATASET ({args.mode.upper()})...")
    print(f"   -> Reading ENTSO-E prices from {PRICES_DIR}")
    price_hourly = load_entsoe_prices()
    load_hourly = load_entsoe_load()
    gen_hourly = load_entsoe_generation()
    gas_daily = load_gas_daily()
    co2_daily = load_co2_daily()

    if args.mode == "daily":
        df_final = process_daily(price_hourly, load_hourly, gen_hourly, gas_daily, co2_daily)
    else:
        df_final = process_hourly(price_hourly, load_hourly, gen_hourly, gas_daily, co2_daily)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.mode}.parquet"
    df_final.to_parquet(out_path, engine="fastparquet")
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
