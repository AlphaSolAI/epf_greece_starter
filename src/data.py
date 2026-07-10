import argparse
import sys
import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PRICES_DIR = RAW_DIR / "dam_prices"
LOAD_DIR = RAW_DIR / "load"
GEN_DIR = RAW_DIR / "generation"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Optional extra covariates (produced by other scripts)
ENTSOE_EXTRA_PARQUET = OUTPUT_DIR / "entsoe_extra_hourly.parquet"

# Weather parquet: accept both names (no refetch needed)
WEATHER_HOURLY_PARQUET_PRIMARY = OUTPUT_DIR / "weather_hourly.parquet"
WEATHER_HOURLY_PARQUET_ALT = OUTPUT_DIR / "weather_gr_hourly.parquet"

# Load forecast parquet (produced by src.eval with --export_predictions)
LOAD_FORECAST_PARQUET = OUTPUT_DIR / "load_forecast_hourly.parquet"

# Vintage (D-1 lead-time) weather forecast, task=load only (GOALS.md G5) —
# produced by src.fetch_open_meteo_vintage (Open-Meteo Previous Runs API).
WEATHER_VINTAGE_PARQUET = OUTPUT_DIR / "weather_vintage_hourly.parquet"

# Cross-border DAM prices (produced by src.fetch_entsoe_xborder) — day-ahead-known
XBORDER_PARQUET = OUTPUT_DIR / "xborder_hourly.parquet"

ENTSOE_EXTRA_COLS = ["solar_fc_dayahead", "wind_onshore_fc_dayahead", "gen_fc_dayahead"]


# ----------------------------
# Helpers
# ----------------------------
def _read_entsoe_csv(path: Path) -> pd.DataFrame:
    # Windows + non-ASCII (ελληνικό) path στο working directory: το πέρασμα ενός
    # Path/str απευθείας στον pandas C/python parser σκάει με κρυπτικό
    # OSError("Invalid argument") σε ΟΛΟΥΣ τους engines. Το άνοιγμα του αρχείου
    # πρώτα με open() (που χειρίζεται σωστά το wide-char path) και το πέρασμα
    # του file handle στο pandas δουλεύει πάντα.
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            df = pd.read_csv(fh, sep=";")
        if df.shape[1] == 1:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
                df = pd.read_csv(fh, sep=",")
    except Exception:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            df = pd.read_csv(fh)
    return df


def _find_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ("mtu", "time", "date", "datetime")):
            return col
    return df.columns[0]


def _find_price_column(df: pd.DataFrame) -> str:
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


def _crop_to_first_complete_row(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    required = [c for c in required_cols if c in df.columns]
    if not required:
        return df
    mask = df[required].notna().all(axis=1)
    if not bool(mask.any()):
        raise ValueError(f"No row has all required features available: {required}")
    first_idx = df.index[int(np.argmax(mask.to_numpy()))]
    if first_idx != df.index[0]:
        print(f"[INFO] Cropping start -> {first_idx} (first complete row for {required})")
    return df.loc[first_idx:]


def _ensure_hourly_index(s: pd.Series) -> pd.Series:
    """Force timestamps to exact hours (floor) and aggregate duplicates by mean."""
    idx = pd.to_datetime(s.index, errors="coerce")
    idx = idx.floor("H")
    s2 = pd.Series(s.to_numpy(), index=idx).sort_index()
    s2 = s2[~s2.index.isna()]
    s2 = s2.groupby(level=0).mean()
    return s2


def _ensure_datetime_index_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for cand in ["timestamp", "ds", "date", "datetime", "time"]:
        if cand in df.columns:
            out = df.copy()
            out[cand] = pd.to_datetime(out[cand], errors="coerce")
            out = out.dropna(subset=[cand]).set_index(cand)
            return out
    return df


def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to UTC naive timestamps (hourly)."""
    df = df.copy()
    df = _ensure_datetime_index_df(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index is not DatetimeIndex.")
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    else:
        idx = pd.to_datetime(idx).tz_localize(None)
    df.index = idx.floor("H")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ---------------------------------------------------------------------------
# TIMEZONE ALIGNMENT FIX (2026-07-04) — βλ. ABLATION_PLAN.md §5.9.
# Κανονικό frame ΟΛΟΥ του pipeline: CET/CEST-naive — αυτό είναι το frame των
# price/load GUI exports ("MTU (CET/CEST)") που ορίζουν το index. Οι πηγές όμως
# έρχονται σε 3 διαφορετικά ρολόγια και ΠΡΕΠΕΙ να μετατραπούν στο join:
#   generation CSVs  -> Europe/Athens naive  (fetch_entsoe_generation, TZ=Athens) => -1h σταθερά
#   entsoe_extra fc  -> UTC                  (fetch_entsoe_dayahead)              => +1h/+2h (DST)
#   weather parquet  -> σταθερό UTC+1        (επιβεβαιωμένο εμπειρικά ανά έτος)   => +0h/+1h (DST)
# Η εμπειρική επαλήθευση (ηλιακά peaks ανά εποχή + corr fc↔gen actual):
# .claude/skills/energy-forecast/scripts/solar_shift_check.py — πρέπει peak_k=0 παντού.
# ---------------------------------------------------------------------------

def _utc_to_cet_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """UTC (tz-aware ή UTC-naive) -> CET/CEST-naive (το frame του index)."""
    df = df.copy()
    df = _ensure_datetime_index_df(df)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("Europe/Brussels").tz_localize(None)
    df.index = idx.floor("H")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fixed_utc1_to_cet_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Σταθερό UTC+1 naive -> CET/CEST-naive (μετατοπίζει +1h ΜΟΝΟ το καλοκαίρι)."""
    df = df.copy()
    df = _ensure_datetime_index_df(df)
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Etc/GMT-1").tz_localize(None)
    # Etc/GMT-1 = UTC+1 σταθερό (αντεστραμμένο πρόσημο POSIX), χωρίς DST -> καμία αμφισημία
    idx = idx.tz_localize("Etc/GMT-1").tz_convert("Europe/Brussels").tz_localize(None)
    df.index = idx.floor("H")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _best_shift_hours(base_idx: pd.DatetimeIndex, extra_idx: pd.DatetimeIndex) -> int:
    base_set = set(base_idx)
    best_h, best_overlap = 0, -1
    for h in range(-3, 4):
        shifted = extra_idx + pd.Timedelta(hours=h)
        overlap = sum((t in base_set) for t in shifted)
        if overlap > best_overlap:
            best_overlap = overlap
            best_h = h
    return best_h


def _align_and_join(base: pd.DataFrame, extra: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Join extra covariates to base, with timezone shift detection (-3..+3h)."""
    base2 = base.copy()
    base2.index = pd.to_datetime(base2.index, errors="coerce").floor("H")
    base2 = base2[~base2.index.isna()].sort_index()

    extra2 = _to_utc_naive_index(extra)

    h = _best_shift_hours(base2.index, extra2.index)
    if h != 0:
        before = len(base2.index.intersection(extra2.index))
        after = len(base2.index.intersection(extra2.index + pd.Timedelta(hours=h)))
        if after > before * 1.1:
            extra2.index = extra2.index + pd.Timedelta(hours=h)
            print(
                f"[INFO] Applied timezone alignment shift for {tag}: "
                f"extra_index += {h}h (overlap {before}->{after})"
            )

    return base2.join(extra2, how="left")


def _fill_entsoe_extra(df: pd.DataFrame) -> pd.DataFrame:
    """Missing flags + conservative filling to avoid dropping rows."""
    df = df.copy()
    for c in ENTSOE_EXTRA_COLS:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype(int)

    if "solar_fc_dayahead" in df.columns:
        df["solar_fc_dayahead"] = df["solar_fc_dayahead"].fillna(0.0)

    for c in ["wind_onshore_fc_dayahead", "gen_fc_dayahead"]:
        if c in df.columns:
            df[c] = df[c].ffill(limit=6)

    for c in ENTSOE_EXTRA_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


def _fill_weather(df: pd.DataFrame, weather_cols: list[str]) -> pd.DataFrame:
    """Missing flags + conservative short forward-fill (limit=6h) then 0-fill."""
    df = df.copy()
    for c in weather_cols:
        df[f"{c}_missing"] = df[c].isna().astype(int)
        df[c] = df[c].ffill(limit=6).fillna(0.0)
    return df


def _print_overlap_diag(base_idx: pd.DatetimeIndex, extra_idx: pd.DatetimeIndex, tag: str) -> None:
    if len(base_idx) == 0 or len(extra_idx) == 0:
        print(f"[WARN] {tag}: empty index -> base_len={len(base_idx)} extra_len={len(extra_idx)}")
        return
    inter = base_idx.intersection(extra_idx)
    print(
        f"[DIAG] {tag}: base_range=[{base_idx.min()}..{base_idx.max()}] n={len(base_idx)} | "
        f"extra_range=[{extra_idx.min()}..{extra_idx.max()}] n={len(extra_idx)} | "
        f"overlap={len(inter)}"
    )


def _warn_all_nan_after_join(df: pd.DataFrame, cols: list[str], tag: str) -> None:
    present = [c for c in cols if c in df.columns]
    if not present:
        return
    all_nan = [c for c in present if df[c].isna().all()]
    if all_nan:
        print(f"[WARN] {tag}: These columns are ALL NaN after join (likely timestamp mismatch): {all_nan}")


# ----------------------------
# Optional covariate loaders
# ----------------------------
def load_entsoe_extra_hourly() -> pd.DataFrame | None:
    if not ENTSOE_EXTRA_PARQUET.exists():
        print(f"[INFO] ENTSO-E extra not found (skipping): {ENTSOE_EXTRA_PARQUET.name}")
        return None
    df = pd.read_parquet(ENTSOE_EXTRA_PARQUET)
    # TZFIX 2026-07-04: το fetch_entsoe_dayahead αποθηκεύει UTC· το index μας είναι CET/CEST.
    # Πριν το fix τα fc έπεφταν 1h (χειμώνα)/2h (καλοκαίρι) ΝΩΡΙΣ (μετρημένο: solar_fc peak
    # 10:00 αντί 11/12 CET — το «ύποπτο 2h shift» του ABLATION_PLAN §7.8).
    df = _utc_to_cet_naive_index(df)
    keep = [c for c in ENTSOE_EXTRA_COLS if c in df.columns]
    if not keep:
        print("[INFO] ENTSO-E extra parquet exists but expected cols not found (skipping).")
        return None
    return df[keep].copy()


def load_weather_hourly() -> pd.DataFrame | None:
    # Prefer primary name, fallback to alt name (your case)
    path = None
    if WEATHER_HOURLY_PARQUET_PRIMARY.exists():
        path = WEATHER_HOURLY_PARQUET_PRIMARY
    elif WEATHER_HOURLY_PARQUET_ALT.exists():
        path = WEATHER_HOURLY_PARQUET_ALT

    if path is None:
        print(
            "[INFO] Weather parquet not found (skipping): "
            f"{WEATHER_HOURLY_PARQUET_PRIMARY.name} or {WEATHER_HOURLY_PARQUET_ALT.name}"
        )
        return None

    df = pd.read_parquet(path)
    # TZFIX 2026-07-04: το weather parquet είναι σε ΣΤΑΘΕΡΟ UTC+1 (επιβεβαιωμένο: radiation
    # peak 11:00 και DJF και JJA, κάθε έτος 2017-2026) -> +1h μόνο το καλοκαίρι για CET/CEST.
    df = _fixed_utc1_to_cet_naive_index(df)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        print(f"[INFO] Weather parquet has no numeric columns (skipping): {path.name}")
        return None

    print(f"[INFO] Weather parquet loaded: {path.name} | cols={df.shape[1]}")
    return df.copy()


def load_weather_vintage_hourly() -> pd.DataFrame | None:
    """
    Vintage (D-1 lead-time) weather forecast for task=load (GOALS.md G5 / last.md
    §2 Α6). Unlike load_weather_hourly() (Archive API = observed/oracle), these
    columns are genuine forecasts with fixed lead-time buckets (wv_*_day1 = value
    predicted 24h before valid time, wv_*_day2 = 48h before) -- see
    src.fetch_open_meteo_vintage docstring. Coverage starts ~2024-02 (no data
    before that; NOT ffilled across the gap -- see _fill_weather_vintage).
    """
    if not WEATHER_VINTAGE_PARQUET.exists():
        print(f"[INFO] Vintage weather parquet not found (skipping): {WEATHER_VINTAGE_PARQUET.name}")
        return None

    df = pd.read_parquet(WEATHER_VINTAGE_PARQUET)
    # Same fixed-UTC+1 quirk as the Archive weather API (confirmed empirically
    # 2026-07-10 via cross-correlation vs w_gr_mean_* at lag k=0, DJF corr=0.969
    # JJA corr=0.988) -> identical fix.
    df = _fixed_utc1_to_cet_naive_index(df)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        print(f"[INFO] Vintage weather parquet has no numeric columns (skipping): {WEATHER_VINTAGE_PARQUET.name}")
        return None

    print(f"[INFO] Vintage weather parquet loaded: {WEATHER_VINTAGE_PARQUET.name} | cols={df.shape[1]}")
    return df.copy()


def load_load_forecast_hourly() -> "pd.DataFrame | None":
    """
    Load the real published day-ahead load forecast (ADMIE/ENTSO-E), extracted
    from the raw load CSVs by src.build_real_load_forecast (no downloads).
    Contains column 'load_fc' indexed by hourly timestamps.
    Used as a LEGITIMATE contemporaneous feature for BOTH tasks
    (published before the DAM auction / gate 12:00 CET D-1, so no leakage).
    """
    if not LOAD_FORECAST_PARQUET.exists():
        print(f"[INFO] Load forecast parquet not found (skipping): {LOAD_FORECAST_PARQUET.name}")
        return None

    df = pd.read_parquet(LOAD_FORECAST_PARQUET)
    df = _ensure_datetime_index_df(df)

    if not isinstance(df.index, pd.DatetimeIndex):
        print("[WARN] Load forecast parquet has no valid datetime index — skipping.")
        return None

    df.index = pd.to_datetime(df.index).floor("H")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        print("[WARN] Load forecast parquet has no numeric columns — skipping.")
        return None

    # Normalise column name -> always 'load_fc'
    if "load_fc" not in df.columns:
        df = df.rename(columns={df.columns[0]: "load_fc"})

    if "load_fc" not in df.columns:
        return None

    print(f"[INFO] Load forecast loaded: {LOAD_FORECAST_PARQUET.name} | rows={len(df)}")
    return df[["load_fc"]].copy()


# ----------------------------
# Loaders (base ENTSO)
# ----------------------------
def load_entsoe_prices() -> pd.Series:
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

        ts_start = df[time_col].astype(str).str.split(" - ").str[0]
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce").dt.floor("H")

        price = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")
        tmp = pd.DataFrame({"timestamp": dt, "price": price}).dropna(subset=["timestamp", "price"])
        frames.append(tmp)

    full = pd.concat(frames, ignore_index=True)
    # NOTE: Do NOT drop_duplicates before _ensure_hourly_index!
    # For periods where ENTSO-E returns 15-min resolution (e.g. Greece 2024+),
    # all 4 quarter-hourly readings are floored to the same hour.
    # _ensure_hourly_index averages them → correct hourly mean price.
    # For older hourly data: only 1 reading per hour → no change.
    full = full.sort_values("timestamp").set_index("timestamp")
    s = full["price"].astype(float)
    return _ensure_hourly_index(s)


def load_entsoe_load() -> pd.Series | None:
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
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce").dt.floor("H")

        load_vals = pd.to_numeric(df[load_col].astype(str).str.replace(",", ""), errors="coerce")
        tmp = pd.DataFrame({"timestamp": dt, "load": load_vals}).dropna(subset=["timestamp", "load"])
        frames.append(tmp)

    full = pd.concat(frames, ignore_index=True)
    # Same fix as load_entsoe_prices: no drop_duplicates before aggregation.
    # 15-min load data (if present) gets properly averaged to hourly.
    full = full.sort_values("timestamp").set_index("timestamp")
    s = full["load"].astype(float)
    return _ensure_hourly_index(s)


def load_entsoe_generation() -> pd.DataFrame | None:
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
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce").dt.floor("H")

        gen_vals = pd.to_numeric(df[val_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)
        tmp = pd.DataFrame(
            {"timestamp": dt, "type": df[type_col].astype(str), "gen": gen_vals}
        ).dropna(subset=["timestamp"])
        frames.append(tmp)

    if not frames:
        return None

    full = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    full["timestamp"] = pd.to_datetime(full["timestamp"], errors="coerce").dt.floor("H")
    full = full.dropna(subset=["timestamp"])

    # NOTE: use MEAN (not SUM) for 15-min -> hourly aggregation, consistent with
    # price/load. SUM produced 4x-scaled generation, corrupting residual_load
    # (load[MEAN] - gen[SUM]). Requires full pipeline re-run after this change.
    pivot = full.pivot_table(index="timestamp", columns="type", values="gen", aggfunc="mean").sort_index()
    pivot = pivot.groupby(level=0).mean()

    idx = pivot.index

    def _col(name: str) -> pd.Series:
        return pivot[name] if name in pivot.columns else pd.Series(0.0, index=idx)

    gas = _col("Fossil Gas")
    lignite = _col("Fossil Brown coal/Lignite")
    hydro = _col("Hydro Pumped Storage") + _col("Hydro Run-of-river and poundage") + _col("Hydro Water Reservoir")
    solar = _col("Solar")
    wind = _col("Wind Onshore") + _col("Wind Offshore")

    gen_df = pd.DataFrame(
        {
            "gen_solar": solar,
            "gen_wind": wind,
            "gen_gas": gas,
            "gen_lignite": lignite,
            "gen_hydro": hydro,
        },
        index=idx,
    ).fillna(0.0)

    gen_df.index = pd.to_datetime(gen_df.index).floor("H")
    gen_df = gen_df.groupby(level=0).mean().sort_index()
    # TZFIX 2026-07-04: τα generation CSVs είναι σε Europe/Athens naive (ο fetcher έχει
    # TZ="Europe/Athens"), το index μας σε CET/CEST. Athens = Brussels + 1h ΠΑΝΤΑ (ίδιες
    # στιγμές αλλαγής DST) -> σταθερή διόρθωση -1h. Πριν το fix το gen_solar peak έπεφτε
    # 12:00/13:00 αντί 11:00/12:00 CET/CEST και το residual_load ανακάτευε load(t) με gen(t-1).
    gen_df.index = gen_df.index - pd.Timedelta(hours=1)
    return gen_df


def load_gas_daily() -> pd.Series | None:
    candidates = sorted(RAW_DIR.glob("*Natural Gas Futures Historical Data*.csv"))
    if not candidates:
        print("[INFO] No gas CSV found in RAW_DIR.")
        return None

    df = pd.read_csv(candidates[0])
    date_col = [c for c in df.columns if "date" in str(c).lower()][0]
    price_col = [c for c in df.columns if "price" in str(c).lower()][0]

    dates = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    prices = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")
    ser = pd.Series(prices.values, index=dates).dropna().sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


def load_co2_daily() -> pd.Series | None:
    candidates = sorted(RAW_DIR.glob("*Carbon Emissions Futures Historical Data*.csv"))
    if not candidates:
        print("[INFO] No CO2 CSV found in RAW_DIR.")
        return None

    df = pd.read_csv(candidates[0])
    date_col = [c for c in df.columns if "date" in str(c).lower()][0]
    price_col = [c for c in df.columns if "price" in str(c).lower()][0]

    dates = pd.to_datetime(df[date_col].astype(str), errors="coerce")
    prices = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")
    ser = pd.Series(prices.values, index=dates).dropna().sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


# ----------------------------
# Feature engineering (HOURLY ONLY)
# ----------------------------
def process_hourly(
    *,
    task: str,
    price_hourly: pd.Series | None,
    load_hourly: pd.Series | None,
    gen_hourly: pd.DataFrame | None,
    gas_daily: pd.Series | None,
    co2_daily: pd.Series | None,
    entsoe_extra_hourly: pd.DataFrame | None,
    weather_hourly: pd.DataFrame | None,
    include_price_feature_for_load: bool,
    load_forecast_hourly: pd.DataFrame | None = None,
    weather_vintage_hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build hourly supervised dataset.

    task="price": y = price
    task="load":  y = load
    """
    if task not in {"price", "load"}:
        raise ValueError(f"Unknown task: {task}")

    print(f"   -> Features: HOURLY (task={task})...")

    # ----------------------------
    # Base target (y) + base index
    # ----------------------------
    if task == "price":
        if price_hourly is None:
            raise ValueError("price_hourly is required for task='price'")
        y = _ensure_hourly_index(price_hourly).rename("y")
    else:
        if load_hourly is None:
            raise ValueError("load_hourly is required for task='load'")
        y = _ensure_hourly_index(load_hourly).rename("y")

    df = y.to_frame().copy()

    # ----------------------------
    # Optional covariates
    # ----------------------------

    # Load as feature (past-only fill) for PRICE forecasting
    if task == "price" and load_hourly is not None:
        load_hourly = _ensure_hourly_index(load_hourly)
        df = df.join(load_hourly.rename("load"), how="left")
        df["load"] = df["load"].ffill()

    # Price as feature (past-only fill) for LOAD forecasting (optional)
    if task == "load" and include_price_feature_for_load and price_hourly is not None:
        price_hourly = _ensure_hourly_index(price_hourly)
        df = df.join(price_hourly.rename("price"), how="left")
        df["price"] = df["price"].ffill()

    # Generation (0-fill safe)
    if gen_hourly is not None:
        df = df.join(gen_hourly, how="left")
        for c in gen_hourly.columns:
            df[c] = df[c].fillna(0.0)

    # Residual load (careful: for task=load, residual_load at time t is a function of y(t))
    # We'll keep it only to create lag features and then DROP the contemporaneous column for task=load.
    if task == "price" and "load" in df.columns:
        df["residual_load"] = df["load"] - (df.get("gen_solar", 0.0) + df.get("gen_wind", 0.0))
    if task == "load":
        # If we have generation, we can form residual signal; otherwise skip.
        if ("gen_solar" in df.columns) or ("gen_wind" in df.columns):
            df["residual_load"] = df["y"] - (df.get("gen_solar", 0.0) + df.get("gen_wind", 0.0))

    # Gas & CO2: included ONLY for price task (market covariates)
    if task == "price":
        day_index = pd.DatetimeIndex(df.index.normalize().unique()).sort_values()

        if gas_daily is not None:
            gas_full = gas_daily.sort_index().reindex(day_index).ffill()
            gas_map = gas_full.to_dict()
            df["gas_price"] = df.index.normalize().map(lambda d: gas_map.get(d, np.nan))
            df["gas_price"] = df["gas_price"].ffill()

        if co2_daily is not None:
            co2_full = co2_daily.sort_index().reindex(day_index).ffill()
            co2_map = co2_full.to_dict()
            df["co2_price"] = df.index.normalize().map(lambda d: co2_map.get(d, np.nan))
            df["co2_price"] = df["co2_price"].ffill()

    # Start from first hour where ALL required base features exist
    required: list[str] = []
    if task == "price":
        if "gas_price" in df.columns:
            required.append("gas_price")
        if "co2_price" in df.columns:
            required.append("co2_price")
        if "load" in df.columns:
            required.append("load")
    else:
        if include_price_feature_for_load and "price" in df.columns:
            required.append("price")

    df = _crop_to_first_complete_row(df, required)

    # Optional: ENTSO-E extra covariates (day-ahead forecasts)
    if entsoe_extra_hourly is not None:
        _print_overlap_diag(df.index, entsoe_extra_hourly.index, "ENTSOE_EXTRA(pre-join)")
        df = _align_and_join(df, entsoe_extra_hourly, tag="ENTSOE_EXTRA")
        _warn_all_nan_after_join(df, ENTSOE_EXTRA_COLS, "ENTSOE_EXTRA(post-join)")
        df = _fill_entsoe_extra(df)

    # Optional: Weather covariates
    if weather_hourly is not None:
        _print_overlap_diag(df.index, weather_hourly.index, "WEATHER(pre-join)")
        df = _align_and_join(df, weather_hourly, tag="WEATHER")
        wcols = list(weather_hourly.columns)
        _warn_all_nan_after_join(df, wcols, "WEATHER(post-join)")
        df = _fill_weather(df, wcols)

    # Optional: VINTAGE weather forecast (task=load only, GOALS.md G5). Inert
    # columns until referenced by a feature group -- storing them does not affect
    # any existing feature set/run. Coverage starts ~2024-02; NOT ffilled across
    # the pre-2024 gap (that would fabricate a forecast that never existed) -- only
    # a short in-season ffill(limit=6) then 0-fill + _missing flag, same convention
    # as _fill_weather / gen_fc_dayahead_missing.
    if task == "load" and weather_vintage_hourly is not None:
        _print_overlap_diag(df.index, weather_vintage_hourly.index, "WEATHER_VINTAGE(pre-join)")
        df = _align_and_join(df, weather_vintage_hourly, tag="WEATHER_VINTAGE")
        wvcols = list(weather_vintage_hourly.columns)
        _warn_all_nan_after_join(df, wvcols, "WEATHER_VINTAGE(post-join)")
        df = _fill_weather(df, wvcols)

    # Optional: cross-border DAM prices — ΜΟΝΟ LAGGED εκδοχές.
    # ΔΙΟΡΘΩΣΗ ΕΓΚΥΡΟΤΗΤΑΣ (2026-07-03): οι τιμές γειτόνων (BG/IT-SUD) για την ημέρα D
    # βγαίνουν από το ΙΔΙΟ SDAC auction με τη δική μας τιμή D (δημοσίευση ~13:00 CET D-1,
    # ΜΕΤΑ το gate closure 12:00) → η same-day τιμή ΔΕΝ είναι γνωστή όταν υποβάλλουμε
    # προσφορές = leakage αν χρησιμοποιηθεί ως feature. Νόμιμο είναι ό,τι δημοσιεύτηκε
    # από ΠΡΟΗΓΟΥΜΕΝΑ auctions: lags 24/48/168h (ίδια ώρα D-1/D-2/D-7), όπως ακριβώς
    # και το y της Ελλάδας (cutoff 23:00 D-1).
    if task == "price" and XBORDER_PARQUET.exists():
        xb = pd.read_parquet(XBORDER_PARQUET)
        xb = _ensure_datetime_index_df(xb)
        _print_overlap_diag(df.index, xb.index, "XBORDER(pre-join)")
        df = _align_and_join(df, xb, tag="XBORDER")
        xb_base_cols = [c for c in xb.columns if c in df.columns]
        for c in xb_base_cols:
            # μικρά κενά (αργίες API): ffill έως 3h πριν τα lags
            df[c] = df[c].ffill(limit=3)
            for lag in (24, 48, 168):
                df[f"{c}_lag{lag}"] = df[c].shift(lag)
        df = df.drop(columns=xb_base_cols)
        print(f"   [LEAKAGE FIX price] Dropped contemporaneous xborder cols: {xb_base_cols} "
              f"(κρατήθηκαν μόνο lag24/48/168 — same-day = ίδιο SDAC auction με το target)")

    # Calendar (known a priori)
    gr_holidays = holidays.Greece()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_holiday"] = df.index.map(lambda x: int(x.date() in gr_holidays))
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # Lags (past-only)
    lags = [1, 2, 3, 6, 12, 24, 48, 168]

    for lag in lags:
        df[f"y_lag{lag}"] = df["y"].shift(lag)

    # For compatibility: provide load_lag* for both tasks
    if task == "price" and "load" in df.columns:
        for lag in lags:
            df[f"load_lag{lag}"] = df["load"].shift(lag)
    if task == "load":
        for lag in lags:
            df[f"load_lag{lag}"] = df["y"].shift(lag)

    if "residual_load" in df.columns:
        for lag in lags:
            df[f"residual_load_lag{lag}"] = df["residual_load"].shift(lag)

    if "gas_price" in df.columns:
        for lag in lags:
            df[f"gas_lag{lag}"] = df["gas_price"].shift(lag)

    if "co2_price" in df.columns:
        for lag in lags:
            df[f"co2_lag{lag}"] = df["co2_price"].shift(lag)

    for col in ["gen_solar", "gen_wind"]:
        if col in df.columns:
            for lag in [1, 2, 24, 48, 168]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df["y_roll24"] = df["y"].shift(1).rolling(24).mean()
    df["y_roll168"] = df["y"].shift(1).rolling(168).mean()

    # ----------------------------------------------------------------
    # load_fc: day-ahead load forecast as legitimate contemporaneous
    # feature for BOTH tasks.
    # A load forecast is published by the TSO before the DAM auction,
    # so it does NOT constitute leakage. For task=load it is the
    # official D-1 forecast of the target itself (standard STLF covariate).
    #
    # Coverage strategy:
    #   - test period : use the forecasted values from load_forecast_hourly
    #   - training period (NaN in forecast file): proxy with load_lag24
    #     (same-hour load from 24 h ago — best available substitute;
    #      for task=load, load_lag24 == y_lag24 by construction)
    # ----------------------------------------------------------------
    if task in ("price", "load"):
        if load_forecast_hourly is not None:
            lfc = load_forecast_hourly.copy()
            lfc.index = pd.to_datetime(lfc.index).floor("H")
            df = df.join(lfc[["load_fc"]], how="left")
            if "load_lag24" in df.columns:
                df["load_fc"] = df["load_fc"].fillna(df["load_lag24"])
            n_fc = int(df["load_fc"].notna().sum())
            print(f"   [INFO] load_fc feature: {n_fc} rows from forecast, rest from load_lag24 proxy")
        elif "load_lag24" in df.columns:
            # No forecast file yet — use load_lag24 as proxy throughout
            df["load_fc"] = df["load_lag24"].copy()
            print("   [INFO] load_fc proxied from load_lag24 (no forecast file available)")

    # ----------------------------------------------------------------
    # LEAKAGE FIX: Drop all contemporaneous columns for both tasks.
    # Lag features (shift >= 1) are kept — they are strictly past-only.
    # ----------------------------------------------------------------

    # PRICE task: load(t), gen(t), residual_load(t) are realized at hour t
    # and are NOT available at DAM auction time (day D-1, ~12:00 noon).
    if task == "price":
        contemporaneous_to_drop = [
            "load",
            "residual_load",
            "gen_solar",
            "gen_wind",
            "gen_gas",
            "gen_lignite",
            "gen_hydro",
        ]
        cols_to_drop = [c for c in contemporaneous_to_drop if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"   [LEAKAGE FIX price] Dropped contemporaneous cols: {cols_to_drop}")

    # LOAD task:
    # - price(t): DAM price is not known at day-ahead load forecast time.
    # - gen_*(t): actual generation at hour t is not known in advance.
    # - residual_load(t): derived directly from y(t) + gen(t) -> double leakage.
    # Lag versions of all of these are kept (shift >= 1 -> past-only).
    if task == "load":
        contemporaneous_to_drop = [
            "price",
            "residual_load",
            "gen_solar",
            "gen_wind",
            "gen_gas",
            "gen_lignite",
            "gen_hydro",
        ]
        cols_to_drop = [c for c in contemporaneous_to_drop if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"   [LEAKAGE FIX load] Dropped contemporaneous cols: {cols_to_drop}")

    df = df.dropna()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="hourly",
        choices=["hourly"],
        help="Dataset mode (hourly only)",
    )
    parser.add_argument(
        "--task",
        default="price",
        choices=["price", "load"],
        help="Prediction task: price (EPF) or load (STLF)",
    )
    parser.add_argument(
        "--include_price_feature_for_load",
        action="store_true",
        help="If set (task=load), include DAM price as an extra covariate feature.",
    )
    args = parser.parse_args()

    print(f"🔨 BUILDING DATASET (HOURLY, task={args.task})...")

    # Base series loading depends on task
    price_hourly: pd.Series | None = None
    load_hourly: pd.Series | None = None

    if args.task == "price":
        print(f"   -> Reading ENTSO-E prices from {PRICES_DIR}")
        price_hourly = load_entsoe_prices()
    else:
        # For load task, price is optional; we only try to load it if explicitly requested.
        if args.include_price_feature_for_load:
            try:
                print(f"   -> Reading ENTSO-E prices from {PRICES_DIR} (as feature)")
                price_hourly = load_entsoe_prices()
            except Exception as e:
                print(f"[WARN] Could not load prices (continuing without price feature): {e}")

    # Load series: required for load task, optional for price task
    load_hourly = load_entsoe_load()
    if args.task == "load" and load_hourly is None:
        raise FileNotFoundError(
            f"Load task requested but no load series found in: {LOAD_DIR} (CSV files missing?)"
        )

    gen_hourly = load_entsoe_generation()

    # Market covariates only for price task
    gas_daily = load_gas_daily() if args.task == "price" else None
    co2_daily = load_co2_daily() if args.task == "price" else None

    entsoe_extra = load_entsoe_extra_hourly()
    weather_hourly = load_weather_hourly()

    # Load forecast: legitimate D-1 forecast feature for both tasks
    # (task=load parquet was missing load_fc entirely — ABLATION_PLAN §5.12γ).
    load_fc_hourly = load_load_forecast_hourly()

    # Vintage weather forecast: task=load only (GOALS.md G5).
    weather_vintage_hourly = load_weather_vintage_hourly() if args.task == "load" else None

    df_final = process_hourly(
        task=args.task,
        price_hourly=price_hourly,
        load_hourly=load_hourly,
        gen_hourly=gen_hourly,
        gas_daily=gas_daily,
        co2_daily=co2_daily,
        entsoe_extra_hourly=entsoe_extra,
        weather_hourly=weather_hourly,
        include_price_feature_for_load=bool(args.include_price_feature_for_load),
        load_forecast_hourly=load_fc_hourly,
        weather_vintage_hourly=weather_vintage_hourly,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / ("hourly.parquet" if args.task == "price" else "hourly_load.parquet")
    df_final.to_parquet(out_path, engine="fastparquet")

    print(f"✅ Saved: {out_path}")
    print(f"[INFO] shape={df_final.shape} | cols={len(df_final.columns)}")


if __name__ == "__main__":
    main()
