"""
Fetch historical weather for Greek cities from Open-Meteo Archive API
and append to the existing weather_gr_hourly.parquet.

Usage:
    python -m src.fetch_weather_2026
    python -m src.fetch_weather_2026 --start 2026-01-01 --end 2026-03-15
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "processed"
WEATHER_PRIMARY = OUTPUT_DIR / "weather_gr_hourly.parquet"
WEATHER_ALT = OUTPUT_DIR / "weather_hourly.parquet"

# Open-Meteo Archive API endpoint
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Greek city coordinates (same as original dataset)
CITIES = {
    "athens":       (37.9838, 23.7275),
    "thessaloniki": (40.6401, 22.9444),
    "patras":       (38.2466, 21.7346),
    "larissa":      (39.6369, 22.4191),
    "heraklion":    (35.3387, 25.1442),
}

# Variables to fetch
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]


def fetch_city_weather(city: str, lat: float, lon: float,
                       start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather for a single city from Open-Meteo Archive API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly["time"])
    df = pd.DataFrame(index=times)
    for var in HOURLY_VARS:
        col_name = f"w_{city}_{var}"
        df[col_name] = hourly.get(var, None)
    df.index.name = "time"
    return df


def build_gr_mean(city_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute gr_mean as the mean of all 5 cities for each variable."""
    result = None
    for city, df in city_dfs.items():
        if result is None:
            result = df.copy()
        else:
            result = result.join(df, how="outer", rsuffix=f"_{city}")

    # Compute gr_mean for each variable
    mean_df = pd.DataFrame(index=list(city_dfs.values())[0].index)
    for var in HOURLY_VARS:
        cols = [f"w_{city}_{var}" for city in city_dfs]
        available = [c for c in cols if c in result.columns]
        mean_df[f"w_gr_mean_{var}"] = result[available].mean(axis=1)
    return mean_df


def main():
    parser = argparse.ArgumentParser(description="Fetch weather 2026 from Open-Meteo")
    parser.add_argument("--start", default="2026-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-15",
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Find existing weather parquet
    wp = WEATHER_PRIMARY if WEATHER_PRIMARY.exists() else WEATHER_ALT
    if not wp.exists():
        print(f"[ERROR] No weather parquet found at {WEATHER_PRIMARY} or {WEATHER_ALT}")
        sys.exit(1)

    print(f"[INFO] Loading existing weather from: {wp.name}")
    existing = pd.read_parquet(wp)
    print(f"  Shape: {existing.shape} | Range: {existing.index.min()} → {existing.index.max()}")

    # Check what dates we actually need
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end) + pd.Timedelta(hours=23)
    existing_end = existing.index.max()

    if start_dt <= existing_end:
        # Start from day after existing end
        start_dt = existing_end + pd.Timedelta(hours=1)
        args.start = start_dt.strftime("%Y-%m-%d")
        print(f"[INFO] Existing data ends {existing_end} — fetching from {args.start}")

    if pd.Timestamp(args.start) > pd.Timestamp(args.end):
        print(f"[INFO] No new data needed (existing data covers up to {existing_end})")
        return

    print(f"\n[INFO] Fetching weather {args.start} → {args.end}")
    print(f"  Cities: {', '.join(CITIES.keys())}")

    city_dfs = {}
    for city, (lat, lon) in CITIES.items():
        print(f"  Fetching {city}...", end=" ", flush=True)
        try:
            df = fetch_city_weather(city, lat, lon, args.start, args.end)
            city_dfs[city] = df
            print(f"✅ {len(df)} rows")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        time.sleep(0.5)  # Be polite to the API

    if not city_dfs:
        print("[ERROR] No city data fetched!")
        sys.exit(1)

    # Build new rows with all cities + gr_mean
    print("\n[INFO] Building combined DataFrame...")
    # Start with gr_mean
    mean_df = build_gr_mean(city_dfs)

    # Join all city dfs
    all_city_df = pd.concat(list(city_dfs.values()), axis=1)
    new_df = pd.concat([mean_df, all_city_df], axis=1)

    # Align columns with existing
    for col in existing.columns:
        if col not in new_df.columns:
            print(f"  [WARN] Column {col} missing in new data → fill with 0")
            new_df[col] = 0.0
    # Keep only columns that exist in existing
    new_df = new_df[existing.columns]

    print(f"  New rows: {len(new_df)}")
    print(f"  Date range: {new_df.index.min()} → {new_df.index.max()}")

    # Concatenate and save
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    print(f"\n[INFO] Combined: {len(combined)} rows | {combined.index.min()} → {combined.index.max()}")
    combined.to_parquet(wp)
    print(f"✅ Saved to: {wp}")


if __name__ == "__main__":
    main()
