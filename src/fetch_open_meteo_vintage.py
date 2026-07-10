"""
Fetch VINTAGE (D-1 forecast) weather from Open-Meteo Previous Runs API
and write data/processed/weather_vintage_hourly.parquet.

Unlike the Archive API (data.fetch_weather_2026 -> observed/oracle values), this
pulls fixed lead-time forecast buckets:
  previous_day1 = value predicted 24h before valid time
  previous_day2 = value predicted 48h before valid time
(no finer lead-time selection is offered by this API — see GOALS.md G5 / last.md
Α6 for why the oracle Archive API cannot be used as a gate-honest meteo feature).

Full 7-variable coverage confirmed empirically to start ~2024-02-01 (temperature_2m
alone goes back to 2021, but the other 6 vars are null before 2024). Data before
that date is NOT fetched -> the merge step marks it with a `_missing` flag, same
pattern as gen_fc_dayahead_missing.

TZ: this script stores RAW timestamps exactly as returned by the API (fixed-UTC+1,
same quirk as the Archive API -- confirmed empirically via cross-correlation vs the
already-trusted w_gr_mean_* archive column: DJF corr=0.969 JJA corr=0.988 exactly at
lag k=0 after applying data._fixed_utc1_to_cet_naive_index). Per project rule
("TZ conversions ONLY in src/data.py loaders"), the CET/CEST-naive conversion is
applied by data.load_weather_vintage_hourly(), NOT here.

Usage:
    python -m src.fetch_open_meteo_vintage
    python -m src.fetch_open_meteo_vintage --start 2024-02-01 --end 2026-07-08
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
OUT_PARQUET = OUTPUT_DIR / "weather_vintage_hourly.parquet"

PREVRUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

CITIES = {
    "athens":       (37.9838, 23.7275),
    "thessaloniki": (40.6401, 22.9444),
    "patras":       (38.2466, 21.7346),
    "larissa":      (39.6369, 22.4191),
    "heraklion":    (35.3387, 25.1442),
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]
OFFSETS = ("previous_day1", "previous_day2")

# Confirmed empirically (2026-07-10): full 7-var coverage starts here.
SAFE_COVERAGE_START = "2024-02-01"


def _date_chunks(start: str, end: str, days: int = 90):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = s
    while cur <= e:
        chunk_end = min(cur + pd.Timedelta(days=days - 1), e)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


def _fetch_chunk(lat: float, lon: float, start: str, end: str, hourly_params: list[str],
                  max_retries: int = 4) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(hourly_params),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(PREVRUNS_URL, params=params, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json().get("hourly", {})
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f" [retry {attempt+1}/{max_retries} after {e!r}, wait {wait}s]", end="", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries for {start}..{end}: {last_err}")


def fetch_city(city: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    hourly_params = [f"{v}_{off}" for v in HOURLY_VARS for off in OFFSETS]
    frames = []
    for c_start, c_end in _date_chunks(start, end, days=90):
        data = _fetch_chunk(lat, lon, c_start, c_end, hourly_params)
        raw_idx = pd.to_datetime(data["time"])  # RAW as returned (fixed-UTC+1 quirk, unconverted)
        cdf = pd.DataFrame(index=raw_idx)
        for v in HOURLY_VARS:
            for off in OFFSETS:
                suffix = "day1" if off == "previous_day1" else "day2"
                cdf[f"wv_{city}_{v}_{suffix}"] = data.get(f"{v}_{off}", None)
        frames.append(cdf)
        time.sleep(0.3)
    df = pd.concat(frames)
    df.index.name = "time"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def build_gr_mean(city_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(list(city_dfs.values()), axis=1)
    mean_df = pd.DataFrame(index=combined.index)
    for v in HOURLY_VARS:
        for suffix in ("day1", "day2"):
            cols = [f"wv_{c}_{v}_{suffix}" for c in city_dfs]
            avail = [c for c in cols if c in combined.columns]
            mean_df[f"wv_gr_mean_{v}_{suffix}"] = combined[avail].mean(axis=1)
    return mean_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=SAFE_COVERAGE_START)
    ap.add_argument("--end", default=None, help="default: today - 2 days (avoid partial last day)")
    args = ap.parse_args()

    end = args.end or (pd.Timestamp.now().normalize() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    print(f"[INFO] Fetching vintage weather (previous_day1/day2) {args.start} -> {end}")
    print(f"  Cities: {', '.join(CITIES)}  Vars: {', '.join(HOURLY_VARS)}")

    city_dfs = {}
    for city, (lat, lon) in CITIES.items():
        print(f"  Fetching {city}...", end=" ", flush=True)
        df = fetch_city(city, lat, lon, args.start, end)
        city_dfs[city] = df
        print(f"OK {len(df)} rows, range {df.index.min()} -> {df.index.max()}")
        time.sleep(0.5)

    mean_df = build_gr_mean(city_dfs)
    all_city_df = pd.concat(list(city_dfs.values()), axis=1)
    combined = pd.concat([mean_df, all_city_df], axis=1)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    print(f"\n[INFO] Combined: {combined.shape} | {combined.index.min()} -> {combined.index.max()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PARQUET)
    print(f"OK saved: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
