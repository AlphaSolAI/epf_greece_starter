"""
fetch_entsoe_dayahead.py — Συμπληρώνει το 2026 gap στο entsoe_extra_hourly.parquet
(day-ahead RES/gen forecasts, ομάδα feature 'forecast') μέσω ENTSO-E API.

Fetch-άρει (entsoe-py):
  - query_wind_and_solar_forecast  → solar_fc_dayahead, wind_onshore_fc_dayahead
  - query_generation_forecast      → gen_fc_dayahead (σύνολο)
15-λεπτη ανάλυση → ωριαία (mean), tz-aware UTC (ίδιο index format με το υπάρχον
parquet). Append-only: κρατά ό,τι υπάρχει ήδη, γεμίζει μόνο μετά το τελευταίο
γνωστό timestamp (ή --start αν δοθεί ρητά).

Χρειάζεται env var ENTSOE_API_KEY.

Usage:
  python -m src.fetch_entsoe_dayahead                  # συνέχεια από το τέλος του parquet
  python -m src.fetch_entsoe_dayahead --start 2026-01-01 --end 2026-07-02
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "processed"
ENTSOE_EXTRA_PARQUET = OUTPUT_DIR / "entsoe_extra_hourly.parquet"
TZ = "Europe/Athens"


def _to_hourly_utc(s: pd.Series) -> pd.Series:
    s = s.tz_convert("UTC")
    return s.groupby(s.index.floor("H")).mean()


def main():
    ap = argparse.ArgumentParser(description="Fetch ENTSO-E day-ahead RES/gen forecast (GR)")
    ap.add_argument("--start", default=None, help="default: συνέχεια από το υπάρχον parquet")
    ap.add_argument("--end", default=None, help="default: σήμερα")
    args = ap.parse_args()

    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        sys.exit("❌ Λείπει το env var ENTSOE_API_KEY.")

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=key)

    existing = None
    if ENTSOE_EXTRA_PARQUET.exists():
        existing = pd.read_parquet(ENTSOE_EXTRA_PARQUET)
        print(f"[INFO] Υπάρχον parquet: {existing.index.min()} → {existing.index.max()} ({len(existing)} rows)")

    if args.start:
        start_local = pd.Timestamp(args.start, tz=TZ)
    elif existing is not None and len(existing):
        start_local = existing.index.max().tz_convert(TZ) + pd.Timedelta(hours=1)
    else:
        start_local = pd.Timestamp("2017-10-30", tz=TZ)
    end_local = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ)

    if start_local >= end_local:
        print(f"[INFO] Τίποτα νέο να κατέβει (start={start_local} >= end={end_local}).")
        return

    print(f"🔌 ENTSO-E day-ahead forecasts GR: {start_local} → {end_local}")

    frames = []
    for year in range(start_local.year, end_local.year + 1):
        y0 = max(start_local, pd.Timestamp(f"{year}-01-01", tz=TZ))
        y1 = min(end_local, pd.Timestamp(f"{year+1}-01-01", tz=TZ))
        if y0 >= y1:
            continue
        print(f"  [{year}] wind/solar forecast ...", end=" ", flush=True)
        try:
            wsf = client.query_wind_and_solar_forecast(country_code="GR", start=y0, end=y1, psr_type=None)
            print(f"✅ {len(wsf)} rows", end="  ")
        except Exception as e:
            print(f"❌ {e}")
            wsf = None
        print("gen forecast ...", end=" ", flush=True)
        try:
            gf = client.query_generation_forecast(country_code="GR", start=y0, end=y1)
            print(f"✅ {len(gf)} rows")
        except Exception as e:
            print(f"❌ {e}")
            gf = None

        if wsf is None and gf is None:
            continue
        df = pd.DataFrame(index=(wsf.index if wsf is not None else gf.index))
        df["solar_fc_dayahead"] = _to_hourly_utc(wsf["Solar"]) if wsf is not None and "Solar" in wsf else pd.NA
        df["wind_onshore_fc_dayahead"] = _to_hourly_utc(wsf["Wind Onshore"]) if wsf is not None and "Wind Onshore" in wsf else pd.NA
        if gf is not None:
            gf_h = _to_hourly_utc(gf)
            df = df.reindex(gf_h.index.union(df.index))
            df["gen_fc_dayahead"] = gf_h
        frames.append(df)

    if not frames:
        sys.exit("❌ Καμία νέα τιμή κατέβηκε.")

    new_df = pd.concat(frames).sort_index()
    new_df = new_df[~new_df.index.duplicated(keep="last")]
    # τα solar/wind χρειάζονται ξεχωριστό hourly resample (έγινε ήδη ανά year-chunk πάνω,
    # εδώ απλά ενοποιούμε index) — ξαναφτιάχνουμε σωστά με groupby σε επίπεδο ώρας
    new_df.index = pd.to_datetime(new_df.index).floor("H")
    new_df = new_df.groupby(level=0).mean()

    if existing is not None:
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(ENTSOE_EXTRA_PARQUET)
    print(f"\n✅ entsoe_extra_hourly.parquet: {len(combined)} rows | "
          f"{combined.index.min()} → {combined.index.max()}")
    print(f"   Αποθηκεύτηκε: {ENTSOE_EXTRA_PARQUET}")


if __name__ == "__main__":
    main()
