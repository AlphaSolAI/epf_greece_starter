"""
fetch_entsoe_xborder.py — Cross-border features (Φάση 3 του ABLATION_PLAN):
DAM τιμές γειτονικών ζωνών (SDAC-coupled με GR) από το ENTSO-E API.

Availability (γι' αυτό είναι leakage-free ΧΩΡΙΣ lag): οι DAM τιμές των γειτόνων
για την ημέρα D δημοσιεύονται την D-1 μαζί/λίγο μετά τις ελληνικές (κοινό SDAC
auction) → στο DAM gate είναι γνωστές για ΟΛΟ τον ορίζοντα, όπως και η y μας.
Ομάδα feature: 'xborder' (στήλες xb_*), εκτός DEFAULT — μπαίνει με
`--features default,xborder`.

Ζώνες: BG (Βουλγαρία), IT_SUD (Ιταλία-Νότος — το coupled border zone με GR).

Γράφει: data/processed/xborder_hourly.parquet (index=naive local hour, στήλες xb_price_*)

Usage:
  python -m src.fetch_entsoe_xborder --start 2017-01-01          # full backfill
  python -m src.fetch_entsoe_xborder                             # συνέχεια από το τέλος
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "xborder_hourly.parquet"
TZ = "Europe/Athens"

ZONES = {
    "bg": "BG",
    "itsud": "IT_SUD",
}


def _to_hourly_naive_local(s: pd.Series) -> pd.Series:
    """tz-aware → ώρα Ελλάδας χωρίς tz (ίδιο index convention με hourly.parquet).

    ΔΙΟΡΘΩΣΗ (2026-07-03): εμπειρικά επιβεβαιώθηκε (cross-correlation lag scan κατά της
    πραγματικής τιμής GR, σταθερό winter/summer → όχι DST) ότι το `query_day_ahead_prices`
    του entsoe-py επιστρέφει index με σταθερή μετατόπιση +1h σε σχέση με τη σύμβαση που
    χρησιμοποιεί το υπόλοιπο pipeline (ίδια σύμβαση με `query_generation`, που είναι σωστή —
    επαληθεύτηκε με peak στο μεσημέρι για το solar). Διορθώνεται εδώ με ρητό −1h.
    (Πρώτη απόπειρα fix είχε λάθος πρόσημο +1h — επιδείνωσε το shift αντί να το διορθώσει·
    επαληθεύτηκε ξανά με το ίδιο cross-correlation script πριν κλειδωθεί αυτή η εκδοχή.)
    """
    s = s.tz_convert(TZ)
    s.index = s.index.tz_localize(None) - pd.Timedelta(hours=1)
    return s.groupby(s.index.floor("h")).mean()


def main():
    ap = argparse.ArgumentParser(description="Fetch γειτονικές DAM τιμές (BG, IT_SUD)")
    ap.add_argument("--start", default=None, help="default: συνέχεια από υπάρχον parquet, αλλιώς 2017-01-01")
    ap.add_argument("--end", default=None, help="default: σήμερα+1d (οι αυριανές είναι ήδη γνωστές)")
    args = ap.parse_args()

    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        sys.exit("❌ Λείπει το env var ENTSOE_API_KEY.")

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=key)

    existing = pd.read_parquet(OUTPUT_PARQUET) if OUTPUT_PARQUET.exists() else None
    if args.start:
        start = pd.Timestamp(args.start, tz=TZ)
    elif existing is not None and len(existing):
        start = existing.index.max().tz_localize(TZ) + pd.Timedelta(hours=1)
    else:
        start = pd.Timestamp("2017-01-01", tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ) + pd.Timedelta(days=1)

    if start >= end:
        print("[INFO] Τίποτα νέο να κατέβει.")
        return

    print(f"🔌 ENTSO-E cross-border DAM prices: {start.date()} → {end.date()} | ζώνες: {list(ZONES.values())}")

    per_zone: dict[str, list[pd.Series]] = {z: [] for z in ZONES}
    for year in range(start.year, end.year + 1):
        y0 = max(start, pd.Timestamp(f"{year}-01-01", tz=TZ))
        y1 = min(end, pd.Timestamp(f"{year+1}-01-01", tz=TZ))
        if y0 >= y1:
            continue
        for short, zone in ZONES.items():
            print(f"  [{year}] {zone} ...", end=" ", flush=True)
            try:
                s = client.query_day_ahead_prices(zone, start=y0, end=y1)
                per_zone[short].append(_to_hourly_naive_local(s))
                print(f"✅ {len(s)}")
            except Exception as e:
                print(f"❌ {type(e).__name__}: {str(e)[:120]}")
            time.sleep(0.5)

    cols = {}
    for short, parts in per_zone.items():
        if parts:
            s = pd.concat(parts).sort_index()
            cols[f"xb_price_{short}"] = s[~s.index.duplicated(keep="last")]
    if not cols:
        sys.exit("❌ Καμία ζώνη δεν επέστρεψε δεδομένα.")

    new_df = pd.DataFrame(cols).sort_index()
    if existing is not None:
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PARQUET)
    print(f"\n✅ xborder_hourly.parquet: {len(combined)} rows | "
          f"{combined.index.min()} → {combined.index.max()} | στήλες: {list(combined.columns)}")
    print("   Επόμενο: rebuild (python -m src.data --task price) ΑΦΟΥ τελειώσει το τρέχον ablation chain.")


if __name__ == "__main__":
    main()
