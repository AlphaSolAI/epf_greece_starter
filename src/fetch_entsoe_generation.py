"""
fetch_entsoe_generation.py — Κατεβάζει actual generation per type (GR) από το
ENTSO-E Transparency Platform API (entsoe-py) και γράφει CSV στο ΙΔΙΟ format
που περιμένει το src.data._read_entsoe_csv/load_entsoe_generation (MTU,
Production Type, Generation [MW]) — έτσι το data/raw/generation/ γεμίζει με
αρχεία 100% συμβατά με το υπάρχον pipeline, χωρίς καμία αλλαγή στο data.py.

Χρειάζεται env var ENTSOE_API_KEY (security token από
https://transparency.entsoe.eu/usrm/user/register — free, ~ημέρες έγκριση).
Σε αυτό το μηχάνημα βρέθηκε ήδη έτοιμο.

Usage:
  python -m src.fetch_entsoe_generation --start 2023-01-01 --end 2026-07-02
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
GEN_DIR = BASE_DIR / "data" / "raw" / "generation"
TZ = "Europe/Athens"


def _fmt_mtu(ts: pd.Timestamp) -> str:
    return ts.strftime("%d.%m.%Y %H:%M")


def fetch_year(client, year: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Actual generation per type για ένα ημερολογιακό έτος (chunked, ENTSO-E limit-safe)."""
    y0 = max(start, pd.Timestamp(f"{year}-01-01", tz=TZ))
    y1 = min(end, pd.Timestamp(f"{year+1}-01-01", tz=TZ))
    if y0 >= y1:
        return None
    df = client.query_generation(country_code="GR", start=y0, end=y1, psr_type=None)
    if df is None or df.empty:
        return None
    # Μερικές στήλες μπορεί να είναι MultiIndex (type, Actual Aggregated/Consumption) —
    # κρατάμε μόνο 'Actual Aggregated' όπου υπάρχει storage διάκριση.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs("Actual Aggregated", axis=1, level=-1, drop_level=True)
    return df


def to_gui_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Μετατρέπει wide (index=time, cols=production type) σε long GUI-compatible format."""
    long = df.stack().reset_index()
    long.columns = ["_t0", "Production Type", "Generation [MW] Actual Aggregated"]
    step = df.index.to_series().diff().dropna().mode()
    freq = step.iloc[0] if len(step) else pd.Timedelta(minutes=15)
    long["_t1"] = long["_t0"] + freq
    long["MTU"] = long["_t0"].apply(_fmt_mtu) + " - " + long["_t1"].apply(_fmt_mtu)
    out = long[["MTU", "Production Type", "Generation [MW] Actual Aggregated"]].copy()
    out.insert(1, "Area", "BZN|GR")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch ENTSO-E actual generation per type (GR)")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None, help="default: σήμερα")
    args = ap.parse_args()

    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        sys.exit("❌ Λείπει το env var ENTSOE_API_KEY (security token από transparency.entsoe.eu).")

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=key)

    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🔌 ENTSO-E generation GR: {start.date()} → {end.date()}")

    for year in range(start.year, end.year + 1):
        out_path = GEN_DIR / f"ENTSOE_GENERATION_GR_{year}.csv"
        print(f"  [{year}] fetching ...", end=" ", flush=True)
        try:
            df = fetch_year(client, year, start, end)
        except Exception as e:
            print(f"❌ {e}")
            continue
        if df is None:
            print("(no data)")
            continue
        gui = to_gui_csv(df)
        gui.to_csv(out_path, index=False)
        print(f"✅ {len(gui)} rows → {out_path.name}")
        time.sleep(1.0)  # ευγενικό rate-limit προς το API

    print(f"\n✅ Ολοκληρώθηκε. Αρχεία στο {GEN_DIR}")
    print("   Επόμενο βήμα: ξανατρέξε το src.data (rebuild) για να μπουν τα gen_* features.")


if __name__ == "__main__":
    main()
