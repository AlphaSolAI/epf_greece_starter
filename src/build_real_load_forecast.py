"""
build_real_load_forecast.py — Αντικαθιστά το ΣΥΝΘΕΤΙΚΟ load_fc με το ΠΡΑΓΜΑΤΙΚΟ
day-ahead load forecast του ENTSO-E.

Εύρημα: το `load_forecast_hourly.parquet` παραγόταν μέχρι τώρα από
`generate_load_forecast.py`, το οποίο τρέχει το ΔΙΚΟ ΜΑΣ LGBM πάνω στο actual
load και αποθηκεύει τις προβλέψεις ΤΟΥ σαν να ήταν «η επίσημη πρόβλεψη» — ούτε
leakage (σέβεται recursive/gate), αλλά ούτε πραγματική εξωτερική πληροφορία.
Κάλυπτε μόνο Οκτ-Δεκ 2025.

Το ίδιο το ENTSO-E raw CSV (data/raw/load/*.csv) περιέχει ΗΔΗ τη στήλη
"Day-ahead Total Load Forecast (MW)" — την πραγματική, δημοσιευμένη πρόβλεψη
του ΑΔΜΗΕ/ENTSO-E, για ΟΛΟ το ιστορικό (2015-σήμερα). Καμία λήψη δεν
χρειάζεται — μόνο parsing του ήδη κατεβασμένου αρχείου.

Usage:
  python -m src.build_real_load_forecast
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LOAD_DIR, OUTPUT_DIR, _read_entsoe_csv, _find_time_column, _ensure_hourly_index

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT_PARQUET = OUTPUT_DIR / "load_forecast_hourly.parquet"


def _find_forecast_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "day-ahead total load forecast" in str(c).lower():
            return c
    for c in df.columns:
        if "forecast" in str(c).lower() and "load" in str(c).lower():
            return c
    return None


def main():
    files = sorted(LOAD_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"❌ Καμία CSV στο {LOAD_DIR}")

    frames = []
    for f in files:
        df = _read_entsoe_csv(f)
        fc_col = _find_forecast_column(df)
        if fc_col is None:
            print(f"   [skip] {f.name}: δεν βρέθηκε στήλη forecast")
            continue
        time_col = _find_time_column(df)
        ts_start = df[time_col].astype(str).str.split(" - ").str[0]
        dt = pd.to_datetime(ts_start, dayfirst=True, errors="coerce").dt.floor("H")
        vals = pd.to_numeric(
            df[fc_col].astype(str).str.replace(",", "").str.replace("-", ""),
            errors="coerce",
        )
        tmp = pd.DataFrame({"timestamp": dt, "load_fc": vals}).dropna(subset=["timestamp", "load_fc"])
        if len(tmp):
            print(f"   [ok] {f.name}: {len(tmp)} τιμές ({tmp['timestamp'].min()} → {tmp['timestamp'].max()})")
        frames.append(tmp)

    if not frames:
        sys.exit("❌ Καμία στήλη forecast βρέθηκε σε κανένα αρχείο.")

    full = pd.concat(frames, ignore_index=True).sort_values("timestamp").set_index("timestamp")
    s = _ensure_hourly_index(full["load_fc"].astype(float))

    fc_df = s.to_frame("load_fc")
    fc_df.index.name = "datetime"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fc_df.to_parquet(OUTPUT_PARQUET)
    print(f"\n✅ Πραγματικό day-ahead load forecast: {len(fc_df)} ωριαίες τιμές")
    print(f"   Περίοδος: {fc_df.index.min()} → {fc_df.index.max()}")
    print(f"   Αποθηκεύτηκε: {OUTPUT_PARQUET}")
    print("   (αντικατέστησε το προηγούμενο συνθετικό load_fc — πλέον πραγματική "
          "δημοσιευμένη πρόβλεψη ΑΔΜΗΕ/ENTSO-E, όχι προϊόν δικού μας μοντέλου)")


if __name__ == "__main__":
    main()
