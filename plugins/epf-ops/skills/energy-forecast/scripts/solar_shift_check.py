# -*- coding: utf-8 -*-
"""
solar_shift_check.py — Οριστικός έλεγχος του ύποπτου solar_fc 2h shift (ABLATION_PLAN §7.8).

Μέθοδος (με τα δεδομένα που ΗΔΗ έχουμε, καμία λήψη): σύγκριση του day-ahead solar forecast
(solar_fc_dayahead) με το ACTUAL solar generation (gen_solar) στο ίδιο parquet.
Αν το fc έχει timestamp shift σε κάποιο κομμάτι του ιστορικού, το corr peak fc↔actual
θα πέσει σε k≠0 για εκείνο το κομμάτι — ανά έτος φαίνεται αμέσως ΠΟΥ (αν) υπάρχει το πρόβλημα.

TIMEZONE (μετά το TZFIX 2026-07-04, βλ. data.py + ABLATION_PLAN §5.9): το parquet index
είναι CET/CEST-naive (το frame των price/load GUI exports). Σωστό solar peak ≈ 11:00 (DJF)
/ 12:00 (JJA) στο index frame. Όλες οι πηγές μετατρέπονται σε αυτό το frame στο data.py.

Χρήση:
  conda run -n epf --no-capture-output python -X utf8 \
    .claude/skills/energy-forecast/scripts/solar_shift_check.py [--max_lag 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[4]

p = argparse.ArgumentParser()
p.add_argument("--fc_col", default="solar_fc_dayahead")
p.add_argument("--act_col", default="gen_solar",
               help="αν λείπει, ανακατασκευάζεται από gen_solar_lag1.shift(-1)")
p.add_argument("--max_lag", type=int, default=6)
args = p.parse_args()

df = pd.read_parquet(BASE / "data" / "processed" / "hourly.parquet")
if args.fc_col not in df.columns:
    sys.exit(f"❌ Στήλη '{args.fc_col}' δεν υπάρχει στο hourly.parquet")

if args.act_col in df.columns:
    act_raw = df[args.act_col].astype(float)
elif f"{args.act_col}_lag1" in df.columns:
    # Το feature store σωστά ΔΕΝ κρατάει same-day actual gen (leakage) — μόνο lags.
    # Ανακατασκευή: act(t) = act_lag1(t+1).
    act_raw = df[f"{args.act_col}_lag1"].astype(float).shift(-1)
    print(f"ℹ️  '{args.act_col}' δεν υπάρχει — ανακατασκευή από {args.act_col}_lag1.shift(-1)")
else:
    sys.exit(f"❌ Ούτε '{args.act_col}' ούτε '{args.act_col}_lag1' υπάρχουν στο parquet")

both = pd.DataFrame({args.fc_col: df[args.fc_col].astype(float),
                     args.act_col: act_raw}).dropna()
fc, act = both[args.fc_col], both[args.act_col]
print(f"solar shift check: corr({args.fc_col}(t), {args.act_col}(t+k)) — {len(both)} κοινές ώρες")
print("Σωστό: peak σε k=0 ΠΑΝΤΟΥ. Peak σε k≠0 σε κάποιο έτος = timestamp shift ΕΚΕΙ.\n")

rows = []
for year, g in both.groupby(both.index.year):
    if len(g) < 24 * 30:  # <1 μήνας δεδομένα — skip
        continue
    f, a = g[args.fc_col], g[args.act_col]
    corrs = {k: f.corr(a.shift(-k)) for k in range(-args.max_lag, args.max_lag + 1)}
    corrs = {k: v for k, v in corrs.items() if pd.notna(v)}
    kbest = max(corrs, key=lambda k: abs(corrs[k]))
    # μέση ώρα-peak (UTC) των δύο σειρών μέσα στο έτος
    fc_peak_h = int(f.groupby(f.index.hour).mean().idxmax())
    act_peak_h = int(a.groupby(a.index.hour).mean().idxmax())
    rows.append((year, len(g), kbest, corrs[kbest], corrs.get(0, np.nan),
                 fc_peak_h, act_peak_h, fc_peak_h - act_peak_h))

out = pd.DataFrame(rows, columns=[
    "year", "hours", "peak_k", "corr@peak", "corr@0",
    "fc_peak_hUTC", "act_peak_hUTC", "Δpeak_h"])
print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

print("-" * 70)
bad = out[(out["peak_k"] != 0) | (out["Δpeak_h"].abs() >= 2)]
if len(bad):
    print(f"⚠️  Ύποπτα έτη: {sorted(bad['year'].tolist())} — peak_k≠0 ή |Δpeak_h|≥2.")
    print("   Επόμενο βήμα: spot-check στις ίδιες ημερομηνίες στο ENTSO-E transparency UI.")
else:
    print("✅ Κανένα έτος με peak_k≠0 ή απόκλιση peak-ώρας fc↔actual ≥2h — ευθυγράμμιση ΟΚ.")
