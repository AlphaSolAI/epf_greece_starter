# -*- coding: utf-8 -*-
"""
lagscan.py — Leakage detector για νέα features (βήματα β+γ του §1 pre-flight πρωτοκόλλου).

Ιστορικό: αυτό το εργαλείο (ως ad-hoc script) έπιασε ΔΥΟ πραγματικά bugs:
  (1) 1h timestamp shift στο fetch_entsoe_xborder.py (peak σε lag+1 αντί lag=0)
  (2) το xborder same-day conceptual leakage (ύποπτα υψηλή same-hour συσχέτιση 0.91)
Κανόνας ερμηνείας: το peak πρέπει να είναι ΕΚΕΙ που προβλέπει η θεωρία δημοσίευσης.
Peak σε «βολικό» σημείο (lag 0/αρνητικό για πηγή που δημοσιεύεται ΜΕΤΑ το gate) = leakage.
Ύποπτα υψηλή συσχέτιση (>0.85) με το y = έλεγξε μήπως μοιράζεται το ίδιο auction/πηγή.

Χρήση:
  conda run -n epf --no-capture-output python -X utf8 \
    .claude/skills/energy-forecast/scripts/lagscan.py --col <στήλη> [--task price] [--max_lag 48]
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
p.add_argument("--col", required=True, help="στήλη προς έλεγχο (πρέπει να υπάρχει στο parquet)")
p.add_argument("--task", default="price", choices=["price", "load"])
p.add_argument("--max_lag", type=int, default=48)
args = p.parse_args()

pq = BASE / "data" / "processed" / ("hourly.parquet" if args.task == "price" else "hourly_load.parquet")
df = pd.read_parquet(pq)
if args.col not in df.columns:
    sys.exit(f"❌ Στήλη '{args.col}' δεν υπάρχει. Διαθέσιμες που μοιάζουν: "
             f"{[c for c in df.columns if args.col.split('_')[0] in c][:10]}")

s = df[args.col].astype(float)
y = df["y"].astype(float)
both = pd.concat([s, y], axis=1).dropna()
s, y = both.iloc[:, 0], both.iloc[:, 1]
print(f"lag-scan: corr(y(t), {args.col}(t−k)) — {len(both)} κοινές ώρες, k∈[−{args.max_lag},{args.max_lag}]")
print("k>0: η στήλη ΠΡΟΗΓΕΙΤΑΙ του y (νόμιμη κατεύθυνση αν δημοσιεύεται πριν) · k<0: η στήλη ΕΠΕΤΑΙ (πάντα ύποπτο)")

rows = []
for k in range(-args.max_lag, args.max_lag + 1):
    rows.append((k, y.corr(s.shift(k))))
res = pd.DataFrame(rows, columns=["lag_k", "corr"]).dropna()
top = res.reindex(res["corr"].abs().sort_values(ascending=False).index).head(7)
print(top.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

peak_k = int(top.iloc[0]["lag_k"])
peak_c = float(top.iloc[0]["corr"])
print("-" * 50)
if abs(peak_c) > 0.85:
    print(f"⚠️  |corr|={abs(peak_c):.2f} > 0.85 στο k={peak_k} — ΠΟΛΥ υψηλή. Έλεγξε μήπως η πηγή "
          f"μοιράζεται το ίδιο auction/μηχανισμό με το target (βλ. xborder post-mortem).")
if peak_k <= 0:
    print(f"⚠️  Peak σε k={peak_k} ≤ 0 — το y «προβλέπει» τη στήλη ή είναι σύγχρονα. "
          f"Νόμιμο ΜΟΝΟ αν η στήλη αποδεδειγμένα δημοσιεύεται ΠΡΙΝ το gate για την ώρα-στόχο.")
else:
    print(f"Peak σε k={peak_k} (η στήλη προηγείται). Συμβατό με νόμιμο lag αν k ≥ το θεωρητικό ελάχιστο.")

hod = s.groupby(s.index.hour).mean()
print(f"\nHour-of-day profile (sanity): peak ώρα {int(hod.idxmax())}:00, min ώρα {int(hod.idxmin())}:00")
print("(index = CET/CEST-naive μετά το TZFIX 2026-07-04 → σωστό solar peak ≈ 11:00 DJF / 12:00 JJA.")
print(" Για οριστικό timestamp-shift check: scripts/solar_shift_check.py — fc vs actual, ανά έτος)")
