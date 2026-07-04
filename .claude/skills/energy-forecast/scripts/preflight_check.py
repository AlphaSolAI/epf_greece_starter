# -*- coding: utf-8 -*-
"""
preflight_check.py — Pre-flight έλεγχος πριν από ΚΑΘΕ batch πειραμάτων (SKILL §pre-flight).

Ελέγχει (γρήγορα, χωρίς training):
  1. OneDrive.exe τρέχει (αλλιώς raw CSV reads σκάνε με OSError 22)
  2. hourly.parquet: υπάρχει, φρέσκο, ΜΟΝΟ lagged xb_* στήλες (same-day = leakage)
  3. hourly_load.parquet: υπάρχει
  4. Βασικές στήλες παρούσες (y, load_fc, gen_solar_lag24, gas_price)

Με --baseline τυπώνει και την εντολή αναπαραγωγής του baseline —
ΔΕΝ την τρέχει (training ~70s, τρέξ' την χωριστά/background).
Baseline ΜΕΤΑ το TZFIX 2026-07-04: default static Q1 = 19.17±0.05 (πριν: 16.10 σε
misaligned δεδομένα — βλ. ABLATION_PLAN §5.9· το 19.17 κουβαλά ακόμα το §5.10 leak).

Χρήση:
  conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/preflight_check.py
Exit code: 0 = όλα PASS, 1 = κάποιο FAIL.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[4]  # project root (skills/energy-forecast/scripts -> root)
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 60)
print("PRE-FLIGHT CHECK (energy-forecast)")
print("=" * 60)

# 1. OneDrive
try:
    out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=30).stdout
    check("OneDrive.exe τρέχει", "OneDrive.exe" in out,
          "αν όχι: Start-Process \"$env:PROGRAMFILES\\Microsoft OneDrive\\OneDrive.exe\"")
except Exception as e:
    check("OneDrive check", False, str(e))

# 2. hourly.parquet
pq = BASE / "data" / "processed" / "hourly.parquet"
if not pq.exists():
    check("hourly.parquet υπάρχει", False, str(pq))
else:
    df = pd.read_parquet(pq)
    check("hourly.parquet υπάρχει", True, f"shape={df.shape}, τέλος={df.index.max()}")
    xb_all = [c for c in df.columns if c.startswith("xb_")]
    xb_bad = [c for c in xb_all if "_lag" not in c]
    check("ΜΟΝΟ lagged xb_* στήλες (same-day = leakage!)", not xb_bad,
          f"same-day βρέθηκαν: {xb_bad}" if xb_bad else f"{len(xb_all)} lagged cols OK")
    needed = ["y", "load_fc", "gen_solar_lag24", "gas_price"]
    missing = [c for c in needed if c not in df.columns]
    check("Βασικές στήλες παρούσες", not missing, f"λείπουν: {missing}" if missing else "")
    # TZFIX guard (2026-07-04, §5.9): index=CET/CEST => solar_fc peak πρέπει 11:00 DJF.
    # Αν ξαναγίνει misalignment (νέο fetch/rebuild με λάθος frame), πιάνεται εδώ.
    if "solar_fc_dayahead" in df.columns:
        s = df["solar_fc_dayahead"].astype(float)
        s = s[s.index.month.isin([12, 1, 2])]
        pk = int(s.groupby(s.index.hour).mean().idxmax())
        check("TZ alignment (solar_fc DJF peak = 11:00 CET)", pk == 11,
              f"peak={pk}:00 — αν ≠11 τρέξε solar_shift_check.py (βλ. ABLATION_PLAN §5.9)")

# 3. hourly_load.parquet
check("hourly_load.parquet υπάρχει", (BASE / "data" / "processed" / "hourly_load.parquet").exists())

print("-" * 60)
if "--baseline" in sys.argv:
    print("Baseline reproduction (τρέξε χωριστά, ~70s, αναμενόμενο MAE=19.17±0.05 — TZFIX data):")
    print('  conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \\')
    print('    --algo lgbm --task price --market dam --strategy recursive --gate strict \\')
    print('    --retrain static --train_end "2025-11-30 23:00" \\')
    print('    --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \\')
    print('    --features default --out_json runs/preflight_out/baseline_check.json')

print(f"ΑΠΟΤΕΛΕΣΜΑ: {'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
sys.exit(0 if not FAILS else 1)
