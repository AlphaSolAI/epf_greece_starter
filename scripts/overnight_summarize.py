# -*- coding: utf-8 -*-
"""
overnight_summarize.py — Πρωινή σύνοψη του overnight_20260705 batch.

Διαβάζει όλα τα runs/overnight_20260705/**/*.json, βγάζει MAE ανά block/config,
γράφει results/overnight_20260705.csv και τυπώνει πίνακες ανά block.
Το MAE υπολογίζεται ΑΠΟ ΤΑ ΙΔΙΑ τα arrays (actual vs series) — όχι από
αντιγραμμένα νούμερα (Α5 traceability).

Χρήση: conda run -n epf --no-capture-output python -X utf8 scripts/overnight_summarize.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs" / "overnight_20260705"
OUT_CSV = BASE / "results" / "overnight_20260705.csv"

rows = []
for f in sorted(RUNS.rglob("*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        rows.append({"block": f.parent.name, "run": f.stem, "mae": None,
                     "note": f"UNREADABLE: {e}"})
        continue
    note = ""
    mae = None
    y = d.get("actual")
    s = d.get("series")
    if isinstance(s, dict):
        # master_forecast: {"LGBM-recursive-dam": [...]} (ένα μοντέλο) ·
        # conformal: προτίμησε p50/median αν υπάρχουν
        s2 = s.get("p50") or s.get("median") or s.get("point")
        if s2 is None and len(s) == 1:
            s2 = next(iter(s.values()))
        elif s2 is not None:
            note = "p50"
        s = s2
    if y is not None and s is not None and len(y) == len(s) and len(y) > 0:
        a = np.asarray(y, dtype=float)
        p = np.asarray(s, dtype=float)
        m = ~(np.isnan(a) | np.isnan(p))
        if m.any():
            mae = float(np.mean(np.abs(a[m] - p[m])))
    if mae is None:
        met = d.get("metrics")
        if isinstance(met, list) and met and isinstance(met[0], dict):
            met = met[0]  # master_forecast: λίστα με ένα dict ανά μοντέλο
        if isinstance(met, dict):
            for k in ("MAE", "mae"):
                if k in met:
                    mae = float(met[k])
                    note = note + " metrics-key"
                    break
    if mae is None:
        note = note + " NO-MAE(κοίτα το JSON χειροκίνητα)"
    rows.append({
        "block": f.parent.name, "run": f.stem, "mae": mae,
        "retrain": d.get("retrain"), "strategy": d.get("strategy"),
        "task": d.get("task"), "n_hours": len(d.get("dates") or []),
        "note": note.strip(),
    })

OUT_CSV.parent.mkdir(exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["block", "run", "mae", "retrain",
                                       "strategy", "task", "n_hours", "note"])
    w.writeheader()
    w.writerows(rows)

cur = None
for r in rows:
    if r["block"] != cur:
        cur = r["block"]
        print(f"\n===== {cur} =====")
    mae = f"{r['mae']:.3f}" if r["mae"] is not None else "  —  "
    print(f"  {mae}  {r['run']}  {r['note']}")

n_ok = sum(1 for r in rows if r["mae"] is not None)
print(f"\nΣύνολο: {len(rows)} runs, {n_ok} με MAE, {len(rows)-n_ok} προβληματικά")
print(f"CSV: {OUT_CSV}")
