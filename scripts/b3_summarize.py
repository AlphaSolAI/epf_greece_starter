# -*- coding: utf-8 -*-
"""
b3_summarize.py — Συγκεντρωτικός πίνακας του Β3 re-ablation (leak-free, TZFIX+AEL).

Διαβάζει όλα τα results/b3_*.csv (έξοδοι run_ablation, 1ο spec = baseline) και
τυπώνει: (α) πίνακα MAE ανά spec × (algo, window, strategy), (β) ΔMAE vs default
ανά συνθήκη, (γ) αυτόματο acceptance check ανά feature-ερώτημα με τον κανόνα
ABLATION_PLAN §2: |ΔMAE| > 0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ανεξάρτητες συνθήκες.

Χρήση: conda run -n epf --no-capture-output python -X utf8 scripts/b3_summarize.py
"""
import csv
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

# b3_<window>_<algo>_<strat>.csv, π.χ. b3_q1_lgbm_rec.csv / b3_summer_xgb_dir.csv
PAT = re.compile(r"^b3_(?P<window>q1|summer)_(?P<algo>lgbm|xgb)_(?P<strat>rec|dir)\.csv$")

conditions = {}  # (window, algo, strat) -> {spec: MAE}
for f in sorted(RESULTS.glob("b3_*.csv")):
    m = PAT.match(f.name)
    if not m:
        continue
    key = (m["window"], m["algo"], m["strat"])
    with open(f, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    conditions[key] = {r["spec"]: (float(r["MAE"]) if r["MAE"] else None) for r in rows}

if not conditions:
    print("Κανένα results/b3_*.csv ακόμα.")
    sys.exit(0)

all_specs = []
for d in conditions.values():
    for s in d:
        if s not in all_specs:
            all_specs.append(s)

keys = sorted(conditions.keys())
hdr = f"{'spec':<28}" + "".join(f"{w}/{a}/{s:<3}".rjust(16) for (w, a, s) in keys)
print("=" * len(hdr)); print("Β3 RE-ABLATION — MAE (€/MWh), leak-free (TZFIX+AEL), static/strict/DAM/price")
print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
for spec in all_specs:
    line = f"{spec:<28}"
    for k in keys:
        v = conditions[k].get(spec)
        line += (f"{v:>16.3f}" if v is not None else f"{'—':>16}")
    print(line)

print("\nΔMAE vs default (θετικό = χειρότερο από default):")
print(hdr); print("-" * len(hdr))
for spec in all_specs:
    if spec == "default":
        continue
    line = f"{spec:<28}"
    for k in keys:
        v, b = conditions[k].get(spec), conditions[k].get("default")
        line += (f"{v-b:>+16.3f}" if (v is not None and b is not None) else f"{'—':>16}")
    print(line)

# Acceptance ανά ερώτημα: αφαίρεση ομάδας → ΔMAE = MAE(χωρίς ομάδα) − MAE(default).
# ΔMAE > +0.15 σε ≥2 συνθήκες (ίδιο πρόσημο) = η ομάδα ΒΟΗΘΑΕΙ (accepted)·
# ΔMAE < −0.15 = η ομάδα ΒΛΑΠΤΕΙ.
QUESTIONS = {
    "resfc": "default,-resfc",
    "meteo": "default,-meteo",
    "genlags+loadlags": "default,-genlags,-loadlags",
    "dense (πρόσθεση)": "default,dense",
    "lean core vs default": "lags,calendar,genlags",
    "bare core vs default": "lags,calendar",
}
print("\nACCEPTANCE (§2: |ΔMAE|>0.15 & ίδιο πρόσημο σε ≥2 συνθήκες):")
for qname, spec in QUESTIONS.items():
    deltas = []
    for k in keys:
        v, b = conditions[k].get(spec), conditions[k].get("default")
        if v is not None and b is not None:
            deltas.append((k, v - b))
    if not deltas:
        continue
    sig = [(k, d) for k, d in deltas if abs(d) > 0.15]
    pos = [x for x in sig if x[1] > 0]
    neg = [x for x in sig if x[1] < 0]
    if len(pos) >= 2 and not neg:
        verdict = "ACCEPTED: αφαίρεση χειροτερεύει σταθερά (η ομάδα/variant προσφέρει)"
    elif len(neg) >= 2 and not pos:
        verdict = "ACCEPTED: variant σταθερά ΚΑΛΥΤΕΡΟ από default"
    elif pos and neg:
        verdict = "MIXED (αντίθετα πρόσημα) → PENDING/εξαρτάται από συνθήκη"
    elif len(sig) == 1:
        verdict = "PENDING: σημαντικό σε 1 μόνο συνθήκη"
    else:
        verdict = "NEUTRAL: |ΔMAE|≤0.15 παντού"
    ds = "  ".join(f"{w}/{a}/{s}:{d:+.2f}" for (w, a, s), d in deltas)
    print(f"  {qname:<24} {verdict}\n      {ds}")
