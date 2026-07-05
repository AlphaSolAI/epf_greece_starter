# -*- coding: utf-8 -*-
"""
overnight_summarize.py — Πρωινή σύνοψη του overnight_20260705 batch.

Διαβάζει όλα τα runs/overnight_20260705/**/*.json, υπολογίζει MAE ΑΠΟ ΤΑ ΙΔΙΑ
τα arrays (actual vs series — Α5 traceability, όχι αντιγραφή), γράφει
results/overnight_20260705.csv και τυπώνει ανά block:
  - a_cadence: πίνακας MAE spec × (window, algo, cadence)
  - b_march:   ΔMAE vs default ανά (algo, strategy) — tie-break υλικό για §2
  - c_load:    ΔMAE vs default ανά (window, algo, strategy) — πρώτο load ablation
  - d_fill / e_ss / f_conformal: λίστα MAE
Ο κανόνας αποδοχής (§2 ABLATION_PLAN) εφαρμόζεται ΣΥΝΔΥΑΣΤΙΚΑ με τα B3 (8
συνθήκες) από το πρωινό session — εδώ τυπώνονται τα Δ, όχι τελικά verdicts.

Χρήση: conda run -n epf --no-capture-output python -X utf8 scripts/overnight_summarize.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs" / "overnight_20260705"
OUT_CSV = BASE / "results" / "overnight_20260705.csv"


def extract_mae(d):
    """MAE από arrays· fallback στο metrics dict/λίστα. Επιστρέφει (mae, note)."""
    note = ""
    y, s = d.get("actual"), d.get("series")
    if isinstance(s, dict):
        s2 = s.get("p50") or s.get("median") or s.get("point")
        if s2 is None and len(s) == 1:
            s2 = next(iter(s.values()))
        elif s2 is not None:
            note = "p50"
        s = s2
    if y is not None and s is not None and len(y) == len(s) and len(y) > 0:
        a, p = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
        m = ~(np.isnan(a) | np.isnan(p))
        if m.any():
            return float(np.mean(np.abs(a[m] - p[m]))), note
    met = d.get("metrics")
    if isinstance(met, list) and met and isinstance(met[0], dict):
        met = met[0]
    if isinstance(met, dict):
        for k in ("MAE", "mae"):
            if k in met:
                return float(met[k]), (note + " metrics-key").strip()
    return None, (note + " NO-MAE").strip()


def parse_name(block, stem):
    """
    Ονοματολογία του overnight_20260705.sh:
      a_cadence: {win}_{algo}_{cadence}_{spec}
      b_march:   march_{algo}_{rec|dir}_{spec}
      c_load:    {win}_{algo}_{rec|dir}_{spec}
      d_fill:    {win}_{algo}_{dir|rec}_{spec}[_seedN]
      e_ss:      {win}_{algo}_rec_ss_{spec}
    """
    t = stem.split("_")
    if len(t) < 4:
        return {}
    info = {"window": t[0], "algo": t[1], "mode": t[2]}
    rest = t[3:]
    if block == "e_ss" and rest and rest[0] == "ss":
        rest = rest[1:]
        info["mode"] = "rec+ss"
    if rest and rest[-1].startswith("seed"):
        info["seed"] = rest[-1]
        rest = rest[:-1]
    info["spec"] = "_".join(rest)
    return info


rows = []
for f in sorted(RUNS.rglob("*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        rows.append({"block": f.parent.name, "run": f.stem, "mae": None,
                     "note": f"UNREADABLE: {e}"})
        continue
    mae, note = extract_mae(d)
    r = {"block": f.parent.name, "run": f.stem, "mae": mae,
         "retrain": d.get("retrain"), "strategy": d.get("strategy"),
         "task": d.get("task"), "n_hours": len(d.get("dates") or []), "note": note}
    r.update(parse_name(f.parent.name, f.stem))
    rows.append(r)

OUT_CSV.parent.mkdir(exist_ok=True)
FIELDS = ["block", "run", "mae", "retrain", "strategy", "task", "n_hours",
          "window", "algo", "mode", "spec", "seed", "note"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

by_block = defaultdict(list)
for r in rows:
    by_block[r["block"]].append(r)


def fmt(v):
    return f"{v:7.3f}" if v is not None else "    —  "


# ---- a_cadence: spec × (window, algo, cadence)
if by_block.get("a_cadence"):
    print("\n===== A. CADENCE (MAE) — υποψήφιο νέο headline =====")
    cell = {}
    for r in by_block["a_cadence"]:
        cell[(r.get("spec"), r.get("window"), r.get("algo"), r.get("mode"))] = r["mae"]
    conds = sorted({(w, a, m) for (_, w, a, m) in cell})
    specs = sorted({s for (s, _, _, _) in cell})
    hdr = "  ".join(f"{w}/{a}/{m}" for (w, a, m) in conds)
    print(f"{'spec':32s}  {hdr}")
    for s in specs:
        vals = "  ".join(fmt(cell.get((s, w, a, m))) for (w, a, m) in conds)
        print(f"{s:32s}  {vals}")

# ---- b_march / c_load: ΔMAE vs default ανά συνθήκη
for blk, title in (("b_march", "B. ΜΑΡΤΙΟΣ tie-break"), ("c_load", "C. LOAD ablation")):
    rs = by_block.get(blk)
    if not rs:
        continue
    print(f"\n===== {title} — ΔMAE vs default (θετικό = χειρότερο από default) =====")
    groups = defaultdict(dict)
    for r in rs:
        groups[(r.get("window"), r.get("algo"), r.get("mode"))][r.get("spec")] = r["mae"]
    for cond in sorted(groups):
        d0 = groups[cond].get("default")
        tag = "/".join(str(c) for c in cond)
        if d0 is None:
            print(f"  {tag}: ΛΕΙΠΕΙ το default baseline — δες log")
            continue
        print(f"  {tag} (default={d0:.3f}):")
        for spec, mae in sorted(groups[cond].items()):
            if spec == "default" or mae is None:
                continue
            print(f"      {mae - d0:+7.3f}  {spec}  (MAE {mae:.3f})")

# ---- υπόλοιπα blocks: flat λίστα
for blk in ("d_fill", "e_ss", "f_conformal"):
    rs = by_block.get(blk)
    if not rs:
        continue
    print(f"\n===== {blk} =====")
    for r in rs:
        extra = f" {r.get('seed')}" if r.get("seed") else ""
        print(f"  {fmt(r['mae'])}  {r['run']}{extra}  {r['note']}")

bad = [r for r in rows if r["mae"] is None]
print(f"\nΣύνολο: {len(rows)} runs, {len(rows) - len(bad)} με MAE, {len(bad)} προβληματικά")
for r in bad:
    print(f"  ⚠️ {r['block']}/{r['run']}: {r['note']}")
print(f"CSV: {OUT_CSV}")
