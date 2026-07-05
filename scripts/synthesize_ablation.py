# -*- coding: utf-8 -*-
"""
synthesize_ablation.py — ΔMAE πίνακες + αυτόματο ABLATION_PLAN §2 pre-gate από run JSONs.

Διαβάζει έναν φάκελο με run JSONs του master_forecast (σύμβαση ονόματος
<window>_<algo>_<mode>_<spec>.json), υπολογίζει ΔMAE κάθε spec έναντι του baseline
ανά συνθήκη (window, algo, mode), και περνάει κάθε spec από το §2 pre-gate:

  |ΔMAE| > 0.15  ΚΑΙ  ίδιο πρόσημο σε ≥2 συνθήκες
  + ο κανόνας ανεξαρτησίας (μάθημα 2026-07-05, validity-reviewer):
    κελιά του ΙΔΙΟΥ window είναι συσχετισμένα — για πλήρες ACCEPT χρειάζονται
    ≥2 ΔΙΑΦΟΡΕΤΙΚΑ windows με συνεπές πρόσημο, αλλιώς PENDING(1-window).

Επίσης κάνει built-in validation (μάθημα spot-check): αν το recomputed MAE από
actual/series αποκλίνει από το metrics MAE του JSON > 0.005 → WARN.

ΠΡΟΣΟΧΗ: το output είναι ΥΠΟΨΗΦΙΑ verdicts (pre-gate). Τελικό ACCEPTED μπαίνει στο
ABLATION_PLAN ΜΟΝΟ μετά από validity-reviewer πάνω στο ΣΥΝΟΛΟ των στοιχείων
(συμπεριλαμβανομένων παλαιότερων windows από άλλα σετ runs).

Χρήση (system python, ΧΩΡΙΣ conda — δεν μπλοκάρει την ουρά training):
  python scripts/synthesize_ablation.py --dir runs/overnight_20260705/b_march
  python scripts/synthesize_ablation.py --dir runs/overnight_20260705/a_cadence --baseline default
  python scripts/synthesize_ablation.py --dir <dir> --csv results/<name>.csv
"""
import argparse
import csv as csvmod
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TOL = 0.15  # §2: |ΔMAE| πάνω από αυτό μετράει ως πραγματικό effect


def get_series(d):
    s = d.get("series")
    if isinstance(s, dict):
        for k in ("p50", "median", "point"):
            if k in s:
                return s[k]
        if len(s) == 1:
            return next(iter(s.values()))
    return s


def get_metrics_mae(d):
    m = d.get("metrics")
    if isinstance(m, list) and m:
        m = m[0]
    if isinstance(m, dict):
        for k in ("MAE", "mae"):
            if k in m:
                return float(m[k])
    return None


def extract_mae(path):
    """(mae, warn) — metrics MAE, validated έναντι recompute από actual/series."""
    d = json.load(open(path, encoding="utf-8"))
    jm = get_metrics_mae(d)
    actual, pred = d.get("actual"), get_series(d)
    rec = None
    if actual and pred:
        pairs = [(a, p) for a, p in zip(actual, pred) if a is not None and p is not None]
        if pairs:
            rec = sum(abs(a - p) for a, p in pairs) / len(pairs)
    warn = None
    if jm is not None and rec is not None and abs(jm - rec) > 0.005:
        warn = f"metrics MAE {jm:.4f} ≠ recomputed {rec:.4f}"
    return (jm if jm is not None else rec), warn


def parse_stem(stem):
    """<window>_<algo>_<mode>_<spec> → (window, algo, mode, spec)."""
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], "_".join(parts[3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="φάκελος με run JSONs")
    ap.add_argument("--baseline", default="default", help="spec-slug του baseline")
    ap.add_argument("--csv", default=None, help="προαιρετικό CSV output")
    args = ap.parse_args()

    runs = {}   # (window, algo, mode) -> {spec: mae}
    warns = []
    for p in sorted(Path(args.dir).glob("*.json")):
        parsed = parse_stem(p.stem)
        if not parsed:
            warns.append(f"SKIP (όνομα εκτός σύμβασης): {p.name}")
            continue
        w, a, m, spec = parsed
        mae, warn = extract_mae(p)
        if mae is None:
            warns.append(f"SKIP (χωρίς MAE): {p.name}")
            continue
        if warn:
            warns.append(f"{p.name}: {warn}")
        runs.setdefault((w, a, m), {})[spec] = mae

    conditions = sorted(runs)
    specs = sorted({s for v in runs.values() for s in v} - {args.baseline})

    # Πίνακας ΔMAE
    print(f"\nΔMAE vs '{args.baseline}' (θετικό = χειρότερο από baseline) — {args.dir}")
    header = f"{'spec':38s}" + "".join(f"{'/'.join(c):>22s}" for c in conditions)
    print(header)
    print("-" * len(header))
    base_row = f"{args.baseline + ' (MAE)':38s}"
    for c in conditions:
        b = runs[c].get(args.baseline)
        base_row += f"{b:22.3f}" if b is not None else f"{'—':>22s}"
    print(base_row)

    deltas = {}  # spec -> {condition: delta}
    for spec in specs:
        row = f"{spec:38s}"
        for c in conditions:
            b, v = runs[c].get(args.baseline), runs[c].get(spec)
            if b is None or v is None:
                row += f"{'—':>22s}"
            else:
                d = v - b
                deltas.setdefault(spec, {})[c] = d
                row += f"{d:+22.3f}"
        print(row)

    # §2 pre-gate ανά spec
    print(f"\n§2 PRE-GATE (|Δ|>{TOL} & ίδιο πρόσημο· ανεξαρτησία = διαφορετικά windows):")
    for spec in specs:
        ds = deltas.get(spec, {})
        strong = {c: d for c, d in ds.items() if abs(d) > TOL}
        if not strong:
            print(f"  {spec:38s} NOISE (κανένα |Δ|>{TOL} σε {len(ds)} συνθήκες)")
            continue
        pos = [c for c, d in strong.items() if d > 0]
        neg = [c for c, d in strong.items() if d < 0]
        if pos and neg:
            print(f"  {spec:38s} MIXED ({len(neg)}− / {len(pos)}+ σε {len(ds)} συνθήκες)")
            continue
        side = neg or pos
        sign = "ΒΟΗΘΑΕΙ (Δ<0)" if neg else "ΒΛΑΠΤΕΙ (Δ>0)"
        wins = {c[0] for c in side}
        n_weak = len(ds) - len(strong)
        weak_note = f", {n_weak} κάτω από {TOL}" if n_weak else ""
        if len(side) >= 2 and len(wins) >= 2:
            print(f"  {spec:38s} ✅ ACCEPT-candidate — {sign}, {len(side)}/{len(ds)} συνθήκες, "
                  f"{len(wins)} ανεξάρτητα windows{weak_note}")
        elif len(side) >= 2:
            print(f"  {spec:38s} ⚠️ PENDING(1-window) — {sign} συνεπές ({len(side)}/{len(ds)}{weak_note}) "
                  f"αλλά ΟΛΑ στο ίδιο window '{list(wins)[0]}' — θέλει 2ο ανεξάρτητο window")
        else:
            print(f"  {spec:38s} ⚠️ PENDING(1-condition) — μόνο 1 συνθήκη με |Δ|>{TOL}")

    if warns:
        print("\n⚠️ WARNINGS:")
        for w in warns:
            print(f"  {w}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wtr = csvmod.writer(f)
            wtr.writerow(["window", "algo", "mode", "spec", "mae", "delta_vs_baseline"])
            for c in conditions:
                for spec, mae in sorted(runs[c].items()):
                    b = runs[c].get(args.baseline)
                    d = (mae - b) if (b is not None and spec != args.baseline) else ""
                    wtr.writerow([*c, spec, f"{mae:.4f}", f"{d:+.4f}" if d != "" else ""])
        print(f"\nCSV → {args.csv}")

    print("\nΥπενθύμιση: pre-gate ≠ τελικό verdict. ACCEPTED στο ABLATION_PLAN μόνο μετά "
          "από validity-reviewer πάνω σε ΟΛΟ το evidence (και παλαιότερα windows).")


if __name__ == "__main__":
    main()
