#!/usr/bin/env python
"""Αριθμητική ισοδυναμία 2 run JSONs (spec §4α — «πιο γρήγορο» μόνο με ίδια νούμερα).

System python, stdlib ΜΟΝΟ. Exit 0 = ισοδύναμα, 1 = όχι (ή μη συγκρίσιμα).
CONFIG-MISMATCH (άλλο window/gate/retrain/features) => μη συγκρίσιμα by design —
ίδιος κανόνας με το validity gate (ΠΟΤΕ σύγκριση across windows/gates).
"""
import argparse
import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CONFIG_KEYS = ["strategy", "method", "task", "market", "gate", "retrain",
               "features", "crosslag_mode", "horizon", "stride"]
QUANTILE_KEYS = ("p10", "p50", "p90")


def fingerprint(d):
    fp = {k: d.get(k) for k in CONFIG_KEYS}
    dates = d.get("dates") or []
    fp["window"] = (dates[0] if dates else d.get("test_start"),
                    dates[-1] if dates else d.get("test_end"))
    return fp


def prediction_arrays(d):
    if isinstance(d.get("series"), dict):
        return dict(d["series"])
    return {k: d[k] for k in QUANTILE_KEYS if k in d}


def mae_table(d):
    out = {}
    for m in d.get("metrics") or []:
        if isinstance(m, dict) and "MAE" in m:
            out[m.get("Model", "?")] = m["MAE"]
    return out


def compare(a, b, tol=1e-9, mae_tol=1e-6):
    reasons = []
    fa, fb = fingerprint(a), fingerprint(b)
    for k in fa:
        if fa[k] != fb[k]:
            reasons.append(f"CONFIG-MISMATCH: {k}: {fa[k]!r} != {fb[k]!r}")
    if reasons:
        return {"equivalent": False, "reasons": reasons}

    pa, pb = prediction_arrays(a), prediction_arrays(b)
    if set(pa) != set(pb):
        reasons.append(f"SERIES-MISMATCH: {sorted(pa)} != {sorted(pb)}")
    for key in sorted(set(pa) & set(pb)):
        xs, ys = pa[key], pb[key]
        if len(xs) != len(ys):
            reasons.append(f"PRED-DIFF: {key}: μήκη {len(xs)} != {len(ys)}")
            continue
        worst, worst_i = 0.0, -1
        for i, (x, y) in enumerate(zip(xs, ys)):
            dxy = abs(x - y)
            if dxy > worst:
                worst, worst_i = dxy, i
        if worst > tol:
            reasons.append(f"PRED-DIFF: {key}: max|diff|={worst:.3e} @ index {worst_i} > tol={tol:.0e}")

    ma, mb = mae_table(a), mae_table(b)
    for key in sorted(set(ma) & set(mb)):
        if abs(ma[key] - mb[key]) > mae_tol:
            reasons.append(f"MAE-DIFF: {key}: {ma[key]} vs {mb[key]} (> {mae_tol})")

    return {"equivalent": not reasons, "reasons": reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ισοδυναμία 2 run JSONs (spec §4α)")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--mae_tol", type=float, default=1e-6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    r = compare(a, b, tol=args.tol, mae_tol=args.mae_tol)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("EQUIVALENT" if r["equivalent"] else "NOT EQUIVALENT")
        for s in r["reasons"]:
            print(" -", s)
    sys.exit(0 if r["equivalent"] else 1)


if __name__ == "__main__":
    main()
