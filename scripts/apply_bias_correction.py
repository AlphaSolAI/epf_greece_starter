# -*- coding: utf-8 -*-
"""Trailing per-hour bias correction πάνω σε ΥΠΑΡΧΟΝΤΑ run JSONs — pure post-processing.

    pred'(d,h) = pred(d,h) + mean_{ημέρες d-lag_days-W+1 .. d-lag_days}(act - pred)(h)

ΚΑΝΕΝΑ training, κανένα μελλοντικό δεδομένο. Gate semantics (κρίσιμο):
η πρόβλεψη για τη μέρα d γίνεται στο gate D-1 (π.χ. 11:00 CET για g12), άρα η
τελευταία ΠΛΗΡΩΣ γνωστή μέρα residuals είναι η d-2 → default --lag_days 2 (gate-safe).
--lag_days 1 (χρησιμοποιεί και την d-1 ολόκληρη) είναι optimistic sensitivity variant,
ΟΧΙ deployable για D-1 gates.

Warmup: όσο δεν υπάρχουν W μέρες ιστορικού χρησιμοποιείται ό,τι υπάρχει (>= min_days),
αλλιώς μηδενική διόρθωση.

Χρήση:
    py -3.11 scripts/apply_bias_correction.py --json runs/load_contest/summer_lgbm_recw_g12_densemv.json \
        [--window 14 28] [--lag_days 2] [--min_days 5] [--out_csv results/bias_corr_<x>.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def load_run(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    idx = pd.to_datetime(d["dates"])
    s = d["series"]
    if isinstance(s, dict):
        first = next(iter(s.values()))
        vals = first["values"] if isinstance(first, dict) else first
    else:
        vals = s[0]["values"] if isinstance(s[0], dict) else s[0]
    pred = pd.Series(vals, index=idx, dtype=float)
    act = pd.Series(d["actual"], index=idx, dtype=float)
    return d, pred, act


def corrected_mae(pred: pd.Series, act: pd.Series, window: int, lag_days: int, min_days: int) -> float:
    err = act - pred
    E = err.to_frame("e")
    E["date"] = E.index.normalize()
    E["hour"] = E.index.hour
    piv = E.pivot_table(index="date", columns="hour", values="e")  # ημέρες x 24

    vals = piv.to_numpy()
    corr = np.zeros_like(vals)
    for i in range(len(piv)):
        hi = i - (lag_days - 1)  # τελευταία χρησιμοποιήσιμη γραμμή (exclusive): lag_days=2 -> έως i-2
        lo = max(0, hi - window)
        hist = vals[lo:hi]
        if hist.shape[0] >= min_days:
            corr[i] = np.nanmean(hist, axis=0)
    corr_df = pd.DataFrame(corr, index=piv.index, columns=piv.columns)
    corr_long = corr_df.stack()
    corr_ser = pd.Series(
        corr_long.values,
        index=[dt + pd.Timedelta(hours=int(h)) for dt, h in corr_long.index],
    ).reindex(pred.index).fillna(0.0)
    return float((act - (pred + corr_ser)).abs().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", required=True, help="run JSON(s) του master_forecast")
    ap.add_argument("--window", nargs="+", type=int, default=[14, 28], help="trailing παράθυρα σε ημέρες")
    ap.add_argument("--lag_days", type=int, default=2,
                    help="2=gate-safe (έως d-2, default) · 1=optimistic (έως d-1, ΟΧΙ deployable)")
    ap.add_argument("--min_days", type=int, default=5)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    if args.lag_days < 2:
        print("!! lag_days<2: optimistic variant — ΜΗ deployable για D-1 gates (μόνο sensitivity).")

    rows = []
    for p in args.json:
        path = Path(p)
        d, pred, act = load_run(path)
        mae0 = float((act - pred).abs().mean())
        for W in args.window:
            mae1 = corrected_mae(pred, act, W, args.lag_days, args.min_days)
            rows.append({"run": path.stem, "gate_gap": d.get("delay_gap"), "retrain": d.get("retrain"),
                         "W": W, "lag_days": args.lag_days,
                         "mae_before": round(mae0, 4), "mae_after": round(mae1, 4),
                         "delta": round(mae1 - mae0, 4)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n→ {out}")


if __name__ == "__main__":
    main()
