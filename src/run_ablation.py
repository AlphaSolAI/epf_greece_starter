"""
run_ablation.py — Feature-group ablation wrapper πάνω στο master_forecast engine.

Τρέχει το ΙΔΙΟ config (algo/task/market/strategy/retrain/gate/χρονικό διάστημα)
με διαφορετικά --features specs και βγάζει καθαρό πίνακα σύγκρισης (ΔMAE vs baseline).

Default πείραμα (μηδέν νέα downloads — ο Δεκ 2025 έχει πλήρη forecast/load_fc/meteo):
    LGBM / price / DAM / recursive / static / strict, Δεκέμβριος 2025
    specs: all | all,-forecast | all,-meteo | lags,calendar

Παραδείγματα:
  python -m src.run_ablation
  python -m src.run_ablation --algo xgb --strategy direct
  python -m src.run_ablation --task load --specs "all;all,-meteo;default"

Σημείωση: τα specs χωρίζονται με ';' γιατί το ',' ανήκει στη σύνταξη του spec.
Το 1ο spec της λίστας είναι το baseline του πίνακα.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .master_forecast import MARKET_PRESETS, add_dense_lags, run_forecast, _metrics
from .feature_availability import GateSpec, parse_feature_spec, select_features
from .split_utils import load_processed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SPECS = "all;all,-forecast;all,-meteo;lags,calendar"


def _slug(spec: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-")


def main():
    p = argparse.ArgumentParser(description="Feature-group ablation πάνω στο master_forecast")
    p.add_argument("--algo", default="lgbm", choices=["lgbm", "xgb", "mlp", "lstm", "lear"])
    p.add_argument("--task", default="price", choices=["price", "load"])
    p.add_argument("--market", default="dam", choices=list(MARKET_PRESETS))
    p.add_argument("--strategy", default="recursive", choices=["recursive", "direct"])
    p.add_argument("--retrain", default="static", choices=["static", "monthly", "weekly"])
    p.add_argument("--gate", default="strict", choices=["strict", "academic"])
    p.add_argument("--specs", default=DEFAULT_SPECS,
                   help="feature specs χωρισμένα με ';' (1ο = baseline)")
    p.add_argument("--train_start", default=None)
    p.add_argument("--train_end", default="2025-11-30 23:00")
    p.add_argument("--test_start", default="2025-12-01 00:00")
    p.add_argument("--test_end", default="2025-12-31 23:00")
    p.add_argument("--outdir", default="ablation_out")
    p.add_argument("--csv", default="results_ablation.csv")
    args = p.parse_args()

    if args.strategy == "direct" and args.algo in ("mlp", "lstm"):
        p.error("--strategy direct υποστηρίζεται μόνο για lgbm/xgb")

    specs = [s.strip() for s in args.specs.split(";") if s.strip()]
    if not specs:
        p.error("κενή λίστα --specs")
    parsed = {s: parse_feature_spec(s) for s in specs}

    df = load_processed("hourly", task=args.task)
    if any("dense" in g for g in parsed.values()):
        df = add_dense_lags(df)
    df = df.dropna(subset=["y"]).sort_index()
    all_cols = [c for c in df.columns if c != "y"]

    hz = MARKET_PRESETS[args.market]["horizon"]
    st = MARKET_PRESETS[args.market]["stride"]
    gate = GateSpec(task=args.task, gate=args.gate, market=args.market)

    ts0, ts1 = pd.to_datetime(args.test_start), pd.to_datetime(args.test_end)
    tr0 = pd.to_datetime(args.train_start) if args.train_start else None
    tr1 = pd.to_datetime(args.train_end) if args.train_end else None

    unit = "€/MWh" if args.task == "price" else "MW"
    outdir = BASE_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"🧪 ABLATION | {args.algo} {args.task} {args.market} {args.strategy} "
          f"retrain={args.retrain} gate={args.gate} (gap={gate.gap_hours()}h)")
    print(f"   test: {args.test_start} → {args.test_end} | {len(specs)} specs")

    rows = []
    t_all = time.time()
    for i, spec in enumerate(specs, 1):
        groups = parsed[spec]
        feats = select_features(all_cols, groups)
        feats = [c for c in feats if np.issubdtype(df[c].dtype, np.number)]
        tag = f"{args.algo}_{args.task}_{args.market}_{args.strategy}_{_slug(spec)}"
        print(f"\n[{i}/{len(specs)}] --features \"{spec}\" → {'+'.join(groups)} | {len(feats)} features")
        t0 = time.time()
        try:
            idx, yt, yp = run_forecast(
                algo=args.algo, task=args.task, strategy=args.strategy, gate=gate,
                horizon=hz, stride=st, retrain=args.retrain, df=df,
                feature_cols=feats, train_start=tr0, train_end=tr1,
                test_start=ts0, test_end=ts1, verbose=False,
            )
            met = _metrics(yt, yp)
            secs = round(time.time() - t0, 1)
            print(f"    MAE={met['MAE']} {unit} | RMSE={met['RMSE']} | sMAPE={met['sMAPE']}% | {secs}s")
            rows.append(dict(spec=spec, groups="+".join(groups), n_features=len(feats),
                             MAE=met["MAE"], RMSE=met["RMSE"], sMAPE=met["sMAPE"],
                             n=met["n"], seconds=secs))
            out = {
                "strategy": args.strategy, "task": args.task, "market": args.market,
                "gate": args.gate, "horizon": hz, "stride": st,
                "retrain": args.retrain, "features": groups, "unit": unit,
                "dates": [t.isoformat() for t in idx],
                "actual": [None if not np.isfinite(v) else round(float(v), 4) for v in yt],
                "series": {tag: [None if not np.isfinite(v) else round(float(v), 4) for v in yp]},
                "metrics": [dict(Model=tag, Type="ml",
                                 MAE=met["MAE"], RMSE=met["RMSE"], sMAPE=met["sMAPE"])],
            }
            (outdir / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"    ❌ FAIL: {e}")
            rows.append(dict(spec=spec, groups="+".join(groups), n_features=len(feats),
                             MAE=None, RMSE=None, sMAPE=None, n=0,
                             seconds=round(time.time() - t0, 1)))

    # ---- πίνακας σύγκρισης ----
    base = next((r for r in rows if r["MAE"] is not None), None)
    print("\n" + "=" * 80)
    print(f"{'spec':<22} {'#feat':>6} {'MAE':>10} {'ΔMAE':>9} {'Δ%':>7} {'RMSE':>10} {'sMAPE%':>8}")
    print("-" * 80)
    for r in rows:
        if r["MAE"] is None:
            print(f"{r['spec']:<22} {r['n_features']:>6} {'FAIL':>10}")
            continue
        d = r["MAE"] - base["MAE"]
        dp = 100.0 * d / base["MAE"] if base["MAE"] else 0.0
        print(f"{r['spec']:<22} {r['n_features']:>6} {r['MAE']:>10.3f} {d:>+9.3f} {dp:>+6.1f}% "
              f"{r['RMSE']:>10.3f} {r['sMAPE']:>8.2f}")
    print("=" * 80)
    print("ΔMAE>0 → χειρότερο από το baseline (1ο spec). Αν π.χ. το 'all,-meteo' έχει "
          "ΔMAE≈0, η ομάδα meteo δεν προσφέρει σε αυτό το setup.")

    csv_path = BASE_DIR / args.csv
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"\n✅ Ablation: {len(specs)} specs σε {round((time.time()-t_all)/60,1)} min")
    print(f"   CSV: {csv_path}")
    print(f"   JSONs: {outdir}")


if __name__ == "__main__":
    main()
