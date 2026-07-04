"""
run_master_grid.py — Batch driver για το master_forecast engine.

Σαρώνει έναν grid (algos × tasks × markets × strategies × retrain) in-process
(χωρίς spawn conda ανά config → γρήγορο) και γράφει:
  - results_master_grid.csv  (μία γραμμή ανά config: MAE/RMSE/sMAPE + metadata)
  - <outdir>/<config>.json   (πλήρη time series ανά config, dashboard-συμβατά)

Παραδείγματα:
  # Πλήρες STATIC grid για Q1 (όλοι οι algos, price+load, dam):
  python -m src.run_master_grid --algos lgbm,xgb,mlp,lstm --tasks price,load \
     --markets dam --strategies recursive,direct --retrain static \
     --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00"

  # Monthly retrain, μόνο tree models, dam+forward:
  # ΠΡΟΣΟΧΗ: με retrain=monthly/weekly ΜΗΝ δίνεις --train_end (expanding window έως cutoff).
  python -m src.run_master_grid --algos lgbm,xgb --tasks price,load \
     --markets dam,forward --strategies recursive --retrain monthly \
     --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .master_forecast import (
    MARKET_PRESETS, add_dense_lags, make_blocks, run_forecast, _metrics,
)
from .feature_availability import GateSpec, parse_feature_spec, select_features
from .split_utils import load_processed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]


def _csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description="Batch grid runner για master_forecast")
    p.add_argument("--algos", default="lgbm,xgb", type=_csv_list)
    p.add_argument("--tasks", default="price,load", type=_csv_list)
    p.add_argument("--markets", default="dam", type=_csv_list)
    p.add_argument("--strategies", default="recursive", type=_csv_list)
    p.add_argument("--retrain", default="static", choices=["static", "monthly", "weekly"])
    p.add_argument("--gate", default="strict", choices=["strict", "academic"])
    p.add_argument("--features", default="default")
    p.add_argument("--train_start", default=None)
    p.add_argument("--train_end", default=None)
    p.add_argument("--test_start", required=True)
    p.add_argument("--test_end", required=True)
    p.add_argument("--outdir", default="master_grid_out")
    p.add_argument("--csv", default="results_master_grid.csv")
    args = p.parse_args()

    outdir = BASE_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = BASE_DIR / args.csv

    ts0 = pd.to_datetime(args.test_start)
    ts1 = pd.to_datetime(args.test_end)
    tr0 = pd.to_datetime(args.train_start) if args.train_start else None
    tr1 = pd.to_datetime(args.train_end) if args.train_end else None
    groups = parse_feature_spec(args.features)

    # cache processed df ανά task (+dense) για να μη φορτώνεται ξανά
    df_cache: dict = {}

    def get_df(task: str):
        if task not in df_cache:
            d = load_processed("hourly", task=task)
            if "dense" in groups:
                d = add_dense_lags(d)
            d = d.dropna(subset=["y"]).sort_index()
            df_cache[task] = d
        return df_cache[task]

    rows = []
    total = len(args.algos) * len(args.tasks) * len(args.markets) * len(args.strategies)
    done = 0
    t_all = time.time()

    for task in args.tasks:
        df = get_df(task)
        all_cols = [c for c in df.columns if c != "y"]
        feats = select_features(all_cols, groups)
        feats = [c for c in feats if np.issubdtype(df[c].dtype, np.number)]
        for market in args.markets:
            hz = MARKET_PRESETS.get(market, {}).get("horizon", 24)
            st = MARKET_PRESETS.get(market, {}).get("stride", 24)
            gate = GateSpec(task=task, gate=args.gate, market=market, delay_override=None)
            for algo in args.algos:
                for strategy in args.strategies:
                    # seq2seq μόνο για lstm· direct όχι για mlp/lstm
                    if strategy == "direct" and algo in ("mlp", "lstm"):
                        continue
                    if strategy == "seq2seq" and algo != "lstm":
                        continue
                    done += 1
                    tag = f"{algo}_{task}_{market}_{strategy}_{args.retrain}"
                    t0 = time.time()
                    print(f"\n[{done}/{total}] {tag} ...", flush=True)
                    try:
                        idx, yt, yp = run_forecast(
                            algo=algo, task=task, strategy=strategy, gate=gate,
                            horizon=hz, stride=st, retrain=args.retrain, df=df,
                            feature_cols=feats, train_start=tr0, train_end=tr1,
                            test_start=ts0, test_end=ts1, verbose=False,
                        )
                        met = _metrics(yt, yp)
                        unit = "€/MWh" if task == "price" else "MW"
                        secs = round(time.time() - t0, 1)
                        print(f"    MAE={met['MAE']} {unit} | RMSE={met['RMSE']} | sMAPE={met['sMAPE']}% | {secs}s", flush=True)
                        rows.append(dict(
                            config=tag, algo=algo, task=task, market=market,
                            strategy=strategy, retrain=args.retrain, gate=args.gate,
                            horizon=hz, stride=st, unit=unit,
                            MAE=met["MAE"], RMSE=met["RMSE"], sMAPE=met["sMAPE"],
                            n=met["n"], seconds=secs,
                        ))
                        out = {
                            "strategy": strategy, "task": task, "market": market,
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
                        print(f"    ❌ FAIL: {e}", flush=True)
                        rows.append(dict(config=tag, algo=algo, task=task, market=market,
                                         strategy=strategy, retrain=args.retrain, gate=args.gate,
                                         horizon=hz, stride=st, unit="", MAE=None, RMSE=None,
                                         sMAPE=None, n=0, seconds=round(time.time() - t0, 1)))

    # write CSV
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"\n✅ Grid done: {done} configs σε {round((time.time()-t_all)/60,1)} min")
    print(f"   CSV: {csv_path}")
    print(f"   JSONs: {outdir}")

    # σύνοψη κατάταξης
    ok = [r for r in rows if r["MAE"] is not None]
    for task in args.tasks:
        sub = sorted([r for r in ok if r["task"] == task], key=lambda r: r["MAE"])
        if sub:
            print(f"\n— {task.upper()} ranking (MAE) —")
            for r in sub[:12]:
                print(f"   {r['MAE']:>10.3f} {r['unit']:<6} | {r['config']}")


if __name__ == "__main__":
    main()
