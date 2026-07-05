"""Run ledger: σαρώνει runs/**/*.json (master_forecast schema) και χτίζει
results/run_ledger.csv — ένα auditability index για τη διπλωματική.

Stdlib μόνο (json/csv) — τρέχει με system Python, ΧΩΡΙΣ conda:
    python scripts/build_run_ledger.py [--root runs] [--out results/run_ledger.csv]

Μία γραμμή ανά forecast JSON: path, mtime, model/algo, task, market, strategy,
gate, retrain, features, crosslag_mode, horizon/stride, test window (από dates),
n_hours, MAE/RMSE/sMAPE. JSONs που δεν ταιριάζουν στο schema αγνοούνται σιωπηλά
(καταμετρώνται στο τέλος).
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

FIELDS = [
    "path", "file_mtime", "model", "task", "market", "strategy", "gate",
    "retrain", "features", "crosslag_mode", "crosslag_gap", "horizon", "stride",
    "test_start", "test_end", "n_hours", "mae", "rmse", "smape",
]


def parse_run(path: str, root: str):
    try:
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None
    metrics = j.get("metrics")
    if isinstance(metrics, list):  # master_forecast: λίστα με 1 entry ανά μοντέλο
        metrics = metrics[0] if metrics and isinstance(metrics[0], dict) else None
    dates = j.get("dates")
    if not isinstance(metrics, dict) or "MAE" not in metrics or not dates:
        return None  # όχι master_forecast schema
    feats = j.get("features")
    if isinstance(feats, list):
        feats = "+".join(feats)
    return {
        "path": os.path.relpath(path, root).replace("\\", "/"),
        "file_mtime": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"),
        "model": metrics.get("Model", ""),
        "task": j.get("task", ""),
        "market": j.get("market", ""),
        "strategy": j.get("strategy", ""),
        "gate": j.get("gate", ""),
        "retrain": j.get("retrain", ""),
        "features": feats or "",
        "crosslag_mode": j.get("crosslag_mode", ""),
        "crosslag_gap": j.get("crosslag_gap", ""),
        "horizon": j.get("horizon", ""),
        "stride": j.get("stride", ""),
        "test_start": dates[0],
        "test_end": dates[-1],
        "n_hours": len(dates),
        "mae": metrics.get("MAE", ""),
        "rmse": metrics.get("RMSE", ""),
        "smape": metrics.get("sMAPE", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs", help="φάκελος για σάρωση (default: runs)")
    ap.add_argument("--out", default="results/run_ledger.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"FAIL: δεν βρέθηκε φάκελος {args.root!r} — τρέξε από το project root", file=sys.stderr)
        return 1

    rows, skipped = [], 0
    for dirpath, _dirnames, filenames in os.walk(args.root):
        for name in sorted(filenames):
            if not name.endswith(".json"):
                continue
            row = parse_run(os.path.join(dirpath, name), os.getcwd())
            if row is None:
                skipped += 1
            else:
                rows.append(row)

    rows.sort(key=lambda r: (r["path"],))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"OK: {len(rows)} runs -> {args.out}  (skipped non-run JSONs: {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
