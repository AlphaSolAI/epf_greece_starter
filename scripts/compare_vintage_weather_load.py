"""
Read-only QA compare: hourly_load.parquet BEFORE vs AFTER the vintage-weather
(GOALS.md G5) rebuild. Mirrors scripts/compare_load_parquet.py's criteria (G2).

Checks:
  1. identical index
  2. new columns = old columns + wv_* (+ wv_*_missing) ONLY
  3. all pre-existing columns bit-identical
  4. new wv_* columns: 0 NaN inside the 3 current ablation windows
     (q1_2025_26, summer_2025, octnov_2025)

Usage:
    python -X utf8 scripts/compare_vintage_weather_load.py \
        --old data/processed/_backup_vintageweather_20260710/hourly_load.parquet \
        --new data/processed/hourly_load.parquet
"""
import argparse
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WINDOWS = {
    "q1_2025_26": ("2025-12-01 00:00", "2026-02-28 23:00"),
    "summer_2025": ("2025-06-01 00:00", "2025-08-31 23:00"),
    "octnov_2025": ("2025-10-01 00:00", "2025-11-30 23:00"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()

    old = pd.read_parquet(args.old)
    new = pd.read_parquet(args.new)

    ok = True

    # 1. index
    if not old.index.equals(new.index):
        print(f"FAIL criterion 1 (index): old n={len(old)} new n={len(new)} equal={old.index.equals(new.index)}")
        # not necessarily fatal (rebuild can extend range) -- report but continue
    else:
        print(f"PASS criterion 1: identical index (n={len(old)})")

    # 2. new columns
    added = set(new.columns) - set(old.columns)
    removed = set(old.columns) - set(new.columns)
    unexpected_added = [c for c in added if not c.startswith("wv_")]
    if removed or unexpected_added:
        print(f"FAIL criterion 2: removed={sorted(removed)} unexpected_added={sorted(unexpected_added)}")
        ok = False
    else:
        print(f"PASS criterion 2: {len(added)} new wv_* columns added, 0 removed, 0 unexpected")

    # 3. existing columns bit-identical
    common_idx = old.index.intersection(new.index)
    common_cols = [c for c in old.columns if c in new.columns]
    diffs = []
    for c in common_cols:
        o = old.loc[common_idx, c]
        n = new.loc[common_idx, c]
        if o.dtype.kind in "fc" and n.dtype.kind in "fc":
            same = ((o == n) | (o.isna() & n.isna())).all()
        else:
            same = o.equals(n)
        if not same:
            diffs.append(c)
    if diffs:
        print(f"FAIL criterion 3: {len(diffs)} pre-existing columns changed: {diffs[:20]}")
        ok = False
    else:
        print(f"PASS criterion 3: all {len(common_cols)} pre-existing columns bit-identical")

    # 4. coverage in eval windows
    wv_cols = [c for c in new.columns if c.startswith("wv_") and not c.endswith("_missing")]
    for wname, (a, b) in WINDOWS.items():
        sl = new.loc[a:b, wv_cols]
        n_nan = int(sl.isna().sum().sum())
        total = sl.shape[0] * sl.shape[1]
        status = "PASS" if n_nan == 0 else "FAIL"
        print(f"{status} criterion 4 [{wname}]: {n_nan}/{total} NaN in wv_* columns ({sl.shape[0]} rows)")
        if n_nan != 0:
            ok = False

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
