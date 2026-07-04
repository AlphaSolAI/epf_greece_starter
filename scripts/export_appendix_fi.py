"""
export_appendix_fi.py
=====================
Consolidate the per-(model, strategy, task) feature-importance data that underlies
the Appendix figures (thesis/content/appendix_fi.tex → ch4_feature_importance/
fig_fi_A_*.png) into a single tidy CSV.

Source of truth: thesis_output/ch4_feature_importance/fi_cache/*.csv
  - lgbm / xgb : normalised gain
  - rf         : impurity-based (MDI)
  - mlp / svr  : permutation importance
  - strategies : Teacher-Forcing (plain), Recursive (openloop), MIMO (mimo_h24),
                 Direct (direct h24)

Output: thesis_output/ch4_feature_importance/appendix_fi_data.csv
  columns: task, strategy, model, importance_type, rank, feature, importance, share_pct

OneDrive-safe: written to a local temp file first, then copied (mtime printed).
"""
from __future__ import annotations
import shutil, tempfile, time
from pathlib import Path
import pandas as pd

ROOT     = Path(__file__).resolve().parent
CACHE    = ROOT / "thesis_output" / "ch4_feature_importance" / "fi_cache"
OUT      = ROOT / "thesis_output" / "ch4_feature_importance" / "appendix_fi_data.csv"

IMP_TYPE = {"lgbm": "gain", "xgb": "gain", "rf": "MDI (impurity)",
            "mlp": "permutation", "svr": "permutation"}


def parse_meta(name: str):
    model = name.split("_", 1)[0]                       # lgbm/xgb/rf/mlp/svr
    task  = "price" if "_price" in name or "hourly_price" in name else "load"
    low = name.lower()
    if "direct" in low:
        strat = "Direct"
    elif "mimo" in low:
        strat = "MIMO"
    elif "openloop" in low:
        strat = "Recursive"
    else:
        strat = "Teacher-Forcing"
    return task, strat, model


def main():
    files = sorted(CACHE.glob("*.csv"))
    if not files:
        raise SystemExit(f"No cache files in {CACHE}")

    rows = []
    combos = []
    for f in files:
        task, strat, model = parse_meta(f.name)
        df = pd.read_csv(f)                              # columns: feature, importance
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        total = df["importance"].sum()
        for rank, r in enumerate(df.itertuples(index=False), start=1):
            rows.append({
                "task": task,
                "strategy": strat,
                "model": model,
                "importance_type": IMP_TYPE.get(model, "?"),
                "rank": rank,
                "feature": r.feature,
                "importance": round(float(r.importance), 6),
                "share_pct": round(100.0 * float(r.importance) / total, 3) if total else 0.0,
                "source_file": f.name,
            })
        combos.append((task, strat, model, len(df)))

    out = pd.DataFrame(rows).sort_values(
        ["task", "strategy", "model", "rank"]).reset_index(drop=True)

    tmp = Path(tempfile.gettempdir()) / OUT.name
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    shutil.copy2(tmp, OUT)
    st = OUT.stat()

    print(f"  → {OUT.relative_to(ROOT)}  ({st.st_size}B, {len(out)} rows, "
          f"mtime {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))})")
    print(f"  {len(combos)} (task, strategy, model) groups:")
    for task in ("price", "load"):
        for c in sorted(x for x in combos if x[0] == task):
            print(f"    {c[0]:5s} {c[1]:15s} {c[2]:5s}  ({c[3]} features)")


if __name__ == "__main__":
    main()
