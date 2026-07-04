"""
src/organise_thesis.py
======================
Organises all thesis figures and analysis outputs into a clean,
chapter-oriented folder structure under thesis_output/.

Folder structure:
  thesis_output/
  ├── ch4_feature_importance/
  │   ├── fi_viz_price.png               — bar charts per model (price)
  │   ├── fi_viz_load.png                — bar charts per model (load)
  │   ├── fi_viz_categories.png          — category breakdown
  │   ├── fi_viz_strategy_shift.png      — strategy shift (CL→OL→MIMO)
  │   ├── fi_viz_topN_price.png          — grouped top-N (price)
  │   ├── fi_viz_topN_load.png           — grouped top-N (load)
  │   ├── fi_viz_mimo_horizons.png       — per-horizon MIMO
  │   ├── feature_importance_price.csv   — raw importances (price)
  │   ├── feature_importance_load.csv    — raw importances (load)
  │   └── fi_summary.csv                 — top-5 per model
  │
  ├── ch5_dm_test/
  │   ├── dm_price_heatmap.png           — DM matrix heatmap (price)
  │   ├── dm_load_heatmap.png            — DM matrix heatmap (load)
  │   ├── dm_price_strategy.png          — champion vs rest (price)
  │   ├── dm_load_strategy.png           — champion vs rest (load)
  │   ├── dm_price_matrix.csv            — DM statistics table
  │   ├── dm_price_pval.csv              — p-values table
  │   ├── dm_price_results.csv           — win/loss summary
  │   ├── dm_load_matrix.csv
  │   ├── dm_load_pval.csv
  │   └── dm_load_results.csv
  │
  └── ch6_forecast_analysis/
      ├── 01_ts_price.png                — time series overlay (price)
      ├── 02_ts_load.png                 — time series overlay (load)
      ├── 03_scatter_price.png           — scatter predicted vs actual
      ├── 04_scatter_load.png
      ├── 05_boxplot_price.png           — error distribution box plots
      ├── 06_boxplot_load.png
      ├── 07_hourly_price.png            — hourly error profile
      ├── 08_hourly_load.png
      ├── 09_dow_price.png               — day-of-week profile
      ├── 10_dow_load.png
      ├── 11_monthly_price.png           — monthly breakdown
      ├── 12_monthly_load.png
      ├── 13_hist_price.png              — error histograms
      ├── 14_hist_load.png
      ├── 15_cumae_price.png             — cumulative error
      ├── 16_cumae_load.png
      ├── 17_heatmap_actual_price.png    — actual heatmap
      ├── 18_heatmap_actual_load.png
      ├── 19_bias_price.png              — bias profile
      ├── 20_bias_load.png
      ├── 21_weekly_price.png            — weekly MAE
      └── 22_weekly_load.png

Usage:
  conda run -n epf --no-capture-output python -m src.organise_thesis
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─── Source folders ───────────────────────────────────────────────────────────
FI_DIR     = ROOT / "feature_importance"
DM_DIR     = ROOT / "dm_test"
THESIS_DIR = ROOT / "thesis_figures"

# ─── Destination ──────────────────────────────────────────────────────────────
OUT     = ROOT / "thesis_output"
CH4     = OUT / "ch4_feature_importance"
CH5     = OUT / "ch5_dm_test"
CH6     = OUT / "ch6_forecast_analysis"

for d in (OUT, CH4, CH5, CH6):
    d.mkdir(exist_ok=True)

# ─── Copy rules ───────────────────────────────────────────────────────────────
COPY_RULES: list[tuple[Path, Path, list[str]]] = [
    # Chapter 4 — Feature Importance
    (FI_DIR, CH4, [
        "fi_viz_price.png",
        "fi_viz_load.png",
        "fi_viz_categories.png",
        "fi_viz_strategy_shift.png",
        "fi_viz_topN_price.png",
        "fi_viz_topN_load.png",
        "fi_viz_mimo_horizons.png",
        "feature_importance_price.csv",
        "feature_importance_load.csv",
        "fi_summary.csv",
    ]),
    # Chapter 5 — DM Test (v2 figures only — the ones the thesis text references)
    (DM_DIR, CH5, [
        "dm_price_heatmap_v2.png",
        "dm_price_wins_v2.png",
        "dm_load_heatmap_v2.png",
        "dm_load_wins_v2.png",
        "dm_price_matrix.csv",
        "dm_price_pval.csv",
        "dm_price_results.csv",
        "dm_load_matrix.csv",
        "dm_load_pval.csv",
        "dm_load_results.csv",
    ]),
    # Chapter 6 — Forecast Analysis (including new 23/24 error heatmaps)
    (THESIS_DIR, CH6, [
        "01_ts_price.png",
        "02_ts_load.png",
        "03_scatter_price.png",
        "04_scatter_load.png",
        "05_boxplot_price.png",
        "06_boxplot_load.png",
        "07_hourly_price.png",
        "08_hourly_load.png",
        "09_dow_price.png",
        "10_dow_load.png",
        "11_monthly_price.png",
        "12_monthly_load.png",
        "13_hist_price.png",
        "14_hist_load.png",
        "15_cumae_price.png",
        "16_cumae_load.png",
        "17_heatmap_actual_price.png",
        "18_heatmap_actual_load.png",
        "19_bias_price.png",
        "20_bias_load.png",
        "21_weekly_price.png",
        "22_weekly_load.png",
        "23_heatmap_error_price.png",
        "24_heatmap_error_load.png",
    ]),
]

# ─── Execute ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Organising thesis outputs")
    print("=" * 60)
    total_ok  = 0
    total_miss= 0

    for src_dir, dst_dir, filenames in COPY_RULES:
        chapter = dst_dir.name
        print(f"\n  {chapter}:")
        for fname in filenames:
            src = src_dir / fname
            dst = dst_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                print(f"    ✓  {fname}")
                total_ok += 1
            else:
                print(f"    ✗  MISSING: {fname} (from {src_dir.name}/)")
                total_miss += 1

    print(f"\n{'─'*60}")
    print(f"  Copied:  {total_ok} files")
    print(f"  Missing: {total_miss} files")
    print(f"  Output:  {OUT}")

    # Print tree
    print(f"\n  📁 thesis_output/")
    for ch in sorted(OUT.iterdir()):
        if ch.is_dir():
            pngs = list(ch.glob("*.png"))
            csvs = list(ch.glob("*.csv"))
            print(f"  ├── {ch.name}/  ({len(pngs)} PNG, {len(csvs)} CSV)")
            for f in sorted(ch.iterdir()):
                print(f"  │   ├── {f.name}")
    print()


if __name__ == "__main__":
    main()
