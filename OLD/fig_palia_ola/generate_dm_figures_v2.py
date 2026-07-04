#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dm_figures_v2.py
=========================
Regenerates the four Diebold-Mariano (HLN-corrected) figures for Chapter 9
from the CSV outputs, with the NEW model naming (CL->TF, OL->Rec, MR->WF) and
strategy-consistent colours.

Outputs (written next to this script, i.e. thesis_output/ch5_dm_test/):
    dm_price_heatmap_v2.png   dm_price_wins_v2.png
    dm_load_heatmap_v2.png    dm_load_wins_v2.png

Expected input CSVs in the same directory:
    dm_price_pval.csv     square p-value matrix, model names on index + columns
    dm_price_matrix.csv   square signed DM-statistic matrix (same labels)
    dm_price_results.csv  per-model summary; needs columns for wins and MAE
    dm_load_pval.csv / dm_load_matrix.csv / dm_load_results.csv  (load equivalents)

IMPORTANT — column-name assumptions (edit COLMAP below if yours differ):
  * the matrices have model names in the first column (index) AND as headers.
  * results.csv has one row per model with a model-name column, a wins column,
    and an MAE column. The script auto-detects common spellings; if detection
    fails it raises a clear error telling you which names it looked for.

Sign convention for the DM statistic (see DM_POSITIVE_MEANS_ROW_WORSE):
  Standard HLN setup compares loss_row - loss_col. A POSITIVE statistic then
  means the row model has the HIGHER loss (is WORSE). We colour a cell GREEN
  when the ROW model is significantly better, RED when the COLUMN model is.
  If your matrix uses the opposite sign, set the flag to False.

Run:
    python generate_dm_figures_v2.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HERE = str(Path(__file__).resolve().parent.parent / "dm_test")
ALPHA = 0.05
DM_POSITIVE_MEANS_ROW_WORSE = True   # flip if your sign convention differs

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# Strategy colours (consistent across the thesis)
STRAT_COLORS = {
    "WF-TF":     "#1a7c3e",   # dark green
    "WF-Rec":    "#52b788",   # light green
    "WF-MIMO":   "#2a9d8f",   # teal (WF variant)
    "WF-Direct": "#ad1457",   # dark magenta (WF Direct — separate strategy)
    "TF":        "#1d3557",   # dark blue
    "Rec":       "#457b9d",   # steel blue
    "MIMO":      "#9d4edd",   # purple
    "Direct":    "#c2185b",   # magenta (Direct — separate strategy, never grouped with MIMO)
    "Baseline":  "#6c757d",   # gray (true Naive/Seasonal baselines only)
}

# Exact-match renames first, then prefix rules.
EXACT_RENAME = {
    "CL-LGBM": "TF-LGBM", "CL-XGB": "TF-XGB", "CL-RF": "TF-RF",
    "CL-MLP-Opt": "TF-MLP", "CL-Ens-Best3": "TF-Ens",
    "OL-LGBM-Daily": "Rec-LGBM", "OL-XGB-SS-Dense": "Rec-XGB-SS",
    "OL-RF-SS": "Rec-RF", "OL-Ens-Top3": "Rec-Ens",
    "MR-Rec-XGB-SS": "WF-Rec-XGB", "MR-Ens-TF-Best3": "WF-TF-Ens",
}
PREFIX_RULES = [("MR-TF-", "WF-TF-"), ("MR-MIMO-", "WF-MIMO-"),
                ("MR-Rec-", "WF-Rec-"), ("MR-", "WF-"),
                ("CL-", "TF-"), ("OL-", "Rec-")]


def rename_model(name: str) -> str:
    name = str(name).strip()
    if name in EXACT_RENAME:
        return EXACT_RENAME[name]
    for old, new in PREFIX_RULES:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def strategy_of(name: str) -> str:
    """Map a (renamed) model to its strategy bucket. Order matters."""
    n = name.upper()
    if n.startswith("WF-TF"):     return "WF-TF"
    if n.startswith("WF-REC"):    return "WF-Rec"
    if n.startswith("WF-MIMO"):   return "WF-MIMO"
    if n.startswith("WF-DIRECT"): return "WF-Direct"    # Direct is separate (not WF-MIMO)
    if n.startswith("WF-"):       return "WF-TF"        # generic WF fallback
    if n.startswith("MIMO"):      return "MIMO"
    if n.startswith("REC"):       return "Rec"
    if n.startswith("TF"):        return "TF"
    if n.startswith("DIRECT"):    return "Direct"       # static Direct (was mislabeled Baseline)
    if "NAIVE" in n or "BASE" in n or "PERSIST" in n: return "Baseline"
    return "Baseline"


def color_of(name: str) -> str:
    return STRAT_COLORS[strategy_of(name)]


# ----------------------------------------------------------------------
# CSV loading helpers
# ----------------------------------------------------------------------
def _read_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df.index = [rename_model(i) for i in df.index]
    df.columns = [rename_model(c) for c in df.columns]
    return df


def _read_results(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(cands, required=True, what=""):
        for cand in cands:
            if cand in cols:
                return cols[cand]
        if required:
            raise KeyError(
                f"{os.path.basename(path)}: could not find a {what} column. "
                f"Looked for {cands}; available columns: {list(df.columns)}")
        return None

    # Directly check exact column names first
    cols = list(df.columns)
    mcol = next((c for c in cols if c.lower().startswith('model')), None) or pick(["Model"], what="model-name")
    wcol = next((c for c in cols if 'win' in c.lower() and 'loss' not in c.lower()), None) or "Wins (p<5%)"
    macol = next((c for c in cols if 'mae' in c.lower()), None) or pick(["mae"], what="mae")
    _ = pick  # keep pick defined for potential future use
    macol = macol or pick(["MAE (€/MWh)", "MAE (MW)", "mae", "mae_mean", "test_mae", "mae_q1", "mae_value"],
                 required=False, what="MAE")
    out = pd.DataFrame({
        "model": [rename_model(m) for m in df[mcol]],
        "wins": pd.to_numeric(df[wcol], errors="coerce"),
    })
    out["mae"] = pd.to_numeric(df[macol], errors="coerce") if macol else np.nan
    out["strategy"] = out["model"].map(strategy_of)
    return out


# ----------------------------------------------------------------------
# Heatmap
# ----------------------------------------------------------------------
def make_heatmap(pval_df, dm_df, quantity, unit, results=None, outname=None):
    models = list(pval_df.index)
    n = len(models)
    pv = pval_df.values.astype(float)
    dm = dm_df.reindex(index=models, columns=models).values.astype(float)

    # category grid: 0 = not significant / diagonal, 1 = row better, 2 = col better
    grid = np.zeros((n, n), dtype=int)
    sig = pv < ALPHA
    row_worse = dm > 0 if DM_POSITIVE_MEANS_ROW_WORSE else dm < 0
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(pv[i, j]):
                grid[i, j] = 0
            elif sig[i, j]:
                grid[i, j] = 2 if row_worse[i, j] else 1
            else:
                grid[i, j] = 0

    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = ListedColormap(["#ffffff", "#bfe3c8", "#f6c0c0"])   # white, green, red
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal")

    # gridlines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    # diagonal: win counts (from results if available, else derived)
    if results is not None:
        wins_map = dict(zip(results["model"], results["wins"]))
    else:
        wins_map = {models[i]: int((grid[i, :] == 1).sum()) for i in range(n)}
    for i in range(n):
        ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fc="#eef2f7",
                     ec="#b8c4d2", lw=0.8, zorder=2))
        w = wins_map.get(models[i], "")
        if w != "" and np.isfinite(w):
            ax.text(i, i, f"{int(w)}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#1d3557", zorder=3)

    # tick labels coloured by strategy
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=90, fontsize=9)
    ax.set_yticklabels(models, fontsize=9)
    for tick, m in zip(ax.get_xticklabels(), models):
        tick.set_color(color_of(m)); tick.set_fontweight("bold")
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(color_of(m)); tick.set_fontweight("bold")

    ax.set_title(f"Diebold–Mariano (HLN) test — {quantity}\n"
                 f"green: row model significantly better (p<{ALPHA}); "
                 f"red: column model better; white: not significant; "
                 f"diagonal: win count",
                 fontsize=13, fontweight="bold", color="#1d3557", pad=14)

    # strategy legend
    seen = []
    for m in models:
        s = strategy_of(m)
        if s not in seen:
            seen.append(s)
    handles = [Patch(fc=STRAT_COLORS[s], label=s) for s in seen]
    ax.legend(handles=handles, title="Strategy (label colour)",
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10)

    fig.tight_layout()
    path = os.path.join(HERE, outname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


# ----------------------------------------------------------------------
# Win bar charts (two panels)
# ----------------------------------------------------------------------
def make_wins(results, quantity, unit, outname):
    df = results.dropna(subset=["wins"]).copy()
    df = df.sort_values("wins", ascending=True)   # ascending -> best on top with barh

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 9),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    # LEFT: individual model wins
    colors = [color_of(m) for m in df["model"]]
    bars = axL.barh(df["model"], df["wins"], color=colors, edgecolor="white")
    axL.set_xlabel("number of significant wins", fontsize=12)
    axL.set_title(f"Per-model wins — {quantity}", fontsize=13,
                  fontweight="bold", color="#1d3557")
    xmax = float(df["wins"].max()) if len(df) else 1.0
    for bar, (_, r) in zip(bars, df.iterrows()):
        label = f"{int(r['wins'])}"
        if np.isfinite(r["mae"]):
            label += f"   (MAE {r['mae']:.2f} {unit})"
        axL.text(bar.get_width() + xmax*0.01, bar.get_y() + bar.get_height()/2,
                 label, va="center", fontsize=9, color="#333333")
    axL.set_xlim(0, xmax * 1.32)
    axL.tick_params(axis="y", labelsize=9)
    for tick, m in zip(axL.get_yticklabels(), df["model"]):
        tick.set_color(color_of(m)); tick.set_fontweight("bold")

    # RIGHT: strategy group mean wins + 95% CI
    grp = df.groupby("strategy")["wins"]
    order = [s for s in STRAT_COLORS if s in grp.groups]
    means = [grp.get_group(s).mean() for s in order]
    sems = []
    for s in order:
        v = grp.get_group(s).values.astype(float)
        sems.append(1.96 * (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    gcolors = [STRAT_COLORS[s] for s in order]
    ypos = np.arange(len(order))
    axR.barh(ypos, means, xerr=sems, color=gcolors, edgecolor="white",
             error_kw=dict(ecolor="#333333", capsize=4, lw=1.2))
    axR.set_yticks(ypos); axR.set_yticklabels(order, fontsize=10, fontweight="bold")
    for tick, s in zip(axR.get_yticklabels(), order):
        tick.set_color(STRAT_COLORS[s])
    axR.set_xlabel("mean wins per model (±95% CI)", fontsize=12)
    axR.set_title(f"Strategy group means — {quantity}", fontsize=13,
                  fontweight="bold", color="#1d3557")
    # Place each value label to the RIGHT of the 95% CI whisker so the black
    # error bar never crosses the number (Fig 7.14/7.16 fix).
    xr_max = max((m + e) for m, e in zip(means, sems)) if means else 1.0
    for yi, mn, se in zip(ypos, means, sems):
        axR.text(mn + se + xr_max * 0.03, yi, f"{mn:.1f}",
                 va="center", ha="left", fontsize=10, fontweight="bold",
                 color="#333333")
    axR.set_xlim(0, xr_max * 1.22)

    for ax in (axL, axR):
        ax.grid(axis="x", alpha=0.25)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(f"Diebold–Mariano win analysis — {quantity}",
                 fontsize=15, fontweight="bold", color="#1d3557", y=1.0)
    fig.tight_layout()
    path = os.path.join(HERE, outname)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def run_one(prefix, quantity, unit):
    pval = _read_matrix(os.path.join(HERE, f"dm_{prefix}_pval.csv"))
    try:
        dm = _read_matrix(os.path.join(HERE, f"dm_{prefix}_matrix.csv"))
    except FileNotFoundError:
        # fall back: no signed matrix -> can't tell direction; treat all sig as row-better
        dm = pd.DataFrame(-np.ones_like(pval.values), index=pval.index, columns=pval.columns)
        print(f"  [warn] dm_{prefix}_matrix.csv not found; heatmap direction may be unreliable")
    results = None
    rpath = os.path.join(HERE, f"dm_{prefix}_results.csv")
    if os.path.exists(rpath):
        results = _read_results(rpath)
    make_heatmap(pval, dm, quantity, unit, results, f"dm_{prefix}_heatmap_v2.png")
    if results is not None:
        make_wins(results, quantity, unit, f"dm_{prefix}_wins_v2.png")
    else:
        print(f"  [warn] dm_{prefix}_results.csv not found; skipping wins chart")


if __name__ == "__main__":
    print("Generating DM figures (v2) in", HERE)
    run_one("price", "Price", "€/MWh")
    run_one("load", "Load", "MW")
    print("Done.")
