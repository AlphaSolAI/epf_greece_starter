"""
src/thesis_plots.py
===================
Kanousis-style thesis visualisations for electricity price & load forecasting.
Uses Q1 2026 (Dec 2025 – Feb 2026, 2160h) dashboard JSON data.

Plot types:
  1. Time-series overlay (actual vs best models per strategy)
  2. Scatter plots: predicted vs actual
  3. Box plots: absolute errors by model
  4. Hourly error profiles (mean |error| by hour-of-day)
  5. Day-of-week error profiles
  6. Monthly error breakdown (Dec / Jan / Feb)
  7. Error distribution histograms
  8. Cumulative error over time
  9. Heat-map: actual price/load matrix (hour-of-day × day-of-week)
 10. Over/under-prediction profiles by model

Usage:
  conda run -n epf --no-capture-output python -m src.thesis_plots

Outputs (in thesis_figures/ folder):
  01_ts_price.png            — time series overlay (price)
  02_ts_load.png             — time series overlay (load)
  03_scatter_price.png       — scatter predicted vs actual (price)
  04_scatter_load.png        — scatter predicted vs actual (load)
  05_boxplot_price.png       — box plots |error| per model (price)
  06_boxplot_load.png        — box plots |error| per model (load)
  07_hourly_price.png        — hourly error profile (price)
  08_hourly_load.png         — hourly error profile (load)
  09_dow_price.png           — day-of-week error profile (price)
  10_dow_load.png            — day-of-week error profile (load)
  11_monthly_price.png       — monthly MAE breakdown (price)
  12_monthly_load.png        — monthly MAE breakdown (load)
  13_hist_price.png          — error distribution histogram (price)
  14_hist_load.png           — error distribution histogram (load)
  15_cumae_price.png         — cumulative absolute error (price)
  16_cumae_load.png          — cumulative absolute error (load)
  17_heatmap_actual_price.png— actual price heatmap (hour × date)
  18_heatmap_actual_load.png — actual load heatmap (hour × date)
  19_bias_price.png          — over/under-prediction per model (price)
  20_bias_load.png           — over/under-prediction per model (load)
"""

from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "thesis_figures"
OUTDIR.mkdir(exist_ok=True)

# ─── Dark theme ───────────────────────────────────────────────────────────────
BG   = "#0f172a"
BG2  = "#1e293b"
BG3  = "#0d1929"
FG   = "#e2e8f0"
GRID = "#334155"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG2,
    "axes.edgecolor":   "#475569",
    "axes.labelcolor":  FG,
    "text.color":       FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "grid.color":       GRID,
    "grid.alpha":       0.5,
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "legend.framealpha": 0.85,
    "legend.facecolor": BG2,
    "legend.edgecolor": "#475569",
})

# ─── Model registry ───────────────────────────────────────────────────────────
# (json_key, series_name, display_name, colour, linestyle, linewidth, monthly_only)
# ONE best STATIC + ONE best NON-STATIC (Walk-Forward Monthly Retrain) per strategy,
# selected from eval_results_summary.xlsx (Q1 2026, all series validated at 2160h).
# Legends state regime (Static / MR) AND the exact winning model.
# Colours match the approved Fig 11/12 family scheme (strat_color / FAMILY_SHADES):
#   TF static #0d47a1 · TF MR #1b5e20 · Rec static #e65100 · Rec MR #ffb74d
#   MIMO static #8e24aa · MIMO MR #ce93d8 · Direct static #c2185b · Direct MR #f06292
# Static = solid line; MR = dashed line. monthly_only kept False so all 8 appear
# in every figure (boxplot / hourly / cumae / bias / monthly).
# NOTE: MIMO/Direct MR read the v2 (…_optuna) JSON — the v1 load JSON is a corrupt
# partial run (1416h) and cannot supply full time-series.
PRICE_MODELS = [
    # Teacher-Forcing
    ("price_cl_monthly_q1_2026",            "Ensemble-Best3 (1/MAE)",         "TF Static — Ensemble-Best3", "#0d47a1", "-",  1.5, False),
    ("price_monthly_retrain",               "TF-LGBM-MR",                     "TF MR — LGBM",               "#1b5e20", "--", 1.5, False),
    # Recursive
    ("price_openloop_h24_monthly_q1_2026",  "XGB-Daily-SS-Optuna Dense (24h)","Rec Static — XGB-SS-Dense",  "#e65100", "-",  1.4, False),
    ("price_monthly_retrain",               "Rec-XGB-SS-DO-MR",               "Rec MR — XGB-SS-DO",         "#ffb74d", "--", 1.4, False),
    # MIMO
    ("price_mimo_h24_monthly_q1_2026",      "Ensemble Best3 24h (1/MAE)",     "MIMO Static — Ensemble-Best3","#8e24aa", "-",  1.4, False),
    ("price_mimo_monthly_retrain_optuna",   "Ensemble-Best3 MIMO MR",         "MIMO MR — Ensemble-Best3",   "#ce93d8", "--", 1.4, False),
    # Direct
    ("price_direct_h24_monthly_q1_2026",    "LGBM DIRECT 24h Optuna",         "Direct Static — LGBM-Optuna","#c2185b", "-",  1.4, False),
    ("price_mimo_monthly_retrain_optuna",   "LGBM Direct Dense MR+CachedOpt", "Direct MR — LGBM-Dense",     "#f06292", "--", 1.4, False),
]

LOAD_MODELS = [
    # Teacher-Forcing
    ("load_cl_monthly_q1_2026",             "Ensemble-Best3 (1/MAE)",         "TF Static — Ensemble-Best3", "#0d47a1", "-",  1.5, False),
    ("load_monthly_retrain",                "Ensemble-TF-Best3 (1/MAE)",      "TF MR — Ensemble-Best3",     "#1b5e20", "--", 1.5, False),
    # Recursive
    ("load_openloop_h24_monthly_q1_2026",   "LGBM-Daily-SS-Optuna (24h)",     "Rec Static — LGBM-SS",       "#e65100", "-",  1.4, False),
    ("load_monthly_retrain",                "Rec-XGB-SS-DO-MR",               "Rec MR — XGB-SS-DO",         "#ffb74d", "--", 1.4, False),
    # MIMO
    ("load_mimo_h24_monthly_q1_2026",       "Ensemble Best3 24h (1/MAE)",     "MIMO Static — Ensemble-Best3","#8e24aa", "-",  1.4, False),
    ("load_mimo_monthly_retrain_optuna",    "Ensemble-Best3 MIMO MR",         "MIMO MR — Ensemble-Best3",   "#ce93d8", "--", 1.4, False),
    # Direct
    ("load_direct_h24_monthly_q1_2026",     "Ensemble All 24h (1/MAE)",       "Direct Static — Ensemble-All","#c2185b", "-",  1.4, False),
    ("load_mimo_monthly_retrain_optuna",    "LGBM Direct Dense MR+CachedOpt", "Direct MR — LGBM-Dense",     "#f06292", "--", 1.4, False),
]

# ─── Data helpers ─────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}

def _jload(key: str) -> dict:
    if key not in _cache:
        p = ROOT / f"dashboard_data_hourly_{key}.json"
        _cache[key] = json.loads(p.read_text(encoding="utf-8"))
    return _cache[key]


def load_task_data(task: str, include_monthly_only: bool = False) -> tuple[pd.Series, pd.DataFrame, list[tuple]]:
    """
    Returns (actual, preds, registry)
    actual : pd.Series (2160h)
    preds  : pd.DataFrame, columns = display names
    registry : list of tuples (display_name, colour, linestyle, lw)

    include_monthly_only : if True, also loads entries with monthly_only=True
                           (used exclusively by plot_monthly_mae for the static vs WF pairs)
    """
    registry = PRICE_MODELS if task == "price" else LOAD_MODELS

    actual_ref = None
    series_dict = {}
    meta = []

    for entry in registry:
        json_key, series_name, display, colour, ls, lw = entry[:6]
        monthly_only = entry[6] if len(entry) > 6 else False
        if monthly_only and not include_monthly_only:
            continue
        try:
            d = _jload(json_key)
        except FileNotFoundError as exc:
            warnings.warn(str(exc)); continue

        if series_name not in d["series"]:
            warnings.warn(f"Series '{series_name}' not in {json_key}"); continue

        dates = pd.to_datetime(d["dates"])
        vals  = np.array(d["series"][series_name], dtype=float)
        # Guard against partial runs where the series is shorter than dates
        if len(vals) != len(dates):
            dates = dates[:len(vals)]
        if actual_ref is None:
            actual_ref = pd.Series(np.array(d["actual"], dtype=float)[:len(dates)],
                                   index=dates, name="actual")
        series_dict[display] = pd.Series(vals, index=dates)
        meta.append((display, colour, ls, lw))

    preds = pd.DataFrame(series_dict)
    common = actual_ref.index.intersection(preds.index)
    return actual_ref.loc[common], preds.loc[common], meta


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 & 2: Time-series overlay
# ═══════════════════════════════════════════════════════════════════════════════

def plot_time_series(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "01" if task == "price" else "02"

    # Two panels: full Q1 overview (top) + a readable 14-day zoom (bottom),
    # avoiding the "spaghetti" of many lines over 2160 hours.
    zoom = (pd.Timestamp("2026-01-19"), pd.Timestamp("2026-02-02"))
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(16, 9))

    for ax in (ax_full, ax_zoom):
        ax.plot(actual.index, actual.values, color=FG, lw=1.4,
                label="Actual", alpha=0.9, zorder=10)
        for (display, colour, ls, lw) in meta:
            ax.plot(preds.index, preds[display].values,
                    color=colour, lw=lw, ls=ls, label=display, alpha=0.8)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y")

    # Month separators on the overview panel
    for m in pd.date_range("2026-01-01", "2026-03-01", freq="MS"):
        ax_full.axvline(m, color="#475569", ls=":", lw=0.8, alpha=0.7)

    # Zoom panel: restrict x and rescale y to the window
    ax_zoom.set_xlim(*zoom)
    mask = (actual.index >= zoom[0]) & (actual.index <= zoom[1])
    if bool(mask.any()):
        win = [actual.values[mask]] + [preds[d].values[mask] for (d, *_ ) in meta]
        allv = np.concatenate([v[~np.isnan(v)] for v in win])
        if allv.size:
            pad = 0.05 * (allv.max() - allv.min() + 1e-9)
            ax_zoom.set_ylim(allv.min() - pad, allv.max() + pad)

    ax_full.set_title(
        f"Electricity {task.title()} Forecast — Q1 2026 (Dec 2025 – Feb 2026)",
        fontsize=11, color=FG, pad=10)
    ax_full.legend(loc="upper left", ncol=4, fontsize=7.5, framealpha=0.9)
    ax_zoom.set_title("Zoom: 19 Jan – 2 Feb 2026", fontsize=10, color=FG, pad=6)
    ax_zoom.set_xlabel("Date", fontsize=9)

    plt.tight_layout()
    out = OUTDIR / f"{num}_ts_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 & 4: Scatter predicted vs actual
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scatter(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "03" if task == "price" else "04"
    n_m   = len(meta)

    ncols = 4
    nrows = (n_m + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    lims = (min(actual.min(), preds.min().min()),
            max(actual.max(), preds.max().max()))

    for ax, (display, colour, ls, lw) in zip(axes_flat, meta):
        ax.scatter(actual.values, preds[display].values,
                   alpha=0.15, s=6, color=colour, linewidths=0)
        ax.plot(lims, lims, color=FG, lw=0.8, ls="--", alpha=0.6)
        mae = np.mean(np.abs(actual.values - preds[display].values))
        ax.set_title(f"{display}\nMAE={mae:.2f} {unit}", fontsize=7.5, color=FG, pad=4)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"Actual ({unit})", fontsize=7)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=7)
        ax.set_facecolor(BG2)

    for ax in axes_flat[len(meta):]:
        ax.set_visible(False)

    fig.suptitle(f"Scatter: Predicted vs Actual — {task.upper()} Q1 2026", fontsize=11, color=FG)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTDIR / f"{num}_scatter_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 & 6: Box plots of absolute errors
# ═══════════════════════════════════════════════════════════════════════════════

def plot_boxplot(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "05" if task == "price" else "06"

    names  = [m[0] for m in meta]
    colours= [m[1] for m in meta]
    errors = [np.abs(actual.values - preds[n].values) for n in names]
    maes   = [np.mean(e) for e in errors]

    # Sort by MAE
    order = np.argsort(maes)
    names_s  = [names[i] for i in order]
    colours_s= [colours[i] for i in order]
    errors_s = [errors[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(errors_s, patch_artist=True, vert=False,
                    whis=[5, 95], showfliers=True,
                    flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                    markerfacecolor=FG, markeredgewidth=0),
                    medianprops=dict(color=FG, linewidth=2),
                    whiskerprops=dict(color=FG, linewidth=1),
                    capprops=dict(color=FG, linewidth=1.2))

    for patch, col in zip(bp["boxes"], colours_s):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
        patch.set_edgecolor(FG)
        patch.set_linewidth(0.8)

    # MAE labels
    for k, (e, col) in enumerate(zip(errors_s, colours_s), start=1):
        mae = np.mean(e)
        ax.text(ax.get_xlim()[1] * 0.98, k, f"MAE={mae:.1f}",
                ha="right", va="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_yticks(range(1, len(names_s) + 1))
    ax.set_yticklabels(names_s, fontsize=8)
    for tick, col in zip(ax.get_yticklabels(), colours_s):
        tick.set_color(col)

    ax.set_xlabel(f"Absolute Error ({unit})", fontsize=9)
    ax.set_title(f"Forecast Error Distribution — {task.upper()} Q1 2026\n"
                 f"(Whiskers: 5th–95th pctile, Box: IQR, Line: Median)",
                 fontsize=10, color=FG)
    ax.grid(axis="x")
    plt.tight_layout()
    out = OUTDIR / f"{num}_boxplot_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 7 & 8: Hourly error profile
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hourly_profile(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "07" if task == "price" else "08"

    # Build hourly profile
    hours = actual.index.hour

    fig, ax = plt.subplots(figsize=(11, 5))
    for (display, colour, ls, lw) in meta:
        err = np.abs(actual.values - preds[display].values)
        hourly_mae = pd.Series(err, index=actual.index).groupby(actual.index.hour).mean()
        ax.plot(hourly_mae.index, hourly_mae.values,
                color=colour, lw=lw, ls=ls, label=display, marker="o", markersize=3)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of Day", fontsize=9)
    ax.set_ylabel(f"Mean Absolute Error ({unit})", fontsize=9)
    ax.set_title(f"MAE by Hour of Day — {task.upper()} Q1 2026", fontsize=10, color=FG)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y")
    plt.tight_layout()
    out = OUTDIR / f"{num}_hourly_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 9 & 10: Day-of-week error profile
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dow_profile(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "09" if task == "price" else "10"
    DOW   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (display, colour, ls, lw) in meta:
        err = np.abs(actual.values - preds[display].values)
        dow_mae = pd.Series(err, index=actual.index).groupby(actual.index.dayofweek).mean()
        ax.plot(dow_mae.index, dow_mae.values,
                color=colour, lw=lw, ls=ls, label=display, marker="s", markersize=4)

    ax.set_xticks(range(7))
    ax.set_xticklabels(DOW, fontsize=9)
    ax.axvspan(4.5, 6.5, alpha=0.08, color="#fbbf24", label="_weekend shade")
    ax.set_ylabel(f"Mean Absolute Error ({unit})", fontsize=9)
    ax.set_title(f"MAE by Day of Week — {task.upper()} Q1 2026", fontsize=10, color=FG)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y")
    plt.tight_layout()
    out = OUTDIR / f"{num}_dow_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 11 & 12: Monthly MAE breakdown — Static vs Walk-Forward MR pairs
# ═══════════════════════════════════════════════════════════════════════════════

# One representative per strategy (static best, MR best).
# Format: (strategy_label, static_display, mr_display, static_colour, mr_colour)
# Colours: dark shade = static; light shade = MR (same hue family).
# Colours match FAMILY_SHADES / strat_color exactly:
#   TF static=#0d47a1, WF-TF MR=#1b5e20, Rec static=#e65100, Rec MR=#ffb74d,
#   MIMO static=#8e24aa, MIMO MR=#ce93d8, Direct static=#c2185b, Direct MR=#f06292
# One representative per strategy: best STATIC vs best NON-STATIC (Walk-Forward MR).
# Keys MUST match the display_name strings in PRICE_MODELS / LOAD_MODELS above.
# Colours = approved Fig 11/12 family scheme (dark = static, light = MR).
_MONTHLY_PAIRS = {
    "price": [
        ("Teacher-Forcing",  "TF Static — Ensemble-Best3",  "TF MR — LGBM",             "#0d47a1", "#1b5e20"),
        ("Recursive",        "Rec Static — XGB-SS-Dense",   "Rec MR — XGB-SS-DO",       "#e65100", "#ffb74d"),
        ("MIMO",             "MIMO Static — Ensemble-Best3","MIMO MR — Ensemble-Best3", "#8e24aa", "#ce93d8"),
        ("Direct",           "Direct Static — LGBM-Optuna", "Direct MR — LGBM-Dense",   "#c2185b", "#f06292"),
    ],
    "load": [
        ("Teacher-Forcing",  "TF Static — Ensemble-Best3",  "TF MR — Ensemble-Best3",   "#0d47a1", "#1b5e20"),
        ("Recursive",        "Rec Static — LGBM-SS",        "Rec MR — XGB-SS-DO",       "#e65100", "#ffb74d"),
        ("MIMO",             "MIMO Static — Ensemble-Best3","MIMO MR — Ensemble-Best3", "#8e24aa", "#ce93d8"),
        ("Direct",           "Direct Static — Ensemble-All","Direct MR — LGBM-Dense",   "#c2185b", "#f06292"),
    ],
}

# All MR series are now validated full-length (2160h) via the v2 (…_optuna) JSONs,
# so no per-month MAE overrides are needed.
_MONTHLY_MAE_OVERRIDE: dict[str, dict[str, dict[str, float]]] = {"price": {}, "load": {}}

def plot_monthly_mae(task: str) -> None:
    actual, preds, _ = load_task_data(task, include_monthly_only=True)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "11" if task == "price" else "12"

    pairs   = _MONTHLY_PAIRS[task]
    months  = sorted(actual.index.to_period("M").unique())
    n_strat = len(pairs)           # 4 strategies
    n_month = len(months)          # 3 months

    # Layout: for each month, 4 strategy-pairs each with 2 bars + small inner gap
    pair_width = 0.12              # width of one bar
    inner_gap  = 0.02             # gap between static and MR bar within a pair
    outer_gap  = 0.06             # gap between strategy pairs within a month
    pair_span  = pair_width * 2 + inner_gap
    group_span = n_strat * pair_span + (n_strat - 1) * outer_gap
    month_step = group_span + 0.3  # spacing between month-groups on x-axis

    month_centres = np.arange(n_month) * month_step
    # compute centre offset for each strategy pair within a month-group
    pair_starts = []
    cur = -group_span / 2
    for _ in range(n_strat):
        pair_starts.append(cur)
        cur += pair_span + outer_gap

    fig, ax = plt.subplots(figsize=(13, 6))

    static_handle = mr_handle = None   # for legend
    for si, (strat_lbl, static_key, mr_key, col_s, col_mr) in enumerate(pairs):
        ps = pair_starts[si]
        for mi, mo in enumerate(months):
            mask = actual.index.to_period("M") == mo
            base = month_centres[mi] + ps

            # static bar
            act_vals = actual.values[mask]
            if static_key in preds.columns:
                mae_s = np.mean(np.abs(act_vals - preds[static_key].values[mask]))
                b = ax.bar(base, mae_s, pair_width,
                           color=col_s, alpha=0.92, edgecolor="#222", linewidth=0.6,
                           label=("Static" if si == 0 and mi == 0 else None))
                if si == 0 and mi == 0:
                    static_handle = b
                # value annotation
                ax.text(base, mae_s + ax.get_ylim()[1] * 0.005,
                        f"{mae_s:.1f}", ha="center", va="bottom",
                        fontsize=6.5, color=col_s, rotation=90)

            # MR bar — prefer hardcoded override over potentially-corrupt JSON series
            mo_key = str(mo)
            override_val = _MONTHLY_MAE_OVERRIDE.get(task, {}).get(mr_key, {}).get(mo_key)
            mae_mr = None
            if override_val is not None:
                mae_mr = override_val
            elif mr_key in preds.columns:
                mr_vals = preds[mr_key].values
                mr_mask = mask & ~np.isnan(mr_vals)
                if mr_mask.sum() > 0:
                    mae_mr = np.mean(np.abs(act_vals[mr_mask[mask]] - mr_vals[mr_mask]))
            if mae_mr is not None:
                b2 = ax.bar(base + pair_width + inner_gap, mae_mr, pair_width,
                            color=col_mr, alpha=0.92, edgecolor="#222", linewidth=0.6,
                            hatch="////",
                            label=("Walk-Forward MR" if si == 0 and mi == 0 else None))
                if si == 0 and mi == 0:
                    mr_handle = b2
                ax.text(base + pair_width + inner_gap, mae_mr + ax.get_ylim()[1] * 0.005,
                        f"{mae_mr:.1f}", ha="center", va="bottom",
                        fontsize=6.5, color=col_mr, rotation=90)

    # Month labels at group centres
    ax.set_xticks(month_centres)
    ax.set_xticklabels([mo.strftime("%b %Y") for mo in
                        pd.PeriodIndex(months).to_timestamp()], fontsize=9)

    # Strategy mini-legend (colour patches) — placed below x-axis tick labels
    from matplotlib.patches import Patch
    strat_patches = [Patch(facecolor=col_s, edgecolor="#222", linewidth=0.6, label=lbl)
                     for lbl, _, _, col_s, _ in pairs]

    # Two-row legend: top row = strategies (colour), bottom row = bar-type (solid/hatch)
    type_patches = [
        Patch(facecolor="#aaaaaa", edgecolor="#222", linewidth=0.6, label="Static (best model)"),
        Patch(facecolor="#aaaaaa", edgecolor="#222", linewidth=0.6, hatch="////",
              label="Walk-Forward MR (best model)"),
    ]
    leg1 = ax.legend(handles=strat_patches, title="Στρατηγική",
                     loc="upper left", fontsize=8, title_fontsize=8.5,
                     framealpha=0.85, edgecolor="#475569",
                     ncol=len(pairs))
    ax.add_artist(leg1)
    ax.legend(handles=type_patches,
              loc="upper right", fontsize=8,
              framealpha=0.85, edgecolor="#475569")

    ax.set_ylabel(f"MAE ({unit})", fontsize=10)
    ax.set_title(
        f"Μηνιαία Ανάλυση MAE — {task.upper()} Q1 2026\n"
        f"Static (σταθερή εκπαίδευση) vs Walk-Forward Monthly Retrain",
        fontsize=10, color=FG, pad=8)
    ax.grid(axis="y", alpha=0.4)
    ax.set_xlim(month_centres[0] - group_span / 2 - 0.15,
                month_centres[-1] + group_span / 2 + 0.15)
    # Give headroom for bar annotations
    ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
    plt.tight_layout()
    out = OUTDIR / f"{num}_monthly_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 13 & 14: Error distribution histograms
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_hist(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "13" if task == "price" else "14"
    n_m   = len(meta)
    ncols = 4
    nrows = (n_m + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (display, colour, ls, lw) in zip(axes_flat, meta):
        err = actual.values - preds[display].values   # signed error
        ax.hist(err, bins=60, color=colour, alpha=0.75, edgecolor="none", density=True)
        ax.axvline(0, color=FG, lw=1.0, ls="--", alpha=0.6)
        ax.axvline(np.mean(err), color="#fbbf24", lw=1.2, ls="--",
                   label=f"Bias={np.mean(err):.1f}")
        mae = np.mean(np.abs(err))
        ax.set_title(f"{display}\nMAE={mae:.2f}, Bias={np.mean(err):.2f} {unit}",
                     fontsize=7.5, color=FG, pad=3)
        ax.set_xlabel(f"Error ({unit})", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_facecolor(BG2)
        ax.legend(fontsize=6.5)

    for ax in axes_flat[len(meta):]:
        ax.set_visible(False)

    fig.suptitle(f"Error Distribution — {task.upper()} Q1 2026", fontsize=11, color=FG)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUTDIR / f"{num}_hist_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 15 & 16: Cumulative absolute error over time
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_error(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "15" if task == "price" else "16"

    fig, ax = plt.subplots(figsize=(13, 5))
    for (display, colour, ls, lw) in meta:
        abs_err = np.abs(actual.values - preds[display].values)
        cum_err = np.cumsum(abs_err) / np.arange(1, len(abs_err) + 1)  # running MAE
        ax.plot(actual.index, cum_err, color=colour, lw=lw, ls=ls, label=display)

    for m in pd.date_range("2026-01-01", "2026-03-01", freq="MS"):
        if m > actual.index[0]:
            ax.axvline(m, color="#475569", ls=":", lw=0.8, alpha=0.7)

    ax.set_title(f"Running Mean Absolute Error — {task.upper()} Q1 2026", fontsize=10, color=FG)
    ax.set_ylabel(f"Running MAE ({unit})", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y")
    plt.tight_layout()
    out = OUTDIR / f"{num}_cumae_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 17 & 18: Actual price/load heat-map (hour-of-day × date)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_actual_heatmap(task: str) -> None:
    actual, _, _ = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "17" if task == "price" else "18"
    cmap  = "YlOrRd" if task == "price" else "Blues"

    # Pivot: rows = hour (0–23), columns = date
    df = pd.DataFrame({"val": actual.values, "hour": actual.index.hour,
                       "date": actual.index.date})
    pivot = df.pivot_table(index="hour", columns="date", values="val", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) * 0.45), 6))
    # Robust colour scaling: clip to 2nd–98th percentile so the bulk of the
    # values (not the few price spikes) span the full colour range.
    vmin, vmax = np.nanpercentile(pivot.values, [2, 98])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower",
                   vmin=vmin, vmax=vmax,
                   extent=[0, len(pivot.columns), -0.5, 23.5])

    # Month separators
    dates  = list(pivot.columns)
    months = pd.Series(dates).apply(lambda d: d.month)
    breaks = np.where(np.diff(months.values) != 0)[0] + 0.5
    for b in breaks:
        ax.axvline(b, color=FG, lw=1.2, ls="--", alpha=0.6)

    # X-tick labels: 1st of each month
    tick_pos   = []
    tick_label = []
    for k, (d_prev, d_curr) in enumerate(zip(dates, dates[1:]), start=0):
        if d_curr.day == 1:
            tick_pos.append(k + 1)
            tick_label.append(pd.Timestamp(d_curr).strftime("%d %b %Y"))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_label, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax.set_ylabel("Hour of Day", fontsize=9)
    ax.set_title(f"Actual {task.title()} ({unit}) — Hourly Heat-map Q1 2026", fontsize=10, color=FG)
    plt.colorbar(im, ax=ax, label=unit, shrink=0.8)
    plt.tight_layout()
    out = OUTDIR / f"{num}_heatmap_actual_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 23 & 24: Signed forecast-error heat-map (best model, hour × date)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_heatmap(task: str) -> None:
    """Signed forecast error of the lowest-MAE model as an hour-of-day × date
    heat-map. Far more informative for an error-analysis chapter than the raw
    actual-value map: it shows *where* and *when* the model fails."""
    actual, preds, _ = load_task_data(task)
    if preds.shape[1] == 0:
        warnings.warn("no predictions for error heatmap"); return
    unit = "€/MWh" if task == "price" else "MW"
    num  = "23" if task == "price" else "24"

    mae  = preds.sub(actual, axis=0).abs().mean()
    best = mae.idxmin()
    err  = preds[best] - actual          # +: over-prediction, -: under-prediction

    dfe = pd.DataFrame({"val": err.values, "hour": err.index.hour,
                        "date": err.index.date})
    pivot = dfe.pivot_table(index="hour", columns="date", values="val", aggfunc="mean")

    lim = np.nanpercentile(np.abs(pivot.values), 98)
    fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) * 0.45), 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", origin="lower",
                   vmin=-lim, vmax=lim, extent=[0, len(pivot.columns), -0.5, 23.5])

    dates  = list(pivot.columns)
    months = pd.Series(dates).apply(lambda d: d.month)
    for b in np.where(np.diff(months.values) != 0)[0] + 0.5:
        ax.axvline(b, color=FG, lw=1.2, ls="--", alpha=0.6)

    tick_pos, tick_label = [], []
    for k, d_curr in enumerate(dates):
        if d_curr.day == 1:
            tick_pos.append(k)
            tick_label.append(pd.Timestamp(d_curr).strftime("%d %b %Y"))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_label, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax.set_ylabel("Hour of Day", fontsize=9)
    ax.set_title(f"Forecast error — {best} ({unit}), Q1 2026\n"
                 f"red = over-prediction, blue = under-prediction",
                 fontsize=10, color=FG)
    plt.colorbar(im, ax=ax, label=f"error ({unit})", shrink=0.8)
    plt.tight_layout()
    out = OUTDIR / f"{num}_heatmap_error_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 19 & 20: Bias (over/under-prediction) per model × hour
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bias_profile(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "19" if task == "price" else "20"

    fig, ax = plt.subplots(figsize=(11, 5))
    for (display, colour, ls, lw) in meta:
        signed_err  = actual.values - preds[display].values   # positive = under-predicted
        hourly_bias = pd.Series(signed_err, index=actual.index).groupby(actual.index.hour).mean()
        ax.plot(hourly_bias.index, hourly_bias.values,
                color=colour, lw=lw, ls=ls, label=display, marker="o", markersize=3)

    ax.axhline(0, color=FG, lw=0.9, ls="--", alpha=0.6)
    ax.fill_between(range(24), 0, 0, alpha=0)  # invisible fill for reference
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of Day", fontsize=9)
    ax.set_ylabel(f"Mean Error (Actual – Predicted) ({unit})", fontsize=9)
    ax.set_title(f"Hourly Bias Profile — {task.upper()} Q1 2026\n"
                 f"Positive = model under-predicts, Negative = over-predicts",
                 fontsize=10, color=FG)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y")
    plt.tight_layout()
    out = OUTDIR / f"{num}_bias_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 21 & 22: Weekly error profile (week number × model)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_weekly_mae(task: str) -> None:
    actual, preds, meta = load_task_data(task)
    unit  = "€/MWh" if task == "price" else "MW"
    num   = "21" if task == "price" else "22"

    fig, ax = plt.subplots(figsize=(14, 5))
    weeks = actual.index.isocalendar().week.values
    unique_weeks = sorted(set(weeks))

    for (display, colour, ls, lw) in meta:
        abs_err = np.abs(actual.values - preds[display].values)
        weekly_mae = [np.mean(abs_err[weeks == w]) for w in unique_weeks]
        ax.plot(range(len(unique_weeks)), weekly_mae,
                color=colour, lw=lw, ls=ls, label=display, marker="o", markersize=4)

    # Month boundaries
    week_dates = [actual.index[weeks == w][0].date() for w in unique_weeks]
    for k, wd in enumerate(week_dates):
        if pd.Timestamp(wd).day <= 7:  # approx. start of month
            ax.axvline(k, color="#475569", ls=":", lw=0.8, alpha=0.7)
            ax.text(k + 0.1, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                    pd.Timestamp(wd).strftime("%b"), fontsize=7.5,
                    color="#94a3b8", va="top")

    ax.set_xticks(range(len(unique_weeks)))
    ax.set_xticklabels([f"W{w}" for w in unique_weeks], rotation=45, fontsize=7)
    ax.set_ylabel(f"Weekly MAE ({unit})", fontsize=9)
    ax.set_title(f"Weekly MAE over Q1 2026 — {task.upper()}", fontsize=10, color=FG)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis="y")
    plt.tight_layout()
    out = OUTDIR / f"{num}_weekly_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Thesis Plots — Q1 2026 Forecast Evaluation")
    print("=" * 60)
    for task in ("price", "load"):
        print(f"\n  Task: {task.upper()}")
        print(f"  {'─' * 40}")
        plot_time_series(task)
        plot_scatter(task)
        plot_boxplot(task)
        plot_hourly_profile(task)
        plot_dow_profile(task)
        plot_monthly_mae(task)
        plot_error_hist(task)
        plot_cumulative_error(task)
        plot_actual_heatmap(task)
        plot_error_heatmap(task)
        plot_bias_profile(task)
        plot_weekly_mae(task)

    print(f"\nAll plots saved to: {OUTDIR}")
    print(f"Total files: {len(list(OUTDIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
