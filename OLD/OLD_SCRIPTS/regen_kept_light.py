"""
regen_kept_light.py
===================
Light-theme restyle of the KEPT Chapter-6 figures, matching the 30/31/32 protocol
(master §7). Data-logic is reused verbatim from src/thesis_plots.py
(load_task_data + the per-figure aggregations are unchanged); only the aesthetics
are swapped (white background, strategy palette, dotted #d9d9d9 grid, no top/right
spines, title13/labels11/ticks9). The boxplots additionally clip the x-axis to the
95th percentile of absolute errors so the IQR boxes are readable.

Restyled (same filenames):
  05/06 boxplot · 07/08 hourly · 11/12 monthly · 15/16 cumae · 19/20 bias
  25/26 cross_regime_delta

NOT touched: 03/04, 09/10, 13/14, 17/18, 21/22 (being removed from the thesis).

OneDrive note: savefig can fail to land on the OneDrive mount, so every figure is
written to a local temp dir first, then copied into ch6_forecast_analysis/ and the
destination mtime/size is printed for confirmation.

Usage:
  conda run -n epf --no-capture-output python regen_kept_light.py
"""
from __future__ import annotations
import os, time, shutil, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reuse the data-logic exactly as the thesis uses it ───────────────────────
from src.thesis_plots import load_task_data  # noqa: E402  (registries baked in)

ROOT = Path(__file__).resolve().parent
CH6  = ROOT / "thesis_output" / "ch6_forecast_analysis"
TMP  = Path(tempfile.gettempdir()) / "ch6_restyle"
TMP.mkdir(parents=True, exist_ok=True)

# ── Light style (master §7 / same as 30/31/32) ──────────────────────────────
GRID_C    = "#aeaeae"   # slightly darker so grid lines read clearly (was #d9d9d9)
DARK_TEXT = "#222222"
SEP_C     = "#999999"   # month separators
TITLE_FS, LABEL_FS, TICK_FS = 13, 11, 9

# User-requested larger, bolder typography for the kept Chapter-6 figures
# (overrides the 13/11/9 baseline for these specific \textwidth thesis plots so
# the labels/titles stay legible when the figure is shrunk to page width).
BIG_TITLE, BIG_SUB, BIG_LABEL, BIG_TICK, BIG_LEG = 21, 14, 17, 14, 15
# Intermediate scale for the cross-regime delta plot (6.5/6.6): the full BIG set
# was too heavy, so dial back toward the 13/11/9 baseline and drop most bold.
MID_TITLE, MID_LABEL, MID_TICK, MID_LEG = 16, 13, 11, 12
# Larger legends across the kept Ch6 figures (user request: fill the green box).
LEG_WIDE = 18

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   DARK_TEXT,
    "text.color":        DARK_TEXT,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "grid.color":        GRID_C,
    "grid.linestyle":    (0, (1, 1.5)),
    "grid.linewidth":    1.0,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    TITLE_FS,
    "axes.labelsize":    LABEL_FS,
    "xtick.labelsize":   TICK_FS,
    "ytick.labelsize":   TICK_FS,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.framealpha": 0.95,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
})

# ── Strategy/regime palette (matches Fig 11/12 family scheme exactly) ─────────
#   per strategy: dark shade = STATIC, light shade = NON-STATIC (Walk-Forward MR)
#   TF static #0d47a1 / TF MR #1b5e20 · Rec static #e65100 / Rec MR #ffb74d
#   MIMO static #8e24aa / MIMO MR #ce93d8 · Direct static #c2185b / Direct MR #f06292
_STRAT_COLORS = {
    "TF MR":          "#1b5e20",   # Teacher-Forcing, Walk-Forward Monthly Retrain
    "TF Static":      "#0d47a1",   # Teacher-Forcing, static
    "Rec MR":         "#ffb74d",   # Recursive, Walk-Forward MR
    "Rec Static":     "#e65100",   # Recursive, static
    "MIMO MR":        "#ce93d8",   # MIMO, Walk-Forward MR
    "MIMO Static":    "#8e24aa",   # MIMO, static
    "Direct MR":      "#f06292",   # Direct, Walk-Forward MR
    "Direct Static":  "#c2185b",   # Direct, static
}

def strat_color(display: str) -> str:
    for prefix, col in _STRAT_COLORS.items():
        if display.startswith(prefix):
            return col
    return DARK_TEXT


# ── Short legend/title labels (user request) ─────────────────────────────────
# Drop the per-model suffix ("— Ensemble-Best3", "— LGBM", …) and keep only the
# strategy + regime: "<Strategy> Static" / "<Strategy> MR / WF".
_LEGEND_SHORT = [
    ("TF MR",         "TF MR / WF"),
    ("TF Static",     "TF Static"),
    ("Rec MR",        "Rec MR / WF"),
    ("Rec Static",    "Rec Static"),
    ("MIMO MR",       "MIMO MR / WF"),
    ("MIMO Static",   "MIMO Static"),
    ("Direct MR",     "Direct MR / WF"),
    ("Direct Static", "Direct Static"),
]

def legend_label(display: str) -> str:
    for prefix, short in _LEGEND_SHORT:
        if display.startswith(prefix):
            return short
    return display

# Monthly-bar / cross-regime strategy labels → short strategy code.
_SHORT_STRAT = {"Teacher-Forcing": "TF", "Recursive": "Rec", "MIMO": "MIMO", "Direct": "Direct"}

def short_strat(strat_lbl: str) -> str:
    return _SHORT_STRAT.get(strat_lbl, strat_lbl)


def _strat_family(display: str) -> str:
    """Coarse strategy family key (same buckets as strat_color)."""
    if display.startswith("TF MR"):      return "wf_tf"
    if display.startswith("TF Static"):  return "tf"
    if display.startswith("Rec"):        return "rec"
    if display.startswith("MIMO"):       return "mimo"
    if display.startswith("Direct"):     return "direct"
    return "other"


# Per-family (dark, light) shade pairs + a hatch on the 2nd member, so that the
# two algorithm variants inside each strategy family stay the SAME hue (→ instantly
# tells you the strategy) yet are unambiguously distinguishable in a bar chart
# (→ tells you which algorithm). Used by the monthly-MAE bars (Fig 6.3/6.4).
FAMILY_SHADES = {
    "wf_tf":  ["#1b5e20", "#66bb6a"],
    "tf":     ["#0d47a1", "#64b5f6"],
    "rec":    ["#e65100", "#ffb74d"],
    "mimo":   ["#8e24aa", "#ce93d8"],
    "direct": ["#c2185b", "#f06292"],
    "other":  [DARK_TEXT, "#888888"],
}
MEMBER_HATCH = [None, "////"]


def bar_styles(meta):
    """Return [(colour, hatch), …] giving each bar in `meta` a distinct look:
    family hue + (dark/light shade, hatch) for the k-th member of that family."""
    seen, styles = {}, []
    for (display, _c, _ls, _lw) in meta:
        fam = _strat_family(display)
        k   = seen.get(fam, 0)
        seen[fam] = k + 1
        shades = FAMILY_SHADES.get(fam, FAMILY_SHADES["other"])
        colour = shades[k] if k < len(shades) else shades[-1]
        hatch  = MEMBER_HATCH[k] if k < len(MEMBER_HATCH) else "xx"
        styles.append((colour, hatch))
    return styles


def _task_gr(task: str) -> str:
    return "Τιμή" if task == "price" else "Φορτίο"


def recolor(meta):
    """Remap only the colour of each (display, colour, ls, lw) tuple."""
    return [(d, strat_color(d), ls, lw) for (d, _c, ls, lw) in meta]


def _save(fig, name: str, tight: bool = True):
    tmp_path = TMP / name
    if tight:
        fig.savefig(tmp_path, dpi=150, bbox_inches="tight", facecolor="white")
    else:
        # Keep the full figure frame (no crop) so the axes stay centered within it.
        fig.savefig(tmp_path, dpi=150, facecolor="white")
    plt.close(fig)
    dst = CH6 / name
    shutil.copy2(tmp_path, dst)
    st = dst.stat()
    print(f"  → {name}  ({st.st_size} bytes, mtime {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))})")


# ═══ 05 / 06 — Box plots of absolute errors (x clipped to 95th pctile) ═══════
def plot_boxplot(task: str):
    actual, preds, meta = load_task_data(task)
    meta = recolor(meta)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "05" if task == "price" else "06"

    names   = [m[0] for m in meta]
    colours = [m[1] for m in meta]
    errors  = [np.abs(actual.values - preds[n].values) for n in names]
    maes    = [np.mean(e) for e in errors]

    order     = np.argsort(maes)
    names_s   = [names[i] for i in order]
    colours_s = [colours[i] for i in order]
    errors_s  = [errors[i] for i in order]

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    bp = ax.boxplot(errors_s, patch_artist=True, vert=False,
                    whis=[5, 95], showfliers=False,
                    medianprops=dict(color=DARK_TEXT, linewidth=2.4),
                    whiskerprops=dict(color="#555555", linewidth=1.3),
                    capprops=dict(color="#555555", linewidth=1.5))

    for patch, col in zip(bp["boxes"], colours_s):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.8)

    # x-limit = the largest per-model 95th pctile whisker (no fliers shown), plus
    # right margin for the MAE labels, so NOTHING is drawn outside the frame.
    whisker_max = float(max(np.percentile(e, 95) for e in errors_s))
    xmax = whisker_max * 1.15
    ax.set_xlim(0, xmax)

    # MAE labels (inside, right edge)
    for k, (e, col) in enumerate(zip(errors_s, colours_s), start=1):
        mae = np.mean(e)
        ax.text(xmax * 0.99, k, f"MAE={mae:.1f}",
                ha="right", va="center", fontsize=BIG_TICK, color=col, fontweight="bold")

    ax.set_yticks(range(1, len(names_s) + 1))
    ax.set_yticklabels([legend_label(n) for n in names_s], fontsize=BIG_TICK)
    for tick, col in zip(ax.get_yticklabels(), colours_s):
        tick.set_color(col)
        tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=BIG_TICK)

    ax.set_xlabel(f"Απόλυτο Σφάλμα ({unit})", fontsize=BIG_LABEL, fontweight="bold")
    # Clean single-line bold title; the whisker/box convention lives in the LaTeX caption.
    ax.set_title(f"Κατανομή Σφάλματος Πρόβλεψης — {_task_gr(task)} (Q1 2026)",
                 fontsize=BIG_TITLE, fontweight="bold", pad=16)
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    # Center the box stack within the figure frame: symmetric vertical margins
    # (8 boxes at y=1..8 → equal pad above/below) and balanced L/R margins. Save
    # WITHOUT a tight crop so the centred axes keep their place inside the frame.
    ax.set_ylim(0.5, len(names_s) + 0.5)
    fig.subplots_adjust(left=0.165, right=0.965, top=0.905, bottom=0.105)
    _save(fig, f"{num}_boxplot_{task}.png", tight=False)


# ═══ 07 / 08 — Hourly error profile ═════════════════════════════════════════
def plot_hourly_profile(task: str):
    actual, preds, meta = load_task_data(task)
    meta = recolor(meta)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "07" if task == "price" else "08"

    fig, ax = plt.subplots(figsize=(19.5, 11.0))
    for (display, colour, ls, lw) in meta:
        err = np.abs(actual.values - preds[display].values)
        hourly_mae = pd.Series(err, index=actual.index).groupby(actual.index.hour).mean()
        # Close at 24 = repeat of hour 0 for visual completeness
        xs = list(hourly_mae.index) + [24]
        ys = list(hourly_mae.values) + [hourly_mae.values[0]]
        ax.plot(xs, ys, color=colour, lw=lw + 2.2, ls=ls, label=legend_label(display), marker="o", markersize=7)

    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(-0.5, 24.5)
    ax.tick_params(axis="both", labelsize=BIG_TICK)
    ax.set_xlabel("Ώρα Ημέρας", fontsize=BIG_LABEL, fontweight="bold")
    ax.set_ylabel(f"Μέσο Απόλυτο Σφάλμα ({unit})", fontsize=BIG_LABEL, fontweight="bold")
    ax.set_title(f"Μέσο Απόλυτο Σφάλμα ανά Ώρα — {_task_gr(task)} (Q1 2026)",
                 fontsize=BIG_TITLE, fontweight="bold", pad=16)
    ax.legend(fontsize=LEG_WIDE+2, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=True, framealpha=0.95,
              edgecolor="#cccccc", columnspacing=1.8+0.4, handlelength=2.4+0.6,
              handletextpad=1.0, borderpad=1.1)
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{num}_hourly_{task}.png")


# ═══ 11 / 12 — Monthly MAE breakdown — Static vs Walk-Forward MR pairs ═══════
# Reuses _MONTHLY_PAIRS defined in thesis_plots for consistency.
from src.thesis_plots import _MONTHLY_PAIRS as _MP, _MONTHLY_MAE_OVERRIDE as _MAE_OVR

def plot_monthly_mae(task: str):
    actual, preds, _ = load_task_data(task, include_monthly_only=True)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "11" if task == "price" else "12"

    pairs   = _MP[task]
    months  = sorted(actual.index.to_period("M").unique())
    n_strat = len(pairs)

    # Clean template look: bars touch within a month group (no inner/outer gaps),
    # with a clear gap only between month groups.
    pair_width = 0.22
    inner_gap  = 0.0    # bars of a strategy pair touch
    outer_gap  = 0.0    # pairs touch → all 8 bars form one clean block per month
    pair_span  = pair_width * 2 + inner_gap
    group_span = n_strat * pair_span + (n_strat - 1) * outer_gap
    month_step = group_span + 0.55   # clear separation between month groups

    month_centres = np.arange(len(months)) * month_step
    pair_starts = []
    cur = -group_span / 2
    for _ in range(n_strat):
        pair_starts.append(cur)
        cur += pair_span + outer_gap

    # Wide template proportions but larger overall with more height (user request).
    fig, ax = plt.subplots(figsize=(17, 9))

    # Per-strategy default colours from the registry: TF blue(Static)/green(MR/WF),
    # Rec orange, MIMO purple, Direct pink — Static solid, MR/WF lighter + hatch.
    for si, (strat_lbl, static_key, mr_key, col_s, col_mr) in enumerate(pairs):
        ps = pair_starts[si]
        for mi, mo in enumerate(months):
            mask = actual.index.to_period("M") == mo
            base = month_centres[mi] + ps
            act_vals = actual.values[mask]

            if static_key in preds.columns:
                mae_s = np.mean(np.abs(act_vals - preds[static_key].values[mask]))
                ax.bar(base, mae_s, pair_width,
                       color=col_s, alpha=1.0, edgecolor="#222222", linewidth=0.8,
                       label=(short_strat(strat_lbl) + " Static" if mi == 0 else None))

            mo_key = str(mo)
            override_val = _MAE_OVR.get(task, {}).get(mr_key, {}).get(mo_key)
            mae_mr = None
            if override_val is not None:
                mae_mr = override_val
            elif mr_key in preds.columns:
                mr_v  = preds[mr_key].values
                mr_mk = mask & ~np.isnan(mr_v)
                if mr_mk.sum() > 0:
                    mae_mr = np.mean(np.abs(act_vals[mr_mk[mask]] - mr_v[mr_mk]))
            if mae_mr is not None:
                ax.bar(base + pair_width + inner_gap, mae_mr, pair_width,
                       color=col_mr, alpha=1.0, edgecolor="#222222", linewidth=0.8,
                       hatch="////",
                       label=(short_strat(strat_lbl) + " MR / WF" if mi == 0 else None))

    ax.set_xticks(month_centres)
    ax.set_xticklabels([mo.strftime("%b %Y") for mo in
                        pd.PeriodIndex(months).to_timestamp()],
                       fontsize=BIG_TICK, fontweight="bold")
    ax.tick_params(axis="y", labelsize=BIG_TICK)
    ax.set_ylabel(f"MAE ({unit})", fontsize=BIG_LABEL, fontweight="bold")
    
    # ── ΑΛΛΑΓΗ ΤΙΤΛΟΥ: Μονός καθαρός τίτλος από το PDF [cite: 158, 182]
    title_text = f"Μηνιαία Ανάλυση ΜΑΕ – Τιμή (Q1 2026)" if task == "price" else f"Μηνιαία Ανάλυση ΜΑΕ – Φορτίο (Q1 2026)"
    ax.set_title(title_text, fontsize=BIG_TITLE + 2, fontweight="bold", pad=18)
    
    ax.set_xlim(month_centres[0] - group_span / 2 - 0.15,
                month_centres[-1] + group_span / 2 + 0.15)
    # Tighter Y-limit (less empty headroom) so the axis fits the data like the
    # reference template — auto-scaled per task (price ~€30, load ~MW250).
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    
    # Ideal compact legend (matches the green-boxed bias legend in the screenshot).
    ax.legend(fontsize=LEG_WIDE, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), frameon=True, framealpha=0.95,
              edgecolor="#cccccc", columnspacing=1.8, handlelength=2.4,
              handletextpad=1.0, borderpad=1.1)

    # ── ΑΛΛΑΓΗ ΓΚΡΙΝΤ: Πιο έντονο πλέγμα με διακεκομμένη γραμμή
    ax.grid(axis="y", linestyle="--", color="#b0b0b0", linewidth=1.2, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{num}_monthly_{task}.png")


# ═══ 15 / 16 — Running (cumulative) MAE over time ═══════════════════════════
def plot_cumulative_error(task: str):
    actual, preds, meta = load_task_data(task)
    meta = recolor(meta)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "15" if task == "price" else "16"

    fig, ax = plt.subplots(figsize=(14, 7.5))
    for (display, colour, ls, lw) in meta:
        abs_err = np.abs(actual.values - preds[display].values)
        cum_err = np.cumsum(abs_err) / np.arange(1, len(abs_err) + 1)  # running MAE
        ax.plot(actual.index, cum_err, color=colour, lw=lw, ls=ls, label=legend_label(display))

    for m in pd.date_range("2026-01-01", "2026-03-01", freq="MS"):
        if m > actual.index[0]:
            ax.axvline(m, color=SEP_C, ls=":", lw=0.9, alpha=0.8)

    ax.set_title(f"Τρέχον Μέσο Απόλυτο Σφάλμα — {_task_gr(task)} (Q1 2026)", fontsize=TITLE_FS)
    ax.set_ylabel(f"Running MAE ({unit})", fontsize=LABEL_FS)
    ax.set_xlabel("Date", fontsize=LABEL_FS)
    # Ideal compact legend (matches the green-boxed bias legend in the screenshot).
    ax.legend(fontsize=LEG_WIDE, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=True, framealpha=0.95,
              edgecolor="#cccccc", columnspacing=1.8, handlelength=2.4,
              handletextpad=1.0, borderpad=1.1)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{num}_cumae_{task}.png")


# ═══ 19 / 20 — Hourly bias profile ══════════════════════════════════════════
def plot_bias_profile(task: str):
    actual, preds, meta = load_task_data(task)
    meta = recolor(meta)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "19" if task == "price" else "20"

    fig, ax = plt.subplots(figsize=(19.5, 11.0))
    for (display, colour, ls, lw) in meta:
        signed_err  = actual.values - preds[display].values   # positive = under-predicted
        hourly_bias = pd.Series(signed_err, index=actual.index).groupby(actual.index.hour).mean()
        # Close at 24 = repeat of hour 0 for visual completeness
        xs = list(hourly_bias.index) + [24]
        ys = list(hourly_bias.values) + [hourly_bias.values[0]]
        ax.plot(xs, ys, color=colour, lw=lw + 2.2, ls=ls, label=legend_label(display), marker="o", markersize=7)

    ax.axhline(0, color=DARK_TEXT, lw=1.1, ls="--", alpha=0.7)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(-0.5, 24.5)
    ax.tick_params(axis="both", labelsize=BIG_TICK)
    ax.set_xlabel("Ώρα Ημέρας", fontsize=BIG_LABEL, fontweight="bold")
    ax.set_ylabel(f"Μέσο Σφάλμα (Πραγματικό – Πρόβλεψη) ({unit})",
                  fontsize=BIG_LABEL, fontweight="bold")
    ax.set_title(f"Προφίλ Ωριαίας Μεροληψίας — {_task_gr(task)} (Q1 2026)",
                 fontsize=BIG_TITLE, fontweight="bold", pad=16)
    ax.legend(fontsize=LEG_WIDE+2, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=True, framealpha=0.95,
              edgecolor="#cccccc", columnspacing=1.8+0.4, handlelength=2.4+0.6,
              handletextpad=1.0, borderpad=1.1)
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{num}_bias_{task}.png")


# ═══ 25 / 26 — Cross-Regime Performance Delta (ΔsMAPE per strategy) ══════════
# Per STRATEGY (TF / Rec / MIMO / Direct): the best static model vs the best
# Walk-Forward Monthly-Retrain model, compared in sMAPE. ΔsMAPE = sMAPE(static) −
# sMAPE(MR); positive ⇒ retraining lowers the error. All series validated 2160h.
# Pairs + colours reuse the approved _MONTHLY_PAIRS scheme (dark = static hue).
def _smape(a: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(p)
    denom = np.abs(a[m]) + np.abs(p[m])
    denom = np.where(denom == 0, 1e-9, denom)
    return float(np.mean(200.0 * np.abs(a[m] - p[m]) / denom))


def plot_cross_regime_delta(task: str):
    actual, preds, _ = load_task_data(task, include_monthly_only=True)
    num = "25" if task == "price" else "26"
    pairs = _MP[task]

    labels, dmae_pct, dsmape, col_mae, col_sm = [], [], [], [], []
    a = actual.values
    for strat_lbl, static_key, mr_key, col_s, col_mr in pairs:
        if static_key not in preds.columns or mr_key not in preds.columns:
            continue
        mae_s  = float(np.mean(np.abs(a - preds[static_key].values)))
        mae_mr = float(np.mean(np.abs(a - preds[mr_key].values)))
        sm_s   = _smape(a, preds[static_key].values)
        sm_mr  = _smape(a, preds[mr_key].values)
        labels.append(strat_lbl)
        dmae_pct.append((mae_s - mae_mr) / mae_s * 100.0)   # % MAE reduction
        dsmape.append(sm_s - sm_mr)                          # pp sMAPE reduction
        col_mae.append(col_s)    # dark family shade = ΔMAE %
        col_sm.append(col_mr)    # light family shade = ΔsMAPE pp

    dmae_pct = np.array(dmae_pct); dsmape = np.array(dsmape)
    x = np.arange(len(labels)); w = 0.38

    fig, axL = plt.subplots(figsize=(11.0, 6.2))
    axR = axL.twinx()

    b1 = axL.bar(x - w / 2, dmae_pct, w, color=col_mae, alpha=0.92,
                 edgecolor="#222", linewidth=0.6, zorder=3)
    b2 = axR.bar(x + w / 2, dsmape, w, color=col_sm, alpha=0.92,
                 edgecolor="#222", linewidth=0.6, hatch="////", zorder=3)
    axL.axhline(0, color=DARK_TEXT, lw=0.9, zorder=4)

    # Annotations: positive → above bar tip, negative → below bar tip
    for bar, v, col in zip(b1, dmae_pct, col_mae):
        bx = bar.get_x() + bar.get_width() / 2
        if v >= 0:
            axL.annotate(f"{v:+.1f}%", xy=(bx, v), xytext=(0, 5),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=MID_TICK + 1, fontweight="bold", color=col)
        else:
            axL.annotate(f"{v:+.1f}%", xy=(bx, v), xytext=(0, -5),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=MID_TICK + 1, fontweight="bold", color=col)
    for bar, v, col in zip(b2, dsmape, col_sm):
        bx = bar.get_x() + bar.get_width() / 2
        if v >= 0:
            axR.annotate(f"{v:+.2f} pp", xy=(bx, v), xytext=(0, 5),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=MID_TICK, color="#444444")
        else:
            axR.annotate(f"{v:+.2f} pp", xy=(bx, v), xytext=(0, -5),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=MID_TICK, color="#444444")

    axL.set_xticks(x)
    axL.set_xticklabels([short_strat(l) for l in labels], fontsize=MID_LABEL)
    axL.set_ylabel("Μείωση MAE (%)", fontsize=MID_LABEL, fontweight="bold")
    axR.set_ylabel("Μείωση sMAPE (ποσοστιαίες μονάδες)",
                   fontsize=MID_LABEL, fontweight="bold")
    axL.tick_params(axis="y", labelsize=MID_TICK)
    axR.tick_params(axis="y", labelsize=MID_TICK)

    # Align zero of both axes at the same pixel: derive a common fraction below zero
    lo_l = min(0.0, float(dmae_pct.min())); hi_l = max(0.0, float(dmae_pct.max()))
    lo_r = min(0.0, float(dsmape.min()));   hi_r = max(0.0, float(dsmape.max()))
    pad_pos_l = hi_l * 0.25 + 1.5;   pad_neg_l = abs(lo_l) * 0.30 + 1.5
    pad_pos_r = hi_r * 0.25 + 0.12;  pad_neg_r = abs(lo_r) * 0.30 + 0.12
    total_l = (hi_l + pad_pos_l) + (abs(lo_l) + pad_neg_l)
    total_r = (hi_r + pad_pos_r) + (abs(lo_r) + pad_neg_r)
    frac_l = (abs(lo_l) + pad_neg_l) / total_l if total_l > 0 else 0.20
    frac_r = (abs(lo_r) + pad_neg_r) / total_r if total_r > 0 else 0.20
    frac = max(frac_l, frac_r, 0.15)   # at least 15% below zero for visual clarity
    span_l = (hi_l + pad_pos_l) / (1.0 - frac)
    span_r = (hi_r + pad_pos_r) / (1.0 - frac)
    axL.set_ylim(-frac * span_l, (1.0 - frac) * span_l)
    axR.set_ylim(-frac * span_r, (1.0 - frac) * span_r)

    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#888", edgecolor="#222", linewidth=0.6, label="Μείωση MAE % (αριστ. άξονας)"),
           Patch(facecolor="#bbb", edgecolor="#222", linewidth=0.6, hatch="////",
                 label="Μείωση sMAPE pp (δεξ. άξονας)")]
    # Ideal compact legend (matches the green-boxed bias legend in the screenshot).
    axL.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.11),
               ncol=2, fontsize=12, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", columnspacing=1.4, handlelength=1.8,
               handletextpad=1.0, borderpad=1.1)

    axL.set_title(f"Διαφορά Επίδοσης μεταξύ Καθεστώτων — {_task_gr(task)} (Q1 2026)",
                  fontsize=MID_TITLE, fontweight="bold", pad=14)
    axL.grid(axis="y"); axL.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{num}_cross_regime_delta_{task}.png")


# ═══ Main ════════════════════════════════════════════════════════════════════
def main():
    print(f"Restyle (light) of kept Ch6 figures → {CH6}")
    print(f"  (temp staging: {TMP})")
    for task in ("price", "load"):
        plot_boxplot(task)
        plot_hourly_profile(task)
        plot_monthly_mae(task)
        plot_cumulative_error(task)
        plot_bias_profile(task)
        plot_cross_regime_delta(task)
    print("Done.")


if __name__ == "__main__":
    main()
