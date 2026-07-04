"""
regen_regime_spike.py
=====================
Light-style thesis figures for Chapter 6 forecast analysis.

(B) Regime figure (price & load) — 2 panels:
      top    = daily mean over the whole Q1 (4 curves) + vertical 1-Feb line
               + regime-drop annotation
      bottom = hourly zoom 26 Jan – 1 Feb 2026
(C) Spike figure (price only) — 2 panels side-by-side:
      the two maximum-actual episodes (one in Dec, one in Feb), +/-1 day window,
      Actual emphasised, arrow on the missed peak annotated with the EUR gap.

Light style: white background, palette of the master context, fonts
title13/labels11/ticks9, dotted grid #d9d9d9, no top/right spines.

Curve colours (master spec):
  Actual #222 · WF-TF #1b5e20 · Recursive #1e88e5 · MIMO #8e24aa

Outputs (thesis_output/ch6_forecast_analysis/):
  27_regime_price.png · 28_regime_load.png · 29_spike_price.png

Usage:
  conda run -n epf --no-capture-output python regen_regime_spike.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent
OUTDIR = ROOT / "thesis_output" / "ch6_forecast_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─── Light style ────────────────────────────────────────────────────────────
TITLE_FS, LABEL_FS, TICK_FS = 13, 11, 9
GRID_C  = "#aeaeae"                            # slightly darker so grid reads clearly
SPLIT   = pd.Timestamp("2026-02-01")          # regime shift / Walk-Forward boundary

COL = {
    "actual": "#222222",   # Actual
    "wf_tf":  "#1b5e20",   # WF-TF  (TF-LGBM-MR)
    "rec":    "#1e88e5",   # Recursive (Rec-XGB-SS-DO-MR)
    "mimo":   "#8e24aa",   # MIMO  (Ensemble-Best3 MIMO MR)
    "direct": "#c2185b",   # Direct (separate strategy — never grouped with MIMO)
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    TITLE_FS,
    "axes.labelsize":    LABEL_FS,
    "xtick.labelsize":   TICK_FS,
    "ytick.labelsize":   TICK_FS,
    "legend.fontsize":   TICK_FS + 0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444444",
    "grid.color":        GRID_C,
    "grid.linestyle":    (0, (1, 1.5)),
    "grid.linewidth":    1.0,
})


def _style_axis(ax):
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


# ─── Data loading ───────────────────────────────────────────────────────────
def _load(fn: str) -> dict:
    return json.loads((ROOT / fn).read_text(encoding="utf-8"))


def load_curves(task: str) -> pd.DataFrame:
    """Return a DataFrame with columns actual / wf_tf / rec / mimo on a common
    hourly index for the given task ('price' or 'load')."""
    mr   = _load(f"dashboard_data_hourly_{task}_monthly_retrain.json")
    mimo = _load(f"dashboard_data_hourly_{task}_mimo_monthly_retrain_optuna.json")

    idx_mr   = pd.to_datetime(mr["dates"])
    idx_mimo = pd.to_datetime(mimo["dates"])

    df = pd.DataFrame({
        "actual": pd.Series(np.asarray(mr["actual"], float),                       index=idx_mr),
        "wf_tf":  pd.Series(np.asarray(mr["series"]["TF-LGBM-MR"], float),          index=idx_mr),
        "rec":    pd.Series(np.asarray(mr["series"]["Rec-XGB-SS-DO-MR"], float),    index=idx_mr),
        "mimo":   pd.Series(np.asarray(mimo["series"]["Ensemble-Best3 MIMO MR"], float),
                            index=idx_mimo),
        # Direct = separate walk-forward strategy (same MR family as the others here)
        "direct": pd.Series(np.asarray(mimo["series"]["LGBM Direct Dense MR+CachedOpt"], float),
                            index=idx_mimo),
    })
    return df.sort_index()


# In-figure text is English (only the figure suptitle stays Greek).
LABELS = {
    "actual": "Actual",
    "wf_tf":  "WF-TF (LGBM)",
    "rec":    "Recursive (XGB-SS)",
    "mimo":   "MIMO (Ens-Best3)",
    "direct": "Direct (LGBM)",
}
ORDER = ["actual", "wf_tf", "rec", "mimo", "direct"]


def _plot_curves(ax, data: pd.DataFrame, *, lw_actual=2.4, lw_model=1.5,
                 markers=False):
    for key in ORDER:
        is_act = key == "actual"
        ax.plot(data.index, data[key].values,
                color=COL[key],
                lw=lw_actual if is_act else lw_model,
                zorder=5 if is_act else 3,
                marker="o" if (markers and is_act) else None,
                ms=2.6 if markers else 0,
                label=LABELS[key])


# ─── (B) Regime figure ──────────────────────────────────────────────────────
def regime_figure(task: str, fname: str):
    unit = "€/MWh" if task == "price" else "MW"
    task_gr = "Τιμής" if task == "price" else "Φορτίου"
    df = load_curves(task)

    daily = df.resample("D").mean()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12.5, 9.2))

    # ── Top: daily mean over all Q1 ─────────────────────────────────────────
    _plot_curves(ax_top, daily, lw_actual=2.6, lw_model=1.7, markers=True)
    ax_top.axvline(SPLIT, color="#c62828", ls="--", lw=1.4, zorder=2)
    ax_top.text(SPLIT, ax_top.get_ylim()[1], "  1 Feb",
                color="#c62828", fontsize=TICK_FS, va="top", ha="left",
                fontweight="bold")

    # Regime-drop annotation (actual Jan vs Feb mean)
    jan_m = df["actual"][df.index.month == 1].mean()
    feb_m = df["actual"][df.index.month == 2].mean()
    pct   = 100.0 * (jan_m - feb_m) / jan_m
    feb_daily = daily["actual"][daily.index >= SPLIT]
    tgt_x = feb_daily.index[min(4, len(feb_daily) - 1)]
    tgt_y = feb_daily.iloc[min(4, len(feb_daily) - 1)]
    ax_top.annotate(
        f"Regime shift\nJan {jan_m:.0f} → Feb {feb_m:.0f} {unit}  (−{pct:.0f}%)",
        xy=(tgt_x, tgt_y),
        xytext=(0.40, 0.30), textcoords="axes fraction",
        fontsize=TICK_FS + 1, color="#222222", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#c62828", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.4),
    )

    ax_top.set_title("(a) Daily mean — Q1 2026", loc="left", pad=8)
    ax_top.set_ylabel(f"{unit}")
    ax_top.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    _style_axis(ax_top)
    ax_top.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95,
                  edgecolor="#cccccc")

    # ── Bottom: hourly zoom 26 Jan – 1 Feb ──────────────────────────────────
    z0, z1 = pd.Timestamp("2026-01-26 00:00"), pd.Timestamp("2026-02-01 23:00")
    zoom = df.loc[z0:z1]
    _plot_curves(ax_bot, zoom, lw_actual=2.2, lw_model=1.4)
    ax_bot.axvline(SPLIT, color="#c62828", ls="--", lw=1.4, zorder=2)
    ax_bot.set_title("(b) Hourly detail — 26 Jan to 1 Feb 2026", loc="left", pad=8)
    ax_bot.set_ylabel(f"{unit}")
    ax_bot.set_xlabel("Date")
    ax_bot.xaxis.set_major_locator(mdates.DayLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax_bot.set_xlim(z0, z1)
    _style_axis(ax_bot)
    ax_bot.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95,
                  edgecolor="#cccccc")

    fig.suptitle(f"Μεταβολή Καθεστώτος Αγοράς — Πρόβλεψη {task_gr} (Walk-Forward MR)",
                 fontsize=TITLE_FS + 1, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUTDIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}   (Ιαν {jan_m:.1f} → Φεβ {feb_m:.1f} {unit}, −{pct:.0f}%)")


# ─── (C) Spike figure (price only) ──────────────────────────────────────────
def spike_figure(fname: str = "29_spike_price.png"):
    unit = "€/MWh"
    df = load_curves("price")
    actual = df["actual"]

    # Two maximum-actual episodes: one in December, one in February
    dec_peak = actual[actual.index.month == 12].idxmax()
    feb_peak = actual[actual.index.month == 2].idxmax()
    episodes = [("Dec 2025", dec_peak), ("Feb 2026", feb_peak)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for ax, (mon, pk) in zip(axes, episodes):
        w0, w1 = pk - pd.Timedelta(days=1), pk + pd.Timedelta(days=1)
        win = df.loc[w0:w1]
        _plot_curves(ax, win, lw_actual=2.8, lw_model=1.5)

        # Missed peak: gap between actual peak and best (closest) model at that hour
        peak_val   = actual.loc[pk]
        model_vals = {k: df[k].loc[pk] for k in ("wf_tf", "rec", "mimo", "direct")}
        best_key   = max(model_vals, key=model_vals.get)   # closest from below
        best_val   = model_vals[best_key]
        gap        = peak_val - best_val

        ax.scatter([pk], [peak_val], color=COL["actual"], s=46, zorder=6)
        ax.annotate(
            f"Missed peak\n−{gap:.0f} {unit}",
            xy=(pk, peak_val), xytext=(pk, best_val - 0.18 * (peak_val - best_val) - 1),
            ha="center", va="top", fontsize=TICK_FS + 1, color="#b71c1c",
            fontweight="bold",
            arrowprops=dict(arrowstyle="<->", color="#b71c1c", lw=1.8),
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#b71c1c", lw=1.0),
        )

        ax.set_title(f"Episode {mon}: peak {peak_val:.0f} {unit}\n"
                     f"({pk:%d/%m %H:00})", loc="left", pad=8)
        ax.set_ylabel(f"{unit}")
        ax.set_xlabel("Hour")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
        ax.set_xlim(w0, w1)
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
        _style_axis(ax)

    axes[0].legend(loc="upper left", ncol=1, frameon=True, framealpha=0.95,
                   edgecolor="#cccccc")
    fig.suptitle("Αιχμές Τιμής DAM — Αδυναμία Αποτύπωσης από τα Μοντέλα",
                 fontsize=TITLE_FS + 1, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUTDIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}   (Δεκ peak {actual.loc[dec_peak]:.0f}, "
          f"Φεβ peak {actual.loc[feb_peak]:.0f} {unit})")


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # NOTE: figures 27/28 (regime) and 29 (spike) were retired — superseded by
    # 30 (regime change-point) and 31a–d (spike). Their generators remain below
    # only because load_curves/_plot_curves/COL/LABELS are imported elsewhere.
    print("27/28/29 are retired (replaced by 30 + 31a–d); nothing to generate.")


if __name__ == "__main__":
    main()
