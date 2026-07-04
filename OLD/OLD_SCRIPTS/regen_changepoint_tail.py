"""
regen_changepoint_tail.py
=========================
Light-style thesis figures (master §7). LANGUAGE RULE: every in-figure element
(axis labels, legends, tick labels, annotations) is ENGLISH; only the figure
suptitle is Greek. Direct is a SEPARATE strategy (#c2185b), never grouped with MIMO.

(1) REGIME  → thesis_output/ch6_forecast_analysis/
        30_regime_changepoint_price.png · 30_regime_changepoint_load.png
    Two stacked panels, shared x (no twin-axis), ACTUAL signal only:
      (a) 30-day rolling mean  → level collapse (price 109→78) + Feb-regime shading
          + ONE PELT/rbf change-point line
      (b) 30-day rolling std   → volatility change
    Legend outside; no diagonal annotation.

(2) SPIKE (price) → thesis_output/ch6_forecast_analysis/  — four separate files,
    5 strategies (Actual, WF-TF, Recursive, MIMO, Direct):
        31a_timeseries.png  full Q1 actual + flags + plain threshold line
        31b_missrate.png    compact horizontal bars of spike-miss % per strategy
        31c_dec.png         December spike episode
        31d_feb.png         February spike episode

Note: Fig 32 (price tail) has been removed.

OneDrive note: each figure is written to a local temp dir, then copied to the
destination (mtime printed).

Usage:
  conda run -n epf --no-capture-output python regen_changepoint_tail.py
"""
from __future__ import annotations
import time, shutil, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import ruptures as rpt

from regen_regime_spike import COL, LABELS, load_curves, _plot_curves, OUTDIR, ROOT, SPLIT

SHADE_C    = "#ededed"     # Feb regime shading
CP_C       = "#b00020"     # change-point vertical line
STD_C      = "#00695c"     # rolling-std line (non-strategy colour)
MEAN_C     = "#1a1a1a"     # rolling-mean line
TITLE_FS, LABEL_FS, TICK_FS = 13, 11, 9
MISS_TOL   = 20.0          # €/MWh — strategy "misses" a flagged spike if |error| > MISS_TOL
STRATS     = ["wf_tf", "rec", "mimo", "direct"]   # forecasting strategies (no Actual)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#222222",
    "text.color":        "#222222",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "grid.color":        "#aeaeae",
    "grid.linestyle":    (0, (1, 1.5)),
    "grid.linewidth":    1.0,
    "font.family":       "DejaVu Sans",
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

TMP = Path(tempfile.gettempdir()) / "ch6_cp_restyle"
TMP.mkdir(parents=True, exist_ok=True)


def _save(fig, dest: Path, extra: str = ""):
    tmp = TMP / dest.name
    fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(tmp, dest)
    st = dest.stat()
    print(f"  → {dest.name}  ({st.st_size}B, mtime {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))}) {extra}")


def _grid(ax, axis="both"):
    ax.grid(True, which="major", axis=axis)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


# ═══ (1) Regime — two stacked panels (rolling mean / rolling std) ════════════
def regime_changepoint(task: str, fname: str):
    unit    = "€/MWh" if task == "price" else "MW"
    task_gr = "Τιμή" if task == "price" else "Φορτίο"
    a = load_curves(task)["actual"]

    roll_mean = a.rolling(720, min_periods=720).mean()
    roll_std  = a.rolling(720, min_periods=720).std()
    first_valid = roll_mean.first_valid_index()

    cp_idx  = rpt.KernelCPD(kernel="rbf").fit(a.to_numpy().reshape(-1, 1)).predict(n_bkps=1)[0]
    cp_date = a.index[cp_idx]

    jan, feb = a[a.index.month == 1], a[a.index.month == 2]
    m_b, s_b = jan.mean(), jan.std()
    m_a, s_a = feb.mean(), feb.std()
    pct = 100.0 * (m_b - m_a) / m_b
    s_lo, s_hi = a[a.index.month == 12].std(), feb.std()

    # Covariate-shift sub-samples (geometric train/test distribution comparison):
    #   A = December 2025 (training);  B = February 2026 after the change-point (test)
    sampA = a[a.index.month == 12]
    sampB = a[a.index >= cp_date]
    mA, sA = sampA.mean(), sampA.std()
    mB, sB = sampB.mean(), sampB.std()

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(12.0, 9.5), sharex=True)

    for ax in (axT, axB):
        ax.axvspan(SPLIT, a.index[-1], color=SHADE_C, zorder=0)
        ax.axvline(cp_date, color=CP_C, ls="--", lw=1.8, zorder=4)

    # (a) rolling mean — level, with covariate-shift μ±σ bands (A=Dec, B=Feb post-CP)
    axT.axhspan(mA - sA, mA + sA, color="#0d47a1", alpha=0.08, zorder=0)
    axT.axhspan(mB - sB, mB + sB, color="#c2185b", alpha=0.08, zorder=0)
    axT.plot(roll_mean.index, roll_mean.values, color=MEAN_C, lw=2.8, zorder=5)
    _lo = min(mA - sA, mB - sB, float(roll_mean.min()))
    _hi = max(mA + sA, mB + sB, float(roll_mean.max()))
    axT.set_ylim(_lo - 0.05 * (_hi - _lo), _hi + 0.05 * (_hi - _lo))
    axT.set_ylabel(f"30-day rolling mean ({unit})")
    axT.set_title("(a) Level — 30-day rolling mean", loc="left", fontsize=LABEL_FS, pad=6)
    axT.annotate(f"Covariate shift (train → test)\n"
                 f"Dec train:  μ {mA:.0f} ± {sA:.0f} {unit}\n"
                 f"Feb post-CP test:  μ {mB:.0f} ± {sB:.0f} {unit}",
                 xy=(0.985, 0.95), xycoords="axes fraction", ha="right", va="top",
                 fontsize=TICK_FS + 1,
                 bbox=dict(boxstyle="round,pad=0.45", fc="#fff8e1", ec="#777777", lw=0.9))

    # (b) rolling std — volatility
    axB.plot(roll_std.index, roll_std.values, color=STD_C, lw=2.4, zorder=5)
    axB.set_ylabel(f"30-day rolling std ({unit})")
    axB.set_title("(b) Volatility — 30-day rolling std", loc="left", fontsize=LABEL_FS, pad=6)
    axB.set_xlabel("Date")

    axB.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axB.set_xlim(first_valid, a.index[-1])
    _grid(axT, "y"); _grid(axB, "y")

    # Legend OUTSIDE (below the panels), English
    handles = [
        mlines.Line2D([], [], color=MEAN_C, lw=2.8, label="30-day rolling mean (level)"),
        mlines.Line2D([], [], color=STD_C, lw=2.4, label="30-day rolling std (volatility)"),
        mlines.Line2D([], [], color=CP_C, lw=1.8, ls="--",
                      label=f"PELT/rbf change-point ({cp_date:%d/%m})"),
        mpatches.Patch(fc=SHADE_C, ec="#cccccc", label="February low-price regime"),
        mpatches.Patch(fc="#0d47a1", alpha=0.30, ec="#0d47a1",
                       label="Training Regime Space (Dec 2025)"),
        mpatches.Patch(fc="#c2185b", alpha=0.30, ec="#c2185b",
                       label="Shifted Regime Space (Feb 2026)"),
    ]
    # Bigger legend so the 6 entries read clearly (Fig 6.1/6.2)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               framealpha=0.95, edgecolor="#cccccc", fontsize=TICK_FS + 3,
               handlelength=2.4, columnspacing=1.8, handletextpad=0.7,
               borderpad=0.8, labelspacing=0.6, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(f"Μεταβολή Καθεστώτος Αγοράς — {task_gr} (Q1 2026)",
                 fontsize=TITLE_FS + 1, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.12, 1, 0.965))
    _save(fig, OUTDIR / fname,
          extra=f"[CP {cp_date:%d/%m}; {m_b:.0f}→{m_a:.0f} {unit} −{pct:.0f}%; σ {s_b:.0f}→{s_a:.0f} (roll {s_lo:.0f}→{s_hi:.0f})]")


# ═══ (2) Spike figures (price) — four separate files ════════════════════════
def _spike_data():
    df = load_curves("price")
    actual = df["actual"]
    thr = float(np.percentile(actual.values, 99))
    flags = actual[actual > thr]
    miss = {k: 100.0 * ((df.loc[flags.index, k] - flags).abs() > MISS_TOL).mean()
            for k in STRATS}
    return df, actual, thr, flags, miss


def spike_timeseries(df, actual, thr, flags, fname="31a_timeseries.png"):
    unit = "€/MWh"
    # Fig 6.13: keep a large/tall plot block, but ENLARGE & DARKEN every text element
    # (ticks, axis labels, threshold annotation) so they stay legible once the figure
    # is scaled to \textwidth, and draw slightly bigger spike markers.
    BIG_TICK, BIG_LABEL, BIG_ANNOT, BIG_TITLE, BIG_LEG = 15, 18, 16, 20, 16
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.plot(actual.index, actual.values, color=COL["actual"], lw=1.2, zorder=3,
            label="Actual")
    ax.axhline(thr, color=CP_C, ls="--", lw=1.4, zorder=2)          # plain threshold line
    ax.scatter(flags.index, flags.values, s=52, color="#d81b60", edgecolor="white",
               linewidth=0.6, zorder=5, label=f"Flagged spike (n={len(flags)})")
    ax.text(actual.index[-1], thr, f" 99th pctile = {thr:.0f} {unit}", color=CP_C,
            fontsize=BIG_ANNOT, fontweight="bold", va="bottom", ha="right")
    ax.set_ylabel(f"Price ({unit})", fontsize=BIG_LABEL, color="#1a1a1a")
    ax.set_xlabel("Date", fontsize=BIG_LABEL, color="#1a1a1a")
    ax.set_xlim(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-01"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    _grid(ax, "y")
    ax.tick_params(axis="both", labelsize=BIG_TICK, colors="#1a1a1a")   # after _grid so it wins
    # Bigger legend, placed close beneath where the plot ends (Fig 6.13)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.11),
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              fontsize=BIG_LEG, markerscale=2.0, handlelength=2.6,
              columnspacing=2.4, handletextpad=0.9, borderpad=1.0)
    fig.suptitle("Επισήμανση Αιχμών Τιμής (Q1 2026)",
                 fontsize=BIG_TITLE, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, OUTDIR / fname, extra=f"[thr {thr:.0f}, n={len(flags)}]")


def spike_missrate(miss, fname="31b_missrate.png"):
    unit = "€/MWh"
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    vals   = [miss[k] for k in STRATS]
    colors = [COL[k] for k in STRATS]
    xpos   = np.arange(len(STRATS))
    ax.bar(xpos, vals, color=colors, edgecolor="#444444", linewidth=0.6, width=0.62)
    for x, v in zip(xpos, vals):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=TICK_FS + 1, fontweight="bold", color="#222222")
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[k] for k in STRATS])
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)
    ax.set_ylim(0, 105)
    ax.set_ylabel(f"Missed flagged spikes (%)   (|error| > {MISS_TOL:.0f} {unit})")
    _grid(ax, "y")
    fig.suptitle("Ποσοστό Αστοχίας Αιχμών Τιμής ανά Στρατηγική",
                 fontsize=TITLE_FS, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.18)
    _save(fig, OUTDIR / fname,
          extra=f"[WF-TF {miss['wf_tf']:.0f}% / Rec {miss['rec']:.0f}% / MIMO {miss['mimo']:.0f}% / Direct {miss['direct']:.0f}%]")


def spike_episode(df, actual, pk, suptitle_gr, fname):
    unit = "€/MWh"
    w0, w1 = pk - pd.Timedelta(days=1), pk + pd.Timedelta(days=1)
    win = df.loc[w0:w1]

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    _plot_curves(ax, win, lw_actual=2.8, lw_model=1.5)

    peak_val   = actual.loc[pk]
    model_vals = {k: df[k].loc[pk] for k in STRATS}
    best_key   = max(model_vals, key=model_vals.get)
    gap        = peak_val - model_vals[best_key]
    ax.scatter([pk], [peak_val], color=COL["actual"], s=46, zorder=6)
    ax.annotate(f"Missed peak −{gap:.0f} {unit}",
                xy=(pk, peak_val), xytext=(0.04, 0.94), textcoords="axes fraction",
                ha="left", va="top", fontsize=TICK_FS + 1, color="#b71c1c",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.1,
                                connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#b71c1c", lw=1.0))

    ax.set_ylabel(f"Price ({unit})")
    ax.set_xlabel("Hour")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.set_xlim(w0, w1)
    _grid(ax, "both")
    # Legend OUTSIDE (below) so it never overlaps the curves / the peak (Fig 6.15/6.16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=TICK_FS)
    fig.suptitle(suptitle_gr, fontsize=TITLE_FS, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, OUTDIR / fname, extra=f"[peak {peak_val:.0f} {unit} @ {pk:%d/%m %H:00}, gap −{gap:.0f}]")


# ═══ Main ════════════════════════════════════════════════════════════════════
def main():
    print(f"Rebuild Ch6 regime + spike figures  (temp staging: {TMP})")
    print(" (1) Regime — stacked rolling mean/std:")
    regime_changepoint("price", "30_regime_changepoint_price.png")
    regime_changepoint("load",  "30_regime_changepoint_load.png")

    print(" (2) Spike (price) — 4 files, 5 strategies:")
    df, actual, thr, flags, miss = _spike_data()
    print(f"   threshold (99th)={thr:.1f} €/MWh, n={len(flags)}, tol={MISS_TOL:.0f}")
    for k in STRATS:
        print(f"     {LABELS[k]:<22s} misses {miss[k]:5.1f}%")
    spike_timeseries(df, actual, thr, flags, "31a_timeseries.png")
    spike_missrate(miss, "31b_missrate.png")
    dec_peak = actual[actual.index.month == 12].idxmax()
    feb_peak = actual[actual.index.month == 2].idxmax()
    spike_episode(df, actual, dec_peak, "Επεισόδιο Αιχμής — Δεκέμβριος 2025", "31c_dec.png")
    spike_episode(df, actual, feb_peak, "Επεισόδιο Αιχμής — Φεβρουάριος 2026", "31d_feb.png")
    print("Done.")


if __name__ == "__main__":
    main()
