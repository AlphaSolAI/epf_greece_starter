"""
regen_load_spikes.py
====================
Spike analysis figures for LOAD (equivalent to price 31a–d).

Figures produced in thesis_output/ch6_forecast_analysis/:
  32a_load_spike_timeseries.png  — full Q1 2026 load series, flagged spikes
  32b_load_spike_missrate.png    — miss-rate bar per strategy
  32c_load_spike_dec.png         — Dec 2025 peak episode (+/- 1 day)
  32d_load_spike_feb.png         — Feb 2026 peak episode (+/- 1 day)

Spike threshold : load > 99th pctile of Q1 2026 actual
Miss threshold  : |error| > 200 MW
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

ROOT   = Path(__file__).resolve().parent
OUTDIR = ROOT / "thesis_output" / "ch6_forecast_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
TITLE_FS, LABEL_FS, TICK_FS = 13, 11, 9
GRID_C = "#aeaeae"

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

COL = {
    "actual": "#222222",
    "wf_tf":  "#1b5e20",   # Walk-Forward TF
    "rec":    "#1e88e5",   # Recursive
    "mimo":   "#8e24aa",   # MIMO
    "direct": "#c2185b",   # Direct
}
LABELS = {
    "actual": "Actual",
    "wf_tf":  "WF-TF (Ens-Best3)",
    "rec":    "Recursive (XGB-SS)",
    "mimo":   "MIMO (Ens-Best3)",
    "direct": "Direct (LGBM)",
}
ORDER = ["actual", "wf_tf", "rec", "mimo", "direct"]

MISS_THR = 200.0   # MW — "missed spike" if |error| > this

# ── Data ─────────────────────────────────────────────────────────────────────
def _load(fn: str) -> dict:
    return json.loads((ROOT / fn).read_text(encoding="utf-8"))


def load_curves() -> pd.DataFrame:
    mr   = _load("dashboard_data_hourly_load_monthly_retrain.json")
    mimo = _load("dashboard_data_hourly_load_mimo_monthly_retrain_optuna.json")

    idx_mr   = pd.to_datetime(mr["dates"])
    idx_mimo = pd.to_datetime(mimo["dates"])

    df = pd.DataFrame({
        "actual": pd.Series(np.asarray(mr["actual"], float),
                            index=idx_mr),
        "wf_tf":  pd.Series(np.asarray(
                      mr["series"]["Ensemble-TF-Best3 (1/MAE)"], float),
                            index=idx_mr),
        "rec":    pd.Series(np.asarray(
                      mr["series"]["Rec-XGB-SS-DO-MR"], float),
                            index=idx_mr),
        "mimo":   pd.Series(np.asarray(
                      mimo["series"]["Ensemble-Best3 MIMO MR"], float),
                            index=idx_mimo),
        "direct": pd.Series(np.asarray(
                      mimo["series"]["LGBM Direct Dense MR+CachedOpt"], float),
                            index=idx_mimo),
    })
    return df.sort_index().dropna(subset=["actual"])


def _style(ax):
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


# ── 32a: Spike detection timeseries ──────────────────────────────────────────
def fig_32a(df: pd.DataFrame):
    actual  = df["actual"]
    thresh  = float(np.percentile(actual.dropna(), 99))
    spikes  = actual[actual >= thresh]

    BIG_TICK, BIG_LABEL, BIG_ANNOT, BIG_TITLE, BIG_LEG = 15, 18, 16, 20, 16
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    ax.plot(actual.index, actual.values, color=COL["actual"], lw=1.2, zorder=3,
            label="Actual")
    ax.axhline(thresh, color="#c62828", ls="--", lw=1.4, zorder=2)
    ax.text(actual.index[-1], thresh, f" 99th pctile = {thresh:.0f} MW",
            color="#c62828", fontsize=BIG_ANNOT, fontweight="bold",
            va="bottom", ha="right")
    ax.scatter(spikes.index, spikes.values, s=52, color="#d81b60",
               edgecolor="white", linewidth=0.6, zorder=5,
               label=f"Flagged spike (n={len(spikes)})")

    ax.set_xlim(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-01"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set_ylabel("Load (MW)", fontsize=BIG_LABEL, color="#1a1a1a")
    ax.set_xlabel("Date", fontsize=BIG_LABEL, color="#1a1a1a")
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=BIG_TICK, colors="#1a1a1a")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.11),
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              fontsize=BIG_LEG, markerscale=2.0, handlelength=2.6,
              columnspacing=2.4, handletextpad=0.9, borderpad=1.0)
    fig.suptitle("Επισήμανση Αιχμών Φορτίου (Q1 2026)",
                 fontsize=BIG_TITLE, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = OUTDIR / "32a_load_spike_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}   (99th pctile = {thresh:.0f} MW, n_spikes = {len(spikes)})")


# ── 32b: Miss-rate bar per strategy ──────────────────────────────────────────
def fig_32b(df: pd.DataFrame):
    actual = df["actual"]
    thresh = float(np.percentile(actual.dropna(), 99))
    spikes_idx = actual[actual >= thresh].index

    keys    = ["wf_tf", "rec", "mimo", "direct"]
    rates   = []
    for k in keys:
        errs = np.abs(actual.loc[spikes_idx] - df[k].loc[spikes_idx])
        rates.append(100.0 * (errs > MISS_THR).mean())

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    xpos   = np.arange(len(keys))
    colors = [COL[k] for k in keys]
    ax.bar(xpos, rates, color=colors, edgecolor="#444444", linewidth=0.6, width=0.62)
    for x, v in zip(xpos, rates):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=TICK_FS + 1, fontweight="bold", color="#222222")
    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[k] for k in keys])
    for tick, c in zip(ax.get_xticklabels(), colors):
        tick.set_color(c)
    ax.set_ylim(0, 105)
    ax.set_ylabel(f"Missed flagged spikes (%)   (|error| > {MISS_THR:.0f} MW)")
    ax.grid(True, which="major", axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")
    fig.suptitle("Ποσοστό Αστοχίας Αιχμών Φορτίου ανά Στρατηγική",
                 fontsize=TITLE_FS, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.18)
    out = OUTDIR / "32b_load_spike_missrate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}   rates: " +
          " | ".join(f"{LABELS[k]}={r:.0f}%" for k, r in zip(keys, rates)))


# ── 32c / 32d: Episode panels ─────────────────────────────────────────────────
def fig_episode(df: pd.DataFrame, month: int, fname: str, month_gr: str):
    actual  = df["actual"]
    seg     = actual[actual.index.month == month]
    pk      = seg.idxmax()
    w0, w1  = pk - pd.Timedelta(days=1), pk + pd.Timedelta(days=1)
    win     = df.loc[w0:w1]

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for key in ORDER:
        is_act = key == "actual"
        ax.plot(win.index, win[key].values,
                color=COL[key],
                lw=2.8 if is_act else 1.5,
                zorder=5 if is_act else 3,
                label=LABELS[key])

    peak_val   = actual.loc[pk]
    model_vals = {k: df[k].loc[pk] for k in ("wf_tf", "rec", "mimo", "direct")}
    best_key   = max(model_vals, key=model_vals.get)
    gap        = peak_val - model_vals[best_key]

    ax.scatter([pk], [peak_val], color=COL["actual"], s=46, zorder=6)
    ax.annotate(f"Missed peak −{gap:.0f} MW",
                xy=(pk, peak_val), xytext=(0.04, 0.94), textcoords="axes fraction",
                ha="left", va="top", fontsize=TICK_FS + 1, color="#b71c1c",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.1,
                                connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#b71c1c", lw=1.0))

    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Hour")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.set_xlim(w0, w1)
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=TICK_FS)
    fig.suptitle(f"Επεισόδιο Αιχμής — {month_gr}", fontsize=TITLE_FS, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUTDIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}   peak={peak_val:.0f} MW  gap={gap:.0f} MW  ({pk})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Load spike figures →", OUTDIR)
    df = load_curves()
    fig_32a(df)
    fig_32b(df)
    fig_episode(df, month=12, fname="32c_load_spike_dec.png",
                month_gr="Δεκέμβριος 2025")
    fig_episode(df, month=1,  fname="32d_load_spike_jan.png",
                month_gr="Ιανουάριος 2026")
    print("Done.")


if __name__ == "__main__":
    main()
