"""
generate_thesis_figures.py
Παράγει τα 4 βασικά figures για τη διπλωματική εργασία.
Style reference: Kanousis (2025), University of Patras.

  Fig 1 – fig_price_overview.png    (3-panel: time series, hourly profile, monthly boxplot)
  Fig 2 – fig_load_overview.png     (3-panel: time series, hourly profile, monthly boxplot)
  Fig 3 – fig_corr_heatmap_price.png (24x24 Pearson, Kanousis style, viridis)
  Fig 4 – fig_corr_heatmap_load.png  (24x24 Pearson, Kanousis style, viridis)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

# ─── Global rcParams (thesis style) ──────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

ROOT = Path(r"C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter")
OUT  = ROOT / "thesis_output"
OUT.mkdir(exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
_price_df = pd.read_parquet(ROOT / "data/processed/hourly.parquet")
price = _price_df["y"].astype(float)
price.index = pd.to_datetime(price.index)

_load_df  = pd.read_parquet(ROOT / "data/processed/hourly_load.parquet")
load_s = _load_df["y"].astype(float)
load_s.index = pd.to_datetime(load_s.index)

# Align load to price start (Jan 2017+)
load_s = load_s[load_s.index >= "2017-01-01"]

print(f"  price : {price.index.min().date()} → {price.index.max().date()}  "
      f"({len(price):,} pts)  y=[{price.min():.0f}, {price.max():.0f}] €/MWh")
print(f"  load  : {load_s.index.min().date()} → {load_s.index.max().date()}  "
      f"({len(load_s):,} pts)  y=[{load_s.min():.0f}, {load_s.max():.0f}] MW")


# ─── Helper: monthly boxplot data ────────────────────────────────────────────
def _monthly_box(series: pd.Series, start_year: int = 2019):
    sub = series[series.index.year >= start_year].dropna()
    periods = sub.index.to_period("M")
    groups  = sub.groupby(periods)
    all_p   = sorted(groups.groups.keys())
    data    = [groups.get_group(p).values for p in all_p]
    return data, all_p


# ─── FIGURE 1: Price overview (3 panels) ─────────────────────────────────────
def fig_price_overview():
    print("Generating fig_price_overview...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.subplots_adjust(hspace=0.42)

    # ── Panel α: Full hourly time series ─────────────────────────────────────
    ax = axes[0]
    ax.plot(price.index, price.values,
            lw=0.3, alpha=0.4, color="#5B9BD5", zorder=2, rasterized=True)
    weekly_ma = price.resample("W").mean()
    ax.plot(weekly_ma.index, weekly_ma.values,
            lw=1.8, color="#1F3864", label="Εβδομαδιαίος μέσος", zorder=3)
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"),
               alpha=0.12, color="#E74C3C", label="Ενεργειακή κρίση 2021–22", zorder=1)
    ax.axhline(0, color="gray", lw=0.5, ls="--", zorder=1)
    ax.set_ylabel("Τιμή (€/MWh)")
    ax.set_ylim(-60, 980)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "(α)  Ωριαίες Τιμές ΗΕ Ελλάδας — Ιαν. 2017 έως Μαρ. 2026",
        fontsize=11, fontweight="bold", pad=8
    )

    # ── Panel β: Mean hourly profile ─────────────────────────────────────────
    ax = axes[1]
    hp    = price.groupby(price.index.hour).agg(["mean", "std"])
    hours = np.arange(24)
    ax.fill_between(hours,
                    hp["mean"] - hp["std"],
                    hp["mean"] + hp["std"],
                    alpha=0.20, color="#5B9BD5", label="±1σ")
    ax.plot(hours, hp["mean"],
            color="#1F3864", lw=2.2, marker="o", ms=5, label="Μέση τιμή")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Ώρα ημέρας")
    ax.set_ylabel("Τιμή (€/MWh)")
    ax.set_title("(β)  Μέσο Ωριαίο Προφίλ Τιμής (2017–2026)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9)

    # ── Panel γ: Monthly boxplots (2019–) ────────────────────────────────────
    ax = axes[2]
    data, periods = _monthly_box(price, start_year=2019)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
    for patch in bp["boxes"]:
        patch.set(facecolor="#AED6F1", edgecolor="#1F3864", lw=0.8)
    for med in bp["medians"]:
        med.set(color="#E74C3C", lw=2.0)
    for wh in bp["whiskers"]:
        wh.set(color="#1F3864", lw=0.7)
    for cap in bp["caps"]:
        cap.set(color="#1F3864", lw=0.7)

    tpos, tlbl = [], []
    for i, p in enumerate(periods):
        if p.month in (1, 7):
            tpos.append(i + 1)
            tlbl.append(str(p))
    ax.set_xticks(tpos)
    ax.set_xticklabels(tlbl, rotation=45, fontsize=8, ha="right")
    ax.set_xlabel("Μήνας")
    ax.set_ylabel("Τιμή (€/MWh)")
    ax.set_title("(γ)  Μηνιαία Κατανομή Τιμής (2019–2026)",
                 fontsize=11, fontweight="bold", pad=8)

    out = OUT / "fig_price_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅  {out.name}")


# ─── FIGURE 2: Load overview (3 panels) ──────────────────────────────────────
def fig_load_overview():
    print("Generating fig_load_overview...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.subplots_adjust(hspace=0.42)

    # ── Panel α ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(load_s.index, load_s.values,
            lw=0.3, alpha=0.4, color="#27AE60", zorder=2, rasterized=True)
    weekly_ma = load_s.resample("W").mean()
    ax.plot(weekly_ma.index, weekly_ma.values,
            lw=1.8, color="#145A32", label="Εβδομαδιαίος μέσος", zorder=3)
    ax.set_ylabel("Φορτίο (MW)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "(α)  Ωριαίο Ηλεκτρικό Φορτίο Ελλάδας — Ιαν. 2017 έως Μαρ. 2026",
        fontsize=11, fontweight="bold", pad=8
    )

    # ── Panel β ──────────────────────────────────────────────────────────────
    ax = axes[1]
    hp    = load_s.groupby(load_s.index.hour).agg(["mean", "std"])
    hours = np.arange(24)
    ax.fill_between(hours,
                    hp["mean"] - hp["std"],
                    hp["mean"] + hp["std"],
                    alpha=0.20, color="#27AE60", label="±1σ")
    ax.plot(hours, hp["mean"],
            color="#145A32", lw=2.2, marker="o", ms=5, label="Μέση τιμή")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Ώρα ημέρας")
    ax.set_ylabel("Φορτίο (MW)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k")
    )
    ax.set_title("(β)  Μέσο Ωριαίο Προφίλ Φορτίου (2017–2026)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9)

    # ── Panel γ ──────────────────────────────────────────────────────────────
    ax = axes[2]
    data, periods = _monthly_box(load_s, start_year=2019)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.65)
    for patch in bp["boxes"]:
        patch.set(facecolor="#A9DFBF", edgecolor="#145A32", lw=0.8)
    for med in bp["medians"]:
        med.set(color="#E74C3C", lw=2.0)
    for wh in bp["whiskers"]:
        wh.set(color="#145A32", lw=0.7)
    for cap in bp["caps"]:
        cap.set(color="#145A32", lw=0.7)

    tpos, tlbl = [], []
    for i, p in enumerate(periods):
        if p.month in (1, 7):
            tpos.append(i + 1)
            tlbl.append(str(p))
    ax.set_xticks(tpos)
    ax.set_xticklabels(tlbl, rotation=45, fontsize=8, ha="right")
    ax.set_xlabel("Μήνας")
    ax.set_ylabel("Φορτίο (MW)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
    )
    ax.set_title("(γ)  Μηνιαία Κατανομή Φορτίου (2019–2026)",
                 fontsize=11, fontweight="bold", pad=8)

    out = OUT / "fig_load_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅  {out.name}")


# ─── Helper: 24×24 Pearson lag-1-day correlation matrix ──────────────────────
def _corr_matrix_24x24(series: pd.Series) -> np.ndarray:
    """
    Row h, column l  =  Pearson( price[day D, hour h], price[day D-1, hour l] )
    No daily demeaning — keeps values in the natural range (same as Kanousis Fig.7).
    """
    df = series.dropna().to_frame("y")
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    pivot = df.pivot_table(index="date", columns="hour", values="y").dropna()

    pivot_lag = pivot.shift(1)
    common    = pivot.index.intersection(pivot_lag.dropna().index)
    A = pivot.loc[common].values      # (N,24) — day D
    B = pivot_lag.loc[common].values  # (N,24) — day D-1

    C = np.zeros((24, 24))
    for h in range(24):
        for l in range(24):
            x, y = A[:, h], B[:, l]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 10:
                C[h, l] = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return C


# ─── Helper: render heatmap in Kanousis Figure 7 style ───────────────────────
def _render_heatmap(corr_mat: np.ndarray,
                    title: str, xlabel: str, ylabel: str,
                    out_path: Path) -> None:
    """
    Kanousis style:
      • viridis colormap, vmin/vmax = actual data range
      • Numbers annotated in every cell (font ≈ 5.5 pt)
      • All 24 tick labels (0-23) on both axes
      • White separator lines every 4 hours
      • No background grid (imshow overrides it)
    """
    fig, ax = plt.subplots(figsize=(9.5, 8.0))

    vmin, vmax = np.nanmin(corr_mat), np.nanmax(corr_mat)
    im = ax.imshow(corr_mat, cmap="viridis", aspect="auto",
                   vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Annotate every cell with the correlation value
    span = vmax - vmin if vmax > vmin else 1.0
    for h in range(24):
        for l in range(24):
            val = corr_mat[h, l]
            brightness = (val - vmin) / span   # 0 = darkest, 1 = lightest
            txt_color  = "white" if brightness < 0.52 else "black"
            ax.text(l, h, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=5.5, color=txt_color, fontweight="normal",
                    fontfamily="monospace")

    # Tick labels 0-23 on both axes
    ticks = list(range(24))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7.5)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7.5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    # Subtle white separator lines every 4 hours
    for x in range(0, 24, 4):
        ax.axvline(x - 0.5, color="white", lw=0.8, alpha=0.55)
        ax.axhline(x - 0.5, color="white", lw=0.8, alpha=0.55)

    ax.grid(False)   # imshow has its own cell borders

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅  {out_path.name}")


# ─── FIGURE 3: Price correlation heatmap ─────────────────────────────────────
def fig_corr_heatmap_price():
    print("Generating fig_corr_heatmap_price (computing 24×24 matrix)...")
    C = _corr_matrix_24x24(price)
    print(f"  Pearson range: [{C.min():.3f}, {C.max():.3f}]")
    _render_heatmap(
        C,
        title=(
            "Ενδοημερήσια Συσχέτιση Τιμής ΗΕ\n"
            "Ώρα ημέρας D  vs.  ίδια ώρα προηγούμενης ημέρας D−1  (Lag-1d)"
        ),
        xlabel="Ώρα Προηγούμενης Ημέρας  D−1",
        ylabel="Ώρα Τρέχουσας Ημέρας  D",
        out_path=OUT / "fig_corr_heatmap_price.png",
    )


# ─── FIGURE 4: Load correlation heatmap ──────────────────────────────────────
def fig_corr_heatmap_load():
    print("Generating fig_corr_heatmap_load (computing 24×24 matrix)...")
    C = _corr_matrix_24x24(load_s)
    print(f"  Pearson range: [{C.min():.3f}, {C.max():.3f}]")
    _render_heatmap(
        C,
        title=(
            "Ενδοημερήσια Συσχέτιση Φορτίου\n"
            "Ώρα ημέρας D  vs.  ίδια ώρα προηγούμενης ημέρας D−1  (Lag-1d)"
        ),
        xlabel="Ώρα Προηγούμενης Ημέρας  D−1",
        ylabel="Ώρα Τρέχουσας Ημέρας  D",
        out_path=OUT / "fig_corr_heatmap_load.png",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_price_overview()
    fig_load_overview()
    fig_corr_heatmap_price()
    fig_corr_heatmap_load()
    print("\nAll 4 figures saved to:", OUT)
