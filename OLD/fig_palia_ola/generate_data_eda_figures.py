"""
EDA figures for Greek electricity market thesis.
12 publication-quality figures saved to thesis_output/.ssasa
Data period: Jan 2017 – Nov 2025 (training window).
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "thesis_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.titlesize":    16,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.dpi":        200,
    "savefig.dpi":       200,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "axes.axisbelow":    True,
})

C_BLUE   = "#2176ae"
C_ORANGE = "#e8871e"
C_GREEN  = "#3a9e4a"
C_RED    = "#c0392b"
C_PURPLE = "#7b2d8b"
C_GRAY   = "#888888"
DPI      = 200

# ── Data load ─────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
df_price_raw = pd.read_parquet(BASE_DIR / "data/processed/hourly.parquet")
df_load_raw  = pd.read_parquet(BASE_DIR / "data/processed/hourly_load.parquet")

TRAIN_START = "2017-01-01"
TRAIN_END   = "2025-11-30 23:00"

df_p = df_price_raw.loc[TRAIN_START:TRAIN_END].copy()
df_l = df_load_raw.loc[TRAIN_START:TRAIN_END].copy()

price  = df_p["y"].rename("price")          # €/MWh
load   = df_l["y"].rename("load")           # MW
gas    = df_p["gas_price"].rename("gas")    # €/MWh gas
solar  = df_p["solar_fc_dayahead"].rename("solar")   # MW
wind   = df_p["wind_onshore_fc_dayahead"].rename("wind")  # MW

# Helper: remove spines
def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Price time series + rolling mean + crisis shading + monthly boxplots
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Price time series...", flush=True)

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 2]})
fig.subplots_adjust(hspace=0.42)

ax1 = axes[0]
# Full available series for the dataset-overview panel (train + test span)
price_full = df_price_raw["y"].dropna()
roll30d = price_full.rolling(24 * 30, min_periods=1).mean()

ax1.plot(price_full.index, price_full.values, color=C_BLUE, alpha=0.30, linewidth=0.4, label="Hourly price")
ax1.plot(roll30d.index, roll30d.values, color=C_RED, linewidth=1.6, label="30-day rolling mean")

# Energy crisis shading (2021 H2 – end 2022)
ax1.axvspan(pd.Timestamp("2021-09-01"), pd.Timestamp("2023-01-31"),
            color="#ffd700", alpha=0.18, label="Energy crisis")
# COVID-19 lockdowns in Greece spanned > 1 year (Mar 2020 – mid 2021)
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"),
            color="#90ee90", alpha=0.20, label="COVID-19 lockdowns")

_t0, _t1 = price_full.index.min(), price_full.index.max()
ax1.set_title(f"Greek Day-Ahead Electricity Price — Hourly ({_t0:%b %Y} – {_t1:%b %Y})", fontweight="bold")
ax1.set_ylabel("Price (€/MWh)")
ax1.set_xlabel("")
ax1.legend(loc="upper left", framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
_clean(ax1)

# Monthly boxplots
ax2 = axes[1]
monthly_price = price_full.resample("ME").apply(list)
months = monthly_price.index
box_data = [np.array(v) for v in monthly_price.values if len(v) > 0]
box_positions = range(len(box_data))
bp = ax2.boxplot(box_data, positions=list(box_positions), widths=0.7,
                 patch_artist=True, showfliers=False,
                 medianprops=dict(color=C_RED, linewidth=1.5),
                 boxprops=dict(facecolor=C_BLUE, alpha=0.55, linewidth=0.8),
                 whiskerprops=dict(linewidth=0.8),
                 capprops=dict(linewidth=0.8))

# Color boxes by year — energy crisis red
for i, (ts, patch) in enumerate(zip(months, bp["boxes"])):
    if pd.Timestamp("2021-09-01") <= ts <= pd.Timestamp("2022-12-31"):
        patch.set_facecolor(C_RED)
        patch.set_alpha(0.55)

# X-axis: show year labels at Jan boundaries
year_ticks = [i for i, ts in enumerate(months) if ts.month == 1]
year_labels = [str(ts.year) for i, ts in enumerate(months) if ts.month == 1]
ax2.set_xticks(year_ticks)
ax2.set_xticklabels(year_labels, fontsize=9)
ax2.set_ylabel("Price (€/MWh)")
ax2.set_title("Monthly Distribution of Day-Ahead Prices", fontweight="bold")
_clean(ax2)

fig.savefig(OUT_DIR / "fig_price_timeseries.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_price_timeseries.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Load time series + rolling mean + monthly boxplots
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Load time series...", flush=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 2]})
fig.subplots_adjust(hspace=0.42)

ax1 = axes[0]
roll30d_l = load.rolling(24 * 30, min_periods=1).mean()

ax1.plot(load.index, load.values, color=C_ORANGE, alpha=0.30, linewidth=0.4, label="Hourly load")
ax1.plot(roll30d_l.index, roll30d_l.values, color=C_RED, linewidth=1.6, label="30-day rolling mean")

ax1.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-08-31"),
            color="#90ee90", alpha=0.20, label="COVID-19 lockdowns")

ax1.set_title("Greek Electricity System Load — Hourly (Jan 2017 – Nov 2025)", fontweight="bold")
ax1.set_ylabel("Load (MW)")
ax1.set_xlabel("")
ax1.legend(loc="upper left", framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
_clean(ax1)

# Monthly boxplots
ax2 = axes[1]
# align load to same period as price for consistency
load_aligned = load.reindex(
    pd.date_range(TRAIN_START, TRAIN_END, freq="h"), fill_value=np.nan
)
monthly_load = load_aligned.resample("ME").apply(
    lambda s: s.dropna().tolist()
)
months_l = monthly_load.index
box_data_l = [np.array(v) for v in monthly_load.values if len(v) > 0]
bp2 = ax2.boxplot(box_data_l, positions=list(range(len(box_data_l))), widths=0.7,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color=C_RED, linewidth=1.5),
                  boxprops=dict(facecolor=C_ORANGE, alpha=0.55, linewidth=0.8),
                  whiskerprops=dict(linewidth=0.8),
                  capprops=dict(linewidth=0.8))

year_ticks_l = [i for i, ts in enumerate(months_l) if ts.month == 1]
year_labels_l = [str(ts.year) for i, ts in enumerate(months_l) if ts.month == 1]
ax2.set_xticks(year_ticks_l)
ax2.set_xticklabels(year_labels_l, fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.set_ylabel("Load (MW)")
ax2.set_title("Monthly Distribution of System Load", fontweight="bold")
_clean(ax2)

fig.savefig(OUT_DIR / "fig_load_timeseries.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_load_timeseries.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Monthly average by year (price + load side by side)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Monthly avg by year...", flush=True)

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
years = range(2017, 2026)  # 2025 only to Nov
cmap = plt.cm.get_cmap("tab10", len(list(years)))

fig, (ax_p, ax_l) = plt.subplots(1, 2, figsize=(15, 6))
fig.subplots_adjust(wspace=0.30)

for idx, yr in enumerate(years):
    yr_price = price[price.index.year == yr]
    monthly_avg_p = yr_price.groupby(yr_price.index.month).mean()
    # only plot months present
    ax_p.plot(monthly_avg_p.index - 1, monthly_avg_p.values,
              marker="o", markersize=4, linewidth=1.6,
              color=cmap(idx), label=str(yr))

    yr_load = load_aligned[load_aligned.index.year == yr]
    monthly_avg_l = yr_load.groupby(yr_load.index.month).mean()
    ax_l.plot(monthly_avg_l.index - 1, monthly_avg_l.values,
              marker="o", markersize=4, linewidth=1.6,
              color=cmap(idx), label=str(yr))

for ax, title, unit in [
    (ax_p, "Average Monthly Price by Year", "€/MWh"),
    (ax_l, "Average Monthly Load by Year",  "MW"),
]:
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(unit)
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.85)
    _clean(ax)

fig.suptitle("Seasonal Patterns: Day-Ahead Price and System Load\n(Greek Electricity Market, 2017–2025)",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_monthly_avg_by_year.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_monthly_avg_by_year.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Daily & weekly profile: Price
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Daily/weekly profile (price)...", flush=True)

dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.30)

# Left: hourly profile by day-of-week
ax = axes[0]
for d in range(7):
    sub = price[price.index.dayofweek == d]
    profile = sub.groupby(sub.index.hour).mean()
    color = C_BLUE if d < 5 else C_ORANGE
    alpha = 0.85 if d < 5 else 1.0
    lw    = 1.2  if d < 5 else 2.0
    ax.plot(profile.index, profile.values, color=color, alpha=alpha,
            linewidth=lw, label=dow_names[d])
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean price (€/MWh)")
ax.set_title("Intraday Price Profile by Day-of-Week", fontweight="bold")
ax.legend(fontsize=8, ncol=2, framealpha=0.9)
_clean(ax)

# Right: weekly heatmap — average by (dow, hour)
ax2 = axes[1]
pivot_p = price.copy().to_frame()
pivot_p["hour"] = pivot_p.index.hour
pivot_p["dow"]  = pivot_p.index.dayofweek
heat_p = pivot_p.groupby(["dow","hour"])["price"].mean().unstack()
im = ax2.imshow(heat_p.values, aspect="auto", cmap="RdYlBu_r",
                interpolation="nearest")
ax2.set_xticks(range(0, 24, 3))
ax2.set_xticklabels(range(0, 24, 3), fontsize=8)
ax2.set_yticks(range(7))
ax2.set_yticklabels(dow_names, fontsize=8)
ax2.set_xlabel("Hour of day")
ax2.set_title("Mean Price Heatmap (DOW × Hour)", fontweight="bold")
fig.colorbar(im, ax=ax2, label="€/MWh", shrink=0.85)
ax2.grid(False)

fig.suptitle("Day-Ahead Price: Intraday and Weekly Patterns", fontsize=13,
             fontweight="bold", y=1.02)
fig.savefig(OUT_DIR / "fig_daily_weekly_profile.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_daily_weekly_profile.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Daily & weekly profile: Load  (3-panel: hourly avg, DOW avg, heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Daily/weekly profile (load)...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.subplots_adjust(wspace=0.33)

# 5a: Hourly avg load by season
seasons = {
    "Winter (Dec–Feb)": [12, 1, 2],
    "Spring (Mar–May)": [3, 4, 5],
    "Summer (Jun–Aug)": [6, 7, 8],
    "Autumn (Sep–Nov)": [9, 10, 11],
}
season_colors = [C_BLUE, C_GREEN, C_RED, C_ORANGE]
ax = axes[0]
for (sname, months), sc in zip(seasons.items(), season_colors):
    sub = load_aligned[load_aligned.index.month.isin(months)]
    profile = sub.groupby(sub.index.hour).mean()
    ax.plot(profile.index, profile.values, color=sc, linewidth=2.0,
            label=sname)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean load (MW)")
ax.set_title("Intraday Load Profile by Season", fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
_clean(ax)

# 5b: Hourly avg by day-of-week
ax2 = axes[1]
for d in range(7):
    sub = load_aligned[load_aligned.index.dayofweek == d]
    profile = sub.groupby(sub.index.hour).mean()
    color = C_ORANGE if d >= 5 else C_BLUE
    lw    = 2.0 if d >= 5 else 1.2
    ax2.plot(profile.index, profile.values, color=color,
             linewidth=lw, alpha=0.85, label=dow_names[d])
ax2.set_xticks(range(0, 24, 3))
ax2.set_xlabel("Hour of day")
ax2.set_ylabel("Mean load (MW)")
ax2.set_title("Intraday Load Profile by Day-of-Week", fontweight="bold")
ax2.legend(fontsize=8, ncol=2, framealpha=0.9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
_clean(ax2)

# 5c: Heatmap DOW × hour
ax3 = axes[2]
pivot_l = load_aligned.to_frame(name="load")
pivot_l["hour"] = pivot_l.index.hour
pivot_l["dow"]  = pivot_l.index.dayofweek
heat_l = pivot_l.groupby(["dow","hour"])["load"].mean().unstack()
im3 = ax3.imshow(heat_l.values, aspect="auto", cmap="YlOrRd",
                 interpolation="nearest")
ax3.set_xticks(range(0, 24, 3))
ax3.set_xticklabels(range(0, 24, 3), fontsize=8)
ax3.set_yticks(range(7))
ax3.set_yticklabels(dow_names, fontsize=8)
ax3.set_xlabel("Hour of day")
ax3.set_title("Mean Load Heatmap (DOW × Hour)", fontweight="bold")
fig.colorbar(im3, ax=ax3, label="MW", shrink=0.85,
             format=mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax3.grid(False)

fig.suptitle("System Load: Seasonal, Intraday and Weekly Patterns", fontsize=13,
             fontweight="bold", y=1.02)
fig.savefig(OUT_DIR / "fig_daily_weekly_profile_load.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_daily_weekly_profile_load.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 — ACF / PACF: Price  (lags up to 168 = 1 week)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6: ACF/PACF price...", flush=True)

MAX_LAGS = 168
price_clean = price.dropna()

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.subplots_adjust(wspace=0.18)

plot_acf(price_clean.values, lags=MAX_LAGS, alpha=0.05,
         ax=axes[0], color=C_BLUE, vlines_kwargs={"colors": C_BLUE})
axes[0].set_title("Autocorrelation Function (ACF) — Day-Ahead Price", fontweight="bold")
axes[0].set_xlabel("Lag (hours)")
axes[0].set_ylabel("ACF")
# annotate key seasonal lags
for lag, note in [(24, "24h"), (48, "48h"), (168, "168h\n(1 week)")]:
    axes[0].axvline(lag, color=C_RED, linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0].text(lag + 1.5, axes[0].get_ylim()[1] * 0.92,
                 note, color=C_RED, fontsize=8)
_clean(axes[0])

plot_pacf(price_clean.values, lags=MAX_LAGS, alpha=0.05,
          method="ywm", ax=axes[1], color=C_ORANGE,
          vlines_kwargs={"colors": C_ORANGE})
axes[1].set_title("Partial Autocorrelation Function (PACF) — Day-Ahead Price", fontweight="bold")
axes[1].set_xlabel("Lag (hours)")
axes[1].set_ylabel("PACF")
for lag, note in [(24, "24h"), (48, "48h"), (168, "168h\n(1 week)")]:
    axes[1].axvline(lag, color=C_RED, linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].text(lag + 1.5, axes[1].get_ylim()[1] * 0.92,
                 note, color=C_RED, fontsize=8)
_clean(axes[1])

fig.suptitle("Autocorrelation Analysis: Day-Ahead Electricity Price", fontsize=13,
             fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_acf_pacf_price.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_acf_pacf_price.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 7 — ACF / PACF: Load
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7: ACF/PACF load...", flush=True)

load_clean = load_aligned.dropna()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.subplots_adjust(hspace=0.45)

plot_acf(load_clean.values, lags=MAX_LAGS, alpha=0.05,
         ax=axes[0], color=C_ORANGE, vlines_kwargs={"colors": C_ORANGE})
axes[0].set_title("Autocorrelation Function (ACF) — System Load", fontweight="bold")
axes[0].set_xlabel("Lag (hours)")
axes[0].set_ylabel("ACF")
for lag, note in [(24, "24h"), (48, "48h"), (168, "168h\n(1 week)")]:
    axes[0].axvline(lag, color=C_RED, linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0].text(lag + 1.5, axes[0].get_ylim()[1] * 0.92,
                 note, color=C_RED, fontsize=8)
_clean(axes[0])

plot_pacf(load_clean.values, lags=MAX_LAGS, alpha=0.05,
          method="ywm", ax=axes[1], color=C_BLUE,
          vlines_kwargs={"colors": C_BLUE})
axes[1].set_title("Partial Autocorrelation Function (PACF) — System Load", fontweight="bold")
axes[1].set_xlabel("Lag (hours)")
axes[1].set_ylabel("PACF")
for lag, note in [(24, "24h"), (48, "48h"), (168, "168h\n(1 week)")]:
    axes[1].axvline(lag, color=C_RED, linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].text(lag + 1.5, axes[1].get_ylim()[1] * 0.92,
                 note, color=C_RED, fontsize=8)
_clean(axes[1])

fig.suptitle("Autocorrelation Analysis: System Load", fontsize=13,
             fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_acf_pacf_load.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_acf_pacf_load.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Hour-by-hour correlation heatmap: Price
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 8: Correlation heatmap price...", flush=True)

price_pivot = price.copy()
price_pivot = price_pivot.to_frame()
price_pivot["date"] = price_pivot.index.date
price_pivot["hour"] = price_pivot.index.hour
price_daily = price_pivot.pivot_table(index="date", columns="hour", values="price")
price_daily = price_daily.dropna()

corr_p = price_daily.corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_p.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
cbar = fig.colorbar(im, ax=ax, shrink=0.82, label="Pearson correlation")
ax.set_xticks(range(24))
ax.set_yticks(range(24))
ax.set_xticklabels(range(24), fontsize=7)
ax.set_yticklabels(range(24), fontsize=7)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Hour of day")
ax.set_title("Hour-by-Hour Price Correlation Heatmap\n(Pearson, daily pivoted, 2017–2025)",
             fontweight="bold")
ax.grid(False)

# Annotate diagonal values
for i in range(24):
    for j in range(24):
        val = corr_p.iloc[i, j]
        if abs(val) > 0.85 and i != j:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color="white" if abs(val) > 0.90 else "black")

fig.savefig(OUT_DIR / "fig_corr_heatmap_price.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_corr_heatmap_price.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Hour-by-hour correlation heatmap: Load
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 9: Correlation heatmap load...", flush=True)

load_pivot = load_aligned.to_frame(name="load")
load_pivot["date"] = load_pivot.index.date
load_pivot["hour"] = load_pivot.index.hour
load_daily = load_pivot.pivot_table(index="date", columns="hour", values="load")
load_daily = load_daily.dropna()
corr_l = load_daily.corr()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_l.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
cbar = fig.colorbar(im, ax=ax, shrink=0.82, label="Pearson correlation")
ax.set_xticks(range(24))
ax.set_yticks(range(24))
ax.set_xticklabels(range(24), fontsize=7)
ax.set_yticklabels(range(24), fontsize=7)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Hour of day")
ax.set_title("Hour-by-Hour Load Correlation Heatmap\n(Pearson, daily pivoted, 2017–2025)",
             fontweight="bold")
ax.grid(False)

fig.savefig(OUT_DIR / "fig_corr_heatmap_load.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_corr_heatmap_load.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Weekly overlay: Price + Load + RES (solar + wind)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 10: Weekly overlay...", flush=True)

# Pick a representative summer week (July 2023) and a winter week (Jan 2024)
week_specs = [
    ("2023-07-03", "2023-07-09", "Summer week — July 2023"),
    ("2024-01-08", "2024-01-14", "Winter week — January 2024"),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.subplots_adjust(hspace=0.45)

for ax, (ws, we, wtitle) in zip(axes, week_specs):
    wp = price.loc[ws:we]
    wl = load_aligned.loc[ws:we]
    ws_fc = solar.loc[ws:we]
    ww_fc = wind.loc[ws:we]
    res   = (ws_fc + ww_fc).clip(lower=0)

    # Stacked-area RES fill under secondary axis
    ax2 = ax.twinx()
    ax2.fill_between(ws_fc.index, ws_fc.values, alpha=0.25, color="#f4c542",
                     label="Solar FC")
    ax2.fill_between(ww_fc.index, ws_fc.values, (ws_fc + ww_fc).values,
                     alpha=0.25, color="#5abaff", label="Wind FC")
    ax2.set_ylabel("Generation forecast (MW)", color=C_GRAY, fontsize=9)
    ax2.tick_params(axis="y", colors=C_GRAY, labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max((ws_fc + ww_fc).max() * 2.5, 1))

    # Load on left axis
    ax.plot(wl.index, wl.values, color=C_ORANGE, linewidth=2.0, label="Load")
    # Price on left axis (scaled)
    ax_p2 = ax.twinx()
    ax_p2.spines["right"].set_position(("outward", 60))
    ax_p2.plot(wp.index, wp.values, color=C_RED, linewidth=1.6,
               linestyle="--", label="Price")
    ax_p2.set_ylabel("Price (€/MWh)", color=C_RED, fontsize=9)
    ax_p2.tick_params(axis="y", colors=C_RED, labelsize=8)
    ax_p2.spines["top"].set_visible(False)

    ax.set_ylabel("Load (MW)", color=C_ORANGE, fontsize=9)
    ax.tick_params(axis="y", colors=C_ORANGE, labelsize=8)
    ax.set_title(wtitle, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shared legend
    handles = [
        Line2D([0],[0], color=C_ORANGE, lw=2,   label="Load (MW)"),
        Line2D([0],[0], color=C_RED,    lw=1.6, ls="--", label="Price (€/MWh)"),
        mpatches.Patch(color="#f4c542", alpha=0.5, label="Solar FC (MW)"),
        mpatches.Patch(color="#5abaff", alpha=0.5, label="Wind FC (MW)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.88)

    # Shade weekends
    ts = pd.date_range(ws, we, freq="h")
    day_starts = ts[ts.hour == 0]
    for ds in day_starts:
        if ds.dayofweek >= 5:
            ax.axvspan(ds, ds + pd.Timedelta(hours=23),
                       color="#cccccc", alpha=0.25)

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d/%m"))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

fig.suptitle("Representative Weekly Snapshot: Price, Load, and RES Generation Forecasts",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_weekly_overlay.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_weekly_overlay.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Price vs Gas price scatter (coloured by year)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 11: Price vs gas scatter...", flush=True)

gas_clean = gas.reindex(price.index)
combined  = pd.DataFrame({"price": price, "gas": gas_clean}).dropna()
# Use daily averages for cleaner scatter
daily = combined.resample("D").mean().dropna()

fig, (ax_sc, ax_ts) = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.30)

years_sc = daily.index.year.unique()
cmap_sc  = plt.cm.get_cmap("plasma", len(years_sc))
for i, yr in enumerate(sorted(years_sc)):
    sub = daily[daily.index.year == yr]
    ax_sc.scatter(sub["gas"], sub["price"], color=cmap_sc(i), s=12, alpha=0.6,
                  label=str(yr))

# Fit line over all data
m, b = np.polyfit(daily["gas"], daily["price"], 1)
xfit = np.linspace(daily["gas"].min(), daily["gas"].max(), 200)
ax_sc.plot(xfit, m * xfit + b, color=C_RED, linewidth=2.0,
           label=f"OLS fit (slope={m:.2f})")
ax_sc.set_xlabel("Gas price (€/MWh)")
ax_sc.set_ylabel("Day-ahead electricity price (€/MWh)")
ax_sc.set_title("Electricity Price vs Natural Gas Price\n(daily averages)", fontweight="bold")
ax_sc.legend(fontsize=8, ncol=2, framealpha=0.88)
_clean(ax_sc)

# Right panel: time series of both (dual axis)
ax_t2 = ax_ts.twinx()
ax_ts.plot(daily.index, daily["price"], color=C_BLUE, linewidth=0.8,
           alpha=0.85, label="Electricity price")
ax_t2.plot(daily.index, daily["gas"], color=C_ORANGE, linewidth=0.8,
           alpha=0.85, label="Gas price")
ax_ts.set_ylabel("Electricity price (€/MWh)")
ax_t2.set_ylabel("Gas price (€/MWh)")
ax_ts.tick_params(axis="y")
ax_t2.tick_params(axis="y")
ax_ts.set_title("Day-Ahead Price & Gas Price Over Time", fontweight="bold")
ax_ts.spines["top"].set_visible(False)
ax_t2.spines["top"].set_visible(False)
handles2 = [
    Line2D([0],[0], color=C_BLUE,   lw=1.5, label="Electricity price"),
    Line2D([0],[0], color=C_ORANGE, lw=1.5, label="Gas price"),
]
ax_ts.legend(handles=handles2, loc="upper left", fontsize=9, framealpha=0.9)

fig.suptitle("Electricity–Gas Price Relationship (Greek Market, 2017–2025)",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_price_vs_gas.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_price_vs_gas.png", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 12 — Train / test split visualisation
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 12: Train/test split...", flush=True)

fig, (ax_p3, ax_l3) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.subplots_adjust(hspace=0.30)

# Daily averages for cleaner display
price_daily_avg = price.resample("D").mean()
load_daily_avg  = load_aligned.resample("D").mean()

CUTOFF = pd.Timestamp("2025-11-30")
TEST_S = pd.Timestamp("2025-12-01")
TEST_E = pd.Timestamp("2026-02-28")

# Price
ax_p3.fill_between(price_daily_avg.index,
                   price_daily_avg.values,
                   where=(price_daily_avg.index <= CUTOFF),
                   color=C_BLUE, alpha=0.45, label="Training (Jan 2017 – Nov 2025)")
# Extend data to include test period from raw
price_te = df_price_raw["y"].loc[TEST_S:TEST_E].resample("D").mean()
ax_p3.fill_between(price_te.index, price_te.values, color=C_RED, alpha=0.5,
                   label="Test period (Dec 2025 – Feb 2026)")
ax_p3.axvline(TEST_S, color=C_RED, linewidth=1.8, linestyle="--")
ax_p3.text(TEST_S + pd.Timedelta(days=4), ax_p3.get_ylim()[1] * 0.88,
           "Test start\nDec 2025", color=C_RED, fontsize=9)
ax_p3.set_ylabel("Price (€/MWh)")
ax_p3.set_title("Day-Ahead Price — Training / Test Split", fontweight="bold")
ax_p3.legend(loc="upper left", fontsize=9, framealpha=0.9)
_clean(ax_p3)

# Load
ax_l3.fill_between(load_daily_avg.index,
                   load_daily_avg.values,
                   where=(load_daily_avg.index <= CUTOFF),
                   color=C_ORANGE, alpha=0.45, label="Training (Jan 2017 – Nov 2025)")
load_te = df_load_raw["y"].loc[TEST_S:TEST_E].resample("D").mean()
ax_l3.fill_between(load_te.index, load_te.values, color=C_RED, alpha=0.5,
                   label="Test period (Dec 2025 – Feb 2026)")
ax_l3.axvline(TEST_S, color=C_RED, linewidth=1.8, linestyle="--")
ax_l3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax_l3.set_ylabel("Load (MW)")
ax_l3.set_title("System Load — Training / Test Split", fontweight="bold")
ax_l3.legend(loc="upper left", fontsize=9, framealpha=0.9)
_clean(ax_l3)

# Annotation: training size
n_tr_h = len(price)
n_te_h = len(price_te) * 24  # approximate
ax_p3.text(pd.Timestamp("2018-01-01"), ax_p3.get_ylim()[1] * 0.78,
           f"Training set: {n_tr_h:,} hourly observations\n({n_tr_h//24:,} days)",
           fontsize=9, color=C_BLUE, alpha=0.9)

fig.suptitle("Dataset Split: Training and Evaluation Periods\n(Greek Electricity Market Forecasting Study)",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_train_test_split.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_train_test_split.png", flush=True)
print('All EDA figures done.', flush=True)
