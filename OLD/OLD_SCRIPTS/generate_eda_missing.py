"""
Generate the 4 missing EDA figures for the thesis:ASDDDDDDDADSDSAADSDSADSAADSADSADSADSADADSD
  1. fig_acf_pacf_price.png   — ACF + PACF for price (side-by-side)
  2. fig_weekly_overlay.png   — 1 week: price + load + solar + wind
  3. fig_price_vs_gas.png     — price vs gas scatter (colored by year) + dual-axis TS
  4. fig_train_test_split.png — train/test timeline for price and load

Saves to thesis_output/. Does NOT overwrite other existing figures.
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
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

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
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
C_SOLAR  = "#f4c542"
C_WIND   = "#5abaff"

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
df_p = pd.read_parquet(BASE_DIR / "data/processed/hourly.parquet")
df_l = pd.read_parquet(BASE_DIR / "data/processed/hourly_load.parquet")

TRAIN_START = "2017-01-01"
TRAIN_END   = "2025-11-30 23:00"

price = df_p["y"].loc[TRAIN_START:TRAIN_END].rename("price")
load  = df_l["y"].reindex(
    pd.date_range(TRAIN_START, TRAIN_END, freq="h"), fill_value=np.nan
).rename("load")

# RES forecasts (from price parquet — available for training period)
solar = df_p["solar_fc_dayahead"].loc[TRAIN_START:TRAIN_END].rename("solar")
wind  = df_p["wind_onshore_fc_dayahead"].loc[TRAIN_START:TRAIN_END].rename("wind")
gas   = df_p["gas_price"].loc[TRAIN_START:TRAIN_END].rename("gas")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — ACF + PACF for Price  (side-by-side, lags=168)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: ACF/PACF price...", flush=True)

MAX_LAGS = 168
price_clean = price.dropna().values

fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.32)

plot_acf(price_clean, lags=MAX_LAGS, alpha=0.05,
         ax=ax_acf, color=C_BLUE,
         vlines_kwargs={"colors": C_BLUE, "linewidth": 0.8})
ax_acf.set_title("Autocorrelation Function (ACF)", fontweight="bold")
ax_acf.set_xlabel("Lag (ώρες)")
ax_acf.set_ylabel("ACF")
for lag, note in [(24,"24h"), (48,"48h"), (168,"168h\n(1 εβδ.)")]:
    ax_acf.axvline(lag, color=C_RED, linestyle="--", linewidth=1.0, alpha=0.75)
    ax_acf.text(lag + 2, ax_acf.get_ylim()[1] * 0.88,
                note, color=C_RED, fontsize=8, va="top")
_clean(ax_acf)

plot_pacf(price_clean, lags=MAX_LAGS, alpha=0.05,
          method="ywm", ax=ax_pacf, color=C_ORANGE,
          vlines_kwargs={"colors": C_ORANGE, "linewidth": 0.8})
ax_pacf.set_title("Μερική Αυτοσυσχέτιση (PACF)", fontweight="bold")
ax_pacf.set_xlabel("Lag (ώρες)")
ax_pacf.set_ylabel("PACF")
for lag, note in [(24,"24h"), (48,"48h"), (168,"168h")]:
    ax_pacf.axvline(lag, color=C_RED, linestyle="--", linewidth=1.0, alpha=0.75)
    ax_pacf.text(lag + 2, ax_pacf.get_ylim()[1] * 0.88,
                 note, color=C_RED, fontsize=8, va="top")
_clean(ax_pacf)

fig.suptitle("Ανάλυση Αυτοσυσχέτισης — Τιμή ΗΕ Αγοράς Επόμενης Ημέρας",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_acf_pacf_price.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_acf_pacf_price.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Weekly Overlay: Price + Load + Solar + Wind
#          Two representative weeks (summer + winter)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Weekly overlay...", flush=True)

weeks = [
    ("2023-07-03", "2023-07-09", "Καλοκαίρι — Ιούλιος 2023"),
    ("2024-01-08", "2024-01-14", "Χειμώνας — Ιανουάριος 2024"),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.subplots_adjust(hspace=0.48)

for ax_main, (ws, we, wtitle) in zip(axes, weeks):
    wp = price.loc[ws:we]
    wl = load.reindex(pd.date_range(ws, we, freq="h")).dropna()
    ws_fc = solar.loc[ws:we]
    ww_fc = wind.loc[ws:we]

    # RES stacked fill on secondary right axis
    ax_res = ax_main.twinx()
    ax_res.fill_between(ws_fc.index, 0, ws_fc.values,
                        alpha=0.30, color=C_SOLAR, label="Ηλιακή παραγωγή (FC)")
    ax_res.fill_between(ww_fc.index, ws_fc.values, (ws_fc + ww_fc).values,
                        alpha=0.30, color=C_WIND, label="Αιολική παραγωγή (FC)")
    ax_res.set_ylabel("Παραγωγή ΑΠΕ (MW)", color=C_SOLAR, fontsize=9)
    ax_res.tick_params(axis="y", colors="#888888", labelsize=8)
    ax_res.set_ylim(0, max((ws_fc + ww_fc).max() * 2.8, 1))
    ax_res.spines["top"].set_visible(False)

    # Load on main left axis
    ax_main.plot(wl.index, wl.values,
                 color=C_ORANGE, linewidth=2.2, label="Φορτίο (MW)", zorder=3)
    ax_main.set_ylabel("Φορτίο (MW)", color=C_ORANGE, fontsize=9)
    ax_main.tick_params(axis="y", colors=C_ORANGE, labelsize=8)

    # Price on third axis (far right)
    ax_price = ax_main.twinx()
    ax_price.spines["right"].set_position(("outward", 58))
    ax_price.plot(wp.index, wp.values,
                  color=C_RED, linewidth=1.8, linestyle="--",
                  label="Τιμή (€/MWh)", zorder=4)
    ax_price.set_ylabel("Τιμή (€/MWh)", color=C_RED, fontsize=9)
    ax_price.tick_params(axis="y", colors=C_RED, labelsize=8)
    ax_price.spines["top"].set_visible(False)

    ax_main.set_title(wtitle, fontweight="bold")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Shade weekends
    for ds in pd.date_range(ws, we, freq="D"):
        if ds.dayofweek >= 5:
            ax_main.axvspan(ds, ds + pd.Timedelta(hours=23, minutes=59),
                            color="#cccccc", alpha=0.22, zorder=0)

    # Legend
    handles = [
        Line2D([0],[0], color=C_ORANGE, lw=2.2,  label="Φορτίο (MW)"),
        Line2D([0],[0], color=C_RED,    lw=1.8, ls="--", label="Τιμή ΗΕ (€/MWh)"),
        mpatches.Patch(color=C_SOLAR, alpha=0.55, label="Ηλιακή παραγωγή (FC)"),
        mpatches.Patch(color=C_WIND,  alpha=0.55, label="Αιολική παραγωγή (FC)"),
        mpatches.Patch(color="#cccccc", alpha=0.5, label="Σαββατοκύριακο"),
    ]
    ax_main.legend(handles=handles, loc="upper left",
                   fontsize=8, framealpha=0.9, ncol=2)

    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    ax_main.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax_main.xaxis.get_majorticklabels(),
             rotation=25, ha="right", fontsize=8)

fig.suptitle(
    "Εβδομαδιαία Επισκόπηση: Φορτίο, Τιμή ΗΕ και Παραγωγή ΑΠΕ",
    fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_weekly_overlay.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_weekly_overlay.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Price vs Gas:  scatter (colored by year)  +  dual-axis time series
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Price vs gas...", flush=True)

# Daily averages for clarity
daily = pd.DataFrame({"price": price, "gas": gas}).resample("D").mean().dropna()

fig, (ax_sc, ax_ts) = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.32)

# --- Scatter ---
years_u = sorted(daily.index.year.unique())
cmap_sc = plt.cm.get_cmap("plasma", len(years_u))

for i, yr in enumerate(years_u):
    sub = daily[daily.index.year == yr]
    ax_sc.scatter(sub["gas"], sub["price"],
                  color=cmap_sc(i), s=14, alpha=0.60, label=str(yr), zorder=3)

# OLS fit
m, b = np.polyfit(daily["gas"], daily["price"], 1)
xfit = np.linspace(daily["gas"].min(), daily["gas"].max(), 200)
ax_sc.plot(xfit, m * xfit + b, color="black", linewidth=1.8, linestyle="--",
           label=f"OLS (κλίση={m:.2f})", zorder=5)

ax_sc.set_xlabel("Τιμή φυσικού αερίου (€/MWh)")
ax_sc.set_ylabel("Τιμή ΗΕ — ημερήσιος μέσος (€/MWh)")
ax_sc.set_title("Τιμή ΗΕ vs Τιμή Φυσικού Αερίου\n(ημερήσιοι μέσοι, 2017–2025)",
                fontweight="bold")
ax_sc.legend(fontsize=8, ncol=2, framealpha=0.9)
_clean(ax_sc)

# --- Dual-axis time series ---
ax_g = ax_ts.twinx()
ax_ts.plot(daily.index, daily["price"],
           color=C_BLUE, linewidth=0.9, alpha=0.85, label="Τιμή ΗΕ")
ax_g.plot(daily.index, daily["gas"],
          color=C_ORANGE, linewidth=0.9, alpha=0.85, label="Τιμή αερίου")

ax_ts.set_ylabel("Τιμή ΗΕ (€/MWh)", color=C_BLUE)
ax_g.set_ylabel("Τιμή αερίου (€/MWh)", color=C_ORANGE)
ax_ts.tick_params(axis="y", colors=C_BLUE)
ax_g.tick_params(axis="y", colors=C_ORANGE)

# Crisis shading
ax_ts.axvspan(pd.Timestamp("2021-09-01"), pd.Timestamp("2022-12-31"),
              color="#ffe066", alpha=0.22, label="Ενεργειακή κρίση")

ax_ts.set_title("Εξέλιξη Τιμής ΗΕ & Φ. Αερίου στον Χρόνο", fontweight="bold")
ax_ts.spines["top"].set_visible(False)
ax_g.spines["top"].set_visible(False)

handles_ts = [
    Line2D([0],[0], color=C_BLUE,   lw=1.5, label="Τιμή ΗΕ (€/MWh)"),
    Line2D([0],[0], color=C_ORANGE, lw=1.5, label="Τιμή αερίου (€/MWh)"),
    mpatches.Patch(color="#ffe066", alpha=0.5, label="Ενεργειακή κρίση"),
]
ax_ts.legend(handles=handles_ts, loc="upper left", fontsize=8, framealpha=0.9)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_ts.xaxis.set_major_locator(mdates.YearLocator())

fig.suptitle("Σχέση Τιμής Ηλεκτρικής Ενέργειας — Φυσικό Αέριο (2017–2025)",
             fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_price_vs_gas.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_price_vs_gas.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Train / Test Split visualization (price + load)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Train/test split...", flush=True)

TEST_S = pd.Timestamp("2025-12-01")
TEST_E = pd.Timestamp("2026-02-28")

price_daily  = price.resample("D").mean()
load_daily   = load.resample("D").mean()

# Test period from raw (extend beyond training end)
price_te = df_p["y"].loc[TEST_S:TEST_E].resample("D").mean()
load_te  = df_l["y"].loc[TEST_S:TEST_E].resample("D").mean()

fig, (ax_p, ax_l) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.subplots_adjust(hspace=0.38)

# — Price panel —
ax_p.fill_between(price_daily.index, price_daily.values,
                  color=C_BLUE, alpha=0.40, label="Σύνολο εκπαίδευσης\n(Ιαν 2017 – Νοε 2025)")
ax_p.fill_between(price_te.index, price_te.values,
                  color=C_RED, alpha=0.55, label="Περίοδος αξιολόγησης\n(Δεκ 2025 – Φεβ 2026)")
ax_p.axvline(TEST_S, color=C_RED, linewidth=2.0, linestyle="--", zorder=5)
ax_p.text(TEST_S + pd.Timedelta(days=5),
          ax_p.get_ylim()[1] if ax_p.get_ylim()[1] > 0 else price_daily.max() * 0.92,
          "Αρχή\nδοκιμής", color=C_RED, fontsize=9, va="top")

n_tr = len(price)
ax_p.text(pd.Timestamp("2018-06-01"),
          price_daily.max() * 0.80,
          f"Σύνολο εκπαίδευσης: {n_tr:,} ωριαίες παρατηρήσεις\n({n_tr//24:,} ημέρες | {n_tr//24//365:.1f} έτη)",
          fontsize=9, color=C_BLUE, alpha=0.9)

ax_p.set_ylabel("Τιμή ΗΕ (€/MWh)")
ax_p.set_title("Τιμή ΗΕ — Διαχωρισμός Εκπαίδευσης / Αξιολόγησης", fontweight="bold")
ax_p.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_p.xaxis.set_major_locator(mdates.YearLocator())
_clean(ax_p)

# — Load panel —
ax_l.fill_between(load_daily.index, load_daily.values,
                  color=C_ORANGE, alpha=0.45,
                  label="Σύνολο εκπαίδευσης\n(Ιαν 2017 – Νοε 2025)")
ax_l.fill_between(load_te.index, load_te.values,
                  color=C_RED, alpha=0.55,
                  label="Περίοδος αξιολόγησης\n(Δεκ 2025 – Φεβ 2026)")
ax_l.axvline(TEST_S, color=C_RED, linewidth=2.0, linestyle="--", zorder=5)
ax_l.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax_l.set_ylabel("Φορτίο (MW)")
ax_l.set_title("Φορτίο Συστήματος — Διαχωρισμός Εκπαίδευσης / Αξιολόγησης",
               fontweight="bold")
ax_l.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax_l.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_l.xaxis.set_major_locator(mdates.YearLocator())
_clean(ax_l)

fig.suptitle(
    "Διαχωρισμός Δεδομένων: Σύνολο Εκπαίδευσης & Αξιολόγησης\n"
    "(Ελληνική Αγορά Ηλεκτρικής Ενέργειας — Αγορά Επόμενης Ημέρας)",
    fontsize=13, fontweight="bold", y=1.01)
fig.savefig(OUT_DIR / "fig_train_test_split.png", bbox_inches="tight")
plt.close(fig)
print("  Saved fig_train_test_split.png", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55, flush=True)
print("Done! 4 new figures saved to:", OUT_DIR, flush=True)
for f in ["fig_acf_pacf_price.png", "fig_weekly_overlay.png",
          "fig_price_vs_gas.png", "fig_train_test_split.png"]:
    p = OUT_DIR / f
    size_kb = p.stat().st_size // 1024 if p.exists() else -1
    status = f"{size_kb} KB" if size_kb >= 0 else "MISSING"
    print(f"  {f:<35} {status}", flush=True)
print("="*55, flush=True)
