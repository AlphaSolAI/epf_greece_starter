"""
src/generate_data_chapter_v2.py
================================
Regenerates Chapter-3 figures to match the 106.pdf (Kanousis prototype).

Output filenames match what thesis/content/data.tex expects:
  fig_price_timeseries.png         → Σχήμα 3.2
  fig_monthly_avg_by_year.png      → Σχήμα 3.3  (stacked, bigger)
  fig_daily_weekly_profile.png     → Σχήμα 3.4
  fig_daily_weekly_profile_load.png→ Σχήμα 3.5  (bigger)
  fig_acf_pacf_price.png           → Σχήμα 3.6  (all-Greek)
  fig_corr_heatmap_price.png       → Σχήμα 3.7  (Kanousis lag-1d spec)
  fig_corr_heatmap_load.png        → Σχήμα 3.8  (Kanousis lag-1d spec)
  fig_price_vs_gas.png             → Σχήμα 3.9
  fig_weekly_overlay.png           → Σχήμα 3.10
  fig_train_test_split.png         → Σχήμα 3.11

Run:
    conda run -n epf --no-capture-output python -m src.generate_data_chapter_v2
"""

import os, sys, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy import stats

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.labelsize'    : 10,
    'axes.titlesize'    : 11,
    'legend.fontsize'   : 8.5,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'axes.grid'         : True,
    'grid.color'        : '#e8e8e8',
    'grid.linewidth'    : 0.5,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : 'white',
})

_BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(_BASE, 'thesis_output')
os.makedirs(OUT, exist_ok=True)

DOW_EN = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
MON_EN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
DOW7C  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']

def _rm(ax):
    ax.spines[['top','right']].set_visible(False)

def _save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    kb = os.path.getsize(path) // 1024
    print(f"    ✓  {name}  ({kb} KB)")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    _d = os.path.join(_BASE, 'data', 'processed')
    dfp = pd.read_parquet(os.path.join(_d, 'hourly.parquet'))
    dfl = pd.read_parquet(os.path.join(_d, 'hourly_load.parquet'))
    dfp.index = pd.to_datetime(dfp.index)
    dfl.index = pd.to_datetime(dfl.index)
    return dfp, dfl, 'y'


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.2 — fig_price_timeseries.png
#   2 panels: top=hourly fill+rolling+shadings, bottom=price-colored boxplots
# ════════════════════════════════════════════════════════════════════════════
def fig_price_timeseries(dfp, pc):
    s    = dfp[pc].dropna()
    roll = s.rolling(720, center=True, min_periods=24).mean()   # ~30-day

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.5),
                                  gridspec_kw={'hspace': 0.38, 'height_ratios': [1.3, 1]})

    # ── Panel 1 ──────────────────────────────────────────────────────────
    a1.fill_between(s.index, s.values, alpha=0.25, color='#5B9BD5', linewidth=0,
                    label='Hourly price')
    a1.plot(s.index, s.values, color='#5B9BD5', lw=0.25, alpha=0.4)
    a1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-05-31'),
               alpha=0.15, color='#27AE60', label='COVID-19 lockdowns')
    a1.axvspan(pd.Timestamp('2021-07-01'), pd.Timestamp('2022-12-31'),
               alpha=0.18, color='#F39C12', label='Energy crisis')
    a1.plot(roll.index, roll.values, color='#C0392B', lw=1.8, label='30-day rolling mean')
    a1.set_ylabel('Price (€/MWh)')
    a1.set_xlim(s.index.min(), s.index.max())
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    a1.xaxis.set_major_locator(mdates.YearLocator())
    _h, _l = a1.get_legend_handles_labels()
    # legend order: Hourly price, 30-day rolling mean, Energy crisis, COVID-19 lockdowns
    a1.legend([_h[i] for i in [0, 3, 2, 1]], [_l[i] for i in [0, 3, 2, 1]],
              loc='upper left', framealpha=0.9, fontsize=8.5)
    a1.set_title('Greek Day-Ahead Electricity Price — Hourly (Jan 2017 – Nov 2025)',
                 fontsize=11, fontweight='bold', pad=8)
    _rm(a1)

    # ── Panel 2 — monthly boxplots, colored by price level ────────────────
    periods   = s.index.to_period('M').unique()
    monthly   = [s[s.index.to_period('M') == p].values for p in periods]
    months_dt = [p.to_timestamp() for p in periods]

    medians   = [np.median(d) for d in monthly]
    norm      = plt.Normalize(vmin=min(medians), vmax=max(medians))
    box_cmap  = plt.get_cmap('RdYlBu_r')
    colors    = [box_cmap(norm(m)) for m in medians]

    bp = a2.boxplot(monthly, positions=range(len(monthly)), widths=0.65,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#C0392B', lw=1.5),
                    whiskerprops=dict(color='#555555', lw=0.7),
                    capprops=dict(color='#555555', lw=0.7),
                    boxprops=dict(color='#333333', lw=0.6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    yticks = [i for i, dt in enumerate(months_dt) if dt.month == 1]
    ylabs  = [dt.strftime('%Y') for dt in months_dt if dt.month == 1]
    a2.set_xticks(yticks)
    a2.set_xticklabels(ylabs)
    a2.set_ylabel('Price (€/MWh)')
    a2.set_title('Monthly Distribution of Day-Ahead Prices',
                 fontsize=11, fontweight='bold', pad=6)
    _rm(a2)

    _save('fig_price_timeseries.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.3 — fig_monthly_avg_by_year.png
#   SIDE-BY-SIDE: price left, load right  (matching 106.pdf screenshot)
# ════════════════════════════════════════════════════════════════════════════
def fig_monthly_avg_by_year(dfp, dfl, pc):
    tcmap = plt.get_cmap('tab10')
    years = sorted(dfp.index.year.unique())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Seasonal Patterns: Day-Ahead Price and System Load\n'
                 '(Greek Electricity Market, 2017–2025)',
                 fontsize=12, fontweight='bold', y=1.02)

    # ── Price (left) ──────────────────────────────────────────────────────
    for i, yr in enumerate(years):
        sub = dfp[pc][dfp.index.year == yr]
        mm  = sub.groupby(sub.index.month).mean()
        lw  = 2.5 if yr in (2021, 2022) else 1.2
        a1.plot(mm.index, mm.values, color=tcmap(i % 10), lw=lw,
                marker='o', ms=3.5, label=str(yr))
    a1.set_xticks(range(1, 13))
    a1.set_xticklabels(MON_EN)
    a1.set_ylabel('€/MWh')
    a1.set_title('Average Monthly Price by Year', fontsize=11, fontweight='bold')
    a1.legend(ncol=2, fontsize=8, loc='upper left', framealpha=0.9)
    _rm(a1)

    # ── Load (right) ──────────────────────────────────────────────────────
    for i, yr in enumerate(years):
        sub = dfl['y'][dfl.index.year == yr]
        mm  = sub.groupby(sub.index.month).mean()
        lw  = 1.5 if yr in (2021, 2022) else 1.1
        a2.plot(mm.index, mm.values, color=tcmap(i % 10), lw=lw,
                marker='o', ms=3.5, label=str(yr))
    a2.set_xticks(range(1, 13))
    a2.set_xticklabels(MON_EN)
    a2.set_ylabel('MW')
    a2.set_title('Average Monthly Load by Year', fontsize=11, fontweight='bold')
    a2.legend(ncol=2, fontsize=8, loc='upper right', framealpha=0.9)
    _rm(a2)

    plt.tight_layout()
    _save('fig_monthly_avg_by_year.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.4 — fig_daily_weekly_profile.png
#   Weekdays blue (Mon–Fri), weekend orange (Sat–Sun) + DOW×Hour heatmap
# ════════════════════════════════════════════════════════════════════════════
# Mon–Fri: shades of blue; Sat–Sun: orange
DOW_PROFILE_COLORS = [
    '#0d47a1',   # Mon
    '#1565c0',   # Tue
    '#1e88e5',   # Wed
    '#42a5f5',   # Thu
    '#90caf9',   # Fri
    '#e65100',   # Sat
    '#ff8f00',   # Sun
]

def fig_daily_weekly_profile(dfp, pc):
    s = dfp[pc].dropna()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Day-Ahead Price: Intraday and Weekly Patterns',
                 fontsize=12, fontweight='bold', y=1.01)

    # ── Left: DOW profiles (weekdays blue, weekend orange) ────────────────
    for d in range(7):
        sub = s[s.index.dayofweek == d]
        hm  = sub.groupby(sub.index.hour).mean()
        hs  = sub.groupby(sub.index.hour).std()
        hrs = hm.index.tolist()
        clr = DOW_PROFILE_COLORS[d]
        a1.fill_between(hrs, hm - hs, hm + hs, alpha=0.05, color=clr)
        a1.plot(hrs, hm.values, color=clr, lw=1.8, label=DOW_EN[d])
    a1.set_xlabel('Hour of day')
    a1.set_ylabel('Mean price (€/MWh)')
    a1.set_xticks(range(0, 24, 3))
    a1.set_title('Intraday Price Profile by Day-of-Week', fontsize=10.5)
    a1.legend(fontsize=8, ncol=2, framealpha=0.9)
    _rm(a1)

    # ── Right: DOW × Hour heatmap (RdBu_r: blue=low, red=high) ───────────
    heat = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            vals = s[(s.index.dayofweek == d) & (s.index.hour == h)]
            heat[d, h] = vals.mean() if len(vals) > 0 else np.nan
    im = a2.imshow(heat, aspect='auto', cmap='RdBu_r',
                   vmin=np.nanmin(heat), vmax=np.nanmax(heat), origin='upper')
    plt.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    a2.set_xticks(range(0, 24, 3))
    a2.set_xticklabels([str(h) for h in range(0, 24, 3)])
    a2.set_yticks(range(7))
    a2.set_yticklabels(DOW_EN)
    a2.set_xlabel('Hour of day')
    a2.set_title('Mean Price Heatmap (DOW × Hour)', fontsize=10.5)

    plt.tight_layout()
    _save('fig_daily_weekly_profile.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.5 — fig_daily_weekly_profile_load.png  (BIGGER — 3 panels)
# ════════════════════════════════════════════════════════════════════════════
def fig_daily_weekly_profile_load(dfl):
    s = dfl['y'].dropna()
    seasons = {
        'Winter (Dec–Feb)': [12, 1, 2],
        'Spring (Mar–May)': [3, 4, 5],
        'Summer (Jun–Aug)': [6, 7, 8],
        'Autumn (Sep–Nov)': [9, 10, 11],
    }
    scols = ['#3498DB', '#27AE60', '#E74C3C', '#E67E22']

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle('System Load: Seasonal, Intraday and Weekly Patterns',
                 fontsize=13, fontweight='bold', y=1.02)

    # ── Left: seasonal ────────────────────────────────────────────────────
    for (sn, sm), sc in zip(seasons.items(), scols):
        sub = s[s.index.month.isin(sm)]
        hm  = sub.groupby(sub.index.hour).mean()
        hs  = sub.groupby(sub.index.hour).std()
        hrs = hm.index.tolist()
        a1.fill_between(hrs, hm - hs, hm + hs, alpha=0.15, color=sc)
        a1.plot(hrs, hm.values, color=sc, lw=2.2, label=sn)
    a1.set_xlabel('Hour of day')
    a1.set_ylabel('Mean load (MW)')
    a1.set_xticks(range(0, 24, 3))
    a1.set_title('Intraday Load Profile by Season', fontsize=11)
    a1.legend(fontsize=8, framealpha=0.9)
    _rm(a1)

    # ── Middle: DOW ───────────────────────────────────────────────────────
    for d in range(7):
        sub = s[s.index.dayofweek == d]
        hm  = sub.groupby(sub.index.hour).mean()
        hs  = sub.groupby(sub.index.hour).std()
        hrs = hm.index.tolist()
        a2.fill_between(hrs, hm - hs, hm + hs, alpha=0.08, color=DOW_PROFILE_COLORS[d])
        a2.plot(hrs, hm.values, color=DOW_PROFILE_COLORS[d], lw=1.8, label=DOW_EN[d])
    a2.set_xlabel('Hour of day')
    a2.set_ylabel('Mean load (MW)')
    a2.set_xticks(range(0, 24, 3))
    a2.set_title('Intraday Load Profile by Day-of-Week', fontsize=11)
    a2.legend(fontsize=8, ncol=2, framealpha=0.9)
    _rm(a2)

    # ── Right: DOW × Hour heatmap ─────────────────────────────────────────
    heat = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            vals = s[(s.index.dayofweek == d) & (s.index.hour == h)]
            heat[d, h] = vals.mean() if len(vals) > 0 else np.nan
    im = a3.imshow(heat, aspect='auto', cmap='YlOrRd',
                   vmin=np.nanmin(heat), vmax=np.nanmax(heat), origin='upper')
    cb = plt.colorbar(im, ax=a3, fraction=0.046, pad=0.04)
    cb.set_label('MW', fontsize=9.5)
    a3.set_xticks(range(0, 24, 3))
    a3.set_xticklabels([str(h) for h in range(0, 24, 3)])
    a3.set_yticks(range(7))
    a3.set_yticklabels(DOW_EN)
    a3.set_xlabel('Hour of day')
    a3.set_title('Mean Load Heatmap (DOW × Hour)', fontsize=11)

    plt.tight_layout()
    _save('fig_daily_weekly_profile_load.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.6 — fig_acf_pacf_price.png  (ALL GREEK, consistent)
# ════════════════════════════════════════════════════════════════════════════
def fig_acf_pacf_price(dfp, pc):
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        print("    ✗ statsmodels missing — skip"); return

    s = dfp[pc].dropna()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(s), size=min(8000, len(s)), replace=False)
    s_sub = s.iloc[sorted(idx)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle('Ανάλυση Αυτοσυσχέτισης — Τιμή ΗΕ Αγοράς Επόμενης Ημέρας',
                 fontsize=12, fontweight='bold')

    # ACF  (English title matching 106.pdf screenshot)
    plot_acf(s, lags=175, ax=a1, alpha=0.05, color='#2176AE',
             vlines_kwargs={'colors': '#2176AE'}, use_vlines=True)
    a1.set_title('Autocorrelation Function (ACF)', fontsize=10.5)
    a1.set_xlabel('Lag (ώρες)')
    a1.set_ylabel('ACF')
    for lag, lbl in [(24,'24h'), (48,'48h'), (168,'168h')]:
        a1.axvline(lag, color='#E8871E', lw=1.0, ls='--', alpha=0.8)
        a1.text(lag+1, a1.get_ylim()[1]*0.90, lbl, fontsize=7.5,
                color='#E8871E', va='top')
    _rm(a1)

    # PACF
    try:
        plot_pacf(s_sub, lags=175, ax=a2, alpha=0.05, color='#E8871E',
                  method='ywmle', use_vlines=True)
    except Exception:
        plot_pacf(s_sub, lags=60, ax=a2, alpha=0.05, color='#E8871E',
                  method='ywmle', use_vlines=True)
    a2.set_title('Μερική Αυτοσυσχέτιση (PACF)', fontsize=10.5)
    a2.set_xlabel('Lag (ώρες)')
    a2.set_ylabel('PACF')
    for lag, lbl in [(24,'24h'), (48,'48h'), (168,'168h')]:
        if lag <= a2.get_xlim()[1]:
            a2.axvline(lag, color='#E8871E', lw=1.0, ls='--', alpha=0.8)
            a2.text(lag+1, a2.get_ylim()[1]*0.90, lbl, fontsize=7.5,
                    color='#E8871E', va='top')
    _rm(a2)

    plt.tight_layout()
    _save('fig_acf_pacf_price.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.7/3.8 — Kanousis lag-1d cross-correlation heatmaps
# ════════════════════════════════════════════════════════════════════════════
def _lag1d_heatmap(s, fname, title1, title2):
    """
    Kanousis-style: 24×24 Pearson between hour-h of day D and
    hour-l of previous day D-1, after daily demeaning.
    """
    # ── Build pivot ───────────────────────────────────────────────────────
    df_tmp = s.to_frame('y')
    df_tmp['hour'] = df_tmp.index.hour
    df_tmp['date'] = df_tmp.index.date
    pivot = df_tmp.pivot_table(index='date', columns='hour', values='y').dropna()

    # (No daily demeaning — raw day-over-day correlation, full positive range)

    # Lag-1d
    pivot_lag1 = pivot.shift(1)
    common = pivot.index.intersection(pivot_lag1.dropna().index)
    A = pivot.loc[common].values       # current day  (N,24)
    B = pivot_lag1.loc[common].values  # previous day (N,24)

    # 24×24 Pearson
    corr_mat = np.zeros((24, 24))
    for h in range(24):
        for l in range(24):
            x, y_v = A[:, h], B[:, l]
            mask = ~(np.isnan(x) | np.isnan(y_v))
            if mask.sum() > 10:
                corr_mat[h, l] = np.corrcoef(x[mask], y_v[mask])[0, 1]

    # Use the actual data range so the full colour scale is used (max contrast)
    vmin = corr_mat.min()
    vmax = corr_mat.max()

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9.5, 8))

    im = ax.imshow(corr_mat, cmap='viridis', aspect='auto',
                   vmin=vmin, vmax=vmax, origin='upper')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ── Cell annotations ──────────────────────────────────────────────────
    norm_range = vmax - vmin
    for h in range(24):
        for l in range(24):
            val = corr_mat[h, l]
            norm_v = (val - vmin) / norm_range if norm_range > 0 else 0.5
            tc = 'white' if norm_v < 0.55 else 'black'
            ax.text(l, h, f'{val:.2f}', ha='center', va='center',
                    fontsize=4.2, color=tc)

    # ── Axis ticks ────────────────────────────────────────────────────────
    ticks = list(range(0, 24, 4))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel('Ώρα Προηγούμενης Ημέρας D−1', fontsize=11)
    ax.set_ylabel('Ώρα Τρέχουσας Ημέρας D', fontsize=11)
    ax.set_title(f'{title1}\n{title2}', fontsize=11, fontweight='bold', pad=12)

    # ── White grid lines every 4 hours ───────────────────────────────────
    for x in range(0, 24, 4):
        ax.axvline(x - 0.5, color='white', lw=0.6, alpha=0.7)
        ax.axhline(x - 0.5, color='white', lw=0.6, alpha=0.7)

    # ── Contour lines ─────────────────────────────────────────────────────
    lvls = [l for l in [0.5, 0.7, 0.9] if vmin < l < vmax]
    if lvls:
        try:
            ax.contour(corr_mat, levels=lvls, colors='k', linewidths=0.6, alpha=0.5)
        except Exception:
            pass

    plt.tight_layout()
    _save(fname)


def fig_corr_heatmap_price(dfp, pc):
    _lag1d_heatmap(dfp[pc].dropna(),
                   'fig_corr_heatmap_price.png',
                   'Ενδοημερήσια Συσχέτιση Τιμής ΗΕ',
                   'Ώρα ημέρας D  vs.  ίδια ώρα προηγούμενης ημέρας D−1  (Lag-1d)')


def fig_corr_heatmap_load(dfl):
    _lag1d_heatmap(dfl['y'].dropna(),
                   'fig_corr_heatmap_load.png',
                   'Ενδοημερήσια Συσχέτιση Φορτίου',
                   'Ώρα ημέρας D  vs.  ίδια ώρα προηγούμενης ημέρας D−1  (Lag-1d)')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.9 — fig_price_vs_gas.png  (scatter + timeseries)
# ════════════════════════════════════════════════════════════════════════════
def fig_price_vs_gas(dfp, pc):
    # Find gas column (prefer non-lagged)
    gas = None
    for candidate in ['gas_price', 'gas_price_lag1', 'gas_price_lag2']:
        if candidate in dfp.columns:
            gas = candidate; break
    if gas is None:
        gas = next((c for c in dfp.columns if 'gas' in c.lower()), None)
    if gas is None:
        print("    ✗ gas column not found"); return

    # Daily aggregation for cleaner scatter (matching 106.pdf style)
    df_d = dfp[[pc, gas]].resample('D').mean().dropna()
    x, y, yrs = df_d[gas].values, df_d[pc].values, df_d.index.year.values
    cmap = plt.get_cmap('plasma')
    yr_min, yr_max = yrs.min(), yrs.max()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Σχέση Τιμής Ηλεκτρικής Ενέργειας — Φυσικό Αέριο (2017–2025)',
                 fontsize=12, fontweight='bold', y=1.01)

    # ── Left: scatter ─────────────────────────────────────────────────────
    for yr in sorted(set(yrs)):
        mask  = yrs == yr
        color = cmap((yr - yr_min) / max(yr_max - yr_min, 1))
        a1.scatter(x[mask], y[mask], c=[color], s=10, alpha=0.65,
                   label=str(yr), edgecolors='none')
    m, b, r, *_ = stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 200)
    a1.plot(xl, m*xl+b, 'k--', lw=1.5, label=f'OLS (κλίση={m:.2f})')
    a1.set_xlabel('Τιμή φυσικού αερίου (€/MWh)')
    a1.set_ylabel('Τιμή ΗΕ — ημερήσιος μέσος (€/MWh)')
    a1.set_title('Τιμή ΗΕ vs Τιμή Φυσικού Αερίου\n(ημερήσιοι μέσοι, 2017–2025)',
                 fontsize=10.5)
    a1.legend(fontsize=7.5, ncol=2, framealpha=0.9)
    _rm(a1)

    # ── Right: dual-axis timeseries ────────────────────────────────────────
    sp = dfp[pc].resample('D').mean().dropna()
    sg = dfp[gas].resample('D').mean().dropna()
    a2b = a2.twinx()
    a2.plot(sp.index, sp.values, color='#2980B9', lw=0.8, alpha=0.85,
            label='Τιμή ΗΕ (€/MWh)')
    a2b.plot(sg.index, sg.values, color='#E67E22', lw=0.8, alpha=0.85,
             label='Τιμή αερίου (€/MWh)')
    a2.axvspan(pd.Timestamp('2021-07-01'), pd.Timestamp('2022-12-31'),
               alpha=0.10, color='#F39C12', label='Ενεργειακή κρίση')
    a2.set_ylabel('Τιμή ΗΕ (€/MWh)', color='#2980B9', fontsize=9.5)
    a2b.set_ylabel('Τιμή αερίου (€/MWh)', color='#E67E22', fontsize=9.5)
    a2.set_title('Εξέλιξη Τιμής ΗΕ & Φ. Αερίου στον Χρόνο', fontsize=10.5)
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    a2.xaxis.set_major_locator(mdates.YearLocator())
    l1, lb1 = a2.get_legend_handles_labels()
    l2, lb2 = a2b.get_legend_handles_labels()
    a2.legend(l1+l2, lb1+lb2, fontsize=7.5, loc='upper left', framealpha=0.9)
    a2.spines[['top']].set_visible(False)

    plt.tight_layout()
    _save('fig_price_vs_gas.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.10 — fig_weekly_overlay.png  (2 weeks: summer + winter)
# ════════════════════════════════════════════════════════════════════════════
def fig_weekly_overlay(dfp, dfl, pc):
    WEEKS = [
        ('Καλοκαίρι — Ιούλιος 2023',   '2023-07-03', '2023-07-09 23:00'),
        ('Χειμώνας — Ιανουάριος 2024',  '2024-01-08', '2024-01-14 23:00'),
    ]

    # RES column discovery (prefer _fc, fall back to _lag1)
    def _find(cols, keyword, fallback):
        c = next((c for c in cols if keyword in c and 'fc' in c), None)
        if c is None:
            c = next((c for c in cols if keyword in c and 'lag1' in c), None)
        return c

    wind_c  = _find(dfp.columns, 'wind',  None)
    solar_c = _find(dfp.columns, 'solar', None)
    hydro_c = _find(dfp.columns, 'hydro', None)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={'hspace': 0.40})
    fig.suptitle('Εβδομαδιαία Επισκόπηση: Φορτίο, Τιμή ΗΕ και Παραγωγή ΑΠΕ',
                 fontsize=12, fontweight='bold')

    _wknd_labeled = [False]  # track weekend label for legend

    for ax, (title, start, end) in zip(axes, WEEKS):
        sp = dfp.loc[start:end, pc].dropna()
        sl = dfl.loc[start:end, 'y'].dropna()
        _wknd_labeled[0] = False

        ax2 = ax.twinx()

        # ── RES fills from 0 (solar=yellow, wind=blue, on load axis) ─────
        solar_vals = (dfp.loc[start:end, solar_c].reindex(sl.index).fillna(0).values
                      if solar_c and solar_c in dfp.columns else None)
        wind_vals  = (dfp.loc[start:end, wind_c].reindex(sl.index).fillna(0).values
                      if wind_c  and wind_c  in dfp.columns else None)

        if solar_vals is not None:
            ax.fill_between(sl.index, 0, solar_vals,
                            alpha=0.60, color='#F4D03F', linewidth=0,
                            label='Ηλιακή παραγωγή (FC)', zorder=2)
        if wind_vals is not None:
            ax.fill_between(sl.index, 0, wind_vals,
                            alpha=0.45, color='#AED6F1', linewidth=0,
                            label='Αιολική παραγωγή (FC)', zorder=1)

        # ── Load line (above fills) ───────────────────────────────────────
        ax.plot(sl.index, sl.values, color='#E67E22', lw=2.0,
                label='Φορτίο (MW)', zorder=5)
        ax.set_ylabel('Φορτίο (MW)', color='#E67E22', fontsize=9.5)
        ax.set_ylim(0, sl.max() * 1.18)  # start from 0 so fills are visible
        ax.yaxis.label.set_color('#E67E22')

        # ── Price (right axis) ────────────────────────────────────────────
        ax2.plot(sp.index, sp.values, color='#C0392B', lw=1.2, ls='--',
                 label='Τιμή ΗΕ (€/MWh)', alpha=0.9, zorder=6)
        ax2.set_ylabel('Τιμή ΗΕ (€/MWh)', color='#C0392B', fontsize=9.5)
        ax2.spines[['top']].set_visible(False)
        ax2.yaxis.label.set_color('#C0392B')

        # ── Weekend shading ───────────────────────────────────────────────
        for day in pd.date_range(start, end, freq='D'):
            if day.weekday() >= 5:
                lbl = 'Σαββατοκύριακο' if not _wknd_labeled[0] else '_nolegend_'
                ax.axvspan(day, day + pd.Timedelta('1D'),
                           alpha=0.07, color='#bdc3c7', zorder=0, label=lbl)
                _wknd_labeled[0] = True

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d/%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.spines[['top', 'right']].set_visible(False)

        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=7.5,
                  framealpha=0.9, ncol=2)

    _save('fig_weekly_overlay.png')


# ════════════════════════════════════════════════════════════════════════════
# Σχήμα 3.11 — fig_train_test_split.png  (2 panels: price + load)
# ════════════════════════════════════════════════════════════════════════════
def fig_train_test_split(dfp, dfl, pc):
    cutoff   = pd.Timestamp('2025-12-01')
    end_test = pd.Timestamp('2026-02-28 23:00')

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7.5),
                                  gridspec_kw={'hspace': 0.42})
    fig.suptitle('Διαχωρισμός Δεδομένων: Σύνολο Εκπαίδευσης & Αξιολόγησης\n'
                 '(Ελληνική Αγορά Ηλεκτρικής Ενέργειας — Αγορά Επόμενης Ημέρας)',
                 fontsize=11.5, fontweight='bold')

    panels = [
        (a1, dfp[pc].dropna(),  '#5B9BD5',
         'Σύνολο εκπαίδευσης\n(Ιαν. 2017 – Νοεμ. 2025)',
         'Περίοδος αξιολόγησης\n(Δεκ. 2025 – Φεβ. 2026)',
         'Τιμή ΗΕ (€/MWh)',
         'Τιμή ΗΕ — Διαχωρισμός Εκπαίδευσης / Αξιολόγησης'),
        (a2, dfl['y'].dropna(), '#E67E22',
         'Σύνολο εκπαίδευσης\n(Ιαν. 2015 – Νοεμ. 2025)',
         'Περίοδος αξιολόγησης\n(Δεκ. 2025 – Φεβ. 2026)',
         'Φορτίο Συστήματος (MW)',
         'Φορτίο Συστήματος — Διαχωρισμός Εκπαίδευσης / Αξιολόγησης'),
    ]

    for ax, s, c_tr, lbl_tr, lbl_te, ylabel, panel_title in panels:
        roll  = s.rolling(168, min_periods=1).mean()
        train = roll[roll.index < cutoff]
        test  = roll[(roll.index >= cutoff) & (roll.index <= end_test)]

        ax.fill_between(train.index, train.values, alpha=0.30, color=c_tr)
        ax.plot(train.index, train.values, color=c_tr, lw=0.5, label=lbl_tr)

        ax.fill_between(test.index, test.values, alpha=0.55, color='#E74C3C')
        ax.plot(test.index, test.values, color='#E74C3C', lw=1.5,
                label=lbl_te)

        # Annotation with size
        n_train = (s.index < cutoff).sum()
        n_years = round(n_train / 8760, 1)
        ax.text(train.index[len(train)//2], train.max() * 0.82,
                f'Σύνολο εκπαίδευσης: {n_train:,} ωριαίες παρατηρήσεις\n'
                f'(~{n_years} έτη)',
                fontsize=7.5, color=c_tr, ha='center', va='top')

        ax.axvline(cutoff, color='#2C3E50', lw=1.5, ls='--')
        ax.text(cutoff + pd.Timedelta(hours=48),
                roll.max() * 0.95,
                'Αρχή\nδοκιμής', fontsize=8, color='#E74C3C', va='top')

        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlim(s.index.min(), end_test)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.set_title(panel_title, fontsize=10.5, fontweight='bold')
        _rm(ax)

    _save('fig_train_test_split.png')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print("  Chapter 3 figures — Kanousis 106.pdf style")
    print("=" * 62)

    print("\nLoading data...")
    dfp, dfl, pc = load_data()
    print(f"  price: {dfp.shape}  load: {dfl.shape}")

    tasks = [
        ("3.2 — Price timeseries",
         lambda: fig_price_timeseries(dfp, pc)),
        ("3.3 — Monthly avg by year (stacked)",
         lambda: fig_monthly_avg_by_year(dfp, dfl, pc)),
        ("3.4 — Daily/weekly profile price",
         lambda: fig_daily_weekly_profile(dfp, pc)),
        ("3.5 — Daily/weekly profile load (bigger)",
         lambda: fig_daily_weekly_profile_load(dfl)),
        ("3.6 — ACF/PACF price (all-Greek)",
         lambda: fig_acf_pacf_price(dfp, pc)),
        ("3.7 — Corr heatmap price (lag-1d Kanousis)",
         lambda: fig_corr_heatmap_price(dfp, pc)),
        ("3.8 — Corr heatmap load (lag-1d Kanousis)",
         lambda: fig_corr_heatmap_load(dfl)),
        ("3.9 — Price vs gas",
         lambda: fig_price_vs_gas(dfp, pc)),
        # ("3.10 — Weekly overlay",
        #  lambda: fig_weekly_overlay(dfp, dfl, pc)),  # αφαιρέθηκε από data.tex

        ("3.11 — Train/test split",
         lambda: fig_train_test_split(dfp, dfl, pc)),
    ]

    ok = fail = 0
    for name, fn in tasks:
        print(f"\n  {name}")
        try:
            fn(); ok += 1
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            traceback.print_exc(); fail += 1

    print(f"\n{'─'*62}")
    print(f"  Done: {ok} ✓   Failed: {fail} ✗")
    print(f"  Output: {OUT}")
    print(f"{'─'*62}\n")


if __name__ == '__main__':
    main()
