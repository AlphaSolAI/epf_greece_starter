"""
generate_data_chapter_figures.py
----------------------------------
Παράγει figures ακαδημαϊκής ποιότητας για το κεφάλαιο δεδομένων
της διπλωματικής (EPF Greece, Jan 2017 – Nov 2025).

Εκτέλεση:
    conda run -n epf --no-capture-output python -m src.generate_data_chapter_figures

Αποθηκεύει PNG στο thesis_output/ (200 DPI).
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from scipy import stats

# ── Στυλ ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

BLUE   = '#2176ae'
ORANGE = '#e8871e'
GREEN  = '#3a9e4a'
RED    = '#c0392b'
PURPLE = '#7b2d8b'
TEAL   = '#4db6ac'
AMBER  = '#ffd54f'
INDIGO = '#5c6bc0'

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_BASE, 'thesis_output')
os.makedirs(OUT_DIR, exist_ok=True)

MONTH_LABELS = ['Ιαν','Φεβ','Μαρ','Απρ','Μάι','Ιούν',
                'Ιούλ','Αύγ','Σεπ','Οκτ','Νοε','Δεκ']
DOW_LABELS   = ['Δευτ','Τρίτ','Τετ','Πέμπ','Παρ','Σάββ','Κυρ']


def remove_spines(ax):
    ax.spines[['top', 'right']].set_visible(False)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ {name}")


def load_data():
    _data = os.path.join(_BASE, 'data', 'processed')
    df_p = pd.read_parquet(os.path.join(_data, 'hourly.parquet'))
    df_l = pd.read_parquet(os.path.join(_data, 'hourly_load.parquet'))
    df_p.index = pd.to_datetime(df_p.index)
    df_l.index = pd.to_datetime(df_l.index)
    # Ορισμός στήλης τιμής
    price_col = 'DA_Price' if 'DA_Price' in df_p.columns else 'y'
    return df_p, df_l, price_col


# ── Figure 1: Πλήρης χρονοσειρά τιμής ──────────────────────────────────────
def fig_da_price_series(df_p, price_col):
    s = df_p[price_col].dropna()
    roll = s.rolling(window=168, center=True, min_periods=24).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), gridspec_kw={'hspace': 0.38})

    # Πάνω πάνελ: ωριαία + rolling mean
    ax1.plot(s.index, s.values, color='#bbbbbb', lw=0.4, alpha=0.7, label='Ωριαία τιμή')
    ax1.plot(roll.index, roll.values, color=BLUE, lw=1.5, label='7-ήμερος ΚΚΟ')
    ax1.axhline(100, linestyle='--', color=ORANGE, lw=1.0, alpha=0.9, label='100 €/MWh')
    ax1.axvspan(pd.Timestamp('2021-06-01'), pd.Timestamp('2022-12-31'),
                alpha=0.07, color='red', label='Ενεργειακή κρίση 2021-22')
    ax1.set_ylabel('Τιμή Ημερήσιας Αγοράς (€/MWh)')
    ax1.set_xlim(s.index.min(), s.index.max())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.legend(loc='upper left', ncol=2, framealpha=0.85)
    remove_spines(ax1)

    # Κάτω πάνελ: monthly boxplots
    monthly = [s[s.index.to_period('M') == p].values
               for p in s.index.to_period('M').unique()]
    months_dt = [pd.Period(p, 'M').to_timestamp() for p in s.index.to_period('M').unique()]
    bp = ax2.boxplot(monthly, positions=range(len(monthly)),
                     widths=0.6, patch_artist=True,
                     medianprops=dict(color=RED, lw=1.5),
                     boxprops=dict(facecolor='#c8dff5', alpha=0.8),
                     whiskerprops=dict(lw=0.8),
                     flierprops=dict(marker='.', ms=1.5, alpha=0.3, color='#888888'),
                     showcaps=True, capwidths=0.4)
    # X-axis labels: ετήσιες
    year_ticks = [i for i, dt in enumerate(months_dt) if dt.month == 1]
    year_labels = [dt.strftime('%Y') for dt in months_dt if dt.month == 1]
    ax2.set_xticks(year_ticks)
    ax2.set_xticklabels(year_labels)
    ax2.set_ylabel('Τιμή (€/MWh)')
    ax2.set_xlabel('Έτος')
    remove_spines(ax2)

    savefig('fig_da_price_series.png')


# ── Figure 2: Πλήρης χρονοσειρά φορτίου ────────────────────────────────────
def fig_load_series(df_l):
    s = df_l['y'].dropna()
    roll = s.rolling(window=168, center=True, min_periods=24).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), gridspec_kw={'hspace': 0.38})

    ax1.plot(s.index, s.values, color='#bbbbbb', lw=0.4, alpha=0.7, label='Ωριαίο φορτίο')
    ax1.plot(roll.index, roll.values, color=GREEN, lw=1.5, label='7-ήμερος ΚΚΟ')
    ax1.set_ylabel('Ηλεκτρική Κατανάλωση (MW)')
    ax1.set_xlim(s.index.min(), s.index.max())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.legend(loc='upper right', framealpha=0.85)
    remove_spines(ax1)

    monthly = [s[s.index.to_period('M') == p].values
               for p in s.index.to_period('M').unique()]
    months_dt = [pd.Period(p, 'M').to_timestamp() for p in s.index.to_period('M').unique()]
    ax2.boxplot(monthly, positions=range(len(monthly)),
                widths=0.6, patch_artist=True,
                medianprops=dict(color=RED, lw=1.5),
                boxprops=dict(facecolor='#c8e6c9', alpha=0.8),
                whiskerprops=dict(lw=0.8),
                flierprops=dict(marker='.', ms=1.5, alpha=0.3, color='#888888'),
                showcaps=True, capwidths=0.4)
    year_ticks  = [i for i, dt in enumerate(months_dt) if dt.month == 1]
    year_labels = [dt.strftime('%Y') for dt in months_dt if dt.month == 1]
    ax2.set_xticks(year_ticks)
    ax2.set_xticklabels(year_labels)
    ax2.set_ylabel('Φορτίο (MW)')
    ax2.set_xlabel('Έτος')
    remove_spines(ax2)

    savefig('fig_load_series.png')


# ── Figure 3: Μηνιαία τιμή ανά έτος ────────────────────────────────────────
def fig_monthly_price_by_year(df_p, price_col):
    s = df_p[price_col].copy()
    cmap = plt.get_cmap('tab10')
    years = sorted(s.index.year.unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, yr in enumerate(years):
        sub = s[s.index.year == yr]
        monthly_mean = sub.groupby(sub.index.month).mean()
        lw = 2.5 if yr in (2021, 2022) else 1.2
        alpha = 1.0
        ax.plot(monthly_mean.index, monthly_mean.values,
                color=cmap(i % 10), lw=lw, alpha=alpha,
                marker='o', ms=3.5, label=str(yr))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel('Μέση Τιμή ΗΑ (€/MWh)')
    ax.set_xlabel('Μήνας')
    ax.legend(ncol=3, loc='upper left', framealpha=0.85, fontsize=8.5)
    remove_spines(ax)

    savefig('fig_monthly_price_by_year.png')


# ── Figure 4: Μηνιαίο φορτίο ανά έτος ──────────────────────────────────────
def fig_monthly_load_by_year(df_l):
    s = df_l['y'].copy()
    cmap = plt.get_cmap('tab10')
    years = sorted(s.index.year.unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, yr in enumerate(years):
        sub = s[s.index.year == yr]
        monthly_mean = sub.groupby(sub.index.month).mean()
        ax.plot(monthly_mean.index, monthly_mean.values,
                color=cmap(i % 10), lw=1.2, marker='o', ms=3.5, label=str(yr))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel('Μέση Κατανάλωση (MW)')
    ax.set_xlabel('Μήνας')
    ax.legend(ncol=3, loc='upper right', framealpha=0.85, fontsize=8.5)
    remove_spines(ax)

    savefig('fig_monthly_load_by_year.png')


# ── Figure 5: Ωριαίο & εβδομαδιαίο μοτίβο τιμής ───────────────────────────
def fig_price_patterns(df_p, price_col):
    s = df_p[price_col].copy()

    hourly_mean = s.groupby(s.index.hour).mean()
    hourly_std  = s.groupby(s.index.hour).std()

    crisis = s[(s.index.year >= 2021) & (s.index.year <= 2022)]
    hourly_crisis = crisis.groupby(crisis.index.hour).mean()

    dow_mean = s.groupby(s.index.dayofweek).mean()
    dow_std  = s.groupby(s.index.dayofweek).std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Ωριαίο
    hours = np.arange(24)
    ax1.fill_between(hours, hourly_mean - hourly_std, hourly_mean + hourly_std,
                     alpha=0.18, color=BLUE, label='±1σ')
    ax1.plot(hours, hourly_mean.values, color=BLUE, lw=2, label='Μέση τιμή')
    ax1.plot(hourly_crisis.index, hourly_crisis.values,
             color=ORANGE, lw=1.5, linestyle='--', label='Κρίση 2021-22')
    ax1.set_xlabel('Ώρα Ημέρας')
    ax1.set_ylabel('Μέση Τιμή ΗΑ (€/MWh)')
    ax1.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax1.legend(framealpha=0.85)
    remove_spines(ax1)

    # Εβδομαδιαίο
    dows = np.arange(7)
    ax2.fill_between(dows, dow_mean - dow_std, dow_mean + dow_std,
                     alpha=0.18, color=BLUE, label='±1σ')
    ax2.plot(dows, dow_mean.values, color=BLUE, lw=2, marker='o', ms=5, label='Μέση τιμή')
    ax2.set_xticks(dows)
    ax2.set_xticklabels(DOW_LABELS)
    ax2.set_ylabel('Μέση Τιμή ΗΑ (€/MWh)')
    ax2.legend(framealpha=0.85)
    remove_spines(ax2)

    plt.tight_layout()
    savefig('fig_price_patterns.png')


# ── Figure 6: Ωριαίο & εβδομαδιαίο μοτίβο φορτίου ─────────────────────────
def fig_load_patterns(df_l):
    s = df_l['y'].copy()

    hourly_mean = s.groupby(s.index.hour).mean()
    hourly_std  = s.groupby(s.index.hour).std()
    dow_mean    = s.groupby(s.index.dayofweek).mean()
    dow_std     = s.groupby(s.index.dayofweek).std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    hours = np.arange(24)
    ax1.fill_between(hours, hourly_mean - hourly_std, hourly_mean + hourly_std,
                     alpha=0.18, color=GREEN)
    ax1.plot(hours, hourly_mean.values, color=GREEN, lw=2, label='Μέση κατανάλωση')
    ax1.set_xlabel('Ώρα Ημέρας')
    ax1.set_ylabel('Μέση Κατανάλωση (MW)')
    ax1.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax1.legend(framealpha=0.85)
    remove_spines(ax1)

    dows = np.arange(7)
    ax2.fill_between(dows, dow_mean - dow_std, dow_mean + dow_std,
                     alpha=0.18, color=GREEN)
    ax2.plot(dows, dow_mean.values, color=GREEN, lw=2, marker='o', ms=5,
             label='Μέση κατανάλωση')
    ax2.set_xticks(dows)
    ax2.set_xticklabels(DOW_LABELS)
    ax2.set_ylabel('Μέση Κατανάλωση (MW)')
    ax2.legend(framealpha=0.85)
    remove_spines(ax2)

    plt.tight_layout()
    savefig('fig_load_patterns.png')


# ── Figure 7: ACF & PACF τιμής ──────────────────────────────────────────────
def fig_acf_pacf_price(df_p, price_col):
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        print("  ✗ statsmodels not available — skip ACF/PACF price")
        return

    s = df_p[price_col].dropna()
    # PACF σε subsample για ταχύτητα
    rng = np.random.default_rng(42)
    idx = rng.choice(len(s), size=min(6000, len(s)), replace=False)
    s_sub = s.iloc[sorted(idx)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(s, lags=175, ax=ax1, alpha=0.05, color=BLUE,
             vlines_kwargs={'colors': BLUE}, use_vlines=True)
    plot_pacf(s_sub, lags=60, ax=ax2, alpha=0.05, color=BLUE,
              method='ywmle', use_vlines=True)

    for ax, ttl in [(ax1, 'Τιμή ΗΑ — ACF'), (ax2, 'Τιμή ΗΑ — PACF (λαγ 0-60)')]:
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('Υστέρηση (ώρες)')
        ax.set_ylabel('Αυτοσυσχέτιση')
        # Ημερήσιες αρμονικές
        for lag in range(24, 176, 24):
            if lag <= ax.get_xlim()[1]:
                ax.axvline(lag, color=ORANGE, alpha=0.35, lw=0.8, linestyle='--')
        remove_spines(ax)

    plt.tight_layout()
    savefig('fig_acf_pacf_price.png')


# ── Figure 8: ACF & PACF φορτίου ────────────────────────────────────────────
def fig_acf_pacf_load(df_l):
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        print("  ✗ statsmodels not available — skip ACF/PACF load")
        return

    s = df_l['y'].dropna()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(s), size=min(6000, len(s)), replace=False)
    s_sub = s.iloc[sorted(idx)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(s, lags=175, ax=ax1, alpha=0.05, color=GREEN, use_vlines=True)
    plot_pacf(s_sub, lags=60, ax=ax2, alpha=0.05, color=GREEN,
              method='ywmle', use_vlines=True)

    for ax, ttl in [(ax1, 'Φορτίο — ACF'), (ax2, 'Φορτίο — PACF (λαγ 0-60)')]:
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('Υστέρηση (ώρες)')
        ax.set_ylabel('Αυτοσυσχέτιση')
        for lag in range(24, 176, 24):
            if lag <= ax.get_xlim()[1]:
                ax.axvline(lag, color=ORANGE, alpha=0.35, lw=0.8, linestyle='--')
        remove_spines(ax)

    plt.tight_layout()
    savefig('fig_acf_pacf_load.png')


# ── Figure 9: Heatmap ωριαίας αυτοσυσχέτισης τιμής ─────────────────────────
def fig_corr_heatmap_price(df_p, price_col):
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    s = df_p[price_col].dropna()
    df_tmp = pd.DataFrame({'y': s.values, 'h': s.index.hour,
                           'date': s.index.normalize()})
    pivot = df_tmp.pivot_table(index='date', columns='h', values='y')
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot[sorted(pivot.columns)]
    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))

    if sns is not None:
        sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=0.3, vmax=1.0,
                    annot=False, square=True, linewidths=0.15,
                    ax=ax, cbar_kws={'label': 'Pearson r', 'shrink': 0.85})
    else:
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=0.3, vmax=1.0,
                       aspect='equal')
        plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.85)
        ax.set_xticks(range(24))
        ax.set_yticks(range(24))
        ax.set_xticklabels(range(24), fontsize=7)
        ax.set_yticklabels(range(24), fontsize=7)

    # Contour lines για r=0.7 και r=0.9
    try:
        cs = ax.contour(corr.values, levels=[0.7, 0.9],
                        colors=['#555555', '#111111'], linewidths=0.8)
        ax.clabel(cs, fmt='%.1f', fontsize=7)
    except Exception:
        pass

    ax.set_xlabel('Ώρα Ημέρας h(t)')
    ax.set_ylabel('Ώρα Ημέρας h(t-24)')

    plt.tight_layout()
    savefig('fig_corr_heatmap_price.png')


# ── Figure 10: Heatmap ωριαίας αυτοσυσχέτισης φορτίου ──────────────────────
def fig_corr_heatmap_load(df_l):
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    s = df_l['y'].dropna()
    df_tmp = pd.DataFrame({'y': s.values, 'h': s.index.hour,
                           'date': s.index.normalize()})
    pivot = df_tmp.pivot_table(index='date', columns='h', values='y')
    pivot.columns = [int(c) for c in pivot.columns]
    pivot = pivot[sorted(pivot.columns)]
    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))

    if sns is not None:
        sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=0.5, vmax=1.0,
                    annot=False, square=True, linewidths=0.15,
                    ax=ax, cbar_kws={'label': 'Pearson r', 'shrink': 0.85})
    else:
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=0.5, vmax=1.0,
                       aspect='equal')
        plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.85)
        ax.set_xticks(range(24))
        ax.set_yticks(range(24))
        ax.set_xticklabels(range(24), fontsize=7)
        ax.set_yticklabels(range(24), fontsize=7)

    try:
        cs = ax.contour(corr.values, levels=[0.85, 0.95],
                        colors=['#555555', '#111111'], linewidths=0.8)
        ax.clabel(cs, fmt='%.2f', fontsize=7)
    except Exception:
        pass

    ax.set_xlabel('Ώρα Ημέρας h(t)')
    ax.set_ylabel('Ώρα Ημέρας h(t-24)')

    plt.tight_layout()
    savefig('fig_corr_heatmap_load.png')


# ── Figure 11: Εβδομαδιαίο overlay τιμή + φορτίο + ΑΠΕ ─────────────────────
def fig_weekly_overlay(df_p, df_l, price_col):
    start, end = '2023-03-06', '2023-03-12 23:00'
    sp = df_p.loc[start:end]
    sl = df_l.loc[start:end]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True,
                             gridspec_kw={'hspace': 0.08})

    # Τιμή
    axes[0].plot(sp.index, sp[price_col].values, color=BLUE, lw=1.5)
    axes[0].set_ylabel('Τιμή (€/MWh)')
    remove_spines(axes[0])

    # Φορτίο
    axes[1].plot(sl.index, sl['y'].values, color=GREEN, lw=1.5)
    axes[1].set_ylabel('Φορτίο (MW)')
    remove_spines(axes[1])

    # ΑΠΕ stacked area
    wind_col  = next((c for c in sp.columns if 'wind' in c.lower() and 'lag' in c.lower()), None)
    solar_col = next((c for c in sp.columns if 'solar' in c.lower() and 'lag' in c.lower()), None)
    hydro_col = next((c for c in sp.columns if 'hydro' in c.lower() and 'lag' in c.lower()), None)

    gen_data = {}
    for lbl, col, color in [('Αιολική', wind_col, TEAL),
                             ('Ηλιακή', solar_col, AMBER),
                             ('Υδροηλεκτρική', hydro_col, INDIGO)]:
        if col and col in sp.columns:
            gen_data[lbl] = (sp[col].values, color)

    if gen_data:
        bottom = np.zeros(len(sp))
        for lbl, (vals, color) in gen_data.items():
            vals_clean = np.nan_to_num(vals, nan=0.0)
            axes[2].fill_between(sp.index, bottom, bottom + vals_clean,
                                 alpha=0.7, color=color, label=lbl)
            bottom += vals_clean
        axes[2].legend(loc='upper right', framealpha=0.85)
    else:
        axes[2].text(0.5, 0.5, 'Δεδομένα ΑΠΕ μη διαθέσιμα', transform=axes[2].transAxes,
                     ha='center', va='center', fontsize=10)
    axes[2].set_ylabel('Παραγωγή ΑΠΕ (MW)')
    remove_spines(axes[2])

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%a %d/%m'))
    axes[2].xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right')

    savefig('fig_weekly_overlay.png')


# ── Figure 12: Τιμή vs Φυσικό Αέριο ────────────────────────────────────────
def fig_price_vs_gas(df_p, price_col):
    gas_col = next((c for c in df_p.columns if 'gas' in c.lower() and 'price' in c.lower()), None)
    if gas_col is None:
        gas_col = next((c for c in df_p.columns if 'gas' in c.lower()), None)
    if gas_col is None:
        print("  ✗ gas_price column not found — skip fig_price_vs_gas")
        return

    df_clean = df_p[[price_col, gas_col]].dropna()
    x = df_clean[gas_col].values
    y = df_clean[price_col].values
    years = df_clean.index.year.values

    # Scatter με χρωματισμό κατά έτος
    cmap = plt.get_cmap('plasma')
    yr_min, yr_max = years.min(), years.max()
    colors = cmap((years - yr_min) / max(yr_max - yr_min, 1))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(x, y, c=years, cmap='plasma', alpha=0.12, s=2.0,
                    vmin=yr_min, vmax=yr_max)
    plt.colorbar(sc, ax=ax, label='Έτος', shrink=0.85)

    # Regression line
    m, b, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color=RED, lw=2.0, label=f'OLS (r={r:.3f})')
    ax.text(0.04, 0.93, f'r = {r:.3f}  (p < 0.001)', transform=ax.transAxes,
            fontsize=9.5, color=RED)

    ax.set_xlabel('Τιμή Φ. Αερίου TTF (€/MWh)')
    ax.set_ylabel('Τιμή ΗΑ (€/MWh)')
    ax.legend(loc='lower right', framealpha=0.85)
    remove_spines(ax)

    savefig('fig_price_vs_gas.png')


# ── Figure 13: Train/Test split ─────────────────────────────────────────────
def fig_train_test_split(df_p, price_col):
    s = df_p[price_col].dropna()
    # 7-ημερος κυλιόμενος για ευκρίνεια
    roll = s.rolling(window=168, min_periods=1).mean()

    cutoff = pd.Timestamp('2025-12-01')
    train = roll[roll.index < cutoff]
    test  = roll[roll.index >= cutoff]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(train.index, train.values, color=BLUE, lw=0.8, alpha=0.75, label='Εκπαίδευση (2017–2025)')
    ax.plot(test.index,  test.values,  color=RED,  lw=1.5, label='Δοκιμή (Δεκ. 2025)')
    ax.axvline(cutoff, color='black', lw=1.5, linestyle='--')
    ax.axvspan(cutoff, s.index.max(), alpha=0.10, color='red')
    ax.text(cutoff + pd.Timedelta(hours=36), ax.get_ylim()[1] * 0.92,
            'Test\n168h', fontsize=9, color=RED, va='top')
    ax.set_ylabel('Τιμή ΗΑ — 7-ήμερος ΚΚΟ (€/MWh)')
    ax.set_xlim(s.index.min(), s.index.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc='upper left', framealpha=0.85)
    remove_spines(ax)

    savefig('fig_train_test_split.png')


# ── Figure 14: Κατανομή τιμής & φορτίου ────────────────────────────────────
def fig_price_load_dist(df_p, df_l, price_col):
    from scipy.stats import gaussian_kde

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, s, color, xlabel in [
        (ax1, df_p[price_col].dropna(), BLUE, 'Τιμή ΗΑ (€/MWh)'),
        (ax2, df_l['y'].dropna(),       GREEN, 'Φορτίο (MW)')
    ]:
        vals = s.values
        ax.hist(vals, bins=80, density=True, color=color, alpha=0.65,
                edgecolor='white', linewidth=0.3)
        kde = gaussian_kde(vals, bw_method='scott')
        x_kde = np.linspace(vals.min(), vals.max(), 400)
        ax.plot(x_kde, kde(x_kde), color=ORANGE, lw=2.0, label='KDE')
        med = np.median(vals)
        ax.axvline(med, linestyle='--', color=RED, lw=1.5,
                   label=f'Διάμεσος: {med:.1f}')
        sk = stats.skew(vals)
        ax.text(0.97, 0.93, f'Λοξότητα: {sk:.2f}', transform=ax.transAxes,
                fontsize=9, ha='right', va='top')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Πυκνότητα')
        ax.legend(framealpha=0.85)
        remove_spines(ax)

    plt.tight_layout()
    savefig('fig_price_load_dist.png')


# ── Figure 15: Μηνιαία boxplots τιμής & φορτίου ─────────────────────────────
def fig_monthly_boxplots(df_p, df_l, price_col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, s, color, ylabel in [
        (ax1, df_p[price_col].dropna(), '#c8dff5', 'Τιμή ΗΑ (€/MWh)'),
        (ax2, df_l['y'].dropna(),       '#c8e6c9', 'Φορτίο (MW)')
    ]:
        monthly_data = [s[s.index.month == m].values for m in range(1, 13)]
        bp = ax.boxplot(monthly_data, positions=range(1, 13),
                        widths=0.55, patch_artist=True,
                        medianprops=dict(color=RED, lw=1.5),
                        boxprops=dict(facecolor=color, alpha=0.85),
                        whiskerprops=dict(lw=0.8),
                        flierprops=dict(marker='.', ms=2, alpha=0.2, color='#888888'),
                        showcaps=True)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Μήνας')
        remove_spines(ax)

    plt.tight_layout()
    savefig('fig_monthly_boxplots.png')


# ── Figure 16: Gas & CO₂ prices ─────────────────────────────────────────────
def fig_fuel_prices(df_p):
    gas_col = next((c for c in df_p.columns if 'gas' in c.lower() and 'price' in c.lower()), None)
    co2_col = next((c for c in df_p.columns if 'co2' in c.lower()), None)

    if gas_col is None and co2_col is None:
        print("  ✗ gas/co2 columns not found — skip fig_fuel_prices")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={'hspace': 0.12})

    if gas_col and gas_col in df_p.columns:
        sg = df_p[gas_col].dropna()
        axes[0].plot(sg.index, sg.values, color=ORANGE, lw=1.0, alpha=0.85)
        axes[0].axvspan(pd.Timestamp('2021-06-01'), pd.Timestamp('2022-12-31'),
                        alpha=0.10, color='red')
        axes[0].set_ylabel('Τιμή ΦΑ TTF (€/MWh)')
        remove_spines(axes[0])
    else:
        axes[0].text(0.5, 0.5, 'Δεδομένα gas_price μη διαθέσιμα',
                     transform=axes[0].transAxes, ha='center', va='center')

    if co2_col and co2_col in df_p.columns:
        sc = df_p[co2_col].dropna()
        axes[1].plot(sc.index, sc.values, color=PURPLE, lw=1.0, alpha=0.85)
        axes[1].set_ylabel('Τιμή CO₂ ETS (€/t)')
        remove_spines(axes[1])
    else:
        axes[1].text(0.5, 0.5, 'Δεδομένα co2_price μη διαθέσιμα',
                     transform=axes[1].transAxes, ha='center', va='center')

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())

    savefig('fig_fuel_prices.png')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Φόρτωση δεδομένων...")
    df_p, df_l, price_col = load_data()
    print(f"  Price: {df_p.shape}, load: {df_l.shape}")
    print(f"  Price col: '{price_col}'")
    print(f"  Price columns: {df_p.columns.tolist()[:12]}")
    print()

    tasks = [
        ("Figure 1  — Χρονοσειρά τιμής",           lambda: fig_da_price_series(df_p, price_col)),
        ("Figure 2  — Χρονοσειρά φορτίου",          lambda: fig_load_series(df_l)),
        ("Figure 3  — Μηνιαία τιμή ανά έτος",       lambda: fig_monthly_price_by_year(df_p, price_col)),
        ("Figure 4  — Μηνιαίο φορτίο ανά έτος",     lambda: fig_monthly_load_by_year(df_l)),
        ("Figure 5  — Μοτίβα τιμής",                lambda: fig_price_patterns(df_p, price_col)),
        ("Figure 6  — Μοτίβα φορτίου",              lambda: fig_load_patterns(df_l)),
        ("Figure 7  — ACF/PACF τιμής",              lambda: fig_acf_pacf_price(df_p, price_col)),
        ("Figure 8  — ACF/PACF φορτίου",            lambda: fig_acf_pacf_load(df_l)),
        ("Figure 9  — Heatmap αυτοσυσχέτισης τιμής",  lambda: fig_corr_heatmap_price(df_p, price_col)),
        ("Figure 10 — Heatmap αυτοσυσχέτισης φορτίου",lambda: fig_corr_heatmap_load(df_l)),
        # ("Figure 11 — Εβδομαδιαίο overlay ΑΠΕ",     lambda: fig_weekly_overlay(df_p, df_l, price_col)),  # αφαιρέθηκε από data.tex

        ("Figure 12 — Τιμή vs Αέριο",               lambda: fig_price_vs_gas(df_p, price_col)),
        ("Figure 13 — Train/Test split",             lambda: fig_train_test_split(df_p, price_col)),
        ("Figure 14 — Κατανομές",                   lambda: fig_price_load_dist(df_p, df_l, price_col)),
        ("Figure 15 — Μηνιαία boxplots",             lambda: fig_monthly_boxplots(df_p, df_l, price_col)),
        ("Figure 16 — Καύσιμα",                     lambda: fig_fuel_prices(df_p)),
    ]

    ok, fail = 0, 0
    for name, fn in tasks:
        print(f"{name}")
        try:
            fn()
            ok += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            traceback.print_exc()
            fail += 1

    print(f"\n{'='*50}")
    print(f"Ολοκλήρωση: {ok} ✓  {fail} ✗")
    print(f"Αρχεία αποθηκευμένα στο: {os.path.abspath(OUT_DIR)}/")


if __name__ == '__main__':
    main()
