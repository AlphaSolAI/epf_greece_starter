"""
src/generate_neosusxestisi.py
=============================
ACF/PACF figures — English titles, fixed y-axis ±1, side-by-side layout.

Outputs (thesis_output/):
  fig_acf_pacf_price_neosusxestisi.png
  fig_acf_pacf_load_neosusxestisi.png

Run:
    conda run -n epf --no-capture-output python -m src.generate_neosusxestisi
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT   = os.path.join(_BASE, 'thesis_output')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'axes.grid'        : True,
    'grid.color'       : '#e0e0e0',
    'grid.linewidth'   : 0.6,
    'axes.axisbelow'   : True,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
})

C_BLUE   = '#2176ae'
C_ORANGE = '#e8871e'
C_RED    = '#c0392b'


def _load():
    _d = os.path.join(_BASE, 'data', 'processed')
    dfp = pd.read_parquet(os.path.join(_d, 'hourly.parquet'))
    dfl = pd.read_parquet(os.path.join(_d, 'hourly_load.parquet'))
    dfp.index = pd.to_datetime(dfp.index)
    dfl.index = pd.to_datetime(dfl.index)
    return dfp['y'].dropna(), dfl['y'].dropna()


def _make(series, fname, suptitle):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    LAGS = 175

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    # ── ACF (left) ────────────────────────────────────────────────────────
    plot_acf(series.values, lags=LAGS, ax=ax1, alpha=0.05,
             color=C_BLUE, vlines_kwargs={'colors': C_BLUE}, use_vlines=True)
    ax1.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Lag (hours)')
    ax1.set_ylabel('ACF')
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xlim(-1, LAGS + 4)
    ax1.spines[['top', 'right']].set_visible(False)
    for lag, lbl in [(24, '24h'), (48, '48h'), (168, '168h\n(1 week)')]:
        ax1.axvline(lag, color=C_RED, lw=1.0, ls='--', alpha=0.8)
        ax1.text(lag + 1.5, 0.92, lbl, fontsize=7.5, color=C_RED, va='top')

    # ── PACF (right) ──────────────────────────────────────────────────────
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(series), size=min(8000, len(series)), replace=False)
    ssub = series.iloc[sorted(idx)]

    try:
        plot_pacf(ssub.values, lags=LAGS, ax=ax2, alpha=0.05,
                  color=C_ORANGE, method='ywmle', use_vlines=True,
                  vlines_kwargs={'colors': C_ORANGE})
        pacf_lags = LAGS
    except Exception:
        plot_pacf(ssub.values, lags=60, ax=ax2, alpha=0.05,
                  color=C_ORANGE, method='ywmle', use_vlines=True,
                  vlines_kwargs={'colors': C_ORANGE})
        pacf_lags = 60

    ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Lag (hours)')
    ax2.set_ylabel('PACF')
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlim(-1, pacf_lags + 4)
    ax2.spines[['top', 'right']].set_visible(False)
    for lag, lbl in [(24, '24h'), (48, '48h'), (168, '168h\n(1 week)')]:
        if lag <= pacf_lags:
            ax2.axvline(lag, color=C_RED, lw=1.0, ls='--', alpha=0.8)
            ax2.text(lag + 1.5, 0.92, lbl, fontsize=7.5, color=C_RED, va='top')

    plt.tight_layout()
    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    kb = os.path.getsize(path) // 1024
    print(f'  ✓  {fname}  ({kb} KB)')


def main():
    print('=' * 60)
    print('  ACF/PACF — neosusxestisi (English, fixed ±1 y-axis)')
    print('=' * 60)

    print('\nLoading data...')
    price, load = _load()
    print(f'  price: {len(price):,}   load: {len(load):,}')

    print('\nPrice ACF/PACF...')
    _make(price,
          'fig_acf_pacf_price_neosusxestisi.png',
          'Autocorrelation Analysis — Day-Ahead Electricity Price')

    print('\nLoad ACF/PACF...')
    _make(load,
          'fig_acf_pacf_load_neosusxestisi.png',
          'Autocorrelation Analysis — System Load')

    print(f'\nDone. Saved to: {OUT}')


if __name__ == '__main__':
    main()
