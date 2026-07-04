"""Grouped horizontal bar charts (Είδος E) — ίδια top-10 features με τα D heatmaps,
ώστε να τοποθετούνται διπλα-διπλα με αυτά. Χρώματα οικογενειών (locked palette).
Aggregation κατά ΤΕΛΙΚΟ label (ενοποιεί y_lagK + load_lagK -> Load_lagK κ.λπ.)."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\thesis_output\ch4_feature_importance\appendix_fi_data.csv'
OUT = r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\thesis_output\ch4_feature_importance' + '\\'

raw_rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))

# Locked family palette (όπως dashboard / skill §7)
FAMILY_COL = {'lgbm': '#38bdf8', 'xgb': '#f97316', 'rf': '#22c55e', 'mlp': '#a855f7', 'svr': '#ef4444'}
MODEL_LABEL = {'lgbm': 'LightGBM (gain)', 'xgb': 'XGBoost (gain)', 'rf': 'RF (MDI)',
               'mlp': 'MLP (perm)', 'svr': 'SVR (perm)'}


def label(f, task):
    f = f.lower()
    p = 'Price' if task == 'price' else 'Load'
    m = {
        'y_lag': p+'_lag', 'y_roll': p+'_roll', 'gen_wind_lag': 'Wind_lag', 'gen_solar_lag': 'Solar_lag',
        'gas_lag': 'Gas_lag', 'load_lag': 'Load_lag', 'res_load_lag': 'Res_load_lag',
        'residual_load_lag': 'Res_load_lag', 'wind_lag': 'Wind_lag', 'solar_lag': 'Solar_lag',
        'gen_fc_dayahead_lag': 'Gen_fc_lag',
    }
    for k, v in m.items():
        if f.startswith(k) and f[len(k):].isdigit():
            return v + f[len(k):]
    nm = {
        'gas_price': 'Gas_price', 'co2_price': 'Co2_price', 'load_fc': 'Load_fc', 'is_holiday': 'Is_holiday',
        'gen_fc_dayahead': 'Gen_fc', 'hour': 'Hour', 'hour_sin': 'Hour_sin', 'hour_cos': 'Hour_cos',
        'dow': 'Dow', 'dow_sin': 'Dow_sin', 'dow_cos': 'Dow_cos',
        'w_larissa_shortwave_radiation': 'Rad_Larissa', 'w_gr_mean_relative_humidity_2m': 'Hum_GR_mean',
    }
    lbl = nm.get(f, f)
    if lbl and lbl[0].islower():
        lbl = lbl[0].upper() + lbl[1:]
    return lbl


# GroupBy & Sum κατά label
aggregated = {}
for x in raw_rows:
    key = (x['task'], x['strategy'], x['model'], label(x['feature'].strip(), x['task']))
    aggregated[key] = aggregated.get(key, 0.0) + float(x['share_pct'])

# rankings ανά (task, strat, model)
columns = {}
for (task, strat, model, feat), v in aggregated.items():
    columns.setdefault((task, strat, model), []).append((feat, v))
ranks = {}
for col_key, lst in columns.items():
    lst.sort(key=lambda z: -z[1])
    for rnk, (feat, v) in enumerate(lst, 1):
        ranks[(col_key, feat)] = rnk

STRATS = ['Teacher-Forcing', 'Recursive', 'MIMO', 'Direct']
MODELS_PER_STRAT = {
    'Teacher-Forcing': ['lgbm', 'xgb', 'rf', 'mlp', 'svr'],
    'Recursive': ['lgbm', 'xgb', 'rf', 'mlp', 'svr'],
    'MIMO': ['xgb', 'rf', 'mlp', 'svr'],
    'Direct': ['lgbm', 'xgb'],
}


def select(task, strat):
    """Ίδιος rank-based κανόνας (union top-3) με τα heatmaps -> ίδια 10 features."""
    models = MODELS_PER_STRAT[strat]
    share = {m: {} for m in models}
    for (t, s, mo, feat), v in aggregated.items():
        if t == task and s == strat and mo in models:
            share[mo][feat] = v
    all_feats = set().union(*[set(share[m]) for m in models])
    nmax = max([len(share[m]) for m in models] + [100]) + 1
    meanrank = {f: np.mean([ranks.get(((task, strat, m), f), nmax) for m in models]) for f in all_feats}
    guaranteed = set()
    for m in models:
        top = sorted(share[m], key=lambda f: ranks.get(((task, strat, m), f), nmax))[:3]
        guaranteed.update(top)
    sel = sorted(guaranteed, key=lambda f: meanrank[f])
    if len(sel) > 10:
        sel = sel[:10]
    else:
        for f in sorted(all_feats, key=lambda f: meanrank[f]):
            if f not in sel:
                sel.append(f)
            if len(sel) == 10:
                break
    sel = sorted(sel, key=lambda f: meanrank[f])     # πιο σημαντικά πάνω
    return sel, models, share


def plot(task, strat, fname, title):
    sel, models, share = select(task, strat)
    sel_disp = list(reversed(sel))                    # barh: πιο σημαντικό στην κορυφή
    n_feat, n_mod = len(sel_disp), len(models)
    y = np.arange(n_feat)
    bar_h = 0.8 / n_mod

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    for j, m in enumerate(models):
        vals = [share[m].get(f, 0.0) for f in sel_disp]
        offs = (j - (n_mod - 1) / 2) * bar_h
        ax.barh(y + offs, vals, height=bar_h, color=FAMILY_COL[m],
                label=MODEL_LABEL[m], edgecolor='white', linewidth=0.4)

    ax.set_yticks(y); ax.set_yticklabels(sel_disp, fontsize=10)
    ax.set_xlabel('Σημαντικότητα (% share)', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(axis='x', color='#d9d9d9', linestyle=(0, (1, 2)), linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#444'); ax.spines['bottom'].set_color('#444')
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, loc='lower right', frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT + fname, dpi=200, facecolor='white', bbox_inches='tight')
    print('saved', fname, '->', sel)


files = {
    ('price', 'Teacher-Forcing'): ('fig_fi_E_tf_price.png', 'Σύγκριση ανά μοντέλο — Teacher-Forcing — Τιμή DAM'),
    ('load', 'Teacher-Forcing'): ('fig_fi_E_tf_load.png', 'Σύγκριση ανά μοντέλο — Teacher-Forcing — Ηλεκτρικό φορτίο'),
    ('price', 'Recursive'): ('fig_fi_E_rec_price.png', 'Σύγκριση ανά μοντέλο — Recursive — Τιμή DAM'),
    ('load', 'Recursive'): ('fig_fi_E_rec_load.png', 'Σύγκριση ανά μοντέλο — Recursive — Ηλεκτρικό φορτίο'),
    ('price', 'MIMO'): ('fig_fi_E_mimo_price.png', 'Σύγκριση ανά μοντέλο — MIMO — Τιμή DAM'),
    ('load', 'MIMO'): ('fig_fi_E_mimo_load.png', 'Σύγκριση ανά μοντέλο — MIMO — Ηλεκτρικό φορτίο'),
    ('price', 'Direct'): ('fig_fi_E_direct_price.png', 'Σύγκριση ανά μοντέλο — Direct — Τιμή DAM'),
    ('load', 'Direct'): ('fig_fi_E_direct_load.png', 'Σύγκριση ανά μοντέλο — Direct — Ηλεκτρικό φορτίο'),
}

for (t, s), (fname, title) in files.items():
    plot(t, s, fname, title)
