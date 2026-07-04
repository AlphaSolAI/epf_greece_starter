import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\thesis_output\ch4_feature_importance\appendix_fi_data.csv'
OUT = r'C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\thesis_output\ch4_feature_importance' + '\\'

raw_rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))


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
        'w_larissa_shortwave_radiation': 'Rad_Larissa',
        'w_gr_mean_relative_humidity_2m': 'Hum_GR_mean',
    }
    lbl = nm.get(f, f)
    if lbl and lbl[0].islower():
        lbl = lbl[0].upper() + lbl[1:]
    return lbl


# 1 & 2. GroupBy & Sum κατά ΤΕΛΙΚΟ LABEL (ενοποιεί y_lagK + load_lagK κ.λπ.)
aggregated = {}
for x in raw_rows:
    task = x['task']
    strat = x['strategy']
    model = x['model']
    imp_type = x['importance_type']
    feat = label(x['feature'].strip(), task)

    key = (task, strat, model, imp_type, feat)
    if key not in aggregated:
        aggregated[key] = {
            'task': task, 'strategy': strat, 'model': model,
            'importance_type': imp_type, 'feature': feat,
            'share_pct': 0.0, 'importance': 0.0
        }
    aggregated[key]['share_pct'] += float(x['share_pct'])
    aggregated[key]['importance'] += float(x['importance'])

# 3. Επανυπολογισμός rankings ανά στήλη μοντέλου
columns = {}
for key, val in aggregated.items():
    col_key = key[:3]  # (task, strat, model)
    columns.setdefault(col_key, []).append(val)

rows = []
for col_key, col_rows in columns.items():
    col_rows.sort(key=lambda x: x['share_pct'], reverse=True)
    for rnk, row_data in enumerate(col_rows, 1):
        row_data['rank'] = rnk
        rows.append(row_data)

STRATS = ['Teacher-Forcing', 'Recursive', 'MIMO', 'Direct']
MODELS_PER_STRAT = {
    'Teacher-Forcing': ['lgbm', 'xgb', 'rf', 'mlp', 'svr'],
    'Recursive': ['lgbm', 'xgb', 'rf', 'mlp', 'svr'],
    'MIMO': ['xgb', 'rf', 'mlp', 'svr'],
    'Direct': ['lgbm', 'xgb']
}
MODEL_LABELS = {
    'lgbm': 'LGBM\n(gain)', 'xgb': 'XGB\n(gain)', 'rf': 'RF\n(MDI)',
    'mlp': 'MLP\n(perm)', 'svr': 'SVR\n(perm)'
}


def build_strat(task, strat):
    models = MODELS_PER_STRAT[strat]
    sub = [x for x in rows if x['task'] == task and x['strategy'] == strat]

    share = {m: {} for m in models}
    rank = {m: {} for m in models}
    for x in sub:
        m = x['model']
        if m in models:
            share[m][x['feature']] = x['share_pct']
            rank[m][x['feature']] = x['rank']

    all_feats = set().union(*[set(share[m]) for m in models])
    nmax = max([len(rank[m]) for m in models] + [100]) + 1
    meanrank = {f: np.mean([rank[m].get(f, nmax) for m in models]) for f in all_feats}

    # 4. Rank-based union of top-3 ανά στήλη μοντέλου
    guaranteed = set()
    for m in models:
        top = sorted(share[m], key=lambda f: rank[m].get(f, nmax))[:3]
        guaranteed.update(top)

    sel = sorted(guaranteed, key=lambda f: meanrank[f])
    if len(sel) > 10:
        sel = sel[:10]
    else:
        pool = sorted(all_feats, key=lambda f: meanrank[f])
        for f in pool:
            if f not in sel: sel.append(f)
            if len(sel) == 10: break

    sel = sorted(sel, key=lambda f: meanrank[f])
    M = np.full((len(sel), len(models)), np.nan)
    for i, f in enumerate(sel):
        for j, m in enumerate(models):
            if f in share[m]: M[i, j] = share[m][f]
    return sel, M, models


def plot_strat(task, strat, fname, title):
    sel, M, models = build_strat(task, strat)
    labels = sel  # ήδη labeled

    width = 3.5 + len(models) * 0.9
    fig, ax = plt.subplots(figsize=(width, 5.6))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    # Κανονικοποίηση ανά στήλη (0..1) — gain/MDI/permutation μη συγκρίσιμα σε απόλυτη κλίμακα
    M_norm = np.full_like(M, np.nan)
    for j in range(len(models)):
        col_max = np.nanmax(M[:, j])
        if col_max > 0:
            M_norm[:, j] = M[:, j] / col_max

    Mm = np.ma.masked_invalid(M_norm)
    cmap = plt.cm.magma_r.copy(); cmap.set_bad('#f2f2f2')
    im = ax.imshow(Mm, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(models))); ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=10)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(sel), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)

    for i in range(len(sel)):
        for j in range(len(models)):
            v = M[i, j]
            v_norm = M_norm[i, j]
            if np.isnan(v): continue
            txt = '' if v < 0.5 else f'{v:.1f}'
            r, g, b, _ = cmap(v_norm)
            ax.text(j, i, txt, ha='center', va='center', fontsize=8.5,
                    color='white' if (0.299*r+0.587*g+0.114*b) < 0.5 else '#222')

    '''ax.set_title(title, fontsize=11, pad=10)
    plt.figtext(0.5, 0.01, 'Xρώμα: κανονικοποιημένο ανά στήλη (μέθοδοι μη συγκρίσιμες σε απόλυτη κλίμακα) · αριθμοί: % share',
                ha='center', fontsize=7.5, color='#555', style='italic')'''''
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig.savefig(OUT+fname, dpi=200, facecolor='white', bbox_inches='tight')
    print('saved', fname, '->', labels)


files_mapping = {
    ('price', 'Teacher-Forcing'): ('fig_fi_D_tf_price.png', 'Σημαντικότητα ανά μοντέλο — Teacher-Forcing — Τιμή DAM'),
    ('load', 'Teacher-Forcing'): ('fig_fi_D_tf_load.png', 'Σημαντικότητα ανά μοντέλο — Teacher-Forcing — Ηλεκτρικό φορτίο'),
    ('price', 'Recursive'): ('fig_fi_D_rec_price.png', 'Σημαντικότητα ανά μοντέλο — Recursive — Τιμή DAM'),
    ('load', 'Recursive'): ('fig_fi_D_rec_load.png', 'Σημαντικότητα ανά μοντέλο — Recursive — Ηλεκτρικό φορτίο'),
    ('price', 'MIMO'): ('fig_fi_D_mimo_price.png', 'Σημαντικότητα ανά μοντέλο — MIMO — Τιμή DAM'),
    ('load', 'MIMO'): ('fig_fi_D_mimo_load.png', 'Σημαντικότητα ανά μοντέλο — MIMO — Ηλεκτρικό φορτίο'),
    ('price', 'Direct'): ('fig_fi_D_direct_price.png', 'Σημαντικότητα ανά μοντέλο — Direct — Τιμή DAM'),
    ('load', 'Direct'): ('fig_fi_D_direct_load.png', 'Σημαντικότητα ανά μοντέλο — Direct — Ηλεκτρικό φορτίο'),
}

for (t, s), (fname, title) in files_mapping.items():
    plot_strat(t, s, fname, title)
