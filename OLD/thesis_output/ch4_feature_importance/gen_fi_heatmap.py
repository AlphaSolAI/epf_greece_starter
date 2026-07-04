import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

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


# 1 & 2. GroupBy & Sum κατά ΤΕΛΙΚΟ LABEL (ενοποιεί y_lagK + load_lagK -> Load_lagK κ.λπ.)
aggregated = {}
for x in raw_rows:
    task = x['task']
    strat = x['strategy']
    model = x['model']
    imp_type = x['importance_type']
    feat = label(x['feature'].strip(), task)   # ομογενοποίηση στο επίπεδο του label

    key = (task, strat, model, imp_type, feat)
    if key not in aggregated:
        aggregated[key] = {
            'task': task, 'strategy': strat, 'model': model,
            'importance_type': imp_type, 'feature': feat,
            'share_pct': 0.0, 'importance': 0.0
        }
    aggregated[key]['share_pct'] += float(x['share_pct'])
    aggregated[key]['importance'] += float(x['importance'])

# 3. Επανυπολογισμός rankings ανά στήλη με βάση το αθροιστικό share_pct
columns = {}
for key, val in aggregated.items():
    col_key = key[:4]  # (task, strat, model, imp_type)
    columns.setdefault(col_key, []).append(val)

rows = []
for col_key, col_rows in columns.items():
    col_rows.sort(key=lambda x: x['share_pct'], reverse=True)
    for rnk, row_data in enumerate(col_rows, 1):
        row_data['rank'] = rnk
        rows.append(row_data)

STRATS = ['Teacher-Forcing', 'Recursive', 'MIMO', 'Direct']
CHAMP = {'Teacher-Forcing': 'lgbm', 'Recursive': 'lgbm', 'MIMO': 'xgb', 'Direct': 'lgbm'}
COLHDR = {'Teacher-Forcing': 'TF\n(LGBM)', 'Recursive': 'Recursive\n(LGBM)', 'MIMO': 'MIMO\n(XGB)', 'Direct': 'Direct\n(LGBM)'}


def build(task):
    share = {}; rank = {}
    for s in STRATS:
        sub = [x for x in rows if x['task'] == task and x['strategy'] == s and x['model'] == CHAMP[s]]
        share[s] = {x['feature']: float(x['share_pct']) for x in sub}
        rank[s] = {x['feature']: int(x['rank']) for x in sub}

    feats = set().union(*[set(share[s]) for s in STRATS])
    nmax = max(len(rank[s]) for s in STRATS) + 1
    meanrank = {f: np.mean([rank[s].get(f, nmax) for s in STRATS]) for f in feats}

    # 4. Rank-based union of top-3 ανά στήλη
    guaranteed = set()
    for s in STRATS:
        top = sorted(share[s], key=lambda f: rank[s].get(f, nmax))[:3]
        guaranteed.update(top)

    sel = sorted(guaranteed, key=lambda f: meanrank[f])
    if len(sel) > 10:
        sel = sel[:10]
    else:
        pool = sorted(feats, key=lambda f: meanrank[f])
        for f in pool:
            if f not in sel: sel.append(f)
            if len(sel) == 10: break

    sel = sorted(sel, key=lambda f: meanrank[f])
    M = np.full((len(sel), len(STRATS)), np.nan)
    for i, f in enumerate(sel):
        for j, s in enumerate(STRATS):
            if f in share[s]: M[i, j] = share[s][f]
    return sel, M


def plot(task, fname, title):
    sel, M = build(task)
    labels = sel  # ήδη labeled
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    Mm = np.ma.masked_invalid(M)
    cmap = plt.cm.magma_r.copy(); cmap.set_bad('#f2f2f2')
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=np.nanmax(M))
    im = ax.imshow(Mm, aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(len(STRATS))); ax.set_xticklabels([COLHDR[s] for s in STRATS], fontsize=10)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(np.arange(-.5, len(STRATS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(sel), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)

    for i in range(len(sel)):
        for j in range(len(STRATS)):
            v = M[i, j]
            if np.isnan(v): continue
            txt = '' if v < 0.5 else f'{v:.1f}'
            r, g, b, _ = cmap(norm(v))
            ax.text(j, i, txt, ha='center', va='center', fontsize=8.5,
                    color='white' if (0.299*r+0.587*g+0.114*b) < 0.5 else '#222')

    ax.set_title(title, fontsize=12, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('Σημαντικότητα (% share)', fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT+fname, dpi=200, facecolor='white', bbox_inches='tight')
    print('saved', fname, '->', labels)


plot('price', 'fig_fi_C_heatmap_price.png', 'Σημαντικότητα χαρακτηριστικών ανά στρατηγική — Τιμή DAM')
plot('load', 'fig_fi_C_heatmap_load.png', 'Σημαντικότητα χαρακτηριστικών ανά στρατηγική — Ηλεκτρικό φορτίο')
