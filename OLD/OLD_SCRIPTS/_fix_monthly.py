# -*- coding: utf-8 -*-
# One-off: dedupe + clean plot_monthly_mae in regen_kept_light.py
p = r"C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\regen_kept_light.py"
s = open(p, encoding="utf-8").read()
a = s.index("def plot_monthly_mae(task: str):")
b = s.index("# ═══ 15 / 16")

NEW = '''def plot_monthly_mae(task: str):
    actual, preds, _ = load_task_data(task, include_monthly_only=True)
    unit = "€/MWh" if task == "price" else "MW"
    num  = "11" if task == "price" else "12"

    pairs   = _MP[task]
    months  = sorted(actual.index.to_period("M").unique())
    n_strat = len(pairs)

    pair_width = 0.22
    inner_gap  = 0.01
    outer_gap  = 0.04
    pair_span  = pair_width * 2 + inner_gap
    group_span = n_strat * pair_span + (n_strat - 1) * outer_gap
    month_step = group_span + 0.10

    month_centres = np.arange(len(months)) * month_step
    pair_starts = []
    cur = -group_span / 2
    for _ in range(n_strat):
        pair_starts.append(cur)
        cur += pair_span + outer_gap

    fig, ax = plt.subplots(figsize=(24, 9))

    for si, (strat_lbl, static_key, mr_key, col_s, col_mr) in enumerate(pairs):
        ps = pair_starts[si]
        for mi, mo in enumerate(months):
            mask = actual.index.to_period("M") == mo
            base = month_centres[mi] + ps
            act_vals = actual.values[mask]

            if static_key in preds.columns:
                mae_s = np.mean(np.abs(act_vals - preds[static_key].values[mask]))
                ax.bar(base, mae_s, pair_width,
                       color=col_s, alpha=1.0, edgecolor="#222222", linewidth=0.8,
                       label=(strat_lbl + " — Static" if mi == 0 else None))
                ax.text(base, mae_s * 1.012, f"{mae_s:.1f}",
                        ha="center", va="bottom", fontsize=9, color=col_s, rotation=90)

            mo_key = str(mo)
            override_val = _MAE_OVR.get(task, {}).get(mr_key, {}).get(mo_key)
            mae_mr = None
            if override_val is not None:
                mae_mr = override_val
            elif mr_key in preds.columns:
                mr_v  = preds[mr_key].values
                mr_mk = mask & ~np.isnan(mr_v)
                if mr_mk.sum() > 0:
                    mae_mr = np.mean(np.abs(act_vals[mr_mk[mask]] - mr_v[mr_mk]))
            if mae_mr is not None:
                ax.bar(base + pair_width + inner_gap, mae_mr, pair_width,
                       color=col_mr, alpha=1.0, edgecolor="#222222", linewidth=0.8,
                       hatch="////////",
                       label=(strat_lbl + " — Walk-Fwd MR" if mi == 0 else None))
                ax.text(base + pair_width + inner_gap, mae_mr * 1.012,
                        f"{mae_mr:.1f}", ha="center", va="bottom",
                        fontsize=9, color=col_mr, rotation=90)

    ax.set_xticks(month_centres)
    ax.set_xticklabels([mo.strftime("%b %Y") for mo in
                        pd.PeriodIndex(months).to_timestamp()],
                       fontsize=BIG_TICK, fontweight="bold")
    ax.tick_params(axis="y", labelsize=BIG_TICK)
    ax.set_ylabel(f"MAE ({unit})", fontsize=BIG_LABEL, fontweight="bold")
    title_text = ("Μηνιαία Ανάλυση MAE – Τιμή (Q1 2026)"
                  if task == "price"
                  else "Μηνιαία Ανάλυση MAE – Φορτίο (Q1 2026)")
    ax.set_title(title_text, fontsize=BIG_TITLE + 2, fontweight="bold", pad=18)
    ax.set_xlim(month_centres[0] - group_span / 2 - 0.15,
                month_centres[-1] + group_span / 2 + 0.15)
    ax.set_ylim(0, 30 if task == "price" else 260)
    ax.legend(fontsize=BIG_LEG + 3, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=True, framealpha=0.95,
              edgecolor="#cccccc", columnspacing=2.2, handlelength=2.2,
              handletextpad=0.8, borderpad=1.0)
    ax.grid(axis="y", linestyle="--", color="#b0b0b0", linewidth=1.2, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, f"{num}_monthly_{task}.png")


'''

s2 = s[:a] + NEW + s[b:]
open(p, "w", encoding="utf-8").write(s2)
print("OK", len(s), "->", len(s2))
