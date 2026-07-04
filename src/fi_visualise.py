"""
Feature Importance — Polished Visualisations
Reads feature_importance/feature_importance_{price,load}.csv and produces:
  fi_viz_price.png          — side-by-side bars, all 8 models (price)
  fi_viz_load.png           — side-by-side bars, all 7 models (load)
  fi_viz_strategy_shift.png — how importance of key features shifts CL→OL→MIMO
  fi_viz_categories.png     — category breakdown comparison (price + load)
  fi_viz_topN_comparison.png— top-10 features, all models overlaid (grouped)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
OUT  = ROOT / "feature_importance"
OUT.mkdir(exist_ok=True)

# ─── global style ─────────────────────────────────────────────────────────────
BG  = "#0f172a"   # slate-900
BG2 = "#1e293b"   # slate-800  (panel bg)
FG  = "#e2e8f0"   # slate-200
GR  = "#334155"   # grid lines

rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     BG2,
    "axes.edgecolor":     GR,
    "axes.labelcolor":    FG,
    "text.color":         FG,
    "xtick.color":        FG,
    "ytick.color":        FG,
    "grid.color":         GR,
    "grid.linewidth":     0.6,
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "figure.titlesize":   13,
})

# ─── category palette ─────────────────────────────────────────────────────────
CAT_COLORS = {
    "Price lag":                  "#38bdf8",   # sky-400
    "Price lag (intraday dense)": "#7dd3fc",   # sky-300
    "Price rolling":              "#0369a1",   # sky-700
    "Load / Load-FC":             "#4ade80",   # green-400
    "Residual load lag":          "#86efac",   # green-300
    "Gas price lag":              "#fb923c",   # orange-400
    "CO₂ price lag":              "#ea580c",   # orange-600
    "Solar gen lag":              "#fbbf24",   # amber-400
    "Wind gen lag":               "#a3e635",   # lime-400
    "Calendar / Time":            "#c084fc",   # purple-400
    "Other":                      "#94a3b8",   # slate-400
}

# Strategy-level accent colours (for comparison charts)
STRAT_COL = {
    "CL — LightGBM":         "#38bdf8",
    "CL — XGBoost":          "#f97316",
    "CL — RandomForest":     "#4ade80",
    "OL — LGBM Daily-Opt":   "#818cf8",
    "OL — LGBM-SS Daily":    "#818cf8",
    "OL — XGB Dense+SS":     "#fb923c",
    "Direct — LGBM Dense":   "#34d399",
    "MIMO — XGB Dense":      "#f59e0b",
    "MIMO — RF (two-stage)": "#f87171",
    "MIMO — RF":             "#f87171",
}

STRATEGY_BADGE = {
    "CL — LightGBM":         ("CL",    "#1d4ed8"),
    "CL — XGBoost":          ("CL",    "#1d4ed8"),
    "CL — RandomForest":     ("CL",    "#1d4ed8"),
    "OL — LGBM Daily-Opt":   ("OL",    "#7c3aed"),
    "OL — LGBM-SS Daily":    ("OL",    "#7c3aed"),
    "OL — XGB Dense+SS":     ("OL",    "#7c3aed"),
    "Direct — LGBM Dense":   ("MIMO",  "#065f46"),
    "MIMO — XGB Dense":      ("MIMO",  "#065f46"),
    "MIMO — RF (two-stage)": ("MIMO",  "#065f46"),
    "MIMO — RF":             ("MIMO",  "#065f46"),
}

def categorize(feat: str) -> str:
    f = feat.lower()
    if f.startswith("y_lag") and f not in (
        "y_lag1","y_lag2","y_lag3","y_lag6","y_lag12","y_lag24","y_lag48","y_lag168"
    ):
        return "Price lag (intraday dense)"
    if f.startswith("y_lag"):        return "Price lag"
    if f.startswith("y_roll"):       return "Price rolling"
    if f.startswith("load_lag") or f == "load_fc":
        return "Load / Load-FC"
    if f.startswith("residual_load"): return "Residual load lag"
    if f.startswith("gas"):          return "Gas price lag"
    if f.startswith("co2"):          return "CO₂ price lag"
    if f.startswith("gen_solar"):    return "Solar gen lag"
    if f.startswith("gen_wind"):     return "Wind gen lag"
    if f in ("hour","dow","is_holiday","hour_sin","hour_cos","dow_sin","dow_cos","month"):
        return "Calendar / Time"
    return "Other"

# ─── load CSVs ────────────────────────────────────────────────────────────────
def load_wide(task: str) -> pd.DataFrame:
    path = OUT / f"feature_importance_{task}.csv"
    df = pd.read_csv(path)
    # drop the avg column if present
    df = df[[c for c in df.columns if c != "avg"]]
    return df

# ─── helper: bar chart for one model ─────────────────────────────────────────
def _draw_model_bar(ax, model_name: str, df_wide: pd.DataFrame, top_n=20):
    if model_name not in df_wide.columns:
        ax.set_visible(False)
        return
    sub = df_wide[["feature", "category", model_name]].copy()
    sub = sub.sort_values(model_name, ascending=False).head(top_n)

    colors = [CAT_COLORS.get(c, "#94a3b8") for c in sub["category"]]
    vals   = sub[model_name].values
    feats  = sub["feature"].values

    bars = ax.barh(range(len(sub)), vals, color=colors,
                   edgecolor=BG, linewidth=0.4, height=0.75)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(feats, fontsize=7.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    ax.grid(axis="x", alpha=0.35)
    ax.set_xlabel("Normalised importance", fontsize=8, color=FG)

    # strategy badge in title
    badge, bcol = STRATEGY_BADGE.get(model_name, ("", BG2))
    ax.set_title(f"[{badge}]  {model_name}", fontsize=8.5, fontweight="bold",
                 pad=5, color=FG,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=bcol, alpha=0.6, edgecolor="none"))

    # annotate top-2 bars with value
    for i, (val, b) in enumerate(zip(vals[:2], bars[:2])):
        ax.text(val + 0.003, i, f"{val*100:.1f}%", va="center",
                fontsize=7, color=FG, fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 & 2: Side-by-side bars per model (price + load)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_model_bars(task: str, model_order: list[str]):
    df = load_wide(task)
    n  = len(model_order)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 7.5 * nrows),
                             facecolor=BG)
    fig.subplots_adjust(hspace=0.55, wspace=0.38)

    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    for i, mname in enumerate(model_order):
        _draw_model_bar(axes_flat[i], mname, df, top_n=18)

    # hide extra axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # ── legend ───────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in CAT_COLORS.items()
        if lbl in df["category"].values
    ]
    fig.legend(handles=legend_handles, title="Feature category",
               title_fontsize=9, fontsize=8,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02),
               facecolor=BG2, edgecolor=GR, labelcolor=FG,
               framealpha=0.9)

    task_label = "Price (€/MWh)" if task == "price" else "Load (MW)"
    fig.suptitle(f"Feature Importance — {task_label}   |   Top-18 features per model",
                 fontsize=13, fontweight="bold", color=FG, y=1.01)

    path = OUT / f"fi_viz_{task}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3: Strategy shift — how key features change CL → OL → MIMO
# ═══════════════════════════════════════════════════════════════════════════════
def plot_strategy_shift():
    price_df = load_wide("price")
    load_df  = load_wide("load")

    # Define ordered sequence for each task
    price_seq = [
        ("CL",   "CL — LightGBM"),
        ("CL",   "CL — XGBoost"),
        ("CL",   "CL — RandomForest"),
        ("OL",   "OL — LGBM Daily-Opt"),
        ("OL",   "OL — XGB Dense+SS"),
        ("MIMO", "Direct — LGBM Dense"),
        ("MIMO", "MIMO — XGB Dense"),
        ("MIMO", "MIMO — RF (two-stage)"),
    ]
    load_seq = [
        ("CL",   "CL — LightGBM"),
        ("CL",   "CL — XGBoost"),
        ("CL",   "CL — RandomForest"),
        ("OL",   "OL — LGBM-SS Daily"),
        ("MIMO", "Direct — LGBM Dense"),
        ("MIMO", "MIMO — XGB Dense"),
        ("MIMO", "MIMO — RF"),
    ]

    # Key features to track
    PRICE_KEY_FEATS = [
        ("y_lag1",     "Price lag",          "#38bdf8"),
        ("y_lag24",    "Price lag",          "#0ea5e9"),
        ("y_lag168",   "Price lag",          "#0369a1"),
        ("y_roll24",   "Price rolling",      "#1d4ed8"),
        ("gen_wind_lag24", "Wind gen lag",   "#a3e635"),
        ("gas_lag168", "Gas price lag",      "#fb923c"),
        ("hour_cos",   "Calendar / Time",    "#c084fc"),
    ]
    LOAD_KEY_FEATS = [
        ("y_lag1",     "Load lag",           "#38bdf8"),
        ("load_lag1",  "Load / Load-FC",     "#4ade80"),
        ("y_lag24",    "Load lag",           "#0ea5e9"),
        ("y_roll24",   "Rolling",            "#1d4ed8"),
        ("y_roll168",  "Rolling",            "#0369a1"),
        ("hour_cos",   "Calendar / Time",    "#c084fc"),
        ("hour",       "Calendar / Time",    "#a78bfa"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.subplots_adjust(wspace=0.08)

    for ax, (task, seq, key_feats, df) in zip(axes, [
        ("Price (€/MWh)", price_seq, PRICE_KEY_FEATS, price_df),
        ("Load (MW)",     load_seq,  LOAD_KEY_FEATS,  load_df),
    ]):
        xlabels = [m for _, m in seq]
        x       = np.arange(len(xlabels))

        for fname, cat, color in key_feats:
            if fname not in df["feature"].values:
                continue
            row = df.loc[df["feature"] == fname]
            vals = []
            for _, mname in seq:
                if mname in row.columns:
                    vals.append(float(row[mname].iloc[0]) if mname in row.columns else 0.0)
                else:
                    vals.append(0.0)
            ax.plot(x, [v * 100 for v in vals],
                    marker="o", markersize=6, linewidth=2.0,
                    color=color, label=fname, zorder=3)
            ax.fill_between(x, [v * 100 for v in vals],
                            alpha=0.08, color=color)

        # vertical strategy separators
        strategy_groups = {}
        for i, (strat, _) in enumerate(seq):
            strategy_groups.setdefault(strat, []).append(i)

        prev_strat = None
        for i, (strat, _) in enumerate(seq):
            if strat != prev_strat and i > 0:
                ax.axvline(i - 0.5, color=GR, linewidth=1.2, linestyle="--", alpha=0.6)
            prev_strat = strat

        # strategy labels at top
        for strat, indices in strategy_groups.items():
            mid = np.mean(indices)
            col = {"CL": "#3b82f6", "OL": "#a855f7", "MIMO": "#10b981"}[strat]
            ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 90,
                    strat, ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=col,
                    transform=ax.get_xaxis_transform())

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" — ", "\n") for m in xlabels],
                           fontsize=7.5, rotation=0, ha="center")
        ax.set_ylabel("Normalised importance (%)", fontsize=9)
        ax.set_title(f"{task}", fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right",
                  facecolor=BG2, edgecolor=GR, labelcolor=FG, framealpha=0.9,
                  ncol=1)
        ax.set_ylim(bottom=0)

    fig.suptitle("Feature Importance Shift: CL → OL → MIMO  (how key features change across strategies)",
                 fontsize=12, fontweight="bold", color=FG, y=1.02)

    path = OUT / "fi_viz_strategy_shift.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4: Category breakdown — polished stacked bars
# ═══════════════════════════════════════════════════════════════════════════════
def plot_category_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor=BG)
    fig.subplots_adjust(wspace=0.12)

    for ax, task in zip(axes, ["price", "load"]):
        df = load_wide(task)
        model_cols = [c for c in df.columns if c not in ("feature", "category", "avg")]

        # Group by category for each model
        cat_data = {}
        for cat in CAT_COLORS:
            rows = df[df["category"] == cat]
            cat_data[cat] = rows[model_cols].sum()

        cat_df = pd.DataFrame(cat_data).T   # shape: (n_cats, n_models)

        x      = np.arange(len(model_cols))
        bottom = np.zeros(len(model_cols))

        for cat, color in CAT_COLORS.items():
            if cat not in cat_df.index:
                continue
            vals = cat_df.loc[cat, model_cols].values.astype(float)
            if vals.sum() < 1e-6:
                continue
            bars = ax.bar(x, vals * 100, bottom=bottom * 100, color=color,
                          label=cat, edgecolor=BG, linewidth=0.5, width=0.72)
            # annotate if > 5%
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.05:
                    ax.text(xi, (b + v / 2) * 100, f"{v*100:.0f}%",
                            ha="center", va="center", fontsize=6.5,
                            color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" — ", "\n") for m in model_cols],
                           fontsize=8, rotation=0, ha="center")
        ax.set_ylabel("Cumulative importance (%)", fontsize=9)
        task_label = "Price (€/MWh)" if task == "price" else "Load (MW)"
        ax.set_title(f"Feature Category Breakdown — {task_label}",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(0, 102)
        ax.grid(axis="y", alpha=0.3)

        # Strategy bracket annotations
        prev, start = None, 0
        model_strats = []
        for m in model_cols:
            if "CL" in m:   model_strats.append("CL")
            elif "OL" in m: model_strats.append("OL")
            else:           model_strats.append("MIMO")

        strat_col_map = {"CL": "#3b82f6", "OL": "#a855f7", "MIMO": "#10b981"}
        for i, s in enumerate(model_strats + [None]):
            if s != prev and prev is not None:
                mid = (start + i - 1) / 2
                ax.annotate(prev,
                            xy=(mid, 103), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color=strat_col_map[prev],
                            xytext=(0, 4), textcoords="offset points")
                if i < len(model_strats):
                    ax.axvline(i - 0.5, color=GR, linewidth=1, linestyle=":")
                start = i
            elif prev is None:
                start = 0
            prev = s

    # Legend (shared)
    handles = [mpatches.Patch(color=c, label=lbl)
               for lbl, c in CAT_COLORS.items()
               if any(lbl in load_wide(t)["category"].values for t in ["price","load"])]
    fig.legend(handles=handles, title="Feature category",
               title_fontsize=9, fontsize=8,
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.07),
               facecolor=BG2, edgecolor=GR, labelcolor=FG, framealpha=0.9)

    fig.suptitle("Feature Category Contribution per Model & Strategy  (Price vs Load)",
                 fontsize=12, fontweight="bold", color=FG, y=1.02)

    path = OUT / "fi_viz_categories.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5: Grouped bar — top-10 features across all models
# ═══════════════════════════════════════════════════════════════════════════════
def plot_topN_grouped(task: str, top_n: int = 10):
    df = load_wide(task)
    model_cols = [c for c in df.columns if c not in ("feature", "category", "avg")]

    # Pick top-N features by max importance across any model
    top_feats = (df.set_index("feature")[model_cols]
                 .max(axis=1)
                 .sort_values(ascending=False)
                 .head(top_n)
                 .index.tolist())

    sub = df[df["feature"].isin(top_feats)].copy()
    sub = sub.set_index("feature")[model_cols]
    # sort rows by mean importance
    sub = sub.loc[sub.mean(axis=1).sort_values(ascending=False).index]

    n_feats  = len(top_feats)
    n_models = len(model_cols)
    bar_w    = 0.8 / n_models
    x        = np.arange(n_feats)

    fig, ax = plt.subplots(figsize=(max(14, n_feats * 1.4), 6), facecolor=BG)
    ax.set_facecolor(BG2)

    for i, mname in enumerate(model_cols):
        offset = (i - n_models / 2 + 0.5) * bar_w
        vals   = [sub.loc[f, mname] * 100 if f in sub.index else 0.0 for f in sub.index]
        color  = STRAT_COL.get(mname, "#94a3b8")
        ax.bar(x + offset, vals, width=bar_w * 0.90,
               color=color, label=mname, edgecolor=BG, linewidth=0.3, alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"[{categorize(f)[:3].upper()}]  {f}" for f in sub.index],
        rotation=30, ha="right", fontsize=8.5
    )
    ax.set_ylabel("Normalised importance (%)", fontsize=9)
    task_label = "Price (€/MWh)" if task == "price" else "Load (MW)"
    ax.set_title(f"Top-{top_n} Features Across All Models — {task_label}",
                 fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # legend with strategy grouping
    cl_patch   = mpatches.Patch(color="#3b82f6", label="── CL (Teacher-Forcing)")
    ol_patch   = mpatches.Patch(color="#a855f7", label="── OL (Recursive)")
    mimo_patch = mpatches.Patch(color="#10b981", label="── MIMO / Direct")
    model_handles = [plt.Rectangle((0,0),1,1, color=STRAT_COL.get(m,"#94a3b8"),
                                    label=m) for m in model_cols]
    ax.legend(handles=[cl_patch, ol_patch, mimo_patch] + model_handles,
              fontsize=7.5, loc="upper right",
              facecolor=BG2, edgecolor=GR, labelcolor=FG, framealpha=0.9,
              ncol=2)

    path = OUT / f"fi_viz_topN_{task}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6: MIMO per-horizon heatmap (polished)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_mimo_horizons_polished():
    """Re-read fi_mimo_horizons data from the LGBM Direct dense bundle."""
    import pickle, joblib
    path = ROOT / "models" / "lgbm_direct_hourly_price_h24_dense.pkl"
    if not path.exists():
        print("  [SKIP] lgbm_direct_hourly_price_h24_dense.pkl not found")
        return

    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    feat_cols = bundle["feature_cols"]
    estimators = model.estimators_

    # per-horizon importance matrix (H × n_features)
    all_imps = []
    for est in estimators:
        imps = est.feature_importances_.astype(float)
        all_imps.append(imps / (imps.sum() + 1e-12))
    mat = np.array(all_imps)   # (24, n_feats)

    # top-20 features by mean
    mean_imp = mat.mean(axis=0)
    top20_idx = np.argsort(mean_imp)[::-1][:20]
    top20_feats = [feat_cols[i] for i in top20_idx]
    mat_top = mat[:, top20_idx].T   # (20, 24)

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
    ax.set_facecolor(BG2)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "h_imp", ["#1e293b", "#0369a1", "#38bdf8", "#fbbf24"]
    )
    im = ax.imshow(mat_top, aspect="auto", cmap=cmap, vmin=0)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"h+{h+1}" for h in range(24)], fontsize=8, rotation=45)
    ax.set_yticks(range(20))
    ylabels = [f"[{categorize(f)[:3].upper()}]  {f}" for f in top20_feats]
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("Forecast horizon (hours ahead)", fontsize=9)
    ax.set_title("LGBM Direct H=24 (Dense) — Feature Importance per Forecast Horizon  (top 20 features)",
                 fontsize=11, fontweight="bold", pad=10)

    # colour y-tick labels by category
    from matplotlib.colors import to_rgba
    for tick, feat in zip(ax.get_yticklabels(), top20_feats):
        cat = categorize(feat)
        tick.set_color(CAT_COLORS.get(cat, "#94a3b8"))

    cb = plt.colorbar(im, ax=ax, label="Norm. importance", shrink=0.7,
                      pad=0.01)
    cb.ax.yaxis.label.set_color(FG)
    cb.ax.tick_params(colors=FG)

    plt.tight_layout()
    path2 = OUT / "fi_viz_mimo_horizons.png"
    fig.savefig(path2, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path2.name}")


# ─── run all ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating polished feature importance visualisations …\n")

    # --- Chart 1: Price bars ---
    price_order = [
        "CL — LightGBM", "CL — XGBoost", "CL — RandomForest",
        "OL — LGBM Daily-Opt", "OL — XGB Dense+SS",
        "Direct — LGBM Dense", "MIMO — XGB Dense", "MIMO — RF (two-stage)",
    ]
    plot_model_bars("price", price_order)

    # --- Chart 2: Load bars ---
    load_order = [
        "CL — LightGBM", "CL — XGBoost", "CL — RandomForest",
        "OL — LGBM-SS Daily",
        "Direct — LGBM Dense", "MIMO — XGB Dense", "MIMO — RF",
    ]
    plot_model_bars("load", load_order)

    # --- Chart 3: Strategy shift ---
    plot_strategy_shift()

    # --- Chart 4: Category breakdown ---
    plot_category_breakdown()

    # --- Chart 5: Top-N grouped ---
    plot_topN_grouped("price", top_n=12)
    plot_topN_grouped("load",  top_n=12)

    # --- Chart 6: MIMO per-horizon ---
    plot_mimo_horizons_polished()

    print(f"\nAll charts saved to: {OUT}")
