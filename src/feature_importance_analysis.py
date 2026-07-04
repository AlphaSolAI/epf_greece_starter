"""
Feature Importance Analysis — Comprehensive comparison across ALL methods,
tasks (price/load), and prediction strategies (CL / OL / MIMO / Direct).

Outputs (saved in feature_importance/ folder):
  - feature_importance_price.csv  : raw importances per model (price)
  - feature_importance_load.csv   : raw importances per model (load)
  - fi_price_heatmap.png          : heatmap — top features × models (price)
  - fi_load_heatmap.png           : heatmap — top features × models (load)
  - fi_price_bars.png             : grouped bar — top 20 features each model (price)
  - fi_load_bars.png              : grouped bar — top 20 features each model (load)
  - fi_price_categories.png       : stacked bar by feature category (price)
  - fi_load_categories.png        : stacked bar by feature category (load)
  - fi_mimo_horizons.png          : LGBM Direct price — per-horizon heatmap
  - fi_summary.csv                : top-5 features per model (quick reference)
"""

import pickle, warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"
OUT   = ROOT / "feature_importance"
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature category mapping
# ─────────────────────────────────────────────────────────────────────────────
def categorize(feat: str) -> str:
    f = feat.lower()
    # Dense intraday price lags (y_lag4..y_lag23)
    if f.startswith("y_lag") and f not in (
        "y_lag1","y_lag2","y_lag3","y_lag6","y_lag12","y_lag24","y_lag48","y_lag168"
    ):
        return "Price lag (intraday dense)"
    if f.startswith("y_lag"):       return "Price lag"
    if f.startswith("y_roll"):      return "Price rolling"
    if f.startswith("load_lag") or f == "load_fc":
        return "Load / Load-FC"
    if f.startswith("residual_load"): return "Residual load lag"
    if f.startswith("gas"):         return "Gas price lag"
    if f.startswith("co2"):         return "CO₂ price lag"
    if f.startswith("gen_solar"):   return "Solar gen lag"
    if f.startswith("gen_wind"):    return "Wind gen lag"
    if f in ("hour","dow","is_holiday","hour_sin","hour_cos","dow_sin","dow_cos","month"):
        return "Calendar / Time"
    return "Other"

CATEGORY_COLORS = {
    "Price lag":                  "#3b82f6",   # blue
    "Price lag (intraday dense)": "#60a5fa",   # light blue
    "Price rolling":              "#1d4ed8",   # dark blue
    "Load / Load-FC":             "#22c55e",   # green
    "Residual load lag":          "#86efac",   # light green
    "Gas price lag":              "#f97316",   # orange
    "CO₂ price lag":              "#ea580c",   # dark orange
    "Solar gen lag":              "#fbbf24",   # yellow
    "Wind gen lag":               "#a3e635",   # lime
    "Calendar / Time":            "#a855f7",   # purple
    "Other":                      "#9ca3af",   # gray
}

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
PRICE_MODELS = [
    # (label, strategy, file, is_bundle, is_mimo)
    # is_bundle=True  → file is a dict; model stored under bundle['model']
    # is_mimo=True    → inner model is MultiOutputRegressor (average across estimators_)
    # is_mimo=False   → inner model is a single estimator (even if predicts multi-output)
    ("CL — LightGBM",         "CL",    "lgbm_hourly_price.pkl",                               False, False),
    ("CL — XGBoost",          "CL",    "xgb_hourly_price.pkl",                                False, False),
    ("CL — RandomForest",     "CL",    "rf_hourly_price.pkl",                                 False, False),
    ("OL — LGBM Daily-Opt",   "OL",    "lgbm_hourly_price_openloop_daily_optuna.pkl",         False, False),
    ("OL — XGB Dense+SS",     "OL",    "xgb_hourly_price_openloop_daily_ss_optuna_dense.pkl", False, False),
    ("Direct — LGBM Dense",   "MIMO",  "lgbm_direct_hourly_price_h24_dense.pkl",              True,  True),
    ("MIMO — XGB Dense",      "MIMO",  "xgb_hourly_price_mimo_h24_dense.pkl",                 True,  True),   # ← bundle
    ("MIMO — RF (two-stage)", "MIMO",  "rf_hourly_price_mimo_h24_with_load.pkl",              True,  False),  # ← bundle, single RF
]

LOAD_MODELS = [
    ("CL — LightGBM",         "CL",    "lgbm_hourly_load.pkl",                                False, False),
    ("CL — XGBoost",          "CL",    "xgb_hourly_load.pkl",                                 False, False),
    ("CL — RandomForest",     "CL",    "rf_hourly_load.pkl",                                  False, False),
    ("OL — LGBM-SS Daily",    "OL",    "lgbm_hourly_load_openloop_daily_ss_optuna.pkl",        False, False),
    ("Direct — LGBM Dense",   "MIMO",  "lgbm_direct_hourly_load_h24_dense.pkl",               True,  True),
    ("MIMO — XGB Dense",      "MIMO",  "xgb_hourly_load_mimo_h24_dense.pkl",                  True,  True),   # ← bundle
    ("MIMO — RF",             "MIMO",  "rf_hourly_load_mimo_h24.pkl",                         True,  False),  # ← bundle, single RF
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: load model + extract importances
# ─────────────────────────────────────────────────────────────────────────────
def load_model(fname: str, is_bundle: bool):
    path = MODELS / fname
    if not path.exists():
        print(f"  [SKIP] {fname} not found")
        return None
    if is_bundle:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return obj  # dict
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _lgbm_feats_imps(m):
    """Returns (feature_names list, importances array) for an LGBMRegressor or raw Booster."""
    # Handle both sklearn LGBMRegressor wrapper and raw lightgbm.Booster
    type_name = type(m).__name__
    if "Booster" in type_name:
        # Raw lightgbm Booster
        feats = list(m.feature_name())   # method call, not attribute
        imps  = m.feature_importance(importance_type="gain").astype(float)
    else:
        # sklearn LGBMRegressor — attribute is feature_name_ (with trailing _)
        try:
            feats = list(m.feature_name_)
        except AttributeError:
            feats = list(m.feature_name())
        imps  = m.feature_importances_.astype(float)
    return feats, imps


def _xgb_feats_imps(m, feat_cols=None):
    """Returns (feature_names list, importances array) for an XGBRegressor."""
    imps = m.feature_importances_.astype(float)
    if feat_cols is not None:
        return list(feat_cols), imps
    try:
        feats = list(m.feature_names_in_)
    except AttributeError:
        fn = m.get_booster().feature_names
        if fn is not None:
            feats = list(fn)
        else:
            # No feature names stored — return index-based placeholders
            feats = [f"f{i}" for i in range(len(imps))]
    return feats, imps


def _rf_feats_imps(m, feat_cols=None):
    """Returns (feature_names list, importances array) for an RF/ExtraTree estimator."""
    imps = m.feature_importances_.astype(float)
    if feat_cols is not None:
        return list(feat_cols), imps
    try:
        feats = list(m.feature_names_in_)
    except AttributeError:
        feats = [f"f{i}" for i in range(len(imps))]
    return feats, imps


def _mimo_feats_imps(estimators, get_fi_fn, feat_cols=None):
    """Average feature importances across H estimators of a MultiOutputRegressor."""
    all_feats, all_imps = None, []
    for est in estimators:
        # Pass feat_cols to XGB helper so it can use bundle names even without named features
        if get_fi_fn is _xgb_feats_imps:
            feats, imps = get_fi_fn(est, feat_cols=feat_cols if all_feats is None else None)
        else:
            feats, imps = get_fi_fn(est)
        if all_feats is None:
            all_feats = feat_cols if feat_cols is not None else feats
        all_imps.append(imps)
    avg_imps = np.mean(all_imps, axis=0)
    return all_feats, avg_imps, all_imps   # also return per-horizon


def extract_importance(label, fname, is_bundle, is_mimo) -> pd.DataFrame | None:
    """
    Load model and return a DataFrame with columns:
      feature | importance | category | model | strategy
    """
    print(f"  Loading {label} …")
    obj = load_model(fname, is_bundle)
    if obj is None:
        return None

    # ── detect algo from filename ─────────────────────────────────────
    fl = fname.lower()
    is_lgbm = "lgbm" in fl
    is_xgb  = "xgb"  in fl
    is_rf   = ("rf_" in fl or fl.startswith("rf"))

    # ── get model + feature extractor ────────────────────────────────
    if is_bundle:
        bundle = obj
        model  = bundle["model"]
        # 'feature_cols' = authoritative list matching model's fitted feature count (incl. load preds)
        # 'price_feature_cols' = original price features only (152, without extra load columns)
        # Always prefer 'feature_cols' since it matches feature_importances_ length
        feat_cols = (bundle.get("feature_cols")
                     or bundle.get("price_feature_cols")
                     or None)
    else:
        model = obj
        feat_cols = None

    # detect actual model type (overrides filename heuristic for bundles)
    model_type_name = type(model).__name__.lower()
    if "multioutput" in model_type_name:
        # model is MultiOutputRegressor → use is_mimo path
        actual_is_mimo = True
    else:
        actual_is_mimo = False

    if actual_is_mimo:
        estimators = model.estimators_
        try:
            first = estimators[0]
        except Exception:
            print(f"  [WARN] {label}: cannot access estimators_")
            return None

        # detect sub-model type from actual estimator, not filename
        sub_type = type(first).__name__.lower()
        if "lgbm" in sub_type or "lightgbm" in sub_type:
            feats, imps, per_h = _mimo_feats_imps(estimators, _lgbm_feats_imps, feat_cols)
        elif "xgb" in sub_type or "xgboost" in sub_type:
            feats, imps, per_h = _mimo_feats_imps(estimators, _xgb_feats_imps, feat_cols)
        elif "forest" in sub_type or "tree" in sub_type:
            feats, imps, per_h = _mimo_feats_imps(estimators, _rf_feats_imps, feat_cols)
        else:
            print(f"  [WARN] {label}: unknown sub-model type {sub_type}")
            return None

        # feat_cols already applied inside _mimo_feats_imps; imps already averaged
        if feat_cols is not None and len(feat_cols) != len(imps):
            print(f"  [WARN] {label}: feat_cols length {len(feat_cols)} != imps length {len(imps)}")
            feats = [f"f{i}" for i in range(len(imps))]

    else:
        # Single-output OR single multi-output model (raw Booster, XGBRegressor, RF, etc.)
        # detect type from actual model object (not just filename)
        m_type = type(model).__name__.lower()
        if "booster" in m_type:
            feats, imps = _lgbm_feats_imps(model)               # raw lightgbm Booster
        elif "lgbm" in m_type or "lightgbm" in m_type:
            feats, imps = _lgbm_feats_imps(model)               # LGBMRegressor
        elif "xgb" in m_type or "xgboost" in m_type:
            feats, imps = _xgb_feats_imps(model, feat_cols)     # XGBRegressor
        elif "forest" in m_type or "randomforest" in m_type:
            feats, imps = _rf_feats_imps(model, feat_cols)      # RandomForestRegressor
        else:
            print(f"  [WARN] {label}: unknown model type '{m_type}'")
            return None
        # For LGBM/Booster: feat_cols override (bundle may have authoritative order)
        if feat_cols is not None and ("booster" in m_type or "lgbm" in m_type):
            feats = list(feat_cols)

    # ── build DataFrame ───────────────────────────────────────────────
    total = imps.sum()
    if total > 0:
        imps_norm = imps / total
    else:
        imps_norm = imps

    df = pd.DataFrame({
        "feature":    feats,
        "importance": imps_norm,
        "category":   [categorize(f) for f in feats],
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def extract_mimo_per_horizon(label, fname, is_bundle) -> pd.DataFrame | None:
    """Return per-horizon feature importances for a MIMO MultiOutputRegressor."""
    obj = load_model(fname, is_bundle)
    if obj is None:
        return None

    model = obj["model"] if is_bundle else obj
    if not hasattr(model, "estimators_"):
        return None
    estimators = model.estimators_
    feat_cols = (obj.get("feature_cols", None) or obj.get("price_feature_cols", None)) if is_bundle else None

    try:
        first = estimators[0]
    except Exception:
        return None

    sub_type = type(first).__name__.lower()
    if "lgbm" in sub_type or "lightgbm" in sub_type:
        get_fi = _lgbm_feats_imps
    elif "xgb" in sub_type or "xgboost" in sub_type:
        get_fi = _xgb_feats_imps
    elif "forest" in sub_type:
        get_fi = _rf_feats_imps
    else:
        return None

    all_feats, all_imps = None, []
    for est in estimators:
        feats, imps = get_fi(est)
        if all_feats is None:
            all_feats = feats if feat_cols is None else list(feat_cols)
        imps_norm = imps / (imps.sum() + 1e-12)
        all_imps.append(imps_norm)

    # shape: (H, n_features)
    mat = np.array(all_imps)
    return pd.DataFrame(mat, columns=all_feats)


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────
def run_task(task: str, model_list):
    print(f"\n{'='*60}")
    print(f" TASK: {task.upper()}")
    print(f"{'='*60}")

    records = {}   # label → DataFrame
    for label, strategy, fname, is_bundle, is_mimo in model_list:
        df = extract_importance(label, fname, is_bundle, is_mimo)
        if df is not None:
            df["model"]    = label
            df["strategy"] = strategy
            records[label] = df

    if not records:
        print("No models loaded!")
        return

    all_df = pd.concat(records.values(), ignore_index=True)

    # ── save raw importances CSV ──────────────────────────────────────
    pivot = all_df.pivot_table(
        index=["feature","category"],
        columns="model",
        values="importance",
        fill_value=0.0
    )
    pivot = pivot.reset_index()
    pivot["avg_importance"] = pivot[list(records.keys())].mean(axis=1)
    pivot = pivot.sort_values("avg_importance", ascending=False)
    csv_path = OUT / f"feature_importance_{task}.csv"
    pivot.to_csv(csv_path, index=False)
    print(f"\n  → Saved: {csv_path.name}")

    # ── top-5 summary ─────────────────────────────────────────────────
    summary_rows = []
    for label, df in records.items():
        for rank, row in df.head(5).iterrows():
            summary_rows.append({
                "task": task, "model": label, "rank": row["rank"],
                "feature": row["feature"], "category": row["category"],
                "importance": row["importance"]
            })
    summary_df = pd.DataFrame(summary_rows)

    # ── Plot 1: Top-20 feature importance per model (individual bars) ─
    n_models = len(records)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 9),
        sharey=False
    )
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Feature Importance — {task.capitalize()} Task  (top 20 per model)",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, (label, df) in zip(axes, records.items()):
        top = df.head(20).copy()
        cats = [categorize(f) for f in top["feature"]]
        colors = [CATEGORY_COLORS.get(c, "#9ca3af") for c in cats]
        bars = ax.barh(range(len(top)), top["importance"], color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Norm. importance", fontsize=8)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        ax.grid(axis="x", alpha=0.25)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=lbl) for lbl, c in CATEGORY_COLORS.items()
               if any(lbl in [categorize(f) for f in df["feature"]] for df in records.values())]
    fig.legend(handles=handles, title="Feature Category", loc="lower center",
               ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    bars_path = OUT / f"fi_{task}_bars.png"
    fig.savefig(bars_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {bars_path.name}")

    # ── Plot 2: Heatmap top-N features × models ───────────────────────
    TOP_N = 30
    # get top features across all models (by max importance in any model)
    top_feat_all = (all_df.groupby("feature")["importance"].max()
                    .sort_values(ascending=False)
                    .head(TOP_N).index.tolist())

    hm_data = []
    for feat in top_feat_all:
        row = {"feature": feat}
        for label, df in records.items():
            match = df.loc[df["feature"] == feat, "importance"]
            row[label] = float(match.iloc[0]) if len(match) > 0 else 0.0
        hm_data.append(row)
    hm_df = pd.DataFrame(hm_data).set_index("feature")

    # Normalize each column to [0,1] for visual clarity
    hm_norm = hm_df.div(hm_df.max(axis=0) + 1e-12)

    fig_h, ax_h = plt.subplots(figsize=(max(8, n_models * 1.6), 10))
    cmap = LinearSegmentedColormap.from_list(
        "importance", ["#1e293b", "#3b82f6", "#fbbf24", "#ef4444"]
    )
    im = ax_h.imshow(hm_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_h.set_xticks(range(len(records)))
    ax_h.set_xticklabels(list(records.keys()), rotation=30, ha="right", fontsize=8)
    ax_h.set_yticks(range(TOP_N))
    labels_y = []
    for feat in top_feat_all:
        cat = categorize(feat)
        labels_y.append(f"[{cat[:3].upper()}]  {feat}")
    ax_h.set_yticklabels(labels_y, fontsize=7.5)
    ax_h.set_title(f"Feature Importance Heatmap — {task.capitalize()} Task  (top {TOP_N})",
                   fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_h, label="Relative importance (column-normalised)", shrink=0.6)

    # color ytick labels by category
    for tick, feat in zip(ax_h.get_yticklabels(), top_feat_all):
        cat = categorize(feat)
        tick.set_color(CATEGORY_COLORS.get(cat, "#9ca3af"))

    plt.tight_layout()
    hm_path = OUT / f"fi_{task}_heatmap.png"
    fig_h.savefig(hm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_h)
    print(f"  → Saved: {hm_path.name}")

    # ── Plot 3: Stacked bar — feature categories per model ────────────
    cat_data = (all_df.groupby(["model","category"])["importance"]
                .sum().reset_index())
    cat_pivot = cat_data.pivot_table(
        index="model", columns="category", values="importance", fill_value=0
    )
    # reorder models to match registry order
    ordered_labels = [l for l, *_ in model_list if l in records]
    cat_pivot = cat_pivot.reindex(ordered_labels)

    fig_c, ax_c = plt.subplots(figsize=(max(8, n_models * 1.3), 5))
    bottom = np.zeros(len(cat_pivot))
    for cat, color in CATEGORY_COLORS.items():
        if cat in cat_pivot.columns:
            vals = cat_pivot[cat].values
            ax_c.bar(cat_pivot.index, vals, bottom=bottom,
                     color=color, label=cat, edgecolor="white", linewidth=0.4)
            bottom += vals

    ax_c.set_xticks(range(len(cat_pivot)))
    ax_c.set_xticklabels(cat_pivot.index, rotation=25, ha="right", fontsize=8)
    ax_c.set_ylabel("Cumulative feature importance")
    ax_c.set_title(f"Feature Category Breakdown — {task.capitalize()} Task",
                   fontsize=12, fontweight="bold")
    ax_c.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax_c.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax_c.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    cat_path = OUT / f"fi_{task}_categories.png"
    fig_c.savefig(cat_path, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"  → Saved: {cat_path.name}")

    return summary_df


def plot_mimo_per_horizon():
    """Per-horizon feature importance heatmap for LGBM Direct Price (H=24)."""
    print("\n  Computing MIMO per-horizon analysis …")
    ph = extract_mimo_per_horizon(
        "Direct LGBM Dense Price",
        "lgbm_direct_hourly_price_h24_dense.pkl",
        is_bundle=True
    )
    if ph is None:
        print("  [SKIP] LGBM Direct Dense price not found")
        return

    H, n_feats = ph.shape
    # Select top 20 features by mean importance across horizons
    mean_imp = ph.mean(axis=0).sort_values(ascending=False)
    top20 = mean_imp.head(20).index.tolist()
    ph_top = ph[top20]

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = LinearSegmentedColormap.from_list(
        "horizon", ["#1e293b", "#0ea5e9", "#fbbf24"]
    )
    im = ax.imshow(ph_top.T.values, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(range(H))
    ax.set_xticklabels([f"h+{h+1}" for h in range(H)], fontsize=7, rotation=45)
    ax.set_yticks(range(len(top20)))
    labels_y2 = [f"[{categorize(f)[:3].upper()}]  {f}" for f in top20]
    ax.set_yticklabels(labels_y2, fontsize=8)
    ax.set_title("LGBM Direct H=24 — Feature Importance per Forecast Horizon  (top 20 features)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast horizon (h+1 = 1 hour ahead … h+24 = 24 hours ahead)")
    plt.colorbar(im, ax=ax, label="Norm. importance", shrink=0.6)

    for tick, feat in zip(ax.get_yticklabels(), top20):
        cat = categorize(feat)
        tick.set_color(CATEGORY_COLORS.get(cat, "#9ca3af"))

    plt.tight_layout()
    path = OUT / "fi_mimo_horizons.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path.name}")


def plot_strategy_comparison(price_records, load_records):
    """Side-by-side: CL vs OL vs MIMO category breakdown for price AND load."""
    if not price_records or not load_records:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    tasks_data = [("Price (€/MWh)", price_records), ("Load (MW)", load_records)]

    for ax, (title, records) in zip(axes, tasks_data):
        cat_rows = []
        for label, df in records.items():
            for _, row in df.iterrows():
                cat_rows.append({
                    "model": label,
                    "category": row["category"],
                    "importance": row["importance"]
                })
        cat_df  = pd.DataFrame(cat_rows)
        cat_piv = (cat_df.groupby(["model","category"])["importance"]
                   .sum()
                   .unstack(fill_value=0))

        ordered = list(records.keys())
        cat_piv = cat_piv.reindex(ordered)

        bottom = np.zeros(len(cat_piv))
        for cat, color in CATEGORY_COLORS.items():
            if cat in cat_piv.columns:
                vals = cat_piv[cat].values
                ax.bar(range(len(cat_piv)), vals, bottom=bottom,
                       color=color, label=cat, edgecolor="white", linewidth=0.4)
                bottom += vals

        ax.set_xticks(range(len(cat_piv)))
        ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Cumulative importance")
        ax.set_title(f"Feature Category Breakdown — {title}", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.grid(axis="y", alpha=0.25)

    handles = [plt.Rectangle((0,0),1,1, color=c) for lbl,c in CATEGORY_COLORS.items()]
    fig.legend(handles=handles,
               labels=list(CATEGORY_COLORS.keys()),
               title="Category", ncol=2,
               loc="center right", fontsize=8, framealpha=0.9,
               bbox_to_anchor=(1.12, 0.5))

    plt.suptitle("Feature Category Contribution — Price vs Load  (CL / OL / MIMO)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT / "fi_combined_strategy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_summaries = []

    # ── PRICE ─────────────────────────────────────────────────────────
    price_records = {}
    for label, strategy, fname, is_bundle, is_mimo in PRICE_MODELS:
        df = extract_importance(label, fname, is_bundle, is_mimo)
        if df is not None:
            df["model"]    = label
            df["strategy"] = strategy
            price_records[label] = df

    # build summary
    for label, df in price_records.items():
        for _, row in df.head(5).iterrows():
            all_summaries.append({
                "task": "price", "model": label, "rank": row["rank"],
                "feature": row["feature"], "category": row["category"],
                "importance_%": f"{row['importance']*100:.2f}%"
            })

    # save CSV
    if price_records:
        all_price = pd.concat(price_records.values(), ignore_index=True)
        pivot_p = (all_price.pivot_table(index=["feature","category"], columns="model",
                                          values="importance", fill_value=0.0).reset_index())
        cols = list(price_records.keys())
        pivot_p["avg"] = pivot_p[cols].mean(axis=1)
        pivot_p.sort_values("avg", ascending=False, inplace=True)
        pivot_p.to_csv(OUT / "feature_importance_price.csv", index=False)
        print(f"\n  → Saved: feature_importance_price.csv")

    # ── LOAD ──────────────────────────────────────────────────────────
    load_records = {}
    for label, strategy, fname, is_bundle, is_mimo in LOAD_MODELS:
        df = extract_importance(label, fname, is_bundle, is_mimo)
        if df is not None:
            df["model"]    = label
            df["strategy"] = strategy
            load_records[label] = df

    for label, df in load_records.items():
        for _, row in df.head(5).iterrows():
            all_summaries.append({
                "task": "load", "model": label, "rank": row["rank"],
                "feature": row["feature"], "category": row["category"],
                "importance_%": f"{row['importance']*100:.2f}%"
            })

    if load_records:
        all_load = pd.concat(load_records.values(), ignore_index=True)
        pivot_l = (all_load.pivot_table(index=["feature","category"], columns="model",
                                         values="importance", fill_value=0.0).reset_index())
        cols = list(load_records.keys())
        pivot_l["avg"] = pivot_l[cols].mean(axis=1)
        pivot_l.sort_values("avg", ascending=False, inplace=True)
        pivot_l.to_csv(OUT / "feature_importance_load.csv", index=False)
        print(f"  → Saved: feature_importance_load.csv")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n Generating plots …")

    if price_records:
        # Rebuild all_df for price
        all_price = pd.concat(price_records.values(), ignore_index=True)
        n_p = len(price_records)
        TOP_N = 30

        # Bar plots
        fig, axes = plt.subplots(1, n_p, figsize=(5*n_p, 9), sharey=False)
        if n_p == 1: axes = [axes]
        fig.suptitle("Feature Importance — Price Task  (top 20 per model)",
                     fontsize=14, fontweight="bold", y=1.01)
        for ax, (label, df) in zip(axes, price_records.items()):
            top = df.head(20)
            colors = [CATEGORY_COLORS.get(categorize(f), "#9ca3af") for f in top["feature"]]
            ax.barh(range(len(top)), top["importance"], color=colors,
                    edgecolor="white", linewidth=0.4)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["feature"], fontsize=7.5)
            ax.invert_yaxis()
            ax.set_title(label, fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Norm. importance", fontsize=8)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
            ax.grid(axis="x", alpha=0.25)
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=lbl) for lbl, c in CATEGORY_COLORS.items()
                   if lbl in all_price["category"].values]
        fig.legend(handles=handles, title="Feature Category", loc="lower center",
                   ncol=4, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.07))
        plt.tight_layout()
        fig.savefig(OUT / "fi_price_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  → fi_price_bars.png")

        # Heatmap
        top_feats = (all_price.groupby("feature")["importance"].max()
                     .sort_values(ascending=False).head(TOP_N).index.tolist())
        hm_rows = []
        for feat in top_feats:
            row = {"feature": feat}
            for lbl, df in price_records.items():
                m = df.loc[df["feature"]==feat, "importance"]
                row[lbl] = float(m.iloc[0]) if len(m) > 0 else 0.0
            hm_rows.append(row)
        hm = pd.DataFrame(hm_rows).set_index("feature")
        hm_norm = hm.div(hm.max(axis=0) + 1e-12)

        fig_h, ax_h = plt.subplots(figsize=(max(8, n_p*1.6), 10))
        cmap = LinearSegmentedColormap.from_list("imp", ["#1e293b","#3b82f6","#fbbf24","#ef4444"])
        im = ax_h.imshow(hm_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax_h.set_xticks(range(len(price_records)))
        ax_h.set_xticklabels(list(price_records.keys()), rotation=30, ha="right", fontsize=8)
        ax_h.set_yticks(range(TOP_N))
        ax_h.set_yticklabels([f"[{categorize(f)[:3].upper()}]  {f}" for f in top_feats], fontsize=7.5)
        ax_h.set_title(f"Feature Importance Heatmap — Price Task  (top {TOP_N})",
                       fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax_h, label="Relative importance (col-normalised)", shrink=0.6)
        for tick, feat in zip(ax_h.get_yticklabels(), top_feats):
            tick.set_color(CATEGORY_COLORS.get(categorize(feat), "#9ca3af"))
        plt.tight_layout()
        fig_h.savefig(OUT / "fi_price_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig_h)
        print("  → fi_price_heatmap.png")

        # Category stacked bar
        cat_rows = []
        for lbl, df in price_records.items():
            for _, r in df.iterrows():
                cat_rows.append({"model": lbl, "category": r["category"], "importance": r["importance"]})
        cat_piv = (pd.DataFrame(cat_rows).groupby(["model","category"])["importance"]
                   .sum().unstack(fill_value=0).reindex([l for l,*_ in PRICE_MODELS if l in price_records]))
        fig_c, ax_c = plt.subplots(figsize=(max(8, n_p*1.3), 5))
        bottom = np.zeros(len(cat_piv))
        for cat, color in CATEGORY_COLORS.items():
            if cat in cat_piv.columns:
                vals = cat_piv[cat].values
                ax_c.bar(cat_piv.index, vals, bottom=bottom, color=color, label=cat,
                         edgecolor="white", linewidth=0.4)
                bottom += vals
        ax_c.set_xticks(range(len(cat_piv)))
        ax_c.set_xticklabels(cat_piv.index, rotation=25, ha="right", fontsize=8)
        ax_c.set_ylabel("Cumulative feature importance")
        ax_c.set_title("Feature Category Breakdown — Price Task", fontsize=12, fontweight="bold")
        ax_c.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax_c.legend(title="Category", bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
        ax_c.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig_c.savefig(OUT / "fi_price_categories.png", dpi=150, bbox_inches="tight")
        plt.close(fig_c)
        print("  → fi_price_categories.png")

    if load_records:
        all_load = pd.concat(load_records.values(), ignore_index=True)
        n_l = len(load_records)

        # Bar plots
        fig, axes = plt.subplots(1, n_l, figsize=(5*n_l, 9), sharey=False)
        if n_l == 1: axes = [axes]
        fig.suptitle("Feature Importance — Load Task  (top 20 per model)",
                     fontsize=14, fontweight="bold", y=1.01)
        for ax, (label, df) in zip(axes, load_records.items()):
            top = df.head(20)
            colors = [CATEGORY_COLORS.get(categorize(f), "#9ca3af") for f in top["feature"]]
            ax.barh(range(len(top)), top["importance"], color=colors,
                    edgecolor="white", linewidth=0.4)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["feature"], fontsize=7.5)
            ax.invert_yaxis()
            ax.set_title(label, fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Norm. importance", fontsize=8)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
            ax.grid(axis="x", alpha=0.25)
        from matplotlib.patches import Patch
        handles2 = [Patch(color=c, label=lbl) for lbl, c in CATEGORY_COLORS.items()
                    if lbl in all_load["category"].values]
        fig.legend(handles=handles2, title="Feature Category", loc="lower center",
                   ncol=4, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.07))
        plt.tight_layout()
        fig.savefig(OUT / "fi_load_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  → fi_load_bars.png")

        # Heatmap load
        top_feats_l = (all_load.groupby("feature")["importance"].max()
                       .sort_values(ascending=False).head(TOP_N).index.tolist())
        hm_rows_l = []
        for feat in top_feats_l:
            row = {"feature": feat}
            for lbl, df in load_records.items():
                m = df.loc[df["feature"]==feat, "importance"]
                row[lbl] = float(m.iloc[0]) if len(m) > 0 else 0.0
            hm_rows_l.append(row)
        hm_l = pd.DataFrame(hm_rows_l).set_index("feature")
        hm_l_norm = hm_l.div(hm_l.max(axis=0) + 1e-12)
        fig_hl, ax_hl = plt.subplots(figsize=(max(8, n_l*1.6), 10))
        im2 = ax_hl.imshow(hm_l_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax_hl.set_xticks(range(n_l))
        ax_hl.set_xticklabels(list(load_records.keys()), rotation=30, ha="right", fontsize=8)
        ax_hl.set_yticks(range(len(top_feats_l)))
        ax_hl.set_yticklabels([f"[{categorize(f)[:3].upper()}]  {f}" for f in top_feats_l], fontsize=7.5)
        ax_hl.set_title(f"Feature Importance Heatmap — Load Task  (top {TOP_N})",
                        fontsize=12, fontweight="bold")
        plt.colorbar(im2, ax=ax_hl, label="Relative importance (col-normalised)", shrink=0.6)
        for tick, feat in zip(ax_hl.get_yticklabels(), top_feats_l):
            tick.set_color(CATEGORY_COLORS.get(categorize(feat), "#9ca3af"))
        plt.tight_layout()
        fig_hl.savefig(OUT / "fi_load_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig_hl)
        print("  → fi_load_heatmap.png")

        # Category stacked bar load
        cat_rows_l = []
        for lbl, df in load_records.items():
            for _, r in df.iterrows():
                cat_rows_l.append({"model": lbl, "category": r["category"], "importance": r["importance"]})
        cat_piv_l = (pd.DataFrame(cat_rows_l).groupby(["model","category"])["importance"]
                     .sum().unstack(fill_value=0).reindex([l for l,*_ in LOAD_MODELS if l in load_records]))
        fig_cl, ax_cl = plt.subplots(figsize=(max(8, n_l*1.3), 5))
        bottom_l = np.zeros(len(cat_piv_l))
        for cat, color in CATEGORY_COLORS.items():
            if cat in cat_piv_l.columns:
                vals = cat_piv_l[cat].values
                ax_cl.bar(cat_piv_l.index, vals, bottom=bottom_l, color=color, label=cat,
                          edgecolor="white", linewidth=0.4)
                bottom_l += vals
        ax_cl.set_xticks(range(len(cat_piv_l)))
        ax_cl.set_xticklabels(cat_piv_l.index, rotation=25, ha="right", fontsize=8)
        ax_cl.set_ylabel("Cumulative feature importance")
        ax_cl.set_title("Feature Category Breakdown — Load Task", fontsize=12, fontweight="bold")
        ax_cl.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax_cl.legend(title="Category", bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
        ax_cl.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig_cl.savefig(OUT / "fi_load_categories.png", dpi=150, bbox_inches="tight")
        plt.close(fig_cl)
        print("  → fi_load_categories.png")

    # MIMO per-horizon
    plot_mimo_per_horizon()

    # Combined strategy comparison
    plot_strategy_comparison(price_records, load_records)

    # Save summary
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(OUT / "fi_summary.csv", index=False)
        print(f"\n  → Saved: fi_summary.csv")

    # ── Print quick summary to console ───────────────────────────────
    print("\n" + "="*70)
    print(" QUICK SUMMARY: Top-3 features per model")
    print("="*70)
    for task_name, records in [("PRICE", price_records), ("LOAD", load_records)]:
        print(f"\n  [{task_name}]")
        for label, df in records.items():
            top3 = df.head(3)
            feats = ", ".join(
                f"{r.feature} ({r.importance*100:.1f}%)"
                for _, r in top3.iterrows()
            )
            print(f"    {label:<30s}  → {feats}")

    print(f"\n All outputs saved to: {OUT}")
