"""
src/dm_test.py
==============
Pairwise Diebold-Mariano (DM) forecast comparison for the thesis.
Uses Q1 2026 (Dec 2025 – Feb 2026, 2160h) dashboard JSON data.

Reference:
  Diebold & Mariano (1995) — original DM test
  Harvey, Leybourne & Newbold (1997) — small-sample HLN correction

Usage:
  conda run -n epf --no-capture-output python -m src.dm_test

Outputs (in dm_test/ folder):
  dm_price_matrix.csv     — DM statistics (price, all key models)
  dm_price_pval.csv       — p-values (price)
  dm_load_matrix.csv      — DM statistics (load)
  dm_load_pval.csv        — p-values (load)
  dm_price_heatmap.png    — heatmap with significance stars
  dm_load_heatmap.png     — heatmap with significance stars
  dm_strategy_summary.png — strategy-level comparison bars
  dm_results_table.csv    — human-readable table: best vs rest
"""

from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "dm_test"
OUTDIR.mkdir(exist_ok=True)

# ─── Academic white theme (Patras ECE style) ─────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "text.color":        "#111111",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "grid.color":        "#cccccc",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─── Model registry ───────────────────────────────────────────────────────────
# (json_key, series_name, display_name, strategy, h_forecast)
# h_forecast: DM bandwidth = forecast horizon
PRICE_MODELS: list[tuple] = [
    # CL teacher-forced (h=1: one-step-ahead)
    ("price_cl_monthly_q1_2026",            "LightGBM",                          "CL-LGBM",          "CL",   1),
    ("price_cl_monthly_q1_2026",            "XGBoost",                           "CL-XGB",           "CL",   1),
    ("price_cl_monthly_q1_2026",            "RandomForest",                      "CL-RF",            "CL",   1),
    ("price_cl_monthly_q1_2026",            "MLP-Optuna",                        "CL-MLP-Opt",       "CL",   1),
    ("price_cl_monthly_q1_2026",            "Ensemble-Best3 (1/MAE)",            "CL-Ens-Best3",     "CL",   1),
    # OL daily 24h recursive (h=24)
    ("price_openloop_h24_monthly_q1_2026",  "LGBM-Daily-Optuna (24h)",           "OL-LGBM-Daily",    "OL",  24),
    ("price_openloop_h24_monthly_q1_2026",  "XGB-Daily-SS-Optuna Dense (24h)",   "OL-XGB-SS-Dense",  "OL",  24),
    ("price_openloop_h24_monthly_q1_2026",  "RF-Daily-SS (24h)",                 "OL-RF-SS",         "OL",  24),
    ("price_openloop_h24_monthly_q1_2026",  "Ensemble-Top3 1/MAE (LGBM+RF+XGB)","OL-Ens-Top3",      "OL",  24),
    # MIMO H=24 (h=24)
    ("price_mimo_h24_monthly_q1_2026",      "XGB MIMO 24h Dense",                "MIMO-XGB-Dense",   "MIMO",24),
    ("price_mimo_h24_monthly_q1_2026",      "RF MIMO 24h+Load Dense",            "MIMO-RF-Dense",    "MIMO",24),
    # Direct multi-horizon (its own strategy, separate JSON)
    ("price_direct_h24_monthly_q1_2026",    "LGBM DIRECT 24h Optuna",            "Direct-LGBM-Opt",  "Direct",24),
    ("price_mimo_h24_monthly_q1_2026",      "Ensemble Best3 24h (1/MAE)",        "MIMO-Ens-Best3",   "MIMO",24),
    # Monthly Retrain TF (h=1) and Rec (h=24)
    ("price_monthly_retrain",               "TF-LGBM-MR",                        "MR-TF-LGBM",       "MR",   1),
    ("price_monthly_retrain",               "TF-XGB-MR",                         "MR-TF-XGB",        "MR",   1),
    ("price_monthly_retrain",               "TF-RF-MR",                          "MR-TF-RF",         "MR",   1),
    ("price_monthly_retrain",               "Rec-XGB-SS-DO-MR",                  "MR-Rec-XGB-SS",    "MR",  24),
    ("price_monthly_retrain",               "Ensemble-TF-Best3 (1/MAE)",         "MR-Ens-TF-Best3",  "MR",   1),
    # Walk-forward MIMO / Direct (monthly retrain + Optuna)
    ("price_mimo_monthly_retrain_optuna",   "Ensemble-Best3 MIMO MR",            "WF-MIMO-Ens3",     "MR-MIMO",24),
    ("price_mimo_monthly_retrain_optuna",   "LGBM Direct Dense MR+CachedOpt",    "WF-Direct-LGBM",   "MR-MIMO",24),
]

LOAD_MODELS: list[tuple] = [
    # CL
    ("load_cl_monthly_q1_2026",             "LightGBM",                          "CL-LGBM",          "CL",   1),
    ("load_cl_monthly_q1_2026",             "XGBoost",                           "CL-XGB",           "CL",   1),
    ("load_cl_monthly_q1_2026",             "RandomForest",                      "CL-RF",            "CL",   1),
    ("load_cl_monthly_q1_2026",             "MLP-Optuna",                        "CL-MLP-Opt",       "CL",   1),
    ("load_cl_monthly_q1_2026",             "Ensemble-Best3 (1/MAE)",            "CL-Ens-Best3",     "CL",   1),
    # OL daily 24h recursive
    ("load_openloop_h24_monthly_q1_2026",   "LGBM-Daily-SS-Optuna (24h)",        "OL-LGBM-SS",       "OL",  24),
    ("load_openloop_h24_monthly_q1_2026",   "XGB-Daily-Optuna (24h)",            "OL-XGB-Daily",     "OL",  24),
    ("load_openloop_h24_monthly_q1_2026",   "RF-Daily-SS (24h)",                 "OL-RF-SS",         "OL",  24),
    ("load_openloop_h24_monthly_q1_2026",   "Ensemble-Top3 1/MAE (LGBM+RF+XGB)","OL-Ens-Top3",      "OL",  24),
    # MIMO H=24
    ("load_mimo_h24_monthly_q1_2026",       "XGB MIMO 24h Dense",                "MIMO-XGB-Dense",   "MIMO",24),
    # Direct multi-horizon (its own strategy, separate JSON)
    ("load_direct_h24_monthly_q1_2026",     "LGBM DIRECT 24h Dense",             "Direct-LGBM-Dense","Direct",24),
    ("load_mimo_h24_monthly_q1_2026",       "RF MIMO 24h",                       "MIMO-RF",          "MIMO",24),
    ("load_mimo_h24_monthly_q1_2026",       "Ensemble Best3 24h (1/MAE)",        "MIMO-Ens-Best3",   "MIMO",24),
    # Monthly Retrain
    ("load_monthly_retrain",                "TF-LGBM-MR",                        "MR-TF-LGBM",       "MR",   1),
    ("load_monthly_retrain",                "TF-XGB-MR",                         "MR-TF-XGB",        "MR",   1),
    ("load_monthly_retrain",                "TF-RF-MR",                          "MR-TF-RF",         "MR",   1),
    ("load_monthly_retrain",                "Rec-XGB-SS-DO-MR",                  "MR-Rec-XGB-SS",    "MR",  24),
    ("load_monthly_retrain",                "Ensemble-TF-Best3 (1/MAE)",         "MR-Ens-TF-Best3",  "MR",   1),
    # Walk-forward MIMO / Direct (monthly retrain + Optuna)
    ("load_mimo_monthly_retrain_optuna",    "Ensemble-Best3 MIMO MR",            "WF-MIMO-Ens3",     "MR-MIMO",24),
    ("load_mimo_monthly_retrain_optuna",    "LGBM Direct Dense MR+CachedOpt",    "WF-Direct-LGBM",   "MR-MIMO",24),
]

# Strategy colours (consistent with dashboard)
STRAT_COL = {
    "CL":      "#38bdf8",   # sky-blue
    "OL":      "#22c55e",   # green
    "MIMO":    "#f97316",   # orange
    "MR":      "#a855f7",   # purple (monthly retrain)
    "Direct":  "#8e24aa",   # violet (direct multi-horizon)
    "MR-MIMO": "#00838f",   # teal (walk-forward MIMO/Direct)
}

# ─── Core DM test ─────────────────────────────────────────────────────────────

def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1,
            loss: str = "absolute") -> tuple[float, float]:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction.

    H0: equal predictive accuracy E[L(e1)] = E[L(e2)]
    H1 (two-sided): unequal predictive accuracy

    Parameters
    ----------
    e1, e2 : forecast errors (actual - predicted) for model 1 and 2
    h      : forecast horizon (bandwidth for NW variance estimator)
    loss   : "absolute" (MAE-based) or "squared" (MSE-based)

    Returns
    -------
    dm_stat : float — DM* statistic (t-distributed under H0 with T-1 df)
    p_value : float — two-sided p-value
    """
    assert len(e1) == len(e2), "e1 and e2 must have equal length"
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)

    # loss differential: d_t = L(e2) - L(e1)
    # Convention: DM > 0 → model 1 (e1) has LOWER loss → model 1 is BETTER.
    if loss == "absolute":
        d = np.abs(e2) - np.abs(e1)
    else:
        d = e2 ** 2 - e1 ** 2

    T      = len(d)
    d_bar  = np.mean(d)

    # Newey-West autocovariance estimate (bandwidth = h-1)
    gamma_0 = np.mean((d - d_bar) ** 2)
    nw_var  = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        nw_var += 2.0 * gamma_k

    if nw_var <= 0:
        return np.nan, np.nan

    dm_raw = d_bar / np.sqrt(nw_var / T)

    # HLN small-sample correction factor
    cf      = np.sqrt((T + 1 - 2*h + h*(h - 1)/T) / T)
    dm_stat = dm_raw * cf

    # t-distribution with T-1 degrees of freedom (HLN 1997)
    p_value = 2.0 * (1.0 - st.t.cdf(abs(dm_stat), df=T - 1))

    return float(dm_stat), float(p_value)


def sig_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if np.isnan(p): return "—"
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ─── Data loading ─────────────────────────────────────────────────────────────

_json_cache: dict[str, dict] = {}

def _load_json(key: str) -> dict:
    if key not in _json_cache:
        path = ROOT / f"dashboard_data_hourly_{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON not found: {path}")
        _json_cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _json_cache[key]


def _load_series(json_key: str, series_name: str) -> pd.Series:
    d = _load_json(json_key)
    dates = pd.to_datetime(d["dates"])
    vals  = np.array(d["series"][series_name], dtype=float)
    return pd.Series(vals, index=dates, name=series_name)


def _load_actual(json_key: str) -> pd.Series:
    d = _load_json(json_key)
    dates = pd.to_datetime(d["dates"])
    return pd.Series(np.array(d["actual"], dtype=float), index=dates, name="actual")


def load_all_series(task: str) -> tuple[pd.Series, pd.DataFrame, list[dict]]:
    """
    Load actual values and all model predictions for the given task.
    Returns:
      actual  (Series, 2160h)
      preds   (DataFrame, columns = display_name)
      meta    (list of dict with strategy/h per model)
    """
    registry = PRICE_MODELS if task == "price" else LOAD_MODELS

    # ── Single canonical ground truth for ALL models ────────────────────────────
    # The dashboard JSONs carry TWO different "actual" arrays for Q1 2026: the
    # *_monthly_retrain / *_mimo* / *_direct* files hold the true market price
    # (Dec 110 / Jan 108,7 / Feb 78,3 — matches the documented regime shift),
    # whereas price_cl / price_openloop hold a stale actual that differs by
    # ~9,8 €/MWh on average. The DM loss differential needs ONE actual per hour,
    # so we anchor every model to the canonical (monthly-retrain) truth instead
    # of whichever JSON happens to be first in the registry. Timestamps are
    # identical across all JSONs, so this is a clean re-alignment.
    canon_key  = "price_monthly_retrain" if task == "price" else "load_monthly_retrain"
    actual_ref = _load_actual(canon_key)

    all_series = {}
    meta       = []

    for json_key, series_name, display, strategy, h in registry:
        try:
            s = _load_series(json_key, series_name)
        except (KeyError, FileNotFoundError) as exc:
            warnings.warn(f"Skipping {display}: {exc}")
            continue

        all_series[display] = s
        meta.append({"display": display, "strategy": strategy, "h": h,
                     "json_key": json_key, "series_name": series_name})

    preds = pd.DataFrame(all_series)
    # Align index to the canonical actual
    common_idx = actual_ref.index.intersection(preds.index)
    actual_ref = actual_ref.loc[common_idx]
    preds      = preds.loc[common_idx]

    return actual_ref, preds, meta


# ─── DM matrix computation ────────────────────────────────────────────────────

def compute_dm_matrix(actual: pd.Series, preds: pd.DataFrame,
                      meta: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise DM statistics and p-values.
    For cross-strategy pairs, h = max(h_i, h_j).
    Returns (dm_stat_df, dm_pval_df) — both shape (n_models, n_models).
    """
    names = preds.columns.tolist()
    h_map = {m["display"]: m["h"] for m in meta}
    n     = len(names)

    stat_mat = np.full((n, n), np.nan)
    pval_mat = np.full((n, n), np.nan)

    errors = (actual.values[:, None] - preds.values)   # shape T × n

    for i in range(n):
        for j in range(n):
            if i == j:
                stat_mat[i, j] = 0.0
                pval_mat[i, j] = 1.0
                continue
            h_ij = max(h_map.get(names[i], 1), h_map.get(names[j], 1))
            dm, pv = dm_test(errors[:, i], errors[:, j], h=h_ij)
            stat_mat[i, j] = dm
            pval_mat[i, j] = pv

    stat_df = pd.DataFrame(stat_mat, index=names, columns=names)
    pval_df = pd.DataFrame(pval_mat, index=names, columns=names)
    return stat_df, pval_df


# ─── Multiple-comparison correction (Holm-Bonferroni / Benjamini-Hochberg) ────

def _adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    """Adjust a 1-D array of p-values. method in {'holm','bh'}."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m)
    if method == "holm":            # step-down FWER
        running = 0.0
        for k in range(m):
            running = max(running, (m - k) * ranked[k])
            adj[k] = min(running, 1.0)
    elif method == "bh":            # step-up FDR
        prev = 1.0
        for k in range(m - 1, -1, -1):
            prev = min(prev, (m / (k + 1)) * ranked[k])
            adj[k] = min(prev, 1.0)
    else:
        raise ValueError(method)
    out = np.empty(m)
    out[order] = adj
    return out


def holm_bh_adjust(pval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Holm and BH correction over the unique off-diagonal pairs
    (153 for 18 models). Returns symmetric (holm_df, bh_df)."""
    names = pval_df.index.tolist()
    n = len(names)
    iu = np.triu_indices(n, k=1)
    pv = pval_df.values[iu]
    holm = _adjust_pvalues(pv, "holm")
    bh = _adjust_pvalues(pv, "bh")
    H = np.full((n, n), np.nan); B = np.full((n, n), np.nan)
    H[iu] = holm; H[(iu[1], iu[0])] = holm
    B[iu] = bh;   B[(iu[1], iu[0])] = bh
    np.fill_diagonal(H, 1.0); np.fill_diagonal(B, 1.0)
    return (pd.DataFrame(H, index=names, columns=names),
            pd.DataFrame(B, index=names, columns=names))


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _strategy_color_list(names: list[str], meta: list[dict]) -> list[str]:
    strat_map = {m["display"]: m["strategy"] for m in meta}
    return [STRAT_COL.get(strat_map.get(n, "CL"), FG) for n in names]


def plot_dm_heatmap(stat_df: pd.DataFrame, pval_df: pd.DataFrame,
                    meta: list[dict], task: str, outdir: Path) -> None:
    """
    Plot DM statistic heatmap with significance annotations.
    Convention: cell (i,j) > 0 → row model MORE accurate than column model.
    """
    names = stat_df.columns.tolist()
    n     = len(names)

    fig, ax = plt.subplots(figsize=(max(11, n * 0.80), max(9, n * 0.70)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Build annotation and colour matrices
    annot  = np.full((n, n), "", dtype=object)
    colors = np.zeros((n, n, 4))  # RGBA

    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j]  = "—"
                colors[i, j] = (0.88, 0.88, 0.88, 1.0)   # light grey diagonal
                continue
            dm = stat_df.iloc[i, j]
            pv = pval_df.iloc[i, j]
            if np.isnan(dm):
                annot[i, j]  = "n/a"
                colors[i, j] = (0.93, 0.93, 0.93, 1.0)
                continue

            stars = sig_stars(pv)
            # Green tones: row beats col; red tones: row loses to col
            intensity = min(abs(dm) / 4.0, 1.0)
            if dm > 0:   # row BETTER → green
                r = 0.85 - 0.55 * intensity
                g = 0.95 - 0.15 * intensity
                b = 0.85 - 0.55 * intensity
            else:        # row WORSE → red
                r = 0.95 - 0.05 * intensity
                g = 0.85 - 0.55 * intensity
                b = 0.85 - 0.55 * intensity
            colors[i, j] = (r, g, b, 1.0)
            annot[i, j]  = f"{dm:+.1f}{stars}"

    ax.imshow(colors, aspect="auto", interpolation="nearest")

    # Text annotations — dark on light cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=7, color="#111111",
                    fontweight="bold" if (i == j) else "normal")

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(names, fontsize=7.5)

    # Colour-code tick labels by strategy
    strat_map = {m["display"]: m["strategy"] for m in meta}
    for tick, name in zip(ax.get_xticklabels(), names):
        tick.set_color(STRAT_COL.get(strat_map.get(name, ""), "#111111"))
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(STRAT_COL.get(strat_map.get(name, ""), "#111111"))

    unit = "€/MWh" if task == "price" else "MW"
    task_label = "Τιμής (€/MWh)" if task == "price" else "Φορτίου (MW)"
    ax.set_title(
        f"Έλεγχος Diebold-Mariano — Πρόβλεψη {task_label} — Q1 2026\n"
        f"Κελί (i,j): στατιστική DM. Θετικό = μοντέλο γραμμής ακριβέστερο.",
        fontsize=10, color="#111111", pad=14,
    )

    # Legend: strategy colours + significance
    patches = [mpatches.Patch(facecolor=c, edgecolor="#555555", label=s)
               for s, c in STRAT_COL.items()]
    patches += [
        mpatches.Patch(facecolor="none", edgecolor="none", label="*** p < 0,01"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="**  p < 0,05"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="*   p < 0,10"),
    ]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.25, 1.0),
              fontsize=7.5, facecolor="white", edgecolor="#bbbbbb")

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    plt.tight_layout()
    out = outdir / f"dm_{task}_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_strategy_comparison(actual: pd.Series, preds: pd.DataFrame,
                              meta: list[dict], task: str, outdir: Path) -> None:
    """
    Horizontal bar chart: best model per strategy vs all others (DM statistic).
    Academic white-background style matching Patras ECE thesis standards.
    """
    unit = "€/MWh" if task == "price" else "MW"
    task_label = "Τιμής (€/MWh)" if task == "price" else "Φορτίου (MW)"

    # Compute MAE per model
    errors_abs = np.abs(actual.values[:, None] - preds.values)
    mae_series  = pd.Series(errors_abs.mean(axis=0), index=preds.columns)

    # Best model per strategy
    strat_map  = {m["display"]: m["strategy"] for m in meta}
    h_map      = {m["display"]: m["h"] for m in meta}
    strategies = ["MR", "MR-MIMO", "CL", "OL", "MIMO", "Direct"]
    best       = {}
    for s in strategies:
        candidates = [n for n in preds.columns if strat_map.get(n) == s]
        if not candidates:
            continue
        best[s] = mae_series[candidates].idxmin()

    fig, axes = plt.subplots(1, len(best), figsize=(4.5 * len(best), 6), sharey=False)
    if len(best) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, (strat, champ) in zip(axes, best.items()):
        others   = [n for n in preds.columns if n != champ]
        dm_stats, p_vals, better = [], [], []

        e_champ = actual.values - preds[champ].values
        for other in others:
            h_ij = max(h_map.get(champ, 1), h_map.get(other, 1))
            dm, pv = dm_test(e_champ, actual.values - preds[other].values, h=h_ij)
            dm_stats.append(dm)
            p_vals.append(pv)
            better.append(dm > 0 and pv < 0.05)

        # Sort descending by DM
        order   = np.argsort(dm_stats)[::-1]
        o_names = [others[i] for i in order]
        o_dm    = [dm_stats[i] for i in order]
        o_pv    = [p_vals[i] for i in order]
        o_bet   = [better[i] for i in order]
        o_strat = [strat_map.get(n, "") for n in o_names]

        # Bar colours from STRAT_COL; bold edge when significant
        bar_colors = [STRAT_COL.get(s2, "#888888") for s2 in o_strat]
        edge_w     = [1.8 if b else 0.6 for b in o_bet]
        ax.barh(range(len(o_names)), o_dm, color=bar_colors,
                edgecolor="#333333", linewidth=edge_w, height=0.65)

        # Significance stars in dark text
        for k, (d, p) in enumerate(zip(o_dm, o_pv)):
            if not np.isnan(p):
                stars = sig_stars(p)
                if stars:
                    xpos  = d + (0.08 if d >= 0 else -0.08)
                    align = "left" if d >= 0 else "right"
                    ax.text(xpos, k, stars, va="center", ha=align,
                            fontsize=8, color="#333333", fontweight="bold")

        ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
        ax.set_yticks(range(len(o_names)))
        ax.set_yticklabels(o_names, fontsize=7.5)
        for tick, s2 in zip(ax.get_yticklabels(), o_strat):
            tick.set_color(STRAT_COL.get(s2, "#111111"))

        ax.set_xlabel("Στατιστική DM  (θετικό = champion ακριβέστερο)", fontsize=8)
        champ_mae = mae_series[champ]
        ax.set_title(
            f"Champion: {champ}\nMAE = {champ_mae:.3f} {unit}  [{strat}]",
            fontsize=8.5, color=STRAT_COL.get(strat, "#111111"), pad=7,
        )
        ax.set_facecolor("white")
        ax.tick_params(colors="#333333")
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)

    # Bottom legend
    patches = [mpatches.Patch(facecolor=c, edgecolor="#555555", label=s)
               for s, c in STRAT_COL.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               facecolor="white", edgecolor="#bbbbbb", bbox_to_anchor=(0.5, -0.04))

    task_gr = "ΤΙΜΗΣ" if task == "price" else "ΦΟΡΤΙΟΥ"
    fig.suptitle(
        f"Σύγκριση DM — Πρόβλεψη {task_label} — Q1 2026\n"
        f"Κάθε πάνελ: καλύτερο μοντέλο στρατηγικής vs όλα τα άλλα. "
        f"Έντονο πλαίσιο = σημαντικό σε 5%.",
        fontsize=9.5, color="#111111", y=1.02,
    )
    plt.tight_layout()
    out = outdir / f"dm_{task}_strategy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


def build_results_table(stat_df: pd.DataFrame, pval_df: pd.DataFrame,
                        actual: pd.Series, preds: pd.DataFrame,
                        meta: list[dict], task: str) -> pd.DataFrame:
    """
    Build human-readable table: for each model, list wins vs losses at 5% significance.
    """
    unit = "€/MWh" if task == "price" else "MW"
    errors_abs = np.abs(actual.values[:, None] - preds.values)
    mae_series  = dict(zip(preds.columns, errors_abs.mean(axis=0)))
    strat_map   = {m["display"]: m["strategy"] for m in meta}

    rows = []
    for name in preds.columns:
        dm_row  = stat_df.loc[name]
        pv_row  = pval_df.loc[name]
        # DM(i,j)>0 → model i is BETTER than model j → WIN for i
        wins    = sum(1 for j, (d, p) in enumerate(zip(dm_row, pv_row))
                      if stat_df.columns[j] != name and d > 0 and p < 0.05)
        losses  = sum(1 for j, (d, p) in enumerate(zip(dm_row, pv_row))
                      if stat_df.columns[j] != name and d < 0 and p < 0.05)
        draws   = len(preds.columns) - 1 - wins - losses
        rows.append({
            "Model":     name,
            "Strategy":  strat_map.get(name, "—"),
            f"MAE ({unit})": round(mae_series[name], 3),
            "Wins (p<5%)":   wins,
            "Draws":         draws,
            "Losses (p<5%)": losses,
            "W-L":           wins - losses,
        })

    df = pd.DataFrame(rows).sort_values("W-L", ascending=False).reset_index(drop=True)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_task(task: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Task: {task.upper()}")
    print(f"{'─'*60}")

    print("  Loading series …")
    actual, preds, meta = load_all_series(task)
    print(f"  {len(preds.columns)} models × {len(actual)} hours")

    print("  Computing DM matrix …")
    stat_df, pval_df = compute_dm_matrix(actual, preds, meta)

    # Save CSVs
    stat_path = OUTDIR / f"dm_{task}_matrix.csv"
    pval_path = OUTDIR / f"dm_{task}_pval.csv"
    stat_df.round(3).to_csv(stat_path)
    pval_df.round(4).to_csv(pval_path)
    print(f"  → {stat_path.name}")
    print(f"  → {pval_path.name}")

    # Multiple-comparison correction (153 pairs) — additive outputs
    holm_df, bh_df = holm_bh_adjust(pval_df)
    holm_df.round(4).to_csv(OUTDIR / f"dm_{task}_pval_holm.csv")
    bh_df.round(4).to_csv(OUTDIR / f"dm_{task}_pval_bh.csv")
    print(f"  → dm_{task}_pval_holm.csv  → dm_{task}_pval_bh.csv")

    def _wins(pmat: pd.DataFrame) -> dict:
        w = {}
        for nm in pval_df.index:
            w[nm] = sum(
                1 for other in pval_df.columns
                if nm != other and stat_df.loc[nm, other] > 0
                and pmat.loc[nm, other] < 0.05
            )
        return w

    raw_w, holm_w, bh_w = _wins(pval_df), _wins(holm_df), _wins(bh_df)
    print("  Significant wins  (raw / Holm / BH):")
    for nm in sorted(raw_w, key=lambda x: -raw_w[x]):
        print(f"    {nm:20s} {raw_w[nm]:2d} / {holm_w[nm]:2d} / {bh_w[nm]:2d}")

    # Plots
    print("  Generating heatmap …")
    plot_dm_heatmap(stat_df, pval_df, meta, task, OUTDIR)

    print("  Generating strategy comparison …")
    plot_strategy_comparison(actual, preds, meta, task, OUTDIR)

    # Results table
    tbl = build_results_table(stat_df, pval_df, actual, preds, meta, task)
    tbl_path = OUTDIR / f"dm_{task}_results.csv"
    tbl.to_csv(tbl_path, index=False)
    print(f"  → {tbl_path.name}")
    print()
    print(tbl.to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("  Diebold-Mariano Forecast Comparison — Q1 2026")
    print("=" * 60)
    for task in ("price", "load"):
        run_task(task)
    print(f"\nAll outputs saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
