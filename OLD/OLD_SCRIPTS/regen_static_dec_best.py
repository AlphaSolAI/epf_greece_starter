"""
regen_static_dec_best.py
========================
ΞΕΧΩΡΙΣΤΟ, αυτόνομο script (δεν αγγίζει το regen_kept_light.py ούτε κανένα άλλο
figure). Παράγει ΜΟΝΟ δύο νέα διαγράμματα:

    thesis_output/ch6_forecast_analysis/static_dec_best_price_mae.png
    thesis_output/ch6_forecast_analysis/static_dec_best_load_mae.png

Περιεχόμενο: το ΚΑΛΥΤΕΡΟ (χαμηλότερο MAE, εξαιρώντας baselines Naive/Seasonal)
μοντέλο από ΚΑΘΕ στρατηγική, για DEC 2025 STATIC ΜΟΝΟ.

Στρατηγικές & πηγές (Dec 2025 static, 744h):
    CL (Teacher-Forcing static) → dashboard_data_hourly_{task}_cl_monthly.json
    OL (Recursive static)       → dashboard_data_hourly_{task}_openloop_h24_monthly.json
    MIMO static                 → dashboard_data_hourly_{task}_mimo_h24_monthly.json
    Direct static               → dashboard_data_hourly_{task}_direct_h24_monthly.json

Η αισθητική (light theme, palette, γραμματοσειρές) αντιγράφεται από το
regen_kept_light.py ώστε να ταιριάζει με τα υπόλοιπα Chapter-6 figures, αλλά
τίποτα δεν εισάγεται/τροποποιείται από εκείνο το script.

Usage:
  conda run -n epf --no-capture-output python regen_static_dec_best.py
"""
from __future__ import annotations
import json, time, shutil, tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CH6  = ROOT / "thesis_output" / "ch6_forecast_analysis"
TMP  = Path(tempfile.gettempdir()) / "ch6_static_dec_best"
TMP.mkdir(parents=True, exist_ok=True)

# ── Light style (ίδιο με regen_kept_light.py) ───────────────────────────────
GRID_C    = "#aeaeae"
DARK_TEXT = "#222222"
BIG_TITLE, BIG_LABEL, BIG_TICK = 21, 17, 14

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   DARK_TEXT,
    "text.color":        DARK_TEXT,
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "grid.color":        GRID_C,
    "grid.linestyle":    (0, (1, 1.5)),
    "grid.linewidth":    1.0,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Static-regime χρώματα ανά στρατηγική (dark shades από regen_kept_light) ──
#   TF static #0d47a1 · Rec static #e65100 · MIMO static #8e24aa · Direct static #c2185b
# Το mimo_h24 JSON περιέχει και Direct-family μοντέλα (design session 23). Για να
# μη δείχνουν MIMO & Direct το ΙΔΙΟ μοντέλο, εξαιρούμε τα "DIRECT"-labeled entries
# από το MIMO pool ώστε να επιλεγεί γνήσιο MIMO μοντέλο/ensemble.
STRATEGIES = [
    ("TF Static",     "cl",           "#0d47a1", None),
    ("Rec Static",    "openloop_h24", "#e65100", None),
    ("MIMO Static",   "mimo_h24",     "#8e24aa", "direct"),
    ("Direct Static", "direct_h24",   "#c2185b", None),
]


def _is_baseline(name: str) -> bool:
    n = name.strip().lower()
    return n.startswith("naive") or n.startswith("seasonal")


def _best_of_json(task: str, key: str, exclude_substr: str | None = None):
    """Επιστρέφει (model_name, mae) του καλύτερου μη-baseline μοντέλου."""
    fp = ROOT / f"dashboard_data_hourly_{task}_{key}_monthly.json"
    d = json.load(open(fp, encoding="utf-8"))
    actual = np.asarray(d["actual"], dtype=float)
    best_name, best_mae = None, np.inf
    for name, vals in d["series"].items():
        if _is_baseline(name):
            continue
        if exclude_substr is not None and exclude_substr in name.lower():
            continue
        arr = np.asarray(vals, dtype=float)
        m = np.isfinite(actual) & np.isfinite(arr)
        if m.sum() == 0:
            continue
        mae = float(np.mean(np.abs(actual[m] - arr[m])))
        if mae < best_mae:
            best_name, best_mae = name, mae
    return best_name, best_mae


def _task_gr(task: str) -> str:
    return "Τιμή" if task == "price" else "Φορτίο"


def plot_static_dec_best(task: str):
    unit = "€/MWh" if task == "price" else "MW"

    labels, colours, maes, models = [], [], [], []
    for disp, key, col, excl in STRATEGIES:
        name, mae = _best_of_json(task, key, excl)
        labels.append(disp)
        colours.append(col)
        maes.append(mae)
        models.append(name)
        print(f"  {task:5s} | {disp:14s} → {name}  MAE={mae:.3f} {unit}")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    bars = ax.bar(x, maes, width=0.62, color=colours, alpha=0.92,
                  edgecolor="#222222", linewidth=1.0, zorder=3)

    ymax = max(maes)
    for bar, mae, name in zip(bars, maes, models):
        bx = bar.get_x() + bar.get_width() / 2
        # MAE (μεγάλο, έντονο, πάνω από τη μπάρα)
        ax.annotate(f"{mae:.2f}", xy=(bx, mae), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=BIG_TICK + 3, fontweight="bold", color=bar.get_facecolor())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=BIG_LABEL, fontweight="bold")
    for tick, col in zip(ax.get_xticklabels(), colours):
        tick.set_color(col)
    ax.tick_params(axis="y", labelsize=BIG_TICK)
    ax.set_ylabel(f"MAE ({unit})", fontsize=BIG_LABEL, fontweight="bold")
    ax.set_ylim(0, ymax * 1.18)
    ax.set_title(f"Καλύτερο Μοντέλο ανά Στρατηγική — {_task_gr(task)} (Δεκέμβριος 2025, Static)",
                 fontsize=BIG_TITLE, fontweight="bold", pad=16)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()

    name = f"static_dec_best_{task}_mae.png"
    tmp_path = TMP / name
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    dst = CH6 / name
    shutil.copy2(tmp_path, dst)
    st = dst.stat()
    print(f"  → {name}  ({st.st_size} bytes, mtime {time.strftime('%H:%M:%S', time.localtime(st.st_mtime))})")


def main():
    print(f"Static Dec-2025 best-per-strategy figures → {CH6}")
    print(f"  (temp staging: {TMP})")
    for task in ("price", "load"):
        plot_static_dec_best(task)
    print("Done.")


if __name__ == "__main__":
    main()
