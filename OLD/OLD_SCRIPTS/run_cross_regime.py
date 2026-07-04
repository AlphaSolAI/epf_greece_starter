"""
Cross-Regime Performance Delta Plot
Same dark theme as src/thesis_plots.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTDIR = Path(r"C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter\thesis_output\ch6_forecast_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ─── Exact same dark theme as src/thesis_plots.py ─────────────────────────────
BG   = "#0f172a"
BG2  = "#1e293b"
FG   = "#e2e8f0"
GRID = "#334155"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    "#475569",
    "axes.labelcolor":   FG,
    "text.color":        FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "grid.color":        GRID,
    "grid.alpha":        0.5,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.framealpha": 0.85,
    "legend.facecolor":  BG2,
    "legend.edgecolor":  "#475569",
})


def plot_cross_regime_delta(task="price"):
    if task == "price":
        models      = ["LightGBM", "Ensemble-Best3", "XGBoost", "Random Forest", "MLP-Optuna"]
        static_mae  = [14.556,  14.478,  15.888,  15.161,  17.917]
        retrain_mae = [10.463,  10.491,  10.913,  12.806,  15.620]
        # Colors from thesis_plots.py palette — map to model type
        colours = ["#38bdf8", "#7dd3fc", "#f97316", "#22c55e", "#a855f7"]
        unit = "€/MWh"
        num  = "25"
    else:
        models      = ["Ensemble-Best3", "LightGBM", "XGBoost", "MLP-Optuna", "Random Forest"]
        static_mae  = [96.751,  98.401,  100.531,  104.120,  113.449]
        retrain_mae = [67.729,  69.435,   72.925,   88.493,  100.177]
        colours = ["#7dd3fc", "#38bdf8", "#f97316", "#a855f7", "#22c55e"]
        unit = "MW"
        num  = "26"

    deltas = np.array(static_mae) - np.array(retrain_mae)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)

    x = np.arange(len(models))
    bars = ax.bar(x, deltas, color=colours, alpha=0.85,
                  edgecolor="#1e293b", width=0.55, linewidth=0.8)

    for bar, col in zip(bars, colours):
        height = bar.get_height()
        ax.annotate(f"+{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5, color=FG)
    ax.tick_params(axis="y", labelsize=8.5, colors=FG)
    ax.set_ylabel(f"MAE Reduction ({unit})", fontsize=9.5, color=FG)
    ax.set_title(
        f"Cross-Regime Performance Delta ({task.upper()})\n"
        f"ΔMAE = MAE(Static) − MAE(Monthly Retrain) — Q1 2026",
        fontsize=10, color=FG, pad=12
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5, color=GRID)
    ax.set_ylim(0, deltas.max() * 1.2)

    plt.tight_layout()
    out = OUTDIR / f"{num}_cross_regime_delta_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out.name}")


plot_cross_regime_delta("price")
plot_cross_regime_delta("load")
print("Done.")
