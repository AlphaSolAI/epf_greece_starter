#!/usr/bin/env python3
"""
generate_missing_diagrams.py
Generates 3 conceptual diagrams missing from thesis_output/:
  fig_multistep_strategies.png
  fig_data_pipeline.png
  xronodiagramma_agorwn.png
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "thesis_output"
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.linewidth": 0,
})

def _save(fig, name):
    p = OUTDIR / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. fig_multistep_strategies.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_multistep_strategies():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Multi-Step Forecasting Strategies", fontsize=14, fontweight="bold", y=1.01)

    HORIZON = 5
    COLORS = {"actual": "#475569", "given": "#3b82f6", "forecast": "#f97316",
              "model_in": "#22c55e", "arrow": "#64748b"}

    for ax_idx, (ax, title, subtitle) in enumerate(zip(
        axes,
        ["Direct", "Recursive", "MIMO"],
        ["H separate models\n(one per horizon)",
         "Single model\n(autoregressive rollout)",
         "Single model\n(all horizons jointly)"]
    )):
        ax.set_xlim(-0.5, HORIZON + 0.5)
        ax.set_ylim(-1.2, 3.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold",
                     color="#1e293b", pad=6)

        # Past observations (blue) — 3 steps
        for i in range(3):
            rect = FancyBboxPatch((i, 1.5), 0.85, 0.85,
                                  boxstyle="round,pad=0.05",
                                  facecolor=COLORS["given"], edgecolor="white", lw=1.5)
            ax.add_patch(rect)
            ax.text(i + 0.42, 1.92, f"$y_{{t-{2-i}}}$", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

        # Forecast steps (orange)
        for h in range(HORIZON):
            fc = COLORS["forecast"]
            rect = FancyBboxPatch((h, 0.0), 0.85, 0.85,
                                  boxstyle="round,pad=0.05",
                                  facecolor=fc, edgecolor="white", lw=1.5, alpha=0.9)
            ax.add_patch(rect)
            ax.text(h + 0.42, 0.42, f"$\\hat{{y}}_{{t+{h+1}}}$", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")

        if title == "Direct":
            # H separate arrows from past → each forecast box
            for h in range(HORIZON):
                ax.annotate("", xy=(h + 0.42, 0.85), xytext=(1.42, 1.5),
                            arrowprops=dict(arrowstyle="->", color=COLORS["arrow"],
                                            lw=1.2, connectionstyle="arc3,rad=0.0"))
            # model boxes
            for h in range(HORIZON):
                ax.text(h + 0.42, -0.55, f"$M_{{{h+1}}}$", ha="center", va="center",
                        fontsize=8.5, color="#6d28d9", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#ede9fe", ec="#7c3aed", lw=0.8))

        elif title == "Recursive":
            # single arrow from last past into h=1, then chained
            ax.annotate("", xy=(0.42, 0.85), xytext=(2.42, 1.5),
                        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.4))
            for h in range(HORIZON - 1):
                ax.annotate("", xy=(h + 1.42, 0.42), xytext=(h + 0.85, 0.42),
                            arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2))
            ax.text(2.42, -0.55, "$M$", ha="center", va="center",
                    fontsize=11, color="#6d28d9", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ede9fe", ec="#7c3aed", lw=0.8))

        else:  # MIMO
            # single wide arrow from all past → all forecast jointly
            ax.annotate("", xy=(2.42, 0.85), xytext=(1.42, 1.5),
                        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.8))
            ax.text(2.42, -0.55, "$M_{MIMO}$", ha="center", va="center",
                    fontsize=10, color="#6d28d9", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ede9fe", ec="#7c3aed", lw=0.8))
            # horizontal brace-like line
            ax.plot([0.0, 4.85], [-0.18, -0.18], color="#7c3aed", lw=1.2, linestyle="--")
            ax.text(2.42, -0.38, "all H jointly", ha="center", fontsize=8, color="#7c3aed")

    # Legend
    legend_els = [
        mpatches.Patch(facecolor=COLORS["given"],    label="Known observations"),
        mpatches.Patch(facecolor=COLORS["forecast"], label="Forecast horizon"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    _save(fig, "fig_multistep_strategies.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. fig_data_pipeline.png
# ══════════════════════════════════════════════════════════════════════════════
def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Data Processing Pipeline", fontsize=14, fontweight="bold",
                 color="#1e293b", pad=10)

    stages = [
        ("Raw\nSources",     "#dbeafe", "#2563eb", 0.7),
        ("Data\nCollection", "#dcfce7", "#16a34a", 2.3),
        ("Cleaning &\nImputation", "#fef9c3", "#ca8a04", 3.9),
        ("Feature\nEngineering", "#fce7f3", "#db2777", 5.5),
        ("Train / Test\nSplit", "#ede9fe", "#7c3aed", 7.1),
        ("Model\nTraining", "#ffedd5", "#ea580c", 8.7),
        ("Evaluation", "#f0fdf4", "#15803d", 10.3),
    ]

    box_w, box_h = 1.35, 1.6
    cy = 3.0

    for i, (label, fc, ec, cx) in enumerate(stages):
        rect = FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                              boxstyle="round,pad=0.12",
                              facecolor=fc, edgecolor=ec, lw=2.0)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#1e293b", multialignment="center")

        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i+1][3] - box_w/2 - 0.05, cy),
                        xytext=(cx + box_w/2 + 0.05, cy),
                        arrowprops=dict(arrowstyle="-|>", color="#64748b",
                                        lw=1.6, mutation_scale=14))

    # Sub-labels
    sublabels = [
        (0.7,  1.55, "ENTSO-E\nHENEX\nECMWF"),
        (2.3,  1.55, "Hourly price\nLoad\nWeather"),
        (3.9,  1.55, "Outliers\nMissing values\nCalendar flags"),
        (5.5,  1.55, "Lags · Rolling\nFourier · Exog.\nHolidays"),
        (7.1,  1.55, "Jan 2017–\nNov 2025 train\n2025 test"),
        (8.7,  1.55, "LGBM / XGB\nRF / MLP / SVR"),
        (10.3, 1.55, "MAE · RMSE\nDM test\nFI analysis"),
    ]
    for cx, cy_sub, txt in sublabels:
        ax.text(cx, cy_sub, txt, ha="center", va="top",
                fontsize=7.5, color="#475569", multialignment="center",
                linespacing=1.4)

    fig.tight_layout()
    _save(fig, "fig_data_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. xronodiagramma_agorwn.png  (electricity market timeline)
# ══════════════════════════════════════════════════════════════════════════════
def xronodiagramma_agorwn():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(-1, 25)
    ax.set_ylim(-2.5, 4.5)
    ax.axis("off")
    ax.set_title("Χρονοδιάγραμμα Λειτουργίας Αγορών Ηλεκτρισμού\n"
                 "Electricity Market Timeline (Day-Ahead to Real-Time)",
                 fontsize=12, fontweight="bold", color="#1e293b", pad=8)

    # Timeline axis
    ax.annotate("", xy=(24, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2.0, mutation_scale=16))

    # Hours D-1 → D+1
    for h in range(0, 25, 4):
        ax.plot([h, h], [-0.15, 0.15], color="#64748b", lw=1.0)
        ax.text(h, -0.45, f"{h:02d}:00", ha="center", fontsize=8, color="#475569")
    ax.text(-0.5, -0.8, "D-1", ha="center", fontsize=9, color="#64748b")
    ax.text(12, -0.8, "Day D", ha="center", fontsize=9, color="#64748b")
    ax.text(24.2, -0.45, "Hour →", ha="left", fontsize=8, color="#475569")

    # Market events
    events = [
        # (x_start, x_end, y_center, color, label, sublabel)
        (0.0,  3.0, 2.5, "#bfdbfe", "#1d4ed8", "Pre-Market\nAnalysis",    "D-1  00:00–03:00"),
        (3.0,  5.0, 2.5, "#bbf7d0", "#15803d", "Day-Ahead\nBid Submission","D-1  03:00–05:00"),
        (5.0,  8.0, 2.5, "#fde68a", "#b45309", "DA Auction\n& Clearing",   "D-1  05:00–08:00"),
        (8.0, 12.0, 2.5, "#fecaca", "#b91c1c", "Intraday\nMarket Opens",   "D-1  08:00 ↔ D 12:00"),
        (12.0,18.0, 2.5, "#ddd6fe", "#6d28d9", "Real-Time\nBalancing",     "D  12:00–18:00"),
        (18.0,24.0, 2.5, "#fed7aa", "#c2410c", "Settlement &\nReporting",  "D  18:00–24:00"),
    ]

    bh = 0.9
    for xs, xe, yc, fc, ec, label, sublabel in events:
        rect = FancyBboxPatch((xs + 0.1, yc - bh/2), xe - xs - 0.2, bh,
                              boxstyle="round,pad=0.06",
                              facecolor=fc, edgecolor=ec, lw=1.5, alpha=0.88)
        ax.add_patch(rect)
        cx = (xs + xe) / 2
        ax.text(cx, yc + 0.02, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#1e293b", multialignment="center")
        ax.text(cx, yc - bh/2 - 0.28, sublabel, ha="center", va="top",
                fontsize=7.0, color="#64748b")

    # Key deadlines
    deadlines = [(5.0, "Bid\nDeadline", "#b91c1c"), (8.0, "Results\nPublished", "#15803d")]
    for xd, lbl, col in deadlines:
        ax.plot([xd, xd], [-0.18, 2.05], color=col, lw=1.4, linestyle="--", alpha=0.7)
        ax.text(xd, -1.8, lbl, ha="center", fontsize=8, color=col, fontweight="bold",
                multialignment="center")

    # HENEX / ENTSO-E label
    ax.text(12, -2.2, "Data sources: HENEX (Greek Day-Ahead market) · ENTSO-E (European grid data)",
            ha="center", fontsize=8, color="#64748b", style="italic")

    fig.tight_layout()
    _save(fig, "xronodiagramma_agorwn.png")


if __name__ == "__main__":
    print("Generating 3 missing conceptual diagrams ...")
    fig_multistep_strategies()
    fig_data_pipeline()
    xronodiagramma_agorwn()
    print("Done.")
