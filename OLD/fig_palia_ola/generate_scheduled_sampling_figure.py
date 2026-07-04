#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scheduled_sampling_figure.py
=====================================
Generates fig_scheduled_sampling.png illustrating the Scheduled-Sampling
curriculum: epsilon = P(feed model's own prediction) ramps from a warmup of
0.0 (pure teacher forcing) up to 0.40 across 3 SS passes. This visualises the
teacher-forcing -> autoregressive curriculum used to mitigate exposure bias.

Output: thesis_output/fig_scheduled_sampling.png
Run:    python generate_scheduled_sampling_figure.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

NAVY   = "#1d3557"
STEEL  = "#457b9d"
GREEN  = "#1a7c3e"
LGREEN = "#52b788"
PURPLE = "#9d4edd"
GRAY   = "#6c757d"

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    # ---- schedule definition -------------------------------------------
    warmup = 5                 # epochs of pure teacher forcing
    pass_len = 10              # epochs per SS pass
    eps_levels = [0.10, 0.25, 0.40]   # epsilon held per pass (warmup -> increase)

    epochs = np.arange(0, warmup + len(eps_levels) * pass_len)
    eps = np.zeros_like(epochs, dtype=float)
    for e in epochs:
        if e < warmup:
            eps[e] = 0.0
        else:
            p = (e - warmup) // pass_len
            p = min(p, len(eps_levels) - 1)
            eps[e] = eps_levels[p]

    # smooth reference curve (inverse-sigmoid style ramp) for intuition
    x_smooth = np.linspace(warmup, epochs[-1], 200)
    k = 0.45
    mid = warmup + 1.5 * pass_len
    smooth = 0.40 / (1 + np.exp(-k * (x_smooth - mid)))

    fig, ax = plt.subplots(figsize=(10, 5.8))

    # warmup band
    ax.axvspan(0, warmup, color=NAVY, alpha=0.06)
    ax.text(warmup / 2, 0.43, "warm-up\n(pure teacher\nforcing)",
            ha="center", va="center", fontsize=9.5, color=NAVY, style="italic")

    # per-pass bands + step line
    colors = [LGREEN, STEEL, PURPLE]
    for i, lvl in enumerate(eps_levels):
        x0 = warmup + i * pass_len
        x1 = warmup + (i + 1) * pass_len
        ax.axvspan(x0, x1, color=colors[i], alpha=0.10)
        ax.text((x0 + x1) / 2, lvl + 0.022, f"SS pass {i+1}\n$\\varepsilon={lvl:.2f}$",
                ha="center", va="bottom", fontsize=10, color=colors[i], fontweight="bold")

    # staircase (actual schedule used in training)
    ax.step(epochs, eps, where="post", color=NAVY, lw=2.6,
            label=r"$\varepsilon$ schedule (per pass)", zorder=4)
    ax.scatter(epochs, eps, s=14, color=NAVY, zorder=5)

    # smooth reference
    ax.plot(x_smooth, smooth, color=GRAY, lw=1.6, ls="--",
            label="smooth ramp (intuition)", zorder=3)

    # curriculum annotations
    ax.annotate("teacher forcing\n($\\varepsilon=0$: feed ground-truth lag)",
                xy=(1.5, 0.0), xytext=(1.5, 0.16),
                fontsize=9.5, color=NAVY, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.3))
    ax.annotate("more autoregressive\n($\\varepsilon\\!\\uparrow$: feed own prediction)",
                xy=(epochs[-1] - 0.5, 0.40), xytext=(15.5, 0.47),
                fontsize=9.5, color=PURPLE, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.3))

    ax.set_xlabel("training epoch", fontsize=12)
    ax.set_ylabel(r"$\varepsilon$  =  P(use model's own prediction)", fontsize=12)
    ax.set_ylim(-0.02, 0.52)
    ax.set_xlim(0, epochs[-1])
    ax.set_title("Scheduled Sampling — teacher-forcing $\\rightarrow$ autoregressive curriculum",
                 fontsize=13.5, fontweight="bold", color=NAVY, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    for s in ax.spines.values():
        s.set_color("#c9d2dc")

    path = os.path.join(OUTDIR, "fig_scheduled_sampling.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
