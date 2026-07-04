#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_algorithm_figures.py
==============================
Generates four 100%-original algorithm diagrams for the thesis, replacing
any textbook/paper-adapted illustrations. Pure matplotlib, no external data,
white background, academic style.

Output (saved next to this script's ../  i.e. thesis_output/):
    fig_random_forest.png
    fig_leafwise_levelwise.png
    fig_svr_etube.png
    fig_mlp_architecture.png

Run:
    python generate_algorithm_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 0.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# Thesis palette
NAVY   = "#1d3557"   # primary dark blue
STEEL  = "#457b9d"   # steel blue
GREEN  = "#1a7c3e"   # accent green
LGREEN = "#52b788"
PURPLE = "#9d4edd"
GRAY   = "#6c757d"
LIGHT  = "#e9eef4"
EDGE   = "#2b3a4a"

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def _box(ax, x, y, w, h, text, fc=LIGHT, ec=EDGE, fs=12, fw="normal", tc="black", rad=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.01,rounding_size={rad}",
                 fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color=tc, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.4, style="-|>", mut=12, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle=style, mutation_scale=mut,
                 lw=lw, color=color, ls=ls, zorder=1,
                 shrinkA=2, shrinkB=2))


# ----------------------------------------------------------------------
# 1. Random Forest  (bagging ensemble for regression)
# ----------------------------------------------------------------------
def fig_random_forest():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # Training set
    _box(ax, 1, 42, 17, 16, "Training set\n$D$  ($N$ samples)", fc=NAVY, tc="white", fw="bold", fs=11.5)

    # Bootstrap samples
    n = 3
    ys = [72, 44, 16]
    labels = [r"$D_1$", r"$D_2$", r"$D_B$"]
    for i, (yy, lab) in enumerate(zip(ys, labels)):
        _box(ax, 27, yy-6, 13, 12, f"Bootstrap\nsample {lab}", fc=LIGHT, fs=11)
        _arrow(ax, 18, 50, 27, yy)
        if i == 1:
            ax.text(46.5, yy, r"$\vdots$", ha="center", va="center", fontsize=20)
    ax.text(8.5, 30, "bootstrap\n(sampling with\nreplacement)",
            ha="center", va="center", fontsize=9, color=GRAY, style="italic")

    # Trees (each fit on a random feature subset)
    def draw_tree(cx, cy, color):
        # tiny CART glyph
        pts = {"r": (cx, cy+7), "l1": (cx-4, cy+1), "r1": (cx+4, cy+1),
               "ll": (cx-6, cy-5), "lr": (cx-2, cy-5),
               "rl": (cx+2, cy-5), "rr": (cx+6, cy-5)}
        edges = [("r","l1"),("r","r1"),("l1","ll"),("l1","lr"),("r1","rl"),("r1","rr")]
        for a, b in edges:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                    color=color, lw=1.6, zorder=2)
        for k, (px, py) in pts.items():
            leaf = k in ("ll","lr","rl","rr")
            ax.add_patch(Circle((px, py), 1.4 if not leaf else 1.6,
                         fc="white" if not leaf else color,
                         ec=color, lw=1.6, zorder=3))

    tree_y = ys
    for yy in tree_y:
        draw_tree(58, yy, STEEL)
        _arrow(ax, 40, yy, 51, yy)
    ax.text(58, 86, "Decision trees\n(random feature subsets)",
            ha="center", va="center", fontsize=10, color=NAVY, fontweight="bold")

    # individual predictions converge into the aggregator
    for yy in tree_y:
        _arrow(ax, 65, yy, 76, 50)
    _box(ax, 76, 38, 14, 24, r"Aggregate" + "\n" + r"$\hat{y}=\frac{1}{B}\sum_{b} \hat{y}_b$",
         fc=GREEN, tc="white", fw="bold", fs=12)

    _arrow(ax, 90, 50, 96, 50)
    ax.text(97.5, 50, r"$\hat{y}$", ha="left", va="center", fontsize=15, fontweight="bold")

    ax.set_title("Random Forest — bagging ensemble for regression",
                 fontsize=14, fontweight="bold", color=NAVY, pad=8)
    return _save(fig, "fig_random_forest.png")


# ----------------------------------------------------------------------
# 2. Level-wise vs Leaf-wise tree growth
# ----------------------------------------------------------------------
def fig_leafwise_levelwise():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    def node(ax, x, y, color, r=0.9, grown=True):
        ax.add_patch(Circle((x, y), r, fc=color if grown else "white",
                     ec=EDGE, lw=1.6, zorder=3))

    def edge(ax, p, q, color=GRAY):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.8, zorder=2)

    # ---- Level-wise (depth-wise) : XGBoost-style ----
    ax = axes[0]
    ax.set_xlim(-3, 16); ax.set_ylim(0, 12); ax.axis("off")
    L = {
        "root": (8, 10),
        "a": (4, 7), "b": (12, 7),
        "c": (2, 4), "d": (6, 4), "e": (10, 4), "f": (14, 4),
    }
    edges = [("root","a"),("root","b"),("a","c"),("a","d"),("b","e"),("b","f")]
    for p, q in edges:
        edge(ax, L[p], L[q])
    for k in L:
        node(ax, *L[k], STEEL)
    # depth braces
    for yy, lab in [(10, "depth 0"), (7, "depth 1"), (4, "depth 2")]:
        ax.text(-2.7, yy, lab, ha="left", va="center", fontsize=10, color=GRAY, style="italic")
    ax.set_title("Level-wise growth (e.g. XGBoost)\nwhole level split at once",
                 fontsize=12.5, fontweight="bold", color=NAVY)

    # ---- Leaf-wise (best-first) : LightGBM-style ----
    ax = axes[1]
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")
    Lf = {
        "root": (8, 10),
        "a": (4, 7), "b": (12, 7),
        "c": (1.5, 4), "d": (6.5, 4),     # split the high-gain leaf 'a'
        "e": (4.5, 1), "f": (8.5, 1),     # deepen further on the best leaf 'd'
    }
    edges = [("root","a"),("root","b"),("a","c"),("a","d"),("d","e"),("d","f")]
    for p, q in edges:
        edge(ax, Lf[p], Lf[q])
    grown = {"root", "a", "b", "c", "d", "e", "f"}
    for k in Lf:
        node(ax, *Lf[k], LGREEN if k in {"a","c","d","e","f"} else STEEL)
    # highlight the max-delta-loss path
    ax.annotate("splits the leaf with\nmax $\\Delta$loss first",
                xy=Lf["d"], xytext=(12.2, 2.2),
                fontsize=10, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    ax.set_title("Leaf-wise growth (e.g. LightGBM)\nbest-first, deeper unbalanced tree",
                 fontsize=12.5, fontweight="bold", color=NAVY)

    fig.suptitle("Decision-tree growth strategies", fontsize=14.5,
                 fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return _save(fig, "fig_leafwise_levelwise.png")


# ----------------------------------------------------------------------
# 3. SVR  epsilon-tube
# ----------------------------------------------------------------------
def fig_svr_etube():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    x = np.linspace(0, 10, 200)
    slope, intercept = 0.85, 1.2
    f = slope * x + intercept
    eps = 1.4

    # tube
    ax.fill_between(x, f - eps, f + eps, color=STEEL, alpha=0.16,
                    label=r"$\varepsilon$-insensitive tube", zorder=1)
    ax.plot(x, f, color=NAVY, lw=2.4, label=r"regression $f(x)=\langle w,x\rangle + b$", zorder=3)
    ax.plot(x, f + eps, color=STEEL, lw=1.3, ls="--", zorder=2)
    ax.plot(x, f - eps, color=STEEL, lw=1.3, ls="--", zorder=2)

    # data points
    xs = rng.uniform(0.5, 9.5, 40)
    noise = rng.normal(0, 1.05, xs.size)
    ys = slope * xs + intercept + noise
    resid = ys - (slope * xs + intercept)
    inside = np.abs(resid) <= eps
    ax.scatter(xs[inside], ys[inside], s=42, color=GRAY, ec="white",
               lw=0.6, zorder=4, label="inside tube (no penalty)")
    ax.scatter(xs[~inside], ys[~inside], s=58, color=PURPLE, ec="black",
               lw=0.7, zorder=5, label=r"outside (slack $\xi$ penalised)")

    # one slack illustration
    j = np.argmax(resid)
    ax.plot([xs[j], xs[j]], [slope*xs[j]+intercept+eps, ys[j]],
            color=PURPLE, lw=1.6, zorder=4)
    ax.annotate(r"$\xi_i$", (xs[j]+0.15, (slope*xs[j]+intercept+eps+ys[j])/2),
                color=PURPLE, fontsize=14, fontweight="bold")

    # epsilon brace
    xa = 1.2
    ax.annotate("", (xa, slope*xa+intercept), (xa, slope*xa+intercept+eps),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.3))
    ax.text(xa+0.18, slope*xa+intercept+eps/2, r"$\varepsilon$",
            color=NAVY, fontsize=15, fontweight="bold", va="center")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(r"Support Vector Regression — the $\varepsilon$-insensitive tube",
                 fontsize=14, fontweight="bold", color=NAVY, pad=8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#c9d2dc")
    return _save(fig, "fig_svr_etube.png")


# ----------------------------------------------------------------------
# 4. MLP feedforward architecture
# ----------------------------------------------------------------------
def fig_mlp_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_aspect("equal")

    layers = [4, 6, 6, 1]                      # input, hidden, hidden, output
    layer_x = [1.2, 4.0, 6.8, 9.2]
    colors = [STEEL, NAVY, NAVY, GREEN]
    names = ["Input layer\n(features)", "Hidden layer 1\n(ReLU)",
             "Hidden layer 2\n(ReLU)", "Output\n$\\hat{y}$"]

    def ys_for(n):
        span = 7.2
        if n == 1:
            return [5.0]
        top = 5.0 + span/2
        return [top - i*(span/(n-1)) for i in range(n)]

    coords = []
    for nx, x in zip(layers, layer_x):
        coords.append([(x, y) for y in ys_for(nx)])

    # edges (fully connected)
    for li in range(len(layers)-1):
        for (x0, y0) in coords[li]:
            for (x1, y1) in coords[li+1]:
                ax.plot([x0, x1], [y0, y1], color="#c4ccd6", lw=0.7, zorder=1)

    # nodes
    for li, (layer, col, nm) in enumerate(zip(coords, colors, names)):
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), 0.30, fc=col, ec="white", lw=1.4, zorder=3))
        ymin = min(p[1] for p in layer)
        ax.text(layer[0][0], ymin-1.1, nm, ha="center", va="top",
                fontsize=10.5, color=NAVY, fontweight="bold")

    # show "..." for omitted input features
    ax.text(layer_x[0], ys_for(4)[-1]-0.55, r"$\vdots$", ha="center", fontsize=16, color=STEEL)

    ax.set_title("Multilayer Perceptron — feedforward architecture",
                 fontsize=14, fontweight="bold", color=NAVY, pad=4)
    return _save(fig, "fig_mlp_architecture.png")


if __name__ == "__main__":
    print("Generating algorithm figures ->", OUTDIR)
    fig_random_forest()
    fig_leafwise_levelwise()
    fig_svr_etube()
    fig_mlp_architecture()
    print("Done.")
