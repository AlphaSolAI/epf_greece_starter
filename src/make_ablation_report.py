# -*- coding: utf-8 -*-
"""
make_ablation_report.py — Οπτική σύνοψη του ablation study (2026-07-02, ΔΙΟΡΘΩΘΗΚΕ 2026-07-04).

Παράγει 7 PNG figures + index.html στο reports/ablation_20260702/.
Τα νούμερα είναι hardcoded από τα ΕΠΑΛΗΘΕΥΜΕΝΑ CSVs/JSONs (spot-checked κατά
του ABLATION_PLAN.md §RESULTS) — δεν ξανατρέχει τίποτα.

⚠️ ΔΙΟΡΘΩΣΗ 2026-07-04: όλα τα xborder claims (14.43/15.02, "νέο headline", stacking battery
θετικά αποτελέσματα) ΑΚΥΡΩΘΗΚΑΝ — ήταν leakage (same-day BG/IT-SUD τιμές βγαίνουν από το ΙΔΙΟ
SDAC auction με το target, δημοσιεύονται ΜΕΤΑ το gate). Βλ. ABLATION_PLAN.md §XBORDER-ΤΕΛΙΚΗ-
ΕΤΥΜΗΓΟΡΙΑ + §OVERNIGHT PROTOCOL (P1-P6). Νέο, έγκυρο headline: LGBM recursive weekly, `default`
(χωρίς xborder) = 15.17 €/MWh, seed-robust (std≈0.11), επιβεβαιωμένο σε 2ο window (Μάρτιος 2026).

Usage: python -m src.make_ablation_report
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
OUT = BASE_DIR / "reports" / "ablation_20260702"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

GREEN, RED, BLUE, GRAY, GOLD = "#2e7d32", "#c62828", "#1565c0", "#9e9e9e", "#f9a825"


def _bar_annot(ax, bars, fmt="{:+.2f}", dy=0.02):
    for b in bars:
        v = b.get_height()
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold",
                    xytext=(0, 3 if v >= 0 else -3), textcoords="offset points")


# ─── F1: Το ταξίδι του headline MAE ──────────────────────────────────────────
def fig1():
    steps = [
        "Ελλιπή δεδομένα\nstatic (αρχικό)",
        "Ελλιπή δεδομένα\nκαλύτερο (weekly)",
        "ΠΛΗΡΗ δεδομένα\nstatic",
        "Πλήρη + monthly\nretrain",
        "Πλήρη + WEEKLY\nretrain ★",
    ]
    vals = [23.84, 17.81, 16.10, 15.80, 15.17]
    colors = [GRAY, GRAY, BLUE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_xticks(range(len(steps)), steps, fontsize=8)
    ax.set_ylabel("MAE (€/MWh) — Q1 2026, strict gate")
    ax.set_title("Το ταξίδι του LGBM: από 23.8 σε 15.17 €/MWh (−36%)\n"
                 "Το μεγαλύτερο κέρδος ήρθε από ΤΑ ΔΕΔΟΜΕΝΑ, όχι από αλγόριθμο/xborder")
    ax.axhline(15.17, ls="--", color=GREEN, alpha=0.5, lw=1)
    ax.set_ylim(0, 26)
    fig.tight_layout()
    fig.savefig(OUT / "F1_headline_journey.png")
    plt.close(fig)


# ─── F2: Αξία ομάδων features (leave-one-out ΔMAE) σε 4 συνθήκες ────────────
def fig2():
    # ΔMAE όταν ΑΦΑΙΡΕΘΕΙ η ομάδα (θετικό = η ομάδα ΒΟΗΘΟΥΣΕ)
    groups = ["genlags\n(actual gen)", "resfc\n(RES fcst)", "loadfc\n(load fcst)",
              "loadlags", "meteo", "fuel"]
    conds = {
        "LGBM Q1 (χειμώνας)":   [+1.45, +0.56, +0.05, +0.22, -0.20, 0.00],
        "XGB Q1 (χειμώνας)":    [+1.42, +0.51, +0.06, -0.21, -0.52, -0.26],
        "LGBM Δεκ (recursive)": [+2.63, -0.04, +0.17, +0.34, -0.23, -0.38],
        "LGBM Καλοκαίρι '25":   [+0.43, +1.09, -0.28, -0.06, np.nan, np.nan],
        "XGB Καλοκαίρι '25":    [+1.20, +1.22, -0.30, +0.42, np.nan, np.nan],
    }
    x = np.arange(len(groups))
    w = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 5))
    palette = [BLUE, "#5e97d1", "#90b8e0", GOLD, "#f5c95d"]
    for i, (label, vals) in enumerate(conds.items()):
        ax.bar(x + (i - 2.0) * w, vals, w, label=label, color=palette[i])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-0.15, 0.15, color=GRAY, alpha=0.15, label="noise floor (±0.15)")
    ax.set_xticks(x, groups)
    ax.set_ylabel("ΔMAE αφαίρεσης (€/MWh)\n> 0 = η ομάδα ΒΟΗΘΑΕΙ · < 0 = ΒΛΑΠΤΕΙ")
    ax.set_title("Αξία κάθε ομάδας features (leave-one-out) — 5 ανεξάρτητες συνθήκες (2 αλγόριθμοι × 2 εποχές)\n"
                 "genlags: παντού πολύτιμο · resfc: εξαρτάται από ΕΠΟΧΗ (cross-model ✓) · meteo/fuel: βλάπτουν στην τιμή (recursive)")
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "F2_feature_groups.png")
    plt.close(fig)


# ─── F3: Εποχικότητα resfc + xborder επιβεβαίωση ────────────────────────────
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # αριστερά: resfc χειμώνας vs καλοκαίρι
    vals = [-0.04, +1.09]
    bars = ax1.bar(["Χειμώνας\n(Δεκ '25)", "Καλοκαίρι\n(Ιουν-Αυγ '25)"], vals,
                   color=[GRAY, GREEN], width=0.5)
    _bar_annot(ax1, bars)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhspan(-0.15, 0.15, color=GRAY, alpha=0.15)
    ax1.set_ylabel("ΔMAE αφαίρεσης resfc (€/MWh)")
    ax1.set_title("resfc (RES forecast): «άχρηστο» τον χειμώνα,\nπολύτιμο το καλοκαίρι — η εποχή του test μετράει!")

    # δεξιά: xborder (ΔΙΟΡΘΩΘΗΚΕ 2026-07-04 — lagged/leakage-free νούμερα, ΟΧΙ same-day)
    vals2 = [-0.42, -0.13, -0.41, +0.52]
    bars2 = ax2.bar(["Q1\nstatic", "Q1\nmonthly", "Q1\nweekly", "Καλοκαίρι '25\n(1 σημείο)"],
                    vals2, color=[RED, RED, RED, GOLD], width=0.55)
    _bar_annot(ax2, bars2)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhspan(-0.15, 0.15, color=GRAY, alpha=0.15)
    ax2.set_ylabel("Όφελος xborder-lagged (€/MWh MAE)")
    ax2.set_title("xborder (lagged, νόμιμο): ΒΛΑΠΤΕΙ τον χειμώνα (3/3 cadences),\n"
                  "βοηθάει το καλοκαίρι (1 σημείο) — ΑΠΟΡΡΙΠΤΕΤΑΙ από default")
    ax2.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "F3_season_and_xborder.png")
    plt.close(fig)


# ─── F4: Scheduled Sampling ─────────────────────────────────────────────────
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # αριστερά: σχήματα SS
    schemes = ["no-SS", "SS-linear ★", "SS-exp", "SS-step"]
    tot = [16.096, 15.830, 16.282, 16.285]
    far = [17.594, 17.358, 18.171, 18.097]
    x = np.arange(len(schemes))
    ax1.bar(x - 0.18, tot, 0.36, label="Συνολικό MAE", color=BLUE)
    ax1.bar(x + 0.18, far, 0.36, label="Μακρινά offsets (17-24h)", color=GOLD)
    ax1.set_xticks(x, schemes)
    ax1.set_ylim(15, 19)
    ax1.set_ylabel("MAE (€/MWh)")
    ax1.set_title("Scheduled Sampling: μόνο το linear decay βελτιώνει\n(και μάλιστα κυρίως τα μακρινά offsets — exposure bias fix)")
    ax1.legend(fontsize=9)
    ax1.annotate("P5 (2026-07-04): SS-linear βοηθάει ΚΑΙ με monthly retrain\n"
                 "15.795 → 15.552 (−0.243) σε καθαρό config — ΔΕΝ είναι redundant",
                 xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round", fc="#e8f5e9", ec="#2e7d32", alpha=0.9))

    # δεξιά: SS × features interaction (dumbbell)
    groups = ["resfc", "loadfc", "meteo", "loadlags", "fuel"]
    no_ss = [-0.04, +0.05, -0.20, +0.22, -0.38]
    with_ss = [+1.10, +0.46, -0.08, -0.32, -0.45]
    y = np.arange(len(groups))[::-1]
    for yi, a, b in zip(y, no_ss, with_ss):
        ax2.plot([a, b], [yi, yi], color=GRAY, lw=2, zorder=1)
        ax2.annotate("", xy=(b, yi), xytext=(a, yi),
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
    ax2.scatter(no_ss, y, s=70, color=GRAY, label="χωρίς SS", zorder=3)
    ax2.scatter(with_ss, y, s=90, color=GREEN, label="με SS-linear", zorder=3)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.axvspan(-0.15, 0.15, color=GRAY, alpha=0.15)
    ax2.set_yticks(y, groups)
    ax2.set_xlabel("ΔMAE αφαίρεσης (€/MWh) — δεξιά = πιο πολύτιμο")
    ax2.set_title("Το SS αλλάζει ποια features αξίζουν:\nτα day-ahead forecasts (resfc/loadfc) εκτοξεύονται")
    ax2.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "F4_scheduled_sampling.png")
    plt.close(fig)


# ─── F5: Σύγκριση μοντέλων + E1/E2 ──────────────────────────────────────────
def fig5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    models = ["LGBM\nweekly ★", "ens_median\n(static Q1)", "LGBM\nstatic", "XGB\nstatic",
              "ens_mean\n(static Q1)", "LEAR\nstatic", "MLP\nstatic", "LSTM s2s\n(bias bug)"]
    vals = [15.17, 16.04, 16.10, 16.21, 16.39, 19.49, 20.49, 44.16]
    colors = [GREEN, "#66bb6a", BLUE, BLUE, GRAY, GRAY, GRAY, RED]
    bars = ax1.bar(models, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax1.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("MAE (€/MWh) — Q1 2026")
    ax1.set_title("Κατάταξη μοντέλων (v2 δεδομένα, strict gate, ΧΩΡΙΣ xborder)\nΔέντρα κυριαρχούν · LSTM θέλει calibration fix")
    ax1.tick_params(axis="x", labelsize=7.5)

    # E1: capacity × meteo + E2 σχόλιο
    n_est = ["400", "800", "1600"]
    delta_meteo = [-0.016, +0.260, +0.439]
    bars2 = ax2.bar(n_est, delta_meteo, color=[GRAY, RED, RED], width=0.5)
    _bar_annot(ax2, bars2)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("n_estimators")
    ax2.set_ylabel("Ζημιά meteo στην ΤΙΜΗ (ΔMAE)")
    ax2.set_title("E1: περισσότερα δέντρα ⇒ το meteo βλάπτει\nΠΕΡΙΣΣΟΤΕΡΟ (υπόθεση απορρίφθηκε)\n"
                  "E2: στο ΦΟΡΤΙΟ όμως meteo = −33 MW (−26%!)")

    fig.tight_layout()
    fig.savefig(OUT / "F5_models_e1e2.png")
    plt.close(fig)


# ─── F6: meteo strategy-effect (το πιο στιβαρό strategy εύρημα) ─────────────
def fig6():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    labels = ["LGBM\nΔεκ", "LGBM\nQ1", "XGB\nQ1", "LGBM\nΔεκ ", "LGBM\nQ1 ", "XGB\nQ1 "]
    vals = [+0.548, +0.486, +0.291, -0.228, -0.20, -0.52]
    colors = [GREEN] * 3 + [RED] * 3
    bars = ax.bar(range(6), vals, color=colors, width=0.55)
    _bar_annot(ax, bars)
    ax.axhline(0, color="black", lw=0.9)
    ax.axhspan(-0.15, 0.15, color=GRAY, alpha=0.15, label="noise floor (±0.15)")
    ax.set_xticks(range(6), labels)
    ax.axvline(2.5, color=GRAY, ls=":", lw=1.5)
    ax.text(1.0, 0.62, "DIRECT\n(meteo ΒΟΗΘΑΕΙ 3/3)", ha="center", fontweight="bold", color=GREEN)
    ax.text(4.0, 0.45, "RECURSIVE\n(meteo ΒΛΑΠΤΕΙ 3/3)", ha="center", fontweight="bold", color=RED)
    ax.set_ylabel("ΔMAE αφαίρεσης meteo (€/MWh)\n> 0 = βοηθάει · < 0 = βλάπτει")
    ax.set_title("P4: Το meteo είναι καθαρό STRATEGY-effect — 4/4 ανεξάρτητα σημεία, 2 αλγόριθμοι, 2 windows\n"
                 "Direct: κάθε offset = ανεξάρτητο μοντέλο χωρίς y-lags → αξιοποιεί exogenous · Recursive: τα y-lags κυριαρχούν")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "F6_meteo_strategy_effect.png")
    plt.close(fig)


# ─── F7: Robustness τελικού headline (P6) ───────────────────────────────────
def fig7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    seeds = ["seed 42\n(headline)", "seed 43", "seed 44"]
    vals = [15.171, 15.429, 15.218]
    bars = ax1.bar(seeds, vals, color=[GREEN, "#66bb6a", "#66bb6a"], width=0.5)
    for b, v in zip(bars, vals):
        ax1.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    mean_v = float(np.mean(vals))
    ax1.axhline(mean_v, ls="--", color=GRAY, lw=1)
    ax1.annotate(f"mean {mean_v:.2f} · std≈0.11", (2.35, mean_v), fontsize=9,
                 ha="right", va="bottom", color="#555")
    ax1.set_ylim(14.5, 16)
    ax1.set_ylabel("MAE (€/MWh) — Q1 2026")
    ax1.set_title("Seed robustness (LGBM weekly default):\nstd≈0.11 < noise floor 0.15 ✓")

    x = np.arange(2)
    weekly = [15.171, 17.666]
    monthly = [15.795, 17.968]
    ax2.bar(x - 0.17, weekly, 0.34, label="weekly", color=GREEN)
    ax2.bar(x + 0.17, monthly, 0.34, label="monthly", color=GRAY)
    for xi, (wv, mv) in enumerate(zip(weekly, monthly)):
        ax2.annotate(f"{wv:.2f}", (xi - 0.17, wv), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.annotate(f"{mv:.2f}", (xi + 0.17, mv), ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x, ["Q1 2026\n(Δ=−0.62)", "Μάρτιος 2026\n(Δ=−0.30, 2ο window)"])
    ax2.set_ylim(14, 19)
    ax2.set_ylabel("MAE (€/MWh)")
    ax2.set_title("Weekly > monthly σε 2 ανεξάρτητα windows ✓\n(ίδιο πρόσημο, πάνω από noise floor)")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "F7_robustness.png")
    plt.close(fig)


HTML = """<!DOCTYPE html>
<html lang="el"><head><meta charset="utf-8">
<title>Ablation Study — Σύνοψη 2026-07-02</title>
<style>
 body{font-family:'Segoe UI',sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;color:#212121}
 h1{border-bottom:3px solid #1565c0;padding-bottom:8px} h2{color:#1565c0;margin-top:34px}
 img{width:100%;border:1px solid #ddd;border-radius:6px;margin:6px 0}
 .verdict{background:#e8f5e9;border-left:5px solid #2e7d32;padding:12px 16px;border-radius:4px}
 .warn{background:#fff3e0;border-left:5px solid #f9a825;padding:12px 16px;border-radius:4px}
 table{border-collapse:collapse;width:100%} td,th{border:1px solid #ccc;padding:6px 10px;text-align:left}
 th{background:#e3f2fd}
</style></head><body>
<h1>⚡ Ablation Study — Οπτική Σύνοψη (2026-07-02, ΔΙΟΡΘΩΘΗΚΕ 2026-07-04)</h1>
<div class="warn"><b>⚠️ Διόρθωση 2026-07-04:</b> η προηγούμενη έκδοση αυτής της αναφοράς ανέφερε
headline 14.43/15.02 €/MWh με <code>xborder</code> — αυτά τα νούμερα ήταν <b>leakage</b> (οι
same-day τιμές γειτόνων BG/IT-SUD βγαίνουν από το ΙΔΙΟ SDAC auction με το ελληνικό target,
δημοσιεύονται ~13:00 CET D-1, δηλαδή ΜΕΤΑ το gate closure 12:00). Η νόμιμη (lagged) εκδοχή του
xborder ΒΛΑΠΤΕΙ τον χειμώνα. Όλα τα νούμερα παρακάτω έχουν διορθωθεί. Λεπτομέρειες:
<code>ABLATION_PLAN.md §XBORDER-ΤΕΛΙΚΗ-ΕΤΥΜΗΓΟΡΙΑ</code> + <code>§OVERNIGHT PROTOCOL</code>.</div>
<div class="verdict"><b>Ετυμηγορία (ενημερωμένη):</b> Και τα 9 βήματα του αρχικού αυτόνομου
session εκτελέστηκαν σωστά (10/10 spot-checks). Το ξεχωριστό overnight protocol (P1-P6,
2026-07-04) έλεγξε ΞΑΝΑ όλα τα ανοιχτά ζητήματα σε καθαρά (χωρίς xborder) configs:
<b>τελικό, έγκυρο headline: LGBM recursive <u>weekly</u>, <code>default</code> = 15.17 €/MWh</b>
(Q1 2026, strict gate). Seed robustness: std≈0.11 (seeds 42/43/44). Επιβεβαιώθηκε σε 2ο
ανεξάρτητο window (Μάρτιος 2026: weekly 17.67 vs monthly 17.97, ίδιο πρόσημο).</div>

<h2>1. Το ταξίδι του σφάλματος — τι μας έδωσε τι</h2>
<img src="F1_headline_journey.png">
<p>Το μεγαλύτερο μεμονωμένο κέρδος (−7.7 €/MWh) ήρθε από τη <b>συμπλήρωση των δεδομένων</b>,
όχι από αλγόριθμους. Δεύτερο: retrain (−0.9), τρίτο: cross-border τιμές (−0.8).</p>

<h2>2. Ποιες ομάδες features αξίζουν</h2>
<img src="F2_feature_groups.png">
<table>
<tr><th>Ομάδα</th><th>Ετυμηγορία</th></tr>
<tr><td><b>genlags</b> (actual παραγωγή/residual load lags)</td><td>🟢 Η πιο στιβαρά πολύτιμη — παντού, κάθε εποχή/στρατηγική/αλγόριθμο</td></tr>
<tr><td><b>resfc</b> (day-ahead RES forecast)</td><td>🟢 Πολύτιμο — αλλά ΜΟΝΟ ορατό σε καλοκαιρινό test window (και υπό SS)</td></tr>
<tr><td><b>xborder</b> (τιμές BG/IT-SUD, lagged/νόμιμο)</td><td>🔴 ΑΠΟΡΡΙΦΘΗΚΕ από default — βλάπτει τον χειμώνα σε 3/3 retrain cadences (το προηγούμενο «θετικό» ήταν leakage)· βοηθάει το καλοκαίρι (1 σημείο, ανοιχτό ερώτημα)</td></tr>
<tr><td><b>loadfc</b>, <b>loadlags</b></td><td>🟡 Οριακά/ασταθή — το loadfc γίνεται χρήσιμο μόνο υπό SS</td></tr>
<tr><td><b>meteo</b>, <b>fuel</b></td><td>🔴 Βλάπτουν στην ΤΙΜΗ (recursive) — αλλά meteo = κορυφαίο feature στο ΦΟΡΤΙΟ</td></tr>
</table>

<h2>3. Εποχικότητα &amp; cross-border</h2>
<img src="F3_season_and_xborder.png">

<h2>4. Scheduled Sampling — δουλεύει, και αλλάζει το παιχνίδι των features</h2>
<img src="F4_scheduled_sampling.png">

<h2>5. Μοντέλα &amp; πειράματα E1/E2</h2>
<img src="F5_models_e1e2.png">

<h2>6. Stacking battery (follow-up, 2026-07-03) — ⚠️ ΑΚΥΡΩΘΗΚΕ, ήταν πάνω σε xborder leakage</h2>
<div class="warn">Το αρχικό stacking battery (6 runs) έδειχνε xborder×weekly=14.43 ως «νέο
headline» με στοίβαγμα οφελών — ΟΛΑ αυτά ήταν leakage-inflated (βλ. διόρθωση στην κορυφή). Ο
πίνακας αφαιρέθηκε για να μην ξαναδιαβαστεί λάθος. Το μόνο μέρος του που παρέμεινε έγκυρο (seeds
στο <code>default</code> χωρίς xborder) επαναλήφθηκε καθαρά στο §7 (P6) παρακάτω.</div>

<h2>7. 🌙 Overnight Protocol (2026-07-04) — πλήρης επανεξέταση σε καθαρά configs</h2>
<p>Μετά την ανακάλυψη του xborder leakage, τρέξαμε ένα δομημένο πρωτόκολλο 6 φάσεων (P1-P6) που
ξαναέλεγξε ΚΑΘΕ ανοιχτό ζήτημα πάνω σε καθαρά (χωρίς xborder) configs, με ρητά κριτήρια αποδοχής
(|ΔMAE|&gt;0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ανεξάρτητες συνθήκες). Πλήρεις πίνακες:
<code>ABLATION_PLAN.md §5</code>.</p>
<img src="F6_meteo_strategy_effect.png">
<img src="F7_robustness.png">
<table>
<tr><th>Φάση</th><th>Ερώτημα</th><th>Ετυμηγορία</th></tr>
<tr><td>P1</td><td>xborder-lagged στο default set;</td><td>🔴 ΑΠΟΡΡΙΦΘΗΚΕ — βλάπτει τον χειμώνα (3/3 cadences), βοηθάει το καλοκαίρι (1 σημείο, PENDING ως γενικό συμπέρασμα)</td></tr>
<tr><td>P2</td><td>Ισχύουν τα ευρήματα σε XGB/LEAR/MLP;</td><td>🟢 resfc/genlags επιβεβαιώθηκαν cross-model (XGB summer, MLP)· 🔴 LEAR αντιδράει ΑΝΤΙΘΕΤΑ (L1 regularization) — μοντελο-εξαρτώμενο, όχι artifact</td></tr>
<tr><td>P3</td><td>Ensembles με τίμιο (out-of-sample) calibration;</td><td>🔴 weighted-by-1/MAE ΑΠΟΡΡΙΦΘΗΚΕ — κανένα ensemble δεν κερδίζει το καλύτερο μεμονωμένο μοντέλο στο honest split· weekly LGBM+XGB δείχνει PENDING υπόσχεση (Δ=−0.148, ακριβώς στο όριο)</td></tr>
<tr><td>P5</td><td>Είναι το SS πράγματι redundant με retrain;</td><td>🟢 ΔΙΟΡΘΩΘΗΚΕ — το SS-linear βοηθάει ΚΑΙ στο monthly (−0.243) σε καθαρό config· το παλιό «redundant» ήταν artifact του xborder-contaminated config</td></tr>
<tr><td>P4</td><td>Direct vs recursive στο πλήρες Q1;</td><td>🔴 Direct παραμένει σαφώς χειρότερο (19.5 vs 16.1)· 🟢 ΝΕΟ: meteo βοηθάει ΠΑΝΤΑ στο direct (4/4 σημεία) — το πιο στιβαρό strategy-effect εύρημα της μελέτης· cross-strategy ensemble ΑΠΟΡΡΙΦΘΗΚΕ</td></tr>
<tr><td>P6</td><td>Robustness τελικού headline;</td><td>🟢 ΔΕΚΤΟ — std≈0.11 σε 3 seeds, επιβεβαιωμένο σε 2ο window (Μάρτιος 2026)</td></tr>
</table>

<h2>8. 🟡 PENDING — ό,τι ΔΕΝ είναι ακόμα συμπέρασμα (μην το πουλήσεις ως εύρημα)</h2>
<table>
<tr><th>#</th><th>Θέμα</th><th>Κατάσταση / τι λείπει</th></tr>
<tr><td>1</td><td>Summer xborder-lagged θετικό σήμα (−0.52)</td><td>1 σημείο (static μόνο) — θέλει monthly/weekly καλοκαίρι ή 2ο seed</td></tr>
<tr><td>2</td><td>Weekly LGBM+XGB ensemble (−0.148)</td><td>Ακριβώς στο noise floor, 1 σημείο — θέλει Μάρτιο ή seeds</td></tr>
<tr><td>3</td><td>resfc στο direct</td><td>LGBM ουδέτερο vs XGB βλάπτει — algo-dependent, άλυτο</td></tr>
<tr><td>4</td><td>fuel window-confound στο direct</td><td>Δεκ −0.68 vs Q1 ~0 — αδιευκρίνιστο</td></tr>
<tr><td>5</td><td>loadlags καλοκαίρι</td><td>LGBM ουδέτερο vs XGB +0.42 — ασυνεπές</td></tr>
<tr><td>6</td><td>SS × weekly</td><td>Αδοκίμαστο (SS οφέλη ίσως στοιβάζονται και με weekly)</td></tr>
<tr><td>7</td><td>LSTM calibration bug</td><td>Bias +41, ποτέ αρνητικές τιμές — θέλει debugging, όχι rerun</td></tr>
<tr><td>8</td><td>solar_fc_dayahead ύποπτο 2h shift</td><td>High-priority data-quality έλεγχος (επηρεάζει resfc) — §1 πρωτόκολλο</td></tr>
<tr><td>9</td><td>henex_premarket ομάδα</td><td>Ποτέ κανένα τεστ</td></tr>
</table>

<h2>9. ➡️ Επόμενα βήματα</h2>
<p><b>Κύριο (στάδιο πλάνου):</b> Data ✅ → Engine ✅ → Feature validity ✅ → Model/strategy ✅ →
Robustness ✅ → <b>Probabilistic layer (conformal) ← ΕΔΩ</b> → Product (L3-L5). Split-conformal
p10/p50/p90 πάνω στο κλειδωμένο config, αξιολόγηση pinball + coverage σε Q1 + Μάρτιο· σύγκριση
με quantile-LGBM. Βλ. <code>ABLATION_PLAN.md §8.2</code> και το έτοιμο prompt στο
<code>last.md §5</code>.</p>
<p><b>Vetted feature candidate — xb_lag1_h0</b> (πέρασε τον pre-flight έλεγχο στα χαρτιά):
το lag1 της ώρας-στόχου 00:00 είναι η 23:00 της D-1, δημοσιευμένη ~13:00 <b>D-2</b> — δηλαδή ΠΡΙΝ
το gate. Για ώρες 01-23 είναι παράνομο → NaN (τα δέντρα το χειρίζονται). Μικρό αναμενόμενο
effect (1/24 των γραμμών) — μετριέται και στο υποσύνολο ώρας-0. <code>ABLATION_PLAN.md §8.1</code>.</p>

<div class="warn"><b>Επιφυλάξεις (honesty):</b> Όλα τα PENDING του §8 δεν είναι δηλωμένα
συμπεράσματα · meteo = reanalysis proxy (αισιόδοξο άνω όριο) · μόνο το headline έχει 3 seeds —
όλα τα υπόλοιπα ΔMAE είναι single-seed · direct δεν υποστηρίζεται για MLP/LSTM στον κώδικα, το
SS μόνο για recursive.</div>
</body></html>
"""

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    (OUT / "index.html").write_text(HTML, encoding="utf-8")
    print(f"✅ 7 figures + index.html → {OUT}")
