# -*- coding: utf-8 -*-
"""Νέα DM heatmaps (v3) στο κλειδωμένο light style (SKILL §7).
Διαφορές από την προηγούμενη έκδοση:
  - Ελληνικός τίτλος + Ελληνικό legend (κάτω), όχι αγγλικό.
  - Locked ορολογία WF- (συμφωνεί με τον Πίνακα tab:dm_models του Κεφ. 7).
  - Ελληνική υποδιαστολή στα p-values.
  - Ετικέτες χρωματισμένες ανά στρατηγική (όπως ζητά η λεζάντα).
Διαβάζει dm_test/*.csv και γράφει thesis_output/ch5_dm_test/dm_{task}_heatmap_v3.png.
Re-runnable.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DM = ROOT / "dm_test"
OUT_DM = ROOT / "thesis_output" / "ch5_dm_test"
OUT_DM.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#444", "axes.linewidth": 0.9,
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

# CSV code -> (εμφανιζόμενο όνομα locked, στρατηγική για χρώμα ετικέτας)
REN = {
    "CL-LGBM": ("TF-LGBM", "TF"), "CL-XGB": ("TF-XGB", "TF"), "CL-RF": ("TF-RF", "TF"),
    "CL-MLP-Opt": ("TF-MLP", "TF"), "CL-Ens-Best3": ("TF-Ens", "TF"),
    "OL-LGBM-Daily": ("Rec-LGBM", "Rec"), "OL-XGB-SS-Dense": ("Rec-XGB-SS", "Rec"),
    "OL-RF-SS": ("Rec-RF", "Rec"), "OL-Ens-Top3": ("Rec-Ens", "Rec"),
    "OL-LGBM-SS": ("Rec-LGBM", "Rec"), "OL-XGB-Daily": ("Rec-XGB", "Rec"),
    "MIMO-XGB-Dense": ("MIMO-XGB", "MIMO"), "MIMO-Ens-Best3": ("MIMO-Ens", "MIMO"),
    "Direct-LGBM-Opt": ("Direct-LGBM", "MIMO"), "Direct-LGBM-Dense": ("Direct-LGBM", "MIMO"),
    "MR-TF-LGBM": ("WF-TF-LGBM", "WF-TF"), "MR-TF-XGB": ("WF-TF-XGB", "WF-TF"),
    "MR-TF-RF": ("WF-TF-RF", "WF-TF"), "MR-Ens-TF-Best3": ("WF-TF-Ens", "WF-TF"),
    "MR-Rec-XGB-SS": ("WF-Rec-XGB", "WF-Rec"),
    "WF-MIMO-Ens3": ("WF-MIMO-Ens", "WF-MIMO"), "WF-Direct-LGBM": ("WF-MIMO-LGBM", "WF-MIMO"),
}
# Χρώματα ετικετών ανά στρατηγική (όπως στη λεζάντα του Κεφ. 7)
STRAT_COL = {
    "WF-TF": "#1b5e20",   # σκ. πράσινο
    "WF-Rec": "#43a047",  # αν. πράσινο
    "WF-MIMO": "#5e35b1", # σκ. μωβ
    "TF": "#1e88e5",      # μπλε
    "Rec": "#f97316",     # πορτοκαλί
    "MIMO": "#8e24aa",    # μωβ
}
NM = {"price": "Τιμή DAM", "load": "Ηλεκτρικό Φορτίο"}


def gr(p):
    """p-value σε ελληνική μορφή με υποδιαστολή."""
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def dm_heatmap(task):
    pv = pd.read_csv(DM / f"dm_{task}_pval.csv", index_col=0)
    dm = pd.read_csv(DM / f"dm_{task}_matrix.csv", index_col=0)
    codes = list(pv.columns)
    n = len(codes)
    lab = [REN.get(c, (c, "MIMO"))[0] for c in codes]
    col = [STRAT_COL[REN.get(c, (c, "MIMO"))[1]] for c in codes]

    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = pv.iloc[i, j]; d = dm.iloc[i, j]
            if pd.isna(p):
                continue
            M[i, j] = (1 if d > 0 else -1) if p < 0.05 else 0

    cmap = LinearSegmentedColormap.from_list("dm", ["#c0392b", "#eef0f2", "#1e8449"])
    fig, ax = plt.subplots(figsize=(13.2, 11.4))
    ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lab, rotation=90, fontsize=9)
    ax.set_yticklabels(lab, fontsize=9)
    for tl, c in zip(ax.get_xticklabels(), col):
        tl.set_color(c); tl.set_fontweight("bold")
    for tl, c in zip(ax.get_yticklabels(), col):
        tl.set_color(c); tl.set_fontweight("bold")

    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor("#333"); s.set_linewidth(1.2)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#888",
                        fontsize=9, fontweight="bold"); continue
            p = pv.iloc[i, j]
            if pd.isna(p):
                continue
            ax.text(j, i, gr(p), ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white" if abs(M[i, j]) == 1 else "#1a1a1a")

    ax.set_title(f"Έλεγχος Diebold–Mariano (HLN) — {NM[task]} (Q1 2026)",
                 fontsize=14, pad=10)
    h = [Patch(fc="#1e8449", ec="#333", label="Η γραμμή υπερτερεί της στήλης (p<0,05)"),
         Patch(fc="#c0392b", ec="#333", label="Η γραμμή υστερεί της στήλης (p<0,05)"),
         Patch(fc="#eef0f2", ec="#333", label="Στατιστική ισοπαλία (p$\\geq$0,05)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=10, handlelength=1.4)
    plt.tight_layout()
    out = OUT_DM / f"dm_{task}_heatmap_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[ok] {task}: {out.name}  ({n}×{n})")


if __name__ == "__main__":
    for t in ["price", "load"]:
        dm_heatmap(t)
    print("DONE →", OUT_DM)
