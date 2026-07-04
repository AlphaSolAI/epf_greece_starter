"""Fig 3.1 — data pipeline, redrawn: clean, space-efficient, tall vertical flow."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
OUT=Path(__file__).resolve().parent/"thesis_output"
plt.rcParams.update({"font.family":"DejaVu Sans"})

GRN="#3a9e4a"; BLU="#2176ae"; DRK="#2c3e50"
fig,ax=plt.subplots(figsize=(8.6,13)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

def box(x,y,w,h,title,sub,ec,fc,tc="#1a1a1a",ts=12,ss=9.5,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.012",
                 lw=1.8,edgecolor=ec,facecolor=fc))
    ax.text(x+w/2,y+h*0.62,title,ha="center",va="center",fontsize=ts,
            fontweight=("bold" if bold else "normal"),color=tc)
    if sub: ax.text(x+w/2,y+h*0.26,sub,ha="center",va="center",fontsize=ss,color="#555")

# ── Sources band (8), compact 2 cols x 4 rows ──
ax.text(0.5,0.985,"Πηγές Δεδομένων (8)",ha="center",fontsize=13,fontweight="bold",color=GRN)
sources=[("ENTSO-E — Τιμές DAM","€/MWh, ωριαία"),
         ("ENTSO-E — Πραγματικό Φορτίο","MW, ωριαία"),
         ("ENTSO-E — Παραγωγή ανά τεχνολογία","5 τύποι, 15ʹ→ωριαία"),
         ("Dutch TTF — Φυσικό αέριο","€/MWh, ημερήσια"),
         ("EU ETS — CO₂","€/t, ημερήσια"),
         ("Open-Meteo — Καιρός","6 τοπ. × 7 μετ., ωριαία"),
         ("ENTSO-E — Προβλέψεις ΑΠΕ","ηλιακή/αιολική, ωριαία"),
         ("ΑΔΜΗΕ — Πρόβλεψη φορτίου","proxy load_lag24")]
x0,w,h,gx,gy=0.04,0.45,0.075,0.02,0.018; ytop=0.93
for i,(t,s) in enumerate(sources):
    r,c=divmod(i,2); x=x0+c*(w+gx); y=ytop-r*(h+gy)
    box(x,y,w,h,t,s,GRN,"#eaf5ec",ts=9.6,ss=8)

# ── converge arrow ──
ax.add_patch(FancyArrowPatch((0.5,ytop-3*(h+gy)-0.005),(0.5,0.595),arrowstyle="-|>",
             mutation_scale=22,lw=2.2,color="#888"))

# ── 5 phases vertical chain ──
ax.text(0.5,0.575,"Αγωγός Επεξεργασίας",ha="center",fontsize=13,fontweight="bold",color=BLU)
phases=[("Φάση 1 — Κανονικοποίηση","UTC-naive δείκτης · 15ʹ→ωριαία (mean)"),
        ("Φάση 2 — Συγχώνευση","left-join στον χρόνο · εναρμόνιση ζωνών ώρας"),
        ("Φάση 3 — Συμπλήρωση κενών","ffill ημερήσιων · δείκτες απόντος"),
        ("Φάση 4 — Μηχανική χαρακτηριστικών","lags · κυλιόμενα · ημερολόγιο · dense lags"),
        ("Φάση 5 — Διόρθωση διαρροής","αφαίρεση σύγχρονων μεταβλητών · dropna")]
pw,ph=0.78,0.072; px=0.11; py=0.485
for i,(t,s) in enumerate(phases):
    y=py-i*(ph+0.028)
    box(px,y,pw,ph,t,s,BLU,"#e8f1f8",ts=12,ss=9.5)
    if i<len(phases)-1:
        ax.add_patch(FancyArrowPatch((0.5,y-0.004),(0.5,y-0.028+0.004),arrowstyle="-|>",
                     mutation_scale=16,lw=1.8,color="#9bb"))

# ── final feature table ──
yf=py-5*(ph+0.028)
ax.add_patch(FancyArrowPatch((0.5,yf+ph+0.028-0.004),(0.5,yf+ph+0.006),arrowstyle="-|>",
             mutation_scale=18,lw=2.0,color="#888"))
ax.add_patch(FancyBboxPatch((px,yf),pw,ph+0.01,boxstyle="round,pad=0.006,rounding_size=0.012",
             lw=2,edgecolor=DRK,facecolor=DRK))
ax.text(0.5,yf+(ph+0.01)*0.64,"Τελικός Πίνακας Χαρακτηριστικών",ha="center",va="center",
        fontsize=12.5,fontweight="bold",color="white")
ax.text(0.5,yf+(ph+0.01)*0.27,"Τιμή: 152 χαρακτ. / ~80K γραμμές   ·   Φορτίο: 133 χαρακτ. / ~95K γραμμές",
        ha="center",va="center",fontsize=9.5,color="#d5dbe0")
fig.savefig(OUT/"fig_data_pipeline.png",dpi=200,bbox_inches="tight",facecolor="white")
print("Saved fig_data_pipeline.png")
