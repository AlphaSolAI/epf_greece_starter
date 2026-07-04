"""Fig 2.5 — juxtaposition: literature taxonomy (left) vs strategies evaluated here (right)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
from pathlib import Path
OUT=Path(__file__).resolve().parent/"thesis_output"
plt.rcParams.update({"font.family":"DejaVu Sans"})
C={"tf":"#2176ae","rec":"#c0392b","direct":"#3a9e4a","mimo":"#7b2d8b"}
GRY="#34495e"

fig=plt.figure(figsize=(15,7.6))
gs=gridspec.GridSpec(1,2,width_ratios=[1,1.18],wspace=0.06)
axL=fig.add_subplot(gs[0,0]); axR=fig.add_subplot(gs[0,1])
for ax in (axL,axR): ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

# ---- LEFT: literature taxonomy ----
def box(ax,x,y,w,h,txt,fc="#ffffff",ec=GRY,fs=15,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.02",
                 lw=2.0,edgecolor=ec,facecolor=fc))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
            fontweight=("bold" if bold else "normal"),color="#1a1a1a")
W,H=0.26,0.17
top=[("Recursive",0.04),("Direct",0.37),("MIMO",0.70)]
for t,x in top: box(axL,x,0.66,W,H,t)
box(axL,0.205,0.12,W,H,"DirRec"); box(axL,0.535,0.12,W,H,"DIRMO")
def arr(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=18,lw=1.8,color=GRY))
arr(axL,0.17,0.66,0.30,0.29)   # Recursive->DirRec
arr(axL,0.47,0.66,0.36,0.29)   # Direct->DirRec
arr(axL,0.52,0.66,0.63,0.29)   # Direct->DIRMO
arr(axL,0.81,0.66,0.69,0.29)   # MIMO->DIRMO
axL.text(0.5,0.96,"Γενική ταξινόμηση στρατηγικών\n(βιβλιογραφία)",ha="center",va="center",
         fontsize=15,fontweight="bold",color="#444")

# ---- RIGHT: strategies evaluated in this work ----
cards=[("tf",0.02,0.50,"Teacher-Forcing (TF)",["πραγματικές υστερήσεις","(open-loop oracle)"]),
       ("rec",0.51,0.50,"Recursive (Rec)",["ανατροφοδότηση προβλέψεων","(closed-loop)"]),
       ("direct",0.02,0.04,"Direct",["$H$ ανεξάρτητα μοντέλα","ένα ανά βήμα $h$"]),
       ("mimo",0.51,0.04,"MIMO",["1 μοντέλο → διάνυσμα","όλων των $H$ βημάτων"])]
cw,ch=0.47,0.40
for k,x,y,title,lines in cards:
    col=C[k]
    axR.add_patch(FancyBboxPatch((x,y),cw,ch,boxstyle="round,pad=0.01,rounding_size=0.03",
                  lw=2.6,edgecolor=col,facecolor=col+"18"))
    axR.text(x+cw/2,y+ch-0.075,title,ha="center",va="center",fontsize=15.5,fontweight="bold",color=col)
    for i,ln in enumerate(lines):
        axR.text(x+cw/2,y+ch-0.175-0.085*i,ln,ha="center",va="center",fontsize=12.5,color="#222")
axR.text(0.5,0.96,"Στρατηγικές που αξιολογούνται στην εργασία",ha="center",va="center",
         fontsize=15,fontweight="bold",color="#444")
fig.savefig(OUT/"fig_multistep_strategies.png",dpi=200,bbox_inches="tight",facecolor="white")
print("Saved 2.5 juxtaposition")
