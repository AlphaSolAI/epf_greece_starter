# -*- coding: utf-8 -*-
"""Ενιαίο restyle DM + Feature Importance στο κλειδωμένο light style (SKILL §7).
Διαβάζει dm_test/*.csv + feature_importance/*.csv και παράγει τελικά PNG
σε thesis_output/ch5_dm_test/ και thesis_output/ch4_feature_importance/.
Re-runnable. conda env: epf (ή οποιοδήποτε python με pandas/matplotlib)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DM   = ROOT / "dm_test"
FI   = ROOT / "feature_importance"
OUT_DM = ROOT / "thesis_output" / "ch5_dm_test"
OUT_FI = ROOT / "thesis_output" / "ch4_feature_importance"
OUT_DM.mkdir(parents=True, exist_ok=True); OUT_FI.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "axes.edgecolor":"#444","axes.linewidth":0.9,
    "font.size":11,"axes.titlesize":13,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9,
})

# locked terminology
REN = {"CL-LGBM":"TF-LGBM","CL-XGB":"TF-XGB","CL-RF":"TF-RF","CL-MLP-Opt":"TF-MLP","CL-Ens-Best3":"TF-Ens",
 "OL-LGBM-Daily":"Rec-LGBM","OL-XGB-SS-Dense":"Rec-XGB-SS","OL-RF-SS":"Rec-RF","OL-Ens-Top3":"Rec-Ens",
 "OL-LGBM-SS":"Rec-LGBM","OL-XGB-Daily":"Rec-XGB",
 "MIMO-XGB-Dense":"MIMO-XGB","MIMO-Ens-Best3":"MIMO-Ens","Direct-LGBM-Opt":"Direct-LGBM",
 "Direct-LGBM-Dense":"Direct-LGBM","WF-MIMO-Ens3":"MIMO-Ens-MR","WF-Direct-LGBM":"Direct-LGBM-MR",
 "MR-TF-LGBM":"TF-LGBM-MR","MR-TF-XGB":"TF-XGB-MR","MR-TF-RF":"TF-RF-MR",
 "MR-Rec-XGB-SS":"Rec-XGB-SS-MR","MR-Ens-TF-Best3":"Ens-TF-Best3-MR"}
def famcol(n):
    n=n.upper()
    return ("#eab308" if "ENS" in n else "#38bdf8" if "LGBM" in n else "#f97316" if "XGB" in n
            else "#22c55e" if "RF" in n else "#a855f7" if "MLP" in n else "#94a3b8")
NM={"price":"Τιμή DAM","load":"Ηλεκτρικό Φορτίο"}
UNIT={"price":"€/MWh","load":"MW"}

def dm_heatmap(task):
    pv=pd.read_csv(DM/f"dm_{task}_pval.csv",index_col=0); dm=pd.read_csv(DM/f"dm_{task}_matrix.csv",index_col=0)
    m=list(pv.columns); n=len(m); lab=[REN.get(x,x) for x in m]
    M=np.full((n,n),np.nan)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            p=pv.iloc[i,j]; d=dm.iloc[i,j]
            if pd.isna(p): continue
            M[i,j]=(1 if d>0 else -1) if p<0.05 else 0
    cmap=LinearSegmentedColormap.from_list("dm",["#c0392b","#eef0f2","#1e8449"])
    fig,ax=plt.subplots(figsize=(13.5,11.6)); ax.imshow(M,cmap=cmap,vmin=-1,vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lab,rotation=90,fontsize=9); ax.set_yticklabels(lab,fontsize=9)
    for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor("#333"); s.set_linewidth(1.2)
    ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=1.3); ax.tick_params(which="minor",length=0)
    for i in range(n):
        for j in range(n):
            if i==j: ax.text(j,i,"—",ha="center",va="center",color="#999",fontsize=9,fontweight="bold"); continue
            p=pv.iloc[i,j]
            if pd.isna(p): continue
            ax.text(j,i,(f"{p:.3f}" if p>=0.001 else "<.001"),ha="center",va="center",
                    fontsize=8.6,fontweight="bold",color="white" if abs(M[i,j])==1 else "#1a1a1a")
    ax.set_title(f"Diebold–Mariano — {NM[task]} (Q1 2026)",fontsize=14,pad=10)
    h=[Patch(fc="#1e8449",ec="#333",label="Row beats column (p<0.05)"),
       Patch(fc="#c0392b",ec="#333",label="Row loses to column (p<0.05)"),
       Patch(fc="#eef0f2",ec="#333",label="Statistical tie (p>=0.05)")]
    ax.legend(handles=h,loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=3,frameon=False,fontsize=10,handlelength=1.4)
    plt.tight_layout(); plt.savefig(OUT_DM/f"dm_{task}_heatmap.png",dpi=145,bbox_inches="tight"); plt.close()

def dm_wins(task):
    r=pd.read_csv(DM/f"dm_{task}_results.csv"); r["disp"]=r["Model"].map(lambda x:REN.get(x,x))
    r=r.sort_values("Wins (p<5%)"); mc=[c for c in r.columns if c.startswith("MAE")][0]
    fig,ax=plt.subplots(figsize=(10.5,8.4))
    ax.grid(axis="x",color="#d9d9d9",linestyle=(0,(1,2)),linewidth=0.7,zorder=0)
    b=ax.barh(r["disp"],r["Wins (p<5%)"],color=[famcol(x) for x in r["disp"]],edgecolor="#333",linewidth=0.7,zorder=3)
    for bar,w,mae in zip(b,r["Wins (p<5%)"],r[mc]):
        ax.text(bar.get_width()+0.12,bar.get_y()+bar.get_height()/2,
                f"{int(w)}   (MAE {mae:.2f} {UNIT[task]})",va="center",fontsize=8.4,color="#222")
    ax.set_xlabel("Number of significant wins (p<0.05)")
    ax.set_title(f"Στατιστικές νίκες Diebold–Mariano — {NM[task]} (Q1 2026)",fontsize=13,pad=8)
    ax.set_xlim(0,r["Wins (p<5%)"].max()+6)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT_DM/f"dm_{task}_wins.png",dpi=145,bbox_inches="tight"); plt.close()

FI_COLS={
 "price":{"CL — LightGBM":"TF\n(LGBM)","OL — LGBM Daily-Opt":"Rec\n(LGBM)",
          "OL — XGB Dense+SS":"Rec\n(XGB-SS)","Direct — LGBM Dense":"Direct\n(LGBM)","MIMO — XGB Dense":"MIMO\n(XGB)"},
 "load":{"CL — LightGBM":"TF\n(LGBM)","OL — LGBM-SS Daily":"Rec\n(LGBM)",
         "Direct — LGBM Dense":"Direct\n(LGBM)","MIMO — XGB Dense":"MIMO\n(XGB)"},
}
import re
def translate(feat, task):
    f=feat; tgt="Price" if task=="price" else "Load"
    for pat,fmt in [(r"y_lag(\d+)$",tgt+" lag-%s"),(r"y_roll(\d+)$",tgt+" roll-%s"),
                    (r"load_lag(\d+)$","Load lag-%s"),(r"load_roll(\d+)$","Load roll-%s"),
                    (r"gas_lag(\d+)$","Gas lag-%s"),(r"co2_lag(\d+)$","CO2 lag-%s"),
                    (r"residual_load_lag(\d+)$","Resid. load lag-%s"),
                    (r"gen_solar_lag(\d+)$","Solar lag-%s"),(r"gen_wind_lag(\d+)$","Wind lag-%s"),
                    (r"load_pred_h(\d+)$","Load fcst h%s")]:
        m=re.match(pat,f)
        if m: return fmt % m.group(1)
    d={"gas_price":"Gas (spot)","co2_price":"CO2 (spot)","gen_fc_dayahead":"Gen fcst D-ahead",
       "solar_fc_dayahead":"Solar fcst D-ahead","wind_onshore_fc_dayahead":"Wind fcst D-ahead",
       "load_fc":"Load D-ahead","hour":"Hour","hour_sin":"Hour (sin)","hour_cos":"Hour (cos)",
       "dow":"Day-of-week","dow_sin":"DoW (sin)","dow_cos":"DoW (cos)","is_holiday":"Holiday"}
    return d.get(f, f)
def fi_heatmap(task):
    df=pd.read_csv(FI/f"feature_importance_{task}.csv"); cmap_cols=FI_COLS[task]
    cols=[c for c in cmap_cols if c in df.columns]
    df["label"]=df["feature"].map(lambda x: translate(x,task))
    g=df.groupby("label",as_index=False)[cols].sum()      # ένωση διπλών (π.χ. y_lag1+load_lag1)
    g["score"]=g[cols].mean(axis=1)
    top=g.sort_values("score",ascending=False).head(12)
    labels=list(top["label"])
    M=top[cols].values*100
    fig,ax=plt.subplots(figsize=(max(8.4,1.7*len(cols)+3),7.8))
    NORM=PowerNorm(gamma=0.45,vmin=0,vmax=90); im=ax.imshow(M,cmap="viridis",norm=NORM,aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([cmap_cols[c] for c in cols],fontsize=10.5)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=10); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor("#222"); s.set_linewidth(1.0)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True); ax.set_yticks(np.arange(-.5,len(labels),1),minor=True)
    ax.grid(which="minor",color="white",linewidth=0.8); ax.tick_params(which="minor",length=0)
    cm=plt.get_cmap("viridis")
    for i in range(len(labels)):
        for j in range(len(cols)):
            v=M[i,j]
            if v<0.5: continue
            dark=float(np.array(cm(NORM(v))[:3]).mean())<0.5
            ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=8.5,color="white" if dark else "#1a1a1a")
    cb=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.025); cb.outline.set_linewidth(0.8)
    cb.set_label("Importance (% Gain)",fontsize=10)
    ax.set_title(f"Σημαντικότητα χαρακτηριστικών ανά στρατηγική — {NM[task]}",fontsize=13,pad=9)
    plt.tight_layout(); plt.savefig(OUT_FI/f"fi_{task}_heatmap.png",dpi=150,bbox_inches="tight"); plt.close()

if __name__=="__main__":
    for t in ["price","load"]:
        dm_heatmap(t); dm_wins(t); fi_heatmap(t)
        print(f"[ok] {t}: dm_heatmap, dm_wins, fi_heatmap")
    print("DONE →", OUT_DM, OUT_FI)
