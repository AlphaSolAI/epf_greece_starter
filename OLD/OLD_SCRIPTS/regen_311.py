"""3.11 train/test split — taller panels (pulled down), bigger fonts."""
import warnings,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from pathlib import Path
warnings.filterwarnings("ignore")
BASE=Path(__file__).resolve().parent; OUT=BASE/"thesis_output"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.titlesize":16,
 "axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11,
 "figure.dpi":200,"savefig.dpi":200,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#e0e0e0","grid.linewidth":0.6,"axes.axisbelow":True})
C_BLUE,C_ORANGE,C_RED="#2176ae","#e8871e","#c0392b"
def _clean(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
TS,TE="2017-01-01","2025-11-30 23:00"
dfp=pd.read_parquet(BASE/"data/processed/hourly.parquet"); dfl=pd.read_parquet(BASE/"data/processed/hourly_load.parquet")
price=dfp.loc[TS:TE]["y"]; load=dfl.loc[TS:TE]["y"]
load_aligned=load.reindex(pd.date_range(TS,TE,freq="h"),fill_value=np.nan)
# TALLER: figsize height 12 (was 8), more hspace
fig,(ax_p3,ax_l3)=plt.subplots(2,1,figsize=(14,12),sharex=True); fig.subplots_adjust(hspace=0.22)
pda=price.resample("D").mean(); lda=load_aligned.resample("D").mean()
CUT=pd.Timestamp("2025-11-30"); TSn=pd.Timestamp("2025-12-01"); TEn=pd.Timestamp("2026-02-28")
ax_p3.fill_between(pda.index,pda.values,where=(pda.index<=CUT),color=C_BLUE,alpha=0.45,label="Training (Jan 2017 – Nov 2025)")
pte=dfp["y"].loc[TSn:TEn].resample("D").mean()
ax_p3.fill_between(pte.index,pte.values,color=C_RED,alpha=0.5,label="Test period (Dec 2025 – Feb 2026)")
ax_p3.axvline(TSn,color=C_RED,lw=1.8,ls="--"); ax_p3.text(TSn+pd.Timedelta(days=4),ax_p3.get_ylim()[1]*0.88,"Test start\nDec 2025",color=C_RED,fontsize=11)
ax_p3.set_ylabel("Price (€/MWh)"); ax_p3.set_title("Day-Ahead Price — Training / Test Split",fontweight="bold")
ax_p3.legend(loc="upper left",framealpha=0.9); _clean(ax_p3)
ax_l3.fill_between(lda.index,lda.values,where=(lda.index<=CUT),color=C_ORANGE,alpha=0.45,label="Training (Jan 2017 – Nov 2025)")
lte=dfl["y"].loc[TSn:TEn].resample("D").mean()
ax_l3.fill_between(lte.index,lte.values,color=C_RED,alpha=0.5,label="Test period (Dec 2025 – Feb 2026)")
ax_l3.axvline(TSn,color=C_RED,lw=1.8,ls="--"); ax_l3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}"))
ax_l3.set_ylabel("Load (MW)"); ax_l3.set_title("System Load — Training / Test Split",fontweight="bold")
ax_l3.legend(loc="upper left",framealpha=0.9); _clean(ax_l3)
ntr=len(price)
ax_p3.text(pd.Timestamp("2018-01-01"),ax_p3.get_ylim()[1]*0.74,f"Training set: {ntr:,} hourly observations\n({ntr//24:,} days)",fontsize=11,color=C_BLUE,alpha=0.9)
fig.suptitle("Dataset Split: Training and Evaluation Periods\n(Greek Electricity Market Forecasting Study)",fontsize=15,fontweight="bold",y=0.97)
fig.savefig(OUT/"fig_train_test_split.png",bbox_inches="tight"); print("Saved 3.11 taller", "size:", end=" ")
