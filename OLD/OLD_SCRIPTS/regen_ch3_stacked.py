"""Regenerate Ch3 figures with user-requested layout:
 - 3.3 fig_monthly_avg_by_year : STACKED (price top / load bottom), larger
 - 3.5 fig_daily_weekly_profile_load : LARGER (2-row: season+dow top, heatmap bottom)
Writes straight to thesis_output/ root. DejaVu fonts (Linux-safe)."""
import warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
OUT  = BASE / "thesis_output"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":13,"axes.titlesize":15,
    "axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11,
    "legend.fontsize":11,"figure.dpi":200,"savefig.dpi":200,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
    "grid.color":"#e0e0e0","grid.linewidth":0.6,"axes.axisbelow":True})
C_BLUE,C_ORANGE,C_GREEN,C_RED="#2176ae","#e8871e","#3a9e4a","#c0392b"
def _clean(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

TRAIN_START,TRAIN_END="2017-01-01","2025-11-30 23:00"
price=pd.read_parquet(BASE/"data/processed/hourly.parquet").loc[TRAIN_START:TRAIN_END]["y"]
load =pd.read_parquet(BASE/"data/processed/hourly_load.parquet").loc[TRAIN_START:TRAIN_END]["y"]
load_aligned=load.reindex(pd.date_range(TRAIN_START,TRAIN_END,freq="h"),fill_value=np.nan)
month_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
dow_names=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ── 3.3 STACKED ──────────────────────────────────────────────
years=range(2017,2026); cmap=plt.cm.get_cmap("tab10",len(list(years)))
fig,(ax_p,ax_l)=plt.subplots(2,1,figsize=(13,12)); fig.subplots_adjust(hspace=0.32)
for idx,yr in enumerate(years):
    yp=price[price.index.year==yr]; mp=yp.groupby(yp.index.month).mean()
    ax_p.plot(mp.index-1,mp.values,marker="o",markersize=5,linewidth=2.0,color=cmap(idx),label=str(yr))
    yl=load_aligned[load_aligned.index.year==yr]; ml=yl.groupby(yl.index.month).mean()
    ax_l.plot(ml.index-1,ml.values,marker="o",markersize=5,linewidth=2.0,color=cmap(idx),label=str(yr))
for ax,title,unit in [(ax_p,"Average Monthly Price by Year","€/MWh"),(ax_l,"Average Monthly Load by Year","MW")]:
    ax.set_xticks(range(12)); ax.set_xticklabels(month_names)
    ax.set_title(title,fontweight="bold"); ax.set_ylabel(unit)
    ax.legend(fontsize=10,ncol=3,loc="upper left",framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}")); _clean(ax)
fig.suptitle("Seasonal Patterns: Day-Ahead Price and System Load\n(Greek Electricity Market, 2017–2025)",
             fontsize=15,fontweight="bold",y=0.995)
fig.savefig(OUT/"fig_monthly_avg_by_year.png",bbox_inches="tight"); plt.close(fig)
print("Saved fig_monthly_avg_by_year.png (stacked)")

# ── 3.5 LARGER (2-row) ───────────────────────────────────────
fig=plt.figure(figsize=(15,11))
gs=gridspec.GridSpec(2,2,height_ratios=[1,1],hspace=0.30,wspace=0.24)
ax0=fig.add_subplot(gs[0,0]); ax1=fig.add_subplot(gs[0,1]); ax2=fig.add_subplot(gs[1,:])
seasons={"Winter (Dec–Feb)":[12,1,2],"Spring (Mar–May)":[3,4,5],"Summer (Jun–Aug)":[6,7,8],"Autumn (Sep–Nov)":[9,10,11]}
for (sname,ms),sc in zip(seasons.items(),[C_BLUE,C_GREEN,C_RED,C_ORANGE]):
    sub=load_aligned[load_aligned.index.month.isin(ms)]; pr=sub.groupby(sub.index.hour).mean()
    ax0.plot(pr.index,pr.values,color=sc,linewidth=2.4,label=sname)
ax0.set_xticks(range(0,24,3)); ax0.set_xlabel("Hour of day"); ax0.set_ylabel("Mean load (MW)")
ax0.set_title("Intraday Load Profile by Season",fontweight="bold"); ax0.legend(framealpha=0.9)
ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}")); _clean(ax0)
for d in range(7):
    sub=load_aligned[load_aligned.index.dayofweek==d]; pr=sub.groupby(sub.index.hour).mean()
    ax1.plot(pr.index,pr.values,color=(C_ORANGE if d>=5 else C_BLUE),linewidth=(2.4 if d>=5 else 1.5),alpha=0.85,label=dow_names[d])
ax1.set_xticks(range(0,24,3)); ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Mean load (MW)")
ax1.set_title("Intraday Load Profile by Day-of-Week",fontweight="bold"); ax1.legend(ncol=2,framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}")); _clean(ax1)
piv=load_aligned.to_frame(name="load"); piv["hour"]=piv.index.hour; piv["dow"]=piv.index.dayofweek
heat=piv.groupby(["dow","hour"])["load"].mean().unstack()
im=ax2.imshow(heat.values,aspect="auto",cmap="YlOrRd",interpolation="nearest")
ax2.set_xticks(range(0,24,3)); ax2.set_xticklabels(range(0,24,3)); ax2.set_yticks(range(7)); ax2.set_yticklabels(dow_names)
ax2.set_xlabel("Hour of day"); ax2.set_title("Mean Load Heatmap (Day-of-Week × Hour)",fontweight="bold")
fig.colorbar(im,ax=ax2,label="MW",shrink=0.9,format=mticker.FuncFormatter(lambda x,_:f"{x:,.0f}")); ax2.grid(False)
fig.suptitle("System Load: Seasonal, Intraday and Weekly Patterns",fontsize=15,fontweight="bold",y=0.97)
fig.savefig(OUT/"fig_daily_weekly_profile_load.png",bbox_inches="tight"); plt.close(fig)
print("Saved fig_daily_weekly_profile_load.png (2-row larger)")
