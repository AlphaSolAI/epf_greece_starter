"""Symmetric Ch3 profile figures (3.4 price / 3.5 load): identical 3-panel 2-row layout.
Both get: Intraday by Season + Intraday by Day-of-Week + Heatmap. Larger, matching look."""
import warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from pathlib import Path
warnings.filterwarnings("ignore")
BASE=Path(__file__).resolve().parent; OUT=BASE/"thesis_output"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.titlesize":15,
 "axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":11,
 "figure.dpi":200,"savefig.dpi":200,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#e0e0e0","grid.linewidth":0.6,"axes.axisbelow":True})
C_BLUE,C_ORANGE,C_GREEN,C_RED="#2176ae","#e8871e","#3a9e4a","#c0392b"
def _clean(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
TS,TE="2017-01-01","2025-11-30 23:00"
price=pd.read_parquet(BASE/"data/processed/hourly.parquet").loc[TS:TE]["y"]
load =pd.read_parquet(BASE/"data/processed/hourly_load.parquet").loc[TS:TE]["y"]
load=load.reindex(pd.date_range(TS,TE,freq="h"),fill_value=np.nan)
dow=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
seasons={"Winter (Dec–Feb)":[12,1,2],"Spring (Mar–May)":[3,4,5],"Summer (Jun–Aug)":[6,7,8],"Autumn (Sep–Nov)":[9,10,11]}
scol=[C_BLUE,C_GREEN,C_RED,C_ORANGE]

def make(series,kind,unit,heatcmap,fname,suptitle):
    # season FULL WIDTH top (big); day-of-week + heatmap side by side below
    fig=plt.figure(figsize=(16,12))
    gs=gridspec.GridSpec(2,2,height_ratios=[1.05,1.0],hspace=0.28,wspace=0.22)
    ax0=fig.add_subplot(gs[0,:]); ax1=fig.add_subplot(gs[1,0]); ax2=fig.add_subplot(gs[1,1])
    fmt=mticker.FuncFormatter(lambda x,_:f"{x:,.0f}")
    for (sn,ms),sc in zip(seasons.items(),scol):
        sub=series[series.index.month.isin(ms)]; pr=sub.groupby(sub.index.hour).mean()
        ax0.plot(pr.index,pr.values,color=sc,linewidth=3.0,label=sn)
    ax0.set_xticks(range(0,24,1)); ax0.set_xlabel("Hour of day"); ax0.set_ylabel(f"Mean {kind} ({unit})")
    ax0.set_title(f"Intraday {kind.capitalize()} Profile by Season",fontweight="bold",fontsize=17)
    ax0.yaxis.set_major_formatter(fmt); _ymin,_ymax=ax0.get_ylim(); ax0.set_ylim(_ymin,_ymax+(_ymax-_ymin)*0.20)
    ax0.legend(framealpha=0.92,ncol=4,loc="upper center"); _clean(ax0)
    for d in range(7):
        sub=series[series.index.dayofweek==d]; pr=sub.groupby(sub.index.hour).mean()
        ax1.plot(pr.index,pr.values,color=(C_ORANGE if d>=5 else C_BLUE),linewidth=(2.6 if d>=5 else 1.7),alpha=0.85,label=dow[d])
    ax1.set_xticks(range(0,24,3)); ax1.set_xlabel("Hour of day"); ax1.set_ylabel(f"Mean {kind} ({unit})")
    ax1.set_title(f"Intraday {kind.capitalize()} Profile by Day-of-Week",fontweight="bold")
    ax1.yaxis.set_major_formatter(fmt); _y0,_y1=ax1.get_ylim(); ax1.set_ylim(_y0,_y1+(_y1-_y0)*0.22)
    ax1.legend(ncol=4,framealpha=0.92,loc="upper center"); _clean(ax1)
    piv=series.to_frame(name="v"); piv["hour"]=piv.index.hour; piv["dow"]=piv.index.dayofweek
    heat=piv.groupby(["dow","hour"])["v"].mean().unstack()
    im=ax2.imshow(heat.values,aspect="auto",cmap=heatcmap,interpolation="nearest")
    ax2.set_xticks(range(0,24,3)); ax2.set_xticklabels(range(0,24,3)); ax2.set_yticks(range(7)); ax2.set_yticklabels(dow)
    ax2.set_xlabel("Hour of day"); ax2.set_title(f"Mean {kind.capitalize()} Heatmap (Day-of-Week × Hour)",fontweight="bold")
    fig.colorbar(im,ax=ax2,label=unit,shrink=0.95,format=fmt); ax2.grid(False)
    fig.suptitle(suptitle,fontsize=16,fontweight="bold",y=0.965)
    fig.savefig(OUT/fname,bbox_inches="tight"); plt.close(fig); print("Saved",fname)

make(price,"price","€/MWh","RdYlBu_r","fig_daily_weekly_profile.png",
     "Day-Ahead Price: Seasonal, Intraday and Weekly Patterns")
make(load,"load","MW","YlOrRd","fig_daily_weekly_profile_load.png",
     "System Load: Seasonal, Intraday and Weekly Patterns")
