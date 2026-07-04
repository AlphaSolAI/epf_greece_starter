"""3.3 monthly avg by year — STACKED, BIGGER (both panels)."""
import warnings,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from pathlib import Path
warnings.filterwarnings("ignore")
BASE=Path(__file__).resolve().parent; OUT=BASE/"thesis_output"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15,"axes.titlesize":18,
 "axes.labelsize":15,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,
 "figure.dpi":200,"savefig.dpi":200,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#e0e0e0","grid.linewidth":0.6,"axes.axisbelow":True})
def _clean(ax): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
TS,TE="2017-01-01","2025-11-30 23:00"
price=pd.read_parquet(BASE/"data/processed/hourly.parquet").loc[TS:TE]["y"]
load=pd.read_parquet(BASE/"data/processed/hourly_load.parquet").loc[TS:TE]["y"]
load=load.reindex(pd.date_range(TS,TE,freq="h"),fill_value=np.nan)
mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
years=range(2017,2026); cmap=plt.cm.get_cmap("tab10",len(list(years)))
fig,(ax_p,ax_l)=plt.subplots(2,1,figsize=(14.5,14)); fig.subplots_adjust(hspace=0.26)
for i,yr in enumerate(years):
    yp=price[price.index.year==yr]; m=yp.groupby(yp.index.month).mean()
    ax_p.plot(m.index-1,m.values,marker="o",markersize=7,linewidth=2.6,color=cmap(i),label=str(yr))
    yl=load[load.index.year==yr]; m=yl.groupby(yl.index.month).mean()
    ax_l.plot(m.index-1,m.values,marker="o",markersize=7,linewidth=2.6,color=cmap(i),label=str(yr))
for ax,t,u in [(ax_p,"Average Monthly Price by Year","€/MWh"),(ax_l,"Average Monthly Load by Year","MW")]:
    ax.set_xticks(range(12)); ax.set_xticklabels(mn); ax.set_title(t,fontweight="bold")
    ax.set_ylabel(u); ax.legend(ncol=3,loc="upper left",framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}")); _clean(ax)
fig.suptitle("Seasonal Patterns: Day-Ahead Price and System Load (Greek Electricity Market, 2017–2025)",
             fontsize=17,fontweight="bold",y=0.995)
fig.savefig(OUT/"fig_monthly_avg_by_year.png",bbox_inches="tight"); print("Saved 3.3 bigger")
