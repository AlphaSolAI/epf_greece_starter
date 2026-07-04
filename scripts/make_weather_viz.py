# -*- coding: utf-8 -*-
"""
make_weather_viz.py — Δημιουργεί αυτόνομο interactive HTML dashboard για τα ωριαία
δεδομένα καιρού (weather_gr_hourly) από το data/processed/hourly.parquet.

Χρήση (από project root):
    conda run -n epf --no-capture-output python make_weather_viz.py

Έξοδος: weather_gr_hourly_viz.html στο project root (άνοιγμα με διπλό κλικ).
Αναπαράγεται όποτε αλλάξουν τα δεδομένα — απλά ξανατρέξε το script.

Περιεχόμενο dashboard:
  - 6 τοποθεσίες (gr_mean, athens, thessaloniki, patras, larissa, heraklion)
  - 7 μεταβλητές (temperature_2m, relative_humidity_2m, precipitation, cloud_cover,
    wind_speed_10m, wind_gusts_10m, shortwave_radiation)
  - Tab «Μηνιαία τάση» (μηνιαίοι μ.ό.), «Ημερήσιο προφίλ» (μ.ό. ανά ώρα), «Στατιστικά»
"""
import json
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "data", "processed", "hourly.parquet")
OUT = os.path.join(HERE, "weather_gr_hourly_viz.html")

LOCATIONS = ["gr_mean", "athens", "thessaloniki", "patras", "larissa", "heraklion"]
VARIABLES = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
             "wind_speed_10m", "wind_gusts_10m", "shortwave_radiation"]


def build_payload(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["month"] = df.index.to_period("M").to_timestamp()
    df["hour"] = df.index.hour

    out = {"locations": LOCATIONS, "variables": VARIABLES}

    g = df.groupby("month")
    monthly = {}
    for loc in LOCATIONS:
        monthly[loc] = {}
        for var in VARIABLES:
            col = f"w_{loc}_{var}"
            miss = f"w_{loc}_{var}_missing"
            vals = g[col].mean().round(2)
            if miss in df.columns:  # κρύψε μήνες που είναι εξ ολοκλήρου imputed
                vals = vals.where(g[miss].mean() < 1.0)
            monthly[loc][var] = [None if pd.isna(v) else v for v in vals.tolist()]
    out["monthly_dates"] = [d.strftime("%Y-%m") for d in g[f"w_gr_mean_temperature_2m"].mean().index]
    out["monthly"] = monthly

    gh = df.groupby("hour")
    hourly = {}
    for loc in LOCATIONS:
        hourly[loc] = {var: gh[f"w_{loc}_{var}"].mean().round(2).tolist() for var in VARIABLES}
    out["hourly_profile"] = hourly

    summary = {}
    for loc in LOCATIONS:
        summary[loc] = {}
        for var in VARIABLES:
            col = f"w_{loc}_{var}"
            summary[loc][var] = {
                "mean": round(float(df[col].mean()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "std": round(float(df[col].std()), 2),
            }
    out["summary"] = summary

    out["meta"] = {
        "rows": int(len(df)),
        "start": df.index.min().strftime("%Y-%m-%d %H:%M"),
        "end": df.index.max().strftime("%Y-%m-%d %H:%M"),
    }
    return out


HTML_TOP = r'''<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Καιρός Ελλάδας — Ωριαία δεδομένα</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root{ --bg:#0f1115; --panel:#171a21; --panel2:#1e2028; --border:#2b2f3a;
         --text:#e5e7eb; --muted:#9aa2b1; --faint:#6b7280; }
  @media (prefers-color-scheme: light){
    :root{ --bg:#f6f7f9; --panel:#ffffff; --panel2:#f0f2f5; --border:#dfe3ea;
           --text:#1a1d24; --muted:#5a6270; --faint:#8a92a0; } }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--text);
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
        line-height:1.5; padding:24px; }
  .wrap{ max-width:1080px; margin:0 auto; }
  h1{ font-size:22px; font-weight:600; margin:0 0 4px; }
  .sub{ color:var(--muted); font-size:14px; margin:0 0 20px; }
  .tabs{ display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
  .tab{ padding:7px 16px; border-radius:8px; border:1px solid var(--border);
        background:transparent; color:var(--text); font-size:14px; cursor:pointer; }
  .tab.on{ background:var(--panel); border-color:var(--faint); font-weight:600; }
  .controls{ display:flex; gap:14px; margin-bottom:18px; flex-wrap:wrap; align-items:center; }
  select{ background:var(--panel); color:var(--text); border:1px solid var(--border);
          border-radius:8px; padding:8px 10px; font-size:14px; min-width:210px; }
  .locs{ display:flex; gap:12px; flex-wrap:wrap; font-size:14px; color:var(--muted); }
  .locs label{ display:flex; align-items:center; gap:5px; cursor:pointer; }
  .sw{ width:10px; height:10px; border-radius:2px; display:inline-block; }
  .cards{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }
  .card{ background:var(--panel2); border-radius:10px; padding:14px 16px; }
  .card .lab{ font-size:13px; color:var(--muted); }
  .card .val{ font-size:26px; font-weight:600; margin-top:2px; }
  .chartbox{ position:relative; width:100%; height:400px; background:var(--panel);
             border:1px solid var(--border); border-radius:12px; padding:14px; }
  .tblwrap{ overflow-x:auto; background:var(--panel); border:1px solid var(--border);
            border-radius:12px; padding:6px 4px; }
  table{ width:100%; border-collapse:collapse; font-size:13px; }
  th{ padding:8px 10px; text-align:right; font-size:11px; color:var(--faint);
      font-weight:600; border-bottom:1px solid var(--border); }
  th.l{ text-align:left; }
  td{ padding:8px 10px; text-align:right; color:var(--muted); }
  td.name{ text-align:left; color:var(--text); font-weight:600; white-space:nowrap; }
  td .rng{ font-size:11px; color:var(--faint); }
  .note{ font-size:12px; color:var(--faint); margin-top:10px; }
  @media (max-width:640px){ .cards{ grid-template-columns:repeat(2,1fr); } }
</style>
</head>
<body>
<div class="wrap">
  <h1>Καιρός Ελλάδας — Ωριαία δεδομένα</h1>
  <p class="sub" id="subtitle"></p>
  <div class="tabs">
    <button class="tab on" data-tab="monthly">Μηνιαία τάση</button>
    <button class="tab" data-tab="hourly">Ημερήσιο προφίλ (24ω)</button>
    <button class="tab" data-tab="summary">Συγκεντρωτικά στατιστικά</button>
  </div>
  <div class="controls">
    <select id="varSel"></select>
    <div class="locs" id="locChecks"></div>
  </div>
  <div class="cards" id="cards"></div>
  <div class="chartbox" id="chartbox"><canvas id="chart"></canvas></div>
  <div class="tblwrap" id="tblwrap" style="display:none"></div>
</div>
<script>
const D = '''

HTML_BOTTOM = r''';
const VAR_LABELS = {
  temperature_2m:"Θερμοκρασία (°C)", relative_humidity_2m:"Σχετική υγρασία (%)",
  precipitation:"Βροχόπτωση (mm)", cloud_cover:"Νεφοκάλυψη (%)",
  wind_speed_10m:"Ταχύτητα ανέμου (km/h)", wind_gusts_10m:"Ριπές ανέμου (km/h)",
  shortwave_radiation:"Ηλιακή ακτινοβολία (W/m²)" };
const LOC_LABELS = {
  gr_mean:"Ελλάδα (μ.ό.)", athens:"Αθήνα", thessaloniki:"Θεσσαλονίκη",
  patras:"Πάτρα", larissa:"Λάρισα", heraklion:"Ηράκλειο" };
const COL = { gr_mean:"#2a78d6", athens:"#1baf7a", thessaloniki:"#eda100",
              patras:"#8b7bff", larissa:"#e34948", heraklion:"#eb6834" };
let tab="monthly", locs=new Set(["gr_mean","athens"]), chart=null;
document.getElementById("subtitle").textContent =
  D.meta.rows.toLocaleString("el-GR")+" ωριαίες γραμμές · "+D.meta.start+" → "+D.meta.end+
  " · 6 τοποθεσίες × 7 μεταβλητές";
function buildControls(){
  const sel=document.getElementById("varSel");
  sel.innerHTML=D.variables.map(v=>`<option value="${v}">${VAR_LABELS[v]}</option>`).join("");
  sel.value="temperature_2m"; sel.onchange=render;
  document.getElementById("locChecks").innerHTML=D.locations.map(l=>`
    <label><input type="checkbox" data-loc="${l}" ${locs.has(l)?"checked":""}/>
    <span class="sw" style="background:${COL[l]}"></span>${LOC_LABELS[l]}</label>`).join("");
  document.querySelectorAll("#locChecks input").forEach(cb=>{
    cb.onchange=()=>{ cb.checked?locs.add(cb.dataset.loc):locs.delete(cb.dataset.loc); render(); };
  });
}
document.querySelectorAll(".tab").forEach(b=>{
  b.onclick=()=>{ tab=b.dataset.tab;
    document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===b));
    document.getElementById("chartbox").style.display=tab==="summary"?"none":"block";
    document.getElementById("tblwrap").style.display=tab==="summary"?"block":"none"; render(); };
});
function cards(vk){ const s=D.summary.gr_mean[vk];
  document.getElementById("cards").innerHTML=`
    <div class="card"><div class="lab">Μ.ό. (Ελλάδα)</div><div class="val">${s.mean}</div></div>
    <div class="card"><div class="lab">Ελάχιστο</div><div class="val">${s.min}</div></div>
    <div class="card"><div class="lab">Μέγιστο</div><div class="val">${s.max}</div></div>
    <div class="card"><div class="lab">Τυπ. απόκλιση</div><div class="val">${s.std}</div></div>`; }
function summaryTable(){ let rows="";
  D.locations.forEach(l=>{ rows+=`<tr><td class="name">${LOC_LABELS[l]}</td>`;
    D.variables.forEach(v=>{ const s=D.summary[l][v];
      rows+=`<td>${s.mean}<br><span class="rng">${s.min} … ${s.max}</span></td>`; });
    rows+=`</tr>`; });
  const head=D.variables.map(v=>`<th>${VAR_LABELS[v]}</th>`).join("");
  document.getElementById("tblwrap").innerHTML=
    `<table><thead><tr><th class="l"></th>${head}</tr></thead><tbody>${rows}</tbody></table>
     <p class="note">Κάθε κελί: μέσος όρος, με εύρος min … max από κάτω. Όλη η περίοδος ${D.meta.start} → ${D.meta.end}.</p>`; }
function render(){
  const vk=document.getElementById("varSel").value; cards(vk);
  if(tab==="summary"){ summaryTable(); return; }
  let labels,datasets; const dark=matchMedia("(prefers-color-scheme: dark)").matches;
  const grid=dark?"#2b2f3a":"#e3e6ec", tick=dark?"#9aa2b1":"#5a6270";
  if(tab==="monthly"){ labels=D.monthly_dates;
    datasets=[...locs].map(l=>({label:LOC_LABELS[l],data:D.monthly[l][vk],
      borderColor:COL[l],backgroundColor:COL[l],borderWidth:2,pointRadius:0,tension:0.15,spanGaps:true})); }
  else { labels=Array.from({length:24},(_,i)=>i+":00");
    datasets=[...locs].map(l=>({label:LOC_LABELS[l],data:D.hourly_profile[l][vk],
      borderColor:COL[l],backgroundColor:COL[l],borderWidth:2,pointRadius:2,tension:0.3})); }
  if(chart) chart.destroy();
  chart=new Chart(document.getElementById("chart"),{ type:"line", data:{labels,datasets},
    options:{ responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false},
      plugins:{ legend:{labels:{color:tick,boxWidth:12,font:{size:12}}},
        tooltip:{callbacks:{title:(it)=>tab==="monthly"?it[0].label:("Ώρα "+it[0].label)}} },
      scales:{ x:{grid:{display:false},ticks:{color:tick,maxRotation:0,autoSkip:true,maxTicksLimit:tab==="monthly"?16:12}},
        y:{grid:{color:grid},ticks:{color:tick},title:{display:true,text:VAR_LABELS[vk],color:tick,font:{size:12}}} } } });
}
buildControls(); render();
</script>
</body>
</html>
'''


def main():
    if not os.path.exists(PARQUET):
        raise SystemExit(f"Δεν βρέθηκε το parquet: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    payload = build_payload(df)
    data_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HTML_TOP + data_str + HTML_BOTTOM)
    print(f"[OK] {OUT}")
    print(f"     {payload['meta']['rows']} γραμμές, {payload['meta']['start']} → {payload['meta']['end']}")


if __name__ == "__main__":
    main()
