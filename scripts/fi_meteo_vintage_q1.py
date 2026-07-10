# -*- coding: utf-8 -*-
"""
FI evidence για το deploy checklist του meteo_vintage (feature-eng Στάδιο 6, T8β).

Fit ΕΝΑ LGBM στο smoke config (task=load, Q1 static, calendar,lags,roll,meteo_vintage,
gate g12, seed 42) και τύπωσε: (α) gain importance ανά ΟΜΑΔΑ features, (β) top-15
μεμονωμένα features, (γ) rank του κορυφαίου wveff_*. ΔΕΝ είναι backtest — καμία
πρόβλεψη/MAE δεν παράγεται ή αναφέρεται από εδώ.

Τρέξιμο (conda, ~1'):
  conda run -n epf --no-capture-output python -X utf8 scripts/fi_meteo_vintage_q1.py
Output: stdout (tee σε reports/fi_meteo_vintage_q1.txt)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.feature_availability import (
    GateSpec, add_meteo_vintage_features, classify_columns,
    parse_feature_spec, select_features,
)
from src.split_utils import load_processed

df = load_processed("hourly", task="load")
gate = GateSpec(task="load", gate="strict", market="dam")
groups = parse_feature_spec("calendar,lags,roll,meteo_vintage")
df = add_meteo_vintage_features(df, gate)
df = df.dropna(subset=["y"]).sort_index()

all_cols = [c for c in df.columns if c != "y"]
feats = [c for c in select_features(all_cols, groups)
         if np.issubdtype(df[c].dtype, np.number)]
train = df.loc[:"2025-11-30 23:00"]
X, y = train[feats].fillna(0.0), train["y"]

m = lgb.LGBMRegressor(n_estimators=400, random_state=42, verbose=-1)
m.fit(X, y)

fi = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=feats)
cls = classify_columns(feats)
col2grp = {c: g for g, cols in cls.items() for c in cols}

print(f"config: task=load Q1-static train_end=2025-11-30 seed=42 spec=calendar,lags,roll,meteo_vintage (#features={len(feats)})")
print("\n=== Gain ανά ομάδα (% συνόλου) ===")
grp = fi.groupby(fi.index.map(col2grp)).sum().sort_values(ascending=False)
for g, v in grp.items():
    print(f"{g:14s} {100*v/fi.sum():6.2f}%  ({sum(1 for c in feats if col2grp[c]==g)} features)")

print("\n=== Top-15 features (gain) ===")
for c, v in fi.sort_values(ascending=False).head(15).items():
    print(f"{c:45s} {100*v/fi.sum():6.2f}%  [{col2grp[c]}]")

ranked = fi.sort_values(ascending=False)
wv_first = next((i + 1 for i, c in enumerate(ranked.index) if c.startswith("wveff_")), None)
print(f"\nΚορυφαίο wveff_ feature: rank {wv_first}/{len(feats)}")
