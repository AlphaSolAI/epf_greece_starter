"""
feature_availability.py — Η καρδιά της leakage-free λογικής του master pipeline.

Δύο ανεξάρτητες ευθύνες:

1) FEATURE-GROUP SELECTION (ablation)
   Ταξινομεί τις στήλες του processed dataframe σε ομάδες (lags, calendar,
   forecast, meteo, fuel, dense, roll, crosslags) και επιστρέφει τη λίστα
   feature που αντιστοιχεί σε ένα include-set.

2) INFORMATION CUTOFF (gate closure)
   Δεδομένου ενός scored block (π.χ. οι 24 ώρες της ημέρας D) υπολογίζει
   ποιά είναι η τελευταία χρονοσφραγίδα της οποίας το actual y επιτρέπεται
   να χρησιμοποιηθεί — διαφορετικά ανά task/gate/market. Ο κανόνας αυτός
   είναι που ξεχωρίζει DAM/IDM/Forward και price/load.

Καμία εξάρτηση από μοντέλα — μόνο ονόματα στηλών & χρονισμός.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

# ----------------------------------------------------------------------------
# 1) FEATURE GROUPS
# ----------------------------------------------------------------------------

# Λεπτόκοκκες (fine) ομάδες — αυτές επιστρέφει το classify_columns.
ALL_GROUPS = [
    "calendar",   # hour/dow/is_holiday/sin-cos — πάντα a priori
    "lags",       # y_lag1,2,3,6,12,24,48,168 (πυρηνικά autoregressive)
    "dense",      # y_lag4..y_lag23 (intraday dense lags)
    "roll",       # y_roll24, y_roll168
    "resfc",      # day-ahead RES/gen forecasts: solar_fc*, wind_onshore_fc*, gen_fc* (+missing)
    "loadfc",     # day-ahead load forecast: load_fc*
    "engfc",      # engineered day-ahead features: resload_fc (load_fc − solar_fc − wind_fc)
    "xborder",    # cross-border: xb_*_lag24/48/168 (γειτονικές DAM τιμές — ΜΟΝΟ lagged·
                  # same-day = ίδιο SDAC auction με το target ⇒ leakage, δεν υπάρχει πια στο parquet)
    "meteo",      # weather (w_*) (+missing flags)
    "fuel",       # gas/co2 price (+lags)
    "genlags",    # actual generation lags: gen_solar_lag*, gen_wind_lag*, residual_load_lag*
    "loadlags",   # actual load lags: load_lag*
    "other",      # ό,τι δεν ταξινομείται ρητά (ουδέτερο exogenous)
]

# Umbrella aliases — τα παλιά χοντρόκοκκα ονόματα επεκτείνονται στα λεπτά.
# Πλήρης backward compatibility με παλιά specs ('all,-forecast', 'crosslags' κ.λπ.).
UMBRELLAS = {
    "forecast": ["resfc", "loadfc"],
    "crosslags": ["genlags", "loadlags", "other"],
}

# Προεπιλογή: ό,τι είναι νόμιμο & χρήσιμο για DAM (χωρίς dense που μπορεί να βλάψει OL).
DEFAULT_GROUPS = ["calendar", "lags", "roll", "resfc", "loadfc",
                  "meteo", "fuel", "genlags", "loadlags", "other"]

_CALENDAR = {"hour", "dow", "is_holiday", "hour_sin", "hour_cos", "dow_sin", "dow_cos"}


def _lag_num(col: str) -> Optional[int]:
    """Επιστρέφει το N αν το col είναι y_lagN, αλλιώς None."""
    m = re.fullmatch(r"y_lag(\d+)", col)
    return int(m.group(1)) if m else None


def classify_columns(all_cols: List[str]) -> Dict[str, List[str]]:
    """Ταξινομεί ΟΛΕΣ τις στήλες (πλην 'y') σε ομάδες feature."""
    groups: Dict[str, List[str]] = {g: [] for g in ALL_GROUPS}

    for c in all_cols:
        if c == "y":
            continue
        cl = c.lower()

        # calendar
        if c in _CALENDAR:
            groups["calendar"].append(c)
            continue

        # target lags: y_lagN  → core (1,2,3,6,12,24,48,168) vs dense (4..23 λοιπά)
        ln = _lag_num(c)
        if ln is not None:
            if ln in (1, 2, 3, 6, 12, 24, 48, 168):
                groups["lags"].append(c)
            else:
                groups["dense"].append(c)
            continue

        # rolling
        if re.fullmatch(r"y_roll\d+", c):
            groups["roll"].append(c)
            continue

        # engineered day-ahead features (πριν τα prefixes για να μην πιαστούν αλλού)
        if cl.startswith("resload_fc"):
            groups["engfc"].append(c)
            continue

        # cross-border (γειτονικές τιμές/flows, day-ahead-known)
        if cl.startswith("xb_"):
            groups["xborder"].append(c)
            continue

        # day-ahead RES/gen forecasts (+missing)
        if cl.startswith("solar_fc") or cl.startswith("wind_onshore_fc") or cl.startswith("gen_fc"):
            groups["resfc"].append(c)
            continue

        # day-ahead load forecast (+τυχόν flags)
        if cl.startswith("load_fc"):
            groups["loadfc"].append(c)
            continue

        # weather (+missing)
        if cl.startswith("w_"):
            groups["meteo"].append(c)
            continue

        # fuel / carbon (price + lags)
        if cl.startswith("gas_") or cl.startswith("co2_"):
            groups["fuel"].append(c)
            continue

        # actual generation lags (residual_load είναι gen-παράγωγο: load − RES)
        if cl.startswith("residual_load_lag") \
                or cl.startswith("gen_solar_lag") or cl.startswith("gen_wind_lag"):
            groups["genlags"].append(c)
            continue

        # actual load lags
        if cl.startswith("load_lag"):
            groups["loadlags"].append(c)
            continue

        # ό,τι δεν ταξινομείται ρητά → 'other' (ουδέτερο exogenous)
        groups["other"].append(c)

    return groups


def select_features(all_cols: List[str], include_groups: List[str]) -> List[str]:
    """
    Επιστρέφει τη λίστα feature (με τη σειρά του all_cols) που ανήκουν στις
    ζητούμενες ομάδες. 'calendar' + 'lags' θεωρούνται πυρηνικά και πάντα μέσα
    (εκτός αν ρητά εξαιρεθούν με πρόθεμα '-' — βλ. parse_feature_spec).
    """
    groups = classify_columns(all_cols)
    keep: set = set()
    for g in include_groups:
        keep.update(groups.get(g, []))
    # διατήρηση αρχικής σειράς
    return [c for c in all_cols if c in keep]


def _expand_umbrellas(names: List[str]) -> List[str]:
    """Επεκτείνει umbrella ονόματα (forecast/crosslags) στα λεπτά τους μέλη."""
    out: List[str] = []
    for n in names:
        out.extend(UMBRELLAS.get(n, [n]))
    return out


def parse_feature_spec(spec: Optional[str]) -> List[str]:
    """
    '--features' parsing.
      None / 'default'       → DEFAULT_GROUPS
      'all'                  → ALL_GROUPS
      'default,-resfc'       → DEFAULT_GROUPS πλην resfc
      'lags,calendar,meteo'  → ακριβώς αυτές
      'all,-meteo,-dense'    → όλες πλην meteo/dense
    Umbrellas: 'forecast' = resfc+loadfc · 'crosslags' = genlags+loadlags+other
    (δουλεύουν και σε include και σε exclude, π.χ. 'all,-forecast').
    """
    if spec is None or spec.strip().lower() in ("", "default"):
        return list(DEFAULT_GROUPS)

    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    include: List[str] = []
    exclude: List[str] = []
    base: Optional[List[str]] = None
    for t in tokens:
        if t == "all":
            base = list(ALL_GROUPS)
        elif t == "default":
            base = list(DEFAULT_GROUPS)
        elif t.startswith("-"):
            exclude.append(t[1:])
        else:
            include.append(t)

    include = _expand_umbrellas(include)
    exclude = _expand_umbrellas(exclude)

    groups = (base + include) if base is not None else include
    groups = [g for g in groups if g not in exclude]

    # πυρηνικά: πάντα calendar + lags εκτός αν ρητά -calendar / -lags
    for core in ("calendar", "lags"):
        if core not in groups and core not in exclude:
            groups.append(core)

    # μοναδικά, με σταθερή σειρά
    seen = set()
    out = []
    for g in ALL_GROUPS:
        if g in groups and g not in seen:
            out.append(g)
            seen.add(g)
    return out


# ----------------------------------------------------------------------------
# 2) INFORMATION CUTOFF (gate closure)
# ----------------------------------------------------------------------------

@dataclass
class GateSpec:
    """Περιγράφει πόσες ώρες ΠΡΙΝ την αρχή ενός scored block είναι το cutoff."""
    task: str          # 'price' | 'load'
    gate: str          # 'strict' | 'academic'
    market: str        # 'dam' | 'idm' | 'forward' | 'custom'
    delay_override: Optional[int] = None  # αν δοθεί ρητά, υπερισχύει (uniform gap)

    def gap_hours(self) -> int:
        """
        Πόσες ώρες κενό (unscored, recursive) υπάρχουν ΜΕΤΑΞΥ cutoff και
        πρώτης scored ώρας. 0 = το cutoff είναι ακριβώς η προηγούμενη ώρα.
        """
        if self.delay_override is not None:
            return int(self.delay_override)

        if self.market in ("dam", "forward"):
            if self.task == "price":
                # Οι τιμές DAM της D-1 είναι γνωστές (δημοσιεύτηκαν D-2) → κενό 0.
                return 0
            # LOAD:
            if self.gate == "strict":
                # actual φορτίο γνωστό ~έως 11:00 D-1 → κενό 12h έως 00:00 D.
                return 12
            return 0  # academic: υποθέτουμε γνωστό όλο το D-1

        if self.market == "idm":
            return 0  # φρέσκα actuals, κοντινός ορίζοντας

        return 0  # custom χωρίς override

    def cutoff_for_block(self, block_start: pd.Timestamp) -> pd.Timestamp:
        """
        Τελευταία χρονοσφραγίδα με χρησιμοποιήσιμο actual y.
        cutoff = block_start - (gap + 1) ώρες.
        """
        return block_start - pd.Timedelta(hours=self.gap_hours() + 1)


def describe_gate(gs: GateSpec) -> str:
    return (f"market={gs.market} task={gs.task} gate={gs.gate} "
            f"delay_override={gs.delay_override} → gap={gs.gap_hours()}h")
