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

import numpy as np
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
    "ramp",       # renewable ramp (docs/features/renewable_ramp): solar_ramp1h, wind_ramp1h
                  # (diff(t,t-1) πάνω σε ήδη day-ahead-known resfc — leakage-free by construction)
    "xborder",    # cross-border: xb_*_lag24/48/168 (γειτονικές DAM τιμές — ΜΟΝΟ lagged·
                  # same-day = ίδιο SDAC auction με το target ⇒ leakage, δεν υπάρχει πια στο parquet)
    "meteo",      # weather (w_*) (+missing flags) — ΠΡΟΣΟΧΗ: observed/oracle (last.md §2 Α6)
    "meteo_vintage",  # gate-aware blend (wveff_*) των D-1/D-2 vintage forecast buckets
                      # (docs/features/meteo_vintage) — task=load parquet μόνο· ΕΚΤΟΣ default.
                      # Τα ΩΜΑ wv_* αποκλείονται δομικά (βλ. classify_columns).
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

        # renewable ramp (docs/features/renewable_ramp) — on-the-fly diff of resfc
        if cl in ("solar_ramp1h", "wind_ramp1h"):
            groups["ramp"].append(c)
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

        # vintage weather buckets (docs/features/meteo_vintage):
        # ΩΜΑ wv_* (day1/day2 + _missing) → ΔΟΜΙΚΑ εκτός ΟΛΩΝ των ομάδων (ούτε
        # 'other', που είναι ΜΕΣΑ στο default): το day1 bucket είναι νόμιμο μόνο
        # για ώρες h ≤ 23−gap — ωμή στήλη χωρίς το ανά-ώρα blend = leak.
        # Features της ομάδας είναι ΜΟΝΟ τα gate-aware wveff_* που χτίζει το
        # add_meteo_vintage_features.
        if cl.startswith("wv_"):
            continue
        if cl.startswith("wveff_"):
            groups["meteo_vintage"].append(c)
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

    def crosslag_gap_hours(self) -> int:
        """
        Gap (ώρες) για τα gen_*/residual_load/load ACTUAL lags (SYSTEM_DESIGN §4.8,
        family 'genlags'/'loadlags') — ο ΙΔΙΟΣ φυσικός reporting-delay ισχύει
        ΑΝΕΞΑΡΤΗΤΩΣ task (price ή load): η πραγματική παραγωγή/φορτίο δημοσιεύεται
        με την ίδια καθυστέρηση όποιο κι αν είναι το target. Ξαναχρησιμοποιεί το
        load-strict gap (ίδιος μηχανισμός, task πάντα 'load' εδώ).
        """
        if self.delay_override is not None:
            return int(self.delay_override)
        return GateSpec(task="load", gate=self.gate, market=self.market).gap_hours()

    def crosslag_cutoff_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Vectorized cutoff διαθεσιμότητας ΓΙΑ gen/residual_load/load actual lags,
        ένα ανά χρονοσφραγίδα του index (§4.8) — ΓΙΑ TRAINING ROWS (recursive/tf/
        lstm): κάθε γραμμή t αντιμετωπίζεται σαν να σερβίρεται μέσα στο block της
        δικής της ημέρας («ίδιο σχήμα» με το serve, §4.8). Για dam/forward
        block_start = midnight της ημέρας του t· για idm/custom block_start = t.
        ΣΤΟ EVAL το σωστό cutoff είναι ΤΟΥ BLOCK ANCHOR (crosslag_cutoff_for_anchor)
        — σε multi-day blocks (forward) το per-day cutoff θα διέρρεε.
        """
        gap = self.crosslag_gap_hours()
        if self.market in ("dam", "forward"):
            block_start = index.normalize()
        else:
            block_start = index
        return block_start - pd.Timedelta(hours=gap + 1)

    def crosslag_cutoff_for_anchor(self, block_start: pd.Timestamp) -> pd.Timestamp:
        """
        Cutoff διαθεσιμότητας crosslag actuals για ΟΛΟΚΛΗΡΟ το block που ξεκινά
        στο block_start (§4.8, anchor-based μηχανισμός) — αυτό χρησιμοποιεί το
        eval (recursive rollout ΚΑΙ direct row@cutoff). Π.χ. DAM price block
        D 00:00 → 11:00 D-1. Σε forward (168h) ισχύει ΈΝΑ cutoff για όλες τις
        168 ώρες: τίποτα μετά το 11:00 D-1 δεν είναι γνωστό στο issue time.
        """
        return block_start - pd.Timedelta(hours=self.crosslag_gap_hours() + 1)


def describe_gate(gs: GateSpec) -> str:
    return (f"market={gs.market} task={gs.task} gate={gs.gate} "
            f"delay_override={gs.delay_override} → gap={gs.gap_hours()}h "
            f"(crosslag_gap={gs.crosslag_gap_hours()}h)")


# ----------------------------------------------------------------------------
# 2β) METEO_VINTAGE — gate-aware blend των D-1/D-2 vintage forecast buckets
#     (docs/features/meteo_vintage/design.md · GOALS G5). Τα ωμά wv_* δεν είναι
#     ΠΟΤΕ features (classify_columns τα αποκλείει δομικά)· η ομάδα meteo_vintage
#     αποτελείται ΜΟΝΟ από τα wveff_* που χτίζει το add_meteo_vintage_features.
#     Exogenous forecast covariates — εκτός AEL crosslag families by construction.
# ----------------------------------------------------------------------------

_WV_DAY1_RE = re.compile(r"^wv_(.+)_day1(_missing)?$")


def meteo_vintage_day1_ok(hours, gap_hours: int) -> np.ndarray:
    """
    Νομιμότητα του day1 bucket στο gate: για στόχο την ώρα h της ημέρας D,
    issue(day1) = valid−24h = D-1 h:00 και cutoff = D-1 (23−gap):00
    (βλ. GateSpec.cutoff_for_block) ⇒ issue ≤ cutoff ⟺ h ≤ 23 − gap.
    g12 (gap=12, cutoff 11:00 D-1): h ≤ 11 · g14 (--delay 14, cutoff 09:00): h ≤ 9.
    Το day2 (issue D-2 h:00) είναι πάντα νόμιμο για dam — δεν χρειάζεται έλεγχο.
    """
    return np.asarray(hours, dtype=int) <= (23 - int(gap_hours))


def add_meteo_vintage_features(df: pd.DataFrame, gate: GateSpec) -> pd.DataFrame:
    """
    Χτίζει τα effective vintage features: wveff_<x>[_missing] = day1 όπου το day1
    είναι νόμιμο στο gate (meteo_vintage_day1_ok), αλλιώς day2 — δηλ. το ΜΙΚΡΟΤΕΡΟ
    νόμιμο lead-time bucket ανά ώρα-στόχο. Τα _missing flags ακολουθούν το
    ΕΠΙΛΕΓΜΕΝΟ bucket. Ισχύει για single-day blocks (dam/idm): σε multi-day blocks
    (forward) οι μέρες D+1.. θα απαιτούσαν day3+ buckets που δεν υπάρχουν στο
    parquet → ρητό ValueError, ΟΧΙ σιωπηλό leak. Unpaired day1 χωρίς day2 sibling
    → ValueError (χωρίς fallback bucket δεν εγγυάται availability).
    Επιστρέφει ΝΕΟ df (copy) όταν υπάρχουν ζεύγη· τα ωμά wv_* μένουν στο frame
    (αδρανή — δεν επιλέγονται ποτέ). Χωρίς wv_* στήλες (π.χ. price parquet) → no-op
    (η ομάδα μένει κενή· ο T4 #features έλεγχος το πιάνει στο πρώτο log).
    """
    pairs = []
    for c in df.columns:
        m = _WV_DAY1_RE.match(c)
        if not m:
            continue
        stem, miss = m.group(1), m.group(2) or ""
        sibling = f"wv_{stem}_day2{miss}"
        if sibling not in df.columns:
            raise ValueError(
                f"meteo_vintage: λείπει το day2 sibling του {c!r} ({sibling!r}) — "
                f"χωρίς fallback bucket δεν εγγυάται availability για h > 23−gap.")
        pairs.append((c, sibling, f"wveff_{stem}{miss}"))
    if not pairs:
        # π.χ. price parquet (κανένα wv_*): καθαρό no-op ΑΝΕΞΑΡΤΗΤΩΣ market —
        # δεν πρέπει να σκάει ένα price run με --features all.
        return df
    if gate.market not in ("dam", "idm"):
        raise ValueError(
            f"meteo_vintage: μόνο market=dam/idm (single-day blocks) — "
            f"market={gate.market!r} θα απαιτούσε day3+ buckets που δεν υπάρχουν στο parquet.")
    df = df.copy()
    day1_ok = meteo_vintage_day1_ok(df.index.hour, gate.gap_hours())
    for day1_col, day2_col, eff in pairs:
        if eff in df.columns:
            continue
        df[eff] = np.where(day1_ok, df[day1_col].astype(float), df[day2_col].astype(float))
    return df


# ----------------------------------------------------------------------------
# 3) AVAILABILITY ENFORCEMENT LAYER (AEL) — freeze-at-cutoff για crosslag actuals
#    SYSTEM_DESIGN §4.8. Καλύπτει gen_solar/gen_wind/residual_load/load lags —
#    τα ΜΟΝΑ families που σήμερα διαβάζονται αυτούσια από actuals χωρίς cutoff
#    enforcement (y lags/rolls ήδη καλύπτονται από recursive running-substitution·
#    resfc/loadfc/meteo/fuel/xborder/ramp είναι ασφαλή by construction —
#    το ramp είναι backward diff δύο ήδη day-ahead-known resfc στηλών, κληρονομεί resfc-safety).
# ----------------------------------------------------------------------------

# base series -> regex της στήλης lag του (καταγράφει το lag σε ώρες)
CROSSLAG_FAMILIES: Dict[str, "re.Pattern"] = {
    "residual_load": re.compile(r"^residual_load_lag(\d+)$"),
    "gen_solar": re.compile(r"^gen_solar_lag(\d+)$"),
    "gen_wind": re.compile(r"^gen_wind_lag(\d+)$"),
    "load": re.compile(r"^load_lag(\d+)$"),
}

# μικρότερο διαθέσιμο lag ανά base series (χρησιμοποιείται για να ανακατασκευαστεί
# η ΠΡΑΓΜΑΤΙΚΗ τιμή lag0 μέσω shift(-min_lag) — βλ. build_frozen_lookup).
_BASE_SERIES_MIN_LAG = {"residual_load": 1, "gen_solar": 1, "gen_wind": 1, "load": 1}


def detect_crosslag_cols(feature_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """base_series -> {col_name: lag_hours} για τις 4 crosslag οικογένειες."""
    out: Dict[str, Dict[str, int]] = {}
    for base, pat in CROSSLAG_FAMILIES.items():
        m = {}
        for c in feature_cols:
            mm = pat.match(c)
            if mm:
                m[c] = int(mm.group(1))
        if m:
            out[base] = m
    return out


def build_frozen_lookup(df_full: pd.DataFrame,
                        crosslag_cols: Dict[str, Dict[str, int]]) -> Dict[str, pd.Series]:
    """
    base_series -> pd.Series ανακατασκευασμένης ΠΡΑΓΜΑΤΙΚΗΣ τιμής (χωρίς lag),
    indexed όπως το df_full. Π.χ. gen_solar_lag1[t] = gen_solar[t-1] ⇒
    gen_solar[t] = gen_solar_lag1.shift(-1)[t]. Χρησιμοποιείται για να «παγώσουμε»
    ΟΠΟΙΟΔΗΠΟΤΕ lag αυτής της οικογένειας στην τιμή που ίσχυε στο cutoff_F.
    """
    lookups: Dict[str, pd.Series] = {}
    for base in crosslag_cols:
        min_lag = _BASE_SERIES_MIN_LAG.get(base, min(crosslag_cols[base].values()))
        col = f"{base}_lag{min_lag}"
        if col in df_full.columns:
            lookups[base] = df_full[col].astype(float).shift(-min_lag)
    return lookups


def apply_crosslag_freeze(
    frame: pd.DataFrame,
    crosslag_cols: Dict[str, Dict[str, int]],
    frozen_lookup: Dict[str, pd.Series],
    cutoffs: pd.DatetimeIndex,
    mode: str = "freeze",
) -> pd.DataFrame:
    """
    Για κάθε γραμμή t στο frame.index (ίδιου μήκους/σειράς με `cutoffs`), για κάθε
    crosslag στήλη (base b, lag k): αν (t-k) > cutoffs[t] → η τιμή ΔΕΝ ήταν ακόμα
    δημοσιευμένη στο cutoff ⇒ αντικατάσταση:
      mode='freeze' (default, deployable) → τιμή του base series ΣΤΟ cutoff (LOCF)
      mode='nan'    (sensitivity variant, μόνο για δέντρα) → NaN
    Mutates in place· επιστρέφει το frame (convenience). `frame` πρέπει να είναι ήδη
    copy αν ο καλών θέλει να κρατήσει το πρωτότυπο.
    """
    if len(frame) != len(cutoffs):
        raise ValueError("frame και cutoffs πρέπει να έχουν το ίδιο μήκος/σειρά.")
    idx = frame.index
    cutoffs_arr = np.asarray(cutoffs.values)
    for base, colmap in crosslag_cols.items():
        lut = frozen_lookup.get(base)
        for col, lag in colmap.items():
            if col not in frame.columns:
                continue
            src_time = (idx - pd.Timedelta(hours=int(lag))).values
            unsafe = src_time > cutoffs_arr
            if not unsafe.any():
                continue
            if mode == "nan" or lut is None:
                frame.loc[unsafe, col] = np.nan
            else:
                frozen_vals = lut.reindex(cutoffs[unsafe]).to_numpy()
                frame.loc[unsafe, col] = frozen_vals
    return frame


def apply_crosslag_freeze_row(
    row: pd.Series,
    t: pd.Timestamp,
    crosslag_cols: Dict[str, Dict[str, int]],
    frozen_lookup: Dict[str, pd.Series],
    cutoff: pd.Timestamp,
    mode: str = "freeze",
) -> pd.Series:
    """
    Ίδια λογική με apply_crosslag_freeze αλλά για ΜΙΑ γραμμή (pd.Series indexed by
    feature name) στο timestamp t — χρησιμοποιείται στο recursive per-step rollout
    όπου το row build είναι ήδη σε for-loop (βλ. recursive_openloop.py). Mutates
    in place· επιστρέφει το row (convenience).
    """
    for base, colmap in crosslag_cols.items():
        lut = frozen_lookup.get(base)
        for col, lag in colmap.items():
            if col not in row.index:
                continue
            src_time = t - pd.Timedelta(hours=int(lag))
            if src_time <= cutoff:
                continue
            if mode == "nan" or lut is None:
                row[col] = np.nan
            else:
                v = lut.reindex([cutoff]).iloc[0] if cutoff in lut.index else np.nan
                row[col] = float(v) if pd.notna(v) and np.isfinite(v) else np.nan
    return row


def freeze_crosslags_for_gate(
    frame: pd.DataFrame,
    feature_cols: List[str],
    gate: "GateSpec",
    df_full: Optional[pd.DataFrame] = None,
    mode: str = "freeze",
    fillna_other: Optional[float] = None,
    cutoffs: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: ανιχνεύει τις crosslag στήλες μέσα στο `feature_cols`,
    υπολογίζει τα per-row cutoffs από το `gate`, χτίζει το frozen lookup από
    `df_full` (ή από το ίδιο το `frame` αν δεν δοθεί df_full) και εφαρμόζει
    apply_crosslag_freeze. Επιστρέφει ΝΕΟ (copy) frame — δεν πειράζει το πρωτότυπο.
    Καλείται από τα σημεία row-build: recursive/direct/training (§4.8).

    cutoffs: explicit per-row cutoffs (π.χ. σταθερό anchor cutoff για eval block,
    ή t−gap για direct training origins). Αν None → gate.crosslag_cutoff_index
    (training σχήμα: ημέρα-του-t).
    fillna_other: αν δοθεί, γεμίζει τα NaN των ΜΗ-crosslag στηλών με αυτή την τιμή
    (π.χ. 0.0, όπως έκανε ήδη ο caller πριν). Στο mode='nan' οι crosslag στήλες
    ΔΕΝ γεμίζουν ποτέ — τα δέντρα χρειάζονται πραγματικό NaN (sensitivity variant).
    """
    crosslag_cols = detect_crosslag_cols(feature_cols)
    if not crosslag_cols or len(frame) == 0:
        return frame.fillna(fillna_other) if fillna_other is not None else frame
    src = df_full if df_full is not None else frame
    frozen_lookup = build_frozen_lookup(src, crosslag_cols)
    if cutoffs is None:
        cutoffs = gate.crosslag_cutoff_index(frame.index)
    out = frame.copy()
    out = apply_crosslag_freeze(out, crosslag_cols, frozen_lookup, cutoffs, mode=mode)
    if fillna_other is not None:
        if mode == "nan":
            protect = set()
            for colmap in crosslag_cols.values():
                protect.update(colmap.keys())
            fill_cols = [c for c in out.columns if c not in protect]
            out[fill_cols] = out[fill_cols].fillna(fillna_other)
        else:
            out = out.fillna(fillna_other)
    return out
