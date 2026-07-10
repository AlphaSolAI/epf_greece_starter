"""Fast, pure-logic unit tests for src/feature_availability.py.

No conda/training required — pandas/numpy only, milliseconds to run:
    conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q

These lock in the contract that ingest-audit/feature-eng rely on: which
columns land in which ablation group, and how the information-cutoff gap
is computed per (task, gate, market). A regression here would silently
change the meaning of every ablation flag — see energy-forecast SKILL.md
rule 12 (Δ=0.000 red flag) for why this contract deserves direct tests.
"""
import numpy as np
import pandas as pd
import pytest

from src.feature_availability import (
    GateSpec,
    add_meteo_vintage_features,
    classify_columns,
    meteo_vintage_day1_ok,
    parse_feature_spec,
    select_features,
)

ALL_COLS = [
    "hour", "dow", "is_holiday", "hour_sin", "hour_cos",
    "y_lag1", "y_lag24", "y_lag168", "y_lag5", "y_lag17",
    "y_roll24", "y_roll168",
    "solar_fc_dayahead", "wind_onshore_fc", "gen_fc_total",
    "load_fc", "load_fc_flag",
    "resload_fc",
    "xb_bg_lag24", "xb_it_lag48",
    "w_temp", "w_wind_speed",
    "gas_price", "co2_price_lag1",
    "gen_solar_lag1", "gen_wind_lag24", "residual_load_lag1",
    "load_lag24",
    "solar_ramp1h", "wind_ramp1h",
    "some_unclassified_col",
]


def test_classify_columns_core_groups():
    groups = classify_columns(ALL_COLS)
    assert set(groups["calendar"]) == {"hour", "dow", "is_holiday", "hour_sin", "hour_cos"}
    assert set(groups["lags"]) == {"y_lag1", "y_lag24", "y_lag168"}
    assert set(groups["dense"]) == {"y_lag5", "y_lag17"}
    assert set(groups["roll"]) == {"y_roll24", "y_roll168"}


def test_classify_columns_forecast_vs_actual_families_stay_separate():
    """resfc/loadfc (day-ahead, legal) must never mix with genlags/loadlags (actuals)."""
    groups = classify_columns(ALL_COLS)
    assert set(groups["resfc"]) == {"solar_fc_dayahead", "wind_onshore_fc", "gen_fc_total"}
    assert set(groups["loadfc"]) == {"load_fc", "load_fc_flag"}
    assert set(groups["engfc"]) == {"resload_fc"}
    assert set(groups["ramp"]) == {"solar_ramp1h", "wind_ramp1h"}
    assert set(groups["xborder"]) == {"xb_bg_lag24", "xb_it_lag48"}
    assert set(groups["genlags"]) == {"gen_solar_lag1", "gen_wind_lag24", "residual_load_lag1"}
    assert set(groups["loadlags"]) == {"load_lag24"}
    # no overlap between any two groups
    seen = set()
    for g, cols in groups.items():
        overlap = seen & set(cols)
        assert not overlap, f"columns {overlap} classified into >1 group (group={g})"
        seen |= set(cols)


def test_classify_columns_unclassified_falls_to_other():
    groups = classify_columns(ALL_COLS)
    assert "some_unclassified_col" in groups["other"]


@pytest.mark.parametrize("spec,expected_present,expected_absent", [
    (None, ["calendar", "lags", "resfc", "loadfc"], ["dense", "xborder", "engfc"]),
    ("all", ["calendar", "lags", "dense", "xborder", "engfc"], []),
    ("default,-resfc", ["calendar", "lags", "loadfc"], ["resfc"]),
    ("lags,calendar,genlags", ["calendar", "lags", "genlags"], ["resfc", "loadfc", "meteo"]),
    ("all,-meteo,-dense", ["calendar", "lags", "resfc"], ["meteo", "dense"]),
])
def test_parse_feature_spec_groups(spec, expected_present, expected_absent):
    groups = parse_feature_spec(spec)
    for g in expected_present:
        assert g in groups, f"spec={spec!r} should include {g}, got {groups}"
    for g in expected_absent:
        assert g not in groups, f"spec={spec!r} should exclude {g}, got {groups}"


def test_parse_feature_spec_core_always_included_unless_explicit_exclude():
    # calendar/lags are core: present even when not named explicitly
    assert set(("calendar", "lags")) <= set(parse_feature_spec("resfc"))
    # ...unless explicitly excluded
    groups = parse_feature_spec("all,-calendar,-lags")
    assert "calendar" not in groups
    assert "lags" not in groups


def test_parse_feature_spec_umbrella_expansion_symmetric():
    inc = parse_feature_spec("forecast")
    assert "resfc" in inc and "loadfc" in inc
    exc = parse_feature_spec("all,-forecast")
    assert "resfc" not in exc and "loadfc" not in exc


def test_select_features_preserves_input_order():
    cols = ["y_lag1", "hour", "solar_fc_dayahead", "load_fc"]
    out = select_features(cols, ["calendar", "lags"])
    assert out == ["y_lag1", "hour"]  # order follows all_cols, not group order


# ---------------------------------------------------------------------------
# meteo_vintage — gate-aware D-1/D-2 vintage weather blend
# (docs/features/meteo_vintage/design.md · GOALS G5). The boundary hour of the
# day1 bucket IS the availability rule: an off-by-one here is a silent 1h leak
# that no poisoning test can catch (exogenous covariate, outside AEL families).
# ---------------------------------------------------------------------------

WV_RAW_COLS = [
    "wv_gr_mean_temperature_2m_day1", "wv_gr_mean_temperature_2m_day2",
    "wv_gr_mean_temperature_2m_day1_missing", "wv_gr_mean_temperature_2m_day2_missing",
    "wv_athens_shortwave_radiation_day1", "wv_athens_shortwave_radiation_day2",
]


def test_classify_columns_raw_vintage_buckets_structurally_excluded():
    """Raw wv_* (day1/day2/_missing) must land in NO group — not even 'other'.
    'other' is inside DEFAULT_GROUPS: an unclassified raw day1 column would
    silently become a default feature and leak for target hours h > 23-gap.
    Only the gate-aware blend (wveff_*) is selectable."""
    groups = classify_columns(ALL_COLS + WV_RAW_COLS)
    for c in WV_RAW_COLS:
        for g, cols in groups.items():
            assert c not in cols, f"raw {c} must not be selectable (found in group {g!r})"


def test_classify_columns_wveff_goes_to_meteo_vintage():
    eff_cols = ["wveff_gr_mean_temperature_2m", "wveff_gr_mean_temperature_2m_missing"]
    groups = classify_columns(ALL_COLS + WV_RAW_COLS + eff_cols)
    assert set(groups["meteo_vintage"]) == set(eff_cols)


def test_meteo_vintage_not_in_default_but_selectable():
    assert "meteo_vintage" not in parse_feature_spec(None)
    assert "meteo_vintage" not in parse_feature_spec("default")
    assert "meteo_vintage" in parse_feature_spec("all")
    assert "meteo_vintage" in parse_feature_spec("calendar,lags,roll,meteo_vintage")


@pytest.mark.parametrize("gap,h,ok", [
    (12, 11, True), (12, 12, False),   # g12: cutoff 11:00 D-1 (δικό μας gate)
    (14, 9, True), (14, 10, False),    # g14 (--delay 14, ΑΔΜΗΕ-aligned): cutoff 09:00 D-1
    (0, 23, True), (0, 0, True),       # gap=0: day1 νόμιμο για όλες τις ώρες
])
def test_meteo_vintage_day1_ok_boundaries(gap, h, ok):
    assert bool(meteo_vintage_day1_ok(np.array([h]), gap)[0]) is ok


def _wv_frame(hours: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2026-01-15 00:00", periods=hours, freq="h")
    return pd.DataFrame({
        "y": 1.0,
        "wv_gr_mean_temperature_2m_day1": 1.0,
        "wv_gr_mean_temperature_2m_day2": 2.0,
        "wv_gr_mean_temperature_2m_day1_missing": 0.0,
        "wv_gr_mean_temperature_2m_day2_missing": 1.0,
    }, index=idx)


def test_add_meteo_vintage_features_blend_g12_and_g14():
    df = _wv_frame()
    out = add_meteo_vintage_features(df, GateSpec(task="load", gate="strict", market="dam"))  # gap 12
    eff = out["wveff_gr_mean_temperature_2m"]
    assert (eff[out.index.hour <= 11] == 1.0).all()   # day1 legal
    assert (eff[out.index.hour >= 12] == 2.0).all()   # day2 fallback
    # missing flags follow the SELECTED bucket
    effm = out["wveff_gr_mean_temperature_2m_missing"]
    assert (effm[out.index.hour <= 11] == 0.0).all()
    assert (effm[out.index.hour >= 12] == 1.0).all()
    # --delay 14 (ΑΔΜΗΕ-aligned) inherits through GateSpec.gap_hours()
    out14 = add_meteo_vintage_features(
        df, GateSpec(task="load", gate="strict", market="dam", delay_override=14))
    eff14 = out14["wveff_gr_mean_temperature_2m"]
    assert (eff14[out14.index.hour <= 9] == 1.0).all()
    assert (eff14[out14.index.hour >= 10] == 2.0).all()
    # original df untouched (copy semantics), raw columns preserved in output
    assert "wveff_gr_mean_temperature_2m" not in df.columns
    assert "wv_gr_mean_temperature_2m_day1" in out.columns


def test_add_meteo_vintage_features_fails_loud():
    df = _wv_frame()
    # forward: multi-day blocks would need day3+ buckets that don't exist → no silent leak
    with pytest.raises(ValueError):
        add_meteo_vintage_features(df, GateSpec(task="load", gate="strict", market="forward"))
    # unpaired day1 without its day2 fallback sibling → availability not guaranteed
    df_unpaired = df.drop(columns=["wv_gr_mean_temperature_2m_day2"])
    with pytest.raises(ValueError):
        add_meteo_vintage_features(df_unpaired, GateSpec(task="load", gate="strict", market="dam"))
    # no wv_* columns at all (e.g. price parquet) → clean no-op, not an error —
    # ΚΑΙ για market=forward (ένα price run με --features all δεν πρέπει να σκάει)
    df_price = pd.DataFrame({"y": [1.0]}, index=pd.date_range("2026-01-15", periods=1, freq="h"))
    out = add_meteo_vintage_features(df_price, GateSpec(task="load", gate="strict", market="dam"))
    assert list(out.columns) == ["y"]
    out_fw = add_meteo_vintage_features(df_price, GateSpec(task="price", gate="strict", market="forward"))
    assert list(out_fw.columns) == ["y"]


# ---------------------------------------------------------------------------
# GateSpec — the price-vs-load, DAM-vs-IDM information cutoff contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task,gate,market,expected_gap", [
    ("price", "strict", "dam", 0),      # D-1 DAM prices published D-2 -> no gap
    ("price", "strict", "forward", 0),
    ("load", "strict", "dam", 12),      # actual load known only to ~11:00 D-1
    ("load", "academic", "dam", 0),     # academic assumes full D-1 known
    ("price", "strict", "idm", 0),
    ("load", "strict", "idm", 0),
])
def test_gatespec_gap_hours(task, gate, market, expected_gap):
    gs = GateSpec(task=task, gate=gate, market=market)
    assert gs.gap_hours() == expected_gap


def test_gatespec_delay_override_wins():
    gs = GateSpec(task="load", gate="strict", market="dam", delay_override=3)
    assert gs.gap_hours() == 3


def test_gatespec_cutoff_for_block():
    gs = GateSpec(task="load", gate="strict", market="dam")  # gap=12
    block_start = pd.Timestamp("2026-01-15 00:00")
    # cutoff = block_start - (gap+1)h = 2026-01-14 11:00
    assert gs.cutoff_for_block(block_start) == pd.Timestamp("2026-01-14 11:00")


def test_gatespec_crosslag_gap_is_load_strict_regardless_of_task():
    """AEL §4.8: crosslag actuals (gen/load) share ONE physical reporting delay,
    independent of whether the target task is price or load."""
    gs_price = GateSpec(task="price", gate="strict", market="dam")
    gs_load = GateSpec(task="load", gate="strict", market="dam")
    assert gs_price.crosslag_gap_hours() == gs_load.crosslag_gap_hours() == 12
