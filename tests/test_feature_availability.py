"""Fast, pure-logic unit tests for src/feature_availability.py.

No conda/training required — pandas/numpy only, milliseconds to run:
    conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q

These lock in the contract that ingest-audit/feature-eng rely on: which
columns land in which ablation group, and how the information-cutoff gap
is computed per (task, gate, market). A regression here would silently
change the meaning of every ablation flag — see energy-forecast SKILL.md
rule 12 (Δ=0.000 red flag) for why this contract deserves direct tests.
"""
import pandas as pd
import pytest

from src.feature_availability import (
    GateSpec,
    classify_columns,
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
