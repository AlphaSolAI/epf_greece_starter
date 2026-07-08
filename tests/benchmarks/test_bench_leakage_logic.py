"""CodSpeed micro-benchmarks — pure-logic leakage/feature paths (spec §4α, φάση 2).

ΤΡΕΧΕΙ ΜΟΝΟ αν είναι εγκατεστημένο το `pytest-codspeed` (CI). Χωρίς αυτό, ολόκληρο
το module γίνεται skip → δεν αγγίζει το ~1s pytest suite. Καμία εξάρτηση σε
training data / conda — μόνο string/list logic, άρα τρέχει καθαρά σε GitHub Actions.

Ενεργοποίηση: βλ. reports/qa/CODSPEED_SETUP.md.
"""
import pytest

pytest.importorskip("pytest_codspeed")  # skip όλο το αρχείο αν λείπει το πακέτο

from src.feature_availability import (  # noqa: E402
    classify_columns,
    detect_crosslag_cols,
    parse_feature_spec,
)


def _realistic_columns(n_lags=48):
    """~200 ρεαλιστικές στήλες: calendar + lags + 4 crosslag οικογένειες + dense."""
    cols = ["hour", "dow", "month", "is_weekend", "is_holiday"]
    cols += [f"y_lag{k}" for k in range(1, n_lags + 1)]
    cols += [f"y_dense_lag{k}" for k in range(4, 24)]
    for base in ("residual_load", "gen_solar", "gen_wind", "load"):
        cols += [f"{base}_lag{k}" for k in (24, 48, 168)]
    cols += [f"roll_mean_{w}" for w in (24, 168)]
    cols += [f"resfc_h{h}" for h in range(24)]
    return cols


COLS = _realistic_columns()
CROSSLAG_COLS = [c for c in COLS if "_lag" in c]


@pytest.mark.benchmark
def test_bench_parse_feature_spec(benchmark):
    benchmark(parse_feature_spec, "default,dense,resfc,loadfc")


@pytest.mark.benchmark
def test_bench_classify_columns(benchmark):
    benchmark(classify_columns, COLS)


@pytest.mark.benchmark
def test_bench_detect_crosslag_cols(benchmark):
    benchmark(detect_crosslag_cols, CROSSLAG_COLS)
