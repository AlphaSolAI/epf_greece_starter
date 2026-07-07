"""Tests για compare_runs (spec §4α) — pure logic, μικρά in-memory runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "qa"))
import compare_runs as cr


def point_run(preds, retrain="weekly"):
    return {
        "strategy": "recursive", "task": "price", "market": "dam", "gate": "strict",
        "retrain": retrain, "features": ["calendar", "dense"],
        "crosslag_mode": "freeze", "horizon": 24, "stride": 24,
        "dates": ["2026-02-01T00:00:00", "2026-02-01T01:00:00", "2026-02-01T02:00:00"],
        "actual": [100.0, 101.0, 102.0],
        "series": {"LGBM-recursive-dam": list(preds)},
        "metrics": [{"Model": "LGBM-recursive-dam", "Type": "ml",
                     "MAE": sum(abs(p - a) for p, a in zip(preds, [100.0, 101.0, 102.0])) / 3}],
    }


def test_identical_runs_equivalent():
    a = point_run([99.0, 100.5, 103.0])
    b = point_run([99.0, 100.5, 103.0])
    r = cr.compare(a, b)
    assert r["equivalent"] is True and r["reasons"] == []


def test_perturbed_predictions_fail():
    a = point_run([99.0, 100.5, 103.0])
    b = point_run([99.0, 100.5, 103.5])
    r = cr.compare(a, b)
    assert r["equivalent"] is False
    assert any("PRED-DIFF" in s for s in r["reasons"])


def test_config_mismatch_not_comparable():
    a = point_run([99.0, 100.5, 103.0], retrain="weekly")
    b = point_run([99.0, 100.5, 103.0], retrain="static")
    r = cr.compare(a, b)
    assert r["equivalent"] is False
    assert any("CONFIG-MISMATCH" in s for s in r["reasons"])


def test_window_mismatch_not_comparable():
    a = point_run([99.0, 100.5, 103.0])
    b = point_run([99.0, 100.5, 103.0])
    b["dates"] = ["2026-03-01T00:00:00", "2026-03-01T01:00:00", "2026-03-01T02:00:00"]
    r = cr.compare(a, b)
    assert any("CONFIG-MISMATCH" in s for s in r["reasons"])


def test_conformal_runs_compared_via_quantiles():
    base = {"method": "quantile_lgbm", "retrain": "weekly", "gate": "strict",
            "market": "dam", "task": "price",
            "test_start": "2025-12-01 00:00", "test_end": "2026-02-28 23:00",
            "dates": ["2025-12-01T00:00:00"], "actual": [100.2],
            "p10": [88.3], "p50": [92.4], "p90": [99.0]}
    other = dict(base, p50=[92.9])
    assert cr.compare(base, dict(base))["equivalent"] is True
    r = cr.compare(base, other)
    assert r["equivalent"] is False and any("PRED-DIFF" in s for s in r["reasons"])


def test_cli_exit_codes(tmp_path):
    import json
    import pytest
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(point_run([99.0, 100.5, 103.0])), encoding="utf-8")
    b.write_text(json.dumps(point_run([99.0, 100.5, 104.0])), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cr.main(["--a", str(a), "--b", str(a)])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        cr.main(["--a", str(a), "--b", str(b)])
    assert e.value.code == 1
