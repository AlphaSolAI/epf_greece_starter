# -*- coding: utf-8 -*-
"""Fast pure-logic tests για src/cycle_runner.py (καμία conda/training)."""
import sys
import textwrap
from pathlib import Path

from src import cycle_runner as cr


def _good_plan():
    return {
        "study": "meteo_check",
        "market": "dam", "task": "price", "strategy": "recursive",
        "retrain": "static", "gate": "strict", "seed": 42,
        "algos": ["lgbm", "xgb"],
        "windows": [
            {"name": "q12026", "test_start": "2025-12-01 00:00",
             "test_end": "2026-02-28 23:00", "train_end": "2025-11-30 23:00"},
            {"name": "summer25", "test_start": "2025-06-01 00:00",
             "test_end": "2025-08-31 23:00", "train_end": "2025-05-31 23:00"},
        ],
        "specs": ["default", "default,meteo"],
    }


# ---- Task 1: plan load/validate ----

def test_valid_plan_passes():
    errors, warnings = cr.validate_plan(_good_plan())
    assert errors == []


def test_missing_required_key_is_error():
    p = _good_plan(); del p["market"]
    errors, _ = cr.validate_plan(p)
    assert any("market" in e for e in errors)


def test_bad_market_is_error():
    p = _good_plan(); p["market"] = "spot"
    errors, _ = cr.validate_plan(p)
    assert any("market" in e for e in errors)


def test_window_name_with_underscore_is_error():
    p = _good_plan(); p["windows"][0]["name"] = "q1_2026"
    errors, _ = cr.validate_plan(p)
    assert any("underscore" in e.lower() or "_" in e for e in errors)


def test_spec_with_underscore_is_error():
    p = _good_plan(); p["specs"] = ["default", "lags_calendar"]
    errors, _ = cr.validate_plan(p)
    assert any("underscore" in e.lower() or "_" in e for e in errors)


def test_tf_strategy_rejected():
    p = _good_plan(); p["strategy"] = "tf"
    errors, _ = cr.validate_plan(p)
    assert any("strategy" in e for e in errors)


def test_single_window_warns_gate2():
    p = _good_plan(); p["windows"] = p["windows"][:1]
    _, warnings = cr.validate_plan(p)
    assert any("§2" in w or "window" in w.lower() for w in warnings)


def test_academic_gate_warns():
    p = _good_plan(); p["gate"] = "academic"
    _, warnings = cr.validate_plan(p)
    assert any("academic" in w.lower() for w in warnings)


def test_train_end_with_weekly_warns():
    p = _good_plan(); p["retrain"] = "weekly"
    _, warnings = cr.validate_plan(p)
    assert any("train_end" in w for w in warnings)


def test_load_plan_reads_yaml(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(textwrap.dedent("""
        study: t
        market: dam
        task: price
        strategy: recursive
        retrain: static
        gate: strict
        seed: 42
        algos: [lgbm]
        windows:
          - {name: q12026, test_start: "2025-12-01 00:00", test_end: "2026-02-28 23:00", train_end: "2025-11-30 23:00"}
          - {name: summer25, test_start: "2025-06-01 00:00", test_end: "2025-08-31 23:00", train_end: "2025-05-31 23:00"}
        specs: [default, "default,meteo"]
    """), encoding="utf-8")
    plan = cr.load_plan(str(f))
    assert plan["study"] == "t"
    assert plan["windows"][0]["name"] == "q12026"


# ---- Task 2: cell expansion + naming ----

def test_expand_cells_count_and_order():
    cells = cr.expand_cells(_good_plan())
    assert len(cells) == 8  # 2 windows × 2 algos × 2 specs
    assert cells[0].window == "q12026" and cells[0].algo == "lgbm" and cells[0].spec == "default"
    assert cells[-1].window == "summer25" and cells[-1].algo == "xgb" and cells[-1].spec == "default,meteo"


def test_cell_json_name_is_synthesize_compatible():
    cells = cr.expand_cells(_good_plan())
    name = cells[-1].json_name()
    assert name == "summer25_xgb_recursive_default,meteo.json"
    # parse όπως ο synthesize: split('_') → 4 πεδία, spec = join(parts[3:])
    stem = name[:-5]
    parts = stem.split("_")
    assert len(parts) >= 4
    window, algo, mode, spec = parts[0], parts[1], parts[2], "_".join(parts[3:])
    assert (window, algo, mode, spec) == ("summer25", "xgb", "recursive", "default,meteo")


def test_cell_carries_window_dates_and_train_end():
    cells = cr.expand_cells(_good_plan())
    c = cells[0]
    assert c.test_start == "2025-12-01 00:00"
    assert c.test_end == "2026-02-28 23:00"
    assert c.train_end == "2025-11-30 23:00"
    assert c.seed == 42 and c.crosslag_mode == "freeze"


def test_expand_cells_weekly_drops_train_end():
    p = _good_plan(); p["retrain"] = "weekly"
    cells = cr.expand_cells(p)
    assert all(c.train_end is None for c in cells)


# ---- Task 3: command building ----

def test_build_command_has_all_flags():
    cells = cr.expand_cells(_good_plan())
    cmd = cr.build_command(cells[-1], "runs/meteo_check", sys.executable)
    joined = " ".join(cmd)
    assert cmd[:4] == [sys.executable, "-m", "src.master_forecast", "--algo"]
    assert "--algo xgb" in joined
    assert "--task price" in joined
    assert "--market dam" in joined
    assert "--strategy recursive" in joined
    assert "--gate strict" in joined
    assert "--retrain static" in joined
    assert "--seed 42" in joined
    assert "--test_start 2025-06-01 00:00" in joined
    assert "--test_end 2025-08-31 23:00" in joined
    assert "--train_end 2025-05-31 23:00" in joined
    assert "--features default,meteo" in joined
    assert "--crosslag_mode freeze" in joined
    assert "--out_json runs/meteo_check/summer25_xgb_recursive_default,meteo.json" in joined
    assert "--quiet" in cmd


def test_build_command_omits_train_end_when_none():
    p = _good_plan(); p["retrain"] = "weekly"
    cells = cr.expand_cells(p)
    cmd = cr.build_command(cells[0], "runs/x", sys.executable)
    assert "--train_end" not in cmd
    assert "--retrain weekly" in " ".join(cmd)


# ---- Task 4: main + dry-run ----

def test_main_dry_run_prints_commands(tmp_path, capsys):
    f = tmp_path / "p.yaml"
    f.write_text(
        "study: t\nmarket: dam\ntask: price\nstrategy: recursive\nretrain: static\n"
        "gate: strict\nseed: 42\nalgos: [lgbm]\n"
        "windows:\n"
        "  - {name: q12026, test_start: \"2025-12-01 00:00\", test_end: \"2026-02-28 23:00\", train_end: \"2025-11-30 23:00\"}\n"
        "  - {name: summer25, test_start: \"2025-06-01 00:00\", test_end: \"2025-08-31 23:00\", train_end: \"2025-05-31 23:00\"}\n"
        "specs: [default, \"default,meteo\"]\n", encoding="utf-8")
    rc = cr.main(["--plan", str(f), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "src.master_forecast" in out
    assert "q12026_lgbm_recursive_default.json" in out
    assert out.count("--out_json") == 4  # 2 windows × 1 algo × 2 specs


def test_main_invalid_plan_returns_2(tmp_path, capsys):
    f = tmp_path / "bad.yaml"
    f.write_text("study: t\nmarket: spot\n", encoding="utf-8")
    rc = cr.main(["--plan", str(f), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR" in out or "λείπει" in out


# ---- Task 5: real execution (μέσω monkeypatch — καμία πραγματική εκτέλεση) ----

def test_run_all_stops_on_preflight_fail(tmp_path, monkeypatch):
    plan = _good_plan(); plan["study"] = "t"; plan["poison"] = False
    cells = cr.expand_cells(plan)
    calls = []
    def fake_run(cmd, log=None):
        calls.append(cmd)
        return 1 if any("preflight_check" in x for x in cmd) else 0
    monkeypatch.setattr(cr, "run_subprocess", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = cr.run_all(plan, cells, "runs/t")
    assert rc == 2
    assert not any("src.master_forecast" in " ".join(c) for c in calls)


def test_run_all_continues_past_failed_cell(tmp_path, monkeypatch):
    plan = _good_plan(); plan["study"] = "t"; plan["poison"] = False
    cells = cr.expand_cells(plan)
    def fake_run(cmd, log=None):
        j = " ".join(cmd)
        if "preflight_check" in j:
            return 0
        if "summer25_xgb" in j:
            return 1
        return 0
    monkeypatch.setattr(cr, "run_subprocess", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = cr.run_all(plan, cells, "runs/t")
    assert rc == 0
    draft = Path("runs/t/DRAFT_deposit.md")
    assert draft.exists()
    assert "FAILED" in draft.read_text(encoding="utf-8")


def test_write_draft_deposit_never_says_accepted(tmp_path, monkeypatch):
    plan = _good_plan(); plan["study"] = "t"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "t").mkdir(parents=True)
    path = cr.write_draft_deposit(plan, "runs/t", "results/t.csv", ok=8, failed=[])
    txt = Path(path).read_text(encoding="utf-8")
    # δεν ισχυρίζεται αποδοχή· αντιθέτως δηλώνει ρητά ότι ΔΕΝ είναι δεκτό
    assert "ACCEPTED" not in txt
    assert "ΔΕΝ είναι ΔΕΚΤΟ" in txt
    assert "validity-reviewer" in txt
