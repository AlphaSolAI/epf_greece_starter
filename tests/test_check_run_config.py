"""Tests για τον QA linter (spec §3) — pure logic, χωρίς conda/training."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "qa"))
import check_run_config as crc

RUN = ("conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast "
       "--algo lgbm --task price --market dam --strategy recursive --gate strict ")


def ids(findings):
    return {f["id"] for f in findings}


def test_clean_command_no_findings():
    cmd = RUN + '--retrain weekly --features "default,dense" --seed 42 --out_json runs/x.json'
    assert crc.scan(cmd) == []


def test_train_end_with_expanding_is_blocker():
    cmd = RUN + '--retrain weekly --train_end "2025-11-30 23:00" --out_json runs/x.json'
    f = crc.scan(cmd)
    assert "train_end_expanding" in ids(f)
    assert any(x["severity"] == "BLOCKER" for x in f if x["id"] == "train_end_expanding")


def test_train_end_with_static_ok():
    cmd = RUN + '--retrain static --train_end "2025-11-30 23:00" --out_json runs/x.json'
    assert "train_end_expanding" not in ids(crc.scan(cmd))


def test_same_day_xborder_is_blocker():
    cmd = RUN + '--retrain weekly --features "default,xb" --out_json runs/x.json'
    assert "same_day_xborder" in ids(crc.scan(cmd))


def test_lagged_xborder_ok():
    cmd = RUN + '--retrain weekly --features "default,xb_lag24" --out_json runs/x.json'
    assert "same_day_xborder" not in ids(crc.scan(cmd))


def test_optuna_is_blocker():
    assert "optuna_forbidden" in ids(crc.scan("python -m src.tune_lgbm_optuna"))
    assert "optuna_forbidden" in ids(crc.scan("python -m src.tune_xgb"))


def test_academic_gate_warning():
    cmd = RUN.replace("--gate strict", "--gate academic") + "--retrain weekly --out_json runs/x.json"
    assert "academic_gate" in ids(crc.scan(cmd))


def test_seed_sweep_with_static_warning():
    cmd = RUN + '--retrain static --seed 7 --out_json runs/x.json'
    assert "seed_sweep_static" in ids(crc.scan(cmd))


def test_seed_42_static_ok():
    cmd = RUN + '--retrain static --seed 42 --out_json runs/x.json'
    assert "seed_sweep_static" not in ids(crc.scan(cmd))


def test_missing_out_json_warning():
    cmd = RUN + "--retrain weekly"
    assert "no_out_json" in ids(crc.scan(cmd))


def test_missing_utf8_minor():
    cmd = ("conda run -n epf --no-capture-output python -m src.master_forecast "
           "--retrain weekly --out_json runs/x.json")
    assert "no_utf8_flag" in ids(crc.scan(cmd))


def test_parallel_conda_background_warning():
    script = (RUN + "--retrain weekly --out_json runs/a.json &\n"
              + RUN + "--retrain weekly --out_json runs/b.json &\n")
    assert "parallel_conda" in ids(crc.scan(script))


def test_multiple_start_process_warning():
    script = ('Start-Process conda -ArgumentList "run..."\n'
              'Start-Process conda -ArgumentList "run..."\n')
    assert "parallel_conda" in ids(crc.scan(script))


def test_nonascii_start_process_warning():
    script = 'Start-Process conda -ArgumentList "τιμη dam"\n'
    assert "nonascii_start_process" in ids(crc.scan(script))


def test_comments_and_blank_lines_ignored():
    script = "# --train_end με weekly εδώ είναι σχόλιο\n\n"
    assert crc.scan(script) == []


def test_main_exit_codes(tmp_path, capsys):
    ok = tmp_path / "ok.sh"
    ok.write_text(RUN + '--retrain weekly --seed 42 --out_json runs/x.json', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        crc.main(["--script", str(ok)])
    assert e.value.code == 0

    bad = tmp_path / "bad.sh"
    bad.write_text(RUN + '--retrain weekly --train_end "2025-11-30 23:00"', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        crc.main(["--script", str(bad)])
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "train_end_expanding" in out
