# Cycle Runner (master orchestrator) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Χτίσε `src/cycle_runner.py` — μία εντολή που, από ένα `plan.yaml`, τρέχει preflight → forecast batch (ανά cell) → §2 πίνακα (synthesize) → run-ledger → draft deposit, χωρίς να αγγίζει δεδομένα και χωρίς να αποφασίζει αποδοχή.

**Architecture:** Ντετερμινιστικός Python orchestrator. Καθαρές, testable συναρτήσεις (plan load/validate, cell expansion, command building) + subprocess-driving συναρτήσεις που καλούν τα ΥΠΑΡΧΟΝΤΑ scripts (`master_forecast`, `preflight_check`, `synthesize_ablation`, `build_run_ledger`). Καμία νέα μοντελιστική/leakage λογική — «glue» γύρω από τον πυρήνα. Spec: `docs/superpowers/specs/2026-07-07-cycle-runner-design.md`.

**Tech Stack:** Python 3 (conda env `epf`), PyYAML 6.0.3 (επιβεβαιωμένα διαθέσιμο), stdlib `subprocess`/`argparse`/`dataclasses`/`pathlib`, pytest (tests/).

## Global Constraints

- **ΟΧΙ νέα leakage/model λογική** — ο runner μόνο *καλεί* validity-περασμένα scripts.
- **Ονόματα cell JSON**: `<window>_<algo>_<strategy>_<spec>.json` — window/algo/strategy slugs **ΧΩΡΙΣ `_`** (ο `synthesize_ablation.py` κάνει `split("_")`). Ο runner το enforce-άρει στο validate.
- **Ένα conda process**: τα cells τρέχουν **σειριακά** μέσα στο ίδιο process.
- **Καμία εγγραφή** σε `data/raw`/`data/processed`/`OLD/`, σε `last.md`/`ABLATION_PLAN`, ούτε `git commit` από τον runner. Deposit = μόνο draft αρχείο.
- **Ποτέ «ΔΕΚΤΟ»**: ο runner τυπώνει το §2 pre-gate του synthesize (candidate/PENDING) και παραπέμπει σε `validity-reviewer` + άνθρωπο.
- **strict gate default**· `--train_end` μόνο με `retrain: static`· `tf` όχι tradeable (δεν επιτρέπεται ως strategy στο plan).
- Outputs: `runs/<study>/` · `results/<study>.csv` · `logs/<study>.log`. Πάντα `--out_json` ανά cell.
- Γλώσσα: σχόλια/docstrings ελληνικά· identifiers αγγλικά (σαν το υπόλοιπο repo).
- Tests: `tests/`, pytest, τρέχουν με `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q` (~δευτερόλεπτα, καμία training).
- Ο runner module ΔΕΝ κάνει import pandas/numpy (μένει light)· η βαριά δουλειά γίνεται μέσω subprocess.

## Existing interfaces (verbatim — μην τα μαντεύεις)

**`src/master_forecast.py`** (invoke: `python -m src.master_forecast ...`):
```
--algo {lgbm,xgb,mlp,lstm,lear} (required)   --task {price,load} (required)
--strategy {recursive,direct,tf,seq2seq}     --market {dam,idm,forward,custom}
--gate {strict,academic}                     --retrain {static,monthly,weekly}
--seed INT (default 42)                       --train_start STR   --train_end STR
--test_start STR (required)                   --test_end STR (required)
--features STR (default "default")            --crosslag_mode {freeze,nan} (default freeze)
--out_json STR                                --quiet
```
Γράφει JSON με `metrics=[{"MAE":..,"Model":..,"RMSE":..,"sMAPE":..}]`, `actual`, `series`, `dates`, `task/market/strategy/gate/retrain/features/crosslag_mode`.

**`.claude/skills/energy-forecast/scripts/preflight_check.py`**: `[--poison] [--baseline]`. Exit **1** σε FAIL.

**`scripts/synthesize_ablation.py`** (system python, stdlib): `--dir DIR (required) --baseline SLUG (default "default") --csv PATH`. Διαβάζει `DIR/*.json`, parse `<window>_<algo>_<mode>_<spec>` με `split("_")` (spec = join(parts[3:], "_")). Τυπώνει ΔMAE πίνακα + §2 pre-gate.

**`scripts/build_run_ledger.py`** (system python, stdlib): `[--root runs] [--out results/run_ledger.csv]`. Σαρώνει `runs/**/*.json`.

---

### Task 1: Plan loading & validation

**Files:**
- Create: `src/cycle_runner.py`
- Test: `tests/test_cycle_runner.py`

**Interfaces:**
- Consumes: PyYAML.
- Produces:
  - `VALID_MARKETS={"dam","idm","forward"}`, `VALID_TASKS={"price","load"}`, `VALID_STRATEGIES={"recursive","direct"}`, `VALID_RETRAIN={"static","monthly","weekly"}`, `VALID_GATES={"strict","academic"}`, `VALID_ALGOS={"lgbm","xgb","mlp","lstm","lear"}`
  - `load_plan(path: str) -> dict`
  - `validate_plan(plan: dict) -> tuple[list[str], list[str]]` — επιστρέφει `(errors, warnings)`. `errors` μη-κενό = fatal (ο caller κάνει exit).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cycle_runner.py
"""Fast pure-logic tests για src/cycle_runner.py (καμία conda/training)."""
import textwrap
import pytest
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
    # underscore σπάει το synthesize_ablation split("_")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.cycle_runner'`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/cycle_runner.py
# -*- coding: utf-8 -*-
"""cycle_runner.py — ντετερμινιστικός orchestrator μιας σύγκρισης πρόβλεψης.

Μία εντολή: preflight → forecast batch (ανά cell) → §2 πίνακας (synthesize)
→ run-ledger → draft deposit. ΔΕΝ αγγίζει δεδομένα, ΔΕΝ αποφασίζει αποδοχή,
ΔΕΝ γράφει σε master MD ή git. Spec: docs/superpowers/specs/2026-07-07-cycle-runner-design.md.

Χρήση:
    conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan configs/cycle_plans/<x>.yaml
    ... --dry-run   # τυπώνει cells + εντολές, χωρίς κανένα training
"""
import argparse
import yaml

VALID_MARKETS = {"dam", "idm", "forward"}
VALID_TASKS = {"price", "load"}
VALID_STRATEGIES = {"recursive", "direct"}   # tf/seq2seq ΟΧΙ tradeable → εκτός plan
VALID_RETRAIN = {"static", "monthly", "weekly"}
VALID_GATES = {"strict", "academic"}
VALID_ALGOS = {"lgbm", "xgb", "mlp", "lstm", "lear"}

_REQUIRED = ("study", "market", "task", "strategy", "retrain", "gate", "algos", "windows", "specs")


def load_plan(path):
    """Διαβάζει το plan.yaml → dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_plan(plan):
    """(errors, warnings). Μη-κενό errors = fatal."""
    errors, warnings = [], []
    for k in _REQUIRED:
        if k not in plan or plan[k] in (None, "", []):
            errors.append(f"λείπει υποχρεωτικό κλειδί: {k}")
    if errors:
        return errors, warnings  # χωρίς τα βασικά, δεν προχωράμε σε βαθύτερους ελέγχους

    if plan["market"] not in VALID_MARKETS:
        errors.append(f"market '{plan['market']}' εκτός {sorted(VALID_MARKETS)}")
    if plan["task"] not in VALID_TASKS:
        errors.append(f"task '{plan['task']}' εκτός {sorted(VALID_TASKS)}")
    if plan["strategy"] not in VALID_STRATEGIES:
        errors.append(f"strategy '{plan['strategy']}' εκτός {sorted(VALID_STRATEGIES)} "
                      f"(tf/seq2seq δεν είναι tradeable)")
    if plan["retrain"] not in VALID_RETRAIN:
        errors.append(f"retrain '{plan['retrain']}' εκτός {sorted(VALID_RETRAIN)}")
    if plan["gate"] not in VALID_GATES:
        errors.append(f"gate '{plan['gate']}' εκτός {sorted(VALID_GATES)}")
    for a in plan["algos"]:
        if a not in VALID_ALGOS:
            errors.append(f"algo '{a}' εκτός {sorted(VALID_ALGOS)}")
    if "_" in str(plan["study"]):
        warnings.append("το 'study' έχει '_' — ΟΚ για φάκελο, αλλά μην το βάζεις σε window/spec")

    # windows
    wins = plan["windows"]
    for w in wins:
        for k in ("name", "test_start", "test_end"):
            if k not in w:
                errors.append(f"window χωρίς '{k}': {w}")
        if "name" in w and "_" in w["name"]:
            errors.append(f"window name '{w['name']}' έχει underscore — σπάει το "
                          f"synthesize_ablation split('_'). Χρησιμοποίησε π.χ. 'q12026'.")
    # specs
    for s in plan["specs"]:
        if "_" in s:
            errors.append(f"spec '{s}' έχει underscore — σπάει το naming. Χρησιμοποίησε ',' "
                          f"(π.χ. 'lags,calendar').")

    # warnings (μη-fatal)
    if plan["gate"] == "academic":
        warnings.append("gate=academic: μόνο για σύγκριση με papers, ΟΧΙ tradeable.")
    if len(wins) < 2:
        warnings.append("μόνο 1 window: το batch ΔΕΝ μπορεί δομικά να πιάσει §2 → θα βγει PENDING.")
    if plan["retrain"] != "static":
        for w in wins:
            if "train_end" in w:
                warnings.append(f"train_end στο window '{w.get('name')}' αγνοείται με "
                                f"retrain={plan['retrain']} (expanding).")
                break
    return errors, warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cycle_runner.py tests/test_cycle_runner.py
git commit -m "feat(cycle-runner): plan loading + validation (Task 1)"
```

---

### Task 2: Cell expansion & JSON naming

**Files:**
- Modify: `src/cycle_runner.py`
- Test: `tests/test_cycle_runner.py`

**Interfaces:**
- Consumes: validated `plan` dict (Task 1).
- Produces:
  - `@dataclass(frozen=True) class Cell` με πεδία `window,algo,strategy,spec,test_start,test_end,train_end(Optional[str]),market,task,gate,retrain,seed,crosslag_mode` και method `json_name() -> str` → `f"{window}_{algo}_{strategy}_{spec}.json"`.
  - `expand_cells(plan: dict) -> list[Cell]` — καρτεσιανό windows × algos × specs (σειρά: window-major, μετά algo, μετά spec· ντετερμινιστικό).

- [ ] **Step 1: Write the failing tests**

```python
# προσθήκη στο tests/test_cycle_runner.py

def test_expand_cells_count_and_order():
    cells = cr.expand_cells(_good_plan())
    # 2 windows × 2 algos × 2 specs = 8
    assert len(cells) == 8
    # window-major order
    assert cells[0].window == "q12026" and cells[0].algo == "lgbm" and cells[0].spec == "default"
    assert cells[-1].window == "summer25" and cells[-1].algo == "xgb" and cells[-1].spec == "default,meteo"


def test_cell_json_name_is_synthesize_compatible():
    cells = cr.expand_cells(_good_plan())
    name = cells[0].json_name()
    assert name == "q12026_lgbm_recursive_default.json"
    # parse όπως ο synthesize: split('_') → 4 πεδία, spec = join(parts[3:])
    stem = name[:-5]
    parts = stem.split("_")
    assert len(parts) >= 4
    window, algo, mode, spec = parts[0], parts[1], parts[2], "_".join(parts[3:])
    assert (window, algo, mode, spec) == ("q12026", "lgbm", "recursive", "default,meteo".split(",")[0] if False else "default")


def test_cell_carries_window_dates_and_train_end():
    cells = cr.expand_cells(_good_plan())
    c = cells[0]
    assert c.test_start == "2025-12-01 00:00"
    assert c.test_end == "2026-02-28 23:00"
    assert c.train_end == "2025-11-30 23:00"
    assert c.seed == 42 and c.crosslag_mode == "freeze"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -k expand_or_cell -q` (ή όλο το αρχείο)
Expected: FAIL (`AttributeError: module 'src.cycle_runner' has no attribute 'expand_cells'`).

- [ ] **Step 3: Write minimal implementation**

```python
# προσθήκη imports στην κορυφή του src/cycle_runner.py
from dataclasses import dataclass
from typing import Optional
import itertools

# προσθήκη μετά το validate_plan
@dataclass(frozen=True)
class Cell:
    window: str
    algo: str
    strategy: str
    spec: str
    test_start: str
    test_end: str
    train_end: Optional[str]
    market: str
    task: str
    gate: str
    retrain: str
    seed: int
    crosslag_mode: str

    def json_name(self):
        return f"{self.window}_{self.algo}_{self.strategy}_{self.spec}.json"


def expand_cells(plan):
    """Καρτεσιανό windows × algos × specs → list[Cell] (window-major, ντετερμινιστικό)."""
    seed = int(plan.get("seed", 42))
    strategy = plan["strategy"]
    crosslag_mode = plan.get("crosslag_mode", "freeze")
    static = plan["retrain"] == "static"
    cells = []
    for w, algo, spec in itertools.product(plan["windows"], plan["algos"], plan["specs"]):
        cells.append(Cell(
            window=w["name"], algo=algo, strategy=strategy, spec=spec,
            test_start=w["test_start"], test_end=w["test_end"],
            train_end=w.get("train_end") if static else None,
            market=plan["market"], task=plan["task"], gate=plan["gate"],
            retrain=plan["retrain"], seed=seed, crosslag_mode=crosslag_mode,
        ))
    return cells
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -q`
Expected: PASS (όλα, 13 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cycle_runner.py tests/test_cycle_runner.py
git commit -m "feat(cycle-runner): cell expansion + synthesize-compatible naming (Task 2)"
```

---

### Task 3: Command building (master_forecast argv)

**Files:**
- Modify: `src/cycle_runner.py`
- Test: `tests/test_cycle_runner.py`

**Interfaces:**
- Consumes: `Cell` (Task 2).
- Produces: `build_command(cell: Cell, out_dir: str, python_exe: str) -> list[str]` — πλήρες argv για `python -m src.master_forecast ...` με `--out_json <out_dir>/<cell.json_name()>` και `--quiet`. `--train_end` ΜΟΝΟ αν `cell.train_end` δεν είναι None.

- [ ] **Step 1: Write the failing test**

```python
# προσθήκη στο tests/test_cycle_runner.py
import sys

def test_build_command_has_all_flags():
    cells = cr.expand_cells(_good_plan())
    cmd = cr.build_command(cells[-1], "runs/meteo_check", sys.executable)
    # cells[-1] = summer25 / xgb / recursive / default,meteo (static → train_end υπάρχει)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -k build_command -q`
Expected: FAIL (`AttributeError: ... 'build_command'`).

- [ ] **Step 3: Write minimal implementation**

```python
# προσθήκη στο src/cycle_runner.py
import os

def build_command(cell, out_dir, python_exe):
    """Πλήρες argv για ένα master_forecast cell. --train_end μόνο αν static+ορισμένο."""
    out_json = os.path.join(out_dir, cell.json_name()).replace("\\", "/")
    cmd = [
        python_exe, "-m", "src.master_forecast",
        "--algo", cell.algo,
        "--task", cell.task,
        "--market", cell.market,
        "--strategy", cell.strategy,
        "--gate", cell.gate,
        "--retrain", cell.retrain,
        "--seed", str(cell.seed),
        "--test_start", cell.test_start,
        "--test_end", cell.test_end,
        "--features", cell.spec,
        "--crosslag_mode", cell.crosslag_mode,
        "--out_json", out_json,
        "--quiet",
    ]
    if cell.train_end:
        cmd += ["--train_end", cell.train_end]
    return cmd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cycle_runner.py tests/test_cycle_runner.py
git commit -m "feat(cycle-runner): master_forecast command builder (Task 3)"
```

---

### Task 4: Orchestration skeleton + `--dry-run`

**Files:**
- Modify: `src/cycle_runner.py`
- Test: `tests/test_cycle_runner.py`

**Interfaces:**
- Consumes: `load_plan`, `validate_plan`, `expand_cells`, `build_command`.
- Produces:
  - `plan_summary(plan, cells) -> str` — human-readable λίστα cells (μία γραμμή/cell: `json_name`).
  - `main(argv=None) -> int` — args `--plan PATH` (required), `--dry-run` (flag). Ροή: load → validate (errors → τύπωσε & return 2· warnings → τύπωσε & συνέχισε) → expand → **dry-run**: τύπωσε summary + κάθε `build_command` (return 0), χωρίς κανένα subprocess. Real-run wiring μπαίνει στο Task 5.

- [ ] **Step 1: Write the failing tests**

```python
# προσθήκη στο tests/test_cycle_runner.py

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
    # 2 windows × 1 algo × 2 specs = 4 cells
    assert out.count("--out_json") == 4


def test_main_invalid_plan_returns_2(tmp_path, capsys):
    f = tmp_path / "bad.yaml"
    f.write_text("study: t\nmarket: spot\n", encoding="utf-8")  # λείπουν κλειδιά + bad market
    rc = cr.main(["--plan", str(f), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "ERROR" in out or "λείπει" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -k main -q`
Expected: FAIL (`AttributeError: ... 'main'`).

- [ ] **Step 3: Write minimal implementation**

```python
# προσθήκη στο src/cycle_runner.py
import sys

def plan_summary(plan, cells):
    lines = [f"STUDY={plan['study']}  market={plan['market']} task={plan['task']} "
             f"strategy={plan['strategy']} retrain={plan['retrain']} gate={plan['gate']}",
             f"{len(cells)} cells:"]
    for c in cells:
        lines.append(f"  - {c.json_name()}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cycle runner — μία εντολή για μια σύγκριση πρόβλεψης")
    ap.add_argument("--plan", required=True, help="διαδρομή σε plan.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="τύπωσε cells + εντολές χωρίς κανένα training")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    errors, warnings = validate_plan(plan)
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 2

    cells = expand_cells(plan)
    out_dir = f"runs/{plan['study']}"
    print(plan_summary(plan, cells))

    if args.dry_run:
        print("\n--- DRY RUN: εντολές που ΘΑ έτρεχαν ---")
        for c in cells:
            print(" ".join(build_command(c, out_dir, sys.executable)))
        return 0

    # Real execution → Task 5
    return run_all(plan, cells, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
```

Σημείωση: το `run_all` ορίζεται στο Task 5· μέχρι τότε, για να τρέχει το αρχείο, πρόσθεσε προσωρινό stub **μόνο αν εκτελείς τα tasks εκτός σειράς** — αλλιώς προχώρα κατευθείαν στο Task 5 (τα dry-run tests δεν καλούν `run_all`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -k main -q`
Expected: PASS (2 tests· τα dry-run/invalid paths δεν αγγίζουν `run_all`).

- [ ] **Step 5: Commit**

```bash
git add src/cycle_runner.py tests/test_cycle_runner.py
git commit -m "feat(cycle-runner): main() + --dry-run orchestration skeleton (Task 4)"
```

---

### Task 5: Real execution — preflight, batch (fail-safe), synthesize, ledger, draft deposit

**Files:**
- Modify: `src/cycle_runner.py`
- Test: `tests/test_cycle_runner.py`

**Interfaces:**
- Consumes: όλα τα προηγούμενα.
- Produces:
  - `run_subprocess(cmd: list[str], log) -> int` — τρέχει με `subprocess.run`, streams σε `log` (file handle ή None=stdout). Επιστρέφει returncode.
  - `run_all(plan, cells, out_dir) -> int` — full pipeline. Επιστρέφει 0 αν preflight PASS και το batch ολοκληρώθηκε (ακόμη κι αν κάποια cells FAILED), 2 αν preflight FAIL.
  - `write_draft_deposit(plan, out_dir, csv_path, ok, failed) -> str` — γράφει `runs/<study>/DRAFT_deposit.md` (πρότυπο, ΟΧΙ εγγραφή σε master MD) και επιστρέφει τη διαδρομή.
- Paths (σταθερές στην κορυφή): `PREFLIGHT=".claude/skills/energy-forecast/scripts/preflight_check.py"`, `SYNTH="scripts/synthesize_ablation.py"`, `LEDGER="scripts/build_run_ledger.py"`.

- [ ] **Step 1: Write the failing tests** (μέσω monkeypatch — καμία πραγματική εκτέλεση)

```python
# προσθήκη στο tests/test_cycle_runner.py
from pathlib import Path

def test_run_all_stops_on_preflight_fail(tmp_path, monkeypatch, capsys):
    plan = _good_plan(); plan["study"] = "t"; plan["poison"] = False
    cells = cr.expand_cells(plan)
    calls = []
    def fake_run(cmd, log=None):
        calls.append(cmd)
        # πρώτη κλήση = preflight → επέστρεψε FAIL (1)
        return 1 if any("preflight_check" in x for x in cmd) else 0
    monkeypatch.setattr(cr, "run_subprocess", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = cr.run_all(plan, cells, "runs/t")
    assert rc == 2
    # δεν έτρεξε κανένα master_forecast
    assert not any("src.master_forecast" in " ".join(c) for c in calls)


def test_run_all_continues_past_failed_cell(tmp_path, monkeypatch):
    plan = _good_plan(); plan["study"] = "t"; plan["poison"] = False
    cells = cr.expand_cells(plan)
    def fake_run(cmd, log=None):
        j = " ".join(cmd)
        if "preflight_check" in j:
            return 0
        if "summer25_xgb" in j:      # ένα cell σκάει
            return 1
        return 0
    monkeypatch.setattr(cr, "run_subprocess", fake_run)
    monkeypatch.chdir(tmp_path)
    rc = cr.run_all(plan, cells, "runs/t")
    assert rc == 0  # ολοκληρώθηκε παρά το 1 FAILED
    draft = Path("runs/t/DRAFT_deposit.md")
    assert draft.exists()
    assert "FAILED" in draft.read_text(encoding="utf-8")


def test_write_draft_deposit_never_says_accepted(tmp_path):
    plan = _good_plan(); plan["study"] = "t"
    (tmp_path / "runs" / "t").mkdir(parents=True)
    import os as _os; _os.chdir(tmp_path)
    path = cr.write_draft_deposit(plan, "runs/t", "results/t.csv", ok=8, failed=[])
    txt = Path(path).read_text(encoding="utf-8")
    assert "ΔΕΚΤΟ" not in txt and "ACCEPTED" not in txt
    assert "validity-reviewer" in txt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -k "run_all or draft" -q`
Expected: FAIL (`AttributeError: ... 'run_all'`).

- [ ] **Step 3: Write minimal implementation**

```python
# προσθήκη imports/σταθερών στην κορυφή του src/cycle_runner.py
import subprocess

PREFLIGHT = ".claude/skills/energy-forecast/scripts/preflight_check.py"
SYNTH = "scripts/synthesize_ablation.py"
LEDGER = "scripts/build_run_ledger.py"


def run_subprocess(cmd, log=None):
    """Τρέχει μία εντολή σειριακά (ένα conda process). Επιστρέφει returncode."""
    stream = log if log is not None else None
    proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    return proc.returncode


def write_draft_deposit(plan, out_dir, csv_path, ok, failed):
    """Γράφει DRAFT_deposit.md (πρότυπο, ΟΧΙ εγγραφή σε master MD)."""
    path = os.path.join(out_dir, "DRAFT_deposit.md").replace("\\", "/")
    lines = [
        f"# DRAFT deposit — {plan['study']} (ΠΡΟΣΧΕΔΙΟ, ΟΧΙ εγγραφή)",
        "",
        "> Παράχθηκε από τον cycle_runner. ΔΕΝ είναι ΔΕΚΤΟ εύρημα. Πέρασέ το από ",
        "> `validity-reviewer` και αποφάσισε ΕΣΥ τι/αν μπαίνει σε last.md/ABLATION_PLAN.",
        "",
        f"- market/task: {plan['market']}/{plan['task']} · strategy={plan['strategy']} "
        f"· retrain={plan['retrain']} · gate={plan['gate']} · seed={plan.get('seed', 42)}",
        f"- specs: {plan['specs']}  (baseline={plan['specs'][0]})",
        f"- windows: {[w['name'] for w in plan['windows']]}",
        f"- cells: {ok} OK / {len(failed)} FAILED",
    ]
    if failed:
        lines.append(f"- FAILED cells: {failed}")
    lines += [
        "",
        f"§2 pre-gate + ΔMAE πίνακας: βλ. `{csv_path}` και το stdout του synthesize.",
        "Επόμενο: validity-reviewer → (αν περνά) ανθρώπινη κατάθεση.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def run_all(plan, cells, out_dir):
    """preflight → batch (fail-safe) → synthesize → ledger → draft deposit."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{plan['study']}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        # 1. PREFLIGHT
        pf = [sys.executable, PREFLIGHT]
        if plan.get("poison"):
            pf.append("--poison")
        print(f"[preflight] {' '.join(pf)}")
        if run_subprocess(pf, log) != 0:
            print("ERROR: preflight FAILED (leakage/env/TZ) — καμία εκτέλεση. Δες το log.")
            return 2

        # 2. BATCH (σειριακά, per-cell fail-safe)
        ok, failed = 0, []
        for c in cells:
            cmd = build_command(c, out_dir, sys.executable)
            print(f"[cell] {c.json_name()}")
            rc = run_subprocess(cmd, log)
            if rc == 0:
                ok += 1
            else:
                failed.append(c.json_name())
                print(f"  FAILED (rc={rc}) — συνεχίζω")

        # 3. SYNTHESIZE (§2 πίνακας — μόνο υπολογισμός)
        csv_path = f"results/{plan['study']}.csv"
        synth = [sys.executable, SYNTH, "--dir", out_dir,
                 "--baseline", plan["specs"][0], "--csv", csv_path]
        print(f"[synthesize] {' '.join(synth)}")
        run_subprocess(synth, None)  # stdout ορατό στον χρήστη

        # 4. LEDGER
        print("[ledger] build_run_ledger")
        run_subprocess([sys.executable, LEDGER], log)

        # 5. DRAFT DEPOSIT + STOP
        draft = write_draft_deposit(plan, out_dir, csv_path, ok, failed)

    print(f"\n=== SUMMARY: {ok}/{len(cells)} cells OK" +
          (f", FAILED: {failed}" if failed else "") + " ===")
    print(f"ΔMAE/§2: {csv_path} · draft: {draft} · log: {log_path}")
    print("STOP: pre-gate ≠ ΔΕΚΤΟ. Τρέξε validity-reviewer, μετά αποφάσισε/κατάθεσε ΕΣΥ.")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_cycle_runner.py -q`
Expected: PASS (όλα τα tests, ~18).

- [ ] **Step 5: Commit**

```bash
git add src/cycle_runner.py tests/test_cycle_runner.py
git commit -m "feat(cycle-runner): real execution — preflight/batch/synthesize/ledger/draft (Task 5)"
```

---

### Task 6: Example plan template + code review + end-to-end smoke

**Files:**
- Create: `configs/cycle_plans/example_dam_meteo.yaml`
- Create: `configs/cycle_plans/README.md`

**Interfaces:** κανένα νέο· χρήση του CLI των Tasks 1-5.

- [ ] **Step 1: Γράψε το example plan**

```yaml
# configs/cycle_plans/example_dam_meteo.yaml
# Παράδειγμα: βοηθάει το meteo το recursive DAM/price; (2 windows × 2 algos)
# Τρέξε: conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan configs/cycle_plans/example_dam_meteo.yaml --dry-run
study: example_dam_meteo        # ΧΩΡΙΣ '_' σε window/spec (όχι στο study)
market: dam
task: price
strategy: recursive
retrain: static
gate: strict
seed: 42
crosslag_mode: freeze
poison: true
algos: [lgbm, xgb]
windows:
  - {name: q12026,   test_start: "2025-12-01 00:00", test_end: "2026-02-28 23:00", train_end: "2025-11-30 23:00"}
  - {name: summer25, test_start: "2025-06-01 00:00", test_end: "2025-08-31 23:00", train_end: "2025-05-31 23:00"}
specs:
  - default            # baseline (1ο)
  - default,meteo
```

- [ ] **Step 2: Γράψε το README**

```markdown
# configs/cycle_plans — plan.yaml για τον cycle_runner

Κάθε αρχείο = μία σύγκριση. Ο runner (`src/cycle_runner.py`) τρέχει preflight →
forecast batch → §2 πίνακα → ledger → draft deposit. ΔΕΝ αγγίζει δεδομένα, ΔΕΝ
αποφασίζει αποδοχή. Spec: `docs/superpowers/specs/2026-07-07-cycle-runner-design.md`.

## Κανόνες ονομάτων
- window `name`, algos, strategy: **ΧΩΡΙΣ underscore** (`q12026`, όχι `q1_2026`) —
  ο synthesize κάνει `split('_')`.
- specs: χώρισε ομάδες με `,` (`lags,calendar`), ποτέ `_`.

## Dry-run πρώτα (χωρίς training)
    conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan <plan> --dry-run

## Πλήρες run (detached, μεγάλα batch)
    Start-Process -WindowStyle Hidden -FilePath "C:\Program Files\Git\bin\bash.exe" `
      -WorkingDirectory "<repo>" -ArgumentList '-c', `
      'conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan <plan> > logs/<study>_boot.log 2>&1'
    # παρακολούθηση: Get-Content logs\<study>.log -Wait -Tail 20
```

- [ ] **Step 3: Dry-run επαλήθευση (καμία training)**

Run: `conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan configs/cycle_plans/example_dam_meteo.yaml --dry-run`
Expected: τυπώνει 8 cells + 8 γραμμές `--out_json runs/example_dam_meteo/<...>.json`, κανένα training.

- [ ] **Step 4: Code review (MODE TOOLING)**

Ο cycle_runner είναι νέο tooling που παράγει artifacts για downstream χρήση → dispatch agent `epf-code-reviewer` (MODE TOOLING) στο diff (`src/cycle_runner.py`, tests). Διόρθωσε ό,τι σηκώσει BLOCK· σώσε το review σε `reports/qa/` αν προκύψει BLOCK/pre-commit finding. (ΟΧΙ validity-reviewer — δεν υπάρχουν claims εδώ.)

- [ ] **Step 5: 1-cell end-to-end smoke (checkpoint — ο άνθρωπος το τρέχει)**

Μικρό plan (1 window Δεκ-2025-only, 1 algo, baseline μόνο) για να επιβεβαιωθεί η αλυσίδα end-to-end **μία φορά** χωρίς μεγάλο κόστος. Επειδή είναι πραγματικό training (>2-3 min), τρέξε **detached** κατά το SKILL.md pattern και δώσε στον χρήστη το copy-paste block + το `Get-Content ... -Wait` για παρακολούθηση. Κριτήριο επιτυχίας: παράγεται `runs/<study>/<cell>.json`, το synthesize τυπώνει πίνακα (baseline-only → NOISE/no-Δ, αναμενόμενο), το `results/run_ledger.csv` ανανεώθηκε, γράφτηκε `DRAFT_deposit.md`. **Anchor**: αν το window/spec ταιριάζει με γνωστό baseline, MAE εντός ±0.05.

- [ ] **Step 6: Commit**

```bash
git add configs/cycle_plans/ src/cycle_runner.py tests/test_cycle_runner.py
git commit -m "feat(cycle-runner): example plan + README + TOOLING review (Task 6)"
```

---

## Self-Review (συμπληρώθηκε)

**Spec coverage:** §0/§1 εύρος 3→7 → Tasks 1-5· §0.1 γενικότητα (market/task params) → Task 2/3 (περνούν ως flags)· §2 plan.yaml schema → Task 1 (validation) + Task 6 (example)· §3 ροή → Task 5 (`run_all`)· §3 naming/underscore → Task 1 (validate) + Task 2 (json_name)· §4 artifacts (JSONs/csv/ledger/draft/log) → Task 5· §5 «ποτέ ΔΕΚΤΟ» → Task 5 (`write_draft_deposit` test)· §6 testing (plan-validation/dry-run/1-cell smoke/anchor) → Tasks 1-6· detached/one-conda → Task 6 README + smoke.

**Placeholder scan:** καμία TBD· όλα τα steps έχουν πλήρη κώδικα/εντολές.

**Type consistency:** `Cell.json_name()`, `validate_plan→(errors,warnings)`, `expand_cells→list[Cell]`, `build_command(cell,out_dir,python_exe)→list[str]`, `run_subprocess(cmd,log)→int`, `run_all(plan,cells,out_dir)→int`, `write_draft_deposit(plan,out_dir,csv_path,ok,failed)→str` — συνεπή σε όλα τα tasks.

**Ανοιχτό (μεταφέρθηκε από spec §7):** επιλέχθηκε **master_forecast ανά cell** (όχι run_master_grid) λόγω naming compatibility με synthesize — κλειδωμένο εδώ.
