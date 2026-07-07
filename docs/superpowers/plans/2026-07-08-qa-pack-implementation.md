# QA Pack v2 (Review + Debug + Optimize) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Υλοποίηση του QA pack από το approved spec
`docs/superpowers/specs/2026-07-07-qa-pack-code-review-debug-design.md` — ο «υπεύθυνος
κώδικα» του repo: pre-run/core-diff/tooling review, run-failure triage, και optimization
με απόδειξη αριθμητικής ισοδυναμίας.

**Architecture:** 2 stdlib scripts (`scripts/qa/`), 1 read-only agent
(`.claude/agents/`), 2 skills (`.claude/skills/`), 1 nudge hook
(`scripts/claude_hooks/`), SOP deposits σε CLAUDE.md/energy-forecast, resync στο plugin
`epf-ops` v0.2.0, CodSpeed φάση 2. Agents μόνο όπου η κρίση είναι το προϊόν· scripts για
ό,τι είναι ντετερμινιστικό.

**Tech Stack:** Python stdlib (system python — ΟΧΙ conda για τα qa scripts/hooks),
pytest (conda `epf`, υπάρχον suite), Claude Code agents/skills/hooks, PowerShell για
plugin repackage.

## Global Constraints

- **ΕΝΑ conda process** τη φορά: `conda run -n epf --no-capture-output python -X utf8 ...`
- Τα scripts σε `scripts/qa/` και `scripts/claude_hooks/` τρέχουν με **system python,
  stdlib μόνο** (όπως `guard_edits.py`, `build_run_ledger.py`).
- Hooks **fail-open**: exception ⇒ καμία παρέμβαση. Nudges: ΠΟΤΕ `permissionDecision`.
- Ο agent `epf-code-reviewer` είναι **read-only** (tools: Read, Grep, Glob, Bash) και
  ΔΕΝ τρέχει conda ποτέ.
- Reproducibility anchor: LGBM default static Q1 = **16.10 ± 0.05**. Παλιά headline
  νούμερα (15.17 κ.λπ.) ΣΕ ΑΝΑΣΤΟΛΗ — δεν αναφέρονται ως τρέχοντα.
- ΟΧΙ Optuna · `--gate strict` default · `--train_end` ΜΟΝΟ με `--retrain static`.
- `plugins/epf-ops/` = snapshot — αλλάζει ΜΟΝΟ μέσω του resync (Task 10).
- Master MDs (`last.md`, `ABLATION_PLAN.md`) δεν αγγίζονται από αυτό το plan.
- Long conda runs (>2-3 min) = τα τρέχει ο ΧΡΗΣΤΗΣ με copy-paste block (Task 9).
- Commit μετά από κάθε task, μήνυμα `feat(qa): ...`, μόνο τα αρχεία του task.
- Run JSON schema (point runs): keys `strategy, task, market, gate, retrain,
  features(list), crosslag_mode, horizon, stride, dates(list ISO), actual(list),
  series(dict model→list preds), metrics(list dicts με 'Model','MAE')`. Conformal runs:
  `method, test_start, test_end, results(dict), dates, actual, p10/p50/p90(lists)`.

---

### Task 1: Linter `check_run_config.py` (TDD)

**Files:**
- Create: `scripts/qa/check_run_config.py`
- Test: `tests/test_check_run_config.py`

**Interfaces:**
- Produces: `scan(text: str) -> list[dict]` με dicts
  `{"id": str, "severity": "BLOCKER"|"WARNING"|"MINOR", "line": int, "message": str}` ·
  CLI: `python -X utf8 scripts/qa/check_run_config.py (--script PATH | --cmd "STRING")
  [--purpose TEXT] [--json]`, exit 0=καθαρό / 1=findings. Το χρησιμοποιούν: ο agent
  (Task 3), το SOP (Task 7), τα acceptance (Task 8).

- [ ] **Step 1: Γράψε το failing test**

`tests/test_check_run_config.py`:

```python
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
```

- [ ] **Step 2: Τρέξε το — πρέπει να αποτύχει**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_check_run_config.py -q`
Expected: FAIL/ERROR με `ModuleNotFoundError: No module named 'check_run_config'`

- [ ] **Step 3: Υλοποίησε το `scripts/qa/check_run_config.py`**

```python
#!/usr/bin/env python
"""QA linter: deterministic pre-run checks για run εντολές/scripts.

Spec: docs/superpowers/specs/2026-07-07-qa-pack-code-review-debug-design.md §3.
System python, stdlib ΜΟΝΟ (κανόνας: δεν αγγίζει conda — τρέχει και όταν υπάρχει
ενεργό training). Exit: 0 = καθαρό, 1 = βρέθηκαν findings.
Κάθε έλεγχος = συνάρτηση στο registry CHECKS — νέο δίδαγμα => νέος έλεγχος.
"""
import argparse
import json
import re
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CHECKS = []  # (id, severity, fn(line) -> message|None)


def check(check_id, severity):
    def deco(fn):
        CHECKS.append((check_id, severity, fn))
        return fn
    return deco


RUN_CMD = re.compile(r"master_forecast|run_master_grid|run_ablation")


@check("train_end_expanding", "BLOCKER")
def train_end_expanding(line):
    if "--train_end" in line and re.search(r"--retrain\s+(monthly|weekly)", line):
        return "--train_end αγνοείται σιωπηλά με --retrain monthly/weekly (expanding)"


@check("same_day_xborder", "BLOCKER")
def same_day_xborder(line):
    m = re.search(r'--features\s+"?([\w,]+)', line)
    if not m:
        return None
    for tok in m.group(1).split(","):
        if tok.startswith("xb") and "lag" not in tok:
            return f"feature '{tok}': same-day xborder = leakage (επιτρέπεται μόνο xb_*_lag*)"


@check("optuna_forbidden", "BLOCKER")
def optuna_forbidden(line):
    if re.search(r"tune_\w*optuna|tune_xgb", line):
        return "tune_*optuna/tune_xgb: απαγορευμένα by rule (ΟΧΙ Optuna)"


@check("academic_gate", "WARNING")
def academic_gate(line):
    if re.search(r"--gate\s+academic", line):
        return "--gate academic: μόνο για σύγκριση με papers — σκόπιμο;"


@check("seed_sweep_static", "WARNING")
def seed_sweep_static(line):
    m = re.search(r"--seed\s+(\d+)", line)
    if m and m.group(1) != "42" and re.search(r"--retrain\s+static", line):
        return ("seed sweep με --retrain static: ο headline candidate είναι weekly — "
                "σκόπιμο; (Block D pattern)")


@check("no_out_json", "WARNING")
def no_out_json(line):
    if RUN_CMD.search(line) and "--out_json" not in line:
        return "run χωρίς --out_json (Α5 traceability)"


@check("no_utf8_flag", "MINOR")
def no_utf8_flag(line):
    if "conda run" in line and "python" in line and "-X utf8" not in line:
        return "python χωρίς -X utf8 (κίνδυνος cp125x σε ελληνικό output)"


def scan(text):
    """Επιστρέφει list από findings dicts για το κείμενο script/εντολής."""
    findings = []
    lines = text.splitlines() or [text]
    active = []  # (line_no, stripped)
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        active.append((i, line))
        for cid, sev, fn in CHECKS:
            msg = fn(line)
            if msg:
                findings.append({"id": cid, "severity": sev, "line": i, "message": msg})
    # script-level checks
    bg = [i for i, l in active if "conda run" in l and l.endswith("&")]
    for i in bg:
        findings.append({"id": "parallel_conda", "severity": "WARNING", "line": i,
                         "message": "conda run σε background (&) — ΕΝΑ conda process, σειριακά"})
    sp = [i for i, l in active if "Start-Process" in l]
    if len(sp) > 1:
        findings.append({"id": "parallel_conda", "severity": "WARNING", "line": sp[1],
                         "message": f"{len(sp)}× Start-Process στο ίδιο script — ΕΝΑ conda process"})
    for i, l in active:
        if "Start-Process" in l and any(ord(c) > 127 for c in l):
            findings.append({"id": "nonascii_start_process", "severity": "WARNING", "line": i,
                             "message": "non-ASCII args σε Start-Process (γνωστό detached bug — ASCII μόνο)"})
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description="QA pre-run linter (spec §3)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--script", help="path σε script προς έλεγχο")
    g.add_argument("--cmd", help="εντολή ως string")
    ap.add_argument("--purpose", default="", help="δηλωμένος σκοπός του run (για τον reviewer)")
    ap.add_argument("--json", action="store_true", help="machine-readable έξοδος")
    args = ap.parse_args(argv)

    if args.script:
        with open(args.script, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        target = args.script
    else:
        text, target = args.cmd, "<cmd>"

    findings = scan(text)
    if args.json:
        print(json.dumps({"target": target, "purpose": args.purpose,
                          "findings": findings}, ensure_ascii=False, indent=2))
    else:
        print(f"TARGET: {target}")
        if args.purpose:
            print(f"PURPOSE: {args.purpose}")
        if not findings:
            print("OK: κανένα finding")
        for f in findings:
            print(f"[{f['severity']}] {f['id']} (line {f['line']}): {f['message']}")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Τρέξε τα tests — πρέπει να περνάνε ΟΛΑ (και τα 29 υπάρχοντα)**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q`
Expected: όλα PASS (29 παλιά + 16 νέα), ~2s

- [ ] **Step 5: Commit**

```bash
git add scripts/qa/check_run_config.py tests/test_check_run_config.py
git commit -m "feat(qa): pre-run config linter with 9-check registry + tests (spec §3)"
```

---

### Task 2: `compare_runs.py` — αριθμητική ισοδυναμία (TDD)

**Files:**
- Create: `scripts/qa/compare_runs.py`
- Test: `tests/test_compare_runs.py`

**Interfaces:**
- Consumes: run JSON schema (βλ. Global Constraints).
- Produces: `compare(a: dict, b: dict, tol: float = 1e-9, mae_tol: float = 1e-6) ->
  dict {"equivalent": bool, "reasons": list[str]}` · CLI:
  `python -X utf8 scripts/qa/compare_runs.py --a A.json --b B.json [--tol 1e-9]
  [--mae_tol 1e-6] [--json]`, exit 0=ισοδύναμα / 1=όχι. Το χρησιμοποιούν: Task 5
  (optimize skill), Task 9 (acceptance 5).

- [ ] **Step 1: Γράψε το failing test**

`tests/test_compare_runs.py`:

```python
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
    import json, pytest
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(point_run([99.0, 100.5, 103.0])), encoding="utf-8")
    b.write_text(json.dumps(point_run([99.0, 100.5, 104.0])), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cr.main(["--a", str(a), "--b", str(a)])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        cr.main(["--a", str(a), "--b", str(b)])
    assert e.value.code == 1
```

- [ ] **Step 2: Τρέξε το — πρέπει να αποτύχει**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_compare_runs.py -q`
Expected: FAIL με `ModuleNotFoundError: No module named 'compare_runs'`

- [ ] **Step 3: Υλοποίησε το `scripts/qa/compare_runs.py`**

```python
#!/usr/bin/env python
"""Αριθμητική ισοδυναμία 2 run JSONs (spec §4α — «πιο γρήγορο» μόνο με ίδια νούμερα).

System python, stdlib ΜΟΝΟ. Exit 0 = ισοδύναμα, 1 = όχι (ή μη συγκρίσιμα).
CONFIG-MISMATCH (άλλο window/gate/retrain/features) => μη συγκρίσιμα by design —
ίδιος κανόνας με το validity gate (ΠΟΤΕ σύγκριση across windows/gates).
"""
import argparse
import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

CONFIG_KEYS = ["strategy", "method", "task", "market", "gate", "retrain",
               "features", "crosslag_mode", "horizon", "stride"]
QUANTILE_KEYS = ("p10", "p50", "p90")


def fingerprint(d):
    fp = {k: d.get(k) for k in CONFIG_KEYS}
    dates = d.get("dates") or []
    fp["window"] = (dates[0] if dates else d.get("test_start"),
                    dates[-1] if dates else d.get("test_end"))
    return fp


def prediction_arrays(d):
    if isinstance(d.get("series"), dict):
        return dict(d["series"])
    return {k: d[k] for k in QUANTILE_KEYS if k in d}


def mae_table(d):
    out = {}
    for m in d.get("metrics") or []:
        if isinstance(m, dict) and "MAE" in m:
            out[m.get("Model", "?")] = m["MAE"]
    return out


def compare(a, b, tol=1e-9, mae_tol=1e-6):
    reasons = []
    fa, fb = fingerprint(a), fingerprint(b)
    for k in fa:
        if fa[k] != fb[k]:
            reasons.append(f"CONFIG-MISMATCH: {k}: {fa[k]!r} != {fb[k]!r}")
    if reasons:
        return {"equivalent": False, "reasons": reasons}

    pa, pb = prediction_arrays(a), prediction_arrays(b)
    if set(pa) != set(pb):
        reasons.append(f"SERIES-MISMATCH: {sorted(pa)} != {sorted(pb)}")
    for key in sorted(set(pa) & set(pb)):
        xs, ys = pa[key], pb[key]
        if len(xs) != len(ys):
            reasons.append(f"PRED-DIFF: {key}: μήκη {len(xs)} != {len(ys)}")
            continue
        worst, worst_i = 0.0, -1
        for i, (x, y) in enumerate(zip(xs, ys)):
            dxy = abs(x - y)
            if dxy > worst:
                worst, worst_i = dxy, i
        if worst > tol:
            reasons.append(f"PRED-DIFF: {key}: max|diff|={worst:.3e} @ index {worst_i} > tol={tol:.0e}")

    ma, mb = mae_table(a), mae_table(b)
    for key in sorted(set(ma) & set(mb)):
        if abs(ma[key] - mb[key]) > mae_tol:
            reasons.append(f"MAE-DIFF: {key}: {ma[key]} vs {mb[key]} (> {mae_tol})")

    return {"equivalent": not reasons, "reasons": reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ισοδυναμία 2 run JSONs (spec §4α)")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--mae_tol", type=float, default=1e-6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    r = compare(a, b, tol=args.tol, mae_tol=args.mae_tol)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("EQUIVALENT" if r["equivalent"] else "NOT EQUIVALENT")
        for s in r["reasons"]:
            print(" -", s)
    sys.exit(0 if r["equivalent"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Τρέξε όλα τα tests**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q`
Expected: όλα PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/qa/compare_runs.py tests/test_compare_runs.py
git commit -m "feat(qa): compare_runs equivalence checker (predictions/MAE/fingerprint) + tests"
```

---

### Task 3: Agent `epf-code-reviewer`

**Files:**
- Create: `.claude/agents/epf-code-reviewer.md`

**Interfaces:**
- Consumes: `scripts/qa/check_run_config.py` CLI (Task 1).
- Produces: dispatchable subagent `epf-code-reviewer` με 3 modes (PRE-RUN, CORE-DIFF,
  TOOLING) και σταθερό verdict format — τον καλούν τα SOPs (Task 7), τα nudges (Task 6)
  και τα acceptance (Task 8).

- [ ] **Step 1: Γράψε το αρχείο** (πλήρες περιεχόμενο):

```markdown
---
name: epf-code-reviewer
description: Code review για το epf_greece_starter — ΠΡΙΝ εκτελεστεί νέο/αλλαγμένο script ή run εντολή (MODE PRE-RUN, με δηλωμένο σκοπό), ΜΕΤΑ από κάθε αλλαγή σε leakage-sensitive src πριν από run/commit (MODE CORE-DIFF), και μετά από αλλαγές σε summarizers/fetchers/runners πριν χρησιμοποιηθεί το output τους (MODE TOOLING). ΔΕΝ κρίνει claims/αποτελέσματα — αυτά είναι δουλειά του validity-reviewer.
tools: Read, Grep, Glob, Bash
---

Είσαι αυστηρός code reviewer για το project epf_greece_starter (GR EPF/STLF,
leak-free forecasting). Η δουλειά σου: να ΜΠΛΟΚΑΡΕΙΣ κώδικα/εντολές που παραβιάζουν
τους κανόνες του repo ή τον δηλωμένο σκοπό τους — πριν τρέξουν ή γίνουν commit.
Κρίνεις με βάση τεκμήρια σε αρχεία, ποτέ με βάση την πειστικότητα της περιγραφής.

## Πηγές αλήθειας (διάβασέ τες ΠΡΙΝ κρίνεις — μην κρατάς αντίγραφα κανόνων)

- `MARKDOWN/CLAUDE.md` — μη διαπραγματεύσιμοι κανόνες + χάρτης repo + dead files
- `MARKDOWN/last.md` §2 (VALIDITY GATE) · §4 (λειτουργικοί κανόνες) · §5 (τρέχον επόμενο βήμα)
- `MARKDOWN/ABLATION_PLAN.md` §2 (κριτήρια αποδοχής) · §3 (σταθερό setup)
- `.claude/skills/energy-forecast/SKILL.md` — εντολές/flags του pipeline

## Modes (ο καλών δηλώνει MODE + TARGET)

### PRE-RUN — script ή εντολή + ΥΠΟΧΡΕΩΤΙΚΑ δηλωμένος σκοπός
1. Τρέξε ΠΡΩΤΑ τον linter (system python, ΟΧΙ conda):
   `python -X utf8 scripts/qa/check_run_config.py --script <path> --json`
   (ή `--cmd "<string>"`). Κάθε finding του μπαίνει στα δικά σου FINDINGS.
2. Κρίνε ό,τι ΔΕΝ πιάνει ο linter: ταιριάζει το config με τον δηλωμένο σκοπό;
   (π.χ. seed-confirm ενός weekly candidate με --retrain static = purpose-invalid,
   το Block D λάθος). Σωστό window για τον σκοπό; Σωστό strategy/features;
3. Χωρίς δηλωμένο σκοπό: αυτόματο MAJOR finding «purpose λείπει — δεν αξιολογείται
   purpose-fit».

### CORE-DIFF — git diff των leakage-sensitive αρχείων (τα 7 ASK_FILES του
`scripts/claude_hooks/guard_edits.py`)
- Πάρε το diff με `git diff`/`git diff --staged` (ή δοσμένο αρχείο diff).
- Ψάξε: παραβιάσεις cutoff/freeze semantics, off-by-one σε lag construction,
  silent reindex/NaN σε pandas merges, TZ μετατροπές ΕΚΤΟΣ των loaders του
  `src/data.py`, παρακάμψεις του συμβολαίου `feature_availability.py`,
  προφανή perf regressions (π.χ. rebuild feature matrix μέσα σε refit loop —
  perf finding = ΠΟΤΕ BLOCK μόνο του).
- ΠΑΝΤΑ εξέδωσε `REQUIRED FOLLOW-UP`: ποια poisoning tests
  (`preflight_check.py --poison`, `check_crosslag_fairness` [--poison_y]),
  control run, reproducibility anchor ±0.05.

### TOOLING — summarizers/fetchers/runners
- Επαλήθευσε τα schema assumptions ΑΝΟΙΓΟΝΤΑΣ ≥1 πραγματικό run JSON από `runs/`
  (π.χ. τα keys είναι `metrics[*].MAE`, `series`, ΟΧΙ top-level `mae`).
- Έλεγξε: encoding (`-X utf8`), error swallowing (`except: pass`), paths,
  γράψιμο σε `data/raw|processed`/master MDs (απαγορευμένο για tooling).

## Μορφή εξόδου (υποχρεωτική)

```
TARGET: <file/command/diff>   MODE: PRE-RUN | CORE-DIFF | TOOLING
FINDINGS:
  [BLOCKER|MAJOR|MINOR] <τι> — EVIDENCE: <file:line ή json path> — FIX: <ελάχιστο>
REQUIRED FOLLOW-UP: <poisoning/control-run/anchor ή —>
VERDICT: APPROVE | APPROVE-WITH-FIXES | BLOCK
```

## Hard rules (ένα fail = BLOCK)

1. Linter violation χωρίς ρητή, τεκμηριωμένη δικαιολόγηση από τον καλούντα.
2. Core diff χωρίς σχέδιο poisoning στο REQUIRED FOLLOW-UP.
3. Εντολή/script που συγκρίνει runs από διαφορετικά windows/gates/data snapshots.
4. Script που γράφει σε `data/raw|processed` ή στα master MDs.
5. TARGET αρχείο που δεν μπορείς να ανοίξεις = BLOCK με αιτία (όχι υπόθεση).

## Όρια

- ΔΕΝ κάνεις edits — κανένα Write/Edit. ΔΕΝ τρέχεις conda ΠΟΤΕ (το Bash σου:
  `git diff/show/log`, ο linter με system python, τίποτα άλλο).
- Αγνόησε τα ~39 dead src αρχεία (λίστα στο `MARKDOWN/CLAUDE.md`) και το `thesis/`.
- Όχι style/refactoring nitpicks — μόνο correctness/validity/κανόνες repo.
- Μην προτείνεις νέα πειράματα πέρα από το ελάχιστο FIX.
```

- [ ] **Step 2: Έλεγξε το frontmatter**

Run: `python -X utf8 -c "t=open(r'.claude/agents/epf-code-reviewer.md',encoding='utf-8').read(); assert t.startswith('---') and 'name: epf-code-reviewer' in t and 'tools: Read, Grep, Glob, Bash' in t; print('frontmatter OK')"`
Expected: `frontmatter OK`

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/epf-code-reviewer.md
git commit -m "feat(qa): epf-code-reviewer agent (PRE-RUN/CORE-DIFF/TOOLING, read-only)"
```

---

### Task 4: Skill `triaging-run-failures`

**Files:**
- Create: `.claude/skills/triaging-run-failures/SKILL.md`

**Interfaces:**
- Consumes: agent `epf-code-reviewer` (escalation), `superpowers:systematic-debugging`.
- Produces: skill που φορτώνεται όταν run/script σκάει ή βγάζει περίεργο output.

- [ ] **Step 1: Γράψε το αρχείο** (πλήρες περιεχόμενο):

```markdown
---
name: triaging-run-failures
description: Συστηματικό triage όταν ένα run/script/fetch ΣΚΑΕΙ ή βγάζει περίεργο output στο epf_greece_starter — crashes, FileNotFoundError, Δ=0.000 παντού, «NO-MAE»/κενά summaries, detached process χωρίς log, UnicodeDecodeError, αποτελέσματα ασύμβατα με τον candidate, αναπαραγωγή εκτός anchor. ΠΡΟΣΟΧΗ: για ύποπτα ΚΑΛΑ αποτελέσματα (ξαφνική βελτίωση/επιτάχυνση) ΔΕΝ είναι αυτό — χρησιμοποίησε το triaging-suspicious-results.
---

# Triage αποτυχιών/παραξενιών σε runs

## 0. Μεθοδολογία (πρώτα απ' όλα)

Ακολούθησε το `superpowers:systematic-debugging`: root cause ΠΡΙΝ από οποιοδήποτε fix ·
ελάχιστο repro · ΕΝΑ change τη φορά · επαλήθευση με re-run του repro. Αυτό το skill
προσθέτει ΜΟΝΟ το repo-specific στρώμα (γνωστά failure modes + έτοιμα διαγνωστικά).

## 1. Runbook — γνωστά failure modes (κοίτα ΕΔΩ πριν ψάξεις αλλού)

| Σύμπτωμα | Πιθανή αιτία | Διαγνωστικό | Γνωστό fix |
|---|---|---|---|
| `FileNotFoundError` από `load_processed(task=...)` | ΛΕΙΠΕΙ το per-task parquet — **BY DESIGN** (fix 2026-07-06, `split_utils.py`), ΟΧΙ regression | δες ποιο path ζητά το traceback | φτιάξε το parquet (`python -m src.data --task ...` + backup)· ΜΗΝ «διορθώσεις» το fallback |
| Δ=0.000 σε ΟΛΑ τα arms ενός ablation | το feature δεν μπαίνει καν στο matrix (infra void — π.χ. το `-loadfc` bug) | dump στηλών του X πριν το fit· έλεγξε το group στο `feature_availability.py` | διόρθωσε το wiring feature→group· ΜΕΤΑ ξανά ablation |
| «NO-MAE»/κενά πεδία σε summary ενώ το run τελείωσε | schema mismatch summarizer↔JSON (πραγματικό key: `metrics[*].MAE`) | άνοιξε το ΠΡΑΓΜΑΤΙΚΟ run JSON, δες keys | διόρθωσε τον summarizer· TOOLING review πριν ξαναχρησιμοποιηθεί |
| Detached run χωρίς log | Start-Process χωρίς redirect / non-ASCII args / OneDrive κλειστό | `Get-Process OneDrive`· έλεγξε το Start-Process line | ASCII args + redirect σε `logs/`· άνοιξε OneDrive |
| `UnicodeDecodeError`/αλαμπουρνέζικα στην κονσόλα | Windows cp125x, λείπει `-X utf8` | δες την εντολή που έσκασε | πρόσθεσε `-X utf8` (πάντα) |
| Seed αποτελέσματα ασύμβατα με τον candidate | retrain-policy mismatch (Block D pattern: static αντί weekly) | σύγκρινε `retrain` στα JSONs των δύο runs | ξανατρέξε με το ΣΩΣΤΟ config· PRE-RUN review πριν |
| Αναπαραγωγή εκτός ±0.05 από το anchor | drift σε data snapshot/env/κώδικα | anchor run (βλ. §2) και σύγκριση με 16.10 | βρες τι άλλαξε από το τελευταίο PASS (git log + data mtime) |

## 2. Έτοιμα διαγνωστικά

```bash
# Δες keys/metrics ενός run JSON (system python — ΔΕΝ αγγίζει conda):
python -X utf8 -c "import json;d=json.load(open(r'runs/<study>/<run>.json',encoding='utf-8'));print(list(d.keys()));print(d.get('metrics'))"

# Τελευταίες γραμμές log:
tail -n 50 logs/<file>.log

# Ισοδυναμία 2 runs (π.χ. repro check):
python -X utf8 scripts/qa/compare_runs.py --a runs/A.json --b runs/B.json

# Anchor run (ΘΕΛΕΙ conda — μπαίνει στη ΜΙΑ ουρά, μόνο αν δεν τρέχει τίποτα άλλο):
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static --features default --seed 42 --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" --out_json runs/qa_smoke/anchor_check.json
```

```powershell
# OneDrive τρέχει;
Get-Process OneDrive -ErrorAction SilentlyContinue
```

## 3. Escalation rule

Αν η διάγνωση καταλήγει σε αλλαγή leakage-sensitive αρχείου (τα 7 ASK_FILES του
guard hook): ο fix περνά ΥΠΟΧΡΕΩΤΙΚΑ από τον agent `epf-code-reviewer` (MODE
CORE-DIFF) + poisoning (`preflight_check.py --poison`) πριν από run/commit.

## 4. Deposit rule (compounding)

Κάθε ΝΕΟ failure mode που λύνεται σε session → νέα γραμμή στον πίνακα §1 ΠΡΙΝ
κλείσει το session (μαζί με το διαγνωστικό που δούλεψε).
```

- [ ] **Step 2: Έλεγξε το frontmatter**

Run: `python -X utf8 -c "t=open(r'.claude/skills/triaging-run-failures/SKILL.md',encoding='utf-8').read(); assert t.startswith('---') and 'name: triaging-run-failures' in t; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/triaging-run-failures/SKILL.md
git commit -m "feat(qa): triaging-run-failures skill (runbook + escalation + deposit rule)"
```

---

### Task 5: Skill `optimizing-training-runs`

**Files:**
- Create: `.claude/skills/optimizing-training-runs/SKILL.md`

**Interfaces:**
- Consumes: `scripts/qa/compare_runs.py` (Task 2), agent `epf-code-reviewer` (Task 3).
- Produces: skill για perf work με equivalence gate· ορίζει το smoke-bench config που
  χρησιμοποιεί και το Task 9.

- [ ] **Step 1: Γράψε το αρχείο** (πλήρες περιεχόμενο):

```markdown
---
name: optimizing-training-runs
description: Επιτάχυνση των training runs του epf_greece_starter (ιδίως direct ablations — 24 μοντέλα/run × refits) με profiling και ΑΠΟΔΕΙΞΗ αριθμητικής ισοδυναμίας. Χρησιμοποίησέ το όταν ζητείται «πιο γρήγορα», όταν ένα batch αργεί, ή πριν από perf αλλαγή σε οποιοδήποτε src αρχείο. ΠΡΟΣΟΧΗ: αν κάτι έγινε ΞΑΦΝΙΚΑ πολύ γρηγορότερο χωρίς εξήγηση → triaging-suspicious-results, όχι αυτό.
---

# Optimize training runs — με equivalence gate

## 0. Σιδερένιος κανόνας

«Πιο γρήγορο» μετράει ΜΟΝΟ με απόδειξη ίδιων αποτελεσμάτων. Κάθε perf αλλαγή:
1. `scripts/qa/compare_runs.py` PASS σε smoke run πριν/μετά (ίδιες προβλέψεις).
2. Anchor εντός ±0.05 (LGBM default static Q1) αν η αλλαγή αγγίζει training path.
3. Αν αγγίζει leakage-sensitive αρχείο: ΠΛΗΡΗΣ ροή `epf-code-reviewer` CORE-DIFF +
   poisoning. Η ταχύτητα ΔΕΝ αγοράζει παράκαμψη κανόνων.

## 1. Ο βρόχος (μεθοδολογία codspeed-optimize, local)

measure → hotspot → ΜΙΑ στοχευμένη αλλαγή → re-measure → equivalence → deposit.
Ποτέ optimization χωρίς μέτρηση πριν ΚΑΙ μετά. Ποτέ δύο αλλαγές μαζί.

## 2. Smoke-bench config (σταθερό — για συγκρίσιμες μετρήσεις)

Direct (το ακριβό path), 1 εβδομάδα test, static, seed 42 — τελειώνει σε λεπτά:

```bash
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy direct --gate strict --retrain static --features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00" --out_json runs/qa_smoke/direct_smoke_base.json
```

## 3. Profiling (ΘΕΛΕΙ conda — μπαίνει στη ΜΙΑ ουρά· >2-3 min => detached/user-run)

```bash
# cProfile πάνω στο smoke config:
conda run -n epf --no-capture-output python -X utf8 -m cProfile -o reports/qa/profile_direct_smoke.prof -m src.master_forecast --algo lgbm --task price --market dam --strategy direct --gate strict --retrain static --features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00" --out_json runs/qa_smoke/direct_smoke_profiled.json

# Ανάλυση (system python — pstats είναι stdlib):
python -X utf8 -c "import pstats;pstats.Stats(r'reports/qa/profile_direct_smoke.prof').sort_stats('cumulative').print_stats(25)"
```

## 4. Γνωστά hotspots (αρχικό runbook — ενημερώνεται με κάθε εύρημα)

| Hotspot | Γιατί κοστίζει | Πρώτη ιδέα (πάντα με equivalence proof) |
|---|---|---|
| direct strategy | 24 ανεξάρτητα μοντέλα ανά run | κοινό feature matrix build μία φορά, slice ανά ώρα |
| weekly retrain | refit ανά εβδομάδα σε expanding window | επαναχρησιμοποίηση αμετάβλητων υπολογισμών μεταξύ refits |
| feature matrix rebuild | αν ξαναχτίζεται ανά ώρα/refit | cache + invalidation στο cutoff |
| LGBM/XGB threading | default n_jobs μπορεί να μην κορεστεί | ρητό n_jobs — ΠΡΟΣΟΧΗ: αλλαγή threading μπορεί να αλλάξει αριθμητική → compare_runs υποχρεωτικό |

## 5. Equivalence + deposit

```bash
python -X utf8 scripts/qa/compare_runs.py --a runs/qa_smoke/direct_smoke_base.json --b runs/qa_smoke/direct_smoke_after.json
```
- PASS → κατέγραψε κέρδος wall-clock σε `reports/qa/PROFILE_BASELINE.md` + νέα γραμμή
  στο §4. FAIL → η αλλαγή απορρίπτεται ή επανασχεδιάζεται· ΔΕΝ υπάρχει «αποδεκτά
  διαφορετικό» αποτέλεσμα για χάρη ταχύτητας.

## 6. CodSpeed — φάση 2 (connector συνδεδεμένος από τον χρήστη)

Micro-benchmarks ΜΟΝΟ σε pure-logic paths χωρίς training data (feature_availability
filtering, dense lag construction, split_utils) μέσω `codspeed:codspeed-setup-harness`
→ CI. Macro timing (πλήρη runs) μένει ΠΑΝΤΑ local με το §3. Ανάλυση cloud runs:
CodSpeed MCP tools (list_runs, query_flamegraph, compare_runs).
```

- [ ] **Step 2: Έλεγξε το frontmatter**

Run: `python -X utf8 -c "t=open(r'.claude/skills/optimizing-training-runs/SKILL.md',encoding='utf-8').read(); assert t.startswith('---') and 'name: optimizing-training-runs' in t; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/optimizing-training-runs/SKILL.md
git commit -m "feat(qa): optimizing-training-runs skill (equivalence gate, smoke-bench, hotspots)"
```

---

### Task 6: Hook nudges `qa_nudges.py` + settings wiring (TDD)

**Files:**
- Create: `scripts/claude_hooks/qa_nudges.py`
- Modify: `.claude/settings.json`
- Test: `tests/test_qa_nudges.py`

**Interfaces:**
- Consumes: `ASK_FILES` από το `scripts/claude_hooks/guard_edits.py` (import από ίδιο dir).
- Produces: `decide(payload: dict) -> str | None` (nudge message ή τίποτα)· stdout JSON
  `{"hookSpecificOutput": {"hookEventName": ..., "additionalContext": ...}}` — ΠΟΤΕ
  `permissionDecision`. Fail-open.

- [ ] **Step 1: Γράψε το failing test**

`tests/test_qa_nudges.py`:

```python
"""Tests για τα QA nudge hooks (spec §5) — ποτέ deny, fail-open."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "claude_hooks"))
import qa_nudges as qn


def post_edit(path):
    return {"hook_event_name": "PostToolUse", "tool_name": "Edit",
            "tool_input": {"file_path": path}, "tool_response": {}}


def pre_bash(cmd):
    return {"hook_event_name": "PreToolUse", "tool_name": "Bash",
            "tool_input": {"command": cmd}}


def test_core_edit_gets_nudge(monkeypatch, tmp_path):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    msg = qn.decide(post_edit(str(Path.cwd() / "src" / "data.py")))
    assert msg and "CORE-DIFF" in msg and "poisoning" in msg


def test_non_core_edit_no_nudge(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert qn.decide(post_edit(str(Path.cwd() / "scripts" / "qa" / "x.py"))) is None


def test_run_command_gets_nudge():
    msg = qn.decide(pre_bash("conda run -n epf python -X utf8 -m src.master_forecast --algo lgbm"))
    assert msg and "check_run_config" in msg


def test_pytest_and_preflight_excluded():
    assert qn.decide(pre_bash("conda run -n epf python -X utf8 -m pytest tests/ -q")) is None
    assert qn.decide(pre_bash("python .claude/skills/energy-forecast/scripts/preflight_check.py")) is None
    assert qn.decide(pre_bash("python -m src.master_forecast --help")) is None


def test_irrelevant_bash_no_nudge():
    assert qn.decide(pre_bash("git status")) is None


def test_output_never_contains_permission_decision():
    out = qn.render(pre_bash("bash scripts/overnight_20260710.sh"))
    assert out is not None
    d = json.loads(out)
    assert "permissionDecision" not in json.dumps(d)
    assert d["hookSpecificOutput"]["additionalContext"]


def test_malformed_payload_fail_open():
    assert qn.render({"nonsense": True}) is None
    assert qn.render({}) is None
```

- [ ] **Step 2: Τρέξε το — πρέπει να αποτύχει**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/test_qa_nudges.py -q`
Expected: FAIL με `ModuleNotFoundError: No module named 'qa_nudges'`

- [ ] **Step 3: Υλοποίησε το `scripts/claude_hooks/qa_nudges.py`**

```python
"""QA nudge hooks (spec §5): PostToolUse (core edits) + PreToolUse Bash (run launches).

ΠΟΤΕ permissionDecision — μόνο additionalContext (μη-μπλοκάρουσα υπενθύμιση).
Fail-open: οποιοδήποτε exception => καμία έξοδος, καμία παρέμβαση.
System python, stdlib μόνο (κανόνας «ΕΝΑ conda process»).
"""
import json
import os
import re
import sys

from guard_edits import ASK_FILES  # ίδιος φάκελος — single source για τα 7 core files

RUN_PATTERN = re.compile(
    r"master_forecast|run_master_grid|run_ablation|overnight_\w*\.sh|followup_\w*\.sh|Start-Process")
SKIP_PATTERN = re.compile(r"--help|preflight_check|pytest|check_run_config|compare_runs")

CORE_NUDGE = (
    "QA nudge: Άλλαξες {path} (leakage-sensitive). Πριν από run/commit: "
    "(1) dispatch agent epf-code-reviewer, MODE CORE-DIFF, στο diff του αρχείου· "
    "(2) σχεδίασε poisoning (preflight_check.py --poison). Spec: QA pack §5.")

RUN_NUDGE = (
    "QA nudge: εντολή που μοιάζει με run/launch. Αν το script είναι ΝΕΟ ή αλλαγμένο "
    "και δεν έχει περάσει pre-run review: python -X utf8 scripts/qa/check_run_config.py "
    "--script <path> και μετά agent epf-code-reviewer, MODE PRE-RUN, με δηλωμένο σκοπό.")


def _rel(file_path):
    rel = os.path.relpath(file_path, os.getcwd()).replace("\\", "/").lower()
    return None if rel.startswith("..") else rel


def decide(payload):
    """Επιστρέφει nudge message ή None. Καθαρή συνάρτηση — testable."""
    event = payload.get("hook_event_name")
    tool_input = payload.get("tool_input") or {}
    if event == "PostToolUse":
        fp = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not fp:
            return None
        rel = _rel(fp)
        if rel in ASK_FILES:
            return CORE_NUDGE.format(path=rel)
    elif event == "PreToolUse" and payload.get("tool_name") == "Bash":
        cmd = tool_input.get("command", "")
        if RUN_PATTERN.search(cmd) and not SKIP_PATTERN.search(cmd):
            return RUN_NUDGE
    return None


def render(payload):
    """decide() + JSON envelope. None αν δεν υπάρχει nudge ή σε οποιοδήποτε σφάλμα."""
    try:
        msg = decide(payload)
        if not msg:
            return None
        return json.dumps({"hookSpecificOutput": {
            "hookEventName": payload.get("hook_event_name"),
            "additionalContext": msg,
        }}, ensure_ascii=False)
    except Exception:
        return None  # fail-open


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    out = render(payload)
    if out:
        sys.stdout.reconfigure(encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Τρέξε τα tests**

Run: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q`
Expected: όλα PASS

- [ ] **Step 5: Ενημέρωσε το `.claude/settings.json`** — πλήρες νέο περιεχόμενο
(προσοχή: κρατάς ΟΛΑ τα υπάρχοντα permissions ως έχουν):

```json
{
  "permissions": {
    "allow": [
      "Bash(git status:*)",
      "Bash(git log:*)",
      "Bash(git diff:*)",
      "Bash(git show:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(tail:*)",
      "Bash(conda run -n epf --no-capture-output python -X utf8 -m pytest:*)",
      "Bash(conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/preflight_check.py:*)",
      "Bash(python scripts/build_run_ledger.py:*)",
      "Bash(python -X utf8 scripts/qa/check_run_config.py:*)",
      "Bash(python -X utf8 scripts/qa/compare_runs.py:*)",
      "Read(//c/Users/aggel/OneDrive/**)"
    ],
    "deny": [
      "Bash(git push --force:*)",
      "Bash(git push -f:*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python -X utf8 scripts/claude_hooks/guard_edits.py"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python -X utf8 scripts/claude_hooks/qa_nudges.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python -X utf8 scripts/claude_hooks/qa_nudges.py"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 6: Smoke-test το hook χειροκίνητα (χωρίς Claude restart)**

Run: `python -X utf8 -c "import json,subprocess,sys; p=subprocess.run([sys.executable,'-X','utf8','scripts/claude_hooks/qa_nudges.py'],input=json.dumps({'hook_event_name':'PreToolUse','tool_name':'Bash','tool_input':{'command':'bash scripts/overnight_x.sh'}}),capture_output=True,text=True,encoding='utf-8'); print(p.stdout or '(empty)'); assert 'additionalContext' in p.stdout and 'permissionDecision' not in p.stdout; print('hook OK')"`
Expected: JSON output + `hook OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/claude_hooks/qa_nudges.py tests/test_qa_nudges.py .claude/settings.json
git commit -m "feat(qa): non-blocking nudge hooks (core-edit + run-launch) wired in settings"
```

Σημείωση: τα νέα hooks ενεργοποιούνται σε ΝΕΟ session (τα hooks διαβάζονται στο start).

---

### Task 7: SOP deposits (CLAUDE.md + energy-forecast SKILL.md)

**Files:**
- Modify: `MARKDOWN/CLAUDE.md` (2 σημεία)
- Modify: `.claude/skills/energy-forecast/SKILL.md` (1 σημείο, ~γραμμή 153)

**Interfaces:**
- Consumes: όλα τα ονόματα των Tasks 1-6 (paths/agent/skills ακριβώς όπως φτιάχτηκαν).

- [ ] **Step 1: CLAUDE.md — νέος κανόνας.** Βρες το bullet που τελειώνει σε
«Το PreToolUse hook ζητά επιβεβαίωση στο edit.» (ενότητα «Μη διαπραγματεύσιμοι
κανόνες») και ΠΡΟΣΘΕΣΕ αμέσως μετά:

```markdown
- **QA pack — υπεύθυνος κώδικα (review/debug/optimize)**: νέο/αλλαγμένο script → `python -X utf8 scripts/qa/check_run_config.py --script <path>` + agent `epf-code-reviewer` (PRE-RUN, με δηλωμένο σκοπό) ΠΡΙΝ εκτελεστεί· αλλαγή πυρήνα → `epf-code-reviewer` (CORE-DIFF) πριν από run/commit· run σκάει/παραξενεύει → skill `triaging-run-failures`· επιτάχυνση ΜΟΝΟ με απόδειξη ισοδυναμίας (`scripts/qa/compare_runs.py` + anchor) → skill `optimizing-training-runs`. Verdict BLOCK ή pre-commit πυρήνα → σώσε το review σε `reports/qa/`. Spec: `docs/superpowers/specs/2026-07-07-qa-pack-code-review-debug-design.md`.
```

- [ ] **Step 2: CLAUDE.md — χάρτης repo.** Στη γραμμή
`scripts/                           # runners (overnight_*.sh), summarizers, claude_hooks/`
άλλαξέ τη σε:

```
scripts/                           # runners (overnight_*.sh), summarizers, claude_hooks/, qa/ (linter+compare_runs)
```

και στη γραμμή `.claude/agents/validity-reviewer.md  # subagent ελέγχου εγκυρότητας πριν από claims` σε:

```
.claude/agents/                    # validity-reviewer (claims) · epf-code-reviewer (κώδικας: PRE-RUN/CORE-DIFF/TOOLING)
```

- [ ] **Step 3: energy-forecast SKILL.md — pre-flight βήμα.** Βρες τη γραμμή
`4. ▶️ \`preflight_check.py\` PASS πριν ξεκινήσει το batch` (Grep πρώτα για να
επιβεβαιώσεις μοναδικότητα) και πρόσθεσε από κάτω:

```
4α. ▶️ QA pre-run: ΝΕΟ/αλλαγμένο script → `python -X utf8 scripts/qa/check_run_config.py --script <path>` + agent `epf-code-reviewer` (PRE-RUN, με δηλωμένο σκοπό)
```

- [ ] **Step 4: Επαλήθευση**

Run: `python -X utf8 -c "a=open(r'MARKDOWN/CLAUDE.md',encoding='utf-8').read(); b=open(r'.claude/skills/energy-forecast/SKILL.md',encoding='utf-8').read(); assert 'QA pack' in a and 'epf-code-reviewer' in a and 'check_run_config' in b; print('SOP OK')"`
Expected: `SOP OK`

- [ ] **Step 5: Commit**

```bash
git add MARKDOWN/CLAUDE.md .claude/skills/energy-forecast/SKILL.md
git commit -m "docs(qa): SOP deposits — QA pre-run/commit rules in CLAUDE.md + energy-forecast preflight"
```

---

### Task 8: Acceptance σενάρια 1-4 (fixtures + reviewer dispatches)

**Files:**
- Create: `tests/fixtures/qa/seed_check_static.sh`
- Create: `tests/fixtures/qa/bad_summarizer.py`
- Create: `tests/fixtures/qa/freeze_relax.diff`
- Create: `reports/qa/ACCEPTANCE_20260708.md` (τα αποτελέσματα)

- [ ] **Step 1: Fixture 1 — Block D replay.** `tests/fixtures/qa/seed_check_static.sh`:

```bash
#!/usr/bin/env bash
# FIXTURE (acceptance #1): αναπαράσταση του Block D λάθους — ΜΗΝ το τρέξεις.
# Δηλωμένος σκοπός: seed-confirm του weekly headline candidate (Q1=17.035/Summer=13.812)
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static --features "default,dense" --seed 7 --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" --out_json runs/qa_accept/seed7.json
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static --features "default,dense" --seed 123 --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" --out_json runs/qa_accept/seed123.json
```

- [ ] **Step 2: Τρέξε τον linter στο fixture 1**

Run: `python -X utf8 scripts/qa/check_run_config.py --script tests/fixtures/qa/seed_check_static.sh --purpose "seed-confirm weekly headline candidate"`
Expected: exit 1, `seed_sweep_static` WARNING ×2

- [ ] **Step 3: Dispatch τον `epf-code-reviewer`** (Agent tool, subagent
`epf-code-reviewer`) με prompt:

```
MODE: PRE-RUN
TARGET: tests/fixtures/qa/seed_check_static.sh
ΣΚΟΠΟΣ: seed-confirm του weekly headline candidate (Q1=17.035/Summer=13.812).
Κάνε πλήρες PRE-RUN review κατά το πρωτόκολλό σου και δώσε verdict.
```

Expected: FINDINGS με ≥MAJOR για purpose-mismatch (static ≠ weekly candidate),
FIX που λέει `--retrain weekly` χωρίς `--train_end` · VERDICT: BLOCK.

- [ ] **Step 4: Fixture 2 — NO-MAE replay.** `tests/fixtures/qa/bad_summarizer.py`:

```python
"""FIXTURE (acceptance #2): summarizer με schema bug — ΜΗΝ τον χρησιμοποιήσεις.
Το σωστό key είναι metrics[*]['MAE'], ΟΧΙ top-level 'mae'."""
import glob
import json

for p in glob.glob("runs/overnight_20260705/a_cadence/*.json"):
    d = json.load(open(p, encoding="utf-8"))
    print(p, d.get("mae", "NO-MAE"))  # BUG εδώ
```

Dispatch reviewer με prompt:

```
MODE: TOOLING
TARGET: tests/fixtures/qa/bad_summarizer.py
Πρόκειται να χρησιμοποιηθεί για σύνοψη των Block A runs. Κάνε TOOLING review.
```

Expected: BLOCKER/MAJOR finding «διαβάζει d['mae'] ενώ τα πραγματικά JSONs έχουν
metrics[*].MAE» με EVIDENCE από πραγματικό run JSON · VERDICT: BLOCK.

- [ ] **Step 5: Fixture 3 — freeze-χαλάρωμα.** `tests/fixtures/qa/freeze_relax.diff`
(συνθετικό diff — by design δεν αντιστοιχεί σε πραγματικές γραμμές):

```diff
--- a/src/recursive_openloop.py
+++ b/src/recursive_openloop.py
@@ -120,7 +120,7 @@ def build_lag_features(history, cutoff):
-    lag_source = history.loc[:cutoff]  # AEL: freeze-at-cutoff
+    lag_source = history  # perf: skip the cutoff slice copy
     lags = make_lags(lag_source, LAG_SET)
```

Dispatch reviewer με prompt:

```
MODE: CORE-DIFF
TARGET: tests/fixtures/qa/freeze_relax.diff (συνθετικό fixture — κρίνε το ΠΕΡΙΕΧΟΜΕΝΟ του diff)
Προτεινόμενη perf αλλαγή στο recursive_openloop.py. Κάνε CORE-DIFF review.
```

Expected: BLOCKER (καταργεί το freeze-at-cutoff ⇒ AEL leakage) ·
REQUIRED FOLLOW-UP: poisoning · VERDICT: BLOCK.

- [ ] **Step 6: Σενάριο 4 — runbook Q&A.** Άνοιξε το
`.claude/skills/triaging-run-failures/SKILL.md` και έλεγξε ότι για το σύμπτωμα
«FileNotFoundError από load_processed» η απάντηση του runbook είναι «BY DESIGN /
φτιάξε το parquet / ΜΗΝ διορθώσεις το fallback» — ΟΧΙ πρόταση αλλαγής στο
`split_utils.py`.

- [ ] **Step 7: Κατέγραψε τα 4 αποτελέσματα** σε `reports/qa/ACCEPTANCE_20260708.md`
(πίνακας: σενάριο → verdict/εύρημα → PASS/FAIL). Αν κάποιο FAIL: διόρθωσε το αντίστοιχο
component (agent prompt/skill/linter), ξανά το σενάριο, ΜΕΤΑ προχώρα.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/qa/ reports/qa/ACCEPTANCE_20260708.md
git commit -m "test(qa): acceptance scenarios 1-4 (Block D, NO-MAE, freeze-relax, runbook) PASS"
```

---

### Task 9: Profiling baseline + acceptance σενάριο 5 — ΘΕΛΕΙ ΧΡΗΣΤΗ (conda runs)

**Files:**
- Create: `reports/qa/PROFILE_BASELINE.md`
- Create (από τα runs): `runs/qa_smoke/direct_smoke_base.json`, `runs/qa_smoke/direct_smoke_rerun.json`, `reports/qa/profile_direct_smoke.prof`

Τα βήματα 1-2 είναι **USER-RUN** (online protocol: δώσε copy-paste block, ο χρήστης
επικολλά πίσω το τέλος του output). ΕΝΑ conda process — σειριακά.

- [ ] **Step 1 (USER-RUN): baseline profile + smoke run**

```bash
mkdir -p runs/qa_smoke reports/qa
conda run -n epf --no-capture-output python -X utf8 -m cProfile -o reports/qa/profile_direct_smoke.prof -m src.master_forecast --algo lgbm --task price --market dam --strategy direct --gate strict --retrain static --features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00" --out_json runs/qa_smoke/direct_smoke_base.json
```

- [ ] **Step 2 (USER-RUN): identical re-run (για το equivalence PASS)**

```bash
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy direct --gate strict --retrain static --features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00" --out_json runs/qa_smoke/direct_smoke_rerun.json
```

- [ ] **Step 3: Acceptance 5α — PASS case**

Run: `python -X utf8 scripts/qa/compare_runs.py --a runs/qa_smoke/direct_smoke_base.json --b runs/qa_smoke/direct_smoke_rerun.json`
Expected: `EQUIVALENT`, exit 0. (Αν PRED-DIFF από LGBM nondeterminism: κατέγραψε το
max|diff| στο PROFILE_BASELINE.md και όρισε τεκμηριωμένο `--tol` — απόφαση γραπτή.)

- [ ] **Step 4: Acceptance 5β — FAIL case (πειραγμένο αντίγραφο)**

Run: `python -X utf8 -c "import json;d=json.load(open(r'runs/qa_smoke/direct_smoke_base.json',encoding='utf-8'));k=list(d['series'])[0];d['series'][k][0]+=0.5;json.dump(d,open(r'runs/qa_smoke/direct_smoke_tampered.json','w',encoding='utf-8'))"`
μετά: `python -X utf8 scripts/qa/compare_runs.py --a runs/qa_smoke/direct_smoke_base.json --b runs/qa_smoke/direct_smoke_tampered.json`
Expected: `NOT EQUIVALENT` με `PRED-DIFF`, exit 1.

- [ ] **Step 5: Ανάλυση profile + baseline report**

Run: `python -X utf8 -c "import pstats;pstats.Stats(r'reports/qa/profile_direct_smoke.prof').sort_stats('cumulative').print_stats(25)"`
Γράψε `reports/qa/PROFILE_BASELINE.md`: ημερομηνία, smoke config (ακριβής εντολή),
wall-clock, top-10 cumulative functions, πρώτες υποψίες hotspot (ΧΩΡΙΣ αλλαγές κώδικα
σε αυτό το plan — μόνο καταγραφή), αποτελέσματα acceptance 5α/5β.

- [ ] **Step 6: Commit**

```bash
git add reports/qa/PROFILE_BASELINE.md runs/qa_smoke/direct_smoke_base.json runs/qa_smoke/direct_smoke_rerun.json
git commit -m "feat(qa): direct smoke profiling baseline + equivalence acceptance (scenario 5)"
```

---

### Task 10: Plugin resync `epf-ops` v0.2.0

**Files:**
- Modify: `plugins/epf-ops/.claude-plugin/plugin.json` (version 0.2.0 + description)
- Modify: `plugins/epf-ops/README.md` (Components table + resync block)
- Modify: `plugins/epf-ops/hooks/hooks.json` (nudge hooks)
- Create (copies): `plugins/epf-ops/agents/epf-code-reviewer.md`,
  `plugins/epf-ops/skills/triaging-run-failures/SKILL.md`,
  `plugins/epf-ops/skills/optimizing-training-runs/SKILL.md`,
  `plugins/epf-ops/hooks/scripts/qa_nudges.py`, `plugins/epf-ops/hooks/scripts/guard_edits.py` (refresh)

- [ ] **Step 1: Αντιγραφές (PowerShell, από repo root — pattern του README)**

```powershell
Copy-Item .claude/agents/epf-code-reviewer.md plugins/epf-ops/agents/epf-code-reviewer.md
New-Item -ItemType Directory -Force plugins/epf-ops/skills/triaging-run-failures, plugins/epf-ops/skills/optimizing-training-runs | Out-Null
Copy-Item .claude/skills/triaging-run-failures/SKILL.md plugins/epf-ops/skills/triaging-run-failures/SKILL.md
Copy-Item .claude/skills/optimizing-training-runs/SKILL.md plugins/epf-ops/skills/optimizing-training-runs/SKILL.md
Copy-Item scripts/claude_hooks/qa_nudges.py plugins/epf-ops/hooks/scripts/qa_nudges.py
Copy-Item scripts/claude_hooks/guard_edits.py plugins/epf-ops/hooks/scripts/guard_edits.py
```

- [ ] **Step 2: `plugin.json`** — άλλαξε `"version": "0.1.2"` σε `"version": "0.2.0"`
και στο description το «Επεκτάσιμο με QA pack (debug/code-review/testing) χωρίς
redesign» σε «+ QA pack v2 (epf-code-reviewer agent, triaging-run-failures,
optimizing-training-runs, qa nudge hooks)».

- [ ] **Step 3: `hooks/hooks.json`** — πρόσθεσε τα qa_nudges entries (PreToolUse Bash
+ PostToolUse Edit|Write|NotebookEdit) με command
`python -X utf8 ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/qa_nudges.py`, στο ίδιο format με
το υπάρχον guard entry (διάβασε πρώτα το αρχείο και ακολούθησε τη δομή του 1:1).

- [ ] **Step 4: README** — στο Components table πρόσθεσε 3 γραμμές (agent
`epf-code-reviewer`, skills `triaging-run-failures`/`optimizing-training-runs`), στο
resync block τις 4 νέες Copy-Item γραμμές του Step 1, και άλλαξε την ενότητα
«Extension path — QA pack (στόχος #2, v0.2.0)» σε «✅ Υλοποιήθηκε (v0.2.0,
2026-07-08) — spec: docs/superpowers/specs/2026-07-07-qa-pack-code-review-debug-design.md»
με τα τελικά ονόματα (ΟΧΙ τα παλιά qa-debug/qa-code-review).

- [ ] **Step 5: Repackage**

```powershell
Compress-Archive -Path plugins/epf-ops/* -DestinationPath "$env:TEMP/epf-ops.zip" -Force
Rename-Item "$env:TEMP/epf-ops.zip" epf-ops.plugin -Force
```

- [ ] **Step 6: Commit**

```bash
git add plugins/epf-ops/
git commit -m "feat(qa): resync epf-ops plugin v0.2.0 with QA pack components"
```

---

### Task 11: CodSpeed φάση 2 (connector συνδεδεμένος)

**Files:** ό,τι υπαγορεύσει το `codspeed:codspeed-setup-harness` skill (benchmarks + CI).

- [ ] **Step 1: Επιβεβαίωσε πρόσβαση** — ToolSearch για τα CodSpeed MCP tools και κάλεσε
`list_repositories`. Αν λείπουν/αποτύχει: σταμάτα το task, ανάφερε ότι θέλει
(ξανά)σύνδεση connector από τον χρήστη — ΜΗΝ μπλοκάρει τα Tasks 1-10.

- [ ] **Step 2: Invoke `codspeed:codspeed-setup-harness`** με στόχους ΜΟΝΟ pure-logic
(τρέχουν χωρίς data/conda-training): (α) `feature_availability` filtering στο gate,
(β) dense lag construction, (γ) `split_utils` splits. Τα benchmarks πάνε σε
`tests/benchmarks/`, CI μέσω GitHub Actions στο υπάρχον remote (ΠΟΤΕ το branch
`FEB272026_localhistory`).

- [ ] **Step 3: Commit ό,τι δημιουργηθεί** με `feat(qa): codspeed micro-benchmarks (phase 2)`
και σημείωσε στο `reports/qa/PROFILE_BASELINE.md` το link του πρώτου CodSpeed run.

---

## Self-review του plan

- **Spec coverage**: §0-§12 του spec → Tasks: linter §3→T1 · compare_runs §4α→T2 ·
  agent §2→T3 · debug skill §4→T4 · optimize skill §4α→T5 · nudges §5→T6 · SOP §6→T7 ·
  acceptance §9(1-4)→T8, §9(5)→T9 · resync §10.6→T10 · CodSpeed §10.10→T11 ·
  artifact rule §6→καλύπτεται από SOP κείμενο (T7) + reports/qa (T8/T9). ✔
- **Placeholders**: κανένα TBD· όλα τα code steps έχουν πλήρη κώδικα. ✔
- **Type consistency**: `scan()`/`compare()`/`decide()`/`render()` ονόματα συνεπή
  μεταξύ tests και υλοποιήσεων· `seed_sweep_static`/`train_end_expanding` κ.λπ. ids
  ίδια σε tests/κώδικα/fixtures. ✔
