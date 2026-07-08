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
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import yaml

VALID_MARKETS = {"dam", "idm", "forward"}
VALID_TASKS = {"price", "load"}
VALID_STRATEGIES = {"recursive", "direct"}   # tf/seq2seq ΟΧΙ tradeable → εκτός plan
VALID_RETRAIN = {"static", "monthly", "weekly"}
VALID_GATES = {"strict", "academic"}
VALID_ALGOS = {"lgbm", "xgb", "mlp", "lstm", "lear"}

_REQUIRED = ("study", "market", "task", "strategy", "retrain", "gate", "algos", "windows", "specs")

PREFLIGHT = ".claude/skills/energy-forecast/scripts/preflight_check.py"
SYNTH = "scripts/synthesize_ablation.py"
LEDGER = "scripts/build_run_ledger.py"


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


def plan_summary(plan, cells):
    lines = [f"STUDY={plan['study']}  market={plan['market']} task={plan['task']} "
             f"strategy={plan['strategy']} retrain={plan['retrain']} gate={plan['gate']}",
             f"{len(cells)} cells:"]
    for c in cells:
        lines.append(f"  - {c.json_name()}")
    return "\n".join(lines)


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

    return run_all(plan, cells, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
