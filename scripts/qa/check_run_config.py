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
