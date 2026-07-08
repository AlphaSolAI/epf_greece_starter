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
