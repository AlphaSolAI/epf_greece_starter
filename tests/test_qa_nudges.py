"""Tests για τα QA nudge hooks (spec §5) — ποτέ deny, fail-open."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "claude_hooks"))
import qa_nudges as qn

REPO_ROOT = Path(__file__).resolve().parents[1]


def post_edit(path):
    return {"hook_event_name": "PostToolUse", "tool_name": "Edit",
            "tool_input": {"file_path": path}, "tool_response": {}}


def pre_bash(cmd):
    return {"hook_event_name": "PreToolUse", "tool_name": "Bash",
            "tool_input": {"command": cmd}}


def test_core_edit_gets_nudge(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    msg = qn.decide(post_edit(str(REPO_ROOT / "src" / "data.py")))
    assert msg and "CORE-DIFF" in msg and "poisoning" in msg


def test_non_core_edit_no_nudge(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert qn.decide(post_edit(str(REPO_ROOT / "scripts" / "qa" / "x.py"))) is None


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
