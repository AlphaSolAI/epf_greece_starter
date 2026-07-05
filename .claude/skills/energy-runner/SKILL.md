---
name: energy-runner
description: Code-first runner for the GR energy forecasting project. Use only when the user wants to continue experiments, run scripts, parse results, or update current experiment docs. Do not use for literature review, thesis prose, or broad redesign.
disable-model-invocation: true
---

# Energy Runner — Claude Code operating skill

## Scope

This skill is for running the current GR energy forecasting / trading-agent work. It is **code-first**.

Use it to:
- identify the next allowed experiment step;
- run or prepare exact commands;
- parse outputs from `runs/`, `results/`, `logs/`;
- update current experiment docs only when the result passes validity checks.

Do **not** use it for:
- literature review;
- thesis/paper prose;
- new connectors/plugins research;
- folder reorganization;
- broad architecture redesign;
- speculative new features unless the current checklist explicitly asks for them.

## Files to read first

Read these before acting:

1. `last.md` (περιέχει το συγχωνευμένο VALIDITY GATE στο §2 — το `VALIDITY_CHECKLIST.md` είναι πλέον pointer stub)
2. `ABLATION_PLAN.md`

Read these only if needed for implementation details:

4. `MASTER_PIPELINE_DESIGN.md`
5. `SYSTEM_DESIGN_TRADING_AGENT.md`
6. `.claude/skills/energy-forecast/SKILL.md`

## Current operating state

Treat the current project state as:

`TZFIX done -> AEL/freeze done -> poisoning tests done -> leak-free B1/B3 clean experiments -> new headline -> conformal later`

The old `15.17 €/MWh` headline is suspended. Never use it as a current result.

The immediate coding priority is to continue the clean leak-free pipeline. If the docs say B1 is incomplete, finish B1. If B1 is complete, proceed to B3 clean re-ablation.

## Interaction rules with the user

When the user is online:

1. Do **not** start by doing a long explanation.
2. Give a short state summary in at most 5 bullets.
3. Give exactly one copy-paste command block.
4. Tell the user exactly what output to paste back: RESULT line, JSON path, CSV path, or error.
5. Use one conda process at a time.

When the user explicitly says “τρέχα εσύ”, “run it”, or uses `/goal` and permissions allow:

1. Run the commands yourself.
2. Keep working until the stated checkpoint is reached or a real blocker occurs.
3. If blocked, report the exact failing command, exact error, and next smallest fix.

## Command rules

Always use:

```bash
conda run -n epf --no-capture-output python -X utf8
```

Never use multi-line Python with `python -c`. If multi-line Python is needed, write a temporary script under `scripts/` and run it.

Use output locations consistently:

- experiment folders: `runs/<study_id>/`
- CSV summaries: `results/`
- logs: `logs/`
- reports/figures: `reports/`

Always prefer explicit `--out_json` when the script supports it.

For static experiments, `--train_end` is allowed. Do not use `--train_end` for monthly/weekly unless the existing script explicitly supports that design.

## Validity gate before any conclusion

Before writing a result as accepted, check:

1. strict gate / correct market / correct task;
2. TZFIX guards still pass if relevant;
3. AEL / crosslag freeze is active;
4. poisoning tests pass if the code path changed;
5. compared methods use the same window, same data snapshot, same gate;
6. no old suspended number is mixed with new clean results;
7. result has JSON/CSV path and reproducible command;
8. acceptance rule: `|ΔMAE| > 0.15` and same sign in at least two independent conditions.

If these do not hold, label the result `PENDING` or `SMOKE`, not accepted.

## Minimal output format

Every response during execution must end with:

```text
DONE:
- ...

RESULT PATHS:
- ...

PENDING:
- ...

NEXT:
<one exact command or one exact user action>
```

## Failure behavior

If a command fails:

1. Do not redesign the project.
2. Do not jump to literature/thesis work.
3. Inspect the smallest relevant file/log.
4. Fix only the minimal issue if safe.
5. Re-run only the failed command or a smaller smoke test.
6. If not fixable quickly, stop with exact error and exact next command.
