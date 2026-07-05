# Plan: Gate-Assist MCP layer (Rung 4, MCP-fronted)

**Status:** Draft idea, not approved · **Date:** 2026-07-05
**Parent spec:** [docs/superpowers/specs/2026-07-05-operating-compounding-layer-design.md](../specs/2026-07-05-operating-compounding-layer-design.md) (§5, Rung 4 — Gate-assist)
**Relation to roadmap:** slots into Rung 4 (after Rung 1 cycle-runner exists), NOT a replacement for it.

## Context

Overnight run 2026-07-05 (stages a_cadence→f_conformal) finished at 13:34:56 — checked directly
via `results/run_ledger.csv` mtimes and `logs/overnight_20260705_master.log` tail, no tooling needed.

Requested capability: an MCP server (built with `anthropic-skills:mcp-builder`) that exposes the
operating-layer's check → validate/analyze → gate stages as callable tools, and branches
DEBUG-agent vs MASTER-agent based on gate result — collapsing the manual "re-read run_ledger +
gate table + preflight logs before every new cycle" work into one call.

This is Rung 4 (**Gate-assist**) of the already-locked cycle-ops roadmap, fronted by MCP instead of
a bare script. It reuses stage semantics already defined in the parent spec's cycle map (stages
3/5/7): Validate, Gate/Accept, Report.

## Components

1. **`check_run_status(batch_id)`** — reads `results/run_ledger.csv` + `runs/<batch_id>/*` mtimes,
   returns which of the 9 cycle stages have completed and their output paths. Read-only.
2. **`run_preflight(task)`** — thin wrapper around `preflight_check.py` [--poison],
   `check_crosslag_fairness.py` (lagscan/TZ guards). Returns pass/fail + evidence paths.
3. **`synthesize_results(batch_id)`** — wraps the `synthesize-ablation` skill: builds ΔMAE tables
   from `runs/<batch_id>/*.json`.
4. **`evaluate_gate(batch_id)`** — computes the §2 validity gate (`|ΔMAE|>0.15`, ≥2 independent
   conditions, same sign across independent windows) from the synthesized table. Returns
   structured `{pass: bool, reasons: [...], evidence_paths: [...]}`. This is a direct port of
   existing gate math — no new gate logic, no re-derivation with generic heuristics.
5. **DEBUG branch** — triggered when `evaluate_gate` returns `pass: false`. Read-only diagnostic
   agent (per cycle-ops "read-only first" rule): reports which condition failed and points at the
   evidence file — does not edit code or config.
6. **MASTER branch** — triggered when `evaluate_gate` returns `pass: true`. Prepares the next run:
   assembles the OS-detached launch command (`Start-Process`, ASCII args, correct
   `-WorkingDirectory`, single conda env) and **hands it to the user to fire** — does not launch
   autonomously.

## Explicit non-goal / constraint carried over from parent spec

Stage 5 (Gate/Accept) stays a human judgment call (parent spec §2, §6: "κρίση ΜΕΝΕΙ ανθρώπινη").
`evaluate_gate` computes the table; a human still accepts. The MASTER branch prepares/stages a run,
it does not auto-launch one — auto-launch would (a) remove the accept checkpoint the parent spec
locks, and (b) risk violating the "ΕΝΑ conda process" rule if a prior run's artifacts aren't fully
settled when a new one is staged.

## Options considered

**A. MCP wraps existing scripts (thin adapter over `preflight_check.py`, gate math, `run_ledger.csv`).**
Complexity low, no new gate logic, no risk to frozen core (read-only). **Recommended.**

**B. Reimplement validate/analyze with generic `data:validate-data`/`data:analyze` skills.**
Duplicates logic already proven in this project's own gate (poisoning tests, leakage checks,
independent-window requirement) with generic heuristics unaware of these constraints — creates two
sources of truth for the same accept/reject decision. Rejected.

## Build order

1. Confirm Rung 1 (cycle-runner, `plan.yaml`) exists or stub `check_run_status` /
   `run_preflight` directly against current script paths in the interim.
2. Implement `evaluate_gate` as a direct port of the §2 gate table logic (no reinterpretation).
3. Build MCP server via `anthropic-skills:mcp-builder` wrapping the four read tools.
4. Add DEBUG-branch agent (read-only) and MASTER-branch command-assembly (staged, not launched).
5. `validity-reviewer` review before any gate-table output is treated as a paper-bound claim.

## Open decisions (need user sign-off before implementation)

- [ ] Proceed as Rung 4 after Rung 1 exists, or pull forward as a standalone MCP shim now?
- [ ] DEBUG agent scope: diagnosis-only (recommended) vs. allowed to propose (not apply) fixes?
- [ ] MCP server hosting: local stdio server invoked from Claude Code only, or broader?

## Explicitly out of scope for this plan

Rung 1 cycle-runner implementation, Rung 2 feature registry, Rung 3 results-auditor/memory
consolidation, Rung 5 daily runner/settle/alerts — all remain as sequenced in the parent spec.
