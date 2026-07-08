# epf-ops — Operating & Compounding Layer Plugin

Plugin για το project **epf_greece_starter** (διπλωματική: leakage-free GR energy
forecasting, DAM/IDM/Forward × price/load/generation). Πακετάρει το στρώμα λειτουργίας
πάνω από τον ντετερμινιστικό πυρήνα, ώστε τα SOPs, ο validity reviewer και ο
single-writer guard να ταξιδεύουν μαζί σε κάθε session (Claude Code ή Cowork).

Βασίζεται σε: `DESIGN.txt` (hybrid ✅ — deterministic core + λίγοι read-only agents) και
`docs/superpowers/specs/2026-07-05-operating-compounding-layer-design.md` (αποφάσεις §7 κλειδωμένες).

## Components

| Component | Όνομα | Ρόλος |
|---|---|---|
| Skill | `energy-forecast` | WHAT + HOW: domain rules, feature groups, εντολές pipeline **+ Operating Protocol** (πρώην energy-runner, merged) |
| Skill | `ingest-audit` | Διαβατήριο νέας πηγής/feature: gate timing → lagscan → availability rule → πληρότητα parquet |
| Skill | `synthesize-ablation` | Run JSONs → ΔMAE πίνακες → §2 pre-gate → verdicts (ACCEPTED/PENDING/MIXED) |
| Skill | `feature-eng` | FeatureENG agent (data_in extension): design με TDD pre-registration (T1-T8) → ingest-audit → υλοποίηση → batch → verdict → deploy checklist + validator script |
| Skill | `cycle-ops` | Ο κανονικός κύκλος (market, task), σκάλα Rungs 1-5, deposit rules, extension contract |
| Agent | `validity-reviewer` | Read-only αυστηρός reviewer — ACCEPT/PENDING/REJECT με hard rules πριν από κάθε ΔΕΚΤΟ (claims) |
| Agent | `epf-code-reviewer` | Read-only code reviewer — PRE-RUN/CORE-DIFF/TOOLING, BLOCK/APPROVE πριν από run/commit (κώδικας ≠ claims) |
| Skill | `triaging-run-failures` | Runbook όταν run/script ΣΚΑΕΙ ή παραξενεύει (crashes, Δ=0.000, NO-MAE, encoding) + escalation + deposit |
| Skill | `optimizing-training-runs` | Επιτάχυνση runs (direct ablations) με equivalence gate (compare_runs + anchor)· CodSpeed φάση 2 |
| Hook | PreToolUse guard | Αντίγραφο του `guard_edits.py`: DENY σε `data/raw|processed`, `OLD/`· ASK σε leakage-sensitive src |
| Hook | QA nudges | `qa_nudges.py`: μη-μπλοκάρουσες υπενθυμίσεις (core-edit → CORE-DIFF review· run-launch → pre-run linter) |
| MCP | — | Κανένα (απόφαση DESIGN.txt §9 — όχι στη research φάση) |

## Setup

- Απαιτεί το repo `epf_greece_starter` τοπικά και conda env `epf`. Όλες οι εντολές
  τρέχουν από το project root.
- `ENTSOE_API_KEY` env var μόνο για τα fetchers (όχι για training/eval).
- Αν το session τρέχει ΜΕΣΑ στο repo, ο repo hook (`.claude/settings.json`) και ο plugin
  hook συνυπάρχουν — ίδια απόφαση, κανένα conflict. Ο plugin hook προσθέτει προστασία σε
  sessions ΕΚΤΟΣ repo root (το κενό που εντοπίστηκε στο brainstorming 2026-07-05).

## Usage

- «τρέξε το Q1 backtest» / «MAE/gate/leakage/retrain» → `energy-forecast`
- «να δοκιμάσουμε νέο feature/πηγή X» → `ingest-audit` (πριν γραφτεί κώδικας)
- «νέο feature end-to-end: σχεδίασε → τέσταρε → deploy» → `feature-eng` (orchestrator· καλεί ingest-audit + synthesize-ablation ως στάδια)
- «σύνοψη/verdict από το batch» → `synthesize-ablation` (→ `validity-reviewer` πριν από ACCEPTED)
- «νέος κύκλος / cycle runner / plan.yaml / αυτοματοποίηση» → `cycle-ops`

## Συντήρηση — source of truth

Τα ζωντανά skills μένουν στο repo (`.claude/skills/`)· το plugin είναι snapshot που
ανασυσκευάζεται. Μετά από αλλαγή στα repo skills:

```powershell
# από το repo root — sync + repackage
Copy-Item .claude/skills/ingest-audit/SKILL.md plugins/epf-ops/skills/ingest-audit/SKILL.md
Copy-Item .claude/skills/synthesize-ablation/SKILL.md plugins/epf-ops/skills/synthesize-ablation/SKILL.md
Copy-Item -Recurse -Force .claude/skills/feature-eng plugins/epf-ops/skills/
Copy-Item .claude/skills/energy-forecast/scripts/*.py plugins/epf-ops/skills/energy-forecast/scripts/
Copy-Item scripts/claude_hooks/guard_edits.py plugins/epf-ops/hooks/scripts/guard_edits.py
# QA pack v2 (2026-07-08):
Copy-Item .claude/agents/epf-code-reviewer.md plugins/epf-ops/agents/epf-code-reviewer.md
Copy-Item .claude/skills/triaging-run-failures/SKILL.md plugins/epf-ops/skills/triaging-run-failures/SKILL.md
Copy-Item .claude/skills/optimizing-training-runs/SKILL.md plugins/epf-ops/skills/optimizing-training-runs/SKILL.md
Copy-Item scripts/claude_hooks/qa_nudges.py plugins/epf-ops/hooks/scripts/qa_nudges.py
Compress-Archive -Path plugins/epf-ops/* -DestinationPath "$env:TEMP/epf-ops.zip" -Force
Rename-Item "$env:TEMP/epf-ops.zip" epf-ops.plugin -Force
```

(Το `energy-forecast/SKILL.md` του plugin είναι η ΣΥΓΧΩΝΕΥΜΕΝΗ εκδοχή forecast+runner —
αν αλλάξει το repo skill, μετέφερε την αλλαγή χειροκίνητα και κράτα την ενότητα
«Operating Protocol».)

## QA pack — ✅ Υλοποιήθηκε (v0.2.0, 2026-07-08)

Spec: `docs/superpowers/specs/2026-07-07-qa-pack-code-review-debug-design.md` ·
plan: `docs/superpowers/plans/2026-07-08-qa-pack-implementation.md`.
Ο «υπεύθυνος κώδικα» — review/debug/optimize — χτισμένος γύρω από τον frozen πυρήνα,
σεβόμενος guard hook + validity-reviewer + «ΕΝΑ conda process»:

- **Agent `epf-code-reviewer`** (read-only): PRE-RUN (linter + purpose-fit), CORE-DIFF
  (leakage-sensitive diffs + poisoning follow-up), TOOLING (schema/encoding).
- **Skill `triaging-run-failures`**: runbook γνωστών failure modes + escalation + deposit.
- **Skill `optimizing-training-runs`**: equivalence gate (compare_runs + anchor)· CodSpeed φάση 2.
- **Hooks `qa_nudges.py`**: μη-μπλοκάρουσες υπενθυμίσεις (PostToolUse core-edit, PreToolUse Bash run).
- **Scripts στο repo** (`scripts/qa/check_run_config.py`, `compare_runs.py`): stdlib, system
  python — τρέχουν από το repo root· δεν πακετάρονται (το plugin απαιτεί ούτως ή άλλως το repo).

## Customization

Συμβατό με το `cowork-plugin-customizer` skill (προσθήκη/αφαίρεση skills, αλλαγή
triggers). Δεν χρησιμοποιεί `~~connector` placeholders — είναι project-specific
(single-user thesis project), όχι για εξωτερική διανομή.
