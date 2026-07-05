---
name: cycle-ops
description: Operating & compounding layer του epf_greece_starter — χάρτης του κανονικού κύκλου (market, task), σκάλα αυτοματοποίησης Rungs 1-5, μηχανισμοί knowledge-deposit και κανόνες frozen core. Χρησιμοποίησέ το όταν ξεκινά νέος κύκλος πρόβλεψης, όταν αναφέρονται "cycle runner"/"plan.yaml"/"intervention cost"/"Rung", όταν σχεδιάζεται αυτοματοποίηση σταδίων, ή όταν πρέπει να αποφασιστεί ΠΟΥ κατατίθεται ένα εύρημα (registry/memory/skill) και τι επιτρέπεται να αυτοματοποιηθεί.
metadata:
  version: "0.1.0"
  spec: "docs/superpowers/specs/2026-07-05-operating-compounding-layer-design.md"
---

# Cycle Ops — το στρώμα λειτουργίας πάνω από τον πυρήνα

Πηγή αλήθειας: `docs/superpowers/specs/2026-07-05-operating-compounding-layer-design.md`
(αποφάσεις §7 κλειδωμένες 2026-07-05). Διάβασέ το πριν από κάθε αρχιτεκτονική απόφαση
στο operating layer. Αυτό το skill είναι η συμπυκνωμένη, εκτελέσιμη εκδοχή του.

## North star

Compounding automation: κάθε κύκλος `(market ∈ {dam,idm,forward}, task ∈ {price,load,generation})`
καταθέτει reusable assets που μειώνουν τον επόμενο. Μετρική: `intervention_cost` ανά κύκλο
(χειροκίνητες αποφάσεις + εντολές), στόχος πτώση C1→C2→C3. Νέα αγορά = νέο plan + gate-spec +
data adapter — ΠΟΤΕ νέα αρχιτεκτονική.

## Frozen core (μη διαπραγματεύσιμο — ΠΡΩΤΑ αυτό)

Engine · AEL/gates · §2 semantics · single-writer master MDs (guard_edits hook) · ΕΝΑ conda
process · artifacts-ως-interface · σκληροί κανόνες energy-forecast (no Optuna, strict gate
default). Κάθε αυτοματοποίηση χτίζεται ΓΥΡΩ από αυτά, ποτέ μέσα. Agents μόνο όπου η κρίση
ΕΙΝΑΙ το προϊόν, και πάντα read-only προς το state.

## Ο κανονικός κύκλος — στάδια & ποιος τα τρέχει

| # | Στάδιο | Εργαλείο | Κόστος σήμερα |
|---|---|---|---|
| 0 | Cycle spec | απόφαση market/task/date/config/features | 🔴 άνθρωπος |
| 1 | Ingest | `fetch_entsoe_*`, `build_real_load_forecast` | 🔴 |
| 2 | Rebuild | `python -m src.data --task ...` + backup + compare | 🔴 |
| 3 | Validate | `preflight_check.py [--poison]`, lagscan, TZ guards | 🟢 script |
| 4 | Forecast | `master_forecast` / `run_master_grid` | 🟢 script |
| 5 | Gate/Accept | §2 validity gate (\|ΔMAE\|>0.15 & ≥2 συνθήκες) | 🔴 κρίση ΜΕΝΕΙ ανθρώπινη |
| 6 | Uncertainty | `conformal.py` (προαιρετικό) | 🟡 |
| 7 | Report | `make_ablation_report`, charts, `build_run_ledger.py` | 🟡 |
| 8 | Deposit | last.md, memory, git commit+push | 🔴 |
| 9 | Settle | forecast vs realized, alerts | ⬜ φάση προϊόντος |

Τα «άπειρα βήματα» ζουν στα 0/5/8 (κρίση & κατάθεση) — στόχοι: (α) collapse των μηχανικών
σταδίων 1-4/6-7 σε 1 εντολή, (β) capture των αποφάσεων σε reusable assets.

## Σκάλα αυτοματοποίησης (build order — ΟΛΗ διαδοχικά, από Rung 1)

1. **Cycle runner** (ΤΡΕΧΟΝ): μία εντολή preflight→run→evaluate→report→registry από
   `plan.yaml` ανά (market,task). Python orchestrator, ίδιο conda env, testable, TDD.
2. **Feature Registry + run-skill-generator**: machine-readable YAML/JSON + thin skill·
   auto-draft ΜΟΝΟ σε `staging/` + promotion review από χρήστη (απόφαση C — καμία σιωπηλή εγγραφή).
3. **results-auditor (read-only agent) + consolidate-memory** (trigger: memory > budget γραμμών).
4. **Gate-assist**: script προ-υπολογίζει τον §2 πίνακα, ο άνθρωπος πατά accept.
5. **Daily runner (cron) + settle + alerts** — φάση προϊόντος, εκτός spec.

Artefacts: specs σε `docs/superpowers/specs/`, plans σε `docs/superpowers/plans/`.
Πριν υλοποιήσεις Rung: γράψε plan file (plan-cycle-runner κ.λπ.), διάβασε τα ακριβή
interfaces (`src/master_forecast.py`, `src/data.py`, `preflight_check.py`,
`scripts/build_run_ledger.py`), μετά TDD.

## Κανόνες κατάθεσης (knowledge-deposit)

- Validated feature → εγγραφή στο Feature Registry (availability rule, lagscan evidence,
  gate-timing proof, verdict, reproduction cmd).
- Απόφαση/προτίμηση χρήστη → memory (με Why/How-to-apply).
- Εύρημα που περνά §2 σε ≥2 windows → draft σε `staging/` → promotion review → registry+memory.
- Σύνθεση/γενίκευση → `plan.yaml` ανά (market,task).
- Κάθε νούμερο που κατατίθεται: JSON/CSV path + εντολή αναπαραγωγής (αλλιώς δεν κατατίθεται).

## Extension contract — QA pack (μελλοντικός στόχος #2, ΜΗΝ υλοποιηθεί πρόωρα)

Το plugin επεκτείνεται ΧΩΡΙΣ redesign: νέα QA skills μπαίνουν ως αδελφοί φάκελοι
`skills/qa-<όνομα>/SKILL.md` (π.χ. qa-debug, qa-code-review, qa-testing-strategy, qa-init)
και οφείλουν να σέβονται:

1. **Frozen core**: κανένα QA skill δεν τροποποιεί engine/gates/AEL/§2 semantics.
2. **Guard hook**: αλλαγές σε leakage-sensitive src (data.py, feature_availability.py,
   master_forecast.py, recursive_openloop.py, conformal.py, scheduled_sampling.py,
   check_crosslag_fairness.py) → ρητή έγκριση + poisoning tests + control run + anchor ±0.05.
3. **Read-only πρώτα**: κάθε review/audit βγάζει ευρήματα με file:line + πρόταση —
   η εφαρμογή fix είναι ξεχωριστό, εγκεκριμένο βήμα.
4. **validity-reviewer**: κάθε QA εύρημα που αγγίζει αποτελέσματα/claims περνά από τον
   agent πριν γραφτεί σε master MDs.
5. Versioning: προσθήκη QA pack = minor bump (0.1.x → 0.2.0) στο plugin.json.

## Λειτουργικά reminders

- Χρήστης online → copy-paste εντολές για το δικό του terminal· long runs (>2-3 λεπτά)
  πάντα OS-detached (`Start-Process`, ASCII-only args, `-WorkingDirectory`), ΠΟΤΕ polling
  loop μέσα στο session.
- Κάθε απάντηση με ανοιχτή δουλειά τελειώνει με `DONE / RESULT PATHS / PENDING / NEXT`.
- intervention_cost: μέτρα (χονδρικά) πόσες χειροκίνητες αποφάσεις+εντολές χρειάστηκε ο
  κύκλος και κατέγραψέ το στο run ledger — αυτό είναι το KPI του στρώματος.
