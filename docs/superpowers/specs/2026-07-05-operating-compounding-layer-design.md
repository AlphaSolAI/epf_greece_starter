# Operating & Compounding Layer — Design Spec (v0)

> Στόχος: να οριστεί το **στρώμα λειτουργίας** πάνω από τον ντετερμινιστικό πυρήνα του
> epf_greece_starter, ώστε **κάθε επόμενος κύκλος (market, task) να απαιτεί λιγότερη
> ανθρώπινη παρέμβαση** — με πλήρη διατήρηση εγκυρότητας. Δεν αλλάζει τον πυρήνα· χτίζει γύρω του.

**Ημερομηνία:** 2026-07-05 · **Κατάσταση:** approved-in-principle (αποφάσεις §7 κλειδωμένες).

## 0. North star & μετρική

North star: *compounding automation* — κάθε κύκλος καταθέτει reusable assets (skills, memory,
scripts, registry) που μειώνουν τον επόμενο. Μετρική: `intervention_cost` ανά κύκλο (χειροκίνητες
αποφάσεις + εντολές), στόχος πτώση C1→C2→C3, καταγραφή στο run-registry.
Γενίκευση: `(market ∈ {dam,idm,forward}, task ∈ {price,load,generation})` = παράμετρος.
DAM/price = πρώτο instance. Νέα αγορά = νέο plan + gate-spec + data adapter, ποτέ νέα αρχιτεκτονική.

## 1. Τα 4 στρώματα

1. **Validity core** (engine/gates/AEL/preflight/poisoning) — FROZEN, δεν πειράζεται.
2. **Operating layer** (στάδια cycle ως script/skill/agent με contract) — εδώ η αυτοματοποίηση.
3. **Knowledge-deposit** (feature→skill, απόφαση→memory, εύρημα→auto-deposit) — μηχανή compounding.
4. **Surface** (plan→dashboard→app→marketing) — μελλοντικό, εκτός spec.

Αρχή: ο πυρήνας κρίσης (τι config/αποδοχή) μένει ανθρώπινος· ό,τι είναι γύρω του γίνεται
ντετερμινιστικό ή deposited-skill. Agents μόνο όπου η κρίση ΕΙΝΑΙ το προϊόν, read-only προς το state.

## 2. Ο κανονικός κύκλος (market, task) — χάρτης & κόστος

| # | Στάδιο | Εργαλείο | Κόστος |
|---|---|---|---|
| 0 | Cycle spec | απόφαση market/task/date/config/features | 🔴 research |
| 1 | Ingest | fetch_entsoe_*, build_real_load_forecast | 🔴 |
| 2 | Rebuild | `python -m src.data --task ...` + backup + compare | 🔴 |
| 3 | Validate | preflight_check.py [--poison], lagscan, TZ guards | 🟢 |
| 4 | Forecast | master_forecast / run_master_grid | 🟢 |
| 5 | Gate/Accept | §2 validity gate (|ΔMAE|>0.15 & ≥2 συνθήκες) | 🔴 κρίση |
| 6 | Uncertainty | conformal.py (προαιρετικό) | 🟡 |
| 7 | Report/artifacts | make_ablation_report, charts, dashboard JSON, build_run_ledger.py | 🟡 |
| 8 | Deposit | last.md, memory, git commit+push | 🔴 |
| 9 | Settle | forecast vs realized, rolling-MAE alerts | ⬜ φάση προϊόντος |

Εύρημα: τα «άπειρα βήματα» είναι στα 0,5,8 (κρίση & κατάθεση), όχι στα 1-4/6-7 (scriptable).
Δύο στόχοι: (α) collapse μηχανικών σταδίων σε 1 εντολή, (β) capture αποφάσεων σε reusable assets.
Υπάρχοντα assets: scripts/build_run_ledger.py, scripts/claude_hooks/guard_edits.py, src/conformal.py,
.claude/skills/energy-forecast/scripts/preflight_check.py, src/check_crosslag_fairness.py.

## 3. Ταξινόμηση & ωριμότητα

0→plan file + research-assistant agent (🟡) · 1-2→script runner (🟢) · 3-4→script υπάρχει (🟢) ·
5→script υπολογίζει, άνθρωπος αποδέχεται (🔴 κρίση μένει) · 6→conformal script (🟡) ·
7→script + agent μόνο για αφήγηση (🟢) · 8→run-skill-generator + consolidate-memory (🟡) · 9→cron+script (⬜).

## 4. Μηχανισμός Knowledge-Deposit

- **4α feature=skill → Feature Registry**: κάθε validated feature = εγγραφή (availability rule,
  lagscan evidence, gate-timing proof, verdict, reproduction cmd). ΑΠΟΦΑΣΗ: **C** = machine-readable
  registry (YAML/JSON) + thin skill (data + narrative μαζί).
- **4β decision=memory → consolidate-memory**: dedupe/compress όταν το memory ξεπερνά budget γραμμών (C).
- **4γ finding=auto-deposit → run-skill-generator**: εύρημα που περνά §2 gate σε ≥2 windows →
  auto draft σε `staging/` → **promotion review** από χρήστη → registry+memory (+ προαιρετικό
  preflight invariant). ΑΠΟΦΑΣΗ: **C** (σέβεται guard_edits.py, όχι σιωπηλή αυτο-εγγραφή).
- **4δ composition=δικό μας πλάνο → plan.yaml** ανά (market,task): στάδια, config, gate spec,
  data adapters. Ο runner εκτελεί το plan. Νέα αγορά = νέο plan file. Αυτή είναι η γενίκευση.

## 5. Σκάλα αυτοματοποίησης (build order)

ΑΠΟΦΑΣΗ: όλη η σκάλα 1→5 διαδοχικά, βήμα-βήμα, ξεκινώντας Rung 1.

1. **Cycle runner** — μία εντολή preflight→run→evaluate→report→registry από plan.yaml
   (Python orchestrator, ίδιο conda env, testable). Μέγιστη πτώση, μηδενικό ρίσκο στον πυρήνα.
2. **Feature Registry + run-skill-generator** (auto→staging + promotion).
3. **results-auditor (read-only agent) + consolidate-memory**.
4. **Gate-assist** — script προ-υπολογίζει §2 πίνακα, άνθρωπος πατά accept.
5. **Daily runner (cron) + settle + alerts** — φάση προϊόντος, εκτός spec.

## 6. Frozen (μη διαπραγματεύσιμο)

Engine · AEL/gates · §2 semantics · single-writer master MDs (guard_edits.py) · ΕΝΑ conda process ·
artifacts-ως-interface · σκληροί κανόνες energy-forecast (no Optuna, strict gate default). Γύρω, ποτέ μέσα.

## 7. Αποφάσεις (κλειδωμένες 2026-07-05)

1. run-skill-generator auto-level: **C** (auto→staging + promotion review).
2. Build order: **όλη η σκάλα 1→5 διαδοχικά, βήμα-βήμα**, ξεκινώντας Rung 1.
3. Artefacts: `docs/superpowers/{specs,plans}/`.

Εκκρεμούν: θέση feature-registry αρχείου (Rung 2) · plan.yaml schema (Rung 1).

## 8. Decomposition σε plans

1. plan-cycle-runner (Rung 1) — πρώτο, ανεξάρτητο
2. plan-feature-registry-and-generator (Rung 2)
3. plan-results-auditor-and-memory-consolidation (Rung 3)
4. plan-gate-assist (Rung 4)
5. plan-daily-runner-settle-alerts (Rung 5 — αργότερα)

---

*Σχετικό asset (2026-07-05): το plugin `plugins/epf-ops/` πακετάρει τα operating-layer skills
(energy-forecast+runner merged, ingest-audit, synthesize-ablation, cycle-ops), τον validity-reviewer
agent και τον guard hook — βλ. `plugins/epf-ops/README.md`.*
