# FeatureENG Agent — Handoff Report (για τον master agent)

> **Ημερομηνία:** 2026-07-06 · **Έκδοση skill:** v1.1 · **Θέση:** `.claude/skills/feature-eng/`
> (mirror: `plugins/epf-ops/skills/feature-eng/`) · **Repo:** epf_greece_starter
> Αυτό το αρχείο είναι αυτόνομο — διαβάζεται χωρίς πρόσβαση στο ιστορικό συνομιλιών.

## 1. Τι είναι

Το **feature-eng** είναι orchestrator skill για **data_in extension**: τον πλήρη κύκλο ζωής
μιας νέας πηγής δεδομένων ή ενός νέου feature στο leakage-free forecasting pipeline του
epf_greece_starter (GR DAM/IDM/Forward × price/load). Δεν είναι ξεχωριστό LLM process —
είναι πρωτόκολλο 7 σταδίων που εκτελεί ο εκάστοτε agent μέσα στο session, δανειζόμενο τα
υπάρχοντα εξειδικευμένα skills αντί να τα αντιγράφει.

**Κεντρική αρχή (TDD pre-registration):** τα acceptance tests **T0-T8** γράφονται και
κλειδώνουν σε design doc **ΠΡΙΝ γραφτεί κώδικας feature**. Ένα «θεαματικά καλό» αποτέλεσμα
θεωρείται failing test (ύποπτο leakage), όχι επιτυχία — κωδικοποίηση του xborder post-mortem
(το «όφελος −0.72» ήταν same-auction leakage).

## 2. Τα 7 στάδια (με STOP gates)

| # | Στάδιο | Τι κάνει | Καλεί | STOP gate |
|---|---|---|---|---|
| 0 | Scope & ουρά | Διαβάζει last.md §1-§2 + ABLATION §2/§7 · ελέγχει conda ουρά · **T0 coverage check** (read-only, system python) | — | <2 πλήρη windows → BLOCKED |
| 1 | DESIGN | Design doc από template: μηχανισμός, pre-registered πρόσημο/μέγεθος ΔMAE, lagscan peak, hour-profile, FI rank + πίνακας T0-T8 + structural breaks | — | Χωρίς πλήρες design doc, τίποτα δεν προχωράει |
| 2 | AUDIT | Διαβατήριο πηγής: T1 gate-timing (**πρωτογενής πηγή**, όχι οικονομική λογική) · T2 lagscan · T3 hour-profile · πληρότητα ανά task parquet | skill `ingest-audit` | T1/T2/T3 fail → ΑΠΟΡΡΙΨΗ με μηδενικό training κόστος |
| 3 | IMPLEMENT | Fetcher → rebuild parquet (backup+σύγκριση) → νέα ομάδα ΜΟΝΟ στο `feature_availability.py` με availability rule → poisoning (T5) + control run ±0.05 (T6) | guard hook (ask) | Poisoning FAIL → δεν προχωράει |
| 4 | BATCH | Πίνακας πειραμάτων: retrain=static, seed 42, `default` vs `default,<group>`, ≥2 windows, LGBM+XGB · ονοματολογία `<window>_<algo>_<mode>_<spec>.json` σε `runs/feat_<name>/` | skill `energy-runner` (κανόνες εκτέλεσης) | T4 fail-fast: `#features` ίδιο = VOID arm → σταμάτα το batch |
| 5 | VERDICT | ΔMAE πίνακες + §2 pre-gate (\|Δ\|>0.15, ίδιο πρόσημο, ≥2 ανεξάρτητα windows) · σύγκριση με pre-registered (T8 too-good red flag) | skill `synthesize-ablation` + subagent `validity-reviewer` | Κανένα ACCEPTED χωρίς validity-reviewer |
| 6 | DEPLOY | FI evidence (T8β) · deploy checklist (leakage proof + FI + expected vs actual + KPI + dangers/rollback) · validator script PASS · deposit σε last.md/ABLATION + commit | validator script | Τελική αποδοχή = ΑΝΘΡΩΠΙΝΗ απόφαση, ποτέ αυτο-έγκριση |

## 3. Acceptance tests T0-T8 (σύνοψη)

- **T0 Coverage**: ≥2 πλήρη ανεξάρτητα windows διαθέσιμα στα δεδομένα της πηγής.
- **T1 Gate timing**: πρωτογενής απόδειξη δημοσίευσης ΠΡΙΝ το gate 12:00 CET D-1 (ή νόμιμο lag).
- **T2 Lagscan**: peak στο θεωρητικά προβλεπόμενο k, |corr|<0.85, όχι «βολικό» k.
- **T3 Hour-profile**: σχήμα λογικό (shift = fetcher/TZ bug).
- **T4 Non-VOID arm**: `#features` αλλάζει baseline↔spec· στήλη υπάρχει σε ΚΑΘΕ target task parquet.
- **T5 Poisoning**: `preflight_check.py --poison` PASS μετά από κάθε αλλαγή στον leakage-sensitive πυρήνα.
- **T6 Reproducibility**: control run, anchor ±0.05.
- **T7 §2 gate**: |ΔMAE|>0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ανεξάρτητα windows (PENDING αλλιώς).
- **T8 Too-good-to-be-true**: Δ ~ pre-registered μέγεθος· FI rank εύλογο· αλλιώς targeted poisoning πριν από claim.

## 4. Artifacts που παράγει

| Artifact | Θέση | Ρόλος |
|---|---|---|
| Design doc | `docs/features/<name>/design.md` | Pre-registration + audit evidence + T0-T8 status |
| Deploy checklist | `docs/features/<name>/deploy.md` | 6 ενότητες: LEAKAGE-FREE PROOF / FEATURE IMPORTANCE / EXPECTED vs ACTUAL / KPI & VERDICT / DANGERS & ROLLBACK / TRACE |
| Validator | `.claude/skills/feature-eng/scripts/validate_deploy_checklist.py` | System python (όχι conda)· FAIL σε placeholders, ελλιπείς ενότητες, ή suspended νούμερα (15.17/16.10/19.17/14.43/15.02) χωρίς suspension marker· smoke-tested |
| Run JSONs/CSV | `runs/feat_<name>/`, `results/feat_<name>.csv` | Συμβατά με `synthesize_ablation.py` |
| Εγγραφές | `ABLATION_PLAN §5/§7`, `last.md §1/§6` | Μόνο μέσω των verdicts (ποτέ σιωπηλή αντικατάσταση) |

Το `deploy.md` είναι σχεδιασμένο ως η μελλοντική πηγή εγγραφής του **Feature Registry**
(Rung 2 του compounding-layer spec, `docs/superpowers/specs/2026-07-05-...md` §4α — approved,
δεν έχει χτιστεί ακόμα).

## 5. Σκληροί κανόνες που κληρονομεί (δεν επαναδιατυπώνονται — ισχύουν αυτούσιοι)

Από `energy-forecast`/CLAUDE.md/last.md §2: ΟΧΙ Optuna · strict gate default · ΕΝΑ conda
process (long runs detached με Start-Process/ASCII args) · `--train_end` μόνο με static ·
data/raw-data/processed-OLD προστατευμένα (guard hook deny) · κανένα νούμερο χωρίς JSON/CSV
trace + εντολή αναπαραγωγής · ΠΟΤΕ suspended νούμερα δίπλα σε leak-free · headline απαιτεί
≥3 seeds + 2ο window.

## 6. Track record (απόδειξη ότι τα STOP gates δουλεύουν)

**Dry-run: henex_premarket (2026-07-06)** — πρώτη πραγματική εκτέλεση, Στάδια 0-2α μόνο,
με την conda ουρά πιασμένη από batch (κανένα conda process δεν χρησιμοποιήθηκε — όλα τα
inspections με standalone Python311+fastparquet):

- **T0 FAIL**: το cached parquet κάλυπτε 2020-11→2026-01-01 μόνο → Q1 2026 35.6%, Μάρτιος 0%,
  μόνο το καλοκαίρι 2025 πλήρες → 1 window, T7 δομικά αδύνατο.
- **Infra ευρήματα**: `data/raw/henex/premarket_summary/` άδειος (0 αρχεία), κανένα fetcher,
  `data_future.py` ορφανό module με ΛΑΘΟΣ raw path — οι στήλες δεν υπάρχουν στο ενεργό
  feature store.
- **T1 OPEN**: καμία πρωτογενής πηγή για την ώρα δημοσίευσης — μόνο οικονομική υπόθεση.
- **Verdict: BLOCKED πριν γραφτεί μία γραμμή κώδικα**, μηδενικό training κόστος. Πλήρης
  τεκμηρίωση: `docs/features/henex_premarket/design.md` + `ABLATION_PLAN §7.9`.

Τα ευρήματα αυτά τροφοδότησαν το v1.1 (βλ. §7) — το skill αυτο-βελτιώνεται από τα δικά του runs.

## 7. Changelog

- **v1.0 (2026-07-06 πρωί)**: αρχική έκδοση — 6 στάδια, T1-T8, templates, validator
  (smoke-tested), mirror στο plugin epf-ops, commit `1769df4`.
- **v1.1 (2026-07-06)**: μαθήματα από το henex dry-run + market research:
  (α) **T0 data-coverage check** στο Στάδιο 0 (BLOCKED πριν από training αν <2 πλήρη windows)·
  (β) **T1 απαιτεί πρωτογενή πηγή** — οικονομική λογική ρητά ΔΕΝ αρκεί·
  (γ) **σειρά για ολοκαίνουρια πηγή**: staging merge στο data.py ΠΡΙΝ το επίσημο lagscan
  (το lagscan.py διαβάζει μόνο task parquet)· προκαταρκτικός scan μόνο ρητά σημειωμένος·
  (δ) **read-only inspections χωρίς conda** (Python311+fastparquet) + «ορφανός loader» έλεγχος·
  (ε) **structural breaks** στο design/deploy template (SDAC 15-min MTU 2025-10-01,
  lignite exit 2026, negative-price regime 2026) — από τη θεματική έρευνα αγοράς 2026-07-06
  (`Greece_DAM_Electricity_Thematic_Research_2026-07.docx`).

## 8. Πώς καλείται

```
Skill: feature-eng   args: "<όνομα πηγής/feature> — <ποια στάδια, τυχόν όρια>"
```

Παράδειγμα: `feature-eng` args `xb_lag1_h0 — full lifecycle, στάδια 3+ μόνο όταν ελευθερωθεί η ουρά`.
Κατάλληλος επόμενος υποψήφιος: **xb_lag1_h0** (ήδη vetted στα χαρτιά, ABLATION §8.1, χωρίς
data-coverage blocker). Το henex_premarket παραμένει BLOCKED μέχρι: πρωτογενής T1 πηγή +
backfill/fetcher + path fix + staging merge.

## 9. Γνωστά όρια

- Δεν τρέχει τίποτα παράλληλα με άλλο conda run (by design — μία ουρά).
- Δεν αυτο-εγκρίνει: default set/headline αποφάσεις είναι ανθρώπινες (compounding spec §1).
- Ο validator ελέγχει δομή/placeholders/suspended numbers — ΔΕΝ ελέγχει την ουσία των
  αριθμών (αυτό το κάνει ο validity-reviewer με τα run artifacts).
- Το Feature Registry (Rung 2) δεν υπάρχει ακόμα — μέχρι τότε το deploy.md είναι η
  αυθεντική εγγραφή ανά feature.
