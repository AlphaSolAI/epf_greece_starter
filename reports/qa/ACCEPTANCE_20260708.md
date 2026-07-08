# QA Pack — Acceptance Results (σενάρια 1-4)

**Ημερομηνία:** 2026-07-08 · **Spec §9** · Component: `epf-code-reviewer` agent + linter +
`triaging-run-failures` runbook. Και τα 4 σενάρια **PASS**.

| # | Σενάριο | Component | Αναμενόμενο | Αποτέλεσμα | Verdict |
|---|---|---|---|---|---|
| 1 | Block D replay (static αντί weekly) | PRE-RUN review + linter | ≥MAJOR + FIX weekly + linter WARNING #5 | BLOCK· 2×BLOCKER (static↔weekly purpose-invalid, δεν αναπαράγει 17.035) + 2×MAJOR + linter `seed_sweep_static`×2· FIX `--retrain weekly`· επιπλέον έπιασε ότι λείπει το Summer window | **PASS** |
| 2 | NO-MAE replay (schema bug) | TOOLING review | εντοπισμός mismatch μέσω πραγματικού JSON | BLOCK· άνοιξε `q1_lgbm_weekly_default.json`, επιβεβαίωσε «top-level mae: False», `metrics[0]['MAE']=17.4312`· FIX σωστό key· +2 MINOR (cwd glob, file handle leak) | **PASS** |
| 3 | Freeze-χαλάρωμα (AEL leakage) | CORE-DIFF review | BLOCK + REQUIRED FOLLOW-UP poisoning | BLOCK· BLOCKER «removal of freeze-at-cutoff → future leakage», διασταύρωσε με `recursive_openloop.py:165-174` (AEL contract)· REQUIRED FOLLOW-UP: `--poison`, `--poison_y`, control run, anchor ±0.05 | **PASS** |
| 4 | FileNotFoundError runbook Q&A | triaging-run-failures | «by design, όχι regression», όχι fix του split_utils | runbook §1 γραμμή: «BY DESIGN (fix 2026-07-08)... φτιάξε το parquet· ΜΗΝ «διορθώσεις» το fallback» | **PASS** |

## Παρατηρήσεις

- Ο reviewer ξεπέρασε τα ελάχιστα κριτήρια: σε κάθε σενάριο διάβασε πραγματικά αρχεία
  (ABLATION_PLAN anchor numbers, run JSON schema, AEL contract lines) αντί να κρίνει
  «στα τυφλά» — ακριβώς το ζητούμενο (evidence-based, όχι πειστικότητα διατύπωσης).
- Επιπλέον ευρήματα εκτός σεναρίου (missing Summer window, file-handle leak, cwd glob)
  δείχνουν ότι δεν κάνει pattern-match στο fixture αλλά πραγματικό review.
- Linter deterministic PASS στο σενάριο 1 (`seed_sweep_static` ×2, exit 1) πριν καν
  τον agent.

## Fixtures (μη εκτελέσιμα — μόνο για review)

- `tests/fixtures/qa/seed_check_static.sh`
- `tests/fixtures/qa/bad_summarizer.py`
- `tests/fixtures/qa/freeze_relax.diff` (συνθετικό diff)
