# QA Pack (Code Review + Debug) — Design Spec

> Στόχος: να κλείσει το κενό ανάμεσα στο guard hook (ρωτάει αλλά δεν διαβάζει κώδικα),
> στα poisoning tests (αποδεικνύουν συμπεριφορά αλλά τρέχουν ΜΕΤΑ το γράψιμο) και στον
> validity-reviewer (κρίνει claims, ποτέ diffs): **static review πριν εκτελεστεί ή
> commit-αριστεί κώδικας + συστηματικό triage όταν ένα run σκάει ή παραξενεύει**.
> Υλοποιεί το προδιαγεγραμμένο «QA pack (debug/code-review/testing)» extension slot του
> `plugins/epf-ops` (plugin.json v0.1.2) χωρίς redesign του operating layer.

**Ημερομηνία:** 2026-07-07 · **Κατάσταση:** approved (brainstorming 2026-07-07 — όλα τα
sections εγκρίθηκαν από τον χρήστη) · **Σχετικό spec:**
`2026-07-05-operating-compounding-layer-design.md` (αρχή: agents μόνο όπου η κρίση ΕΙΝΑΙ
το προϊόν, read-only προς το state· deterministic όπου γίνεται).

## 0. Success criteria — 4 πραγματικά περιστατικά ως οδηγός

Το design κρίνεται στο αν θα είχε πιάσει τα εξής (όλα συνέβησαν — βλ. `last.md`,
`ABLATION_PLAN §5.12`):

| # | Περιστατικό | Ποιο component το πιάνει |
|---|---|---|
| Ι | Block D seed-check έτρεξε με `--retrain static` αντί weekly — κάηκε ολόκληρο batch | PRE-RUN review (purpose-fit) + linter #5 |
| ΙΙ | Summarizer έγραφε ψευδώς «NO-MAE» (schema mismatch με τα run JSONs) | TOOLING review (άνοιγμα πραγματικού JSON) |
| ΙΙΙ | Subtle leakage αλλαγή σε πυρήνα που περνά το ask του hook | CORE-DIFF review + REQUIRED FOLLOW-UP poisoning |
| ΙV | `-loadfc` VOID (8/8 Δ=0.000, infra bug) · FileNotFoundError · detached χωρίς log | triaging-run-failures runbook |

Coverage: ΟΛΕΣ οι 4 κατηγορίες (απόφαση χρήστη 2026-07-07), έμφαση σε Ι και ΙΙ.

## 1. Αρχιτεκτονική — 4 components, τοποθέτηση κατά μηχανισμό

| Component | Μηχανισμός | Γιατί έτσι |
|---|---|---|
| `epf-code-reviewer` | subagent, read-only | το review θέλει φρέσκα μάτια σε απομονωμένο context· η κρίση είναι το προϊόν |
| `scripts/qa/check_run_config.py` | script (stdlib, system python) | μηχανικοί έλεγχοι δεν χρειάζονται LLM· μηδέν tokens, μηδέν conda |
| `triaging-run-failures` | skill (κύριο context) | το debugging είναι διαδραστικό· σέβεται το «ΕΝΑ conda process» |
| hook nudges | PostToolUse + PreToolUse, fail-open | υπενθύμιση στο τελευταίο χρήσιμο σημείο· ΠΟΤΕ deny |

Trigger model (απόφαση χρήστη): **SOP + nudges** — κανόνες στα SKILL.md/CLAUDE.md +
hook υπενθυμίσεις. Όχι hard gate, όχι σκέτο on-demand. Το debug σκέλος είναι on-demand
από τη φύση του (ενεργοποιείται όταν κάτι σκάει).

## 2. Agent `epf-code-reviewer` (`.claude/agents/epf-code-reviewer.md`)

Δομικός καθρέφτης του validity-reviewer: read-only tools (Read, Grep, Glob, Bash),
αυστηρό verdict format, **πηγές-κανόνων αντί για αντίγραφα** — διαβάζει at runtime:
`MARKDOWN/CLAUDE.md` (μη διαπραγματεύσιμοι κανόνες) · `MARKDOWN/last.md` §2/§4/§5 ·
`MARKDOWN/ABLATION_PLAN.md` §2/§3 · `.claude/skills/energy-forecast/SKILL.md`
(εντολές/flags). Όταν οι κανόνες αλλάζουν, ο reviewer ενημερώνεται δωρεάν — κανένα
sync χρέος.

### Modes (δηλώνονται στην κλήση)

- **PRE-RUN** — input: script path ή command string + **υποχρεωτικά δηλωμένος σκοπός**
  του run (π.χ. «seed-confirm του weekly headline candidate»). Χωρίς purpose: αυτόματο
  MAJOR finding (το purpose-fit δεν αξιολογείται). Βήματα: (1) τρέχει τον linter,
  (2) κρίνει ό,τι δεν πιάνει linter — κυρίως αν το config εξυπηρετεί τον δηλωμένο σκοπό
  (περιστατικό Ι: μηχανικά έγκυρο, purpose-invalid).
- **CORE-DIFF** — input: `git diff` (unstaged/staged ή range) των leakage-sensitive
  αρχείων (τα 7 ASK_FILES του `guard_edits.py`). Ψάχνει: παραβιάσεις cutoff/freeze
  semantics, off-by-one σε lag construction, silent reindex/NaN σε merges, TZ μετατροπές
  εκτός των loaders του `data.py`, παρακάμψεις του feature_availability συμβολαίου.
  Εκδίδει ΠΑΝΤΑ γραμμή `REQUIRED FOLLOW-UP` (poisoning/control-run/anchor ή «—»).
- **TOOLING** — summarizers/fetchers/runners: schema assumptions επαληθεύονται
  ανοίγοντας ≥1 πραγματικό run JSON (περιστατικό ΙΙ), encoding (`-X utf8`), error
  swallowing (`except: pass`), paths, γράψιμο σε προστατευμένα μονοπάτια.

### Verdict format (υποχρεωτικό)

```
TARGET: <file/command/diff>   MODE: PRE-RUN | CORE-DIFF | TOOLING
FINDINGS:
  [BLOCKER|MAJOR|MINOR] <τι> — EVIDENCE: <file:line ή json path> — FIX: <ελάχιστο>
REQUIRED FOLLOW-UP: <poisoning/control-run/anchor ή —>
VERDICT: APPROVE | APPROVE-WITH-FIXES | BLOCK
```

### Hard rules & όρια

BLOCK αν: linter violation χωρίς ρητή δικαιολόγηση · core diff χωρίς poisoning plan ·
εντολή/script που συγκρίνει runs across windows/gates/data snapshots · script που γράφει
σε `data/raw|processed` ή στα master MDs · target αρχείο που δεν ανοίγει (fail, όχι
υπόθεση — ίδιος κανόνας με validity-reviewer).

Όρια: ΔΕΝ κάνει edits (κανένα Write/Edit tool) · ΔΕΝ τρέχει conda ΠΟΤΕ — Bash μόνο για
`git diff/show/log` + τον linter με system python, ώστε ο κανόνας «ΕΝΑ conda process»
να μένει άθικτος ακόμα κι όταν τρέχει run · αγνοεί τα ~39 dead src αρχεία (λίστα στο
CLAUDE.md) και το `thesis/*.tex` · όχι style/refactoring nitpicks — μόνο
correctness/validity/κανόνες repo · δεν προτείνει νέα πειράματα πέρα από το FIX.

## 3. Linter `scripts/qa/check_run_config.py`

Stdlib-only, **system python** (όπως `guard_edits.py`, `build_run_ledger.py` — δεν
αγγίζει conda). CLI: `--script <path>` | `--cmd "<string>"` · προαιρετικά
`--purpose "<text>"` (τυπώνεται στο report για τον reviewer), `--json`. Exit code:
0 = καθαρό, 1 = findings. Κάθε έλεγχος = μικρή συνάρτηση σε registry (id, severity,
message, γραμμή-evidence) — **νέο δίδαγμα ⇒ νέος έλεγχος** (compounding).

| # | Έλεγχος | Severity |
|---|---|---|
| 1 | `--train_end` μαζί με `--retrain monthly/weekly` (αγνοείται σιωπηλά) | BLOCKER |
| 2 | features με `xb` χωρίς `_lag` (same-day xborder = leakage) | BLOCKER |
| 3 | αναφορά `tune_*optuna` / `tune_xgb` (απαγορευμένα by rule) | BLOCKER |
| 4 | `--gate academic` (μόνο για σύγκριση με papers) | WARNING |
| 5 | seed sweep (`--seed` ≠ 42 ή πολλαπλά seeds) με `--retrain static` | WARNING |
| 6 | πολλαπλά ταυτόχρονα conda invocations (`&`, παράλληλα Start-Process) | WARNING |
| 7 | training run χωρίς `--out_json` (Α5 traceability) | WARNING |
| 8 | conda εντολή χωρίς `-X utf8` | MINOR |
| 9 | non-ASCII args σε Start-Process detached pattern | WARNING |

Tests: `tests/test_check_run_config.py` — pytest, μπαίνει στο υπάρχον ~1s suite
(`conda run -n epf ... -m pytest tests/ -q`, ήδη permitted).

## 4. Skill `triaging-run-failures` (`.claude/skills/triaging-run-failures/`)

Όνομα συμμετρικό με το υπάρχον `triaging-suspicious-results`: εκείνο πιάνει τα ύποπτα
ΚΑΛΑ νέα, αυτό τα σκασίματα/περίεργα. Το description κάνει ρητό cross-reference στο
άλλο skill για σωστό auto-triggering.

Δομή SKILL.md:

1. **Μεθοδολογία**: δένει με `superpowers:systematic-debugging` (root cause πριν από
   fix, ελάχιστο repro, ένα change τη φορά) — το skill προσθέτει ΜΟΝΟ το repo-specific
   στρώμα, δεν αντικαθιστά τη μεθοδολογία.
2. **Runbook** «σύμπτωμα → αιτία → διαγνωστικό → γνωστό fix», αρχικό περιεχόμενο από
   πραγματικά περιστατικά:
   - `FileNotFoundError` από `load_processed(task=...)` → BY DESIGN (fix 2026-07-06,
     `split_utils.py`) — φτιάξε το per-task parquet, ΜΗΝ «διορθώσεις» το fallback.
   - Δ=0.000 σε όλα τα arms → feature δεν μπαίνει καν στο matrix (infra void τύπου
     `-loadfc`) — διαγνωστικό: dump των στηλών του X πριν το fit.
   - «NO-MAE»/κενά πεδία σε summary ενώ το run τελείωσε → schema mismatch
     summarizer↔JSON — άνοιξε το πραγματικό JSON, δες τα keys.
   - Detached run χωρίς log → Start-Process χωρίς redirect / non-ASCII args /
     OneDrive κλειστό.
   - `UnicodeDecodeError`/αλαμπουρνέζικα κονσόλας → cp125x, λείπει `-X utf8`.
   - Seed αποτελέσματα ασύμβατα με τον candidate → retrain-policy mismatch
     (το Block D pattern).
   - Αναπαραγωγή εκτός ±0.05 → anchor check (LGBM default static Q1 = 16.10±0.05).
3. **Έτοιμα διαγνωστικά one-liners** — system python όπου γίνεται· ό,τι θέλει conda
   σημαδεύεται ρητά και μπαίνει σειριακά στη μία-και-μοναδική ουρά.
4. **Escalation rule**: αν η διάγνωση καταλήγει σε αλλαγή leakage-sensitive αρχείου →
   ο fix περνά ΥΠΟΧΡΕΩΤΙΚΑ από `epf-code-reviewer` (CORE-DIFF) + poisoning.
5. **Deposit rule** (compounding): κάθε νέο failure mode που λύνεται → νέα γραμμή στο
   runbook πριν κλείσει το session.

Skill και όχι agent: το debugging απαιτεί διαδραστικά διαγνωστικά μέσα στο κύριο
context, με συντονισμό της μίας ουράς conda — πράγμα αδύνατο για απομονωμένο subagent.

## 5. Hook nudges (επέκταση enforcement layer — ποτέ deny)

- **PostToolUse σε `Edit|Write`** για τα 7 ASK-files: μετά από επιτυχές edit, έγχυση
  υπενθύμισης στο context — «Άλλαξες {path}: πριν από run/commit ⇒ epf-code-reviewer
  (CORE-DIFF) + poisoning plan». PostToolUse (όχι PreToolUse) ώστε να μιλά μόνο όταν
  η αλλαγή όντως έγινε.
- **PreToolUse σε `Bash`**, matcher σε
  `master_forecast|run_master_grid|run_ablation|overnight_*.sh|followup_*.sh|Start-Process`:
  μη-μπλοκάρουσα υπενθύμιση «νέο/αλλαγμένο script; αν δεν πέρασε pre-run review, τρέξε
  πρώτα τον linter». Anti-spam εξαιρέσεις: `--help`, `preflight_check`, `pytest`.
- **Μηχανισμός έγχυσης**: `hookSpecificOutput.additionalContext` (προτιμώμενο)· fallback
  `systemMessage`. Η επιλογή οριστικοποιείται στο rollout βήμα 4 με βάση την
  εγκατεστημένη έκδοση Claude Code — αμφότερα non-blocking, το design δεν εξαρτάται.
- Πολιτική: fail-open (exception ⇒ καμία παρέμβαση) · τα nudges δεν κάνουν deny ΠΟΤΕ ·
  `guard_edits.py` παραμένει stdlib/system python · matchers σε `.claude/settings.json`
  (+ `plugins/epf-ops/hooks/hooks.json` στο resync).

## 6. SOP deposits + review artifacts

- `MARKDOWN/CLAUDE.md`: +2 γραμμές στους μη διαπραγματεύσιμους κανόνες — (α) pre-run
  review νέων/αλλαγμένων scripts πριν εκτελεστούν, (β) CORE-DIFF review πριν από commit
  πυρήνα — και ενημέρωση χάρτη repo (`scripts/qa/`, νέος agent, νέο skill).
- `energy-forecast` SKILL.md: linter+reviewer ως βήμα στο pre-batch pre-flight.
- **Artifact rule**: σε verdict BLOCK ή σε review πριν από commit πυρήνα, ο ΚΥΡΙΟΣ agent
  (ο reviewer είναι read-only) σώζει το verdict σε
  `reports/qa/qa_review_<YYYYMMDD_HHMM>_<target>.md` — Α5 traceability, μηδενικό
  επιπλέον process. Τα υπόλοιπα reviews δεν αρχειοθετούνται.

## 7. Ροές στην πράξη

- **Pre-run**: ετοιμάζεται script → linter (δευτερόλεπτα, μηδέν tokens) →
  `epf-code-reviewer` PRE-RUN με δηλωμένο σκοπό → APPROVE ⇒ detached launch · BLOCK ⇒
  fix → ξανά linter → σύντομο re-review ΜΟΝΟ των findings.
- **Core-diff**: edit σε πυρήνα → υπάρχον ask του hook → PostToolUse nudge → πριν από
  run/commit: CORE-DIFF → `REQUIRED FOLLOW-UP: poisoning` → `preflight_check.py
  --poison` → commit. Ο `validity-reviewer` για claims παραμένει ανεξάρτητο στάδιο.
- **Debug**: run σκάει/παραξενεύει → `triaging-run-failures` → runbook match ⇒ γνωστό
  fix· αλλιώς systematic-debugging loop → fix σε πυρήνα ⇒ escalation σε CORE-DIFF +
  poisoning → deposit νέου failure mode στο runbook.

## 8. Error handling του ίδιου του pack

- Linter crash → ο reviewer το αναφέρει ως MAJOR finding και συνεχίζει χειροκίνητα —
  ποτέ σιωπηλή παράλειψη ελέγχου.
- Target που δεν ανοίγει → BLOCK με αιτία.
- Hooks fail-open· nudges ποτέ deny.
- Σύγκρουση verdicts: reviewer APPROVE αλλά poisoning FAIL ⇒ **το poisoning υπερισχύει
  πάντα** (behavioral απόδειξη > static ανάγνωση).

## 9. Acceptance tests (το pack ΔΕΝ είναι DONE πριν περάσουν)

1. **Block D replay**: PRE-RUN σε seed-sweep script με `--retrain static` + purpose
   «seed-confirm weekly candidate» ⇒ ≥MAJOR με σωστό FIX (και linter WARNING #5).
2. **NO-MAE replay**: TOOLING σε summarizer με λάθος JSON key ⇒ εντοπισμός του mismatch
   μέσω πραγματικού run JSON.
3. **Freeze-χαλάρωμα**: τεχνητό diff στο `recursive_openloop.py` ⇒ CORE-DIFF απαιτεί
   poisoning και δίνει BLOCK χωρίς αυτό.
4. **FileNotFoundError**: το runbook απαντά «by design, όχι regression» χωρίς πρόταση
   «διόρθωσης» του `split_utils`.

Συν: pytest για τον linter · unit test ότι τα nudges δεν μπλοκάρουν ποτέ.

## 10. Rollout — build order (commit ανά βήμα)

1. Linter + tests (αυτόνομη αξία, μηδενικό ρίσκο)
2. Agent `epf-code-reviewer`
3. Skill `triaging-run-failures`
4. Hook nudges + matchers στο `.claude/settings.json` (εδώ οριστικοποιείται
   additionalContext vs systemMessage)
5. SOP deposits (`MARKDOWN/CLAUDE.md`, energy-forecast SKILL.md)
6. Resync στο `plugins/epf-ops/` (per README — ποτέ direct edit στο snapshot)
7. Acceptance validation (τα 4 σενάρια του §9)

## 11. Frozen / εκτός scope

- Ο πυρήνας (engine/AEL/gates/§2 semantics) δεν αλλάζει σε τίποτα — το pack χτίζει γύρω.
- Single-writer στα master MDs μένει ως έχει· ο reviewer δεν γράφει πουθενά.
- Dead src (~39 αρχεία) και `thesis/*.tex` εκτός review by rule.
- Testing workstream (επέκταση coverage του `tests/`) εκτός — μόνο τα tests του linter.
- Ο `validity-reviewer` παραμένει ανέγγιχτος και ανεξάρτητος (claims ≠ code).

## 12. Κλειδωμένες αποφάσεις (2026-07-07)

1. Coverage: ΟΛΕΣ οι 4 κατηγορίες περιστατικών· προτεραιότητα Ι (pre-run config) και
   ΙΙ (tooling bugs).
2. Trigger: SOP + nudges — όχι hard gate, όχι σκέτο on-demand.
3. Αρχιτεκτονική: Approach Β (agent+script+skill+nudges) + review artifacts σε
   `reports/qa/` μόνο για BLOCK/pre-commit.
4. Ονόματα: `epf-code-reviewer` (αποφυγή σύγκρουσης με feature-dev:code-reviewer) ·
   `triaging-run-failures` (συμμετρία με triaging-suspicious-results) ·
   `scripts/qa/check_run_config.py`.
5. Debug = skill, όχι agent (διαδραστικότητα + ΕΝΑ conda process).

---

*Επόμενο βήμα: implementation plan σε `docs/superpowers/plans/` (writing-plans skill),
με τη σειρά του §10.*
