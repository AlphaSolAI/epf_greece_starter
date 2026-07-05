# CLAUDE.md — epf_greece_starter (GR EPF/STLF · διπλωματική + trading agent)

## Αποστολή

Leak-free, auditable προβλέψεις τιμής (EPF) και φορτίου (STLF) για την ελληνική αγορά
(DAM / IDM / Forward ≤1 εβδ.). Κανένα νούμερο δεν δημοσιεύεται/γράφεται ως claim αν δεν
περνά το VALIDITY GATE. Αυτό το αρχείο είναι guidance — το enforcement ζει σε hooks
(`scripts/claude_hooks/`), poisoning scripts και στα κριτήρια αποδοχής των docs.

## Διάβασε ΠΡΩΤΑ (με αυτή τη σειρά)

1. `last.md` — τρέχουσα κατάσταση (§1), **VALIDITY GATE §2** (invariants Α1-Α5 + Β-σειρά), επόμενο βήμα (§5)
2. `ABLATION_PLAN.md` — έγκυρα ευρήματα (§5) + **κριτήρια αποδοχής §2** + PENDING (§7)
3. Skill `energy-forecast` — σκληροί κανόνες, εντολές, pre-flight scripts (φορτώνεται αυτόματα)
4. Skill `energy-runner` — operating loop για εκτέλεση πειραμάτων

Design specs (μόνο όταν χρειάζεται βάθος): `MASTER_PIPELINE_DESIGN.md` (information clock,
gate, στρατηγικές) · `SYSTEM_DESIGN_TRADING_AGENT.md` (αρχιτεκτονική L0-L5, AEL §4.8,
TZ contract §4.9, validation harness §4.10).

## Μη διαπραγματεύσιμοι κανόνες (σύνοψη — πλήρης μορφή στο SKILL.md + last.md §2)

- **ΟΧΙ Optuna.** `--gate strict` default (`academic` μόνο για σύγκριση με papers). `tf` = oracle, ποτέ tradeable.
- **ΕΝΑ conda process** τη φορά, πάντα: `conda run -n epf --no-capture-output python -X utf8 ...` — μία ουρά, σειριακά, ανεξαρτήτως ποιος το ξεκίνησε (Claude/χρήστης/script).
- **Long runs (>2-3 min) = detached** (Start-Process pattern στο SKILL.md), ποτέ μέσα στο Bash tool.
- `--train_end` ΜΟΝΟ με `--retrain static` (monthly/weekly = expanding, το αγνοούν).
- **Κανένα αποτέλεσμα χωρίς trace**: JSON/CSV path σε `runs/`-`results/` + εντολή αναπαραγωγής. Κανένα claim χωρίς ABLATION_PLAN §2 (|ΔMAE|>0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 συνθήκες· headline: ≥3 seeds + 2ο window).
- **ΠΟΤΕ σύγκριση** runs από διαφορετικά windows/gates/data snapshots. ΠΟΤΕ suspended νούμερα (π.χ. το 15.17) δίπλα σε leak-free νέα.
- **Νέα πηγή δεδομένων**: πρώτα το 3-βήμα pre-flight (πότε ΑΚΡΙΒΩΣ δημοσιεύεται vs gate 12:00 CET D-1 · lagscan · hour-profile)· μπαίνει ΜΟΝΟ ως ομάδα στο `src/feature_availability.py` με ρητό availability rule.
- **Leakage-sensitive πυρήνας** (`src/data.py`, `feature_availability.py`, `master_forecast.py`, `recursive_openloop.py`, `conformal.py`, `scheduled_sampling.py`): μετά από ΚΑΘΕ αλλαγή → poisoning tests (`preflight_check.py --poison`) + control run + reproducibility anchor ±0.05. Το PreToolUse hook ζητά επιβεβαίωση στο edit.
- **`data/raw/`, `data/processed/`, `OLD/` = προστατευμένα** (hook: deny). Αλλαγές μόνο μέσω scripts (fetch/rebuild με backup+σύγκριση).
- **Timezone**: κανονικό frame CET/CEST-naive· μετατροπές ΜΟΝΟ στους loaders του `src/data.py`· TZ guards στο preflight.
- Outputs: `runs/<study>/` · `results/*.csv` · `logs/` · `reports/` · αρχειοθέτηση → `OLD/`. Πάντα `--out_json`.
- Git: commit+push στο τέλος κάθε session · **ΠΟΤΕ push το branch `FEB272026_localhistory`** (4.3GB).

## Standard εντολές

```bash
# Preflight (πριν από ΚΑΘΕ batch — env/parquet/TZ guards· --poison = AEL poisoning tests):
conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/preflight_check.py [--poison] [--baseline]

# Lagscan νέου feature (πριν μπει σε τεστ):
conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/lagscan.py --col <στήλη>

# Runs/grids/ablations/ensemble/conformal: βλ. energy-forecast SKILL.md
#   (src.master_forecast · src.run_master_grid · src.run_ablation · src.make_ensemble)

# Run ledger — auditability index όλων των run JSONs (system python, ΧΩΡΙΣ conda):
python scripts/build_run_ledger.py            # → results/run_ledger.csv
```

## Χάρτης repo

```
last.md, ABLATION_PLAN.md          # τα 2 master αρχεία τρέχουσας αλήθειας
MASTER_PIPELINE_DESIGN.md          # ερευνητικό design (clock/gate/στρατηγικές)
SYSTEM_DESIGN_TRADING_AGENT.md     # προϊοντικό design (L0-L5, AEL, TZ, harness)
src/                               # engine: data.py, feature_availability.py (ΤΟ συμβόλαιο),
                                   #   master_forecast.py, conformal.py, check_crosslag_fairness.py, fetchers
scripts/                           # runners (overnight_*.sh), summarizers, claude_hooks/
runs/ results/ logs/ reports/      # πειράματα · CSV πίνακες · logs · αναφορές/figures
data/raw/ data/processed/          # ΠΡΟΣΤΑΤΕΥΜΕΝΑ — μόνο μέσω scripts
OLD/                               # αρχειοθετημένο ιστορικό (docs/runs) — read-only
thesis/                            # υλικό διπλωματικής
dashboard.html, dashboard/         # οπτικοποίηση forecast JSONs
.claude/skills/                    # energy-forecast (κανόνες) · energy-runner (loop)
.claude/agents/validity-reviewer.md  # subagent ελέγχου εγκυρότητας πριν από claims
```

## Πρωτόκολλο session

- **Online χρήστης**: δώσε copy-paste block να τρέξει ΕΚΕΙΝΟΣ στο terminal + πες τι να επικολλήσει πίσω (RESULT line / JSON path / error). Autonomous (`/goal`, overnight): τρέξε detached με log + progress.
- Πριν γραφτεί εύρημα ως ΔΕΚΤΟ: πέρασέ το από το validity gate (`last.md §2`) — αλλιώς PENDING/SMOKE. Για paper-bound claims: subagent `validity-reviewer`.
- Τέλος session: ενημέρωση `last.md` (ΜΟΝΟ τρέχουσα αλήθεια· παλιά → `OLD/docs/`) + commit+push.
- Κάθε απάντηση με ανοιχτή δουλειά τελειώνει με: `DONE / RESULT PATHS / PENDING / NEXT` (βλ. energy-runner).
