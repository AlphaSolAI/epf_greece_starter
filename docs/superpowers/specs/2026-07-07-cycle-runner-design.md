# Cycle Runner (master orchestrator) — Design Spec (v0)

> Στόχος: **μία εντολή** που τρέχει ολόκληρη τη μηχανική αλυσίδα μιας σύγκρισης
> πρόβλεψης — `preflight → forecast/grid → [conformal] → §2 πίνακας → report → ledger` —
> από ένα αρχείο-συνταγή (`plan.yaml`), **χωρίς να αγγίζει δεδομένα και χωρίς να αποφασίζει
> αποδοχή**. Είναι το **Rung 1 «Cycle runner»** της σκάλας αυτοματοποίησης του
> `2026-07-05-operating-compounding-layer-design.md` (§5.1, §8.1) — το πρώτο, ανεξάρτητο βήμα.

**Ημερομηνία:** 2026-07-07 · **Κατάσταση:** approved (brainstorm ολοκληρώθηκε, αποφάσεις §7 κλειδωμένες).
**Θέση κώδικα (προτεινόμενη):** `src/cycle_runner.py` (ίδιο conda env `epf`).

## 0. Τι είναι — και τι ΔΕΝ είναι

Ντετερμινιστικός Python orchestrator. **Δεν** είναι LLM agent, **δεν** περιέχει καμία νέα
μοντελιστική/leakage λογική. Είναι «glue»: καλεί με τη σωστή σειρά τα **ήδη δοκιμασμένα,
validity-περασμένα** εργαλεία του repo. Κάθε νέα γραμμή λογικής στον πυρήνα = νέο ρίσκο
διαρροής· γι' αυτό ο runner μένει σκόπιμα «χαζός» και ελεγχόμενος.

Ρητά **εκτός** αυτού του runner (v0):
- **Ingest / rebuild parquet** (στάδια 1-2 του cycle) — αγγίζουν `data/raw`, `data/processed`
  (guard hook = deny, backup+compare υποχρεωτικό). Μπαίνουν σε μεταγενέστερη βαθμίδα.
- **Απόφαση αποδοχής** (στάδιο 5) — ο runner *υπολογίζει* τον §2 πίνακα αλλά **ποτέ** δεν
  γράφει «ΔΕΚΤΟ». Αυτό μένει ανθρώπινο (+ `validity-reviewer`).
- **Deposit** (στάδιο 8) — καμία εγγραφή σε `last.md`/`ABLATION_PLAN`, κανένα `git commit`.

### 0.1 Γενικότητα: ο runner γενικεύει τη ρουτίνα, ΟΧΙ το availability

Ο runner είναι γενικός ως προς `(market, task)` επειδή τα περνάει **ως παραμέτρους** στη μηχανή
(το engine ήδη υποστηρίζει `dam`/`idm`/`forward` με presets — dam=H24/s24, idm=H6/s3,
forward=H168/s24). Βάζεις `market: idm` στο plan και δουλεύει.

**Δεν** αποφασίζει όμως ο runner τι δεδομένα/features είναι διαθέσιμα ανά αγορά — αυτό είναι, και
**πρέπει** να παραμείνει, δουλειά **μόνο** του `src/feature_availability.py` (το leakage συμβόλαιο:
πότε δημοσιεύεται τι σε σχέση με το gate κάθε αγοράς). Ο runner λέει «τρέξε για IDM»· η μηχανή +
`feature_availability.py` ξέρουν ήδη τι επιτρέπεται. Αν προσπαθούσε ο runner να «μαντέψει» το
availability, θα εισήγαγε ρίσκο διαρροής σε κάθε νέα αγορά (το xborder same-day «όφελος» ήταν
ακριβώς τέτοια διαρροή). Άρα:
- Νέα αγορά με έτοιμα δεδομένα **και** σωστό availability rule → αλλάζεις **μόνο το plan**.
- Νέα αγορά που χρειάζεται νέο gate-timing/ingest/availability rule → στάδια 0-2 (εκτός runner),
  δουλειά ανθρώπου / `feature-eng` skill. Ο runner δεν τα εφευρίσκει.

**Χρησιμότητα ακόμη κι όταν τα δεδομένα είναι έτοιμα:** η αξία δεν είναι η προετοιμασία δεδομένων,
αλλά η σύμπτυξη ~8 χειροκίνητων εντολών (2 windows × 2 algos × 2 specs + preflight + synthesize +
ledger) σε μία επαναλήψιμη — που εξαλείφει ακριβώς τα λάθη εγκυρότητας που απαγορεύουν οι κανόνες
(λάθος gate, ξεχασμένο window, σύγκριση ετερογενών runs) και αφήνει το `plan.yaml` ως γραπτή
απόδειξη του τι έτρεξε (auditability).

## 1. Εύρος (ποια στάδια του κανονικού κύκλου καλύπτει)

Χαρτογράφηση προς τον 10-σταδιακό κύκλο του compounding spec (§2 εκεί):

| # | Στάδιο | Στον runner; | Υλοποίηση (υπάρχον εργαλείο) |
|---|---|---|---|
| 0 | Cycle spec | ❌ (input) | ο άνθρωπος γράφει το `plan.yaml` |
| 1 | Ingest | ❌ εκτός v0 | — |
| 2 | Rebuild parquet | ❌ εκτός v0 | — |
| 3 | **Validate** | ✅ | `preflight_check.py [--poison]` |
| 4 | **Forecast (batch)** | ✅ | `src.run_master_grid` (ή `src.master_forecast` ανά cell) |
| 5 | Gate/Accept | ⚠️ *μόνο υπολογισμός* | `synthesize_ablation.py` (§2 pre-gate) — **STOP για άνθρωπο** |
| 6 | Uncertainty | ✅ προαιρετικό | `src.conformal` (αν `conformal: true` στο plan) |
| 7 | **Report/artifacts** | ✅ | `synthesize_ablation.py` πίνακες + `scripts/build_run_ledger.py` |
| 8 | Deposit | ❌ (draft μόνο) | ετοιμάζει draft κείμενο, δεν το γράφει |
| 9 | Settle | ❌ εκτός v0 | — |

Δηλαδή ο runner καλύπτει **3 → 4 → (6) → 5(calc) → 7**, με καθαρά STOP στα ανθρώπινα σημεία.

## 2. Είσοδος — το `plan.yaml`

Ένα batch = μία σύγκριση (ώστε να μπορεί να πιάσει τον §2 κανόνα). Το plan περιγράφει τι να
συγκριθεί· ο runner παράγει το καρτεσιανό γινόμενο των cells.

```yaml
study: meteo_recursive_check        # όνομα → runs/<study>/, results/<study>.csv, logs/<study>.log
market: dam                         # dam|idm|forward
task: price                         # price|load
strategy: recursive                 # recursive|direct
retrain: static                     # static|monthly|weekly  (static στα ablations, §3 ABLATION)
gate: strict                        # strict default (μόνο tradeable)
seed: 42
algos: [lgbm, xgb]                  # ≥1
windows:                            # ≥2 ανεξάρτητα για να μπορεί να κριθεί §2
  - {name: q1_2026,  test_start: "2025-12-01 00:00", test_end: "2026-02-28 23:00", train_end: "2025-11-30 23:00"}
  - {name: summer25, test_start: "2025-06-01 00:00", test_end: "2025-08-31 23:00", train_end: "2025-05-31 23:00"}
specs:                              # 1ο = baseline· τα υπόλοιπα συγκρίνονται vs baseline
  - default
  - default,meteo
conformal: false                    # true → τρέχει src.conformal στα forecast JSONs
poison: true                        # true → preflight_check.py --poison (AEL) πριν το batch
```

Κανόνες εγκυρότητας στο ίδιο το plan (ο runner τα ελέγχει και σταματά με σαφές μήνυμα):
- `train_end` επιτρέπεται **μόνο** με `retrain: static` (αλλιώς αγνοείται → warn).
- `gate` ≠ `strict` → ρητό warn «academic = μόνο για σύγκριση με papers, όχι tradeable».
- `windows` < 2 **ή** (`algos` × `strategy`) δεν δίνουν ≥2 ανεξάρτητες συνθήκες → warn
  «αυτό το batch δεν μπορεί δομικά να πιάσει §2· θα βγει PENDING».

Το cell naming ακολουθεί την ήδη υπάρχουσα σύμβαση: `<window>_<algo>_<strategy>_<spec>.json`
στο `runs/<study>/` — συμβατό με `synthesize_ablation.py`.

## 3. Ροή εκτέλεσης (ντετερμινιστική)

```
1. LOAD & VALIDATE plan.yaml         → σφάλμα plan = exit πριν τρέξει οτιδήποτε
2. PREFLIGHT                         → preflight_check.py  (+ --poison αν poison:true)
                                       FAIL → exit αμέσως (leakage/env/TZ), κανένα training
3. BATCH (σειριακά, ΕΝΑ conda process):
     for window × algo × spec:
        run master_forecast/grid cell → runs/<study>/<cell>.json  (--out_json, --quiet)
        cell crash → κατέγραψε FAILED, ΣΥΝΕΧΙΣΕ (δεν σταματά όλο το batch)
     [conformal:true] → src.conformal πάνω στα point-forecast JSONs
4. SYNTHESIZE                        → synthesize_ablation.py: πίνακας ΔMAE + §2 pre-gate
5. LEDGER                            → scripts/build_run_ledger.py  (system python)
6. SUMMARY & STOP                    → τυπώνει: cells ok/failed, ΔMAE πίνακα, §2 status
                                       (PENDING/candidate — ΠΟΤΕ «ΔΕΚΤΟ»), και ρητή προτροπή:
                                       «Τρέξε validity-reviewer, μετά αποφάσισε/κατάθεσε ΕΣΥ.»
```

**Ένα conda process:** ο runner τρέχει τα cells **σειριακά** μέσα στο ίδιο process — τιμά τον
σκληρό κανόνα «μία ουρά», ανεξαρτήτως ποιος τον εκκίνησε.

**Detached εκκίνηση (μεγάλα batch):** το batch συνήθως >2-3 λεπτά, άρα εκκινείται OS-detached
με log, κατά το SKILL.md pattern (`Start-Process -WindowStyle Hidden ... -WorkingDirectory
<repo> -ArgumentList '-c', 'conda run ... python -m src.cycle_runner --plan <p> > logs/<study>.log 2>&1'`,
ASCII-only args). Παρακολούθηση με `Get-Content logs\<study>.log -Wait -Tail 20`.

**Failure handling:** ανά cell fail-safe (σημείωσε & συνέχισε). Το τελικό summary δίνει
`N_ok/N_total` + λίστα FAILED cells. Ένα preflight FAIL είναι η **μόνη** άμεση διακοπή.

## 4. Έξοδοι (artifacts)

| Artifact | Θέση | Ρόλος |
|---|---|---|
| Forecast JSONs | `runs/<study>/<cell>.json` | ωμά αποτελέσματα ανά cell (`--out_json`) |
| ΔMAE πίνακας + §2 | `results/<study>.csv` + stdout | ό,τι παράγει το `synthesize_ablation.py` |
| Conformal JSONs | `runs/<study>/*conformal*` | μόνο αν `conformal: true` |
| Run ledger | `results/run_ledger.csv` | ανανεωμένο auditability index |
| Log | `logs/<study>.log` | πλήρες trace της detached εκτέλεσης |
| Draft deposit | `runs/<study>/DRAFT_deposit.md` | προτεινόμενο κείμενο για last.md/ABLATION — **όχι** αυτόματη εγγραφή |

Το `DRAFT_deposit.md` είναι **βοήθημα**, όχι εγγραφή: ο άνθρωπος (μετά τον `validity-reviewer`)
αποφασίζει τι/αν θα μπει στα master MD. Σέβεται το single-writer principle (guard_edits.py).

## 5. Τι εγγυάται / τι όχι

**Εγγυάται:** ίδιο plan → ίδια cells/εντολές (reproducible)· κανένα cell χωρίς JSON trace·
preflight πριν από κάθε batch· καμία εγγραφή σε προστατευμένα paths ή master MD.

**ΔΕΝ εγγυάται (by design, μένει ανθρώπινο):** ουσιαστική εγκυρότητα των αριθμών (→ `validity-reviewer`)·
απόφαση αποδοχής/headline· επιλογή τι feature/window αξίζει (→ στάδιο 0, ο άνθρωπος γράφει το plan).

## 6. Testing (πριν εμπιστευτούμε τον runner)

Ο ίδιος ο runner είναι απλός· τεστάρεται **χωρίς** ακριβά trainings:
1. **plan-validation unit tests**: κακό plan (train_end με weekly, <2 windows, άγνωστο market)
   → σωστό warn/exit. Καθαρά συναρτησιακό, system python.
2. **dry-run mode** (`--dry-run`): τυπώνει τη λίστα cells + τις ακριβείς εντολές που ΘΑ έτρεχε,
   χωρίς να τρέξει κανένα training — έλεγχος ότι το wiring/naming είναι σωστό.
3. **1-cell smoke** σε μικρό window (π.χ. Δεκ-2025-only, 1 algo, 1 spec) → επιβεβαίωση ότι
   παράγει JSON + περνά στο synthesize + ledger, end-to-end, μία φορά.
4. **Reproducibility anchor**: baseline cell αναπαράγει το γνωστό ±0.05 (π.χ. default static Q1).

## 7. Αποφάσεις (κλειδωμένες 2026-07-07, brainstorm)

1. Φύση: **ντετερμινιστικός Python orchestrator** (όχι LLM agent). *(Επιλογή Α χρήστη)*
2. Εύρος v0: **στάδια 3→7**· ingest/rebuild (1-2) **εκτός**· accept (5) μόνο υπολογισμός· deposit (8) draft. *(Επιλογή Α χρήστη)*
3. Μονάδα εκτέλεσης: **ένα πάτημα = όλη η σύγκριση** (batch από plan.yaml), όχι μία μεμονωμένη πρόβλεψη. *(Επιλογή Β χρήστη)*
4. Εκτέλεση: σειριακά (ένα conda process), detached για μεγάλα batch, per-cell fail-safe.
5. Καμία αυτο-αποδοχή, καμία αυτόματη εγγραφή σε master MD ή git — μόνο draft.
6. Γενικότητα: market/task = παράμετροι· availability μένει στο `feature_availability.py` (§0.1).

Εκκρεμούν προς το plan υλοποίησης: ακριβές CLI του `cycle_runner.py`· αν το batch layer
θα καλεί `run_master_grid` (ένα process, πολλά cells) ή `master_forecast` ανά cell (πιο
granular fail-safe)· ακριβές schema του `DRAFT_deposit.md`.

## 8. Σχέση με υπάρχοντα assets

Καλούμενα (αμετάβλητα): `.claude/skills/energy-forecast/scripts/preflight_check.py` ·
`src/run_master_grid.py` / `src/master_forecast.py` · `scripts/synthesize_ablation.py` ·
`src/conformal.py` · `scripts/build_run_ledger.py`. Σεβαστά: guard_edits.py (προστατευμένα
paths + single-writer master MD)· σκληροί κανόνες energy-forecast (no Optuna, strict gate,
ΕΝΑ conda). Ο runner **γύρω** από τον πυρήνα, ποτέ **μέσα** του.
