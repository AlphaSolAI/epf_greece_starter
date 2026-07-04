# SESSION HANDOFF — energy trading agent (GR EPF/STLF)

## ✅ UPDATE 2026-07-04(γ) — OVERNIGHT PROTOCOL ΟΛΟΚΛΗΡΩΘΗΚΕ (P1→P2→P3→P5→P4→P6, autonomous /goal)

Το πλήρες 6-φασικό πρωτόκολλο (`ABLATION_PLAN.md §OVERNIGHT PROTOCOL`) εκτελέστηκε αυτόνομα,
χωρίς διακοπή, ένα conda process τη φορά. Πλήρη νούμερα/πίνακες: `ABLATION_PLAN.md §P1-RESULTS`
έως `§P6-RESULTS`. Το report (`reports/ablation_20260702/index.html`) ξαναπαράχθηκε με τα
διορθωμένα νούμερα, και το `.claude/skills/energy-forecast/SKILL.md §11` ενημερώθηκε.

### 🔒 Τι ΚΛΕΙΔΩΣΕ (ΔΕΚΤΟ κατά τα κριτήρια του §3)

1. **Οριστικό headline: LGBM recursive weekly, `default` = 15.17 €/MWh** (Q1 2026, strict gate,
   χωρίς xborder). Seed-robust (std≈0.11, seeds 42/43/44) και επιβεβαιωμένο σε 2ο ανεξάρτητο
   window (Μάρτιος 2026: weekly 17.67 vs monthly 17.97, ίδιο πρόσημο υπεροχής retrain cadence).
2. **xborder-lagged ΑΠΟΡΡΙΠΤΕΤΑΙ από το default set** — βλάπτει τον χειμώνα σε 3/3 retrain
   cadences (static +0.42, monthly +0.13, weekly +0.41), παρότι βοηθάει το καλοκαίρι (−0.52, 1
   σημείο — δες PENDING παρακάτω).
3. **meteo = καθαρό strategy-effect, οριστικά επιβεβαιωμένο (4/4 σημεία)**: βοηθάει ΠΑΝΤΑ στο
   direct (LGBM-Δεκ +0.55, LGBM-Q1 +0.49, XGB-Q1 +0.29) και βλάπτει ΠΑΝΤΑ στο recursive. Το πιο
   στιβαρό strategy-dependent εύρημα όλης της μελέτης.
4. **genlags/resfc πλέον cross-model επιβεβαιωμένα σε 3 αλγόριθμους** (LGBM, XGB, MLP) — genlags
   πολύτιμο παντού, resfc πολύτιμο ειδικά το καλοκαίρι (XGB summer επιβεβαιώνει το LGBM εύρημα).
5. **SS-linear ΔΕΝ είναι redundant με retrain** (διόρθωση προηγούμενου λάθους συμπεράσματος) —
   βοηθάει και στο monthly (−0.243) σε καθαρό config, όχι μόνο static.
6. **Weighted-by-1/MAE ensemble ΑΠΟΡΡΙΠΤΕΤΑΙ** (με τίμιο out-of-sample calibration, Δεκ→Ιαν-Φεβ,
   κανένα ensemble δεν κερδίζει το καλύτερο μεμονωμένο μοντέλο) — κλείνει οριστικά το §4.3.
7. **Direct στρατηγική παραμένει σαφώς χειρότερη από recursive** στο πλήρες Q1 (19.5 vs 16.1) —
   retrain cadence δεν βοηθάει το direct (αντίθετα με recursive). Cross-strategy ensemble
   (direct+recursive) χάνει καθαρά (+0.523). **Απόφαση προϊόντος: recursive για DAM.**
8. **LEAR αντιδράει γνήσια αντίθετα από τα δέντρα** (αφαίρεση resfc/fuel το βελτιώνει, L1
   regularization effect) — μοντελο-εξαρτώμενο, όχι artifact κακής μέτρησης.

### 🟡 Τι έμεινε PENDING (ρητά, όχι εγκαταλελειμμένο)

- **Καλοκαιρινό xborder θετικό σήμα** (−0.52, static μόνο) — χρειάζεται cadence-robustness
  (monthly/weekly στο καλοκαίρι) ή 2ο seed πριν δηλωθεί ως γνήσιο εύρημα εποχιακότητας.
- **weekly LGBM+XGB ensemble** (15.023, Δ=−0.148 vs LGBM 15.171) — ακριβώς στο όριο noise floor,
  1 μόνο σημείο· χρειάζεται 2ο window ή seeds πριν δηλωθεί κέρδος.
- **resfc-in-direct**: LGBM (ουδέτερο, −0.04) vs XGB (βλάπτει, −0.32) διαφωνούν — algo-dependent,
  χρειάζεται 3ο σημείο (π.χ. MLP δεν υποστηρίζει direct, οπότε μόνο LGBM/XGB δυνατά εδώ).
- **fuel window-size confound ΜΕΣΑ στο direct strategy** (Δεκ: −0.68 βλάπτει· Q1: +0.02 ουδέτερο)
  — νέο, αδιευκρίνιστο ερώτημα, όχι στρατηγικό αλλά μέγεθος-window εφέ.
- **LSTM seq2seq calibration bug** (MAE=44, ποτέ αρνητικές τιμές) — παραμένει ανοιχτό, χαμηλή
  προτεραιότητα μέχρι στοχευμένο debugging.
- Ένα seed παντού εκτός του headline (P6) — οριακά ΔMAE σε άλλα σημεία (π.χ. ensemble median στο
  πλήρες window) χρειάζονται ακόμα 2-3 seeds πριν δηλωθούν στη διπλωματική.

### ➡️ Έτοιμο επόμενο βήμα (κατά το §4 του πρωτοκόλλου)

Με το feature/model/strategy selection πλέον κλειδωμένο (LGBM recursive weekly, `default`,
15.17 €/MWh, robust), η μεγάλη εικόνα (§5) λέει το επόμενο στάδιο είναι **probabilistic layer
(conformal prediction) πάνω στο κλειδωμένο point-forecast config** — το σημείο πρόβλεψης έχει
πλέον αρκετή σιγουριά (seed-robust, 2 windows, cross-model confirmed features) για να αξίζει η
επένδυση σε calibrated intervals πριν προχωρήσει το product στο layer L3-L5 του
`SYSTEM_DESIGN_TRADING_AGENT.md`. Εναλλακτικά, χαμηλότερης προτεραιότητας αλλά ανοιχτά:
(α) το προϋπάρχον ύποπτο `solar_fc_dayahead` 2ωρο shift (§last.md, code review 2026-07-03) με το
ίδιο §1 πρωτόκολλο ελέγχου δημοσίευσης/lag-scan· (β) πλήρες `task=load` ablation (μόνο το E2
meteo έχει γίνει εκεί μέχρι τώρα).

## 🌙 UPDATE 2026-07-04(β) — OVERNIGHT PROTOCOL ΣΕ ΕΞΕΛΙΞΗ (autonomous /goal, δεν ρωτάει) — ΙΣΤΟΡΙΚΟ, βλ. σύνοψη πάνω

**Pre-flight §1: ΠΕΡΑΣΕ.** OneDrive.exe ξεκινήθηκε (δεν έτρεχε), parquet επιβεβαιώθηκε ΜΟΝΟ με
`xb_*_lag{24,48,168}` (καμία same-day στήλη), LGBM default static Q1 reproduced MAE=16.0963
(αναμενόμενο 16.10±0.05). Git παραμένει σπασμένο (γνωστό, δεν το άγγιξα, εκτός scope).

**P1 (xborder-lagged closure): ΟΛΟΚΛΗΡΩΘΗΚΕ.** Δεν έκλεισε καθαρά με «≥0 παντού» — winter (Q1)
βλάπτει σταθερά σε static/monthly/weekly (+0.42/+0.13/+0.41) αλλά **summer βοηθάει** (−0.52,
ΝΕΟ εύρημα, μόνο 1 σημείο). Συντηρητική απόφαση: xborder ΑΠΟΡΡΙΠΤΕΤΑΙ από το default set (η
headline σεζόν είναι χειμώνας, εκεί βλάπτει με συνέπεια)· το summer θετικό αποτέλεσμα μένει
ανοιχτό ερώτημα για μελλοντικό session. **Headline παραμένει: LGBM recursive weekly, `default`
= 15.17 €/MWh.** Πλήρες: `ABLATION_PLAN.md §P1-RESULTS`.

**P2 (cross-model ablation): ΟΛΟΚΛΗΡΩΘΗΚΕ.** XGB summer 10-spec: επιβεβαιώνει πλήρως resfc/
genlags value το καλοκαίρι (2ος αλγόριθμος συμφωνεί, ΔΕΚΤΟ). LEAR mini 6-spec: **γνήσια
αντίθετη συμπεριφορά** — αφαίρεση resfc/fuel ΒΕΛΤΙΩΝΕΙ το LEAR (L1 regularization effect, model-
specific, όχι artifact). MLP mini 3-spec: genlags πολύτιμο (3ος αλγόριθμος συμφωνεί, οριστικά
ΔΕΚΤΟ), bare core νικάει καθαρά το πλήρες default (kitchen-sink βλάπτει ακόμα πιο έντονα στο
MLP). Πλήρες: `ABLATION_PLAN.md §P2-RESULTS`.

**P3 (ensembles σωστά): ΟΛΟΚΛΗΡΩΘΗΚΕ.** Με τίμιο out-of-sample calibration (weights από Δεκ,
eval στο Ιαν-Φεβ), το weighted-by-1/MAE ensemble ΔΕΝ κερδίζει κανένα μεμονωμένο μοντέλο —
**ΑΠΟΡΡΙΦΘΗΚΕ**, κλείνει το §4.3 ανοιχτό ζήτημα. Νέο: weekly LGBM+XGB (2 μέλη σχεδόν ισοδύναμα)
ensemble δίνει 15.023 (Δ=−0.148 vs LGBM 15.171) — PENDING, ακριβώς κάτω από noise floor,
ενθαρρυντικό αλλά όχι ακόμα δηλωμένο κέρδος. Πλήρες: `ABLATION_PLAN.md §P3-RESULTS`.

**P5 (SS re-check σε έγκυρο config): ΟΛΟΚΛΗΡΩΘΗΚΕ.** Διορθώνει το προηγούμενο συμπέρασμα: το
SS-linear ΔΕΝ είναι redundant με το retrain — σε καθαρό `default` (χωρίς xborder) βοηθάει και
στο monthly (ΔMAE=−0.243) όπως και στο static (−0.266), και τα δύο πάνω από noise floor. Το
παλιό «redundant» (§Stacking-RESULTS) ήταν artifact του xborder-contaminated config. Πλήρες:
`ABLATION_PLAN.md §P5-RESULTS`.

**P4 (direct πυλώνας): ΟΛΟΚΛΗΡΩΘΗΚΕ.** Πρώτη φορά direct στρατηγική στο ΠΛΗΡΕΣ Q1 (πριν μόνο
Δεκ-only). direct-LGBM Q1 baseline=19.495 (πολύ χειρότερο από recursive 16.096). **Πιο στιβαρό
νέο εύρημα: meteo βοηθάει ΠΑΝΤΑ στο direct (4/4: LGBM-Δεκ/Q1, XGB-Q1) και βλάπτει ΠΑΝΤΑ στο
recursive** — καθαρό strategy-effect, οριστικά ΔΕΚΤΟ. Retrain cadence ΔΕΝ βοηθάει στο direct
(αντίθετα με recursive). Cross-strategy ensemble (direct+recursive) ΑΠΟΡΡΙΦΘΗΚΕ — χάνει (+0.523),
το direct μέλος πολύ πιο αδύναμο. **Απόφαση: recursive παραμένει η στρατηγική επιλογή για DAM.**
Πλήρες: `ABLATION_PLAN.md §P4-RESULTS`.

Επόμενο: P6 (robustness: seeds + 2ο test window) → τελικό report/SKILL.md/last.md.

## ⛔ UPDATE 2026-07-04 — XBORDER ΑΚΥΡΩΘΗΚΕ (leakage) — ΥΠΕΡΙΣΧΥΕΙ των παρακάτω xborder claims

Ερώτηση χρήστη («πότε δημοσιεύονται; έχουμε κοινό gate;») αποκάλυψε ότι το xborder same-day
ήταν **εννοιολογικό leakage**: BG/IT-SUD τιμές ημέρας D βγαίνουν από το ΙΔΙΟ SDAC auction με το
target (δημοσίευση ~13:00 CET D-1 = ΜΕΤΑ το gate 12:00). Fix δομικό στο `data.py`: same-day xb
στήλες δεν μπαίνουν καν στο parquet — μόνο `xb_*_lag{24,48,168}`. **Έντιμο αποτέλεσμα: lagged
xborder ΒΛΑΠΤΕΙ (static Q1: 16.52 vs default 16.10).** Το −0.72 «όφελος» ήταν εξ ολοκλήρου
leakage. **Headline επανέρχεται: LGBM recursive weekly `default` = 15.17 €/MWh.** ΑΚΥΡΑ τα
14.43/15.02 και όλες οι xborder-θετικές αναφορές στα docs/report (το report θα ξαναπαραχθεί).
ΑΝΕΠΗΡΕΑΣΤΑ: όλα τα core ablations/E1-E2/SS/καλοκαίρι (δεν είδαν ποτέ xb στήλες). Το «SS
redundant» θέλει re-check (μετρήθηκε σε xborder configs). Πλήρης ετυμηγορία + νέο πρωτόκολλο
pre-flight ελέγχου features + OVERNIGHT PROTOCOL (5-φάσεις): `ABLATION_PLAN.md` (κορυφή).

## ✅ UPDATE 2026-07-03(γ) — Code review βρήκε πραγματικό bug, διορθώθηκε, headline άντεξε

Το `/code-review` (χειροκίνητο, git δεν λειτουργεί) βρήκε **1ωρη μετατόπιση timestamps** στο
`fetch_entsoe_xborder.py` (BG/IT-SUD DAM τιμές) — ανιχνεύθηκε με cross-correlation lag-scan
κατά της πραγματικής τιμής GR (peak σε lag+1 αντί lag=0). **Κυρίως ΟΧΙ leakage** (23/24 ώρες/
ημέρα: ίδιο D-1 auction, απλά λάθος αντιστοίχιση ώρας) — ΜΟΝΟ η ώρα 23:00 κάθε ημέρας (~4% των
γραμμών) είχε γνήσιο μικρό leakage (τιμή από το ΕΠΟΜΕΝΟ auction). Διορθώθηκε (`-1h`, μετά από
μία αποτυχημένη πρώτη απόπειρα με λάθος πρόσημο +1h — ξανα-ελέγχθηκε πριν κλειδωθεί), πλήρες
re-fetch+rebuild, verified (peak τώρα ακριβώς lag=0). **Headline ξανατρέχτηκε: 14.423 (πριν:
14.426) — διαφορά 0.003, ουσιαστικά μηδενική.** Το εύρημα άντεξε (τιμές ρεύματος κινούνται ομαλά
ώρα-προς-ώρα, η "λάθος" ώρα ήταν ήδη σχεδόν ταυτόσημη πληροφορία). Πλήρη λεπτομέρεια:
`ABLATION_PLAN.md` (νέα ενότητα πριν το `§Stacking-RESULTS`).
⚠️ Δεν ξανατρέχτηκαν με διορθωμένα δεδομένα: αρχικό xborder ablation, confirmation Μάρτιος/
καλοκαίρι, SS×xborder, seeds — μόνο το headline verified explicit (effect size της διόρθωσης
αμελητέο, οπότε πρακτικά έγκυρα, τυπικά "not re-verified post-fix").

**Επίσης βρέθηκε (code review), ΠΡΟΫΠΑΡΧΟΝ, ΟΧΙ κάτι που έφτιαξα σήμερα**: `solar_fc_dayahead`
(2024-2025, μέρος του entsoe_extra_hourly.parquet από ΠΡΙΝ αυτό το session) έχει peak στις 10:00
ενώ το πραγματικό solar (gen_solar_lag24) στις 12:00 — πιθανή 2ωρη μετατόπιση στο ΠΑΛΙΟ κομμάτι
των δεδομένων forecast. ΔΕΝ διερευνήθηκε περαιτέρω (εκτός scope σημερινού review), σημειώνεται
ως ανοιχτό εύρημα υψηλής προτεραιότητας — επηρεάζει την ομάδα `resfc` που μόλις βρήκαμε πολύτιμη
το καλοκαίρι.

## 🔴 UPDATE 2026-07-03(β) — Infrastructure gaps (σημαντικό, πριν συνεχίσεις)

**1. Το git repo φαίνεται σπασμένο.** `.git/` υπάρχει με πραγματικά εσωτερικά (HEAD, objects,
config, COMMIT_EDITMSG, worktrees/) — δεν είναι άδειο ή fake. Αλλά `git status`/`git log` από
το project root αποτυγχάνουν σταθερά με `fatal: not a git repository (or any of the parent
directories): .git`, ακόμα και με clean retry+timeout. Δεν το άγγιξα (repair git = ρίσκο αν γίνει
λάθος, δεν μου ζητήθηκε). **Πρακτική συνέπεια**: καμία από τις σημερινές αλλαγές (νέα src modules,
docs, ablation results) δεν μπορεί να γίνει commit αυτή τη στιγμή — κανένα safety net version
control. Σύσταση: άνοιξε plain terminal (όχι μέσα από αυτό το session) και δοκίμασε `git status`
εκεί· αν αποτύχει κι εκεί, πιθανό αίτιο stale `index.lock` (υπάρχει ένα, dated Jun 12) ή
προβληματικό `worktrees/` entry — μην κάνεις `rm -rf .git` χωρίς να καταλάβεις πρώτα τι έφταιξε.
**2. Δεν υπάρχει `environment.yml`/`requirements.txt` πουθενά στο repo.** Το conda env `epf`
απέκτησε ΤΟΥΛΑΧΙΣΤΟΝ 2 νέα πακέτα σήμερα (`entsoe-py`, `matplotlib`) χωρίς να καταγραφούν πουθενά.
Για reproducibility (και για το παράρτημα της διπλωματικής) θα άξιζε `conda env export -n epf >
environment.yml` όποτε βρεθεί ευκαιρία.
**3. Υπάρχει ξεχωριστό `.venv/`** στο project root, παράλληλα με το conda env `epf` — πιθανή πηγή
σύγχυσης (ποιο env τρέχει π.χ. το dashboard/VS Code). Δεν το άγγιξα, απλά το σημειώνω.
**4. Root-level debris** (άσχετα με σημερινή δουλειά, pre-existing): `nul` (105B, στραβό
redirection artifact, Μάρτιος), `texput.log` (LaTeX artifact, 1 Ιουλ), δύο
`Νέο Έγγραφο κειμένου*.txt` (είναι απλά αντίγραφα δύο παλιότερων replies μου που αποθήκευσες
χειροκίνητα — όχι πηγαίο υλικό, ασφαλή). Χαμηλή προτεραιότητα cleanup.
**5. Πλήρης πίνακας κάλυψης ablation ανά μοντέλο/στρατηγική** (ποιο μοντέλο πέρασε από ablation,
ποιο όχι, γιατί) → `ABLATION_PLAN.md §9` (ενημερώθηκε 2026-07-03) — μη το ξαναγράψεις εδώ.

## CHANGELOG (τι άλλαξε σε αρχεία/πλάνο — σύνοψη όλου του session, 2026-07-03)

**Νέα src/ modules**: `feature_availability.py` (leakage gate + feature groups, αργότερα
έσπασε forecast/crosslags σε resfc/loadfc/genlags/loadlags/other + πρόσθεσε engfc/xborder) ·
`master_forecast.py` (κύριο engine· αργότερα +`--ss/--ss_decay/--ss_rounds`, +`--n_estimators`,
+`--seed`, +`--algo lear`, fix retrain-window bug για monthly/weekly) · `scheduled_sampling.py`
(νέο, SS engine 3 σχήματα) · `run_master_grid.py`, `run_ablation.py` (batch drivers) ·
`make_ensemble.py` · `lstm_models.py` · `fetch_entsoe_generation.py`, `fetch_entsoe_dayahead.py`,
`fetch_entsoe_xborder.py`, `build_real_load_forecast.py` (fetchers) · `make_ablation_report.py`
(matplotlib report generator, νέο σήμερα).

**Τροποποιήσεις σε υπάρχον κώδικα**: `data.py` — `_read_entsoe_csv` fix (Windows OSError σε
non-ASCII path + wrong-sep CSVs μέσω open()-handle αντί για path)· join του `xborder_hourly.parquet`
στο `process_hourly`. `master_forecast.py` — bug fix: `retrain=monthly/weekly` αγνοούσε το
`--train_end` λάθος (χρησιμοποιούσε το ως cap αντί για expanding window) → διορθώθηκε.

**Docs — δημιουργήθηκαν**: `MASTER_PIPELINE_DESIGN.md`, `SYSTEM_DESIGN_TRADING_AGENT.md`,
`.claude/skills/energy-forecast/SKILL.md`, `ABLATION_PLAN.md`, `last.md` (αυτό εδώ) ·
`reports/ablation_20260702/` (5 PNG + index.html, νέο σήμερα).

**Docs — εξέλιξη πλάνου (`ABLATION_PLAN.md`)**: ξεκίνησε ως 3-μερές σχέδιο (LGBM/XGB Q1 +
direct-LGBM Δεκ) → προστέθηκε Μέρος 1b (καλοκαίρι, μετά από ένσταση χρήστη για season bias) →
προστέθηκαν E1/E2, Scheduled Sampling (Φάση 2, με SS×features πολιτική retest), Φάση 3
(engfc/xborder) → μετά την αυτόνομη εκτέλεση προστέθηκαν 8 `§X-RESULTS` ενότητες (μία ανά
βήμα, με ωμά αριθμητικά + ερμηνεία) → μετά την αξιολόγηση προστέθηκε `§Stacking-RESULTS`
(follow-up battery: xborder×weekly/SS/seeds/καλοκαίρι) → σήμερα ενημερώθηκε το `§9` (honesty
section) με πίνακα κάλυψης ανά μοντέλο/στρατηγική.

**Docs — ενημερώθηκαν σήμερα (2026-07-03) για να μην είναι stale**: `SYSTEM_DESIGN_TRADING_AGENT.md`
(§3 data table: xborder row done· §4.3 ensemble: πρόσθεσε το median-οριακά-κερδίζει εύρημα·
§6 trade-offs: LEAR row αναθεωρήθηκε πλήρως· §7 Roadmap: P0/P1 όλα ✅, νέα ανοιχτά items) ·
`MASTER_PIPELINE_DESIGN.md` (status pointer note: feature-group split, LEAR ενσωμάτωση) ·
`.claude/skills/energy-forecast/SKILL.md` (§11 headline config, ×2 ενημερώσεις: πρώτα 15.02
monthly+xborder, μετά 14.43 weekly+xborder).

**Data artifacts (νέα, όχι κώδικας)**: `data/raw/generation/*.csv` (2015-2026, ήταν άδειο) ·
`data/processed/entsoe_extra_hourly.parquet` (επεκτάθηκε έως 2026-07-02) ·
`data/processed/load_forecast_hourly.parquet` (αντικαταστάθηκε: συνθετικό→πραγματικό, 2015-2026) ·
`data/processed/xborder_hourly.parquet` (νέο, 2017-2026) · `data/processed/hourly.parquet` /
`hourly_load.parquet` (rebuilt με backup+σύγκριση, 2 φορές: μία για generation/entsoe_extra/
load_fc, μία για xborder).

## ⭐ UPDATE 2026-07-03 — Αξιολόγηση autonomous session + Stacking battery

Η αξιολόγηση του αυτόνομου session ΠΕΡΑΣΕ (10/10 spot-checks των νούμερων στα ωμά JSONs).
Οπτική αναφορά για τον χρήστη: **`reports/ablation_20260702/index.html`** (5 figures + πίνακες
— παράγεται από `src/make_ablation_report.py`, τα νούμερα hardcoded-verified).

Follow-up stacking battery (6 runs, `confirm_out2/`) — βλ. `ABLATION_PLAN.md §Stacking-RESULTS`:
- **ΤΕΛΙΚΟ HEADLINE: LGBM recursive WEEKLY + `default,xborder` = 14.43 €/MWh** (Q1 2026, strict).
- xborder όφελος σταθερό ≈−0.7 σε κάθε cadence, −1.20 καλοκαίρι → 5/5, 3 windows → **μπαίνει
  οριστικά στο default set** (η αλλαγή στο `feature_availability.DEFAULT_GROUPS` ΔΕΝ έχει γίνει
  ακόμα — συνειδητά, θέλει απόφαση χρήστη γιατί αλλάζει τη σημασία του "default" σε όλα τα specs).
- **SS = redundant με φρέσκο retrain** (−0.09 πάνω σε xborder+monthly, εντός seed noise ±0.05).
  Χρήσιμο μόνο σε static καθεστώς. Τα SS×features interactions (resfc boost κ.λπ.) παραμένουν
  έγκυρα επιστημονικά ευρήματα για τη διπλωματική.
- Seed robustness: std≈0.05 (seeds 42/43/44) → noise floor 0.15 ήταν συντηρητικά σωστό.

Ανοιχτά (νέα λίστα): LSTM bias fix · weighted/median ensemble με 2-3 seeds · weekly×xborder
καλοκαίρι · summer meteo replication · xborder→DEFAULT_GROUPS απόφαση · task=load με xborder;

**Κάλυψη ανά μοντέλο (κρίσιμο, βλ. `ABLATION_PLAN.md §9` για πλήρη πίνακα)**: μόνο το LGBM έχει
πλήρη ablation. Το XGB πήρε δική του ανεξάρτητη 18-spec Q1 ablation (όχι αντιγραφή) αλλά ΔΕΝ
ξανατεστάρε μετά (καλοκαίρι/E1-E2/SS/xborder/stacking = LGBM-only). LEAR/MLP/LSTM ΠΟΤΕ δεν πέρα-
σαν από ablation — έτρεξαν μόνο με `default`, ούτε καν με xborder. Στρατηγική: όλα recursive
πλην μιας εξαίρεσης (direct-LGBM Δεκ)· SS καλωδιωμένο ΜΟΝΟ για recursive στον κώδικα.

> Σκοπός αυτού του αρχείου: να μη χρειαστεί μια νέα συνομιλία να ξαναδιαβάσει 400+ μηνύματα.
> Διαβάζεται ΜΑΖΙ με: `.claude/skills/energy-forecast/SKILL.md` (κανόνες/εντολές),
> `ABLATION_PLAN.md` (το ενεργό πρόγραμμα πειραμάτων + RESULTS), `MASTER_PIPELINE_DESIGN.md`
> (anti-leakage θεωρία), `SYSTEM_DESIGN_TRADING_AGENT.md` (προϊοντική αρχιτεκτονική).
> Δεν επαναλαμβάνω κώδικα εδώ — μόνο ευρήματα, αποφάσεις, και το «γιατί».

## ΤΕΛΙΚΗ ΣΥΝΟΨΗ — αυτόνομο session 2026-07-02 (όλα τα 9 βήματα ΟΛΟΚΛΗΡΩΘΗΚΑΝ)

Αυτό το session έτρεξε αυτόνομα (`/goal`, χωρίς διακοπή) όλα τα 9 βήματα της λίστας
προτεραιότητας που είχε μείνει εκκρεμής. Σειρά αποτελεσμάτων:

**1. Data (xborder rebuild)** — `hourly.parquet` πήρε τις τιμές γειτόνων BG/IT-SUD
(`xb_price_bg`, `xb_price_itsud`). Backup+σύγκριση: 0 αλλαγές σε υπάρχουσα στήλη/γραμμή.

**2. Strategy vs window (meteo confound, ΛΥΘΗΚΕ)** — controlled recursive-LGBM στο ΙΔΙΟ Δεκ
window με το ήδη-τρεγμένο direct-Δεκ: το meteo αντιστρέφει πρόσημο (recursive −0.23 vs direct
+0.55) ΜΕ ΙΔΙΟ window → **η ασυμφωνία ήταν strategy effect, ΟΧΙ season/window effect.**
genlags/fuel/loadlags συνεπή και στις δύο στρατηγικές.

**3. Season effect (resfc, ΕΠΙΒΕΒΑΙΩΘΗΚΕ η ανησυχία χρήστη)** — καλοκαιρινή αναπαραγωγή (Ιουν-Αυγ
2025): το `resfc` (DA solar/wind/gen forecast) ήταν σχεδόν άχρηστο τον χειμώνα αλλά είναι από τις
πιο πολύτιμες ομάδες το καλοκαίρι (ΔMAE leave-one-out +1.09, additive −2.45 — μεγαλύτερο κέρδος
όλης της μελέτης). Το χειμωνιάτικο Q1-window ΥΠΟΕΚΤΙΜΟΥΣΕ συστηματικά τις solar-ευαίσθητες ομάδες.

**4+8. xborder — μεγάλο νέο θετικό feature, ΕΠΙΒΕΒΑΙΩΜΕΝΟ σε 2 windows** — Q1 στατικό: ΔMAE
−0.72 (−4.5%). Monthly retrain: −0.78 (Q1) / −0.35 (Μάρτιος 2026, ανεξάρτητο 2ο out-of-sample
μήνα). **Νέο headline config: LGBM recursive monthly, `default,xborder` → 15.02 €/MWh**
(αντικαθιστά το προηγούμενο "weekly, default, 15.17" — το xborder gain ξεπερνά το κέρδος από
αναβάθμιση retrain cadence monthly→weekly).

**5. E1 (capacity×meteo, υπόθεση χρήστη ΑΠΟΡΡΙΦΘΗΚΕ)** — περισσότερα estimators ΔΕΝ ξεκλειδώνουν
το meteo· το βλάπτουν ΟΛΟ ΚΑΙ ΠΕΡΙΣΣΟΤΕΡΟ, μονότονα (−0.02→+0.26→+0.44) — overfitting στο
reanalysis noise, όχι βαθύτερο σήμα. **E2 (meteo στο load, ΕΠΙΒΕΒΑΙΩΘΗΚΕ)** — meteo ΔMAE=−33 MW
(~26%) στο φορτίο, η πιο πολύτιμη ομάδα ΟΛΟΥ του study — επιβεβαιώνει ότι το meteo δρα έμμεσα
στην τιμή μέσω load/RES.

**6+7. Scheduled Sampling — πρώτο ποτέ end-to-end τρέξιμο, δούλεψε, και έδωσε το πιο εντυπωσιακό
εύρημα του session** — `SS-linear` νικητής (μόνο αυτό βελτιώνει ΚΑΙ το συνολικό MAE ΚΑΙ ειδικά
τα μακρινά offsets 17-24h, ακριβώς η exposure-bias υπογραφή). SS×features interaction
**επιβεβαιώθηκε δραματικά**: `resfc`/`loadfc` (πάντα-αξιόπιστα day-ahead exogenous) γίνονται
ΠΟΛΥ πιο πολύτιμα υπό SS (resfc: −0.04→+1.10, loadfc: οριακό→+0.46) — το μοντέλο ακουμπάει σε
ό,τι δεν corrupted όταν τα y-lags γίνονται θορυβώδη. `loadlags` κάνει το αντίθετο ταξίδι
(+0.22→−0.32, helpful→harmful).

**9. LEAR/LSTM/ensemble ξανατρέξιμο** — LEAR σταθερό (19.49, όχι artifact, παραμένει χειρότερο
από δέντρα). Ensemble: mean ΞΑΝΑ χάνει (επιβεβαιώνει παλιό εύρημα)· median κερδίζει οριακά
(16.042 vs LGBM 16.096, κάτω από noise floor, κατευθυντικό μόνο). **LSTM seq2seq**: πρώτη φορά
ΟΛΟΚΛΗΡΩΝΕΤΑΙ χωρίς exception (90 blocks, 445.6s) αλλά ΚΑΚΗ ποιότητα — MAE=44.16 (χειρότερο
ΚΑΙ από LEAR), corr(actual,pred)=0.69 (μαθαίνει σήμα αλλά με ισχυρό θετικό bias, ποτέ δεν
προβλέπει αρνητικές τιμές). **ΔΕΝ είναι tradeable ακόμα** — χρειάζεται calibration fix.

**Τι μένει ανοιχτό για επόμενο session** (ρητά, όχι εγκαταλελειμμένο):
- Weekly retrain του νέου headline config (`default,xborder`) — δοκιμάστηκε μόνο static/monthly.
- Καλοκαιρινή αναπαραγωγή του `meteo` (§2b specs 6-7 δεν συμπεριλήφθηκαν σκόπιμα εκεί).
- SS × monthly/weekly retrain combination — μόνο SS×static δοκιμάστηκε.
- Ένα μόνο seed παντού· οριακά ΔMAE (<0.15, π.χ. ensemble median) χρειάζονται 2-3 seeds πριν
  γίνουν δηλωμένο συμπέρασμα στη διπλωματική.
- LSTM seq2seq calibration bug (θετικό bias, καμία αρνητική τιμή) — debugging, όχι απλό ξανατρέξιμο.
- SS-informed feature set (`default,xborder,-loadlags` + SS-linear) δοκιμάστηκε μόνο static
  (15.35) — δεν συγκρίθηκε ακόμα με monthly retrain του ίδιου feature set.
- Νέο `default` σύνολο (με xborder μέσα μόνιμα) δεν έχει ενημερωθεί ακόμα στο `feature_availability.py`
  — το xborder παραμένει explicit opt-in ομάδα, όχι μέρος του baseline `default`.

Λεπτομέρειες/αριθμοί κάθε βήματος: `ABLATION_PLAN.md §RESULTS` (κάθε ενότητα). Ενημερώθηκε και
το `.claude/skills/energy-forecast/SKILL.md §11` με το νέο headline config.

## TL;DR κατάσταση (2026-07-02, βράδυ) — ΠΑΛΙΟ, πριν το autonomous session (βλ. σύνοψη πάνω)

Το νέο leakage-free pipeline (`master_forecast.py` + `feature_availability.py`) είναι πλήρες
και δουλεύει (lgbm/xgb/mlp/lear, recursive/direct/seq2seq, static/monthly/weekly). Τα δεδομένα
έγιναν πλήρη (2015-2026: generation, DA forecasts, real load_fc, +cross-border). Το bulletproof
feature-ablation (LGBM+XGB σε Q1, direct-LGBM σε Δεκ) μόλις ολοκληρώθηκε — βλ. `ABLATION_PLAN.md
§RESULTS`. Εκκρεμούν: xborder rebuild+test, Μέρος 1b (καλοκαίρι), E1/E2, Scheduled Sampling runs,
SS-retest οριακών ομάδων, LEAR/LSTM/ensemble δεν έχουν ξανατρέξει με τα v2 δεδομένα.

## Η διαδρομή — ευρήματα ΚΑΙ ανατροπές (με σειρά, με το «γιατί»)

1. **Ξεκίνημα**: ο χρήστης υποψιαζόταν ότι παλιά μοντέλα ίσως παραβίαζαν το DAM gate closure
   (D-1 12:00). Επιβεβαιώθηκε εν μέρει: CL/teacher-forced στρατηγικές ΗΤΑΝ oracle/μη-tradeable.
   Το «MIMO» ήταν στην πραγματικότητα Direct (MultiOutputRegressor = Ν ανεξάρτητα μοντέλα).
   → Χτίστηκε ΝΕΟ engine από το μηδέν με ρητό `GateSpec` (strict/academic) ως το μοναδικό
   σημείο αλήθειας για το τι επιτρέπεται πότε.

2. **Πρώτο Q1-2026 grid (πριν τα data fixes)**: το LEAR (γραμμικό, LASSO-AR) φάνηκε να ΚΕΡΔΙΖΕΙ
   τα δέντρα δραματικά, ειδικά Ιαν-Φεβ (regime shift, τιμή έπεσε 110→78 €/MWh). Η ερμηνεία τότε:
   «τα δέντρα δεν κάνουν extrapolation, το LEAR ναι». **Αυτό αποδείχτηκε ΛΑΘΟΣ ερμηνεία** —
   βλ. σημείο 5. Επίσης: naive ensemble (mean/median LGBM+XGB+MLP+LEAR) ΕΧΑΝΕ από το καλύτερο
   μεμονωμένο μοντέλο σε κάθε δοκιμή — δεν ξαναδοκιμάστηκε ακόμα με weighted-by-1/MAE εκδοχή.

3. **Data audit**: βρέθηκε ότι `data/raw/generation/` ήταν ΑΔΕΙΟΣ (τα gen features στο parquet
   ήταν από παλιό χαμένο download), το `entsoe_extra_hourly.parquet` σταματούσε 31/12/2025, και
   το `load_fc` ήταν ΣΥΝΘΕΤΙΚΟ (recursive πρόβλεψη του ΔΙΚΟΥ ΜΑΣ LGBM πάνω στο actual load,
   αποθηκευμένη σαν να ήταν «η επίσημη πρόβλεψη» — όχι leakage, αλλά ούτε πραγματική πληροφορία).

4. **Data fix (μεγάλο side-quest, ένα βράδυ)**: `ENTSOE_API_KEY` υπήρχε ήδη στο env — μηδέν
   χρειάστηκε από τον χρήστη. Βρέθηκαν και διορθώθηκαν 2 bugs στην πορεία: (α) το OneDrive
   client δεν έτρεχε → πολλά raw CSV ήταν "cloud-only" placeholders (attribute `O`), reads
   έσκαγαν με `OSError: [Errno 22] Invalid argument` — λύση: ξεκίνησε το OneDrive.exe· (β)
   `_read_entsoe_csv` στο `data.py` είχε latent parsing bug σε 10/12 raw load CSV. Μετά:
   - `fetch_entsoe_generation.py`: actual gen/type 2015-2026 (ήταν 100% κενό).
   - `fetch_entsoe_dayahead.py`: DA RES/gen forecast έως 2026-07-02 (ήταν κενό 2026).
   - `build_real_load_forecast.py`: ΑΝΑΚΑΛΥΨΗ ότι η ΠΡΑΓΜΑΤΙΚΗ δημοσιευμένη ΑΔΜΗΕ/ENTSO-E
     day-ahead load forecast ήταν ΗΔΗ κρυμμένη σε στήλη μέσα στα ήδη-κατεβασμένα raw load CSVs
     — καμία λήψη, μόνο σωστό parsing· αντικατέστησε το συνθετικό, κάλυψη 2015-2026 αντί 3 μηνών.
   - Rebuild έγινε με backup+σύγκριση πριν/μετά (2018-2022 ταίριαξαν σχεδόν ψηφίο-προς-ψηφίο
     με το χαμένο παλιό parquet → μηδενική απώλεια ιστορικού).

5. **ΑΝΑΤΡΟΠΗ — ξανατρέξιμο Q1-2026 με πλήρη δεδομένα**: LGBM static 23.84 → **16.10** €/MWh
   (-32%!). Η «κατάρρευση Φεβρουαρίου» των δέντρων (32.35 MAE) εξαφανίστηκε (17.57 με πλήρη
   δεδομένα). **Το «LEAR κερδίζει σε regime shift» ήταν κυρίως artifact των κενών 2026 στηλών
   που δηλητηρίαζαν τα δέντρα, όχι πραγματική ιδιότητα extrapolation.** Με πλήρη δεδομένα το
   LEAR είναι σαφώς χειρότερο (19.0 vs 15.2-15.8 σε weekly) σε Q1-2026 ΚΑΙ σε full-2025 (19.6 vs
   17.0-17.2). Ρόλος LEAR τώρα: φθηνό robustness/fallback baseline, όχι κύριο μοντέλο.
   Νέο headline config: **LGBM recursive weekly, default features → 15.17 €/MWh** (v2 δεδομένα,
   Q1 2026, strict gate). Retrain βοηθάει μονότονα αλλά με λογικό όφελος πλέον (16.10→15.80→15.17).
   → «forecast group βλάπτει» / «no-forecast καλύτερο» επίσης ΑΝΑΤΡΑΠΗΚΕ (ήταν artifact του gap).

6. **Bulletproof ablation (μόλις τώρα)**: ζητήθηκε λεπτότερη ανάλυση πηγών (gen_act, for_gen,
   for_load, ξεχωριστά+συνδυασμοί) — οι παλιές χοντρές ομάδες `forecast`/`crosslags` δεν το
   επέτρεπαν. Έσπασαν σε `resfc`/`loadfc`/`genlags`/`loadlags`/`other` (τα παλιά ονόματα μένουν
   ως umbrella aliases, 100% backward-compatible). Αποτελέσματα πλήρη στο `ABLATION_PLAN.md
   §RESULTS` — headline: `genlags` στιβαρά η πιο πολύτιμη ομάδα (συμφωνούν LGBM+XGB+direct)·
   λιτός πυρήνας `lags+calendar+resfc+genlags` (39-48 feat) ΚΕΡΔΙΖΕΙ το πλήρες `default` (152
   feat)· `dense` βοηθάει (ανατρέπει παλιά παραδοχή)· `meteo`/`loadlags`/`fuel` ασυνεπή μεταξύ
   LGBM/XGB/στρατηγικών → χρειάζονται SS-retest + το νέο controlled πείραμα (recursive-Δεκ-μόνο,
   ίδιο window με το direct-Δεκ, για να διαχωριστεί strategy-effect από window/season-effect).

7. **Ιδέες που εξετάστηκαν και αναβλήθηκαν (όχι εγκαταλελειμμένες, conditional)**:
   - Δικός μας RES-generation forecaster: **αντικαταστάθηκε** από φθηνότερη λύση —
     `resload_fc = load_fc − solar_fc − wind_fc` (ομάδα `engfc`, on-the-fly, μηδέν training).
     Δικός μας forecaster αξίζει ΜΟΝΟ αν αποδειχτεί ότι το ENTSO-E resfc είναι αδύναμο ενώ το
     gen_act δυνατό (θα φανεί από τα ήδη τρεγμένα ablations — προς το παρόν resfc ΕΙΝΑΙ χρήσιμο).
   - Cross-border (BG, IT_SUD DAM prices): fetcher χτίστηκε, backfill 2017-2026 ΚΑΤΕΒΗΚΕ
     (`xborder_hourly.parquet`, 83k rows) αλλά **ΔΕΝ έχει μπει ακόμα στο κύριο parquet ούτε
     τεσταριστεί** — σκόπιμα, για να μην ξαναγραφτεί το `hourly.parquet` κάτω από τα πόδια του
     τρέχοντος ablation chain. Rebuild εκκρεμεί.
   - Ensemble (weighted by 1/MAE αντί για plain mean): προτάθηκε, δεν υλοποιήθηκε ακόμα.

## Τι υπάρχει τώρα (engines, όχι κώδικας — δες τα ίδια τα αρχεία)

`src/feature_availability.py` (leakage gate + fine-grained feature groups) ·
`src/master_forecast.py` (κύριο engine, όλοι οι algos/strategies/retrain, +SS +n_estimators) ·
`src/scheduled_sampling.py` (SS engine, 3 decay σχήματα — μόλις καλωδιώθηκε, ΔΕΝ έχει τρέξει
ποτέ ακόμα end-to-end) · `src/run_master_grid.py`, `src/run_ablation.py` (batch drivers) ·
`src/make_ensemble.py` · `src/lstm_models.py` (seq2seq/recursive, validation ΔΕΝ έχει
ολοκληρωθεί ποτέ end-to-end) · `src/fetch_entsoe_generation.py`, `fetch_entsoe_dayahead.py`,
`fetch_entsoe_xborder.py`, `build_real_load_forecast.py` (data fetchers/builders).

## Autonomous run log (2026-07-02, ξεκίνησε από /goal, ΧΩΡΙΣ διακοπή για ερωτήσεις)

**Βήμα 1 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** `python -m src.data --task price`: backup (`_backup_20260702_prexborder/`),
rebuild, σύγκριση. Αποτέλεσμα: shape 153→155 στήλες, ΜΟΝΟ προστέθηκαν `xb_price_bg` +
`xb_price_itsud` (BG/IT-SUD DAM τιμές γειτόνων), ΜΗΔΕΝ αλλαγή σε καμία υπάρχουσα στήλη/γραμμή
(0 mismatched cells σε y/gen_solar_lag24/gen_wind_lag24/residual_load_lag24/load_fc/y_lag1/
gas_price/co2_price, όλα τα 80126 rows κοινά). Backup διαγράφηκε. `xborder` ομάδα τώρα έτοιμη
για χρήση σε `--features`.

**Βήμα 2 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** Controlled πείραμα recursive-LGBM σε Δεκ-2025-μόνο, ίδιες 10 specs με
direct-Δεκ. **Εύρημα-κλειδί: το meteo confound ήταν STRATEGY effect, όχι window effect** — με
ΙΔΙΟ window (Δεκ) το meteo βοηθάει στο direct (+0.55) αλλά βλάπτει στο recursive (−0.23), άρα η
recursive-Q1 vs direct-Δεκ ασυμφωνία δεν ήταν artifact εποχής. fuel/genlags/resfc συνεπή
πρόσημα και στις δύο στρατηγικές (μόνο μέγεθος διαφέρει)· loadlags συνεπές θετικό και στις δύο
(η LGBM/XGB Q1 ασυμφωνία μάλλον algo effect, όχι strategy effect)· loadfc παραμένει αδιάγνωστο
(και τα δύο κάτω από noise floor 0.15). Πλήρης πίνακας: `ABLATION_PLAN.md §RESULTS`.

**Βήμα 3 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** Καλοκαιρινή αναπαραγωγή (LGBM recursive static, Ιουν-Αυγ 2025, 10
solar-ευαίσθητα specs). **Επιβεβαιώθηκε η ανησυχία του χρήστη**: το `resfc` ήταν σχεδόν άχρηστο
τον χειμώνα αλλά είναι από τις πιο πολύτιμες ομάδες το καλοκαίρι (ΔMAE leave-one-out +1.09,
additive πάνω σε core −2.45 MAE — το μεγαλύτερο κέρδος όλης της μελέτης). `genlags` παραμένει
πολύτιμο και στις δύο εποχές (μικρότερο μέγεθος καλοκαίρι). `loadfc` τώρα 3/3 σημεία δείχνουν
ασθενές/αρνητικό — υποψήφιο για αφαίρεση. Πλήρης πίνακας: `ABLATION_PLAN.md §2b-RESULTS`.

## Επόμενα βήματα — ΑΚΡΙΒΩΣ όπως στο `ABLATION_PLAN.md`, με σειρά προτεραιότητας

**Βήμα 4 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** `default` vs `default,xborder`, LGBM recursive static Q1 2026:
**16.096 → 15.376 (ΔMAE −0.72, −4.5%) — μεγάλο θετικό εύρημα**, δεύτερο μεγαλύτερο effect όλης
της μελέτης μετά το resfc-καλοκαίρι. Οι τιμές γειτόνων (BG/IT-SUD DAM) φαίνεται να προσθέτουν
πραγματική πληροφορία. Σύσταση: xborder υποψήφιο για το default set, θέλει confirmation
(Βήμα 8 ή SS-retest). Πλήρες: `ABLATION_PLAN.md §8-RESULTS`.

1. ~~`python -m src.data --task price`~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω.
2. ~~Controlled πείραμα recursive-Δεκ~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω.
3. ~~Μέρος 1b καλοκαιρινή αναπαραγωγή~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω.
4. ~~default,xborder vs default~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω. xborder δείχνει μεγάλο θετικό effect.

**Βήμα 5 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** E1 (capacity×meteo, 6 runs): υπόθεση χρήστη ΑΠΟΡΡΙΦΘΗΚΕ καθαρά —
meteo βλάπτει ΟΛΟ ΚΑΙ ΠΕΡΙΣΣΟΤΕΡΟ με μεγαλύτερο n_estimators (μονότονη τάση −0.02→+0.26→+0.44),
overfitting στο reanalysis noise, όχι "ξεκλείδωμα" σήματος. E2 (meteo στο load, 3 runs):
ΕΠΙΒΕΒΑΙΩΘΗΚΕ πλήρως η θεωρητική πρόβλεψη — meteo ΔMAE=−33 MW (~26%) στο load, η πιο πολύτιμη
ομάδα σε ΟΛΟ το ablation study, επιβεβαιώνει ότι το meteo δρα έμμεσα στην τιμή μέσω load/RES.
Πλήρες: `ABLATION_PLAN.md §5-RESULTS & 6-RESULTS`.
5. ~~E1+E2~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω.

**Βήμα 6 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** Πρώτο end-to-end τρέξιμο `scheduled_sampling.py` — δούλεψε σωστά
από την πρώτη φορά, κανένα σφάλμα (⚠️ σημείωση: χρειάστηκε να γραφτεί το debug script σε αρχείο
αντί για `conda run python -c <multiline>`, όπως προειδοποιεί το SKILL.md — το ξέχασα αρχικά και
έσκασε με `NotImplementedError`, το ξαναδιόρθωσα). **SS-linear νικητής καθαρά**: μόνο αυτό
βελτιώνει ΚΑΙ το συνολικό MAE (16.10→15.83) ΚΑΙ ειδικά τα μακρινά offsets (17-24h: 17.59→17.36,
Δ_far > Δ_near — ακριβώς η υπογραφή exposure-bias). SS-exp/step χειροτερεύουν και τα δύο.
Πλήρες: `ABLATION_PLAN.md §7-RESULTS`.
6. ~~SS πρώτο end-to-end τρέξιμο~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω. Winner: `linear`.

**Βήμα 7 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** SS-linear retest 5 οριακών ομάδων. **Εντυπωσιακή επιβεβαίωση** της
υπόθεσης χρήστη #7 (interaction SS×features): `resfc` κάνει δραματική αναστροφή από άχρηστο
(no-SS) σε ΠΟΛΥ πολύτιμο (SS ΔMAE=+1.10) — το μοντέλο ακουμπάει στα πάντα-αξιόπιστα day-ahead
forecasts μόλις τα y-lags γίνουν θορυβώδη. `loadfc` λύνεται από θόρυβο σε καθαρά θετικό
(+0.46). `loadlags` κάνει το ΑΝΤΙΘΕΤΟ ταξίδι — από βοηθητικό σε επιζήμιο (−0.32) υπό SS.
`fuel`/`meteo` συνεπή (fuel βλάπτει σταθερά, meteo βλάπτει αλλά λιγότερο υπό SS).
Πλήρες: `ABLATION_PLAN.md §7b-RESULTS`.
7. ~~SS-retest οριακών ομάδων~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω.

**Βήμα 8 — ΟΛΟΚΛΗΡΩΘΗΚΕ.** Confirmation battery (A: default-monthly, B: default+xborder-monthly,
C: SS-informed static): **B νικητής, 15.02 €/MWh** — καλύτερο ΑΚΟΜΑ κι από το προηγούμενο
headline (weekly retrain, 15.17), δηλαδή το xborder gain > το κέρδος από weekly retrain.
Επιβεβαιώθηκε σε 2ο out-of-sample μήνα (Μάρτιος 2026): xborder βοηθάει ξανά (ΔMAE −0.35),
μικρότερο μέγεθος αλλά ίδιο πρόσημο. **Νέο headline config: LGBM recursive monthly,
`default,xborder` → 15.02 €/MWh.** Ενημερώθηκε και το SKILL.md §11. Πλήρες:
`ABLATION_PLAN.md §Confirmation-RESULTS`.
8. ~~Confirmation winner×monthly×2ος μήνας~~ **ΟΛΟΚΛΗΡΩΘΗΚΕ** — βλ. πάνω.

**Βήμα 9 — ΣΕ ΕΞΕΛΙΞΗ.** LEAR ξανατρέξιμο με v2 δεδομένα: 19.49 €/MWh, σταθερό εύρημα (ταιριάζει
με το ήδη καταγεγραμμένο ~19.0) — παραμένει σαφώς χειρότερο από τα δέντρα, όχι artifact των
παλιών ελλιπών δεδομένων. Ensemble (LGBM+XGB+LEAR): ο **mean ΞΑΝΑ χάνει** (επιβεβαιώνει το ήδη
γνωστό εύρημα)· ΝΕΟ: ο **median ΚΕΡΔΙΖΕΙ οριακά** (16.042 vs LGBM 16.096, Δ=−0.055, κάτω από
noise floor — κατευθυντικό μόνο). LSTM seq2seq: **τρέχει τώρα** (πρώτη ποτέ απόπειρα πλήρους
ολοκλήρωσης end-to-end). Πλήρες: `ABLATION_PLAN.md §Βήμα 9-RESULTS`.
3. Μέρος 1b: 10 solar-ευαίσθητα specs σε καλοκαίρι 2025 (Ιουν-Αυγ, train_end Μάιος '25).
4. `default,xborder` vs `default` test (μόλις γίνει το rebuild του βήματος 1).
5. E1 (n_estimators × meteo, 6 runs) + E2 (meteo σε task=load, 3 runs).
6. SS runs (no-SS/linear/exp/step, LGBM recursive static Q1) — πρώτο πραγματικό τρέξιμο του
   scheduled_sampling.py engine, verify ότι δουλεύει σωστά πριν τα «οριακά» retests.
7. SS-retest των οριακών ομάδων (meteo/loadlags/fuel/resfc/loadfc) με το καλύτερο SS σχήμα.
8. Confirmation: winner config(s) × monthly retrain, +2ο test-μήνα.
9. (χαμηλότερη προτεραιότητα) Ξανατρέξιμο LEAR/LSTM/ensemble με v2 δεδομένα — δεν έχει γίνει
   ακόμα μετά το data fix, οπότε τα LEAR/LSTM νούμερα στα παλιά docs μπορεί να είναι stale.

## Λειτουργικές σημειώσεις για autonomous/overnight εκτέλεση

- **Windows: ΕΝΑ conda process τη φορά.** Sequential μόνο — μη ξεκινήσεις δεύτερο background
  training run πριν τελειώσει το προηγούμενο (θα καθυστερήσουν αμφότερα ή θα σκάσουν).
- **OneDrive πρέπει να τρέχει** (`Get-Process OneDrive`) αλλιώς raw CSV reads σκάνε με
  κρυπτικό `OSError: [Errno 22]`. Αν σκάσει: `Start-Process "$env:PROGRAMFILES\Microsoft
  OneDrive\OneDrive.exe"`, περίμενε ~15s.
- **`conda run python -c "<multi-line>"` σπάει** (`NotImplementedError` σε newlines) — γράψε
  πάντα σε προσωρινό `.py` αρχείο και τρέξε `conda run -n epf --no-capture-output python -X
  utf8 <path>`.
- Πριν από ΟΠΟΙΟΔΗΠΟΤΕ rebuild parquet: backup (`cp` σε `_backup_*`), rebuild, σύγκρινε
  gen-related στήλες ανά έτος (regression-check όπως στο βήμα 4), ΜΕΤΑ διάγραψε backup.
- `--retrain monthly/weekly` ΔΕΝ παίρνει `--train_end` (αγνοείται, expanding window) — μόνο
  `static` το σέβεται.
- Μη ξεχάσεις: το ablation chain τρέχει με `--retrain static` σκόπιμα (απομόνωση αξίας
  features, όχι retrain policy) — μην το αλλάξεις σε monthly/weekly μέσα στο ίδιο batch.
