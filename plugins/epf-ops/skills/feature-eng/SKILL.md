---
name: feature-eng
description: FeatureENG agent — end-to-end lifecycle για data_in extension (νέα πηγή δεδομένων ή νέο feature) με TDD pre-registration. Χρησιμοποίησέ το όταν ο χρήστης θέλει να προσθέσει/σχεδιάσει/αξιολογήσει νέο feature ή νέα πηγή ΑΠΟ ΤΗΝ ΑΡΧΗ ΩΣ ΤΟ ΤΕΛΟΣ (design → audit → implementation → batch → verdict → deploy). Για μεμονωμένο audit πηγής υπάρχει το ingest-audit· για σκέτη σύνοψη runs το synthesize-ablation — αυτό το skill τα ΚΑΛΕΙ ως στάδια.
---

# FeatureENG Agent — data_in extension, leakage-free by construction

Ιεραρχία: αυτό το skill είναι ο **orchestrator**. Δεν επαναλαμβάνει κανόνες — τους
δανείζεται: `energy-forecast` (σκληροί κανόνες/εντολές), `ingest-audit` (στάδιο 2),
`energy-runner` (στάδιο 4), `synthesize-ablation` (στάδιο 5), subagent
`validity-reviewer` (πριν από κάθε ACCEPTED). Πηγές αλήθειας: `last.md §2` + `ABLATION_PLAN §2`.

**Αρχή TDD (μη διαπραγματεύσιμη):** τα acceptance tests T1-T8 γράφονται και
κλειδώνουν στο design doc **ΠΡΙΝ γραφτεί μία γραμμή feature κώδικα**. Ένα
«θεαματικά καλό» αποτέλεσμα ΔΕΝ είναι επιτυχία — είναι failing test (T8, μάθημα
xborder: το −0.72 ήταν leakage).

## Στάδιο 0 — Scope & ουρά

1. Διάβασε `last.md` §1-§2 + `ABLATION_PLAN.md` §2/§7 (μην ξανασχεδιάσεις ό,τι είναι ήδη PENDING/ΑΚΥΡΟ).
2. Έλεγξε την conda ουρά: αν τρέχει batch (π.χ. overnight), τα στάδια 2β/3/4 ΠΕΡΙΜΕΝΟΥΝ — τα στάδια 1/2α (docs) τρέχουν ελεύθερα.
3. Ένα feature τη φορά (narrow scope). Ομάδα-στόχος = ένα flag στο `feature_availability.py`.

## Στάδιο 1 — DESIGN (system-design × TDD)

1. Αντίγραψε το `references/design-template.md` σε `docs/features/<name>/design.md` και συμπλήρωσέ το ΟΛΟ.
2. Pre-registration (πριν από κώδικα): μηχανισμός, αναμενόμενο πρόσημο & μέγεθος ΔMAE
   (πρέπει να δικαιολογεί |Δ|>0.15), εποχιακότητα/στρατηγική όπου αναμένεται να δρα,
   προβλεπόμενο lagscan peak k, hour-profile σχήμα, αναμενόμενη τάξη FI rank.
3. Κλείδωσε τον πίνακα T1-T8. STOP: χωρίς συμπληρωμένο design doc, κανένα επόμενο στάδιο.

## Στάδιο 2 — AUDIT (διαβατήριο πηγής)

1. Κάλεσε το skill **ingest-audit** (βήματα 1-4: gate timing, lagscan, hour-profile, πληρότητα ανά task parquet).
2. Κατάγραψε τα evidence paths στο design doc (T1-T3).
3. STOP RULE: T1 (δημοσίευση ΜΕΤΑ το gate 12:00 CET D-1 χωρίς νόμιμη lagged εκδοχή) ή
   T2/T3 ύποπτα → το feature ΑΠΟΡΡΙΠΤΕΤΑΙ εδώ, με μηδενικό κόστος training. Γράψε το verdict στο ABLATION_PLAN §7.

## Στάδιο 3 — IMPLEMENT (feature-dev)

1. Fetcher (αν χρειάζεται νέο download) → `src/fetch_*.py` pattern, μετά rebuild parquet
   με backup+σύγκριση (κανόνας energy-forecast — data/processed προστατευμένο, μόνο μέσω scripts).
2. Νέα ομάδα ΜΟΝΟ στο `src/feature_availability.py` με ρητό availability rule — ποτέ κατευθείαν στο engine.
   Το guard hook θα ζητήσει επιβεβαίωση: αναμενόμενο.
3. Μετά την αλλαγή (leakage-sensitive πυρήνας): `preflight_check.py --poison` PASS (T5) +
   control run + reproducibility anchor ±0.05 (T6) + έλεγχος ότι η στήλη υπάρχει στο σωστό parquet ΑΝΑ task (μάθημα loadfc).

## Στάδιο 4 — BATCH (testing-strategy)

1. Πίνακας πειραμάτων κατά `ABLATION_PLAN §3`: retrain=**static** (απομόνωση αξίας feature),
   seed 42, ζεύγος specs `default` vs `default,<group>` (ή `all` vs `all,-<group>`),
   **≥2 ανεξάρτητα windows** (Q1 + καλοκαίρι· Μάρτιος ως 3ο/tiebreak), LGBM + XGB.
2. Ονοματολογία out_json ΥΠΟΧΡΕΩΤΙΚΑ `<window>_<algo>_<mode>_<spec>.json` σε
   `runs/feat_<name>/` (ώστε να τη διαβάζει το synthesize_ablation.py) + `--csv results/feat_<name>.csv`.
3. Εκτέλεση με τους κανόνες **energy-runner**: ΕΝΑ conda process, long runs detached
   (Start-Process, ASCII args), progress από logs.
4. **Fail-fast (T4):** στο ΠΡΩΤΟ log επαλήθευσε ότι το `#features` ΑΛΛΑΖΕΙ μεταξύ των δύο
   specs. Δ=0.000 bit-for-bit = VOID arm → σταμάτα το batch, ψάξε το parquet, μην κάψεις ώρες.

## Στάδιο 5 — VERDICT

1. Κάλεσε το skill **synthesize-ablation** (ΔMAE πίνακες + §2 pre-gate + validity-reviewer
   πριν από κάθε ACCEPTED + εγγραφή στο ABLATION_PLAN §5/§7).
2. Σύγκρινε με τα pre-registered του σταδίου 1: πρόσημο/μέγεθος όπως προβλέφθηκε; Αν το
   αποτέλεσμα είναι ΠΟΛΥ καλύτερο απ' το αναμενόμενο → T8 red flag, ζήτα targeted poisoning πριν από claim.

## Στάδιο 6 — DEPLOY CHECKLIST

1. Feature importance evidence: τρέξε/εντόπισε FI για config με τη νέα ομάδα
   (`src/feature_strategy_compare.py` → πίνακας FI· οπτικά `src/fi_visualise.py`) — η νέα
   ομάδα εμφανίζεται με εύλογο rank; (T8 μέρος β).
2. Αντίγραψε `references/deploy-checklist.md` σε `docs/features/<name>/deploy.md`, συμπλήρωσε ΟΛΕΣ τις ενότητες.
3. Validation: `python .claude/skills/feature-eng/scripts/validate_deploy_checklist.py docs/features/<name>/deploy.md`
   (system python, όχι conda) — πρέπει PASS.
4. Απόφαση default set ΜΟΝΟ για ACCEPTED· αλλαγή headline επιπλέον ≥3 seeds + 2ο window.
   Η τελική αποδοχή είναι ΑΝΘΡΩΠΙΝΗ απόφαση (compounding spec §1) — παρουσίασε, μην αυτο-εγκρίνεις.
5. Κατάθεση (deposit): ενημέρωσε `last.md` §1/§6 + ABLATION_PLAN + commit. Όταν υπάρξει το
   Feature Registry (Rung 2 του compounding spec), το deploy.md είναι η πηγή της εγγραφής.

## Έξοδος κάθε απάντησης με ανοιχτή δουλειά

Κλείσε πάντα με το format του energy-runner: `DONE / RESULT PATHS / PENDING / NEXT`.
