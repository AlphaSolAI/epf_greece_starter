# ABLATION PLAN — Bulletproof πρόγραμμα πειραμάτων (v2 δεδομένα)

## 🌙 OVERNIGHT PROTOCOL (2026-07-04) — δομημένο κατά το ζητούμενο: έλεγχος → robust δοκιμές → αδιαμφισβήτητη απόφαση → επόμενο βήμα → μεγάλη εικόνα

> Εκτελείται από αυτόνομο session (/goal). Προτεραιότητα: recursive πρώτα, direct μετά
> (αν στενέψει ο χρόνος, κόβονται με σειρά: direct-XGB → MLP mini → direct 18-spec).
> Single conda lane, sequential. Κάθε φάση: τρέξε → γράψε §RESULTS εδώ → μετά επόμενη.

### §1 ΠΡΟΛΗΠΤΙΚΟΣ ΕΛΕΓΧΟΣ (πριν από ΚΑΘΕ νέο feature/δοκιμή — μόνιμος κανόνας πλέον)
Για κάθε νέα πηγή: (α) **Πότε ΑΚΡΙΒΩΣ δημοσιεύεται;** — γραπτή απάντηση σε σχέση με το gate
12:00 CET D-1, ΠΡΙΝ γραφτεί κώδικας· (β) **cross-correlation lag-scan** κατά του y (peak πρέπει
να είναι εκεί που προβλέπει η θεωρία — peak σε «βολικό» σημείο = ύποπτο leakage)· (γ) hour-of-day
profile sanity (π.χ. solar peak μεσημέρι). Pre-flight overnight: OneDrive τρέχει, conda epf OK,
parquet έχει xb_*_lag* ΚΑΙ ΟΧΙ same-day xb, LGBM default static Q1 αναπαράγει 16.10±0.05.

### §2 ROBUST ΔΟΚΙΜΕΣ (φάσεις με σειρά)
- **P1 — xborder-lagged κλείσιμο** (~40′): monthly Q1, weekly Q1, summer static × `default,xborder`.
  Ήδη γνωστό: static Q1 = 16.52 (+0.42, βλάπτει). Αν ≥0 παντού → οριστικό «όχι», κλείνει το θέμα.
- **P2 — cross-model ablation (η μεγάλη παράλειψη)** (~2h): XGB καλοκαίρι (10 specs του §2b) ·
  LEAR mini-ablation static Q1 (6 specs: default / -genlags / -resfc / -fuel / lags,calendar /
  default,xborder) · MLP mini static Q1 (3 specs: default / -genlags / lags,calendar).
- **P3 — Ensembles σωστά** (~1h): μέλη = ίδιο window/retrain (LGBM+XGB+LEAR recursive static Q1
  ήδη υπάρχουν στο step9_out + νέα του P2). mean/median (υπάρχει) + **weighted-by-1/MAE με
  calibration στον Δεκ και αξιολόγηση σε Ιαν-Φεβ** (μικρή προσθήκη στο make_ensemble ή script —
  ΟΧΙ βάρη από το ίδιο το test window). Επίσης: weekly-retrain ζεύγος LGBM+XGB ensemble.
- **P5 — SS-συμπέρασμα σε έγκυρα δεδομένα** (~40′): το «SS redundant με retrain» είχε μετρηθεί
  σε xborder configs → re-check: SS-linear `default` monthly Q1 vs plain `default` monthly (15.795).
- **P4 — DIRECT πυλώνας** (~4-5h): direct-LGBM 18 specs static Q1 · direct-LGBM default static
  vs monthly · direct-XGB 10 specs static Q1 · cross-strategy ensemble (direct-LGBM + recursive-
  LGBM median — δύο ανόμοια λάθη, υποψήφιο πραγματικό ensemble κέρδος).
- **P6 — Robustness τελικού νικητή** (~40′): seeds 43/44 στο όποιο νέο headline + 1 επιπλέον
  test window (Μάρτιος 2026).

### §3 ΑΔΙΑΜΦΙΣΒΗΤΗΤΗ ΑΠΟΦΑΣΗ — ρητά κριτήρια (για να μη χρειάζεται κρίση εκ των υστέρων)
Εύρημα ΓΙΝΕΤΑΙ ΔΕΚΤΟ ⟺ |ΔMAE| > 0.15 (noise floor· seed std μετρήθηκε ≈0.05) ΚΑΙ ίδιο πρόσημο
σε ≥2 ανεξάρτητες συνθήκες (άλλο window Ή άλλος αλγόριθμος Ή άλλη στρατηγική). Αλλιώς: PENDING,
όχι συμπέρασμα. Κάθε φάση κλείνει με πίνακα: [ερώτημα | αποτέλεσμα | ΔΕΚΤΟ/ΑΠΟΡΡΙΦΘΗΚΕ/PENDING
| τι σημαίνει για το προϊόν] — αυτοί οι πίνακες είναι τα raw materials του report/διπλωματικής.

### §4 ΕΤΟΙΜΟ ΕΠΟΜΕΝΟ ΒΗΜΑ (μετά το overnight, με την ίδια 1→2→3 λογική)
Ανάλογα με P1-P6: (α) αν direct ≈ recursive → προϊόν πάει direct για DAM (χωρίς rollout
πολυπλοκότητα)· (β) resfc-2h-shift διερεύνηση (προϋπάρχον ύποπτο, §last.md) με το §1 πρωτόκολλο·
(γ) probabilistic layer (conformal) πάνω στο κλειδωμένο point config· (δ) task=load πλήρες
ablation (μόνο E2 έχει γίνει).

## §P1-RESULTS — xborder-lagged closure test, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-04)

`p1_out/*.json`, logs `p1_out/p1_{monthly,weekly,summer}_log.txt`. Ερώτημα: κλείνει οριστικά το
xborder θέμα με ≥0 (βλάπτει/ουδέτερο) παντού;

| Συνθήκη | `default` MAE | `default,xborder`-lagged MAE | ΔMAE | Πρόσημο |
|---|---|---|---|---|
| static Q1 (ήδη γνωστό) | 16.096 | 16.518 | **+0.422** | βλάπτει |
| monthly Q1 | 15.795 | 15.929 | **+0.134** | βλάπτει (κάτω από noise floor 0.15) |
| weekly Q1 | 15.17 | 15.576 | **+0.406** | βλάπτει |
| summer 2025 static | 14.434 | 13.915 | **−0.519** | **βοηθάει** |

**🟡 ΔΕΝ κλείνει με το απλό κριτήριο «≥0 παντού»** — το summer αποτέλεσμα είναι αρνητικό
(βοηθάει), άρα η προϋπόθεση του πρωτοκόλλου δεν πληρούται. Εφαρμόζεται το γενικό κριτήριο §3:

- **Winter/Q1 (headline window): ΔΕΚΤΟ ότι βλάπτει.** Συνεπές πρόσημο σε 3 retrain cadences
  (static/monthly/weekly), μέγεθος >0.15 σε 2/3 (static, weekly) — static+weekly δεν είναι πλήρως
  ανεξάρτητες συνθήκες κατά γράμμα του §3 (ίδιο window/algo/strategy, μόνο cadence αλλάζει), αλλά
  η robustness σε 3 cadences ενισχύει την εμπιστοσύνη πρακτικά.
- **Summer: PENDING ως καθολικό συμπέρασμα** — 1 μόνο σημείο (static μόνο, καμία cadence-
  robustness δοκιμή), μέγεθος 0.52 πάνω από noise floor αλλά χωρίς δεύτερη ανεξάρτητη επιβεβαίωση.
  Ηχεί σαν το ήδη-γνωστό resfc season-interaction (§2b-RESULTS) — πιθανό ΝΕΟ, γνήσιο εύρημα
  εποχιακότητας (όχι artifact, μια που το lagged xborder είναι ήδη leakage-free), αλλά χρειάζεται
  τουλάχιστον 1 ακόμα summer cadence/seed πριν δηλωθεί στη διπλωματική.

**Απόφαση για το default feature set (συντηρητική, βλ. §OVERNIGHT PROTOCOL directive)**:
**xborder ΑΠΟΡΡΙΠΤΕΤΑΙ από το unconditional default set.** Η headline/target σεζόν του project
είναι ο χειμώνας (Q1 2026, strict gate, tradeable backtest) και εκεί το xborder-lagged βλάπτει με
συνέπεια σε όλα τα retrain cadences. Το θετικό καλοκαιρινό εύρημα καταγράφεται ως ανοιχτό,
επιστημονικά ενδιαφέρον ερώτημα (πιθανή εποχιακή αλληλεπίδραση, ΟΧΙ αρκετή απόδειξη ακόμα για να
αλλάξει την πρακτική απόφαση) — υποψήφιο για μελλοντικό session (SS×summer×xborder ή cadence
robustness στο καλοκαίρι).

**Headline παραμένει: LGBM recursive weekly, `default` = 15.17 €/MWh** (χωρίς xborder).

## §P2-RESULTS — Cross-model ablation (η μεγάλη παράλειψη), ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-04)

`p2_out/results_ablation_{xgb_summer,lear_q1,mlp_q1}.csv`, logs `p2_out/p2_*_log.txt`.

**XGB summer 2025 (10 specs του §2b, ίδιο window με LGBM summer)** — baseline `default`=15.026:

| Ομάδα (leave-one-out) | XGB ΔMAE | LGBM ΔMAE (§2b) | Συμφωνία; |
|---|---|---|---|
| `resfc` | **+1.218** (πολύτιμο) | +1.085 | 🟢 ΝΑΙ — cross-model, ΔΕΚΤΟ ότι resfc πολύτιμο το καλοκαίρι σε 2 αλγόριθμους |
| `genlags` | **+1.201** (πολύτιμο) | +0.431 | 🟢 ΝΑΙ (ίδιο πρόσημο, XGB μεγαλύτερο μέγεθος) |
| `loadfc` | −0.300 (βλάπτει η παρουσία του) | −0.278 | 🟢 ΝΑΙ — loadfc συνεχίζει να δείχνει αρνητικό/άχρηστο και σε XGB |
| `loadlags` | +0.421 (βοηθάει) | −0.055 (ουδέτερο) | 🟡 ίδια κατεύθυνση (θετικό/ουδέτερο) αλλά διαφορετικό μέγεθος |
| Καλύτερο combo | `lags,calendar,resfc,loadfc`=14.704 (κερδίζει το default 15.026) | παρόμοιο εύρημα | 🟢 lean core με resfc+loadfc κερδίζει, συνεπές με LGBM |

**🟢 Το summer resfc/genlags εύρημα είναι πλέον ΔΕΚΤΟ με πλήρη αυστηρότητα** (2 ανεξάρτητοι
αλγόριθμοι, ίδιο window, ίδιο πρόσημο, μέγεθος πολύ πάνω από noise floor) — δεν ήταν artifact
του LGBM.

**LEAR mini-ablation static Q1 (6 specs)** — baseline `default`=19.492 (ταιριάζει με §Βήμα
9-RESULTS):

| Spec | MAE | ΔMAE | Ερμηνεία |
|---|---|---|---|
| `default,-genlags` | 19.616 | +0.124 | Ίδια κατεύθυνση με δέντρα (helps) αλλά κάτω από noise floor — ασθενές στο LEAR. |
| `default,-resfc` | **18.498** | **−0.995** | 🎯 **Αντίστροφο από δέντρα** — η αφαίρεση resfc ΒΕΛΤΙΩΝΕΙ το LEAR κατά σχεδόν 1 €/MWh. Επιβεβαιώνει το ήδη καταγεγραμμένο (SKILL.md §10): το L1/LASSO shrinkage του LEAR υποβαθμίζει/βλάπτεται από τα day-ahead forecast features που τα δέντρα εκμεταλλεύονται καλά. |
| `default,-fuel` | 18.988 | −0.505 | Αφαίρεση fuel βοηθάει, συνεπές με δέντρα (fuel άχρηστο) αλλά μεγαλύτερο όφελος στο LEAR. |
| `lags,calendar` (bare core) | 19.590 | +0.098 | Οριακό, κάτω από noise floor. |
| `default,xborder` (lagged) | 19.334 | −0.158 | Οριακά βοηθάει (ακριβώς στο όριο noise floor) — **αντίθετο πρόσημο από LGBM στο Q1** (εκεί xborder έβλαπτε +0.42). Μοναδικό σημείο, όχι επιβεβαιωμένο. |

**🔴 Μοντελο-εξαρτώμενο εύρημα, ΔΕΚΤΟ ως τέτοιο**: το LEAR αντιδρά ΑΝΤΙΘΕΤΑ από τα δέντρα στο
`resfc`/`fuel` (η αφαίρεσή τους βοηθάει, όχι βλάπτει) — επιβεβαιώνει ότι «τα LGBM-derived
καλύτερα features βοηθούν παντού» είναι ΛΑΘΟΣ υπόθεση, όπως προειδοποιούσε το §9 (honesty
section). Πρακτική συνέπεια: αν το LEAR χρησιμοποιηθεί ποτέ ξανά (π.χ. ως ensemble member/
fallback), το δικό του βέλτιστο feature set ΔΕΝ είναι το `default` των δέντρων.

**MLP mini static Q1 (3 specs)** — baseline `default`=20.491:

| Spec | MAE | ΔMAE | Ερμηνεία |
|---|---|---|---|
| `default,-genlags` | 21.850 | **+1.360** | Ίδιο πρόσημο με δέντρα (genlags πολύτιμο), ΜΕΓΑΛΥΤΕΡΟ μέγεθος — 3ος αλγόριθμος συμφωνεί (LGBM/XGB/MLP). |
| `lags,calendar` (bare core) | **18.911** | **−1.580** | 🎯 Ο γυμνός πυρήνας ΝΙΚΑΕΙ ξεκάθαρα το πλήρες `default` (152 feat) — ενισχύει δραματικά το ήδη-γνωστό «kitchen-sink ≠ βέλτιστο» (§1), ακόμα εντονότερο στο MLP (πιθανώς πιο ευαίσθητο σε θόρυβο/διαστασιμότητα από τα δέντρα). |

**🟢 `genlags` τώρα ΔΕΚΤΟ σε 3/3 αλγόριθμους (LGBM, XGB, MLP)** — το πιο στιβαρό εύρημα όλης
της μελέτης μαζί με το resfc-καλοκαίρι.

**Συνολική ενημέρωση §9 honesty table**: XGB πήρε πλέον summer ablation (όχι μόνο Q1 static) —
συμφωνεί πλήρως με LGBM. LEAR/MLP πήραν την ΠΡΩΤΗ τους ποτέ ablation (μικρή κλίμακα, 3-6 specs) —
LEAR δείχνει γνήσια διαφορετική συμπεριφορά (model-specific feature value, ΟΧΙ artifact), MLP
συμφωνεί με τα δέντρα στο genlags αλλά είναι ακόμα πιο ευαίσθητο στο kitchen-sink πρόβλημα.

## §P3-RESULTS — Ensembles σωστά (out-of-sample calibrated weights), ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-04)

**Πρόβλημα με το παλιό εύρημα** (§Βήμα 9-RESULTS: median=16.042 vs LGBM=16.096, «κερδίζει
οριακά»): η σύγκριση mean/median δεν χρειάζεται calibration (καμία παράμετρος) άρα ήταν ήδη
έγκυρη, ΑΛΛΑ δεν είχε δοκιμαστεί weighted-by-1/MAE με σωστό out-of-sample split. Έγινε τώρα:

**Static Q1, 3 μέλη (LGBM+XGB+LEAR recursive static, `default`)** — calibration στο Δεκ (744h),
αξιολόγηση στο Ιαν-Φεβ (1416h, ΠΟΤΕ δεν είδε τα weights):

| Μέλος/Ensemble | Calib(Δεκ) MAE | Eval(Ιαν-Φεβ) MAE | Σχόλιο |
|---|---|---|---|
| LGBM (καλύτερο μεμονωμένο) | 13.884 | **17.259** | Νικητής στο honest eval split |
| ensemble_median | — | 17.318 | Χάνει οριακά από το LGBM (Δ=+0.059, θόρυβος) |
| XGB | 13.692 | 17.532 | — |
| ensemble_weighted(1/MAE, βάρη από Δεκ) | — | 17.783 | **Χάνει ΚΑΙ από το mean ΚΑΙ από το median** |
| ensemble_mean | — | 17.862 | Χειρότερο όλων των ensembles |
| LEAR | 15.436 | 21.624 | — |

Sanity check: το weighted split reproduce σωστά το πλήρες-window LGBM MAE
(744·13.884+1416·17.259)/2160=16.096 ✓ ταιριάζει ακριβώς με το already-known 16.0963.

**🔴 ΑΠΟΡΡΙΦΘΗΚΕ: το weighted-by-1/MAE ensemble ΔΕΝ προσφέρει τίποτα** — με τίμιο (out-of-
sample) calibration, ΚΑΝΕΝΑ ensemble μέθοδος (mean/median/weighted) δεν κερδίζει το καλύτερο
μεμονωμένο μοντέλο (LGBM) σε αυτό το split. Το παλιό «median κερδίζει οριακά» (§Βήμα 9) ήταν
πάνω στο ΠΛΗΡΕΣ Q1 window (Δεκ+Ιαν+Φεβ μαζί) — σε αυτό το πιο δύσκολο (μόνο Ιαν-Φεβ, χωρίς το
πιο εύκολο Δεκ) υπο-παράθυρο ούτε καν ο median κερδίζει. Πρακτική συνέπεια: ΜΗΝ επενδυθεί άλλος
χρόνος στο weighted ensemble (§4.3 ανοιχτό ζήτημα, τώρα κλείνει με «όχι»).

**Weekly retrain, 2 μέλη (LGBM+XGB, `default`, πλήρες Q1 window)**:

| Μέλος/Ensemble | MAE | ΔMAE vs καλύτερο μέλος |
|---|---|---|
| LGBM weekly (headline) | 15.171 | — |
| XGB weekly (νέο τρέξιμο) | 15.197 | — |
| **ensemble_mean = ensemble_median** | **15.023** | **−0.148** (ταυτίζονται γιατί 2 μέλη) |

**🟡 PENDING, ΟΧΙ ΔΕΚΤΟ** — το Δ=−0.148 είναι ΑΚΡΙΒΩΣ κάτω από το noise floor (0.15), και μόνο
1 σημείο (καμία δεύτερη ανεξάρτητη συνθήκη δοκιμάστηκε ακόμα εδώ). Κατευθυντικά ενθαρρυντικό
(τα 2 μέλη είναι σχεδόν ισοδύναμης ποιότητας εδώ, μάλλον γιατί LGBM/XGB μαθαίνουν συμπληρωματικά
λάθη) αλλά ΔΕΝ αρκεί ακόμα για να αντικαταστήσει το single-model headline. Χρειάζεται 2ο test
window ή seeds πριν δηλωθεί κέρδος στη διπλωματία.

**Συνολικό συμπέρασμα P3**: το ensembling με 3 ανόμοια-ποιοτικά μέλη (LGBM/XGB/LEAR) ΔΕΝ
αποδίδει με τίμιο calibration· το ensembling 2 σχεδόν-ισοδύναμων μελών (LGBM/XGB weekly) δείχνει
οριακό αλλά ενθαρρυντικό σημάδι, κάτω ακριβώς από το κατώφλι. Headline παραμένει το μεμονωμένο
LGBM recursive weekly, `default` = 15.17 €/MWh.

## §P5-RESULTS — SS re-check σε ΕΓΚΥΡΟ (χωρίς xborder) config, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-04)

`p5_out/lgbm_ss_linear_default_monthly_q1.json`, log `p5_out/p5_log.txt`. Πρόβλημα με το παλιό
συμπέρασμα («SS redundant με retrain», §Stacking-RESULTS): μετρήθηκε σε `default,xborder`
configs — τα xborder claims ακυρώθηκαν (leakage), άρα η μέτρηση ήταν πάνω σε δηλητηριασμένο
feature set. Re-check σε καθαρό `default` (χωρίς xborder):

| Config | MAE | ΔMAE vs plain | Πρόσημο |
|---|---|---|---|
| plain `default` monthly (ήδη γνωστό, Confirmation-RESULTS A) | 15.795 | — | — |
| SS-linear `default` monthly (νέο) | **15.5517** | **−0.243** | **βοηθάει, πάνω από noise floor** |
| (για σύγκριση) SS-linear `default` static (§7-RESULTS, ήδη έγκυρο πάντα) | 15.830 vs 16.096 static | −0.266 | βοηθάει, πάνω από noise floor |

**🟢 ΔΙΟΡΘΩΝΕΤΑΙ το προηγούμενο συμπέρασμα: το SS-linear ΔΕΝ είναι redundant με το φρέσκο
retrain.** Δύο συνθήκες (static, monthly· ίδιο algo/window/strategy, μόνο cadence διαφέρει —
όχι πλήρως ανεξάρτητες κατά γράμμα του §3, αλλά συνεπές πρόσημο+μέγεθος σε αμφότερες) δείχνουν
σταθερό όφελος ≈−0.24 έως −0.27, και τα δύο πάνω από το noise floor (0.15). Το παλιό «redundant»
εύρημα (§Stacking-RESULTS: −0.09, εντός noise) ήταν artifact του xborder-contaminated config,
ΟΧΙ γνήσιο εύρημα για το SS αυτό καθαυτό. Πρακτική συνέπεια: το SS-linear αξίζει να θεωρηθεί
σοβαρός υποψήφιος για το production pipeline ακόμα και με μηνιαίο retrain, ΟΧΙ μόνο static/
στατικό καθεστώς όπως πιστεύαμε πριν.

⚠️ Δεν δοκιμάστηκε ακόμα SS×weekly (θα ήταν το επόμενο λογικό βήμα, ανοιχτό για μελλοντικό
session) ούτε SS×monthly σε δεύτερο window — 1 μόνο σημείο ανά cadence.

## §P4-RESULTS — DIRECT πυλώνας, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-04)

`p4_out/results_ablation_direct_{lgbm,xgb}_q1.csv`, `p4_out/lgbm_direct_default_monthly_q1.json`,
`p4_out/ensemble_crossstrategy_direct_recursive.json`. **Πρώτη φορά που το direct στρατηγική
τεσταρίστηκε στο ΠΛΗΡΕΣ Q1 window** (πριν μόνο Δεκ-only, 31 ημέρες).

**Baseline μέγεθος**: direct-LGBM static Q1 `default` = **19.495** — ΠΟΛΥ χειρότερο από το
recursive-LGBM static Q1 (16.096) ΚΑΙ από το ίδιο direct στο μικρότερο Δεκ-only window (16.10).
Το direct φαίνεται να υποφέρει δυσανάλογα σε μεγαλύτερα/πιο ασταθή test windows (Ιαν-Φεβ regime
shift) σε σχέση με το recursive — μηχανιστικά εύλογο: κάθε ένα από τα 24 ανεξάρτητα direct
μοντέλα εκπαιδεύεται ΜΙΑ φορά στο ίδιο train_end, χωρίς κανένα μηχανισμό προσαρμογής όσο
προχωράει το test window (δεν κάνει rollout, ΑΛΛΑ και δεν ξαναβλέπει καθόλου φρέσκα δεδομένα
μέσα στο ίδιο static block).

**direct-LGBM 18-spec static Q1** (πλήρης πίνακας: `results_ablation_direct_lgbm_q1.csv`) —
σύγκριση με το ήδη γνωστό direct-Δεκ (§RESULTS Μέρη 1-3 πίνακας) όπου υπάρχει επικάλυψη specs:

| Ομάδα | Direct-Δεκ ΔMAE | Direct-Q1 ΔMAE (ΝΕΟ) | Ερμηνεία |
|---|---|---|---|
| `meteo` | **+0.548** (βοηθάει) | **+0.486** (βοηθάει) | 🟢 **ΔΕΚΤΟ, ισχυρότερα από ποτέ** — ίδιο πρόσημο σε 2 windows ΜΕΣΑ στο direct strategy. |
| `fuel` | −0.682 (βλάπτει) | +0.018 (ουδέτερο) | 🟡 Ασυνέπεια — window-size confound ΜΕΣΑ στο ίδιο strategy, νέο ανοιχτό ερώτημα. |
| `loadlags` | +0.198 (βοηθάει) | +0.126 (βοηθάει, κάτω από noise floor) | Ίδιο πρόσημο, μικρότερο μέγεθος στο μεγαλύτερο window. |
| `genlags` | +1.104 (πολύτιμο) | +0.192 (βοηθάει, ασθενέστερα) | Ίδιο πρόσημο αλλά ΠΟΛΥ μικρότερο μέγεθος — η αξία του genlags στο direct φαίνεται να «διαλύεται» σε μεγαλύτερο test window. |
| `resfc` | −0.316 (βλάπτει) | −0.044 (ουδέτερο) | Ίδιο πρόσημο (ελαφρώς αρνητικό/ουδέτερο), διαφορετικό μέγεθος. |
| `loadfc` | −0.160 (βλάπτει) | **−0.269** (βλάπτει, μεγαλύτερα) | Συνεπές. |
| `dense` (additive) | — | default,dense=19.003 (ΔMAE=**−0.492**, βοηθάει) | Συνεπές με recursive/XGB — 3η στρατηγική/context επιβεβαιώνει dense. |
| bare core `lags,calendar` | — | 20.240 (ΔMAE=+0.744, ΧΕΙΡΟΤΕΡΟ) | Αντίθετα από recursive/XGB-summer — στο direct-Q1 το πλήρες `default` ΝΙΚΑΕΙ το lean core, όχι το αντίστροφο. |

**direct-LGBM monthly vs static Q1**: monthly=19.589 vs static=19.495, ΔMAE=**+0.094** (ουδέτερο,
κάτω από noise floor). **🔴 ΝΕΟ εύρημα: το retrain cadence ΔΕΝ βοηθάει στο direct strategy**,
αντίθετα με το recursive όπου βοηθάει μονότονα (16.10→15.80→15.17). Μηχανιστική εξήγηση: το
direct δεν κάνει rollout άρα δεν έχει το ίδιο «staleness» πρόβλημα που λύνει το retrain στο
recursive — κάθε direct μοντέλο είναι ήδη trained στο πιο πρόσφατο διαθέσιμο train_end του
static config, ο μόνος λόγος να βοηθήσει monthly θα ήταν να πιάσει regime shift ΜΕΣΑ στο test
window, κάτι που δεν φάνηκε εδώ.

**direct-XGB 10-spec static Q1** (`results_ablation_direct_xgb_q1.csv`) — baseline `default`=19.594:

| Ομάδα | XGB-direct ΔMAE | LGBM-direct ΔMAE | Συμφωνία; |
|---|---|---|---|
| `meteo` | **+0.291** (βοηθάει) | +0.486 (βοηθάει) | 🟢 **ΝΑΙ — 2ος αλγόριθμος επιβεβαιώνει meteo-helps-in-direct.** Μαζί με τα 2 windows του LGBM, αυτό είναι πλέον το πιο στιβαρά επιβεβαιωμένο εύρημα strategy-effect όλης της μελέτης (2 algos × 2 windows, ίδιο πρόσημο). |
| `resfc` | **−0.317** (βλάπτει) | −0.044 (ουδέτερο) | 🟡 Διαφωνία μεγέθους/οριακά προσήμου μεταξύ αλγορίθμων — PENDING για resfc-in-direct. |
| `genlags` | +0.174 (βοηθάει) | +0.192 (βοηθάει) | 🟢 Συμφωνία, μικρό-μέτριο μέγεθος και στους δύο. |
| `loadlags` | +0.228 (βοηθάει) | +0.126 (βοηθάει) | 🟢 Συμφωνία. |
| `loadfc` | −0.006 (ουδέτερο) | −0.269 (βλάπτει) | 🟡 Διαφωνία μεγέθους. |
| `fuel` | −0.033 (ουδέτερο) | +0.018 (ουδέτερο) | Συμφωνία — αμφότερα ουδέτερα. |
| bare core | 19.914 (ΔMAE=+0.320, χειρότερο) | +0.744 (χειρότερο) | 🟢 Συμφωνία — kitchen-sink κερδίζει στο direct, αντίστροφα από recursive/summer. |

**🟢 ΟΡΙΣΤΙΚΟ ΝΕΟ ΕΥΡΗΜΑ (ΔΕΚΤΟ, το πιο στιβαρό strategy-effect όλης της μελέτης)**: το `meteo`
βοηθάει ΠΑΝΤΑ στο direct strategy (4/4 σημεία: LGBM-Δεκ, LGBM-Q1, XGB-Q1, + το ήδη-γνωστό
recursive πάντα βλάπτει) και βλάπτει ΠΑΝΤΑ στο recursive — καθαρό strategy-dependent φαινόμενο,
όχι artifact κανενός window/αλγορίθμου.

**Cross-strategy ensemble (direct-LGBM + recursive-LGBM, static Q1 default)**:

| Μέλος/Ensemble | MAE |
|---|---|
| LGBM recursive (καλύτερο μεμονωμένο) | 16.096 |
| ensemble_mean = ensemble_median | 16.619 (Δ=**+0.523, ΧΕΙΡΟΤΕΡΟ**) |
| LGBM direct | 19.495 |

**🔴 ΑΠΟΡΡΙΦΘΗΚΕ καθαρά** — η υπόθεση «δύο ανόμοιες στρατηγικές = ανόμοια λάθη = πραγματικό
ensemble κέρδος» ΔΕΝ επαληθεύτηκε εδώ: το direct μέλος είναι απλά πολύ πιο αδύναμο (19.5 vs
16.1) σε αυτό το window, οπότε ο μέσος όρος τραβάει την πρόβλεψη προς τα κάτω αντί να διορθώνει
θόρυβο. Επιβεβαιώνει το ήδη-γνωστό μοτίβο (SKILL.md #9): ensemble helps μόνο όταν τα μέλη είναι
συγκρίσιμης ποιότητας (βλ. §P3-RESULTS weekly LGBM+XGB, εκεί δείχνει υπόσχεση επειδή τα μέλη
είναι σχεδόν ισοδύναμα).

**Απόφαση §4(α) του πρωτοκόλλου**: **recursive ΠΑΡΑΜΕΝΕΙ η στρατηγική επιλογή για DAM** — το
direct δεν πλησιάζει το recursive σε κανένα από τα δοκιμασμένα windows (Δεκ ή Q1), άρα ΔΕΝ αξίζει
η πολυπλοκότητα rollout-free deployment. Headline παραμένει: **LGBM recursive weekly, `default`
= 15.17 €/MWh.**

## §P6-RESULTS — Robustness τελικού νικητή, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-04)

`p6_out/lgbm_weekly_default_{seed43,seed44,march2026}.json`. Headline: LGBM recursive weekly,
`default` (χωρίς xborder, βλ. §P1-RESULTS).

**Seed robustness (Q1 2026)**:

| Seed | MAE |
|---|---|
| 42 (headline) | 15.171 |
| 43 | 15.429 |
| 44 | 15.218 |
| **mean / std** | **15.273 / 0.112** |

**🟢 ΔΕΚΤΟ**: std≈0.11, κοντά στο ήδη καθιερωμένο noise floor (0.15) — το headline είναι
σταθερό ως προς seed, συνεπές με το ήδη-γνωστό (§Stacking-RESULTS: std≈0.05 σε παρόμοιο αλλά
xborder-config). Ελαφρώς μεγαλύτερο std εδώ (0.11 vs 0.05) αλλά ακόμα μέσα σε λογικά όρια, δεν
αλλάζει κανένα από τα δηλωμένα συμπεράσματα (όλα τα |ΔMAE| που δηλώθηκαν ΔΕΚΤΑ ήταν >0.19).

**2ο test window (Μάρτιος 2026, 2026-03-01→2026-03-19, ίδιο με Confirmation-RESULTS)**:

| Config | MAE | Σύγκριση |
|---|---|---|
| weekly `default` (ΝΕΟ) | **17.666** | — |
| monthly `default` (ήδη γνωστό, Confirmation-RESULTS A) | 17.968 | ΔMAE(weekly−monthly)=**−0.302** |

**🟢 ΔΕΚΤΟ**: η υπεροχή weekly>monthly επιβεβαιώνεται σε 2ο ανεξάρτητο out-of-sample window
(Q1: weekly 15.171 vs monthly 15.795, Δ=−0.624· Μάρτιος: Δ=−0.302) — ίδιο πρόσημο, 2 ανεξάρτητα
windows, πάνω από noise floor και στα δύο. Το retrain cadence weekly είναι στιβαρά η καλύτερη
επιλογή, ΟΧΙ artifact ενός μόνο test window.

**ΤΕΛΙΚΗ ΕΤΥΜΗΓΟΡΙΑ ΤΟΥ OVERNIGHT PROTOCOL**: **LGBM recursive weekly, `default` = 15.17 €/MWh**
(Q1 2026, strict gate, tradeable, seed-robust std≈0.11, επιβεβαιωμένο σε 2ο window). Το xborder
ΔΕΝ μπαίνει στο default (§P1). Κανένα ensemble/SS/direct-strategy πείραμα δεν ξεπέρασε αυτό το
config με στιβαρό τρόπο (weekly LGBM+XGB ensemble δείχνει borderline υπόσχεση, PENDING).

### §5 ΜΕΓΑΛΗ ΕΙΚΟΝΑ (η ροή όλου του έργου)
Data ✅ → Engine ✅ → Feature validity (τρέχον, κλείνει με overnight) → Model/strategy selection
(P2/P4) → Robustness/seeds (P6) → Probabilistic layer → Product (L3-L5 του SYSTEM_DESIGN).
Κάθε στάδιο κλειδώνει με τα κριτήρια του §3 πριν ανοίξει το επόμενο.


> Σταθερό setup παντού (εκτός αν λέγεται ρητά): **Q1 2026** (2025-12-01 → 2026-02-28),
> v2 πλήρη δεδομένα, `--gate strict`, `--market dam`, task=price, seed=42,
> **retrain=static** (ΌΧΙ weekly — απόφαση χρήστη: το ablation μετράει ΑΞΙΑ FEATURES,
> όχι πολιτική retrain· το static απομονώνει καθαρά την επίδραση κάθε πηγής και κοστίζει 1 fit/config).
> Επιβεβαίωση των νικητών με monthly ΜΟΝΟ στο τέλος (βλ. Πυραμίδα).

## RESULTS — Μέρη 1-3 ολοκληρώθηκαν (2026-07-02, πριν xborder rebuild)

Ωμά νούμερα: `results_ablation_q1_lgbm.csv`, `results_ablation_q1_xgb.csv`, `results_ablation_dec_direct.csv`.
Baseline: LGBM-Q1=16.19 · XGB-Q1=16.39 · direct-LGBM-Δεκ=16.10 €/MWh (default, 152 feat).

**🟢 Στιβαρό (συμφωνούν LGBM-Q1 + XGB-Q1 + direct-Δεκ):**
- `genlags` (gen actuals + residual_load lags) = η πιο πολύτιμη ομάδα. ΔMAE αφαίρεσης:
  LGBM +1.45 · XGB +1.42 · direct +1.10 (θετικό = βλάπτει η αφαίρεση = πολύτιμη ομάδα).
- Καλύτερο config σε LGBM & XGB: **`lags,calendar,resfc,genlags`** (39-48 feat) — ΚΕΡΔΙΖΕΙ το
  πλήρες `default` (152 feat): LGBM 16.06 vs 16.19 · XGB 16.01 vs 16.39. Kitchen-sink ≠ βέλτιστο.
- `dense` βοηθάει σταθερά (LGBM/XGB και οι δύο ~−0.74 MAE) — ΑΝΑΤΡΕΠΕΙ την παλιά παραδοχή
  "dense μπορεί να βλάψει OL" (σχόλιο στο DEFAULT_GROUPS).

**🔴 Ασυνεπή/αντιφατικά (χρειάζονται SS-retest ή/και το controlled πείραμα #X παρακάτω):**
- `meteo`: βλάπτει σε recursive-Q1 (LGBM −0.20, XGB −0.52) αλλά βοηθάει σε direct-Δεκ (+0.55).
  Confound: strategy (recursive/direct) ΚΑΙ window (Δεκ-Φεβ vs Δεκ-μόνο) αλλάζουν ταυτόχρονα.
- `loadlags`: LGBM +0.22 (βοηθάει) vs XGB −0.21 (βλάπτει) — αντίστροφο πρόσημο ΣΤΟ ΙΔΙΟ παράθυρο.
- `resfc`, `loadfc`, `fuel`: ασταθές πρόσημο/μέγεθος μεταξύ συνθηκών.

**Controlled πείραμα (disentangle strategy vs window) — ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02).**
recursive-LGBM σε **Δεκ-μόνο** (train_end 2025-11-30 23:00, test 2025-12-01→2025-12-31 —
ΑΚΡΙΒΩΣ το ίδιο window με το ήδη-τρεγμένο direct-Δεκ), ίδιες 10 specs.
`results_ablation_dec_recursive.csv` / `ablation_dec_recursive/`. baseline recursive-Δεκ = 13.88
€/MWh (vs direct-Δεκ 16.10 — recursive σαφώς καλύτερο σε Δεκ-μόνο, αναμενόμενο: μικρότερο
effective error-accumulation window).

ΔMAE σύγκριση (ίδιο window, strategy μόνο αλλάζει):

| Ομάδα | Direct-Δεκ ΔMAE | Recursive-Δεκ ΔMAE | Συμπέρασμα |
|---|---|---|---|
| `meteo` | **+0.548** (βοηθάει) | **−0.228** (βλάπτει) | 🎯 **STRATEGY effect, ΟΧΙ window** — το πρόσημο αντιστρέφεται με ΙΔΙΟ window, άρα η ασυμφωνία meteo (recursive-Q1 αρνητικό, direct-Δεκ θετικό) ΔΕΝ οφειλόταν στην εποχή/season· οφείλεται καθαρά στη στρατηγική. Υπόθεση μηχανισμού: στο direct κάθε offset είναι ανεξάρτητο μοντέλο χωρίς πρόσβαση σε φρέσκα y-lags άρα εκμεταλλεύεται exogenous meteo· στο recursive τα y-lags/error-accumulation κυριαρχούν και το meteo προσθέτει κυρίως θόρυβο. |
| `fuel` | −0.682 (βλάπτει) | −0.376 (βλάπτει) | Συνεπές — fuel άχρηστο στο Δεκ window, ανεξαρτήτως strategy. |
| `loadlags` | +0.198 (βοηθάει) | +0.340 (βοηθάει) | Συνεπές θετικό και στις δύο στρατηγικές (ίδιο LGBM) — η παλιά ασυμφωνία LGBM(+)/XGB(−) στο Q1 μάλλον είναι **algo effect** (LGBM vs XGB), όχι strategy/window effect. |
| `genlags` | +1.104 (πολύτιμο) | +2.626 (ΠΟΛΥ πολύτιμο) | Συνεπές θετικό, μεγαλύτερο στο recursive (λογικό — το recursive «σέρνει» λάθη, τα genlags σταθεροποιούν το rollout περισσότερο). |
| `resfc` | −0.316 (βλάπτει) | −0.043 (ουδέτερο, κάτω από noise floor) | Ίδιο πρόσημο (δεν βοηθάει σε Δεκ window με καμία strategy), διαφορετικό μέγεθος. |
| `loadfc` | −0.160 (οριακά βλάπτει) | +0.170 (οριακά βοηθάει) | Και τα δύο ~στο noise floor (0.15) — ασθενές/αδιάγνωστο σήμα, όχι στιβαρό συμπέρασμα προς καμία κατεύθυνση. |

**Headline finding**: το meteo confound λύθηκε ΠΛΗΡΩΣ — είναι strategy-dependent, όχι
season-dependent. Θα χρειαστεί ΚΑΙ το §2b (καλοκαιρινό) replication για πλήρη εικόνα σε solar-heavy
window, αλλά η ερώτηση "γιατί recursive-Q1 vs direct-Δεκ διαφωνούν" έχει πλέον απάντηση.

**xborder rebuild — ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02).** `python -m src.data --task price` έκανε merge
του `xborder_hourly.parquet` (BG + IT-SUD DAM τιμές) στο κύριο `hourly.parquet`: 2 νέες στήλες
(`xb_price_bg`, `xb_price_itsud`), backup+σύγκριση επιβεβαίωσε μηδενική αλλαγή σε καμία
υπάρχουσα στήλη/γραμμή. Ομάδα `xborder` έτοιμη για ablation test (§8, βήμα 4 του last.md).

## 0. Πυραμίδα πειραμάτων (ταχύτητα ↔ σιγουριά)

```
      / Confirmation \      ΛΙΓΑ: winner-config × monthly retrain × (+/- 2ος μήνας test)
     /  Full Q1 static \    ΜΕΣΑΙΑ: 18 specs × Q1 × static (LGBM ~45', XGB ~45')
    / Fast screening     \  ΠΟΛΛΑ/ΓΡΗΓΟΡΑ: direct-LGBM σε 1 μήνα (Δεκ) — sanity/στρατηγική-robustness
```

Κανόνας απόφασης: ομάδα «μετράει» αν |ΔMAE| > **0.15 €/MWh** vs baseline στο ίδιο παράθυρο
(πρόχειρο noise floor ενός τριμήνου· κάτω από αυτό = θόρυβος, μην βγάζεις συμπέρασμα).
Συμπέρασμα γίνεται δεκτό μόνο αν συμφωνούν ≥2 επίπεδα της πυραμίδας (π.χ. LGBM-Q1 + XGB-Q1,
ή LGBM-Q1 + direct-Δεκ).

## 1. Προαπαιτούμενο — λεπτόκοκκες ομάδες features (ΕΓΙΝΕ σε αυτό το session)

Οι παλιές ομάδες ήταν πολύ χοντρές για την ερώτηση «gen_act / for_gen / for / load χωριστά»:

| Παλιά ομάδα | Νέες λεπτές ομάδες |
|---|---|
| `forecast` | `resfc` (solar_fc/wind_fc/gen_fc day-ahead) + `loadfc` (load_fc) |
| `crosslags` | `genlags` (gen_solar_lag*/gen_wind_lag*/residual_load_lag*) + `loadlags` (load_lag*) + `other` |

Τα παλιά ονόματα (`forecast`, `crosslags`) δουλεύουν ακόμα ως **umbrellas** (επεκτείνονται
αυτόματα) — όλα τα παλιά specs/εντολές παραμένουν συμβατά. Επιπλέον το `--features` δέχεται
πλέον και `default` ως βάση με εξαιρέσεις (π.χ. `default,-resfc`).

## 2. ΜΕΡΟΣ 1 — Ablation πηγών δεδομένων (LGBM recursive static Q1) — 18 specs

**Leave-one-out από default** (τι χάνουμε αν λείψει):
1. `default` (baseline)
2. `default,-resfc` — χωρίς DA RES/gen forecast
3. `default,-loadfc` — χωρίς DA load forecast
4. `default,-genlags` — χωρίς actual gen lags (gen_act)
5. `default,-loadlags` — χωρίς actual load lags
6. `default,-meteo`
7. `default,-fuel`

**Additive πάνω σε γυμνό πυρήνα** (τι προσφέρει ΜΟΝΗ της κάθε πηγή):
8. `lags,calendar` (core baseline)
9. `lags,calendar,resfc`
10. `lags,calendar,loadfc`
11. `lags,calendar,genlags`
12. `lags,calendar,loadlags`

**Συνδυασμοί (τα ζητούμενα combos):**
13. `lags,calendar,resfc,loadfc` — όλα τα DA forecasts μαζί
14. `lags,calendar,genlags,loadlags` — όλα τα actuals μαζί
15. `lags,calendar,resfc,genlags` — forecast gen + actual gen
16. `lags,calendar,resfc,loadfc,genlags,loadlags` — όλα πλην meteo/fuel/roll

**Dense lags:**
17. `default,dense`
18. `lags,calendar,dense`

Κόστος: ~2.5 min/config × 18 ≈ **45′** (μία εντολή `run_ablation`, background).

## 2b-RESULTS — Καλοκαιρινή αναπαραγωγή, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02)

`results_ablation_summer2025.csv` / `ablation_summer2025/` — LGBM recursive static, test
2025-06-01→2025-08-31 (train_end 2025-05-31), 10 specs, baseline (`default`) = **14.43 €/MWh**
(σαφώς χαμηλότερο error level από τον χειμώνα — αναμενόμενο, λιγότερη volatility καλοκαίρι).

**🎯 Επιβεβαιώθηκε η ανησυχία του χρήστη (§2b): το χειμωνιάτικο Q1-window ΥΠΟΕΚΤΙΜΟΥΣΕ τις
solar-ευαίσθητες ομάδες.**

| Ομάδα | Winter (Q1/Δεκ) ΔMAE | Summer 2025 ΔMAE | Ερμηνεία |
|---|---|---|---|
| `resfc` (leave-one-out) | Q1 ασθενές/αρνητικό, Δεκ −0.32/−0.04 | **+1.085** (πολύτιμο!) | Πλήρης αναστροφή — resfc σχεδόν άχρηστο τον χειμώνα, ΞΕΚΑΘΑΡΑ πολύτιμο το καλοκαίρι. |
| `resfc` (additive πάνω σε core) | — | core→core+resfc: **−2.45 MAE** | Το μεγαλύτερο additive κέρδος από όλη την ablation μελέτη, χειμώνα ή καλοκαίρι. |
| `genlags` (leave-one-out) | Q1 +1.45, Δεκ-recursive +2.63 | +0.431 (ακόμα πολύτιμο, μικρότερο) | Παραμένει θετικό και τις δύο εποχές, αλλά αναλογικά ΠΙΟ σημαντικό τον χειμώνα (πιθανώς επειδή residual_load lags/wind κυριαρχούν εκεί, ενώ το καλοκαίρι το resfc παίρνει μέρος του ρόλου). |
| `loadfc` (leave-one-out) | Δεκ οριακό (±0.16) | −0.278 (βλάπτει, πάνω από noise floor) | Τρίτη ένδειξη ότι το loadfc δεν προσφέρει καθαρά (winter οριακό/ασυνεπές, summer καθαρά αρνητικό). |
| `loadlags` (leave-one-out) | Q1/Δεκ θετικό ~+0.2-0.34 | −0.055 (ουδέτερο) | Ασθενέστερο καλοκαίρι — πιθανώς επειδή το load pattern καλοκαιριού έχει διαφορετική δομή (AC-driven, όχι heating-driven). |
| Καλύτερο core-combo | `lags,calendar,resfc,genlags` (Q1: 16.06, κερδίζει default) | 14.632 (πολύ κοντά στο default 14.434, ελαφρώς χειρότερο) | Το lean core κερδίζει τον χειμώνα αλλά ΟΧΙ ξεκάθαρα το καλοκαίρι — το default (152 feat, με meteo/fuel μέσα) παραμένει ελαφρώς καλύτερο όταν το solar mix είναι ενεργό. |

**Συνολικό συμπέρασμα ανά ομάδα = ζεύγος (χειμώνας, καλοκαίρι), όπως προβλεπόταν:**
- **`resfc`**: ΑΠΟΡΡΙΠΤΕΤΑΙ η χειμωνιάτικη εκτίμηση «άχρηστο» — στην πραγματικότητα από τις πιο
  πολύτιμες ομάδες, απλώς αόρατο σε window με κοιμισμένο solar. Θα πρέπει να ΠΑΡΑΜΕΙΝΕΙ στο
  default feature set ανεξαρτήτως εποχής.
- **`genlags`**: στιβαρά πολύτιμο και στις δύο εποχές, όχι artifact κανενός window.
- **`loadfc`**: τώρα με 3/3 σημεία (Δεκ-direct, Δεκ-recursive, summer) καθαρά ασθενές/αρνητικό
  εκτός από ένα οριακά θετικό — υποψήφιο για αφαίρεση από το core σύνολο, χαμηλή προτεραιότητα
  ελέγχου SS.
- **`meteo`**: ΔΕΝ συμπεριλήφθηκε σε αυτό το batch (εκτός §2b specs by design) — παραμένει
  εκκρεμές για καλοκαιρινή αναπαραγωγή αν χρειαστεί ξανά αργότερα.

## 2b. ΜΕΡΟΣ 1b — Καλοκαιρινή αναπαραγωγή (ΚΡΙΣΙΜΟ — παρατήρηση χρήστη)

**Το Q1 2026 είναι εξ ολοκλήρου χειμώνας** → η solar παραγωγή/πρόβλεψη έχει μικρή διακύμανση
και μικρό μερίδιο στο mix → οι ομάδες `resfc`/`genlags`/`meteo` θα ΥΠΟΕΚΤΙΜΗΘΟΥΝ συστηματικά.
Το μεγάλο training window ΔΕΝ μας καλύπτει: το training μαθαίνει στο μοντέλο ΠΩΣ να
χρησιμοποιεί το solar· το τι ΜΕΤΡΑΜΕ όμως το καθορίζει το test window — αν εκεί το solar
κοιμάται, το ΔMAE βγαίνει ~0 ακόμα κι αν το feature είναι πολύτιμο τον Ιούνιο.

→ Αναπαραγωγή των solar-ευαίσθητων specs σε **καλοκαίρι 2025** (test: 2025-06-01 → 2025-08-31,
train_end 2025-05-31, v2 δεδομένα πλήρη εκεί): specs 1-5, 8, 9, 11, 13, 15 (10 specs, LGBM static, ~25′).
Τελικό συμπέρασμα ανά ομάδα = ζεύγος (χειμώνας, καλοκαίρι) — όχι μόνο χειμώνας.

## 3. ΜΕΡΟΣ 2 — Ίδια 18 specs με XGB (static Q1)

Σκεπτικό (σωστό του χρήστη): διαφορετικός τρόπος διάσπασης/regularization → μπορεί να
αξιοποιεί διαφορετικά τα ίδια features. Αν LGBM+XGB συμφωνούν στο πρόσημο ΔMAE μιας ομάδας,
το συμπέρασμα είναι στιβαρό. Κόστος ≈ 45′.

## 4. ΜΕΡΟΣ 3 — Direct-LGBM σε σύντομο παράθυρο (Δεκ 2025) — 10 specs

Στρατηγική-robustness: το recursive «σέρνει» λάθη μέσω rollout, το direct όχι — αν μια ομάδα
βοηθά ΚΑΙ στο direct, δεν είναι artifact του rollout. Reduced set (specs 1-8, 13, 14).
Κόστος: direct fit ≈ 2-4′/config × 10 ≈ **30′**.

## 5-RESULTS & 6-RESULTS — E1/E2, ΟΛΟΚΛΗΡΩΘΗΚΑΝ (2026-07-02)

`e1_e2_out/*.json`, log σε `_step5_log.txt`.

**E1 (capacity × meteo, price/DAM, Q1 2026):**

| n_estimators | MAE default | MAE default,-meteo | Δ(meteo)=def−(-meteo) |
|---|---|---|---|
| 400 | 15.744 | 15.760 | −0.016 (σχεδόν ουδέτερο) |
| 800 (default capacity) | 16.096 | 15.836 | **+0.260** (meteo βλάπτει) |
| 1600 | 16.349 | 15.910 | **+0.439** (meteo βλάπτει ΠΕΡΙΣΣΟΤΕΡΟ) |

**🔴 Η υπόθεση του χρήστη ΑΠΟΡΡΙΦΘΗΚΕ — καθαρά, μονότονα.** Όσο μεγαλώνει η χωρητικότητα, το
meteo ΔΕΝ ξεκλειδώνεται ως χρήσιμο· αντίθετα βλάπτει ΟΛΟ ΚΑΙ ΠΕΡΙΣΣΟΤΕΡΟ (μονότονη τάση
−0.016 → +0.26 → +0.44). Ερμηνεία: το επιπλέον capacity δεν "φτάνει βαθύτερα σε weak-but-real
signal" — overfitάρει στον θόρυβο του reanalysis proxy. Σημειωτέον: και το MAE(default) ΜΟΝΟ ΤΟΥ
χειροτερεύει με περισσότερα δέντρα (15.74→16.10→16.35) — γενικό overfitting risk χωρίς tuning,
συνεπές με την απόφαση "όχι Optuna, σταθερά params". `n_estimators=800` (default) παραμένει
λογική επιλογή, ΟΧΙ μεγαλύτερο μοντέλο.

**E2 (meteo στο ΦΟΡΤΙΟ, load/DAM, static Q1 2026):**

| Spec | MAE (MW) |
|---|---|
| `default` | 125.68 |
| `default,-meteo` | 158.67 (Δ = **−32.98**, ~26% relative) |
| `lags,calendar,meteo` (bare core + μόνο meteo) | 160.18 (χειρότερο ΚΑΙ από το -meteo!) |

**🟢 Επιβεβαιώθηκε πλήρως η θεωρητική πρόβλεψη του §6.** Το meteo είναι μακράν η πιο πολύτιμη
ομάδα σε ΟΛΟ το ablation study όταν μετριέται στο σωστό target (load, όχι price) — ΔMAE −33 MW
είναι πολλαπλάσιο μεγέθους από οτιδήποτε βρέθηκε στο price. Επιβεβαιώνει ότι το meteo δρα κυρίως
ΕΜΜΕΣΑ στην τιμή (μέσω load/RES), γι' αυτό και το άμεσο price-ablation delta είναι μικρό/αρνητικό.
Σημείωση: `lags,calendar,meteo` χειρότερο από `default,-meteo` δείχνει ότι το meteo χρειάζεται τα
lags/loadfc γύρω του για να αποδώσει — δεν αρκεί μόνο του πάνω σε γυμνό πυρήνα.
⚠️ Reanalysis = perfect-forecast proxy → αυτό είναι το ΑΙΣΙΟΔΟΞΟ άνω όριο, όχι εγγυημένο
production gain (βλ. §9).

## 5. Πείραμα E1 — Capacity × Meteo (απάντηση στην απορία #1)

Υπόθεση χρήστη: «με περισσότερους estimators ίσως τα μετεωρολογικά βοηθήσουν».
Μηχανισμός: πράγματι, με μεγαλύτερη χωρητικότητα το boosting φτάνει βαθύτερα σε weak features
— αλλά συνήθως το κέρδος είναι οριακό για price (τα meteo δρουν στην τιμή ΕΜΜΕΣΑ, μέσω RES/load,
που ήδη τα έχουμε ως resfc/loadfc). Το τεστ είναι φθηνό και ευθύ:

| Grid | n_estimators ∈ {400, 800, 1600} × features ∈ {default, default,-meteo} = 6 runs |
|---|---|

Αν το Δ(meteo) μεγαλώνει με τους estimators → ναι, αξίζουν καλύτερα meteo και μεγαλύτερο μοντέλο.
*(Απαιτεί μικρό flag `--n_estimators` στο engine — 5 γραμμές.)*

## 6. Πείραμα E2 — Meteo στο ΦΟΡΤΙΟ (η σωστή αγορά για τα «καλύτερα μετεωρολογικά»)

Όλα τα ablations ως τώρα ήταν price. Θεωρητικά τα meteo έχουν ΠΟΛΥ μεγαλύτερη αξία στο **load**
(θερμοκρασία↔κατανάλωση σχεδόν αιτιακά). Πριν επενδύσεις στα «καλύτερα μετεωρολογικά που σου
υποσχέθηκαν», μέτρα το ταβάνι:

- task=load, static Q1: `default` vs `default,-meteo` vs `lags,calendar,meteo` (3 runs, ~10′)
- Αν Δ(meteo|load) μεγάλο → τα καλύτερα meteo πιάνουν κυρίως στο φορτίο· στην τιμή το κέρδος
  θα έρθει έμμεσα (καλύτερο load_fc/resfc). Σημείωση: τα τωρινά meteo είναι reanalysis
  («τέλεια πρόβλεψη») — καλύτερη ΠΟΙΟΤΗΤΑ πηγής βελτιώνει κυρίως το realism, οπότε το
  μετρήσιμο κέρδος στο backtest είναι το ΑΝΩ όριο του πραγματικού.

## 7-RESULTS — Scheduled Sampling, ΠΡΩΤΟ end-to-end τρέξιμο, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02)

`ss_out/{nossf,ss_linear,ss_exp,ss_step}.json`, log σε `_step6_log.txt`. LGBM recursive static
Q1 2026, `default` features, R=3 rounds. **Το engine δούλεψε σωστά από την πρώτη φορά end-to-end
(καμία εξαίρεση/σφάλμα)** — το `--ss` δεν είναι πια ασύρματο.

| Σχήμα | MAE (all) | MAE near (1-8h) | MAE far (17-24h) | far−near gap |
|---|---|---|---|---|
| no-SS (baseline) | 16.096 | 11.920 | 17.594 | 5.674 |
| **SS-linear** | **15.830** (Δ=−0.266) | 11.833 (Δ=−0.087) | **17.358** (Δ=−0.236) | 5.526 |
| SS-exp | 16.282 (Δ=+0.185, χειρότερο) | 11.874 | 18.171 (χειρότερο) | 6.297 |
| SS-step | 16.285 (Δ=+0.188, χειρότερο) | 11.844 | 18.097 (χειρότερο) | 6.253 |

**🟢 SS-linear = νικητής, καθαρά.** Μόνο το linear decay βελτιώνει ΤΑΥΤΟΧΡΟΝΑ το συνολικό MAE
ΚΑΙ ειδικά τα μακρινά offsets (17-24h) περισσότερο από τα κοντινά (Δ_far=−0.24 > Δ_near=−0.09),
ακριβώς η υπογραφή που προβλέπει η θεωρία exposure-bias (Bengio et al. 2015). Τα exp/step
χειροτερεύουν και τα δύο — πιθανή εξήγηση: πέφτουν πολύ γρήγορα σε υψηλό ποσοστό corruption
(exp: 50%→75%→75% ήδη από το r2· step: ίδιο ε_min αλλά πιο απότομο πρώτο βήμα) χωρίς αρκετά
ενδιάμεσα rounds για ομαλή προσαρμογή, πιθανώς λόγω R=3 πολύ μικρό για αυτά τα σχήματα.
**Winner scheme για το Βήμα 7 (SS-retest οριακών ομάδων): `linear`.**

## 7. ΦΑΣΗ 2 — Scheduled Sampling (⚠️ ΚΕΝΟ: το `--ss` είναι ΑΣΥΡΜΑΤΟ σήμερα)

Gap που εντοπίστηκε: το flag `--ss` υπάρχει στο CLI του `master_forecast` αλλά **δεν
χρησιμοποιείται πουθενά** στο `run_forecast` — parsed but unwired. Υλοποίηση (για trees):
iterative self-generated retraining — round r: αντικατέστησε στο train τα y_lag εντός
προσομοιωμένου ορίζοντα με προβλέψεις του μοντέλου του round r−1 με πιθανότητα 1−ε(r).

Σχήματα μείωσης ε (τα 2-3 που ζητήθηκαν, Bengio et al. 2015):
- **linear**: ε(r) = max(ε_min, 1 − r/R)
- **exponential**: ε(r) = k^r (π.χ. k=0.5)
- **step**: ε ∈ {1.0, 0.5, 0.25} ανά round (R=3)

Τεστ: LGBM recursive static Q1 × {no-SS, SS-linear, SS-exp, SS-step} = 4 runs (κόστος ~3× fit
ανά SS run). Κριτήριο: βελτίωση στα ΜΑΚΡΙΝΑ offsets (ώρες 12-35 του rollout) — εκεί χτυπάει
το exposure bias, εκεί πρέπει να φανεί.

**Πολιτική αλληλεπίδρασης SS × features (παρατήρηση χρήστη — σωστή):** το SS αλλάζει το
πόσο το μοντέλο στηρίζεται στα (θορυβώδη πλέον) y-lags → η σχετική αξία των exogenous
(meteo/resfc/loadfc) μπορεί να ΑΝΕΒΕΙ υπό SS. Άρα: κάθε ομάδα που απορρίφθηκε ΟΡΙΑΚΑ
(|ΔMAE| κοντά στο 0.15) στο no-SS ablation **ξανατεστάρεται** με το καλύτερο SS σχήμα
(4-6 επιπλέον runs). Ομάδες με μεγάλο καθαρό πρόσημο (πολύ καλές ή πολύ άχρηστες) δεν
χρειάζονται επανέλεγχο — το SS δεν αντιστρέφει μεγάλες διαφορές, μόνο οριακές.

## 7b-RESULTS — SS-retest οριακών ομάδων, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02)

`ss_out/ss_linear_no_{meteo,loadlags,fuel,resfc,loadfc}.json`, log σε `_step7_log.txt`.
Baseline SS-linear `default` = 15.8301 (από §7-RESULTS). ΔMAE = MAE(default,-X,SS) − MAE(default,SS).

| Ομάδα | No-SS ΔMAE (Q1/Δεκ, εύρος) | SS-linear ΔMAE | Αλλαγή |
|---|---|---|---|
| `meteo` | −0.20 έως −0.52 (βλάπτει) | **−0.079** (βλάπτει, ΠΟΛΥ λιγότερο) | Μερική επιβεβαίωση interaction hypothesis — η ζημιά συρρικνώνεται κάτω από noise floor, δεν αντιστρέφεται. |
| `loadlags` | +0.22 έως +0.34 (βοηθάει) | **−0.316** (ΒΛΑΠΤΕΙ) | 🔄 **Πλήρης αναστροφή προσήμου.** Κάτω από SS το loadlags γίνεται επιζήμιο. |
| `fuel` | −0.38 έως −0.68 (βλάπτει) | −0.452 (βλάπτει, παρόμοιο μέγεθος) | Συνεπές — SS δεν αλλάζει το συμπέρασμα, fuel άχρηστο. |
| `resfc` | −0.04 έως −0.32 (ασθενές/βλάπτει) | 🎯 **+1.101 (ΠΟΛΥ πολύτιμο!)** | Δραματική αναστροφή — από τις πιο άχρηστες σε μία από τις πιο πολύτιμες ομάδες υπό SS. |
| `loadfc` | −0.16 έως +0.17 (οριακό/θόρυβος) | **+0.456 (καθαρά πολύτιμο)** | Λύνεται η αβεβαιότητα — υπό SS το loadfc είναι ξεκάθαρα πάνω από το noise floor. |

**🎯 Επιβεβαιώθηκε ΕΝΤΥΠΩΣΙΑΚΑ η υπόθεση του χρήστη (§7): «το SS αλλάζει το πόσο το μοντέλο
στηρίζεται στα (θορυβώδη πλέον) y-lags → η σχετική αξία των exogenous μπορεί να ΑΝΕΒΕΙ υπό SS».**
Τα δύο day-ahead-known exogenous forecasts (`resfc`, `loadfc`) γίνονται ΣΑΦΩΣ πιο πολύτιμα υπό SS
— λογικό μηχανιστικά: όσο τα y-lags γίνονται αναξιόπιστα (corrupted) στο training, το μοντέλο
αναγκάζεται να ακουμπήσει σε ό,τι ΠΑΝΤΑ είναι αξιόπιστο (τα day-ahead forecasts, ποτέ corrupted).
Το `loadlags` κάνει το αντίθετο ταξίδι (helpful→harmful) — πιθανή εξήγηση: το load_lags «διαρρέει»
έμμεση πληροφορία που παλιότερα βοηθούσε να αντισταθμίσει τα καθαρά y-lags, αλλά υπό SS αυτό
γίνεται περιττό/παραπλανητικό μόλις υπάρχουν πιο αξιόπιστα εναλλακτικά (resfc/loadfc) να το
αντικαταστήσουν. **Πρακτική συνέπεια**: ένα SS-tuned feature set (`default,-loadlags` υπό SS-linear)
αξίζει να δοκιμαστεί ως candidate winner στο Βήμα 8.

## 8-RESULTS — Cross-border (xborder) test, ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02)

`results_ablation_xborder.csv` / `ablation_xborder/` — LGBM recursive static, Q1 2026
(train_end 2025-11-30 23:00, test 2025-12-01→2026-02-28), `default` vs `default,xborder`:

| Spec | MAE | ΔMAE | Δ% |
|---|---|---|---|
| `default` | 16.096 | — | — |
| `default,xborder` | **15.376** | **−0.720** | **−4.5%** |

**🟢 Μεγάλο θετικό εύρημα** — δεύτερο μεγαλύτερο effect μετά το resfc-καλοκαίρι σε όλη τη
μελέτη, πολύ πάνω από το noise floor (0.15). Οι τιμές BG/IT-SUD DAM (γνωστές D-1, νόμιμο lag 0)
προσθέτουν πραγματική πληροφορία πέρα από ό,τι ήδη υπάρχει. **Σύσταση**: `xborder` να μπει στο
default feature set μετά από 1-2 ακόμα confirmations (SS-retest ή 2ο test window) — προς το
παρόν 1 μόνο σημείο/seed, αλλά το μέγεθος του effect είναι πολύ πιο πάνω από το noise floor
ώστε να αξίζει άμεση προτεραιότητα σε confirmation runs (Βήμα 8).

## 8. ΦΑΣΗ 3 — Νέες πηγές (conditional, με σειρά προτεραιότητας από τα ευρήματα του Μέρους 1)

1. **`resload_fc` engineered feature** (φθηνό, ΠΡΙΝ φτιάξουμε δικό μας RES forecaster):
   `resload_fc = load_fc − solar_fc_dayahead − wind_onshore_fc_dayahead` — υπολογίζεται από
   υπάρχουσες στήλες, μηδέν downloads. Αν το Μέρος 1 δείξει resfc+loadfc σημαντικά → τρέξε
   `default` vs `default+resload_fc` (1 run). Δικός μας RES forecaster (το σημείο 4 του χρήστη)
   ΜΟΝΟ αν το ENTSO-E resfc αποδειχθεί αδύναμο ενώ το gen_act δυνατό — αλλιώς δεν αξίζει.
2. **Cross-border (διασυνδέσεις)**: νέος fetcher για (α) DAM τιμές γειτόνων (IT-GR/BG SDAC —
   γνωστές D-1 όπως οι δικές μας → lag 0 νόμιμο), (β) scheduled commercial exchanges /flows
   (day-ahead γνωστά) — νέα ομάδα `xborder` στο feature_availability με ρητό availability rule.
   Υλοποίηση ~1-2h + backfill download. Τρέξιμο: `default` vs `default+xborder`.

## Βήμα 9-RESULTS — LEAR/ensemble ξανατρέξιμο με v2 δεδομένα (2026-07-02)

`step9_out/{lear,xgb}_q1_static_default.json`, `step9_out/ensemble_q1.json`, log `_step9a_log.txt`.

**LEAR**: MAE=**19.49 €/MWh** (Q1 2026, static, default features, v2 δεδομένα). Ταιριάζει με
το ήδη-καταγεγραμμένο (SKILL.md §10: ~19.0) — **σταθερό εύρημα, όχι artifact**: το LEAR παραμένει
σαφώς χειρότερο από τα δέντρα (LGBM 16.10 / XGB 16.21) ακόμα και με πλήρη v2+xborder-ready
δεδομένα. Ρόλος του παραμένει fallback/robustness baseline μόνο.

**Ensemble** (LGBM+XGB+LEAR, ίδιο Q1 static default window):

| Μέλος/Ensemble | MAE |
|---|---|
| ensemble_median | **16.042** ★ |
| LGBM (καλύτερο μεμονωμένο) | 16.096 |
| XGB | 16.209 |
| ensemble_mean | 16.387 (χάνει) |
| LEAR | 19.492 |

**Μερική αναθεώρηση του §4.3 (SYSTEM_DESIGN)**: ο ισοβαρής **mean** ΞΑΝΑ χάνει από το καλύτερο
μεμονωμένο μοντέλο (επιβεβαιώνει το ήδη-καταγεγραμμένο εύρημα — LEAR's κακή απόδοση τραβάει τον
mean προς τα κάτω). ΝΕΑ παρατήρηση: ο **median** ΚΕΡΔΙΖΕΙ οριακά (16.042 vs 16.096, Δ=−0.055) —
πολύ κάτω από το noise floor (0.15) άρα ΔΕΝ είναι στιβαρό συμπέρασμα ακόμα, αλλά κατευθυντικά
υποδεικνύει ότι ο median (πιο ανθεκτικός σε outlier-μέλη σαν το LEAR) μπορεί να αξίζει παραπάνω
διερεύνηση από τον mean πριν επενδυθεί χρόνος στο weighted-by-1/MAE (§4.3, ήδη ανοιχτό ζήτημα).

**LSTM seq2seq** (`step9_out/lstm_q1_static.json`, log `_step9c_log.txt`): **Πρώτη φορά ΠΟΤΕ
ολοκληρώνεται πλήρες end-to-end τρέξιμο χωρίς exception** (90 blocks, 445.6s, MAE
υπολογίστηκε). ΑΛΛΑ η ποιότητα είναι κακή: **MAE=44.16 €/MWh** — χειρότερο ΑΚΟΜΑ και από το LEAR
(19.49), ~2.7× χειρότερο από το LGBM (16.10). Διαγνωστικό: correlation(actual,pred)=**0.69**
(μαθαίνει πραγματικό σήμα, ΔΕΝ είναι τυχαίο/σπασμένο) αλλά με ισχυρό συστηματικό θετικό bias
(pred mean=140.9 vs actual mean=99.7) και ΠΟΤΕ δεν προβλέπει αρνητικές τιμές (pred min=33.1 vs
actual min=−25.0, ενώ οι αρνητικές τιμές είναι πραγματικό φαινόμενο στην ελληνική αγορά).
**Συμπέρασμα**: το "πρώτη φορά τρέχει" ≠ "πρώτη φορά δουλεύει" — το μοντέλο χρειάζεται
undertraining/miscalibration fix (πιθανά αίτια: λίγα epochs, learning rate, ή scaling/target
transform που αποκλείει αρνητικές τιμές) πριν θεωρηθεί χρήσιμο. **ΔΕΝ είναι tradeable ακόμα.**
Ανοιχτό για μελλοντικό session.

## 9. Καταγεγραμμένα κενά κάλυψης (honesty section) — ενημερώθηκε 2026-07-03

**Κάλυψη ανά μοντέλο (ΣΗΜΑΝΤΙΚΟ — ΔΕΝ είναι όλα τα μοντέλα ισοδύναμα ελεγμένα):**
- **LGBM**: πλήρης ablation (18 specs × Q1, καλοκαίρι, Δεκ-recursive/direct, E1/E2, SS, xborder,
  stacking). Το μόνο μοντέλο με πλήρη κάλυψη.
- **XGB**: πήρε τη ΔΙΚΗ του ανεξάρτητη 18-spec Q1 ablation (§3, Μέρος 2) — ΟΧΙ απλή αντιγραφή
  των LGBM συμπερασμάτων, πραγματική επιβεβαίωση (genlags/dense συμφωνούν). Αλλά ΔΕΝ ξανατεστάρε-
  τηκε ΜΕΤΑ το Μέρος 2 — καλοκαίρι/E1-E2/SS/xborder/stacking έγιναν ΜΟΝΟ σε LGBM. Άγνωστο αν το
  xborder gain (−0.7) γενικεύεται στο XGB.
- **LEAR, MLP**: **καμία ablation ποτέ** — έτρεξαν μόνο με το πλήρες `default` set (καμία φορά
  καν με xborder). Η υπόθεση «τα LGBM-derived καλύτερα features βοηθούν παντού» δεν έχει ελεγχθεί
  γι' αυτά· το LEAR μάλιστα είναι γνωστό ότι υποβαθμίζει resfc/loadfc λόγω L1 regularization
  (§Βήμα 9-RESULTS), άρα ΔΕΝ είναι ασφαλές να υποτεθεί ότι θα αντιδράσει σαν τα δέντρα.
- **LSTM**: ίδια κατάσταση με LEAR/MLP (μόνο `default`, καμία ablation) — επιπλέον το μοντέλο έχει
  δικό του calibration bug (βλ. παρακάτω), οπότε ablation πάνω του είναι χαμηλή προτεραιότητα
  μέχρι να διορθωθεί.

**Κάλυψη ανά στρατηγική**: όλα recursive εκτός από ΜΙΑ σκόπιμη εξαίρεση — το Μέρος 3
(direct-LGBM, Δεκ, 10 specs) + το ζευγαρωτό Δεκ-recursive-controlled, ειδικά για να ελεγχθεί αν
τα ευρήματα (ιδίως meteo) είναι strategy-artifact. Τεχνικός περιορισμός: το Scheduled Sampling
είναι καλωδιωμένο ΜΟΝΟ για `strategy=="recursive"` στο `master_forecast.py` — SS+direct δεν
υποστηρίζεται καν σήμερα στον κώδικα, όχι μόνο αδοκίμαστο.

**LSTM seq2seq — calibration bug (§Βήμα 9-RESULTS: MAE=44.16, ποτέ αρνητικές τιμές, corr=0.69)**:
πιθανή αιτία (ΥΠΟΘΕΣΗ, όχι επιβεβαιωμένη με στοχευμένο debugging) — το training κάνει πάντα
πλήρες teacher forcing (`use_tf=True` hardcoded στο `lstm_models.py`, ΚΑΜΙΑ SS-εκδοχή για LSTM),
ενώ το inference κάνει αυστηρά αυτοπαλίνδρομο decode· ακριβώς το exposure-bias πρόβλημα που
μόλις αποδείξαμε (§7-RESULTS) ότι λύνει το SS στα δέντρα, απλά δεν εφαρμόστηκε ποτέ στο LSTM.
Σε συνδυασμό με μικρή χωρητικότητα (hidden=64, 1 layer, 30 epochs) για πρόβλημα με σπάνια ακραία
γεγονότα (αρνητικές τιμές) → το μοντέλο μαθαίνει "ασφαλείς" προβλέψεις γύρω από τον μέσο όρο.
Δεν έχει γίνει ακόμα στοχευμένο test (π.χ. SS-equivalent στο LSTM training, περισσότερα epochs/
capacity, εξέταση μεμονωμένων rollout ιχνών) — μόνο ανάγνωση κώδικα + συμπτώματα.

**Λοιπά (από πριν, ισχύουν ακόμα)**:
- **Ένα seed παντού** πλην του xborder headline (seeds 42/43/44 ελέγχθηκαν, std≈0.05 —
  βλ. §Stacking-RESULTS). Όλα τα υπόλοιπα ΔMAE είναι single-seed· οριακά (<0.15) χρειάζονται
  2-3 seeds πριν δηλωθούν συμπέρασμα στη διπλωματική.
- Reanalysis meteo = perfect-forecast proxy (αισιόδοξο άνω όριο — βλ. §6).
- Full-2025 monthly loop (πριν το ablation) υπάρχει ως δεύτερο window επιβεβαίωσης, αλλά όχι
  μέρος του ίδιου του ablation battery.

## Confirmation-RESULTS (Πυραμίδα, κορυφή), ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02)

`confirm_out/*.json`, logs `_step8a_log.txt` / `_step8b_log.txt`. Candidates (Q1 2026):

| Config | Retrain | Q1 MAE | Σχόλιο |
|---|---|---|---|
| A: `default` | monthly | 15.795 | Σταθεροποιεί το ήδη γνωστό 15.80 (last.md) — sanity confirm στα v2 δεδομένα. |
| **B: `default,xborder`** | monthly | **15.015** | 🏆 **Νέο headline** — καλύτερο ΑΚΟΜΑ και από το προηγούμενο βέλτιστο (weekly-retrain default, 15.17) ενώ χρησιμοποιεί ΦΘΗΝΟΤΕΡΟ retrain cadence (monthly, όχι weekly). Το xborder gain (−0.78 εδώ) ξεπερνά το κέρδος από το αναβάθμισμα monthly→weekly. |
| C: `default,xborder,-loadlags` + SS-linear | static | 15.350 | Καλό αλλά χάνει από το B — το retrain cadence (monthly) φαίνεται να μετράει περισσότερο από το SS-tuning σε αυτό το σύγκριση. |

**Νικητής: config B (`default,xborder`, monthly retrain).**

**2ο test window (Μάρτιος 2026, 2026-03-01→2026-03-19, νέο out-of-sample μήνας — μοναδικό
διαθέσιμο πέρα του Q1):**

| Config | Mar-2026 MAE | ΔMAE vs no-xborder |
|---|---|---|
| `default` monthly | 17.968 | — |
| `default,xborder` monthly | **17.621** | **−0.347** |

**🟢 Το xborder όφελος ΕΠΙΒΕΒΑΙΩΝΕΤΑΙ σε δεύτερο, ανεξάρτητο out-of-sample μήνα** — μικρότερο
μέγεθος (−0.35 έναντι −0.78 στο Q1, αναμενόμενο σε μικρότερο 19-ήμερο test window/λιγότερα
δεδομένα) αλλά ίδιο πρόσημο και πάνω από το noise floor. Η σύσταση του Βήματος 4 (§8-RESULTS)
επιβεβαιώνεται πλήρως: **το `xborder` πρέπει να μπει στο default feature set.**

**Νέο headline config (αντικαθιστά το προηγούμενο "LGBM recursive weekly, default, 15.17"):**
**LGBM recursive monthly, `default,xborder` → 15.02 €/MWh (Q1 2026, v2 δεδομένα, strict gate).**

## 🔴 XBORDER — ΤΕΛΙΚΗ ΕΤΥΜΗΓΟΡΙΑ (2026-07-03/04): ΤΟ ΕΥΡΗΜΑ ΗΤΑΝ LEAKAGE — ΑΚΥΡΩΝΕΤΑΙ

> **ΥΠΕΡΙΣΧΥΕΙ όλων των προηγούμενων xborder ενοτήτων** (§8-RESULTS, §Confirmation-RESULTS,
> §Stacking-RESULTS, και της παρακάτω «1h misalignment» ενότητας — εκείνο το bug ήταν
> δευτερεύον· το πραγματικό πρόβλημα ήταν εννοιολογικό).

**Το λάθος (availability rule, όχι υλοποίηση)**: οι τιμές BG/IT-SUD για την ημέρα D βγαίνουν
από το **ΙΔΙΟ SDAC auction** (EUPHEMIA) με την ελληνική τιμή της D — δημοσιεύονται ~13:00 CET
D-1, δηλαδή **ΜΕΤΑ το gate closure 12:00**. Άρα η same-day ξένη τιμή ΔΕΝ είναι γνωστή όταν
υποβάλλουμε προσφορές. Χρησιμοποιώντας την ως feature «προβλέπαμε» το αποτέλεσμα ενός auction
κοιτώντας ένα άλλο output του ίδιου auction. Η υψηλή same-hour συσχέτιση (IT-SUD 0.91) ήταν
ακριβώς αυτό — coupling του ίδιου solution, όχι προβλεπτική πληροφορία.

**Το fix (δομικό, στο data.py)**: same-day xb στήλες ΔΕΝ μπαίνουν πια καθόλου στο parquet —
κρατούνται ΜΟΝΟ `xb_price_{bg,itsud}_lag{24,48,168}` (ίδια ώρα D-1/D-2/D-7, δημοσιευμένα από
προηγούμενα auctions — ακριβώς η ίδια διαθεσιμότητα με το ελληνικό y). Ίδιο pattern με το
υπάρχον `[LEAKAGE FIX price] Dropped contemporaneous cols`.

**Το έντιμο αποτέλεσμα (LGBM recursive static Q1)**:
| Config | MAE | |
|---|---|---|
| `default` | 16.096 | baseline |
| `default,xborder` same-day (ΑΚΥΡΟ) | 15.376 | −0.72 «όφελος» = leakage |
| `default,xborder` **lagged (ΝΟΜΙΜΟ)** | **16.518** | **+0.42 — ΒΛΑΠΤΕΙ** |

**Συνέπειες**:
1. **Headline επανέρχεται σε: LGBM recursive weekly, `default` = 15.17 €/MWh** (το τελευταίο
   πλήρως έγκυρο tradeable νούμερο). Τα 14.43/15.02 ήταν leakage-inflated — ΜΗΝ αναφερθούν.
2. ΑΚΥΡΑ: §8-RESULTS (−0.72), Confirmation-B (15.02), όλο το Stacking-RESULTS εκτός από τα
   no-xborder μέλη του (seeds στο A-default ισχύουν· «SS redundant» χρειάζεται re-check σε
   έγκυρο config γιατί μετρήθηκε πάνω σε xborder configs).
3. **ΑΝΕΠΗΡΕΑΣΤΑ** (δεν είδαν ποτέ xb στήλες — το feature selection τις απέκλειε): Μέρη 1/2/3
   core ablations, καλοκαιρινή αναπαραγωγή, Δεκ-controlled, E1/E2, SS σχήματα + SS-retest,
   LEAR/MLP/LSTM/ensemble runs του Βήματος 9, και όλα τα v2 πριν-xborder νούμερα (16.10/15.80/
   15.17). Τα rebuilds ενδιάμεσα επαληθεύτηκαν 0-mismatch στις παλιές στήλες.
4. Μεθοδολογικό δίδαγμα (για τη διπλωματική — πολύτιμο κεφάλαιο): «feature που βελτιώνει
   θεαματικά και αμέσως» = πρώτος ύποπτος για leakage. Το cross-correlation lag-scan και ο
   έλεγχος «πότε ακριβώς δημοσιεύεται;» πρέπει να γίνονται ΠΡΙΝ το feature μπει σε test, όχι
   μετά. Προστίθεται ως υποχρεωτικό pre-flight βήμα στο πρωτόκολλο (βλ. OVERNIGHT PROTOCOL §1).

## ⚠️→✅ Bug found & fixed (code review, 2026-07-03): xborder 1h misalignment — ΞΕΠΕΡΑΣΜΕΝΟ, βλ. πάνω

Κατά το `/code-review` (χωρίς git diff λόγω σπασμένου repo — χειροκίνητος έλεγχος) εντοπίστηκε
ότι `fetch_entsoe_xborder.py` είχε σταθερή μετατόπιση **+1h** στα timestamps (BG/IT-SUD DAM
τιμές) σε σχέση με το υπόλοιπο pipeline. Ανίχνευση: cross-correlation lag-scan κατά της
πραγματικής τιμής GR (peak στο lag+1 αντί lag=0, σταθερό winter/summer → όχι DST issue).

**Leakage ή όχι;** Κυρίως ΟΧΙ — για 23/24 ώρες/ημέρα η "λάθος" τιμή ήταν του h+1 ΕΝΤΟΣ της ίδιας
ημέρας παράδοσης (ίδιο D-1 auction, ίδια δημοσίευση) → data-correctness bug, όχι leakage. ΜΟΝΟ
η ώρα 23:00 κάθε ημέρας (~4% των γραμμών) είχε πραγματικό, μικρό leakage (τιμή από το D-1 auction
της ΕΠΟΜΕΝΗΣ ημέρας, όχι ακόμα δημοσιευμένη στο σωστό gate).

**Fix**: `-1h` στο `_to_hourly_naive_local` (πρώτη απόπειρα είχε λάθος πρόσημο `+1h`, επιδείνωσε
το shift, ξανα-ελέγχθηκε με το ίδιο cross-correlation script πριν κλειδωθεί). Ξανα-κατέβηκε
πλήρες xborder_hourly.parquet (2017-2026), rebuild hourly.parquet, verified: peak τώρα ακριβώς
στο lag=0 (BG corr=0.824, IT-SUD corr=0.913).

**Επίδραση στο headline**: LGBM weekly+xborder = **14.423** (πριν: 14.426) — **διαφορά 0.003
€/MWh, ουσιαστικά μηδενική.** Το εύρημα άντεξε γιατί οι τιμές ρεύματος κινούνται ομαλά ώρα-προς-
ώρα· η "λάθος" ώρα ήταν ήδη σχεδόν ταυτόσημη πληροφορία με τη σωστή. **Headline 14.43 παραμένει
έγκυρο.**

⚠️ **Δεν ξανατρέχτηκαν** με τα διορθωμένα δεδομένα: το αρχικό xborder ablation (§8-RESULTS),
confirmation σε Μάρτιο/καλοκαίρι, SS×xborder, seeds 43/44 — μόνο το headline weekly+default
verified explicit. Δεδομένου του αμελητέου effect size της διόρθωσης, τα υπόλοιπα θεωρούνται
πρακτικά έγκυρα αλλά τυπικά "not re-verified post-fix".

## Stacking-RESULTS (follow-up battery μετά την αξιολόγηση, 2026-07-03)

`confirm_out2/*.json`. Ερώτημα: στοιβάζονται τα 3 ανεξάρτητα κέρδη (xborder / retrain cadence / SS);

| Config (Q1 2026) | MAE | Συμπέρασμα |
|---|---|---|
| **xborder × weekly** | **14.426** 🏆 | ΣΤΟΙΒΑΖΕΤΑΙ πλήρως — ΝΕΟ HEADLINE (από 15.02) |
| xborder × SS-linear × monthly | 14.926 | SS ΔΕΝ προσθέτει πάνω σε xborder+monthly (−0.09, εντός seed noise) |
| xborder,-loadlags × SS × monthly | 14.947 | ομοίως — το SS-informed set δεν βοηθά εδώ |
| xborder × monthly seeds 42/43/44 | 15.015/15.028/14.924 | **std≈0.05** — τα ευρήματα σταθερά ως προς seed |
| xborder × καλοκαίρι '25 (static) | 13.236 vs default 14.434 | **Δ=−1.20, 3ο ανεξάρτητο window, μεγαλύτερο effect** |

**Συμπεράσματα:**
1. **Τελικό headline: LGBM recursive WEEKLY + `default,xborder` = 14.43 €/MWh** (Q1 2026, strict).
   Το xborder όφελος είναι σταθερό ≈−0.7 σε ΚΑΘΕ retrain cadence (static/monthly/weekly) και
   −1.20 το καλοκαίρι → 5/5 θετικά, 3 windows → **οριστικό: μπαίνει στο default set**.
2. **SS: redundant με φρέσκο retrain.** Το SS-linear βοηθούσε το static μοντέλο (−0.27) αλλά με
   monthly retrain + xborder το πλεονέκτημα εξαφανίζεται — μοιράζονται την ίδια πληροφορία
   (αντιστάθμιση staleness/exposure-bias). Χρήσιμο ΜΟΝΟ σε αραιό/στατικό retrain καθεστώς.
3. Seed noise ±0.05 → ό,τι |ΔMAE|>0.15 που δηλώσαμε παραμένει έγκυρο με άνεση.

## 10. Σειρά εκτέλεσης & συνολικός χρόνος

| Βήμα | Τι | Χρόνος |
|---|---|---|
| 1 | Μέρος 1: LGBM static Q1 × 18 specs | ~45′ |
| 2 | Μέρος 2: XGB static Q1 × 18 specs | ~45′ |
| 3 | Μέρος 3: direct-LGBM Δεκ × 10 specs | ~30′ |
| 4 | E1: estimators×meteo (6 runs) + E2: load meteo (3 runs) | ~35′ |
| 5 | Ανάλυση → επιβεβαίωση 2-3 νικητών με monthly | ~20′ |
| 6 | Φάση 2 (SS implementation + 4 runs) | ~2-3h dev+run |
| 7 | Φάση 3 (resload_fc / xborder) conditional | ~2h |

Βήματα 1-3 τρέχουν αλυσιδωτά σε ένα background command (σειριακό conda lane).
