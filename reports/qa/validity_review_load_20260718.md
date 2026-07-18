# VALIDITY REVIEW — LOAD (STLF) · 5 ευρήματα · 2026-07-18

> Subagent `validity-reviewer`, session 2026-07-18 (goal mode). Κοινό πλαίσιο όλων των
> συγκρίσεων επαληθευμένο ΜΕΣΑ στα JSONs (task=load, dam, strict, crosslag=freeze,
> H24/s24, ζεύγη ίδιο window+gap+strategy+retrain — asserted προγραμματιστικά).
> ΑΔΜΗΕ 146.8095 octnov επαληθευμένο live από scripts/load_contest_benchmarks.py
> (persisted πλέον: results/load_contest_benchmarks.csv). Α6: τα base/dense/loadfc/genlags
> arms δεν περιέχουν meteo → καθαρά. Dense y-path poisoning Test D PASS (last.md §2 Α1).
> **Οριστικοποίηση ΔΕΚΤΟ στο ABLATION_PLAN = απόφαση χρήστη** — εδώ είναι τα verdicts
> του reviewer.

## Ε1 — dense (y_lag4..23) βοηθάει load, recursive weekly, cross-algo cross-gate
**VERDICT: ACCEPT** (με 2 σημειώσεις διατύπωσης)

ΔMAE = dense − base (MW):

| algo | q1 g12 | q1 g14 | summer g12 | summer g14 | octnov g12 | octnov g14 |
|---|---|---|---|---|---|---|
| LGBM | −8.52 | −8.47 | −15.38 | −17.61 | −10.67 | −11.18 |
| XGB | −10.40 | −5.56 | −1.71 | −12.71 | −8.57 | −7.16 |
| MLP | **+0.74** | −17.64 | −7.25 | −53.74 | −6.80 | −1.95 |
| LEAR | −53.63 | −50.28 | −17.62 | −14.15 | −53.92 | −53.74 |

23/24 κελιά αρνητικά, 3 ανεξάρτητα windows. Seeds: LGBM 12/12 κελιά-seed-σετ dense<base
(std 2.3–4.3)· XGB **18/18**· MLP q1-g12 flip = seed noise (3 seeds/arm: base mean 264.5
vs dense mean 256.5, Δmean −8.0)· LEAR seed αρχεία **bit-identical** → seed-invariant
by construction, **verified** εμπειρικά.
**FIX διατύπωσης**: (α) MLP = «5/6 κελιά, q1-g12 flip = seed noise», ΟΧΙ «24/24»·
(β) LEAR = «seed-invariant by construction — verified identical outputs».

## Ε2 — LSTM (recursive static): calendar+loadfc ≪ calendar+lags+roll
**VERDICT: ACCEPT** ΜΟΝΟ με 3 scoping flags:
1. **static cadence only** (weekly αδοκίμαστο — δεν γενικεύεται).
2. **non-tradeable LSTM eval pipeline** (eval encoder δεν φιλτράρεται — τα απόλυτα MAE
   ΔΕΝ μπαίνουν σε leaderboard/vs-ΑΔΜΗΕ δίπλα σε tradeable runs· το feature-effect
   είναι internally fair γιατί και τα 2 arms τρέχουν στο ίδιο pipeline).
3. **anchored σε q1+summer** (Δ −47.6/−50.6 και −166.0/−104.8)· octnov (−5.5/−2.6)
   συνεπές πρόσημο αλλά εντός LSTM seed-noise (~10 MW).
Επίσης: το arm είναι `calendar,lags,loadfc` vs `calendar,lags,roll` — arm-vs-arm,
όχι «+loadfc μόνο του». Το loadfc = δημοσιευμένο ΑΔΜΗΕ D-1 forecast, Α6-καθαρό.

## Ε3 — direct LGBM weekly g12: dense βοηθάει 3/3, genlags βλάπτει 3/3
**VERDICT: ACCEPT** scoped «direct LGBM weekly g12»

| window | base | dense (Δ) | genlags (Δ) |
|---|---|---|---|
| q1 | 269.61 | 258.88 (−10.73) | 280.97 (+11.36) |
| summer | 315.70 | 299.64 (−16.06) | 332.72 (+17.01) |
| octnov | 162.86 | 144.86 (−18.00) | 193.64 (+30.78) |

⚠️ Το genlags-hurts είναι **direct-specific**: στο recursive έχει ΑΝΤΙΘΕΤΟ πρόσημο
σε q1/octnov (−6.2/−10.2, summer +17.8) — δεν γενικεύεται.

## Ε4 — «recursive > direct για load (weekly)» ως default
**VERDICT: PENDING** ως γενικός κανόνας (τεκμηριωμένη αναστροφή προσήμου στο summer:
dir κερδίζει +26..+44 σε 4/5 specs). Γράφεται ΔΕΚΤΟ ΜΟΝΟ σε μία από τις μορφές:
1. **Interaction finding**: «rec κερδίζει q1+octnov (9/10 σημαντικά κελιά)· dir κερδίζει
   summer (4/5)» — αυτό περνάει §2 ως interaction.
2. **loadfc-scoped**: «με loadfc στο spec, recursive > direct **3/3 windows**
   (−81.8/−84.1/−23.8)» — μόνο αυτή δικαιολογεί «recursive default», μόνο για
   config με loadfc.

D = rec − dir (g12): base −13.6/+42.9/−9.2 · dense −11.4/+43.6/−1.9 ·
genlags −31.1/+43.7/−50.2 · noroll −25.6/+26.0/−16.0 · loadfc **−81.8/−84.1/−23.8**
(σειρά q1/summer/octnov).

## Ε5 — XGB recursive weekly dense (no-meteo) < ΑΔΜΗΕ στο octnov
**VERDICT: ACCEPT** ως **window-specific (octnov 2025 ΜΟΝΟ)** — ΟΧΙ headline.
- ΑΔΜΗΕ octnov 146.8095 (n=1463, 1h NaN αμελητέο) — persisted CSV.
- XGB dense: g12 137.46/139.55/138.76 · g14 143.81/142.72/142.92 — **6/6 < 146.81**,
  min περιθώριο 2.99 MW ≫ XGB seed std (0.6–1.1).
- Α6 ΟΚ (κανένα meteo). Υποχρεωτική δίπλα η φράση: **«ΑΔΜΗΕ κερδίζει q1 (175.6 vs
  242-245) και summer (170.8 vs 335-357)»**. Headline εκδοχή ΑΠΟΡΡΙΠΤΕΤΑΙ (το «2ο
  winning window» δεν λείπει απλώς — αντικρούεται).

## Σύνοψη
| # | Εύρημα | Verdict |
|---|---|---|
| Ε1 | dense cross-algo cross-gate | ACCEPT (+2 διατυπώσεις) |
| Ε2 | LSTM calendar+loadfc | ACCEPT (+3 flags) |
| Ε3 | direct dense/genlags | ACCEPT (scoped direct-LGBM-weekly-g12) |
| Ε4 | recursive default για load | PENDING (interaction ή loadfc-scoped μόνο) |
| Ε5 | XGB < ΑΔΜΗΕ octnov | ACCEPT (window-specific, όχι headline) |
