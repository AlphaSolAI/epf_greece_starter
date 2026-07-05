---
name: ingest-audit
description: Υποχρεωτικό 3-βήμα pre-flight audit πριν μπει ΟΠΟΙΑΔΗΠΟΤΕ νέα πηγή δεδομένων ή νέο feature στο pipeline (gate timing, lagscan, hour-profile) + έλεγχος πληρότητας στηλών ανά task parquet. Χρησιμοποίησέ το όταν προτείνεται νέο feature/πηγή (ENTSO-E, HENEX, weather, xborder κ.λπ.), πριν από fetch/parquet rebuild, ή όταν ένα ablation δείχνει ύποπτα αποτελέσματα σε νέα ομάδα.
---

# Ingest Audit — καμία νέα πηγή χωρίς διαβατήριο

Ιστορικό που το επιβάλλει: το same-day xborder πέρασε ως «feature» ενώ ήταν leakage
(ίδιο SDAC auction, δημοσίευση ΜΕΤΑ το gate) — κόστισε μέρες. Το lagscan έπιασε 2
πραγματικά bugs (1h shift στο fetcher, same-day leak). Το 2026-07-05 βρέθηκε ότι το
`hourly_load.parquet` δεν είχε καν `load_fc` → ολόκληρο ablation arm ήταν VOID.

## Βήμα 1 — Γραπτή απάντηση: πότε ΑΚΡΙΒΩΣ δημοσιεύεται;

Πριν γραφτεί ΜΙΑ γραμμή κώδικα, γραπτά (στο chat ή στο ABLATION_PLAN §7):
- Ποιος τη δημοσιεύει, τι ώρα (με timezone!), για ποιο διάστημα αναφοράς;
- Σε σχέση με το gate **12:00 CET D-1**: διαθέσιμη ΠΡΙΝ ή ΜΕΤΑ; Αν «εξαρτάται» → ΜΕΤΑ.
- Προβλέψεις (fc) = day-ahead-known ομάδες· actuals = ΜΟΝΟ ως lags με ρητό reporting
  delay (crosslag family, AEL freeze).
- Ύποπτο: πηγή που βγαίνει από τον ΙΔΙΟ μηχανισμό/auction με το target (βλ. xborder).

## Βήμα 2 — Lagscan + hour-profile (τρέχει σε δευτερόλεπτα)

```bash
conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/lagscan.py --col <στήλη>
```
- Peak ΕΚΕΙ που προβλέπει η θεωρία του βήματος 1 — peak σε «βολικό» k ή |corr|>0.85 = ύποπτο.
- Hour-of-day profile λογικό (π.χ. solar peak ~12:00, όχι 14:00 → shift bug).
⚠️ Σεβάσου την ουρά: ΕΝΑ conda process — αν τρέχει batch, περίμενε.

## Βήμα 3 — Ένταξη ΜΟΝΟ μέσω feature_availability.py

- Νέα ομάδα (ή ένταξη σε υπάρχουσα) στο `src/feature_availability.py` με ρητό
  availability rule — ΠΟΤΕ κατευθείαν στο engine.
- Το `feature_availability.py` είναι leakage-sensitive πυρήνας → μετά την αλλαγή:
  `preflight_check.py --poison` + control run + reproducibility anchor ±0.05.

## Βήμα 4 — Έλεγχος πληρότητας ανά task (μάθημα loadfc 2026-07-05)

Η στήλη πρέπει να υπάρχει στο parquet ΚΑΘΕ task που θα τη χρησιμοποιήσει:
- task=price → `data/processed/hourly.parquet` · task=load → `hourly_load.parquet`
  (επιλογή αρχείου: `split_utils._pick_processed_path`).
- Μετά το rebuild, γρήγορο check (system python): διάβασε τα columns και των δύο parquet
  και βεβαιώσου ότι η νέα στήλη είναι εκεί που πρέπει.
- Σε πρώτο ablation με τη νέα ομάδα: βεβαιώσου ότι το `#features` στο log ΑΛΛΑΖΕΙ
  μεταξύ baseline και ablated spec (αλλιώς η ομάδα είναι κενή = VOID arm).

## Θυμήσου

- `data/raw/`, `data/processed/` προστατευμένα — αλλαγές ΜΟΝΟ μέσω scripts με
  backup+σύγκριση πριν σβηστεί το παλιό.
- Rebuild: `python -m src.data --task price` και `--task load` (conda), μετά preflight.
- Κατέγραψε το αποτέλεσμα του audit στο ABLATION_PLAN §7 (PENDING) μέχρι το πρώτο τεστ.
