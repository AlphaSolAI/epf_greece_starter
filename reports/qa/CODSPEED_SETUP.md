# CodSpeed — φάση 2 ✅ ΕΝΕΡΓΗ (2026-07-08)

**Κατάσταση 2026-07-08:** ✅ **ΔΟΥΛΕΥΕΙ end-to-end.** CodSpeed GitHub App εγκατεστημένο,
workflow `.github/workflows/codspeed.yml` active, πρώτο run πέρασε (commit a4eba37):

| Benchmark | Πρώτη μέτρηση |
|---|---|
| `parse_feature_spec` | 65.8 µs |
| `detect_crosslag_cols` | 247.2 µs |
| `classify_columns` | 1.4 ms |

Public repo → **δεν χρειάστηκε `CODSPEED_TOKEN`** (δουλεύει μέσω του GitHub App). Κάθε
push/PR στο FEB272026 πλέον τρέχει τα benchmarks και το CodSpeed flag-άρει regressions.

**(Ιστορικό ενεργοποίησης — έγινε):** install GitHub App μέσω codspeed.io → activate workflow
(rename από `.template`) → push σκανδάλισε το πρώτο run. Τοπική δοκιμή (προαιρετικό):
`pip install pytest-codspeed` στο env `epf` + `python -X utf8 -m pytest tests/benchmarks/ --codspeed`.

## Τι μετρᾶται (pure-logic, χωρίς training data — τρέχει σε CI)

`tests/benchmarks/test_bench_leakage_logic.py`:
- `parse_feature_spec` — feature-spec parsing (το `--features` path).
- `classify_columns` — ταξινόμηση ~200 στηλών σε feature groups.
- `detect_crosslag_cols` — ανίχνευση των 4 crosslag οικογενειών (leakage-critical).

Το macro timing (πλήρη training runs) ΔΕΝ πάει στο CodSpeed — μένει local
(`optimizing-training-runs` skill §3, `reports/qa/PROFILE_BASELINE.md`). Το CI δεν έχει
δεδομένα/ώρες για training· μόνο τα pure-logic hot paths.

## Μετά την ενεργοποίηση

Ανάλυση cloud runs από εδώ: CodSpeed MCP tools (`list_runs`, `get_run`, `query_flamegraph`,
`compare_runs`). Κάθε regression στα pure-logic paths εμφανίζεται στο PR ως CodSpeed check.
