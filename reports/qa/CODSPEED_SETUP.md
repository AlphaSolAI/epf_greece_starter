# CodSpeed — ενεργοποίηση φάσης 2 (βήματα χρήστη)

**Κατάσταση 2026-07-08:** connector συνδεδεμένος (MCP `list_repositories` απαντά), αλλά το
repo `AlphaSolAI/epf_greece_starter` **δεν είναι ακόμα CodSpeed-enabled** (κενή λίστα) και
το `pytest-codspeed` δεν είναι στο conda env. Τα benchmarks + το CI template είναι έτοιμα
— μένουν 4 βήματα που απαιτούν εσένα (GitHub/CodSpeed side, δεν γίνονται από headless session):

1. **Install CodSpeed GitHub App** στο `AlphaSolAI/epf_greece_starter`:
   https://codspeed.io → Sign in with GitHub → Add repository → epf_greece_starter.
2. **Πρόσθεσε secret** `CODSPEED_TOKEN` (το δίνει το CodSpeed dashboard μετά το install):
   GitHub repo → Settings → Secrets and variables → Actions → New repository secret.
3. **Ενεργοποίησε το workflow**: rename
   `.github/workflows/codspeed.yml.template` → `.github/workflows/codspeed.yml`, commit+push.
4. (τοπικά, προαιρετικό για δοκιμή πριν το CI) `pip install pytest-codspeed` στο env `epf`
   και τρέξε: `conda run -n epf --no-capture-output python -X utf8 -m pytest tests/benchmarks/ --codspeed`.

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
