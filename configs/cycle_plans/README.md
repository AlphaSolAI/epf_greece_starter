# configs/cycle_plans — plan.yaml για τον cycle_runner

Κάθε αρχείο = μία σύγκριση. Ο runner (`src/cycle_runner.py`) τρέχει preflight →
forecast batch → §2 πίνακα → ledger → draft deposit. ΔΕΝ αγγίζει δεδομένα, ΔΕΝ
αποφασίζει αποδοχή. Spec: `docs/superpowers/specs/2026-07-07-cycle-runner-design.md`.

## Κανόνες ονομάτων
- window `name`, algos, strategy: **ΧΩΡΙΣ underscore** (`q12026`, όχι `q1_2026`) —
  ο synthesize κάνει `split('_')`.
- specs: χώρισε ομάδες με `,` (`lags,calendar`), ποτέ `_`.

## Dry-run πρώτα (χωρίς training)

    conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan <plan> --dry-run

## Πλήρες run (detached, μεγάλα batch)

    Start-Process -WindowStyle Hidden -FilePath "C:\Program Files\Git\bin\bash.exe" `
      -WorkingDirectory "<repo>" -ArgumentList '-c', `
      'conda run -n epf --no-capture-output python -X utf8 -m src.cycle_runner --plan <plan> > logs/<study>_boot.log 2>&1'
    # παρακολούθηση: Get-Content logs\<study>.log -Wait -Tail 20

## Στάδια που καλύπτει (3→7 του cycle)
preflight (+poison) → forecast batch (master_forecast ανά cell, σειριακά, ένα conda
process, per-cell fail-safe) → synthesize_ablation (ΔMAE + §2 pre-gate) → build_run_ledger
→ `runs/<study>/DRAFT_deposit.md`. ΣΤΑΜΑΤΑ εκεί: pre-gate ≠ ΔΕΚΤΟ → validity-reviewer + άνθρωπος.
