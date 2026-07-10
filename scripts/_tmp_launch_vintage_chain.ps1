# Detached launcher: G7 chain = vintage LGBM batch (12 runs) -> XGB Batch 2 (30 runs).
# One bash process, sequential -> ONE conda at a time. Marker file on completion.
# NOTE (2026-07-10): multi-word ArgumentList items MUST carry embedded double quotes —
# PS 5.1 Start-Process joins args unquoted, so bash would receive only the first word
# after -c (silent no-op launch). Single-token lists (netload launcher) never hit this.
$repo = Split-Path -Parent $PSScriptRoot
$p = Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' `
  -ArgumentList '-c','"bash scripts/load_contest_vintage_lgbm.sh > logs/load_contest_vintage_lgbm.log 2>&1; bash scripts/load_contest_xgb_weekly.sh > logs/load_contest_xgb_weekly.log 2>&1; echo CHAIN_DONE > logs/vintage_chain.done"' `
  -WorkingDirectory $repo -PassThru
Write-Host ("launched pid=" + $p.Id)
