# Detached overnight chain (2026-07-11): octnov seed hardening -> SS x weekly probe.
# Small idempotent pieces (SKIP existing), sequential -> ONE conda. Marker on completion.
# Embedded double quotes REQUIRED (PS 5.1 Start-Process joins args unquoted — else silent no-op).
$repo = Split-Path -Parent $PSScriptRoot
$p = Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' `
  -ArgumentList '-c','"bash scripts/load_contest_octnov_seeds.sh > logs/load_contest_octnov_seeds.log 2>&1; bash scripts/load_contest_ss_lgbm_weekly.sh > logs/load_contest_ss_lgbm_weekly.log 2>&1; echo DONE > logs/overnight_20260711.done"' `
  -WorkingDirectory $repo -PassThru
Write-Host ("launched pid=" + $p.Id)
