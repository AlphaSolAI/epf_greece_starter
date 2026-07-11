# Detached: SS x weekly XGB (dense) confirm. 2 idempotent runs, ONE conda. Marker on done.
# Embedded double quotes REQUIRED (PS 5.1 Start-Process joins args unquoted).
$repo = Split-Path -Parent $PSScriptRoot
$p = Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' `
  -ArgumentList '-c','"bash scripts/load_contest_ss_xgb_dense.sh > logs/load_contest_ss_xgb_dense.log 2>&1; echo DONE > logs/ssxgb.done"' `
  -WorkingDirectory $repo -PassThru
Write-Host ("launched pid=" + $p.Id)
