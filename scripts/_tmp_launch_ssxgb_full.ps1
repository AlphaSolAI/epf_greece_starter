# Detached: SS x weekly XGB full grid (6 new runs, 2 SKIP). ONE conda. Marker on done.
# Embedded double quotes REQUIRED (PS 5.1 Start-Process joins args unquoted).
$repo = Split-Path -Parent $PSScriptRoot
$p = Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' `
  -ArgumentList '-c','"bash scripts/load_contest_ss_xgb_weekly.sh > logs/load_contest_ss_xgb_weekly.log 2>&1; echo DONE > logs/ssxgbfull.done"' `
  -WorkingDirectory $repo -PassThru
Write-Host ("launched pid=" + $p.Id)
