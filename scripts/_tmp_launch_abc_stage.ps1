# Detached: LOAD ABC stage (A weekly seas confirm x4, B direct probe x1). Waits for queue.
# ASCII-only file (PS 5.1); embedded double quotes REQUIRED in multi-word -ArgumentList item.
$repo = Split-Path -Parent $PSScriptRoot
$p = Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' `
  -ArgumentList '-c','"bash scripts/load_abc_stage.sh > logs/load_abc_stage.log 2>&1"' `
  -WorkingDirectory $repo -PassThru
Write-Host ("launched pid=" + $p.Id)
