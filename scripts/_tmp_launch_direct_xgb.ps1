# Detached launcher — direct XGB queue (3 windows). ASCII-only ($PSScriptRoot).
$repo = Split-Path $PSScriptRoot -Parent
$bash = "C:\Program Files\Git\bin\bash.exe"
$p = Start-Process -FilePath $bash `
     -WorkingDirectory $repo `
     -ArgumentList '-c','"bash scripts/load_direct_xgb_queue.sh"' `
     -RedirectStandardOutput (Join-Path $repo 'logs\load_direct_xgb_queue.log') `
     -RedirectStandardError  (Join-Path $repo 'logs\load_direct_xgb_queue.err.log') `
     -WindowStyle Hidden -PassThru
Write-Output "PID=$($p.Id)"
