# Detached launcher — direct LGBM octnov g12 (1 stage). ASCII-only ($PSScriptRoot).
# OS-detached: επιβιωνει αν κλεισει το Claude desktop app ΚΑΙ το VS Code.
$repo = Split-Path $PSScriptRoot -Parent
$bash = "C:\Program Files\Git\bin\bash.exe"
$p = Start-Process -FilePath $bash `
     -WorkingDirectory $repo `
     -ArgumentList '-c','"bash scripts/load_direct_lgbm_octnov.sh"' `
     -RedirectStandardOutput (Join-Path $repo 'logs\load_direct_lgbm_octnov.log') `
     -RedirectStandardError  (Join-Path $repo 'logs\load_direct_lgbm_octnov.err.log') `
     -WindowStyle Hidden -PassThru
Write-Output "PID=$($p.Id)"
