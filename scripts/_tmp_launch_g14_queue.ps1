# Detached launcher — G14+DIRECT queue (4 stages). ASCII-only ($PSScriptRoot).
# OS-detached: επιβιωνει αν κλεισει το Claude desktop app ΚΑΙ το VS Code.
$repo = Split-Path $PSScriptRoot -Parent
$bash = "C:\Program Files\Git\bin\bash.exe"
$p = Start-Process -FilePath $bash `
     -WorkingDirectory $repo `
     -ArgumentList '-c','"bash scripts/load_g14_direct_queue.sh"' `
     -RedirectStandardOutput (Join-Path $repo 'logs\load_g14_direct_queue.log') `
     -RedirectStandardError  (Join-Path $repo 'logs\load_g14_direct_queue.err.log') `
     -WindowStyle Hidden -PassThru
Write-Output "PID=$($p.Id)"
