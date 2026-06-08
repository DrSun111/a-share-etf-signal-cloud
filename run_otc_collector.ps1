$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\python -m pip install -r requirements.txt

$interval = 120
Write-Host "OTC watchlist collector started. Refresh interval: $interval seconds."
while ($true) {
    $started = Get-Date
    Write-Host "[$($started.ToString('yyyy-MM-dd HH:mm:ss'))] collecting OTC watch snapshot..."
    .\.venv\Scripts\python otc_collector.py
    Start-Sleep -Seconds $interval
}
