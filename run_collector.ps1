$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python collector.py
