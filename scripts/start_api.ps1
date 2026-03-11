$ErrorActionPreference = "Stop"

$Port = 8080
$Root = Split-Path -Parent $PSScriptRoot
$Python = "python"

Write-Host "Cleaning old listeners on port $Port ..."
& "$PSScriptRoot\stop_api.ps1"

Write-Host "Starting API on port $Port ..."
Set-Location $Root
& $Python -m uvicorn api:app --host 0.0.0.0 --port $Port --reload
