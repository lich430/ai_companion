$ErrorActionPreference = "Stop"

$Port = 8080

Write-Host "Stopping processes listening on port $Port ..."

$listeningPids = @()
$lines = netstat -ano | Select-String ":$Port"
foreach ($line in $lines) {
    $parts = ($line -split "\s+") | Where-Object { $_ -ne "" }
    if ($parts.Length -ge 5 -and $parts[3] -eq "LISTENING") {
        $targetPid = $parts[-1]
        if ($targetPid -match "^\d+$" -and $targetPid -ne "0") {
            $listeningPids += [int]$targetPid
        }
    }
}

$listeningPids = $listeningPids | Sort-Object -Unique

if (-not $listeningPids) {
    Write-Host "No LISTENING process found on port $Port."
    exit 0
}

foreach ($targetPid in $listeningPids) {
    try {
        $proc = Get-Process -Id $targetPid -ErrorAction Stop
        Write-Host "Stopping PID=$targetPid Name=$($proc.ProcessName)"
        Stop-Process -Id $targetPid -Force
    } catch {
        Write-Host "Skip PID=$targetPid (already exited or inaccessible)"
    }
}

Write-Host "Done."
