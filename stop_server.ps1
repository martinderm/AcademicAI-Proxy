# AcademicAI Proxy Server - Stop Script
# Stoppt den Server kontrolliert und bereinigt die PID-Datei.

$port = 11435
$pidFile = "$PSScriptRoot\server.pid"

$stopped = $false
if (Test-Path $pidFile) {
    $rawPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($rawPid -match '^\d+$') {
        $proc = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Host "AcademicAI Proxy gestoppt (PID: $rawPid)"
            $stopped = $true
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "AcademicAI Proxy Prozess ueber Port $port gestoppt (PID: $($conn.OwningProcess))"
        $stopped = $true
    }
}

if (-not $stopped) {
    Write-Host "Kein laufender AcademicAI Proxy gefunden."
}
