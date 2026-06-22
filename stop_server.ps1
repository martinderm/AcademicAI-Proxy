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

# Always verify if the port is still listening, in case of orphaned child processes
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) {
    $owningPid = $conn.OwningProcess
    Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
    Write-Host "Orphaned AcademicAI Proxy Prozess ueber Port $port gestoppt (PID: $owningPid)"
    $stopped = $true
}

if (-not $stopped) {
    Write-Host "Kein laufender AcademicAI Proxy gefunden."
}
