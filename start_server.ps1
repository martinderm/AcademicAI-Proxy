# AcademicAI Proxy Server - Start Script
# Startet den Server robust und bereinigt stale PID-Dateien.

$port = 11435
$pidFile = "$PSScriptRoot\server.pid"
$logFile = "$PSScriptRoot\server.log"
$errFile = "$PSScriptRoot\server.err.log"
$serverScript = "$PSScriptRoot\server.py"

if (Test-Path $pidFile) {
    $rawPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($rawPid -match '^\d+$') {
        $existing = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
        if ($existing) {
            Write-Host "AcademicAI Proxy laeuft bereits (PID: $rawPid)"
            exit 0
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    Write-Host "Stale PID-Datei bereinigt."
}

$inUse = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    Write-Host "Port $port ist bereits belegt. Start abgebrochen."
    exit 1
}

$pythonExe = "$PSScriptRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "py"
}

Write-Host "Starte AcademicAI Proxy auf Port $port..."
$proc = Start-Process -FilePath $pythonExe `
    -ArgumentList $serverScript `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -PassThru `
    -WindowStyle Hidden

$proc.Id | Out-File $pidFile -Encoding UTF8
Write-Host "AcademicAI Proxy gestartet (PID: $($proc.Id))"
