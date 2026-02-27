# AcademicAI Proxy Server - Start Script
# Startet den Server, falls er nicht laeuft

$port = 11435
$pidFile = "$PSScriptRoot\server.pid"
$logFile = "$PSScriptRoot\server.log"
$serverScript = "$PSScriptRoot\server.py"

# Pruefen ob Port bereits belegt
$inUse = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    Write-Host "AcademicAI Proxy bereits aktiv auf Port $port"
    exit 0
}

# Server starten
Write-Host "Starte AcademicAI Proxy auf Port $port..."
$proc = Start-Process -FilePath "py" `
    -ArgumentList $serverScript `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$PSScriptRoot\server.err.log" `
    -PassThru `
    -WindowStyle Hidden

$proc.Id | Out-File $pidFile -Encoding UTF8
Write-Host "AcademicAI Proxy gestartet (PID: $($proc.Id))"
