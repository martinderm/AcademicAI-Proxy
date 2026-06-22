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
$envPrefix = ""
Get-ChildItem env: | Where-Object { $_.Name -like 'ACADEMICAI_*' -or $_.Name -eq 'TENANT_ID' } | ForEach-Object {
    $val = $_.Value -replace "'", "''"
    $envPrefix += "`$env:$($_.Name) = '$val'; "
}
$cmd = "powershell.exe -NoProfile -Command `"$envPrefix & '$pythonExe' -u '$serverScript' > '$logFile' 2> '$errFile'`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $PSScriptRoot }

if ($result.ReturnValue -eq 0) {
    # Wait up to 5 seconds for the server to start listening and bind to the port
    $realPid = $null
    for ($i = 0; $i -lt 10; $i++) {
        Start-Sleep -Milliseconds 500
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn) {
            $realPid = $conn.OwningProcess
            break
        }
    }
    if (-not $realPid) {
        $realPid = $result.ProcessId
    }
    $realPid | Out-File $pidFile -Encoding UTF8
    Write-Host "AcademicAI Proxy gestartet (PID: $realPid)"
} else {
    Write-Error "Fehler beim Starten des AcademicAI Proxy (WMI ReturnValue: $($result.ReturnValue))"
    exit 1
}
