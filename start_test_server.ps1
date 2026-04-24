# AcademicAI Proxy - lokaler Teststart mit isolierten Defaults.

$envFile = "$PSScriptRoot\.env.localtest"
if (-not (Test-Path $envFile)) {
    Write-Host ".env.localtest fehlt. Bitte .env.localtest.example kopieren und anpassen."
    exit 1
}

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
        return
    }
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
        [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
    }
}

& "$PSScriptRoot\start_server.ps1"
