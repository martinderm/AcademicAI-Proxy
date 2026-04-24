# AcademicAI Proxy - lokaler Teststart mit isolierten Defaults.

function Import-EnvFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
            return
        }
        $parts = $_ -split '=', 2
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
        }
    }
}

function Import-BackendCredsFromEnvFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }

    $allowed = @{
        "ACADEMICAI_BASE_URL"      = $true
        "ACADEMICAI_CLIENT_ID"     = $true
        "ACADEMICAI_CLIENT_SECRET" = $true
    }

    Get-Content $Path | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
            return
        }
        $parts = $_ -split '=', 2
        if ($parts.Count -ne 2) {
            return
        }
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (-not $allowed.ContainsKey($key)) {
            return
        }
        if (-not [System.Environment]::GetEnvironmentVariable($key, 'Process')) {
            [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
}

$envFile = "$PSScriptRoot\.env.localtest"
if (-not (Test-Path $envFile)) {
    Write-Host ".env.localtest fehlt. Bitte .env.localtest.example kopieren und anpassen."
    exit 1
}

Import-EnvFile -Path $envFile

$backendMissing = (-not $env:ACADEMICAI_BASE_URL) -or (-not $env:ACADEMICAI_CLIENT_ID) -or (-not $env:ACADEMICAI_CLIENT_SECRET)
if ($backendMissing) {
    Import-BackendCredsFromEnvFile -Path "$PSScriptRoot\.env"
}

& "$PSScriptRoot\start_server.ps1"
