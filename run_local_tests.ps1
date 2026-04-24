param(
    [ValidateSet('offline', 'e2e', 'all')]
    [string]$Mode = 'offline'
)

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

$python = "c:/Users/dagobert-ai/.openclaw/.venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    $python = "py"
}

$envFile = "$PSScriptRoot\.env.localtest"
Import-EnvFile -Path $envFile

# Backend-Credentials optional aus .env nachladen, wenn sie lokaltest-seitig fehlen.
$backendMissing = (-not $env:ACADEMICAI_BASE_URL) -or (-not $env:ACADEMICAI_CLIENT_ID) -or (-not $env:ACADEMICAI_CLIENT_SECRET)
if ($backendMissing) {
    Import-EnvFile -Path "$PSScriptRoot\.env"
}

$offlineTests = @(
    'tests/test_post_tool_guard.py',
    'tests/test_hardening_security_runtime.py',
    'tests/test_humanization_flow.py'
)



if ($Mode -in @('offline', 'all')) {
    & $python -m pytest -q @offlineTests
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($Mode -in @('e2e', 'all')) {
    if (-not $env:ACADEMICAI_BASE_URL -or -not $env:ACADEMICAI_CLIENT_ID -or -not $env:ACADEMICAI_CLIENT_SECRET) {
        Write-Host "E2E-Modus braucht ACADEMICAI_BASE_URL, ACADEMICAI_CLIENT_ID und ACADEMICAI_CLIENT_SECRET in .env.localtest oder .env."
        exit 1
    }
    & "$PSScriptRoot\start_test_server.ps1"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    try {
        & $python -m pytest -q -m e2e
        exit $LASTEXITCODE
    }
    finally {
        & "$PSScriptRoot\stop_server.ps1" | Out-Null
    }
}
