param(
    [ValidateSet('offline', 'e2e', 'all')]
    [string]$Mode = 'offline'
)

$python = "c:/Users/dagobert-ai/.openclaw/.venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    $python = "py"
}

$envFile = "$PSScriptRoot\.env.localtest"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
            return
        }
        $parts = $_ -split '=', 2
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
        }
    }
}

$offlineTests = @(
    'tests/test_post_tool_guard.py',
    'tests/test_hardening_security_runtime.py',
    'tests/test_humanization_flow.py'
)

$e2eTests = @(
    'tests/test_tool_emulation.py',
    'tests/test_strategy_a.py',
    'tests/test_memory_tools.py',
    'tests/test_models.py',
    'tests/test_openclaw_style.py'
)

if ($Mode -in @('offline', 'all')) {
    & $python -m pytest -q @offlineTests
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($Mode -in @('e2e', 'all')) {
    if (-not $env:ACADEMICAI_BASE_URL -or -not $env:ACADEMICAI_CLIENT_ID -or -not $env:ACADEMICAI_CLIENT_SECRET) {
        Write-Host "E2E-Modus braucht ACADEMICAI_BASE_URL, ACADEMICAI_CLIENT_ID und ACADEMICAI_CLIENT_SECRET in .env.localtest."
        exit 1
    }
    & "$PSScriptRoot\start_test_server.ps1"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    try {
        & $python @e2eTests
        exit $LASTEXITCODE
    }
    finally {
        & "$PSScriptRoot\stop_server.ps1" | Out-Null
    }
}
