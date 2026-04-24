import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from _local_env import BASE, MODEL, auth_headers


def _backend_env_available() -> bool:
    return bool(
        os.environ.get("ACADEMICAI_BASE_URL")
        and os.environ.get("ACADEMICAI_CLIENT_ID")
        and os.environ.get("ACADEMICAI_CLIENT_SECRET")
    )


def _health_url() -> str:
    return f"{BASE}/health"


def _is_server_up(timeout: float = 1.0) -> bool:
    try:
        response = httpx.get(_health_url(), timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _shell_executable() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "pwsh"


def _run_script(script_path: Path) -> int:
    proc = subprocess.run(
        [_shell_executable(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=str(script_path.parent),
        check=False,
    )
    return proc.returncode


@pytest.fixture(scope="session")
def e2e_server() -> dict:
    if not _backend_env_available():
        pytest.skip("E2E requires ACADEMICAI_BASE_URL, ACADEMICAI_CLIENT_ID and ACADEMICAI_CLIENT_SECRET")

    repo_root = Path(__file__).resolve().parents[1]
    start_script = repo_root / "start_test_server.ps1"
    stop_script = repo_root / "stop_server.ps1"

    started_here = False
    if not _is_server_up():
        rc = _run_script(start_script)
        if rc != 0:
            pytest.fail("Could not start local proxy for E2E tests")
        started_here = True

        deadline = time.time() + 20
        while time.time() < deadline:
            if _is_server_up(timeout=2.0):
                break
            time.sleep(0.5)
        else:
            pytest.fail("Local proxy did not become healthy in time")

    try:
        yield {"base": BASE, "headers": auth_headers(), "model": MODEL}
    finally:
        if started_here:
            _run_script(stop_script)
