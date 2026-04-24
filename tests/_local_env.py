import os


BASE = os.environ.get("ACADEMICAI_TEST_BASE_URL", "http://127.0.0.1:11435").rstrip("/")
API_KEY = os.environ.get("ACADEMICAI_TEST_PROXY_API_KEY", "test-proxy-key-123456")
MODEL = os.environ.get("ACADEMICAI_TEST_MODEL", "gpt-5-mini")


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
