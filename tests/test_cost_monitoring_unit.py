"""
Unit tests for academicai.cost_monitoring.

Tests:
- _safe_float and _parse_iso_ts parsing
- _extract_cost_summary with valid and malformed payloads
- is_cost_cache_stale with fresh and expired timestamps
- build_cost_headers formatting and disable-toggle behavior
- Cache read and write roundtrip with file locking/safety (using tmp_path)
- Concurrency and backward compatibility re-exports on server.py
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import server
from academicai.cost_monitoring import (
    _cost_lock,
    _extract_cost_summary,
    _is_cost_cache_stale,
    _parse_iso_ts,
    _read_cost_cache,
    _safe_float,
    _write_cost_cache,
    build_cost_headers,
    get_cost_status_payload,
    is_cost_cache_stale,
    read_cost_cache,
    write_cost_cache,
)


# ---------------------------------------------------------------------------
# 1. Parsing: _safe_float & _parse_iso_ts
# ---------------------------------------------------------------------------


def test_safe_float():
    assert _safe_float(12.34) == 12.34
    assert _safe_float(10) == 10.0
    assert _safe_float("42.5") == 42.5
    assert _safe_float("100") == 100.0
    assert _safe_float("0") == 0.0
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float("invalid") is None
    assert _safe_float({}) is None
    assert _safe_float([]) is None


def test_parse_iso_ts():
    # UTC with Z
    dt_z = _parse_iso_ts("2026-09-06T12:00:00Z")
    assert dt_z is not None
    assert dt_z.tzinfo is not None
    assert dt_z.year == 2026
    assert dt_z.month == 9
    assert dt_z.day == 6
    assert dt_z.hour == 12

    # With offset
    dt_offset = _parse_iso_ts("2026-09-06T14:00:00+02:00")
    assert dt_offset is not None
    assert dt_offset.tzinfo is not None
    assert dt_offset.hour == 14

    # Naive timestamp
    dt_naive = _parse_iso_ts("2026-09-06T12:00:00")
    assert dt_naive is not None
    assert dt_naive.hour == 12

    # Malformed / empty / invalid types
    assert _parse_iso_ts(None) is None
    assert _parse_iso_ts("") is None
    assert _parse_iso_ts("not-a-timestamp") is None
    assert _parse_iso_ts(12345) is None
    assert _parse_iso_ts([]) is None


# ---------------------------------------------------------------------------
# 2. _extract_cost_summary
# ---------------------------------------------------------------------------


def test_extract_cost_summary_valid_nested():
    payload = {
        "data": {
            "totalCost": "123.456",
            "totalClients": "7",
            "costs": [{"id": 1}, {"id": 2}, {"id": 3}],
        }
    }
    summary = _extract_cost_summary(payload)
    assert summary == {
        "total_cost": 123.456,
        "total_clients": 7,
        "cost_entries": 3,
    }


def test_extract_cost_summary_valid_flat():
    payload = {
        "totalCost": 99.0,
        "totalClients": 2,
        "costs": [{"id": 1}],
    }
    summary = _extract_cost_summary(payload)
    assert summary == {
        "total_cost": 99.0,
        "total_clients": 2,
        "cost_entries": 1,
    }


def test_extract_cost_summary_malformed_payloads():
    # Non-dict inputs
    assert _extract_cost_summary(None) == {
        "total_cost": None,
        "total_clients": None,
        "cost_entries": 0,
    }
    assert _extract_cost_summary("unexpected string") == {
        "total_cost": None,
        "total_clients": None,
        "cost_entries": 0,
    }
    assert _extract_cost_summary([]) == {
        "total_cost": None,
        "total_clients": None,
        "cost_entries": 0,
    }

    # Dict with invalid types for fields
    malformed_fields = {
        "totalCost": "not_a_number",
        "totalClients": "abc",
        "costs": "not_a_list",
    }
    assert _extract_cost_summary(malformed_fields) == {
        "total_cost": None,
        "total_clients": None,
        "cost_entries": 0,
    }


# ---------------------------------------------------------------------------
# 3. is_cost_cache_stale
# ---------------------------------------------------------------------------


def test_is_cost_cache_stale_fresh_and_expired():
    now = datetime.now(timezone.utc)

    # Fresh cache (within default 600s TTL)
    fresh_cache = {"updated_at": (now - timedelta(seconds=10)).isoformat()}
    assert is_cost_cache_stale(fresh_cache) is False

    # Expired cache (> 600s TTL)
    expired_cache = {"updated_at": (now - timedelta(seconds=700)).isoformat()}
    assert is_cost_cache_stale(expired_cache) is True

    # Custom explicit ttl_seconds override
    mid_cache = {"updated_at": (now - timedelta(seconds=50)).isoformat()}
    assert is_cost_cache_stale(mid_cache, ttl_seconds=30) is True
    assert is_cost_cache_stale(mid_cache, ttl_seconds=100) is False

    # Missing or empty updated_at
    assert is_cost_cache_stale({}) is True
    assert is_cost_cache_stale({"updated_at": ""}) is True
    assert is_cost_cache_stale({"updated_at": "invalid"}) is True

    # Non-dict inputs
    assert is_cost_cache_stale(None) is True
    assert is_cost_cache_stale([]) is True


def test_is_cost_cache_stale_naive_timestamp():
    # Naive timestamp string without tzinfo should not raise TypeError
    now = datetime.now(timezone.utc)
    naive_recent = (now - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S")
    assert is_cost_cache_stale({"updated_at": naive_recent}, ttl_seconds=60) is False


# ---------------------------------------------------------------------------
# 4. build_cost_headers
# ---------------------------------------------------------------------------


def test_build_cost_headers_formatting():
    now = datetime.now(timezone.utc)
    cache = {
        "updated_at": now.isoformat(),
        "total_cost": 42.500000,
        "total_clients": 3,
        "cost_entries": 10,
    }
    headers = build_cost_headers(cache, enabled=True)

    assert headers["X-AcademicAI-Cost-Stale"] == "false"
    assert headers["X-AcademicAI-Cost-Updated-At"] == now.isoformat()
    assert headers["X-AcademicAI-Total-Cost"] == "42.5"
    assert headers["X-AcademicAI-Total-Clients"] == "3"
    assert headers["X-AcademicAI-Cost-Entries"] == "10"


def test_build_cost_headers_number_formatting():
    cache_whole = {"total_cost": 100.0, "updated_at": datetime.now(timezone.utc).isoformat()}
    headers = build_cost_headers(cache_whole, enabled=True)
    assert headers["X-AcademicAI-Total-Cost"] == "100"

    cache_precise = {"total_cost": 0.123456, "updated_at": datetime.now(timezone.utc).isoformat()}
    headers_precise = build_cost_headers(cache_precise, enabled=True)
    assert headers_precise["X-AcademicAI-Total-Cost"] == "0.123456"


def test_build_cost_headers_disabled_toggle():
    cache = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_cost": 10.0,
        "total_clients": 1,
        "cost_entries": 2,
    }

    # Explicitly disabled
    assert build_cost_headers(cache, enabled=False) == {}

    # Empty or None cache
    assert build_cost_headers({}, enabled=True) == {}
    assert build_cost_headers(None, enabled=True) == {}


def test_build_cost_headers_dynamic_server_setting(monkeypatch):
    cache = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_cost": 10.0,
        "total_clients": 1,
        "cost_entries": 2,
    }
    monkeypatch.setattr(server, "ENABLE_COST_MONITORING", False)
    assert build_cost_headers(cache) == {}

    monkeypatch.setattr(server, "ENABLE_COST_MONITORING", True)
    assert "X-AcademicAI-Total-Cost" in build_cost_headers(cache)


# ---------------------------------------------------------------------------
# 5. Cache read/write roundtrip & atomic safety
# ---------------------------------------------------------------------------


def test_cache_read_and_write_roundtrip(tmp_path: Path):
    target_file = tmp_path / "subdir" / "nested" / "cost_cache.json"

    # Reading nonexistent file returns empty dict
    assert read_cost_cache(target_file) == {}

    # Write cache payload (creates parent directories automatically)
    payload = {
        "updated_at": "2026-09-06T18:00:00+00:00",
        "total_cost": 77.88,
        "total_clients": 4,
        "cost_entries": 12,
        "source": "live",
    }
    write_cost_cache(payload, target_file)

    assert target_file.exists()
    cached = read_cost_cache(target_file)
    assert cached == payload


def test_cache_read_corrupt_file(tmp_path: Path):
    target_file = tmp_path / "corrupt_cost_cache.json"
    target_file.write_text("NOT_JSON_DATA!{{{", encoding="utf-8")

    assert read_cost_cache(target_file) == {}


def test_cache_concurrent_writes(tmp_path: Path):
    target_file = tmp_path / "concurrent_cost_cache.json"

    def worker(i: int):
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_cost": float(i),
            "total_clients": i,
            "cost_entries": i,
            "source": f"worker_{i}",
        }
        write_cost_cache(payload, target_file)
        return read_cost_cache(target_file)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(20)))

    assert target_file.exists()
    final_cache = read_cost_cache(target_file)
    assert isinstance(final_cache, dict)
    assert "total_cost" in final_cache
    assert "source" in final_cache


# ---------------------------------------------------------------------------
# 6. Server backward compatibility re-exports & status payload
# ---------------------------------------------------------------------------


def test_server_reexports_backward_compatibility():
    expected_attributes = [
        "_get_cost_cache_with_lazy_refresh",
        "_build_cost_headers",
        "_is_cost_cache_stale",
        "_safe_float",
        "_read_cost_cache",
        "_write_cost_cache",
        "_cost_lock",
    ]
    for attr in expected_attributes:
        assert hasattr(server, attr), f"server is missing expected re-export: {attr}"


def test_get_cost_status_payload():
    cache = {
        "total_cost": 55.25,
        "total_clients": 6,
        "cost_entries": 14,
        "updated_at": "2026-09-06T15:00:00+00:00",
        "source": "cache",
    }
    payload = get_cost_status_payload(cache=cache, enabled=True)
    assert payload == {
        "enabled": True,
        "total_cost": 55.25,
        "total_clients": 6,
        "cost_entries": 14,
        "updated_at": "2026-09-06T15:00:00+00:00",
        "is_stale": True,  # timestamp is from the past
        "source": "cache",
    }

    # Empty cache fallback
    empty_payload = get_cost_status_payload(cache={}, enabled=False)
    assert empty_payload == {
        "enabled": False,
        "total_cost": None,
        "total_clients": None,
        "cost_entries": None,
        "updated_at": None,
        "is_stale": True,
        "source": "none",
    }

