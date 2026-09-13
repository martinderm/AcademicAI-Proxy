"""
Unit tests for LocalCostStore and local cost persistence (academicai/local_cost_tracker.py).
"""

import threading
from decimal import Decimal
from pathlib import Path

import pytest

from academicai.cost_calculation import RequestCost
from academicai.local_cost_tracker import (
    CostAggregationBucket,
    LocalCostStore,
    hash_client_key,
)


def _sample_cost(model="gpt-4o", prompt=100, comp=50, cost="0.001", is_est=False):
    d_cost = Decimal(cost)
    return RequestCost(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=comp,
        total_tokens=prompt + comp,
        input_cost=d_cost * Decimal("0.4"),
        output_cost=d_cost * Decimal("0.6"),
        per_request_cost=Decimal("0"),
        request_cost=d_cost,
        currency="EUR",
        is_estimated=is_est,
    )


def test_hash_client_key():
    k1 = "sk-live-1234567890abcdef"
    h1 = hash_client_key(k1)
    assert h1.startswith("client_")
    assert len(h1) == 15  # "client_" (7) + 8 hex chars
    assert hash_client_key(k1) == h1  # Deterministic
    assert hash_client_key(None) == "client_anonymous"
    assert hash_client_key("") == "client_anonymous"


def test_record_single_request_updates_all_time(tmp_path):
    cache_file = tmp_path / "cost_test.json"
    store = LocalCostStore(cache_file=str(cache_file))

    cost = _sample_cost(model="gpt-4o", prompt=100, comp=50, cost="0.0015")
    store.record_request(cost, client_key="test-key")

    status = store.get_status_payload()
    assert status["enabled"] is True
    all_time = status["all_time"]
    assert all_time["request_count"] == 1
    assert all_time["prompt_tokens"] == 100
    assert all_time["completion_tokens"] == 50
    assert all_time["total_tokens"] == 150
    assert pytest.approx(all_time["request_cost"], 1e-6) == 0.0015


def test_record_updates_by_model_and_client(tmp_path):
    cache_file = tmp_path / "cost_test.json"
    store = LocalCostStore(cache_file=str(cache_file))

    c1 = _sample_cost(model="gpt-4o", prompt=100, comp=50, cost="0.001")
    c2 = _sample_cost(model="gpt-5-mini", prompt=200, comp=100, cost="0.0005")

    store.record_request(c1, client_key="key-alpha")
    store.record_request(c2, client_key="key-alpha")

    status = store.get_status_payload()
    assert "gpt-4o" in status["by_model"]
    assert "gpt-5-mini" in status["by_model"]
    assert status["by_model"]["gpt-4o"]["request_count"] == 1
    assert status["by_model"]["gpt-5-mini"]["request_count"] == 1

    client_id = hash_client_key("key-alpha")
    assert client_id in status["by_client"]
    assert status["by_client"][client_id]["request_count"] == 2
    assert status["by_client"][client_id]["total_tokens"] == 450


def test_ring_buffer_limit(tmp_path):
    cache_file = tmp_path / "cost_test.json"
    store = LocalCostStore(cache_file=str(cache_file), history_limit=3)

    for i in range(5):
        c = _sample_cost(model="gpt-4o", prompt=10 * i, comp=5 * i, cost=f"0.00{i}")
        store.record_request(c, client_key="test-key")

    status = store.get_status_payload()
    recent = status["recent_requests"]
    assert len(recent) == 3
    # Should contain requests 2, 3, 4
    assert recent[0]["prompt_tokens"] == 20
    assert recent[1]["prompt_tokens"] == 30
    assert recent[2]["prompt_tokens"] == 40


def test_privacy_sanitization(tmp_path):
    cache_file = tmp_path / "cost_test.json"
    store = LocalCostStore(cache_file=str(cache_file))

    raw_key = "sk-super-secret-api-key-12345"
    c = _sample_cost()
    store.record_request(c, client_key=raw_key)

    status = store.get_status_payload()
    recent = status["recent_requests"][0]

    # Verify no sensitive or content fields exist
    assert "prompt" not in recent
    assert "completion" not in recent
    assert "messages" not in recent
    assert "api_key" not in recent
    assert raw_key not in str(status)
    assert recent["client_id"].startswith("client_")


def test_atomic_persistence_and_reload(tmp_path):
    cache_file = tmp_path / "cost_test.json"
    store1 = LocalCostStore(cache_file=str(cache_file))

    c1 = _sample_cost(model="gpt-4o", prompt=100, comp=50, cost="0.001")
    c2 = _sample_cost(model="gpt-4o", prompt=200, comp=100, cost="0.002")
    store1.record_request(c1, client_key="key-1")
    store1.record_request(c2, client_key="key-2")

    assert cache_file.exists()

    # Re-instantiate from file (simulating server restart)
    store2 = LocalCostStore(cache_file=str(cache_file))
    status2 = store2.get_status_payload()

    assert status2["all_time"]["request_count"] == 2
    assert status2["all_time"]["total_tokens"] == 450
    assert pytest.approx(status2["all_time"]["request_cost"], 1e-6) == 0.003
    assert len(status2["recent_requests"]) == 2


def test_concurrent_recording(tmp_path):
    cache_file = tmp_path / "cost_test.json"
    store = LocalCostStore(cache_file=str(cache_file))

    def worker(worker_id):
        for _ in range(20):
            c = _sample_cost(model="gpt-4o", prompt=10, comp=5, cost="0.0001")
            store.record_request(c, client_key=f"worker-{worker_id}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status = store.get_status_payload()
    assert status["all_time"]["request_count"] == 100
    assert status["all_time"]["total_tokens"] == 1500
    assert pytest.approx(status["all_time"]["request_cost"], 1e-6) == 0.01
