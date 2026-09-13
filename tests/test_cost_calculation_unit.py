"""
Unit tests for ModelCatalog and Decimal request cost calculation (academicai/cost_calculation.py).
"""

import json
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from academicai.cost_calculation import (
    ModelCatalog,
    ModelEntry,
    ModelPricing,
    RequestCost,
    calculate_request_cost,
    parse_model_costs,
    get_model_catalog,
)


def test_parse_model_costs_standard():
    raw_costs = [
        {"costType": "input_tokens", "cost": 0.00275},
        {"costType": "output_tokens", "cost": 0.011},
    ]
    pricing = parse_model_costs("gpt-4o", raw_costs)
    assert pricing.model_id == "gpt-4o"
    assert pricing.input_cost_per_token == Decimal("0.00000275")
    assert pricing.output_cost_per_token == Decimal("0.000011")
    assert pricing.per_request_cost == Decimal("0")
    assert pricing.is_tiered is False
    assert pricing.currency == "EUR"
    assert len(pricing.raw_costs) == 2


def test_parse_model_costs_tiered():
    raw_costs = [
        {"costType": "input_tokens", "cost": 0.00125},
        {"costType": "input_tokens", "cost": 0.0025},
        {"costType": "output_tokens", "cost": 0.005},
        {"costType": "output_tokens", "cost": 0.01},
    ]
    pricing = parse_model_costs("gemini-2.5-pro", raw_costs)
    assert pricing.model_id == "gemini-2.5-pro"
    # Base/first tier selected
    assert pricing.input_cost_per_token == Decimal("0.00000125")
    assert pricing.output_cost_per_token == Decimal("0.000005")
    assert pricing.is_tiered is True


def test_parse_model_costs_per_request():
    raw_costs = [
        {"costType": "input_tokens", "cost": 0.001},
        {"costType": "output_tokens", "cost": 0.001},
        {"costType": "per_request", "cost": 0.006},
    ]
    pricing = parse_model_costs("sonar-pro", raw_costs)
    assert pricing.per_request_cost == Decimal("0.006")
    assert pricing.is_tiered is False


def test_parse_model_costs_empty():
    pricing = parse_model_costs("unknown-custom", [])
    assert pricing.input_cost_per_token == Decimal("0")
    assert pricing.output_cost_per_token == Decimal("0")
    assert pricing.per_request_cost == Decimal("0")
    assert pricing.is_tiered is False


def test_calculate_request_cost_exact():
    pricing_map = {
        "gpt-4o": ModelPricing(
            model_id="gpt-4o",
            input_cost_per_token=Decimal("0.00000275"),
            output_cost_per_token=Decimal("0.000011"),
            per_request_cost=Decimal("0"),
            currency="EUR",
            is_tiered=False,
        )
    }
    # 1000 prompt tokens, 500 completion tokens
    cost = calculate_request_cost("gpt-4o", 1000, 500, pricing_map=pricing_map)
    assert cost.model == "gpt-4o"
    assert cost.prompt_tokens == 1000
    assert cost.completion_tokens == 500
    assert cost.total_tokens == 1500
    assert cost.input_cost == Decimal("0.00275")
    assert cost.output_cost == Decimal("0.0055")
    assert cost.per_request_cost == Decimal("0")
    assert cost.request_cost == Decimal("0.00825")
    assert cost.is_estimated is False
    assert cost.currency == "EUR"


def test_calculate_request_cost_decimal_precision():
    pricing_map = {
        "gpt-5-nano": ModelPricing(
            model_id="gpt-5-nano",
            input_cost_per_token=Decimal("0.00000006"),
            output_cost_per_token=Decimal("0.00000044"),
            per_request_cost=Decimal("0"),
            currency="EUR",
            is_tiered=False,
        )
    }
    cost = calculate_request_cost("gpt-5-nano", 7, 3, pricing_map=pricing_map)
    expected_input = Decimal("7") * Decimal("0.00000006")
    expected_output = Decimal("3") * Decimal("0.00000044")
    assert cost.input_cost == expected_input
    assert cost.output_cost == expected_output
    assert cost.request_cost == expected_input + expected_output
    assert isinstance(cost.request_cost, Decimal)


def test_calculate_request_cost_unknown_model_fallback():
    cost = calculate_request_cost("unregistered-model", 500, 100, pricing_map={})
    assert cost.model == "unregistered-model"
    assert cost.input_cost == Decimal("0")
    assert cost.output_cost == Decimal("0")
    assert cost.request_cost == Decimal("0")
    assert cost.is_estimated is True


def test_calculate_request_cost_tiered_model_sets_estimated():
    pricing_map = {
        "gemini-2.5-pro": ModelPricing(
            model_id="gemini-2.5-pro",
            input_cost_per_token=Decimal("0.00000125"),
            output_cost_per_token=Decimal("0.000005"),
            per_request_cost=Decimal("0"),
            currency="EUR",
            is_tiered=True,
        )
    }
    cost = calculate_request_cost("gemini-2.5-pro", 100, 50, pricing_map=pricing_map)
    assert cost.is_estimated is True
    assert cost.request_cost > Decimal("0")


def test_calculate_request_cost_zero_tokens():
    pricing_map = {
        "gpt-4o": ModelPricing(
            model_id="gpt-4o",
            input_cost_per_token=Decimal("0.00000275"),
            output_cost_per_token=Decimal("0.000011"),
            per_request_cost=Decimal("0"),
            currency="EUR",
            is_tiered=False,
        )
    }
    cost = calculate_request_cost("gpt-4o", 0, 0, pricing_map=pricing_map)
    assert cost.prompt_tokens == 0
    assert cost.completion_tokens == 0
    assert cost.total_tokens == 0
    assert cost.request_cost == Decimal("0")
    assert cost.is_estimated is False


def test_calculate_request_cost_with_per_request():
    pricing_map = {
        "sonar-pro": ModelPricing(
            model_id="sonar-pro",
            input_cost_per_token=Decimal("0.000001"),
            output_cost_per_token=Decimal("0.000001"),
            per_request_cost=Decimal("0.006"),
            currency="EUR",
            is_tiered=False,
        )
    }
    cost = calculate_request_cost("sonar-pro", 100, 200, pricing_map=pricing_map)
    # (100 * 0.000001) + (200 * 0.000001) + 0.006 = 0.0001 + 0.0002 + 0.006 = 0.0063
    assert cost.request_cost == Decimal("0.0063")


def test_request_cost_to_headers():
    cost = RequestCost(
        model="gpt-4o",
        prompt_tokens=120,
        completion_tokens=45,
        total_tokens=165,
        input_cost=Decimal("0.00033"),
        output_cost=Decimal("0.000495"),
        per_request_cost=Decimal("0"),
        request_cost=Decimal("0.000825"),
        currency="EUR",
        is_estimated=False,
    )
    headers = cost.to_headers()
    assert headers["X-AcademicAI-Request-Cost"] == "0.000825"
    assert headers["X-AcademicAI-Input-Cost"] == "0.00033"
    assert headers["X-AcademicAI-Output-Cost"] == "0.000495"
    assert headers["X-AcademicAI-Prompt-Tokens"] == "120"
    assert headers["X-AcademicAI-Completion-Tokens"] == "45"
    assert headers["X-AcademicAI-Cost-Currency"] == "EUR"
    assert headers["X-AcademicAI-Cost-Estimated"] == "false"


def test_request_cost_to_headers_estimated():
    cost = RequestCost(
        model="gpt-5.5",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        input_cost=Decimal("0.0001"),
        output_cost=Decimal("0.0002"),
        per_request_cost=Decimal("0"),
        request_cost=Decimal("0.0003"),
        currency="EUR",
        is_estimated=True,
    )
    headers = cost.to_headers()
    assert headers["X-AcademicAI-Cost-Estimated"] == "true"


def test_model_catalog_staleness_and_fallback(tmp_path):
    catalog = ModelCatalog(cache_file=tmp_path / "catalog.json", ttl_seconds=60)
    assert catalog.is_stale() is True

    # Manually populate catalog
    catalog._pricing_map = {
        "gpt-4o": ModelPricing("gpt-4o", Decimal("0.00000275"), Decimal("0.000011"), Decimal("0"), "EUR", False)
    }
    catalog._last_refreshed_at = datetime.now(timezone.utc)
    assert catalog.is_stale() is False
    assert catalog.get_pricing("gpt-4o") is not None

    # Simulate expired TTL
    catalog._last_refreshed_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    assert catalog.is_stale() is True

    # If refresh fails, fallback returns existing cached pricing
    with patch.object(catalog, "_fetch_from_backend", side_effect=RuntimeError("Network error")):
        pricing = catalog.get_pricing_with_refresh("gpt-4o")
        assert pricing is not None
        assert pricing.model_id == "gpt-4o"


def test_model_pricing_to_dict_and_from_dict():
    pricing = ModelPricing(
        model_id="gpt-4o",
        input_cost_per_token=Decimal("0.00000275"),
        output_cost_per_token=Decimal("0.000011"),
        per_request_cost=Decimal("0.005"),
        currency="EUR",
        is_tiered=True,
        raw_costs=[{"costType": "input_tokens", "cost": 0.00275}],
    )
    d = pricing.to_dict()
    assert d["model_id"] == "gpt-4o"
    assert d["input_cost_per_token"] == "0.00000275"
    assert d["output_cost_per_token"] == "0.000011"
    assert d["per_request_cost"] == "0.005"
    assert d["currency"] == "EUR"
    assert d["is_tiered"] is True
    assert len(d["raw_costs"]) == 1

    restored = ModelPricing.from_dict(d)
    assert restored.model_id == pricing.model_id
    assert restored.input_cost_per_token == pricing.input_cost_per_token
    assert restored.output_cost_per_token == pricing.output_cost_per_token
    assert restored.per_request_cost == pricing.per_request_cost
    assert restored.currency == pricing.currency
    assert restored.is_tiered == pricing.is_tiered
    assert restored.raw_costs == pricing.raw_costs


def test_model_catalog_default_24h_ttl(tmp_path):
    catalog = ModelCatalog(cache_file=tmp_path / "catalog.json", ttl_seconds=None)
    assert catalog.ttl_seconds == 86400

    now = datetime.now(timezone.utc)
    catalog._pricing_map = {
        "gpt-4o": ModelPricing("gpt-4o", Decimal("0.00000275"), Decimal("0.000011"), Decimal("0"), "EUR", False)
    }

    # Within 24 hours: fresh
    catalog._last_refreshed_at = now - timedelta(hours=23)
    assert catalog.is_stale() is False

    # After 24 hours: stale
    catalog._last_refreshed_at = now - timedelta(hours=25)
    assert catalog.is_stale() is True


def test_model_catalog_disk_persistence(tmp_path):
    catalog_file = tmp_path / "model_catalog.json"
    cat1 = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400)

    now = datetime.now(timezone.utc)
    cat1._pricing_map = {
        "gpt-4o": ModelPricing(
            "gpt-4o",
            Decimal("0.00000275"),
            Decimal("0.000011"),
            Decimal("0"),
            "EUR",
            False,
            [{"costType": "input_tokens", "cost": 0.00275}],
        )
    }
    cat1._last_refreshed_at = now
    cat1._save_to_disk()

    assert catalog_file.exists()

    # Create a new catalog instance pointing to the same file
    cat2 = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400)
    assert cat2.is_stale() is False
    pricing = cat2.get_pricing("gpt-4o")
    assert pricing is not None
    assert pricing.model_id == "gpt-4o"
    assert pricing.input_cost_per_token == Decimal("0.00000275")
    assert pricing.output_cost_per_token == Decimal("0.000011")


def test_model_catalog_get_status(tmp_path):
    catalog_file = tmp_path / "status_test_catalog.json"
    catalog = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400)
    status = catalog.get_status()

    assert status["models_cached"] == 0
    assert status["last_refreshed_at"] is None
    assert status["is_stale"] is True
    assert status["ttl_seconds"] == 86400
    assert status["currency"] == "EUR"
    assert status["catalog_file"] == str(catalog_file)


def test_model_catalog_currency_ssot(tmp_path):
    catalog_file = tmp_path / "catalog_ssot.json"
    cat1 = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400, currency="EUR")
    cat1._pricing_map = {
        "gpt-4o": ModelPricing("gpt-4o", Decimal("0.00000275"), Decimal("0.000011"), Decimal("0"))
    }
    cat1._last_refreshed_at = datetime.now(timezone.utc)
    cat1._save_to_disk()

    # Verify JSON file has currency defined once at the root level
    raw = json.loads(catalog_file.read_text(encoding="utf-8"))
    assert raw["currency"] == "EUR"

    # Modify currency directly in the JSON file to USD
    raw["currency"] = "USD"
    catalog_file.write_text(json.dumps(raw), encoding="utf-8")

    # Reload in a new catalog instance
    cat2 = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400)
    assert cat2.currency == "USD"
    pricing = cat2.get_pricing("gpt-4o")
    assert pricing is not None
    assert pricing.currency == "USD"

    # Calculation dynamically picks up the currency from the catalog SSOT
    cost = calculate_request_cost("gpt-4o", 100, 50, catalog=cat2)
    assert cost.currency == "USD"
    assert cost.to_headers()["X-AcademicAI-Cost-Currency"] == "USD"


def test_parse_model_costs_with_metadata():
    raw_costs = [
        {"costType": "input_tokens", "cost": 0.0025},
        {"costType": "output_tokens", "cost": 0.010},
    ]
    entry = parse_model_costs(
        "gpt-5",
        raw_costs,
        context_window=1048576,
        output_token_limit=65535,
    )
    assert entry.model_id == "gpt-5"
    assert entry.context_window == 1048576
    assert entry.output_token_limit == 65535
    assert "context_window=1048576" in repr(entry)

    # Serialization roundtrip
    d = entry.to_dict()
    assert d["context_window"] == 1048576
    assert d["output_token_limit"] == 65535

    restored = ModelEntry.from_dict(d)
    assert restored.model_id == "gpt-5"
    assert restored.context_window == 1048576
    assert restored.output_token_limit == 65535


def test_model_catalog_to_openai_models_response(tmp_path):
    catalog_file = tmp_path / "catalog_test.json"
    catalog = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400)
    catalog._pricing_map = {
        "gpt-5": ModelEntry(
            model_id="gpt-5",
            input_cost_per_token=Decimal("0.0000025"),
            output_cost_per_token=Decimal("0.00001"),
            per_request_cost=Decimal("0"),
            currency="EUR",
            is_tiered=False,
            raw_costs=[{"costType": "input_tokens", "cost": 0.0025}],
            context_window=1048576,
            output_token_limit=65535,
        ),
        "gpt-5-mini": ModelEntry(
            model_id="gpt-5-mini",
            input_cost_per_token=Decimal("0.00000028"),
            output_cost_per_token=Decimal("0.0000022"),
            per_request_cost=Decimal("0"),
            currency="EUR",
            is_tiered=False,
            raw_costs=[{"costType": "input_tokens", "cost": 0.00028}],
            context_window=400000,
            output_token_limit=128000,
        ),
    }

    resp = catalog.to_openai_models_response()
    assert resp["object"] == "list"
    data = resp["data"]
    assert len(data) == 2

    m_map = {item["id"]: item for item in data}
    assert "gpt-5" in m_map
    assert m_map["gpt-5"]["object"] == "model"
    assert m_map["gpt-5"]["owned_by"] == "academicai"
    assert m_map["gpt-5"]["context_window"] == 1048576
    assert m_map["gpt-5"]["max_tokens"] == 65535
    assert len(m_map["gpt-5"]["costs"]) == 1

    assert "gpt-5-mini" in m_map
    assert m_map["gpt-5-mini"]["context_window"] == 400000
    assert m_map["gpt-5-mini"]["max_tokens"] == 128000


def test_model_catalog_get_model_and_get_models(tmp_path):
    catalog_file = tmp_path / "catalog_test.json"
    catalog = ModelCatalog(cache_file=catalog_file, ttl_seconds=86400)
    catalog._pricing_map = {
        "gpt-4o": ModelEntry("gpt-4o", Decimal("0.00000275"), Decimal("0.000011"), Decimal("0"))
    }
    assert catalog.get_model("gpt-4o") is not None
    assert catalog.get_model("unknown") is None
    all_models = catalog.get_models()
    assert "gpt-4o" in all_models
    assert len(all_models) == 1


def test_model_catalog_default_file_path(monkeypatch):
    monkeypatch.delenv("ACADEMICAI_MODEL_CATALOG_FILE", raising=False)
    catalog = ModelCatalog()
    assert catalog.cache_file_path.name == "model_catalog.json"
    assert "data" in str(catalog.cache_file_path)



