"""Tests for pricing catalog and cost calculation."""

from llm_bench.cost import calculate_cost, estimate_cost_per_1k_requests
from llm_bench.models import TokenUsage
from llm_bench.pricing.catalog import get_catalog, get_model, resolve_model_id


def test_catalog_has_models():
    catalog = get_catalog()
    assert len(catalog) >= 15


def test_get_model_by_id():
    model = get_model("gpt-4o-mini")
    assert model is not None
    assert model.input_per_1m == 0.15


def test_resolve_alias():
    assert resolve_model_id("claude-haiku") == "claude-3-5-haiku-20241022"
    assert resolve_model_id("groq-llama-70b") == "llama-3.3-70b-versatile"


def test_calculate_cost():
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    cost = calculate_cost("gpt-4o-mini", usage)
    # 1000/1M * 0.15 + 500/1M * 0.60 = 0.00015 + 0.0003 = 0.00045
    assert 0.0004 < cost < 0.0005


def test_estimate_cost_per_1k():
    model = get_model("gpt-4o-mini")
    assert model is not None
    cost_1k = estimate_cost_per_1k_requests(model)
    assert 0.10 < cost_1k < 1.0
