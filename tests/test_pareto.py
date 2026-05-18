"""Tests for Pareto frontier calculation."""

from llm_bench.models import BenchmarkResult, Provider
from llm_bench.pareto import compute_pareto_frontier


def _result(name: str, quality: float, cost: float) -> BenchmarkResult:
    return BenchmarkResult(
        model_id=name,
        provider=Provider.OPENAI,
        display_name=name,
        task="test",
        quality_score=quality,
        latency_p50_ms=100,
        latency_p95_ms=200,
        cost_per_1k_requests=cost,
        cost_per_quality_point=cost / quality if quality else 0,
        total_cost_usd=0.01,
        sample_count=5,
    )


def test_pareto_frontier_excludes_dominated():
    results = [
        _result("cheap-low", 60, 0.10),
        _result("cheap-high", 85, 0.15),
        _result("expensive-high", 90, 2.00),
        _result("dominated", 70, 1.00),  # worse quality AND more expensive than cheap-high
    ]
    frontier = compute_pareto_frontier(results)
    names = [r.model_id for r in frontier]
    assert "cheap-low" in names
    assert "cheap-high" in names
    assert "expensive-high" in names
    assert "dominated" not in names


def test_pareto_empty_when_no_scores():
    assert compute_pareto_frontier([]) == []
