"""Pareto frontier calculation for cost vs quality."""

from __future__ import annotations

from llm_bench.models import BenchmarkResult


def compute_pareto_frontier(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Return models on the Pareto frontier (not dominated on cost AND quality)."""
    scored = [r for r in results if r.quality_score > 0]
    if not scored:
        return []

    sorted_by_cost = sorted(scored, key=lambda r: r.cost_per_1k_requests)
    frontier: list[BenchmarkResult] = []
    max_quality = 0.0

    for model in sorted_by_cost:
        if model.quality_score > max_quality:
            frontier.append(model)
            max_quality = model.quality_score

    return frontier
