"""Cost calculation from token usage and pricing catalog."""

from __future__ import annotations

from llm_bench.models import ModelPricing, TokenUsage
from llm_bench.pricing.catalog import get_model


def calculate_cost(model_id: str, usage: TokenUsage) -> float:
    pricing = get_model(model_id)
    if not pricing:
        return 0.0
    input_cost = (usage.input_tokens / 1_000_000) * pricing.input_per_1m
    output_cost = (usage.output_tokens / 1_000_000) * pricing.output_per_1m
    return input_cost + output_cost


def estimate_cost_per_1k_requests(
    pricing: ModelPricing,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 150,
) -> float:
    usage = TokenUsage(input_tokens=avg_input_tokens, output_tokens=avg_output_tokens)
    per_request = calculate_cost(pricing.model_id, usage)
    return per_request * 1000


def cost_per_quality_point(total_cost: float, quality: float) -> float:
    if quality <= 0:
        return float("inf")
    return total_cost / quality
