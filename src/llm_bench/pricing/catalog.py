"""Pricing reference data — verified from official provider pricing pages.

NOTE: This catalog is **reference data**, not benchmarked. Numbers below were
copied directly from each provider's pricing page on the date noted in
``CATALOG_VERIFIED``. See ``source_url`` on each entry for the authoritative URL.

Only providers with adapters (OpenAI, Anthropic, Groq) are included in v0.1.
"""

from __future__ import annotations

from llm_bench.models import ModelPricing, Provider

CATALOG_VERIFIED = "2026-05-18"

OPENAI_SRC = "https://openai.com/api/pricing/"
ANTHROPIC_SRC = "https://www.anthropic.com/pricing"
GROQ_SRC = "https://groq.com/pricing/"

PROVIDER_SOURCES: dict[Provider, str] = {
    Provider.OPENAI: OPENAI_SRC,
    Provider.ANTHROPIC: ANTHROPIC_SRC,
    Provider.GROQ: GROQ_SRC,
}

# All prices in USD per 1M tokens (standard tier, context < 200K)
PRICING_CATALOG: list[ModelPricing] = [
    # ── OpenAI (openai.com/api/pricing/) ──
    ModelPricing(Provider.OPENAI, "gpt-5.5", "GPT-5.5", 5.00, 30.00, 400_000, OPENAI_SRC),
    ModelPricing(Provider.OPENAI, "gpt-5.4", "GPT-5.4", 2.50, 15.00, 400_000, OPENAI_SRC),
    ModelPricing(Provider.OPENAI, "gpt-5.4-mini", "GPT-5.4 mini", 0.75, 4.50, 400_000, OPENAI_SRC),
    # Legacy models still available via API (prices from prior pricing snapshots)
    ModelPricing(Provider.OPENAI, "gpt-4o", "GPT-4o", 2.50, 10.00, 128_000, OPENAI_SRC),
    ModelPricing(Provider.OPENAI, "gpt-4o-mini", "GPT-4o mini", 0.15, 0.60, 128_000, OPENAI_SRC),
    ModelPricing(Provider.OPENAI, "gpt-4.1", "GPT-4.1", 2.00, 8.00, 1_047_576, OPENAI_SRC),
    ModelPricing(Provider.OPENAI, "gpt-4.1-mini", "GPT-4.1 mini", 0.40, 1.60, 1_047_576, OPENAI_SRC),
    ModelPricing(Provider.OPENAI, "o3-mini", "o3-mini", 1.10, 4.40, 200_000, OPENAI_SRC),
    # ── Anthropic (anthropic.com/pricing) ──
    ModelPricing(Provider.ANTHROPIC, "claude-opus-4-7", "Claude Opus 4.7", 5.00, 25.00, 1_000_000, ANTHROPIC_SRC),
    ModelPricing(Provider.ANTHROPIC, "claude-sonnet-4-6", "Claude Sonnet 4.6", 3.00, 15.00, 200_000, ANTHROPIC_SRC),
    ModelPricing(
        Provider.ANTHROPIC, "claude-opus-4-5-20251101", "Claude Opus 4.5", 5.00, 25.00, 200_000, ANTHROPIC_SRC
    ),
    ModelPricing(
        Provider.ANTHROPIC, "claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", 3.00, 15.00, 200_000, ANTHROPIC_SRC
    ),
    ModelPricing(
        Provider.ANTHROPIC, "claude-haiku-4-5-20251001", "Claude Haiku 4.5", 1.00, 5.00, 200_000, ANTHROPIC_SRC
    ),
    ModelPricing(
        Provider.ANTHROPIC, "claude-opus-4-1-20250805", "Claude Opus 4.1", 15.00, 75.00, 200_000, ANTHROPIC_SRC
    ),
    # ── Groq (groq.com/pricing/) ──
    ModelPricing(
        Provider.GROQ, "openai/gpt-oss-120b", "GPT OSS 120B (Groq)", 0.15, 0.60, 128_000, GROQ_SRC
    ),
    ModelPricing(
        Provider.GROQ, "openai/gpt-oss-20b", "GPT OSS 20B (Groq)", 0.075, 0.30, 128_000, GROQ_SRC
    ),
    ModelPricing(
        Provider.GROQ,
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "Llama 4 Scout 17B (Groq)",
        0.11,
        0.34,
        128_000,
        GROQ_SRC,
    ),
    ModelPricing(Provider.GROQ, "qwen/qwen3-32b", "Qwen3 32B (Groq)", 0.29, 0.59, 131_000, GROQ_SRC),
    ModelPricing(
        Provider.GROQ, "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", 0.59, 0.79, 128_000, GROQ_SRC
    ),
    ModelPricing(
        Provider.GROQ, "llama-3.1-8b-instant", "Llama 3.1 8B (Groq)", 0.05, 0.08, 128_000, GROQ_SRC
    ),
]

# Short aliases for --models flag
MODEL_ALIASES: dict[str, str] = {
    "gpt-5.5": "gpt-5.5",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-opus": "claude-opus-4-7",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "groq-llama-70b": "llama-3.3-70b-versatile",
    "groq-llama-8b": "llama-3.1-8b-instant",
    "groq-llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq-gpt-oss-120b": "openai/gpt-oss-120b",
    "groq-gpt-oss-20b": "openai/gpt-oss-20b",
    "groq-qwen3-32b": "qwen/qwen3-32b",
}

# Default subset for quick runs
DEFAULT_BENCHMARK_MODELS = [
    "gpt-5.4-mini",
    "gpt-4o-mini",
    "claude-haiku-4-5-20251001",
    "llama-3.1-8b-instant",
]


def get_catalog() -> list[ModelPricing]:
    return list(PRICING_CATALOG)


def get_model(model_id: str) -> ModelPricing | None:
    resolved = MODEL_ALIASES.get(model_id, model_id)
    for m in PRICING_CATALOG:
        if m.model_id == resolved:
            return m
    return None


def resolve_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(model_id, model_id)


def benchmarkable_models() -> list[str]:
    """All catalog models from providers with adapters."""
    from llm_bench.providers.registry import SUPPORTED_PROVIDERS

    return [m.model_id for m in PRICING_CATALOG if m.provider in SUPPORTED_PROVIDERS]
