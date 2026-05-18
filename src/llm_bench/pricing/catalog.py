"""Hardcoded pricing catalog — updated manually; live scraping in v0.2."""

from __future__ import annotations

from llm_bench.models import ModelPricing, Provider

PRICING_CATALOG: list[ModelPricing] = [
    # OpenAI
    ModelPricing(Provider.OPENAI, "gpt-4o", "GPT-4o", 2.50, 10.00, 128_000),
    ModelPricing(Provider.OPENAI, "gpt-4o-mini", "GPT-4o mini", 0.15, 0.60, 128_000),
    ModelPricing(Provider.OPENAI, "gpt-4.1", "GPT-4.1", 2.00, 8.00, 1_047_576),
    ModelPricing(Provider.OPENAI, "gpt-4.1-mini", "GPT-4.1 mini", 0.40, 1.60, 1_047_576),
    ModelPricing(Provider.OPENAI, "o3-mini", "o3-mini", 1.10, 4.40, 200_000),
    # Anthropic
    ModelPricing(Provider.ANTHROPIC, "claude-sonnet-4-20250514", "Claude Sonnet 4", 3.00, 15.00, 200_000),
    ModelPricing(
        Provider.ANTHROPIC, "claude-3-5-haiku-20241022", "Claude 3.5 Haiku", 0.80, 4.00, 200_000
    ),
    ModelPricing(Provider.ANTHROPIC, "claude-3-haiku-20240307", "Claude 3 Haiku", 0.25, 1.25, 200_000),
    # Google
    ModelPricing(Provider.GOOGLE, "gemini-2.0-flash", "Gemini 2.0 Flash", 0.10, 0.40, 1_048_576),
    ModelPricing(Provider.GOOGLE, "gemini-1.5-flash", "Gemini 1.5 Flash", 0.075, 0.30, 1_048_576),
    ModelPricing(Provider.GOOGLE, "gemini-1.5-pro", "Gemini 1.5 Pro", 1.25, 5.00, 2_097_152),
    # Groq
    ModelPricing(Provider.GROQ, "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", 0.59, 0.79, 128_000),
    ModelPricing(Provider.GROQ, "llama-3.1-8b-instant", "Llama 3.1 8B (Groq)", 0.05, 0.08, 128_000),
    ModelPricing(Provider.GROQ, "mixtral-8x7b-32768", "Mixtral 8x7B (Groq)", 0.24, 0.24, 32_768),
    # Mistral
    ModelPricing(Provider.MISTRAL, "mistral-large-latest", "Mistral Large", 2.00, 6.00, 128_000),
    ModelPricing(Provider.MISTRAL, "mistral-small-latest", "Mistral Small", 0.20, 0.60, 128_000),
    # DeepSeek
    ModelPricing(Provider.DEEPSEEK, "deepseek-chat", "DeepSeek Chat", 0.27, 1.10, 64_000),
    ModelPricing(Provider.DEEPSEEK, "deepseek-reasoner", "DeepSeek Reasoner", 0.55, 2.19, 64_000),
]

# Short aliases for CLI --models flag
MODEL_ALIASES: dict[str, str] = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-3-5-haiku-20241022",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "gemini-flash": "gemini-2.0-flash",
    "groq-llama-70b": "llama-3.3-70b-versatile",
    "groq-llama-8b": "llama-3.1-8b-instant",
    "mistral-large": "mistral-large-latest",
    "mistral-small": "mistral-small-latest",
}

# Default models for benchmark runs (providers with adapters in v0.1)
DEFAULT_BENCHMARK_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "llama-3.3-70b-versatile",
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
