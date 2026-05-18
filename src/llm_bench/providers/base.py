"""Provider adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_bench.models import ModelResponse, Provider


class ProviderAdapter(ABC):
    provider: Provider

    @abstractmethod
    async def complete(self, model_id: str, prompt: str, system: str = "") -> ModelResponse:
        """Send a completion request and return response with usage + latency."""

    def is_configured(self) -> bool:
        return True


def get_adapter(provider: Provider) -> ProviderAdapter:
    from llm_bench.providers.anthropic import AnthropicAdapter
    from llm_bench.providers.groq import GroqAdapter
    from llm_bench.providers.openai import OpenAIAdapter

    adapters = {
        Provider.OPENAI: OpenAIAdapter,
        Provider.ANTHROPIC: AnthropicAdapter,
        Provider.GROQ: GroqAdapter,
    }
    cls = adapters.get(provider)
    if not cls:
        raise ValueError(f"No adapter for provider: {provider.value}")
    return cls()
