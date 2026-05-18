"""OpenAI provider adapter."""

from __future__ import annotations

import os
import time

import httpx

from llm_bench.models import ModelResponse, Provider, TokenUsage
from llm_bench.providers.base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    provider = Provider.OPENAI

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = "https://api.openai.com/v1"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def complete(self, model_id: str, prompt: str, system: str = "") -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Reasoning models (o-series, gpt-5+) don't accept `temperature=0`;
        # they only support the default (1).
        omit_temperature = model_id.startswith(("o1", "o3", "o4", "gpt-5"))
        payload: dict = {"model": model_id, "messages": messages}
        if not omit_temperature:
            payload["temperature"] = 0

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        return ModelResponse(
            text=data["choices"][0]["message"]["content"].strip(),
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            latency_ms=latency_ms,
            model_id=model_id,
            provider=self.provider,
        )
