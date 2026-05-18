"""Anthropic provider adapter."""

from __future__ import annotations

import os
import time

import httpx

from llm_bench.models import ModelResponse, Provider, TokenUsage
from llm_bench.providers.base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    provider = Provider.ANTHROPIC

    def __init__(self) -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = "https://api.anthropic.com/v1"

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def complete(self, model_id: str, prompt: str, system: str = "") -> ModelResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        # Newer Claude (Opus 4.7+) deprecated `temperature`; older models still accept it.
        omit_temperature = "opus-4-7" in model_id or "opus-4-6" in model_id or "sonnet-4-6" in model_id
        body: dict = {
            "model": model_id,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not omit_temperature:
            body["temperature"] = 0
        if system:
            body["system"] = system

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        return ModelResponse(
            text=text.strip(),
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            latency_ms=latency_ms,
            model_id=model_id,
            provider=self.provider,
        )
