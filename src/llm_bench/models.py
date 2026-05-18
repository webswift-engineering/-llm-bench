"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"


@dataclass(frozen=True)
class ModelPricing:
    provider: Provider
    model_id: str
    display_name: str
    input_per_1m: float
    output_per_1m: float
    context_window: int = 128_000

    @property
    def avg_per_1m(self) -> float:
        """Blended cost assuming 3:1 input/output token ratio."""
        return (3 * self.input_per_1m + self.output_per_1m) / 4


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelResponse:
    text: str
    usage: TokenUsage
    latency_ms: float
    model_id: str
    provider: Provider


@dataclass
class SampleResult:
    quality: float
    latency_ms: float
    usage: TokenUsage
    raw_output: str = ""


@dataclass
class BenchmarkResult:
    model_id: str
    provider: Provider
    display_name: str
    task: str
    quality_score: float
    latency_p50_ms: float
    latency_p95_ms: float
    cost_per_1k_requests: float
    cost_per_quality_point: float
    total_cost_usd: float
    sample_count: int
    tags: list[str] = field(default_factory=list)


@dataclass
class PricingSnapshot:
    captured_at: datetime
    models: list[ModelPricing]

    def to_dict(self) -> dict:
        return {
            "captured_at": self.captured_at.isoformat(),
            "models": [
                {
                    "provider": m.provider.value,
                    "model_id": m.model_id,
                    "display_name": m.display_name,
                    "input_per_1m": m.input_per_1m,
                    "output_per_1m": m.output_per_1m,
                    "context_window": m.context_window,
                }
                for m in self.models
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PricingSnapshot:
        return cls(
            captured_at=datetime.fromisoformat(data["captured_at"]),
            models=[
                ModelPricing(
                    provider=Provider(m["provider"]),
                    model_id=m["model_id"],
                    display_name=m["display_name"],
                    input_per_1m=m["input_per_1m"],
                    output_per_1m=m["output_per_1m"],
                    context_window=m.get("context_window", 128_000),
                )
                for m in data["models"]
            ],
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
