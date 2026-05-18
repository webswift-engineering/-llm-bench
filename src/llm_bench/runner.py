"""Benchmark runner — executes tasks across models with budget control."""

from __future__ import annotations

import asyncio
from statistics import median

from llm_bench.cost import calculate_cost, cost_per_quality_point, estimate_cost_per_1k_requests
from llm_bench.evaluator import get_judge_model, score_sample
from llm_bench.models import BenchmarkResult, Provider, SampleResult, TokenUsage
from llm_bench.pricing.catalog import get_model, resolve_model_id
from llm_bench.providers.base import get_adapter
from llm_bench.providers.registry import SUPPORTED_PROVIDERS
from llm_bench.tasks.loader import TaskSuite, load_task


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


async def run_model_benchmark(
    model_id: str,
    task: TaskSuite,
    budget_remaining: float,
) -> BenchmarkResult | None:
    resolved_id = resolve_model_id(model_id)
    pricing = get_model(resolved_id)
    if not pricing:
        raise ValueError(f"Unknown model: {model_id}")

    if pricing.provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Provider {pricing.provider.value} not supported in v0.1 "
            f"(adapters: openai, anthropic, groq)"
        )

    adapter = get_adapter(pricing.provider)
    if not adapter.is_configured():
        raise RuntimeError(
            f"{pricing.provider.value.upper()}_API_KEY not set for {resolved_id}"
        )

    judge_model = task.judge_model or get_judge_model()
    sample_results: list[SampleResult] = []
    total_cost = 0.0
    total_usage = TokenUsage()

    for sample in task.samples:
        if budget_remaining <= 0:
            break

        prompt = sample.input
        if task.name == "classification":
            prompt = (
                f"{task.system_prompt}\n\n"
                f"Text to classify:\n{sample.input}\n\n"
                f"Respond with ONLY the category label."
            )
        elif task.name == "summarization":
            prompt = (
                f"{task.system_prompt}\n\n"
                f"Summarize the following text concisely:\n\n{sample.input}"
            )

        response = await adapter.complete(resolved_id, prompt, system=task.system_prompt)
        sample_cost = calculate_cost(resolved_id, response.usage)

        # Judge cost (subjective tasks only)
        judge_cost = 0.0
        if task.scoring == "llm_judge":
            judge_cost = 0.001  # approximate; tracked separately in full accounting

        total_cost += sample_cost + judge_cost
        budget_remaining -= sample_cost + judge_cost
        total_usage.input_tokens += response.usage.input_tokens
        total_usage.output_tokens += response.usage.output_tokens

        quality = await score_sample(task.name, task.scoring, response.text, sample, judge_model)
        sample_results.append(
            SampleResult(quality=quality, latency_ms=response.latency_ms, usage=response.usage)
        )

    if not sample_results:
        return None

    latencies = [s.latency_ms for s in sample_results]
    avg_quality = sum(s.quality for s in sample_results) / len(sample_results)
    cost_1k = estimate_cost_per_1k_requests(pricing)

    tags: list[str] = []
    if avg_quality >= 85 and cost_1k < 1.0:
        tags.append("value")
    if _percentile(latencies, 50) < 500:
        tags.append("speed")
    if cost_1k < 0.20:
        tags.append("cheapest")

    return BenchmarkResult(
        model_id=resolved_id,
        provider=pricing.provider,
        display_name=pricing.display_name,
        task=task.name,
        quality_score=round(avg_quality, 1),
        latency_p50_ms=round(median(latencies), 1),
        latency_p95_ms=round(_percentile(latencies, 95), 1),
        cost_per_1k_requests=round(cost_1k, 2),
        cost_per_quality_point=round(cost_per_quality_point(total_cost, avg_quality), 4),
        total_cost_usd=round(total_cost, 4),
        sample_count=len(sample_results),
        tags=tags,
    )


async def run_benchmark(
    task_name: str,
    model_ids: list[str],
    budget: float = 5.0,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    task = load_task(task_name)
    results: list[BenchmarkResult] = []
    remaining = budget

    if dry_run:
        for mid in model_ids:
            pricing = get_model(resolve_model_id(mid))
            if pricing:
                est = estimate_cost_per_1k_requests(pricing) * len(task.samples) / 1000
                results.append(
                    BenchmarkResult(
                        model_id=pricing.model_id,
                        provider=pricing.provider,
                        display_name=pricing.display_name,
                        task=task.name,
                        quality_score=0.0,
                        latency_p50_ms=0.0,
                        latency_p95_ms=0.0,
                        cost_per_1k_requests=round(estimate_cost_per_1k_requests(pricing), 2),
                        cost_per_quality_point=0.0,
                        total_cost_usd=round(est, 4),
                        sample_count=len(task.samples),
                    )
                )
        return results

    for model_id in model_ids:
        if remaining <= 0:
            break
        try:
            result = await run_model_benchmark(model_id, task, remaining)
            if result:
                results.append(result)
                remaining -= result.total_cost_usd
        except (RuntimeError, ValueError) as exc:
            # Skip models without keys or unsupported providers
            import sys

            print(f"  Skipping {model_id}: {exc}", file=sys.stderr)

    return results


def run_benchmark_sync(
    task_name: str,
    model_ids: list[str],
    budget: float = 5.0,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    return asyncio.run(run_benchmark(task_name, model_ids, budget, dry_run))
