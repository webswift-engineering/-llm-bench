"""Persist benchmark results for dashboard generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from llm_bench.models import BenchmarkResult, Provider, utc_now

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "model_id": r.model_id,
        "display_name": r.display_name,
        "provider": r.provider.value,
        "task": r.task,
        "quality_score": r.quality_score,
        "latency_p50_ms": r.latency_p50_ms,
        "latency_p95_ms": r.latency_p95_ms,
        "cost_per_1k_requests": r.cost_per_1k_requests,
        "cost_per_quality_point": r.cost_per_quality_point,
        "total_cost_usd": r.total_cost_usd,
        "sample_count": r.sample_count,
        "tags": r.tags,
    }


def _dict_to_result(d: dict) -> BenchmarkResult:
    return BenchmarkResult(
        model_id=d["model_id"],
        provider=Provider(d["provider"]),
        display_name=d["display_name"],
        task=d["task"],
        quality_score=d["quality_score"],
        latency_p50_ms=d["latency_p50_ms"],
        latency_p95_ms=d["latency_p95_ms"],
        cost_per_1k_requests=d["cost_per_1k_requests"],
        cost_per_quality_point=d.get("cost_per_quality_point", 0),
        total_cost_usd=d["total_cost_usd"],
        sample_count=d["sample_count"],
        tags=d.get("tags", []),
    )


def save_benchmark_results(task: str, results: list[BenchmarkResult]) -> Path:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARKS_DIR / f"{task}.json"
    payload = {
        "task": task,
        "updated_at": utc_now().isoformat(),
        "results": [_result_to_dict(r) for r in results if r.quality_score > 0 or r.sample_count > 0],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_benchmark_results(task: str) -> list[BenchmarkResult]:
    path = BENCHMARKS_DIR / f"{task}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_dict_to_result(r) for r in data.get("results", [])]


def load_all_benchmarks() -> dict[str, list[BenchmarkResult]]:
    if not BENCHMARKS_DIR.exists():
        return {}
    out: dict[str, list[BenchmarkResult]] = {}
    for path in sorted(BENCHMARKS_DIR.glob("*.json")):
        task = path.stem
        results = load_benchmark_results(task)
        if results:
            out[task] = results
    return out
