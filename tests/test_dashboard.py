"""Tests for dashboard generation."""

from pathlib import Path
from unittest.mock import patch

from llm_bench.models import BenchmarkResult, Provider
from llm_bench.reports.html import generate_dashboard


def _sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        model_id="gpt-4o-mini",
        provider=Provider.OPENAI,
        display_name="GPT-4o mini",
        task="classification",
        quality_score=100.0,
        latency_p50_ms=500.0,
        latency_p95_ms=800.0,
        cost_per_1k_requests=0.17,
        cost_per_quality_point=0.001,
        total_cost_usd=0.01,
        sample_count=8,
        tags=["value"],
    )


def test_generate_dashboard_pricing_only(tmp_path: Path):
    index = generate_dashboard(tmp_path / "docs")
    assert index.exists()
    html = index.read_text(encoding="utf-8")
    assert "llm-bench" in html
    assert "GPT-4o mini" in html
    assert (tmp_path / "docs" / "data" / "prices.json").exists()


def test_generate_dashboard_with_benchmarks(tmp_path: Path):
    with patch(
        "llm_bench.reports.html.load_all_benchmarks",
        return_value={"classification": [_sample_result()]},
    ):
        index = generate_dashboard(tmp_path / "docs")
    html = index.read_text(encoding="utf-8")
    assert "Classification" in html
    assert (tmp_path / "docs" / "data" / "benchmarks.json").exists()


def test_generate_dashboard_injects_ga4_when_configured(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GA_MEASUREMENT_ID", "G-TEST12345")
    index = generate_dashboard(tmp_path / "docs")
    html = index.read_text(encoding="utf-8")
    assert "https://www.googletagmanager.com/gtag/js?id=G-TEST12345" in html
    assert "gtag('config', 'G-TEST12345')" in html
