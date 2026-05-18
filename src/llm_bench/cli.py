"""CLI entry point."""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv

from llm_bench import __version__
from llm_bench.pareto import compute_pareto_frontier
from llm_bench.pricing.catalog import (
    DEFAULT_BENCHMARK_MODELS,
    benchmarkable_models,
    get_catalog,
    resolve_model_id,
)
from llm_bench.reports.terminal import (
    print_benchmark_table,
    print_pricing_table,
    print_recommendations,
)
from llm_bench.benchmark_store import save_benchmark_results
from llm_bench.reports.html import generate_dashboard
from llm_bench.runner import run_benchmark_sync
from llm_bench.snapshot import save_snapshot


def _load_env() -> None:
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@click.group()
@click.version_option(__version__, prog_name="llm-bench")
def main() -> None:
    """Run LLM benchmarks across providers — cost, latency, and quality."""
    _load_env()


@main.command()
@click.option("--provider", "-p", help="Filter by provider (openai, anthropic, google, groq)")
@click.option("--sort", "sort_by", type=click.Choice(["input", "output", "avg"]), default="avg")
def prices(provider: str | None, sort_by: str) -> None:
    """Show current model pricing across providers."""
    models = get_catalog()
    if provider:
        models = [m for m in models if m.provider.value == provider.lower()]

    key = {"input": "input_per_1m", "output": "output_per_1m", "avg": "avg_per_1m"}[sort_by]
    models = sorted(models, key=lambda m: getattr(m, key))
    print_pricing_table(models)


@main.command()
@click.option("--task", "-t", default="summarization", help="Task type for recommendations")
@click.option("--top", "-n", default=3, help="Number of models to show")
def recommend(task: str, top: int) -> None:
    """Suggest cheapest models for a task (pricing-based)."""
    print_recommendations(task, get_catalog(), top_n=top)


@main.command()
def snapshot() -> None:
    """Save a pricing snapshot to data/snapshots/."""
    path = save_snapshot()
    click.echo(f"Snapshot saved: {path}")


@main.command()
@click.option("--task", "-t", required=True, help="Task name (classification, summarization)")
@click.option("--models", "-m", default=None, help="Comma-separated model IDs or aliases")
@click.option("--budget", "-b", default=5.0, type=float, help="Max spend in USD")
@click.option("--dry-run", is_flag=True, help="Estimate costs without API calls")
@click.option("--output", "-o", default=None, help="Save JSON results to file")
@click.option("--save", is_flag=True, help="Save results to data/benchmarks/ for dashboard")
@click.option("--all", "all_models", is_flag=True, help="Benchmark every catalog model with an adapter")
def run(
    task: str,
    models: str | None,
    budget: float,
    dry_run: bool,
    output: str | None,
    save: bool,
    all_models: bool,
) -> None:
    """Run benchmark across models for a task."""
    if all_models:
        model_ids = benchmarkable_models()
    elif models:
        model_ids = [m.strip() for m in models.split(",")]
    else:
        model_ids = DEFAULT_BENCHMARK_MODELS

    if dry_run:
        click.echo(f"[dry-run] Would benchmark {len(model_ids)} models on '{task}'")

    results = run_benchmark_sync(task, model_ids, budget=budget, dry_run=dry_run)
    print_benchmark_table(results, task, budget)

    frontier = compute_pareto_frontier(results)
    if frontier and not dry_run:
        click.echo("\nPareto frontier: " + " → ".join(r.display_name for r in frontier))

    if save and not dry_run and results:
        path = save_benchmark_results(task, results)
        click.echo(f"Benchmark data saved: {path}")

    if output:
        import json

        data = [
            {
                "model_id": r.model_id,
                "display_name": r.display_name,
                "provider": r.provider.value,
                "quality_score": r.quality_score,
                "latency_p50_ms": r.latency_p50_ms,
                "cost_per_1k_requests": r.cost_per_1k_requests,
                "total_cost_usd": r.total_cost_usd,
            }
            for r in results
        ]
        Path(output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        click.echo(f"Results saved: {output}")


@main.command()
@click.option("--output", "-o", default=None, help="Output directory (default: docs/)")
def dashboard(output: str | None) -> None:
    """Generate static dashboard for GitHub Pages."""
    out = Path(output) if output else None
    path = generate_dashboard(out)
    click.echo(f"Dashboard generated: {path}")
    click.echo("Enable GitHub Pages: Settings → Pages → Source: GitHub Actions")


@main.command("compare")
@click.option("--baseline", "-b", required=True, help="Baseline model ID")
@click.option("--challenger", "-c", required=True, help="Challenger model ID")
@click.option("--task", "-t", required=True, help="Task name")
@click.option("--budget", default=2.0, type=float, help="Max spend in USD")
def compare_cmd(baseline: str, challenger: str, task: str, budget: float) -> None:
    """Head-to-head comparison of two models."""
    baseline_id = resolve_model_id(baseline)
    challenger_id = resolve_model_id(challenger)
    click.echo(f"Comparing {baseline_id} vs {challenger_id} on '{task}'...\n")

    results = run_benchmark_sync(task, [baseline_id, challenger_id], budget=budget)
    print_benchmark_table(results, task, budget)

    scored = [r for r in results if r.quality_score > 0]
    if len(scored) == 2:
        a, b = scored
        delta_q = b.quality_score - a.quality_score
        delta_cost = b.cost_per_1k_requests - a.cost_per_1k_requests
        click.echo(f"\nQuality delta: {delta_q:+.1f} pts | Cost delta: ${delta_cost:+.2f}/1K")


if __name__ == "__main__":
    main()
