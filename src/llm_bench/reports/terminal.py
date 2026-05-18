"""Rich terminal output for pricing and benchmark results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_bench.models import BenchmarkResult, ModelPricing

console = Console()


def print_pricing_table(models: list[ModelPricing], provider_filter: str | None = None) -> None:
    filtered = models
    if provider_filter:
        filtered = [m for m in models if m.provider.value == provider_filter]

    filtered = sorted(filtered, key=lambda m: m.avg_per_1m)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="dim", width=12)
    table.add_column("Model", width=28)
    table.add_column("Input/1M", justify="right")
    table.add_column("Output/1M", justify="right")
    table.add_column("Avg/1M", justify="right", style="green")

    for m in filtered:
        table.add_row(
            m.provider.value.ljust(12),
            m.display_name,
            f"${m.input_per_1m:.2f}",
            f"${m.output_per_1m:.2f}",
            f"${m.avg_per_1m:.2f}",
        )

    console.print(table)


def print_benchmark_table(results: list[BenchmarkResult], task: str, budget: float) -> None:
    if not results:
        console.print("[yellow]No results — check API keys or use --dry-run[/yellow]")
        return

    console.print(f'\n[bold]Running "{task}" benchmark[/bold] (budget: ${budget:.2f})\n')

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", width=22)
    table.add_column("Quality", justify="right", width=8)
    table.add_column("Latency p50", justify="right", width=12)
    table.add_column("Cost/1K", justify="right", width=10)
    table.add_column("$/quality", justify="right", width=10)
    table.add_column("Best for", width=12)

    sorted_results = sorted(results, key=lambda r: r.quality_score, reverse=True)

    for r in sorted_results:
        tags = []
        if "value" in r.tags:
            tags.append("* Value")
        if "speed" in r.tags:
            tags.append("* Speed")
        if "cheapest" in r.tags:
            tags.append("* Cheapest")
        if r.quality_score >= 95:
            tags.append("Max qual.")

        best_for = tags[0] if tags else ""
        latency = f"{r.latency_p50_ms:.0f}ms" if r.latency_p50_ms else "—"
        quality = f"{r.quality_score:.0f}" if r.quality_score else "—"

        table.add_row(
            r.display_name[:22],
            quality,
            latency,
            f"${r.cost_per_1k_requests:.2f}",
            f"${r.cost_per_quality_point:.3f}" if r.cost_per_quality_point else "—",
            best_for,
        )

    console.print(table)
    _print_recommendations(sorted_results)


def _print_recommendations(results: list[BenchmarkResult]) -> None:
    if not results or all(r.quality_score == 0 for r in results):
        return

    scored = [r for r in results if r.quality_score > 0]
    if not scored:
        return

    best_quality = max(scored, key=lambda r: r.quality_score)
    best_value = min(
        [r for r in scored if r.quality_score >= 70],
        key=lambda r: r.cost_per_quality_point,
        default=best_quality,
    )
    best_speed = min(scored, key=lambda r: r.latency_p50_ms or float("inf"))

    console.print("\n[bold]Recommendations:[/bold]")
    console.print(
        f"  → Best value: [green]{best_value.display_name}[/green] "
        f"({best_value.quality_score:.0f}% quality, ${best_value.cost_per_1k_requests:.2f}/1K)"
    )
    console.print(
        f"  → Best speed: [cyan]{best_speed.display_name}[/cyan] "
        f"({best_speed.latency_p50_ms:.0f}ms p50)"
    )
    console.print(
        f"  → Best quality: [magenta]{best_quality.display_name}[/magenta] "
        f"({best_quality.quality_score:.0f}%)"
    )


def print_recommendations(task: str, models: list[ModelPricing], top_n: int = 3) -> None:
    """Recommend cheapest models for a task (pricing-only, no benchmark run)."""
    sorted_models = sorted(models, key=lambda m: m.avg_per_1m)[:top_n]
    console.print(f"\n[bold]Top {top_n} cheapest models for {task}:[/bold]")
    for i, m in enumerate(sorted_models, 1):
        console.print(
            f"  {i}. {m.display_name} ({m.provider.value}) — "
            f"${m.avg_per_1m:.2f}/1M tokens (blended)"
        )
