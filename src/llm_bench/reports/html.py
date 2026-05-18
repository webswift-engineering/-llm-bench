"""Static HTML dashboard for GitHub Pages."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

from llm_bench.benchmark_store import load_all_benchmarks
from llm_bench.models import BenchmarkResult, ModelPricing, utc_now
from llm_bench.pareto import compute_pareto_frontier
from llm_bench.pricing.catalog import CATALOG_VERIFIED, get_catalog
from llm_bench.snapshot import load_latest_snapshot

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
PROVIDER_COLORS = {
    "openai": "#10b981",
    "anthropic": "#f59e0b",
    "groq": "#818cf8",
    "google": "#60a5fa",
    "aws": "#f97316",
    "mistral": "#ec4899",
    "deepseek": "#f43f5e",
}


def _quality_per_dollar(result: BenchmarkResult) -> float:
    if result.cost_per_1k_requests <= 0:
        return 0.0
    return result.quality_score / result.cost_per_1k_requests


def _cached_input_per_1m(model: ModelPricing) -> float | None:
    if model.provider.value == "anthropic":
        return model.input_per_1m * 0.1
    if model.provider.value == "openai":
        return model.input_per_1m * 0.5
    return None


def _format_money(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return ""
    return f"${value:.{decimals}f}"


def _provider_badge(provider: str) -> str:
    color = PROVIDER_COLORS.get(provider, "#94a3b8")
    return (
        f'<span class="provider-badge" style="--provider-color: {color}">'
        f"{escape(provider)}</span>"
    )


def _quality_class(quality: float) -> str:
    if quality >= 95:
        return "quality-good"
    if quality >= 85:
        return "quality-mid"
    return "quality-low"


def _short_name(display_name: str) -> str:
    """Shorten a model name for use as a compact in-chart label (~14 chars)."""
    name = display_name
    if "(" in name and name.endswith(")"):
        name = name[: name.rfind("(")].strip()
    name = (
        name.replace("Claude ", "")
        .replace("GPT OSS ", "OSS ")
        .replace("Llama ", "L")
    )
    return name if len(name) <= 14 else name[:13] + "…"


def _pricing_tier(avg_per_1m: float) -> tuple[str, str, str]:
    """Return (tier_key, label, header_text) for a pricing tier."""
    if avg_per_1m < 1.0:
        return "budget", "Budget", "Budget (< $1 / 1M tokens)"
    if avg_per_1m < 5.0:
        return "mid", "Mid", "Mid ($1 – $5 / 1M tokens)"
    return "premium", "Premium", "Premium (≥ $5 / 1M tokens)"


def _benchmark_picks(results: list[BenchmarkResult]) -> dict[str, BenchmarkResult]:
    scored = [r for r in results if r.quality_score > 0]
    if not scored:
        return {}
    fast_candidates = [r for r in scored if r.quality_score >= 80] or scored
    return {
        "Best Quality": max(scored, key=lambda r: (r.quality_score, -r.cost_per_1k_requests)),
        "Best Value": max(scored, key=_quality_per_dollar),
        "Fastest": min(fast_candidates, key=lambda r: (r.latency_p50_ms, -r.quality_score)),
    }


def _quick_picks_html(results: list[BenchmarkResult]) -> str:
    picks = _benchmark_picks(results)
    if not picks:
        return ""
    stats = {
        "Best Quality": lambda r: f"{r.quality_score:.0f} quality, ${r.cost_per_1k_requests:.2f}/1K",
        "Best Value": lambda r: f"{_quality_per_dollar(r):.1f} quality/$",
        "Fastest": lambda r: f"{r.latency_p50_ms:.0f}ms p50, {r.quality_score:.0f} quality",
    }
    cards = []
    for label, result in picks.items():
        cards.append(
            f"""<article class="pick-card">
              <div class="pick-card__label">{escape(label)}</div>
              <div class="pick-card__model">{escape(result.display_name)}</div>
              <div class="pick-card__provider">{_provider_badge(result.provider.value)}</div>
              <div class="pick-card__stat">{escape(stats[label](result))}</div>
            </article>"""
        )
    return f'<div class="pick-cards" aria-label="Quick picks">{"".join(cards)}</div>'


def _benchmark_chart(results: list[BenchmarkResult]) -> str:
    """Return chart HTML, possibly with a Quality↔Latency view toggle.

    - When all models share the same quality score, render a single horizontal
      cost bar chart (no toggle).
    - Otherwise render a Quality-vs-Cost scatter plus a Latency-vs-Cost scatter,
      with toggle buttons to switch views.
    """
    scored = [r for r in results if r.quality_score > 0]
    if len(scored) < 2:
        return "<p class='muted'>Need at least 2 models with scores for a chart.</p>"
    qualities = {round(r.quality_score, 6) for r in scored}
    if len(qualities) == 1:
        return _cost_bar_svg(scored)
    quality_svg = _scatter_svg(scored)
    latency_svg = _latency_scatter_svg(scored)
    return f"""<div class="chart-toggle" role="tablist" aria-label="Chart view">
        <button type="button" class="chart-toggle-btn is-active" data-view="quality-cost"
                role="tab" aria-selected="true">Quality vs Cost</button>
        <button type="button" class="chart-toggle-btn" data-view="latency-cost"
                role="tab" aria-selected="false">Latency vs Cost</button>
      </div>
      <div class="chart-view" data-view="quality-cost">{quality_svg}</div>
      <div class="chart-view" data-view="latency-cost" hidden>{latency_svg}</div>"""


def _cost_bar_svg(results: list[BenchmarkResult], width: int = 720) -> str:
    sorted_results = sorted(results, key=lambda r: r.cost_per_1k_requests)
    row_height = 30
    margin_left = 190
    margin_right = 110
    margin_top = 28
    margin_bottom = 48
    height = margin_top + margin_bottom + row_height * len(sorted_results)
    costs = [r.cost_per_1k_requests for r in sorted_results]
    max_cost = max(costs) or 1
    median_cost = sorted(costs)[len(costs) // 2]
    plot_width = width - margin_left - margin_right
    plot_x_end = margin_left + plot_width
    cheapest_id = id(sorted_results[0])

    def bar_x(cost: float) -> float:
        return margin_left + (cost / max_cost) * plot_width

    bands = []
    for index in range(len(sorted_results)):
        if index % 2 == 1:
            y = margin_top + index * row_height
            bands.append(
                f'<rect class="bar-band" x="{margin_left - 4}" y="{y:.1f}" '
                f'width="{plot_width + 8}" height="{row_height}" fill="#1e293b" '
                f'opacity="0.35"/>'
            )

    bars = []
    for index, result in enumerate(sorted_results):
        y = margin_top + index * row_height
        bar_width = max(result.cost_per_1k_requests / max_cost * plot_width, 2)
        color = PROVIDER_COLORS.get(result.provider.value, "#94a3b8")
        cheapest_marker = ""
        if id(result) == cheapest_id:
            cheapest_marker = (
                f'<text x="{margin_left + bar_width + 56:.1f}" y="{y + 19}" '
                f'class="cheapest-tag">★ cheapest</text>'
            )
        bars.append(
            f"""<g class="bar-row">
              <text x="{margin_left - 10}" y="{y + 19}" text-anchor="end" class="bar-label">
                {escape(result.display_name)}
              </text>
              <rect class="bar-rect" x="{margin_left}" y="{y + 6}" width="{bar_width:.1f}"
                    height="16" rx="8" fill="{color}">
                <title>{escape(result.display_name)} | {escape(result.provider.value)} |
                  Cost: ${result.cost_per_1k_requests:.2f}/1K | Quality: {result.quality_score:.0f} |
                  Latency: {result.latency_p50_ms:.0f}ms</title>
              </rect>
              <text x="{margin_left + bar_width + 8:.1f}" y="{y + 19}" class="bar-value">
                ${result.cost_per_1k_requests:.2f}
              </text>
              {cheapest_marker}
            </g>"""
        )

    axis_y_top = margin_top
    axis_y_bottom = margin_top + len(sorted_results) * row_height
    median_x = bar_x(median_cost)
    median_line = (
        f'<line class="median-line" x1="{median_x:.1f}" y1="{axis_y_top}" '
        f'x2="{median_x:.1f}" y2="{axis_y_bottom}" stroke="#f0f2f5" stroke-width="1" '
        f'stroke-dasharray="4,3" opacity="0.45"/>'
        f'<text x="{median_x:.1f}" y="{axis_y_top - 8}" text-anchor="middle" '
        f'class="median-label">median ${median_cost:.2f}</text>'
    )
    tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ticks = []
    for frac in tick_fracs:
        tx = margin_left + frac * plot_width
        ticks.append(
            f'<line x1="{tx:.1f}" y1="{axis_y_bottom}" x2="{tx:.1f}" '
            f'y2="{axis_y_bottom + 5}" stroke="#475569"/>'
            f'<text x="{tx:.1f}" y="{axis_y_bottom + 18}" text-anchor="middle" '
            f'class="axis-tick">${frac * max_cost:.2g}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" class="benchmark-chart bar-chart" role="img"
      aria-label="Cost per model sorted cheapest first">
      {"".join(bands)}
      <line x1="{margin_left}" y1="{axis_y_bottom}" x2="{plot_x_end}" y2="{axis_y_bottom}"
            stroke="#363b4a"/>
      {"".join(ticks)}
      {median_line}
      <text x="{margin_left}" y="{height - 8}" class="axis-label">Cost / 1K reqs ($)</text>
      {"".join(bars)}
    </svg>"""


def _place_labels(
    points: list[tuple[float, float, str]],
    chart_width: float,
    min_dy: float = 16.0,
) -> list[tuple[float, float, str, str]]:
    """Simple label-placement that nudges overlapping labels vertically.

    Returns a list of (x, y, text, text_anchor).
    """
    placed: list[tuple[float, float, str, str]] = []
    for x, y, text in sorted(points, key=lambda p: p[1]):
        anchor = "start" if x < chart_width * 0.7 else "end"
        dx = 12 if anchor == "start" else -12
        final_y = y
        for _, py, _, _ in placed:
            if abs(final_y - py) < min_dy:
                final_y = py + min_dy
        placed.append((x + dx, final_y, text, anchor))
    return placed


def _scatter_svg(results: list[BenchmarkResult], width: int = 720, height: int = 460) -> str:
    """Quality (y) vs Cost (x, log scale) scatter with labeled Pareto frontier."""
    import math

    costs = [max(r.cost_per_1k_requests, 0.001) for r in results]
    qualities = [r.quality_score for r in results]
    min_c, max_c = min(costs), max(costs)
    min_q, max_q = min(qualities), max(qualities)
    pad_q = (max_q - min_q) * 0.1 or 5
    min_q -= pad_q
    max_q += pad_q
    min_log = math.log10(min_c)
    max_log = math.log10(max_c)
    if min_log == max_log:
        min_log -= 0.5
        max_log += 0.5

    margin_left = 56
    margin_right = 36
    margin_top = 56
    margin_bottom = 58

    def px(cost: float) -> float:
        value = math.log10(max(cost, 0.001))
        return margin_left + (value - min_log) / (max_log - min_log) * (
            width - margin_left - margin_right
        )

    def py(quality: float) -> float:
        return height - margin_bottom - (quality - min_q) / (max_q - min_q) * (
            height - margin_top - margin_bottom
        )

    frontier_list = compute_pareto_frontier(results)
    frontier = {id(r) for r in frontier_list}
    median_cost = sorted(costs)[len(costs) // 2]
    median_quality = sorted(qualities)[len(qualities) // 2]

    # Horizontal quality gridlines at 25/50/75% of the quality range
    grid_y = []
    q_range = max_q - min_q
    for frac in (0.25, 0.5, 0.75):
        q_val = min_q + frac * q_range
        gy = py(q_val)
        grid_y.append(
            f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{width - margin_right}" '
            f'y2="{gy:.1f}" stroke="#334155" stroke-opacity="0.25" stroke-width="1"/>'
        )

    # Pareto frontier line: faint glow + dashed primary
    sorted_frontier = sorted(frontier_list, key=lambda r: r.cost_per_1k_requests)
    if len(sorted_frontier) >= 2:
        frontier_pts = " ".join(
            f"{px(r.cost_per_1k_requests):.1f},{py(r.quality_score):.1f}"
            for r in sorted_frontier
        )
        frontier_line = (
            f'<polyline class="frontier-glow" points="{frontier_pts}" fill="none" '
            f'stroke="#22d3ee" stroke-width="6" opacity="0.18"/>'
            f'<polyline class="frontier-line" points="{frontier_pts}" fill="none" '
            f'stroke="#22d3ee" stroke-width="2" stroke-dasharray="6,4" opacity="0.85"/>'
        )
    else:
        frontier_line = ""

    dots = []
    label_seeds: list[tuple[float, float, str]] = []
    for result in results:
        is_frontier = id(result) in frontier
        color = PROVIDER_COLORS.get(result.provider.value, "#94a3b8")
        radius = 9 if is_frontier else 6
        opacity = "1" if is_frontier else "0.6"
        classes = "chart-point chart-point--frontier" if is_frontier else "chart-point"
        cx = px(result.cost_per_1k_requests)
        cy = py(result.quality_score)
        dots.append(
            f'<circle class="{classes}" cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" '
            f'fill="{color}" stroke="#0f1117" stroke-width="1.5" opacity="{opacity}" '
            f'data-name="{escape(result.display_name)}" '
            f'data-provider="{escape(result.provider.value)}" '
            f'data-quality="{result.quality_score:.0f}" '
            f'data-cost="{result.cost_per_1k_requests:.2f}" '
            f'data-latency="{result.latency_p50_ms:.0f}">'
            f"<title>{escape(result.display_name)} | {escape(result.provider.value)} | "
            f"Quality: {result.quality_score:.0f} | Cost: ${result.cost_per_1k_requests:.2f}/1K | "
            f"Latency: {result.latency_p50_ms:.0f}ms</title></circle>"
        )
        if is_frontier:
            label_seeds.append((cx, cy, _short_name(result.display_name)))

    labels = []
    for x, y, text, anchor in _place_labels(label_seeds, width):
        labels.append(
            f'<text class="point-label" x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{escape(text)}</text>'
        )

    providers = sorted({r.provider.value for r in results})
    legend = "".join(
        f'<g transform="translate(0,{i * 22})"><circle cx="0" cy="0" r="5" '
        f'fill="{PROVIDER_COLORS.get(provider, "#94a3b8")}"/>'
        f'<text x="12" y="4" class="legend-label">{escape(provider)}</text></g>'
        for i, provider in enumerate(providers)
    )
    ticks = sorted({min_c, (min_c * max_c) ** 0.5, max_c})
    tick_marks = "".join(
        f'<g><line x1="{px(tick):.1f}" y1="{height - margin_bottom:.1f}" '
        f'x2="{px(tick):.1f}" y2="{height - margin_bottom + 5:.1f}" '
        f'stroke="#475569"/><text x="{px(tick):.1f}" y="{height - margin_bottom + 20:.1f}" '
        f'text-anchor="middle" fill="#9ca3b4" font-size="11">${tick:.2g}</text></g>'
        for tick in ticks
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="benchmark-chart scatter-chart" role="img"
      aria-label="Cost vs quality chart using a logarithmic cost scale">
      {"".join(grid_y)}
      <rect x="{margin_left}" y="{margin_top}" width="{px(median_cost) - margin_left:.1f}"
            height="{py(median_quality) - margin_top:.1f}" fill="rgba(16, 185, 129, 0.08)"/>
      <line x1="{px(median_cost):.1f}" y1="{margin_top}" x2="{px(median_cost):.1f}"
            y2="{height - margin_bottom}" class="guide-line"/>
      <line x1="{margin_left}" y1="{py(median_quality):.1f}" x2="{width - margin_right}"
            y2="{py(median_quality):.1f}" class="guide-line"/>
      <text x="{margin_left + 12}" y="{margin_top + 18}" class="sweet-spot-label">Sweet spot</text>
      <text x="{margin_left + 8}" y="{margin_top - 12}" class="quadrant-hint">
        ← Higher quality, lower cost</text>
      <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}"
            y2="{height - margin_bottom}" stroke="#363b4a"/>
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}"
            y2="{height - margin_bottom}" stroke="#363b4a"/>
      {tick_marks}
      <text x="{width // 2}" y="{height - 8}" text-anchor="middle" fill="#9ca3b4" font-size="12">
        Cost / 1K reqs ($, log scale)</text>
      <text x="16" y="{height // 2}" text-anchor="middle" fill="#9ca3b4" font-size="12"
            transform="rotate(-90 16 {height // 2})">Quality</text>
      {frontier_line}
      {"".join(dots)}
      {"".join(labels)}
      <g class="chart-legend" transform="translate({width - 150}, {margin_top + 8})">{legend}</g>
    </svg>"""


def _latency_scatter_svg(
    results: list[BenchmarkResult], width: int = 720, height: int = 460
) -> str:
    """Latency (y, inverted so 'fast & cheap' sits top-left) vs Cost (x, log scale)."""
    import math

    costs = [max(r.cost_per_1k_requests, 0.001) for r in results]
    latencies = [max(r.latency_p50_ms, 1.0) for r in results]
    min_c, max_c = min(costs), max(costs)
    min_l, max_l = min(latencies), max(latencies)
    pad_l = (max_l - min_l) * 0.1 or 50
    min_l -= pad_l
    max_l += pad_l
    if min_l < 0:
        min_l = 0
    min_log = math.log10(min_c)
    max_log = math.log10(max_c)
    if min_log == max_log:
        min_log -= 0.5
        max_log += 0.5

    margin_left = 56
    margin_right = 36
    margin_top = 56
    margin_bottom = 58

    def px(cost: float) -> float:
        value = math.log10(max(cost, 0.001))
        return margin_left + (value - min_log) / (max_log - min_log) * (
            width - margin_left - margin_right
        )

    def py(latency: float) -> float:
        # Lower latency = higher on chart (good is up)
        return margin_top + (latency - min_l) / (max_l - min_l) * (
            height - margin_top - margin_bottom
        )

    # Pareto frontier for cost vs latency (both lower is better)
    sorted_by_cost = sorted(results, key=lambda r: r.cost_per_1k_requests)
    front_ids: set[int] = set()
    best_lat = float("inf")
    front_seq: list[BenchmarkResult] = []
    for r in sorted_by_cost:
        if r.latency_p50_ms < best_lat:
            best_lat = r.latency_p50_ms
            front_ids.add(id(r))
            front_seq.append(r)

    grid_y_lines = []
    l_range = max_l - min_l
    for frac in (0.25, 0.5, 0.75):
        l_val = min_l + frac * l_range
        gy = py(l_val)
        grid_y_lines.append(
            f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{width - margin_right}" '
            f'y2="{gy:.1f}" stroke="#334155" stroke-opacity="0.25" stroke-width="1"/>'
        )

    if len(front_seq) >= 2:
        pts = " ".join(
            f"{px(r.cost_per_1k_requests):.1f},{py(r.latency_p50_ms):.1f}" for r in front_seq
        )
        frontier_line = (
            f'<polyline class="frontier-glow" points="{pts}" fill="none" '
            f'stroke="#22d3ee" stroke-width="6" opacity="0.18"/>'
            f'<polyline class="frontier-line" points="{pts}" fill="none" '
            f'stroke="#22d3ee" stroke-width="2" stroke-dasharray="6,4" opacity="0.85"/>'
        )
    else:
        frontier_line = ""

    dots = []
    label_seeds: list[tuple[float, float, str]] = []
    for result in results:
        is_frontier = id(result) in front_ids
        color = PROVIDER_COLORS.get(result.provider.value, "#94a3b8")
        radius = 9 if is_frontier else 6
        opacity = "1" if is_frontier else "0.6"
        classes = "chart-point chart-point--frontier" if is_frontier else "chart-point"
        cx = px(result.cost_per_1k_requests)
        cy = py(result.latency_p50_ms)
        dots.append(
            f'<circle class="{classes}" cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" '
            f'fill="{color}" stroke="#0f1117" stroke-width="1.5" opacity="{opacity}" '
            f'data-name="{escape(result.display_name)}" '
            f'data-provider="{escape(result.provider.value)}" '
            f'data-quality="{result.quality_score:.0f}" '
            f'data-cost="{result.cost_per_1k_requests:.2f}" '
            f'data-latency="{result.latency_p50_ms:.0f}">'
            f"<title>{escape(result.display_name)} | {escape(result.provider.value)} | "
            f"Latency: {result.latency_p50_ms:.0f}ms | Cost: ${result.cost_per_1k_requests:.2f}/1K | "
            f"Quality: {result.quality_score:.0f}</title></circle>"
        )
        if is_frontier:
            label_seeds.append((cx, cy, _short_name(result.display_name)))

    labels = []
    for x, y, text, anchor in _place_labels(label_seeds, width):
        labels.append(
            f'<text class="point-label" x="{x:.1f}" y="{y:.1f}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{escape(text)}</text>'
        )

    providers = sorted({r.provider.value for r in results})
    legend = "".join(
        f'<g transform="translate(0,{i * 22})"><circle cx="0" cy="0" r="5" '
        f'fill="{PROVIDER_COLORS.get(provider, "#94a3b8")}"/>'
        f'<text x="12" y="4" class="legend-label">{escape(provider)}</text></g>'
        for i, provider in enumerate(providers)
    )
    ticks = sorted({min_c, (min_c * max_c) ** 0.5, max_c})
    tick_marks = "".join(
        f'<g><line x1="{px(tick):.1f}" y1="{height - margin_bottom:.1f}" '
        f'x2="{px(tick):.1f}" y2="{height - margin_bottom + 5:.1f}" '
        f'stroke="#475569"/><text x="{px(tick):.1f}" y="{height - margin_bottom + 20:.1f}" '
        f'text-anchor="middle" fill="#9ca3b4" font-size="11">${tick:.2g}</text></g>'
        for tick in ticks
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="benchmark-chart scatter-chart" role="img"
      aria-label="Cost vs latency chart using a logarithmic cost scale">
      {"".join(grid_y_lines)}
      <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}"
            y2="{height - margin_bottom}" stroke="#363b4a"/>
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}"
            y2="{height - margin_bottom}" stroke="#363b4a"/>
      <text x="{margin_left + 8}" y="{margin_top - 12}" class="quadrant-hint">
        ← Faster &amp; cheaper</text>
      {tick_marks}
      <text x="{width // 2}" y="{height - 8}" text-anchor="middle" fill="#9ca3b4" font-size="12">
        Cost / 1K reqs ($, log scale)</text>
      <text x="16" y="{height // 2}" text-anchor="middle" fill="#9ca3b4" font-size="12"
            transform="rotate(-90 16 {height // 2})">Latency p50 (ms, lower is better)</text>
      {frontier_line}
      {"".join(dots)}
      {"".join(labels)}
      <g class="chart-legend" transform="translate({width - 150}, {margin_top + 8})">{legend}</g>
    </svg>"""


def _scatter_svg(results: list[BenchmarkResult], width: int = 720, height: int = 460) -> str:
    import math

    costs = [max(r.cost_per_1k_requests, 0.001) for r in results]
    qualities = [r.quality_score for r in results]
    min_c, max_c = min(costs), max(costs)
    min_q, max_q = min(qualities), max(qualities)
    pad_q = (max_q - min_q) * 0.1 or 5
    min_q -= pad_q
    max_q += pad_q
    min_log = math.log10(min_c)
    max_log = math.log10(max_c)
    if min_log == max_log:
        min_log -= 0.5
        max_log += 0.5

    margin_left = 56
    margin_right = 36
    margin_top = 48
    margin_bottom = 58

    def px(cost: float) -> float:
        value = math.log10(max(cost, 0.001))
        return margin_left + (value - min_log) / (max_log - min_log) * (
            width - margin_left - margin_right
        )

    def py(quality: float) -> float:
        return height - margin_bottom - (quality - min_q) / (max_q - min_q) * (
            height - margin_top - margin_bottom
        )

    frontier = {id(r) for r in compute_pareto_frontier(results)}
    cheapest = min(results, key=lambda r: (r.cost_per_1k_requests, -r.quality_score))
    highest_quality = max(results, key=lambda r: (r.quality_score, -r.cost_per_1k_requests))
    best_value = max(results, key=_quality_per_dollar)
    labels: dict[int, list[str]] = {}
    labels.setdefault(id(cheapest), []).append("Cheapest")
    labels.setdefault(id(highest_quality), []).append("Best quality")
    labels.setdefault(id(best_value), []).append("Best value")
    median_cost = sorted(costs)[len(costs) // 2]
    median_quality = sorted(qualities)[len(qualities) // 2]

    dots = []
    for result in results:
        color = PROVIDER_COLORS.get(result.provider.value, "#94a3b8")
        classes = "chart-point chart-point--frontier" if id(result) in frontier else "chart-point"
        dots.append(
            f'<circle class="{classes}" cx="{px(result.cost_per_1k_requests):.1f}" '
            f'cy="{py(result.quality_score):.1f}" r="10" fill="{color}" '
            f'stroke="#0f1117" stroke-width="1.5">'
            f"<title>{escape(result.display_name)} | {escape(result.provider.value)} | "
            f"Quality: {result.quality_score:.0f} | Cost: ${result.cost_per_1k_requests:.2f}/1K | "
            f"Latency: {result.latency_p50_ms:.0f}ms</title></circle>"
        )
        if id(result) in labels:
            label_text = " + ".join(labels[id(result)])
            x = px(result.cost_per_1k_requests)
            y = py(result.quality_score)
            dx = 12 if x < width - 220 else -12
            anchor = "start" if dx > 0 else "end"
            dots.append(
                f'<text class="point-label" x="{x + dx:.1f}" y="{y - 16:.1f}" '
                f'text-anchor="{anchor}">{escape(label_text)}: {escape(result.display_name)}</text>'
            )

    providers = sorted({r.provider.value for r in results})
    legend = "".join(
        f'<g transform="translate(0,{i * 22})"><circle cx="0" cy="0" r="5" '
        f'fill="{PROVIDER_COLORS.get(provider, "#94a3b8")}"/>'
        f'<text x="12" y="4" class="legend-label">{escape(provider)}</text></g>'
        for i, provider in enumerate(providers)
    )
    ticks = sorted({min_c, (min_c * max_c) ** 0.5, max_c})
    tick_marks = "".join(
        f'<g><line x1="{px(tick):.1f}" y1="{height - margin_bottom:.1f}" '
        f'x2="{px(tick):.1f}" y2="{height - margin_bottom + 5:.1f}" '
        f'stroke="#475569"/><text x="{px(tick):.1f}" y="{height - margin_bottom + 20:.1f}" '
        f'text-anchor="middle" fill="#9ca3b4" font-size="11">${tick:.2g}</text></g>'
        for tick in ticks
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="benchmark-chart scatter-chart" role="img"
      aria-label="Cost vs quality chart using a logarithmic cost scale">
      <rect x="{margin_left}" y="{margin_top}" width="{px(median_cost) - margin_left:.1f}"
            height="{py(median_quality) - margin_top:.1f}" fill="rgba(16, 185, 129, 0.08)"/>
      <line x1="{px(median_cost):.1f}" y1="{margin_top}" x2="{px(median_cost):.1f}"
            y2="{height - margin_bottom}" class="guide-line"/>
      <line x1="{margin_left}" y1="{py(median_quality):.1f}" x2="{width - margin_right}"
            y2="{py(median_quality):.1f}" class="guide-line"/>
      <text x="{margin_left + 12}" y="{margin_top + 18}" class="sweet-spot-label">Sweet spot</text>
      <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}"
            y2="{height - margin_bottom}" stroke="#363b4a"/>
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}"
            y2="{height - margin_bottom}" stroke="#363b4a"/>
      {tick_marks}
      <text x="{width // 2}" y="{height - 8}" text-anchor="middle" fill="#9ca3b4" font-size="12">
        Cost / 1K reqs ($, log scale)</text>
      <text x="16" y="{height // 2}" text-anchor="middle" fill="#9ca3b4" font-size="12"
            transform="rotate(-90 16 {height // 2})">Quality</text>
      <g class="chart-legend" transform="translate({width - 150}, {margin_top + 8})">{legend}</g>
      {"".join(dots)}
    </svg>"""


def _pricing_rows(models: list[ModelPricing]) -> str:
    rows: list[str] = []
    last_tier: str | None = None
    for m in sorted(models, key=lambda x: x.avg_per_1m):
        tier_key, _, tier_header = _pricing_tier(m.avg_per_1m)
        if tier_key != last_tier:
            rows.append(
                f"<tr class='pricing-tier-header pricing-tier--{tier_key}'>"
                f"<td colspan='8'>{escape(tier_header)}</td></tr>"
            )
            last_tier = tier_key
        cached = _cached_input_per_1m(m)
        source = (
            f"<a href='{escape(m.source_url)}' target='_blank' rel='noopener'>source</a>"
            if m.source_url
            else ""
        )
        rows.append(
            f"<tr class='pricing-row pricing-tier--{tier_key}'>"
            f"<td>{escape(m.display_name)}</td>"
            f"<td>{_provider_badge(m.provider.value)}</td>"
            f"<td class='highlight'>${m.avg_per_1m:.3f}</td>"
            f"<td class='muted'>{source}</td>"
            f"<td class='pricing-detail'>${m.input_per_1m:.3f}</td>"
            f"<td class='pricing-detail'>${m.output_per_1m:.3f}</td>"
            f"<td class='pricing-detail'>{_format_money(cached)}</td>"
            f"<td class='pricing-detail'>${m.avg_per_1m * 0.5:.3f}</td></tr>"
        )
    return "\n".join(rows)


def _average_quality_by_model(benchmarks: dict[str, list[BenchmarkResult]]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for results in benchmarks.values():
        for result in results:
            if result.quality_score > 0:
                scores.setdefault(result.model_id, []).append(result.quality_score)
    return {model_id: sum(values) / len(values) for model_id, values in scores.items()}


def _calculator_rows(models: list[ModelPricing], average_quality: dict[str, float]) -> str:
    rows = []
    for m in sorted(models, key=lambda x: (5000 * x.input_per_1m + 2000 * x.output_per_1m)):
        monthly = 100 * 30 * (
            (5000 * m.input_per_1m / 1_000_000) + (2000 * m.output_per_1m / 1_000_000)
        )
        quality = average_quality.get(m.model_id)
        quality_value = f"{quality:.1f}" if quality is not None else ""
        quality_class = _quality_class(quality) if quality is not None else ""
        rows.append(
            f"<tr data-provider='{escape(m.provider.value)}' "
            f"data-input-price='{m.input_per_1m:.6f}' "
            f"data-output-price='{m.output_per_1m:.6f}' "
            f"data-quality='{quality or 0:.6f}'>"
            f"<td>{escape(m.display_name)} <span class='recommended-badge'>Recommended</span></td>"
            f"<td>{_provider_badge(m.provider.value)}</td>"
            f"<td class='calculator-cost'>${monthly:.2f}</td>"
            f"<td class='{quality_class}'>{quality_value}</td></tr>"
        )
    return "\n".join(rows)


def _benchmark_table(results: list[BenchmarkResult], table_id: str) -> str:
    if not results:
        return "<p class='muted'>No results yet. Run <code>llm-bench run --save</code> locally.</p>"
    sorted_results = sorted(results, key=_quality_per_dollar, reverse=True)
    providers = sorted({r.provider.value for r in sorted_results})
    min_cost = min(r.cost_per_1k_requests for r in sorted_results)
    max_cost = max(r.cost_per_1k_requests for r in sorted_results)
    provider_chips = "".join(
        f"<button type='button' class='chip is-active' data-provider='{escape(provider)}'>"
        f"{escape(provider)}</button>"
        for provider in providers
    )
    rows = []
    visible_initial = 8
    for index, r in enumerate(sorted_results):
        tags = ", ".join(r.tags) if r.tags else ""
        value = _quality_per_dollar(r)
        quality_class = _quality_class(r.quality_score)
        hidden_class = " bench-row-hidden" if index >= visible_initial else ""
        title = (
            f"{r.display_name}: Quality {r.quality_score:.0f}, "
            f"Latency {r.latency_p50_ms:.0f}ms, "
            f"Cost ${r.cost_per_1k_requests:.2f}/1K"
        )
        rows.append(
            f"<tr class='bench-row{hidden_class}' data-default-index='{index}' "
            f"data-provider='{escape(r.provider.value)}' "
            f"data-cost='{r.cost_per_1k_requests:.6f}' "
            f"data-quality='{r.quality_score:.6f}' title='{escape(title)}'>"
            f"<td data-value='{escape(r.display_name.lower())}'>{escape(r.display_name)}</td>"
            f"<td data-value='{escape(r.provider.value)}'>{_provider_badge(r.provider.value)}</td>"
            f"<td class='{quality_class}' data-value='{r.quality_score:.6f}'>{r.quality_score:.0f}</td>"
            f"<td data-value='{r.latency_p50_ms:.6f}'>{r.latency_p50_ms:.0f}ms</td>"
            f"<td data-value='{r.cost_per_1k_requests:.6f}'>${r.cost_per_1k_requests:.2f}</td>"
            f"<td data-value='{value:.6f}'>{value:.1f}</td>"
            f"<td data-value='{escape(tags.lower())}'>{escape(tags)}</td></tr>"
        )

    hidden_count = max(0, len(sorted_results) - visible_initial)
    show_all_button = (
        f'<button type="button" class="show-all-btn" data-hidden-count="{hidden_count}">'
        f"Show all {len(sorted_results)} models ▾</button>"
        if hidden_count > 0
        else ""
    )
    return f"""<div class="benchmark-card bench-section" data-table-id="{escape(table_id)}">
      {_quick_picks_html(results)}
      <div class="filter-bar" aria-label="Filters for {escape(table_id)} benchmark">
        <div class="filter-group">
          <span class="filter-label">Provider</span>
          <div class="chip-group">{provider_chips}</div>
        </div>
        <label class="filter-field">Min cost
          <input type="number" class="filter-min-cost" value="{min_cost:.2f}" min="0" step="0.01"/>
        </label>
        <label class="filter-field">Max cost
          <input type="number" class="filter-max-cost" value="{max_cost:.2f}" min="0" step="0.01"/>
        </label>
        <label class="filter-field filter-field--wide">
          <span>Quality ≥ <output class="quality-output">0</output></span>
          <input type="range" class="filter-min-quality" value="0" min="0" max="100" step="1"/>
        </label>
      </div>
      <div class="table-wrap"><table class="benchmark-table">
      <thead><tr>
        <th><button type="button" data-sort="model" data-type="text">Model</button></th>
        <th><button type="button" data-sort="provider" data-type="text">Provider</button></th>
        <th><button type="button" data-sort="quality" data-type="number">Quality</button></th>
        <th><button type="button" data-sort="latency" data-type="number">Latency p50</button></th>
        <th><button type="button" data-sort="cost" data-type="number">Cost / 1K reqs</button></th>
        <th><button type="button" data-sort="value" data-type="number" data-default-sort="desc">
          Quality / $</button></th>
        <th><button type="button" data-sort="tags" data-type="text">Tags</button></th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>{show_all_button}</div>"""


def _ga4_script() -> str:
    measurement_id = os.environ.get("GA_MEASUREMENT_ID", "").strip()
    if not measurement_id:
        return ""
    safe_id = escape(measurement_id)
    return f"""  <script async src="https://www.googletagmanager.com/gtag/js?id={safe_id}"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());
    gtag('config', '{safe_id}');
  </script>
"""


def generate_dashboard(output_dir: Path | None = None) -> Path:
    out = output_dir or DOCS_DIR
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    models = get_catalog()
    snapshot = load_latest_snapshot()
    snapshot_date = ""
    if snapshot:
        models = snapshot.models
        snapshot_date = snapshot.captured_at.strftime("%Y-%m-%d")

    benchmarks = load_all_benchmarks()
    average_quality = _average_quality_by_model(benchmarks)
    updated = utc_now().strftime("%Y-%m-%d %H:%M UTC")

    task_sections = []
    for task, results in benchmarks.items():
        task_id = f"task-{task}"
        task_sections.append(
            f"""<section class="card" id="{escape(task_id)}">
              <h2>{escape(task.replace("_", " ").title())}</h2>
              {_benchmark_table(results, task)}
              <div class="chart-wrap">{_benchmark_chart(results)}</div>
            </section>"""
        )

    benchmarks_json = {
        task: [
            {
                "model_id": r.model_id,
                "display_name": r.display_name,
                "provider": r.provider.value,
                "quality_score": r.quality_score,
                "latency_p50_ms": r.latency_p50_ms,
                "cost_per_1k_requests": r.cost_per_1k_requests,
            }
            for r in results
        ]
        for task, results in benchmarks.items()
    }

    prices_json = [
        {
            "provider": m.provider.value,
            "model_id": m.model_id,
            "display_name": m.display_name,
            "input_per_1m": m.input_per_1m,
            "output_per_1m": m.output_per_1m,
            "cached_input_per_1m": (
                round(cached, 3) if (cached := _cached_input_per_1m(m)) is not None else None
            ),
            "avg_per_1m": round(m.avg_per_1m, 3),
            "batch_avg_per_1m": round(m.avg_per_1m * 0.5, 3),
        }
        for m in models
    ]
    (data_dir / "prices.json").write_text(json.dumps(prices_json, indent=2), encoding="utf-8")
    (data_dir / "benchmarks.json").write_text(json.dumps(benchmarks_json, indent=2), encoding="utf-8")

    tasks_html = (
        "\n".join(task_sections)
        if task_sections
        else "<p class='muted'>No benchmark data yet. Run benchmarks locally and commit "
        "<code>data/benchmarks/*.json</code>.</p>"
    )
    task_nav_links = "\n".join(
        f'<a href="#task-{escape(task)}">{escape(task.replace("_", " ").title())}</a>'
        for task in benchmarks
    )
    dashboard_script = """
  <div id="chart-tooltip" class="chart-tooltip" role="tooltip" aria-hidden="true"></div>
  <script>
    (() => {
      const sortIndexes = {
        model: 0,
        provider: 1,
        quality: 2,
        latency: 3,
        cost: 4,
        value: 5,
        tags: 6,
      };

      const numberValue = (value) => Number.parseFloat(value) || 0;

      function cellValue(row, key) {
        const cell = row.cells[sortIndexes[key]];
        return cell ? cell.dataset.value || cell.textContent.trim() : "";
      }

      function setSortLabels(card, sortKey, direction) {
        card.querySelectorAll("th button[data-sort]").forEach((button) => {
          button.dataset.label ||= button.textContent.trim();
          button.classList.toggle("is-sorted", button.dataset.sort === sortKey);
          const arrow = direction === "asc" ? " ▲" : " ▼";
          button.textContent = button.dataset.label + (button.dataset.sort === sortKey ? arrow : "");
        });
      }

      function initBenchmark(card) {
        const tbody = card.querySelector("tbody");
        const rows = Array.from(tbody.rows);
        const chips = Array.from(card.querySelectorAll(".chip"));
        const minCost = card.querySelector(".filter-min-cost");
        const maxCost = card.querySelector(".filter-max-cost");
        const minQuality = card.querySelector(".filter-min-quality");
        const qualityOutput = card.querySelector(".quality-output");
        const showAllBtn = card.querySelector(".show-all-btn");
        let sortKey = "value";
        let direction = "desc";
        let truncated = !!showAllBtn;

        function applyTruncation() {
          rows.forEach((row) => {
            const initiallyHidden = row.classList.contains("bench-row-hidden");
            row.dataset.truncated = (truncated && initiallyHidden) ? "1" : "0";
          });
        }

        function render() {
          const sorted = [...rows].sort((a, b) => {
            if (!sortKey) {
              return numberValue(a.dataset.defaultIndex) - numberValue(b.dataset.defaultIndex);
            }
            const type = card.querySelector(`[data-sort="${sortKey}"]`).dataset.type;
            const aValue = cellValue(a, sortKey);
            const bValue = cellValue(b, sortKey);
            const result = type === "number"
              ? numberValue(aValue) - numberValue(bValue)
              : aValue.localeCompare(bValue);
            return direction === "asc" ? result : -result;
          });
          sorted.forEach((row) => tbody.appendChild(row));
          const activeProviders = new Set(
            chips.filter((chip) => chip.classList.contains("is-active")).map((chip) => chip.dataset.provider)
          );
          const minCostValue = numberValue(minCost.value);
          const maxCostValue = numberValue(maxCost.value) || Number.POSITIVE_INFINITY;
          const minQualityValue = numberValue(minQuality.value);
          qualityOutput.value = minQuality.value;
          applyTruncation();
          rows.forEach((row) => {
            const matchesFilter = activeProviders.has(row.dataset.provider)
              && numberValue(row.dataset.cost) >= minCostValue
              && numberValue(row.dataset.cost) <= maxCostValue
              && numberValue(row.dataset.quality) >= minQualityValue;
            const isTruncated = row.dataset.truncated === "1";
            row.hidden = !matchesFilter || isTruncated;
          });
          setSortLabels(card, sortKey, direction);
        }

        card.querySelectorAll("th button[data-sort]").forEach((button) => {
          button.addEventListener("click", () => {
            const nextKey = button.dataset.sort;
            if (sortKey !== nextKey) {
              sortKey = nextKey;
              direction = "asc";
            } else if (direction === "asc") {
              direction = "desc";
            } else {
              sortKey = null;
              direction = "asc";
            }
            render();
          });
        });
        chips.forEach((chip) => chip.addEventListener("click", () => {
          chip.classList.toggle("is-active");
          render();
        }));
        [minCost, maxCost, minQuality].forEach((input) => input.addEventListener("input", render));
        if (showAllBtn) {
          showAllBtn.addEventListener("click", () => {
            truncated = false;
            showAllBtn.remove();
            render();
          });
        }
        render();
      }

      function flashCell(cell) {
        cell.classList.remove("cell-changed");
        // Force reflow so the animation restarts even on rapid edits.
        void cell.offsetWidth;
        cell.classList.add("cell-changed");
      }

      function initCalculator() {
        const calculator = document.querySelector(".calculator");
        if (!calculator) return;
        const tasks = calculator.querySelector("#tasks-per-day");
        const inputTokens = calculator.querySelector("#input-tokens");
        const outputTokens = calculator.querySelector("#output-tokens");
        const tbody = calculator.querySelector("tbody");
        const rows = Array.from(tbody.rows);
        function render() {
          const taskCount = numberValue(tasks.value);
          const inputCount = numberValue(inputTokens.value);
          const outputCount = numberValue(outputTokens.value);
          const monthlies = [];
          rows.forEach((row) => {
            const monthly = taskCount * 30 * (
              inputCount * numberValue(row.dataset.inputPrice) / 1000000
              + outputCount * numberValue(row.dataset.outputPrice) / 1000000
            );
            const previous = numberValue(row.dataset.monthlyCost);
            row.dataset.monthlyCost = monthly.toFixed(6);
            const costCell = row.querySelector(".calculator-cost");
            const next = `$${monthly.toFixed(2)}`;
            if (costCell.textContent !== next) {
              costCell.textContent = next;
              if (previous !== 0) flashCell(costCell);
            }
            row.classList.remove("is-recommended");
            monthlies.push(monthly);
          });
          // "Recommended" = quality >= 90 AND monthly cost in bottom 30%
          // among models with a quality score. Multiple rows can qualify.
          const ranked = [...monthlies].sort((a, b) => a - b);
          const threshold = ranked[Math.max(0, Math.floor(ranked.length * 0.3) - 1)] ?? Infinity;
          rows.forEach((row, idx) => {
            const quality = numberValue(row.dataset.quality);
            if (quality >= 90 && monthlies[idx] <= threshold) {
              row.classList.add("is-recommended");
            }
          });
          rows
            .sort((a, b) => numberValue(a.dataset.monthlyCost) - numberValue(b.dataset.monthlyCost))
            .forEach((row) => tbody.appendChild(row));
        }
        [tasks, inputTokens, outputTokens].forEach((input) =>
          input.addEventListener("input", render)
        );
        render();
      }

      function initPricingDetails() {
        const button = document.querySelector(".pricing-toggle");
        const table = document.querySelector(".pricing-table");
        if (!button || !table) return;
        button.addEventListener("click", () => {
          const expanded = table.classList.toggle("show-details");
          button.setAttribute("aria-expanded", String(expanded));
          button.textContent = expanded ? "Hide details" : "Show details";
        });
      }

      function initChartToggles() {
        document.querySelectorAll(".chart-wrap").forEach((wrap) => {
          const buttons = wrap.querySelectorAll(".chart-toggle-btn");
          const views = wrap.querySelectorAll(".chart-view");
          if (!buttons.length) return;
          buttons.forEach((btn) => {
            btn.addEventListener("click", () => {
              const target = btn.dataset.view;
              buttons.forEach((b) => {
                const active = b === btn;
                b.classList.toggle("is-active", active);
                b.setAttribute("aria-selected", String(active));
              });
              views.forEach((v) => {
                v.hidden = v.dataset.view !== target;
              });
            });
          });
        });
      }

      function initChartTooltip() {
        const tip = document.getElementById("chart-tooltip");
        if (!tip) return;
        document.querySelectorAll(".chart-point").forEach((dot) => {
          dot.addEventListener("mouseenter", () => {
            const name = dot.dataset.name || "";
            const provider = dot.dataset.provider || "";
            const quality = dot.dataset.quality || "";
            const cost = dot.dataset.cost || "";
            const latency = dot.dataset.latency || "";
            tip.innerHTML =
              `<strong>${name}</strong>` +
              `<span class="tt-row">Provider: <em>${provider}</em></span>` +
              `<span class="tt-row">Quality: <em>${quality}</em></span>` +
              `<span class="tt-row">Cost: <em>$${cost}/1K</em></span>` +
              `<span class="tt-row">Latency p50: <em>${latency}ms</em></span>`;
            tip.style.display = "block";
            tip.setAttribute("aria-hidden", "false");
            const rect = dot.getBoundingClientRect();
            const tipRect = tip.getBoundingClientRect();
            let left = rect.left + window.scrollX + 14;
            let top = rect.top + window.scrollY - tipRect.height - 12;
            if (top < window.scrollY + 8) top = rect.bottom + window.scrollY + 12;
            if (left + tipRect.width > window.scrollX + window.innerWidth - 8) {
              left = window.scrollX + window.innerWidth - tipRect.width - 8;
            }
            tip.style.left = left + "px";
            tip.style.top = top + "px";
          });
          dot.addEventListener("mouseleave", () => {
            tip.style.display = "none";
            tip.setAttribute("aria-hidden", "true");
          });
        });
      }

      function initScrollSpy() {
        const links = Array.from(document.querySelectorAll(".top-nav a[href^='#']"));
        const sections = links
          .map((link) => document.querySelector(link.getAttribute("href")))
          .filter(Boolean);
        const observer = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            links.forEach((link) => {
              link.classList.toggle("is-active", link.getAttribute("href") === `#${entry.target.id}`);
            });
          });
        }, { rootMargin: "-25% 0px -60% 0px" });
        sections.forEach((section) => observer.observe(section));
      }

      document.querySelectorAll(".benchmark-card").forEach(initBenchmark);
      initCalculator();
      initPricingDetails();
      initChartToggles();
      initChartTooltip();
      initScrollSpy();
    })();
  </script>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>llm-bench — LLM Cost &amp; Quality Dashboard</title>
{_ga4_script()}
  <style>
    :root {{
      --bg-base: #0f1117; --bg-surface: #1a1d27; --bg-elevated: #242836;
      --bg-input: #2a2e3b; --text-primary: #f0f2f5; --text-secondary: #9ca3b4;
      --text-muted: #5c6478; --color-openai: #10b981; --color-anthropic: #f59e0b;
      --color-groq: #818cf8; --color-google: #60a5fa; --color-aws: #f97316;
      --color-accent: #6366f1; --color-success: #10b981; --color-warning: #f59e0b;
      --color-danger: #ef4444; --border-subtle: #2a2e3b; --border-default: #363b4a;
      --text-xs: 0.75rem; --text-sm: 0.875rem; --text-base: 1rem; --text-lg: 1.25rem;
      --text-xl: 1.5rem; --text-2xl: 2rem; --font-normal: 400; --font-medium: 500;
      --font-semibold: 600; --font-bold: 700; --space-1: 4px; --space-2: 8px;
      --space-3: 12px; --space-4: 16px; --space-5: 24px; --space-6: 32px;
      --space-7: 48px; --space-8: 64px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
      background: var(--bg-base); color: var(--text-primary); margin: 0; line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: var(--space-6) var(--space-6) var(--space-8); }}
    h1 {{ font-size: var(--text-2xl); font-weight: var(--font-bold); margin: 0 0 var(--space-1); }}
    .lede {{ color: var(--text-secondary); margin: 0 0 var(--space-5); }}
    .meta {{ font-size: var(--text-sm); color: var(--text-secondary); margin-bottom: var(--space-5); }}
    .top-nav {{
      position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: var(--space-2);
      margin: 0 0 var(--space-6); padding: var(--space-2) 0;
      background: color-mix(in srgb, var(--bg-base) 90%, transparent);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
    }}
    .top-nav a {{
      color: var(--text-secondary); padding: var(--space-2) var(--space-4);
      text-decoration: none; font-size: var(--text-sm); font-weight: var(--font-medium);
      border-bottom: 2px solid transparent; transition: color 0.15s ease, border-color 0.15s ease;
    }}
    .top-nav a:hover, .top-nav a.is-active {{
      color: var(--text-primary); border-bottom-color: var(--color-accent);
    }}
    .card {{
      background: var(--bg-surface); border: 1px solid var(--border-subtle);
      border-radius: 12px; padding: var(--space-5); margin-bottom: var(--space-8);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }}
    h2 {{ font-size: var(--text-lg); margin: 0 0 var(--space-4); color: var(--text-primary); font-weight: var(--font-semibold); }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: var(--text-sm); }}
    th, td {{
      text-align: left; padding: var(--space-3) var(--space-4);
      border-bottom: 1px solid var(--border-subtle); font-variant-numeric: tabular-nums;
    }}
    th {{
      color: var(--text-secondary); font-size: var(--text-xs); font-weight: var(--font-medium);
      text-transform: uppercase; letter-spacing: 0.05em; border-bottom-color: var(--border-default);
    }}
    tbody tr:hover td {{ background: var(--bg-elevated); }}
    tr[hidden] {{ display: none; }}
    th button {{
      appearance: none; border: 0; background: transparent; color: inherit; cursor: pointer;
      font: inherit; padding: 0; text-align: left; white-space: nowrap;
    }}
    th button:hover, th button.is-sorted {{ color: var(--text-primary); }}
    button, input, a {{ transition: 0.15s ease; }}
    button:focus-visible, input:focus-visible, a:focus-visible {{
      outline: 2px solid var(--color-accent); outline-offset: 2px;
    }}
    .table-wrap {{ overflow-x: auto; }}
    .highlight, .quality-good {{ color: var(--color-success); font-weight: var(--font-semibold); }}
    .quality-mid {{ color: var(--color-warning); font-weight: var(--font-semibold); }}
    .quality-low {{ color: var(--color-danger); font-weight: var(--font-semibold); }}
    .muted {{ color: var(--text-secondary); }}
    code {{ background: var(--bg-input); padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }}
    .chart-wrap {{
      background: var(--bg-surface); border: 1px solid var(--border-subtle);
      border-radius: 12px; padding: var(--space-5); margin-top: var(--space-5);
      min-height: 400px; overflow-x: auto;
    }}
    .benchmark-chart {{ width: 100%; max-width: 720px; height: auto; }}
    .chart-point {{ transition: r 0.15s ease, opacity 0.15s ease; cursor: pointer; }}
    .chart-point:hover {{ r: 12; opacity: 1 !important; }}
    .chart-point--frontier {{ filter: drop-shadow(0 0 7px rgba(34, 211, 238, 0.55)); stroke: #f0f2f5; stroke-width: 2.5; }}
    .guide-line {{ stroke: var(--border-default); stroke-dasharray: 6 6; }}
    .frontier-line {{ pointer-events: none; }}
    .frontier-glow {{ pointer-events: none; }}
    .point-label, .legend-label, .bar-label, .bar-value, .axis-label, .axis-tick,
    .sweet-spot-label, .quadrant-hint, .median-label, .cheapest-tag {{
      fill: var(--text-primary); font-size: 11px; font-weight: var(--font-semibold);
    }}
    .point-label {{
      paint-order: stroke; stroke: rgba(15, 17, 23, 0.85); stroke-width: 3;
      stroke-linejoin: round; pointer-events: none;
    }}
    .bar-label, .legend-label, .axis-tick {{ fill: var(--text-secondary); }}
    .bar-value, .axis-label {{ fill: var(--text-secondary); }}
    .sweet-spot-label {{ fill: var(--color-success); }}
    .quadrant-hint {{ fill: #22d3ee; font-size: 11px; opacity: 0.85; }}
    .median-label {{ fill: #cbd5e1; font-size: 10px; opacity: 0.8; }}
    .cheapest-tag {{ fill: var(--color-success); font-size: 11px; }}
    .chart-toggle {{
      display: inline-flex; gap: 4px; padding: 4px; margin-bottom: var(--space-3);
      background: var(--bg-input); border: 1px solid var(--border-default); border-radius: 999px;
    }}
    .chart-toggle-btn {{
      appearance: none; border: 0; background: transparent; color: var(--text-secondary);
      cursor: pointer; padding: 6px 14px; font-size: var(--text-xs); font-weight: var(--font-semibold);
      border-radius: 999px; transition: background 0.15s ease, color 0.15s ease;
    }}
    .chart-toggle-btn:hover {{ color: var(--text-primary); }}
    .chart-toggle-btn.is-active {{ background: var(--color-accent); color: #fff; }}
    .chart-tooltip {{
      position: absolute; display: none; pointer-events: none; z-index: 100;
      background: #0f172aee; border: 1px solid #475569; border-radius: 8px;
      padding: 8px 12px; font-size: 12px; color: #e2e8f0; line-height: 1.5;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); max-width: 260px;
      backdrop-filter: blur(8px);
    }}
    .chart-tooltip strong {{ display: block; color: #f1f5f9; margin-bottom: 4px; font-size: 13px; }}
    .chart-tooltip .tt-row {{ display: block; color: #94a3b8; font-size: 11px; }}
    .chart-tooltip .tt-row em {{ font-style: normal; color: #e2e8f0; font-weight: 500; }}
    footer {{ margin-top: var(--space-6); font-size: var(--text-xs); color: var(--text-secondary); }}
    a {{ color: var(--color-accent); }}
    .small {{ font-size: var(--text-sm); margin: calc(-1 * var(--space-2)) 0 var(--space-4); }}
    .pick-cards {{ display: flex; gap: var(--space-4); margin: 0 0 var(--space-5); }}
    .pick-card {{
      background: var(--bg-surface); border: 1px solid var(--border-default);
      border-radius: 12px; padding: var(--space-4) var(--space-5); flex: 1; min-width: 200px;
    }}
    .pick-card__label {{
      font-size: var(--text-xs); color: var(--text-secondary); text-transform: uppercase;
      letter-spacing: 0.05em; margin-bottom: var(--space-1);
    }}
    .pick-card__model {{ font-size: var(--text-base); font-weight: var(--font-semibold); color: var(--text-primary); }}
    .pick-card__provider {{ margin-top: var(--space-2); }}
    .pick-card__stat {{ font-size: var(--text-sm); color: var(--text-secondary); margin-top: var(--space-1); }}
    .provider-badge {{
      display: inline-flex; align-items: center; gap: 6px; font-size: var(--text-xs);
      padding: 2px 8px; border-radius: 4px; background: var(--bg-input); color: var(--text-primary);
      text-transform: capitalize;
    }}
    .provider-badge::before {{
      content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--provider-color);
    }}
    .filter-bar, .calculator-controls {{
      display: flex; flex-wrap: wrap; gap: var(--space-4); align-items: end; margin: 0 0 var(--space-4);
      padding: var(--space-4); border: 1px solid var(--border-default); border-radius: 8px;
      background: var(--bg-elevated);
    }}
    .filter-group {{ display: grid; gap: 0.35rem; }}
    .filter-label, .filter-field span, .filter-field {{
      color: var(--text-secondary); font-size: var(--text-xs); font-weight: var(--font-semibold);
    }}
    .chip-group {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
    .chip {{
      border: 1px solid var(--border-default); border-radius: 999px; background: transparent;
      color: var(--text-secondary); cursor: pointer; padding: var(--space-1) var(--space-3);
      text-transform: capitalize; font-size: var(--text-xs);
    }}
    .chip:hover {{ transform: scale(1.02); }}
    .chip.is-active {{ border-color: var(--color-accent); color: white; background: var(--color-accent); }}
    input {{
      border: 1px solid var(--border-default); border-radius: 8px; background: var(--bg-input);
      color: var(--text-primary); padding: 0.45rem 0.55rem;
    }}
    input[type="range"] {{ min-width: 160px; padding: 0; }}
    .filter-field {{ display: grid; gap: 0.35rem; }}
    .filter-field--wide {{ min-width: 200px; }}
    .calculator-controls label {{ display: grid; gap: 0.35rem; color: var(--text-secondary); font-size: var(--text-xs); font-weight: var(--font-semibold); }}
    .recommended-badge {{
      display: none; margin-left: var(--space-2); border-radius: 999px; padding: 2px 8px;
      background: rgba(16, 185, 129, 0.15); color: var(--color-success); font-size: var(--text-xs);
      font-weight: var(--font-semibold);
    }}
    tr.is-recommended .recommended-badge {{ display: inline-flex; }}
    tr.is-recommended td.calculator-cost {{ color: var(--color-success); font-weight: var(--font-semibold); }}
    @keyframes cell-flash {{
      0%   {{ background: rgba(99, 102, 241, 0.35); }}
      100% {{ background: transparent; }}
    }}
    .cell-changed {{ animation: cell-flash 0.7s ease-out; }}
    .pricing-toggle {{
      border: 1px solid var(--border-default); border-radius: 8px; background: var(--bg-input);
      color: var(--text-primary); padding: var(--space-2) var(--space-4); cursor: pointer;
      margin-bottom: var(--space-4);
    }}
    .pricing-detail {{ display: none; }}
    .pricing-table.show-details .pricing-detail {{ display: table-cell; }}
    .pricing-tier-header td {{
      background: var(--bg-elevated); color: var(--text-secondary);
      text-transform: uppercase; letter-spacing: 0.06em; font-size: var(--text-xs);
      font-weight: var(--font-semibold); padding: var(--space-3) var(--space-4);
      border-top: 1px solid var(--border-default);
    }}
    .pricing-tier-header.pricing-tier--budget td  {{ color: #4ade80; border-left: 3px solid #10b981; }}
    .pricing-tier-header.pricing-tier--mid td     {{ color: #fbbf24; border-left: 3px solid #f59e0b; }}
    .pricing-tier-header.pricing-tier--premium td {{ color: #f87171; border-left: 3px solid #ef4444; }}
    .pricing-row.pricing-tier--budget  td:first-child {{ border-left: 3px solid #10b981; }}
    .pricing-row.pricing-tier--mid     td:first-child {{ border-left: 3px solid #f59e0b; }}
    .pricing-row.pricing-tier--premium td:first-child {{ border-left: 3px solid #ef4444; }}
    .bench-row-hidden {{ display: none; }}
    .show-all-btn {{
      display: block; margin: var(--space-4) auto 0; padding: var(--space-2) var(--space-5);
      background: transparent; border: 1px solid var(--border-default); border-radius: 999px;
      color: var(--text-secondary); cursor: pointer; font-size: var(--text-sm);
      font-weight: var(--font-medium); transition: background 0.15s ease, color 0.15s ease;
    }}
    .show-all-btn:hover {{
      background: var(--bg-elevated); color: var(--text-primary); border-color: var(--color-accent);
    }}
    .quality-good, .quality-mid, .quality-low {{ transition: transform 0.1s ease; display: inline-block; }}
    td.quality-good:hover, td.quality-mid:hover, td.quality-low:hover {{ transform: scale(1.05); }}
    .card {{
      background:
        linear-gradient(180deg, rgba(99,102,241,0.04) 0%, transparent 120px),
        var(--bg-surface);
    }}
    .badge {{
      display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.5rem;
      border-radius: 999px; vertical-align: middle; margin-left: 0.5rem;
      font-weight: 500; letter-spacing: 0.02em;
    }}
    .badge--live {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; }}
    .badge--ref {{ background: rgba(148, 163, 184, 0.18); color: #cbd5e1; }}
    @keyframes skeleton-pulse {{
      0%, 100% {{ opacity: 0.5; }}
      50% {{ opacity: 1; }}
    }}
    .skeleton-row {{ animation: skeleton-pulse 1.2s ease-in-out infinite; }}
    @media (max-width: 1024px) {{
      .chart-wrap {{ min-height: 300px; }}
    }}
    @media (max-width: 768px) {{
      .benchmark-table th:nth-child(n+6),
      .benchmark-table td:nth-child(n+6) {{ display: none; }}
      .pick-cards {{ flex-direction: column; }}
      .chart-wrap svg {{ width: 100%; height: auto; }}
      .top-nav {{ top: 0; }}
      th:first-child, td:first-child {{
        position: sticky; left: 0; z-index: 1; background: var(--bg-surface);
      }}
    }}
    @media (max-width: 639px) {{
      .wrap {{ padding-inline: var(--space-4); }}
      .filter-bar {{ flex-wrap: wrap; }}
      table {{ font-size: var(--text-xs); }}
      th, td {{ padding: var(--space-2) var(--space-3); }}
      .chart-toggle {{ width: 100%; justify-content: center; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>llm-bench</h1>
      <p class="lede">Cost, latency, and quality across LLM providers.</p>
      <p class="meta">Dashboard updated: {escape(updated)}
        {f" · Pricing snapshot: {escape(snapshot_date)}" if snapshot_date else ""}</p>
    </header>

    <nav class="top-nav" aria-label="Dashboard sections">
      <a href="#calculator">Calculator</a>
      {task_nav_links}
      <a href="#pricing">Pricing</a>
    </nav>

    <section class="card calculator" id="calculator">
      <h2>Agentic cost calculator</h2>
      <p class="muted small">Estimate monthly spend for multi-turn tasks using provider
      input/output token pricing. <strong>Edit any value below — costs recalculate live.</strong>
      Defaults assume 100 tasks/day, 5K input tokens, and 2K output tokens per task.</p>
      <div class="calculator-controls">
        <label>Tasks per day
          <input id="tasks-per-day" type="number" min="0" step="1" value="100"
                 placeholder="100"/>
        </label>
        <label>Avg input tokens/task
          <input id="input-tokens" type="number" min="0" step="100" value="5000"
                 placeholder="5000"/>
        </label>
        <label>Avg output tokens/task
          <input id="output-tokens" type="number" min="0" step="100" value="2000"
                 placeholder="2000"/>
        </label>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Model</th><th>Provider</th><th>Estimated monthly cost</th><th>Avg quality</th></tr></thead>
          <tbody>{_calculator_rows(models, average_quality)}</tbody>
        </table>
      </div>
    </section>

    <section id="benchmarks">
      <h2 style="margin-bottom:1rem;">Benchmark results
        <span class="badge badge--live" title="Benchmarks re-run daily when publishing succeeds.">
          updated daily
        </span>
      </h2>
      <p class="muted small">Quality, latency, and cost measured by running each model
      against the task suites in <code>benchmarks/</code>. Cost reflects observed token
      usage at provider list prices. Tags: <strong>value</strong> means Pareto-optimal;
      <strong>cheapest</strong> means lowest cost in the category with quality ≥ 80.</p>
      {tasks_html}
    </section>

    <section class="card" id="pricing">
      <h2>Model pricing <span class="badge badge--ref">reference</span></h2>
      <p class="muted small">Reference data — <strong>not benchmarked by llm-bench</strong>.
      Numbers below were copied verbatim from each provider's pricing page on
      <strong>{escape(CATALOG_VERIFIED)}</strong>. Click <em>source</em> on any row to
      verify the latest rate. Prices in USD per 1M tokens. Cached input discounts are
      shown where the provider publishes a broad discount; batch average assumes 50% off.</p>
      <button type="button" class="pricing-toggle" aria-expanded="false">Show details</button>
      <div class="table-wrap"><table class="pricing-table">
        <thead><tr>
          <th>Model</th><th>Provider</th><th>Avg/1M</th><th>Source</th>
          <th class="pricing-detail">Input/1M</th><th class="pricing-detail">Output/1M</th>
          <th class="pricing-detail">Cached Input/1M</th><th class="pricing-detail">Batch Avg/1M</th>
        </tr></thead>
        <tbody>{_pricing_rows(models)}</tbody>
      </table></div>
    </section>

    <footer>
      <p>Data: <a href="data/prices.json">prices.json</a> ·
         <a href="data/benchmarks.json">benchmarks.json</a></p>
      <p>Generated by <a href="https://github.com/webswift-engineering/-llm-bench">llm-bench</a></p>
    </footer>
  </div>
{dashboard_script}
</body>
</html>"""

    index_path = out / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
