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
    scored = [r for r in results if r.quality_score > 0]
    if len(scored) < 2:
        return "<p class='muted'>Need at least 2 models with scores for a chart.</p>"
    qualities = {round(r.quality_score, 6) for r in scored}
    if len(qualities) == 1:
        return _cost_bar_svg(scored)
    return _scatter_svg(scored)


def _cost_bar_svg(results: list[BenchmarkResult], width: int = 720) -> str:
    sorted_results = sorted(results, key=lambda r: r.cost_per_1k_requests)
    row_height = 28
    margin_left = 190
    margin_right = 92
    margin_top = 24
    margin_bottom = 34
    height = margin_top + margin_bottom + row_height * len(sorted_results)
    max_cost = max(r.cost_per_1k_requests for r in sorted_results) or 1
    plot_width = width - margin_left - margin_right
    bars = []
    for index, result in enumerate(sorted_results):
        y = margin_top + index * row_height
        bar_width = max(result.cost_per_1k_requests / max_cost * plot_width, 2)
        color = PROVIDER_COLORS.get(result.provider.value, "#94a3b8")
        bars.append(
            f"""<g>
              <text x="{margin_left - 10}" y="{y + 17}" text-anchor="end" class="bar-label">
                {escape(result.display_name)}
              </text>
              <rect x="{margin_left}" y="{y + 5}" width="{bar_width:.1f}" height="14"
                    rx="7" fill="{color}">
                <title>{escape(result.display_name)} | {escape(result.provider.value)} |
                  Cost: ${result.cost_per_1k_requests:.2f}/1K | Quality: {result.quality_score:.0f} |
                  Latency: {result.latency_p50_ms:.0f}ms</title>
              </rect>
              <text x="{margin_left + bar_width + 8:.1f}" y="{y + 17}" class="bar-value">
                ${result.cost_per_1k_requests:.2f}
              </text>
            </g>"""
        )
    return f"""<svg viewBox="0 0 {width} {height}" class="benchmark-chart bar-chart" role="img"
      aria-label="Cost per model sorted cheapest first">
      <text x="{margin_left}" y="{height - 8}" class="axis-label">Cost / 1K reqs</text>
      {"".join(bars)}
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
    rows = []
    for m in sorted(models, key=lambda x: x.avg_per_1m):
        cached = _cached_input_per_1m(m)
        source = (
            f"<a href='{escape(m.source_url)}' target='_blank' rel='noopener'>source</a>"
            if m.source_url
            else ""
        )
        rows.append(
            f"<tr><td>{escape(m.display_name)}</td>"
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
    for index, r in enumerate(sorted_results):
        tags = ", ".join(r.tags) if r.tags else ""
        value = _quality_per_dollar(r)
        quality_class = _quality_class(r.quality_score)
        rows.append(
            f"<tr data-default-index='{index}' data-provider='{escape(r.provider.value)}' "
            f"data-cost='{r.cost_per_1k_requests:.6f}' data-quality='{r.quality_score:.6f}'>"
            f"<td data-value='{escape(r.display_name.lower())}'>{escape(r.display_name)}</td>"
            f"<td data-value='{escape(r.provider.value)}'>{_provider_badge(r.provider.value)}</td>"
            f"<td class='{quality_class}' data-value='{r.quality_score:.6f}'>{r.quality_score:.0f}</td>"
            f"<td data-value='{r.latency_p50_ms:.6f}'>{r.latency_p50_ms:.0f}ms</td>"
            f"<td data-value='{r.cost_per_1k_requests:.6f}'>${r.cost_per_1k_requests:.2f}</td>"
            f"<td data-value='{value:.6f}'>{value:.1f}</td>"
            f"<td data-value='{escape(tags.lower())}'>{escape(tags)}</td></tr>"
        )
    return f"""<div class="benchmark-card" data-table-id="{escape(table_id)}">
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
    </table></div></div>"""


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
        let sortKey = "value";
        let direction = "desc";

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
          rows.forEach((row) => {
            const visible = activeProviders.has(row.dataset.provider)
              && numberValue(row.dataset.cost) >= minCostValue
              && numberValue(row.dataset.cost) <= maxCostValue
              && numberValue(row.dataset.quality) >= minQualityValue;
            row.hidden = !visible;
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
        render();
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
          rows.forEach((row) => {
            const monthly = taskCount * 30 * (
              inputCount * numberValue(row.dataset.inputPrice) / 1000000
              + outputCount * numberValue(row.dataset.outputPrice) / 1000000
            );
            row.dataset.monthlyCost = monthly.toFixed(6);
            row.querySelector(".calculator-cost").textContent = `$${monthly.toFixed(2)}`;
            row.classList.remove("is-recommended");
          });
          const recommended = rows
            .filter((row) => numberValue(row.dataset.monthlyCost) <= 50 && numberValue(row.dataset.quality) > 0)
            .sort((a, b) => numberValue(b.dataset.quality) - numberValue(a.dataset.quality))[0];
          if (recommended) recommended.classList.add("is-recommended");
          rows
            .sort((a, b) => numberValue(a.dataset.monthlyCost) - numberValue(b.dataset.monthlyCost))
            .forEach((row) => tbody.appendChild(row));
        }
        [tasks, inputTokens, outputTokens].forEach((input) => input.addEventListener("input", render));
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
    .chart-point--frontier {{ filter: drop-shadow(0 0 7px rgba(240, 242, 245, 0.75)); stroke: #f0f2f5; stroke-width: 2.5; }}
    .guide-line {{ stroke: var(--border-default); stroke-dasharray: 6 6; }}
    .point-label, .legend-label, .bar-label, .bar-value, .axis-label, .sweet-spot-label {{
      fill: var(--text-primary); font-size: 11px; font-weight: var(--font-semibold);
    }}
    .bar-label, .legend-label {{ fill: var(--text-secondary); }}
    .bar-value, .axis-label {{ fill: var(--text-secondary); }}
    .sweet-spot-label {{ fill: var(--color-success); }}
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
    .pricing-toggle {{
      border: 1px solid var(--border-default); border-radius: 8px; background: var(--bg-input);
      color: var(--text-primary); padding: var(--space-2) var(--space-4); cursor: pointer;
      margin-bottom: var(--space-4);
    }}
    .pricing-detail {{ display: none; }}
    .pricing-table.show-details .pricing-detail {{ display: table-cell; }}
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
    @media (max-width: 639px) {{
      .wrap {{ padding-inline: var(--space-4); }}
      .pick-cards {{ flex-direction: column; }}
      .filter-bar {{ flex-wrap: wrap; }}
      table {{ font-size: var(--text-xs); }}
      th, td {{ padding: var(--space-2) var(--space-3); }}
    }}
    @media (max-width: 768px) {{
      th:first-child, td:first-child {{
        position: sticky; left: 0; z-index: 1; background: var(--bg-surface);
      }}
      .top-nav {{ top: 0; }}
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
      input/output token pricing. Defaults assume 100 tasks/day, 5K input tokens, and
      2K output tokens per task.</p>
      <div class="calculator-controls">
        <label>Tasks per day
          <input id="tasks-per-day" type="number" min="0" step="1" value="100"/>
        </label>
        <label>Avg input tokens/task
          <input id="input-tokens" type="number" min="0" step="100" value="5000"/>
        </label>
        <label>Avg output tokens/task
          <input id="output-tokens" type="number" min="0" step="100" value="2000"/>
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
