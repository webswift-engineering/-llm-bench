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
    "groq": "#6366f1",
    "google": "#38bdf8",
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


def _pareto_svg(results: list[BenchmarkResult], width: int = 720, height: int = 420) -> str:
    scored = [r for r in results if r.quality_score > 0]
    if len(scored) < 2:
        return "<p class='muted'>Need at least 2 models with scores for a Pareto chart.</p>"

    import math

    costs = [max(r.cost_per_1k_requests, 0.001) for r in scored]
    qualities = [r.quality_score for r in scored]
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

    frontier = compute_pareto_frontier(scored)
    frontier_pts = " ".join(f"{px(r.cost_per_1k_requests):.1f},{py(r.quality_score):.1f}" for r in frontier)
    cheapest = min(scored, key=lambda r: (r.cost_per_1k_requests, -r.quality_score))
    highest_quality = max(scored, key=lambda r: (r.quality_score, -r.cost_per_1k_requests))
    best_value = max(scored, key=_quality_per_dollar)
    labels: dict[int, list[str]] = {}
    labels.setdefault(id(cheapest), []).append("Cheapest")
    labels.setdefault(id(highest_quality), []).append("Highest quality")
    labels.setdefault(id(best_value), []).append("Best value")

    dots = []
    for r in scored:
        on_frontier = r in frontier
        color = PROVIDER_COLORS.get(r.provider.value, "#94a3b8")
        classes = "chart-point chart-point--frontier" if on_frontier else "chart-point"
        dots.append(
            f'<circle class="{classes}" cx="{px(r.cost_per_1k_requests):.1f}" '
            f'cy="{py(r.quality_score):.1f}" r="6.5" fill="{color}" '
            f'stroke="#0f172a" stroke-width="1.5">'
            f"<title>{escape(r.display_name)}: {r.quality_score:.0f}% @ "
            f"${r.cost_per_1k_requests:.2f}/1K</title></circle>"
        )
        if id(r) in labels:
            label_text = " + ".join(labels[id(r)])
            x = px(r.cost_per_1k_requests)
            y = py(r.quality_score)
            dx = 10 if x < width - 200 else -10
            anchor = "start" if dx > 0 else "end"
            dots.append(
                f'<text class="point-label" x="{x + dx:.1f}" y="{y - 10:.1f}" '
                f'text-anchor="{anchor}">{escape(label_text)}: {escape(r.display_name)}</text>'
            )

    providers = sorted({r.provider.value for r in scored})
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
        f'text-anchor="middle" fill="#94a3b8" font-size="11">${tick:.2g}</text></g>'
        for tick in ticks
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="pareto-chart" role="img"
      aria-label="Cost vs quality chart using a logarithmic cost scale">
      <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}"
            y2="{height - margin_bottom}" stroke="#334155"/>
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}"
            y2="{height - margin_bottom}" stroke="#334155"/>
      {tick_marks}
      <text x="{width // 2}" y="{height - 8}" text-anchor="middle" fill="#94a3b8" font-size="12">
        Cost / 1K reqs ($, log scale)</text>
      <text x="16" y="{height // 2}" text-anchor="middle" fill="#94a3b8" font-size="12"
            transform="rotate(-90 16 {height // 2})">Quality</text>
      <polyline points="{frontier_pts}" fill="none" stroke="#e2e8f0" stroke-width="2"
                stroke-dasharray="5,4"/>
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
            f"<tr><td>{escape(m.provider.value)}</td>"
            f"<td>{escape(m.display_name)}</td>"
            f"<td>${m.input_per_1m:.3f}</td>"
            f"<td>${m.output_per_1m:.3f}</td>"
            f"<td>{_format_money(cached)}</td>"
            f"<td class='highlight'>${m.avg_per_1m:.3f}</td>"
            f"<td>${m.avg_per_1m * 0.5:.3f}</td>"
            f"<td class='muted'>{source}</td></tr>"
        )
    return "\n".join(rows)


def _calculator_rows(models: list[ModelPricing]) -> str:
    rows = []
    for m in sorted(models, key=lambda x: (5000 * x.input_per_1m + 2000 * x.output_per_1m)):
        monthly = 100 * 30 * (
            (5000 * m.input_per_1m / 1_000_000) + (2000 * m.output_per_1m / 1_000_000)
        )
        rows.append(
            f"<tr data-provider='{escape(m.provider.value)}' "
            f"data-input-price='{m.input_per_1m:.6f}' "
            f"data-output-price='{m.output_per_1m:.6f}'>"
            f"<td>{escape(m.display_name)}</td><td>{escape(m.provider.value)}</td>"
            f"<td class='calculator-cost'>${monthly:.2f}</td></tr>"
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
        rows.append(
            f"<tr data-default-index='{index}' data-provider='{escape(r.provider.value)}' "
            f"data-cost='{r.cost_per_1k_requests:.6f}' data-quality='{r.quality_score:.6f}'>"
            f"<td data-value='{escape(r.display_name.lower())}'>{escape(r.display_name)}</td>"
            f"<td data-value='{escape(r.provider.value)}'>{escape(r.provider.value)}</td>"
            f"<td data-value='{r.quality_score:.6f}'>{r.quality_score:.0f}</td>"
            f"<td data-value='{r.latency_p50_ms:.6f}'>{r.latency_p50_ms:.0f}ms</td>"
            f"<td data-value='{r.cost_per_1k_requests:.6f}'>${r.cost_per_1k_requests:.2f}</td>"
            f"<td data-value='{value:.6f}'>{value:.1f}</td>"
            f"<td data-value='{escape(tags.lower())}'>{escape(tags)}</td></tr>"
        )
    return f"""<div class="benchmark-card" data-table-id="{escape(table_id)}">
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
    updated = utc_now().strftime("%Y-%m-%d %H:%M UTC")

    task_sections = []
    for task, results in benchmarks.items():
        task_id = f"task-{task}"
        task_sections.append(
            f"""<section class="card" id="{escape(task_id)}">
              <h2>{escape(task.replace("_", " ").title())}</h2>
              {_benchmark_table(results, task)}
              <div class="chart-wrap">{_pareto_svg(results)}</div>
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
          });
          rows
            .sort((a, b) => numberValue(a.dataset.monthlyCost) - numberValue(b.dataset.monthlyCost))
            .forEach((row) => tbody.appendChild(row));
        }
        [tasks, inputTokens, outputTokens].forEach((input) => input.addEventListener("input", render));
        render();
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
      --bg: #0b1120; --card: #131c31; --text: #e2e8f0; --muted: #94a3b8;
      --accent: #818cf8; --green: #10b981; --border: #1e293b;
      --amber: #f59e0b; --indigo: #6366f1;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); margin: 0; line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.25rem; }}
    .lede {{ color: var(--muted); margin: 0 0 2rem; }}
    .meta {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 2rem; }}
    .top-nav {{
      position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 0.5rem;
      margin: 0 -0.25rem 1.5rem; padding: 0.65rem 0.25rem;
      background: color-mix(in srgb, var(--bg) 88%, transparent);
      backdrop-filter: blur(12px);
    }}
    .top-nav a {{
      border: 1px solid var(--border); border-radius: 999px; color: var(--muted);
      padding: 0.3rem 0.7rem; text-decoration: none; font-size: 0.82rem;
    }}
    .top-nav a:hover, .top-nav a.is-active {{
      border-color: var(--accent); color: var(--text); background: rgba(129, 140, 248, 0.12);
    }}
    .card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
    }}
    h2 {{ font-size: 1.1rem; margin: 0 0 1rem; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 500; }}
    tbody tr:hover {{ background: rgba(255, 255, 255, 0.05); }}
    tr[hidden] {{ display: none; }}
    th button {{
      appearance: none; border: 0; background: transparent; color: inherit; cursor: pointer;
      font: inherit; padding: 0; text-align: left; white-space: nowrap;
    }}
    th button:hover, th button.is-sorted {{ color: var(--text); }}
    .table-wrap {{ overflow-x: auto; }}
    .highlight {{ color: var(--green); font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    code {{ background: #1e293b; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }}
    .chart-wrap {{ margin-top: 1rem; overflow-x: auto; }}
    .pareto-chart {{ width: 100%; max-width: 720px; height: auto; }}
    .chart-point--frontier {{ filter: drop-shadow(0 0 4px rgba(226, 232, 240, 0.5)); }}
    .point-label, .legend-label {{ fill: #e2e8f0; font-size: 11px; font-weight: 600; }}
    .legend-label {{ fill: #cbd5e1; text-transform: capitalize; }}
    footer {{ margin-top: 2rem; font-size: 0.8rem; color: var(--muted); }}
    a {{ color: var(--accent); }}
    .small {{ font-size: 0.85rem; margin: -0.5rem 0 1rem; }}
    .filter-bar, .calculator-controls {{
      display: flex; flex-wrap: wrap; gap: 0.8rem; align-items: end; margin: 0 0 1rem;
      padding: 0.85rem; border: 1px solid var(--border); border-radius: 10px;
      background: rgba(15, 23, 42, 0.45);
    }}
    .filter-group {{ display: grid; gap: 0.35rem; }}
    .filter-label, .filter-field span, .filter-field {{
      color: var(--muted); font-size: 0.78rem; font-weight: 600;
    }}
    .chip-group {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
    .chip {{
      border: 1px solid var(--border); border-radius: 999px; background: transparent;
      color: var(--muted); cursor: pointer; padding: 0.3rem 0.6rem; text-transform: capitalize;
    }}
    .chip.is-active {{ border-color: var(--accent); color: var(--text); background: rgba(99, 102, 241, 0.18); }}
    input {{
      border: 1px solid var(--border); border-radius: 8px; background: #0f172a; color: var(--text);
      padding: 0.45rem 0.55rem;
    }}
    input[type="range"] {{ min-width: 160px; padding: 0; }}
    .filter-field {{ display: grid; gap: 0.35rem; }}
    .filter-field--wide {{ min-width: 200px; }}
    .calculator-controls label {{ display: grid; gap: 0.35rem; color: var(--muted); font-size: 0.78rem; font-weight: 600; }}
    .badge {{
      display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.5rem;
      border-radius: 999px; vertical-align: middle; margin-left: 0.5rem;
      font-weight: 500; letter-spacing: 0.02em;
    }}
    .badge--live {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; }}
    .badge--ref {{ background: rgba(148, 163, 184, 0.18); color: #cbd5e1; }}
    @media (max-width: 768px) {{
      .wrap {{ padding-inline: 1rem; }}
      th:first-child, td:first-child {{
        position: sticky; left: 0; z-index: 1; background: var(--card);
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
          <thead><tr><th>Model</th><th>Provider</th><th>Estimated monthly cost</th></tr></thead>
          <tbody>{_calculator_rows(models)}</tbody>
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
      <div class="table-wrap"><table>
        <thead><tr>
          <th>Provider</th><th>Model</th><th>Input/1M</th><th>Output/1M</th>
          <th>Cached Input/1M</th><th>Avg/1M</th><th>Batch Avg/1M</th><th>Source</th>
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
