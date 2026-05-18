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


def _pareto_svg(results: list[BenchmarkResult], width: int = 640, height: int = 360) -> str:
    scored = [r for r in results if r.quality_score > 0]
    if len(scored) < 2:
        return "<p class='muted'>Need at least 2 models with scores for a Pareto chart.</p>"

    costs = [r.cost_per_1k_requests for r in scored]
    qualities = [r.quality_score for r in scored]
    min_c, max_c = min(costs), max(costs)
    min_q, max_q = min(qualities), max(qualities)
    pad_c = (max_c - min_c) * 0.1 or 0.1
    pad_q = (max_q - min_q) * 0.1 or 5
    min_c -= pad_c
    max_c += pad_c
    min_q -= pad_q
    max_q += pad_q

    margin = 48

    def px(cost: float) -> float:
        return margin + (cost - min_c) / (max_c - min_c) * (width - 2 * margin)

    def py(quality: float) -> float:
        return height - margin - (quality - min_q) / (max_q - min_q) * (height - 2 * margin)

    frontier = compute_pareto_frontier(scored)
    frontier_pts = " ".join(f"{px(r.cost_per_1k_requests):.1f},{py(r.quality_score):.1f}" for r in frontier)

    dots = []
    for r in scored:
        on_frontier = r in frontier
        color = "#22c55e" if on_frontier else "#6366f1"
        dots.append(
            f'<circle cx="{px(r.cost_per_1k_requests):.1f}" cy="{py(r.quality_score):.1f}" '
            f'r="6" fill="{color}" stroke="#0f172a" stroke-width="1">'
            f"<title>{escape(r.display_name)}: {r.quality_score:.0f}% @ "
            f"${r.cost_per_1k_requests:.2f}/1K</title></circle>"
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="pareto-chart" role="img"
      aria-label="Cost vs quality chart">
      <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
            stroke="#334155"/>
      <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#334155"/>
      <text x="{width // 2}" y="{height - 8}" text-anchor="middle" fill="#94a3b8" font-size="12">
        Cost per 1K requests ($)</text>
      <text x="14" y="{height // 2}" text-anchor="middle" fill="#94a3b8" font-size="12"
            transform="rotate(-90 14 {height // 2})">Quality</text>
      <polyline points="{frontier_pts}" fill="none" stroke="#22c55e" stroke-width="2"
                stroke-dasharray="5,4"/>
      {"".join(dots)}
    </svg>"""


def _pricing_rows(models: list[ModelPricing]) -> str:
    rows = []
    for m in sorted(models, key=lambda x: x.avg_per_1m):
        source = (
            f"<a href='{escape(m.source_url)}' target='_blank' rel='noopener'>source</a>"
            if m.source_url
            else "—"
        )
        rows.append(
            f"<tr><td>{escape(m.provider.value)}</td>"
            f"<td>{escape(m.display_name)}</td>"
            f"<td>${m.input_per_1m:.3f}</td>"
            f"<td>${m.output_per_1m:.3f}</td>"
            f"<td class='highlight'>${m.avg_per_1m:.3f}</td>"
            f"<td class='muted'>{source}</td></tr>"
        )
    return "\n".join(rows)


def _benchmark_table(results: list[BenchmarkResult]) -> str:
    if not results:
        return "<p class='muted'>No results yet. Run <code>llm-bench run --save</code> locally.</p>"
    rows = []
    for r in sorted(results, key=lambda x: x.quality_score, reverse=True):
        tags = ", ".join(r.tags) if r.tags else "—"
        rows.append(
            f"<tr><td>{escape(r.display_name)}</td>"
            f"<td>{escape(r.provider.value)}</td>"
            f"<td>{r.quality_score:.0f}</td>"
            f"<td>{r.latency_p50_ms:.0f}ms</td>"
            f"<td>${r.cost_per_1k_requests:.2f}</td>"
            f"<td>{escape(tags)}</td></tr>"
        )
    return f"""<table>
      <thead><tr>
        <th>Model</th><th>Provider</th><th>Quality</th>
        <th>Latency p50</th><th>Cost/1K</th><th>Tags</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


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
        task_sections.append(
            f"""<section class="card" id="task-{escape(task)}">
              <h2>{escape(task.replace("_", " ").title())}</h2>
              {_benchmark_table(results)}
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
            "avg_per_1m": round(m.avg_per_1m, 3),
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
      --accent: #6366f1; --green: #22c55e; --border: #1e293b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); margin: 0; line-height: 1.5;
    }}
    .wrap {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.25rem; }}
    .lede {{ color: var(--muted); margin: 0 0 2rem; }}
    .meta {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 2rem; }}
    .card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
    }}
    h2 {{ font-size: 1.1rem; margin: 0 0 1rem; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 500; }}
    .highlight {{ color: var(--green); font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    code {{ background: #1e293b; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }}
    .chart-wrap {{ margin-top: 1rem; overflow-x: auto; }}
    .pareto-chart {{ width: 100%; max-width: 640px; height: auto; }}
    footer {{ margin-top: 2rem; font-size: 0.8rem; color: var(--muted); }}
    a {{ color: var(--accent); }}
    .small {{ font-size: 0.85rem; margin: -0.5rem 0 1rem; }}
    .badge {{
      display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.5rem;
      border-radius: 999px; vertical-align: middle; margin-left: 0.5rem;
      font-weight: 500; letter-spacing: 0.02em;
    }}
    .badge--live {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; }}
    .badge--ref {{ background: rgba(148, 163, 184, 0.18); color: #cbd5e1; }}
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

    <section id="benchmarks">
      <h2 style="margin-bottom:1rem;">Benchmark results <span class="badge badge--live">live</span></h2>
      <p class="muted small">Quality, latency, and cost measured by running each model
      against the task suites in <code>benchmarks/</code>. Cost reflects observed token
      usage at provider list prices.</p>
      {tasks_html}
    </section>

    <section class="card" id="pricing">
      <h2>Model pricing <span class="badge badge--ref">reference</span></h2>
      <p class="muted small">Reference data — <strong>not benchmarked by llm-bench</strong>.
      Numbers below were copied verbatim from each provider's pricing page on
      <strong>{escape(CATALOG_VERIFIED)}</strong>. Click <em>source</em> on any row to
      verify the latest rate. Prices in USD per 1M tokens.</p>
      <table>
        <thead><tr>
          <th>Provider</th><th>Model</th><th>Input/1M</th><th>Output/1M</th><th>Avg/1M</th><th>Source</th>
        </tr></thead>
        <tbody>{_pricing_rows(models)}</tbody>
      </table>
    </section>

    <footer>
      <p>Data: <a href="data/prices.json">prices.json</a> ·
         <a href="data/benchmarks.json">benchmarks.json</a></p>
      <p>Generated by <a href="https://github.com/webswift-engineering/-llm-bench">llm-bench</a></p>
    </footer>
  </div>
</body>
</html>"""

    index_path = out / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
