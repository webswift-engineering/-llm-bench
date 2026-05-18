"""Quality scoring — deterministic and LLM-as-judge."""

from __future__ import annotations

import json
import os
import re

from llm_bench.models import Provider
from llm_bench.pricing.catalog import resolve_model_id
from llm_bench.providers.base import get_adapter
from llm_bench.tasks.loader import TaskSample


def _strip_reasoning_blocks(text: str) -> str:
    """Remove <think>...</think> and similar reasoning tags (Qwen, DeepSeek, etc)."""
    return re.sub(r"<think[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def score_classification(output: str, expected: str) -> float:
    """Return 0-100 based on label match (case-insensitive, ignoring reasoning tags)."""
    cleaned = _strip_reasoning_blocks(output)
    normalized = cleaned.strip().lower()
    target = expected.strip().lower()
    # Try last line first (reasoning models often answer last), then first
    lines = [ln.strip().rstrip(".") for ln in normalized.split("\n") if ln.strip()]
    for line in [lines[-1], lines[0]] if lines else []:
        if line == target or target in line:
            return 100.0
    return 100.0 if target in normalized else 0.0


async def score_llm_judge(
    output: str,
    sample: TaskSample,
    judge_model: str,
) -> float:
    """Use a separate model to score subjective tasks (0-100)."""
    judge_id = resolve_model_id(judge_model)
    criteria_text = "\n".join(f"- {c}" for c in sample.criteria)
    prompt = f"""Rate the following response on a scale of 1-10 for the criteria listed.
Return ONLY a JSON object: {{"score": <number 1-10>}}

Criteria:
{criteria_text}

Original input (for context):
{sample.input[:2000]}

Response to evaluate:
{output}
"""

    # Judge always runs on OpenAI (cheapest reliable judge)
    adapter = get_adapter(Provider.OPENAI)
    if not adapter.is_configured():
        raise RuntimeError("OPENAI_API_KEY required for LLM-judge scoring")

    response = await adapter.complete(judge_id, prompt, system="You are an impartial evaluator.")
    match = re.search(r"\{[^}]+\}", response.text)
    if not match:
        return 50.0
    try:
        data = json.loads(match.group())
        score = float(data.get("score", 5))
        return min(100.0, max(0.0, score * 10))
    except (json.JSONDecodeError, ValueError):
        return 50.0


async def score_sample(
    task_name: str,
    scoring: str,
    output: str,
    sample: TaskSample,
    judge_model: str,
) -> float:
    if scoring == "deterministic":
        if task_name == "classification":
            return score_classification(output, sample.expected)
        return 100.0 if sample.expected.lower() in output.lower() else 0.0
    return await score_llm_judge(output, sample, judge_model)


def get_judge_model() -> str:
    return os.environ.get("LLM_BENCH_JUDGE_MODEL", "gpt-4o-mini")
