"""Load task suites from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

def _benchmarks_dir() -> Path:
    # Editable install: repo root/benchmarks
    repo_dir = Path(__file__).resolve().parents[3] / "benchmarks"
    if repo_dir.exists():
        return repo_dir
    # Wheel install: packaged alongside module
    packaged = Path(__file__).resolve().parent.parent / "benchmarks"
    return packaged if packaged.exists() else repo_dir


@dataclass
class TaskSample:
    input: str
    expected: str = ""
    criteria: list[str] = field(default_factory=list)


@dataclass
class TaskSuite:
    name: str
    scoring: str  # "deterministic" | "llm_judge"
    judge_model: str = "gpt-4o-mini"
    system_prompt: str = ""
    samples: list[TaskSample] = field(default_factory=list)


def load_task(task_name: str) -> TaskSuite:
    path = _benchmarks_dir() / f"{task_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task not found: {task_name} (expected {path})")

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    samples = [
        TaskSample(
            input=s["input"].strip(),
            expected=s.get("expected", ""),
            criteria=s.get("criteria", []),
        )
        for s in data.get("samples", [])
    ]

    return TaskSuite(
        name=data.get("task", task_name),
        scoring=data.get("scoring", "deterministic"),
        judge_model=data.get("judge_model", "gpt-4o-mini"),
        system_prompt=data.get("system_prompt", "").strip(),
        samples=samples,
    )
