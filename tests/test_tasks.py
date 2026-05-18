"""Tests for task loading."""

from llm_bench.tasks.loader import load_task


def test_load_classification_task():
    task = load_task("classification")
    assert task.name == "classification"
    assert task.scoring == "deterministic"
    assert len(task.samples) >= 5


def test_load_summarization_task():
    task = load_task("summarization")
    assert task.name == "summarization"
    assert task.scoring == "llm_judge"
    assert task.samples[0].criteria
