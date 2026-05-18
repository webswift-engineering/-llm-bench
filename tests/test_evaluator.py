"""Tests for deterministic scoring."""

from llm_bench.evaluator import score_classification


def test_classification_exact_match():
    assert score_classification("positive", "positive") == 100.0


def test_classification_case_insensitive():
    assert score_classification("Positive.", "positive") == 100.0


def test_classification_mismatch():
    assert score_classification("negative", "positive") == 0.0
