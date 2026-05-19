"""Unit tests for evaluation dataset shape."""

from __future__ import annotations

from evals.eval_dataset import RAG_EVAL_CASES


def test_eval_dataset_has_required_fields() -> None:
    assert len(RAG_EVAL_CASES) >= 5
    for case in RAG_EVAL_CASES:
        assert case["question"].strip()
        assert case["reference"].strip()
