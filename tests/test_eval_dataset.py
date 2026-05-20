"""Unit tests for evaluation dataset shape."""

from __future__ import annotations

from evals.eval_dataset import RAG_EVAL_CASES
from evals.scorecard_dataset import RAG_SCORECARD_CASES


def test_eval_dataset_has_required_fields() -> None:
    assert len(RAG_EVAL_CASES) >= 10
    for case in RAG_EVAL_CASES:
        assert case["id"].strip()
        assert case["question"].strip()
        assert case["reference"].strip()


def test_eval_dataset_covers_filters() -> None:
    assert any(case.get("topic_filter") for case in RAG_EVAL_CASES)
    assert any(case.get("domain_filter") for case in RAG_EVAL_CASES)


def test_rag_scorecard_expects_direct_hybrid_retrieval() -> None:
    assert RAG_SCORECARD_CASES
    assert all(case["expected_tool"] == "hybrid_retrieval" for case in RAG_SCORECARD_CASES)
