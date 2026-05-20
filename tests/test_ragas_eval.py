"""Unit tests for RAGAS evaluation plumbing."""

from __future__ import annotations

import pytest

from evals.ragas_eval import build_ragas_dataset, import_ragas_metrics


def test_build_ragas_dataset_contains_old_and_new_columns() -> None:
    pytest.importorskip("datasets")
    dataset = build_ragas_dataset(
        [
            {
                "id": "case-1",
                "question": "What is RAG?",
                "answer": "RAG retrieves context before generation.",
                "contexts": ["RAG combines retrieval and generation."],
                "ground_truth": "RAG retrieves relevant context before generation.",
                "reference": "RAG retrieves relevant context before generation.",
                "retrieval_queries": ["What is RAG? (retrieval augmented generation)"],
                "effective_topic_filter": "rag",
                "cached": False,
                "cache_type": None,
                "cache_score": 0.0,
            }
        ]
    )
    row = dataset[0]
    assert row["question"] == row["user_input"]
    assert row["answer"] == row["response"]
    assert row["contexts"] == row["retrieved_contexts"]
    assert row["ground_truth"] == row["reference"]
    assert row["retrieval_queries"] == ["What is RAG? (retrieval augmented generation)"]
    assert row["effective_topic_filter"] == "rag"


def test_import_ragas_metrics_is_safe_without_hard_failure() -> None:
    metrics = import_ragas_metrics()
    assert isinstance(metrics, list)
    try:
        import ragas  # noqa: F401
    except Exception:
        return
    assert len(metrics) >= 4
