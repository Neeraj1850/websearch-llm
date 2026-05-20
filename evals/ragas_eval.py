"""Evaluate the upgraded RAG pipeline with all compatible RAGAS metrics.

Run:
    python -m evals.ragas_eval
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.rag_eval_runner import RESULTS_DIR, build_rag_records

DEFAULT_EVAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_judge_llm():
    load_dotenv()
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("RAGAS_GROQ_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
        temperature=0,
        max_retries=2,
        timeout=60,
    )


def build_judge_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAG_EVAL_EMBEDDING_MODEL", DEFAULT_EVAL_EMBEDDING_MODEL),
        model_kwargs={"device": os.getenv("RAG_EVAL_EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def import_metric(name: str) -> Any | None:
    """Import a RAGAS metric across old and new package layouts."""
    for module_name in ("ragas.metrics.collections", "ragas.metrics"):
        try:
            module = __import__(module_name, fromlist=[name])
            metric = getattr(module, name, None)
            if metric is not None:
                return metric
        except Exception:
            continue
    return None


def instantiate_metric(name: str, class_name: str) -> Any | None:
    try:
        import ragas.metrics as metrics

        cls = getattr(metrics, class_name, None)
        if cls is not None:
            return cls()
    except Exception:
        pass
    return import_metric(name)


def import_ragas_metrics() -> list[Any]:
    """Return all compatible RAGAS metrics available in the installed version."""
    metric_specs = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "ResponseRelevancy"),
        ("answer_correctness", "AnswerCorrectness"),
        ("answer_similarity", "SemanticSimilarity"),
        ("context_precision", "LLMContextPrecisionWithReference"),
        ("context_recall", "LLMContextRecall"),
        ("context_entity_recall", "ContextEntityRecall"),
        ("noise_sensitivity_relevant", "NoiseSensitivity"),
        ("noise_sensitivity_irrelevant", "NoiseSensitivity"),
    ]

    imported = []
    seen = set()
    for metric_name, class_name in metric_specs:
        metric = import_metric(metric_name) or instantiate_metric(metric_name, class_name)
        if metric is None:
            continue
        name = getattr(metric, "name", metric_name)
        if name in seen:
            continue
        seen.add(name)
        imported.append(metric)
    return imported


def build_ragas_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "id": record["id"],
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
                "reference": record["reference"],
                "retrieval_queries": record.get("retrieval_queries", [record["question"]]),
                "effective_topic_filter": record.get("effective_topic_filter"),
                "cached": record.get("cached", False),
                "cache_type": record.get("cache_type"),
                "cache_score": record.get("cache_score", 0.0),
                "retrieved_context_count": record.get("retrieved_context_count", len(record["contexts"])),
                "unique_source_count": record.get("unique_source_count", len(set(record.get("sources", [])))),
                # Newer RAGAS versions use these names for single-turn samples.
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["contexts"],
            }
            for record in records
        ]
    )


def run_ragas() -> dict[str, Any]:
    try:
        from ragas import evaluate
    except ImportError as error:
        raise RuntimeError(
            "RAGAS is not installed. Uncomment ragas in requirements.txt, then run pip install -r requirements.txt."
        ) from error

    records = build_rag_records(
        use_cache=False,
        k=int(os.getenv("RAGAS_RETRIEVAL_K", "5")),
    )
    metrics = import_ragas_metrics()
    if not metrics:
        raise RuntimeError("No compatible RAGAS metrics were found in the installed ragas version.")

    result = evaluate(
        dataset=build_ragas_dataset(records),
        metrics=metrics,
        llm=build_judge_llm(),
        embeddings=build_judge_embeddings(),
    )

    try:
        dataframe = result.to_pandas()
        csv_path = Path(RESULTS_DIR) / "ragas_results.csv"
        json_path = Path(RESULTS_DIR) / "ragas_results.json"
        dataframe.to_csv(csv_path, index=False)
        json_path.write_text(dataframe.to_json(orient="records", indent=2), encoding="utf-8")
        summary = dataframe.mean(numeric_only=True).to_dict()
    except Exception:
        summary = dict(result)

    summary["metrics_requested"] = [
        getattr(metric, "name", metric.__class__.__name__) for metric in metrics
    ]
    summary["num_cases"] = len(records)
    summary_path = Path(RESULTS_DIR) / "ragas_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ragas(), indent=2))
