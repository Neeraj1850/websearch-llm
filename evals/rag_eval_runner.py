"""Shared RAG evaluation runner used by DeepEval, RAGAS, and smoke checks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from evals.eval_dataset import RAG_EVAL_CASES
from langchain_compat import install_cachebacked_embeddings_shim

install_cachebacked_embeddings_shim()

from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_VECTOR_DB_DIR,
    DEFAULT_WEB_LINKS,
    index_sources,
    load_vector_store,
    prepare_retrieval_queries,
    retrieve_context,
    run_rag,
    _vector_store_is_empty,
)


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def lexical_answer_from_context(question: str, contexts: list[str], reference: str) -> str:
    """Token-free fallback answer for eval plumbing when LLM quota is exhausted."""
    context_preview = " ".join(" ".join(context.split()) for context in contexts)[:600]
    return (
        f"Answer: {reference}\n"
        f"Sources: Retrieved context from indexed document\n"
        f"Rationale: LLM generation was skipped; retrieved context preview: {context_preview}"
    )


def ensure_indexed(
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
) -> Any:
    vector_store = load_vector_store(collection_name, persist_directory)
    if _vector_store_is_empty(vector_store):
        vector_store = index_sources(DEFAULT_WEB_LINKS, collection_name, persist_directory)
    return vector_store


def build_rag_records(
    output_path: str | Path = RESULTS_DIR / "rag_eval_records.json",
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
    use_cache: bool = True,
    no_llm: bool | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    if use_cache and output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))

    if no_llm is None:
        no_llm = os.getenv("RAG_EVAL_NO_LLM", "0") == "1"

    vector_store = ensure_indexed(collection_name, persist_directory)
    records: list[dict[str, Any]] = []

    for case in RAG_EVAL_CASES:
        started = time.perf_counter()
        topic_filter = case.get("topic_filter")
        domain_filter = case.get("domain_filter")
        retrieval_queries, detected_topic = prepare_retrieval_queries(case["question"])
        effective_topic = topic_filter or detected_topic
        if no_llm:
            docs = retrieve_context(
                vector_store,
                retrieval_queries,
                k=k,
                topic=effective_topic,
                domain=domain_filter,
            )
            rag_result = {
                "answer": lexical_answer_from_context(
                    case["question"],
                    [doc.page_content for doc in docs],
                    case["reference"],
                ),
                "contexts": [doc.page_content for doc in docs],
                "sources": [doc.metadata.get("source", "unknown") for doc in docs],
                "retrieval_queries": retrieval_queries,
                "topic_filter": effective_topic,
                "cached": False,
                "cache_type": None,
                "cache_score": 0.0,
            }
        else:
            rag_result = run_rag(
                case["question"],
                vector_store,
                collection_name=collection_name,
                k=k,
                topic_filter=topic_filter,
                domain_filter=domain_filter,
                use_cache=os.getenv("RAGAS_USE_RESPONSE_CACHE", "0") == "1",
            )
            retrieval_queries = rag_result.get("retrieval_queries", retrieval_queries)
            effective_topic = rag_result.get("topic_filter", effective_topic)
        latency = time.perf_counter() - started
        records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
                "ground_truth": case["reference"],
                "reference": case["reference"],
                "sources": rag_result["sources"],
                "topic_filter": topic_filter,
                "effective_topic_filter": effective_topic,
                "domain_filter": domain_filter,
                "retrieval_queries": retrieval_queries,
                "cached": rag_result.get("cached", False),
                "cache_type": rag_result.get("cache_type"),
                "cache_score": rag_result.get("cache_score", 0.0),
                "retrieved_context_count": len(rag_result["contexts"]),
                "unique_source_count": len(set(rag_result["sources"])),
                "latency_seconds": round(latency, 3),
                "generation_mode": "no_llm_reference_fallback" if no_llm else "llm",
            }
        )

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    records = build_rag_records()
    print(f"Wrote {len(records)} records to {RESULTS_DIR / 'rag_eval_records.json'}")
