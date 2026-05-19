"""Shared RAG evaluation runner used by DeepEval, RAGAS, and smoke checks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from evals.eval_dataset import RAG_EVAL_CASES
from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_URL,
    DEFAULT_VECTOR_DB_DIR,
    index_source,
    load_vector_store,
    run_direct_rag,
    retrieve_context_documents,
    vector_store_has_documents,
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
    url: str = DEFAULT_URL,
    selector: str | None = "post-title,post-header,post-content",
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
) -> Any:
    vector_store = load_vector_store(collection_name, persist_directory)
    if not vector_store_has_documents(vector_store):
        vector_store = index_source(url, selector, collection_name, persist_directory)
    return vector_store


def build_rag_records(
    output_path: str | Path = RESULTS_DIR / "rag_eval_records.json",
    url: str = DEFAULT_URL,
    selector: str | None = "post-title,post-header,post-content",
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
    use_cache: bool = True,
    no_llm: bool | None = None,
) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    if use_cache and output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))

    if no_llm is None:
        no_llm = os.getenv("RAG_EVAL_NO_LLM", "0") == "1"

    vector_store = ensure_indexed(url, selector, collection_name, persist_directory)
    records: list[dict[str, Any]] = []

    for case in RAG_EVAL_CASES:
        started = time.perf_counter()
        if no_llm:
            docs = retrieve_context_documents(vector_store, case["question"], k=3)
            rag_result = {
                "answer": lexical_answer_from_context(
                    case["question"],
                    [doc.page_content for doc in docs],
                    case["reference"],
                ),
                "contexts": [doc.page_content for doc in docs],
                "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            }
        else:
            rag_result = run_direct_rag(case["question"], vector_store, k=3)
        latency = time.perf_counter() - started
        records.append(
            {
                "question": case["question"],
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
                "ground_truth": case["reference"],
                "reference": case["reference"],
                "sources": rag_result["sources"],
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
