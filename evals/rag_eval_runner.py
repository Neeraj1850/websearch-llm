"""Shared RAG evaluation runner.

Builds one record per eval case containing everything RAGAS, DeepEval,
and the smoke-check scorecard need:

  question / answer / contexts / ground_truth / reference / sources /
  retrieval_queries / latency_seconds / cached / cache_type / cache_score /
  retrieved_context_count / unique_source_count / effective_topic_filter

Run standalone to pre-build records without scoring:
    python -m evals.rag_eval_runner
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from evals.eval_dataset import RAG_EVAL_CASES

from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_VECTOR_DB_DIR,
    DEFAULT_WEB_LINKS,
    index_sources,
    load_vector_store,
    prepare_retrieval_queries,
    retrieve_context,
    run_rag,
    get_raw_embeddings,
    _vector_store_is_empty,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _lexical_answer(contexts: list[str], reference: str) -> str:
    """Fallback answer when LLM generation is skipped (RAG_EVAL_NO_LLM=1).

    Uses the reference directly so RAGAS answer_correctness / answer_similarity
    still get a meaningful signal even without a live LLM call.
    """
    preview = " ".join(" ".join(c.split()) for c in contexts)[:400]
    return (
        f"Answer: {reference}\n"
        f"Sources: Retrieved context from indexed document\n"
        f"Rationale: LLM generation skipped; context preview: {preview}"
    )


def ensure_indexed(
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
) -> Any:
    """Return the vector store, indexing from DEFAULT_WEB_LINKS if empty."""
    vs = load_vector_store(collection_name, persist_directory)
    if _vector_store_is_empty(vs):
        vs = index_sources(DEFAULT_WEB_LINKS, collection_name, persist_directory,
                           show_progress=True)
    return vs


# ── main builder ──────────────────────────────────────────────────────────────

def build_rag_records(
    output_path: str | Path = RESULTS_DIR / "rag_eval_records.json",
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
    use_cache: bool = True,
    no_llm: bool | None = None,
    k: int = 5,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Return eval records, reading from disk cache unless force_rebuild=True.

    Parameters
    ----------
    use_cache:
        If True and output_path already exists, return the cached records
        (skips re-running the RAG pipeline).  Set to False for RAGAS eval
        so every run reflects the current pipeline state.
    no_llm:
        Skip LLM generation and use the reference as the answer.  Useful
        for fast retrieval-only checks.  Defaults to RAG_EVAL_NO_LLM env var.
    k:
        Number of chunks to retrieve per question.
    force_rebuild:
        Ignore any cached records on disk and always re-run.
    """
    # Trigger model load with visible feedback before any silent blocking call.
    print("  Loading embedding model (may take 20-30s on first run) ...", flush=True)
    get_raw_embeddings()
    print("  Embedding model ready.", flush=True)

    output_path = Path(output_path)
    if use_cache and not force_rebuild and output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))

    if no_llm is None:
        no_llm = os.getenv("RAG_EVAL_NO_LLM", "0") == "1"

    vs = ensure_indexed(collection_name, persist_directory)
    records: list[dict[str, Any]] = []

    for case in RAG_EVAL_CASES:
        t0 = time.perf_counter()
        topic_filter  = case.get("topic_filter")
        domain_filter = case.get("domain_filter")

        # Always run retrieval so context metrics are populated regardless of no_llm.
        retrieval_queries, detected_topic = prepare_retrieval_queries(case["question"])
        effective_topic = topic_filter or detected_topic

        if no_llm:
            docs = retrieve_context(
                vs, retrieval_queries,
                k=k,
                topic=effective_topic,
                domain=domain_filter,
            )
            answer   = _lexical_answer([d.page_content for d in docs], case["reference"])
            contexts = [d.page_content for d in docs]
            sources  = [d.metadata.get("source", "unknown") for d in docs]
            cached, cache_type, cache_score = False, None, 0.0
        else:
            result = run_rag(
                case["question"], vs,
                collection_name=collection_name,
                k=k,
                topic_filter=topic_filter,
                domain_filter=domain_filter,
                # Never read from the response cache during eval — we want fresh
                # pipeline output so scores reflect current behaviour.
                use_cache=False,
            )
            answer            = result["answer"]
            contexts          = result["contexts"]
            sources           = result["sources"]
            retrieval_queries = result.get("retrieval_queries", retrieval_queries)
            effective_topic   = result.get("topic_filter", effective_topic)
            cached            = result.get("cached", False)
            cache_type        = result.get("cache_type")
            cache_score       = result.get("cache_score", 0.0)

        records.append({
            # ── identity ──────────────────────────────────────────────────
            "id":       case["id"],
            "question": case["question"],
            # ── pipeline output ───────────────────────────────────────────
            "answer":   answer,
            "contexts": contexts,
            "sources":  sources,
            # ── ground truth (both names for RAGAS v0.1 and v0.2 compat) ──
            "ground_truth": case["reference"],
            "reference":    case["reference"],
            # ── retrieval metadata ────────────────────────────────────────
            "topic_filter":          topic_filter,
            "domain_filter":         domain_filter,
            "effective_topic_filter": effective_topic,
            "retrieval_queries":     retrieval_queries,
            # ── cache metadata ────────────────────────────────────────────
            "cached":      cached,
            "cache_type":  cache_type,
            "cache_score": cache_score,
            # ── stats ─────────────────────────────────────────────────────
            "retrieved_context_count": len(contexts),
            "unique_source_count":     len(set(sources)),
            "latency_seconds":         round(time.perf_counter() - t0, 3),
            "generation_mode":         "no_llm" if no_llm else "llm",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    recs = build_rag_records(force_rebuild=True)
    print(f"Built {len(recs)} records → {RESULTS_DIR / 'rag_eval_records.json'}")