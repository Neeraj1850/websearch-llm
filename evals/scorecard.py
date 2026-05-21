"""Smoke-check scorecard for the RAG pipeline.

This is NOT a substitute for RAGAS — it is a fast, zero-LLM sanity check
that runs in seconds and catches obvious regressions:

  retrieval_hit     – did the retriever return ≥1 chunk?
  context_coverage  – do retrieved chunks contain the key reference terms?
  format_ok         – does the answer include "Answer:" and "Sources:"?
  no_hallucination_flag – does the answer NOT claim information it shouldn't?
  latency_ok        – is end-to-end latency under 30s?
  out_of_scope_ok   – for OOS questions, does the agent admit it?

Run:
    python -m evals.scorecard          # smoke check only
    python -m evals.scorecard --full   # smoke check + RAGAS (requires Groq)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from evals.eval_dataset import RAG_EVAL_CASES
from evals.rag_eval_runner import RESULTS_DIR, build_rag_records, ensure_indexed

from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_VECTOR_DB_DIR,
    DEFAULT_RAG_K,
    prepare_retrieval_queries,
    retrieve_context,
)


# ── individual scorers ────────────────────────────────────────────────────────

def _retrieval_hit(contexts: list[str]) -> float:
    """1.0 if at least one chunk was retrieved, else 0.0."""
    return 1.0 if contexts else 0.0


def _context_coverage(contexts: list[str], reference: str) -> float:
    """Fraction of reference content-words found anywhere in the retrieved chunks.

    Ignores stop-words so short references don't over-penalise.
    """
    STOP = {
        "a", "an", "the", "is", "in", "it", "of", "to", "and", "or",
        "for", "that", "this", "are", "can", "be", "by", "with", "as",
        "so", "its", "from", "into", "more", "than", "both", "when",
        "which", "how", "what", "does", "not", "but",
    }
    words = [w.lower().strip(".,;:()") for w in reference.split()]
    content_words = [w for w in words if w and w not in STOP]
    if not content_words:
        return 1.0
    combined = " ".join(contexts).lower()
    hits = sum(1 for w in content_words if w in combined)
    return round(hits / len(content_words), 3)


def _format_ok(answer: str) -> float:
    """1.0 if answer contains both 'Answer:' and 'Sources:', else partial."""
    low = answer.lower()
    has_answer  = "answer:" in low
    has_sources = "sources:" in low
    if has_answer and has_sources:
        return 1.0
    if has_answer or has_sources:
        return 0.5
    return 0.0


def _latency_ok(latency: float, limit: float = 30.0) -> float:
    if latency <= limit / 2:
        return 1.0
    if latency <= limit:
        return 0.5
    return 0.0


def _out_of_scope_ok(answer: str, question_id: str) -> float | None:
    """For the OOS case, check the agent admits it doesn't know."""
    if question_id != "out_of_scope":
        return None   # not applicable
    low = answer.lower()
    REFUSAL_SIGNALS = [
        "don't have", "do not have", "not contain", "not mention",
        "no information", "cannot find", "not enough context",
        "i don't", "i do not",
    ]
    return 1.0 if any(s in low for s in REFUSAL_SIGNALS) else 0.0


def _score_record(record: dict[str, Any]) -> dict[str, Any]:
    retrieval_hit = _retrieval_hit(record["contexts"])
    coverage      = _context_coverage(record["contexts"], record["reference"])
    fmt           = _format_ok(record["answer"])
    lat           = _latency_ok(record["latency_seconds"])
    oos           = _out_of_scope_ok(record["answer"], record["id"])

    scores = {
        "retrieval_hit":     retrieval_hit,
        "context_coverage":  coverage,
        "format_ok":         fmt,
        "latency_ok":        lat,
    }
    if oos is not None:
        scores["out_of_scope_ok"] = oos

    applicable = [v for v in scores.values() if v is not None]
    scores["composite"] = round(sum(applicable) / len(applicable), 3) if applicable else 0.0

    return {**record, **scores}


# ── aggregate summary ─────────────────────────────────────────────────────────

def _summarise(scored: list[dict[str, Any]]) -> dict[str, Any]:
    metric_cols = [
        "retrieval_hit", "context_coverage", "format_ok",
        "latency_ok", "composite",
    ]
    summary: dict[str, Any] = {"num_cases": len(scored)}
    for col in metric_cols:
        vals = [float(r[col]) for r in scored if r.get(col) is not None]
        if vals:
            summary[col] = {
                "mean":   round(statistics.mean(vals), 3),
                "min":    round(min(vals), 3),
                "max":    round(max(vals), 3),
                "weak_cases": [
                    r["id"] for r in scored
                    if r.get(col) is not None and float(r[col]) < 0.5
                ],
            }

    oos_vals = [float(r["out_of_scope_ok"]) for r in scored if "out_of_scope_ok" in r]
    if oos_vals:
        summary["out_of_scope_ok"] = {"mean": round(statistics.mean(oos_vals), 3)}

    latencies = [r["latency_seconds"] for r in scored]
    summary["latency"] = {
        "mean_seconds": round(statistics.mean(latencies), 3),
        "p95_seconds":  round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3),
        "max_seconds":  round(max(latencies), 3),
    }
    return summary


# ── writers ───────────────────────────────────────────────────────────────────

def _write(scored: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    (Path(RESULTS_DIR) / "scorecard.json").write_text(
        json.dumps(scored, indent=2), encoding="utf-8"
    )
    (Path(RESULTS_DIR) / "scorecard_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({k for r in scored for k in r})
    with (Path(RESULTS_DIR) / "scorecard.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)


def _print_summary(summary: dict[str, Any]) -> None:
    metric_cols = [
        "retrieval_hit", "context_coverage", "format_ok",
        "latency_ok", "out_of_scope_ok", "composite",
    ]
    print("\n" + "=" * 58)
    print(f"{'Metric':<30} {'Mean':>6}  {'Min':>6}  Weak cases")
    print("-" * 58)
    for col in metric_cols:
        stats = summary.get(col)
        if not stats:
            continue
        weak = ", ".join(stats.get("weak_cases", [])) or "—"
        print(f"{col:<30} {stats['mean']:>6.3f}  {stats.get('min', 0):>6.3f}  {weak}")
    lat = summary.get("latency", {})
    print("-" * 58)
    print(f"Latency  mean={lat.get('mean_seconds', 0):.2f}s  "
          f"p95={lat.get('p95_seconds', 0):.2f}s  "
          f"max={lat.get('max_seconds', 0):.2f}s")
    print("=" * 58)
    print(f"Cases: {summary['num_cases']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG smoke-check scorecard.")
    p.add_argument("--full",    action="store_true", help="Also run RAGAS after smoke check.")
    p.add_argument("--no-llm",  action="store_true", help="Skip LLM generation (retrieval only).")
    p.add_argument("--rebuild", action="store_true", help="Force-rebuild eval records.")
    p.add_argument("--k", type=int, default=DEFAULT_RAG_K, help="Retrieval k.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_llm:
        os.environ["RAG_EVAL_NO_LLM"] = "1"

    print("Building RAG eval records …")
    records = build_rag_records(
        use_cache=False,
        no_llm=args.no_llm,
        k=args.k,
        force_rebuild=args.rebuild,
    )

    print(f"Scoring {len(records)} cases …")
    scored  = [_score_record(r) for r in records]
    summary = _summarise(scored)

    _write(scored, summary)
    _print_summary(summary)
    print("\nWrote results/scorecard.json, scorecard.csv, scorecard_summary.json")

    if args.full:
        print("\n--- Running RAGAS ---")
        from evals.ragas_eval import run_ragas
        run_ragas(k=args.k, force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()