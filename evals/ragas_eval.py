"""RAGAS evaluation for the RAG pipeline.

Evaluates all compatible RAGAS metrics against the 10-case eval dataset.

Metrics evaluated
-----------------
  Generator  : faithfulness, answer_relevancy, answer_correctness,
               answer_similarity
  Retriever  : context_precision, context_recall, context_entity_recall
  Robustness : noise_sensitivity

Field mapping (handles both RAGAS v0.1 and v0.2 SingleTurnSample API):
  v0.1  question / contexts / answer / ground_truth
  v0.2  user_input / retrieved_contexts / response / reference

Run:
    python -m evals.ragas_eval                  # full LLM eval
    RAG_EVAL_NO_LLM=1 python -m evals.ragas_eval  # retrieval-only (no Groq)
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.rag_eval_runner import RESULTS_DIR, build_rag_records

EVAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── LLM / embedding judges ────────────────────────────────────────────────────

def _judge_llm():
    load_dotenv()
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("RAGAS_GROQ_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
        temperature=0,
        max_retries=2,
        timeout=60,
    )


def _judge_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAG_EVAL_EMBEDDING_MODEL", EVAL_EMBEDDING_MODEL),
        model_kwargs={"device": os.getenv("RAG_EVAL_EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── RAGAS metric loading ──────────────────────────────────────────────────────

# Each entry: (instance_attr_name_in_ragas.metrics, class_name_in_new_API)
_METRIC_SPECS = [
    ("faithfulness",                "Faithfulness"),
    ("answer_relevancy",            "ResponseRelevancy"),
    ("answer_correctness",          "AnswerCorrectness"),
    ("answer_similarity",           "SemanticSimilarity"),
    ("context_precision",           "LLMContextPrecisionWithReference"),
    ("context_recall",              "LLMContextRecall"),
    ("context_entity_recall",       "ContextEntityRecall"),
    ("noise_sensitivity_relevant",  "NoiseSensitivity"),
]


def _load_metrics() -> list[Any]:
    """Return instantiated RAGAS metrics compatible with the installed version.

    Tries the new class-based API (v0.2+) first, then falls back to the
    pre-instantiated module-level singletons from v0.1.
    """
    import ragas.metrics as _rm

    loaded: list[Any] = []
    seen_names: set[str] = set()

    for attr_name, class_name in _METRIC_SPECS:
        metric = None

        # v0.2+: class exists → instantiate it
        cls = getattr(_rm, class_name, None)
        if cls is not None:
            try:
                metric = cls()
            except Exception:
                pass

        # v0.1 fallback: pre-instantiated singleton on the module
        if metric is None:
            metric = getattr(_rm, attr_name, None)

        if metric is None:
            print(f"  [skip] {attr_name} not available in this RAGAS version.")
            continue

        name = getattr(metric, "name", attr_name)
        if name in seen_names:
            continue
        seen_names.add(name)
        loaded.append(metric)
        print(f"  [ok]   {name}")

    return loaded


# ── Dataset builder ───────────────────────────────────────────────────────────

def _build_dataset(records: list[dict[str, Any]]):
    """Convert records into the correct RAGAS dataset format.

    Supports both RAGAS v0.1 (Dataset-based) and v0.2 (EvaluationDataset /
    SingleTurnSample-based) by detecting what is importable.
    """
    rows = [
        {
            # v0.1 field names
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["reference"],
            # v0.2 field names (same data, different keys)
            "user_input":          r["question"],
            "response":            r["answer"],
            "retrieved_contexts":  r["contexts"],
            "reference":           r["reference"],
        }
        for r in records
    ]

    # Try v0.2 EvaluationDataset + SingleTurnSample first
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        samples = [
            SingleTurnSample(
                user_input         = row["user_input"],
                response           = row["response"],
                retrieved_contexts = row["retrieved_contexts"],
                reference          = row["reference"],
            )
            for row in rows
        ]
        return EvaluationDataset(samples=samples)
    except ImportError:
        pass

    # Fall back to v0.1 HuggingFace Dataset
    from datasets import Dataset
    return Dataset.from_list(rows)


# ── Main eval ─────────────────────────────────────────────────────────────────

def run_ragas(
    k: int | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Run: pip install ragas"
        ) from exc

    k = k or int(os.getenv("RAGAS_RETRIEVAL_K", "5"))

    print("Building eval records …")
    records = build_rag_records(
        use_cache=False,
        no_llm=os.getenv("RAG_EVAL_NO_LLM", "0") == "1",
        k=k,
        force_rebuild=force_rebuild,
    )
    print(f"  {len(records)} cases ready.\n")

    print("Loading RAGAS metrics …")
    metrics = _load_metrics()
    if not metrics:
        raise RuntimeError(
            "No compatible RAGAS metrics found. Check your ragas installation."
        )
    print(f"  {len(metrics)} metrics loaded.\n")

    print("Building RAGAS dataset …")
    dataset = _build_dataset(records)

    print("Running RAGAS evaluate() — this calls the judge LLM once per metric per case …")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_judge_llm(),
        embeddings=_judge_embeddings(),
    )

    # ── persist results ───────────────────────────────────────────────────────
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    try:
        df = result.to_pandas()

        # Attach pipeline metadata columns for debugging
        meta_cols = [
            "id", "effective_topic_filter", "retrieval_queries",
            "retrieved_context_count", "unique_source_count",
            "latency_seconds", "cached", "cache_type",
        ]
        for col in meta_cols:
            vals = [r.get(col) for r in records]
            if any(v is not None for v in vals):
                df[col] = vals

        csv_path = Path(RESULTS_DIR) / "ragas_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nPer-case results → {csv_path}")

        # Per-case JSON for programmatic inspection
        per_case = df.to_dict(orient="records")
        (Path(RESULTS_DIR) / "ragas_results.json").write_text(
            json.dumps(per_case, indent=2), encoding="utf-8"
        )

        # Aggregate summary
        metric_names = [getattr(m, "name", m.__class__.__name__) for m in metrics]
        summary: dict[str, Any] = {}
        for name in metric_names:
            if name in df.columns:
                col = df[name].dropna()
                summary[name] = {
                    "mean":   round(float(col.mean()), 4),
                    "min":    round(float(col.min()),  4),
                    "max":    round(float(col.max()),  4),
                    "stddev": round(float(col.std()),  4),
                    # Flag cases that drag scores down (< 0.5)
                    "weak_cases": [
                        records[i]["id"]
                        for i, v in enumerate(df[name])
                        if v is not None and float(v) < 0.5
                    ],
                }

    except Exception:
        traceback.print_exc()
        summary = {
            getattr(m, "name", m.__class__.__name__): None for m in metrics
        }

    summary["num_cases"]         = len(records)
    summary["metrics_evaluated"] = [getattr(m, "name", m.__class__.__name__) for m in metrics]
    summary["generation_mode"]   = records[0].get("generation_mode", "llm") if records else "unknown"
    summary["retrieval_k"]       = k

    summary_path = Path(RESULTS_DIR) / "ragas_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary            → {summary_path}\n")

    # ── print table ───────────────────────────────────────────────────────────
    print("=" * 52)
    print(f"{'Metric':<35} {'Mean':>6}  {'Min':>6}  {'Max':>6}")
    print("-" * 52)
    for name, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        print(f"{name:<35} {stats['mean']:>6.3f}  {stats['min']:>6.3f}  {stats['max']:>6.3f}")
        if stats["weak_cases"]:
            print(f"  ↳ weak (<0.5): {', '.join(stats['weak_cases'])}")
    print("=" * 52)
    print(f"Cases: {len(records)}  |  Metrics: {len(metrics)}  |  k={k}")

    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline.")
    p.add_argument("--k",      type=int, default=None, help="Retrieval k (default: RAGAS_RETRIEVAL_K env or 5).")
    p.add_argument("--rebuild", action="store_true",   help="Force-rebuild eval records even if cached.")
    args = p.parse_args()
    print(json.dumps(run_ragas(k=args.k, force_rebuild=args.rebuild), indent=2))