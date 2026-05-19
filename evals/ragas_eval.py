"""Evaluate the RAG pipeline with RAGAS metrics.

Run:
    py -m evals.ragas_eval
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.rag_eval_runner import RESULTS_DIR, build_rag_records
from rag_agent import DEFAULT_EMBEDDING_MODEL


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
        model_name=os.getenv("RAG_EVAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        encode_kwargs={"normalize_embeddings": True},
    )


def import_ragas_metrics() -> list[Any]:
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        return [faithfulness, answer_relevancy, context_precision, context_recall]
    except ImportError:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        return [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
            LLMContextRecall(),
        ]


def run_ragas() -> dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as error:
        raise RuntimeError("RAGAS is not installed. Run: pip install -r requirements-eval.txt") from error

    records = build_rag_records(use_cache=True)
    dataset = Dataset.from_list(
        [
            {
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["ground_truth"],
                "reference": record["reference"],
            }
            for record in records
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=import_ragas_metrics(),
        llm=build_judge_llm(),
        embeddings=build_judge_embeddings(),
    )

    try:
        dataframe = result.to_pandas()
        output_path = Path(RESULTS_DIR) / "ragas_results.csv"
        dataframe.to_csv(output_path, index=False)
        summary = dataframe.mean(numeric_only=True).to_dict()
    except Exception:
        summary = dict(result)

    summary_path = Path(RESULTS_DIR) / "ragas_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_ragas(), indent=2))
