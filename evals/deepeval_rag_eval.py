"""Evaluate the RAG pipeline with DeepEval metrics.

Run:
    py -m evals.deepeval_rag_eval

This uses Groq as the judge model through a DeepEval custom LLM wrapper.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.rag_eval_runner import RESULTS_DIR, build_rag_records


class GroqDeepEvalLLM:
    """DeepEval custom model wrapper backed by LangChain ChatGroq."""

    def __init__(self) -> None:
        from deepeval.models import DeepEvalBaseLLM
        from langchain_groq import ChatGroq

        class _GroqDeepEvalLLM(DeepEvalBaseLLM):
            def __init__(self) -> None:
                load_dotenv()
                self.model = ChatGroq(
                    model=os.getenv("DEEPEVAL_GROQ_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
                    temperature=0,
                    max_retries=2,
                    timeout=60,
                )

            def load_model(self):
                return self.model

            def generate(self, prompt: str, schema: Any | None = None):
                time.sleep(float(os.getenv("DEEPEVAL_THROTTLE_SECONDS", "4")))
                model = self.load_model()
                if schema is not None:
                    structured_model = model.with_structured_output(schema)
                    return structured_model.invoke(prompt)
                response = model.invoke(prompt)
                return response.content if isinstance(response.content, str) else str(response.content)

            async def a_generate(self, prompt: str, schema: Any | None = None):
                return await asyncio.to_thread(self.generate, prompt, schema)

            def get_model_name(self) -> str:
                return os.getenv("DEEPEVAL_GROQ_MODEL", os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

        self.instance = _GroqDeepEvalLLM()


def run_deepeval() -> dict[str, Any]:
    try:
        from deepeval import evaluate
        from deepeval.evaluate import AsyncConfig
        from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as error:
        raise RuntimeError(
            "DeepEval is not installed. Run: pip install -r requirements-eval.txt"
        ) from error

    records = build_rag_records(use_cache=True)
    judge = GroqDeepEvalLLM().instance

    test_cases = [
        LLMTestCase(
            input=record["question"],
            actual_output=record["answer"],
            expected_output=record["ground_truth"],
            retrieval_context=record["contexts"],
        )
        for record in records
    ]

    metrics = [
        AnswerRelevancyMetric(model=judge, threshold=0.5, async_mode=False),
        FaithfulnessMetric(model=judge, threshold=0.5, async_mode=False),
        ContextualRelevancyMetric(model=judge, threshold=0.5, async_mode=False),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=False,
            max_concurrent=1,
            throttle_value=int(os.getenv("DEEPEVAL_THROTTLE_SECONDS", "4")),
        ),
    )
    summary = {
        "num_cases": len(test_cases),
        "metrics": [metric.__class__.__name__ for metric in metrics],
        "note": "See DeepEval console output for detailed scores.",
    }
    output_path = Path(RESULTS_DIR) / "deepeval_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"raw_result": result, "summary": summary}


if __name__ == "__main__":
    output = run_deepeval()
    print(json.dumps(output["summary"], indent=2))
