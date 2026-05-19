"""Consolidated scorecard for the tool agent and RAG agent.

Run:
    python -m evals.scorecard
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from agent import ask_agent
from evals.scorecard_dataset import RAG_SCORECARD_CASES, TOOL_AGENT_CASES
from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_URL,
    DEFAULT_VECTOR_DB_DIR,
    ask_rag_agent,
    load_vector_store,
    retrieve_context_documents,
    vector_store_has_documents,
)


RESULTS_DIR = Path("results")
GROQ_FREE_TIER_LIMITS = {
    "llama-3.1-8b-instant": {
        "rpm": 30,
        "rpd": 14400,
        "tpm": 6000,
        "tpd": 500000,
        "recommended_delay_seconds": 8.0,
        "recommended_max_cases": 4,
    },
    "llama-3.3-70b-versatile": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 12000,
        "tpd": 100000,
        "recommended_delay_seconds": 12.0,
        "recommended_max_cases": 2,
    },
}
DEFAULT_EVAL_MODEL = "llama-3.1-8b-instant"


def contains_any(answer: str, terms: list[str]) -> bool:
    lowered = answer.lower()
    return any(term.lower() in lowered for term in terms)


def source_score(answer: str) -> float:
    lowered = answer.lower()
    if "sources:" not in lowered:
        return 0.0
    if "http" in lowered or "source:" in lowered or "recent conversation memory" in lowered:
        return 1.0
    if "no external source used" in lowered or "no reliable source" in lowered:
        return 0.7
    return 0.5


def format_score(answer: str) -> float:
    lowered = answer.lower()
    required = ["answer:", "sources:", "rationale:"]
    return sum(1 for item in required if item in lowered) / len(required)


def latency_score(latency_seconds: float, target_seconds: float = 10.0) -> float:
    if latency_seconds <= target_seconds:
        return 1.0
    if latency_seconds <= target_seconds * 2:
        return 0.5
    return 0.0


def tool_efficiency_score(tool_calls: list[str], expected_tool: str | None = None) -> float:
    real_calls = [call for call in tool_calls if call != "no_tool"]
    if expected_tool == "no_tool":
        return 1.0 if not real_calls else 0.0
    if len(real_calls) <= 1:
        return 1.0
    if len(real_calls) == 2:
        return 0.5
    return 0.0


def expected_tool_score(tool_calls: list[str], expected_tool: str) -> float:
    if expected_tool == "no_tool":
        return 1.0 if tool_calls == ["no_tool"] else 0.0
    return 1.0 if any(call.startswith(expected_tool) for call in tool_calls) else 0.0


def error_handling_score(answer: str) -> float:
    lowered = answer.lower()
    if "traceback" in lowered or "exception" in lowered:
        return 0.0
    if "failed" in lowered or "error" in lowered:
        return 0.5
    return 1.0


def is_groq_rate_limit_error(error: Exception | str) -> bool:
    text = str(error).lower()
    return "ratelimiterror" in text or "rate limit" in text or "rate_limit_exceeded" in text


def extract_retry_after(error: Exception | str) -> str:
    text = str(error)
    match = re.search(r"try again in ([^.]+(?:\.\d+)?s|[^.]+?m[^.]+?s)", text, re.IGNORECASE)
    return match.group(1) if match else "the reset window shown by Groq"


def sanitize_error(error: Exception | str) -> str:
    text = str(error)
    if is_groq_rate_limit_error(text):
        return f"Groq rate limit exceeded. Retry after {extract_retry_after(text)}."
    return f"{error.__class__.__name__}: {error}" if isinstance(error, Exception) else text


def rate_limit_record(
    agent: str,
    case: dict[str, Any],
    latency: float,
    error: Exception,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "agent": agent,
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "answer": (
            "Skipped: Groq daily or per-minute rate limit was reached. "
            "This case was not scored. Wait for reset, reduce max cases, or use Ollama for RAG."
        ),
        "tool_calls": [],
        "expected_tool": case.get("expected_tool", "retrieve_context"),
        "latency_seconds": round(latency, 3),
        "status": "skipped_rate_limit",
        "tool_selection_score": None,
        "content_match_score": None,
        "format_score": None,
        "source_score": None,
        "tool_efficiency_score": None,
        "latency_score": latency_score(latency),
        "error_handling_score": 1.0,
        "error": sanitize_error(error),
    }
    if extra:
        record.update(extra)
    return record


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def run_tool_agent_cases(skip_live: bool, max_cases: int | None, delay_seconds: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    thread_id = "scorecard"
    executed = 0

    for case in TOOL_AGENT_CASES:
        if skip_live and case["category"] in {"current", "research", "factual"}:
            continue
        if max_cases is not None and executed >= max_cases:
            break

        started = time.perf_counter()
        try:
            if skip_live:
                answer = (
                    "Answer: Neeraj is testing this agent.\n"
                    "Sources: No external source used\n"
                    "Rationale: This deterministic skip-live answer validates memory/no-tool scoring without using Groq."
                )
                tool_calls = ["no_tool"]
                latency = time.perf_counter() - started
            else:
                result = ask_agent(case["question"], thread_id=thread_id)
                answer = result["answer"]
                tool_calls = result["tool_calls"]
                latency = result["latency_seconds"]
            error = ""
        except Exception as exc:
            latency = time.perf_counter() - started
            if is_groq_rate_limit_error(exc):
                records.append(rate_limit_record("tool_agent", case, latency, exc))
                break

            answer = f"Exception: {exc}"
            tool_calls = []
            error = f"{exc.__class__.__name__}: {exc}"

        records.append(
            {
                "agent": "tool_agent",
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "answer": answer,
                "tool_calls": tool_calls,
                "expected_tool": case["expected_tool"],
                "latency_seconds": round(latency, 3),
                "tool_selection_score": expected_tool_score(tool_calls, case["expected_tool"]),
                "content_match_score": 1.0 if contains_any(answer, case["must_contain_any"]) else 0.0,
                "format_score": format_score(answer),
                "source_score": source_score(answer),
                "tool_efficiency_score": tool_efficiency_score(tool_calls, case["expected_tool"]),
                "latency_score": latency_score(latency),
                "error_handling_score": error_handling_score(answer),
                "error": error,
                "status": "completed" if not error else "failed",
            }
        )
        executed += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return records


def run_rag_cases(skip_live: bool, max_cases: int | None, delay_seconds: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    vector_store = load_vector_store(DEFAULT_COLLECTION, DEFAULT_VECTOR_DB_DIR)
    if skip_live and not vector_store_has_documents(vector_store):
        return records
    executed = 0

    for case in RAG_SCORECARD_CASES:
        if max_cases is not None and executed >= max_cases:
            break
        started = time.perf_counter()
        try:
            if skip_live:
                docs = retrieve_context_documents(vector_store, case["question"], k=3)
                contexts = [doc.page_content for doc in docs]
                sources = [doc.metadata.get("source", "unknown") for doc in docs]
                preview = " ".join(" ".join(context.split()) for context in contexts)[:400]
                answer = (
                    f"Answer: Retrieved relevant context for offline inspection. {preview}\n"
                    f"Sources: {', '.join(sorted(set(sources))) if sources else 'No external source used'}\n"
                    "Rationale: This skip-live mode validates retrieval without using Groq tokens."
                )
                tool_calls = [f"retrieve_context({case['question']!r})"]
                latency = time.perf_counter() - started
            else:
                result = ask_rag_agent(
                    case["question"],
                    DEFAULT_URL,
                    "post-title,post-header,post-content",
                    DEFAULT_COLLECTION,
                    DEFAULT_VECTOR_DB_DIR,
                    auto_index=True,
                )
                answer = result["answer"]
                tool_calls = result["tool_calls"]
                latency = result["latency_seconds"]
                contexts = result.get("contexts", [])
                sources = result.get("sources", [])
            error = ""
        except Exception as exc:
            latency = time.perf_counter() - started
            if is_groq_rate_limit_error(exc):
                records.append(
                    rate_limit_record(
                        "rag_agent",
                        case,
                        latency,
                        exc,
                        extra={
                            "retrieved_context_count": 0,
                            "unique_source_count": 0,
                            "avg_context_chars": 0,
                        },
                    )
                )
                break

            answer = f"Exception: {exc}"
            tool_calls = []
            contexts = []
            sources = []
            error = f"{exc.__class__.__name__}: {exc}"

        records.append(
            {
                "agent": "rag_agent",
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "answer": answer,
                "tool_calls": tool_calls,
                "expected_tool": "retrieve_context",
                "latency_seconds": round(latency, 3),
                "tool_selection_score": 1.0 if tool_calls else 0.0,
                "content_match_score": 1.0 if contains_any(answer, case["must_contain_any"]) else 0.0,
                "format_score": format_score(answer),
                "source_score": source_score(answer),
                "tool_efficiency_score": tool_efficiency_score(tool_calls),
                "latency_score": latency_score(latency),
                "error_handling_score": error_handling_score(answer),
                "retrieved_context_count": len(contexts),
                "unique_source_count": len(set(sources)),
                "avg_context_chars": round(statistics.mean([len(context) for context in contexts]), 1)
                if contexts
                else 0,
                "error": error,
                "status": "completed" if not error else "failed",
            }
        )
        executed += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "tool_selection_score",
        "content_match_score",
        "format_score",
        "source_score",
        "tool_efficiency_score",
        "latency_score",
        "error_handling_score",
    ]
    by_agent: dict[str, Any] = {}
    for agent_name in sorted(set(record["agent"] for record in records)):
        agent_records = [record for record in records if record["agent"] == agent_name]
        scored_records = [record for record in agent_records if record.get("status") != "skipped_rate_limit"]
        by_agent[agent_name] = {
            "cases": len(agent_records),
            "scored_cases": len(scored_records),
            "skipped_rate_limit": sum(
                1 for record in agent_records if record.get("status") == "skipped_rate_limit"
            ),
            "avg_latency_seconds": round(
                statistics.mean(record["latency_seconds"] for record in agent_records), 3
            ),
            "p95_latency_seconds": round(
                sorted(record["latency_seconds"] for record in agent_records)[
                    max(0, int(len(agent_records) * 0.95) - 1)
                ],
                3,
            ),
            **{
                metric: average(
                    [
                        float(record[metric])
                        for record in scored_records
                        if record.get(metric) is not None
                    ]
                )
                for metric in metric_names
            },
        }

    scored_records = [record for record in records if record.get("status") != "skipped_rate_limit"]
    overall = {
        "cases": len(records),
        "scored_cases": len(scored_records),
        "skipped_rate_limit": sum(1 for record in records if record.get("status") == "skipped_rate_limit"),
        "avg_latency_seconds": round(statistics.mean(record["latency_seconds"] for record in records), 3)
        if records
        else 0,
        **{
            metric: average(
                [float(record[metric]) for record in scored_records if record.get(metric) is not None]
            )
            for metric in metric_names
        },
    }
    return {"overall": overall, "by_agent": by_agent}


def write_results(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "scorecard.json"
    csv_path = RESULTS_DIR / "scorecard.csv"
    summary_path = RESULTS_DIR / "scorecard_summary.json"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = sorted({key for record in records for key in record.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a consolidated agent/RAG scorecard.")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Avoid network/Groq-heavy cases where possible.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EVAL_MODEL,
        help="Groq model to use for the scorecard. Defaults to the higher-quota free-tier 8B model.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum cases per agent. Defaults to a Groq-free-tier-safe value for the selected model.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help="Delay between cases. Defaults to a Groq-free-tier-safe value for the selected model.",
    )
    parser.add_argument(
        "--no-force-model",
        action="store_true",
        help="Do not override GROQ_MODEL for this scorecard run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model
    if not args.no_force_model:
        os.environ["GROQ_MODEL"] = model
    limits = GROQ_FREE_TIER_LIMITS.get(model, GROQ_FREE_TIER_LIMITS[DEFAULT_EVAL_MODEL])
    max_cases = args.max_cases if args.max_cases is not None else limits["recommended_max_cases"]
    delay_seconds = (
        args.delay_seconds
        if args.delay_seconds is not None
        else float(limits["recommended_delay_seconds"])
    )

    print(
        "Groq-aware scorecard plan: "
        f"model={model}, free-tier limits={limits}, "
        f"max_cases_per_agent={max_cases}, delay_seconds={delay_seconds}"
    )

    records = run_tool_agent_cases(args.skip_live, max_cases=max_cases, delay_seconds=delay_seconds)
    records.extend(run_rag_cases(args.skip_live, max_cases=max_cases, delay_seconds=delay_seconds))
    summary = summarize(records)
    write_results(records, summary)
    print_summary(summary)
    print("\nWrote results/scorecard.json, results/scorecard.csv, and results/scorecard_summary.json")


if __name__ == "__main__":
    main()
