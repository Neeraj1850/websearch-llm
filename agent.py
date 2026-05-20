"""Groq-powered LangChain agent with free research/search tools.

Run:
    python agent.py "Find recent papers about retrieval augmented generation."
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

try:
    from langchain_classic.agents import AgentExecutor, create_react_agent
except ImportError:  # Older LangChain installs expose these from langchain.agents.
    from langchain.agents import AgentExecutor, create_react_agent


DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_WEB_RESULTS = 3
MAX_WIKI_RESULTS = 2
MAX_ARXIV_RESULTS = 3
MEMORY_PATH = Path(".memory") / "short_term_memory.json"
DEFAULT_THREAD_ID = "default"
MAX_MEMORY_MESSAGES = 6
DEFAULT_MAX_ITERATIONS = 3

REACT_PROMPT = """Answer the following question as best you can. You have access to the following tools:

{tools}

Context:
- Current date: {current_date}
- Recent conversation memory:
{chat_history}

Tool selection guidance:
- Use web_search for current events, recent or latest facts, news, sports results, live or time-sensitive information.
- Use wikipedia_search for stable facts, definitions, history, biographies, places, concepts, and background knowledge.
- Use arxiv_search for academic papers, research literature, abstracts, authors, technical methods, scientific studies, ML/AI papers, and scholarly topics.
- If the answer is conversational or can be answered from recent conversation memory, do not use a tool.
- Prefer one tool call unless another is truly needed.

Final answer requirements:
- Keep the final answer under 180 words unless the user asks for detail.
- Include source names or URLs when a tool was used.
- Use this exact final answer shape:
Answer: <direct response>
Sources: <source names or URLs, or "No external source used">
Rationale: <one sentence explaining why the answer follows>

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If no tool is needed, use:
Thought: I can answer without a tool
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def compact_text(value: str, limit: int = 900) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def format_error(tool_name: str, error: Exception) -> str:
    return f"{tool_name} failed: {error.__class__.__name__}: {error}"


@tool("web_search")
def web_search(query: str) -> str:
    """Search the live web for current events, weather, sports results, recent news, latest facts, products, or time-sensitive information."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=MAX_WEB_RESULTS)
        if not results:
            return f"No web search results found for: {query}"

        formatted = []
        for index, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            url = item.get("href") or item.get("url") or "No URL"
            body = compact_text(item.get("body", "No snippet available."), 350)
            formatted.append(f"{index}. {title}\nURL: {url}\nSnippet: {body}")
        return "\n\n".join(formatted)
    except Exception as error:  # LangChain should receive tool failures as observations.
        return format_error("web_search", error)


@tool("wikipedia_search")
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for stable general facts, definitions, biographies, history, and background context."""
    try:
        import wikipedia

        titles = wikipedia.search(query, results=MAX_WIKI_RESULTS)
        if not titles:
            return f"No Wikipedia results found for: {query}"

        formatted = []
        for index, title in enumerate(titles, start=1):
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = wikipedia.summary(title, sentences=4, auto_suggest=False)
                formatted.append(
                    f"{index}. {page.title}\nURL: {page.url}\nSummary: {compact_text(summary)}"
                )
            except wikipedia.DisambiguationError as error:
                options = ", ".join(error.options[:5])
                formatted.append(f"{index}. {title}\nDisambiguation options: {options}")
            except wikipedia.PageError:
                formatted.append(f"{index}. {title}\nPage could not be loaded.")
        return "\n\n".join(formatted)
    except Exception as error:
        return format_error("wikipedia_search", error)


@tool("arxiv_search")
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers, research literature, technical methods, authors, and abstracts."""
    try:
        import arxiv

        search = arxiv.Search(
            query=query,
            max_results=MAX_ARXIV_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client(page_size=MAX_ARXIV_RESULTS, delay_seconds=1, num_retries=2)
        results = list(client.results(search))
        if not results:
            return f"No arXiv results found for: {query}"

        formatted = []
        for index, paper in enumerate(results, start=1):
            authors = ", ".join(author.name for author in paper.authors[:4])
            if len(paper.authors) > 4:
                authors += ", et al."
            published = paper.published.date().isoformat() if paper.published else "Unknown"
            formatted.append(
                "\n".join(
                    [
                        f"{index}. {paper.title}",
                        f"Authors: {authors or 'Unknown'}",
                        f"Published: {published}",
                        f"URL: {paper.entry_id}",
                        f"Summary: {compact_text(paper.summary)}",
                    ]
                )
            )
        return "\n\n".join(formatted)
    except Exception as error:
        return format_error("arxiv_search", error)


def build_model() -> ChatGroq:
    load_dotenv()
    setup_langchain_llm_cache()

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file or export it in your shell."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
        max_retries=2,
        timeout=30,
    )


def setup_langchain_llm_cache() -> None:
    """Enable LangChain Core exact LLM caching for repeated prompts in this process."""
    if os.getenv("ENABLE_LLM_CACHE", "1") == "0":
        return
    try:
        from langchain_core.caches import InMemoryCache
        from langchain_core.globals import get_llm_cache, set_llm_cache

        if get_llm_cache() is None:
            set_llm_cache(InMemoryCache())
    except Exception:
        return


def load_memory_store() -> dict[str, list[dict[str, Any]]]:
    if not MEMORY_PATH.exists():
        return {}

    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_memory_store(store: dict[str, list[dict[str, Any]]]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(store, file, indent=2)


def get_recent_memory(thread_id: str) -> list[dict[str, Any]]:
    store = load_memory_store()
    messages = store.get(thread_id, [])
    return messages[-MAX_MEMORY_MESSAGES:]


def format_memory_for_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "No previous messages in this thread."

    formatted = []
    for message in messages:
        role = message.get("role", "unknown")
        content = compact_text(str(message.get("content", "")), 500)
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def append_memory(
    thread_id: str,
    question: str,
    answer: str,
    tool_name: str,
    latency_seconds: float,
) -> None:
    store = load_memory_store()
    thread_messages = store.setdefault(thread_id, [])
    timestamp = date.today().isoformat()
    thread_messages.extend(
        [
            {
                "role": "user",
                "content": question,
                "timestamp": timestamp,
            },
            {
                "role": "assistant",
                "content": answer,
                "timestamp": timestamp,
                "tool": tool_name,
                "latency_seconds": round(latency_seconds, 3),
            },
        ]
    )
    store[thread_id] = thread_messages[-MAX_MEMORY_MESSAGES:]
    save_memory_store(store)


def clear_memory(thread_id: str) -> None:
    store = load_memory_store()
    if thread_id in store:
        del store[thread_id]
        save_memory_store(store)


TOOLS = {
    "web_search": web_search,
    "wikipedia_search": wikipedia_search,
    "arxiv_search": arxiv_search,
}


def build_react_executor(model: ChatGroq) -> AgentExecutor:
    tools = list(TOOLS.values())
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=prompt,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))),
        handle_parsing_errors=(
            "Invalid ReAct format. Use exactly either "
            "'Action:' with 'Action Input:' or 'Final Answer:'."
        ),
        return_intermediate_steps=True,
        verbose=False,
    )


def extract_tool_calls(intermediate_steps: list[Any]) -> list[str]:
    calls = []
    for action, _observation in intermediate_steps:
        if getattr(action, "tool", "") == "_Exception":
            continue
        calls.append(f"{action.tool}({action.tool_input!r})")
    return calls


def ask_agent(question: str, thread_id: str = DEFAULT_THREAD_ID) -> dict[str, Any]:
    started = time.perf_counter()
    model = build_model()
    memory_messages = get_recent_memory(thread_id)
    executor = build_react_executor(model)
    current_date = os.getenv("CURRENT_DATE", date.today().isoformat())
    result = executor.invoke(
        {
            "input": question,
            "chat_history": format_memory_for_prompt(memory_messages),
            "current_date": current_date,
        }
    )
    answer = result.get("output", str(result))
    intermediate_steps = result.get("intermediate_steps", [])
    tool_calls = extract_tool_calls(intermediate_steps)
    tool_name = tool_calls[-1].split("(", maxsplit=1)[0] if tool_calls else "no_tool"
    latency_seconds = time.perf_counter() - started
    append_memory(thread_id, question, answer, tool_name, latency_seconds)
    return {
        "answer": answer,
        "tool_calls": tool_calls or ["no_tool"],
        "latency_seconds": latency_seconds,
        "intermediate_steps": intermediate_steps,
        "thread_id": thread_id,
        "memory_messages_used": len(memory_messages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the Groq/LangChain tool agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Who won in yesterday's ipl match",
        help="Question for the agent.",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print the tool calls selected by the agent.",
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help="Conversation thread id for short-term memory.",
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear short-term memory for the selected thread before answering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clear_memory:
        clear_memory(args.thread_id)

    response = ask_agent(args.question, thread_id=args.thread_id)

    print(textwrap.dedent(response["answer"]).strip())
    print(f"\nLatency: {response['latency_seconds']:.2f}s")
    print(f"Thread: {response['thread_id']}")
    print(f"Memory messages used: {response['memory_messages_used']}")

    if args.show_tools:
        calls = response["tool_calls"] or ["No tool calls."]
        print("\nTool calls:")
        for call in calls:
            print(f"- {call}")


if __name__ == "__main__":
    main()
