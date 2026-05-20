"""Streamlit dashboard for the WebSearch/RAG agent project.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agent import ask_agent
from langchain_compat import install_cachebacked_embeddings_shim

install_cachebacked_embeddings_shim()

from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_VECTOR_DB_DIR,
    DEFAULT_WEB_LINKS,
    ask,
    index_sources,
    load_chunks_jsonl,
    load_vector_store,
    _load_cache,
)


RESULTS_DIR = Path("results")
VECTOR_DB_DIR = Path(DEFAULT_VECTOR_DB_DIR)
CHUNKS_PATH = Path(".vector_db/chunks.jsonl")
RESPONSE_CACHE_PATH = Path(".cache/rag_responses.json")
EMBEDDING_CACHE_DIR = Path(".cache/embeddings")
HTML_CACHE_DIR = Path(".cache/html")


st.set_page_config(
    page_title="WebSearch LLM Agents",
    page_icon="🔎",
    layout="wide",
)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return round(total / (1024 * 1024), 2)


def vector_count() -> int:
    try:
        store = load_vector_store(DEFAULT_COLLECTION, DEFAULT_VECTOR_DB_DIR)
        return store._collection.count()
    except Exception:
        return 0


def format_tool_calls(tool_calls: list[str]) -> str:
    return "\n".join(f"- {call}" for call in tool_calls) if tool_calls else "No tool calls."


def metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def render_pipeline() -> None:
    st.subheader("End-to-End Pipeline")
    st.code(
        """
20 web pages
  -> cached HTML
  -> UnstructuredHTMLLoader
  -> semantic chunking
  -> sentence-transformers embeddings with cache
  -> semantic response cache lookup
  -> local router + query enhancer + optional HyDE + optional multi-query
  -> parallel Chroma dense search + BM25 sparse search
  -> RRF fusion
  -> metadata filtering
  -> cross-encoder reranking
  -> limited context window
  -> answer draft
  -> self-reflection
  -> final answer + sources
        """.strip(),
        language="text",
    )
    st.caption("RAG uses a direct routed pipeline. It does not run a LangChain agent loop.")

    st.subheader("Core Techniques")
    techniques = pd.DataFrame(
        [
            ["Document loading", "UnstructuredHTMLLoader over cached HTML files"],
            ["Preprocessing", "HTML cache, text extraction, metadata enrichment"],
            ["Chunking", "SemanticChunker with RecursiveCharacterTextSplitter fallback"],
            ["Embeddings", "sentence-transformers/all-MiniLM-L6-v2"],
            ["Embedding cache", ".cache/embeddings via CacheBackedEmbeddings"],
            ["Dense index", "Chroma persistent vector store"],
            ["Sparse index", "BM25 over persisted chunks.jsonl"],
            ["Hybrid search", "Parallel dense candidates + BM25 candidates"],
            ["Fusion", "Reciprocal Rank Fusion across dense, sparse, and multi-query result lists"],
            ["Router", "Local topic routing before retrieval"],
            ["Query enhancer", "Cheap synonym/acronym expansion before search"],
            ["HyDE", "Optional hypothetical document expansion"],
            ["Multi-query", "Optional LLM-generated query variants"],
            ["Metadata filtering", "Automatic topic filter from router; no manual UI filter needed"],
            ["Reranking", "BAAI/bge-reranker-base CrossEncoder"],
            ["Limited context", "Only the top retrieved text within the context budget is sent to the LLM"],
            ["LLM cache", "LangChain Core InMemoryCache for exact prompt/model repeats"],
            ["Self-reflection", "LLM reviewer revises answer for grounding and format"],
            ["Response cache", ".cache/rag_responses.json with semantic question matching"],
            ["Evaluation", "Custom scorecard, RAGAS, DeepEval-compatible records"],
        ],
        columns=["Feature", "Implementation"],
    )
    st.dataframe(techniques, use_container_width=True, hide_index=True)


def render_status() -> None:
    st.subheader("Project Status")
    chunks = load_chunks_jsonl()
    response_cache = _load_cache()
    cols = st.columns(5)
    with cols[0]:
        metric_card("Indexed vectors", vector_count())
    with cols[1]:
        metric_card("BM25 chunks", len(chunks))
    with cols[2]:
        metric_card("Cached responses", len(response_cache))
    with cols[3]:
        metric_card("Embedding cache", f"{directory_size_mb(EMBEDDING_CACHE_DIR)} MB")
    with cols[4]:
        metric_card("HTML cache", f"{directory_size_mb(HTML_CACHE_DIR)} MB")

    st.caption("Opening this dashboard does not call Groq. Buttons below trigger model calls explicitly.")


def render_corpus() -> None:
    st.subheader("20-Link Corpus")
    corpus = pd.DataFrame(DEFAULT_WEB_LINKS)
    st.dataframe(corpus, use_container_width=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reindex Corpus", type="primary"):
            with st.spinner("Indexing corpus. First run may download models and take a while..."):
                started = time.perf_counter()
                store = index_sources(
                    DEFAULT_WEB_LINKS,
                    DEFAULT_COLLECTION,
                    DEFAULT_VECTOR_DB_DIR,
                    reindex=True,
                    force_rescrape=False,
                )
                st.success(
                    f"Indexed {store._collection.count()} chunks in {time.perf_counter() - started:.2f}s."
                )
    with col2:
        st.info(
            "Reindexing refreshes Chroma and BM25 chunks. Use the CLI with "
            "`--force-rescrape` when you also want to refresh cached HTML."
        )


def render_tool_agent() -> None:
    st.subheader("Tool-Calling Agent")
    st.write("Uses ReAct with web search, Wikipedia, arXiv, and short-term memory.")

    question = st.text_input(
        "Tool-agent question",
        value="Who is Ada Lovelace?",
        key="tool_question",
    )
    thread_id = st.text_input("Thread ID", value="streamlit-demo")

    if st.button("Ask Tool Agent"):
        with st.spinner("Running tool agent..."):
            try:
                response = ask_agent(question, thread_id=thread_id)
                st.markdown(response["answer"])
                st.caption(f"Latency: {response['latency_seconds']:.2f}s")
                st.code(format_tool_calls(response["tool_calls"]), language="text")
            except Exception as exc:
                st.error(f"Tool agent failed: {exc}")


def render_rag_agent() -> None:
    st.subheader("RAG Agent")
    st.write("Uses routing, hybrid retrieval, reranking, self-reflection, and response caching.")

    question = st.text_input(
        "RAG question",
        value="How does hybrid search work?",
        key="rag_question",
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        use_cache = st.toggle("Semantic response cache", value=True)
    with col_b:
        use_query_enhancer = st.toggle("Query enhancer", value=os.getenv("RAG_ENABLE_QUERY_ENHANCER", "1") == "1")
    with col_c:
        use_hyde = st.toggle("HyDE", value=os.getenv("RAG_ENABLE_HYDE", "0") == "1")
    with col_d:
        use_multi_query = st.toggle("Multi-query", value=os.getenv("RAG_ENABLE_MULTI_QUERY", "0") == "1")

    col_e, col_f = st.columns(2)
    with col_e:
        use_self_reflection = st.toggle(
            "Self-reflection",
            value=os.getenv("RAG_ENABLE_SELF_REFLECTION", "1") == "1",
        )
    with col_f:
        st.caption("Topic/domain filters are automatic through the router.")

    context_limit = st.slider(
        "Context character budget",
        min_value=1500,
        max_value=12000,
        value=int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "5000")),
        step=500,
        help="Caps retrieved context sent to the LLM, reducing latency and token usage.",
    )

    if st.button("Ask RAG Agent"):
        with st.spinner("Running RAG agent..."):
            try:
                os.environ["RAG_ENABLE_QUERY_ENHANCER"] = "1" if use_query_enhancer else "0"
                os.environ["RAG_ENABLE_HYDE"] = "1" if use_hyde else "0"
                os.environ["RAG_ENABLE_MULTI_QUERY"] = "1" if use_multi_query else "0"
                os.environ["RAG_ENABLE_SELF_REFLECTION"] = "1" if use_self_reflection else "0"
                os.environ["RAG_CONTEXT_CHAR_LIMIT"] = str(context_limit)
                response = ask(
                    question,
                    url="",
                    collection_name=DEFAULT_COLLECTION,
                    persist_directory=DEFAULT_VECTOR_DB_DIR,
                    auto_index=True,
                    use_cache=use_cache,
                )
                st.markdown(response["answer"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latency", f"{response['latency_seconds']:.2f}s")
                c2.metric("Cached", str(response.get("cached", False)))
                c3.metric("Cache type", str(response.get("cache_type") or "none"))
                c4.metric("Contexts", len(response["contexts"]))
                st.caption(f"Cache score: {response.get('cache_score', 0.0)}")
                st.caption(f"Topic filter: {response.get('topic_filter') or 'none'}")
                with st.expander("Retrieval Queries"):
                    for query in response.get("retrieval_queries", [question]):
                        st.write(query)
                tool_calls = [
                    f"hybrid_retrieval(queries={response.get('retrieval_queries', [question])!r}, "
                    f"k={len(response.get('contexts', []))}, topic={response.get('topic_filter')!r})"
                ]
                st.code(format_tool_calls(tool_calls), language="text")
                with st.expander("Retrieved Contexts"):
                    for index, context in enumerate(response["contexts"], start=1):
                        st.markdown(f"**Context {index}**")
                        st.write(context[:1500])
                with st.expander("Sources"):
                    st.write(response["sources"])
            except Exception as exc:
                st.error(f"RAG agent failed: {exc}")


def render_evaluation() -> None:
    st.subheader("Evaluation Results")
    summary = read_json(RESULTS_DIR / "scorecard_summary.json", default={})
    scorecard = read_json(RESULTS_DIR / "scorecard.json", default=[])
    ragas_summary = read_json(RESULTS_DIR / "ragas_summary.json", default={})
    ragas_results = read_csv(RESULTS_DIR / "ragas_results.csv")

    if summary:
        st.markdown("### Custom Scorecard Summary")
        overall = summary.get("overall", {})
        cols = st.columns(4)
        cols[0].metric("Cases", overall.get("cases", "-"))
        cols[1].metric("Scored", overall.get("scored_cases", "-"))
        cols[2].metric("Avg latency", f"{overall.get('avg_latency_seconds', '-')}s")
        cols[3].metric("Error handling", overall.get("error_handling_score", "-"))

        by_agent = summary.get("by_agent", {})
        if by_agent:
            st.dataframe(pd.DataFrame(by_agent).T, use_container_width=True)
    else:
        st.warning("No scorecard summary found. Run `python -m evals.scorecard`.")

    if scorecard:
        st.markdown("### Per-Case Scorecard")
        scorecard_df = pd.DataFrame(scorecard)
        preferred_columns = [
            "agent",
            "id",
            "status",
            "expected_tool",
            "tool_calls",
            "latency_seconds",
            "cached",
            "cache_type",
            "cache_score",
            "retrieval_queries",
            "topic_filter",
            "tool_selection_score",
            "content_match_score",
            "tool_efficiency_score",
            "error_handling_score",
        ]
        visible_columns = [column for column in preferred_columns if column in scorecard_df.columns]
        remaining_columns = [column for column in scorecard_df.columns if column not in visible_columns]
        st.dataframe(scorecard_df[visible_columns + remaining_columns], use_container_width=True)

    if ragas_summary:
        st.markdown("### RAGAS Summary")
        st.json(ragas_summary)
    else:
        st.warning("No RAGAS summary found. Run `python -m evals.ragas_eval`.")

    if not ragas_results.empty:
        st.markdown("### RAGAS Per-Case Results")
        preferred_columns = [
            "id",
            "question",
            "retrieval_queries",
            "effective_topic_filter",
            "cached",
            "cache_type",
            "cache_score",
            "retrieved_context_count",
            "unique_source_count",
        ]
        visible_columns = [column for column in preferred_columns if column in ragas_results.columns]
        remaining_columns = [column for column in ragas_results.columns if column not in visible_columns]
        st.dataframe(ragas_results[visible_columns + remaining_columns], use_container_width=True)


def render_caches() -> None:
    st.subheader("Caches And Local Artifacts")
    data = pd.DataFrame(
        [
            [".cache/html", HTML_CACHE_DIR.exists(), f"{directory_size_mb(HTML_CACHE_DIR)} MB"],
            [".cache/embeddings", EMBEDDING_CACHE_DIR.exists(), f"{directory_size_mb(EMBEDDING_CACHE_DIR)} MB"],
            [".cache/rag_responses.json", RESPONSE_CACHE_PATH.exists(), f"{directory_size_mb(RESPONSE_CACHE_PATH)} MB"],
            [".vector_db/chroma", VECTOR_DB_DIR.exists(), f"{directory_size_mb(VECTOR_DB_DIR)} MB"],
            [".vector_db/chunks.jsonl", CHUNKS_PATH.exists(), f"{directory_size_mb(CHUNKS_PATH)} MB"],
            ["results/", RESULTS_DIR.exists(), f"{directory_size_mb(RESULTS_DIR)} MB"],
        ],
        columns=["Path", "Exists", "Size"],
    )
    st.dataframe(data, use_container_width=True, hide_index=True)

    if st.button("Clear Response Cache"):
        if RESPONSE_CACHE_PATH.exists():
            RESPONSE_CACHE_PATH.unlink()
            st.success("Response cache cleared.")
        else:
            st.info("No response cache found.")


def main() -> None:
    st.title("WebSearch LLM Agents Dashboard")
    st.caption("A Streamlit view of the tool agent, RAG pipeline, corpus, caches, and evaluation results.")

    render_status()

    tab_names = [
        "Pipeline",
        "Corpus & Index",
        "Tool Agent",
        "RAG Agent",
        "Evaluation",
        "Caches",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_pipeline()
    with tabs[1]:
        render_corpus()
    with tabs[2]:
        render_tool_agent()
    with tabs[3]:
        render_rag_agent()
    with tabs[4]:
        render_evaluation()
    with tabs[5]:
        render_caches()


if __name__ == "__main__":
    main()
