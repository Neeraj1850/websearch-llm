"""Streamlit dashboard for testing rag_agent.py.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from langchain_compat import install_cachebacked_embeddings_shim
install_cachebacked_embeddings_shim()

from rag_agent import (
    DEFAULT_COLLECTION,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_RESPONSE_CACHE_DB,
    DEFAULT_VECTOR_DB_DIR,
    DEFAULT_WEB_LINKS,
    ask,
    clear_all_local_artifacts,
    clear_response_cache,
    index_sources,
)

RESULTS_DIR      = Path("results")
VECTOR_DB_DIR    = Path(DEFAULT_VECTOR_DB_DIR)
CHUNKS_PATH      = Path(DEFAULT_CHUNKS_PATH)
CACHE_DB_PATH    = Path(os.getenv("RAG_RESPONSE_CACHE_DB", DEFAULT_RESPONSE_CACHE_DB))
EMBEDDING_CACHE  = Path(".cache/embeddings")
HTML_CACHE       = Path(".cache/html")

st.set_page_config(
    page_title="RAG Agent Tester",
    page_icon="🔍",
    layout="wide",
)

# ── minimal custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  div[data-testid="metric-container"] { background: #f8f9fa; border-radius: 8px; padding: 0.5rem 1rem; }
  .answer-box { background: #f0f4ff; border-left: 4px solid #4f6ef7;
                padding: 1rem 1.25rem; border-radius: 6px; margin-top: 0.5rem; }
  .tag { display: inline-block; background: #e8eeff; color: #111;
         font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dir_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / 1_048_576, 2)
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576, 2)


def _vector_count() -> int | str:
    if not VECTOR_DB_DIR.exists():
        return 0
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        return client.get_collection(DEFAULT_COLLECTION).count()
    except Exception:
        return "?"


def _chunk_count() -> int:
    if not CHUNKS_PATH.exists():
        return 0
    try:
        return sum(1 for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip())
    except OSError:
        return 0


def _cache_count() -> int:
    if not CACHE_DB_PATH.exists():
        return 0
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            row = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ── status bar ────────────────────────────────────────────────────────────────

def _render_status() -> None:
    cols = st.columns(5)
    cols[0].metric("Indexed vectors",   _vector_count())
    cols[1].metric("BM25 chunks",       _chunk_count())
    cols[2].metric("Cached responses",  _cache_count())
    cols[3].metric("Embedding cache",   f"{_dir_mb(EMBEDDING_CACHE)} MB")
    cols[4].metric("HTML cache",        f"{_dir_mb(HTML_CACHE)} MB")


# ── tabs ──────────────────────────────────────────────────────────────────────

def tab_ask() -> None:
    """Main RAG query interface."""
    st.subheader("Ask the RAG Agent")

    question = st.text_input(
        "Question",
        value="What is task decomposition?",
        placeholder="Ask anything about the indexed corpus…",
    )

    # ── feature toggles ───────────────────────────────────────────────────────
    with st.expander("⚙️  Pipeline options", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        use_cache         = c1.toggle("Response cache",   value=True)
        use_enhancer      = c2.toggle("Query enhancer",   value=os.getenv("RAG_ENABLE_QUERY_ENHANCER","1")=="1")
        use_hyde          = c3.toggle("HyDE",             value=os.getenv("RAG_ENABLE_HYDE","0")=="1")
        use_multi_query   = c4.toggle("Multi-query",      value=os.getenv("RAG_ENABLE_MULTI_QUERY","0")=="1")

        c5, c6, _ , __ = st.columns(4)
        use_reflection    = c5.toggle("Self-reflection",  value=os.getenv("RAG_ENABLE_SELF_REFLECTION","1")=="1")
        use_reranking     = c6.toggle("Reranking",        value=os.getenv("RAG_ENABLE_RERANKING","1")=="1")

        context_limit = st.slider(
            "Context character budget", 1500, 12000,
            int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "6000")), step=500,
        )

    if st.button("Ask", type="primary", use_container_width=True):
        # apply env toggles before calling ask()
        os.environ["RAG_ENABLE_QUERY_ENHANCER"]   = "1" if use_enhancer    else "0"
        os.environ["RAG_ENABLE_HYDE"]             = "1" if use_hyde        else "0"
        os.environ["RAG_ENABLE_MULTI_QUERY"]      = "1" if use_multi_query else "0"
        os.environ["RAG_ENABLE_SELF_REFLECTION"]  = "1" if use_reflection  else "0"
        os.environ["RAG_ENABLE_RERANKING"]        = "1" if use_reranking   else "0"
        os.environ["RAG_CONTEXT_CHAR_LIMIT"]      = str(context_limit)

        with st.spinner("Retrieving and generating…"):
            try:
                resp = ask(
                    question,
                    collection_name=DEFAULT_COLLECTION,
                    persist_directory=DEFAULT_VECTOR_DB_DIR,
                    auto_index=True,
                    use_cache=use_cache,
                )
            except Exception as exc:
                st.error(f"RAG agent error: {exc}")
                return

        # ── answer ────────────────────────────────────────────────────────────
        st.markdown("#### Answer")
        st.markdown(
            f'<div class="answer-box">{resp["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # ── stats row ─────────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Latency",      f"{resp['latency_seconds']:.2f}s")
        m2.metric("Contexts",     len(resp["contexts"]))
        m3.metric("Cached",       "✓" if resp.get("cached") else "✗")
        m4.metric("Cache type",   resp.get("cache_type") or "—")
        m5.metric("Cache score",  f"{resp.get('cache_score', 0.0):.3f}")

        # ── topic filter + queries ────────────────────────────────────────────
        topic = resp.get("topic_filter") or "none"
        st.caption(f"Topic filter: **{topic}**")

        queries = resp.get("retrieval_queries", [question])
        if len(queries) > 1:
            with st.expander(f"Retrieval queries ({len(queries)})"):
                for q in queries:
                    st.markdown(f'<span class="tag">{q}</span>', unsafe_allow_html=True)

        # ── sources ───────────────────────────────────────────────────────────
        if resp.get("sources"):
            with st.expander("Sources"):
                for src in resp["sources"]:
                    st.markdown(f"- {src}")

        # ── contexts ──────────────────────────────────────────────────────────
        with st.expander(f"Retrieved contexts ({len(resp['contexts'])})"):
            for i, ctx in enumerate(resp["contexts"], 1):
                st.markdown(f"**Chunk {i}**")
                st.text(ctx[:800] + ("…" if len(ctx) > 800 else ""))
                st.divider()


def tab_index() -> None:
    """Corpus inspection and reindexing."""
    st.subheader("Indexed Corpus")
    st.dataframe(pd.DataFrame(DEFAULT_WEB_LINKS), use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reindex", type="primary"):
            with st.spinner("Indexing… first run downloads models and may take a minute."):
                t0 = time.perf_counter()
                store = index_sources(
                    DEFAULT_WEB_LINKS,
                    DEFAULT_COLLECTION,
                    DEFAULT_VECTOR_DB_DIR,
                    reindex=True,
                    show_progress=False,
                )
                st.success(
                    f"Indexed {store._collection.count()} chunks "
                    f"in {time.perf_counter() - t0:.1f}s."
                )
    with col2:
        st.info("Use the CLI `--force-rescrape` flag when you also want to refresh cached HTML.")


def tab_eval() -> None:
    """Show the latest eval results from disk."""
    st.subheader("Evaluation Results")

    # ── smoke-check scorecard ─────────────────────────────────────────────────
    sm = _read_json(RESULTS_DIR / "scorecard_summary.json", {})
    if sm:
        st.markdown("#### Smoke-check scorecard")
        metric_cols = [
            "retrieval_hit", "context_coverage", "format_ok",
            "latency_ok", "out_of_scope_ok", "composite",
        ]
        rows = []
        for col in metric_cols:
            stats = sm.get(col)
            if isinstance(stats, dict):
                rows.append({
                    "Metric": col,
                    "Mean":   stats.get("mean", "—"),
                    "Min":    stats.get("min",  "—"),
                    "Weak cases": ", ".join(stats.get("weak_cases", [])) or "—",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        lat = sm.get("latency", {})
        if lat:
            st.caption(
                f"Latency  mean={lat.get('mean_seconds',0):.2f}s  "
                f"p95={lat.get('p95_seconds',0):.2f}s  "
                f"max={lat.get('max_seconds',0):.2f}s"
            )
    else:
        st.warning("No scorecard results yet. Run: `python -m evals.scorecard`")

    # ── RAGAS summary ─────────────────────────────────────────────────────────
    rs = _read_json(RESULTS_DIR / "ragas_summary.json", {})
    if rs:
        st.markdown("#### RAGAS summary")
        ragas_rows = []
        for key, val in rs.items():
            if isinstance(val, dict) and "mean" in val:
                ragas_rows.append({
                    "Metric": key,
                    "Mean":   val["mean"],
                    "Min":    val["min"],
                    "Max":    val["max"],
                    "Weak cases": ", ".join(val.get("weak_cases", [])) or "—",
                })
        if ragas_rows:
            st.dataframe(pd.DataFrame(ragas_rows), use_container_width=True, hide_index=True)
        else:
            st.json(rs)
    else:
        st.warning("No RAGAS results yet. Run: `python -m evals.ragas_eval`")

    # ── per-case RAGAS CSV ────────────────────────────────────────────────────
    ragas_csv = RESULTS_DIR / "ragas_results.csv"
    if ragas_csv.exists():
        df = pd.read_csv(ragas_csv)
        with st.expander("Per-case RAGAS results"):
            st.dataframe(df, use_container_width=True)

    # ── per-case scorecard ────────────────────────────────────────────────────
    sc = _read_json(RESULTS_DIR / "scorecard.json", [])
    if sc:
        with st.expander("Per-case smoke-check scores"):
            keep = [
                "id", "question", "retrieval_hit", "context_coverage",
                "format_ok", "latency_ok", "composite", "latency_seconds",
                "effective_topic_filter",
            ]
            df = pd.DataFrame(sc)
            st.dataframe(df[[c for c in keep if c in df.columns]], use_container_width=True)


def tab_caches() -> None:
    """Cache stats and clear buttons."""
    st.subheader("Caches & Artifacts")

    data = [
        ["HTML cache",       str(HTML_CACHE),       HTML_CACHE.exists(),       f"{_dir_mb(HTML_CACHE)} MB"],
        ["Embedding cache",  str(EMBEDDING_CACHE),  EMBEDDING_CACHE.exists(),  f"{_dir_mb(EMBEDDING_CACHE)} MB"],
        ["Response cache",   str(CACHE_DB_PATH),    CACHE_DB_PATH.exists(),    f"{_dir_mb(CACHE_DB_PATH)} MB"],
        ["Chroma DB",        str(VECTOR_DB_DIR),    VECTOR_DB_DIR.exists(),    f"{_dir_mb(VECTOR_DB_DIR)} MB"],
        ["BM25 chunks",      str(CHUNKS_PATH),      CHUNKS_PATH.exists(),      f"{_dir_mb(CHUNKS_PATH)} MB"],
        ["Results",          "results/",            RESULTS_DIR.exists(),      f"{_dir_mb(RESULTS_DIR)} MB"],
    ]
    st.dataframe(
        pd.DataFrame(data, columns=["Cache", "Path", "Exists", "Size"]),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear response cache"):
            clear_response_cache()
            st.success("Response cache cleared.")
    with c2:
        if st.button("Clear ALL RAG artifacts", type="secondary"):
            clear_all_local_artifacts()
            st.success("All RAG artifacts cleared. Reindex before querying.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("RAG Agent Tester")
    _render_status()
    st.divider()

    ask_tab, index_tab, eval_tab, cache_tab = st.tabs([
        "💬 Ask", "📚 Corpus & Index", "📊 Evaluation", "🗄️ Caches",
    ])
    with ask_tab:
        tab_ask()
    with index_tab:
        tab_index()
    with eval_tab:
        tab_eval()
    with cache_tab:
        tab_caches()


if __name__ == "__main__":
    main()