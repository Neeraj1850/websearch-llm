"""Production RAG agent — latency-optimised, accuracy-preserved.

Latency fixes applied (see BOTTLENECKS section in README):
  B1  CrossEncoder:          singleton, loaded once at first rerank call
  B2  HuggingFaceEmbeddings: singleton, loaded once
  B7  ChatLLM:               singleton, loaded once
  B3  BM25Retriever:         LRU-cached keyed on (path-mtime, topic, domain)
  B4  Response cache:        SQLite WAL + in-process LRU dict — no full JSON parse per call
  B5  Double embedding:      question vector computed once in run_rag, threaded through
  B9  Cache store race:      single INSERT, no read-modify-write cycle
  B6  Async serving:         full asyncio pipeline; sync helpers run in executor
  B8  HyDE + multi-query:    concurrent asyncio.gather when both enabled

Indexing fixes applied:
  I1  Parallel scraping:     ThreadPoolExecutor fetches all URLs concurrently
  I2  Default chunker:       RecursiveCharacterTextSplitter used by default (SemanticChunker
                             opt-in via RAG_USE_SEMANTIC_CHUNKING=1); avoids O(sentences)
                             embedding calls during index build
  I3  BS4-first parsing:     UnstructuredHTMLLoader skipped entirely by default;
                             opt-in via RAG_USE_UNSTRUCTURED=1
  I4  Batched Chroma writes: add_documents called in batches of RAG_INDEX_BATCH_SIZE (default 500)
                             to avoid SQLite write contention on large corpora

Usage:
    python rag_agent.py "What is task decomposition?"
    python rag_agent.py --index
    python rag_agent.py --reindex --force-rescrape
    python rag_agent.py "Explain HyDE" --url https://example.com --single-url
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bs4
import numpy as np
import requests
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_core.embeddings import CacheBackedEmbeddings
except ImportError:
    try:
        from langchain_classic.embeddings import CacheBackedEmbeddings  # type: ignore[no-redef]
    except ImportError:
        from langchain.embeddings import CacheBackedEmbeddings          # type: ignore[no-redef]

try:
    from langchain_core.stores import LocalFileStore
except ImportError:
    try:
        from langchain_classic.storage import LocalFileStore             # type: ignore[no-redef]
    except ImportError:
        from langchain.storage import LocalFileStore                     # type: ignore[no-redef]

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None                                           # type: ignore[assignment,misc]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_agent")

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_URL                      = "https://lilianweng.github.io/posts/2023-06-23-agent/"
DEFAULT_GROQ_MODEL               = "llama-3.1-8b-instant"
DEFAULT_OLLAMA_MODEL             = "llama3.1:8b"
DEFAULT_EMBEDDING_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL           = "BAAI/bge-reranker-base"
DEFAULT_VECTOR_DB_DIR            = ".vector_db/chroma"
DEFAULT_COLLECTION               = "rag_documents_minilm"
DEFAULT_HTML_CACHE_DIR           = ".cache/html"
DEFAULT_CHUNKS_PATH              = ".vector_db/chunks.jsonl"
DEFAULT_EMBEDDING_CACHE_DIR      = ".cache/embeddings"
DEFAULT_RESPONSE_CACHE_DB        = ".cache/rag_responses.db"   # SQLite replaces JSON
DEFAULT_RESPONSE_CACHE_THRESHOLD = 0.92
DEFAULT_RAG_K                    = 6
DEFAULT_RAG_CONTEXT_CHAR_LIMIT   = 6000
DEFAULT_MULTI_QUERY_COUNT        = 3
DEFAULT_INDEX_BATCH_SIZE         = 500   # I4: Chroma write batch size
DEFAULT_SCRAPE_WORKERS           = 8     # I1: parallel scrape thread count
_RESPONSE_CACHE_LRU_SIZE         = 512   # in-process LRU dict cap

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "agents":     ["agent", "agents", "tool use", "autonomous"],
    "rag":        ["rag", "retrieval augmented", "retrieval-augmented"],
    "tools":      ["tool", "tools", "function calling"],
    "memory":     ["memory", "long-term memory", "short-term memory"],
    "retrieval":  ["retrieval", "search", "bm25", "hybrid search"],
    "embeddings": ["embedding", "embeddings", "sentence transformer"],
    "chunking":   ["chunk", "chunking", "text split"],
    "reranking":  ["rerank", "reranking", "cross-encoder"],
}

DEFAULT_WEB_LINKS: list[dict[str, str]] = [
    {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/",           "topic": "agents"},
    {"url": "https://python.langchain.com/docs/how_to/semantic-chunker/",     "topic": "chunking"},
    {"url": "https://python.langchain.com/docs/how_to/MultiQueryRetriever/",  "topic": "retrieval"},
    {"url": "https://python.langchain.com/docs/how_to/contextual_compression/","topic": "reranking"},
    {"url": "https://python.langchain.com/docs/how_to/ensemble_retriever/",   "topic": "retrieval"},
    {"url": "https://www.pinecone.io/learn/retrieval-augmented-generation/",  "topic": "rag"},
    {"url": "https://www.pinecone.io/learn/hybrid-search-intro/",             "topic": "retrieval"},
    {"url": "https://weaviate.io/blog/hybrid-search-explained",               "topic": "retrieval"},
    {"url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2", "topic": "embeddings"},
    {"url": "https://huggingface.co/BAAI/bge-reranker-base",                 "topic": "reranking"},
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SINGLETONS  (B1, B2, B7)
#
# Every heavyweight object is initialised exactly once per process, behind a
# threading.Lock so concurrent first-calls don't double-load.  Subsequent
# calls return the cached object in O(1) with no locking overhead.
# ══════════════════════════════════════════════════════════════════════════════

_singleton_lock = threading.Lock()

# ── Embedding model (B2) ──────────────────────────────────────────────────────
_raw_embeddings: HuggingFaceEmbeddings | None = None
_cached_embeddings: CacheBackedEmbeddings | None = None

def get_raw_embeddings() -> HuggingFaceEmbeddings:
    """Return the process-wide HuggingFaceEmbeddings singleton (lazy-loaded)."""
    global _raw_embeddings
    if _raw_embeddings is None:
        with _singleton_lock:
            if _raw_embeddings is None:
                model_name = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
                log.info("Loading embedding model %s (may take 20–30s on first run) …", model_name)
                t0 = time.perf_counter()
                _raw_embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": os.getenv("RAG_EMBEDDING_DEVICE", "cpu")},
                    encode_kwargs={"normalize_embeddings": True},
                )
                log.info("Embedding model ready in %.1fs.", time.perf_counter() - t0)
    return _raw_embeddings


def get_cached_embeddings() -> CacheBackedEmbeddings:
    """Return the disk-cached embedding wrapper (lazy-loaded)."""
    global _cached_embeddings
    if _cached_embeddings is None:
        with _singleton_lock:
            if _cached_embeddings is None:
                cache_dir = os.getenv("RAG_EMBEDDING_CACHE_DIR", DEFAULT_EMBEDDING_CACHE_DIR)
                namespace  = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).replace("/", "__")
                _cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                    get_raw_embeddings(), LocalFileStore(cache_dir), namespace=namespace
                )
    return _cached_embeddings


# ── LLM (B7) ─────────────────────────────────────────────────────────────────
_llm_instance = None
_llm_provider: str | None = None

def _setup_llm_cache() -> None:
    if os.getenv("RAG_ENABLE_LLM_CACHE", "1") == "0":
        return
    try:
        from langchain_core.caches import InMemoryCache
        from langchain_core.globals import get_llm_cache, set_llm_cache
        if get_llm_cache() is None:
            set_llm_cache(InMemoryCache())
    except Exception:
        pass


def get_model():
    """Return the process-wide chat LLM singleton (lazy-loaded)."""
    global _llm_instance, _llm_provider
    load_dotenv()
    provider = os.getenv("RAG_MODEL_PROVIDER", "groq").lower()
    if _llm_instance is None or provider != _llm_provider:
        with _singleton_lock:
            if _llm_instance is None or provider != _llm_provider:
                _setup_llm_cache()
                log.info("Loading LLM provider=%s …", provider)
                if provider == "ollama":
                    from langchain_ollama import ChatOllama
                    _llm_instance = ChatOllama(
                        model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                        temperature=0,
                    )
                else:
                    api_key = os.getenv("GROQ_API_KEY")
                    if not api_key:
                        raise RuntimeError(
                            "GROQ_API_KEY is not set. Add it to .env or set RAG_MODEL_PROVIDER=ollama."
                        )
                    from langchain_groq import ChatGroq
                    _llm_instance = ChatGroq(
                        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                        temperature=0,
                        max_retries=3,
                        timeout=30,
                    )
                _llm_provider = provider
    return _llm_instance


# ── CrossEncoder (B1) ────────────────────────────────────────────────────────
_cross_encoder = None

def get_cross_encoder():
    """Return the process-wide CrossEncoder singleton (lazy-loaded on first rerank call)."""
    global _cross_encoder
    if _cross_encoder is None:
        with _singleton_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    model_name = os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
                    log.info("Loading reranker %s …", model_name)
                    _cross_encoder = CrossEncoder(
                        model_name,
                        device=os.getenv("RAG_RERANKER_DEVICE", "cpu"),
                    )
                except Exception as exc:
                    log.warning("CrossEncoder unavailable (%s); reranking disabled.", exc)
                    _cross_encoder = None   # stays None → fallback path in _rerank
    return _cross_encoder


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RESPONSE CACHE  (B4, B5, B9)
#
# Architecture:
#   • SQLite with WAL mode — O(1) INSERT, concurrent readers don't block writer.
#   • In-process LRU dict (_cache_lru) — cache hits never touch disk at all.
#   • Question embedding computed ONCE in run_rag and passed into both
#     _lookup_cache and _store_cache (eliminates the double-embed of B5).
#   • _store_cache does a single INSERT OR REPLACE — no read-modify-write (B9).
# ══════════════════════════════════════════════════════════════════════════════

# In-process LRU: maps cache_key -> row dict
_cache_lru: dict[str, dict[str, Any]] = {}
_cache_lru_lock = threading.Lock()

# Semantic scan index: kept in memory after first DB load
# Maps cache_key -> (question_embedding, settings_json_str, answer, contexts, sources)
_semantic_index: list[tuple[str, list[float], str, dict[str, Any]]] = []
_semantic_index_loaded = False
_semantic_index_lock = threading.Lock()


def _db_path() -> str:
    return os.getenv("RAG_RESPONSE_CACHE_DB", DEFAULT_RESPONSE_CACHE_DB)


def _open_db() -> sqlite3.Connection:
    """Open (and initialise) the SQLite response cache with WAL mode."""
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster than FULL
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key          TEXT PRIMARY KEY,
            question     TEXT NOT NULL,
            q_embedding  TEXT NOT NULL,          -- JSON float array
            answer       TEXT NOT NULL,
            contexts     TEXT NOT NULL,          -- JSON array
            sources      TEXT NOT NULL,          -- JSON array
            settings     TEXT NOT NULL           -- JSON object
        )
    """)
    conn.commit()
    return conn


# Module-level DB connection (one per process; WAL allows concurrent readers)
_db_conn: sqlite3.Connection | None = None
_db_lock = threading.Lock()

def _get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        with _db_lock:
            if _db_conn is None:
                _db_conn = _open_db()
    return _db_conn


def _current_settings(
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> dict[str, Any]:
    return {
        "collection_name":  collection_name,
        "topic_filter":     topic_filter,
        "domain_filter":    domain_filter,
        "k":                k,
        "embedding_model":  os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "reranking":        os.getenv("RAG_ENABLE_RERANKING",        "1"),
        "self_reflection":  os.getenv("RAG_ENABLE_SELF_REFLECTION",  "1"),
        "hyde":             os.getenv("RAG_ENABLE_HYDE",             "0"),
        "multi_query":      os.getenv("RAG_ENABLE_MULTI_QUERY",      "0"),
        "query_enhancer":   os.getenv("RAG_ENABLE_QUERY_ENHANCER",   "1"),
        "context_char_limit": os.getenv("RAG_CONTEXT_CHAR_LIMIT",    str(DEFAULT_RAG_CONTEXT_CHAR_LIMIT)),
    }


def _cache_key(
    question: str,
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> str:
    payload = {"question": question,
               **_current_settings(collection_name, topic_filter, domain_filter, k)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _load_semantic_index() -> None:
    """Load all question embeddings + settings from SQLite into memory once.

    Subsequent semantic scans iterate over this in-memory list — no DB I/O
    per query.  New entries are appended on cache store so the index stays
    current without a reload.
    """
    global _semantic_index, _semantic_index_loaded
    with _semantic_index_lock:
        if _semantic_index_loaded:
            return
        db = _get_db()
        rows = db.execute(
            "SELECT key, q_embedding, settings, answer, contexts, sources FROM cache"
        ).fetchall()
        _semantic_index = [
            (
                row[0],
                json.loads(row[1]),
                row[2],          # settings JSON string — compare as string, fast
                {
                    "answer":   row[3],
                    "contexts": json.loads(row[4]),
                    "sources":  json.loads(row[5]),
                },
            )
            for row in rows
        ]
        _semantic_index_loaded = True
        log.debug("Semantic index loaded: %d entries.", len(_semantic_index))


def _lookup_cache(
    question: str,
    q_emb: list[float],             # pre-computed — eliminates B5 double-embed
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> dict[str, Any] | None:
    """Two-tier lookup: O(1) LRU dict → O(1) SQLite exact → O(n) semantic scan.

    The semantic scan runs over the in-memory index, never touching disk.
    """
    key = _cache_key(question, collection_name, topic_filter, domain_filter, k)

    # ── Tier 1: in-process LRU (sub-microsecond) ─────────────────────────────
    with _cache_lru_lock:
        if key in _cache_lru:
            log.info("Cache hit: LRU (exact).")
            return {**_cache_lru[key], "cached": True, "cache_type": "lru", "cache_score": 1.0}

    # ── Tier 2: SQLite exact key (single indexed lookup, ~0.1 ms) ────────────
    db = _get_db()
    row = db.execute(
        "SELECT answer, contexts, sources FROM cache WHERE key = ?", (key,)
    ).fetchone()
    if row:
        item = {"answer": row[0], "contexts": json.loads(row[1]), "sources": json.loads(row[2])}
        with _cache_lru_lock:
            if len(_cache_lru) >= _RESPONSE_CACHE_LRU_SIZE:
                _cache_lru.pop(next(iter(_cache_lru)))   # evict oldest
            _cache_lru[key] = item
        log.info("Cache hit: SQLite exact.")
        return {**item, "cached": True, "cache_type": "exact", "cache_score": 1.0}

    # ── Tier 3: semantic scan over in-memory index ────────────────────────────
    _load_semantic_index()
    threshold      = float(os.getenv("RAG_RESPONSE_CACHE_THRESHOLD", DEFAULT_RESPONSE_CACHE_THRESHOLD))
    settings_str   = json.dumps(_current_settings(collection_name, topic_filter, domain_filter, k),
                                sort_keys=True)
    best_score, best_item = 0.0, None

    for _key, emb, s_str, payload in _semantic_index:
        if s_str != settings_str:          # string compare is O(len) but cheap
            continue
        score = _cosine(q_emb, emb)
        if score > best_score:
            best_score, best_item = score, payload

    if best_item and best_score >= threshold:
        log.info("Cache hit: semantic (score=%.4f).", best_score)
        return {**best_item, "cached": True, "cache_type": "semantic",
                "cache_score": round(best_score, 4)}

    return None


def _store_cache(
    key: str,
    question: str,
    q_emb: list[float],             # pre-computed — eliminates B5 double-embed
    result: dict[str, Any],
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> None:
    """Single INSERT OR REPLACE — no read-modify-write, no full-file rewrite (B9)."""
    settings = _current_settings(collection_name, topic_filter, domain_filter, k)
    settings_str = json.dumps(settings, sort_keys=True)

    db = _get_db()
    with _db_lock:
        db.execute(
            """INSERT OR REPLACE INTO cache
               (key, question, q_embedding, answer, contexts, sources, settings)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                question,
                json.dumps(q_emb),
                result["answer"],
                json.dumps(result.get("contexts", [])),
                json.dumps(result.get("sources", [])),
                settings_str,
            ),
        )
        db.commit()

    # Update in-process caches
    payload = {
        "answer":   result["answer"],
        "contexts": result.get("contexts", []),
        "sources":  result.get("sources", []),
    }
    with _cache_lru_lock:
        if len(_cache_lru) >= _RESPONSE_CACHE_LRU_SIZE:
            _cache_lru.pop(next(iter(_cache_lru)))
        _cache_lru[key] = payload

    with _semantic_index_lock:
        _semantic_index.append((key, q_emb, settings_str, payload))


def clear_response_cache() -> None:
    """Wipe all cache tiers: SQLite table, LRU dict, semantic index."""
    global _semantic_index, _semantic_index_loaded
    db = _get_db()
    with _db_lock:
        db.execute("DELETE FROM cache")
        db.commit()
    with _cache_lru_lock:
        _cache_lru.clear()
    with _semantic_index_lock:
        _semantic_index.clear()
        _semantic_index_loaded = False
    log.info("Response cache cleared.")


def clear_all_local_artifacts(
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    html_cache_dir: str = DEFAULT_HTML_CACHE_DIR,
    embedding_cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR,
) -> None:
    """Delete all local RAG artifacts so the next run starts from a clean slate."""
    global _cached_embeddings, _raw_embeddings, _cross_encoder, _semantic_index_loaded

    clear_response_cache()
    for directory in (persist_directory, html_cache_dir, embedding_cache_dir):
        if os.path.exists(directory):
            shutil.rmtree(directory)
            log.info("Removed %s", directory)
    if os.path.exists(chunks_path):
        os.remove(chunks_path)
        log.info("Removed %s", chunks_path)

    _build_bm25_retriever.cache_clear()
    with _singleton_lock:
        _cached_embeddings = None
        _raw_embeddings = None
        _cross_encoder = None
    with _semantic_index_lock:
        _semantic_index.clear()
        _semantic_index_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BM25 CACHE  (B3)
#
# BM25Retriever.from_documents() tokenises the entire corpus on every call.
# We cache the built retriever keyed on (file mtime, topic, domain).
# A corpus change triggers mtime change → automatic cache invalidation.
# ══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=32)
def _build_bm25_retriever(
    chunks_path: str,
    mtime: float,           # included in key so stale cache auto-invalidates
    topic: str | None,
    domain: str | None,
    candidate_k: int,
) -> BM25Retriever | None:
    """Build and cache a BM25Retriever for the given filter combination.

    The `mtime` parameter is intentionally part of the cache key — when
    chunks.jsonl is updated the mtime changes and a fresh retriever is built.
    """
    docs = _load_chunks_jsonl(chunks_path)
    filtered = [
        d for d in docs
        if (not topic  or d.metadata.get("topic")  == topic)
        and (not domain or d.metadata.get("domain") == domain)
    ]
    if not filtered:
        return None
    ret = BM25Retriever.from_documents(filtered)
    ret.k = candidate_k
    return ret


def _get_bm25_retriever(
    chunks_path: str,
    topic: str | None,
    domain: str | None,
    candidate_k: int,
) -> BM25Retriever | None:
    """Return a cached BM25Retriever, invalidating automatically on file change."""
    try:
        mtime = round(Path(chunks_path).stat().st_mtime, 2)
    except FileNotFoundError:
        return None
    return _build_bm25_retriever(chunks_path, mtime, topic, domain, candidate_k)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WEB SCRAPING & HTML LOADING
#
# I1: Parallel scraping — all URLs fetched concurrently via ThreadPoolExecutor.
# I3: BS4-first parsing — UnstructuredHTMLLoader skipped by default (opt-in
#     via RAG_USE_UNSTRUCTURED=1). BS4 is ~5–10× faster for plain HTML.
# ══════════════════════════════════════════════════════════════════════════════

def _url_to_cache_path(url: str, cache_dir: str = DEFAULT_HTML_CACHE_DIR) -> Path:
    parsed = urlparse(url)
    slug   = f"{parsed.netloc}{parsed.path}".strip("/").replace("/", "_") or parsed.netloc
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"{slug}_{digest}.html"


def _scrape_url(url: str, cache_dir: str = DEFAULT_HTML_CACHE_DIR, force: bool = False) -> Path:
    path = _url_to_cache_path(url, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    log.info("Scraping %s", url)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "rag-agent/1.0"})
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return path


def _parse_html_bs4(path: Path, metadata: dict[str, str]) -> list[Document]:
    """I3: Default fast parser — pure BS4, no Unstructured overhead."""
    soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return [Document(page_content=text, metadata={**metadata, "source": metadata["source_url"]})]


def _parse_html_unstructured(path: Path, metadata: dict[str, str]) -> list[Document]:
    """Opt-in richer parser — use RAG_USE_UNSTRUCTURED=1 to enable."""
    from langchain_community.document_loaders import UnstructuredHTMLLoader
    loader = UnstructuredHTMLLoader(str(path), mode="single")
    docs   = loader.load()
    for doc in docs:
        doc.metadata.update(metadata)
        doc.metadata.setdefault("source", metadata["source_url"])
    return docs


def _load_single_source(
    src: dict[str, str],
    cache_dir: str,
    force_rescrape: bool,
    use_unstructured: bool,
) -> list[Document]:
    """Scrape + parse one URL. Called concurrently by load_web_sources (I1)."""
    url = src["url"]
    metadata = {
        "source_url": url,
        "source":     url,
        "domain":     urlparse(url).netloc,
        "topic":      src.get("topic", "general"),
    }
    try:
        cached = _scrape_url(url, cache_dir=cache_dir, force=force_rescrape)
        if use_unstructured:
            try:
                loaded = _parse_html_unstructured(cached, metadata)
            except Exception:
                loaded = _parse_html_bs4(cached, metadata)
        else:
            loaded = _parse_html_bs4(cached, metadata)
    except Exception as exc:
        log.warning("Skipping %s – %s", url, exc)
        return []
    return [d for d in loaded if d.page_content.strip()]


def load_web_sources(
    sources: list[dict[str, str]],
    cache_dir: str = DEFAULT_HTML_CACHE_DIR,
    force_rescrape: bool = False,
) -> list[Document]:
    """I1: Fetch and parse all URLs concurrently.

    Uses a ThreadPoolExecutor so network-bound scraping runs in parallel.
    Maintains source ordering in the returned document list.
    """
    use_unstructured = os.getenv("RAG_USE_UNSTRUCTURED", "0") == "1"
    max_workers = int(os.getenv("RAG_SCRAPE_WORKERS", str(DEFAULT_SCRAPE_WORKERS)))

    documents_by_index: dict[int, list[Document]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_load_single_source, src, cache_dir, force_rescrape, use_unstructured): idx
            for idx, src in enumerate(sources)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                documents_by_index[idx] = future.result()
            except Exception as exc:
                log.warning("Source %d failed: %s", idx, exc)
                documents_by_index[idx] = []

    # Preserve original source order
    documents: list[Document] = []
    for idx in range(len(sources)):
        documents.extend(documents_by_index.get(idx, []))
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CHUNKING
#
# I2: RecursiveCharacterTextSplitter is the default. SemanticChunker requires
#     embedding every sentence to find breakpoints — O(sentences) embed calls
#     per document, which dominates indexing time on any non-trivial corpus.
#     Enable with RAG_USE_SEMANTIC_CHUNKING=1 when chunking quality matters
#     more than indexing speed.
# ══════════════════════════════════════════════════════════════════════════════

def chunk_documents(docs: list[Document]) -> list[Document]:
    use_semantic = (
        os.getenv("RAG_USE_SEMANTIC_CHUNKING", "0") == "1"   # I2: default OFF
        and SemanticChunker is not None
    )
    if use_semantic:
        splitter = SemanticChunker(
            get_raw_embeddings(),     # singleton — no reload
            breakpoint_threshold_type=os.getenv("SEMANTIC_BREAKPOINT_TYPE",   "percentile"),
            breakpoint_threshold_amount=float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "90")),
        )
        chunks = list(splitter.split_documents(docs))
        log.info("SemanticChunker produced %d chunks.", len(chunks))
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks   = splitter.split_documents(docs)
        log.info("RecursiveCharacterTextSplitter produced %d chunks.", len(chunks))

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_chars"] = len(chunk.page_content)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VECTOR STORE & INDEXING
#
# I4: add_documents called in batches of RAG_INDEX_BATCH_SIZE (default 500).
#     Avoids SQLite write contention and lets Chroma pipeline embedding + write.
# ══════════════════════════════════════════════════════════════════════════════

def _doc_id(source: str, index: int, content: str) -> str:
    return hashlib.sha256(f"{source}:{index}:{content}".encode()).hexdigest()[:32]


def _save_chunks_jsonl(chunks: list[Document], path: str = DEFAULT_CHUNKS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps({"page_content": chunk.page_content,
                                  "metadata": chunk.metadata}) + "\n")
    # Invalidate BM25 LRU cache — new file means new mtime on next query
    _build_bm25_retriever.cache_clear()


def _load_chunks_jsonl(path: str = DEFAULT_CHUNKS_PATH) -> list[Document]:
    p = Path(path)
    if not p.exists():
        return []
    docs: list[Document] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                docs.append(Document(page_content=item["page_content"],
                                     metadata=item.get("metadata", {})))
    return docs


def load_vector_store(collection_name: str, persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_cached_embeddings(),   # singleton
        persist_directory=persist_directory,
    )


def _vector_store_is_empty(vs: Chroma) -> bool:
    try:
        return vs._collection.count() == 0
    except Exception:
        return not bool(vs.similarity_search("test", k=1))


def _progress(enabled: bool, message: str) -> None:
    """Print flushed indexing progress for CLI runs."""
    if enabled:
        print(f"[index] {message}", flush=True)


def _add_documents_batched(
    vs: Chroma,
    chunks: list[Document],
    ids: list[str],
    batch_size: int,
    show_progress: bool,
) -> None:
    """I4: Embed all chunks in one shot via the raw model, then write to Chroma
    in batches.

    Why bypass CacheBackedEmbeddings here:
      CacheBackedEmbeddings performs a LocalFileStore lookup (a filesystem open())
      for every single chunk before embedding it.  On a cold index with N chunks
      that is N sequential disk probes before a single vector is produced.
      For indexing we want one bulk embed_documents() call on the raw model so
      the sentence-transformer can batch the whole corpus in one pass, then we
      write the resulting vectors straight into Chroma.  The embedding cache is
      still valuable at query time (single-vector lookups are cheap there).
    """
    total = len(chunks)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    _progress(show_progress, f"  Embedding {total} chunks in one batch pass …")
    embeddings = get_raw_embeddings().embed_documents(texts)   # one bulk call

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        vs._collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        _progress(show_progress, f"  Stored chunks {start + 1}–{end} / {total}.")


def index_sources(
    sources: list[dict[str, str]],
    collection_name: str,
    persist_directory: str,
    reindex: bool = False,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    force_rescrape: bool = False,
    show_progress: bool = False,
) -> Chroma:
    _progress(show_progress, "Starting index build.")
    if reindex:
        _progress(show_progress, "Clearing existing vector DB and BM25 chunks.")
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

    _progress(show_progress, "Loading embedding model (may take 20-30s on first run) ...")
    get_raw_embeddings()   # trigger model load now so the log above is visible before the hang
    _progress(show_progress, "Embedding model ready. Opening Chroma.")
    vs   = load_vector_store(collection_name, persist_directory)

    # I1: parallel fetch — all URLs scraped concurrently
    _progress(show_progress,
              f"Fetching {len(sources)} source page(s) in parallel "
              f"(workers={os.getenv('RAG_SCRAPE_WORKERS', str(DEFAULT_SCRAPE_WORKERS))}).")
    docs = load_web_sources(sources, force_rescrape=force_rescrape)
    if not docs:
        raise RuntimeError("No documents loaded – check sources and network access.")

    # I2: fast chunker by default
    _progress(show_progress, f"Loaded {len(docs)} document(s). Chunking documents.")
    chunks = chunk_documents(docs)

    # I4: batched writes
    batch_size = int(os.getenv("RAG_INDEX_BATCH_SIZE", str(DEFAULT_INDEX_BATCH_SIZE)))
    _progress(show_progress,
              f"Generated {len(chunks)} chunk(s). Writing to Chroma in batches of {batch_size}.")
    ids = [
        _doc_id(c.metadata.get("source_url", c.metadata.get("source", "unknown")), i, c.page_content)
        for i, c in enumerate(chunks)
    ]
    _add_documents_batched(vs, chunks, ids, batch_size, show_progress)

    _progress(show_progress, "Saving BM25 chunk sidecar.")
    _save_chunks_jsonl(chunks, chunks_path)
    log.info("Indexed %d chunks into collection '%s'.", len(chunks), collection_name)
    _progress(show_progress, "Index build complete.")
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — QUERY PREPARATION  (B8: concurrent HyDE + multi-query)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_topic(question: str) -> str | None:
    lowered = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return topic
    return None


def _enhance_query(question: str) -> str:
    """Expand common acronyms — zero LLM calls, ~0 ms."""
    if os.getenv("RAG_ENABLE_QUERY_ENHANCER", "1") == "0":
        return question
    replacements = {
        " rag ":   " retrieval augmented generation (RAG) ",
        " bm25 ":  " BM25 sparse keyword retrieval ",
        " hyde ":  " Hypothetical Document Embeddings (HyDE) ",
        " llm ":   " large language model (LLM) ",
    }
    padded = f" {question.lower()} "
    for short, expanded in replacements.items():
        if short in padded and expanded.strip().lower() not in question.lower():
            return question + f" ({expanded.strip()})"
    return question


async def _async_generate_hyde(question: str, loop: asyncio.AbstractEventLoop) -> str:
    """HyDE: generate a short hypothetical answer for dense retrieval (async)."""
    if os.getenv("RAG_ENABLE_HYDE", "0") == "0":
        return question
    model = get_model()     # singleton, no reload

    def _invoke() -> str:
        resp = model.invoke([
            ("system", "Write a short, factual hypothetical answer for retrieval only. "
                       "Under 80 words. No sources."),
            ("user", question),
        ])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    content = await loop.run_in_executor(None, _invoke)
    return f"{question}\n\nHypothetical answer:\n{content.strip()}"


async def _async_generate_multi_queries(
    question: str,
    n: int,
    loop: asyncio.AbstractEventLoop,
) -> list[str]:
    """Multi-query: generate n query variants for broader retrieval (async)."""
    if os.getenv("RAG_ENABLE_MULTI_QUERY", "0") == "0":
        return [question]
    model = get_model()

    def _invoke() -> list[str]:
        prompt = (
            f"Generate {n} distinct search queries covering different aspects of:\n"
            f"Question: {question}\n"
            "Return one query per line, no numbering, no preamble."
        )
        resp     = model.invoke([("user", prompt)])
        content  = resp.content if isinstance(resp.content, str) else str(resp.content)
        variants = [q.strip() for q in content.strip().splitlines() if q.strip()]
        return [question] + variants[:n]

    return await loop.run_in_executor(None, _invoke)


async def prepare_retrieval_queries_async(
    question: str,
    loop: asyncio.AbstractEventLoop,
) -> tuple[list[str], str | None]:
    """Prepare retrieval queries — HyDE and multi-query run concurrently (B8)."""
    topic    = _detect_topic(question)
    enhanced = _enhance_query(question)

    # B8: run HyDE and multi-query concurrently if both enabled
    hyde_enabled  = os.getenv("RAG_ENABLE_HYDE",        "0") == "1"
    multi_enabled = os.getenv("RAG_ENABLE_MULTI_QUERY", "0") == "1"

    if hyde_enabled and multi_enabled:
        hyde_q, variants = await asyncio.gather(
            _async_generate_hyde(enhanced, loop),
            _async_generate_multi_queries(enhanced, DEFAULT_MULTI_QUERY_COUNT, loop),
        )
        # HyDE query replaces the base; multi-query variants are appended
        queries = [hyde_q] + [v for v in variants if v != enhanced]
    elif hyde_enabled:
        hyde_q  = await _async_generate_hyde(enhanced, loop)
        queries = [hyde_q]
    elif multi_enabled:
        queries = await _async_generate_multi_queries(enhanced, DEFAULT_MULTI_QUERY_COUNT, loop)
    else:
        queries = [enhanced]

    return queries, topic


def prepare_retrieval_queries(question: str) -> tuple[list[str], str | None]:
    """Sync wrapper — always runs in a dedicated thread with its own event loop.

    Why a new thread every time:
      Streamlit, Jupyter, and any other framework that owns an event loop will
      cause "Future attached to a different loop" or "This event loop is already
      running" errors if we schedule coroutines onto their loop.  The safest
      cross-environment approach is to spin up a throwaway thread that owns its
      own loop, run the coroutine there to completion, and return the result.
      The overhead is one thread creation (~0.1 ms) which is negligible vs the
      LLM call that HyDE/multi-query would make.
    """
    import concurrent.futures as _cf

    def _run() -> tuple[list[str], str | None]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                prepare_retrieval_queries_async(question, loop)
            )
        finally:
            loop.close()

    with _cf.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RETRIEVAL  (B3, B6: cached BM25 + async parallel retrieval)
# ══════════════════════════════════════════════════════════════════════════════

def _chroma_filter(topic: str | None, domain: str | None) -> dict[str, Any] | None:
    parts = []
    if topic:
        parts.append({"topic":  {"$eq": topic}})
    if domain:
        parts.append({"domain": {"$eq": domain}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _bm25_search(
    query: str,
    chunks_path: str,
    k: int,
    topic: str | None,
    domain: str | None,
) -> list[Document]:
    """Sparse keyword search using the cached BM25Retriever (B3)."""
    retriever = _get_bm25_retriever(chunks_path, topic, domain, k)
    if retriever is None:
        return []
    return retriever.invoke(query)


def _rrf_fuse(ranked_lists: list[list[Document]], k_rrf: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: score = Σ 1/(k_rrf + rank_i)."""
    scores:     dict[str, float]    = {}
    docs_by_id: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(
                doc.metadata.get("source_url", doc.metadata.get("source", "")),
                int(doc.metadata.get("chunk_index", 0)),
                doc.page_content,
            )
            scores[doc_id]     = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
            docs_by_id[doc_id] = doc
    return [docs_by_id[d] for d in sorted(scores, key=lambda d: scores[d], reverse=True)]


def _rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    if not docs or os.getenv("RAG_ENABLE_RERANKING", "1") == "0":
        return docs[:k]

    reranker = get_cross_encoder()
    if reranker is None:
        return docs[:k]

    # 1. Only rerank top-N fused candidates
    rerank_top_n = int(os.getenv("RAG_RERANK_TOP_N", "20"))
    docs_to_rerank = docs[:rerank_top_n]

    try:
        max_chars = int(os.getenv("RAG_RERANK_MAX_CHARS", "1000"))
        pairs = [(query, doc.page_content[:max_chars]) for doc in docs_to_rerank]

        scores = reranker.predict(
            pairs,
            batch_size=int(os.getenv("RAG_RERANK_BATCH_SIZE", "8")),
            show_progress_bar=False,
        )

        ranked = sorted(zip(docs_to_rerank, scores), key=lambda x: float(x[1]), reverse=True)

        # Keep non-reranked docs after reranked docs as fallback
        return [doc for doc, _ in ranked[:k]]

    except Exception as exc:
        log.warning("Reranking failed (%s); using fused order.", exc)
        return docs[:k]

async def retrieve_context_async(
    vector_store: Chroma,
    queries: list[str],
    k: int = DEFAULT_RAG_K,
    topic: str | None = None,
    domain: str | None = None,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> list[Document]:
    """Async hybrid retrieval: all dense + sparse calls run concurrently (B6).

    For n query variants this fires 2n tasks simultaneously instead of
    executing them in serial pairs.
    """
    loop        = asyncio.get_event_loop()
    chroma_flt  = _chroma_filter(topic, domain)
    candidate_k = max(k * 2, 12)
    executor    = ThreadPoolExecutor(max_workers=min(len(queries) * 2, 8))

    def _dense(q: str) -> list[Document]:
        return vector_store.similarity_search(q, k=candidate_k, filter=chroma_flt)

    def _sparse(q: str) -> list[Document]:
        return _bm25_search(q, chunks_path, candidate_k, topic, domain)

    # Fire ALL dense + sparse tasks simultaneously
    tasks = []
    for q in queries:
        tasks.append(loop.run_in_executor(executor, _dense,  q))
        tasks.append(loop.run_in_executor(executor, _sparse, q))

    results = await asyncio.gather(*tasks)
    executor.shutdown(wait=False)

    ranked_lists = list(results)
    fused = _rrf_fuse(ranked_lists)
    log.info("RRF fused %d candidates; reranking to top %d.", len(fused), k)

    # Reranking is CPU-bound; run in executor to avoid blocking the event loop
    reranked = await loop.run_in_executor(None, _rerank, queries[0], fused, k)
    return reranked


def retrieve_context(
    vector_store: Chroma,
    queries: list[str],
    k: int = DEFAULT_RAG_K,
    topic: str | None = None,
    domain: str | None = None,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> list[Document]:
    """Sync wrapper around retrieve_context_async — always uses a dedicated thread.

    Same reasoning as prepare_retrieval_queries: never touch a running loop
    owned by Streamlit, Jupyter, or FastAPI.
    """
    import concurrent.futures as _cf

    def _run() -> list[Document]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                retrieve_context_async(vector_store, queries, k, topic, domain, chunks_path)
            )
        finally:
            loop.close()

    with _cf.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _serialize_context(docs: list[Document], char_limit: int | None = None) -> str:
    limit = char_limit or int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", str(DEFAULT_RAG_CONTEXT_CHAR_LIMIT)))
    parts: list[str] = []
    used = 0
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        budget = max(0, limit - used - len(source) - 20)
        if budget <= 0:
            break
        content = doc.page_content[:budget]
        part    = f"[Source: {source}]\n{content}"
        parts.append(part)
        used += len(part)
    return "\n\n---\n\n".join(parts)


_ANSWER_SYSTEM = """\
You are a precise RAG assistant. Answer the user's question using ONLY the
provided context. If the context does not contain enough information, say
"I don't have enough context to answer this." Do not fabricate information.
Cite the source URL(s) that support your answer.

Format your response as:
Answer: <direct, well-structured answer>
Sources: <comma-separated source URLs>
"""

_REFLECTION_SYSTEM = """\
You are a strict answer reviewer. Check:
1. Is the answer fully grounded in the context? (no hallucinations)
2. Does it actually address the question?
3. Are sources cited correctly?

Return ONLY the corrected final answer in this format:
Answer: <improved answer>
Sources: <comma-separated source URLs>
Rationale: <one-sentence explanation of any change>
"""


def generate_answer(question: str, docs: list[Document]) -> str:
    """Generate and optionally self-reflect on the answer. Uses LLM singleton (B7)."""
    model   = get_model()
    context = _serialize_context(docs)

    resp  = model.invoke([
        ("system", _ANSWER_SYSTEM),
        ("user",   f"Question: {question}\n\nContext:\n{context}"),
    ])
    draft = resp.content if isinstance(resp.content, str) else str(resp.content)

    if os.getenv("RAG_ENABLE_SELF_REFLECTION", "1") != "0":
        resp2 = model.invoke([
            ("system", _REFLECTION_SYSTEM),
            ("user",   f"Question:\n{question}\n\nContext:\n{context}\n\nDraft answer:\n{draft}"),
        ])
        return resp2.content if isinstance(resp2.content, str) else str(resp2.content)

    return draft


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_rag(
    question: str,
    vector_store: Chroma,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = DEFAULT_RAG_K,
    topic_filter: str | None = None,
    domain_filter: str | None = None,
    use_cache: bool = True,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    question_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Full RAG pipeline.

    Latency profile:
      • Cache hits:      ~0.1 ms (LRU) / ~0.5 ms (SQLite) / ~5 ms (semantic)
      • Cache miss path: embedding(1×) + retrieval(parallel) + LLM + reflection
    """
    # B5: embed once, reuse in both _lookup_cache and _store_cache
    q_emb: list[float] = []
    if use_cache:
        q_emb = question_embedding or [float(v) for v in get_raw_embeddings().embed_query(question)]
        hit   = _lookup_cache(question, q_emb, collection_name, topic_filter, domain_filter, k)
        if hit:
            return hit

    queries, detected_topic = prepare_retrieval_queries(question)
    effective_topic         = topic_filter or detected_topic

    docs = retrieve_context(
        vector_store,
        queries=queries,
        k=k,
        topic=effective_topic,
        domain=domain_filter,
        chunks_path=chunks_path,
    )

    if not docs:
        return {
            "answer":            "I could not find relevant context to answer your question.",
            "contexts":          [],
            "sources":           [],
            "cached":            False,
            "cache_type":        None,
            "cache_score":       0.0,
            "retrieval_queries": queries,
            "topic_filter":      effective_topic,
        }

    answer = generate_answer(question, docs)
    key    = _cache_key(question, collection_name, topic_filter, domain_filter, k)
    result: dict[str, Any] = {
        "answer":            answer,
        "contexts":          [d.page_content for d in docs],
        "sources":           list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs)),
        "cached":            False,
        "cache_type":        None,
        "cache_score":       0.0,
        "retrieval_queries": queries,
        "topic_filter":      effective_topic,
    }

    if use_cache:
        # B5: reuse q_emb computed above — no second embed call
        if not q_emb:
            q_emb = [float(v) for v in get_raw_embeddings().embed_query(question)]
        _store_cache(key, question, q_emb, result, collection_name, topic_filter, domain_filter, k)

    return result


def ask(
    question: str,
    url: str = DEFAULT_URL,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
    auto_index: bool = True,
    use_corpus: bool = True,
    topic_filter: str | None = None,
    domain_filter: str | None = None,
    use_cache: bool = True,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    force_rescrape: bool = False,
) -> dict[str, Any]:
    """Top-level entrypoint: manages index lifecycle, then delegates to run_rag."""
    started = time.perf_counter()
    question_embedding: list[float] | None = None

    if use_cache:
        question_embedding = [float(v) for v in get_raw_embeddings().embed_query(question)]
        hit = _lookup_cache(
            question,
            question_embedding,
            collection_name,
            topic_filter,
            domain_filter,
            DEFAULT_RAG_K,
        )
        if hit:
            hit["latency_seconds"] = round(time.perf_counter() - started, 3)
            hit["indexed_now"] = False
            hit["collection_name"] = collection_name
            hit["persist_directory"] = persist_directory
            return hit

    vs      = load_vector_store(collection_name, persist_directory)
    indexed_now = False

    if auto_index and _vector_store_is_empty(vs):
        sources = DEFAULT_WEB_LINKS if use_corpus else [{"url": url, "topic": "general"}]
        vs      = index_sources(sources, collection_name, persist_directory,
                                chunks_path=chunks_path, force_rescrape=force_rescrape)
        indexed_now = True

    result = run_rag(
        question, vs,
        collection_name=collection_name,
        topic_filter=topic_filter,
        domain_filter=domain_filter,
        use_cache=use_cache,
        chunks_path=chunks_path,
        question_embedding=question_embedding,
    )
    result["latency_seconds"]   = round(time.perf_counter() - started, 3)
    result["indexed_now"]       = indexed_now
    result["collection_name"]   = collection_name
    result["persist_directory"] = persist_directory
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production RAG agent.")
    p.add_argument("question",        nargs="?", default="What is task decomposition?")
    p.add_argument("--url",           default=DEFAULT_URL)
    p.add_argument("--collection",    default=DEFAULT_COLLECTION)
    p.add_argument("--persist-dir",   default=DEFAULT_VECTOR_DB_DIR)
    p.add_argument("--chunks-path",   default=DEFAULT_CHUNKS_PATH)
    p.add_argument("--topic-filter",  default=None)
    p.add_argument("--domain-filter", default=None)
    p.add_argument("--index",         action="store_true", help="Build index and exit.")
    p.add_argument("--reindex",       action="store_true", help="Destroy and rebuild index.")
    p.add_argument("--no-auto-index", action="store_true")
    p.add_argument("--single-url",    action="store_true")
    p.add_argument("--force-rescrape",action="store_true")
    p.add_argument("--no-cache",      action="store_true")
    p.add_argument("--clear-cache",   action="store_true")
    p.add_argument("--clear-all-cache", action="store_true",
                   help="Delete response cache, HTML cache, embedding cache, Chroma DB, and BM25 chunks.")
    p.add_argument("--show-context",  action="store_true")
    p.add_argument("--show-queries",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.clear_cache:
        clear_response_cache()

    if args.clear_all_cache:
        clear_all_local_artifacts(
            persist_directory=args.persist_dir,
            chunks_path=args.chunks_path,
        )
        print("Cleared response cache, HTML cache, embedding cache, vector DB, and BM25 chunks.")
        return

    if args.index or args.reindex:
        started = time.perf_counter()
        sources = ([{"url": args.url, "topic": "general"}] if args.single_url
                   else DEFAULT_WEB_LINKS)
        vs    = index_sources(sources, args.collection, args.persist_dir,
                              reindex=args.reindex, chunks_path=args.chunks_path,
                              force_rescrape=args.force_rescrape,
                              show_progress=True)
        count = vs._collection.count()
        print(f"Indexed {count} chunks → collection '{args.collection}' ({args.persist_dir})")
        print(f"Indexing time: {time.perf_counter() - started:.2f}s")
        return

    response = ask(
        args.question,
        url=args.url,
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        auto_index=not args.no_auto_index,
        use_corpus=not args.single_url,
        topic_filter=args.topic_filter,
        domain_filter=args.domain_filter,
        use_cache=not args.no_cache,
        chunks_path=args.chunks_path,
        force_rescrape=args.force_rescrape,
    )

    print("\n" + "=" * 70)
    print(textwrap.dedent(response["answer"]).strip())
    print("=" * 70)
    print(f"\nLatency      : {response['latency_seconds']:.3f}s")
    print(f"Collection   : {response['collection_name']}")
    print(f"Indexed now  : {response['indexed_now']}")
    cache_type  = response.get("cache_type", "—")
    cache_score = response.get("cache_score", 0.0)
    print(f"Cached       : {response['cached']}  [{cache_type}  score={cache_score:.3f}]")
    print(f"Topic filter : {response.get('topic_filter', '—')}")
    if response.get("sources"):
        print("\nSources:")
        for src in response["sources"]:
            print(f"  • {src}")
    if args.show_queries:
        print("\nRetrieval queries:")
        for q in response.get("retrieval_queries", []):
            print(f"  • {q}")
    if args.show_context:
        print("\nContext snippets:")
        for i, ctx in enumerate(response.get("contexts", []), 1):
            print(f"  [{i}] {ctx[:200]}…")


if __name__ == "__main__":
    main()