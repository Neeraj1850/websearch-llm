"""Production-ready RAG agent with advanced retrieval architecture.

Features:
- Hybrid retrieval: dense (Chroma) + sparse (BM25) with RRF fusion
- Cross-encoder reranking (BAAI/bge-reranker-base)
- Multi-query retrieval for query diversification
- Semantic chunking with SemanticChunker (fallback to RecursiveCharacterTextSplitter)
- HyDE (Hypothetical Document Embeddings) for abstract queries
- Query routing with topic/domain metadata filters
- Response cache (exact SHA-256 key + semantic cosine-similarity fallback)
- LangChain InMemoryCache for repeated LLM calls within a session
- Disk-backed embedding cache (CacheBackedEmbeddings)
- Parallel dense + sparse retrieval via ThreadPoolExecutor
- Context budget enforcement to control token cost and latency
- Stable Chroma document IDs for idempotent re-indexing

Usage:
    python rag_agent.py "What is task decomposition?"
    python rag_agent.py "Explain HyDE" --url https://example.com --single-url
    python rag_agent.py --index
    python rag_agent.py --reindex --force-rescrape
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bs4
import numpy as np
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ── Optional / graceful imports ────────────────────────────────────────────────
try:
    from langchain_core.embeddings import CacheBackedEmbeddings
except ImportError:
    from langchain.embeddings import CacheBackedEmbeddings  # type: ignore[no-redef]

try:
    from langchain_core.stores import LocalFileStore
except ImportError:
    from langchain.storage import LocalFileStore  # type: ignore[no-redef]

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None  # type: ignore[assignment,misc]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag_agent")

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_VECTOR_DB_DIR = ".vector_db/chroma"
DEFAULT_COLLECTION = "rag_documents_minilm"
DEFAULT_HTML_CACHE_DIR = ".cache/html"
DEFAULT_CHUNKS_PATH = ".vector_db/chunks.jsonl"
DEFAULT_EMBEDDING_CACHE_DIR = ".cache/embeddings"
DEFAULT_RESPONSE_CACHE_PATH = ".cache/rag_responses.json"
DEFAULT_RESPONSE_CACHE_THRESHOLD = 0.92   # raised for fewer false hits
DEFAULT_RAG_K = 6                          # wider top-k; reranker trims to best
DEFAULT_RAG_CONTEXT_CHAR_LIMIT = 6000
DEFAULT_MULTI_QUERY_COUNT = 3              # extra query variants for multi-query

# Topic keywords used by the router
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "agents": ["agent", "agents", "tool use", "autonomous"],
    "rag": ["rag", "retrieval augmented", "retrieval-augmented"],
    "tools": ["tool", "tools", "function calling"],
    "memory": ["memory", "long-term memory", "short-term memory"],
    "retrieval": ["retrieval", "search", "bm25", "hybrid search"],
    "embeddings": ["embedding", "embeddings", "sentence transformer"],
    "chunking": ["chunk", "chunking", "text split"],
    "reranking": ["rerank", "reranking", "cross-encoder"],
}

DEFAULT_WEB_LINKS: list[dict[str, str]] = [
    {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/", "topic": "agents"},
    {"url": "https://python.langchain.com/docs/how_to/semantic-chunker/", "topic": "chunking"},
    {"url": "https://python.langchain.com/docs/how_to/MultiQueryRetriever/", "topic": "retrieval"},
    {"url": "https://python.langchain.com/docs/how_to/contextual_compression/", "topic": "reranking"},
    {"url": "https://python.langchain.com/docs/how_to/ensemble_retriever/", "topic": "retrieval"},
    {"url": "https://www.pinecone.io/learn/retrieval-augmented-generation/", "topic": "rag"},
    {"url": "https://www.pinecone.io/learn/hybrid-search-intro/", "topic": "retrieval"},
    {"url": "https://weaviate.io/blog/hybrid-search-explained", "topic": "retrieval"},
    {"url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2", "topic": "embeddings"},
    {"url": "https://huggingface.co/BAAI/bge-reranker-base", "topic": "reranking"},
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL & EMBEDDING BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _setup_llm_cache() -> None:
    """Install a process-local InMemoryCache for repeated identical LLM calls."""
    if os.getenv("RAG_ENABLE_LLM_CACHE", "1") == "0":
        return
    try:
        from langchain_core.caches import InMemoryCache
        from langchain_core.globals import get_llm_cache, set_llm_cache
        if get_llm_cache() is None:
            set_llm_cache(InMemoryCache())
    except Exception:
        pass


def build_model():
    """Return the chat LLM (Groq or Ollama) with temperature=0 for reproducibility."""
    load_dotenv()
    _setup_llm_cache()
    provider = os.getenv("RAG_MODEL_PROVIDER", "groq").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            temperature=0,
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env or set RAG_MODEL_PROVIDER=ollama."
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=0,
        max_retries=3,
        timeout=30,
    )


def build_raw_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace sentence-transformer embedding model (L2-normalized)."""
    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        model_kwargs={"device": os.getenv("RAG_EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_embeddings() -> CacheBackedEmbeddings:
    """Wrap raw embeddings with a disk-backed LangChain cache under .cache/embeddings."""
    raw = build_raw_embeddings()
    cache_dir = os.getenv("RAG_EMBEDDING_CACHE_DIR", DEFAULT_EMBEDDING_CACHE_DIR)
    namespace = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).replace("/", "__")
    store = LocalFileStore(cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(raw, store, namespace=namespace)


# ══════════════════════════════════════════════════════════════════════════════
# 2. WEB SCRAPING & HTML LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _url_to_cache_path(url: str, cache_dir: str = DEFAULT_HTML_CACHE_DIR) -> Path:
    parsed = urlparse(url)
    slug = f"{parsed.netloc}{parsed.path}".strip("/").replace("/", "_") or parsed.netloc
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"{slug}_{digest}.html"


def _scrape_url(url: str, cache_dir: str = DEFAULT_HTML_CACHE_DIR, force: bool = False) -> Path:
    """Download a URL to the local HTML cache; skips download if cached."""
    path = _url_to_cache_path(url, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    log.info("Scraping %s", url)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "rag-agent/1.0"})
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    return path


def _parse_html_unstructured(path: Path, metadata: dict[str, str]) -> list[Document]:
    loader = UnstructuredHTMLLoader(str(path), mode="single")
    docs = loader.load()
    for doc in docs:
        doc.metadata.update(metadata)
        doc.metadata.setdefault("source", metadata["source_url"])
    return docs


def _parse_html_bs4(path: Path, metadata: dict[str, str]) -> list[Document]:
    soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return [Document(page_content=text, metadata={**metadata, "source": metadata["source_url"]})]


def load_web_sources(
    sources: list[dict[str, str]],
    cache_dir: str = DEFAULT_HTML_CACHE_DIR,
    force_rescrape: bool = False,
) -> list[Document]:
    """Scrape → cache → parse each source; returns non-empty Documents."""
    documents: list[Document] = []
    for src in sources:
        url = src["url"]
        metadata = {
            "source_url": url,
            "source": url,
            "domain": urlparse(url).netloc,
            "topic": src.get("topic", "general"),
        }
        try:
            cached = _scrape_url(url, cache_dir=cache_dir, force=force_rescrape)
            try:
                loaded = _parse_html_unstructured(cached, metadata)
            except Exception:
                loaded = _parse_html_bs4(cached, metadata)
        except Exception as exc:
            log.warning("Skipping %s – %s", url, exc)
            continue
        documents.extend(d for d in loaded if d.page_content.strip())
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into retrievable chunks.

    Priority:
      1. SemanticChunker (embedding-driven boundary detection).
      2. RecursiveCharacterTextSplitter (size-based fallback).

    Every chunk gets chunk_index and chunk_chars metadata for deduplication
    and debugging.
    """
    use_semantic = os.getenv("RAG_USE_SEMANTIC_CHUNKING", "1") == "1" and SemanticChunker is not None
    if use_semantic:
        splitter = SemanticChunker(
            build_raw_embeddings(),
            breakpoint_threshold_type=os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile"),
            breakpoint_threshold_amount=float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "90")),
        )
        chunks = list(splitter.split_documents(docs))
        log.info("SemanticChunker produced %d chunks", len(chunks))
    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        log.info("RecursiveCharacterTextSplitter produced %d chunks", len(chunks))

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_chars"] = len(chunk.page_content)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4. VECTOR STORE & INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def _doc_id(source: str, index: int, content: str) -> str:
    """Stable 32-char SHA-256 document ID for idempotent Chroma inserts."""
    digest = hashlib.sha256(f"{source}:{index}:{content}".encode()).hexdigest()
    return digest[:32]


def save_chunks_jsonl(chunks: list[Document], path: str = DEFAULT_CHUNKS_PATH) -> None:
    """Persist chunks as JSONL for BM25 sparse retrieval."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps({"page_content": chunk.page_content, "metadata": chunk.metadata}) + "\n")


def load_chunks_jsonl(path: str = DEFAULT_CHUNKS_PATH) -> list[Document]:
    """Load BM25 sidecar chunks from disk."""
    p = Path(path)
    if not p.exists():
        return []
    docs: list[Document] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                docs.append(Document(page_content=item["page_content"], metadata=item.get("metadata", {})))
    return docs


def load_vector_store(collection_name: str, persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=build_embeddings(),
        persist_directory=persist_directory,
    )


def _vector_store_is_empty(vs: Chroma) -> bool:
    try:
        return vs._collection.count() == 0
    except Exception:
        return not bool(vs.similarity_search("test", k=1))


def index_sources(
    sources: list[dict[str, str]],
    collection_name: str,
    persist_directory: str,
    reindex: bool = False,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    force_rescrape: bool = False,
) -> Chroma:
    """Build dense (Chroma) + sparse (JSONL/BM25) indexes for the given corpus."""
    if reindex:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

    vs = load_vector_store(collection_name, persist_directory)
    docs = load_web_sources(sources, force_rescrape=force_rescrape)
    if not docs:
        raise RuntimeError("No documents loaded – check sources and network access.")

    chunks = chunk_documents(docs)
    ids = [
        _doc_id(c.metadata.get("source_url", c.metadata.get("source", "unknown")), i, c.page_content)
        for i, c in enumerate(chunks)
    ]
    vs.add_documents(chunks, ids=ids)
    save_chunks_jsonl(chunks, chunks_path)
    log.info("Indexed %d chunks into collection '%s'.", len(chunks), collection_name)
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# 5. QUERY PREPARATION (Routing, Enhancement, HyDE, Multi-Query)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_topic(question: str) -> str | None:
    lowered = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return topic
    return None


def enhance_query(question: str) -> str:
    """Expand common acronyms inline without an extra LLM call."""
    if os.getenv("RAG_ENABLE_QUERY_ENHANCER", "1") == "0":
        return question
    replacements = {
        " rag ": " retrieval augmented generation (RAG) ",
        " bm25 ": " BM25 sparse keyword retrieval ",
        " hyde ": " Hypothetical Document Embeddings (HyDE) ",
        " llm ": " large language model (LLM) ",
    }
    padded = f" {question.lower()} "
    for short, expanded in replacements.items():
        if short in padded and expanded.strip().lower() not in question.lower():
            question = question + f" ({expanded.strip()})"
            break
    return question


def generate_multi_queries(question: str, n: int = DEFAULT_MULTI_QUERY_COUNT) -> list[str]:
    """Generate n query variants via LLM for multi-query retrieval.

    Multi-query retrieval counteracts embedding sensitivity by fetching
    documents for several paraphrases of the same question, then merging
    results before reranking. Disabled by default (env RAG_ENABLE_MULTI_QUERY=1).
    """
    if os.getenv("RAG_ENABLE_MULTI_QUERY", "0") == "0":
        return [question]
    model = build_model()
    prompt = (
        f"Generate {n} distinct search queries that cover different aspects of the "
        f"following question. Return one query per line, no numbering, no preamble.\n\n"
        f"Question: {question}"
    )
    resp = model.invoke([("user", prompt)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    variants = [q.strip() for q in content.strip().splitlines() if q.strip()]
    return [question] + variants[:n]


def generate_hyde_doc(question: str) -> str:
    """Generate a hypothetical answer document for improved dense retrieval.

    HyDE embeds a plausible answer instead of the sparse question, which often
    aligns better with the dense index of factual documents. Enabled via
    env RAG_ENABLE_HYDE=1.
    """
    if os.getenv("RAG_ENABLE_HYDE", "0") == "0":
        return question
    model = build_model()
    resp = model.invoke([
        ("system", "Write a short, factual hypothetical answer for retrieval only. Under 80 words. No sources."),
        ("user", question),
    ])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return f"{question}\n\nHypothetical answer:\n{content.strip()}"


def prepare_retrieval_queries(question: str) -> tuple[list[str], str | None]:
    """Return (list_of_retrieval_queries, topic_filter).

    Pipeline:
      1. Detect topic for metadata filtering.
      2. Expand acronyms in the question.
      3. Optionally apply HyDE on the enhanced question.
      4. Optionally generate additional query variants (multi-query).
    """
    topic = _detect_topic(question)
    enhanced = enhance_query(question)
    hyde_query = generate_hyde_doc(enhanced)
    queries = generate_multi_queries(hyde_query)
    return queries, topic


# ══════════════════════════════════════════════════════════════════════════════
# 6. RETRIEVAL (Dense + Sparse + RRF Fusion + Reranking)
# ══════════════════════════════════════════════════════════════════════════════

def _metadata_matches(doc: Document, topic: str | None, domain: str | None) -> bool:
    if topic and doc.metadata.get("topic") != topic:
        return False
    if domain and doc.metadata.get("domain") != domain:
        return False
    return True


def _chroma_filter(topic: str | None, domain: str | None) -> dict[str, Any] | None:
    """Convert optional filters into Chroma's `$and`/$eq query syntax."""
    parts = []
    if topic:
        parts.append({"topic": {"$eq": topic}})
    if domain:
        parts.append({"domain": {"$eq": domain}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _bm25_search(
    query: str,
    docs: list[Document],
    k: int,
    topic: str | None,
    domain: str | None,
) -> list[Document]:
    filtered = [d for d in docs if _metadata_matches(d, topic, domain)]
    if not filtered:
        return []
    ret = BM25Retriever.from_documents(filtered)
    ret.k = k
    return ret.invoke(query)


def _rrf_fuse(
    ranked_lists: list[list[Document]],
    k_rrf: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion across multiple ranked document lists.

    RRF score = Σ 1/(k_rrf + rank_i). Higher score → better fused rank.
    Deduplication is by stable doc ID so equal chunks from dense+sparse
    accumulate scores instead of appearing twice.
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(
                doc.metadata.get("source_url", doc.metadata.get("source", "")),
                int(doc.metadata.get("chunk_index", 0)),
                doc.page_content,
            )
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
            docs_by_id[doc_id] = doc
    ordered = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [docs_by_id[d] for d in ordered]


def _rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Cross-encoder reranking with BAAI/bge-reranker-base.

    The cross-encoder evaluates every (query, chunk) pair jointly, which is
    more accurate than vector similarity alone. Falls back to fused order on
    error or when reranking is disabled.
    """
    if not docs or os.getenv("RAG_ENABLE_RERANKING", "1") == "0":
        return docs[:k]
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(
            os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            device=os.getenv("RAG_RERANKER_DEVICE", "cpu"),
        )
        pairs = [(query, doc.page_content[:2500]) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [doc for doc, _ in ranked[:k]]
    except Exception as exc:
        log.warning("Reranking failed (%s); using fused order.", exc)
        return docs[:k]


def retrieve_context(
    vector_store: Chroma,
    queries: list[str],
    k: int = DEFAULT_RAG_K,
    topic: str | None = None,
    domain: str | None = None,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> list[Document]:
    """Full hybrid retrieval pipeline for a list of query variants.

    Steps:
      1. For each query variant, run dense (Chroma) and sparse (BM25) in parallel.
      2. Aggregate all result lists and fuse with RRF.
      3. Rerank the fused pool with a cross-encoder and return top-k.
    """
    chroma_flt = _chroma_filter(topic, domain)
    bm25_docs = load_chunks_jsonl(chunks_path)
    candidate_k = max(k * 4, 12)  # wider pool for reranker to work with

    ranked_lists: list[list[Document]] = []

    def _dense(q: str) -> list[Document]:
        return vector_store.similarity_search(q, k=candidate_k, filter=chroma_flt)

    def _sparse(q: str) -> list[Document]:
        return _bm25_search(q, bm25_docs, candidate_k, topic, domain)

    with ThreadPoolExecutor(max_workers=min(len(queries) * 2, 8)) as ex:
        futures = [(ex.submit(_dense, q), ex.submit(_sparse, q)) for q in queries]
        for dense_fut, sparse_fut in futures:
            ranked_lists.append(dense_fut.result())
            ranked_lists.append(sparse_fut.result())

    fused = _rrf_fuse(ranked_lists)
    log.info("RRF fused %d candidates; reranking to top %d.", len(fused), k)
    return _rerank(queries[0], fused, k=k)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ANSWER GENERATION WITH SELF-REFLECTION
# ══════════════════════════════════════════════════════════════════════════════

def _serialize_context(docs: list[Document], char_limit: int | None = None) -> str:
    """Serialize top-ranked docs into the LLM prompt; respects a char budget."""
    limit = char_limit or int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", str(DEFAULT_RAG_CONTEXT_CHAR_LIMIT)))
    parts: list[str] = []
    used = 0
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        budget = max(0, limit - used - len(source) - 20)
        if budget <= 0:
            break
        content = doc.page_content[:budget]
        part = f"[Source: {source}]\n{content}"
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
    """Generate an answer from retrieved context, with optional self-reflection."""
    model = build_model()
    context = _serialize_context(docs)

    resp = model.invoke([
        ("system", _ANSWER_SYSTEM),
        ("user", f"Question: {question}\n\nContext:\n{context}"),
    ])
    draft = resp.content if isinstance(resp.content, str) else str(resp.content)

    if os.getenv("RAG_ENABLE_SELF_REFLECTION", "1") != "0":
        resp2 = model.invoke([
            ("system", _REFLECTION_SYSTEM),
            ("user", f"Question:\n{question}\n\nContext:\n{context}\n\nDraft answer:\n{draft}"),
        ])
        return resp2.content if isinstance(resp2.content, str) else str(resp2.content)

    return draft


# ══════════════════════════════════════════════════════════════════════════════
# 8. RESPONSE CACHE (Exact + Semantic)
# ══════════════════════════════════════════════════════════════════════════════

def _current_settings(
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> dict[str, Any]:
    return {
        "collection_name": collection_name,
        "topic_filter": topic_filter,
        "domain_filter": domain_filter,
        "k": k,
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "reranking": os.getenv("RAG_ENABLE_RERANKING", "1"),
        "self_reflection": os.getenv("RAG_ENABLE_SELF_REFLECTION", "1"),
        "hyde": os.getenv("RAG_ENABLE_HYDE", "0"),
        "multi_query": os.getenv("RAG_ENABLE_MULTI_QUERY", "0"),
        "query_enhancer": os.getenv("RAG_ENABLE_QUERY_ENHANCER", "1"),
        "context_char_limit": os.getenv("RAG_CONTEXT_CHAR_LIMIT", str(DEFAULT_RAG_CONTEXT_CHAR_LIMIT)),
    }


def _cache_key(
    question: str,
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> str:
    payload = {"question": question, **_current_settings(collection_name, topic_filter, domain_filter, k)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_cache(path: str = DEFAULT_RESPONSE_CACHE_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, Any], path: str = DEFAULT_RESPONSE_CACHE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _lookup_cache(
    question: str,
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> dict[str, Any] | None:
    """Check exact cache key first; then fall back to semantic similarity lookup."""
    cache = _load_cache()
    if not cache:
        return None

    key = _cache_key(question, collection_name, topic_filter, domain_filter, k)
    if key in cache:
        item = cache[key]
        return {**item, "cached": True, "cache_type": "exact", "cache_score": 1.0}

    threshold = float(os.getenv("RAG_RESPONSE_CACHE_THRESHOLD", DEFAULT_RESPONSE_CACHE_THRESHOLD))
    settings_now = _current_settings(collection_name, topic_filter, domain_filter, k)
    q_emb = [float(v) for v in build_raw_embeddings().embed_query(question)]

    best_score, best_item = 0.0, None
    for item in cache.values():
        if item.get("settings") != settings_now:
            continue
        score = _cosine(q_emb, item.get("question_embedding", []))
        if score > best_score:
            best_score, best_item = score, item

    if best_item and best_score >= threshold:
        return {**best_item, "cached": True, "cache_type": "semantic", "cache_score": round(best_score, 4)}
    return None


def _store_cache(
    question: str,
    result: dict[str, Any],
    collection_name: str,
    topic_filter: str | None,
    domain_filter: str | None,
    k: int,
) -> None:
    key = _cache_key(question, collection_name, topic_filter, domain_filter, k)
    q_emb = [float(v) for v in build_raw_embeddings().embed_query(question)]
    cache = _load_cache()
    cache[key] = {
        "question": question,
        "question_embedding": q_emb,
        "answer": result["answer"],
        "contexts": result.get("contexts", []),
        "sources": result.get("sources", []),
        "settings": _current_settings(collection_name, topic_filter, domain_filter, k),
    }
    _save_cache(cache)


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_rag(
    question: str,
    vector_store: Chroma,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = DEFAULT_RAG_K,
    topic_filter: str | None = None,
    domain_filter: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Execute the full RAG pipeline for a question.

    Pipeline:
      1. Cache lookup (exact → semantic).
      2. Query preparation (enhance → HyDE → multi-query).
      3. Hybrid retrieval (dense + sparse → RRF fusion → cross-encoder rerank).
      4. Answer generation + optional self-reflection.
      5. Cache store.

    Returns a dict with: answer, contexts, sources, cached, cache_type,
    cache_score, retrieval_queries, topic_filter.
    """
    if use_cache:
        hit = _lookup_cache(question, collection_name, topic_filter, domain_filter, k)
        if hit:
            log.info("Cache hit (%s, score=%.3f).", hit.get("cache_type"), hit.get("cache_score", 1.0))
            return hit

    queries, detected_topic = prepare_retrieval_queries(question)
    effective_topic = topic_filter or detected_topic

    docs = retrieve_context(
        vector_store,
        queries=queries,
        k=k,
        topic=effective_topic,
        domain=domain_filter,
    )

    if not docs:
        return {
            "answer": "I could not find relevant context to answer your question.",
            "contexts": [],
            "sources": [],
            "cached": False,
            "cache_type": None,
            "cache_score": 0.0,
            "retrieval_queries": queries,
            "topic_filter": effective_topic,
        }

    answer = generate_answer(question, docs)
    result: dict[str, Any] = {
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "sources": list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs)),
        "cached": False,
        "cache_type": None,
        "cache_score": 0.0,
        "retrieval_queries": queries,
        "topic_filter": effective_topic,
    }

    if use_cache:
        _store_cache(question, result, collection_name, topic_filter, domain_filter, k)

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
    """High-level entrypoint: load/build index, then run the RAG pipeline.

    Returns the run_rag result enriched with latency_seconds and indexed_now.
    """
    started = time.perf_counter()
    vs = load_vector_store(collection_name, persist_directory)
    indexed_now = False

    if auto_index and _vector_store_is_empty(vs):
        sources = DEFAULT_WEB_LINKS if use_corpus else [{"url": url, "topic": "general"}]
        vs = index_sources(
            sources,
            collection_name,
            persist_directory,
            chunks_path=chunks_path,
            force_rescrape=force_rescrape,
        )
        indexed_now = True

    result = run_rag(
        question,
        vs,
        collection_name=collection_name,
        topic_filter=topic_filter,
        domain_filter=domain_filter,
        use_cache=use_cache,
    )
    result["latency_seconds"] = round(time.perf_counter() - started, 3)
    result["indexed_now"] = indexed_now
    result["collection_name"] = collection_name
    result["persist_directory"] = persist_directory
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production RAG agent – index or query a web corpus.")
    p.add_argument("question", nargs="?", default="What is task decomposition?")
    p.add_argument("--url", default=DEFAULT_URL, help="Single-page URL (used with --single-url).")
    p.add_argument("--show-context", action="store_true", help="Print retrieved context snippets.")
    p.add_argument("--show-queries", action="store_true", help="Print retrieval query variants.")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--persist-dir", default=DEFAULT_VECTOR_DB_DIR)
    p.add_argument("--chunks-path", default=DEFAULT_CHUNKS_PATH)
    p.add_argument("--index", action="store_true", help="Build index and exit.")
    p.add_argument("--reindex", action="store_true", help="Destroy and rebuild the index.")
    p.add_argument("--no-auto-index", action="store_true", help="Disable automatic indexing.")
    p.add_argument("--single-url", action="store_true", help="Index/query only --url, not the full corpus.")
    p.add_argument("--topic-filter", default=None)
    p.add_argument("--domain-filter", default=None)
    p.add_argument("--force-rescrape", action="store_true", help="Re-download cached HTML files.")
    p.add_argument("--no-cache", action="store_true", help="Bypass response cache.")
    p.add_argument("--clear-cache", action="store_true", help="Delete the response cache before running.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.clear_cache and os.path.exists(DEFAULT_RESPONSE_CACHE_PATH):
        os.remove(DEFAULT_RESPONSE_CACHE_PATH)
        log.info("Response cache cleared.")

    if args.index or args.reindex:
        started = time.perf_counter()
        sources = (
            [{"url": args.url, "topic": "general"}]
            if args.single_url
            else DEFAULT_WEB_LINKS
        )
        vs = index_sources(
            sources,
            args.collection,
            args.persist_dir,
            reindex=args.reindex,
            chunks_path=args.chunks_path,
            force_rescrape=args.force_rescrape,
        )
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
    print(f"Cached       : {response['cached']}  [{response.get('cache_type', '—')}  score={response.get('cache_score', 0):.3f}]")
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