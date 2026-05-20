"""Unit tests for RAG indexing and retrieval helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_agent import (
    _cache_key,
    _chroma_filter,
    _cosine,
    _doc_id,
    _load_cache,
    _metadata_matches,
    _rrf_fuse,
    _save_cache,
    _serialize_context,
    chunk_documents,
    enhance_query,
    prepare_retrieval_queries,
)


def test_chunk_documents_splits_long_document() -> None:
    import os

    os.environ["RAG_USE_SEMANTIC_CHUNKING"] = "0"
    doc = Document(page_content="Sentence about agents. " * 200, metadata={"source": "unit-test"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "unit-test" for chunk in chunks)


def test_build_document_id_is_stable() -> None:
    first = _doc_id("source", 1, "content")
    second = _doc_id("source", 1, "content")
    third = _doc_id("source", 2, "content")
    assert first == second
    assert first != third


def test_serialize_context_includes_source_and_content() -> None:
    docs = [Document(page_content="hello world", metadata={"source": "source-url"})]
    serialized = _serialize_context(docs)
    assert "[Source: source-url]" in serialized
    assert "hello world" in serialized


def test_serialize_context_respects_budget() -> None:
    docs = [Document(page_content="x" * 500, metadata={"source": "source-url"})]
    serialized = _serialize_context(docs, char_limit=80)
    assert "[Source: source-url]" in serialized
    assert len(serialized) <= 90


def test_query_preparation_routes_and_enhances_rag() -> None:
    queries, topic = prepare_retrieval_queries("What is RAG")
    assert topic == "rag"
    assert any("retrieval augmented generation" in query.lower() for query in queries)
    assert "retrieval augmented generation" in enhance_query("What is RAG").lower()


def test_metadata_filter_helpers() -> None:
    doc = Document(
        page_content="hybrid search",
        metadata={"topic": "retrieval", "domain": "docs.langchain.com"},
    )
    assert _metadata_matches(doc, topic="retrieval", domain=None)
    assert _metadata_matches(doc, topic=None, domain="docs.langchain.com")
    assert not _metadata_matches(doc, topic="agents", domain=None)
    assert _chroma_filter(topic="retrieval", domain=None) == {"topic": {"$eq": "retrieval"}}
    assert _chroma_filter(topic=None, domain="docs.langchain.com") == {
        "domain": {"$eq": "docs.langchain.com"}
    }
    assert _chroma_filter(topic="retrieval", domain="docs.langchain.com") == {
        "$and": [{"topic": {"$eq": "retrieval"}}, {"domain": {"$eq": "docs.langchain.com"}}]
    }


def test_rrf_fuse_merges_duplicate_chunks() -> None:
    docs = [
        Document(page_content="same content", metadata={"source_url": "source", "chunk_index": 1}),
        Document(page_content="same content", metadata={"source_url": "source", "chunk_index": 1}),
        Document(page_content="different", metadata={"source_url": "source", "chunk_index": 2}),
    ]
    assert len(_rrf_fuse([docs[:2], docs[1:]])) == 2


def test_response_cache_round_trip(tmp_path) -> None:
    cache_path = tmp_path / "responses.json"
    cache = {"abc": {"answer": "cached"}}
    _save_cache(cache, path=str(cache_path))
    assert _load_cache(path=str(cache_path)) == cache


def test_response_cache_key_changes_with_filter() -> None:
    first = _cache_key("question", "collection", topic_filter="retrieval", domain_filter=None, k=3)
    second = _cache_key("question", "collection", topic_filter="agents", domain_filter=None, k=3)
    assert first != second


def test_cosine_similarity_for_cache() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
