"""Unit tests for RAG indexing and retrieval helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_agent import build_document_id, chunk_documents, serialize_docs


def test_chunk_documents_splits_long_document() -> None:
    doc = Document(page_content="Sentence about agents. " * 200, metadata={"source": "unit-test"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "unit-test" for chunk in chunks)


def test_build_document_id_is_stable() -> None:
    first = build_document_id("source", 1, "content")
    second = build_document_id("source", 1, "content")
    third = build_document_id("source", 2, "content")
    assert first == second
    assert first != third


def test_serialize_docs_includes_source_and_content() -> None:
    docs = [Document(page_content="hello world", metadata={"source": "source-url"})]
    serialized = serialize_docs(docs)
    assert "Source: source-url" in serialized
    assert "Content: hello world" in serialized
