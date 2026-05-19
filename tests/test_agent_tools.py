"""Unit tests for individual search/research tools in agent.py."""

from __future__ import annotations

import os

import pytest

from agent import arxiv_search, compact_text, format_error, web_search, wikipedia_search


def test_compact_text_truncates_long_text() -> None:
    text = "word " * 300
    result = compact_text(text, limit=80)
    assert len(result) <= 80
    assert result.endswith("...")


def test_format_error_includes_tool_name() -> None:
    result = format_error("sample_tool", ValueError("bad input"))
    assert "sample_tool failed" in result
    assert "ValueError" in result


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TOOL_TESTS") != "1",
    reason="Live tool tests require network access. Set RUN_LIVE_TOOL_TESTS=1 to run.",
)
def test_web_search_live() -> None:
    result = web_search.invoke({"query": "LangChain RAG"})
    assert result
    assert "URL:" in result or "failed" in result


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TOOL_TESTS") != "1",
    reason="Live tool tests require network access. Set RUN_LIVE_TOOL_TESTS=1 to run.",
)
def test_wikipedia_search_live() -> None:
    result = wikipedia_search.invoke({"query": "Ada Lovelace"})
    assert result
    assert "Ada" in result or "failed" in result


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TOOL_TESTS") != "1",
    reason="Live tool tests require network access. Set RUN_LIVE_TOOL_TESTS=1 to run.",
)
def test_arxiv_search_live() -> None:
    result = arxiv_search.invoke({"query": "attention is all you need"})
    assert result
    assert "URL:" in result or "failed" in result
