"""Compatibility helpers for LangChain package layout changes."""

from __future__ import annotations


def install_cachebacked_embeddings_shim() -> None:
    """Expose classic cache classes where older local imports expect them.

    Some LangChain versions moved `CacheBackedEmbeddings` and `LocalFileStore`
    into `langchain_classic`, while older project code may import them from
    `langchain_core` or `langchain`. This shim lets tests, evals, and the UI
    import the existing agent without changing `rag_agent.py`.
    """
    try:
        from langchain_classic.embeddings import CacheBackedEmbeddings
        import langchain_core.embeddings as core_embeddings

        if not hasattr(core_embeddings, "CacheBackedEmbeddings"):
            core_embeddings.CacheBackedEmbeddings = CacheBackedEmbeddings
    except Exception:
        pass

    try:
        from langchain_classic.storage import LocalFileStore
        import langchain_core.stores as core_stores

        if not hasattr(core_stores, "LocalFileStore"):
            core_stores.LocalFileStore = LocalFileStore
    except Exception:
        pass
