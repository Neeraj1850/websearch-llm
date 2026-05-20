"""RAG evaluation dataset for the upgraded 20-link corpus."""

from __future__ import annotations


RAG_EVAL_CASES = [
    {
        "id": "agent_task_decomposition",
        "question": "What is task decomposition in LLM-powered agents?",
        "topic_filter": "agents",
        "reference": (
            "Task decomposition breaks a complex task into smaller, manageable subgoals or steps "
            "that an agent can solve more reliably."
        ),
    },
    {
        "id": "agent_task_decomposition_methods",
        "question": "What are common ways to perform task decomposition?",
        "topic_filter": "agents",
        "reference": (
            "Task decomposition can be done by prompting the LLM directly, using task-specific "
            "instructions, or with human input."
        ),
    },
    {
        "id": "agent_reflection",
        "question": "How does the article describe reflection for autonomous agents?",
        "topic_filter": "agents",
        "reference": (
            "Reflection lets agents review past actions, identify mistakes, and improve future plans "
            "through self-criticism or feedback."
        ),
    },
    {
        "id": "agent_memory",
        "question": "What role does memory play in LLM-powered agents?",
        "topic_filter": "memory",
        "reference": (
            "Memory helps agents retain short-term context and long-term knowledge so they can use "
            "past information during planning and execution."
        ),
    },
    {
        "id": "agent_tool_use",
        "question": "What is the purpose of tool use in LLM-powered agents?",
        "topic_filter": "tools",
        "reference": (
            "Tool use lets agents call external APIs, search, code interpreters, or other systems to "
            "access capabilities and information beyond the base model."
        ),
    },
    {
        "id": "rag_definition",
        "question": "What is retrieval augmented generation?",
        "topic_filter": "rag",
        "reference": (
            "Retrieval augmented generation is a technique that retrieves relevant external context "
            "and provides it to a language model so the model can generate a more grounded answer."
        ),
    },
    {
        "id": "hybrid_search",
        "question": "Why combine dense vector search with BM25 keyword search?",
        "topic_filter": "retrieval",
        "reference": (
            "Hybrid search combines semantic matching from dense vectors with exact-term matching "
            "from BM25, improving retrieval when queries include both concepts and specific keywords."
        ),
    },
    {
        "id": "sentence_transformers_embeddings",
        "question": "What are sentence-transformer embeddings useful for in retrieval systems?",
        "topic_filter": "embeddings",
        "reference": (
            "Sentence-transformer embeddings are useful for retrieval because they convert text into "
            "dense vectors that can be compared semantically for similarity search."
        ),
    },
    {
        "id": "metadata_filtering",
        "question": "How can metadata filtering improve RAG retrieval?",
        "domain_filter": "docs.langchain.com",
        "reference": (
            "Metadata filtering narrows retrieval to documents matching attributes such as source, "
            "domain, or topic, which can reduce irrelevant context before generation."
        ),
    },
    {
        "id": "out_of_scope",
        "question": "What does the indexed corpus say about the nutritional value of mangoes?",
        "reference": (
            "The indexed RAG corpus is about agents, LangChain, retrieval, embeddings, and reranking; "
            "it does not provide information about the nutritional value of mangoes."
        ),
    },
]
